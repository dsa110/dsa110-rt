"""``TransportTx`` — fast-vis cube transmit (M3 chunk 8).

Loopback / unicast UDP transmitter for fast-vis cubes; satisfies the
chunk-4 :class:`dsart.services.corr_fast_integration.TransportTxStage`
Protocol so the orchestrator can drop this in without code change.

# Cube shape contract (F26 in M3_PLAN_FIXES.md)

The cubes that arrive at :meth:`TransportTx.transmit` come from the
chunk-4 Stage-2 FIFO (no-op stub today, real impl in chunk 3b). Their
shape depends on which upstream stage is active:

* **Sparse-COO** ``(N_DM, n_fast_vis, N_filled)`` — current chunk-4
  output. The ``N_filled`` axis is the filled-cell list of the
  cached :class:`SparsityPattern` (chunk 3a). The receiver scatters
  these back to a dense ``(N_grid, N_grid)`` image-cube via the
  same pattern.
* **Image cube** ``(N_DM, n_fast_vis, N_grid, N_grid)`` — when an
  upstream stage has already done the iFFT2. Useful for the chunk-9
  full-pipeline orchestrator and as a forward-compatible shape.

``TransportTx`` auto-detects via ``cube.ndim``:

* ``ndim == 3`` → sparse-COO; payload per ``(dm_idx, t_idx)`` slice
  is the 1-D ``(N_filled,)`` complex slice.
* ``ndim == 4`` → image cube; payload is the flattened
  ``(N_grid * N_grid,)`` complex slice.

In both cases one frame is sent per ``(dm_idx, t_idx)`` tile (so a
cube of shape ``(N_DM=4, n_fv=64, ...)`` produces ``256`` frames).

# Sequence numbers

``TransportTx`` keeps an internal ``self._seq`` counter, scoped to one
``(host, port, chgroup)`` flow. Each frame transmitted bumps it by one.
The counter persists across :meth:`transmit` calls (production search
side relies on monotonic ordering for drop accounting). After a
service restart the counter resets to 0; the receiver tracks per-
``chgroup`` sequence and treats any seq < last_seq as a wrap or fresh
TX restart.

# Reliability semantics

UDP fire-and-forget; **drops on send failure are LOGGED + COUNTED but
not retried**. Production §4.3 specifies the same semantics — a TCP-
style retransmit would couple a slow receiver back into the corr-side
gridder, exactly what this transport layer is designed to avoid.
"""

from __future__ import annotations

import logging
import socket
from typing import Final

import numpy as np
import torch

from dsart.transport.frame import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DTYPE_CFP16,
    DTYPE_CINT8,
    FLAG_RFI_WARMING_UP,
    FastVisFrame,
    FramePayloadOversizeError,
)


LOG = logging.getLogger("dsart.transport.tx")


# Default payload dtype. cfp16 mirrors plan §4.3 ``bits_per_cell=32``
# (fp16 re + fp16 im = 32 bits per complex cell). cint8 is the
# operational §9 ops-table dtype (16 bits per cell), but requires a
# scale+offset that the chunk-8 frame doesn't yet carry — deferred to
# M4a's 72-byte production header.
DEFAULT_DTYPE_CODE: Final[int] = DTYPE_CFP16


class TransportTx:
    """UDP unicast/loopback transmitter for fast-vis cubes.

    Args:
        host: destination IP. Loopback bench uses ``127.0.0.1``;
            production uses each search node's ``nic.search`` IP from
            ``configs/host_phase.yaml``.
        port: destination UDP port. Production: ``9000 + chgroup``
            (per plan §4.3); chunk-8 loopback bench uses
            ``49500 + os.getpid() % 100``.
        chgroup: this TX's chgroup id (0..15) — written into every
            frame's ``chgroup`` field.
        max_payload_bytes: per-frame payload cap. Default 65000 fits
            comfortably inside the 65507-byte UDP loopback MTU.
        dtype_code: payload encoding (:data:`DTYPE_CFP16` (default) or
            :data:`DTYPE_CINT8`). For cint8 the cube is normalised by
            ``cint8_scale`` (default 127.0 / max(|re|, |im|)) before
            quantisation.
        cint8_scale: optional fixed scale factor applied before
            quantisation when ``dtype_code == DTYPE_CINT8``. ``None``
            means per-frame max-absolute scaling. (Chunk-8 doesn't
            transmit the scale; M4a will via the production header.)
    """

    def __init__(
        self,
        host: str,
        port: int,
        chgroup: int,
        *,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        dtype_code: int = DEFAULT_DTYPE_CODE,
        cint8_scale: float | None = None,
    ) -> None:
        if not (0 <= chgroup <= 0xFF):
            raise ValueError(f"chgroup={chgroup} out of u8 range")
        if dtype_code not in (DTYPE_CFP16, DTYPE_CINT8):
            raise ValueError(
                f"dtype_code={dtype_code} must be {DTYPE_CFP16} (cfp16) "
                f"or {DTYPE_CINT8} (cint8)"
            )
        self.host = host
        self.port = int(port)
        self.chgroup = int(chgroup)
        self.max_payload_bytes = int(max_payload_bytes)
        self.dtype_code = int(dtype_code)
        self.cint8_scale = cint8_scale

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Loopback / single-host: large enough sndbuf to absorb a burst
        # of cubes without `sendto` returning EAGAIN.
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024,
            )
        except OSError:
            LOG.warning("could not raise SO_SNDBUF; loopback bursts may EAGAIN")

        self._addr = (self.host, self.port)
        self._seq: int = 0
        self.n_send_fail: int = 0
        self.n_payload_oversize: int = 0
        self.n_sent: int = 0
        self.bytes_sent: int = 0

    # ---- Public API ------------------------------------------------------

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "TransportTx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def next_seq(self) -> int:
        """Next seq number that :meth:`transmit` would emit. Tests
        introspect this to verify monotonicity across calls."""
        return self._seq

    def reset_seq(self, *, to: int = 0) -> None:
        """Force-set the seq counter. Used by tests to inject seq gaps;
        production never calls this."""
        self._seq = int(to)

    # ---- Stage Protocol --------------------------------------------------

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
    ) -> int:
        """Satisfy :class:`TransportTxStage.transmit` Protocol.

        Iterates each cube's ``(dm_idx, t_idx)`` tile, packs the
        complex slice as cfp16/cint8 wire bytes, wraps in a
        :class:`FastVisFrame`, and ``sendto``s the destination. On
        send failure logs + counts; does NOT retry.

        Args:
            cubes_for_tx: list of per-block cubes from the Stage-2
                FIFO. Each cube is either ``(N_DM, n_fast_vis,
                N_filled)`` (sparse-COO; chunk-4 output today) or
                ``(N_DM, n_fast_vis, N_grid, N_grid)`` (image cube;
                forward-compatible). The first axis MAY be size 0
                (no cubes ready); :meth:`transmit` no-ops on those.
            block_n: chunk-4 block counter (logged for diagnostics;
                NOT carried in the frame in chunk 8 — the chunk-4
                ``IntegrationOutput.block_n`` is the canonical
                book-keeper).
            rfi_warming_up: when ``True`` the
                :data:`FLAG_RFI_WARMING_UP` bit is set on every frame
                emitted by this call.

        Returns:
            Number of frames actually sent (== expected_total -
            send-failure count).

        Notes:
            Per the chunk-4 Protocol the return value is "count of
            cubes sent". Chunk 8 returns the **frame** count instead
            (= cubes × N_DM × n_fast_vis), because that's what
            production §4.3 mon-keys watch (``tx_payloads_sent``,
            not ``tx_cubes_sent``). The chunk-4 orchestrator
            aggregates this into ``IntegrationOutput.n_tx`` and the
            ``corr_fast_integration.run`` summary's ``n_tx_total``,
            both of which are documented as "frames sent" downstream.
        """
        if not cubes_for_tx:
            return 0

        flags = FLAG_RFI_WARMING_UP if rfi_warming_up else 0
        n_sent_this_call = 0

        for cube in cubes_for_tx:
            n_sent_this_call += self._transmit_one_cube(
                cube, flags=flags, block_n=block_n,
            )
        return n_sent_this_call

    # ---- Internals -------------------------------------------------------

    def _transmit_one_cube(
        self,
        cube: torch.Tensor,
        *,
        flags: int,
        block_n: int,
    ) -> int:
        if not isinstance(cube, torch.Tensor):
            raise TypeError(
                f"TransportTx.transmit: each cube must be torch.Tensor; "
                f"got {type(cube).__name__}"
            )
        if not cube.is_complex():
            raise TypeError(
                f"TransportTx.transmit: cube must be complex; got "
                f"dtype={cube.dtype}"
            )

        if cube.ndim == 3:
            # Sparse-COO: (N_DM, n_fv, N_filled). One payload per (dm, t).
            n_dm, n_fv, _n_filled = cube.shape
            n_grid = 0                                                   # encoded as 0 for sparse-COO
            mode = "sparse"
        elif cube.ndim == 4:
            # Image cube: (N_DM, n_fv, N_grid, N_grid). One payload
            # per (dm, t) — the flattened (N_grid * N_grid,) slice.
            n_dm, n_fv, n_grid_a, n_grid_b = cube.shape
            if n_grid_a != n_grid_b:
                raise ValueError(
                    f"TransportTx.transmit: image cube must be square "
                    f"in last two axes; got {(n_grid_a, n_grid_b)}"
                )
            n_grid = int(n_grid_a)
            mode = "image"
        else:
            raise ValueError(
                f"TransportTx.transmit: cube ndim={cube.ndim} must be 3 "
                f"(sparse-COO (N_DM, n_fv, N_filled)) or 4 (image "
                f"(N_DM, n_fv, N_grid, N_grid))"
            )

        if n_dm > 0xFF or n_fv > 0xFFFF:
            raise ValueError(
                f"TransportTx.transmit: cube shape ({n_dm}, {n_fv}, ...) "
                f"exceeds dm_idx u8 / t_idx u16 range"
            )
        if n_grid > 0xFFFF:
            raise ValueError(
                f"TransportTx.transmit: n_grid={n_grid} exceeds u16 range"
            )

        cube_cpu = cube.detach().to("cpu", copy=False).contiguous()
        n_emitted = 0
        for dm_idx in range(n_dm):
            for t_idx in range(n_fv):
                if mode == "sparse":
                    slice_complex = cube_cpu[dm_idx, t_idx]              # (N_filled,) complex
                else:
                    slice_complex = cube_cpu[dm_idx, t_idx].reshape(-1)  # (N_grid*N_grid,) complex
                payload = self._pack_payload(slice_complex)
                if len(payload) > self.max_payload_bytes:
                    self.n_payload_oversize += 1
                    LOG.error(
                        "TX payload oversize: %d > %d (block_n=%d, "
                        "chgroup=%d, dm_idx=%d, t_idx=%d); dropping",
                        len(payload), self.max_payload_bytes, block_n,
                        self.chgroup, dm_idx, t_idx,
                    )
                    continue
                frame = FastVisFrame(
                    seq=self._seq,
                    chgroup=self.chgroup,
                    dm_idx=int(dm_idx),
                    t_idx=int(t_idx),
                    n_grid=int(n_grid),
                    dtype_code=self.dtype_code,
                    flags=int(flags),
                    payload=payload,
                )
                self._seq = (self._seq + 1) & 0xFFFF_FFFF
                if self._send_one(frame):
                    n_emitted += 1
        return n_emitted

    def _send_one(self, frame: FastVisFrame) -> bool:
        try:
            wire = frame.pack(max_payload_bytes=self.max_payload_bytes)
        except FramePayloadOversizeError:
            self.n_payload_oversize += 1
            return False
        try:
            self._sock.sendto(wire, self._addr)
        except OSError as exc:
            self.n_send_fail += 1
            LOG.warning(
                "TX sendto(%s) failed: %s (seq=%d chgroup=%d dm_idx=%d "
                "t_idx=%d); dropping (UDP fire-and-forget per §4.3)",
                self._addr, exc, frame.seq, frame.chgroup, frame.dm_idx,
                frame.t_idx,
            )
            return False
        self.n_sent += 1
        self.bytes_sent += len(wire)
        return True

    def _pack_payload(self, slice_complex: torch.Tensor) -> bytes:
        """Encode one (N,) complex slice → wire payload bytes per
        ``self.dtype_code``."""
        # complex64 view as (N, 2) float32 — re/im interleaved.
        c = slice_complex.detach().to(torch.complex64, copy=False).contiguous()
        re_im = torch.view_as_real(c)                                    # (N, 2) float32
        re_im_np = re_im.numpy()                                         # zero-copy when contiguous

        if self.dtype_code == DTYPE_CFP16:
            return re_im_np.astype(np.float16).tobytes()
        if self.dtype_code == DTYPE_CINT8:
            scale = self.cint8_scale
            if scale is None:
                amax = float(np.abs(re_im_np).max(initial=0.0))
                scale = 127.0 / max(amax, 1e-12)
            qf = np.clip(re_im_np * scale, -128.0, 127.0)
            return qf.astype(np.int8).tobytes()
        raise AssertionError(f"unhandled dtype_code={self.dtype_code}")
