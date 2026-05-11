"""``TransportTx`` — fast-vis cube transmit (M3 chunk 8 + M4a chunk 2).

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

# M4a chunk-2 production path

When ``use_prod_frame=True``, :meth:`transmit` routes each cube to
:meth:`_transmit_one_cube_prod` which emits production 72-byte
ProdFrame headers with per-payload ``scale``/``offset`` quantisation,
MTU-aware fragmentation, and a per-``dm_idx`` token-bucket pacer.
The chunk-8 path is preserved for the M3 loopback bench.
"""

from __future__ import annotations

import collections
import logging
import socket
import time
from dataclasses import dataclass, field
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
from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    BITS_CFP16_COMPLEX,
    DEFAULT_MAX_FRAG_PAYLOAD_BYTES,
    FLAG_LAST_IN_BLOCK,
    FLAG_QUANTIZED,
    FLAG_RFI_WARMING_UP as PROD_FLAG_RFI_WARMING_UP,
    VALID_BITS_PER_CELL,
    VALID_T_INT_FACTORS,
    ProdFrameHeader,
    pack_frame,
    split_payload_into_fragments,
)


LOG = logging.getLogger("dsart.transport.tx")


# Default payload dtype. cfp16 mirrors plan §4.3 ``bits_per_cell=32``
# (fp16 re + fp16 im = 32 bits per complex cell). cint8 is the
# operational §9 ops-table dtype (16 bits per cell), but requires a
# scale+offset that the chunk-8 frame doesn't yet carry — deferred to
# M4a's 72-byte production header.
DEFAULT_DTYPE_CODE: Final[int] = DTYPE_CFP16


# ---------------------------------------------------------------------------
# M4a chunk-2: TransportTxProdConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransportTxProdConfig:
    """M4a chunk-2 production-path config.

    Mirrors the chunk-8 :class:`TransportTx` knobs, plus production-
    specific fields for the 72-byte ProdFrame header path.

    Args:
        target_gbps_per_flow: token-bucket fill rate in Gbps per DM-idx
            flow. The pacer token rate is
            ``target_gbps_per_flow * 1e9 / 8 * pacer_headroom``
            bytes/sec (plan §4.3 line 1447).
        pacer_headroom: 5% headroom multiplier (default 1.05, per plan
            §4.3 line 1447). Applied on top of ``target_gbps_per_flow``
            so the token refill slightly over-provisions the link rate,
            giving the burst capacity to absorb short bursts without
            drops.
        max_frag_payload_bytes: per-fragment payload cap (bytes). The
            total wire frame is header (72 B) + fragment payload; this
            should be ≤ MTU - 28 B (IPv4 + UDP headers). Default
            :data:`~dsart.transport.prod_frame.DEFAULT_MAX_FRAG_PAYLOAD_BYTES`
            (= 8964 B for 9000 B jumbo MTU).
        t_int_factor: time-integration factor applied upstream.
            Must be in
            :data:`~dsart.transport.prod_frame.VALID_T_INT_FACTORS`.
        bits_per_cell: wire encoding — 16 for cint8 complex (operational
            default per plan §9) or 32 for cfp16 complex (debug / wider
            dynamic range).
        corr_idx: corr-node index 0..N_CORR-1; written into mon-key
            paths ``/mon/corr/<corr_idx>/transport/*``.
        bucket_fifo_depth: maximum number of pending fragments in each
            per-dm_idx FIFO queue. Drop-oldest semantics when full
            (plan §4.3 line 1447).
    """

    target_gbps_per_flow: float
    pacer_headroom: float = 1.05
    max_frag_payload_bytes: int = DEFAULT_MAX_FRAG_PAYLOAD_BYTES
    t_int_factor: int = 1
    bits_per_cell: int = BITS_CINT8_COMPLEX
    corr_idx: int = 0
    bucket_fifo_depth: int = 4

    def __post_init__(self) -> None:
        if self.target_gbps_per_flow < 0.0:
            raise ValueError(
                f"target_gbps_per_flow={self.target_gbps_per_flow} must be >= 0"
            )
        if self.pacer_headroom < 1.0:
            raise ValueError(
                f"pacer_headroom={self.pacer_headroom} must be >= 1.0"
            )
        if self.t_int_factor not in VALID_T_INT_FACTORS:
            raise ValueError(
                f"t_int_factor={self.t_int_factor} not in {VALID_T_INT_FACTORS}"
            )
        if self.bits_per_cell not in VALID_BITS_PER_CELL:
            raise ValueError(
                f"bits_per_cell={self.bits_per_cell} not in {VALID_BITS_PER_CELL}"
            )


# ---------------------------------------------------------------------------
# M4a chunk-2: _TokenBucket — per-flow drop-oldest pacer
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Per-flow drop-oldest token-bucket rate limiter (plan §4.3 line 1447).

    Token rate = ``rate_bytes_per_sec`` bytes/sec (set by the caller to
    ``target_gbps_per_flow * 1e9 / 8 * pacer_headroom``).

    Bucket capacity = ``capacity_bytes`` (set to
    ``max_frag_payload_bytes * 4`` for a 4-fragment burst).

    On each :meth:`try_send` call:

    1. Refill tokens from wall-clock elapsed time (clamped to capacity).
    2. Flush any previously queued fragments (FIFO order) as long as
       there are sufficient tokens.
    3. If balance >= ``len(wire_bytes)``: debit + send immediately;
       return ``True``.
    4. Else: bucket is exhausted.

       * If the FIFO queue is full (``len(queue) >= max_fifo``): pop
         the oldest item from the queue and increment :attr:`drop_count`
         (drop-oldest semantics; the new item takes the oldest slot).
       * Append the new item to the FIFO for a future flush.
       * Return ``False`` (not sent yet).

    Calling code MUST NOT block on a ``False`` return.

    Args:
        rate_bytes_per_sec: token refill rate in bytes per second.
        capacity_bytes: maximum token balance (burst capacity).
        max_fifo: maximum items in the pending-send FIFO. When full,
            the oldest is dropped (never blocked).
    """

    __slots__ = (
        "_rate",
        "_capacity",
        "_balance",
        "_last_ns",
        "_fifo",
        "_max_fifo",
        "drop_count",
    )

    def __init__(
        self,
        rate_bytes_per_sec: float,
        capacity_bytes: int,
        *,
        max_fifo: int = 4,
    ) -> None:
        self._rate: float = max(0.0, float(rate_bytes_per_sec))
        self._capacity: float = float(max(1, capacity_bytes))
        self._balance: float = float(capacity_bytes)  # start full
        self._last_ns: int = time.monotonic_ns()
        self._max_fifo: int = max(1, max_fifo)
        # Each entry is (wire_bytes, sock, addr).
        self._fifo: collections.deque = collections.deque()
        self.drop_count: int = 0

    # ------------------------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic_ns()
        elapsed_s = (now - self._last_ns) * 1e-9
        self._last_ns = now
        self._balance = min(
            self._capacity,
            self._balance + self._rate * elapsed_s,
        )

    def _flush_fifo(self) -> None:
        """Send queued items while there are sufficient tokens."""
        while self._fifo:
            item_bytes, item_sock, item_addr = self._fifo[0]
            n = len(item_bytes)
            if self._balance < n:
                break
            self._balance -= n
            self._fifo.popleft()
            try:
                item_sock.sendto(item_bytes, item_addr)
            except OSError:
                pass  # UDP fire-and-forget; log at call-site level

    def try_send(
        self,
        wire_bytes: bytes,
        sock: socket.socket,
        addr: tuple,
    ) -> bool:
        """Attempt to send ``wire_bytes`` under rate-limiting.

        Returns ``True`` if the bytes were sent immediately; ``False``
        if they were queued (or dropped due to a full FIFO). Never
        blocks.
        """
        self._refill()
        self._flush_fifo()
        frag_len = len(wire_bytes)

        if self._balance >= frag_len:
            self._balance -= frag_len
            try:
                sock.sendto(wire_bytes, addr)
            except OSError:
                pass
            return True

        # Bucket exhausted: drop oldest if FIFO is full, then queue.
        if len(self._fifo) >= self._max_fifo:
            self._fifo.popleft()
            self.drop_count += 1

        self._fifo.append((wire_bytes, sock, addr))
        return False


# ---------------------------------------------------------------------------
# TransportTx (chunk 8 + M4a chunk-2 extension)
# ---------------------------------------------------------------------------


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
        max_payload_bytes: per-frame payload cap for the chunk-8 path.
            Default 65000 fits comfortably inside the 65507-byte UDP
            loopback MTU.
        dtype_code: payload encoding (:data:`DTYPE_CFP16` (default) or
            :data:`DTYPE_CINT8`) for the chunk-8 path. For cint8 the
            cube is normalised by ``cint8_scale`` before quantisation.
        cint8_scale: optional fixed scale factor applied before
            quantisation when ``dtype_code == DTYPE_CINT8``. ``None``
            means per-frame max-absolute scaling. (Chunk-8 doesn't
            transmit the scale; M4a will via the production header.)
        use_prod_frame: when ``True``, :meth:`transmit` routes to
            :meth:`_transmit_one_cube_prod` (72-byte ProdFrame headers
            + fragmentation + token-bucket pacer). Default ``False``
            preserves the chunk-8 behaviour.
        prod_config: required when ``use_prod_frame=True``; ignored
            otherwise. Carries token-bucket knobs + ``bits_per_cell`` +
            ``t_int_factor`` for the production path.
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
        use_prod_frame: bool = False,
        prod_config: TransportTxProdConfig | None = None,
    ) -> None:
        if not (0 <= chgroup <= 0xFF):
            raise ValueError(f"chgroup={chgroup} out of u8 range")
        if dtype_code not in (DTYPE_CFP16, DTYPE_CINT8):
            raise ValueError(
                f"dtype_code={dtype_code} must be {DTYPE_CFP16} (cfp16) "
                f"or {DTYPE_CINT8} (cint8)"
            )
        if use_prod_frame and prod_config is None:
            raise ValueError(
                "prod_config required when use_prod_frame=True"
            )

        self.host = host
        self.port = int(port)
        self.chgroup = int(chgroup)
        self.max_payload_bytes = int(max_payload_bytes)
        self.dtype_code = int(dtype_code)
        self.cint8_scale = cint8_scale
        self.use_prod_frame = use_prod_frame
        self.prod_config: TransportTxProdConfig | None = prod_config

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
        # Chunk-8 seq counter (single per-TX counter):
        self._seq: int = 0
        # Chunk-8 stats:
        self.n_send_fail: int = 0
        self.n_payload_oversize: int = 0
        self.n_sent: int = 0
        self.bytes_sent: int = 0

        # ---- M4a chunk-2 prod-frame state ----
        # per-dm_idx seq counters (monotone u64)
        self._seq_by_flow: dict[int, int] = {}
        # per-dm_idx token buckets (lazily created in _get_bucket)
        self._bucket_by_flow: dict[int, _TokenBucket] = {}
        # pattern_id cache populated at prepare_prod() time
        self._pattern_id_by_chgroup: dict[int, int] = {}
        self._n_grid: int = 0

        # M4a mon-key counters (in-process; drained by tools.mon_key_emitter)
        # /mon/corr/<n>/transport/tx_dropped_payloads
        self.tx_dropped_payloads: int = 0
        # /mon/corr/<n>/transport/cube_seq_emitted (last emitted seq)
        self.cube_seq_emitted: int = 0

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
        """Next seq number that :meth:`transmit` would emit (chunk-8 path).
        Tests introspect this to verify monotonicity across calls."""
        return self._seq

    def reset_seq(self, *, to: int = 0) -> None:
        """Force-set the chunk-8 seq counter. Used by tests to inject seq
        gaps; production never calls this."""
        self._seq = int(to)

    def prepare_prod(
        self,
        pattern_id_by_chgroup: dict[int, int],
        n_grid: int,
    ) -> None:
        """Set prod-frame state at ``cmd: prepare`` time.

        Caches the per-chgroup ``pattern_id`` values (precomputed from
        :func:`~dsart.transport.prod_frame.predict_pattern_id`) and the
        grid side length. Must be called before
        :meth:`_transmit_one_cube_prod` (or :meth:`transmit` with
        ``use_prod_frame=True``).

        Also resets per-dm_idx seq counters and token buckets so that
        a fresh ``cmd: prepare`` starts clean (plan §M4a env-var reload
        semantics).

        Args:
            pattern_id_by_chgroup: mapping ``{chgroup: pattern_id}``
                for this corr node. The TX picks up
                ``pattern_id_by_chgroup[self.chgroup]`` on each send.
            n_grid: grid side length (e.g. 256 at default ops).
        """
        if n_grid <= 0:
            raise ValueError(f"n_grid={n_grid} must be > 0")
        self._pattern_id_by_chgroup = dict(pattern_id_by_chgroup)
        self._n_grid = int(n_grid)
        # Reset per-flow state so a re-prepare starts clean.
        self._seq_by_flow.clear()
        self._bucket_by_flow.clear()
        # Reset prod mon-key counters on prepare (matches existing TX
        # lifecycle: stats reset on service restart / re-prepare).
        self.tx_dropped_payloads = 0
        self.cube_seq_emitted = 0

    # ---- Stage Protocol --------------------------------------------------

    def transmit(
        self,
        cubes_for_tx: list[torch.Tensor],
        *,
        block_n: int,
        rfi_warming_up: bool,
        specnum: int | None = None,
    ) -> int:
        """Satisfy :class:`TransportTxStage.transmit` Protocol.

        Iterates each cube's ``(dm_idx, t_idx)`` tile, packs the
        complex slice, and ``sendto``s the destination. On send failure
        logs + counts; does NOT retry.

        When ``use_prod_frame=False`` (default), uses the chunk-8
        :class:`~dsart.transport.frame.FastVisFrame` 32-byte path.
        When ``use_prod_frame=True``, uses the M4a production 72-byte
        ProdFrame path.

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
            rfi_warming_up: when ``True`` the RFI warm-up flag is set
                on every frame emitted by this call.
            specnum: SNAP block-start counter. Required when
                ``use_prod_frame=True``; ignored for the chunk-8 path.

        Returns:
            Number of frames actually sent.

        Raises:
            NotImplementedError: if ``use_prod_frame=True`` and
                ``specnum is None``.
        """
        if not cubes_for_tx:
            return 0

        if self.use_prod_frame:
            if specnum is None:
                raise NotImplementedError(
                    "TransportTx.transmit: specnum=None but "
                    "use_prod_frame=True. The production ProdFrame path "
                    "requires the SNAP block-start specnum (uint64 F-engine "
                    "packet counter at the start of this block). Wire it from "
                    "the corr-side block metadata upstream of this call. "
                    "F-item logged in M4a_PLAN_FIXES.md."
                )
            flags = PROD_FLAG_RFI_WARMING_UP if rfi_warming_up else 0
            n_sent_this_call = 0
            for cube in cubes_for_tx:
                n_sent_this_call += self._transmit_one_cube_prod(
                    cube, specnum=specnum, flags=flags,
                )
            return n_sent_this_call

        # Chunk-8 path (default).
        flags = FLAG_RFI_WARMING_UP if rfi_warming_up else 0
        n_sent_this_call = 0
        for cube in cubes_for_tx:
            n_sent_this_call += self._transmit_one_cube(
                cube, flags=flags, block_n=block_n,
            )
        return n_sent_this_call

    # ---- Chunk-8 internals -----------------------------------------------

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

    # ---- M4a chunk-2: prod-frame internals ------------------------------

    def _get_bucket(self, dm_idx: int) -> _TokenBucket:
        """Lazily create and return the token bucket for ``dm_idx``."""
        if dm_idx not in self._bucket_by_flow:
            cfg = self.prod_config
            assert cfg is not None  # guaranteed by __init__ check
            rate = cfg.target_gbps_per_flow * 1e9 / 8.0 * cfg.pacer_headroom
            capacity = cfg.max_frag_payload_bytes * 4
            self._bucket_by_flow[dm_idx] = _TokenBucket(
                rate,
                capacity,
                max_fifo=cfg.bucket_fifo_depth,
            )
        return self._bucket_by_flow[dm_idx]

    def _next_seq(self, dm_idx: int) -> int:
        """Return and bump the per-dm_idx seq counter."""
        seq = self._seq_by_flow.get(dm_idx, 0)
        self._seq_by_flow[dm_idx] = seq + 1
        return seq

    @staticmethod
    def _compute_scale_offset(
        cells_re_im: np.ndarray,
    ) -> tuple[np.float32, np.float32]:
        """Compute cint8 scale/offset over filled cells only.

        Plan §4.2 pin: "dynamic range tracks actual data, not zeros."
        Inputs are the complex-valued filled cells as a ``(N_filled, 2)``
        float32 array (re in column 0, im in column 1). Outputs are
        float32 for wire-encoding accuracy — plan §4.3 field is f32.

        Returns:
            (scale, offset) where the dequantisation rule is
            ``x = scale * q + offset``. For symmetric cint8:
            ``offset = 0``, ``scale = amax / 127``.
        """
        amax = float(np.abs(cells_re_im).max(initial=0.0))
        if amax == 0.0:
            return np.float32(1.0), np.float32(0.0)
        scale = np.float32(amax / 127.0)
        return scale, np.float32(0.0)

    @staticmethod
    def _encode_payload(
        cells_complex: np.ndarray,
        bits_per_cell: int,
        scale: np.float32,
    ) -> bytes:
        """Encode filled cells to wire bytes.

        Args:
            cells_complex: ``(N_filled,)`` complex64 array of filled
                cell values.
            bits_per_cell: 16 for cint8, 32 for cfp16.
            scale: dequant scale (only used for cint8; cfp16 ignores).

        Returns:
            Wire payload bytes: ``N_filled * bits_per_cell // 8`` bytes.
        """
        re_im = np.stack(
            [cells_complex.real, cells_complex.imag], axis=1,
        ).astype(np.float32)                                             # (N, 2) f32

        if bits_per_cell == BITS_CINT8_COMPLEX:
            # Quantise to int8: q = clip(round(x / scale), -128, 127).
            inv_scale = float(1.0 / scale) if float(scale) != 0.0 else 1.0
            q = np.clip(np.round(re_im * inv_scale), -128.0, 127.0)
            return q.astype(np.int8).tobytes()

        if bits_per_cell == BITS_CFP16_COMPLEX:
            return re_im.astype(np.float16).tobytes()

        raise AssertionError(f"unhandled bits_per_cell={bits_per_cell}")

    def _transmit_one_cube_prod(
        self,
        cube: torch.Tensor,
        *,
        specnum: int,
        flags: int,
    ) -> int:
        """Emit production 72-byte ProdFrame datagrams for one cube.

        For each ``(dm_idx, t_idx)`` tile in ``cube``:

        1. Extract the ``(N_filled,)`` complex slice (sparse-COO path).
        2. Compute ``scale`` / ``offset`` over filled cells only.
        3. Encode to cint8 or cfp16 bytes.
        4. Fragment via :func:`split_payload_into_fragments`.
        5. For each fragment: build :class:`ProdFrameHeader`, pack, and
           pass through the per-dm_idx token bucket.
        6. Update ``tx_dropped_payloads`` and ``cube_seq_emitted``
           mon-key counters.

        The cube must be a sparse-COO ``(N_DM, n_fv, N_filled)``
        complex torch.Tensor. Image cubes (ndim=4) are not currently
        supported by the prod-frame path (NotImplementedError).

        Args:
            cube: ``(N_DM, n_fv, N_filled)`` complex torch.Tensor.
            specnum: SNAP block-start counter (uint64 F-engine packet
                seq); cross-corr time alignment depends on this.
            flags: prod-frame flags bitfield (RFI warm-up, etc.);
                ``FLAG_LAST_IN_BLOCK`` is set internally by this method.

        Returns:
            Number of fragments actually sent (== total fragments -
            dropped by pacer).

        Raises:
            NotImplementedError: if ``cube.ndim != 3`` (image-cube
                support deferred to chunk 7 bench integration) or if
                ``prepare_prod`` has not been called (``_n_grid == 0``).
        """
        if not isinstance(cube, torch.Tensor):
            raise TypeError(
                f"_transmit_one_cube_prod: expected torch.Tensor, "
                f"got {type(cube).__name__}"
            )
        if not cube.is_complex():
            raise TypeError(
                f"_transmit_one_cube_prod: cube must be complex, "
                f"got dtype={cube.dtype}"
            )
        if cube.ndim != 3:
            raise NotImplementedError(
                f"_transmit_one_cube_prod: cube.ndim={cube.ndim} "
                f"(expected 3 for sparse-COO (N_DM, n_fv, N_filled)); "
                f"image-cube (ndim=4) support is deferred to the chunk-7 "
                f"net-loopback bench. F-item logged in M4a_PLAN_FIXES.md."
            )
        if self._n_grid == 0:
            raise NotImplementedError(
                "_transmit_one_cube_prod: n_grid=0 — call prepare_prod() "
                "with the SparsityPattern n_grid and pattern_id_by_chgroup "
                "before transmitting. F-item logged in M4a_PLAN_FIXES.md."
            )

        cfg = self.prod_config
        assert cfg is not None

        chgroup = self.chgroup
        if chgroup not in self._pattern_id_by_chgroup:
            raise NotImplementedError(
                f"_transmit_one_cube_prod: pattern_id not cached for "
                f"chgroup={chgroup}. Call prepare_prod(pattern_id_by_chgroup, "
                f"n_grid) before transmitting. F-item logged in "
                f"M4a_PLAN_FIXES.md."
            )

        pattern_id = self._pattern_id_by_chgroup[chgroup]
        n_grid = self._n_grid
        bits_per_cell = cfg.bits_per_cell
        t_int_factor = cfg.t_int_factor
        max_frag = cfg.max_frag_payload_bytes

        n_dm, n_fv, n_filled = cube.shape
        cube_cpu = cube.detach().to("cpu", copy=False).contiguous()

        # D1 (M4a chunk-2): FLAG_QUANTIZED is set only for cint8; for
        # cfp16 the payload is the "no-quantisation" fp16 path and
        # FLAG_QUANTIZED = 0. Plan §4.3 line 1414 defines bit0 as
        # "quantized cint8 (vs cfp16)". The mission's spec of always
        # setting FLAG_QUANTIZED is a shorthand for the cint8 case.
        base_flags = flags
        if bits_per_cell == BITS_CINT8_COMPLEX:
            base_flags |= FLAG_QUANTIZED

        n_frags_sent = 0

        for dm_idx in range(n_dm):
            bucket = self._get_bucket(dm_idx)
            # Allocate seq before we know if any fragment will be sent;
            # seq advances even if all fragments are dropped (per plan
            # §4.3 line 1421 monotone contract — the receiver uses seq
            # gaps for drop accounting, not for replay).
            seq = self._next_seq(dm_idx)
            payload_dropped = False

            for t_idx in range(n_fv):
                slice_np = (
                    cube_cpu[dm_idx, t_idx]
                    .numpy()
                    .astype(np.complex64, copy=False)
                )                                                        # (N_filled,) complex64

                # scale/offset over filled cells only (plan §4.2 pin).
                re_im_f32 = np.stack(
                    [slice_np.real, slice_np.imag], axis=1,
                )                                                        # (N_filled, 2) f32
                if bits_per_cell == BITS_CINT8_COMPLEX:
                    # Dynamic range tracks actual filled-cell data.
                    scale, offset = self._compute_scale_offset(re_im_f32)
                else:
                    # cfp16: identity dequant (D1: FLAG_QUANTIZED=0).
                    scale = np.float32(1.0)
                    offset = np.float32(0.0)

                payload_bytes = self._encode_payload(
                    slice_np, bits_per_cell, scale,
                )

                frags = split_payload_into_fragments(
                    payload_bytes,
                    max_frag_payload_bytes=max_frag,
                )
                n_frags = len(frags)

                for frag_idx, frag in enumerate(frags):
                    last_frag = (frag_idx == n_frags - 1)
                    frag_flags = base_flags
                    if last_frag:
                        frag_flags |= FLAG_LAST_IN_BLOCK

                    hdr = ProdFrameHeader(
                        seq=seq,
                        specnum=int(specnum),
                        chgroup=chgroup,
                        dm_idx=int(dm_idx),
                        frag_idx=frag_idx,
                        n_frags=n_frags,
                        n_grid=n_grid,
                        n_filled=n_filled,
                        pattern_id=pattern_id,
                        bits_per_cell=bits_per_cell,
                        t_int_factor=t_int_factor,
                        scale=float(scale),
                        offset=float(offset),
                        payload_bytes_in_frag=len(frag),
                        flags=frag_flags,
                    )
                    wire = pack_frame(hdr, frag)

                    if bucket.try_send(wire, self._sock, self._addr):
                        n_frags_sent += 1
                    else:
                        # Pacer dropped (or queued for later); count
                        # the PAYLOAD (not fragment) as dropped once.
                        if not payload_dropped:
                            payload_dropped = True
                            self.tx_dropped_payloads += 1

            # Update cube_seq_emitted mon-key with last seq for this
            # dm_idx. The mon-key tracks the highest seq emitted.
            if seq > self.cube_seq_emitted:
                self.cube_seq_emitted = seq

        return n_frags_sent
