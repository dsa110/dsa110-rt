"""``TransportRx`` — fast-vis cube receive (M3 chunk 8).

Loopback / unicast UDP receiver for fast-vis cubes; the chunk-8 peer
to :class:`dsart.transport.tx.TransportTx`. Validates magic + CRC,
tracks per-``chgroup`` sequence gaps for drop accounting (mirrors plan
§4.3 ``rx_seq_gap_count`` mon-key), and optionally captures payloads
to disk for the loopback-bench `.cfp16` set.

# Sequence-gap detection

The transmitter keeps a per-``(host, port, chgroup)`` strictly-
monotonic ``seq``. On the receive side we track ``next_expected_seq``
per ``chgroup`` and count any incoming ``seq != next_expected``:

* ``seq == next_expected``  → in-order; bump expected.
* ``seq > next_expected``   → gap; ``n_seq_gaps += seq -
  next_expected`` (count of MISSING seq values, not skipped frames),
  then ``next_expected = seq + 1``.
* ``seq < next_expected``   → out-of-order or TX restart. Counted as
  ``n_out_of_order``; expected unchanged. (Loopback never reorders
  in the kernel; this branch is for safety.)

Per-``chgroup`` because production has 16 chgroups multiplexed onto
one search-side process; chunk-8 only operates on a single chgroup
per RX instance, but the dict-keyed counter is forward-compatible.

# Capture

:meth:`recv_into_capture` writes each incoming payload to
``capture_dir/seq_<seq:08d>_chg<g>_dm<d>_t<t>.<ext>`` and a
``meta.json`` index at the end. Used by the loopback bench + by the
M3 voltage-fixture sub-DoDs (chunks 5/6) once they wire transport-TX.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from dsart.transport.frame import (
    DTYPE_CFP16,
    DTYPE_CINT8,
    HEADER_BYTES,
    FastVisFrame,
    FrameCRCError,
    FrameMagicError,
)


LOG = logging.getLogger("dsart.transport.rx")


# Max UDP datagram on a 64 KiB-MTU loopback (65535 − 28).
_MAX_UDP_DATAGRAM: int = 65507


@dataclass
class RxStats:
    """Per-RX-instance counters. Mirrors plan §4.3 mon-key set."""

    n_received: int = 0
    n_crc_fail: int = 0
    n_magic_fail: int = 0
    n_seq_gaps: int = 0
    n_out_of_order: int = 0
    bytes_received: int = 0
    # Per-chgroup state: chgroup → next_expected_seq.
    _next_seq: dict[int, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "n_received": self.n_received,
            "n_crc_fail": self.n_crc_fail,
            "n_magic_fail": self.n_magic_fail,
            "n_seq_gaps": self.n_seq_gaps,
            "n_out_of_order": self.n_out_of_order,
            "bytes_received": self.bytes_received,
        }


class TransportRx:
    """UDP receiver for fast-vis cubes.

    Args:
        host: bind IP. Loopback bench: ``127.0.0.1``. ``0.0.0.0`` for
            multi-interface listeners (chunk 8 doesn't use this).
        port: bind UDP port. Pass ``0`` to let the kernel pick an
            ephemeral free port (used by acceptance tests). After
            ``__init__`` the assigned port is in :attr:`port`.
        recv_timeout_s: ``socket.SO_RCVTIMEO``. ``receive_one``
            returns ``None`` rather than raising on timeout.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        recv_timeout_s: float = 1.0,
    ) -> None:
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Production tunes SO_RCVBUF to 256 MiB (plan §4.3); for
        # loopback / acceptance tests, 8 MiB is plenty.
        try:
            self._sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024,
            )
        except OSError:
            LOG.warning("could not raise SO_RCVBUF; bursts may drop")
        self._sock.bind((host, int(port)))
        bound_host, bound_port = self._sock.getsockname()
        self.port: int = int(bound_port)
        self.host_actual: str = str(bound_host)
        self.recv_timeout_s = float(recv_timeout_s)
        self._sock.settimeout(self.recv_timeout_s)
        self.stats = RxStats()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "TransportRx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---- Public API ------------------------------------------------------

    def receive_one(self) -> FastVisFrame | None:
        """Receive ONE frame off the socket.

        Returns:
            The decoded :class:`FastVisFrame` on success; ``None`` if
            the socket timed out (no frame within
            :attr:`recv_timeout_s`).

        Raises:
            FrameMagicError: incoming buffer doesn't carry the expected
                magic. Counters are bumped before raise so production
                can choose to swallow this in a non-strict capture loop.
            FrameCRCError: CRC mismatch. Same handling as above.
            ValueError: malformed / truncated buffer.

        On any of these errors the per-(host, port) stats are still
        updated (``n_magic_fail``, ``n_crc_fail``).
        """
        try:
            buf, _addr = self._sock.recvfrom(_MAX_UDP_DATAGRAM)
        except socket.timeout:
            return None
        if len(buf) < HEADER_BYTES:
            self.stats.n_magic_fail += 1
            raise ValueError(
                f"TransportRx.receive_one: short buffer: {len(buf)} bytes "
                f"< {HEADER_BYTES}"
            )
        try:
            frame = FastVisFrame.unpack(buf)
        except FrameMagicError:
            self.stats.n_magic_fail += 1
            raise
        except FrameCRCError:
            self.stats.n_crc_fail += 1
            raise
        # Stats / seq accounting (only for valid frames).
        self.stats.n_received += 1
        self.stats.bytes_received += len(buf)
        self._update_seq_accounting(frame)
        return frame

    def recv_into_capture(
        self,
        capture_dir: Path,
        max_frames: int,
        *,
        progress_every: int = 0,
    ) -> dict[str, int]:
        """Receive up to ``max_frames`` frames; persist each payload
        to disk + write a ``meta.json`` index. Returns the
        :class:`RxStats` dict.

        Output layout:

            capture_dir/
              seq_<seq:08d>_chg<g>_dm<d>_t<t>.<ext>     (one per frame)
              meta.json                                 (final index)

        Where ``<ext>`` is ``cfp16`` for ``dtype_code=0`` or ``cint8``
        for ``dtype_code=1``. The capture dir is created if missing.

        Args:
            capture_dir: target dir.
            max_frames: stop after this many *valid* frames received
                (not counting CRC / magic failures).
            progress_every: log every N frames (0 = silent).
        """
        capture_dir = Path(capture_dir)
        capture_dir.mkdir(parents=True, exist_ok=True)
        index: list[dict] = []
        t0 = time.monotonic()
        while self.stats.n_received < max_frames:
            try:
                frame = self.receive_one()
            except (FrameMagicError, FrameCRCError, ValueError) as exc:
                LOG.warning("RX frame rejected: %s", exc)
                continue
            if frame is None:
                continue                                                 # timeout; loop again
            ext = self._dtype_ext(frame.dtype_code)
            fname = (
                f"seq_{frame.seq:08d}_chg{frame.chgroup}_"
                f"dm{frame.dm_idx}_t{frame.t_idx:04d}.{ext}"
            )
            (capture_dir / fname).write_bytes(frame.payload)
            entry = frame.to_dict()
            entry["filename"] = fname
            index.append(entry)
            if progress_every and self.stats.n_received % progress_every == 0:
                LOG.info(
                    "rx progress: n=%d gaps=%d crc_fail=%d",
                    self.stats.n_received, self.stats.n_seq_gaps,
                    self.stats.n_crc_fail,
                )
        elapsed = time.monotonic() - t0
        meta = {
            "stats": self.stats.to_dict(),
            "frames": index,
            "elapsed_s": elapsed,
            "max_frames": max_frames,
        }
        (capture_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return self.stats.to_dict()

    # ---- Internals -------------------------------------------------------

    def _update_seq_accounting(self, frame: FastVisFrame) -> None:
        next_seq = self.stats._next_seq.get(frame.chgroup)
        if next_seq is None:
            # First frame on this chgroup: anchor.
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
            return
        if frame.seq == next_seq:
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
        elif frame.seq > next_seq:
            self.stats.n_seq_gaps += frame.seq - next_seq
            self.stats._next_seq[frame.chgroup] = (frame.seq + 1) & 0xFFFF_FFFF
        else:
            # frame.seq < next_seq → out-of-order or TX restart.
            self.stats.n_out_of_order += 1

    @staticmethod
    def _dtype_ext(dtype_code: int) -> str:
        if dtype_code == DTYPE_CFP16:
            return "cfp16"
        if dtype_code == DTYPE_CINT8:
            return "cint8"
        return f"raw{dtype_code:02d}"
