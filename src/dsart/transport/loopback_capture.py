"""``LoopbackCaptureService`` — RX-side capture loop (M3 chunk 8).

Spawns a :class:`dsart.transport.rx.TransportRx`, runs it for
``max_frames`` frames, persists each payload + a ``meta.json`` index
under ``capture_dir/``. Used by:

* ``bench/transport_loopback.py`` (the chunk-8 bench).
* ``bench/voltage_fixture_fast_corr_continuum.py`` /
  ``...burst.py`` (chunks 5/6) once they wire transport-TX into the
  full M3 pipeline.

CLI:

    python -m dsart.transport.loopback_capture \\
        --host 127.0.0.1 --port 49555 \\
        --capture-dir /tmp/dsart-loopback \\
        --max-frames 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from dsart.transport.rx import TransportRx


LOG = logging.getLogger("dsart.transport.loopback_capture")


@dataclass
class LoopbackCaptureService:
    """Headless capture-and-persist service. One-shot — call
    :meth:`run` once and discard.

    Args:
        recv_timeout_s: per-``receive_one`` timeout. Loopback is
            fast; 1 s is plenty.
        progress_every: log every N frames received (0 = silent).
    """

    recv_timeout_s: float = 1.0
    progress_every: int = 16

    def run(
        self,
        host: str,
        port: int,
        capture_dir: Path,
        *,
        max_frames: int,
    ) -> dict[str, int]:
        """Spawn RX, capture ``max_frames`` frames, write index.

        Returns the :class:`RxStats` dict augmented with the bound
        host/port the RX actually picked (useful when ``port=0`` is
        passed to let the kernel choose).
        """
        capture_dir = Path(capture_dir)
        rx = TransportRx(host, port, recv_timeout_s=self.recv_timeout_s)
        try:
            LOG.info(
                "loopback capture: bound %s:%d → %s (max_frames=%d)",
                rx.host_actual, rx.port, capture_dir, max_frames,
            )
            stats = rx.recv_into_capture(
                capture_dir, max_frames=max_frames,
                progress_every=self.progress_every,
            )
        finally:
            rx.close()
        out = dict(stats)
        out["bound_host"] = rx.host_actual
        out["bound_port"] = rx.port
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind IP (default: %(default)s)")
    p.add_argument("--port", type=int, default=49555,
                   help="bind UDP port; 0 = ephemeral (default: %(default)d)")
    p.add_argument("--capture-dir", type=Path, required=True,
                   help="dir for per-frame payload files + meta.json")
    p.add_argument("--max-frames", type=int, default=100,
                   help="stop after N valid frames (default: %(default)d)")
    p.add_argument("--recv-timeout-s", type=float, default=1.0,
                   help="per-receive_one timeout (default: %(default)s)")
    p.add_argument("--progress-every", type=int, default=16,
                   help="log every N frames; 0 = silent (default: %(default)d)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    svc = LoopbackCaptureService(
        recv_timeout_s=args.recv_timeout_s,
        progress_every=args.progress_every,
    )
    out = svc.run(
        args.host, args.port, args.capture_dir, max_frames=args.max_frames,
    )
    LOG.info("capture done: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
