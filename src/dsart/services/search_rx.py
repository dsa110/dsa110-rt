"""Search-node UDP receive + ring-publish service (M7.2).

Long-running process owned by a single search node. Binds N UDP sockets
(one per chgroup the search node is responsible for; production = all
16 chgroups on ports ``6625 + chgroup``), drains them via the C epoll
loop (:mod:`dsart.transport.recv_epoll`), and publishes reassembled
cube slots into a POSIX-shm ring (:mod:`dsart.transport.recv_ring`)
that the two ``search_compute`` halves attach as read-only consumers.

This module is the operator-facing CLI; the C-side primitives are
already covered by the transport unit tests
(``tests/transport/test_recv_epoll*.py``,
``tests/transport/test_recv_ring_spmc.py``). The Python wrapper here
adds: argparse, optional pattern_id table loading, etcd-friendly
status logging, and SIGTERM/SIGINT handling so the dsart_rt
orchestrator can clean-up via ``systemctl --user stop dsart-rt``.

Lifecycle:
    1. ``open`` first socket (primary port).
    2. ``add_port`` the remaining sockets (default: 15 more for the
       full chgroup 0..15 fan-in).
    3. ``attach_ring`` as ring owner (creates / zero-inits the shm
       segment; the two ``search_compute`` consumers attach
       read-only).
    4. Register expected pattern_ids per chgroup (from
       ``--pattern-id-file`` JSON if supplied; otherwise we run with
       no pattern check and every frame is accepted).
    5. ``start`` the C drainer pthread.
    6. Poll counters every ``--status-interval-s`` and log a single
       line of progress (so the dsart_rt mon-dict can scrape the
       most recent values).
    7. On SIGTERM/SIGINT: ``stop`` + ``close``; the ring shm is
       unlinked at exit so a re-start gets a fresh segment.

Production wire-format constants and topology references:
   * :mod:`dsart.transport.prod_frame` — 72-byte ProdFrame header.
   * ``tools/dod/corner_turn.sh`` — corr→search IP/port topology.
   * ``configs/dsart_search_rt.yaml`` — etcd-pushed routine config.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..transport.recv_epoll import RxEpoll
from ..transport.recv_ring import (
    BYTES_CFP16_COMPLEX,
    BYTES_CINT8_COMPLEX,
    RxRing,
)


LOG = logging.getLogger("dsart.services.search_rx")


# ---------------------------------------------------------------------------
# Pattern-ID table loader
# ---------------------------------------------------------------------------


def _load_pattern_ids(path: Path) -> Dict[int, int]:
    """Load a per-chgroup pattern_id table from a JSON file.

    File layout (one of):
        ``{"0": 12345, "1": 67890, ...}``  (str keys)
        ``{"chgroup_0": 12345, ...}``      (named keys)
        ``[12345, 67890, ...]``            (list, index = chgroup)
    """
    raw = json.loads(path.read_text())
    out: Dict[int, int] = {}
    if isinstance(raw, list):
        for chg, pid in enumerate(raw):
            out[int(chg)] = int(pid)
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            chg = int(k.replace("chgroup_", "").strip())
            out[chg] = int(v)
        return out
    raise ValueError(
        f"Unrecognised pattern_id file format at {path}: type={type(raw)}"
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SearchRxService:
    """Convenience wrapper around RxEpoll for the search-node fan-in role."""

    def __init__(
        self,
        *,
        bind_host: str,
        port_base: int,
        n_ports: int,
        shm_name: str,
        n_corr: int,
        n_coarse_dm: int,
        t_buf_samples: int,
        n_filled: int,
        bytes_per_cell: int,
        so_rcvbuf_bytes: int,
        pattern_ids: Optional[Dict[int, int]] = None,
        status_interval_s: float = 5.0,
        unlink_shm_on_exit: bool = True,
    ) -> None:
        self.bind_host = bind_host
        self.port_base = port_base
        self.n_ports = int(n_ports)
        self.shm_name = shm_name
        self.n_corr = int(n_corr)
        self.n_coarse_dm = int(n_coarse_dm)
        self.t_buf_samples = int(t_buf_samples)
        self.n_filled = int(n_filled)
        self.bytes_per_cell = int(bytes_per_cell)
        self.so_rcvbuf_bytes = int(so_rcvbuf_bytes)
        self.pattern_ids = dict(pattern_ids or {})
        self.status_interval_s = float(status_interval_s)
        self.unlink_shm_on_exit = bool(unlink_shm_on_exit)

        self._rx: Optional[RxEpoll] = None
        self._stopping = False

    def setup(self) -> None:
        if self._rx is not None:
            raise RuntimeError("SearchRxService already set up")
        rx = RxEpoll.open(
            bind_host=self.bind_host,
            bind_port=self.port_base,
            so_rcvbuf_bytes=self.so_rcvbuf_bytes,
        )
        LOG.info(
            "primary UDP socket bound: %s:%d (sockfd in C)",
            self.bind_host, rx.port,
        )
        for i in range(1, self.n_ports):
            port = self.port_base + i
            rx.add_port(
                bind_host=self.bind_host,
                bind_port=port,
                so_rcvbuf_bytes=self.so_rcvbuf_bytes,
            )
            LOG.info("additional UDP socket bound: %s:%d", self.bind_host, port)
        LOG.info("total bound sockets: %d", rx.n_sockets)

        rx.attach_ring(
            shm_name=self.shm_name,
            owner=True,
            n_corr=self.n_corr,
            n_coarse_dm=self.n_coarse_dm,
            t_buf_samples=self.t_buf_samples,
            n_filled=self.n_filled,
            bytes_per_cell=self.bytes_per_cell,
        )
        LOG.info(
            "ring attached: shm_name=%s dims=(n_corr=%d, n_coarse_dm=%d, "
            "t_buf=%d, n_filled=%d, bpc=%d)",
            self.shm_name, self.n_corr, self.n_coarse_dm,
            self.t_buf_samples, self.n_filled, self.bytes_per_cell,
        )

        for chg, pid in self.pattern_ids.items():
            rx.set_expected_pattern_id(chg, pid)
        if self.pattern_ids:
            LOG.info(
                "registered %d expected pattern_ids: %s",
                len(self.pattern_ids),
                {chg: hex(pid) for chg, pid in self.pattern_ids.items()},
            )
        else:
            LOG.warning(
                "no expected pattern_ids registered; every frame will be "
                "accepted (use --pattern-id-file in production)."
            )

        self._rx = rx

    def start(self) -> None:
        assert self._rx is not None, "call setup() first"
        self._rx.start()
        LOG.info("search-rx drainer pthread started")

    def request_stop(self) -> None:
        self._stopping = True

    def run_status_loop(self) -> None:
        """Block until ``request_stop`` is called, logging counter
        snapshots once per ``status_interval_s``. The C drainer is in
        its own pthread; this loop is purely operator-facing telemetry."""
        assert self._rx is not None
        next_status = time.monotonic() + self.status_interval_s
        last_received = 0
        last_committed = 0
        last_ring_writes = 0
        last_bytes = 0
        last_ts = time.monotonic()
        while not self._stopping:
            time.sleep(0.1)
            now = time.monotonic()
            if now < next_status:
                continue
            dt = max(now - last_ts, 1e-9)
            c = self._rx.counters()
            d_recv = c.n_received - last_received
            d_cmt = c.n_committed - last_committed
            d_ring = c.ring_slots_written - last_ring_writes
            d_bytes = c.bytes_received_total - last_bytes
            LOG.info(
                "rx_status: bound=%d ring=%s n_recv=%d (+%d, %.1f/s) "
                "n_cmt=%d (+%d, %.1f/s) ring_writes=%d (+%d, %.1f/s) "
                "bad_magic=%d bad_len=%d pid_mismatch=%d zerofill=%d "
                "ring_err=%d bytes=%d (+%.1f MB/s = %.2f Gb/s)",
                self._rx.n_sockets, self._rx.ring_attached,
                c.n_received, d_recv, d_recv / dt,
                c.n_committed, d_cmt, d_cmt / dt,
                c.ring_slots_written, d_ring, d_ring / dt,
                c.bad_magic_count, c.bad_length_count,
                c.pattern_mismatch_count,
                c.window_slide_zerofill_count,
                c.ring_write_error_count,
                c.bytes_received_total,
                d_bytes / dt / 1e6,
                d_bytes * 8 / dt / 1e9,
            )
            last_received = c.n_received
            last_committed = c.n_committed
            last_ring_writes = c.ring_slots_written
            last_bytes = c.bytes_received_total
            last_ts = now
            next_status = now + self.status_interval_s

    def shutdown(self) -> None:
        if self._rx is None:
            return
        try:
            self._rx.stop()
            LOG.info("drainer pthread joined")
        finally:
            try:
                self._rx.close()
                LOG.info("RxEpoll closed (sockets + ring released)")
            finally:
                if self.unlink_shm_on_exit:
                    try:
                        RxRing.unlink_name(self.shm_name)
                        LOG.info("unlinked shm %s", self.shm_name)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(
                            "shm unlink %s failed: %s (ignoring)",
                            self.shm_name, exc,
                        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _install_signals(svc: SearchRxService) -> None:
    def _handler(signum, _frame):
        LOG.info("signal %d received; requesting stop…", signum)
        svc.request_stop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bind-host", required=True,
                   help="IPv4 address to bind on. Production: the "
                        "search node's nic.search address (e.g. "
                        "10.41.0.205 for n01). Use 0.0.0.0 to bind all.")
    p.add_argument("--port-base", type=int, default=6625,
                   help="first listen port (default: 6625 = production "
                        "topology base; chgroup K → port 6625+K).")
    p.add_argument("--n-ports", type=int, default=16,
                   help="how many consecutive ports to bind starting at "
                        "--port-base. Default: 16 (full 0..15 chgroup "
                        "fan-in on production; pass 2 for smoke tests).")
    p.add_argument("--so-rcvbuf-bytes", type=int, default=256 * 1024 * 1024,
                   help="SO_RCVBUF size per socket (default: 256 MiB; "
                        "production value tuned for jumbo MTU bursts).")

    p.add_argument("--shm-name", required=True,
                   help="POSIX-shm name to create as ring owner "
                        "(e.g. /dsart-rxring-n01).")
    p.add_argument("--n-corr", type=int, default=16,
                   help="ring dim: number of correlator groups (default: 16)")
    p.add_argument("--n-coarse-dm", type=int, default=5,
                   help="ring dim: number of coarse DM slabs (default: 5 = "
                        "M7.1 op-point).")
    p.add_argument("--t-buf-samples", type=int, default=4096,
                   help="ring dim: time-axis depth in search-cadence "
                        "samples (default 4096 ≈ 16 cube cadences of "
                        "headroom). MUST be >= the slowest compute half's "
                        "drain cadence to avoid overrun.")
    p.add_argument("--n-filled", type=int, default=5000,
                   help="ring dim: cells per (corr, dm) slot.")
    p.add_argument("--bytes-per-cell", type=int, default=2, choices=(2, 4),
                   help="ring dim: 2=cint8 complex (prod default), "
                        "4=cfp16 complex (debug).")

    p.add_argument("--pattern-id-file", type=Path, default=None,
                   help="optional JSON file mapping chgroup→pattern_id "
                        "(list-of-ints OR {chgroup_K: pid}). When set, "
                        "frames with a mismatching pattern_id are published "
                        "as zero-payload slots with VF_PATTERN_MISMATCH "
                        "and the mismatch counter bumps.")
    p.add_argument("--status-interval-s", type=float, default=5.0,
                   help="how often to log counter snapshots (default 5s).")
    p.add_argument("--keep-shm-on-exit", action="store_true",
                   help="don't shm_unlink the segment on exit (useful "
                        "for post-mortem of a crashed compute half).")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    bpc = BYTES_CINT8_COMPLEX if args.bytes_per_cell == 2 else BYTES_CFP16_COMPLEX

    pids: Optional[Dict[int, int]] = None
    if args.pattern_id_file is not None:
        if not args.pattern_id_file.exists():
            LOG.error("pattern_id_file %s not found", args.pattern_id_file)
            return 2
        pids = _load_pattern_ids(args.pattern_id_file)

    svc = SearchRxService(
        bind_host=args.bind_host,
        port_base=args.port_base,
        n_ports=args.n_ports,
        shm_name=args.shm_name,
        n_corr=args.n_corr,
        n_coarse_dm=args.n_coarse_dm,
        t_buf_samples=args.t_buf_samples,
        n_filled=args.n_filled,
        bytes_per_cell=bpc,
        so_rcvbuf_bytes=args.so_rcvbuf_bytes,
        pattern_ids=pids,
        status_interval_s=args.status_interval_s,
        unlink_shm_on_exit=not args.keep_shm_on_exit,
    )
    _install_signals(svc)
    rc = 0
    try:
        svc.setup()
        svc.start()
        svc.run_status_loop()
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt; shutting down…")
    except Exception:  # noqa: BLE001
        LOG.exception("search_rx crashed")
        rc = 1
    finally:
        svc.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
