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
            # M7.2 G4 observability: expose per-(chgroup, coarse-DM)
            # ingress counters so operators can distinguish
            # intentionally-empty cells (remain zero under ownership mask)
            # from unexpected silence in cells that should be active.
            matrix = self._rx.ring_data_present_matrix(
                n_corr=self.n_corr,
                n_coarse_dm=self.n_coarse_dm,
            )
            LOG.info(
                "rx_ingress_by_corr_dm: n_corr=%d n_coarse_dm=%d data_present=%s",
                self.n_corr,
                self.n_coarse_dm,
                matrix,
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
# x2: ring v4 mode (frame v2; SPEC §5/§6). The v3 service above is unchanged.
# ---------------------------------------------------------------------------


def build_search_rx_mon_key(cn: int) -> str:
    """``/mon/search_rt/<cn>/rx`` (WS-D monitoring, D11)."""
    return f"/mon/search_rt/{int(cn)}/rx"


class RxMonPublisher:
    """Small ``/mon/search_rt/<cn>/rx`` payload at a few Hz: rx_gbps,
    datagrams/s, row loss fraction, rejected frames, stalled streams.
    ``store`` is a DsaStore-like object (``put_dict``); created lazily from
    dsautils when None. An etcd failure never stops the RX."""

    def __init__(self, cn: int, *, store=None) -> None:
        self.cn = int(cn)
        self.key = build_search_rx_mon_key(cn)
        self._store = store
        self._prev: Optional[dict] = None
        self.n_published = 0
        self.n_errors = 0

    @staticmethod
    def snapshot(rx: RxEpoll, reader) -> dict:
        c = rx.counters()
        rc = reader.counters()
        w = rc["writer"]
        rows = sum(d["writer"]["rows_written"] for d in rc["per_dm"].values())
        holes = sum(d["writer"]["rows_hole"] for d in rc["per_dm"].values())
        rejected = sum(v for k, v in w.items()
                       if k not in ("datagrams", "ok", "bytes", "sweeps", "pattern_mismatch"))
        return {"t": time.monotonic(), "bytes": c.bytes_received_total,
                "datagrams": c.n_received, "rows": rows, "holes": holes,
                "rejected": rejected, "pattern_mismatch": w["pattern_mismatch"],
                "heartbeat_age_s": rc["heartbeat_age_s"]}

    def payload(self, snap: dict, stalled_streams: int) -> Optional[dict]:
        prev, self._prev = self._prev, snap
        if prev is None:
            return None
        dt = max(1e-9, snap["t"] - prev["t"])
        d_rows = snap["rows"] - prev["rows"]
        d_holes = snap["holes"] - prev["holes"]
        return {
            "rx_gbps": round((snap["bytes"] - prev["bytes"]) * 8 / dt / 1e9, 4),
            "rx_datagrams_per_s": round((snap["datagrams"] - prev["datagrams"]) / dt, 1),
            "loss_frac": round(d_holes / (d_rows + d_holes), 7) if d_rows + d_holes else 0.0,
            "rejected_per_s": round((snap["rejected"] - prev["rejected"]) / dt, 2),
            "pattern_mismatch_per_s": round(
                (snap["pattern_mismatch"] - prev["pattern_mismatch"]) / dt, 2),
            "stalled_streams": int(stalled_streams),
            "heartbeat_age_s": round(float(snap["heartbeat_age_s"]), 3),
            "ring": "v4",
            "ts_wall_unix": round(time.time(), 3),
        }

    def publish(self, payload: dict) -> bool:
        try:
            if self._store is None:
                from dsautils.dsa_store import DsaStore
                self._store = DsaStore()
            self._store.put_dict(self.key, payload)
            self.n_published += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self.n_errors += 1
            if self.n_errors == 1:
                LOG.warning("rx mon publish to %s failed: %s (further failures silent)",
                            self.key, exc)
            return False


def _load_stream_pattern_ids(path: Path) -> Dict[int, int]:
    """Per-STREAM pattern ids (v4): list (index = stream) or {"<stream>": pid}."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {i: int(p) for i, p in enumerate(raw)}
    return {int(str(k).replace("stream_", "")): int(v) for k, v in raw.items()}


class SearchRxV4Service:
    """search_rx in ring v4 mode: frame-v2 UDP -> ring v4 (owned DMs only)."""

    def __init__(self, *, bind_host: str, port_base: int, n_ports: int, shm_name: str,
                 spec, so_rcvbuf_bytes: int, n_threads: int = 2, strict_ports: bool = True,
                 prefault: bool = True, pattern_ids: Optional[Dict[int, int]] = None,
                 status_interval_s: float = 5.0, mon_cn: Optional[int] = None,
                 mon_interval_s: float = 0.5, mon_store=None,
                 stall_warn_s: float = 3.0, unlink_shm_on_exit: bool = True) -> None:
        self.bind_host = bind_host
        self.port_base = int(port_base)
        self.n_ports = int(n_ports)
        self.shm_name = shm_name
        self.spec = spec
        self.so_rcvbuf_bytes = int(so_rcvbuf_bytes)
        self.n_threads = int(n_threads)
        self.strict_ports = bool(strict_ports)
        self.prefault = bool(prefault)
        self.pattern_ids = dict(pattern_ids or {})
        self.status_interval_s = float(status_interval_s)
        self.mon = RxMonPublisher(mon_cn, store=mon_store) if mon_cn is not None else None
        self.mon_interval_s = float(mon_interval_s)
        self.stall_warn_s = float(stall_warn_s)
        self.unlink_shm_on_exit = bool(unlink_shm_on_exit)
        self._rx: Optional[RxEpoll] = None
        self._reader = None
        self._stopping = False

    def setup(self) -> None:
        from ..transport.recv_ring import unlink_ring_v4
        rx = RxEpoll.open(bind_host=self.bind_host, bind_port=self.port_base,
                          so_rcvbuf_bytes=self.so_rcvbuf_bytes)
        for i in range(1, self.n_ports):
            rx.add_port(bind_host=self.bind_host, bind_port=self.port_base + i,
                        so_rcvbuf_bytes=self.so_rcvbuf_bytes)
        try:
            rx.attach_ring4(shm_name=self.shm_name, spec=self.spec, prefault=self.prefault,
                            n_threads=self.n_threads, strict_ports=self.strict_ports)
        except Exception:
            rx.close()
            unlink_ring_v4(self.shm_name)
            raise
        for s, pid in self.pattern_ids.items():
            rx.set_expected_pattern_id4(s, pid)
        LOG.info("search_rx v4: %d ports from %s:%d, %d threads, ring %s: dms=%s tif=%s "
                 "t_buf=%s n_stream=%d row_bytes=%d (%.2f GB), %d expected pattern ids",
                 rx.n_sockets, self.bind_host, self.port_base, self.n_threads, self.shm_name,
                 list(self.spec.dms), list(self.spec.t_int_factor), list(self.spec.t_buf),
                 self.spec.n_stream, self.spec.row_bytes_eff, self.spec.size_bytes / 1e9,
                 len(self.pattern_ids))
        if not self.pattern_ids:
            LOG.warning("search_rx v4: no expected pattern ids; every value map is accepted")
        self._rx = rx
        self._reader = rx.ring4_reader(access="pread")

    def start(self) -> None:
        assert self._rx is not None, "call setup() first"
        self._rx.start()

    def request_stop(self) -> None:
        self._stopping = True

    def status_once(self) -> dict:
        """One status record (also what the periodic log line prints)."""
        r = self._reader
        rep = r.stall_report(stall_s=self.stall_warn_s)
        stalled = sum(len(v["stalled"]) for v in rep["dms"].values())
        edges = {d.dm: r.live_edge(d.dm) for d in r.dms()}
        return {"counters": r.counters(), "stalled_streams": stalled,
                "stall_report": rep, "live_edge": edges}

    def run_status_loop(self) -> None:
        assert self._rx is not None
        next_status = time.monotonic() + self.status_interval_s
        next_mon = time.monotonic()
        while not self._stopping:
            time.sleep(0.05)
            now = time.monotonic()
            if self.mon is not None and now >= next_mon:
                next_mon = now + self.mon_interval_s
                rep = self._reader.stall_report(stall_s=self.stall_warn_s)
                stalled = sum(len(v["stalled"]) for v in rep["dms"].values())
                p = self.mon.payload(RxMonPublisher.snapshot(self._rx, self._reader), stalled)
                if p is not None:
                    self.mon.publish(p)
            if now >= next_status:
                next_status = now + self.status_interval_s
                st = self.status_once()
                w = st["counters"]["writer"]
                LOG.info("rx4_status: %s", json.dumps({
                    k: w[k] for k in ("datagrams", "ok", "bad_version", "v1_frames", "not_owned",
                                      "format_mismatch", "stream_mismatch", "pattern_mismatch",
                                      "port_mismatch", "tier_mismatch", "bad_length")}))
                per_dm = {d: {k: v["writer"][k] for k in ("rows_written", "rows_hole", "rows_late",
                                                           "resyncs", "streams_started")}
                          for d, v in st["counters"]["per_dm"].items()}
                LOG.info("rx4_dm: live_edge=%s per_dm=%s", st["live_edge"], per_dm)
                if st["stalled_streams"]:
                    LOG.warning("rx4_stall: %d stream(s) without progress > %.1f s: %s",
                                st["stalled_streams"], self.stall_warn_s,
                                {d: v["stalled"] for d, v in st["stall_report"]["dms"].items()
                                 if v["stalled"]})

    def shutdown(self) -> None:
        from ..transport.recv_ring import unlink_ring_v4
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._rx is not None:
            try:
                self._rx.stop()
            finally:
                self._rx.close()
                self._rx = None
        if self.unlink_shm_on_exit:
            unlink_ring_v4(self.shm_name)


def _spec_from_args(args):
    """``(RingV4Spec, {stream: expected pattern_id} or None)``: from ``--ring4-spec`` JSON
    (+ ``--pattern-id-file``), or built in-process from ``--plan`` + ``--opoint`` +
    ``--node-index`` + ``--antpos-path`` + ``--obs-dec-deg`` exactly as corr_fast and the
    owner engine build their value maps (``dsart.x2.prod_inputs``)."""
    from ..transport.recv_ring import RingV4Spec
    if args.ring4_spec is not None:
        if args.plan is not None:
            raise SystemExit("--ring4-spec and --plan are exclusive")
        spec = RingV4Spec.from_dict(json.loads(args.ring4_spec.read_text()))
        pids = None
        if args.pattern_id_file is not None:
            pids = _load_stream_pattern_ids(args.pattern_id_file)
        return spec, pids
    if args.plan is not None:
        from ..x2 import prod_inputs as pi
        from ..x2.opoint import opoint_from_id
        need = {"--opoint": args.opoint, "--node-index": args.node_index,
                "--antpos-path": args.antpos_path, "--obs-dec-deg": args.obs_dec_deg}
        missing = [k for k, v in need.items() if v is None]
        if missing:
            raise SystemExit(f"--ring-version 4 --plan also needs {' '.join(missing)}")
        if args.pattern_id_file is not None:
            raise SystemExit("--pattern-id-file with --plan: the ids come from the value map")
        if args.mon_cn is not None and pi.node_index_of_cn(args.mon_cn) != args.node_index:
            raise SystemExit(f"--node-index {args.node_index} is not search node cn "
                             f"{args.mon_cn} (index {pi.node_index_of_cn(args.mon_cn)})")
        op = opoint_from_id(args.opoint)
        plan = pi.load_plan(args.plan, op, args.plan_id)
        spec, vm, pids = pi.node_ring4(op, plan, node_index=args.node_index,
                                       antpos_path=args.antpos_path,
                                       obs_dec_deg=args.obs_dec_deg,
                                       retention_s=args.retention_s)
        LOG.info("search_rx v4: %s node %d, plan 0x%016x, value map 0x%016x (DEC %.4f -> "
                 "%.2f), %d streams", op.opoint_id, args.node_index, int(plan.plan_id),
                 int(vm.pattern_id), float(args.obs_dec_deg), vm.dec_deg_quant, vm.n_streams)
        return spec, pids
    raise SystemExit("--ring-version 4 needs --plan (with --opoint --node-index --antpos-path "
                     "--obs-dec-deg) or --ring4-spec (see `search_rx make-ring4-spec`)")


def make_ring4_spec(argv: Optional[List[str]] = None) -> int:
    """``search_rx make-ring4-spec``: the v4 ring spec JSON of one search node
    from WS-P's plan + value map (the production cal blob or an antenna
    positions npz with antpos_e/n + station, e.g.
    tests/fixtures/x2/dsa110_antpos.npz), built by ``dsart.x2.prod_inputs`` like
    search_rx's in-process path."""
    from ..x2 import prod_inputs as pi
    from ..x2.opoint import opoint_from_id
    from ..x2.plan import TieredDmPlan

    p = argparse.ArgumentParser(prog="search_rx make-ring4-spec")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--opoint", required=True, help="e.g. 2x-gridded-ns2-cint4")
    p.add_argument("--node-index", type=int, required=True, help="0..3 in N_SEARCH order")
    p.add_argument("--antpos-npz", "--antpos-path", dest="antpos_npz", type=Path, required=True,
                   help="cal blob (antennas.out) or antpos npz")
    p.add_argument("--dec-deg", type=float, required=True)
    p.add_argument("--retention-s", type=float, default=20.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern-ids-out", type=Path, default=None)
    a = p.parse_args(argv)
    op = opoint_from_id(a.opoint)
    plan = TieredDmPlan.from_npz(a.plan)
    # SPEC §3 production inputs (antpos_u = None, the canonical DEC), shared with corr and the engine
    vm = pi.value_map_for(op, a.antpos_npz, a.dec_deg)
    spec = pi.ring4_spec(plan, vm, op, node_index=a.node_index, retention_s=a.retention_s)
    a.out.write_text(json.dumps(spec.to_dict(), indent=1))
    if a.pattern_ids_out is not None:
        a.pattern_ids_out.write_text(json.dumps([int(x) for x in vm.stream_pattern_ids]))
    print(json.dumps({"dms": list(spec.dms), "size_gb": spec.size_bytes / 1e9,
                      "pattern_id": f"0x{vm.pattern_id:016x}"}))
    return 0


def main_v4(args) -> int:
    spec, pids = _spec_from_args(args)
    svc = SearchRxV4Service(
        bind_host=args.bind_host, port_base=args.port_base, n_ports=args.n_ports,
        shm_name=args.shm_name, spec=spec,
        so_rcvbuf_bytes=args.so_rcvbuf_bytes, n_threads=args.rx_threads,
        strict_ports=not args.no_strict_ports, prefault=not args.no_prefault,
        pattern_ids=pids, status_interval_s=args.status_interval_s, mon_cn=args.mon_cn,
        mon_interval_s=args.mon_interval_s, unlink_shm_on_exit=not args.keep_shm_on_exit)
    _install_signals(svc)
    rc = 0
    try:
        svc.setup()
        svc.start()
        svc.run_status_loop()
    except KeyboardInterrupt:
        LOG.info("KeyboardInterrupt; shutting down…")
    except Exception:  # noqa: BLE001
        LOG.exception("search_rx v4 crashed")
        rc = 1
    finally:
        svc.shutdown()
    return rc


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "make-ring4-spec":
        return make_ring4_spec(argv[1:])
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
    # x2 ring v4 mode (frame v2). Default stays the v3 ring.
    p.add_argument("--ring-version", type=int, default=3, choices=(3, 4),
                   help="3 = today's ring (frame v1); 4 = x2 ring v4 (frame v2, owned DMs "
                        "only; --n-corr/--n-coarse-dm/--t-buf-samples/--n-filled ignored)")
    p.add_argument("--ring4-spec", type=Path, default=None,
                   help="v4: RingV4Spec JSON (``search_rx make-ring4-spec``)")
    # v4 in-process spec (production, SPEC §6): plan + op-point + node + the value map's
    # inputs; the expected per-stream pattern ids come from the same value map
    p.add_argument("--opoint", default=None,
                   help="op-point id (2x-<kind>-ns<n>-cint<b>; 2x needs --ring-version 4)")
    p.add_argument("--plan", type=Path, default=None, help="v4: tiered DM plan v3 npz")
    p.add_argument("--plan-id", default=None,
                   help="v4: refuse a plan whose plan_id differs (the set's ${plan_id})")
    p.add_argument("--node-index", type=int, default=None, choices=(0, 1, 2, 3),
                   help="v4: search node index in SEARCH_NODES order (n01 n02 n09 n13)")
    p.add_argument("--antpos-path", default=None,
                   help="v4: cal blob (antennas.out, as corr_fast --apply-cal) or antpos npz")
    p.add_argument("--obs-dec-deg", type=float, default=None,
                   help="v4: the start verb's DEC (CUSTOMDEC; enters pattern_id)")
    p.add_argument("--retention-s", type=float, default=20.0,
                   help="v4: minimum ring retention per DM (s)")
    p.add_argument("--rx-threads", type=int, default=2,
                   help="v4: receive threads (socket i -> thread i %% N)")
    p.add_argument("--no-strict-ports", action="store_true",
                   help="v4 with 1 thread: accept a stream on any port")
    p.add_argument("--no-prefault", action="store_true",
                   help="v4: do not touch every ring page at create")
    p.add_argument("--mon-cn", type=int, default=None,
                   help="v4: publish /mon/search_rt/<cn>/rx (rx_gbps, loss_frac, ...)")
    p.add_argument("--mon-interval-s", type=float, default=0.5)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.ring_version == 4:
        return main_v4(args)
    if args.opoint not in (None, "1x"):
        LOG.error("--opoint %s needs --ring-version 4", args.opoint)
        return 2

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
