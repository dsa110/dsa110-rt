#!/usr/bin/env python3
"""C2 ingest throughput at 1x vs 2x C1 batch rates (x2 D6 replay benchmark).

Drives a private :class:`CoincidencerService` (offline name allocator, no
etcd, no sockets, no plots, a no-op broadcaster) with a synthetic C1 batch
stream and measures how long C2's per-batch pipeline (window, graph,
clustering, criteria, fire path) takes compared with the simulated time.

Stream model (per 201.33 ms cycle, aligned like the search engine):
  * 8 halves; each emits ``--n-dm-per-half`` batches per cycle (1x: 1,
    2x: 3 (n_sub 4) to 6 (n_sub 2) per-DM cubes);
  * each batch carries Poisson(``--rows``) noise rows (``--rows`` is the
    MEAN rows per batch; the C1 cap is 8 per block today) at random DMs /
    pixels; heartbeats (0-row batches) are included, as in production;
  * every ``--burst-every-s`` a bright burst lands on ``--burst-halves``
    halves with ``--burst-rows`` rows each (its DM response).

Pure Python (no torch / GPU imports); safe on h23's CPU at low priority::

    nice -n 19 python tools/c2/rate_bench.py --opoint 2x --n-dm-per-half 6 --seconds 60

Prints one JSON line: offered batches/s and rows/s, processing time per
batch (mean / p99), and ``realtime_factor`` = processing time / simulated
time (must stay well below 1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

CYCLE_S = 192 * 1048.576e-6          # 201.326592 ms: 1x cube hop = 2x cycle
HALVES = [(s, g) for s in (1, 2, 9, 13) for g in (0, 1)]


def _crit(path: Path) -> Path:
    p = path / "crit.yaml"
    p.write_text(
        "trigger_classes:\n"
        "  - name: bright_frb_extragalactic\n"
        "    require: {snr_max_min: 12.0, n_events_min: 1, dm_median_min_pc_cc: 115.0}\n"
        "    action: dump_all_gpus\n"
        "    holdoff_s: 30.0\n"
        "  - name: log_only\n"
        "    require: {n_events_min: 1}\n"
        "    action: log_only\n")
    return p


class _NullBc:
    def broadcast(self, **kw):
        return {sg: True for sg in HALVES}

    def send_to(self, blob, halves):
        return {tuple(h): True for h in halves}

    def close(self):
        pass


class _Store:
    def get_dict(self, key):
        return {"enabled": True, "ts": 0.0} if key.endswith("dumps_enabled") else None

    def put_dict(self, key, value):
        pass


def _make_service(tmp: Path, opoint: str):
    import dsart.services.coincidencer as c2mod
    from dsart.coinc.cutout_lifecycle import CutoutConfig
    from dsart.services.coincidencer import CoincidencerConfig, CoincidencerService

    c2mod.enqueue_event = lambda *a, **k: None            # no plotting
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1", bind_port=0, window_s=5.0,
        csv_dir_c1=tmp / "c1", csv_dir_c2=tmp / "c2",
        event_archive_root=tmp / "events", trigger_criteria_path=_crit(tmp),
        name_allocator_offline=True, gal_dm_poll_interval_s=60.0,
        startup_grace_s=0.0,
        dump_broadcast_hosts={1: "127.0.0.1", 2: "127.0.0.1",
                              9: "127.0.0.1", 13: "127.0.0.1"},
        cutout=CutoutConfig(enabled=(opoint == "2x")),
        archive_schema=2 if opoint == "2x" else 1,
        plotter_use_process_pool=False,
    )
    return CoincidencerService(config=cfg, mon_store=_Store(), broadcaster=_NullBc())


def _batches(args, rng: random.Random):
    from dsart.coinc import wire

    period = 524.288 if args.opoint == "2x" else 1048.576
    rows_per_cycle = round(CYCLE_S / (period * 1e-6))       # 384 or 192
    n_cycles = int(args.seconds / CYCLE_S)
    mjd0 = 61335.0
    cube_id = 0
    burst_every = max(1, int(args.burst_every_s / CYCLE_S)) if args.burst_every_s > 0 else 0
    for c in range(n_cycles):
        spec0 = 10_000_000 + c * rows_per_cycle
        mjd_c = mjd0 + c * CYCLE_S / 86400.0
        burst = burst_every and c % burst_every == burst_every // 2
        for h, (sid, g) in enumerate(HALVES):
            for k in range(args.n_dm_per_half):
                cube_id += 1
                n = _poisson(rng, args.rows)
                rows = []
                for _ in range(n):
                    t = rng.randrange(rows_per_cycle)
                    rows.append(wire.C1CandidateRow(
                        snr=9.6 + rng.expovariate(1.5), l_rad=rng.uniform(-0.05, 0.05),
                        m_rad=rng.uniform(-0.05, 0.05), l_pix=rng.randrange(256),
                        m_pix=rng.randrange(256), dm_pc_cc=rng.uniform(50, 3000),
                        dm_idx_global=rng.randrange(758), fine_dm_idx=rng.randrange(30),
                        event_specnum=spec0 + t, width_samples=rng.choice((1, 2, 4, 8, 16)),
                        kernel_id="unit:d1:b4", flags=0))
                if burst and h < args.burst_halves and k == 0:
                    for j in range(args.burst_rows):
                        rows.append(wire.C1CandidateRow(
                            snr=40.0 - j, l_rad=0.01, m_rad=0.02, l_pix=140, m_pix=97,
                            dm_pc_cc=812.0 + 3 * j, dm_idx_global=300 + j, fine_dm_idx=j,
                            event_specnum=spec0 + 100 + (j % 3), width_samples=2,
                            kernel_id="unit:d1:b2", flags=0))
                hdr = wire.build_header(
                    cube_id=cube_id, event_specnum_start=spec0, mjd_start=mjd_c,
                    sample_period_specnum=8 if args.opoint == "2x" else 16,
                    sample_period_us=period, n_grid=256, n_fdm_in_cube=17,
                    search_node_id=sid, gpu_half=g, n_candidates=len(rows))
                yield wire.C1Batch(header=hdr, candidates=tuple(rows))


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    # Knuth; lam is small (<= ~10)
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


async def _run(args) -> dict:
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="c2bench_") as td:
        svc = _make_service(Path(td), args.opoint)
        per = []
        n_rows = 0
        n_batches = 0
        t_all = time.perf_counter()
        for b in _batches(args, rng):
            t0 = time.perf_counter()
            await svc._on_batch(b, peer_repr="bench")
            per.append(time.perf_counter() - t0)
            n_rows += len(b.candidates)
            n_batches += 1
        total = time.perf_counter() - t_all
        sim = int(args.seconds / CYCLE_S) * CYCLE_S
        per.sort()
        return {
            "opoint": args.opoint, "n_dm_per_half": args.n_dm_per_half,
            "rows_mean_per_batch": args.rows, "simulated_s": round(sim, 2),
            "batches": n_batches, "rows": n_rows,
            "offered_batches_per_s": round(n_batches / sim, 1),
            "offered_rows_per_s": round(n_rows / sim, 1),
            "ms_per_batch_mean": round(1e3 * statistics.fmean(per), 3),
            "ms_per_batch_p99": round(1e3 * per[int(0.99 * (len(per) - 1))], 3),
            "processing_s": round(total, 2),
            "realtime_factor": round(total / sim, 3),
            "dumps": svc._counters.get("triggers_dump", 0),
            "components_evaluated": svc._counters.get("components_evaluated", 0),
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--opoint", choices=("1x", "2x"), default="2x")
    p.add_argument("--n-dm-per-half", type=int, default=None)
    p.add_argument("--rows", type=float, default=1.0, help="mean rows per batch")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--burst-every-s", type=float, default=10.0)
    p.add_argument("--burst-halves", type=int, default=4)
    p.add_argument("--burst-rows", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-nice", action="store_true",
                   help="do not lower this process to nice 19")
    args = p.parse_args(argv)
    if not args.no_nice:
        import os
        os.nice(19)          # shares h23 with the live C2/C3
    if args.n_dm_per_half is None:
        args.n_dm_per_half = 6 if args.opoint == "2x" else 1
    print(json.dumps(asyncio.run(_run(args))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
