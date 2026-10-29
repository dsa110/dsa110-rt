#!/usr/bin/env python3
"""x2 line-rate fada replay: recorded voltage blocks -> a (TEST) DADA buffer, no SNAPs.

``bench/replay_voltage_dump.py`` reads every block from the file and copies it with one
thread: 245 ms per block on a corr node (measured on n04, 2026-10-28) -- 1.8x too slow for
real time (one block per 134.218 ms). This writer preloads the fixture's blocks into RAM once
(15 blocks = 4.5 GB) and fills each DADA page with a multi-threaded copy (numpy releases the
GIL), so it runs well under the block period and G1 can drive the whole corr -> search chain
in real time from recordings.

    python bench/timeres/x2_fada_replay.py --file .../250924mptq_sb01_data.out --key d2f0 \\
        --n-blocks 600 --rate native [--threads 8] [--dec-deg 53.85]

Blocks cycle through the file (block b = file block b % n). The fada header follows
``bench/replay_voltage_dump.build_fada_header``. Prints one JSON summary line at the end:
blocks written, elapsed, period, per-block fill time (mean / p99 / max) and late blocks
(rate native: a block written more than one period after its slot).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FADA_BYTES = 301989888
BLOCK_S = 4096 * 32.768e-6


def _header(dec_deg: float, label: str) -> dict:
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from replay_voltage_dump import build_fada_header  # noqa: PLC0415
    return build_fada_header(source_label=label, dec_deg=dec_deg,
                             extra={"X2_REPLAY": "1", "X2_REPLAY_SOURCE": label})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True)
    ap.add_argument("--key", default="d2f0")
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--rate", default="native", choices=("native", "fast"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dec-deg", type=float, default=0.0)
    ap.add_argument("--wait-for", default=None,
                    help="start writing only once this file exists (a consumer's ready sentinel)")
    a = ap.parse_args(argv)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    from psrdada import Writer
    t_load = time.perf_counter()
    blocks = np.fromfile(a.file, dtype=np.uint8)
    if blocks.size % FADA_BYTES:
        raise SystemExit(f"{a.file}: {blocks.size} B is not a whole number of fada blocks")
    blocks = blocks.reshape(-1, FADA_BYTES)
    load_s = time.perf_counter() - t_load
    n_file = blocks.shape[0]
    n_thr = max(1, int(a.threads))
    edges = np.linspace(0, FADA_BYTES, n_thr + 1).astype(np.int64)
    pool = ThreadPoolExecutor(max_workers=n_thr)
    writer = Writer(int(a.key, 16))
    writer.setHeader(_header(a.dec_deg, os.path.basename(a.file)))
    # the header goes out at once (a reader blocks on it before warming up); data waits for
    # the consumer's ready sentinel, as the production capture routines do
    if a.wait_for:
        while not os.path.exists(a.wait_for) and not stop["flag"]:
            time.sleep(0.05)
    fill_ms, late = [], 0
    t0 = None
    n = 0
    try:
        for page in writer:
            if n >= a.n_blocks or stop["flag"]:
                writer.markEndOfData()
                break
            if t0 is None:
                t0 = time.perf_counter()
            dst = np.asarray(page)
            src = blocks[n % n_file]
            ts = time.perf_counter()
            list(pool.map(lambda i: np.copyto(dst[edges[i]:edges[i + 1]], src[edges[i]:edges[i + 1]]),
                          range(n_thr)))
            writer.markFilled()
            te = time.perf_counter()
            fill_ms.append((te - ts) * 1e3)
            n += 1
            if a.rate == "native":
                target = t0 + n * BLOCK_S
                if te - target > BLOCK_S:
                    late += 1
                if target > te:
                    time.sleep(target - te)
    finally:
        try:
            writer.disconnect()
        except Exception:  # noqa: BLE001
            pass
        pool.shutdown()
    el = time.perf_counter() - (t0 or time.perf_counter())
    f = np.asarray(fill_ms) if fill_ms else np.zeros(1)
    print(json.dumps({"x2_fada_replay": True, "file": a.file, "key": a.key, "rate": a.rate,
                      "threads": n_thr, "n_blocks": n, "elapsed_s": round(el, 3),
                      "ms_per_block": round(1e3 * el / max(1, n), 3), "fill_ms_mean": round(float(f.mean()), 3),
                      "fill_ms_p99": round(float(np.percentile(f, 99)), 3), "fill_ms_max": round(float(f.max()), 3),
                      "late_blocks": late, "preload_s": round(load_s, 2), "file_blocks": n_file}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
