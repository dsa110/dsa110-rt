"""Shared helpers for the time-resolution benchmark suite (bench/timeres).

Run ONLY in an observing gap: every script refuses to start while any
process holds a GPU (override with --force).  Results are appended as JSON
lines to --out (default ~/timeres_bench/<UTC>/results.jsonl).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import subprocess
import sys
import time

import torch


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="results .jsonl path")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="run even if other processes hold a GPU (NOT during observing)")
    ap.add_argument("--selftest", action="store_true",
                    help="tiny CPU-sized correctness check, no timing")
    return ap


def gpu_busy_guard(force: bool) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        out = ""
    procs = [ln for ln in out.splitlines() if ln.strip() and str(os.getpid()) not in ln]
    if procs and not force:
        sys.exit("refusing to run: GPU(s) in use (observing?)\n" + "\n".join(procs) +
                 "\nRun in an observing gap with the pipeline stopped, or pass --force.")


def default_out() -> str:
    d = os.path.expanduser("~/timeres_bench/" + _dt.datetime.utcnow().strftime("%Y%m%dT%H%MZ"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "results.jsonl")


def env_info(device: str) -> dict:
    info = {"host": socket.gethostname(), "utc": _dt.datetime.utcnow().isoformat() + "Z",
            "torch": torch.__version__}
    if device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, sm=f"{p.major}.{p.minor}", n_sm=p.multi_processor_count,
                    mem_gib=round(p.total_memory / 2 ** 30, 2))
    return info


def cuda_time_ms(fn, reps: int = 5, warmup: int = 2) -> float:
    """Mean wall time per call (ms) measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


class Recorder:
    def __init__(self, bench: str, args):
        if args.device.startswith("cuda"):
            # CUDA events, synchronize() and Triton launches all use the CURRENT
            # device; without this every --device cuda:1 timing measured cuda:0.
            torch.cuda.set_device(args.device)
        self.bench = bench
        self.path = args.out or default_out()
        self.meta = env_info(args.device)

    def row(self, **kw) -> None:
        rec = {"bench": self.bench, **self.meta, **kw}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in kw.items()}), flush=True)


def try_run(rec: Recorder, label: str, fn, **meta):
    """Run one configuration; record OOM/errors instead of aborting the suite."""
    try:
        res = fn()
        rec.row(case=label, ok=True, **meta, **res)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec.row(case=label, ok=False, error="OOM", **meta)
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        rec.row(case=label, ok=False, error=repr(exc)[:300], **meta)


def now() -> float:
    return time.perf_counter()
