#!/usr/bin/env python
"""Benchmark rx_recorder's pread path on a SYNTHETIC v3 ring (never production).

Creates a private ring with production slot geometry (64 corrs x 6144 rows x
5216 B slots; ``--n-dm`` coarse DMs, default 2 = the DMs a node records),
prefaulted like the RX does, advances its write_seq at the production row
rate (953.67 rows/s, both DMs) from a ticker thread, and runs the recorder
as a SUBPROCESS so its CPU time and page faults are measured on their own
(``/usr/bin/time -v`` when installed, else ``getrusage(RUSAGE_CHILDREN)``).
Refuses to start unless /dev/shm keeps ``--shm-margin-gib`` free after the
ring is created.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROWS_PER_S = 1.0 / (32 * 32.768e-6)          # 953.67 search samples/s (1x)
RECORDER = Path(__file__).with_name("rx_recorder.py")


def _shm_free_bytes() -> int:
    st = os.statvfs("/dev/shm")
    return st.f_bavail * st.f_frsize


def run_bench(*, seconds: float = 20.0, n_corr: int = 64, n_dm: int = 2, t_buf: int = 6144,
              n_filled: int = 2600, write_dir: str | None = None,
              shm_margin_gib: float = 8.0, chunk_rows: int = 64) -> dict:
    from dsart.transport.recv_ring import RxRing, RxRingDims, VF_DATA_PRESENT

    stride = ((n_filled * 2 + 10) + 7) // 8 * 8
    need = 4096 + n_corr * n_dm * t_buf * stride
    free = _shm_free_bytes()
    if free - need < shm_margin_gib * (1 << 30):
        return {"skipped": f"/dev/shm free {free / 2**30:.1f} GiB - ring {need / 2**30:.1f} GiB "
                           f"< margin {shm_margin_gib} GiB"}
    name = f"/x2recbench-{uuid.uuid4().hex[:10]}"
    t_create = time.monotonic()
    ring = RxRing.open_or_create(name, RxRingDims(n_corr, n_dm, t_buf, n_filled, 2))  # memsets all
    create_s = time.monotonic() - t_create
    out = None
    try:
        payload = np.arange(n_filled * 2, dtype=np.int64).astype(np.int8)
        for d in range(n_dm):                              # mark the DMs active (rows 0..63)
            for t in range(64):
                ring.write_slot(0, d, t, payload, VF_DATA_PRESENT, scale=1.0)
        # Synthetic writer: drive write_seq (our own segment) at the production rate.
        wseq = np.memmap("/dev/shm/" + name.lstrip("/"), dtype="<u8", mode="r+",
                         offset=32, shape=(n_corr,))
        start_row = t_buf // 2
        stop = threading.Event()
        t0 = time.monotonic()

        def tick() -> None:
            while not stop.is_set():
                rows = start_row + int((time.monotonic() - t0) * ROWS_PER_S)
                wseq[:] = rows * n_dm
                time.sleep(0.001)

        th = threading.Thread(target=tick, daemon=True)
        th.start()
        time.sleep(0.05)
        cmd = [sys.executable, str(RECORDER), "--shm-name", name, "--no-guard",
               "--duration-s", str(seconds), "--dms", ",".join(map(str, range(n_dm))),
               "--n-active-dms", str(n_dm), "--fan-in-min", str(n_corr),
               "--chunk-rows", str(chunk_rows), "--stall-s", str(seconds + 10)]
        if write_dir:
            out = Path(tempfile.mkdtemp(prefix="recbench-", dir=write_dir))
            cmd += ["--out-root", str(out), "--max-gib", "100", "--max-disk-frac", "0.9"]
        else:
            cmd += ["--no-write"]
        timev = shutil.which("time") or ("/usr/bin/time" if os.path.exists("/usr/bin/time") else None)
        if timev:
            cmd = [timev, "-v"] + cmd
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 120)
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        stop.set()
        th.join()
        rec = {}
        m = re.search(r"\{.*\}", proc.stdout, re.S)
        if m:
            rec = json.loads(m.group(0))
        tv = {}
        for line in proc.stderr.splitlines():
            line = line.strip()
            for key in ("Minor (reclaiming a frame) page faults", "Major (requiring I/O) page faults",
                        "User time (seconds)", "System time (seconds)",
                        "Percent of CPU this job got", "Maximum resident set size (kbytes)",
                        "Elapsed (wall clock) time (h:mm:ss or m:ss)"):
                if line.startswith(key):
                    tv[key] = line.split(":", 1)[1].strip() if "Elapsed" not in key else line.rsplit(" ", 1)[1]
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
        rows = sum(v.get("n_rows", 0) for v in rec.get("per_dm", {}).values())
        read_bytes = rows * n_corr * stride
        res = {
            "ring": {"n_corr": n_corr, "n_dm": n_dm, "t_buf": t_buf, "slot_stride": stride,
                     "size_gib": need / 2**30, "create_s": create_s},
            "seconds": seconds, "write": bool(write_dir), "rc": proc.returncode,
            "rows_recorded": rows, "rows_expected": int(seconds * ROWS_PER_S * n_dm),
            "read_MB_per_s": read_bytes / seconds / 1e6,
            "child_cpu_s": cpu, "child_cpu_frac": cpu / seconds,
            "child_user_s": ru1.ru_utime - ru0.ru_utime, "child_sys_s": ru1.ru_stime - ru0.ru_stime,
            "child_minflt": ru1.ru_minflt - ru0.ru_minflt,
            "child_majflt": ru1.ru_majflt - ru0.ru_majflt,
            "minflt_per_GB_read": (ru1.ru_minflt - ru0.ru_minflt) / max(1e-9, read_bytes / 1e9),
            "time_v": tv, "recorder": rec, "stderr_tail": proc.stderr[-400:] if proc.returncode else "",
        }
        return res
    finally:
        ring.close()
        RxRing.unlink_name(name)
        if out is not None:
            shutil.rmtree(out, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--n-dm", type=int, default=2)
    p.add_argument("--write-dir", default=None)
    a = p.parse_args(argv)
    print(json.dumps(run_bench(seconds=a.seconds, n_dm=a.n_dm, write_dir=a.write_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
