"""Recorder pread-path benchmark on a SYNTHETIC production-size ring.

Runs only when selected explicitly (``-m x2_bench``); it creates a ~4 GiB
/dev/shm segment, so normal CI runs skip it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "tools" / "x2" / "rx_recorder_bench.py"


@pytest.mark.x2_bench
def test_recorder_pread_bench(request):
    if "x2_bench" not in (request.config.option.markexpr or ""):
        pytest.skip("benchmark: run with -m x2_bench")
    spec = importlib.util.spec_from_file_location("rx_recorder_bench", _P)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rx_recorder_bench"] = mod
    spec.loader.exec_module(mod)
    read = mod.run_bench(seconds=20.0)
    if "skipped" in read:
        pytest.skip(read["skipped"])
    wr = mod.run_bench(seconds=8.0, write_dir=str(Path.home() / "x2_scratch"))
    keys = ("rows_recorded", "rows_expected", "read_MB_per_s", "child_cpu_frac", "child_user_s",
            "child_sys_s", "child_minflt", "child_majflt", "minflt_per_GB_read", "time_v", "rc")
    short = {name: {k: v.get(k) for k in keys} for name, v in (("read_only", read), ("with_write", wr))}
    warnings.warn("RECBENCH " + json.dumps(short))
    assert read["rc"] == 0 and wr["rc"] == 0, (read.get("stderr_tail"), wr.get("stderr_tail"))
    assert read["rows_recorded"] >= 0.95 * read["rows_expected"]
    assert read["child_cpu_frac"] < 1.0
