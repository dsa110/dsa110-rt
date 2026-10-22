"""The CP1 corr bench (``bench/timeres/b8_corr2x.py``) dry-runs on the CPU:
plans, value maps, CSRs and launch geometries for a gridded and a
per-baseline format build, fit the kernel limits, and the memory estimate
stays inside the plan's corr budget."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.parametrize("fmt", ["2x-gridded-ns2-cint4", "2x-baselines-ns4-cint4"])
def test_b8_describe(fmt) -> None:
    sys.path.insert(0, os.path.join(ROOT, "bench", "timeres"))
    from types import SimpleNamespace
    import b8_corr2x
    d = b8_corr2x.describe(fmt, b8_corr2x._antpos_for(SimpleNamespace(antpos="fixture")), 15)
    assert d["values"] > 0 and d["sources"] >= d["values"]
    assert d["ring_gib"] < 3.0
    for t in d["tiers"]:
        assert t["threads"] <= 1024 and t["smem_kib"] <= 48 and t["chunk"] >= 1
