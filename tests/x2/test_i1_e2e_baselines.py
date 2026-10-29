"""x2 I1, per-baseline leg: the whole 2x chain for ``2x-baselines-ns4-cint5``.

Same bursts, chain and assertions as ``tests/x2/test_i1_e2e.py`` (gridded
n_sub 2 cint4; see its docstring and ``tests/x2/i1_util.py``). Its own file
because the per-baseline n_sub 4 format (64 streams, 30-35 fines per coarse DM)
costs ~9 min on one core and the harness gives each file 15 min.
"""
from __future__ import annotations

from . import i1_util as U


def test_i1_2x_chain_baselines_ns4_cint5(tmp_path, monkeypatch) -> None:
    U.run_2x_chain("2x-baselines-ns4-cint5", tmp_path, monkeypatch)
