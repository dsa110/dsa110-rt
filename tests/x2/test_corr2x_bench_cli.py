"""CPU dry run of the CP1 corr bench's command-line plumbing: the production
corr_fast argv is taken from configs/dsart_pipeline_rt.yaml, tokens are
substituted, the 1x TX / monitoring / etcd-touching options are dropped,
and corr_fast's OWN CLI parser accepts the result (``run`` stubbed) -- for
the 1x profile/golden runs and for the 2x live run."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bench", "timeres"))


@pytest.fixture
def b8(tmp_path, monkeypatch):
    import b8_corr2x
    cal = tmp_path / "antennas.out"
    cal.write_bytes(b"")
    real = b8_corr2x._prod_argv

    def with_local_cal(**kw):
        argv, env = real(**kw)
        i = argv.index("--apply-cal")
        argv[i + 1] = str(cal)
        if "--dm-plan-path" in argv:
            j = argv.index("--dm-plan-path")
            plan = tmp_path / "plan.npz"
            plan.write_bytes(b"")
            argv[j + 1] = str(plan)
        return argv, env

    monkeypatch.setattr(b8_corr2x, "_prod_argv", with_local_cal)
    monkeypatch.setattr(b8_corr2x, "_cn_chgroup", lambda: (4, 1))
    return b8_corr2x


def test_live_argv_is_a_2x_drain(b8) -> None:
    argv, env = b8._prod_argv(
        dec=71.63,
        drop=("--transport-tx-*", "--sky-export-*", "--rfi-mon-*", "--inject-watch", "--stage2-mode",
              "--dm-plan-path", "--ready-sentinel-path", "--output-dir", "--max-blocks"),
        set_={"--t-int-fast-native": 8, "--chan-sum-factor": 2, "--n-sub": 4,
              "--opoint": "2x-baselines-ns4-cint4", "--max-blocks": 8941,
              "--output-dir": "/tmp/dsart-x2-test-out", "--blocks-output-mode": "none",
              "--ready-sentinel-path": "/tmp/dsart-x2-corr-fast-live.ready"})
    joined = " ".join(argv)
    for bad in ("--transport-tx", "--sky-export", "--rfi-mon", "--inject-watch", "--stage2-mode",
                "/tmp/dsart-corr-fast-", "CHGROUP", "CUSTOMDEC"):
        assert bad not in joined, bad
    assert env.get("CUDA_VISIBLE_DEVICES") == "1"
    cap = b8._capture_cfg(argv)
    cfg, kw = cap["cfg"], cap["kw"]
    assert (cfg.t_int_fast_native, cfg.chan_sum_factor, cfg.n_sub, cfg.chgroup) == (8, 2, 4, 1)
    assert kw["opoint"] == "2x-baselines-ns4-cint4" and kw["max_blocks"] == 8941
    assert kw["transport_tx_host"] == "" and kw["sky_export_url"] == ""
    assert kw["rfi_mon_cn_id"] is None and not cfg.inject_watch_enabled
    assert cfg.rfi_enabled and cfg.rfi_persist_enabled and cfg.rfi_array_burst_mode == "flag"


def test_1x_argv_keeps_production_parameters(b8) -> None:
    argv, _ = b8._prod_argv(dec=71.63, drop=("--transport-tx-*", "--sky-export-*", "--rfi-mon-*",
                                              "--inject-watch", "--ready-sentinel-path",
                                              "--output-dir", "--max-blocks"))
    cap = b8._capture_cfg(argv)
    cfg, kw = cap["cfg"], cap["kw"]
    assert (cfg.t_int_fast_native, cfg.chan_sum_factor, cfg.n_sub) == (32, 8, 4)
    assert cfg.sliding_window and cfg.cell_lambda_mode == "tee45"
    assert kw["opoint"] == "1x" and kw["stage2_mode"] == "per_coarse_dm"
