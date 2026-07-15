"""Voltage-dump timing guards (docs/voltage_dumps/VOLTAGE_DUMP_TIMING_FIX.md).

Three invariants:

1. the deployed dump window contains the full dispersion sweep at the
   deployed dm-plan's fine-DM maximum (the event label is the arrival
   at the BOTTOM of the band, so the burst extends BEFORE it);
2. the C1→SNAP specnum conversion derives its factor from the peak
   member's batch header, not a compile-time constant;
3. end-to-end: every chgroup's earliest signal falls inside the staged
   window at the worst-case DM.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from dsart.common.constants import (
    BLOCK_DURATION_S,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
    SPECNUM_PERIOD_US,
    freq_GHz,
)
from dsart.coinc.stats import compute_stats
from dsart.coinc.window import WindowEntry
from dsart.services.coincidencer import search_to_snap_specnum

_REPO = Path(__file__).resolve().parents[1]
_YAML = _REPO / "configs" / "dsart_pipeline_rt.yaml"

K_DM_S = 4.148808e-3  # s GHz^2 / (pc cm^-3)


def _sweep_s(dm: float, nu_hi_ghz: float, nu_lo_ghz: float) -> float:
    return K_DM_S * dm * (nu_lo_ghz ** -2 - nu_hi_ghz ** -2)


def _deployed_retention_args() -> dict:
    """Parse --n-pre/--n-post/--dm-plan-path out of the pipeline yaml."""
    text = _YAML.read_text(encoding="utf-8")
    out = {}
    for key in ("n-pre", "n-post"):
        m = re.search(rf"--{key}\s+(\d+)", text)
        assert m, f"--{key} not found in {_YAML}"
        out[key] = int(m.group(1))
    m = re.search(r"--dm-plan-path\s+(\S+)", text)
    assert m, "--dm-plan-path not found"
    out["dm_plan"] = Path(m.group(1))
    return out


def _fine_dm_max(plan_path: Path) -> float:
    if not plan_path.is_file():
        pytest.skip(f"deployed dm plan not present on this host: {plan_path}")
    with np.load(plan_path) as d:
        return float(np.asarray(d["fine_dm"]).max())


def test_window_contains_full_sweep() -> None:
    args = _deployed_retention_args()
    dm_max = _fine_dm_max(args["dm_plan"])
    tau = _sweep_s(dm_max, NU_TOP_PROC_GHZ, NU_BOT_PROC_GHZ)
    # +1 block: the event can land at the very start of its block.
    assert args["n-pre"] * BLOCK_DURATION_S >= tau + BLOCK_DURATION_S, (
        f"n_pre={args['n-pre']} does not contain the {tau:.3f}s sweep at "
        f"fine-DM max {dm_max:.2f} — re-derive n_pre "
        "(VOLTAGE_DUMP_TIMING_FIX.md §3)")
    assert args["n-post"] * BLOCK_DURATION_S >= 0.2


def test_c1_to_native_specnum_conversion() -> None:
    S = 4935704
    # production op-point: 1048.576 us / 65.536 us = 16
    assert search_to_snap_specnum(S, 1048.576) == S * 16
    # the case the old fixed x16 got wrong: 524.288 us -> 8
    assert search_to_snap_specnum(S, 524.288) == S * 8
    # guards: heartbeat placeholder + non-integer multiples fall back to
    # the production factor instead of raising
    assert search_to_snap_specnum(S, 1.0) == S * 16
    assert search_to_snap_specnum(S, 1000.0) == S * 16


def test_peak_sample_period_reaches_stats() -> None:
    def entry(spec: int, period: float, snr: float) -> WindowEntry:
        kw = dict(
            mjd=61236.0, event_specnum=spec, snr=snr, dm_pc_cc=500.0,
            dm_idx_global=10, fine_dm_idx=1, l_rad=0.001, m_rad=0.002,
            l_pix=10, m_pix=12, width_samples=4, kernel_id="unit:d1:b4",
            flags=0, search_node_id=1, gpu_half=0, cube_id=7,
            sample_period_us=period,
        )
        # WindowEntry fields may differ across versions; filter to the
        # ones it actually declares.
        import dataclasses
        names = {f.name for f in dataclasses.fields(WindowEntry)}
        return WindowEntry(**{k: v for k, v in kw.items() if k in names})

    members = [entry(100, 524.288, 10.0), entry(200, 524.288, 30.0)]
    stats = compute_stats(members, gal_dm_max_los=1000.0)
    assert stats.peak_event_specnum == 200
    assert stats.peak_sample_period_us == pytest.approx(524.288)
    # the full chain: broadcast value uses the peak member's own unit
    assert search_to_snap_specnum(
        stats.peak_event_specnum, stats.peak_sample_period_us) == 200 * 8


def test_dump_window_reference_frequency() -> None:
    args = _deployed_retention_args()
    dm_max = _fine_dm_max(args["dm_plan"])
    n_pre, n_post = args["n-pre"], args["n-post"]
    block_specnums = BLOCK_DURATION_S / (SPECNUM_PERIOD_US * 1e-6)  # 2048

    # worst case: event at the very first specnum of its block
    for T in (38560 * 2048, 38560 * 2048 + 2047):
        target_block = T // 2048
        win_lo = (target_block - n_pre) * 2048
        win_hi = (target_block + n_post + 1) * 2048   # exclusive
        for g in range(16):
            nu_g_top = freq_GHz(g, 0)
            tau = _sweep_s(dm_max, nu_g_top, NU_BOT_PROC_GHZ)
            earliest = T - tau / 65.536e-6
            assert win_lo <= earliest < win_hi, (
                f"chgroup {g}: earliest signal {earliest:.0f} outside "
                f"[{win_lo}, {win_hi}) at DM {dm_max:.1f} (T={T})")
    assert block_specnums == 2048
