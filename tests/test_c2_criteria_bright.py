"""Bright bursts are classified on their PEAK member, not the medians.

Live 2026-09-24 (DM 500, W = 1 ms injections): up to 100 sigma the
cluster medians matched the burst, but at 150-200 sigma width_median was
48 and dm_median +36.5 (sidelobe members at wide boxcars / nearby DMs)
and at 300 sigma width_median was 64, so the production
``bright_frb_extragalactic`` class (width_median <= 16) stopped matching
and the bursts were filed log_only / bright_pulsar. These use the
PRODUCTION criteria file.
"""
from __future__ import annotations

from pathlib import Path

from dsart.coinc.criteria import BRIGHT_PEAK_SNR, CriteriaEvaluator
from dsart.coinc.stats import ClusterStats, compute_stats
from dsart.coinc.window import WindowEntry

PROD = Path(__file__).resolve().parents[1] / "configs" / "c2_trigger_criteria.yaml"
GAL = 630.0          # NE2001 max-LOS DM at the DEC +16 pointing


def _cls(**kw) -> str:
    base = dict(
        n_events=40, n_search_nodes=2, n_gpu_halves=3,
        snr_max=150.0, snr_sum=2000.0, snr_mean=40.0,
        dm_min=450.0, dm_max=700.0, dm_median=536.5, dm_iqr=60.0,
        l_median=0.004, m_median=-0.003, lm_diag_rad=2e-3,
        width_min=2, width_max=64, width_median=48.0,
        t_start_mjd=61307.2, t_end_mjd=61307.2, t_peak_mjd=61307.2,
        kernel_ids_distinct=("unit:d1:b2",), peak_event_specnum=1,
        gal_dm_max_los=GAL, dm_galactic_fraction=536.5 / GAL,
        dm_peak=500.3, width_peak=2,
    )
    base.update(kw)
    hit = CriteriaEvaluator(PROD).evaluate(ClusterStats(**base))
    return hit.name if hit is not None else "none"


def test_bright_burst_with_sidelobe_medians_is_an_frb():
    # the 150-200 sigma live case
    assert _cls() == "bright_frb_extragalactic"
    assert _cls(snr_max=200.0) == "bright_frb_extragalactic"


def test_300_sigma_burst_is_not_a_pulsar():
    # many members at nearly one DM used to fall into bright_pulsar
    assert _cls(snr_max=300.0, width_median=64.0, dm_iqr=1.0,
                dm_median=500.3, dm_galactic_fraction=500.3 / GAL) \
        == "bright_frb_extragalactic"


def test_faint_clusters_are_judged_exactly_as_before():
    # same sidelobe medians below the bright threshold: still rejected
    assert _cls(snr_max=30.0) == "log_only"
    # and a normal faint FRB still passes on its medians
    assert _cls(snr_max=30.0, width_median=2.0, dm_median=500.3,
                dm_iqr=5.0, dm_galactic_fraction=500.3 / GAL) \
        == "bright_frb_extragalactic"


def test_bright_but_wide_peak_is_still_rejected():
    # broadband RFI whose brightest detection is itself wide
    assert _cls(width_peak=48) != "bright_frb_extragalactic"


def test_bright_galactic_split_uses_the_peak_dm():
    # medians say extragalactic (536/630 = 0.85), the burst itself is at
    # DM 300 (0.48): must NOT be called extragalactic
    got = _cls(dm_peak=300.0)
    assert got != "bright_frb_extragalactic"
    assert got == "bright_galactic"


def _entry(snr, dm, width):
    return WindowEntry(
        mjd=61307.2, snr=snr, l_rad=0.004, m_rad=-0.003, l_pix=0, m_pix=0,
        dm_pc_cc=dm, dm_idx_global=0, fine_dm_idx=0, event_specnum=1,
        width_samples=width, kernel_id="unit:d1:b2", flags=0,
        search_node_id=9, gpu_half=0, cube_id=0, sample_period_us=1048.576)


def test_compute_stats_takes_peak_fields_from_the_max_snr_member():
    members = [_entry(40.0, 540.0, 48), _entry(150.0, 500.3, 2),
               _entry(60.0, 560.0, 64)]
    s = compute_stats(members, gal_dm_max_los=GAL)
    assert s.dm_peak == 500.3 and s.width_peak == 2
    assert s.width_median == 48.0
    assert s.snr_max >= BRIGHT_PEAK_SNR
