"""faint_frb_extragalactic -- the -1 sigma trigger for narrow bursts (2026-10-05).

The C2 FRB trigger was 12 sigma for width <= 16. This class adds 11-12
sigma for width <= 8 only. Narrow-only because the extra wide (w=16)
clusters a whole-class cut would add are a population C3's R10 rejects
regardless (its single-sample tz_trig scales ~2.5/sqrt(w) of the detector
SNR, so at w=16 it cannot tell real from RFI), while for w <= 8 R10
confirms real bursts. See the class description in the yaml.
"""
from __future__ import annotations

from pathlib import Path

from dsart.coinc.criteria import CriteriaEvaluator
from dsart.coinc.stats import ClusterStats

PROD = Path(__file__).resolve().parents[1] / "configs" / "c2_trigger_criteria.yaml"
GAL = 46.1           # NE2001 max-LOS DM at the current pointing


def _base() -> dict:
    return dict(
        n_events=1, n_search_nodes=1, n_gpu_halves=1,
        snr_max=11.5, snr_sum=11.5, snr_mean=11.5,
        dm_min=500.0, dm_max=500.0, dm_median=500.0, dm_iqr=0.0,
        l_median=0.004, m_median=-0.003, lm_diag_rad=0.0,
        width_min=2, width_max=2, width_median=2.0,
        t_start_mjd=61318.2, t_end_mjd=61318.2, t_peak_mjd=61318.2,
        kernel_ids_distinct=("unit:d1:b2",), peak_event_specnum=1,
        gal_dm_max_los=GAL, dm_galactic_fraction=500.0 / GAL,
    )


def _cls(**kw) -> str:
    base = _base()
    base.update(kw)
    hit = CriteriaEvaluator(PROD).evaluate(ClusterStats(**base))
    return hit.name if hit is not None else "none"


def test_a_narrow_11p5_sigma_burst_now_triggers() -> None:
    for w in (1, 2, 4, 8):
        assert _cls(width_median=float(w)) == "faint_frb_extragalactic", w


def test_wide_11p5_sigma_clusters_still_do_not() -> None:
    assert _cls(width_median=16.0) == "log_only"
    assert _cls(width_median=12.0) == "log_only"


def test_below_11_sigma_nothing_changes() -> None:
    assert _cls(snr_max=10.9) == "log_only"


def test_12_sigma_and_up_is_still_the_bright_class() -> None:
    """First match wins: anything the old class took, it still takes."""
    assert _cls(snr_max=12.0) == "bright_frb_extragalactic"
    assert _cls(snr_max=12.5, width_median=16.0) == "bright_frb_extragalactic"


def test_the_faint_class_keeps_every_other_gate() -> None:
    assert _cls(dm_median=110.0, dm_galactic_fraction=110.0 / GAL) == "log_only"   # floor rail
    assert _cls(dm_galactic_fraction=0.5) == "log_only"                           # galactic
    assert _cls(lm_diag_rad=0.2) == "log_only"                                    # extended
    assert _cls(dm_iqr=150.0) == "log_only"                                       # DM-smeared


def test_it_dumps_and_has_its_own_holdoff() -> None:
    ev = CriteriaEvaluator(PROD)
    c = {x.name: x for x in ev.classes}["faint_frb_extragalactic"]
    assert c.action == "dump_all_gpus"
    assert c.holdoff_s == 30.0
    names = [x.name for x in ev.classes]
    assert names.index("faint_frb_extragalactic") == names.index("bright_frb_extragalactic") + 1


def test_a_bright_burst_in_bright_holdoff_is_not_dumped_again() -> None:
    """2026-10-05 07:00 live: a 100 sigma probe dumped as
    bright_frb_extragalactic, then its grown cluster fell through the
    bright class's holdoff into this class and dumped a second time."""
    from dsart.coinc.criteria import CriteriaEvaluator
    clock = {"t": 1000.0}
    ev = CriteriaEvaluator(PROD, now=lambda: clock["t"])
    first = ClusterStats(**{**_base(), "snr_max": 88.4, "width_median": 18.0,
                            "width_peak": 2, "dm_peak": 971.0})
    assert ev.evaluate(first).name == "bright_frb_extragalactic"
    clock["t"] += 0.7
    grown = ClusterStats(**{**_base(), "snr_max": 88.4, "n_events": 4,
                            "width_median": 20.0, "width_peak": 2, "dm_peak": 971.0})
    again = ev.evaluate(grown)
    assert again is None or again.action != "dump_all_gpus", again


def test_the_faint_class_is_11_to_12_sigma_only() -> None:
    assert _cls(snr_max=11.99) == "faint_frb_extragalactic"
    assert _cls(snr_max=12.0) == "bright_frb_extragalactic"
