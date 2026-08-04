"""Tests for :mod:`dsart.coinc.cube_veto` decide() + event_is_injection()."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dsart.coinc.cube_veto import (
    CubeMetrics,
    CubeVetoThresholds,
    decide,
    event_is_injection,
)


def _clean_metrics(**ov) -> CubeMetrics:
    """A metrics object that passes every rule (a real-looking burst)."""
    base = dict(
        ok=True,
        snr_c1=30.0, dm_c1=500.0, fdm_trig=4, l_pix_trig=128, m_pix_trig=128,
        t_img=100, t_apex=100, fdm_apex=4, t_shift=0, dm_shift_trials=0,
        tz_trig=20.0, tz_apex=20.0, dmz=20.0, dm_edge=0,
        imgz_apex=20.0, img_off_apex=1.0, n_fdm=8,
        g_apex_cube_same=1, g_apex_val=20.0, n_cubes=8,
    )
    base.update(ov)
    return CubeMetrics(**base)


def test_clean_burst_keeps() -> None:
    d = decide(_clean_metrics(), is_injection=False)
    assert d.keep is True
    assert d.action == "KEEP"
    assert d.rules_fired == ()


def test_injection_always_kept() -> None:
    # Even a metrics object that trips every rule is kept when injection.
    bad = _clean_metrics(
        img_off_apex=999, t_shift=999, dm_shift_trials=999,
        tz_apex=0.0, imgz_apex=0.0, dm_edge=1, g_apex_cube_same=0, tz_trig=0.0,
    )
    d = decide(bad, is_injection=True)
    assert d.keep is True
    assert d.is_injection is True
    assert d.rules_fired == ()


def test_metrics_not_ok_fail_open() -> None:
    d = decide(CubeMetrics(ok=False, reason="missing npz"), is_injection=False)
    assert d.keep is True
    assert "fail-open" in d.notes


def test_r1_image_offset() -> None:
    d = decide(_clean_metrics(img_off_apex=41.0), is_injection=False)
    assert not d.keep and "R1_image_offset" in d.rules_fired


def test_r2_time_shift() -> None:
    d = decide(_clean_metrics(t_shift=31), is_injection=False)
    assert not d.keep and "R2_time_shift" in d.rules_fired


def test_r3_dm_shift() -> None:
    d = decide(_clean_metrics(dm_shift_trials=9), is_injection=False)
    assert not d.keep and "R3_dm_shift" in d.rules_fired


def test_r4_no_peak() -> None:
    d = decide(_clean_metrics(tz_apex=3.9, imgz_apex=3.9), is_injection=False)
    assert not d.keep and "R4_no_peak" in d.rules_fired


def test_r5_dm_edge_rail() -> None:
    d = decide(
        _clean_metrics(dm_edge=1, dm_shift_trials=5), is_injection=False,
    )
    assert not d.keep and "R5_dm_edge_rail" in d.rules_fired


def test_r10_cube_unconfirmed() -> None:
    # 2026-08-04: t_shift/dm_shift must be outside the coherence
    # exemption, otherwise the cube HAS confirmed the trigger and R10 is
    # deliberately suppressed (see test_r10_exempts_... below).
    d = decide(
        _clean_metrics(g_apex_cube_same=0, tz_trig=9.9,
                       t_shift=40, dm_shift_trials=5),
        is_injection=False,
    )
    assert not d.keep and "R10_cube_unconfirmed" in d.rules_fired


def test_thresholds_respected() -> None:
    # With a stricter R1 threshold, a previously-clean offset now trips.
    th = CubeVetoThresholds(r1_img_off_px=0.5)
    d = decide(_clean_metrics(img_off_apex=1.0), is_injection=False,
               thresholds=th)
    assert not d.keep and "R1_image_offset" in d.rules_fired


# --- event_is_injection ------------------------------------------------


def _make_event(tmp: Path, name: str, *, l3_inj=None, csv_inj_id="",
                csv_row=None) -> Path:
    ev = tmp / name
    (ev / "Level2").mkdir(parents=True)
    (ev / "Level3").mkdir(parents=True)
    doc = {"event_name": name}
    if l3_inj is not None:
        doc["injection"] = l3_inj
    (ev / "Level3" / f"{name}.json").write_text(json.dumps(doc))
    row = csv_row or {"mjd": "0", "dm_pc_cc": "0", "l_rad": "0", "m_rad": "0"}
    row = dict(row)
    row["inj_id"] = csv_inj_id
    with (ev / "Level2" / f"C1_window_{name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    return ev


def test_event_is_injection_l3_marker(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "aaaa",
                     l3_inj={"is_injection": True, "inj_ids": ["x"]})
    assert event_is_injection(ev, "aaaa") is True


def test_event_is_injection_csv_inj_id(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "bbbb", csv_inj_id="probe1")
    assert event_is_injection(ev, "bbbb") is True


def test_event_not_injection_when_no_marker(tmp_path: Path) -> None:
    ev = _make_event(tmp_path, "cccc")
    assert event_is_injection(ev, "cccc") is False


def test_event_is_injection_coincidence_fallback(tmp_path: Path) -> None:
    import time

    from dsart.coinc.inject_log import FiredInjection, append_fired_injection

    now = time.time()
    log = tmp_path / "fired.jsonl"
    append_fired_injection(log, FiredInjection(
        inj_id="homxlike", dm_pc_cm3=500.0, l_rad=0.01, m_rad=0.0,
        apply_at_specnum=1, fired_at_unix=now, ttl_s=60.0,
    ))
    mjd = 40587.0 + now / 86400.0
    ev = _make_event(
        tmp_path, "dddd",
        csv_row={"mjd": f"{mjd:.11f}", "dm_pc_cc": "503.0",
                 "l_rad": "0.011", "m_rad": "0.0"},
    )
    # No L3 marker, no CSV inj_id → only the durable-log coincidence flags it.
    assert event_is_injection(ev, "dddd") is False
    assert event_is_injection(ev, "dddd", fired_log_path=log) is True


# ---------------------------------------------------------------------------
# R10 coherence exemption + R11 cube-quality (2026-08-04)
# ---------------------------------------------------------------------------


def _r10_only(**ov) -> CubeMetrics:
    """Metrics that trip R10 and nothing else: the global apex is in
    another half and the trigger feature is weak."""
    base = dict(g_apex_cube_same=0, tz_trig=6.6, tz_apex=6.6, imgz_apex=6.6,
                t_shift=0, dm_shift_trials=0)
    base.update(ov)
    return _clean_metrics(**base)


def test_r10_exempts_a_temporally_and_spectrally_coherent_trigger():
    """260803doen: the cube's apex sits exactly on the trigger's time and
    DM, so the cube HAS confirmed it — a louder feature in another half
    (which just means that half has worse RFI) must not reject it."""
    d = decide(_r10_only(), is_injection=False)
    assert d.keep, d.rules_fired
    assert "R10_cube_unconfirmed" not in d.rules_fired


def test_r10_still_fires_when_the_cube_is_incoherent():
    """260803szgu: apex 82 samples and 25 DM trials away — the exemption
    must not rescue that."""
    d = decide(_r10_only(t_shift=82, dm_shift_trials=25), is_injection=False)
    assert not d.keep
    assert "R10_cube_unconfirmed" in d.rules_fired


def test_r10_exemption_is_bounded_by_its_thresholds():
    th = CubeVetoThresholds()
    # just inside on both axes -> exempt
    assert decide(
        _r10_only(t_shift=th.r10_coherent_t_samples,
                  dm_shift_trials=th.r10_coherent_dm_trials),
        is_injection=False,
    ).keep
    # one sample too far in time -> R10 bites again
    assert "R10_cube_unconfirmed" in decide(
        _r10_only(t_shift=th.r10_coherent_t_samples + 1), is_injection=False,
    ).rules_fired
    # one trial too far in DM -> likewise
    assert "R10_cube_unconfirmed" in decide(
        _r10_only(dm_shift_trials=th.r10_coherent_dm_trials + 1),
        is_injection=False,
    ).rules_fired


def _streaky(**ov) -> CubeMetrics:
    """A broad candidate in a demonstrably non-Gaussian half, weak and
    incoherent enough that no exemption applies."""
    base = dict(width_samples=16, streak_ac1_trig=0.28, frac_z5_trig=0.012,
                tz_trig=6.0, t_shift=10, dm_shift_trials=3)
    base.update(ov)
    return _clean_metrics(**base)


def test_r11_is_off_by_default_but_says_so():
    """Uncalibrated, so it must not reject — but the decision has to
    record that it would have, or we can never calibrate it."""
    d = decide(_streaky(), is_injection=False)
    assert d.keep
    assert "R11_nongaussian_cube" not in d.rules_fired
    assert "R11 would fire" in d.notes


def test_r11_rejects_when_enabled():
    th = CubeVetoThresholds(r11_enabled=True)
    d = decide(_streaky(), is_injection=False, thresholds=th)
    assert not d.keep
    assert "R11_nongaussian_cube" in d.rules_fired


def test_r11_spares_a_narrow_candidate_in_a_streaky_half():
    """Interference correlated on ~10 ms scales does not manufacture
    1-sample spikes, so width gates the rule."""
    th = CubeVetoThresholds(r11_enabled=True)
    d = decide(_streaky(width_samples=1), is_injection=False, thresholds=th)
    assert "R11_nongaussian_cube" not in d.rules_fired


def test_r11_spares_a_strong_trigger_in_a_streaky_half():
    """260802ohco: SNR 18, tz_trig 11.6, coherent, but its trigger half
    is genuinely streaky. A noisy half makes a candidate less
    trustworthy; it does not outweigh direct evidence."""
    th = CubeVetoThresholds(r11_enabled=True)
    assert "R11_nongaussian_cube" not in decide(
        _streaky(tz_trig=11.6), is_injection=False, thresholds=th,
    ).rules_fired
    assert "R11_nongaussian_cube" not in decide(
        _streaky(t_shift=0, dm_shift_trials=0), is_injection=False,
        thresholds=th,
    ).rules_fired


def test_r11_needs_a_width_to_fire():
    """Callers with no width (width_samples unset) keep the old rule set."""
    th = CubeVetoThresholds(r11_enabled=True)
    d = decide(_streaky(width_samples=0), is_injection=False, thresholds=th)
    assert "R11_nongaussian_cube" not in d.rules_fired


def test_r11_never_overrides_the_injection_exemption():
    th = CubeVetoThresholds(r11_enabled=True)
    d = decide(_streaky(), is_injection=True, thresholds=th)
    assert d.keep and d.rules_fired == ()


# ---------------------------------------------------------------------------
# live_span / cube_noise_character
# ---------------------------------------------------------------------------


def test_live_span_ignores_a_zero_filled_overlap_tail():
    import numpy as np
    from dsart.coinc.cube_veto import live_span

    wf = np.abs(np.random.RandomState(0).normal(size=(256, 34))) + 1.0
    assert live_span(wf) == 256          # fully populated cube untouched
    wf[192:] = 0.0                       # the production 64-sample overlap
    assert live_span(wf) == 192


def test_noise_character_separates_white_from_streaky():
    import numpy as np
    from dsart.coinc.cube_veto import cube_noise_character

    rng = np.random.RandomState(1)
    white = np.abs(rng.normal(size=(256, 34)))
    ac1_w, fz_w = cube_noise_character(white)
    # heavily time-correlated: each sample carries the previous one
    streaky = np.cumsum(rng.normal(size=(256, 34)), axis=0)
    ac1_s, _ = cube_noise_character(streaky)
    assert abs(ac1_w) < 0.10 < ac1_s
    assert fz_w < 5.0e-3


def test_noise_character_measures_only_the_live_span():
    import numpy as np
    from dsart.coinc.cube_veto import cube_noise_character

    wf = np.abs(np.random.RandomState(2).normal(size=(256, 34))) + 1.0
    ac1_full, fz_full = cube_noise_character(wf)
    wf2 = wf.copy()
    wf2[192:] = 0.0
    ac1_tail, fz_tail = cube_noise_character(wf2)
    # the zero block must not register as structure
    assert abs(ac1_tail - ac1_full) < 0.05
    assert abs(fz_tail - fz_full) < 2.0e-3
