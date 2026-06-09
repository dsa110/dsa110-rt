"""Tests for the M7.6 C1→C2 metering: candidate selection
(``meter_candidates``) + the per-half metering mon publisher
(``SearchComputeMonPublisher``)."""

from __future__ import annotations

import os

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate  # noqa: E402
from dsart.services.search_compute import (  # noqa: E402
    boxcar_noise_color_factor,
    derate_noise_color,
    dm_smear_samples,
    filter_unphysical_narrow,
    meter_candidates,
)
from dsart.services.search_compute_mon import (  # noqa: E402
    SearchComputeMonPublisher,
    build_search_compute_mon_key,
)


def _cand(
    width: int, snr: float, *, dm_idx: int = 0, dm_fine: float = 1.0,
) -> Candidate:
    return Candidate(
        l=0.0, m=0.0, dm_fine=dm_fine, dm_idx=dm_idx, event_specnum=100,
        width_samples=width, kernel_id="unit:d1:b4", snr=snr,
        detector_version="t", flags=0, search_node_id=0, gpu_half=0,
    )


def test_meter_disabled_passthrough() -> None:
    cands = [_cand(1, 20.0), _cand(8, 99.0)]
    for cap in (None, 0, -1):
        kept, dropped = meter_candidates(cands, cap)
        assert kept is cands
        assert dropped == 0


def test_meter_under_cap_is_noop_identity() -> None:
    cands = [_cand(1, 20.0), _cand(2, 30.0)]
    kept, dropped = meter_candidates(cands, 8)
    # At/under cap: returned unchanged (no sort, dump/retention unaffected).
    assert kept is cands
    assert dropped == 0


def test_meter_narrow_first_then_bright() -> None:
    # 6 candidates, cap 3. Ordering priority: width asc, then snr desc.
    cands = [
        _cand(4, 50.0),   # widest
        _cand(1, 10.0),   # narrowest, low snr
        _cand(2, 40.0),
        _cand(1, 30.0),   # narrowest, high snr
        _cand(2, 35.0),
        _cand(8, 99.0),   # widest + brightest -> still dropped (too wide)
    ]
    kept, dropped = meter_candidates(cands, 3)
    assert dropped == 3
    widths = [c.width_samples for c in kept]
    snrs = [c.snr for c in kept]
    # Keep both width-1 (ordered by snr desc), then the best width-2.
    assert widths == [1, 1, 2]
    assert snrs == [30.0, 10.0, 40.0]
    # The bright wide (8,99) is shed despite top SNR — width wins first.
    assert all(c.width_samples <= 2 for c in kept)


def test_meter_ties_keep_count_exact() -> None:
    cands = [_cand(1, float(i)) for i in range(20)]
    kept, dropped = meter_candidates(cands, 5)
    assert len(kept) == 5
    assert dropped == 15
    # All width 1 → top-5 by snr desc.
    assert sorted((c.snr for c in kept), reverse=True) == [19, 18, 17, 16, 15]


def test_meter_always_keep_predicate_exempts_cal_probes() -> None:
    """T3 (2026-06-07): a candidate matching ``always_keep_predicate``
    bypasses the cap. Only the rest of the list is metered."""
    # 5 cal-probe matches (wide+bright; would be dropped by the cap)
    # plus 5 RFI singletons that under the legacy ordering would
    # narrowly beat them out.
    probes = [_cand(8, 99.0, dm_fine=500.0) for _ in range(5)]
    rfi = [_cand(1, 20.0 + i, dm_fine=1500.0) for i in range(5)]
    cands = rfi + probes
    # Without the predicate, cap=4 keeps the 4 brightest narrow
    # candidates and drops every probe.
    kept_legacy, dropped_legacy = meter_candidates(cands, 4)
    assert dropped_legacy == 6
    assert all(c.width_samples == 1 for c in kept_legacy)

    # With the predicate, every probe is kept AND the cap is applied
    # to the rest (so we keep all 5 probes + the top 4 RFI singles).
    def is_probe(c):
        return abs(c.dm_fine - 500.0) < 1.0

    kept_t3, dropped_t3 = meter_candidates(
        cands, 4, always_keep_predicate=is_probe,
    )
    assert dropped_t3 == 1, "1 RFI single shed; all 5 probes kept"
    n_probes_kept = sum(1 for c in kept_t3 if abs(c.dm_fine - 500.0) < 1.0)
    n_rfi_kept = sum(1 for c in kept_t3 if abs(c.dm_fine - 1500.0) < 1.0)
    assert n_probes_kept == 5
    assert n_rfi_kept == 4


def test_meter_always_keep_predicate_no_op_when_under_cap() -> None:
    """When the post-split rest is under the cap, the entire list is
    returned (no sort, no drops) regardless of the predicate."""
    probes = [_cand(8, 99.0, dm_fine=500.0) for _ in range(2)]
    rfi = [_cand(1, 20.0 + i, dm_fine=1500.0) for i in range(2)]
    cands = rfi + probes

    def is_probe(c):
        return abs(c.dm_fine - 500.0) < 1.0

    kept, dropped = meter_candidates(cands, 5, always_keep_predicate=is_probe)
    assert dropped == 0
    assert len(kept) == 4


def test_meter_always_keep_predicate_swallows_predicate_errors() -> None:
    """A buggy predicate must not sink the metering hot path —
    candidates whose predicate raises are simply not exempted."""

    def bad_predicate(c):
        raise RuntimeError("operator-supplied predicate is broken")

    cands = [_cand(1, float(i)) for i in range(6)]
    kept, dropped = meter_candidates(cands, 3, always_keep_predicate=bad_predicate)
    assert len(kept) == 3
    assert dropped == 3


# ---------------------------------------------------------------------------
# DM-smearing-floor filter (2026-05-30)
# ---------------------------------------------------------------------------


# Production detection-sample period (32 native samples = 1048.576 us);
# the service passes this via geom.sample_period_us.
_T_SEARCH_PROD_US = 1048.576


def test_dm_smear_samples_scales_and_floor_at_zero() -> None:
    # Non-positive DM => 0; smearing grows ~linearly with DM.
    assert dm_smear_samples(0.0) == 0.0
    assert dm_smear_samples(-5.0) == 0.0
    # At the PRODUCTION op-point (1048.576 us/sample) the band-averaged
    # smear at DM 2500 is ~1.8 search-samples (8x chan sum). (At the
    # legacy 524.288 default it is ~3.6 — twice as many samples.)
    s_prod = dm_smear_samples(2500.0, t_search_us=_T_SEARCH_PROD_US)
    assert 1.5 < s_prod < 2.3
    assert abs(dm_smear_samples(524.288) / dm_smear_samples(
        524.288, t_search_us=_T_SEARCH_PROD_US) - 2.0) < 0.01
    # Linear in DM.
    assert abs(dm_smear_samples(5000.0) - 2.0 * dm_smear_samples(2500.0)) < (
        0.2 * dm_smear_samples(2500.0)
    )


def test_filter_disabled_passthrough() -> None:
    cands = [_cand(1, 20.0, dm_fine=2500.0)]
    for frac in (None, 0.0, -1.0):
        kept, dropped = filter_unphysical_narrow(
            cands, frac, t_search_us=_T_SEARCH_PROD_US,
        )
        assert kept is cands
        assert dropped == 0


def test_filter_drops_width1_highdm_keeps_width2() -> None:
    # At the production cadence the smear floor at DM~2538 is ~1.8 samples;
    # with frac 0.6 the threshold is ~1.1, so only the width-1 high-DM
    # detection is unphysical and dropped. Width-2 is CONSISTENT with
    # smearing there (so kept), as is width-4.
    cands = [
        _cand(1, 13.0, dm_fine=2538.0),   # unphysically narrow (drop)
        _cand(2, 12.6, dm_fine=2538.0),   # consistent w/ smearing (keep)
        _cand(4, 11.0, dm_fine=2538.0),   # plausible real burst (keep)
    ]
    kept, dropped = filter_unphysical_narrow(
        cands, 0.6, t_search_us=_T_SEARCH_PROD_US,
    )
    assert dropped == 1
    assert [c.width_samples for c in kept] == [2, 4]


def test_filter_keeps_low_dm_narrow_events() -> None:
    # At low DM the smear floor is sub-sample, so genuine narrow low-DM
    # events must pass untouched even at strict frac.
    cands = [_cand(1, 25.0, dm_fine=50.0), _cand(1, 18.0, dm_fine=200.0)]
    kept, dropped = filter_unphysical_narrow(
        cands, 0.8, t_search_us=_T_SEARCH_PROD_US,
    )
    assert dropped == 0
    assert kept == cands


# ---------------------------------------------------------------------------
# DM-aware noise-color SNR de-rating (2026-06-02 s13.1 fix)
# ---------------------------------------------------------------------------


def test_color_factor_unity_low_dm_and_width1() -> None:
    # Width-1 is never inflated (boxcar SUM over a single sample has no
    # cross-correlation term), regardless of DM.
    assert boxcar_noise_color_factor(2538.0, 1, t_search_us=_T_SEARCH_PROD_US) == 1.0
    # At low DM the smear floor is sub-sample (L <= 1) => no correlation =>
    # factor 1.0 even for wide boxcars. Low/mid-DM detections untouched.
    assert boxcar_noise_color_factor(50.0, 8, t_search_us=_T_SEARCH_PROD_US) == 1.0
    assert boxcar_noise_color_factor(200.0, 4, t_search_us=_T_SEARCH_PROD_US) == 1.0
    # Non-positive DM => 1.0.
    assert boxcar_noise_color_factor(0.0, 4) == 1.0


def test_color_factor_inflates_high_dm_width2() -> None:
    # At DM~2538 the smear floor is ~1.8 samples, so a width-2 boxcar sums
    # correlated samples => score scatter inflated above the white-noise
    # (sqrt(w)) expectation that the sigma-clipped sigma_k converges to.
    f2 = boxcar_noise_color_factor(2538.0, 2, t_search_us=_T_SEARCH_PROD_US)
    assert f2 > 1.0
    # Closed form: L = dm_smear_samples(2538); inflation = 1 + (L-1)/L for
    # w=2 (single k=1 term, weight w-k=1, /w=2 => 2*1*(L-1)/L / 2).
    L = dm_smear_samples(2538.0, t_search_us=_T_SEARCH_PROD_US)
    expected = (1.0 + (L - 1.0) / L) ** 0.5
    assert abs(f2 - expected) < 1e-9
    # Inflation grows with DM (longer correlation length).
    assert boxcar_noise_color_factor(
        5000.0, 2, t_search_us=_T_SEARCH_PROD_US,
    ) > f2


def test_derate_disabled_passthrough() -> None:
    cands = [_cand(2, 13.8, dm_fine=2538.0)]
    for strength in (None, 0.0, -1.0):
        kept, dropped = derate_noise_color(
            cands, strength, 8.0, t_search_us=_T_SEARCH_PROD_US,
        )
        assert kept is cands
        assert dropped == 0


def test_derate_drops_inflated_highdm_noise() -> None:
    # The s13.1 noise family: width-2 high-DM singles whose sigma-clip-
    # inflated SNR sits a little above the floor. De-rating drops them
    # below the floor; a genuinely bright high-DM burst survives, carrying
    # the corrected (lower-but-still-significant) SNR. Magnitudes are
    # derived from the factor itself so the test is robust to the exact
    # smearing length.
    strength = 4.0
    factor = boxcar_noise_color_factor(2538.0, 2, t_search_us=_T_SEARCH_PROD_US)
    applied = 1.0 + strength * (factor - 1.0)
    assert applied > 1.0
    noise = _cand(2, 8.0 * applied * 0.9, dm_fine=2538.0)    # de-rated -> 7.2
    bright = _cand(2, 8.0 * applied * 3.0, dm_fine=2538.0)   # de-rated -> 24
    kept, dropped = derate_noise_color(
        [noise, bright], strength, 8.0, t_search_us=_T_SEARCH_PROD_US,
    )
    assert dropped == 1
    assert len(kept) == 1
    # Survivor is the bright burst with the CORRECTED (lower) SNR.
    assert abs(kept[0].snr - 8.0 * 3.0) < 1e-6
    assert kept[0].snr >= 8.0


def test_derate_leaves_low_dm_and_width1_untouched() -> None:
    # Factor is 1.0 here => SNR unchanged, nothing dropped (identity values).
    cands = [
        _cand(1, 13.8, dm_fine=2538.0),   # width-1 high-DM: factor 1.0
        _cand(4, 9.0, dm_fine=100.0),     # low-DM: factor 1.0
    ]
    kept, dropped = derate_noise_color(
        cands, 2.0, 8.0, t_search_us=_T_SEARCH_PROD_US,
    )
    assert dropped == 0
    assert [c.snr for c in kept] == [13.8, 9.0]


def test_derate_no_floor_keeps_all_but_corrects_snr() -> None:
    # snr_floor None => nothing dropped, but high-DM width-2 SNR is still
    # corrected downward (so C2 sees the true significance).
    cands = [_cand(2, 13.8, dm_fine=2538.0)]
    kept, dropped = derate_noise_color(
        cands, 2.0, None, t_search_us=_T_SEARCH_PROD_US,
    )
    assert dropped == 0
    assert kept[0].snr < 13.8


class _MockStore:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict]] = []

    def put_dict(self, key: str, payload: dict) -> None:
        self.puts.append((key, dict(payload)))


def test_mon_key_shape() -> None:
    assert build_search_compute_mon_key(13, 1) == "/mon/search_rt/13/compute/1"


def test_mon_publisher_averages_window() -> None:
    store = _MockStore()
    pub = SearchComputeMonPublisher(
        search_node_id=2, gpu_half=0, store=store,
    )
    # 16-block window: 4 blocks metered, dropping 2,4,6,8 (sum 20, max 8),
    # total candidates seen 160.
    ok = pub.publish_metering(
        n_blocks=16, n_metered_blocks=4, dropped_sum=20,
        dropped_max=8, cands_sum=160, cap=8,
    )
    assert ok
    assert len(store.puts) == 1
    key, payload = store.puts[0]
    assert key == "/mon/search_rt/2/compute/0"
    assert payload["c1_metering_active"] == 1
    assert payload["c1_metering_frac"] == 0.25
    assert payload["c1_metered_dropped_mean"] == 1.25      # 20/16
    assert payload["c1_metered_dropped_max"] == 8
    assert payload["c1_cands_per_block_mean"] == 10.0      # 160/16
    assert payload["c1_max_candidates_per_block"] == 8
    assert payload["n_blocks"] == 16


def test_mon_publisher_inactive_window() -> None:
    store = _MockStore()
    pub = SearchComputeMonPublisher(
        search_node_id=9, gpu_half=1, store=store,
    )
    pub.publish_metering(
        n_blocks=16, n_metered_blocks=0, dropped_sum=0,
        dropped_max=0, cands_sum=32, cap=8,
    )
    _, payload = store.puts[0]
    assert payload["c1_metering_active"] == 0
    assert payload["c1_metering_frac"] == 0.0
    assert payload["c1_metered_dropped_mean"] == 0.0


def test_mon_publisher_noise_per_kernel_flattening() -> None:
    """2026-06-09: ``publish_noise`` flattens per-kernel σ_k into
    ``s_k_<kernel_id with ':'→'_'>`` payload fields and carries the
    clamp-escape counters."""
    store = _MockStore()
    pub = SearchComputeMonPublisher(
        search_node_id=2, gpu_half=0, store=store,
    )
    ok = pub.publish_noise(
        s_k_median=1.05, s_k_p95=1.42, s_k_max=7.7, s_k_min=0.97,
        n_clamped_high_total=17, n_clamped_high_max_per_kernel=9,
        sigma_max_ratio=4.0, cube_count=200, is_warming_up=False,
        s_k_per_kernel={
            "unit:d1:b1": 1.01,
            "unit:d1:b16": 7.7,
        },
        n_clamp_escapes_total=2,
        clamp_streak_max=64,
    )
    assert ok
    key, payload = store.puts[0]
    assert key == "/mon/search_rt/2/noise/0"
    assert payload["s_k_unit_d1_b1"] == 1.01
    assert payload["s_k_unit_d1_b16"] == 7.7
    assert payload["n_clamp_escapes_total"] == 2
    assert payload["clamp_streak_max"] == 64
    # Rollup fields unchanged.
    assert payload["s_k_median"] == 1.05
    assert payload["n_clamped_high_total"] == 17


def test_mon_publisher_noise_defaults_omit_per_kernel() -> None:
    """Calling publish_noise without the new kwargs (legacy call
    shape) stays valid: no s_k_<kernel> fields, zeroed counters."""
    store = _MockStore()
    pub = SearchComputeMonPublisher(
        search_node_id=2, gpu_half=0, store=store,
    )
    pub.publish_noise(
        s_k_median=1.0, s_k_p95=1.0, s_k_max=1.0, s_k_min=1.0,
        n_clamped_high_total=0, n_clamped_high_max_per_kernel=0,
        sigma_max_ratio=4.0, cube_count=10, is_warming_up=False,
    )
    _, payload = store.puts[0]
    assert payload["n_clamp_escapes_total"] == 0
    assert payload["clamp_streak_max"] == 0
    per_kernel = [
        k for k in payload
        if k.startswith("s_k_") and k not in (
            "s_k_median", "s_k_p95", "s_k_max", "s_k_min",
        )
    ]
    assert per_kernel == []
