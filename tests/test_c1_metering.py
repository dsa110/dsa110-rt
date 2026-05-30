"""Tests for the M7.6 C1→C2 metering: candidate selection
(``meter_candidates``) + the per-half metering mon publisher
(``SearchComputeMonPublisher``)."""

from __future__ import annotations

import os

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import Candidate  # noqa: E402
from dsart.services.search_compute import (  # noqa: E402
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
