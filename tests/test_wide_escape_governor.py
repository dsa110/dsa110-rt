"""C1 wide-escape governor (2026-10-05).

The C1->C2 width cap has a brightness escape (``c1.max_c1c2_width_snr_
escape``) so a bright wide burst is not discarded. Over 2026-10-03..05
that escape was the door for RFI / satellite-transit floods: 8.5% of the
time produced 99.4% of all C2 clusters, 91% of them b64 above the escape
threshold, and their volume tripped C2's cluster-rate limiter 5.5% of
the time -- during which no dump at all could fire. The governor keeps
the brightest escape per block and spends a per-half token bucket, so an
isolated wide burst always ships and a sustained flood becomes a trickle.
"""
from __future__ import annotations

import pytest

from dsart.common.contracts import Candidate, CandidateFlags
from dsart.services.c1_emit import C1EmitConfig, WideEscapeGovernor


def _c(snr: float, width: int = 64, l: float = 10.0, m: float = 20.0) -> Candidate:
    return Candidate(
        l=l, m=m, dm_fine=600.0, dm_idx=3, event_specnum=1024,
        width_samples=width, kernel_id=f"unit:d1:b{width}", snr=snr,
        detector_version="v1.M5", flags=int(CandidateFlags.NONE),
        search_node_id=0, gpu_half=0,
    )


def test_an_isolated_bright_wide_burst_always_ships() -> None:
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    adm, thr = g.admit([_c(85.0)], now_s=0.0)
    assert [c.snr for c in adm] == [85.0] and thr == 0


def test_brightest_escape_per_block_wins_sidelobes_are_dropped() -> None:
    """A real wide burst's PSF sidelobes are fainter than its main lobe."""
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    adm, thr = g.admit([_c(24.0, l=40), _c(150.0), _c(31.0, m=90)], now_s=0.0)
    assert [c.snr for c in adm] == [150.0] and thr == 2


def test_a_burst_straddling_the_cube_overlap_still_ships_twice() -> None:
    """Cubes overlap by 64 samples, so one burst can appear in two."""
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    assert len(g.admit([_c(40.0)], now_s=0.0)[0]) == 1
    assert len(g.admit([_c(41.0)], now_s=0.2)[0]) == 1


def test_a_sustained_flood_is_cut_to_a_trickle() -> None:
    """The measured flood shape: escapes in every block for minutes.

    At ~5 blocks/s for 10 minutes, with several escapes per block, the
    governor ships the bucket (3) plus one per refill period (600/20 =
    30) -- ~33 of ~9000, i.e. the ~2600 clusters/min floods drop to a
    few per minute per half.
    """
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    shipped = offered = 0
    t = 0.0
    while t < 600.0:
        block = [_c(30.0 + k, l=k * 7.0) for k in range(3)]
        adm, _ = g.admit(block, now_s=t)
        shipped += len(adm)
        offered += len(block)
        t += 0.2013  # one cube per 201.3 ms
    assert offered > 8000
    assert 30 <= shipped <= 36, shipped


def test_the_bucket_refills_after_the_flood_ends() -> None:
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    for k in range(10):
        g.admit([_c(30.0)], now_s=0.2 * k)
    assert g.admit([_c(30.0)], now_s=2.1)[1] == 1          # drained
    adm, thr = g.admit([_c(90.0)], now_s=2.1 + 3 * 20.0)   # quiet for 60 s
    assert len(adm) == 1 and thr == 0
    assert g.tokens == pytest.approx(2.0, abs=1e-6)


def test_tokens_never_exceed_the_burst() -> None:
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    g.admit([], now_s=0.0)
    g.admit([], now_s=1e6)
    assert g.tokens == pytest.approx(3.0)


def test_an_empty_block_costs_nothing() -> None:
    g = WideEscapeGovernor(per_block=1, burst=3, refill_s=20.0)
    assert g.admit([], now_s=0.0) == ([], 0)
    assert g.tokens == pytest.approx(3.0)


def test_clock_going_backwards_does_not_mint_tokens() -> None:
    g = WideEscapeGovernor(per_block=1, burst=1, refill_s=20.0)
    g.admit([_c(30.0)], now_s=100.0)
    assert g.admit([_c(30.0)], now_s=50.0)[1] == 1


@pytest.mark.parametrize("kw", [dict(per_block=0), dict(burst=0), dict(refill_s=0.0)])
def test_bad_parameters_are_rejected(kw) -> None:
    args = dict(per_block=1, burst=3, refill_s=20.0)
    args.update(kw)
    with pytest.raises(ValueError):
        WideEscapeGovernor(**args)


def test_governor_defaults_off_in_the_dataclass() -> None:
    cfg = C1EmitConfig(host="h", port=1, search_node_id=1, gpu_half=0)
    assert cfg.max_width_escape_per_block is None


def test_yaml_keys_reach_the_config() -> None:
    import inspect

    from dsart.services import search_compute

    src = inspect.getsource(search_compute._build_search_config_from_yaml)
    for key in ("max_c1c2_width_escape_per_block",
                "max_c1c2_width_escape_burst",
                "max_c1c2_width_escape_refill_s"):
        assert f'c1.get("{key}"' in src, key


def test_the_service_applies_it_only_to_escapes() -> None:
    """Candidates within the cap must never be touched by the governor."""
    import inspect

    from dsart.services import search_compute

    src = inspect.getsource(search_compute.SearchComputeService._submit_c1_batch)
    assert "self._wide_escape_gov.admit(" in src
    assert "kept = narrow + admitted" in src
