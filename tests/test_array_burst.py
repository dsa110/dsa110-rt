"""Unit tests for :mod:`dsart.rfi.array_burst`.

These are CPU-only and synthetic: they build ``S1`` moments directly
from Gamma statistics rather than going through the voltage unpack, so
they run anywhere torch does.

What is actually being pinned:

* the arm split reproduces DSA-110's 47 / 35 core geometry and is
  insensitive to the tolerance;
* the null false-alarm rate is ~0 over a few hundred quiet cubes,
  *including* dead antennas and a realistic gain spread — this is the
  property that broke when dead antennas were left in the sum during
  the offline study (all-96 null 0.978 ± 0.026 vs 1.000 ± 0.0013);
* a broadband burst at the measured amplitude fires;
* a dispersed FRB above DM ≈ 150 does **not** fire, because it lights
  too few coarse bins in any one 2.097 ms sample.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsart.rfi.array_burst import (  # noqa: E402
    BIN_CHANS_DEFAULT,
    GROUP_NAMES,
    ArrayBurstDetector,
    build_groups_from_antpos,
    excision_mask,
)

NANTS = 96
NCHAN = 384
NPOL = 2
M_FINE = 64
N_ACC = 64

#: Dead antenna-pols, mirroring the four voltage indices that carry
#: exactly zero power on the real array.
DEAD_ANTS = (43, 89, 93, 94)


# ---------------------------------------------------------------------------
# Synthetic array
# ---------------------------------------------------------------------------


def _dsa_like_antpos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(E, N, station) for a DSA-110-shaped array: an offset T core of
    47 E-W + 35 N-S antennas plus 14 outriggers."""
    e = np.zeros(NANTS, dtype=np.float64)
    n = np.zeros(NANTS, dtype=np.float64)
    st = np.zeros(NANTS, dtype=np.int64)
    # 47 on the E-W line at N = -295.3, spanning E -190..207
    e[:47] = np.linspace(-190.3, 206.6, 47)
    n[:47] = -295.3
    st[:47] = np.arange(1, 48)
    # 35 on the N-S line at E = +8.2, spanning N -148..146
    e[47:82] = 8.2
    n[47:82] = np.linspace(-148.2, 146.1, 35)
    st[47:82] = np.arange(48, 83)
    # 14 outriggers, far off both axes, stations 103-116
    rng = np.random.default_rng(7)
    e[82:] = rng.uniform(-1200, 1200, 14)
    n[82:] = rng.uniform(-1200, 1200, 14)
    st[82:] = np.arange(103, 117)
    return e, n, st


def _gains(rng: np.random.Generator) -> np.ndarray:
    """Per-(ant, ch, pol) mean auto-power with a realistic spread and
    the usual handful of stone-dead antenna-pols."""
    g = rng.uniform(0.5, 2.0, (NANTS, 1, NPOL)) * np.ones((1, NCHAN, 1))
    # A gentle bandpass shape, so the per-channel reference is doing
    # real work rather than being a constant.
    g = g * (1.0 + 0.15 * np.cos(np.linspace(0, np.pi, NCHAN)))[None, :, None]
    for a in DEAD_ANTS:
        g[a] = 0.0
    return g.astype(np.float64)


def _draw_cube(
    rng: np.random.Generator,
    gain: np.ndarray,
    *,
    excess: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one cube of ``S1`` moments.

    ``S1 = sum over M samples of |E|^2`` is Gamma(M, P/M * M) = Gamma
    with shape ``M`` and mean ``M * P``; the fractional scatter is
    ``1 / sqrt(M)``.

    Args:
        excess: optional ``[N_ACC, NCHAN]`` fractional power excess,
            applied to every antenna and pol (i.e. array-common).

    Returns:
        ``(s1_fine [N_ACC, NANTS, NCHAN, NPOL], s1_full [NANTS, NCHAN, NPOL])``
    """
    p = np.broadcast_to(gain, (N_ACC, NANTS, NCHAN, NPOL)).copy()
    if excess is not None:
        p = p * (1.0 + excess[:, None, :, None])
    s1 = rng.gamma(shape=M_FINE, scale=p / M_FINE, size=p.shape) * M_FINE
    s1 = np.where(p > 0, s1, 0.0)
    return (
        torch.as_tensor(s1, dtype=torch.float32),
        torch.as_tensor(s1.sum(axis=0), dtype=torch.float32),
    )


def _make_detector(**kw) -> ArrayBurstDetector:
    e, n, st = _dsa_like_antpos()
    groups = build_groups_from_antpos(e, n, st)
    kw.setdefault("warmup_cubes", 8)
    return ArrayBurstDetector(groups, n_chan=NCHAN, n_pol=NPOL, **kw)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_arm_split_matches_dsa110_geometry():
    e, n, st = _dsa_like_antpos()
    g = build_groups_from_antpos(e, n, st)
    assert g.names == GROUP_NAMES
    sizes = {nm: int(s) for nm, s in zip(g.names, g.sizes.tolist())}
    assert sizes == {
        "all": 96, "core": 82, "ew_arm": 47, "ns_arm": 35, "outriggers": 14,
    }
    # The arms partition the core exactly — no antenna in both, none
    # left over. The E-W / N-S comparison is meaningless otherwise.
    ew = g.member[:, g.index("ew_arm")].bool()
    ns = g.member[:, g.index("ns_arm")].bool()
    core = g.member[:, g.index("core")].bool()
    assert not bool((ew & ns).any())
    assert bool(((ew | ns) == core).all())


@pytest.mark.parametrize("tol", [5.0, 10.0, 20.0, 30.0, 50.0])
def test_arm_split_insensitive_to_tolerance(tol):
    e, n, st = _dsa_like_antpos()
    g = build_groups_from_antpos(e, n, st, arm_tol_m=tol)
    assert int(g.sizes[g.index("ew_arm")]) == 47
    assert int(g.sizes[g.index("ns_arm")]) == 35


def test_degenerate_split_raises():
    """A single-arm array must fail loudly rather than silently
    publish an arm ratio with an empty denominator."""
    e = np.linspace(-200, 200, NANTS)
    n = np.zeros(NANTS)
    st = np.arange(1, NANTS + 1)
    with pytest.raises(ValueError, match="degenerate arm split"):
        build_groups_from_antpos(e, n, st, arm_tol_m=20.0)


# ---------------------------------------------------------------------------
# Null behaviour
# ---------------------------------------------------------------------------


def test_null_false_alarm_rate_is_negligible():
    """Quiet data, realistic gain spread, four dead antenna-pols: the
    gate must essentially never fire."""
    rng = np.random.default_rng(20260908)
    gain = _gains(rng)
    det = _make_detector()
    n_fired = 0
    n_samples = 0
    for _ in range(60):
        s1f, s1c = _draw_cube(rng, gain)
        r = det.detect(s1f, s1c)
        # A NaN would make every comparison False and fake a perfect
        # false-alarm rate. That is exactly how the dead-antenna 0/0
        # bug hid, so check explicitly.
        assert not bool(torch.isnan(r.z).any()), "NaN in z"
        assert not bool(torch.isnan(r.band_frac).any()), "NaN in band_frac"
        if not r.warmup:
            core = r.fired[:, det.groups.index("core"), :]
            n_fired += int(core.sum())
            n_samples += core.numel()
    assert n_samples > 3_000
    # Zero fires in ~3300 samples puts the 95% upper limit at ~9e-4.
    assert n_fired == 0, f"{n_fired}/{n_samples} samples fired on quiet data"


def test_dead_antennas_are_excluded_not_downweighted():
    """n_live must drop by exactly the dead count, and the normalised
    group power must still sit at 1.0 — the failure mode that put the
    offline all-96 null at 0.978 +- 0.026."""
    rng = np.random.default_rng(11)
    gain = _gains(rng)
    det = _make_detector(warmup_cubes=0)
    for _ in range(20):
        s1f, s1c = _draw_cube(rng, gain)
        r = det.detect(s1f, s1c)
    n_live = r.n_live[det.groups.index("all")]
    assert torch.allclose(
        n_live, torch.full_like(n_live, float(NANTS - len(DEAD_ANTS))),
    )
    # band_frac is (P/mu - 1); on quiet data it is the scatter about
    # zero, so its own scatter is what the sigma is derived from.
    bf = r.band_frac[:, det.groups.index("core"), :]
    assert abs(float(bf.mean())) < 5e-3
    # Predicted fractional noise on the core sum:
    # 1/sqrt(M * n_live * NCHAN). Allow a factor of 2 either way.
    pred = 1.0 / np.sqrt(M_FINE * (NANTS - len(DEAD_ANTS)) * NCHAN)
    got = float(bf.std())
    assert 0.5 * pred < got < 2.0 * pred, (pred, got)


# ---------------------------------------------------------------------------
# Response to a real burst
# ---------------------------------------------------------------------------


def _broadband_excess(rng, amp: float, duty: float) -> np.ndarray:
    """[N_ACC, NCHAN] array-common flat excess in a random subset of
    samples — the 260812imek burst shape."""
    exc = np.zeros((N_ACC, NCHAN))
    hot = rng.random(N_ACC) < duty
    exc[hot, :] = amp
    return exc


def test_broadband_burst_fires():
    """A 1.45%-of-band-power flat burst in 5% of samples — the measured
    260812imek amplitude — must be caught."""
    rng = np.random.default_rng(4242)
    gain = _gains(rng)
    det = _make_detector()
    for _ in range(20):                       # settle the baseline
        det.detect(*_draw_cube(rng, gain))
    exc = _broadband_excess(rng, amp=0.0145, duty=0.05)
    s1f, s1c = _draw_cube(rng, gain, excess=exc)
    r = det.detect(s1f, s1c)
    hot = torch.as_tensor(exc[:, 0] > 0)
    core = det.groups.index("core")
    fired = r.fired[:, core, 0]
    # Every injected sample caught, and nothing else.
    assert bool((fired[hot]).all()), r.z[hot, core, 0]
    assert not bool((fired[~hot]).any())
    # And it should be comfortably significant, not marginal.
    assert float(r.z[hot, core, 0].median()) > 10.0


def test_dispersed_frb_does_not_fire():
    """An FRB at DM 300 sweeps 10.5 ms across the sub-band, so in any
    one 2.097 ms sample it lights ~20% of the band — far below the
    75% occupancy the gate demands. It must survive."""
    rng = np.random.default_rng(99)
    gain = _gains(rng)
    det = _make_detector()
    for _ in range(20):
        det.detect(*_draw_cube(rng, gain))

    # Sweep: 5 consecutive samples, each lighting a fifth of the band,
    # and BRIGHT (10x the RFI burst) so the only thing rejecting it is
    # the occupancy term.
    exc = np.zeros((N_ACC, NCHAN))
    span = NCHAN // 5
    for i in range(5):
        exc[20 + i, i * span:(i + 1) * span] = 0.145
    s1f, s1c = _draw_cube(rng, gain, excess=exc)
    r = det.detect(s1f, s1c)
    core = det.groups.index("core")
    assert not bool(r.fired[:, core, :].any()), (
        "a dispersed FRB was flagged; occupancy gate is not working"
    )
    # The band-summed significance IS large — that is the point of the
    # occupancy term, and why detect_k alone would not be safe.
    assert float(r.z[20:25, core, 0].max()) > 6.0


def test_zero_dm_burst_fires_and_is_reported_as_such():
    """The flip side: a dispersion-free burst of the same brightness
    lights the whole band and IS flagged. Documented behaviour."""
    rng = np.random.default_rng(1234)
    gain = _gains(rng)
    det = _make_detector()
    for _ in range(20):
        det.detect(*_draw_cube(rng, gain))
    exc = np.zeros((N_ACC, NCHAN))
    exc[20, :] = 0.145
    r = det.detect(*_draw_cube(rng, gain, excess=exc))
    core = det.groups.index("core")
    assert bool(r.fired[20, core, :].all())
    assert float(r.occupancy[20, core, 0]) > 0.9


# ---------------------------------------------------------------------------
# Shapes, warmup and the arm diagnostic
# ---------------------------------------------------------------------------


def test_shapes_and_warmup_gate():
    rng = np.random.default_rng(5)
    gain = _gains(rng)
    det = _make_detector(warmup_cubes=3)
    n_bin = NCHAN // BIN_CHANS_DEFAULT
    n_group = len(GROUP_NAMES)
    for i in range(5):
        r = det.detect(*_draw_cube(rng, gain))
        assert r.fired.shape == (N_ACC, n_group, NPOL)
        assert r.z.shape == (N_ACC, n_group, NPOL)
        assert r.coarse_z.shape == (N_ACC, n_group, n_bin, NPOL)
        assert r.group_spec.shape == (n_group, NCHAN, NPOL)
        assert r.n_live.shape == (n_group, NPOL)
        if i < 3:
            assert r.warmup
            assert r.time_chan_mask is None
            assert not bool(r.fired.any())
        else:
            assert not r.warmup
            assert r.time_chan_mask.shape == (N_ACC, NCHAN, NPOL)


def test_arm_asymmetry_is_visible_in_band_frac():
    """A burst injected only on the E-W arm must show up as an arm
    asymmetry in band_frac — the local vs far-field discriminant. It
    is a diagnostic, not part of the gate."""
    rng = np.random.default_rng(808)
    gain = _gains(rng)
    e, n, st = _dsa_like_antpos()
    groups = build_groups_from_antpos(e, n, st)
    det = ArrayBurstDetector(
        groups, n_chan=NCHAN, n_pol=NPOL, warmup_cubes=8,
    )
    for _ in range(20):
        det.detect(*_draw_cube(rng, gain))

    # Hand-build a cube where only the E-W antennas see the burst.
    p = np.broadcast_to(gain, (N_ACC, NANTS, NCHAN, NPOL)).copy()
    ew = groups.member[:, groups.index("ew_arm")].numpy().astype(bool)
    p[20][ew] *= 1.05
    s1 = rng.gamma(shape=M_FINE, scale=p / M_FINE, size=p.shape) * M_FINE
    s1 = np.where(p > 0, s1, 0.0)
    r = det.detect(
        torch.as_tensor(s1, dtype=torch.float32),
        torch.as_tensor(s1.sum(axis=0), dtype=torch.float32),
    )
    bf = r.band_frac[20, :, 0]
    f_ew = float(bf[groups.index("ew_arm")])
    f_ns = float(bf[groups.index("ns_arm")])
    assert f_ew > 0.04, f_ew
    assert abs(f_ns) < 0.01, f_ns
    assert f_ew / max(abs(f_ns), 1e-4) > 4.0


def test_no_host_sync_in_detect():
    """The hot path must not synchronise. On CPU we cannot observe a
    sync directly, so pin the contract structurally: detect() returns
    tensors and a python bool, and never a python float derived from
    device data."""
    rng = np.random.default_rng(3)
    gain = _gains(rng)
    det = _make_detector(warmup_cubes=0)
    r = det.detect(*_draw_cube(rng, gain))
    for name in ("fired", "z", "band_frac", "occupancy", "coarse_z",
                 "group_spec", "n_live"):
        assert torch.is_tensor(getattr(r, name)), name
    assert isinstance(r.warmup, bool)


def test_baseline_is_read_before_the_cube_is_folded_in():
    """z must be measured against the PRE-update EMA.

    The update is in place, so folding first would let a burst move
    the reference it is then compared against. Pinned with a short
    time constant, where the effect is large enough to see: with
    ema_cubes=2 the pre-update mean is the previous cube's, so a step
    change of known size must show its FULL size in z, not half of it.
    """
    rng = np.random.default_rng(31)
    gain = _gains(rng)
    det = _make_detector(warmup_cubes=0, ema_cubes=2)
    for _ in range(12):                      # settle on quiet data
        det.detect(*_draw_cube(rng, gain))
    core = det.groups.index("core")

    # A whole cube lifted by a known amount, in every sample.
    exc = np.full((N_ACC, NCHAN), 0.02)
    r = det.detect(*_draw_cube(rng, gain, excess=exc))
    got = float(np.median(r.band_frac[:, core, 0].numpy()))
    # band_frac = P/mu - 1 with mu the PREVIOUS baseline, so a +2%
    # cube reads +2%. Had the update run first, mu would already have
    # absorbed half the step (alpha=0.5) and this would read ~1%.
    assert 0.017 < got < 0.023, got


# ---------------------------------------------------------------------------
# Band-limited (impulsive) gate — 2026-09-14
# ---------------------------------------------------------------------------
#
# This gate exists because the broadband one collapses its 24 per-bin
# significances into a single occupancy number and demands >= 90 % of
# them, so an impulse confined to a few bins is rejected. Measured on
# archive dumps that population is real, is array-common, and nothing
# else in the chain can reach it: ~1 sigma in any one antenna, 8-10
# sigma in the core sum, and SK does not respond at all because an
# impulse that fills the 2.097 ms accumulation is Gaussian inside it.
#
# The tests below pin the two halves of the safety argument, both of
# which are needed: a bin must have fired REPEATEDLY (a pulse visits
# each bin once), and the contributing cubes must have been BAND-
# LIMITED (a pulse lights every bin as it sweeps).


def _bin_detector(**kw) -> ArrayBurstDetector:
    kw.setdefault("bin_mode", "flag")
    kw.setdefault("bin_persist_n", 3)
    kw.setdefault("bin_window_cubes", 8)
    kw.setdefault("warmup_cubes", 2)
    return _make_detector(**kw)


def _narrow_excess(bins, samples, amp: float) -> np.ndarray:
    """[N_ACC, NCHAN] array-common excess confined to whole coarse bins."""
    exc = np.zeros((N_ACC, NCHAN))
    for b in bins:
        lo = b * BIN_CHANS_DEFAULT
        exc[np.asarray(samples)[:, None],
            np.arange(lo, lo + BIN_CHANS_DEFAULT)[None, :]] = amp
    return exc


def test_bin_gate_off_by_default_and_computes_nothing():
    rng = np.random.default_rng(1)
    gain = _gains(rng)
    det = _make_detector()
    r = det.detect(*_draw_cube(rng, gain))
    assert det.bin_mode == "off"
    assert r.bin_fired is None
    assert r.bin_armed is None
    assert r.bin_chan_mask is None


def test_bin_gate_arms_only_after_repeated_hits():
    """The causal read: the first cubes must fire and NOT excise."""
    rng = np.random.default_rng(11)
    gain = _gains(rng)
    det = _bin_detector()
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    excised, armed_at = [], None
    for c in range(12):
        r = det.detect(*_draw_cube(rng, gain, excess=exc))
        n = 0 if r.bin_chan_mask is None else int(r.bin_chan_mask.sum())
        excised.append(n)
        if armed_at is None and n > 0:
            armed_at = c
        assert bool(r.bin_fired[:, det.groups.index("core")].any()) or r.warmup

    assert armed_at is not None, "never armed on repeating band-limited RFI"
    # warmup is 2 cubes and persist_n is 3, so the earliest possible
    # arming is cube 4 (cubes 2 and 3 fold, cube 4 reads the history).
    assert armed_at >= 4, f"armed too early at cube {armed_at}"
    assert all(n == 0 for n in excised[:armed_at])
    assert all(n > 0 for n in excised[armed_at:])


def test_bin_gate_excises_only_the_offending_bins():
    rng = np.random.default_rng(12)
    gain = _gains(rng)
    det = _bin_detector()
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    for _ in range(8):
        r = det.detect(*_draw_cube(rng, gain, excess=exc))
    m = r.bin_chan_mask
    assert m is not None and bool(m.any())
    # Exactly bins 5 and 6, exactly samples 10 and 11, both pols.
    want_ch = np.zeros(NCHAN, dtype=bool)
    for b in (5, 6):
        want_ch[b * BIN_CHANS_DEFAULT:(b + 1) * BIN_CHANS_DEFAULT] = True
    got_ch = m.any(dim=0).any(dim=-1).cpu().numpy()
    assert np.array_equal(got_ch, want_ch), "wrong channels excised"
    got_t = m.any(dim=1).any(dim=-1).cpu().numpy()
    want_t = np.zeros(N_ACC, dtype=bool)
    want_t[[10, 11]] = True
    assert np.array_equal(got_t, want_t), "wrong samples excised"


def test_bin_gate_ignores_a_one_off_event():
    """A single band-limited event — the shape a pulse has — fires the
    per-bin test but must never arm, because it has no history."""
    rng = np.random.default_rng(13)
    gain = _gains(rng)
    det = _bin_detector()
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    total = 0
    for c in range(12):
        use = exc if c == 9 else None
        r = det.detect(*_draw_cube(rng, gain, excess=use))
        if r.bin_chan_mask is not None:
            total += int(r.bin_chan_mask.sum())
    assert total == 0, f"a one-off event was excised ({total} cells)"


def test_bin_gate_does_not_excise_a_repeating_dispersed_pulse():
    """The half of the safety argument the revisit counter alone does
    NOT give you.

    A dispersed sweep lights every bin once as it crosses the band, so
    a bright source recurring inside the arming window — a giant-pulse
    train, a repeater — would otherwise accumulate exactly the history
    the gate reads and arm its own excision. The cube only counts
    towards arming if it was band-limited, which a sweep never is.
    """
    rng = np.random.default_rng(14)
    gain = _gains(rng)
    det = _bin_detector()
    # A bright sweep in EVERY cube: 5 samples, each lighting a fifth of
    # the band, at 10x the measured RFI amplitude.
    exc = np.zeros((N_ACC, NCHAN))
    span = NCHAN // 5
    for i in range(5):
        exc[20 + i, i * span:(i + 1) * span] = 0.145
    total = 0
    fired_any = False
    for _ in range(14):
        r = det.detect(*_draw_cube(rng, gain, excess=exc))
        if r.bin_fired is not None:
            fired_any |= bool(r.bin_fired[:, det.groups.index("core")].any())
        if r.bin_chan_mask is not None:
            total += int(r.bin_chan_mask.sum())
    assert fired_any, "the sweep was not even detected; test is vacuous"
    assert total == 0, (
        f"a repeating dispersed pulse was excised ({total} cells); the "
        f"band-limited arming gate is not working"
    )


def test_bin_monitor_mode_computes_but_the_caller_excises_nothing():
    rng = np.random.default_rng(15)
    gain = _gains(rng)
    det = _bin_detector(bin_mode="monitor")
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    saw_mask = False
    for _ in range(10):
        r = det.detect(*_draw_cube(rng, gain, excess=exc))
        if r.bin_chan_mask is not None and bool(r.bin_chan_mask.any()):
            saw_mask = True
        # monitor means the CALLER does not arm it
        assert excision_mask(r, broadband=False, band_limited=False) is None
    assert saw_mask, "monitor mode computed nothing"


def test_excision_mask_ors_the_two_gates():
    rng = np.random.default_rng(16)
    gain = _gains(rng)
    det = _bin_detector()
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    for _ in range(8):
        r = det.detect(*_draw_cube(rng, gain, excess=exc))
    both = excision_mask(r, broadband=True, band_limited=True)
    only_bin = excision_mask(r, broadband=False, band_limited=True)
    assert both is not None and only_bin is not None
    assert bool((only_bin <= both).all())
    assert excision_mask(r, broadband=False, band_limited=False) is None


@pytest.mark.parametrize("kw", [
    {"bin_mode": "nonsense"},
    {"bin_window_cubes": 0},
    {"bin_persist_n": 0},
    {"bin_persist_n": 9, "bin_window_cubes": 8},
    {"bin_arm_max_frac": 0.0},
    {"bin_arm_max_frac": 1.5},
])
def test_bin_gate_rejects_bad_configuration(kw):
    args = {"bin_mode": "flag"}
    args.update(kw)
    with pytest.raises(ValueError):
        _make_detector(**args)


def _autos_from_s1(s1_fine: "torch.Tensor", s1_full: "torch.Tensor"):
    """Build an :class:`AutoSpectra` from the M=64 moments.

    ``s2`` is set to ``2 S1^2 / (M+1)``, the value that makes SK
    exactly 1, so the SK detector contributes nothing and the test is
    about the array-burst path alone.
    """
    from dsart.rfi.autos import AutoSpectra

    s1: dict[int, torch.Tensor] = {}
    s2: dict[int, torch.Tensor] = {}
    for m in (64, 256, 1024, 4096):
        k = m // M_FINE
        acc = s1_fine.reshape(
            N_ACC // k, k, *s1_fine.shape[1:],
        ).sum(dim=1)
        if m == 4096:
            acc = s1_full.unsqueeze(0)
        s1[m] = acc.contiguous()
        s2[m] = (2.0 * acc * acc / (m + 1.0)).contiguous()
    return AutoSpectra(s1=s1, s2=s2)


def test_bin_gate_runs_with_the_broadband_gate_off():
    """`array_burst_mode=off` + `bin_mode=flag` is a legitimate
    configuration and must not silently do nothing."""
    from dsart.rfi import RFIFlagger

    rng = np.random.default_rng(17)
    gain = _gains(rng)
    e, n, st = _dsa_like_antpos()
    det = ArrayBurstDetector(
        build_groups_from_antpos(e, n, st), n_chan=NCHAN, n_pol=NPOL,
        warmup_cubes=2, bin_mode="flag", bin_persist_n=3,
        bin_window_cubes=8,
    )
    flg = RFIFlagger(
        flagants_path=None, m_values=(64, 256, 1024, 4096),
        warmup_cubes=0, run_sum_threshold=False,
        array_burst=det, array_burst_mode="off",
    )
    exc = _narrow_excess((5, 6), (10, 11), 0.45)
    saw = 0
    for _ in range(9):
        s1f, s1c = _draw_cube(rng, gain, excess=exc)
        # RFIFlagger wants voltages; feed the moments it would build.
        res = flg.flag_block(
            None, None,
            autos_override=_autos_from_s1(s1f, s1c),
        )
        ab = res.array_burst
        assert ab is not None, "detector never ran with broadband off"
        if ab.bin_chan_mask is not None:
            saw += int(ab.bin_chan_mask.sum())
    assert saw > 0, "band-limited gate produced nothing"
