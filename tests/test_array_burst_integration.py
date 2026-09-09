"""Integration tests for the array-burst detector's wiring.

Covers the seams the unit tests do not: the time-resolved voltage
zero-fill, the shm v2 round trip, and the slow-path settings parser.
CPU-only.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsart.rfi import FlagSourceBit                              # noqa: E402
from dsart.services.rfi_mon_shm import (                        # noqa: E402
    RFIMonShmReader,
    RFIMonShmWriter,
    group_section_bytes,
)
from dsart.services.rfi_window import RFIWindow                 # noqa: E402

NCHAN, NTP, NPOL, NPKT, NANTS = 8, 2, 2, 16, 4
N_ACC = 4


def _voltages(fill: float = 1.0):
    """(real, imag) in the M2 GEMM layout, contiguous."""
    shape = (NCHAN, NTP, NPOL, NPKT, NANTS)
    r = torch.full(shape, fill, dtype=torch.float32)
    i = torch.full(shape, fill, dtype=torch.float32)
    return r, i


# ---------------------------------------------------------------------------
# Source-tag bit
# ---------------------------------------------------------------------------


def test_source_tag_bits_are_all_distinct_and_fit_uint8():
    """No two detectors may share a bit, and the whole set must fit in
    the uint8 source-tag plane.

    Pinned generally rather than against a hardcoded value: this test
    was originally written asserting ARRAY_BURST == 32, and it caught
    a real collision when main independently took bit 5 for the
    persistence latch. The next detector added should fail here too if
    it reuses a bit, not silently alias an existing one — the monitor
    decomposes these bitwise, so an alias would report one detector's
    flags as another's.
    """
    bits = [b for b in FlagSourceBit if int(b) != 0]
    values = [int(b) for b in bits]
    assert len(set(values)) == len(values), (
        "duplicate source-tag bits: %s"
        % {b.name: int(b) for b in bits}
    )
    for v in values:
        assert v & (v - 1) == 0, "not a single bit: %d" % v
    combined = 0
    for v in values:
        assert combined & v == 0
        combined |= v
    assert combined <= 255, "source tags no longer fit in uint8"
    assert int(FlagSourceBit.ARRAY_BURST) & int(FlagSourceBit.PERSISTENCE) == 0


# ---------------------------------------------------------------------------
# Time-resolved zero-fill
# ---------------------------------------------------------------------------


def test_time_mask_zeroes_only_its_own_accumulations():
    """The whole point of the time axis: a burst in one 2.097 ms
    sample must cost that sample, not the whole 134.2 ms cube."""
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    r, i = _voltages()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    tmask = torch.zeros(N_ACC, NCHAN, NPOL, dtype=torch.bool)
    tmask[1] = True                                  # flag accumulation 1
    apply_rfi_mask_to_voltages(r, i, cube, time_chan_mask=tmask)

    pkts_per_acc = NPKT // N_ACC
    v = r.view(NCHAN, NTP, NPOL, N_ACC, pkts_per_acc, NANTS)
    assert float(v[:, :, :, 1].abs().max()) == 0.0
    for k in (0, 2, 3):
        assert float(v[:, :, :, k].min()) == 1.0
    # And both t_sub values within the flagged accumulation went, since
    # native_t = 2*packet + t_sub and an accumulation spans both.
    assert float(v[:, 0, :, 1].abs().max()) == 0.0
    assert float(v[:, 1, :, 1].abs().max()) == 0.0


def test_cube_and_time_masks_compose():
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    r, i = _voltages()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    cube[2] = True                                   # antenna 2, all chans
    tmask = torch.zeros(N_ACC, NCHAN, NPOL, dtype=torch.bool)
    tmask[0, 3, :] = True                            # acc 0, channel 3
    apply_rfi_mask_to_voltages(r, i, cube, time_chan_mask=tmask)
    v = r.view(NCHAN, NTP, NPOL, N_ACC, NPKT // N_ACC, NANTS)
    assert float(v[..., 2].abs().max()) == 0.0       # dead antenna
    assert float(v[3, :, :, 0].abs().max()) == 0.0   # flagged sample
    assert float(v[3, :, :, 1, :, 0].min()) == 1.0   # neighbouring acc kept
    assert float(v[4, :, :, 0, :, 0].min()) == 1.0   # neighbouring chan kept


def test_imag_is_masked_too():
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    r, i = _voltages()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    tmask = torch.zeros(N_ACC, NCHAN, NPOL, dtype=torch.bool)
    tmask[2] = True
    apply_rfi_mask_to_voltages(r, i, cube, time_chan_mask=tmask)
    vi = i.view(NCHAN, NTP, NPOL, N_ACC, NPKT // N_ACC, NANTS)
    assert float(vi[:, :, :, 2].abs().max()) == 0.0


def test_non_dividing_n_acc_raises():
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    r, i = _voltages()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    bad = torch.zeros(5, NCHAN, NPOL, dtype=torch.bool)   # 5 ∤ 16
    with pytest.raises(ValueError, match="does not divide"):
        apply_rfi_mask_to_voltages(r, i, cube, time_chan_mask=bad)


def test_non_contiguous_voltages_raise_rather_than_silently_no_op():
    """A reshape-copy would discard the flags without a word. The
    function must refuse instead."""
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    _, i = _voltages()
    # A permuted view has the right shape but the wrong strides, so
    # splitting the packet axis would have to copy.
    big = torch.ones(NANTS, NCHAN, NTP, NPOL, NPKT, dtype=torch.float32)
    r = big.permute(1, 2, 3, 4, 0)
    assert r.shape == (NCHAN, NTP, NPOL, NPKT, NANTS)
    assert not r.is_contiguous()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    tmask = torch.zeros(N_ACC, NCHAN, NPOL, dtype=torch.bool)
    with pytest.raises(ValueError, match="contiguous"):
        apply_rfi_mask_to_voltages(r, i, cube, time_chan_mask=tmask)


def test_no_time_mask_is_the_old_behaviour():
    from dsart.services.corr_fast_integration import (
        apply_rfi_mask_to_voltages,
    )
    r, i = _voltages()
    cube = torch.zeros(NANTS, NCHAN, NPOL, dtype=torch.bool)
    cube[1, 2, 0] = True
    apply_rfi_mask_to_voltages(r, i, cube)
    assert float(r[2, :, 0, :, 1].abs().max()) == 0.0
    assert float(r[2, :, 1, :, 1].min()) == 1.0


# ---------------------------------------------------------------------------
# shm v2
# ---------------------------------------------------------------------------


def _window(*, n_ants, n_chan_ds, n_pol, w, g, n_acc, **kw):
    sc = (0.1, 0.2, 0.15)
    masks = {
        "mask_count_" + k: np.zeros((n_ants, n_chan_ds, n_pol), np.uint8)
        for k in ("final", "sk", "bp", "grp", "sumthr", "fa")
    }
    return RFIWindow(
        block_n_start=1, block_n_end=w, n_cubes=w, n_cubes_warmup=0,
        s1_full_mean=np.ones((n_ants, n_chan_ds, n_pol), np.float32),
        total_flag_fraction=sc, bandpass_channel_fraction=sc,
        ant_fraction_flagged=sc, frac_sk=sc, frac_bp=sc, frac_grp=sc,
        frac_sumthr=sc, frac_fa=sc, **masks, **kw
    )


def test_shm_v2_round_trip_is_exact(tmp_path, monkeypatch):
    import dsart.services.rfi_mon_shm as shm
    monkeypatch.setattr(shm, "_SHM_DIR", str(tmp_path))
    n_ants, n_chan_ds, n_pol, w, g, n_acc = 8, 6, 2, 4, 5, 4
    t = w * n_acc
    rng = np.random.default_rng(0)
    z = rng.normal(size=(t, g, n_pol)).astype(np.float32)
    bf = rng.normal(size=(t, g, n_pol)).astype(np.float32)
    fired = (z > 1.0).astype(np.uint8)
    spec = rng.normal(1.0, 0.01, (g, n_chan_ds, n_pol)).astype(np.float32)
    live = rng.uniform(1, 96, (g, n_pol)).astype(np.float32)
    names = ("all", "core", "ew_arm", "ns_arm", "outriggers")

    win = _window(
        n_ants=n_ants, n_chan_ds=n_chan_ds, n_pol=n_pol, w=w, g=g,
        n_acc=n_acc,
        array_burst_mode="flag", array_burst_flag_group="ew_arm",
        group_names=names,
        group_sizes=np.array([96, 82, 47, 35, 14], np.int32),
        group_z=z, group_band_frac=bf, group_fired=fired,
        group_spec_mean=spec, group_n_live=live, n_acc_per_cube=n_acc,
    )
    writer = shm.RFIMonShmWriter(
        cn_id=4242, n_ants=n_ants, n_chan_ds=n_chan_ds, n_pol=n_pol,
        window_size=w, freq_downsample=1, n_slots=3,
        n_group=g, n_acc_per_cube=n_acc, group_names=names,
        array_burst_flag_group="ew_arm",
    )
    try:
        writer.publish(win)
        reader = shm.RFIMonShmReader(4242)
        assert reader.version == 2
        assert reader.n_group == g
        assert reader.n_acc_per_cube == n_acc
        rec = reader.read_latest()
        assert np.array_equal(rec.group_z, z)
        assert np.array_equal(rec.group_band_frac, bf)
        assert np.array_equal(rec.group_fired, fired)
        assert np.array_equal(rec.group_spec_mean, spec)
        assert np.array_equal(rec.group_n_live, live)
        assert list(rec.group_sizes) == [96, 82, 47, 35, 14]
        assert rec.array_burst_mode == "flag"
        assert rec.array_burst_flag_group == "ew_arm"
        # The pre-existing planes must be untouched by the new section.
        assert rec.s1_full_mean.shape == (n_ants, n_chan_ds, n_pol)
        assert rec.scalars["total_flag_fraction"][0] == pytest.approx(0.1)
        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_shm_without_groups_is_still_valid(tmp_path, monkeypatch):
    """n_group=0 must produce a record with no group section and no
    group fields — the detector-off case."""
    import dsart.services.rfi_mon_shm as shm
    monkeypatch.setattr(shm, "_SHM_DIR", str(tmp_path))
    n_ants, n_chan_ds, n_pol, w = 8, 6, 2, 4
    win = _window(
        n_ants=n_ants, n_chan_ds=n_chan_ds, n_pol=n_pol, w=w, g=0, n_acc=0,
    )
    writer = shm.RFIMonShmWriter(
        cn_id=4243, n_ants=n_ants, n_chan_ds=n_chan_ds, n_pol=n_pol,
        window_size=w, freq_downsample=1, n_slots=2,
    )
    try:
        writer.publish(win)
        reader = shm.RFIMonShmReader(4243)
        rec = reader.read_latest()
        assert reader.n_group == 0
        assert rec.group_z is None
        assert rec.array_burst_mode == "off"
        assert rec.s1_full_mean.shape == (n_ants, n_chan_ds, n_pol)
        reader.close()
    finally:
        writer.close()
        writer.unlink()


def test_group_section_size_is_what_we_claim():
    """96,040 B at the production op-point — quoted in the ABI docs."""
    assert group_section_bytes(
        n_group=5, n_acc_total=16 * 64, n_chan_ds=96, n_pol=2,
    ) == 96_040
    assert group_section_bytes(
        n_group=0, n_acc_total=1024, n_chan_ds=96, n_pol=2,
    ) == 0


def test_too_many_groups_is_rejected(tmp_path, monkeypatch):
    import dsart.services.rfi_mon_shm as shm
    monkeypatch.setattr(shm, "_SHM_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="_MAX_GROUPS"):
        shm.RFIMonShmWriter(
            cn_id=4244, n_ants=4, n_chan_ds=4, n_pol=2, window_size=2,
            freq_downsample=1, n_slots=2, n_group=9, n_acc_per_cube=4,
        )


# ---------------------------------------------------------------------------
# Slow-path settings parser
# ---------------------------------------------------------------------------


def test_slow_rfi_settings_ignore_malformed_payloads():
    """A garbled etcd payload must never be able to switch flagging
    ON: it changes what every downstream calibration sees."""
    from dsart.services.corr_slow_compute import SlowRfiSettings
    off = SlowRfiSettings(enabled=False)
    for junk in (None, [], "true", 1, {"enabled": "yes"}, {"enabled": 1}):
        assert SlowRfiSettings.from_payload(junk, current=off).enabled is False
    on = SlowRfiSettings.from_payload({"enabled": True}, current=off)
    assert on.enabled is True


def test_slow_rfi_settings_keep_current_for_absent_fields():
    from dsart.services.corr_slow_compute import SlowRfiSettings
    cur = SlowRfiSettings(
        enabled=True, array_burst_mode="monitor", hi_guard=False,
    )
    got = SlowRfiSettings.from_payload({"enabled": False}, current=cur)
    assert got.enabled is False
    assert got.array_burst_mode == "monitor"
    assert got.hi_guard is False
    # An unknown mode must not be accepted.
    got2 = SlowRfiSettings.from_payload(
        {"array_burst_mode": "banana"}, current=cur,
    )
    assert got2.array_burst_mode == "monitor"


# ---------------------------------------------------------------------------
# End to end through RFIFlagger
# ---------------------------------------------------------------------------


def _dsa_geometry():
    e = np.concatenate([
        np.linspace(-190.3, 206.6, 47), np.full(35, 8.2),
        np.linspace(-1200, 1200, 14),
    ])
    n = np.concatenate([
        np.full(47, -295.3), np.linspace(-148.2, 146.1, 35),
        np.linspace(-900, 900, 14),
    ])
    st = np.concatenate([np.arange(1, 83), np.arange(103, 117)])
    return e, n, st


def _tiny_autos(n_acc=4, n_ant=96, n_ch=32, n_pol=2, burst_at=None, seed=0):
    """AutoSpectra with the M values RFIFlagger expects, built directly.

    ``burst_at`` injects an array-common flat excess into one
    accumulation.
    """
    from dsart.rfi.autos import AutoSpectra
    rng = np.random.default_rng(seed)
    m_fine = 64
    total = n_acc * m_fine
    p = np.ones((n_acc, n_ant, n_ch, n_pol))
    if burst_at is not None:
        p[burst_at] *= 1.30
    s1 = rng.gamma(m_fine, p / m_fine, p.shape) * m_fine
    s2 = s1 * s1 / m_fine * 2.0
    s1_full = s1.sum(axis=0, keepdims=True)
    s2_full = s2.sum(axis=0, keepdims=True)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)   # noqa: E731
    return AutoSpectra(
        s1={m_fine: t(s1), total: t(s1_full)},
        s2={m_fine: t(s2), total: t(s2_full)},
    )


def test_flagger_rejects_wrong_base_m():
    """The detector runs on the BASE accumulation, so a caller who
    drops M=64 from m_values would silently move it to a different
    time scale. That must fail loudly at construction."""
    from dsart.rfi import ArrayBurstDetector, RFIFlagger, build_groups_from_antpos
    groups = build_groups_from_antpos(*_dsa_geometry())
    det = ArrayBurstDetector(groups, n_chan=32, n_pol=2)
    with pytest.raises(ValueError, match="M=64"):
        RFIFlagger(
            flagants_path=None, array_burst=det, array_burst_mode="monitor",
            m_values=(128, 512),
        )


def test_flagger_monitor_mode_reports_but_does_not_tag():
    from dsart.rfi import (
        ArrayBurstDetector, FlagSourceBit, RFIFlagger,
        build_groups_from_antpos,
    )
    groups = build_groups_from_antpos(*_dsa_geometry())
    det = ArrayBurstDetector(groups, n_chan=32, n_pol=2, warmup_cubes=0)
    fl = RFIFlagger(
        flagants_path=None, array_burst=det, array_burst_mode="monitor",
        m_values=(64, 256), warmup_cubes=0,
    )
    for _ in range(6):
        r = fl.flag_block(None, None, autos_override=_tiny_autos(seed=1))
    assert r.array_burst is not None
    assert r.array_burst.z.shape == (4, 5, 2)
    # Monitor mode: no tag bit, no time mask reaches the caller as a flag.
    assert int((r.source_tags & int(FlagSourceBit.ARRAY_BURST)).sum()) == 0


def test_flagger_flag_mode_sets_bit5_on_a_real_burst():
    from dsart.rfi import (
        ArrayBurstDetector, FlagSourceBit, RFIFlagger,
        build_groups_from_antpos,
    )
    groups = build_groups_from_antpos(*_dsa_geometry())
    det = ArrayBurstDetector(groups, n_chan=32, n_pol=2, warmup_cubes=0)
    fl = RFIFlagger(
        flagants_path=None, array_burst=det, array_burst_mode="flag",
        m_values=(64, 256), warmup_cubes=0,
    )
    for i in range(8):                       # settle the baseline
        fl.flag_block(None, None, autos_override=_tiny_autos(seed=100 + i))
    r = fl.flag_block(
        None, None, autos_override=_tiny_autos(burst_at=2, seed=7),
    )
    ab = r.array_burst
    assert ab is not None and ab.time_chan_mask is not None
    assert bool(ab.fired[2, det.groups.index("core"), :].all())
    assert not bool(ab.fired[0, det.groups.index("core"), :].any())
    # Bit 5 marks the (ant, ch, pol) cells touched, on EVERY antenna:
    # the signal is array-common and cannot be attributed to one.
    tagged = (r.source_tags & int(FlagSourceBit.ARRAY_BURST)) != 0
    assert bool(tagged.all())
    # But the cube mask is NOT polluted by it — the array-burst verdict
    # is time-resolved and travels separately.
    assert not bool(r.mask.all())


def test_flagger_without_detector_is_unchanged():
    from dsart.rfi import FlagSourceBit, RFIFlagger
    fl = RFIFlagger(flagants_path=None, m_values=(64, 256), warmup_cubes=0)
    r = fl.flag_block(None, None, autos_override=_tiny_autos(seed=3))
    assert r.array_burst is None
    assert int((r.source_tags & int(FlagSourceBit.ARRAY_BURST)).sum()) == 0


# ---------------------------------------------------------------------------
# SK threshold warm-up gating (slow path)
# ---------------------------------------------------------------------------


def test_sk_warmup_is_idempotent_and_sets_ready(monkeypatch):
    """The warm-up must run once, off the block loop, and always end
    with ready set — including on failure, or a feature the operator
    turned on would silently stay off forever."""
    import dsart.rfi.sk as sk
    from dsart.services.corr_slow_compute import SkWarmup

    calls: list[int] = []
    monkeypatch.setattr(
        sk, "sk_thresholds", lambda m, far=1e-4: calls.append(m) or (0.9, 1.1),
    )
    w = SkWarmup()
    assert not w.ready.is_set()
    w.ensure()
    w.ensure()                               # second call must be a no-op
    assert w.wait(timeout=10.0)
    from dsart.rfi.autos import DEFAULT_M_VALUES
    assert sorted(calls) == sorted(DEFAULT_M_VALUES)


def test_sk_warmup_sets_ready_even_when_it_fails(monkeypatch):
    import dsart.rfi.sk as sk
    from dsart.services.corr_slow_compute import SkWarmup

    def _boom(m, far=1e-4):
        raise MemoryError("33 GB, as it happens")

    monkeypatch.setattr(sk, "sk_thresholds", _boom)
    w = SkWarmup()
    w.ensure()
    assert w.wait(timeout=10.0)
    assert w.ready.is_set()


def test_build_slow_flagger_does_no_sk_work(monkeypatch):
    """Constructing the flagger must not touch the Monte Carlo: the
    default is flagging OFF, and corr_slow shares a node with
    corr_fast, so an unconditional warm would have both processes
    reaching for tens of GB at startup."""
    import dsart.rfi.sk as sk
    from dsart.services.corr_slow_compute import build_slow_flagger

    called: list[int] = []
    monkeypatch.setattr(
        sk, "sk_thresholds",
        lambda m, far=1e-4: called.append(m) or (0.9, 1.1),
    )
    flagger, guard = build_slow_flagger(
        device=torch.device("cpu"), chgroup=6, flagants_path=None,
        array_burst_mode="off", cal_path=None,
    )
    assert flagger is not None
    assert int(guard.sum()) == 86          # chgroup 6 carries the HI band
    assert called == []
