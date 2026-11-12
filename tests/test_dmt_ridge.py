"""DM-time ridge features (C3 R12 study, 2026-11-12).

A narrow-band impulse draws a straight line through the trigger in the 2x
DM-time plane whose slope gives its frequency (261111lmnm: 1337 MHz, dapz:
1341 MHz); a broadband burst is compact at its DM and a faint fan away from
it. These tests pin the geometry and the features on synthetic planes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsart.coinc import dmt_ridge as R

DMS = np.linspace(50.0, 3000.0, 758)
T_TRIG = 256


def _noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(DMS.size, 512)).astype(np.float32)


def _line(z: np.ndarray, *, dm0: float, nu_mhz: float, amp: float, t0: int = T_TRIG) -> None:
    s = float(R.slope_of_freq(nu_mhz))
    for i, dm in enumerate(DMS):
        t = t0 + int(round(s * (dm - dm0) / R.T_EXT_MS))
        if 0 <= t < z.shape[1]:
            z[i, t] += amp


def test_slope_frequency_round_trip() -> None:
    assert float(R.slope_of_freq(R.NU_LO_MHZ)) == pytest.approx(0.0, abs=1e-12)
    assert float(R.slope_of_freq(R.NU_HI_MHZ)) == pytest.approx(0.5659, abs=1e-3)
    nu = np.array([1320.0, 1337.0, 1341.0, 1496.0])
    assert np.allclose(R.freq_of_slope(R.slope_of_freq(nu)), nu)
    # the slopes measured on the 11-11 events map to their voltage spikes
    assert float(R.freq_of_slope(0.095)) == pytest.approx(1337.9, abs=0.2)
    assert float(R.freq_of_slope(0.105)) == pytest.approx(1340.8, abs=0.2)


def test_narrowband_line_is_found_at_its_frequency() -> None:
    z = _noise()
    _line(z, dm0=500.0, nu_mhz=1340.0, amp=6.0)
    z[np.argmin(np.abs(DMS - 500.0)), T_TRIG] = 12.0
    f, spec = R.ridge_features(z, DMS, dm_trig=500.0, t_trig=T_TRIG, width=2)
    assert f.ok
    assert f.freq_mhz == pytest.approx(1340.0, abs=1.5)
    assert f.peak_sigma > 30 and f.occupied_mhz < 6.0
    assert 0.3 < f.ridge_frac < 0.6
    assert spec["freq_mhz"].shape == spec["arm"].shape
    # in focus at every DM: the profile fit puts it below 2 MHz, carrying half the core
    assert f.bw_fit_mhz < 2.0 and 0.35 < f.f_fit < 0.65 and f.fit_sigma > 50


def test_compact_broadband_burst_has_no_ridge() -> None:
    z = _noise(1)
    i0 = int(np.argmin(np.abs(DMS - 800.0)))
    z[i0 - 1:i0 + 2, T_TRIG - 1:T_TRIG + 2] += 30.0
    f, _ = R.ridge_features(z, DMS, dm_trig=800.0, t_trig=T_TRIG, width=2)
    assert f.ok and f.peak_sigma < 6 and f.ridge_frac < 0.05


def test_broadband_fan_is_broad_not_a_line() -> None:
    # away from its DM a bright broadband burst smears over every physical slope,
    # weaker by w / (0.566 |dDM|): elevated arms at all frequencies, no single line
    z = _noise(2)
    dm0, core = 800.0, 200.0
    for i, dm in enumerate(DMS):
        lo = T_TRIG + R.slope_of_freq(R.NU_LO_MHZ) * (dm - dm0) / R.T_EXT_MS
        hi = T_TRIG + R.slope_of_freq(R.NU_HI_MHZ) * (dm - dm0) / R.T_EXT_MS
        a, b = sorted((int(round(lo)), int(round(hi))))
        a, b = max(a, 0), min(b + 1, z.shape[1])
        if b > a:
            z[i, a:b] += core / max(1, b - a)
    f, _ = R.ridge_features(z, DMS, dm_trig=dm0, t_trig=T_TRIG, width=2)
    assert f.ok and f.ridge_frac < 0.1
    assert f.occupied_mhz > 20.0
    assert f.bw_fit_mhz > 20.0                 # falls off like a broadband burst, not a line


def test_arm_window_moves_out_with_width() -> None:
    z = _noise(3)
    f2, _ = R.ridge_features(z, DMS, dm_trig=1000.0, t_trig=T_TRIG, width=2)
    f32, _ = R.ridge_features(z, DMS, dm_trig=1000.0, t_trig=T_TRIG, width=32)
    assert f2.arm_dm_lo == pytest.approx(30.0)
    assert f32.arm_dm_lo == pytest.approx(8 * 32 * R.T_EXT_MS / 0.5659, rel=1e-3)


def test_missing_rows_and_no_data() -> None:
    z = _noise(4)
    _line(z, dm0=500.0, nu_mhz=1400.0, amp=6.0)
    z[(DMS > 900) & (DMS < 2000)] = np.nan                     # absent halves
    f, _ = R.ridge_features(z, DMS, dm_trig=500.0, t_trig=T_TRIG, width=2)
    assert f.ok and f.freq_mhz == pytest.approx(1400.0, abs=2.0)
    z[:] = np.nan
    f, _ = R.ridge_features(z, DMS, dm_trig=500.0, t_trig=T_TRIG, width=2)
    assert not f.ok


def test_boxcar_and_robust_rows() -> None:
    x = np.zeros((2, 10), np.float32)
    x[0, 4] = 2.0
    x[1, 7] = np.nan
    b = R.boxcar_rows(x, 2)
    assert b[0, 3] == pytest.approx(2 / np.sqrt(2)) and b[0, 4] == pytest.approx(2 / np.sqrt(2))
    assert np.isnan(b[1, 6]) and np.isnan(b[1, 7]) and np.isnan(b[0, 9])
    r = R.robust_rows(np.random.default_rng(5).normal(3.0, 2.0, size=(3, 4000)))
    assert np.allclose(np.nanmedian(r, axis=1), 0, atol=0.05)
    assert np.allclose(1.4826 * np.nanmedian(np.abs(r), axis=1), 1, atol=0.05)


def _cutout(path: Path, *, t0: int, fdm, pix, ivf: float = 1.0) -> None:
    np.savez(path, pix_lc=np.asarray(pix, np.float32), peak_grid=np.asarray(pix, np.float32) + 3,
             fine_dm_pc_cc=np.asarray(fdm, np.float32), t0_ext=np.int64(t0), t_peak_ext=np.int64(1000 + 5),
             width_ext_req=np.int32(2), dm_req=np.float32(fdm[0]),
             input_valid_frac=np.full(len(fdm), ivf, np.float32))


def test_dm_time_from_cutouts_stacks_halves_by_dm(tmp_path: Path) -> None:
    cut = tmp_path / "ev" / "cutouts"
    cut.mkdir(parents=True)
    _cutout(cut / "cutout_s1_g0.npz", t0=1000, fdm=[300.0, 100.0], pix=np.ones((16, 2)))
    _cutout(cut / "cutout_s2_g0.npz", t0=1004, fdm=[200.0], pix=2 * np.ones((16, 1)))
    _cutout(cut / "cutout_s9_g0.npz", t0=1000, fdm=[150.0], pix=np.ones((16, 1)), ivf=0.0)   # no input
    d = R.dm_time_from_cutouts(tmp_path / "ev", kind="pix")
    assert d["n_halves"] == 2 and list(d["dms"]) == [100.0, 200.0, 300.0]
    assert d["plane"].shape == (3, 20)
    assert np.isnan(d["plane"][1, :4]).all() and (d["plane"][1, 4:] == 2).all()
    assert d["t_trig"] == 5 and d["width"] == 2
    assert R.dm_time_from_cutouts(tmp_path / "nothing") is None
