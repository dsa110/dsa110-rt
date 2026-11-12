"""DM-time ridge: does a candidate's signal sit at ONE frequency? (2x, C3 R12 study)

In a dedispersed DM-time plane whose time axis is referenced to the bottom of
the band (the 2x search: stage-2 reads ``stream[t - a2 * DM]`` with ``a2`` the
delay from each sub-band down to ``nu_bot``), a signal component at frequency
``nu`` moves by

    dt = s(nu) * (DM - DM_trig),   s(nu) = k (nu_lo^-2 - nu^-2)   [ms per pc/cc]

as the trial DM changes: a straight line through the trigger. ``s`` runs from
0 (``nu_lo``) to 0.566 ms per pc/cc (``nu_hi``), so a line's slope IS its
emission frequency. A narrow-band impulse (1335-1341 MHz interferer: 261111lmnm,
dapz) stays in focus for hundreds of pc/cc along one slope; a broadband burst
is in focus only near its DM (~w / 0.566 pc/cc) and away from it smears into a
fan covering every physical slope, weaker by ~w / (0.566 |dDM|).

So along each slope we take the "arm": the median, over DM rows away from the
trigger, of the plane's max within +-1 sample of the line. Scanning slope maps
the arm against frequency (the ridge spectrum); slopes outside the physical
range (negative, or beyond ``nu_hi``) cross the same trigger but can carry no
dispersed signal, and give the noise reference. Features:

* ``peak_sigma``: best physical arm above the null, in null-MAD units;
* ``freq_mhz``: the frequency of that best line;
* ``occupied_mhz``: participation-ratio bandwidth of the significant excess
  (``(sum e)^2 / sum e^2`` x bin) -- one line gives ~the frequency resolution,
  a broadband fan gives most of the band;
* ``ridge_frac``: excess arm / trigger core (what the eye reads as "the
  stripe is as bright as the burst");
* ``core``: the trigger's own significance.

Real bursts can draw faint lines too (261110zzeg: scintles act as narrow-band
components), so no single feature decides; see ``docs/x2/dmt_ridge_study.md``.
Pure numpy; no torch (runs on h23 inside C3).
"""

from __future__ import annotations

import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["RidgeConfig", "RidgeFeatures", "ridge_features", "slope_of_freq", "freq_of_slope",
           "dm_time_from_cutouts", "boxcar_rows", "robust_rows", "event_ridge", "narrowband_verdict"]

K_DM_MS_GHZ2 = 4.148808          # ms GHz^2 cm^3 / pc
NU_LO_MHZ = 1311.28              # 2x time reference (bottom of the processed band)
NU_HI_MHZ = 1498.75
T_EXT_MS = 0.524288


def slope_of_freq(nu_mhz, nu_lo_mhz: float = NU_LO_MHZ):
    """ms per pc/cc of a line drawn by a component at ``nu_mhz``."""
    nu = np.asarray(nu_mhz, np.float64) / 1e3
    return K_DM_MS_GHZ2 * ((nu_lo_mhz / 1e3) ** -2 - nu ** -2)


def freq_of_slope(s, nu_lo_mhz: float = NU_LO_MHZ):
    s = np.asarray(s, np.float64)
    return 1e3 * ((nu_lo_mhz / 1e3) ** -2 - s / K_DM_MS_GHZ2) ** -0.5


@dataclass(frozen=True)
class RidgeConfig:
    nu_lo_mhz: float = NU_LO_MHZ
    nu_hi_mhz: float = NU_HI_MHZ
    t_samp_ms: float = T_EXT_MS
    freq_step_mhz: float = 0.5
    #: arm rows: |dDM| in [max(arm_dm_min, arm_widths * w_ms / s_max), arm_dm_max]
    arm_dm_min: float = 30.0
    arm_widths: float = 8.0
    arm_dm_max: float = 600.0
    t_tol: int = 1                # +- samples around the line
    min_arm_rows: int = 30
    excess_sigma: float = 5.0     # a frequency bin counts toward occupied_mhz above this


@dataclass(frozen=True)
class RidgeFeatures:
    ok: bool
    reason: str = ""
    core: float = 0.0
    arm: float = 0.0
    null_median: float = 0.0
    null_mad: float = 0.0
    peak_sigma: float = 0.0
    freq_mhz: float = 0.0
    occupied_mhz: float = 0.0
    ridge_frac: float = 0.0
    frac_rows_gt4: float = 0.0
    n_arm_rows: int = 0
    arm_dm_lo: float = 0.0
    #: profile fit along the best line: z(dDM) = f * core * min(1, dDM_c / |dDM|),
    #: dDM_c = w nu^3 / (2 k B): the bandwidth B that keeps the line in focus
    bw_fit_mhz: float = 0.0
    bw_fit_lo: float = 0.0
    bw_fit_hi: float = 0.0
    f_fit: float = 0.0
    fit_sigma: float = 0.0
    n_prof_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def boxcar_rows(z: np.ndarray, w: int) -> np.ndarray:
    """Sum of ``w`` samples / sqrt(w) along time, aligned to the window start; NaN-aware."""
    if w <= 1:
        return z.astype(np.float32, copy=True)
    ok = np.isfinite(z)
    x = np.where(ok, z, 0.0).astype(np.float64)
    c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
    n = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(ok, axis=1)], axis=1)
    out = np.full(x.shape, np.nan)
    s = (c[:, w:] - c[:, :-w]) / np.sqrt(w)
    full = (n[:, w:] - n[:, :-w]) == w
    out[:, :x.shape[1] - w + 1] = np.where(full, s, np.nan)
    return out.astype(np.float32)


def robust_rows(x: np.ndarray) -> np.ndarray:
    """Per-row (median, 1.4826 MAD) z-score; rows with no spread become NaN."""
    med = np.nanmedian(x, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((x - med) / np.where(mad > 0, mad, np.nan)).astype(np.float32)


def _max3(z: np.ndarray, tol: int) -> np.ndarray:
    out = z.copy()
    for d in range(1, tol + 1):
        out[:, d:] = np.fmax(out[:, d:], z[:, :-d])
        out[:, :-d] = np.fmax(out[:, :-d], z[:, d:])
    return out


def ridge_features(
    z: np.ndarray,
    dms: np.ndarray,
    *,
    dm_trig: float,
    t_trig: int,
    width: int,
    cfg: RidgeConfig = RidgeConfig(),
) -> Tuple[RidgeFeatures, Dict[str, np.ndarray]]:
    """Ridge features of a per-row normalised, width-matched DM-time plane.

    ``z[row, t]`` (NaN = no data), ``dms[row]`` ascending, ``t_trig`` the
    trigger's column, ``width`` its boxcar width (samples). Returns the
    features and the ridge spectrum (``freq_mhz``, ``arm``, ``excess``).
    """
    z = np.asarray(z, np.float32)
    dms = np.asarray(dms, np.float64)
    n_rows, n_t = z.shape
    i_trig = int(np.argmin(np.abs(dms - dm_trig)))
    # the core: max within +-(w + 2) samples of the trigger (a boxcar's labelled
    # sample is not its centre: 261111dbuj, w 64, read 0.75 at the label); the
    # lines are anchored where the core is
    reach = max(1, int(width)) + 2
    lo_t, hi_t = max(0, t_trig - reach), min(n_t, t_trig + reach + 1)
    r0 = max(0, i_trig - 1)
    core_box = z[r0:i_trig + 2, lo_t:hi_t]
    if not np.isfinite(core_box).any():
        return RidgeFeatures(False, "no data at the trigger"), {}
    ci, ct = np.unravel_index(int(np.nanargmax(core_box)), core_box.shape)
    core = float(core_box[ci, ct])
    t_trig = lo_t + int(ct)

    s_max = float(slope_of_freq(cfg.nu_hi_mhz, cfg.nu_lo_mhz))
    w_ms = max(1, int(width)) * cfg.t_samp_ms
    arm_lo = max(cfg.arm_dm_min, cfg.arm_widths * w_ms / s_max)
    ddm = np.abs(dms - dm_trig)
    rows = np.nonzero((ddm >= arm_lo) & (ddm <= max(cfg.arm_dm_max, 2 * arm_lo))
                      & np.isfinite(z).any(axis=1))[0]
    if rows.size < cfg.min_arm_rows:
        return RidgeFeatures(False, f"{rows.size} arm rows (< {cfg.min_arm_rows})", core=core,
                             n_arm_rows=int(rows.size), arm_dm_lo=arm_lo), {}

    nu = np.arange(cfg.nu_lo_mhz, cfg.nu_hi_mhz + 1e-9, cfg.freq_step_mhz)
    s_phys = slope_of_freq(nu, cfg.nu_lo_mhz)
    s_null = np.concatenate([-np.linspace(0.25 * s_max, 3.0 * s_max, nu.size),
                             np.linspace(1.25 * s_max, 3.0 * s_max, nu.size)])
    zm = _max3(z, cfg.t_tol)
    rel = dms[rows] - dm_trig

    def arms(slopes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = t_trig + np.rint(slopes[:, None] * rel[None, :] / cfg.t_samp_ms).astype(np.int64)
        inside = (t >= 0) & (t < n_t)
        v = np.full(t.shape, np.nan, np.float32)
        rr = np.broadcast_to(rows[None, :], t.shape)
        v[inside] = zm[rr[inside], t[inside]]
        n_ok = np.isfinite(v).sum(axis=1)
        with np.errstate(all="ignore"):
            med = np.where(n_ok >= cfg.min_arm_rows, np.nanmedian(v, axis=1), np.nan)
        return med, v

    arm_phys, v_phys = arms(s_phys)
    arm_null, _ = arms(s_null)
    null = arm_null[np.isfinite(arm_null)]
    if null.size < 10 or not np.isfinite(arm_phys).any():
        return RidgeFeatures(False, "too little coverage along the lines", core=core,
                             n_arm_rows=int(rows.size), arm_dm_lo=arm_lo), {}
    n_med = float(np.median(null))
    n_mad = max(float(1.4826 * np.median(np.abs(null - n_med))), 0.02)
    excess = (arm_phys - n_med) / n_mad
    j = int(np.nanargmax(excess))
    sig = np.where(np.isfinite(excess) & (excess > cfg.excess_sigma), excess, 0.0)
    occ = float((sig.sum() ** 2 / (sig ** 2).sum()) * cfg.freq_step_mhz) if sig.any() else 0.0
    best = v_phys[j]
    frac4 = float(np.mean(best[np.isfinite(best)] > 4.0)) if np.isfinite(best).any() else 0.0
    fit = _profile_fit(zm, dms, dm_trig=dm_trig, t_trig=t_trig, slope=float(s_phys[j]), nu_mhz=float(nu[j]),
                       w_ms=w_ms, core=core, null=n_med, cfg=cfg)
    feats = RidgeFeatures(
        ok=True, core=round(core, 3), arm=round(float(arm_phys[j]), 3), null_median=round(n_med, 3),
        null_mad=round(n_mad, 3), peak_sigma=round(float(excess[j]), 2), freq_mhz=round(float(nu[j]), 2),
        occupied_mhz=round(occ, 2), ridge_frac=round(float((arm_phys[j] - n_med) / max(core, 1e-6)), 4),
        frac_rows_gt4=round(frac4, 3), n_arm_rows=int(rows.size), arm_dm_lo=round(arm_lo, 1), **fit)
    return feats, {"freq_mhz": nu, "arm": arm_phys, "excess": excess}


#: bandwidth grid of the profile fit (MHz)
_BW_GRID = np.geomspace(0.1, 300.0, 90)


def _profile_fit(zm, dms, *, dm_trig, t_trig, slope, nu_mhz, w_ms, core, null, cfg) -> Dict[str, Any]:
    """Fit z along the best line to ``A * min(1, dDM_c(B) / |dDM|)`` over every row.

    A component of bandwidth B at nu, seen at trial DM off by dDM, smears over
    2 k dDM B / nu^3; a boxcar of the burst width keeps ~min(1, w / smear) of
    it. Rows within 2 broadband smear-lengths of the trigger are left out (a
    broadband burst's own core). ``A`` is linear (least squares per B);
    ``fit_sigma`` = sqrt(chi2(A=0) - chi2_min) with unit noise per row.
    """
    n_t = zm.shape[1]
    s_max = float(slope_of_freq(cfg.nu_hi_mhz, cfg.nu_lo_mhz))
    rel = dms - dm_trig
    t = t_trig + np.rint(slope * rel / cfg.t_samp_ms).astype(np.int64)
    ok = (np.abs(rel) >= max(2.0 * w_ms / s_max, 3.0)) & (t >= 0) & (t < n_t)
    y = np.full(dms.size, np.nan)
    y[ok] = zm[np.nonzero(ok)[0], t[ok]] - null
    ok &= np.isfinite(y)
    if ok.sum() < cfg.min_arm_rows:
        return {"n_prof_rows": int(ok.sum())}
    y, d = y[ok], np.abs(rel[ok])
    nu = nu_mhz / 1e3
    ddm_c = w_ms * nu ** 3 / (2.0 * K_DM_MS_GHZ2 * (_BW_GRID / 1e3))       # (n_B,)
    m = np.minimum(1.0, ddm_c[:, None] / d[None, :])                     # (n_B, n_rows)
    a = np.clip((m @ y) / np.einsum("ij,ij->i", m, m), 0.0, None)
    chi = ((y[None, :] - a[:, None] * m) ** 2).sum(axis=1)
    k = int(np.argmin(chi))
    chi0 = float((y ** 2).sum())
    if a[k] <= 0.0:                                                     # no line: B undefined
        return {"n_prof_rows": int(y.size)}
    good = _BW_GRID[chi <= chi[k] + 4.0]
    return {"bw_fit_mhz": round(float(_BW_GRID[k]), 3), "bw_fit_lo": round(float(good.min()), 3),
            "bw_fit_hi": round(float(good.max()), 3), "f_fit": round(float(a[k] / max(core, 1e-6)), 4),
            "fit_sigma": round(float(np.sqrt(max(chi0 - chi[k], 0.0))), 2), "n_prof_rows": int(y.size)}


# ---------------------------------------------------------------------------
# 2x cutouts -> DM-time plane
# ---------------------------------------------------------------------------


def dm_time_from_cutouts(event_dir, *, kind: str = "pix") -> Optional[Dict[str, Any]]:
    """Stack every half's fines by DM on one T_ext grid (as the dm_time plot).

    ``kind="pix"``: the light curve at the requested pixel (``pix_lc``, a
    beam on the candidate); ``"max"``: the image maximum (``peak_grid``, what
    the plot shows). A half whose window carried no input at all is left out.
    Returns ``{"plane", "dms", "t_trig", "width", "dm_trig", "n_halves"}`` or
    None when there is no usable cutout.
    """
    key = {"pix": "pix_lc", "max": "peak_grid"}[kind]
    planes = []
    for f in sorted(glob.glob(str(Path(event_dir) / "cutouts" / "cutout_s*_g*.npz"))):
        try:
            with np.load(f) as zf:
                arr = np.asarray(zf[key], np.float32)
                ivf = np.asarray(zf["input_valid_frac"])
                if arr.size == 0 or (ivf.size and float(ivf.max()) <= 0.0):
                    continue
                planes.append((int(zf["t0_ext"]), arr, np.asarray(zf["fine_dm_pc_cc"], np.float64),
                               int(zf["t_peak_ext"]), int(zf["width_ext_req"]), float(zf["dm_req"])))
        except (OSError, KeyError, ValueError):
            continue
    if not planes:
        return None
    t_lo = min(p[0] for p in planes)
    t_hi = max(p[0] + p[1].shape[0] for p in planes)
    rows, dms = [], []
    for t0, arr, fdm, *_ in planes:
        for j in range(arr.shape[1]):
            r = np.full(t_hi - t_lo, np.nan, np.float32)
            r[t0 - t_lo:t0 - t_lo + arr.shape[0]] = arr[:, j]
            rows.append(r)
            dms.append(fdm[j])
    order = np.argsort(dms)
    return {"plane": np.stack([rows[i] for i in order]), "dms": np.asarray(dms)[order],
            "t_trig": planes[0][3] - t_lo, "width": max(1, planes[0][4]), "dm_trig": planes[0][5],
            "n_halves": len(planes)}


def narrowband_verdict(f: RidgeFeatures, *, bw_max_mhz: float = 6.0, f_min: float = 0.45,
                       sigma_min: float = 8.0) -> bool:
    """R12: a line narrower than ``bw_max_mhz`` that carries the trigger.

    Defaults from docs/x2/dmt_ridge_study (0/26 legacy FRBs, 0 of 4 other
    FRB records, B1933+16 and 11 injections flagged; 10/11 voltage-confirmed narrow-band).
    ``bw_fit_mhz == 0`` means no line was found.
    """
    return bool(f.ok and 0.0 < f.bw_fit_mhz <= bw_max_mhz and f.f_fit >= f_min and f.fit_sigma >= sigma_min)


def event_ridge(event_dir, *, kind: str = "pix", cfg: RidgeConfig = RidgeConfig()):
    """Cutouts -> boxcar at the trigger width -> robust rows -> features."""
    d = dm_time_from_cutouts(event_dir, kind=kind)
    if d is None:
        return RidgeFeatures(False, "no cutouts"), {}, None
    w = int(d["width"])
    z = robust_rows(boxcar_rows(d["plane"], w))
    t_trig = int(d["t_trig"])          # boxcar output index t covers [t, t + w)
    return (*ridge_features(z, d["dms"], dm_trig=d["dm_trig"], t_trig=t_trig, width=w, cfg=cfg), d)
