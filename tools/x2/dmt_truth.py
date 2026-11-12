#!/usr/bin/env python3
"""Truth set for the C3 narrow-band ridge rule (R12 study, P0).

Recreates the 2x search's DM-time plane -- the light curve at the candidate's
position for each of the 758 fine DMs, on the 0.524 ms T_ext grid, time
referenced to the bottom of the band -- from a beamformed voltage filterbank,
by applying the tiered plan's own tables to the dynamic spectrum:

    stage 1 (per coarse d, q1 = 0.262 ms):
        x[t, g, s] = sum_{k in sub-band s} I[t + stage1_bins[g, k, d], g, k]
    decimate x2 (tier A) / x4 (tier B)
    stage 2 (per fine f of d, tier samples):
        y_f[j] = sum_{g, s} x[j - stage2_bins[f, g*4+s], g, s]

A beam on a point source is linear in power, so this is the search's
``pix_lc`` up to the per-stream weighting. Inputs: dsart C3 filterbanks
(dispersed, 0.262 ms) and the legacy pipeline's ``<ev>_dev_polcal_I.fil``
(dedispersed at the catalogued DM, 32.768 us): a dedispersed input is first
re-dispersed at its DM (state detected, not assumed).

Also measures the objective spectral label the August study used: the
fraction of the on-pulse S/N^2 inside the best 1 MHz (real bursts ~0.05,
the 1335 MHz interferer 0.85-0.92) and the bandwidth holding 90% of it.

    python tools/x2/dmt_truth.py --plan PLAN.npz --out truth.jsonl \
        [--legacy-keep /dataz/dsa110/saveDirs.keep] [--fil EV=PATH:DM ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dsart.coinc.dmt_ridge import (K_DM_MS_GHZ2, RidgeConfig, boxcar_rows,  # noqa: E402
                                   ridge_features, robust_rows)
from dsart.x2.plan import TieredDmPlan  # noqa: E402

Q1_MS = 0.262144
NCH, NG, NK, NSUBK = 6144, 16, 192, 48


def read_fil(path) -> Tuple[Dict[str, Any], np.ndarray]:
    h: Dict[str, Any] = {}
    with open(path, "rb") as fh:
        def rstr():
            n = struct.unpack("<i", fh.read(4))[0]
            return fh.read(n).decode("ascii", "replace")
        assert rstr() == "HEADER_START"
        ints = {"nchans", "nbits", "nifs", "telescope_id", "machine_id", "data_type", "nbeams", "ibeam",
                "barycentric", "pulsarcentric", "nsamples"}
        dbls = {"tsamp", "fch1", "foff", "tstart", "src_raj", "src_dej", "az_start", "za_start", "refdm"}
        while True:
            key = rstr()
            if key == "HEADER_END":
                break
            if key in ints:
                h[key] = struct.unpack("<i", fh.read(4))[0]
            elif key in dbls:
                h[key] = struct.unpack("<d", fh.read(8))[0]
            else:
                h[key] = rstr()
        off = fh.tell()
    if h.get("nbits", 32) != 32 or h.get("nchans") != NCH:
        raise ValueError(f"{path}: nbits {h.get('nbits')} nchans {h.get('nchans')}")
    data = np.memmap(path, dtype=np.float32, mode="r", offset=off)
    return h, data[: data.size // NCH * NCH].reshape(-1, NCH)


def to_q1_summed(h, data) -> np.ndarray:
    """(n_q1, 16, 192) channel-pair means at 0.262 ms, per-channel robust z (bad channels -> 0)."""
    f = int(round(Q1_MS / (h["tsamp"] * 1e3)))
    if f < 1 or abs(f * h["tsamp"] * 1e3 - Q1_MS) > 1e-6:
        raise ValueError(f"tsamp {h['tsamp']} is not a divisor of 0.262 ms")
    n = data.shape[0] // f * f
    x = np.asarray(data[:n], np.float32).reshape(n // f, f, NG, NK, 2).mean(axis=(1, 4))
    med = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - med), axis=0)
    bad = ~np.isfinite(mad) | (mad <= 0)
    z = (x - med) / np.where(bad, 1.0, mad)
    z[:, bad] = 0.0
    return z.astype(np.float32)


def _nu_summed() -> np.ndarray:
    fil = 1498.75 - 0.03051757812 * np.arange(NCH)
    return fil.reshape(NG, NK, 2).mean(axis=2) / 1e3                   # GHz


def disp_samples(dm: float) -> np.ndarray:
    """(16, 192) q1-sample delay of each summed channel w.r.t. the top of the band."""
    nu = _nu_summed()
    return np.rint(K_DM_MS_GHZ2 * dm * (nu ** -2 - nu[0, 0] ** -2) / Q1_MS).astype(np.int64)


def band_series(x: np.ndarray, shifts: Optional[np.ndarray]) -> np.ndarray:
    """Band sum, each channel advanced by ``shifts`` (dedisperse) or as is."""
    if shifts is None:
        return x.reshape(x.shape[0], -1).sum(axis=1)
    n = x.shape[0] - int(shifts.max())
    t = np.arange(n)
    return x[t[:, None] + shifts.reshape(-1)[None, :], np.arange(NG * NK)[None, :] // NK,
             np.arange(NG * NK)[None, :] % NK].sum(axis=1)


def peak_snr(series: np.ndarray, widths=(1, 2, 4, 8, 16, 32, 64)) -> Tuple[float, int, int]:
    best = (-np.inf, 1, 0)
    for w in widths:
        b = np.convolve(series, np.ones(w), "valid") / np.sqrt(w)
        med = np.median(b)
        mad = 1.4826 * np.median(np.abs(b - med)) or 1.0
        zz = (b - med) / mad
        j = int(np.argmax(zz))
        if zz[j] > best[0]:
            best = (float(zz[j]), w, j)
    return best


def redisperse(x: np.ndarray, dm: float) -> np.ndarray:
    """Dedispersed (top-referenced) -> dispersed; NaN where no data."""
    d = disp_samples(dm)
    out = np.full((x.shape[0] + int(d.max()), NG, NK), np.nan, np.float32)
    for g in range(NG):
        for k in range(NK):
            out[d[g, k]:d[g, k] + x.shape[0], g, k] = x[:, g, k]
    return out


def emulate_2x(x: np.ndarray, plan: TieredDmPlan, *, dm_range: Optional[Tuple[float, float]] = None,
               t_window: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Dispersed q1 dynamic spectrum (n_q1, 16, 192) -> (758, hi - lo) on the T_ext grid.

    ``t_window = (lo, hi)`` in T_ext samples (default: everything); rows outside
    ``dm_range`` stay NaN. Each coarse DM computes stage 1 only over the q1 span
    its fines' stage-2 reads need. Tier-B samples (1.049 ms) fill two T_ext cells.
    """
    n_q1 = x.shape[0]
    lo_e, hi_e = t_window if t_window is not None else (0, n_q1 // 2)
    out = np.full((plan.fine_dm.size, hi_e - lo_e), np.nan, np.float32)
    gk = np.arange(NK)
    for d in range(plan.coarse_dm.size):
        f0, f1 = int(plan.fine_offset[d]), int(plan.fine_offset[d + 1])
        fsel = np.arange(f0, f1)
        if dm_range is not None:
            fsel = fsel[(plan.fine_dm[fsel] >= dm_range[0]) & (plan.fine_dm[fsel] <= dm_range[1])]
        if fsel.size == 0:
            continue
        D = int(plan.coarse_t_native[d]) // int(plan.tier_q1_native[int(plan.coarse_tier[d])])
        r = D // 2                                                          # T_ext cells per tier sample
        j_lo, j_hi = lo_e // r, -(-hi_e // r)
        s2 = np.asarray(plan.stage2_bins[fsel], np.int64)                  # (nf, 64)
        a = max(0, j_lo - int(s2.max()))
        b = j_hi - int(s2.min())
        s1 = np.asarray(plan.stage1_bins[:, :, d], np.int64)
        q_lo, q_hi = a * D, min(b * D, n_q1 - int(s1.max()))
        n1 = q_hi - q_lo
        if n1 < D:
            continue
        t = q_lo + np.arange(n1)
        x1 = np.empty((n1, NG, 4), np.float32)
        for g in range(NG):
            v = x[t[:, None] + s1[g][None, :], g, gk[None, :]]             # (n1, 192)
            x1[:, g, :] = v.reshape(n1, 4, NSUBK).sum(axis=2)
        nd = n1 // D
        x2 = x1[:nd * D].reshape(nd, D, NG * 4).sum(axis=1)               # x2[i] = tier sample a + i
        j = np.arange(j_lo, j_hi)
        cols = np.arange(64)[None, :]
        for fi, f in enumerate(fsel):
            idx = j[:, None] - s2[fi][None, :] - a
            ok = (idx >= 0) & (idx < nd)
            y = np.where(ok, x2[np.clip(idx, 0, nd - 1), cols], np.nan).sum(axis=1)
            te = np.repeat(j * r, r) + np.tile(np.arange(r), j.size)       # T_ext cells of each sample
            yy = np.repeat(y, r)
            keep = (te >= lo_e) & (te < hi_e)
            out[f, te[keep] - lo_e] = yy[keep]
    return out


def spectral_label(h, data, dm: float, dispersed: bool, t_top_q1: int, w_q1: int) -> Dict[str, float]:
    """On-pulse S/N per native channel at ``dm``: best-1-MHz S^2 fraction + 90% bandwidth."""
    f = int(round(Q1_MS / (h["tsamp"] * 1e3)))
    n = data.shape[0] // f * f
    x = np.asarray(data[:n], np.float32).reshape(n // f, f, NCH).mean(axis=1)
    if dispersed:
        nu = (1498.75 - 0.03051757812 * np.arange(NCH)) / 1e3
        sh = np.rint(K_DM_MS_GHZ2 * dm * (nu ** -2 - nu[0] ** -2) / Q1_MS).astype(np.int64)
        m = x.shape[0] - int(sh.max())
        x = x[np.arange(m)[:, None] + sh[None, :], np.arange(NCH)[None, :]]
    on = slice(max(0, t_top_q1), t_top_q1 + w_q1)
    off = np.r_[0:max(0, t_top_q1 - 4 * w_q1), t_top_q1 + 5 * w_q1:x.shape[0]]
    med = np.median(x[off], axis=0)
    sd = 1.4826 * np.median(np.abs(x[off] - med), axis=0)
    good = np.isfinite(sd) & (sd > 0)
    s = np.where(good, (x[on].mean(axis=0) - med) / np.where(good, sd, 1.0) * np.sqrt(max(1, w_q1)), 0.0)
    p = np.clip(s, 0, None) ** 2
    tot = p.sum()
    if tot <= 0:
        return {"frac_best_1mhz": float("nan"), "bw90_mhz": float("nan"), "peak_chan_snr": float(s.max())}
    win = 33                                                               # 1 MHz of 30.5 kHz channels
    c = np.convolve(p, np.ones(win), "valid")
    srt = np.sort(p)[::-1]
    n90 = int(np.searchsorted(np.cumsum(srt), 0.9 * tot) + 1)
    return {"frac_best_1mhz": round(float(c.max() / tot), 4), "bw90_mhz": round(n90 * 0.0305176, 2),
            "peak_chan_snr": round(float(s.max()), 2),
            "freq_peak_mhz": round(float(1498.75 - 0.0305176 * (int(np.argmax(c)) + win // 2)), 2)}


def save_plane(planes_dir, name: str, z: np.ndarray, dms: np.ndarray, *, t_trig: int, width: int,
               dm_trig: float, **meta) -> None:
    """Normalised plane (float16) + geometry, for re-scoring without re-emulating."""
    if not planes_dir:
        return
    Path(planes_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(planes_dir) / f"plane_{name}.npz", z=z.astype(np.float16), dms=dms,
                        t_trig=np.int64(t_trig), width=np.int64(width), dm_trig=np.float64(dm_trig),
                        meta=json.dumps(meta))


def truth_one(path, dm: float, plan: TieredDmPlan, cfg: RidgeConfig = RidgeConfig(), *,
              name: str = "", planes_dir: Optional[str] = None) -> Dict[str, Any]:
    h, data = read_fil(path)
    x = to_q1_summed(h, data)
    sh = disp_samples(dm)
    s_as, w_as, _ = peak_snr(band_series(x, None))
    if x.shape[0] - int(sh.max()) < 128:
        # shorter than the sweep: it can only be the dedispersed (legacy polcal) product
        s_dd, w_dd, j_dd = -np.inf, 1, 0
    else:
        s_dd, w_dd, j_dd = peak_snr(band_series(x, sh))
    dispersed = s_dd > s_as
    if dispersed:
        xd, t_top, w_q1 = x, j_dd, w_dd
    else:
        s_as2, w_q1, t_top = peak_snr(band_series(x, None))
        xd = redisperse(x, dm)
    # the burst on the bottom-referenced T_ext grid: top-referenced q1 + the sweep to nu_bot
    sweep_q1 = K_DM_MS_GHZ2 * dm * (1.311280517578125 ** -2 - 1.4987347412109375 ** -2) / Q1_MS
    t_ext_pred = int(round((t_top + sweep_q1) / 2.0))
    i_trig = int(np.argmin(np.abs(plan.fine_dm - dm)))
    w = max(1, int(round(w_q1 / 2.0)))
    lo, hi = max(0, t_ext_pred - 256), min(xd.shape[0] // 2, t_ext_pred + 256)
    win = emulate_2x(xd, plan, t_window=(lo, hi))
    z = robust_rows(boxcar_rows(win, w))
    row = z[i_trig]
    near = slice(max(0, t_ext_pred - lo - 8), t_ext_pred - lo + 8)
    t_trig = near.start + int(np.nanargmax(row[near])) if np.isfinite(row[near]).any() else t_ext_pred - lo
    feats, _ = ridge_features(z, plan.fine_dm, dm_trig=float(plan.fine_dm[i_trig]), t_trig=t_trig,
                              width=w, cfg=cfg)
    lab = spectral_label(h, data, dm, dispersed, t_top, w_q1)
    save_plane(planes_dir, name, z, plan.fine_dm, t_trig=t_trig, width=w, dm_trig=float(plan.fine_dm[i_trig]),
               **lab)
    return {"dispersed_input": bool(dispersed), "snr_fil": round(float(max(s_as, s_dd)), 2),
            "w_ext": w, "coverage": round(float(np.isfinite(win).mean()), 3), **lab,
            **{f"r_{k}": v for k, v in feats.to_dict().items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy-keep", help="keep-list: legacy events with Level3/<ev>_dev_polcal_I.fil")
    ap.add_argument("--dsart-glob", default="", help="glob of dsart C3 .fil files (DM from Level3 c2)")
    ap.add_argument("--fil", nargs="*", default=[], help="EV=PATH:DM")
    ap.add_argument("--planes", default="", help="directory for the normalised planes (re-scoring)")
    a = ap.parse_args(argv)
    plan = TieredDmPlan.from_npz(a.plan)
    jobs = []
    if a.legacy_keep:
        for ev in Path(a.legacy_keep).read_text().split():
            d = f"/dataz/dsa110/candidates/{ev}"
            p = f"{d}/Level3/{ev}_dev_polcal_I.fil"
            try:
                meta = json.load(open(f"{d}/Level3/{ev}.json"))
            except (OSError, ValueError):
                continue
            if os.path.exists(p) and meta.get("dm"):
                jobs.append((ev, p, float(meta["dm"]), "legacy", {"snr_cat": meta.get("snr"),
                                                                    "injected": meta.get("injected")}))
    if a.dsart_glob:
        for p in sorted(glob.glob(a.dsart_glob)):
            if p.endswith("_rfi.fil"):
                continue
            ev = os.path.basename(p)[:-4]
            evdir = p.split("/filterbank/")[0]
            try:
                c2 = json.load(open(f"{evdir}/Level3/{ev}.json")).get("c2", {})
            except (OSError, ValueError):
                continue
            if c2.get("dm_median"):
                jobs.append((ev, p, float(c2["dm_median"]), "dsart", {"snr_cat": c2.get("snr_max")}))
    for spec in a.fil:
        ev, rest = spec.split("=", 1)
        p, dm = rest.rsplit(":", 1)
        jobs.append((ev, p, float(dm), "manual", {}))
    with open(a.out, "a") as fh:
        for ev, p, dm, src, extra in jobs:
            t0 = time.time()
            try:
                rec = {"event": ev, "source": src, "dm": dm, **extra,
                       **truth_one(p, dm, plan, name=ev, planes_dir=a.planes)}
            except Exception as exc:  # noqa: BLE001
                rec = {"event": ev, "source": src, "dm": dm, "error": repr(exc)[:200]}
            rec["sec"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps({k: rec.get(k) for k in ("event", "source", "dm", "snr_fil", "dispersed_input",
                  "frac_best_1mhz", "bw90_mhz", "r_core", "r_peak_sigma", "r_freq_mhz", "r_occupied_mhz",
                  "r_ridge_frac", "r_reason", "error", "sec")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
