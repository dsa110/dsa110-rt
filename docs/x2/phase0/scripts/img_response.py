"""Phase 0, study 4: point-source imaging response of per-unique-baseline entries vs gridded cells.

At the 2x op-point (csf 2: 16 chgroups x 192 summed channels, 3321 core baselines), compare
  A  today's semantics (gridded-cells format): every (baseline, summed channel) at its OWN nearest
     cell (pillbox, production convention: u = -dE/lambda, v = -dN/lambda, no dec projection,
     45" pixel / 17.905-lambda cell, 256 grid, fftshift, real image == Hermitian half-spectrum);
  B  per-unique-baseline entries (n_sub 2 or 4): each exact 3-D redundancy class (5 cm) summed over
     the sub-band's channels and members (natural weighting = SUM), placed at the nearest cell of
     the class member-mean uv at the sub-band CENTRE frequency;
  Bx the same entries at their EXACT (unrounded) uv (DFT) -- isolates bandwidth smearing;
  ideal: every (baseline, channel) at its exact uv -> peak exactly 1 at the true position.

Sky model: unit point source at apparent image offset (x, y) pixels (x along l / columns / E-W,
y along m / rows / N-S): V_bc = exp(+2 pi i (dE_b l + dN_b m) f_c / c), l = x * pix, m = y * pix.
Working in the image's own (apparent) coordinates is exact for the A-vs-B comparison: both schemes
sum the identical V_bc, so any per-(baseline, channel) phase common to both (the planar-w fringe
stop, the declination foreshortening of m, the (n - 1) term) cancels from the comparison; the only
term not captured, the core's height scatter (|dU| < 1 cm), is a < 1 cm baseline error.
The primary-beam weighting uses true sky offsets: l = x pix, m_true = y pix / cos(dec - lat).

    python img_response.py        -> results/imaging_per_baseline.json, imaging_per_baseline_map.npz
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from valuemaps import (C_M_S, CELL, N_GRID, RES, core_baselines, redundancy_classes,  # noqa: E402
                       summed_freqs_ghz)

PIX = 1.0 / (N_GRID * CELL)                       # rad per pixel (45 arcsec)
ARCSEC_PER_PIX = PIX * 180 / np.pi * 3600
COSZ = np.cos(np.radians(71.63 - 37.234))
H = N_GRID // 2
TWO_PI_N = 2 * np.pi / N_GRID


class Geometry:
    def __init__(self, csf=2, nsubs=(2, 4)):
        du, dv, _dw, _, _ = core_baselines()
        cls, flip, mult = redundancy_classes()
        sgn = np.where(flip, -1.0, 1.0)
        # canonical orientation (identical real image to the raw xGPU orientation: rint is odd)
        self.du, self.dv = sgn * du, sgn * dv
        self.cls, self.mult = cls, mult
        nus = summed_freqs_ghz(csf)                       # (16, nsum)
        nsum = nus.shape[1]
        self.f = nus.ravel() * 1e9                        # channel c = g * nsum + k
        lam = C_M_S / self.f
        col = np.rint(-self.du[:, None] / lam[None, :] / CELL).astype(np.int64) + H
        row = np.rint(-self.dv[:, None] / lam[None, :] / CELL).astype(np.int64) + H
        assert ((col >= 0) & (col < N_GRID) & (row >= 0) & (row < N_GRID)).all()
        self.keyA = (row * N_GRID + col).ravel()
        self.N = float(self.keyA.size)                    # contributions per image
        self.B = {}
        nb_cls = len(mult)
        rep_du = np.bincount(cls, self.du) / mult         # member-mean uv per class (m)
        rep_dv = np.bincount(cls, self.dv) / mult
        for n_sub in nsubs:
            per = nsum // n_sub
            ci = np.arange(self.f.size)
            stream_of_c = (ci // nsum) * n_sub + (ci % nsum) // per
            n_st = N_CHG_ * n_sub
            entry = (cls[:, None] * n_st + stream_of_c[None, :]).ravel()
            f_ctr = np.array([nus[g, s * per:(s + 1) * per].mean() for g in range(N_CHG_)
                              for s in range(n_sub)]) * 1e9
            e_cls = np.repeat(np.arange(nb_cls), n_st)
            e_st = np.tile(np.arange(n_st), nb_cls)
            lam_e = C_M_S / f_ctr[e_st]
            kc = -rep_du[e_cls] / lam_e / CELL            # exact cell coordinates (not rounded)
            kr = -rep_dv[e_cls] / lam_e / CELL
            cell = (np.rint(kr).astype(np.int64) + H) * N_GRID + (np.rint(kc).astype(np.int64) + H)
            self.B[n_sub] = dict(entry=entry, n_entries=nb_cls * n_st, cell=cell, kc=kc, kr=kr,
                                 n_e=mult[e_cls] * per)

    # -- visibilities and grids ----------------------------------------------------------
    def vis(self, x, y):
        phi = 2 * np.pi * (self.du * x * PIX + self.dv * y * PIX) / C_M_S        # rad / Hz
        ph = np.outer(phi, self.f).ravel()
        return np.cos(ph), np.sin(ph)

    def grid_A(self, re, im):
        return (np.bincount(self.keyA, re, minlength=N_GRID ** 2)
                + 1j * np.bincount(self.keyA, im, minlength=N_GRID ** 2))

    def entries(self, n_sub, re, im):
        b = self.B[n_sub]
        return (np.bincount(b["entry"], re, minlength=b["n_entries"])
                + 1j * np.bincount(b["entry"], im, minlength=b["n_entries"]))

    def grid_B(self, n_sub, ev):
        c = self.B[n_sub]["cell"]
        return (np.bincount(c, ev.real, minlength=N_GRID ** 2)
                + 1j * np.bincount(c, ev.imag, minlength=N_GRID ** 2))


N_CHG_ = 16


def fft_image(G, norm):
    """Real dirty image [row, col] (row = 128 + y, col = 128 + x), peak 1 for an ideal unit source."""
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G.reshape(N_GRID, N_GRID)))).real
    return img * (N_GRID ** 2) / norm


class Evaluator:
    """Continuous dirty image Re sum_k v_k exp(2 pi i (kc x + kr y) / N) / norm, with derivatives."""

    def __init__(self, kc, kr, v, norm):
        self.kc, self.kr, self.v, self.norm = kc, kr, v, norm

    @classmethod
    def from_grid(cls, G, norm):
        nz = np.flatnonzero(G)
        return cls((nz % N_GRID - H).astype(float), (nz // N_GRID - H).astype(float), G[nz], norm)

    def __call__(self, xs, ys):
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        out = np.empty(xs.size)
        for i0 in range(0, xs.size, 64):
            th = TWO_PI_N * (np.outer(xs[i0:i0 + 64], self.kc) + np.outer(ys[i0:i0 + 64], self.kr))
            out[i0:i0 + 64] = (np.exp(1j * th) @ self.v).real
        return out / self.norm

    def newton(self, x, y, it=8):
        for _ in range(it):
            e = np.exp(1j * TWO_PI_N * (self.kc * x + self.kr * y)) * self.v
            a, b = 1j * TWO_PI_N * self.kc, 1j * TWO_PI_N * self.kr
            gx, gy = (e * a).sum().real, (e * b).sum().real
            hxx, hyy, hxy = (e * a * a).sum().real, (e * b * b).sum().real, (e * a * b).sum().real
            det = hxx * hyy - hxy ** 2
            if det <= 0 or hxx >= 0:
                break
            dx = -(hyy * gx - hxy * gy) / det
            dy = -(-hxy * gx + hxx * gy) / det
            dx, dy = np.clip(dx, -0.2, 0.2), np.clip(dy, -0.2, 0.2)
            x, y = x + dx, y + dy
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                break
        return float(self(x, y)[0]), x, y

    def peak(self, x0, y0):
        g = np.arange(-3, 4) * 0.2
        X, Y = np.meshgrid(x0 + g, y0 + g)
        v = self(X.ravel(), Y.ravel())
        j = int(np.argmax(v))
        return self.newton(X.ravel()[j], Y.ravel()[j])


def pixel_peak(img, x0, y0, r=3):
    """Max pixel near the expected position + 1-D parabolic sub-pixel refinement per axis."""
    r0, c0 = H + int(round(y0)), H + int(round(x0))
    sub = img[r0 - r:r0 + r + 1, c0 - r:c0 + r + 1]
    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    R, Cc = r0 - r + i, c0 - r + j

    def para(m1, z, p1):
        d = m1 - 2 * z + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d

    dy = para(img[R - 1, Cc], img[R, Cc], img[R + 1, Cc])
    dx = para(img[R, Cc - 1], img[R, Cc], img[R, Cc + 1])
    return float(img[R, Cc]), Cc - H + dx, R - H + dy


def beam2_weight(x, y):
    sig = np.radians(3.2) / 2.3548
    l, m = x * PIX, y * PIX / COSZ
    return np.exp(-(l ** 2 + m ** 2) / sig ** 2)


def analyse_position(G, x, y, nsubs=(2, 4), want_images=False):
    re, im = G.vis(x, y)
    out = {}
    GA = G.grid_A(re, im)
    evA = Evaluator.from_grid(GA, G.N)
    pa = evA.peak(x, y)
    imgA = fft_image(GA, G.N)
    out["A"] = dict(peak=pa[0], dx=pa[1] - x, dy=pa[2] - y, at_truth=float(evA(x, y)[0]))
    pp = pixel_peak(imgA, x, y)
    out["A"].update(pix_peak=pp[0], pix_dx=pp[1] - x, pix_dy=pp[2] - y)
    imgs = {"A": imgA}
    for n_sub in nsubs:
        b = G.B[n_sub]
        ev = G.entries(n_sub, re, im)
        GB = G.grid_B(n_sub, ev)
        evB = Evaluator.from_grid(GB, G.N)
        pb = evB.peak(x, y)
        imgB = fft_image(GB, G.N)
        evX = Evaluator(b["kc"], b["kr"], ev, G.N)
        bx_truth = float(evX(x, y)[0])
        rec = dict(peak=pb[0], dx=pb[1] - x, dy=pb[2] - y, at_truth=float(evB(x, y)[0]),
                   exact_at_truth=bx_truth)
        pp = pixel_peak(imgB, x, y)
        rec.update(pix_peak=pp[0], pix_dx=pp[1] - x, pix_dy=pp[2] - y,
                   maxdiff_vs_A=float(np.abs(imgB - imgA).max() / imgA.max()))
        if want_images:
            px = evX.peak(x, y)
            rec.update(exact_peak=px[0], exact_dx=px[1] - x, exact_dy=px[2] - y)
        out[f"B{n_sub}"] = rec
        imgs[f"B{n_sub}"] = imgB
    return (out, imgs) if want_images else out


def lobe(ev, x0, y0):
    """FWHM and first-minimum radius along x and y through (x0, y0) of a continuous image."""
    t = np.arange(0, 8, 0.005)
    res = {}
    for ax in ("x", "y"):
        vals_p = ev(x0 + t, np.full_like(t, y0)) if ax == "x" else ev(np.full_like(t, x0), y0 + t)
        vals_m = ev(x0 - t, np.full_like(t, y0)) if ax == "x" else ev(np.full_like(t, x0), y0 - t)
        pk = vals_p[0]
        hw = []
        fm = []
        for vals in (vals_p, vals_m):
            i = int(np.argmax(vals < 0.5 * pk))
            hw.append(t[i - 1] + (vals[i - 1] - 0.5 * pk) / (vals[i - 1] - vals[i]) * 0.005)
            d = np.diff(vals)
            fm.append(t[int(np.argmax(d > 0))])
        res[f"fwhm_{ax}_px"] = float(hw[0] + hw[1])
        res[f"first_min_{ax}_px"] = float(min(fm))
    return res


def sidelobes(img, x0, y0, rx, ry):
    yy, xx = np.mgrid[0:N_GRID, 0:N_GRID]
    main = ((xx - H - x0) / rx) ** 2 + ((yy - H - y0) / ry) ** 2 <= 1.0
    s = img[~main]
    pk = img[main].max()
    return dict(peak_sidelobe=float(s.max() / pk), most_negative=float(s.min() / pk),
                rms_sidelobe=float(s.std() / pk))


def noise_check(G, nsubs=(2, 4), n_real=2, seed=7):
    rng = np.random.default_rng(seed)
    out = {"theory_std": float(1.0 / np.sqrt(G.N))}
    acc = {k: [] for k in ["A"] + [f"B{n}" for n in nsubs]}
    for _ in range(n_real):
        re = rng.standard_normal(int(G.N))
        im = rng.standard_normal(int(G.N))
        acc["A"].append(fft_image(G.grid_A(re, im), G.N).std())
        for n in nsubs:
            acc[f"B{n}"].append(fft_image(G.grid_B(n, G.entries(n, re, im)), G.N).std())
    for k, v in acc.items():
        out[f"{k}_std"] = float(np.mean(v))
        out[f"{k}_std_over_theory"] = float(np.mean(v) * np.sqrt(G.N))
    return out


def bw_smear_formula(x, y, n_sub, cosz=False):
    """docs/timeres_study/model/bw_smear.py: mean over baselines of sinc((u l + v m) B / nu)."""
    du, dv, _, _, _ = core_baselines()
    B = 11.71875e6 / n_sub
    v = dv * (COSZ if cosz else 1.0)
    return float(np.mean(np.sinc((du * x * PIX + v * y * PIX) * B / C_M_S)))


NAMED = {"centre": (0, 0), "half-way EW": (64, 0), "half-way NS": (0, 64), "edge EW": (120, 0),
         "edge NS": (0, 120), "corner": (120, 120), "anti-corner": (120, -120),
         "centre +0.5px": (0.5, 0.5), "corner +0.5px": (120.5, 120.5)}


def main():
    os.nice(10)
    t0 = time.time()
    G = Geometry()
    print(f"geometry built: N = {G.N:.0f} contributions, entries n_sub2 {G.B[2]['n_entries']}, "
          f"n_sub4 {G.B[4]['n_entries']} [{time.time() - t0:.1f}s]", flush=True)
    out = {"pix_arcsec": ARCSEC_PER_PIX, "N": G.N,
           "N_B": {str(n): float(G.B[n]["n_e"].sum()) for n in (2, 4)}}

    # -- named positions --------------------------------------------------------------------
    named = {}
    for name, (x, y) in NAMED.items():
        r, imgs = analyse_position(G, x, y, want_images=True)
        for n in (2, 4):
            r[f"B{n}"]["bw_smear_formula_nocosz"] = bw_smear_formula(x, y, n)
            r[f"B{n}"]["bw_smear_formula_cosz"] = bw_smear_formula(x, y, n, cosz=True)
        named[name] = r
        print(name, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.items()}),
              f"[{time.time() - t0:.0f}s]", flush=True)
    out["named"] = named

    # -- dirty beam (centre) and off-centre lobes -------------------------------------------
    beams = {}
    for name in ("centre", "edge EW", "edge NS", "corner"):
        x, y = NAMED[name]
        re, im = G.vis(x, y)
        GA = G.grid_A(re, im)
        grids = {"A": GA}
        for n in (2, 4):
            grids[f"B{n}"] = G.grid_B(n, G.entries(n, re, im))
        rec = {}
        lobeA = None
        for k, Gk in grids.items():
            ev = Evaluator.from_grid(Gk, G.N)
            pk = ev.peak(x, y)
            lb = lobe(ev, pk[1], pk[2])
            if k == "A":
                lobeA = lb
            img = fft_image(Gk, G.N)
            sl = sidelobes(img, x, y, lobeA["first_min_x_px"], lobeA["first_min_y_px"])
            rec[k] = {**lb, **sl}
        beams[name] = rec
        print("lobe", name, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rec.items()}),
              flush=True)
    out["beams"] = beams

    # -- noise per unit signal ---------------------------------------------------------------
    out["noise"] = noise_check(G)
    print("noise", out["noise"], flush=True)

    # -- 2-D map over the field (half plane computed, mirrored: the real image of a source at
    #    -(x, y) is the mirror image) ------------------------------------------------------------
    grid = np.arange(-120, 121, 15)
    n = grid.size
    keys = ["A_peak", "A_dx", "A_dy", "A_truth"]
    for b in (2, 4):
        keys += [f"B{b}_peak", f"B{b}_dx", f"B{b}_dy", f"B{b}_truth", f"B{b}_exact"]
    M = {k: np.full((n, n), np.nan) for k in keys}
    for iy, y in enumerate(grid):
        for ix, x in enumerate(grid):
            if y < 0 or (y == 0 and x < 0):
                continue
            r = analyse_position(G, float(x), float(y))
            vals = {"A_peak": r["A"]["peak"], "A_dx": r["A"]["dx"], "A_dy": r["A"]["dy"],
                    "A_truth": r["A"]["at_truth"]}
            for b in (2, 4):
                q = r[f"B{b}"]
                vals.update({f"B{b}_peak": q["peak"], f"B{b}_dx": q["dx"], f"B{b}_dy": q["dy"],
                             f"B{b}_truth": q["at_truth"], f"B{b}_exact": q["exact_at_truth"]})
            for k, v in vals.items():
                M[k][iy, ix] = v
                sgn = -1.0 if k.endswith(("_dx", "_dy")) else 1.0
                M[k][n - 1 - iy, n - 1 - ix] = sgn * v
        print(f"map row y={y} [{time.time() - t0:.0f}s]", flush=True)
    X, Y = np.meshgrid(grid, grid)
    w = beam2_weight(X, Y)
    summ = {}
    for b in (2, 4):
        ratio = M[f"B{b}_peak"] / M["A_peak"]
        offB = np.hypot(M[f"B{b}_dx"], M[f"B{b}_dy"])
        offBA = np.hypot(M[f"B{b}_dx"] - M["A_dx"], M[f"B{b}_dy"] - M["A_dy"])
        summ[f"n_sub{b}"] = dict(
            ratio_BA_beam2=float((ratio * w).sum() / w.sum()), ratio_BA_min=float(ratio.min()),
            ratio_BA_uniform_mean=float(ratio.mean()),
            Bexact_beam2=float((M[f"B{b}_exact"] * w).sum() / w.sum()), Bexact_min=float(M[f"B{b}_exact"].min()),
            B_over_ideal_beam2=float((M[f"B{b}_peak"] * w).sum() / w.sum()),
            offset_B_vs_truth_max_px=float(offB.max()), offset_B_vs_truth_rms_px=float(np.sqrt((offB ** 2).mean())),
            offset_B_vs_A_max_px=float(offBA.max()), offset_B_vs_A_rms_px=float(np.sqrt((offBA ** 2).mean())),
            n_positions_offset_B_vs_A_gt_0p1=int((offBA > 0.1).sum()),
            n_positions_offset_B_vs_truth_gt_0p1=int((offB > 0.1).sum()))
    offA = np.hypot(M["A_dx"], M["A_dy"])
    summ["A"] = dict(A_over_ideal_beam2=float((M["A_peak"] * w).sum() / w.sum()),
                     A_over_ideal_min=float(M["A_peak"].min()),
                     offset_A_vs_truth_max_px=float(offA.max()),
                     offset_A_vs_truth_rms_px=float(np.sqrt((offA ** 2).mean())),
                     n_positions_offset_A_vs_truth_gt_0p1=int((offA > 0.1).sum()))
    out["map_summary"] = summ
    out["map_grid_px"] = grid.tolist()
    print("map summary", json.dumps(summ, indent=1), flush=True)
    np.savez_compressed(os.path.join(RES, "imaging_per_baseline_map.npz"), grid=grid, w=w, **M)
    json.dump(out, open(os.path.join(RES, "imaging_per_baseline.json"), "w"), indent=1)
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
