"""Study 1e: hot-cell robustness and the exactly-zero-value bias of a mid-rise code.

(a) Hot value: one value of a row carries H sigma (H = 10, 50, 1000) on top of noise (a narrow RFI
    line on one short baseline, a bad correlator cell).  Row efficiency (point-source S/N of the
    OTHER values of the row) for today's per-row max-abs cint8, a hypothetical 4-bit max-abs, and
    the 2x rule (static sigma_hat); plus the image artefact of the clipped hot value per sample.
(b) Zero bias: a value whose sources are all flagged is exactly 0.  A mid-rise code has no zero
    level, so it decodes to (+0.5 + 0.5i) step -> a CONSTANT, coherent offset every sample.  Cases:
    one whole stream zero (chgroup sub-band flagged / zero-coded fill), one antenna flagged
    (per-baseline: its singleton entries are zero; gridded: cells fed only by it are zero).
    Offset at the phase centre in image sigma, and the S/N it produces in a b64 boxcar
    (sigma_k = 1.163 sqrt(64), audit live value).  Mid-tread: exactly 0.
-> results/rowrule_hot.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from rr_lib import Quant
from valuemaps import RES, core_baselines, load, redundancy_classes, summed_freqs_ghz, cell_of

OPT = {4: 0.3534, 5: 0.1932}
OPT_MR = {4: 0.3352, 5: 0.1881}


def row_eff(n, enc, hot_idx, H, n_rows=400, seed=5):
    """Efficiency of a row's point-source projection EXCLUDING the hot value, vs unquantised."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, 2, n.size)) * np.sqrt(n)
    x[:, 0, hot_idx] += H * np.sqrt(n[hot_idx])
    keep = np.ones(n.size, bool)
    keep[hot_idx] = False
    d = enc(x, n)                                          # decoded, natural units (x-like)
    p_opt = (x[:, 0, keep]).sum(axis=1) - 0.0
    p_q = d[:, 0, keep].sum(axis=1)
    p_opt = p_opt - p_opt.mean()
    p_q = p_q - p_q.mean()
    den = math.sqrt(np.sum(p_q ** 2) * np.sum(p_opt ** 2))
    return float(np.sum(p_q * p_opt) / den) if den > 0 else 0.0     # crushed row: every code 0


def enc_maxabs(bits):
    top = 2 ** (bits - 1) - 1
    def f(x, n):
        amax = np.abs(x).reshape(x.shape[0], -1).max(axis=1)
        return np.clip(np.round(x * (top / amax)[:, None, None]), -top - 1, top)
    return f


def enc_2x(bits, s_h, thr):
    qm, qh = Quant("midtread", bits, OPT[bits]), Quant("midtread", 8, s_h)
    def f(x, n):
        y = x / np.sqrt(n)
        return np.where(n >= thr, qh(y), qm(y)) * np.sqrt(n)
    return f


def zero_bias(fmt, bits=4, n_flag_ants=1):
    """Image offset (sigma) at the phase centre from exactly-zero values under a mid-rise code."""
    vm = load(fmt)
    n_all = vm["n"].astype(float)
    N = n_all.sum()
    s = OPT_MR[bits]
    sigma_img = math.sqrt(N)                      # decoded-unit image noise x N_grid^2 (V ~ 1)
    out = {}
    # (i) one whole stream zero
    sid = vm["stream"] == 0
    off = 0.5 * s * np.sqrt(n_all[sid]).sum()     # Re part at the centre; Im adds elsewhere
    out["one_stream_zero_sigma"] = off / sigma_img
    # (ii) flagged antennas
    du, dv, dw, a_l, b_l = core_baselines()
    ants = np.unique(np.concatenate([a_l, b_l]))
    worst, mean = 0.0, []
    kind = str(vm["kind"])
    if kind == "baselines":
        cls, flip, mult = redundancy_classes()
        per = int(vm["n"].min() // 1) if False else None
        n_sub = int(vm["n_sub"])
        nch = 192 // n_sub
        for ant in ants:
            hit = (a_l == ant) | (b_l == ant)
            # classes whose EVERY member touches the antenna -> zero
            cnt_hit = np.bincount(cls[hit], minlength=len(mult))
            zero_cls = np.where(cnt_hit == mult)[0]
            # all 16 x n_sub streams, value n = mult * nch
            o = 0.5 * s * (16 * n_sub) * np.sqrt(mult[zero_cls] * nch).sum() / sigma_img
            mean.append(o)
            worst = max(worst, o)
    else:
        csf, n_sub = 2, int(vm["n_sub"])
        nus = summed_freqs_ghz(csf)
        per = nus.shape[1] // n_sub
        for ant in ants[:: max(1, len(ants) // 12)]:            # a sample of antennas (cost)
            hit = (a_l == ant) | (b_l == ant)
            o = 0.0
            for g in range(16):
                for sb in range(n_sub):
                    f = nus[g, sb * per:(sb + 1) * per]
                    row, col, ok = cell_of(du[:, None], dv[:, None], f[None, :])
                    key = row * 256 + col
                    allk, cnt_all = np.unique(key[ok], return_counts=True)
                    hk, cnt_hit = np.unique(key[ok & hit[:, None]], return_counts=True)
                    m = dict(zip(allk.tolist(), cnt_all.tolist()))
                    zero = [k for k, c in zip(hk.tolist(), cnt_hit.tolist()) if m[k] == c]
                    o += 0.5 * s * sum(math.sqrt(m[k]) for k in zero)
            o /= sigma_img
            mean.append(o)
            worst = max(worst, o)
    out["one_antenna_flagged_sigma_mean"] = float(np.mean(mean))
    out["one_antenna_flagged_sigma_max"] = float(worst)
    out["b64_snr_per_sigma_offset"] = 64 / (1.163 * 8.0)
    return out


def main():
    out = {"hot": {}, "zero_bias_midrise": {}}
    # (a) hot value: today (1x n's) and 2x g_ns2 / b_ns4, on stream 0; hot value = the heaviest
    #     and a median one
    for fmt in ("today_g_ns4_csf8", "g_ns2", "b_ns4"):
        vm = load(fmt)
        n = vm["n"][vm["stream"] == 0].astype(float)
        N = vm["n"].sum()
        thr = 1e-4 * N if not fmt.startswith("today") else np.inf
        s_h = 2 * 250 * math.sqrt(n.max() / N) / 127
        for which, idx in (("heaviest", int(np.argmax(n))), ("median", int(np.argsort(n)[n.size // 2]))):
            for H in (0.0, 10.0, 50.0, 1000.0):
                row = {"n_hot": float(n[idx])}
                if fmt.startswith("today"):
                    row["maxabs8"] = row_eff(n, enc_maxabs(8), idx, H)
                    row["maxabs4"] = row_eff(n, enc_maxabs(4), idx, H)
                else:
                    row["rule2x_4bit"] = row_eff(n, enc_2x(4, s_h, thr), idx, H)
                    q = Quant("midtread", 8, s_h) if n[idx] >= thr else Quant("midtread", 4, OPT[4])
                    row["hot_clip_sigma"] = q.clip
                    row["artefact_image_sigma_per_sample"] = q.clip * math.sqrt(n[idx]) / math.sqrt(N)
                out["hot"][f"{fmt}|{which}|H{H:g}"] = row
                print(fmt, which, H, row, flush=True)
    for fmt in ("g_ns2", "g_ns4", "b_ns2", "b_ns4"):
        out["zero_bias_midrise"][fmt] = zero_bias(fmt)
        print(fmt, out["zero_bias_midrise"][fmt], flush=True)
    json.dump(out, open(os.path.join(RES, "rowrule_hot.json"), "w"), indent=1)


if __name__ == "__main__":
    os.nice(10)
    main()
