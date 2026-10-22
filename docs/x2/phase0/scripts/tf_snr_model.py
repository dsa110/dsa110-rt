"""COPY of docs/timeres_study/snr2x/snr_model.py (Phase 0 study 5), extended at the end with general
matched-filter banks (boxcar / B2 triangle / B3 quadratic B-spline / Gaussian) and extra pulse shapes.
The original model code above the "Phase 0 extension" marker is unchanged except for the WT path.

S/N loss of a two-stage dedispersion + boxcar search, against an IDEAL matched filter.

One generic model of the dsart chain, used for both the deployed scheme and the proposed
2x scheme, so the comparison is like for like:

  native channels (16 x 384, dsart.common.constants.freq_GHz) carry their own dispersion
  smear (box of 8.2976 us DM dnu / nu^3);
  `csf` native channels are SUMMED before any shift (as the fast GEMM does), at the
  band-centre frequency production uses (build_summed_chgroup_freq_table_GHz);
  stage 1 (corr) advances each summed channel to its SUB-BAND reference (the sub-band's
  first summed channel) at the COARSE DM, rounded twice exactly as
  DMPlan.delay_bins_per_subband does: to native samples, then to the stage-1 quantum q1;
  stage 2 delays each (chgroup, sub-band) stream to the band bottom at the FINE DM with
  ONE rounding to the stream sample q2 (the merged-rounding form,
  compute_time_shift_search(merge_coarse_rounding=True, n_sub>1));
  the stream is binned at q2 with a random arrival phase and searched with boxcars of
  1, 2, ..., 64 samples at every start; the best of the nearest DM trials is kept.

For the deployed plan, the integer shifts this produces are checked bit-for-bit against
the production functions (validate.py).

Loss factors, each a ratio of S/N (white noise, unit-fluence pulse, Gaussian FWHM W):
  smear   - channel smearing only: exact per-summed-channel shifts at the TRUE DM, ideal
            continuous matched filter on the smeared profile;
  sample  - the same profile binned at the search sample and searched with the boxcar
            ladder, averaged over arrival phase;
  dmgrid  - exact (unrounded) shifts but at the best nearby TRIAL DM;
  integer - the real two-stage integer shifts (stage-1 + stage-2 rounding, residual
            sweep inside each sub-band between the coarse and fine DM).
The total is the product; each factor is reported as the ratio to the previous step.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

WT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))   # worktree root (copy lives 4 deep)
_PKG = os.path.join(WT, "src", "dsart")
if "dsart" not in sys.modules:          # bind dsart to THIS worktree, not an editable install elsewhere
    _spec = importlib.util.spec_from_file_location("dsart", _PKG + "/__init__.py",
                                                   submodule_search_locations=[_PKG])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["dsart"] = _mod
    _spec.loader.exec_module(_mod)
    # coarse_dm/ and fine_dm/ __init__ pull in torch kernels; the plan/shift modules used here
    # are pure numpy, so register bare packages and import those modules directly.
    import types
    for _sub in ("coarse_dm", "fine_dm"):
        _p = types.ModuleType("dsart." + _sub)
        _p.__path__ = [os.path.join(_PKG, _sub)]
        sys.modules["dsart." + _sub] = _p

from dsart.coarse_dm.dm_plan import build_summed_chgroup_freq_table_GHz  # noqa: E402
from dsart.common.constants import (DELTA_NU_CH_GHZ, K_DM_MS_GHZ2_PC, N_CHGROUP,  # noqa: E402
                                    NATIVE_SAMPLE_US, NCHAN_PER_CHGROUP, NU_BOT_PROC_GHZ, freq_GHz)

K = K_DM_MS_GHZ2_PC
T_NAT = NATIVE_SAMPLE_US
NU = np.array([[freq_GHz(g, c) for c in range(NCHAN_PER_CHGROUP)] for g in range(N_CHGROUP)])
NU_BOT = float(NU_BOT_PROC_GHZ)
A_TRUE = 1e3 * K * (1.0 / NU ** 2 - 1.0 / NU_BOT ** 2)          # (16, 384) us per pc/cc, <= 0
SMEAR_COEF = 8.297616 * DELTA_NU_CH_GHZ * 1e3 / NU ** 3          # us per pc/cc, native channel
DT = T_NAT / 8.0                                                  # 4.096 us profile grid
NGRID = 1 << 16                                                   # 268 ms window
LADDER = (1, 2, 4, 8, 16, 32, 64)                                 # live detector boxcars
N_PHASE = 16


class Scheme:
    """A two-stage plan: per-tier (fine, coarse, f2c, q1, q2) at one channel width."""

    def __init__(self, name, csf, n_sub, tiers, ladder=LADDER):
        """tiers: list of dict(fine, coarse, f2c, q1, q2); q2 is that tier's search sample."""
        self.name, self.csf, self.n_sub, self.tiers, self.ladder = name, csf, n_sub, tiers, tuple(ladder)
        nus = build_summed_chgroup_freq_table_GHz(csf)                 # (16, 384/csf)
        nsum = nus.shape[1]
        per = nsum // n_sub
        refs = nus[:, ::per][:, :n_sub]                                # (16, n_sub)
        k = np.arange(NCHAN_PER_CHGROUP) // csf                        # native -> summed
        s = k // per                                                   # native -> sub-band
        self.nu_sum = nus[:, k]                                        # (16, 384) per native ch
        self.a1 = 1e3 * K * (1.0 / nus ** 2 - 1.0 / np.repeat(refs, per, axis=1) ** 2)   # (16, nsum)
        self.a2 = 1e3 * K * (1.0 / NU_BOT ** 2 - 1.0 / refs ** 2)      # (16, n_sub)
        self.k, self.s = k, s
        self.trials = []                                               # (tier, fine index, dm)
        for ti, t in enumerate(tiers):
            for fi, d in enumerate(t["fine"]):
                self.trials.append((ti, fi, float(d)))
        self.trial_dm = np.array([d for _, _, d in self.trials])

    # -- integer shifts (what the pipeline applies) --------------------------------------
    def stage1_bins(self, ti, c):
        """(16, nsum) int stage-1 advance in q1 units for coarse index c of tier ti."""
        t = self.tiers[ti]
        nat = np.rint(self.a1 * t["coarse"][c] / T_NAT)
        return np.rint(nat / (t["q1"] / T_NAT))

    def stage2_bins(self, ti, fi):
        """(16, n_sub) int stage-2 delay in q2 units (one merged rounding at the fine DM)."""
        t = self.tiers[ti]
        return np.rint(self.a2 * t["fine"][fi] / t["q2"])

    def applied_us(self, ti, fi):
        """(16, 384) applied delay per native channel, us (positive = delayed)."""
        t = self.tiers[ti]
        s1 = self.stage1_bins(ti, t["f2c"][fi])[:, self.k] * t["q1"]
        s2 = self.stage2_bins(ti, fi)[:, self.s] * t["q2"]
        return s2 - s1

    def exact_us(self, dm):
        """(16, 384) exact per-SUMMED-channel delay to the band bottom (no rounding)."""
        return 1e3 * K * dm * (1.0 / NU_BOT ** 2 - 1.0 / self.nu_sum ** 2)

    def tier_T(self, dm):
        """Search sample of the tier whose fine range contains dm (nearest tier otherwise)."""
        best = None
        for t in self.tiers:
            lo, hi = t["fine"][0], t["fine"][-1]
            dist = 0.0 if lo <= dm <= hi else min(abs(dm - lo), abs(dm - hi))
            if best is None or dist < best[0]:
                best = (dist, t["q2"])
        return best[1]

    def dm_range(self):
        return min(t["fine"][0] for t in self.tiers), max(t["fine"][-1] for t in self.tiers)


class OneStage(Scheme):
    """Single-shot incoherent dedispersion of a filterbank: every (summed) channel is delayed
    by ONE integer number of samples, rint(delay(trial DM, channel centre) / T)."""

    def __init__(self, name, csf, fine, T_us, ladder=LADDER):
        tier = dict(fine=np.asarray(fine, float), coarse=np.zeros(1), f2c=np.zeros(len(fine), int),
                    q1=T_us, q2=T_us)
        super().__init__(name, csf, 1, [tier], ladder)
        nus = build_summed_chgroup_freq_table_GHz(csf)
        self.a_full = 1e3 * K * (1.0 / NU_BOT ** 2 - 1.0 / nus ** 2)         # (16, nsum) us per pc/cc

    def applied_us(self, ti, fi):
        t = self.tiers[ti]
        return (t["q2"] * np.rint(self.a_full * t["fine"][fi] / t["q2"]))[:, self.k]


# -- profile + S/N engine ----------------------------------------------------------------
_F = np.fft.rfftfreq(NGRID)


def _gauss_ft(W_us):
    sig = W_us / 2.354820045 / DT
    return np.exp(-2.0 * (np.pi * _F * sig) ** 2)


def _tophat_ft(W_us):
    n = max(1, int(round(W_us / DT)))
    box = np.zeros(NGRID)
    box[:n] = 1.0 / n
    return np.fft.rfft(np.roll(box, -(n // 2)))          # box centred on 0 (keep the phase)


def ideal_norm(W_us, shape="gauss"):
    """sqrt(integral g^2 dt) of a unit-fluence pulse, per sqrt(us)."""
    if shape == "gauss":
        return (1.0 / (2.0 * np.sqrt(np.pi) * W_us / 2.354820045)) ** 0.5
    return (1.0 / max(W_us, DT)) ** 0.5


def histogram(resid_us, dm):
    """Unit-fluence spectrum of the dedispersed series before the intrinsic pulse:
    per-channel arrivals (linear interpolation on the grid) convolved with each chgroup's
    native-channel smear box.  Returns rfft of the (NGRID,) profile, centred."""
    r = resid_us - np.median(resid_us)
    pos = np.clip(r / DT + NGRID / 2, 0, NGRID - 2)
    prof = np.zeros(NGRID)
    for g in range(N_CHGROUP):
        p = pos[g]
        i0 = np.floor(p).astype(int)
        fr = p - i0
        h = np.bincount(i0, weights=1 - fr, minlength=NGRID + 1)[:NGRID]
        h += np.bincount(i0 + 1, weights=fr, minlength=NGRID + 1)[:NGRID]
        wbox = max(1, int(round(SMEAR_COEF[g].mean() * dm / DT)))
        if wbox > 1:
            c = np.concatenate([[0.0], np.cumsum(h)])
            h = c[wbox:] - c[:-wbox]
            h = np.concatenate([np.zeros(wbox // 2), h, np.zeros(wbox - 1 - wbox // 2)]) / wbox
        prof += h
    return np.fft.rfft(prof / prof.sum())


def snr_binned(H, kern_ft, T_us, W_us, shape="gauss", ladder=LADDER):
    """Phase-averaged best-boxcar S/N / ideal, for spectrum H with pulse kernel kern_ft."""
    prof = np.fft.irfft(H * kern_ft, n=NGRID)
    m = int(round(T_us / DT))
    cf = np.concatenate([[0.0], np.cumsum(prof)])
    nb = (NGRID - m) // m
    ph = (np.arange(N_PHASE) * m) // N_PHASE
    C = cf[ph[:, None] + m * np.arange(nb + 1)[None, :]]                # (P, nb+1) cumulative
    best = np.zeros(N_PHASE)
    for w in ladder:
        s = (C[:, w:] - C[:, :-w]).max(axis=1) / np.sqrt(w * T_us)
        best = np.maximum(best, s)
    return float(best.mean()) / ideal_norm(W_us, shape)


def snr_matched(H, kern_ft, W_us, shape="gauss"):
    """Continuous matched-filter S/N / ideal (smearing only, no sampling)."""
    prof = np.fft.irfft(H * kern_ft, n=NGRID)
    return float(np.sqrt(np.sum(prof ** 2) / DT)) / ideal_norm(W_us, shape)


def kernels(widths_us, shape="gauss"):
    return [(_gauss_ft(w) if shape == "gauss" else _tophat_ft(w)) for w in widths_us]


def evaluate(scheme, dm, widths_us, n_near=8, shape="gauss"):
    """Loss chain at one true DM for all widths -> dict of (n_w,) arrays."""
    KS = kernels(widths_us, shape)
    nw = len(widths_us)
    T = scheme.tier_T(dm)
    H = histogram(A_TRUE * dm + scheme.exact_us(dm), dm)
    smear = np.array([snr_matched(H, k, w, shape) for k, w in zip(KS, widths_us)])
    sample = np.array([snr_binned(H, k, T, w, shape, scheme.ladder) for k, w in zip(KS, widths_us)])
    near = np.argsort(np.abs(scheme.trial_dm - dm))[:n_near]
    grid = np.zeros(nw)
    full = np.zeros(nw)
    for j in near:
        ti, fi, d = scheme.trials[j]
        Tt = scheme.tiers[ti]["q2"]
        He = histogram(A_TRUE * dm + scheme.exact_us(d), dm)
        Hi = histogram(A_TRUE * dm + scheme.applied_us(ti, fi), dm)
        for i, (k, w) in enumerate(zip(KS, widths_us)):
            grid[i] = max(grid[i], snr_binned(He, k, Tt, w, shape, scheme.ladder))
            full[i] = max(full[i], snr_binned(Hi, k, Tt, w, shape, scheme.ladder))
    return {"smear": smear, "sample": sample, "dmgrid": grid, "total": full}


# ======================================================================================
# Phase 0 extension: matched-filter banks (study 5, docs/x2/phase0/time_filters.md)
# ======================================================================================
import itertools  # noqa: E402

BANK_WIDTHS = (1, 2, 4, 8, 16, 32, 64)        # nominal widths (detector samples) = today's boxcar ladder


def fwhm_discrete(h):
    """FWHM (samples) of a non-negative unimodal sequence; half-max crossings by linear interpolation
    with zeros outside (a w-sample boxcar has FWHM w, a delta 1, the triangle [1..a..1] exactly a)."""
    h = np.concatenate([[0.0], np.asarray(h, float), [0.0]])
    hm = 0.5 * h.max()
    above = np.nonzero(h >= hm)[0]
    i0, i1 = above[0], above[-1]
    lo = (i0 - 1) + (hm - h[i0 - 1]) / (h[i0] - h[i0 - 1])
    hi = i1 + (h[i1] - hm) / (h[i1] - h[i1 + 1])
    return float(hi - lo)


def cascade(lengths):
    """Integer cascade of running sums = convolution of boxcars of the given lengths."""
    h = np.ones(1, dtype=np.int64)
    for a in lengths:
        h = np.convolve(h, np.ones(int(a), dtype=np.int64))
    return h


def bspline_lengths(n_stage, w):
    """Integer stage lengths of an n-stage B-spline whose FWHM is closest to w.

    B2 (triangle): equal lengths (w, w) -> FWHM exactly w (a trapezoid (a < w, w) also has FWHM w,
    so FWHM alone does not pin it; the true triangle is the equal-knot B-spline).
    B3 (quadratic): near-equal lengths (each a or a+1, the closest integer approximation of equal
    knots), searched around the continuous optimum a = w / 1.2679; ties broken by more active
    stages (smoother), then shorter support."""
    if n_stage == 2:
        return (int(w), int(w)), fwhm_discrete(cascade((w, w)))
    a0 = max(1, int(round(w / 1.2679)))
    best = None
    for a in range(max(1, a0 - 2), a0 + 3):
        for k in range(n_stage + 1):                   # k stages of length a+1, the rest a
            ls = tuple([a] * (n_stage - k) + [a + 1] * k)
            f = fwhm_discrete(cascade(ls))
            key = (round(abs(f - w), 9), -sum(1 for x in ls if x > 1), sum(ls))
            if best is None or key < best[0]:
                best = (key, ls, f)
    return tuple(int(a) for a in best[1]), best[2]


def make_bank(kind, widths=BANK_WIDTHS, gauss_trunc=3.0):
    """List of filter dicts: name, h (un-normalised), stage lengths, support, fwhm, ||h||_2, cost."""
    out = []
    if kind == "b3x":                               # 8 filters: boxcar 1, 2 + triangle [1,2,1] + B3 4..64
        out = make_bank("boxcar", (1, 2), gauss_trunc) + make_bank("b2", (2,), gauss_trunc) \
            + make_bank("b3", tuple(w for w in widths if w >= 4), gauss_trunc)
        for f in out:
            f["name"] = "b3x:" + f["name"]
        return out
    if kind in ("b2h", "b3h"):                      # hybrid: boxcar where the B-spline degenerates (w <= 2)
        for w in widths:
            f = make_bank("boxcar" if w <= 2 else kind[:2], (w,), gauss_trunc)[0]
            f["name"] = f"{kind}{w}"
            out.append(f)
        return out
    for w in widths:
        if kind == "boxcar":
            ls = (w,)
            h = cascade(ls).astype(float)
        elif kind in ("b2", "b3"):
            ls, _ = bspline_lengths(2 if kind == "b2" else 3, w)
            h = cascade(ls).astype(float)
        elif kind == "gauss":
            sig = w / 2.354820045
            half = int(np.ceil(gauss_trunc * sig))
            j = np.arange(-half, half + 1)
            h = np.exp(-0.5 * (j / sig) ** 2)
            ls = None
        else:
            raise ValueError(kind)
        if ls is not None:
            stages = sum(1 for a in ls if a > 1)
            adds = 2 * stages                       # S_t = S_{t-1} + x_t - x_{t-a} per stage
        else:
            stages = None
            adds = 2 * len(h)                       # FIR multiply-add per tap
        out.append(dict(name=f"{kind}{w}", w=int(w), h=h, lengths=ls, support=int(len(h)),
                        fwhm=fwhm_discrete(h), l2=float(np.sqrt(np.sum(h ** 2))),
                        stages=stages, flops_per_output=int(adds)))
    return out


BANKS = {k: make_bank(k) for k in ("boxcar", "b2", "b3", "gauss", "b2h", "b3h", "b3x")}


# -- extra pulse shapes -----------------------------------------------------------------
def _exp_ft(tau_us):
    """One-sided exponential (scattering tail) exp(-t/tau)/tau, tau = the 'width' argument."""
    return 1.0 / (1.0 + 2j * np.pi * _F * (tau_us / DT))


def ideal_norm_x(W_us, shape):
    if shape == "exp":
        return (1.0 / (2.0 * max(W_us, DT))) ** 0.5
    return ideal_norm(W_us, shape)


def kernel_x(W_us, shape):
    return {"gauss": _gauss_ft, "tophat": _tophat_ft, "exp": _exp_ft}[shape](W_us)


_FFT_CACHE = {}


def _bank_fft(bank_key, nfft):
    key = (bank_key, nfft)
    if key not in _FFT_CACHE:
        fl = BANKS[bank_key]
        arr = np.zeros((len(fl), nfft))
        for i, f in enumerate(fl):
            arr[i, :f["support"]] = (f["h"] / f["l2"])[::-1]      # correlation via convolution
        _FFT_CACHE[key] = (np.fft.rfft(arr, axis=1), np.array([f["support"] for f in fl]))
    return _FFT_CACHE[key]


def snr_banks(H, kern_ft, T_us, W_us, shape, bank_keys, per_filter=False):
    """Phase-averaged best-filter S/N / ideal per bank: filters ||h||_2-normalised (unit white-noise
    sigma), evaluated at every lag with full overlap, best filter per arrival phase, mean over phase
    (the snr_binned convention).  With per_filter, also each filter's phase-mean S/N / ideal."""
    prof = np.fft.irfft(H * kern_ft, n=NGRID)
    m = int(round(T_us / DT))
    cf = np.concatenate([[0.0], np.cumsum(prof)])
    nb = (NGRID - m) // m
    ph = (np.arange(N_PHASE) * m) // N_PHASE
    C = cf[ph[:, None] + m * np.arange(nb + 1)[None, :]]
    X = np.diff(C, axis=1)                                         # (P, nb) binned fluence
    nfft = 1 << int(np.ceil(np.log2(nb + 256)))
    Xf = np.fft.rfft(X, n=nfft, axis=1)
    out, pf = {}, {}
    norm = ideal_norm_x(W_us, shape) * np.sqrt(T_us)
    for bk in bank_keys:
        Hf, sup = _bank_fft(bk, nfft)
        Y = np.fft.irfft(Xf[:, None, :] * Hf[None, :, :], n=nfft, axis=2)   # (P, F, nfft)
        best_f = np.empty((N_PHASE, len(sup)))
        for k, L in enumerate(sup):
            best_f[:, k] = Y[:, k, L - 1:nb].max(axis=1)           # windows fully inside the series
        best_f /= norm
        out[bk] = float(best_f.max(axis=1).mean())
        if per_filter:
            pf[bk] = best_f.mean(axis=0)
    return (out, pf) if per_filter else out


def evaluate_banks(scheme, dm, widths_us, shapes=("gauss", "tophat", "exp"),
                   bank_keys=("boxcar", "b2", "b3", "gauss"), n_near=8):
    """Total chain (real two-stage integer shifts, best of the n_near nearest trials) S/N / ideal for
    every bank.  Returns (n_shape, n_bank, n_width)."""
    near = np.argsort(np.abs(scheme.trial_dm - dm))[:n_near]
    res = np.zeros((len(shapes), len(bank_keys), len(widths_us)))
    KS = {(s, w): kernel_x(w, s) for s in shapes for w in widths_us}
    for j in near:
        ti, fi, d = scheme.trials[j]
        Tt = scheme.tiers[ti]["q2"]
        Hi = histogram(A_TRUE * dm + scheme.applied_us(ti, fi), dm)
        for a, s in enumerate(shapes):
            for c, w in enumerate(widths_us):
                r = snr_banks(Hi, KS[(s, w)], Tt, w, s, bank_keys)
                for b, bk in enumerate(bank_keys):
                    res[a, b, c] = max(res[a, b, c], r[bk])
    return res
