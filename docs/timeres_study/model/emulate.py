"""Monte-Carlo emulator of incoherent dedispersion + boxcar search.

Models what reaches the detector for a burst of intrinsic Gaussian FWHM W at
true DM D, for a pipeline described by `Cfg`:

  * native channels (30.5 kHz) carry the unavoidable intra-channel smear;
  * channels are summed `csf` at a time BEFORE any shift (as corr_fast does);
  * shifts are integer multiples of a quantum: q1 for the per-channel
    (stage-1) shift, T for the per-stream / per-channel final shift;
  * one-stage: one shift per summed channel at the fine DM;
  * two-stage: stage-1 aligns each summed channel to its sub-band top at the
    COARSE DM (quantum q1), stage-2 aligns the sub-band top at the FINE DM
    (single merged rounding at T, as deployed with
    --merge-coarse-rounding-in-search-shifts);
  * the dedispersed sum is binned at T with a random phase and searched with
    power-of-two boxcars; the best of the nearest fine DM trials is kept.

Recovered fraction = boxcar S/N / ideal matched-filter S/N of the intrinsic
profile (no smear, continuous sampling).  Everything is linear, so channels
are histogrammed onto a fine grid and convolved once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics import (DNU_NATIVE_MHZ, FREQS_MHZ, K_DM_MS, N_CH,
                     N_CH_PER_CHGROUP, N_CHGROUP, T_NATIVE_US, F_TOP_MHZ,
                     chan_smear_us)

DT_US = T_NATIVE_US / 8.0          # 4.096 us fine grid; every T is a multiple
NGRID = 1 << 15                    # 134 ms window
WIDTHS = 2 ** np.arange(0, 9)      # boxcar widths in output samples


@dataclass
class Cfg:
    name: str
    T_us: float                    # output (search) sample
    csf: int = 8                   # native channels summed before shifting
    stages: int = 2
    n_sub: int = 4                 # sub-bands per chgroup (two-stage only)
    q1_us: float | None = None     # stage-1 shift quantum (None -> T)
    fine_dm: np.ndarray = field(default=None, repr=False)
    coarse_dm: np.ndarray = field(default=None, repr=False)
    fine_to_coarse: np.ndarray = field(default=None, repr=False)
    q2_us: float | None = None     # stage-2 (per-stream) shift quantum (None -> T)
    coherent: bool = False         # coherent dedispersion within channels: no intra-channel smear

    def __post_init__(self):
        if self.q1_us is None:
            self.q1_us = self.T_us
        if self.q2_us is None:
            self.q2_us = self.T_us


_nu = FREQS_MHZ / 1e3
_chgroup = np.arange(N_CH) // N_CH_PER_CHGROUP


def _summed_channel_freq(csf):
    """Centre frequency of the csf-summed channel each native channel is in."""
    grp = np.arange(N_CH) // csf
    centres = FREQS_MHZ.reshape(-1, csf).mean(axis=1)
    return centres[grp] / 1e3


def _subband_top(n_sub):
    per = N_CH_PER_CHGROUP // n_sub
    sb = np.arange(N_CH) // per
    tops = FREQS_MHZ[::per][sb] / 1e3          # first (highest) channel of the sub-band
    return tops


def applied_shift_ms(cfg: Cfg, dm_f: float, dm_c: float) -> np.ndarray:
    """Delay removed from each native channel (ms, relative to band top)."""
    nu_s = _summed_channel_freq(cfg.csf)
    T = cfg.q2_us * 1e-3
    if cfg.stages == 1:
        q = cfg.q1_us * 1e-3
        return q * np.rint(K_DM_MS * dm_f * (nu_s ** -2 - (F_TOP_MHZ / 1e3) ** -2) / q)
    q1 = cfg.q1_us * 1e-3
    nu_top_sb = _subband_top(cfg.n_sub)
    s1 = q1 * np.rint(K_DM_MS * dm_c * (nu_s ** -2 - nu_top_sb ** -2) / q1)
    s2 = T * np.rint(K_DM_MS * dm_f * (nu_top_sb ** -2 - (F_TOP_MHZ / 1e3) ** -2) / T)
    return s1 + s2


def _gauss_kernel_ft(W_us):
    sig = W_us / 2.354820045 / DT_US
    f = np.fft.rfftfreq(NGRID)
    return np.exp(-2.0 * (np.pi * f * sig) ** 2)


def recovered(cfg: Cfg, D: float, W_us: float, rng, n_trial_nbrs=3):
    """Best boxcar S/N fraction over the nearest fine trials, random phase."""
    nu = _nu
    true_ms = K_DM_MS * D * (nu ** -2 - (F_TOP_MHZ / 1e3) ** -2)
    smear = chan_smear_us(D, DNU_NATIVE_MHZ, FREQS_MHZ)        # us, per native ch
    if cfg.coherent:
        smear = np.zeros_like(smear)
    gft = _gauss_kernel_ft(W_us)
    # ideal: unit-fluence Gaussian per channel, matched filter, no smear
    sig_us = W_us / 2.354820045
    ideal_norm = (1.0 / (2.0 * np.sqrt(np.pi) * sig_us)) ** 0.5   # ||g||_2, per sqrt(us)
    T = cfg.T_us
    m = int(round(T / DT_US))
    phase = rng.integers(0, m)
    order = np.argsort(np.abs(cfg.fine_dm - D))[:n_trial_nbrs]
    best = 0.0
    for fi in order:
        dm_f = cfg.fine_dm[fi]
        dm_c = cfg.coarse_dm[cfg.fine_to_coarse[fi]] if cfg.stages == 2 else dm_f
        resid_us = (true_ms - applied_shift_ms(cfg, dm_f, dm_c)) * 1e3
        resid_us -= np.median(resid_us)
        # per-chgroup smear box (smear varies <4% inside a chgroup)
        prof = np.zeros(NGRID)
        pos = np.clip(resid_us / DT_US + NGRID / 2, 0, NGRID - 2)   # out-of-window -> edge (S/N ~0)
        for g in range(N_CHGROUP):
            sl = slice(g * N_CH_PER_CHGROUP, (g + 1) * N_CH_PER_CHGROUP)
            p = pos[sl]
            i0 = np.floor(p).astype(int)
            fr = p - i0
            h = np.bincount(i0, weights=1 - fr, minlength=NGRID + 1)[:NGRID]
            h += np.bincount(i0 + 1, weights=fr, minlength=NGRID + 1)[:NGRID]
            wbox = max(1, int(round(smear[sl].mean() / DT_US)))
            if wbox > 1:
                c = np.cumsum(np.concatenate([[0.0], h]))
                h = (c[wbox:] - c[:-wbox])
                h = np.concatenate([np.zeros(wbox // 2), h, np.zeros(wbox - 1 - wbox // 2)]) / wbox
            prof += h
        prof = np.fft.irfft(np.fft.rfft(prof) * gft, n=NGRID)
        prof /= N_CH                               # per-channel fluence units
        # bin at T with random phase (dt units)
        nb = (NGRID - phase) // m
        binned = prof[phase:phase + nb * m].reshape(nb, m).sum(axis=1)   # fluence fraction per bin
        c = np.concatenate([[0.0], np.cumsum(binned)])
        for w in WIDTHS:
            if w >= nb:
                break
            s = (c[w:] - c[:-w]).max() / np.sqrt(w * T)
            if s > best:
                best = s
    return best / ideal_norm


def mean_recovered(cfg, dms, W_us, n_draw=24, seed=0, jitter=0.03):
    rng = np.random.default_rng(seed)
    vals = []
    for d0 in np.atleast_1d(dms):
        for _ in range(n_draw):
            D = d0 * (1 + jitter * rng.uniform(-1, 1))
            vals.append(recovered(cfg, D, W_us, rng))
    return float(np.mean(vals)), float(np.percentile(vals, 10))
