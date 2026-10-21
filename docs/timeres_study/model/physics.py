"""Shared physics for the time-resolution study (numpy only; runs on h23).

Band / channel conventions follow REALTIME_FRB_SEARCH.md §15.18 and the
deployed DM-plan metadata: 16 chgroups x 384 native channels of
250/8192 MHz, channel i (0..6143) at 1498.75 - i*dnu MHz, native sample
32.768 us.
"""
from __future__ import annotations

import numpy as np

K_DM_MS = 4.148808          # ms GHz^2 / (pc cm^-3)
DNU_NATIVE_MHZ = 250.0 / 8192.0
T_NATIVE_US = 32.768
N_CHGROUP = 16
N_CH_PER_CHGROUP = 384
N_CH = N_CHGROUP * N_CH_PER_CHGROUP
F_TOP_MHZ = 1498.75
FREQS_MHZ = F_TOP_MHZ - np.arange(N_CH) * DNU_NATIVE_MHZ
F_BOT_MHZ = FREQS_MHZ[-1]                      # 1311.2805
BAND_COEF_MS = K_DM_MS * ((F_BOT_MHZ / 1e3) ** -2 - (F_TOP_MHZ / 1e3) ** -2)


def delay_ms(dm, f_mhz, f_ref_mhz):
    """Dispersion delay of f relative to f_ref (positive when f < f_ref)."""
    return K_DM_MS * dm * ((np.asarray(f_mhz) / 1e3) ** -2 - (f_ref_mhz / 1e3) ** -2)


def chan_smear_us(dm, dnu_mhz, f_mhz):
    """Intra-channel dispersion smear (full width) in microseconds."""
    return 8.297616e-3 * 1e3 * dm * dnu_mhz / (np.asarray(f_mhz) / 1e3) ** 3


def gen_dm_list(dm_min, dm_max, tsamp_us, dnu_mhz, nchans, f_ctr_mhz,
                ti_us=40.0, tol=1.25):
    """Barsdell `dedisp_generate_dm_list` recursion (== legacy gen_dmtrials).

    Spacing such that the DM-error smear adds in quadrature to the
    effective width (tsamp, intrinsic ti, intra-channel smear) by `tol`.
    """
    f = f_ctr_mhz * 1e-3
    tol2 = tol * tol
    a = 8.3 * dnu_mhz / f ** 3
    a2 = a * a
    b2 = a2 * (nchans * nchans / 16.0)
    c = (tsamp_us ** 2 + ti_us ** 2) * (tol2 - 1.0)
    out = [dm_min]
    while out[-1] < dm_max:
        prev = out[-1]
        prev2 = prev * prev
        k = c + tol2 * a2 * prev2
        out.append((b2 * prev + np.sqrt(-a2 * b2 * prev2 + (a2 + b2) * k)) / (a2 + b2))
    return np.array(out)


def n_dm_trials(dm_min, dm_max, tsamp_us, dnu_mhz, tol=1.25, ti_us=40.0):
    nch = int(round(187.469 / dnu_mhz))
    return len(gen_dm_list(dm_min, dm_max, tsamp_us, dnu_mhz, nch,
                           0.5 * (F_TOP_MHZ + F_BOT_MHZ), ti_us=ti_us, tol=tol))
