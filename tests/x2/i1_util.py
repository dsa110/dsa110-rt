"""x2 I1 helpers: the 2x chain end to end on the CPU (``tests/x2/test_i1_e2e.py``).

Per 2x format (``dsart.x2.opoint.FORMATS_2X``), production geometry (WS-P
fixture antennas, 16 chgroups x n_sub streams, N = 256, the frozen tiered
plan and value map)::

    point-source bursts with EXACT integer native delays (snr_model style)
      -> fast-visibility signal per (baseline, native channel), placed into
         the corr's values with the float64 reference's geometric placement
         (``corr2x_ref.value_dest_from_vm``) and the corr plan's stage-1
         shifts + x2 / x4 decimation  (+ value-domain Gaussian noise)
      -> BatchedStaticSky -> RowRulePacker (frozen row rule, 8-block warm-up)
      -> FramedTx2x -> frame-v2 datagrams -> RxRingV4Writer.ingest (in process)
      -> RingV4Source (WS-S binding of RxRingV4Reader, pread)
      -> OwnerEngine (CPU path) -> OwnerService + C1CyclePool -> C1 rows
      -> C2 test instance -> CutoutRequest -> CutoutProducer per half
         (ring snapshot + re-imaging) -> Level3 v2 products -> C3 -> plots.

Reductions (all statistical or cost-only; see the test module docstring):

* noise is added in the VALUE domain after stage 1 (complex Gaussian, var
  ``n_v * D`` per value per tier sample, ``D`` q1 samples summed): exactly
  the distribution of independent fast-visibility noise summed by stage 1 +
  decimation, without materialising 10 M (baseline, channel) samples per q1
  sample;
* the signal's stage 1 is computed separably (``B_c[v]`` per native channel
  x the pulse's per-channel time profile) -- cross-checked against
  ``corr2x_ref.stage1_values`` on dense visibilities in the test;
* each owner half searches only the fed coarse DM(s); halves owning none of
  them answer cutouts from :class:`NoiseRing` (one owned DM each);
* the WS-S CPU twins of K0 unpack, K1 and K3b are swapped for numerically
  equivalent vectorised versions (:func:`k1_fast`, :func:`k3b_fast`,
  :func:`unpack_frame2_rows_fast`) -- the twins are Python loops built for
  bit-exactness at N = 32 (K1 17 s / 64 rows, K3b 128 s / tier-A cycle at
  N = 256 on n02); the test checks each replacement against its twin.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SNR2X = REPO / "docs" / "timeres_study" / "snr2x"
MODEL = REPO / "docs" / "timeres_study" / "model"
FIXTURE = REPO / "tests" / "fixtures" / "x2" / "dsa110_antpos.npz"

N_GRID = 256
DEC_DEG = 71.63
BLOCK_NATIVE = 4096
Q1_NATIVE = 8
CYCLE_NATIVE = 6144
EXT_NATIVE = 16
NATIVE_S = 32.768e-6
MJD0 = 61341.0                      # test clock: native sample 0 = 2026-10-28T00:00 UTC
SHM_PREFIX = "/dsart-x2-i1-"
SEARCH_SIDS = (1, 2, 9, 13)         # SEARCH_NODES n01 n02 n09 n13
#: Phase 0 rowrule.md §8 "quant" column (mid-tread, heavy 3e-5 N, tau 3 s) by
#: bits_main. Dedispersion and imaging enter the prediction exactly, so only
#: the wire factor is applied on top (the "total vs model" column would count
#: them twice).
QUANT_FACTOR = {8: 0.9962, 10: 0.9986, 12: 0.9996}
#: today's v1 wire (per-row max-abs cint8), Phase 0 SUMMARY "contradictions" 2
QUANT_FACTOR_1X = 0.988


def sid_g_of_half(h: int) -> Tuple[int, int]:
    return SEARCH_SIDS[int(h) // 2], int(h) % 2


def mjd_of_native(n: float) -> float:
    return MJD0 + float(n) * NATIVE_S / 86400.0


# ---------------------------------------------------------------------------
# oracle + fixture
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def oracle():
    """``(schemes, snr_model)`` from docs/timeres_study/snr2x bound to THIS
    dsart (the tests/x2/test_plan_oracle.py recipe)."""
    import dsart

    added = [str(SNR2X), str(MODEL)]
    for p in added:
        sys.path.insert(0, p)
    try:
        spec = importlib.util.spec_from_file_location("x2_i1_oracle_schemes", SNR2X / "schemes.py")
        schemes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schemes)
    finally:
        for p in added:
            sys.path.remove(p)
    if sys.modules["dsart"] is not dsart:
        raise AssertionError("snr_model rebound dsart to another checkout")
    return schemes, sys.modules["snr_model"]


@lru_cache(maxsize=None)
def antpos():
    """WS-P fixture: cal-blob E/N (as stored) + the corr_setup_96 core mask."""
    from dsart.grid.sparsity_pattern import core_baseline_mask_from_station_numbers

    z = np.load(FIXTURE)
    return (np.asarray(z["antpos_e"]), np.asarray(z["antpos_n"]),
            core_baseline_mask_from_station_numbers(z["station"]))


def kept_rows(vm_kept_bl, nants: int) -> Tuple[np.ndarray, np.ndarray]:
    from dsart.grid.baseline_grid import kept_core_baselines

    keep = np.zeros(nants * (nants + 1) // 2, bool)
    keep[np.asarray(vm_kept_bl)] = True
    _, a, b = kept_core_baselines(keep, nants)
    return np.asarray(a, np.int64), np.asarray(b, np.int64)


# ---------------------------------------------------------------------------
# a format (op-point) and its static tables
# ---------------------------------------------------------------------------


class Format:
    """One op-point: plan, corr view, value map, row rule, owner inputs and
    the reference's geometric (baseline, summed channel) -> value placement
    of every stream (``corr2x_ref.value_dest_from_vm``, independent of the
    CSR the kernels use)."""

    def __init__(self, op) -> None:
        from dsart.grid.baseline_grid import build_value_map_for_opoint
        from dsart.x2 import quant
        from dsart.x2.corr.plan2x import CorrPlan2x
        from dsart.x2.plan import build_tiered_plan_for_opoint
        from dsart.x2.ref import corr2x_ref as cref
        from dsart.x2.search.valuemap import dm_inputs_from_value_map

        e, n, mask = antpos()
        self.op = op
        self.e, self.n = e, n
        self.plan = build_tiered_plan_for_opoint(op)
        self.cplan = CorrPlan2x.from_tiered_plan(self.plan)
        self.vm = build_value_map_for_opoint(op, e, n, is_core_baseline_mask=mask, dec_deg=DEC_DEG)
        self.rule = quant.RowRule.for_format(op.bits_main, op.n_sub)
        self.inp = dm_inputs_from_value_map(self.vm, s_main=float(self.rule.s_main_f32),
                                            s_heavy=float(self.rule.s_heavy_f32),
                                            bits_main=op.bits_main)
        self.n_sub, self.csf = int(op.n_sub), int(op.chan_sum_factor)
        self.nsum = int(self.vm.nchan_eff)
        self.per = self.nsum // self.n_sub
        self.S = int(self.vm.n_streams)
        self.ant_a, self.ant_b = kept_rows(self.vm.kept_bl, len(e))
        self.dest, self.conj = [], []
        for sid in range(self.S):
            d_, c_ = cref.value_dest_from_vm(self.vm, sid, e, n)
            st = self.vm.streams[sid]
            cnt = np.bincount(d_[d_ >= 0], minlength=int(st.n_values))
            if not np.array_equal(cnt, np.asarray(st.n_contrib)):
                raise AssertionError(f"stream {sid}: geometric placement != value map n_contrib")
            self.dest.append(np.asarray(d_, np.int64))
            self.conj.append(np.asarray(c_, bool))
        self.cell_rad = 1.0 / (N_GRID * float(self.vm.cell_lambda))
        self.n_src_native = int(self.vm.n_image) * self.csf

    @property
    def opoint_id(self) -> str:
        return self.op.opoint_id

    def nv(self, sid: int) -> int:
        return int(self.vm.streams[sid].n_values)

    def chgroup_layout(self, g: int):
        """``(sids, value_offsets, n_contrib, n_heavy)`` of chgroup ``g``."""
        sids = [g * self.n_sub + s for s in range(self.n_sub)]
        nvs = [self.nv(s) for s in sids]
        voff = np.concatenate([[0], np.cumsum(nvs)]).astype(np.int64)
        ncon = np.concatenate([np.asarray(self.vm.streams[s].n_contrib) for s in sids])
        nh = [int(self.vm.streams[s].n_heavy) for s in sids]
        return sids, voff, ncon, nh

    def tier(self, d: int):
        ti, _ = self.cplan.tier_of(int(d))
        return self.cplan.tiers[ti]

    def owner_half(self, d: int) -> int:
        return int(self.plan.owner_half[int(d)])

    def node_of(self, d: int) -> int:
        return sid_g_of_half(self.owner_half(d))[0]

    def nearest_fine(self, dm: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.plan.fine_dm) - float(dm))))

    def coarse_of_dm(self, dm: float) -> int:
        return int(self.plan.fine_to_coarse[self.nearest_fine(dm)])


# ---------------------------------------------------------------------------
# bursts: exact integer native delays, point source at a pixel centre
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Burst:
    name: str
    dm: float
    width: int           # native samples (top-hat)
    n_bot: int           # absolute native index of the pulse start at NU_BOT_PROC
    dcol: int            # image column offset from the centre = sky l / pixel (east)
    drow: int            # image row offset = sky m / pixel (north)
    snr: float           # target best-filter S/N (sets the amplitude)

    def lm_rad(self, cell_rad: float) -> Tuple[float, float]:
        return self.dcol * cell_rad, self.drow * cell_rad

    @property
    def pixel(self) -> Tuple[int, int]:
        """``(row, col)`` image pixel (Candidate ``(l, m)`` indices)."""
        return N_GRID // 2 + self.drow, N_GRID // 2 + self.dcol


def native_arrivals(burst: Burst) -> np.ndarray:
    """``(16, 384)`` int64 absolute native arrival per native channel:
    ``n_bot + rint(A_TRUE dm / T_nat)`` (snr_model's per-channel delay to
    NU_BOT_PROC, rounded to whole native samples -- no intra-channel smear)."""
    _, sm = oracle()
    return int(burst.n_bot) + np.rint(sm.A_TRUE * float(burst.dm) / sm.T_NAT).astype(np.int64)


def stream_source_sums(fmt: Format, sid: int, burst: Burst,
                       rows: Optional[np.ndarray] = None) -> np.ndarray:
    """``B[c, v]`` complex128 (the stream's native channels x values): the sum
    over the (baseline) sources of value ``v`` of the unit point-source
    visibility of native channel ``c``. ``V(a, b) = conj(E_b) E_a =
    exp(-2 pi i (u l + v m))`` with ``u = -dE / lambda``, ``v = -dN / lambda``
    (F18/F20, lambda of the NATIVE channel); per-baseline members the
    reference conjugates are conjugated."""
    from dsart.common.constants import SPEED_OF_LIGHT_M_S

    _, sm = oracle()
    g, s = divmod(int(sid), fmt.n_sub)
    c0 = s * fmt.per * fmt.csf
    chans = np.arange(c0, c0 + fmt.per * fmt.csf)
    kloc = chans // fmt.csf - s * fmt.per
    rsel = np.arange(fmt.dest[sid].shape[0]) if rows is None else np.asarray(rows, np.int64)
    dest = fmt.dest[sid][rsel][:, kloc].T                      # (n_c, R)
    conj = fmt.conj[sid][rsel][:, kloc].T
    nv = fmt.nv(sid)
    l_, m_ = burst.lm_rad(fmt.cell_rad)
    if l_ == 0.0 and m_ == 0.0:
        vis = np.ones(dest.shape, np.complex128)
    else:
        e = np.asarray(fmt.e, np.float64)
        n = np.asarray(fmt.n, np.float64)
        a, b = fmt.ant_a[rsel], fmt.ant_b[rsel]
        proj = (e[a] - e[b]) * l_ + (n[a] - n[b]) * m_                                # (R,) m
        nu = sm.NU[g, chans] * 1e9                                                   # Hz
        vis = np.exp(2j * np.pi * (nu[:, None] / SPEED_OF_LIGHT_M_S) * proj[None, :])
        vis = np.where(conj, np.conj(vis), vis)
    ok = dest >= 0
    idx = (np.arange(len(chans))[:, None] * nv + dest)[ok]
    re = np.bincount(idx, weights=vis.real[ok], minlength=len(chans) * nv)
    im = np.bincount(idx, weights=vis.imag[ok], minlength=len(chans) * nv)
    return (re + 1j * im).reshape(len(chans), nv)


def stage1_signal(fmt: Format, sid: int, d: int, burst: Burst, B: np.ndarray,
                  amp: float) -> Tuple[int, np.ndarray]:
    """The burst in stream ``sid``'s decimated values at coarse DM ``d``:
    ``(j0, X [n_j, V])`` complex128 for tier rows ``j0 ..``. Every native
    sample ``n`` of the pulse in native channel ``c`` (summed channel
    ``k``) is a fast-vis sample ``q' = n // 8``; stage 1 ADVANCES ``k`` by
    ``stage1_bins[g, k, d]`` (``out[q] = vis_k[q + s1]``) and the tier sums
    ``D = T / 8`` q1 samples aligned to the absolute clock: row
    ``j = (q' - s1) // D``. Equivalent to ``corr2x_ref.stage1_values`` +
    ``decimate`` on the dense visibilities (checked in the test)."""
    g, s = divmod(int(sid), fmt.n_sub)
    T = int(fmt.tier(d).T_native)
    D = T // Q1_NATIVE
    arr = native_arrivals(burst)[g]
    c0 = s * fmt.per * fmt.csf
    chans = np.arange(c0, c0 + fmt.per * fmt.csf)
    s1 = np.asarray(fmt.cplan.stage1_global[g, chans // fmt.csf, int(d)], np.int64)
    n = arr[chans][:, None] + np.arange(int(burst.width))[None, :]
    j = np.floor_divide(n // Q1_NATIVE - s1[:, None], D)
    j0 = int(j.min())
    W = np.zeros((int(j.max()) - j0 + 1, len(chans)))
    np.add.at(W, (j - j0, np.broadcast_to(np.arange(len(chans))[:, None], j.shape)), 1.0)
    return j0, float(amp) * (W @ B)


# ---------------------------------------------------------------------------
# prediction (snr_model style: exact integer shifts, actual arrival phase)
# ---------------------------------------------------------------------------


def applied_native(fmt: Format, f: int) -> np.ndarray:
    """``(16, 384)`` int64 applied delay of every native channel at global
    fine ``f`` in native samples, ``T s2 - 8 s1`` (snr_model
    ``Scheme.applied_us / T_nat``)."""
    d = int(fmt.plan.fine_to_coarse[int(f)])
    T = int(fmt.tier(d).T_native)
    k = np.arange(384) // fmt.csf
    sb = k // fmt.per
    s1 = np.asarray(fmt.cplan.stage1_global[:, :, d], np.int64)[:, k]          # (16, 384)
    s2 = np.asarray(fmt.plan.stage2_bins[int(f)], np.int64).reshape(16, fmt.n_sub)[:, sb]
    return T * s2 - Q1_NATIVE * s1


def out_rows(fmt: Format, burst: Burst, f: int) -> Tuple[np.ndarray, int]:
    """``(t [16, 384, w], T)``: the output tier row every native sample of the
    pulse lands in after stage 1 + decimation + stage 2 at fine ``f``
    (``t = floor((n + applied) / T)`` = ``(n // 8 - s1) // D + s2``)."""
    d = int(fmt.plan.fine_to_coarse[int(f)])
    T = int(fmt.tier(d).T_native)
    n = native_arrivals(burst)[:, :, None] + np.arange(int(burst.width))[None, None, :]
    return np.floor_divide(n + applied_native(fmt, f)[:, :, None], T), T


def imaging_response(fmt: Format, Bs: Sequence[np.ndarray], pixels: Sequence[Tuple[int, int]]
                     ) -> np.ndarray:
    """``rho [n_pix, 16, 384]``: the image (pixel, native channel) response of
    the unit source, ``Re(sum_v B_c[v] exp(+2 pi i (U'_v y' + W'_v x') / N))``
    with the value's OWNER cell ``(U', W')`` (today's imager, natural
    weighting: every source weighs 1) -- includes the exact gridding loss."""
    rho = np.zeros((len(pixels), 16, 384))
    py = np.array([p[0] - N_GRID // 2 for p in pixels], np.float64)
    px = np.array([p[1] - N_GRID // 2 for p in pixels], np.float64)
    for sid, B in enumerate(Bs):
        g, s = divmod(sid, fmt.n_sub)
        st = fmt.vm.streams[sid]
        U = np.asarray(st.cell_row, np.float64) - N_GRID // 2
        W = np.asarray(st.cell_col, np.float64) - N_GRID // 2
        E = np.exp(2j * np.pi * (U[:, None] * py[None, :] + W[:, None] * px[None, :]) / N_GRID)
        c0 = s * fmt.per * fmt.csf
        rho[:, g, c0:c0 + fmt.per * fmt.csf] = (B @ E).real.T
    return rho


def filter_bank(bank_name: str = "b3x"):
    from dsart.x2.search.filters import FilterBank

    return FilterBank.b3x() if bank_name == "b3x" else FilterBank.build((1, 2, 4, 8, 16, 32, 64),
                                                                        ("b",))


def profile_snr(prof: np.ndarray, t0: int, sigma_row: float, bank) -> dict:
    """Best (filter, centre) of a deterministic row profile ``prof`` (rows
    ``t0 ..``) under the engine's filter taps (``score = sum_j h[j]
    x[c - a + j]``, S/N = score / (sigma_row ||h||))."""
    best = dict(snr=-np.inf)
    pad = 160
    x = np.concatenate([np.zeros(pad), prof, np.zeros(pad)])
    for k, flt in enumerate(bank.filters):
        h = flt.taps()
        sc = np.convolve(x, h[::-1], mode="full")          # sc[i] = sum_j h[j] x[i - L + 1 + j]
        L = len(h)
        # centre c (x index) uses x[c - a + j]: window end c - a + L - 1
        c_idx = np.arange(len(x))
        e_idx = c_idx - flt.a + L - 1
        ok = (e_idx >= 0) & (e_idx < len(sc))
        snr = np.full(len(x), -np.inf)
        snr[ok] = sc[e_idx[ok]] / (sigma_row * float(np.sqrt(np.sum(h * h))))
        i = int(np.argmax(snr))
        if snr[i] > best["snr"]:
            best = dict(snr=float(snr[i]), filt=flt.name, k=k, t=int(t0 + i - pad))
    return best


def predict(fmt: Format, burst: Burst, rho: np.ndarray, pixels, amp: float, *,
            fines: Sequence[int], bank) -> dict:
    """Best predicted (fine, pixel, filter, centre) over ``fines`` x ``pixels``
    for amplitude ``amp`` (fast-vis noise sigma_V = 1: per native element
    sigma^2 = 1/16, per output row sigma_row^2 = N_src T / 32)."""
    best = dict(snr=-np.inf)
    for f in fines:
        t, T = out_rows(fmt, burst, f)
        sig_row = math.sqrt(fmt.n_src_native * T / 32.0)
        tmin = int(t.min())
        n_t = int(t.max()) - tmin + 1
        for ip, p in enumerate(pixels):
            w = np.broadcast_to(rho[ip][:, :, None], t.shape)
            prof = np.bincount((t - tmin).ravel(), weights=w.ravel(), minlength=n_t)
            r = profile_snr(amp * prof, tmin, sig_row, bank)
            if r["snr"] > best["snr"]:
                best = dict(r, f=int(f), pixel=tuple(p), T=T)
    return best


@lru_cache(maxsize=None)
def _scheme(n_sub: int):
    schemes, _ = oracle()
    return schemes.proposed(n_sub=int(n_sub))


def snr_model_reference(fmt: Format, burst: Burst, f: int) -> dict:
    """snr_model's own numbers for this burst at fine ``f``: the phase-averaged
    boxcar-ladder efficiency of ``histogram`` (exact integer arrivals, no
    smear box: ``dm = 0``) and the ideal S/N per unit amplitude. Also checks
    the plan's applied shifts against ``schemes.proposed(n_sub)``."""
    _, sm = oracle()
    sch = _scheme(fmt.n_sub)
    d = int(fmt.plan.fine_to_coarse[int(f)])
    ti = int(fmt.plan.coarse_tier[d])
    fi = int(f) - int(fmt.plan.tier_fine_offsets[ti])
    ap = applied_native(fmt, f)
    if np.max(np.abs(sch.applied_us(ti, fi) - ap * sm.T_NAT)) > 1e-6:
        raise AssertionError("plan applied shifts != snr_model Scheme.applied_us")
    arr = native_arrivals(burst) - int(burst.n_bot)
    resid_us = (arr + ap).astype(np.float64) * sm.T_NAT
    H = sm.histogram(resid_us, 0.0)
    W_us = int(burst.width) * sm.T_NAT
    T_us = float(fmt.tier(d).T_native) * sm.T_NAT
    eta = sm.snr_binned(H, sm._tophat_ft(W_us), T_us, W_us, "tophat", sm.LADDER)
    return dict(eta_phase_avg=float(eta), ideal_per_amp=math.sqrt(32.0 * fmt.n_src_native
                                                                  * int(burst.width)))


def phase_avg_ladder(fmt: Format, burst: Burst, f: int) -> float:
    """My predictor with rho = 1 and today's boxcar ladder, averaged over
    snr_model's 16 arrival phases -- must equal ``snr_model_reference``'s
    ``eta_phase_avg`` (ties the exact-phase predictor to the model)."""
    _, sm = oracle()
    d = int(fmt.plan.fine_to_coarse[int(f)])
    T = int(fmt.tier(d).T_native)
    m = T * 8                                               # profile grid DT = T_nat / 8
    phases = (np.arange(sm.N_PHASE) * m) // sm.N_PHASE // 8
    bank = filter_bank("box")
    vals = []
    ideal = math.sqrt(32.0 * fmt.n_src_native * int(burst.width))
    for ph in phases:
        b = Burst(burst.name, burst.dm, burst.width, int(burst.n_bot) - int(ph), 0, 0, 1.0)
        t, _ = out_rows(fmt, b, f)
        tmin = int(t.min())
        prof = np.bincount((t - tmin).ravel(), minlength=int(t.max()) - tmin + 1).astype(float)
        prof *= fmt.n_src_native / 6144.0                   # sources per native channel
        sig_row = math.sqrt(fmt.n_src_native * T / 32.0)
        vals.append(profile_snr(prof, tmin, sig_row, bank)["snr"] / ideal)
    return float(np.mean(vals))


def neighbourhood(pixel: Tuple[int, int]) -> List[Tuple[int, int]]:
    """The pixel and its 8 neighbours that the edge mask keeps (the image
    centre -- the phase centre -- is masked: ``compute_edge_mask(drop_dc)``)."""
    from dsart.image.imager import compute_edge_mask

    mask = compute_edge_mask(n_grid=N_GRID, kernel_support=1, dtype=np.float64)
    r0, c0 = pixel
    return [(r0 + dr, c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if mask[r0 + dr, c0 + dc] != 0]


@dataclass
class Prepared:
    burst: Burst
    amp: float
    pixels: list
    rho: np.ndarray
    pred_unit: dict                      # best prediction at amp = 1, no wire factor
    signals: dict                        # (d, sid) -> (j0, X)
    quant_factor: float
    true_pixel_ratio: float = 1.0        # best kept pixel / true pixel (masked or not)
    compression: float = 1.0             # bright response of the wire (1x: today's curve)

    @property
    def pred_snr(self) -> float:
        return float(self.pred_unit["snr"] * self.amp * self.quant_factor * self.compression)


def prepare_burst(fmt: Format, burst: Burst, dms: Sequence[int], *, bank, quant_factor: float,
                  n_near: int = 3) -> Prepared:
    """Amplitude for ``burst.snr`` (prediction x wire factor), the imaging
    response around the true pixel, and the stage-1 signal into every fed
    coarse DM; the per-stream ``B`` (up to ~250 MB) is dropped here."""
    Bs = [stream_source_sums(fmt, sid, burst) for sid in range(fmt.S)]
    pixels = neighbourhood(burst.pixel)
    rho = imaging_response(fmt, Bs, pixels)
    f0 = fmt.nearest_fine(burst.dm)
    d0 = int(fmt.plan.fine_to_coarse[f0])
    sl = fmt.plan.fines_of(d0)
    fines = [f for f in range(f0 - n_near, f0 + n_near + 1) if sl.start <= f < sl.stop]
    pred = predict(fmt, burst, rho, pixels, 1.0, fines=fines, bank=bank)
    # the true pixel even if the edge mask drops it (the phase centre): what the
    # masked centre costs an on-axis burst
    rho_t = imaging_response(fmt, Bs, [burst.pixel])
    pred_t = predict(fmt, burst, rho_t, [burst.pixel], 1.0, fines=fines, bank=bank)
    amp = float(burst.snr) / (pred["snr"] * float(quant_factor))
    signals = {(int(d), sid): stage1_signal(fmt, sid, d, burst, Bs[sid], amp)
               for d in dms for sid in range(fmt.S)}
    return Prepared(burst=burst, amp=amp, pixels=pixels, rho=rho, pred_unit=pred,
                    signals=signals, quant_factor=float(quant_factor),
                    true_pixel_ratio=float(pred["snr"] / pred_t["snr"]))


# ---------------------------------------------------------------------------
# fast, numerically equivalent CPU kernels (the twins are exactness tools)
# ---------------------------------------------------------------------------

_K1_MATS: dict = {}


def _k1_mats(csr, S: int, V: int):
    """``(P, Q)`` sparse float32 ``[n_act, S * V]``: direct + conjugate entries
    and direct - conjugate entries (``Re`` and ``Im`` of the Hermitian fold)."""
    import scipy.sparse as sp

    key = (id(csr), S, V)
    if key not in _K1_MATS:
        ne = np.diff(np.asarray(csr.cell_ptr, np.int64))
        cell = np.repeat(np.arange(csr.n_active), ne)
        col = np.asarray(csr.ent_stream, np.int64) * V + np.asarray(csr.ent_value, np.int64)
        sign = np.where(np.asarray(csr.ent_conj).astype(bool), -1.0, 1.0).astype(np.float32)
        shape = (csr.n_active, S * V)
        p = sp.csr_matrix((np.ones(cell.size, np.float32), (cell, col)), shape=shape)
        q = sp.csr_matrix((sign, (cell, col)), shape=shape)
        _K1_MATS[key] = (p, q, csr)
    return _K1_MATS[key][:2]


def k1_fast(codes, *, c_v, offset, csr, shifts, t_lo, n_t, w_v=None):
    """``combine.k1_twin`` as two sparse matrix products per fine DM: every
    stream's rows of ALL fines are decoded once (same fp32 ``((code + off)
    * c) * w``), each fine takes its shifted window, ``Re = P x_re``, ``Im =
    Q x_im``, times ``0.5 * sign``; only the fp32 summation order differs."""
    S, V, R, _ = codes.shape
    F = shifts.shape[0]
    P, Q = _k1_mats(csr, S, V)
    cv = np.asarray(c_v, np.float32)
    wv = np.ones_like(cv) if w_v is None else np.asarray(w_v, np.float32)
    off = np.float32(offset)
    kk = (np.float32(0.5) * csr.sign).astype(np.float32)
    sh = np.asarray(shifts, np.int64)
    smax, smin = sh.max(axis=0), sh.min(axis=0)
    n_u = int(n_t) + int((smax - smin).max())
    xr = np.empty((S, V, n_u), np.float32)
    xi = np.empty((S, V, n_u), np.float32)
    for s in range(S):
        slots = np.mod(int(t_lo) - int(smax[s]) + np.arange(n_u), R)
        lo = int(slots[0])
        c = codes[s, :, lo:lo + n_u] if lo + n_u <= R else codes[s][:, slots]
        xr[s] = ((c[..., 0].astype(np.float32) + off) * cv[s][:, None]) * wv[s][:, None]
        xi[s] = ((c[..., 1].astype(np.float32) + off) * cv[s][:, None]) * wv[s][:, None]
    bre = np.empty((S, V, int(n_t)), np.float32)
    bim = np.empty((S, V, int(n_t)), np.float32)
    out = np.empty((F, int(n_t), csr.n_active), np.complex64)
    for f in range(F):
        for s in range(S):
            o = int(smax[s] - sh[f, s])
            bre[s] = xr[s, :, o:o + int(n_t)]
            bim[s] = xi[s, :, o:o + int(n_t)]
        out[f].real = (P @ bre.reshape(S * V, -1)).T * kk[None, :]
        out[f].imag = (Q @ bim.reshape(S * V, -1)).T * kk[None, :]
    return out


def _box0(cs: np.ndarray, L: int) -> np.ndarray:
    """Trailing box sums of length ``L`` along axis 0 from the prefix sums
    ``cs`` of the series (zero before the start)."""
    out = cs.copy()
    out[L:] -= cs[:-L]
    return out


def k3b_fast(ring, *, ring_row0, c_first, n_c, sigma_inv, s_inv, bank, clip=250.0, chunk=64,
             tap_pixels=None, pix_block=128):
    """``filters.k3b_twin`` vectorised: the input ``clip(fl32(x16) *
    sigma_inv)`` exactly as the twin; every filter's box cascade as trailing
    prefix-sum differences in fp32 along time (axis 0, so each prefix step is
    one vector add over a cache-sized pixel block; the first stage of every
    filter shares one prefix sum); windows of centres ``>= c_first`` never
    reach rows before the filter's first needed row, so no per-filter zeroing
    is needed; ``snr = score * s_inv[k]``; chunk maxima filter by filter with
    strict ``>`` (ties: earlier filter, then earlier centre -- the twin's
    order). Differs from the twin by fp32 summation order only."""
    R, F, P = ring.shape
    if n_c % chunk:
        raise ValueError(f"n_c={n_c} not a multiple of chunk={chunk}")
    n_ch = n_c // chunk
    K = len(bank)
    si = np.asarray(sigma_inv, np.float32)[None, :, None]
    sk = np.asarray(s_inv, np.float32)
    clip32 = np.float32(clip)
    lo = int(c_first) - bank.max_a
    hi = int(c_first) + int(n_c) + bank.max_d
    slots = np.mod(np.arange(lo, hi) - int(ring_row0), R)
    best = np.full((n_ch, F, P), -np.inf, np.float32)
    tk = np.zeros((n_ch, F, P), np.int32)
    taps_out = None
    if tap_pixels is not None:
        tap_pixels = np.asarray(tap_pixels, np.int64)
        taps_out = np.zeros((K, F, tap_pixels.size, n_c), np.float32)
    c0 = int(c_first) - lo
    for p0 in range(0, P, pix_block):
        p1 = min(P, p0 + pix_block)
        pb = p1 - p0
        x = ring[slots, :, p0:p1].astype(np.float32)                    # (T, F, pb)
        x *= si
        np.minimum(np.maximum(x, -clip32, out=x), clip32, out=x)
        cs1 = np.cumsum(x, axis=0, dtype=np.float32)
        sel = None
        if taps_out is not None:
            sel = np.nonzero((tap_pixels >= p0) & (tap_pixels < p1))[0]
        bb = best[:, :, p0:p1]
        tt = tk[:, :, p0:p1]
        for k, flt in enumerate(bank.filters):
            if not flt.stages:
                y = x
            else:
                y = _box0(cs1, int(flt.stages[0]))
                for L in flt.stages[1:]:
                    y = _box0(np.cumsum(y, axis=0, dtype=np.float32), int(L))
            e0 = c0 + flt.d
            score = y[e0:e0 + int(n_c)]                                   # (n_c, F, pb)
            if sel is not None and sel.size:
                taps_out[k][:, sel, :] = np.transpose(score[:, :, tap_pixels[sel] - p0],
                                                      (1, 2, 0))
            v = (score * sk[k]).reshape(n_ch, chunk, F, pb)
            vm = v.max(axis=1)
            ia = np.argmax(v == vm[:, None], axis=1).astype(np.int32)
            better = vm > bb
            np.copyto(bb, vm, where=better)
            np.copyto(tt, (ia << 8) | np.int32(k), where=better)
    return best, tk, taps_out


def unpack_frame2_rows_fast(raw, fmt):
    """``codec.unpack_frame2_rows`` without the uint64 bit expansion (4-bit:
    nibbles; other widths: ``unpackbits`` -> ``packbits`` per component)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)[:, :fmt.row_bytes]
    rows = raw.shape[0]
    nh, nm, b = fmt.n_heavy, fmt.n_main, fmt.bits
    re = np.empty((rows, fmt.n_values), dtype=np.int8)
    im = np.empty((rows, fmt.n_values), dtype=np.int8)
    if nh:
        pair = raw[:, :2 * nh].view(np.int8).reshape(rows, nh, 2)
        re[:, :nh] = pair[..., 0]
        im[:, :nh] = pair[..., 1]
    if nm:
        if b == 4:
            m = raw[:, 2 * nh:2 * nh + nm]
            lo_ = (m & 0x0F).astype(np.int16)
            hi_ = (m >> 4).astype(np.int16)
            re[:, nh:] = np.where(lo_ >= 8, lo_ - 16, lo_)
            im[:, nh:] = np.where(hi_ >= 8, hi_ - 16, hi_)
        else:
            bits = np.unpackbits(raw[:, 2 * nh:], axis=1, bitorder="little")[:, :2 * nm * b]
            pad = np.zeros((rows, nm, 2, 8), np.uint8)
            pad[..., :b] = bits.reshape(rows, nm, 2, b)
            val = np.packbits(pad, axis=-1, bitorder="little")[..., 0].astype(np.int16)
            val = np.where(val >= (1 << (b - 1)), val - (1 << b), val)
            re[:, nh:] = val[..., 0]
            im[:, nh:] = val[..., 1]
    return re, im


def install_fast_kernels(monkeypatch) -> None:
    """Swap the WS-S CPU twins used by the engine (K0 unpack, K1, K3b) for the
    equivalents above for the duration of a test."""
    import dsart.x2.search.combine as comb
    import dsart.x2.search.engine as eng

    monkeypatch.setattr(eng, "k1_twin", k1_fast)
    monkeypatch.setattr(eng, "k3b_twin", k3b_fast)
    monkeypatch.setattr(comb, "unpack_frame2_rows", unpack_frame2_rows_fast)


# ---------------------------------------------------------------------------
# corr leg: values -> static sky -> row rule -> frame v2 -> ring v4
# ---------------------------------------------------------------------------


class _RingSink:
    """``FramedTx2x`` sink that ingests every datagram into a ring v4 writer
    (no socket)."""

    def __init__(self, writer, rr) -> None:
        self.w, self.rr = writer, rr
        self.n_ok = 0
        self.bad: Dict[int, int] = {}

    def submit(self, dgs, **_kw) -> int:
        buf, offs, lens = dgs
        ok = 0
        for o, n in zip(np.asarray(offs).tolist(), np.asarray(lens).tolist()):
            rc = self.w.ingest(buf[o:o + n].tobytes())
            if rc == self.rr.RX4_OK:
                ok += 1
            else:
                self.bad[rc] = self.bad.get(rc, 0) + 1
        self.n_ok += ok
        return ok

    def counters(self) -> dict:
        return {"submitted": self.n_ok, "accepted": self.n_ok, "unrouted": 0, "per_dest": []}

    def close(self) -> None:
        pass


class CorrLeg:
    """The corr side of one format for a set of coarse DMs, block by block.

    ``first_block[d]``: the first corr block of DM ``d``'s flows (the row
    rule then spends 8 blocks in warm-up = PRESENT rows of NO_DATA codes).
    One ring v4 per search node, holding that node's fed DMs (as
    ``search_rx`` would)."""

    def __init__(self, fmt: Format, first_block: Dict[int, int], last_block: int,
                 prepared: Sequence["Prepared"], *, seed: int) -> None:
        import torch

        from dsart.transport import recv_ring as rr
        from dsart.x2.corr.framer2x import StreamFormat
        from dsart.x2.corr.packer2x import RowRulePacker
        from dsart.x2.corr.static_sky2x import BatchedStaticSky
        from dsart.x2.corr.tx2x import FramedTx2x

        torch.set_num_threads(1)
        rr._rx4_lib()
        self.rr = rr
        self.fmt = fmt
        self.first = {int(d): int(b) for d, b in first_block.items()}
        self.seed = int(seed)
        self.next_block = {d: b for d, b in self.first.items()}
        # ---- rings, one per search node
        by_node: Dict[int, List[int]] = {}
        for d in sorted(self.first):
            by_node.setdefault(fmt.node_of(d), []).append(d)
        self.ring_name: Dict[int, str] = {}
        self.writers: Dict[int, object] = {}
        self.sinks: Dict[int, _RingSink] = {}
        nvs = tuple(fmt.nv(s) for s in range(fmt.S))
        nhs = tuple(int(fmt.vm.streams[s].n_heavy) for s in range(fmt.S))
        tag = uuid.uuid4().hex[:8]
        for node, ds in by_node.items():
            tbuf = tuple((last_block + 4 - self.first[d]) * (BLOCK_NATIVE // int(fmt.tier(d).T_native))
                         for d in ds)
            spec = rr.RingV4Spec(
                dms=tuple(ds), t_int_factor=tuple(int(fmt.tier(d).T_native) for d in ds),
                t_buf=tbuf, n_stream=fmt.S, n_sub=fmt.n_sub, bits_main=fmt.rule.bits_main,
                bits_heavy=16, value_kind=0 if fmt.vm.kind == "gridded" else 1, n_grid=N_GRID,
                s_main=float(fmt.rule.s_main_f32), s_heavy=float(fmt.rule.s_heavy_f32),
                stream_n_values=nvs, stream_n_heavy=nhs, min_retention_s=0.0)
            name = f"{SHM_PREFIX}{tag}-n{node:02d}"
            w = rr.RxRingV4Writer.create(name, spec)
            for sid in range(fmt.S):
                w.set_expected_pattern_id(sid, int(fmt.vm.stream_pattern_ids[sid]))
            self.ring_name[node] = name
            self.writers[node] = w
            self.sinks[node] = _RingSink(w, rr)
        # ---- per (d, chgroup): static sky, packer, framer
        self.state: Dict[Tuple[int, int], tuple] = {}
        for d in self.first:
            sink = self.sinks[fmt.node_of(d)]
            for g in range(16):
                sids, voff, ncon, nh = fmt.chgroup_layout(g)
                sky = BatchedStaticSky(n_d=1, n_values=int(voff[-1]), window_blocks=8,
                                       warmup_cubes=8)
                pk = RowRulePacker(n_d=1, value_offsets=voff, n_contrib=ncon, n_heavy=nh,
                                   rule=fmt.rule)
                tx = FramedTx2x(sink, [StreamFormat.from_value_map(fmt.vm, s, fmt.rule)
                                       for s in sids])
                std = np.sqrt(ncon.astype(np.float64) * (int(fmt.tier(d).T_native) // Q1_NATIVE)
                              / 2.0).astype(np.float32)
                self.state[(d, g)] = (sids, voff, sky, pk, tx, std)
        # ---- the bursts' exact stage-1 signal per (d, stream)
        self.signal: Dict[Tuple[int, int], list] = {}
        for pb in prepared:
            for key, sig in pb.signals.items():
                self.signal.setdefault(key, []).append(sig)
        self.packets = 0
        self.guarded_rows = 0
        self.tm = dict(values_s=0.0, sky_s=0.0, pack_s=0.0, tx_s=0.0)

    def push_until(self, block_end: int) -> None:
        """Run every DM's flows through blocks ``< block_end``."""
        import torch

        for d in sorted(self.first):
            T = int(self.fmt.tier(d).T_native)
            tier = self.fmt.tier(d).name
            rows = BLOCK_NATIVE // T
            while self.next_block[d] < block_end:
                b = self.next_block[d]
                j_lo = b * rows
                for g in range(16):
                    sids, voff, sky, pk, tx, std = self.state[(d, g)]
                    c0 = time.perf_counter()
                    rng = np.random.default_rng([self.seed, d, g, b])
                    z = rng.standard_normal((rows, int(voff[-1]), 2), dtype=np.float32)
                    x = (z[..., 0] + 1j * z[..., 1]).astype(np.complex64) * std[None, :]
                    for s, sid in enumerate(sids):
                        for j0, X in self.signal.get((d, sid), ()):
                            a = max(j0, j_lo)
                            e = min(j0 + X.shape[0], j_lo + rows)
                            if e > a:
                                x[a - j_lo:e - j_lo, voff[s]:voff[s + 1]] += \
                                    X[a - j0:e - j0].astype(np.complex64)
                    xt = torch.from_numpy(x[None])
                    c1 = time.perf_counter()
                    sky.apply_(xt)
                    c2 = time.perf_counter()
                    res = pk.pack(xt)
                    c3 = time.perf_counter()
                    self.guarded_rows += sum(int(r["guarded"].sum()) for r in res)
                    self.packets += tx.transmit_tier(
                        block_n=b, tier=tier, dms=np.array([d]), t0_native=b * BLOCK_NATIVE,
                        T_native=T, stream_ids=sids, packets=res, rfi_warming_up=False)
                    c4 = time.perf_counter()
                    for key, dt in (("values_s", c1 - c0), ("sky_s", c2 - c1), ("pack_s", c3 - c2),
                                    ("tx_s", c4 - c3)):
                        self.tm[key] += dt
                self.next_block[d] = b + 1

    def bad_ingests(self) -> dict:
        out = {}
        for s in self.sinks.values():
            for k, v in s.bad.items():
                out[k] = out.get(k, 0) + v
        return out

    def close(self) -> None:
        for w in self.writers.values():
            try:
                w.close()
            finally:
                w.unlink()
        self.writers.clear()


# ---------------------------------------------------------------------------
# search leg
# ---------------------------------------------------------------------------


def engine_config(**kw):
    """CPU owner engine at the 2x defaults (b3x, Phase 0 amplitude regime),
    except Layer-2 burn-in 0: one processed cycle per DM in I1, and C2 drops
    NOISE_WARMUP rows; s_k then starts at the white-noise seeds ||h|| (what
    the EMA converges to on white noise)."""
    from dsart.x2.search.engine import EngineConfig

    base = dict(backend="cpu", fft_mode="pruned", layer2_n_burnin=0, wait_timeout_s=0.0)
    base.update(kw)
    return EngineConfig(**base)


def owner_plan(fmt: Format, half: int, dms: Sequence[int]):
    """The half's owner plan (``owner_plan_from_tiered``) restricted to ``dms``."""
    from dsart.x2.search.geometry import OwnerPlan, owner_plan_from_tiered

    full = owner_plan_from_tiered(fmt.plan, half=int(half))
    return OwnerPlan(half=full.half, dms=tuple(full.by_d(d) for d in sorted(dms)),
                     n_stream=full.n_stream, ext_native=full.ext_native, opoint=full.opoint,
                     plan_id=full.plan_id, extra=dict(full.extra))


class HalfPlanView:
    """The TieredDmPlan as seen by an I1 half whose engine owns only some of
    its coarse DMs (``fines_of_half`` / ``dms_of_half`` restricted; every
    other attribute is the plan's) -- what ``fine_axis_for_half`` reads."""

    def __init__(self, plan, dms_by_half: Dict[int, Sequence[int]]) -> None:
        self._plan = plan
        self._dms = {int(h): sorted(int(d) for d in ds) for h, ds in dms_by_half.items()}

    def dms_of_half(self, h: int) -> np.ndarray:
        return np.asarray(self._dms[int(h)], np.int64)

    def fines_of_half(self, h: int) -> np.ndarray:
        out = []
        for d in self._dms[int(h)]:
            sl = self._plan.fines_of(d)
            out.append(np.arange(sl.start, sl.stop))
        return np.concatenate(out).astype(np.int64)

    def __getattr__(self, k):
        return getattr(self._plan, k)


class NoiseRing:
    """Frame-v2 rows of pure noise (unit per-component sigma_hat, quantised
    with the frozen mid-tread steps), served for ANY owned DM: the stand-in
    ring of the halves that own none of the burst DMs (they must still
    answer every CutoutRequest for Level3 products to be complete)."""

    def __init__(self, fmt: Format, dms: Sequence[int], row0: int, n_rows: int, seed: int) -> None:
        import torch

        from dsart.x2.corr.packer2x import pack_bits_torch

        self._dms = sorted(int(d) for d in dms)
        self._fmt = fmt.inp.fmt
        self.row0, self.n = int(row0), int(n_rows)
        rb = self._fmt.row_bytes
        self.raw = np.zeros((fmt.S, self.n, rb), np.uint8)
        rng = np.random.default_rng(seed)
        lo, hi = fmt.rule.main_range
        sm, sh = float(fmt.rule.s_main_f32), float(fmt.rule.s_heavy_f32)
        cb = fmt.rule.comp_bits
        for sid in range(fmt.S):
            nv, nh = fmt.nv(sid), int(fmt.vm.streams[sid].n_heavy)
            z = rng.standard_normal((self.n, nv, 2), dtype=np.float32)
            heavy = np.clip(np.rint(z[:, :nh] / sh), -127, 127).astype(np.int8)
            main = np.clip(np.rint(z[:, nh:] / sm), lo, hi).astype(np.int8)
            self.raw[sid, :, :2 * nh] = heavy.reshape(self.n, 2 * nh).view(np.uint8)
            pb = pack_bits_torch(torch.from_numpy(np.ascontiguousarray(main.reshape(self.n, -1))),
                                 cb).numpy()
            self.raw[sid, :, 2 * nh:2 * nh + pb.shape[1]] = pb

    def dms(self) -> list:
        return list(self._dms)

    def row_format(self, d: int):
        return self._fmt

    def write_pos(self, d: int) -> int:
        return self.row0 + self.n

    def wait_window(self, d, t_first, n_rows, timeout_s=None):
        out = np.zeros((self.raw.shape[0], int(n_rows), self.raw.shape[2]), np.uint8)
        valid = np.zeros((self.raw.shape[0], int(n_rows)), bool)
        a = max(int(t_first), self.row0)
        e = min(int(t_first) + int(n_rows), self.row0 + self.n)
        if e > a:
            out[:, a - t_first:e - t_first] = self.raw[:, a - self.row0:e - self.row0]
            valid[:, a - t_first:e - t_first] = True
        return out, valid

    def counters(self) -> dict:
        return dict(row0=self.row0, n_rows=self.n)


def c1_header(*, k: int, half: int, n_rows: int, n_fdm: int):
    """C1 batch header of cycle ``k`` (2x external clock: ``sample_period_us``
    524.288, ``sample_period_specnum`` 8, event_specnum_start = the cycle's
    first external row). WS-S's ``OwnerService.emit`` hands rows only."""
    from dsart.coinc import wire

    sid, g = sid_g_of_half(half)
    e0 = int(k) * (CYCLE_NATIVE // EXT_NATIVE)
    return wire.build_header(cube_id=int(k), event_specnum_start=e0,
                             mjd_start=mjd_of_native(e0 * EXT_NATIVE), sample_period_specnum=8,
                             sample_period_us=EXT_NATIVE * NATIVE_S * 1e6, n_grid=N_GRID,
                             n_fdm_in_cube=int(n_fdm), search_node_id=sid, gpu_half=g,
                             n_candidates=int(n_rows))


# ---------------------------------------------------------------------------
# 1x-compat leg: today's op-point (csf 8, n_sub 4, 1.048576 ms, v2 DM plan,
# corr stage 1 + stage-2 FIFO, v1 per-row max-abs cint8 wire) -> the same
# owner engine in 1x-compat mode
# ---------------------------------------------------------------------------


class Format1x:
    """Today's op-point: ``OPPOINT_1X`` value map (== today's sub-band
    patterns), the deployed v2 plan (snr_model ``schemes.current()``), its
    corr stage-1 (``Scheme.stage1_bins``, validated bit-for-bit against
    ``DMPlan.delay_bins_per_subband`` by snr2x/validate.py) and stage-2 FIFO
    (``compute_stage2_shifts``)."""

    T_NATIVE = 32

    def __init__(self) -> None:
        from dsart.coarse_dm.stage2_shifts import compute_stage2_shifts
        from dsart.grid.baseline_grid import build_value_map_for_opoint
        from dsart.x2.opoint import OPPOINT_1X
        from dsart.x2.ref import corr2x_ref as cref
        from dsart.x2.search.valuemap import dm_inputs_from_value_map

        schemes, sm = oracle()
        e, n, mask = antpos()
        self.op = OPPOINT_1X
        self.e, self.n = e, n
        self.vm = build_value_map_for_opoint(OPPOINT_1X, e, n, is_core_baseline_mask=mask,
                                             dec_deg=DEC_DEG)
        self.inp = dm_inputs_from_value_map(self.vm)
        self.n_sub, self.csf = 4, 8
        self.nsum = int(self.vm.nchan_eff)
        self.per = self.nsum // self.n_sub
        self.S = int(self.vm.n_streams)
        self.ant_a, self.ant_b = kept_rows(self.vm.kept_bl, len(e))
        self.dest, self.conj = [], []
        for sid in range(self.S):
            d_, c_ = cref.value_dest_from_vm(self.vm, sid, e, n)
            self.dest.append(np.asarray(d_, np.int64))
            self.conj.append(np.asarray(c_, bool))
        self.cell_rad = 1.0 / (N_GRID * float(self.vm.cell_lambda))
        self.n_src_native = int(self.vm.n_image) * self.csf
        self.scheme = schemes.current()
        t = self.scheme.tiers[0]
        self.T_us = float(t["q2"])
        self.coarse = np.asarray(t["coarse"], float)
        self.fine = np.asarray(t["fine"], float)
        self.f2c = np.asarray(t["f2c"], np.int64)
        self.s1 = np.stack([self.scheme.stage1_bins(0, c) for c in range(self.coarse.size)],
                           axis=-1).astype(np.int64)                       # (16, 48, C)
        self.s2c = np.stack([np.asarray(compute_stage2_shifts(
            chgroup=g, coarse_dm_pc_cm3=self.coarse, t_int_corr_us=self.T_us).shifts_samples)
            for g in range(16)]).astype(np.int64)                          # (16, C)

    def nv(self, sid: int) -> int:
        return int(self.vm.streams[sid].n_values)

    def coarse_of_dm(self, dm: float) -> int:
        return int(self.f2c[int(np.argmin(np.abs(self.fine - float(dm))))])

    def owner_plan(self, c: int, half: int):
        from dsart.x2.search.geometry import owner_plan_1x

        return owner_plan_1x(coarse_dm=self.coarse, fine_dm=self.fine, fine_to_coarse=self.f2c,
                             owned=[int(c)], half=int(half), n_sub=4, csf=8, t_native=32)

    def applied_native(self, c: int, f_loc: int, spec) -> np.ndarray:
        """``(16, 384)`` native delay at local fine ``f_loc`` of coarse ``c``:
        ``32 (s2_fifo + s_fine - s1)``; checks that the FIFO + the search's
        signed shift is snr_model's merged rounding."""
        k = np.arange(384) // self.csf
        sb = k // self.per
        sf = np.asarray(spec.shifts[f_loc], np.int64).reshape(16, 4)
        f_glob = int(np.nonzero(self.f2c == c)[0][f_loc])
        merged = np.asarray(self.scheme.stage2_bins(0, f_glob), np.int64)
        if not np.array_equal(self.s2c[:, c][:, None] + sf, merged):
            raise AssertionError("1x FIFO + search shift != merged stage-2 rounding")
        return self.T_NATIVE * ((self.s2c[:, c][:, None] + sf)[:, sb] - self.s1[:, k, c])


def stage1_signal_1x(f1: Format1x, sid: int, c: int, burst: Burst, B: np.ndarray,
                     amp: float) -> Tuple[int, np.ndarray]:
    """1x stream rows of the burst: fast-vis sample ``n // 32``, stage 1
    advances summed channel ``k`` by ``s1`` samples, the FIFO delays the
    chgroup by ``s2``: ``r = n // 32 - s1 + s2``."""
    g, s = divmod(int(sid), f1.n_sub)
    arr = native_arrivals(burst)[g]
    c0 = s * f1.per * f1.csf
    chans = np.arange(c0, c0 + f1.per * f1.csf)
    s1 = f1.s1[g, chans // f1.csf, int(c)]
    n = arr[chans][:, None] + np.arange(int(burst.width))[None, :]
    r = n // f1.T_NATIVE - s1[:, None] + int(f1.s2c[g, int(c)])
    r0 = int(r.min())
    W = np.zeros((int(r.max()) - r0 + 1, len(chans)))
    np.add.at(W, (r - r0, np.broadcast_to(np.arange(len(chans))[:, None], r.shape)), 1.0)
    return r0, float(amp) * (W @ B)


@lru_cache(maxsize=None)
def today_bright_curve():
    """Phase 0 study 2 (``docs/x2/phase0/results/rowrule_bright.json``, MC
    ``today|random``): today's per-row max-abs wire response L(rho) vs the
    per-sample S/N rho (1.000 at low rho; the 1.2% row-weight jitter is the
    separate ``QUANT_FACTOR_1X``)."""
    d = json.loads((REPO / "docs" / "x2" / "phase0" / "results" / "rowrule_bright.json")
                   .read_text())["mc"]["today|random"]
    pairs = sorted((float(k), float(v)) for k, v in d.items())
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def today_compression(rho: np.ndarray) -> np.ndarray:
    r, L = today_bright_curve()
    x = np.clip(np.abs(np.asarray(rho, float)), r[0], r[-1])
    return np.interp(np.log(x), np.log(r), L)


def predict_1x(f1: Format1x, burst: Burst, rho, pixels, amp: float, *, c: int, spec, f_locs,
               bank, compress: bool = False) -> dict:
    """As :func:`predict` for the 1x shifts; ``compress`` scales every row of
    the deterministic profile by today's bright response at its own S/N."""
    best = dict(snr=-np.inf)
    T = f1.T_NATIVE
    sig_row = math.sqrt(f1.n_src_native * T / 32.0)
    n = native_arrivals(burst)[:, :, None] + np.arange(int(burst.width))[None, None, :]
    for fl in f_locs:
        t = np.floor_divide(n + f1.applied_native(c, fl, spec)[:, :, None], T)
        tmin = int(t.min())
        for ip, p in enumerate(pixels):
            w = np.broadcast_to(rho[ip][:, :, None], t.shape)
            prof = amp * np.bincount((t - tmin).ravel(), weights=w.ravel(),
                                     minlength=int(t.max()) - tmin + 1)
            if compress:
                prof = prof * today_compression(prof / sig_row)
            r = profile_snr(prof, tmin, sig_row, bank)
            if r["snr"] > best["snr"]:
                best = dict(r, f=int(fl), pixel=tuple(p), T=T)
    return best


def prepare_burst_1x(f1: Format1x, burst: Burst, c: int, spec, *, bank, n_near: int = 3
                     ) -> Prepared:
    Bs = [stream_source_sums(f1, sid, burst) for sid in range(f1.S)]
    pixels = neighbourhood(burst.pixel)
    rho = imaging_response(f1, Bs, pixels)
    f0 = int(np.argmin(np.abs(spec.fine_dm - float(burst.dm))))
    f_locs = [f for f in range(f0 - n_near, f0 + n_near + 1) if 0 <= f < spec.n_fdm]
    pred = predict_1x(f1, burst, rho, pixels, 1.0, c=c, spec=spec, f_locs=f_locs, bank=bank)
    rho_t = imaging_response(f1, Bs, [burst.pixel])
    pred_t = predict_1x(f1, burst, rho_t, [burst.pixel], 1.0, c=c, spec=spec, f_locs=f_locs,
                        bank=bank)
    amp = float(burst.snr) / (pred["snr"] * QUANT_FACTOR_1X)
    signals = {(int(c), sid): stage1_signal_1x(f1, sid, c, burst, Bs[sid], amp)
               for sid in range(f1.S)}
    pred_c = predict_1x(f1, burst, rho, pixels, amp, c=c, spec=spec, f_locs=f_locs, bank=bank,
                        compress=True)
    return Prepared(burst=burst, amp=amp, pixels=pixels, rho=rho, pred_unit=pred,
                    signals=signals, quant_factor=QUANT_FACTOR_1X,
                    true_pixel_ratio=float(pred["snr"] / pred_t["snr"]),
                    compression=float(pred_c["snr"] / (pred["snr"] * amp)))


def corr_rows_1x(f1: Format1x, c: int, prepared: Sequence[Prepared], *, row_lo: int,
                 row_hi: int, warmup_blocks: int = 8, seed: int = 0) -> np.ndarray:
    """1x stream rows ``[row_lo, row_hi)`` of coarse ``c`` as today's RX ring
    holds them: value-domain noise (var ``16 n`` per value = 32 natives x 8
    channels of the same native noise as the 2x leg) + the burst, the
    corr's static sky (8-block window, warm-up 8: ``warmup_blocks`` earlier
    blocks are run first), then today's v1 rule per row
    (``corr2x_ref.quantise_v1_rows``: per-row max-abs cint8, scale dropped).
    Returns ``uint8 [S, n, row_bytes]``."""
    import torch

    from dsart.x2.corr.static_sky2x import BatchedStaticSky
    from dsart.x2.ref.corr2x_ref import quantise_v1_rows

    rows_blk = BLOCK_NATIVE // f1.T_NATIVE
    b_first = row_lo // rows_blk - int(warmup_blocks)
    b_last = -(-row_hi // rows_blk)
    rb = int(f1.inp.fmt.row_bytes)
    out = np.zeros((f1.S, row_hi - row_lo, rb), np.uint8)
    sig: Dict[int, list] = {}
    for pb in prepared:
        for (cc, sid), v in pb.signals.items():
            if cc == int(c):
                sig.setdefault(sid, []).append(v)
    for g in range(16):
        sids = [g * 4 + s for s in range(4)]
        nvs = [f1.nv(s) for s in sids]
        voff = np.concatenate([[0], np.cumsum(nvs)]).astype(np.int64)
        ncon = np.concatenate([np.asarray(f1.vm.streams[s].n_contrib) for s in sids])
        std = np.sqrt(8.0 * ncon).astype(np.float32)
        sky = BatchedStaticSky(n_d=1, n_values=int(voff[-1]), window_blocks=8, warmup_cubes=8)
        for b in range(b_first, b_last):
            j_lo = b * rows_blk
            z = np.random.default_rng([seed, int(c), g, b]).standard_normal(
                (rows_blk, int(voff[-1]), 2), dtype=np.float32)
            x = (z[..., 0] + 1j * z[..., 1]).astype(np.complex64) * std[None, :]
            for s, sid in enumerate(sids):
                for j0, X in sig.get(sid, []):
                    a = max(j0, j_lo)
                    e = min(j0 + X.shape[0], j_lo + rows_blk)
                    if e > a:
                        x[a - j_lo:e - j_lo, voff[s]:voff[s + 1]] += X[a - j0:e - j0].astype(
                            np.complex64)
            xt = torch.from_numpy(x[None])
            sky.apply_(xt)
            x = xt[0].numpy()
            a = max(j_lo, row_lo)
            e = min(j_lo + rows_blk, row_hi)
            if e <= a:
                continue
            for s, sid in enumerate(sids):
                q, _scales = quantise_v1_rows(x[a - j_lo:e - j_lo, voff[s]:voff[s + 1]])
                out[sid, a - row_lo:e - row_lo, :2 * nvs[s]] = q.reshape(e - a, -1).view(np.uint8)
    return out


# ---------------------------------------------------------------------------
# the I1 chain driver (tests/x2/test_i1_e2e.py, tests/x2/test_i1_e2e_baselines.py)
# ---------------------------------------------------------------------------

REPORT = Path(os.environ.get("DSART_X2_I1_REPORT", "/tmp/dsart-x2-i1/report.json"))
FORMATS = ("2x-gridded-ns2-cint4", "2x-baselines-ns4-cint5")
SNR_2X = 100.0
SNR_1X = 30.0
TOL_SNR = 0.05
B0 = 1000                      # first corr block of the earliest flow
MARGIN_NATIVE = 2 * BLOCK_NATIVE


@lru_cache(maxsize=None)
def get_format(opoint_id: str) -> Format:
    from dsart.x2.opoint import FORMATS_2X

    return Format(FORMATS_2X[opoint_id])


def write_report(key: str, value) -> None:
    import json

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    doc[key] = value
    REPORT.write_text(json.dumps(doc, indent=1, default=float))


def _timeline(fmt: Format, dms, history: int):
    """Burst cycle k, first block per DM and the block that makes cycle k the
    newest complete one on every fed DM."""
    need = {}
    for d in dms:
        spec = owner_plan(fmt, fmt.owner_half(d), [d]).dms[0]
        need[d] = (history + spec.max_pos) * spec.t_native
    k = math.ceil(((B0 + 8) * BLOCK_NATIVE + max(need.values()) + MARGIN_NATIVE)
                  / CYCLE_NATIVE)
    first = {d: (k * CYCLE_NATIVE - need[d] - MARGIN_NATIVE) // BLOCK_NATIVE - 8 for d in dms}
    end = math.ceil(1.5 * (k + 1))
    return k, first, end


def _bursts(k: int, snr: float, *, tier_b: bool = True):
    a0, b0 = k * 384, k * 192
    out = [
        Burst("dm300_w2_on", 300.0, 2, (a0 - 40) * 16 + 5, 0, 0, snr),
        Burst("dm300_w2_off", 300.0, 2, (a0 + 50) * 16 + 11, 37, -23, snr),
        Burst("dm300_w16_off", 300.0, 16, (a0 + 140) * 16 + 3, -52, 41, snr),
        Burst("dm300_w16_on", 300.0, 16, (a0 + 230) * 16 + 9, 0, 0, snr),
    ]
    if tier_b:
        out += [
            Burst("dm2000_w2_on", 2000.0, 2, (b0 - 60) * 32 + 7, 0, 0, snr),
            Burst("dm2000_w2_off", 2000.0, 2, (b0 - 20) * 32 + 21, 61, 18, snr),
            Burst("dm2000_w16_off", 2000.0, 16, (b0 + 30) * 32 + 13, -29, -47, snr),
            Burst("dm2000_w16_on", 2000.0, 16, (b0 + 90) * 32 + 29, 0, 0, snr),
        ]
    return out


def _match(pb: Prepared, cands, d: int):
    """The candidate of burst ``pb``: on its coarse DM, within one pixel of the
    true pixel and 3 rows of the predicted centre; brightest wins."""
    r0, c0 = pb.burst.pixel
    t_pred = pb.pred_unit["t"]
    hit = [c for c in cands if c.d == d and abs(c.l - r0) <= 1 and abs(c.m - c0) <= 1
           and abs(c.t_tier - t_pred) <= 3]
    return max(hit, key=lambda c: c.snr) if hit else None


def _check_burst(fmt, pb: Prepared, cand, row, *, ext_native: int, ext_factor: int,
                 fine_ok) -> dict:
    """The report row of one detected burst (assertions in ``_assert_rec``).
    Time: centre of the detecting filter's window (``event_specnum`` is the
    filter centre ``c``; the window spans rows ``[c - L//2, c - L//2 + L)``)
    vs the pulse centre at nu_bot_proc; position from the C1 row's (l_rad,
    m_rad)."""
    b = pb.burst
    l_true, m_true = b.lm_rad(fmt.cell_rad)
    # the filter's window [c - a, c - a + L) rows is centred at c - a + L / 2:
    # c + 1/2 for odd L (b1, t2, q*), c itself for even L (b2, today's b2..b64)
    L = int(cand.support_ext) // int(ext_factor)
    t_rep = cand.event_specnum + (0.5 * ext_factor if L % 2 else 0.0)
    t_true = (b.n_bot + 0.5 * b.width) / ext_native
    pred = pb.pred_snr
    return dict(
        burst=b.name, dm_true=b.dm, width_native=b.width, pixel_true=list(b.pixel),
        detected_half=int(cand.half), d=int(cand.d), tier=cand.tier,
        dm_det=float(cand.dm), fine_det=int(cand.fine_global), fine_ok=sorted(fine_ok),
        t_reported_ext=float(t_rep), t_true_ext=float(t_true), dt_ext=float(t_rep - t_true),
        pixel_det=[int(cand.l), int(cand.m)],
        l_rad=float(row.l_rad), m_rad=float(row.m_rad), l_true=l_true, m_true=m_true,
        dl_px=float((row.l_rad - l_true) / fmt.cell_rad),
        dm_px=float((row.m_rad - m_true) / fmt.cell_rad),
        snr=float(cand.snr), snr_c1=float(row.snr), snr_pred=pred,
        snr_ratio=float(cand.snr / pred), filt=cand.filt, filt_pred=pb.pred_unit["filt"],
        fine_pred=int(pb.pred_unit["f"]), pixel_pred=list(pb.pred_unit["pixel"]),
        t_pred_row=int(pb.pred_unit["t"]), t_det_row=int(cand.t_tier), amp=pb.amp,
        best_over_true_pixel=pb.true_pixel_ratio, wire_compression=pb.compression)


def _assert_rec(rec: dict, *, owner_half: int, cell_tol: float = 1.0 + 1e-6) -> list:
    bad = []
    if rec["detected_half"] != owner_half:
        bad.append(f"half {rec['detected_half']} != owner {owner_half}")
    if rec["fine_det"] not in rec["fine_ok"]:
        bad.append(f"fine {rec['fine_det']} not within one trial of {rec['fine_ok']}")
    if abs(rec["dt_ext"]) > 1.0:
        bad.append(f"time off by {rec['dt_ext']:.2f} T_ext")
    if abs(rec["dl_px"]) > cell_tol or abs(rec["dm_px"]) > cell_tol:
        bad.append(f"position off by ({rec['dl_px']:.2f}, {rec['dm_px']:.2f}) px")
    if abs(rec["snr_ratio"] - 1.0) > rec.get("snr_tol", TOL_SNR):
        bad.append(f"S/N {rec['snr']:.2f} vs predicted {rec['snr_pred']:.2f}")
    return bad


class _Store:
    def get_dict(self, key):
        return {"enabled": True, "ts": 0.0} if key.endswith("dumps_enabled") else None

    def put_dict(self, key, value):
        pass


class _Bc:
    """C2's broadcaster: records the 128-B CutoutRequest blobs it sends."""

    def __init__(self):
        self.sent = []

    def broadcast(self, **kw):
        raise AssertionError("2x must not send the 64-byte cube trigger")

    def send_to(self, blob, halves):
        self.sent.append((bytes(blob), [tuple(h) for h in halves]))
        return {tuple(h): True for h in halves}

    def close(self):
        pass


def _c2(fmt: Format, root: Path, monkeypatch):
    import dsart.services.coincidencer as c2mod
    from dsart.coinc.cutout_lifecycle import CutoutConfig
    from dsart.services.coincidencer import CoincidencerConfig, CoincidencerService

    pp = root / "plan.npz"
    fmt.plan.to_npz(pp)
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1", bind_port=0, window_s=5.0,
        csv_dir_c1=root / "c1", csv_dir_c2=root / "c2", event_archive_root=root / "events",
        trigger_criteria_path=REPO / "configs" / "opoints" / "2x" / "c2_trigger_criteria.yaml",
        name_allocator_offline=True, gal_dm_poll_interval_s=60.0, gal_dm_max_los_override=40.0,
        startup_grace_s=0.0, dump_broadcast_hosts={s: f"10.0.0.{s}" for s in SEARCH_SIDS},
        archive_schema=2, opoint=fmt.opoint_id,
        cutout=CutoutConfig(enabled=True, plan_path=pp, rows=64, raw_pre_ms=20.0,
                            raw_post_ms=20.0, deadline_s=1),
        partial_event_policy="archive", plotter_use_process_pool=False)
    rendered = []
    monkeypatch.setattr(c2mod, "enqueue_event", lambda w, name, root_, **kw: rendered.append(name))
    bc = _Bc()
    return CoincidencerService(config=cfg, mon_store=_Store(), broadcaster=bc), cfg, bc, rendered


def run_2x_chain(opoint_id: str, tmp_path, monkeypatch) -> None:
    """The 2x chain of one format, every assertion of the I1 test (see the
    test module docstring)."""
    from dsart.coinc import plotter as PL
    from dsart.coinc import wire
    from dsart.coinc.cal_hdf5_archive import CalHdf5Config
    from dsart.services import c3 as c3mod
    from dsart.x2.cutout import schema as CS
    from dsart.x2.cutout.raw_archive import is_raw_complete, read_raw_dir, stream_window
    from dsart.x2.ref.search2x_ref import reimage_from_raw
    from dsart.x2.search.c1_cycle import C1CycleConfig, C1CyclePool
    from dsart.x2.search.cutout_service import ACCEPTED, DONE, CutoutProducer
    from dsart.x2.search.engine import OwnerEngine
    from dsart.x2.search.ring import RingV4Source
    from dsart.x2.search.service import OwnerService, mon_payloads

    install_fast_kernels(monkeypatch)
    t_start = time.perf_counter()
    tm = {}
    fmt = get_format(opoint_id)
    bank = filter_bank("b3x")
    dA, dB = fmt.coarse_of_dm(300.0), fmt.coarse_of_dm(2000.0)
    dms = [dA, dB]
    k, first, end = _timeline(fmt, dms, bank.history())
    bursts = _bursts(k, SNR_2X)
    qf = QUANT_FACTOR[fmt.op.bits_main]
    prepared = [prepare_burst(fmt, b, dms, bank=bank, quant_factor=qf) for b in bursts]
    tm["prepare_s"] = time.perf_counter() - t_start
    write_report(f"{opoint_id}/timings", tm)

    # ---- corr -> frame v2 -> ring v4
    t0 = time.perf_counter()
    corr = CorrLeg(fmt, first, end + 2, prepared, seed=11)
    sources = []
    try:
        corr.push_until(end)
        tm["corr_s"] = time.perf_counter() - t0
        tm["corr_parts_s"] = dict(corr.tm)
        write_report(f"{opoint_id}/timings", tm)
        assert corr.bad_ingests() == {}, corr.bad_ingests()
        assert corr.packets > 0

        # ---- owner halves: ring v4 reader -> engine -> C1
        by_half = {}
        for d in dms:
            by_half.setdefault(fmt.owner_half(d), []).append(d)
        geom = dict(n_grid=N_GRID, cell_l_rad=fmt.cell_rad, cell_m_rad=fmt.cell_rad,
                    l0_rad=0.0, m0_rad=0.0)
        engines, out = {}, {}
        t0 = time.perf_counter()
        for h, hd in sorted(by_half.items()):
            sid, g = sid_g_of_half(h)
            oplan = owner_plan(fmt, h, hd)
            src = RingV4Source.open(corr.ring_name[sid], owned=hd, access="pread")
            sources.append(src)
            inputs = {d: fmt.inp for d in hd}
            assert src.validate(oplan, inputs, vmap=fmt.vm) == []
            eng = OwnerEngine(oplan, inputs, src, engine_config())
            assert all(s["ok"] for s in eng.status()["dms"].values())
            emitted = []
            svc = OwnerService(eng, search_node_id=sid, gpu_half=g,
                               c1_pool=C1CyclePool(C1CycleConfig(chan_sum_factor=fmt.csf,
                                                                 ext_period_us=524.288)),
                               emit=lambda kk, kept, rows, _e=emitted: _e.append((kk, kept, rows)),
                               geom=geom)
            kk, res, kept = svc.step(timeout_s=0.0)
            assert kk == k, (kk, k)
            mon = mon_payloads(svc, opoint=opoint_id)
            assert mon[f"/mon/search_rt/{sid}/compute/{g}"]["pattern_build_ok"] == 1
            engines[h] = eng
            out[h] = dict(res=res, kept=kept, rows=emitted[0][2], n_fdm=sum(s.n_fdm for s in
                                                                            oplan.dms))
        tm["search_s"] = time.perf_counter() - t0
        tm["engine_ms"] = {str(d): r.timings_ms for h in out for d, r in out[h]["res"].per_dm.items()}
        write_report(f"{opoint_id}/timings", tm)

        # ---- physics per burst
        recs, problems = [], []
        for pb in prepared:
            d_true = fmt.coarse_of_dm(pb.burst.dm)
            h = fmt.owner_half(d_true)
            kept, rows = out[h]["kept"], out[h]["rows"]
            cand = _match(pb, kept, d_true)
            row = rows[kept.index(cand)] if cand is not None else None
            if cand is None:                                  # diagnose: pre-C1 list
                pre = _match(pb, out[h]["res"].candidates, d_true)
                problems.append(f"{pb.burst.name}: not in C1 rows (engine: {pre})")
                continue
            f0 = fmt.nearest_fine(pb.burst.dm)
            rec = _check_burst(fmt, pb, cand, row, ext_native=EXT_NATIVE,
                               ext_factor=int(fmt.tier(d_true).T_native) // EXT_NATIVE,
                               fine_ok={f0 - 1, f0, f0 + 1})
            rec["search_node_id"], rec["gpu_half"] = sid_g_of_half(h)
            ref = snr_model_reference(fmt, pb.burst, int(cand.fine_global))
            rec["snr_model_eta_phase_avg"] = ref["eta_phase_avg"]
            rec["snr_ideal"] = ref["ideal_per_amp"] * pb.amp
            recs.append(rec)
            problems += [f"{pb.burst.name}: {p}" for p in _assert_rec(rec, owner_half=h)]
        write_report(f"{opoint_id}/bursts", recs)
        assert not problems, problems

        # ---- C2: every owner batch of cycle k
        root = tmp_path
        c2, cfg, bc, rendered = _c2(fmt, root, monkeypatch)
        # gridded: tier A first, per-baseline: tier B first, so both tiers' events go
        # through cutouts / C3 / plots across the two formats (C2 holds off the class
        # for 30 s after the first dump and does not evaluate a batch's clusters in
        # row order). A half owning both DMs (per-baseline n_sub 4: d 2 and 18 on
        # half 2) sends its cycle as two batches, preferred tier first.
        pref = "A" if fmt.vm.kind == "gridded" else "B"
        batches = []
        for h in sorted(out):
            for tier in (pref, "B" if pref == "A" else "A"):
                rows = tuple(r for c, r in zip(out[h]["kept"], out[h]["rows"]) if c.tier == tier)
                if rows:
                    batches.append((tier != pref, h, rows))
        for _later, h, rows in sorted(batches, key=lambda b: b[0]):
            hdr = c1_header(k=k, half=h, n_rows=len(rows), n_fdm=out[h]["n_fdm"])
            asyncio.run(c2._on_batch(wire.C1Batch(header=hdr, candidates=rows), peer_repr="i1"))
        assert bc.sent, "C2 sent no CutoutRequest"
        blob, halves = bc.sent[0]
        req = wire.decode_cutout_request(blob)
        assert sorted(halves) == sorted(sid_g_of_half(h) for h in range(8))
        assert req.plan_id == fmt.plan.plan_id
        trig = [pb for pb in prepared
                if fmt.coarse_of_dm(pb.burst.dm) == int(req.coarse_global_peak)
                and abs(pb.burst.pixel[0] - req.l_pix) <= 1
                and abs(pb.burst.pixel[1] - req.m_pix) <= 1
                and abs(pb.pred_unit["t"] * fmt.tier(req.coarse_global_peak).T_native
                        // EXT_NATIVE - int(req.t_peak_ext)) <= 8]
        assert trig, "the request is not centred on an injected burst"
        assert fmt.tier(int(req.coarse_global_peak)).name == pref, req
        name = req.event_name
        ev = cfg.event_archive_root / name

        # ---- time passes: the corr fills the raw window after the trigger
        corr.push_until(end + 2)

        # ---- every half answers: owner halves from ring v4, the rest from a noise ring
        dms_by_half = {h: list(v) for h, v in by_half.items()}
        noise_dm = {h: h for h in range(8) if h not in by_half}
        dms_by_half.update({h: [d] for h, d in noise_dm.items()})
        view = HalfPlanView(fmt.plan, dms_by_half)
        lo, hi = None, None
        for h, d in noise_dm.items():
            spec = owner_plan(fmt, h, [d]).dms[0]
            s_lo, n_s = stream_window(req.t_peak_ext, req.raw_pre_ext, req.raw_post_ext,
                                      ext_factor=1, max_pos=spec.max_pos, max_neg=spec.max_neg)
            a, z = min(s_lo, req.t0_ext - spec.max_pos), max(s_lo + n_s, req.t0_ext + req.n_t + 1)
            lo, hi = (a, z) if lo is None else (min(lo, a), max(hi, z))
        nring = NoiseRing(fmt, list(noise_dm.values()), lo - 4, hi - lo + 8, seed=5)
        for h, d in noise_dm.items():
            engines[h] = OwnerEngine(owner_plan(fmt, h, [d]), {d: fmt.inp}, nring,
                                     engine_config())
        t0 = time.perf_counter()
        states = {}
        for h in range(8):
            sid, g = sid_g_of_half(h)
            prod = CutoutProducer(engines[h], search_node_id=sid, gpu_half=g, out_root=root,
                                  plan=view, plan_id=fmt.plan.plan_id, geometry=geom,
                                  clock_mjd=lambda t0e: mjd_of_native(t0e * EXT_NATIVE))
            got = prod.offer(blob)
            assert got is not None and got[1] == ACCEPTED, got
            st, why, info = prod.serve(prod.next_request(), event_dir=ev)
            states[h] = (st, why, info)
        tm["cutouts_s"] = time.perf_counter() - t0
        write_report(f"{opoint_id}/timings", tm)
        assert all(s[0] == DONE for s in states.values()), states

        # ---- C2 finalises Level3 v2: complete, nothing deleted
        c2._scan_pending_events(time.monotonic())
        l3 = json.loads((ev / "Level3" / f"{name}.json").read_text())
        prod_blk = l3["products"]
        assert prod_blk["status"] == "complete", prod_blk
        assert len(prod_blk["present"]) == 8 and not prod_blk["missing"]
        assert l3["c2"]["sample_period_us"] == pytest.approx(524.288)
        assert rendered == [name]
        for h in range(8):
            sid, g = sid_g_of_half(h)
            assert (ev / "cutouts" / CS.cutout_filename(sid, g)).is_file()
            assert is_raw_complete(ev / "raw" / f"raw_s{sid}_g{g}")

        # ---- the owner's cutout shows the burst; the raw archive re-images to it
        h_t = int(req.coarse_global_peak) % 8
        sid_t, g_t = sid_g_of_half(h_t)
        cut = CS.read_cutout(ev / "cutouts" / CS.cutout_filename(sid_t, g_t))
        fi = cut.fine_index_of_global(int(req.fine_global_peak))
        assert fi is not None
        r_pk = int(req.t_peak_ext - req.t0_ext)
        lc = np.asarray(cut["pix_lc"])[:, fi]
        sig = np.asarray(cut["sigma_applied"])[:, fi]
        assert abs(int(np.argmax(lc)) - r_pk) <= 2 and float(lc.max()) > 20.0, lc
        slabs = read_raw_dir(ev / "raw" / f"raw_s{sid_t}_g{g_t}")
        d_t = int(req.coarse_global_peak)
        spec_t = owner_plan(fmt, h_t, [d_t]).dms[0]
        f_loc = int(np.nonzero(spec_t.fine_global == int(req.fine_global_peak))[0][0])
        ext_f = spec_t.t_native // EXT_NATIVE
        img = reimage_from_raw(slabs[d_t], f_loc, int(req.t0_ext), int(req.n_t), ext_factor=ext_f,
                               value_map=fmt.vm, sigma=sig)
        ref_lc = img[:, req.l_pix, req.m_pix]
        corr_lc = float(np.corrcoef(ref_lc, lc)[0, 1])
        peak_ratio = float(lc.max() / ref_lc.max())
        assert corr_lc > 0.99 and abs(peak_ratio - 1.0) < 0.02, (corr_lc, peak_ratio)

        # ---- C3 on the cutouts, then the plots
        c3 = c3mod.C3Service(c3mod.C3Config(
            archive_root=cfg.event_archive_root, rejected_root=root / "rejected",
            state_path=root / "c3_state.json", fired_injection_log=None, corr_nodes={},
            flag_only=True, cal_hdf5=CalHdf5Config(enabled=False)), mon_store=_Store())
        rec3 = c3.process_event(name)
        assert rec3["metrics"]["ok"] and rec3["metrics"]["source"] == "cutouts", rec3["metrics"]
        assert rec3["action"] == "KEEP", rec3
        paths = PL.render_event_plots(PL.PlotJob(event_name=name,
                                                 archive_root=cfg.event_archive_root))
        assert len(paths) == 4 and all(p.is_file() and p.stat().st_size > 1000 for p in paths)
        assert ev.is_dir() and (ev / "Level3" / f"{name}.json").is_file()      # never deleted
        tm["total_s"] = time.perf_counter() - t_start
        write_report(f"{opoint_id}/event", dict(
            name=name, trigger=trig[0].burst.name, request=dict(
                t_peak_ext=int(req.t_peak_ext), fine=int(req.fine_global_peak),
                coarse=int(req.coarse_global_peak), l_pix=int(req.l_pix), m_pix=int(req.m_pix),
                snr=float(req.snr_peak)),
            products=prod_blk, c3=dict(action=rec3["action"], rules=rec3["rules_fired"],
                                       tz_trig=rec3["metrics"].get("tz_trig")),
            raw_reimage=dict(corr=corr_lc, peak_ratio=peak_ratio),
            plots=[p.name for p in paths], timings_s=tm, cycle=k))
    finally:
        for s in sources:
            s.close()
        corr.close()


# ---------------------------------------------------------------------------
# 1x-compat leg
# ---------------------------------------------------------------------------


def run_1x_leg(monkeypatch) -> None:
    """The same DM-300 bursts through today's op-point: deployed v2 plan, corr
    stage 1 + stage-2 FIFO at 1.048576 ms / csf 8 / n_sub 4, today's v1
    per-row max-abs cint8 wire, the owner engine in 1x-compat mode (today's
    boxcars, amplitude regime, lagged Layer-1, local fine index) and today's
    C1 chain. 1x still detects every burst at the right DM / time / pixel. The
    S/N prediction carries today's wire: the 1.2% row-weight jitter
    (``QUANT_FACTOR_1X``) and the per-row max-abs bright response at each
    row's own S/N (Phase 0 study 2 curve, ~0.98-0.99 at the 20-30 sigma per
    sample of these bursts). Target 30 sigma, tolerance +-10% = 3 sigma of the
    measurement noise at 30 sigma (the 2x legs run at 100 sigma for +-5%)."""
    from dsart.x2.search.c1_cycle import C1CycleConfig, C1CyclePool
    from dsart.x2.search.engine import EngineConfig, OwnerEngine
    from dsart.x2.search.ring import MemoryRing
    from dsart.x2.search.service import OwnerService

    install_fast_kernels(monkeypatch)
    t0 = time.perf_counter()
    f1 = Format1x()
    bank = filter_bank("box")
    c = f1.coarse_of_dm(300.0)
    h = c                                                   # 1x: coarse DM c on half c
    sid, g = sid_g_of_half(h)
    oplan = f1.owner_plan(c, h)
    spec = oplan.dms[0]
    cfg = EngineConfig.compat_1x(backend="cpu", fft_mode="pruned", layer2_n_burnin=0,
                                 wait_timeout_s=0.0)
    k = 700
    bursts = _bursts(k, SNR_1X, tier_b=False)
    prepared = [prepare_burst_1x(f1, b, c, spec, bank=bank) for b in bursts]
    history = 32 + bank.max_a                               # compat_1x delay 32
    row_lo = k * 192 - history - spec.max_pos - 8
    row_hi = (k + 1) * 192 + spec.max_neg
    raw = corr_rows_1x(f1, c, prepared, row_lo=row_lo, row_hi=row_hi, seed=13)
    ring = MemoryRing({c: f1.inp.fmt}, f1.S, t_buf=row_hi - row_lo + 64)
    ring.push(c, row_lo, raw)
    eng = OwnerEngine(oplan, {c: f1.inp}, ring, cfg)
    emitted = []
    geom = dict(n_grid=N_GRID, cell_l_rad=f1.cell_rad, cell_m_rad=f1.cell_rad, l0_rad=0.0,
                m0_rad=0.0)
    svc = OwnerService(eng, search_node_id=sid, gpu_half=g,
                       c1_pool=C1CyclePool(C1CycleConfig.today_1x()),
                       emit=lambda kk, kept, rows: emitted.append(rows), geom=geom)
    kk, res, kept = svc.step(timeout_s=0.0)
    assert kk == k
    rows = emitted[0]
    recs, problems = [], []
    for pb in prepared:
        cand = _match(pb, kept, c)
        if cand is None:
            problems.append(f"{pb.burst.name}: not in C1 rows "
                            f"(engine: {_match(pb, res.candidates, c)})")
            continue
        f0 = int(np.argmin(np.abs(spec.fine_dm - pb.burst.dm)))
        rec = _check_burst(f1, pb, cand, rows[kept.index(cand)], ext_native=32, ext_factor=1,
                           fine_ok={f0 - 1, f0, f0 + 1})
        rec["snr_tol"] = 0.10
        rec["search_node_id"], rec["gpu_half"] = sid, g
        recs.append(rec)
        problems += [f"{pb.burst.name}: {p}" for p in _assert_rec(rec, owner_half=h)]
    write_report("1x/bursts", recs)
    write_report("1x/timings", dict(total_s=time.perf_counter() - t0, cycle=k))
    assert not problems, problems
