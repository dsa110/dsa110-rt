"""Tiered DM plan v3 for the 2x search (docs/x2/SPEC.md §2).

A :class:`TieredDmPlan` is the 2x replacement of the v2 ``DmPlan``: several
DM tiers, each searched at its own stream sample, with a variable number of
fine DMs per coarse DM and every integer shift table the corr and the search
apply. ``DmPlan`` v2 (``dsart.common.contracts``) is not touched and keeps
loading; :func:`load_any_plan` dispatches on the metadata version.

Oracle
======
The tables reproduce ``docs/timeres_study/snr2x/schemes.py::proposed(q1=262.144,
n_sub=...)`` bit for bit (``tests/x2/test_plan_oracle.py``):

* fine grid per tier: ``gen_dm_list`` (Barsdell / legacy ``gen_dmtrials``
  recursion; a verbatim port of ``docs/timeres_study/model/physics.py``) with
  tsamp = the tier sample, dnu = csf native channels, nchans = 6144 / csf,
  f_ctr = the processed-band centre (:data:`F_CTR_MHZ`), ti 40 us, tol 1.25;
  values above the tier ceiling dropped;
* coarse DMs per tier: ``n_c = max(1, ceil(density * dDM * (T_ref / T) / n_sub))``
  uniform-width buckets over [dm_lo, dm_hi], coarse = bucket centre,
  ``fine_to_coarse`` = the bucket holding the fine DM;
* stage 1 (corr), per chgroup g, summed channel k, coarse d::

      a1[g, k]  = 1e3 K (1/nu[g,k]^2 - 1/nu_ref[g, k // per]^2)      us per pc/cc
      nat       = rint(a1 * DM_c / 32.768)                            native samples
      stage1    = rint(nat / q1_native)                               q1 samples, >= 0

  nu = band-centre summed-channel frequencies
  (``build_summed_chgroup_freq_table_GHz(csf)``), nu_ref = the FIRST summed
  channel of the sub-band (``DMPlan.delay_bins_per_subband`` semantics);
* stage 2 (search), per fine f, stream = g * n_sub + s::

      a2[g, s]  = 1e3 K (1/nu_bot_proc^2 - 1/nu_ref[g, s]^2)          us per pc/cc
      stage2    = rint(a2 * DM_f / T_tier)                            tier samples

  one rounding, from the sub-band reference to nu_bot_proc (corr stage-2
  FIFO off). A stream sample t of stream (g, s) holds the sub-band-reference
  arrival time t; the search reads ``stream[t - stage2[f, stream]]`` to align
  every stream to nu_bot_proc (the same read convention as today's combine).

Global coarse index and ownership
=================================
Coarse DMs are numbered tier A first, then tier B: ``d = tier_coarse_offsets[i]
+ c``. Fine DMs likewise (``fine_dm`` is strictly increasing). Half ``h = d % 8``
owns coarse d; half h is search node ``SEARCH_NODES[h // 2]`` GPU ``h % 2``.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsart.common.constants import (
    DELTA_NU_CH_GHZ,
    K_DM_MS_GHZ2_PC,
    N_CHAN_PROC_NATIVE,
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NU_BOT_PROC_GHZ,
    NU_TOP_PROC_GHZ,
    freq_GHz,
)

__all__ = [
    "F_CTR_MHZ",
    "PLAN_SCHEMA",
    "PLAN_VERSION",
    "SEARCH_NODES",
    "TieredDmPlan",
    "build_tiered_plan",
    "build_tiered_plan_for_opoint",
    "gen_dm_list",
    "load_any_plan",
    "plan_metadata_version",
    "stage1_coeff_us",
    "stage2_coeff_us",
    "subband_ref_freqs_GHz",
    "summed_freqs_GHz",
    "tier_grid",
]

PLAN_SCHEMA = "dsart.x2.TieredDmPlan"
PLAN_VERSION = 3

#: Search nodes in N_SEARCH order; half h -> SEARCH_NODES[h // 2], gpu h % 2.
SEARCH_NODES: tuple[str, ...] = ("n01", "n02", "n09", "n13")
N_HALVES = N_SEARCH * N_SEARCH_GPU

#: Processed band in MHz, edge convention, exactly as physics.py builds it:
#: top = 1498.75, bottom = top - 6143 dnu, centre = their mean (1405.0152587890625).
#: (Rounded from the GHz constants: NU_TOP_PROC_GHZ carries 1.530 - 1024 dnu
#: rounding at the 1e-16 level, which would make the fine grid differ in the
#: last ulp from the oracle's.)
_F_TOP_MHZ: float = round(NU_TOP_PROC_GHZ * 1e3, 6)
_DNU_NATIVE_MHZ: float = DELTA_NU_CH_GHZ * 1e3          # 250/8192, exact
_F_BOT_MHZ: float = _F_TOP_MHZ - (N_CHAN_PROC_NATIVE - 1) * _DNU_NATIVE_MHZ
F_CTR_MHZ: float = 0.5 * (_F_TOP_MHZ + _F_BOT_MHZ)

_K = K_DM_MS_GHZ2_PC
_NU_BOT = float(NU_BOT_PROC_GHZ)


# ---------------------------------------------------------------------------------------
# Grid generation (oracle ports)
# ---------------------------------------------------------------------------------------
def gen_dm_list(dm_min, dm_max, tsamp_us, dnu_mhz, nchans, f_ctr_mhz, ti_us=40.0, tol=1.25):
    """Barsdell ``dedisp_generate_dm_list`` recursion (== legacy gen_dmtrials).

    Verbatim port of ``docs/timeres_study/model/physics.py::gen_dm_list`` (do not
    "simplify" the arithmetic: the oracle test compares bit for bit). The list
    runs until the first value >= dm_max, which is included.
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


def tier_grid(dm_lo: float, dm_hi: float, t_native: int, *, n_sub: int, csf: int, tol: float,
              ti_us: float, coarse_density: float, t_ref_native: int = 32):
    """``(fine, coarse, f2c)`` of one tier, exactly as ``schemes._tier``.

    f2c is the tier-LOCAL coarse index (int64).
    """
    T = t_native * NATIVE_SAMPLE_US
    T_ref = t_ref_native * NATIVE_SAMPLE_US
    fine = gen_dm_list(dm_lo, dm_hi, T, _DNU_NATIVE_MHZ * csf, int(N_CHAN_PROC_NATIVE // csf),
                       F_CTR_MHZ, ti_us=ti_us, tol=tol)
    fine = fine[fine <= dm_hi + 1e-9]
    n_c = max(1, math.ceil(coarse_density * (dm_hi - dm_lo) * (T_ref / T) / n_sub))
    edges = np.linspace(dm_lo, dm_hi, n_c + 1)
    coarse = 0.5 * (edges[:-1] + edges[1:])
    f2c = np.clip(np.searchsorted(edges, fine, side="right") - 1, 0, n_c - 1)
    return np.asarray(fine, dtype=np.float64), np.asarray(coarse, dtype=np.float64), f2c


def summed_freqs_GHz(csf: int) -> np.ndarray:
    """``(16, 384/csf)`` band-centre summed-channel frequencies.

    The same arithmetic as ``dsart.coarse_dm.dm_plan.build_summed_chgroup_freq_table_GHz``
    (test-pinned bit-identical), restated here because importing ``dsart.coarse_dm``
    pulls in torch and this module must load on torch-free hosts (h23 tools, dashboard).
    """
    csf = int(csf)
    if csf < 1 or NCHAN_PER_CHGROUP % csf:
        raise ValueError(f"chan_sum_factor={csf} does not divide {NCHAN_PER_CHGROUP}")
    full = np.asarray([[freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)] for g in range(N_CHGROUP)],
                      dtype=np.float64)
    if csf == 1:
        return full
    return full.reshape(N_CHGROUP, NCHAN_PER_CHGROUP // csf, csf).mean(axis=2)


def subband_ref_freqs_GHz(csf: int, n_sub: int) -> np.ndarray:
    """``(16, n_sub)`` stage-1 reference = first summed channel of each sub-band."""
    nus = summed_freqs_GHz(csf)
    per = nus.shape[1] // n_sub
    if per * n_sub != nus.shape[1]:
        raise ValueError(f"n_sub={n_sub} does not divide {nus.shape[1]}")
    return np.ascontiguousarray(nus[:, ::per][:, :n_sub])


def stage1_coeff_us(csf: int, n_sub: int) -> np.ndarray:
    """``(16, nsum)`` a1: stage-1 advance per unit DM (us), sub-band referenced, >= 0."""
    nus = summed_freqs_GHz(csf)
    per = nus.shape[1] // n_sub
    refs = subband_ref_freqs_GHz(csf, n_sub)
    return 1e3 * _K * (1.0 / nus ** 2 - 1.0 / np.repeat(refs, per, axis=1) ** 2)


def stage2_coeff_us(csf: int, n_sub: int) -> np.ndarray:
    """``(16, n_sub)`` a2: delay from each sub-band reference to nu_bot_proc per unit DM (us)."""
    refs = subband_ref_freqs_GHz(csf, n_sub)
    return 1e3 * _K * (1.0 / _NU_BOT ** 2 - 1.0 / refs ** 2)


def _stage1_table(a1: np.ndarray, coarse: np.ndarray, q1_native: int) -> np.ndarray:
    nat = np.rint(a1[:, :, None] * coarse[None, None, :] / NATIVE_SAMPLE_US)
    return np.rint(nat / float(q1_native)).astype(np.int32)


def _stage2_table(a2: np.ndarray, fine: np.ndarray, t_native: int) -> np.ndarray:
    T = t_native * NATIVE_SAMPLE_US
    flat = a2.reshape(-1)                                         # stream = g * n_sub + s
    return np.rint(flat[None, :] * fine[:, None] / T).astype(np.int32)


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------
#: (name, dtype, shape-spec). Shape symbols: T tiers, C coarse, F fine, G chgroups,
#: K summed channels per chgroup, S streams (G * n_sub), N n_sub.
_ARRAYS: tuple[tuple[str, str, tuple], ...] = (
    ("tier_name", "<U8", ("T",)),
    ("tier_dm_lo", "float64", ("T",)),
    ("tier_dm_hi", "float64", ("T",)),
    ("tier_t_native", "int32", ("T",)),
    ("tier_q1_native", "int32", ("T",)),
    ("tier_coarse_offsets", "int32", ("T+1",)),
    ("tier_fine_offsets", "int32", ("T+1",)),
    ("coarse_dm", "float64", ("C",)),
    ("coarse_tier", "int32", ("C",)),
    ("coarse_t_native", "int32", ("C",)),
    ("fine_offsets", "int32", ("C+1",)),
    ("n_fdm", "int32", ("C",)),
    ("owner_half", "int32", ("C",)),
    ("stage2_max", "int32", ("C",)),
    ("stage2_min", "int32", ("C",)),
    ("fine_dm", "float64", ("F",)),
    ("fine_to_coarse", "int32", ("F",)),
    ("fine_tier", "int32", ("F",)),
    ("stage1_bins", "int32", ("G", "K", "C")),
    ("stage2_bins", "int32", ("F", "S")),
    ("summed_freq_GHz", "float64", ("G", "K")),
    ("subband_ref_GHz", "float64", ("G", "N")),
    ("stage1_coeff_us", "float64", ("G", "K")),
    ("stage2_coeff_us", "float64", ("G", "N")),
)


@dataclass(frozen=True)
class TieredDmPlan:
    """DM plan v3 (SPEC §2). Arrays are read-only numpy arrays; see :data:`_ARRAYS`.

    Per tier i (``tier_*[i]``): DM range, stream sample ``tier_t_native`` (16 | 32),
    stage-1 quantum ``tier_q1_native`` (8), and the global index ranges
    ``[tier_coarse_offsets[i], tier_coarse_offsets[i+1])`` /
    ``[tier_fine_offsets[i], tier_fine_offsets[i+1])``.

    Per global coarse d: ``coarse_dm``, ``coarse_tier``, ``coarse_t_native``, the CSR
    ``fine_offsets`` (fines of d = ``fine_offsets[d]:fine_offsets[d+1]``), ``n_fdm``
    (VARIES per d), ``owner_half`` (= d % 8), and ``stage2_max`` / ``stage2_min``
    (extreme stage-2 shift over d's fines and all streams, tier samples; ring sizing).

    Per global fine f: ``fine_dm`` (strictly increasing), ``fine_to_coarse`` (global d),
    ``fine_tier``; ``stage2_bins[f, g * n_sub + s]`` int32 (tier samples, >= 0 here;
    signed by contract).

    ``stage1_bins[g, k, d]`` int32: stage-1 advance of summed channel k of chgroup g at
    coarse d, in units of ``tier_q1_native`` native samples (>= 0; 0 on every
    sub-band's first channel).
    """

    n_sub: int
    chan_sum_factor: int
    tol: float
    ti_us: float
    f_ctr_MHz: float
    coarse_density: float
    coarse_density_t_ref_native: int
    n_halves: int
    tier_name: np.ndarray
    tier_dm_lo: np.ndarray
    tier_dm_hi: np.ndarray
    tier_t_native: np.ndarray
    tier_q1_native: np.ndarray
    tier_coarse_offsets: np.ndarray
    tier_fine_offsets: np.ndarray
    coarse_dm: np.ndarray
    coarse_tier: np.ndarray
    coarse_t_native: np.ndarray
    fine_offsets: np.ndarray
    n_fdm: np.ndarray
    owner_half: np.ndarray
    stage2_max: np.ndarray
    stage2_min: np.ndarray
    fine_dm: np.ndarray
    fine_to_coarse: np.ndarray
    fine_tier: np.ndarray
    stage1_bins: np.ndarray
    stage2_bins: np.ndarray
    summed_freq_GHz: np.ndarray
    subband_ref_GHz: np.ndarray
    stage1_coeff_us: np.ndarray
    stage2_coeff_us: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        for name, _, _ in _ARRAYS:
            arr = getattr(self, name)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
        self.validate()

    # -- sizes --------------------------------------------------------------------------
    @property
    def n_tiers(self) -> int:
        return int(self.tier_name.shape[0])

    @property
    def n_coarse(self) -> int:
        return int(self.coarse_dm.shape[0])

    @property
    def n_fine(self) -> int:
        return int(self.fine_dm.shape[0])

    @property
    def n_streams(self) -> int:
        return N_CHGROUP * self.n_sub

    @property
    def nchan_eff(self) -> int:
        return NCHAN_PER_CHGROUP // self.chan_sum_factor

    # -- per-DM helpers ------------------------------------------------------------------
    def fines_of(self, d: int) -> slice:
        """Global fine indices of coarse d (contiguous)."""
        return slice(int(self.fine_offsets[d]), int(self.fine_offsets[d + 1]))

    def t_us(self, d: int) -> float:
        """Stream sample of coarse d (us)."""
        return int(self.coarse_t_native[d]) * NATIVE_SAMPLE_US

    def owner_node(self, d: int) -> str:
        return SEARCH_NODES[int(self.owner_half[d]) // N_SEARCH_GPU]

    def owner_gpu(self, d: int) -> int:
        return int(self.owner_half[d]) % N_SEARCH_GPU

    def dms_of_half(self, h: int) -> np.ndarray:
        """Global coarse indices owned by half h (ascending)."""
        return np.flatnonzero(self.owner_half == int(h)).astype(np.int32)

    def stage2_for_coarse(self, d: int) -> np.ndarray:
        """``(n_fdm[d], n_streams)`` int32 stage-2 shifts of coarse d's fines."""
        return self.stage2_bins[self.fines_of(d)]

    def tier_dict(self, i: int) -> dict:
        """Tier i in the oracle's ``schemes._tier`` form (fine, coarse, f2c local, q1, q2 us)."""
        c0, c1 = int(self.tier_coarse_offsets[i]), int(self.tier_coarse_offsets[i + 1])
        f0, f1 = int(self.tier_fine_offsets[i]), int(self.tier_fine_offsets[i + 1])
        return dict(fine=self.fine_dm[f0:f1], coarse=self.coarse_dm[c0:c1],
                    f2c=(self.fine_to_coarse[f0:f1] - c0).astype(np.int64),
                    q1=int(self.tier_q1_native[i]) * NATIVE_SAMPLE_US,
                    q2=int(self.tier_t_native[i]) * NATIVE_SAMPLE_US)

    # -- validation ---------------------------------------------------------------------
    def validate(self) -> None:
        """Structural invariants; raises ValueError/TypeError. Always runs on construction."""
        nT, nC, nF = self.n_tiers, self.n_coarse, self.n_fine
        dims = {"T": nT, "T+1": nT + 1, "C": nC, "C+1": nC + 1, "F": nF, "G": N_CHGROUP,
                "K": self.nchan_eff, "S": self.n_streams, "N": self.n_sub}
        for name, dtype, spec in _ARRAYS:
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"TieredDmPlan.{name} must be ndarray, got {type(arr).__name__}")
            if np.dtype(arr.dtype) != np.dtype(dtype):
                raise TypeError(f"TieredDmPlan.{name}.dtype={arr.dtype}, expected {dtype}")
            shape = tuple(dims[s] for s in spec)
            if arr.shape != shape:
                raise ValueError(f"TieredDmPlan.{name}.shape={arr.shape}, expected {shape}")
        if self.metadata.get("version") != PLAN_VERSION or self.metadata.get("schema") != PLAN_SCHEMA:
            raise ValueError(f"metadata version/schema {self.metadata.get('version')}/"
                             f"{self.metadata.get('schema')} != {PLAN_VERSION}/{PLAN_SCHEMA}")
        if self.nchan_eff % self.n_sub:
            raise ValueError(f"n_sub={self.n_sub} does not divide nchan_eff={self.nchan_eff}")
        if self.n_halves != N_HALVES:
            raise ValueError(f"n_halves={self.n_halves} != {N_HALVES}")
        for off, n in ((self.tier_coarse_offsets, nC), (self.tier_fine_offsets, nF)):
            if off[0] != 0 or off[-1] != n or np.any(np.diff(off) < 1):
                raise ValueError("tier offsets must start at 0, end at the total, every tier non-empty")
        if np.any(np.diff(self.fine_dm) <= 0):
            raise ValueError("fine_dm must be strictly increasing (tiers in DM order)")
        if np.any(np.diff(self.coarse_dm) <= 0):
            raise ValueError("coarse_dm must be strictly increasing")
        for i in range(nT):
            c0, c1 = self.tier_coarse_offsets[i], self.tier_coarse_offsets[i + 1]
            f0, f1 = self.tier_fine_offsets[i], self.tier_fine_offsets[i + 1]
            if np.any(self.coarse_tier[c0:c1] != i) or np.any(self.fine_tier[f0:f1] != i):
                raise ValueError(f"tier {i}: coarse_tier/fine_tier inconsistent with offsets")
            if np.any(self.coarse_t_native[c0:c1] != self.tier_t_native[i]):
                raise ValueError(f"tier {i}: coarse_t_native != tier_t_native")
            if np.any((self.fine_to_coarse[f0:f1] < c0) | (self.fine_to_coarse[f0:f1] >= c1)):
                raise ValueError(f"tier {i}: a fine maps to a coarse DM of another tier")
            lo, hi = self.tier_dm_lo[i], self.tier_dm_hi[i]
            if np.any((self.fine_dm[f0:f1] < lo - 1e-9) | (self.fine_dm[f0:f1] > hi + 1e-9)):
                raise ValueError(f"tier {i}: fine DM outside [{lo}, {hi}]")
            if self.tier_t_native[i] % self.tier_q1_native[i]:
                raise ValueError(f"tier {i}: stream sample not a multiple of q1")
        if np.any(np.diff(self.fine_to_coarse) < 0):
            raise ValueError("fine_to_coarse must be non-decreasing (contiguous fines per coarse)")
        counts = np.bincount(self.fine_to_coarse, minlength=nC)
        if counts.shape[0] != nC or np.any(counts != self.n_fdm) or np.any(self.n_fdm < 1):
            raise ValueError("n_fdm must equal the fines per coarse and be >= 1 everywhere")
        if self.fine_offsets[0] != 0 or np.any(np.diff(self.fine_offsets) != self.n_fdm):
            raise ValueError("fine_offsets must be the CSR of n_fdm")
        if np.any(self.owner_half != np.arange(nC) % self.n_halves):
            raise ValueError("owner_half must be d % n_halves")
        if np.any(self.stage1_bins < 0):
            raise ValueError("stage1_bins must be >= 0 (advance toward the sub-band reference)")
        per = self.nchan_eff // self.n_sub
        if np.any(self.stage1_bins[:, ::per, :] != 0):
            raise ValueError("stage1_bins must be 0 on every sub-band's reference channel")
        for d in range(nC):
            s2 = self.stage2_bins[self.fines_of(d)]
            if s2.max() != self.stage2_max[d] or s2.min() != self.stage2_min[d]:
                raise ValueError(f"stage2_max/min inconsistent at coarse {d}")

    def verify_rebuild(self) -> None:
        """Rebuild from ``metadata['inputs']`` and require bit-identical arrays.

        The "fail loudly" check a service runs at start: a plan file whose tables were
        edited, or built by different code/constants, refuses to load.
        """
        inp = self.metadata["inputs"]
        ref = build_tiered_plan(
            n_sub=inp["n_sub"], chan_sum_factor=inp["chan_sum_factor"],
            tiers=[(t["name"], t["dm_lo"], t["dm_hi"], t["t_native"], t["q1_native"]) for t in inp["tiers"]],
            tol=inp["tol"], ti_us=inp["ti_us"], coarse_density=inp["coarse_density"],
            coarse_density_t_ref_native=inp["coarse_density_t_ref_native"],
            git_sha=self.metadata.get("git_sha"), build_utc_ns=self.metadata.get("build_utc_ns"))
        for name, _, _ in _ARRAYS:
            a, b = getattr(self, name), getattr(ref, name)
            if not np.array_equal(a, b):
                raise ValueError(f"TieredDmPlan.{name} differs from a rebuild from its inputs")
        if float(self.f_ctr_MHz) != F_CTR_MHZ:
            raise ValueError(f"f_ctr_MHz={self.f_ctr_MHz} != {F_CTR_MHZ}")

    # -- npz -----------------------------------------------------------------------------
    def to_npz(self, path) -> None:
        scal = dict(n_sub=self.n_sub, chan_sum_factor=self.chan_sum_factor, tol=self.tol, ti_us=self.ti_us,
                    f_ctr_MHz=self.f_ctr_MHz, coarse_density=self.coarse_density,
                    coarse_density_t_ref_native=self.coarse_density_t_ref_native, n_halves=self.n_halves)
        np.savez(path, metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype="U"),
                 scalars=np.asarray(json.dumps(scal, sort_keys=True), dtype="U"),
                 **{name: getattr(self, name) for name, _, _ in _ARRAYS})

    @classmethod
    def from_npz(cls, path, *, verify: bool = False) -> "TieredDmPlan":
        """Load + validate. ``verify=True`` also rebuilds from the stored inputs."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta.get("version") != PLAN_VERSION or meta.get("schema") != PLAN_SCHEMA:
                raise ValueError(f"{path}: not a TieredDmPlan v{PLAN_VERSION} file "
                                 f"(metadata version={meta.get('version')!r}, schema={meta.get('schema')!r})")
            scal = json.loads(str(z["scalars"]))
            arrays = {name: np.array(z[name]) for name, _, _ in _ARRAYS}
        plan = cls(n_sub=int(scal["n_sub"]), chan_sum_factor=int(scal["chan_sum_factor"]),
                   tol=float(scal["tol"]), ti_us=float(scal["ti_us"]), f_ctr_MHz=float(scal["f_ctr_MHz"]),
                   coarse_density=float(scal["coarse_density"]),
                   coarse_density_t_ref_native=int(scal["coarse_density_t_ref_native"]),
                   n_halves=int(scal["n_halves"]), metadata=meta, **arrays)
        if verify:
            plan.verify_rebuild()
        return plan

    def summary(self) -> dict:
        out = {"version": PLAN_VERSION, "n_sub": self.n_sub, "csf": self.chan_sum_factor,
               "n_coarse": self.n_coarse, "n_fine": self.n_fine, "tiers": []}
        for i in range(self.n_tiers):
            c = slice(int(self.tier_coarse_offsets[i]), int(self.tier_coarse_offsets[i + 1]))
            f = slice(int(self.tier_fine_offsets[i]), int(self.tier_fine_offsets[i + 1]))
            out["tiers"].append(dict(
                name=str(self.tier_name[i]), dm=[float(self.tier_dm_lo[i]), float(self.tier_dm_hi[i])],
                T_us=int(self.tier_t_native[i]) * NATIVE_SAMPLE_US, n_coarse=c.stop - c.start,
                n_fine=f.stop - f.start, fine_range=[float(self.fine_dm[f][0]), float(self.fine_dm[f][-1])],
                n_fdm=[int(self.n_fdm[c].min()), int(self.n_fdm[c].max())],
                stage2_max=int(self.stage2_max[c].max())))
        out["dms_per_half"] = [int(np.sum(self.owner_half == h)) for h in range(self.n_halves)]
        out["fines_per_half"] = [int(self.n_fdm[self.owner_half == h].sum()) for h in range(self.n_halves)]
        return out


# ---------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------
def build_tiered_plan(*, n_sub: int, chan_sum_factor: int = 2,
                      tiers=(("A", 50.0, 1600.0, 16, 8), ("B", 1600.0, 3000.0, 32, 8)),
                      tol: float = 1.25, ti_us: float = 40.0, coarse_density: float = 0.0206,
                      coarse_density_t_ref_native: int = 32, opoint_id: str | None = None,
                      git_sha: str | None = None, build_utc_ns: int | None = None) -> TieredDmPlan:
    """Build the v3 plan. ``tiers`` = sequence of (name, dm_lo, dm_hi, t_native, q1_native)."""
    csf = int(chan_sum_factor)
    n_sub = int(n_sub)
    a1 = stage1_coeff_us(csf, n_sub)
    a2 = stage2_coeff_us(csf, n_sub)
    t_rows, fines, coarses, f2cs, s1, s2 = [], [], [], [], [], []
    c_off, f_off = [0], [0]
    for name, lo, hi, t_native, q1_native in tiers:
        fine, coarse, f2c = tier_grid(float(lo), float(hi), int(t_native), n_sub=n_sub, csf=csf, tol=tol,
                                      ti_us=ti_us, coarse_density=coarse_density,
                                      t_ref_native=coarse_density_t_ref_native)
        t_rows.append((str(name), float(lo), float(hi), int(t_native), int(q1_native)))
        fines.append(fine)
        coarses.append(coarse)
        f2cs.append(f2c + c_off[-1])
        s1.append(_stage1_table(a1, coarse, int(q1_native)))
        s2.append(_stage2_table(a2, fine, int(t_native)))
        c_off.append(c_off[-1] + coarse.shape[0])
        f_off.append(f_off[-1] + fine.shape[0])
    nT = len(t_rows)
    coarse_dm = np.concatenate(coarses)
    fine_dm = np.concatenate(fines)
    f2c = np.concatenate(f2cs).astype(np.int32)
    nC = coarse_dm.shape[0]
    coarse_tier = np.repeat(np.arange(nT, dtype=np.int32), np.diff(c_off))
    fine_tier = np.repeat(np.arange(nT, dtype=np.int32), np.diff(f_off))
    t_nat = np.array([r[3] for r in t_rows], dtype=np.int32)
    n_fdm = np.bincount(f2c, minlength=nC).astype(np.int32)
    fine_offsets = np.zeros(nC + 1, dtype=np.int32)
    fine_offsets[1:] = np.cumsum(n_fdm)
    stage2 = np.concatenate(s2, axis=0)
    s2max = np.array([stage2[fine_offsets[d]:fine_offsets[d + 1]].max() if n_fdm[d] else 0
                      for d in range(nC)], dtype=np.int32)
    s2min = np.array([stage2[fine_offsets[d]:fine_offsets[d + 1]].min() if n_fdm[d] else 0
                      for d in range(nC)], dtype=np.int32)
    if build_utc_ns is None:
        build_utc_ns = time.time_ns()
    metadata = {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "git_sha": git_sha if git_sha is not None else _git_sha(),
        "build_utc_ns": int(build_utc_ns),
        "build_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(build_utc_ns) / 1e9)),
        "opoint_id": opoint_id,
        "inputs": {
            "n_sub": n_sub, "chan_sum_factor": csf, "tol": float(tol), "ti_us": float(ti_us),
            "coarse_density": float(coarse_density),
            "coarse_density_t_ref_native": int(coarse_density_t_ref_native),
            "tiers": [dict(name=r[0], dm_lo=r[1], dm_hi=r[2], t_native=r[3], q1_native=r[4]) for r in t_rows],
        },
        "constants": {
            "NATIVE_SAMPLE_US": NATIVE_SAMPLE_US, "K_DM_MS_GHZ2_PC": K_DM_MS_GHZ2_PC,
            "NU_TOP_PROC_GHZ": NU_TOP_PROC_GHZ, "NU_BOT_PROC_GHZ": NU_BOT_PROC_GHZ,
            "F_CTR_MHZ": F_CTR_MHZ, "N_CHGROUP": N_CHGROUP, "NCHAN_PER_CHGROUP": NCHAN_PER_CHGROUP,
            "N_SEARCH": N_SEARCH, "N_SEARCH_GPU": N_SEARCH_GPU,
        },
        "search_nodes": list(SEARCH_NODES),
        "oracle": "docs/timeres_study/snr2x/schemes.py::proposed(q1=262.144, n_sub=n_sub)",
        "conventions": (
            "stage1_bins[g,k,d] = rint(rint(a1[g,k]*DM_d/32.768)/q1_native), a1 referenced to the first "
            "summed channel of the sub-band, >= 0 (corr advances later channels); stage2_bins[f,g*n_sub+s] "
            "= rint(a2[g,s]*DM_f/T_tier), a2 from the sub-band reference to nu_bot_proc, one rounding, "
            "applied by the search (reads stream[t - stage2]); owner half = d % 8 -> "
            "search_nodes[h//2] gpu h%2"),
    }
    return TieredDmPlan(
        n_sub=n_sub, chan_sum_factor=csf, tol=float(tol), ti_us=float(ti_us), f_ctr_MHz=F_CTR_MHZ,
        coarse_density=float(coarse_density), coarse_density_t_ref_native=int(coarse_density_t_ref_native),
        n_halves=N_HALVES,
        tier_name=np.array([r[0] for r in t_rows], dtype="<U8"),
        tier_dm_lo=np.array([r[1] for r in t_rows], dtype=np.float64),
        tier_dm_hi=np.array([r[2] for r in t_rows], dtype=np.float64),
        tier_t_native=t_nat,
        tier_q1_native=np.array([r[4] for r in t_rows], dtype=np.int32),
        tier_coarse_offsets=np.array(c_off, dtype=np.int32),
        tier_fine_offsets=np.array(f_off, dtype=np.int32),
        coarse_dm=coarse_dm, coarse_tier=coarse_tier, coarse_t_native=t_nat[coarse_tier],
        fine_offsets=fine_offsets, n_fdm=n_fdm,
        owner_half=(np.arange(nC) % N_HALVES).astype(np.int32),
        stage2_max=s2max, stage2_min=s2min,
        fine_dm=fine_dm, fine_to_coarse=f2c, fine_tier=fine_tier,
        stage1_bins=np.concatenate(s1, axis=2), stage2_bins=stage2,
        summed_freq_GHz=np.ascontiguousarray(summed_freqs_GHz(csf), dtype=np.float64),
        subband_ref_GHz=subband_ref_freqs_GHz(csf, n_sub).astype(np.float64),
        stage1_coeff_us=np.ascontiguousarray(a1, dtype=np.float64),
        stage2_coeff_us=np.ascontiguousarray(a2, dtype=np.float64),
        metadata=metadata,
    )


def build_tiered_plan_for_opoint(op, **kw) -> TieredDmPlan:
    """:func:`build_tiered_plan` with every parameter taken from a 2x :class:`OpPoint`."""
    if op.plan_version != 3:
        raise ValueError(f"op-point {op.opoint_id} uses a v{op.plan_version} plan, not a tiered one")
    return build_tiered_plan(
        n_sub=op.n_sub, chan_sum_factor=op.chan_sum_factor,
        tiers=[(t.name, t.dm_lo, t.dm_hi, t.t_native, op.stage1_q_native) for t in op.tiers],
        tol=op.dm_tol, ti_us=op.dm_ti_us, coarse_density=op.coarse_density,
        coarse_density_t_ref_native=op.coarse_density_t_ref_native, opoint_id=op.opoint_id, **kw)


def plan_metadata_version(path) -> int:
    """The ``metadata['version']`` of a v2 or v3 plan file."""
    with np.load(path, allow_pickle=False) as z:
        return int(json.loads(str(z["metadata"]))["version"])


def load_any_plan(path):
    """v2 -> :class:`dsart.common.contracts.DmPlan`; v3 -> :class:`TieredDmPlan`."""
    v = plan_metadata_version(path)
    if v == PLAN_VERSION:
        return TieredDmPlan.from_npz(path)
    if v == 2:
        from dsart.common.contracts import DmPlan
        return DmPlan.from_npz(str(path))
    raise ValueError(f"{path}: unsupported DM plan metadata version {v}")
