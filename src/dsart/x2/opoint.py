"""Op-point parameters for the 1x and 2x searches (docs/x2/SPEC.md §1).

One frozen :class:`OpPoint` holds every parameter that differs between
today's search (``OPPOINT_1X``) and the 2x search (:func:`opoint_2x`). The
2x op-point is ``docs/timeres_study/snr2x/schemes.py::proposed()``: fast
visibilities at 8 native samples (262.144 us), csf 2 (61 kHz), two DM tiers
(A: 50-1600 at 524.288 us, B: 1600-3000 at 1048.576 us), stage 1 rounded
native -> 8 native, ONE stage-2 rounding at the fine DM applied by the
search. The corner-turn *format* (value kind, n_sub, bits) is chosen at CP1
from measured capacity (plan §Corner-turn format); every row of that table
is one call of :func:`opoint_2x`.

Pure python: no numpy/torch import, so services can read an op-point
before deciding what to load.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from dsart.common.constants import (
    N_CHGROUP,
    N_SEARCH,
    N_SEARCH_GPU,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
)

__all__ = [
    "FORMATS_2X",
    "OPPOINT_1X",
    "OPPOINT_2X",
    "OpPoint",
    "STAGE2_CORR_FIFO_MERGED",
    "STAGE2_SEARCH_SINGLE_RINT",
    "TierSpec",
    "VALUE_KINDS",
    "opoint_2x",
    "opoint_from_id",
]

#: Corner-turn value kinds (SPEC §3). ``gridded``: one value per uv cell
#: (today's semantics); ``baselines``: one value per exact-redundancy class.
VALUE_KINDS: tuple[str, ...] = ("gridded", "baselines")

#: Today's stage 2: the corr-side FIFO aligns chgroups, the search applies
#: the residual with the merged rounding (compute_time_shift_search
#: merge_coarse_rounding=True).
STAGE2_CORR_FIFO_MERGED = "corr_fifo_merged"
#: 2x stage 2: corr FIFO off; one rint at the fine DM, at the tier sample,
#: from each sub-band's reference to nu_bot_proc, applied by the search.
STAGE2_SEARCH_SINGLE_RINT = "search_single_rint"

#: bits per COMPLEX value on the wire (re + im): cint4/5/6/8.
_BITS_MAIN_VALID = (8, 10, 12, 16)


@dataclass(frozen=True)
class TierSpec:
    """One DM tier of the 2x plan: fine DMs in [dm_lo, dm_hi] searched at a
    stream sample of ``t_native`` native samples."""

    name: str
    dm_lo: float
    dm_hi: float
    t_native: int

    @property
    def t_us(self) -> float:
        """Stream sample period (us); exact (power-of-two multiple of 32.768)."""
        return self.t_native * NATIVE_SAMPLE_US

    def __post_init__(self) -> None:
        if not (0.0 <= self.dm_lo < self.dm_hi):
            raise ValueError(f"tier {self.name}: need 0 <= dm_lo < dm_hi, got {self.dm_lo}, {self.dm_hi}")
        if self.t_native < 1:
            raise ValueError(f"tier {self.name}: t_native={self.t_native} < 1")


@dataclass(frozen=True)
class OpPoint:
    """Every parameter that differs between the 1x and 2x op-points.

    Field units: ``*_native`` are native samples (32.768 us); ``bits_*`` are
    bits per complex value (re + im).
    """

    name: str                    #: "1x" | "2x"
    t_int_fast_native: int       #: fast-vis integration (32 | 8)
    chan_sum_factor: int         #: native channels per summed channel (8 | 2)
    n_sub: int                   #: sub-band streams per chgroup (4 | 2, 4)
    stage1_q_native: int         #: stage-1 quantum (32 | 8)
    stage2_mode: str             #: STAGE2_CORR_FIFO_MERGED | STAGE2_SEARCH_SINGLE_RINT
    value_kind: str              #: "gridded" | "baselines"
    bits_main: int               #: 16 (cint8) | 8/10/12 (cint4/5/6)
    bits_heavy: int              #: 0 (no heavy class) | 16 (cint8)
    t_ext_native: int            #: downstream ("external") sample (32 | 16)
    plan_version: int            #: 2 (DmPlan v2 file) | 3 (TieredDmPlan)
    tiers: tuple[TierSpec, ...] = ()   #: () for 1x (the v2 file defines the grid)
    dm_tol: float | None = None        #: gen_dm_list tol (2x: 1.25)
    dm_ti_us: float | None = None      #: gen_dm_list intrinsic width (2x: 40 us)
    #: coarse count per tier = ceil(coarse_density * dDM * (T_ref / T) / n_sub),
    #: T_ref = coarse_density_t_ref_native (32 native = 1048.576 us).
    coarse_density: float | None = None
    coarse_density_t_ref_native: int = 32
    normalise_values: bool = False     #: values carry 1/sqrt(n) on the wire (2x)
    #: heavy values: static count n >= heavy_frac * N_image; None = no heavy class.
    heavy_frac: float | None = None
    redundancy_tol_m: float = 0.05     #: exact-redundancy tolerance (3-D)
    n_grid: int = 256
    cell_lambda_mode: str = "tee45"    #: corr_fast --cell-lambda-mode (production)
    n_halves: int = field(default=N_SEARCH * N_SEARCH_GPU)

    def __post_init__(self) -> None:
        if self.name not in ("1x", "2x"):
            raise ValueError(f"name={self.name!r}, expected '1x' or '2x'")
        for k in ("t_int_fast_native", "chan_sum_factor", "n_sub", "stage1_q_native", "t_ext_native"):
            if int(getattr(self, k)) < 1:
                raise ValueError(f"{k}={getattr(self, k)} < 1")
        if NCHAN_PER_CHGROUP % self.chan_sum_factor:
            raise ValueError(f"chan_sum_factor={self.chan_sum_factor} does not divide {NCHAN_PER_CHGROUP}")
        if self.nchan_eff % self.n_sub:
            raise ValueError(f"n_sub={self.n_sub} does not divide nchan_eff={self.nchan_eff}")
        if self.stage1_q_native % self.t_int_fast_native:
            raise ValueError("stage-1 quantum must be a multiple of the fast-vis sample")
        if self.stage2_mode not in (STAGE2_CORR_FIFO_MERGED, STAGE2_SEARCH_SINGLE_RINT):
            raise ValueError(f"stage2_mode={self.stage2_mode!r}")
        if self.value_kind not in VALUE_KINDS:
            raise ValueError(f"value_kind={self.value_kind!r}, expected one of {VALUE_KINDS}")
        if self.bits_main not in _BITS_MAIN_VALID:
            raise ValueError(f"bits_main={self.bits_main}, expected one of {_BITS_MAIN_VALID}")
        if self.bits_heavy not in (0, 16):
            raise ValueError(f"bits_heavy={self.bits_heavy}, expected 0 or 16")
        if (self.heavy_frac is None) != (self.bits_heavy == 0):
            raise ValueError("heavy_frac and bits_heavy must be both set (2x) or both off (1x)")
        if self.plan_version == 2:
            if self.tiers:
                raise ValueError("plan_version 2 takes its grid from the v2 file: tiers must be ()")
        elif self.plan_version == 3:
            if not self.tiers:
                raise ValueError("plan_version 3 needs at least one tier")
            for a, b in zip(self.tiers[:-1], self.tiers[1:]):
                if a.dm_hi != b.dm_lo:
                    raise ValueError(f"tiers must be contiguous: {a.name}.dm_hi != {b.name}.dm_lo")
            for t in self.tiers:
                if t.t_native % self.stage1_q_native:
                    raise ValueError(f"tier {t.name}: t_native not a multiple of the stage-1 quantum")
            if None in (self.dm_tol, self.dm_ti_us, self.coarse_density):
                raise ValueError("plan_version 3 needs dm_tol, dm_ti_us and coarse_density")
        else:
            raise ValueError(f"plan_version={self.plan_version}, expected 2 or 3")
        if self.n_grid < 2 or self.n_grid & 1:
            raise ValueError(f"n_grid={self.n_grid} must be even")

    # -- derived -------------------------------------------------------------------------
    @property
    def nchan_eff(self) -> int:
        """Summed channels per chgroup (48 | 192)."""
        return NCHAN_PER_CHGROUP // self.chan_sum_factor

    @property
    def n_streams(self) -> int:
        """Corner-turn streams per coarse DM (16 x n_sub); stream = g * n_sub + s."""
        return N_CHGROUP * self.n_sub

    @property
    def t_fast_us(self) -> float:
        return self.t_int_fast_native * NATIVE_SAMPLE_US

    @property
    def stage1_q_us(self) -> float:
        return self.stage1_q_native * NATIVE_SAMPLE_US

    @property
    def t_ext_us(self) -> float:
        """Downstream sample (1048.576 | 524.288 us), SPEC §8."""
        return self.t_ext_native * NATIVE_SAMPLE_US

    @property
    def chan_width_khz(self) -> float:
        return 250e3 / 8192 * self.chan_sum_factor

    @property
    def opoint_id(self) -> str:
        """Stable string id published to etcd / frame-independent logs."""
        if self.name == "1x":
            return "1x"
        return f"2x-{self.value_kind}-ns{self.n_sub}-cint{self.bits_main // 2}"


#: Today's search (deployed 2026-09-23): 1.05 ms, 244 kHz, n_sub 4, gridded
#: cells at cint8 with per-row max-abs scale, corr stage-2 FIFO + merged
#: rounding, DM grid from the v2 plan file.
OPPOINT_1X = OpPoint(
    name="1x",
    t_int_fast_native=32,
    chan_sum_factor=8,
    n_sub=4,
    stage1_q_native=32,
    stage2_mode=STAGE2_CORR_FIFO_MERGED,
    value_kind="gridded",
    bits_main=16,
    bits_heavy=0,
    t_ext_native=32,
    plan_version=2,
)

#: Tier layout of proposed(): A 50-1600 at 16 native, B 1600-3000 at 32 native.
TIERS_2X: tuple[TierSpec, ...] = (
    TierSpec("A", 50.0, 1600.0, 16),
    TierSpec("B", 1600.0, 3000.0, 32),
)


def opoint_2x(value_kind: str = "gridded", n_sub: int = 2, bits_main: int = 8) -> OpPoint:
    """The 2x op-point for one corner-turn format (plan §Corner-turn format)."""
    if n_sub not in (2, 4):
        raise ValueError(f"2x n_sub={n_sub}, expected 2 or 4")
    if bits_main not in (8, 10, 12):
        raise ValueError(f"2x bits_main={bits_main}, expected 8, 10 or 12 (cint4/5/6)")
    return OpPoint(
        name="2x",
        t_int_fast_native=8,
        chan_sum_factor=2,
        n_sub=int(n_sub),
        stage1_q_native=8,
        stage2_mode=STAGE2_SEARCH_SINGLE_RINT,
        value_kind=value_kind,
        bits_main=int(bits_main),
        bits_heavy=16,
        t_ext_native=16,
        plan_version=3,
        tiers=TIERS_2X,
        dm_tol=1.25,
        dm_ti_us=40.0,
        coarse_density=0.0206,
        normalise_values=True,
        heavy_frac=3e-5,     # Phase 0 study 1: 1e-4 compressed n_sub 4 values 2.8-2.9% at the 250-sigma rail
    )


#: SPEC §1 spelling.
OPPOINT_2X = opoint_2x

#: The six rows of the plan's format table, by id.
FORMATS_2X: dict[str, OpPoint] = {
    op.opoint_id: op
    for op in (
        opoint_2x("gridded", 2, 8),
        opoint_2x("gridded", 4, 10),
        opoint_2x("gridded", 4, 8),
        opoint_2x("baselines", 4, 10),
        opoint_2x("baselines", 2, 8),
        opoint_2x("baselines", 4, 8),
    )
}


def opoint_from_id(opoint_id: str) -> OpPoint:
    """Inverse of :attr:`OpPoint.opoint_id` (any 2x format, not only the table rows)."""
    if opoint_id == "1x":
        return OPPOINT_1X
    parts = opoint_id.split("-")
    if len(parts) != 4 or parts[0] != "2x" or not parts[2].startswith("ns") or not parts[3].startswith("cint"):
        raise ValueError(f"unknown op-point id {opoint_id!r}")
    return opoint_2x(parts[1], int(parts[2][2:]), 2 * int(parts[3][4:]))
