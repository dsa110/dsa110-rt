"""Slim coarse-DM-focused view over the canonical :class:`DmPlan` (M3 chunk 3b).

Wraps the canonical :class:`dsart.common.contracts.DmPlan` (produced by
:mod:`tools.build_dm_plan`) with a minimal interface tailored to the
**coarse-DM dedispersion** consumers:

* :func:`dsart.coarse_dm.dedisp.coarse_dedisp` (corr-side; this chunk).
* :mod:`dsart.fine_dm.combiner` (search-side; M5; **read-only consumer**).

Both consumers only need:

1. The coarse-DM trial values (``dm_pc_cc``, the coarse-DM grid).
2. The bin width (``t_int_fast_us``) of the input cube's time axis.
3. The full per-(chgroup, channel) frequency grid (``chgroup_freqs_GHz``).
4. The pre-computed per-(chgroup, channel, dm-trial) time delays
   needed for the integer-sample shift-and-sum.

The canonical :class:`DmPlan` carries a much larger schema (fine-DM
grid, CSR fine→coarse map, search-side DM-range partitioning, etc.)
that the corr-side coarse-DM module does **not** need; exposing only
the focused view here keeps the coarse-DM consumer surface small +
makes the M5 import-time coupling explicit (M5 imports just
:class:`DMPlan` from this module rather than reaching into
``contracts.DmPlan``).

Sign convention (Convention A — "reference = chgroup TOP")
==========================================================

The canonical :class:`DmPlan` stores ``time_shift_corr_stage1[g, ch, c]``
referenced to **chgroup BOT** (the lowest freq in the chgroup, which has
the largest dispersion delay). That convention is suited to stage-1's
job of producing chgroup-output already aligned to ν_chgroup_bot ready
for stage-2's cross-chgroup alignment to ν_bot_proc.

This module's :meth:`DMPlan.delay_native_samples` instead uses
**Convention A: reference = chgroup TOP**, so that:

* ``delay_native_samples(g, ch=0, dm) == 0`` for every ``g`` and ``dm``
  (the chgroup's top channel has zero shift by definition).
* ``delay_native_samples(g, ch, dm) > 0`` for every other ``ch`` (lower
  freq channels are delayed more by cold-plasma dispersion).

This Convention A choice matches the **injector** (see
:func:`dsart.inject.online.build_dispersion_delay_table_ms`) which
references ``NU_TOP_PROC_GHZ``; the chunk-3b dedisperser composes
end-to-end with the injector when the injector and dedisperser share a
chgroup (``inj.NU_TOP_PROC_GHZ`` is ``freq_GHz(0, 0)`` and our chgroup
0 reference is ``freq_GHz(0, 0)`` — same value).

The two conventions are equivalent up to a constant ``max_delay``
offset across channels at fixed ``(g, dm)``; converting between them is
trivial via ``delay_B = max_delay_in_chgroup - delay_A``. F-item F24
(proposed; see test ``test_F24_coarse_dm_uses_native_t_axis``) pins
this convention choice.

Native-sample units, not bin units (F24)
========================================

The stored delays are in **native sample units** (32.768 µs per sample;
:data:`dsart.common.constants.NATIVE_SAMPLE_US`). The dedisperser
converts to fast-vis bin units at apply time via
:meth:`DMPlan.delay_bins`. This is more accurate than rounding once at
the bin level because it preserves sub-bin alignment information that
composes with downstream stage-2 cross-chgroup shifts (which are *also*
computed in native sample units in the canonical DmPlan).

References
==========

* Plan §3.2 (DM plan: coarse + fine).
* Plan §3.6.1 lines ~702 (dispersion constant).
* Plan §3.6.2 lines ~726-770 (DEDISP architecture).
* Plan §4.2 lines ~1283-1346 (streaming pipeline placement).
* :class:`dsart.common.contracts.DmPlan` — canonical schema.
* :mod:`tools.build_dm_plan` — producer of ``configs/dm_plan.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from dsart.common.constants import (
    K_DM_MS_GHZ2_PC,
    NATIVE_SAMPLE_US,
    N_CHGROUP,
    NCHAN_PER_CHGROUP,
    freq_GHz,
)


__all__ = [
    "DMPlan",
    "build_chgroup_freq_table_GHz",
    "compute_delay_native_samples_table",
    "load_dm_plan",
]


# Reference convention for the per-(g, ch, dm) delay table:
#   delay[g, ch, c] = round( K * dm[c] * (1/freq(g, ch)² - 1/freq(g, 0)²) * 1e3 / NATIVE_SAMPLE_US )
# (units: ms · GHz² · cm³ / pc cancels with dm pc/cc · 1/GHz² to give ms;
#  ×1e3 → µs; ÷ NATIVE_SAMPLE_US µs → integer native samples.)
_REF_PER_CHGROUP_TOP: Final[str] = "chgroup_top"


def build_chgroup_freq_table_GHz() -> np.ndarray:
    """Build the full ``(N_chgroup, NCHAN_PER_CHGROUP)`` freq table (GHz).

    Pinned to the system :func:`dsart.common.constants.freq_GHz`. Rows
    are descending in frequency within each chgroup (chan_ascending=False
    in ``corr_setup_96.yaml``: higher local_ch → lower frequency).

    Returns:
        ``(N_CHGROUP, NCHAN_PER_CHGROUP) float64``.
    """
    return np.asarray(
        [
            [freq_GHz(g, ch) for ch in range(NCHAN_PER_CHGROUP)]
            for g in range(N_CHGROUP)
        ],
        dtype=np.float64,
    )


def compute_delay_native_samples_table(
    coarse_dm: np.ndarray,
    chgroup_freqs_GHz: np.ndarray,
) -> np.ndarray:
    """Compute the canonical ``(N_chgroup, NCHAN_PER_CHGROUP, N_coarse)`` delay table.

    Per Convention A (reference = each chgroup's own TOP channel):

    .. math::

       \\Delta_\\text{nat}[g, c, k] = \\text{round}\\left(
           K_\\text{DM} \\cdot \\text{dm}[k] \\cdot \\left(
               \\frac{1}{\\nu_{g,c}^2} - \\frac{1}{\\nu_{g,0}^2}
           \\right) \\cdot \\frac{10^3}{\\Delta t_\\text{nat}}
       \\right)

    where ``ν_{g,0}`` is the chgroup's TOP frequency (highest ν within
    the chgroup, ``freq_GHz(g, 0)``) and ``Δt_nat`` is
    :data:`dsart.common.constants.NATIVE_SAMPLE_US`.

    All entries are non-negative (lower ν → larger τ); ``ch=0`` row is
    identically zero by construction (any DM × any chgroup) — the
    ``test_dm_plan_delay_zero_at_top_freq`` invariant.

    Args:
        coarse_dm: ``(N_coarse,) float64`` coarse-DM trial values
            (pc / cm³).
        chgroup_freqs_GHz: ``(N_chgroup, NCHAN_PER_CHGROUP) float64``
            per-(chgroup, ch) frequency grid in GHz.

    Returns:
        ``(N_chgroup, NCHAN_PER_CHGROUP, N_coarse) int64``.
    """
    if coarse_dm.ndim != 1:
        raise ValueError(f"coarse_dm must be 1-D; got shape {coarse_dm.shape}")
    if chgroup_freqs_GHz.ndim != 2:
        raise ValueError(
            f"chgroup_freqs_GHz must be 2-D; got shape {chgroup_freqs_GHz.shape}"
        )
    n_chgroup, nchan = chgroup_freqs_GHz.shape
    nu_g_ch = chgroup_freqs_GHz.astype(np.float64)                # (G, C)
    nu_g_top = nu_g_ch[:, 0:1]                                    # (G, 1) top of chgroup
    inv_diff = (1.0 / (nu_g_ch ** 2)) - (1.0 / (nu_g_top ** 2))   # (G, C); ≥ 0
    delay_us = (
        K_DM_MS_GHZ2_PC
        * coarse_dm.astype(np.float64)[None, None, :]              # (1, 1, K)
        * inv_diff[:, :, None] * 1e3                               # (G, C, 1)
    )                                                              # (G, C, K) µs
    delay_native = np.rint(delay_us / NATIVE_SAMPLE_US).astype(np.int64)
    return delay_native


@dataclass(frozen=True, slots=True)
class DMPlan:
    """Coarse-DM dedispersion's slim view over the canonical
    :class:`dsart.common.contracts.DmPlan`.

    Construct via :meth:`from_canonical` (when you already have the
    canonical plan in memory) or :meth:`from_npz` /
    :func:`load_dm_plan` (to read ``configs/dm_plan.npz``).

    Hot-path lookup is :meth:`delay_native_samples` (precomputed
    table; O(1) per lookup) or its bin-quantised counterpart
    :meth:`delay_bins`. The vectorised forms
    :meth:`delay_native_samples_per_chgroup` and
    :meth:`delay_bins_per_chgroup` return entire ``(NCHAN_PER_CHGROUP,
    N_coarse)`` slices for a single chgroup with no per-call Python
    overhead — these are what the dedisp kernel calls.

    Attributes
    ----------
    dm_pc_cc:
        ``(N_coarse,) float64`` — coarse-DM trial values in pc / cm³.
    n_fine_per_coarse:
        Mean fine-DM grid points per coarse interval, taken from the
        canonical plan's CSR ``fine_offsets_idx`` (rounded down to int).
        Used by the search-side fine-DM combiner (M5).
    t_int_fast_us:
        Fast-vis bin width (µs). Matches the cube's time-axis cadence.
    chgroup_freqs_GHz:
        ``(N_CHGROUP, NCHAN_PER_CHGROUP) float64`` frequency table.
    """

    dm_pc_cc: np.ndarray
    n_fine_per_coarse: int
    t_int_fast_us: float
    chgroup_freqs_GHz: np.ndarray

    # Internal precomputed delay table — referenced via the public
    # API methods below. Kept in a private slot so callers can't mutate
    # it accidentally.
    _delay_native_samples_table: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.dm_pc_cc.ndim != 1:
            raise ValueError(
                f"DMPlan.dm_pc_cc must be 1-D; got shape "
                f"{self.dm_pc_cc.shape}"
            )
        if self.dm_pc_cc.dtype != np.float64:
            raise TypeError(
                f"DMPlan.dm_pc_cc dtype must be float64; got "
                f"{self.dm_pc_cc.dtype}"
            )
        if self.chgroup_freqs_GHz.shape != (N_CHGROUP, NCHAN_PER_CHGROUP):
            raise ValueError(
                f"DMPlan.chgroup_freqs_GHz shape must be "
                f"({N_CHGROUP}, {NCHAN_PER_CHGROUP}); got "
                f"{self.chgroup_freqs_GHz.shape}"
            )
        if self.t_int_fast_us <= 0:
            raise ValueError(
                f"DMPlan.t_int_fast_us={self.t_int_fast_us}, expected > 0"
            )
        if self.n_fine_per_coarse < 1:
            raise ValueError(
                f"DMPlan.n_fine_per_coarse={self.n_fine_per_coarse}, "
                f"expected ≥ 1"
            )
        # Strictly increasing dm grid (allow N_coarse == 1).
        n_coarse = self.dm_pc_cc.shape[0]
        if n_coarse >= 2 and not np.all(np.diff(self.dm_pc_cc) > 0):
            raise ValueError("DMPlan.dm_pc_cc must be strictly increasing")
        if self.dm_pc_cc[0] < 0:
            raise ValueError(
                f"DMPlan.dm_pc_cc[0]={self.dm_pc_cc[0]} < 0; "
                f"negative DMs are not physical"
            )

        # Precomputed delay table dimensions.
        if self._delay_native_samples_table.shape != (
            N_CHGROUP, NCHAN_PER_CHGROUP, n_coarse,
        ):
            raise ValueError(
                f"DMPlan._delay_native_samples_table shape must be "
                f"({N_CHGROUP}, {NCHAN_PER_CHGROUP}, {n_coarse}); got "
                f"{self._delay_native_samples_table.shape}"
            )
        if self._delay_native_samples_table.dtype != np.int64:
            raise TypeError(
                f"DMPlan._delay_native_samples_table dtype must be "
                f"int64; got {self._delay_native_samples_table.dtype}"
            )
        # Convention A invariant: top channel of every chgroup has 0 shift.
        if not np.all(self._delay_native_samples_table[:, 0, :] == 0):
            raise ValueError(
                "DMPlan._delay_native_samples_table: chgroup-top channel "
                "(local_ch=0) must have zero shift for every (g, dm); "
                "Convention A invariant broken."
            )
        # Non-negative invariant.
        if int(self._delay_native_samples_table.min()) < 0:
            raise ValueError(
                "DMPlan._delay_native_samples_table contains negatives; "
                "Convention A guarantees ≥ 0 (lower freq → larger τ)."
            )

    # ------------------------------------------------------------------
    # Properties + scalar lookups
    # ------------------------------------------------------------------

    @property
    def n_coarse(self) -> int:
        """Number of coarse-DM trials (= ``len(dm_pc_cc)``)."""
        return int(self.dm_pc_cc.shape[0])

    @property
    def t_int_fast_native(self) -> float:
        """Fast-vis bin width expressed in NATIVE sample units.

        Equals ``t_int_fast_us / NATIVE_SAMPLE_US``. Often an integer
        in the production pipeline (default ``T_INT_FAST_NATIVE = 8``)
        but the schema allows non-integer values for benches that use
        fractional cadences.
        """
        return float(self.t_int_fast_us) / NATIVE_SAMPLE_US

    def delay_native_samples(
        self, chgroup: int, ch_idx: int, dm_idx: int
    ) -> int:
        """Per-(chgroup, ch, dm) delay in NATIVE samples.

        Convention A: relative to the chgroup's TOP channel
        (``freq_GHz(chgroup, 0)``). Always ≥ 0; identically zero when
        ``ch_idx == 0``.
        """
        return int(self._delay_native_samples_table[chgroup, ch_idx, dm_idx])

    def delay_bins(
        self, chgroup: int, ch_idx: int, dm_idx: int
    ) -> int:
        """Per-(chgroup, ch, dm) delay in fast-vis BIN units.

        Equals ``round(delay_native_samples / t_int_fast_native)``
        — the integer shift the dedisp kernel applies to the cube's
        time axis. Always ≥ 0.

        Note: this rounds the *native* delay to bins (not the µs delay
        to bins). The two are equivalent up to floating-point rounding
        at integer ``t_int_fast_native``, but the native-units form
        composes with the canonical DmPlan stage-2 inter-chgroup
        shifts (also stored in native units) without compound rounding.
        F24 pins this convention.
        """
        nat = self.delay_native_samples(chgroup, ch_idx, dm_idx)
        return int(round(nat / self.t_int_fast_native))

    # ------------------------------------------------------------------
    # Vectorised lookups (hot-path)
    # ------------------------------------------------------------------

    def delay_native_samples_per_chgroup(self, chgroup: int) -> np.ndarray:
        """Return ``(NCHAN_PER_CHGROUP, N_coarse) int64`` slice for one chgroup.

        Zero-copy view into the precomputed table. The dedisp kernel
        calls this once per chgroup and walks all (ch, dm) pairs from
        the returned slice.
        """
        if not 0 <= chgroup < N_CHGROUP:
            raise IndexError(
                f"chgroup={chgroup}, expected 0..{N_CHGROUP - 1}"
            )
        return self._delay_native_samples_table[chgroup]

    def delay_bins_per_chgroup(self, chgroup: int) -> np.ndarray:
        """Return ``(NCHAN_PER_CHGROUP, N_coarse) int64`` bin shifts for one chgroup.

        Allocates one ``(NCHAN_PER_CHGROUP, N_coarse)`` int64 array
        per call. Cheap (≤ 100 KB at default ops); the dedisp kernel
        caches the per-chgroup result for its loop body.
        """
        nat = self.delay_native_samples_per_chgroup(chgroup)
        return np.rint(nat / self.t_int_fast_native).astype(np.int64)

    def max_delay_bins_per_chgroup(self, chgroup: int) -> int:
        """Largest bin shift across (ch, dm) for one chgroup.

        Used by the dedisperser to size the output ``T_dedisp`` axis
        (``T_dedisp = T_fast - max_delay_bins_per_chgroup`` is the
        usable output range per Convention A).
        """
        return int(self.delay_bins_per_chgroup(chgroup).max())

    # ------------------------------------------------------------------
    # NPZ I/O via the canonical DmPlan
    # ------------------------------------------------------------------

    @classmethod
    def from_canonical(cls, plan) -> "DMPlan":
        """Build a :class:`DMPlan` from an existing :class:`DmPlan`.

        Args:
            plan: a :class:`dsart.common.contracts.DmPlan`. Lazily
                imported inside this classmethod so this module's
                top-level import doesn't create a cycle with
                ``contracts``.
        """
        from dsart.common.contracts import DmPlan
        if not isinstance(plan, DmPlan):
            raise TypeError(
                f"from_canonical expected DmPlan; got {type(plan).__name__}"
            )
        coarse_dm = plan.coarse_dm.astype(np.float64, copy=True)
        n_coarse = coarse_dm.shape[0]
        n_fine = plan.fine_dm.shape[0]
        # n_fine_per_coarse: mean per-cell count from the CSR offsets.
        # Floor to int (≥ 1; n_coarse ≥ 1 by DmPlan invariant).
        n_fine_per_coarse = max(1, int(round(n_fine / max(1, n_coarse))))
        t_int_fast_us = float(plan.metadata["t_int_fast_us"])
        chgroup_freqs_GHz = build_chgroup_freq_table_GHz()
        delay_table = compute_delay_native_samples_table(
            coarse_dm, chgroup_freqs_GHz,
        )
        return cls(
            dm_pc_cc=coarse_dm,
            n_fine_per_coarse=n_fine_per_coarse,
            t_int_fast_us=t_int_fast_us,
            chgroup_freqs_GHz=chgroup_freqs_GHz,
            _delay_native_samples_table=delay_table,
        )

    @classmethod
    def from_npz(cls, path: str) -> "DMPlan":
        """Load a :class:`DMPlan` from a ``configs/dm_plan.npz``.

        Reads the canonical :class:`DmPlan` schema and projects to the
        coarse-DM-only view. Round-trips with :meth:`to_npz` (which
        delegates to the canonical writer for write parity).
        """
        from dsart.common.contracts import DmPlan
        return cls.from_canonical(DmPlan.from_npz(path))

    def to_npz(self, path: str) -> None:
        """Write a :class:`DMPlan` out as a coarse-only ``.npz``.

        This writes a *strict subset* of the canonical schema — only
        the coarse-DM fields, the bin width, and the metadata. It is
        intended for **bench / test fixtures** that don't need the
        fine-DM tables (notably the burst-fixture single-DM plan
        described in ``PARALLEL_AGENTS.md`` §5.1).

        For round-trip with the production ``configs/dm_plan.npz`` use
        the canonical :meth:`dsart.common.contracts.DmPlan.to_npz`
        followed by :meth:`from_npz` here.
        """
        np.savez(
            path,
            dm_pc_cc=self.dm_pc_cc,
            n_fine_per_coarse=np.asarray(self.n_fine_per_coarse, dtype=np.int64),
            t_int_fast_us=np.asarray(self.t_int_fast_us, dtype=np.float64),
            chgroup_freqs_GHz=self.chgroup_freqs_GHz,
            delay_native_samples_table=self._delay_native_samples_table,
            schema_tag=np.asarray("DMPlan-v1", dtype="U16"),
        )

    @classmethod
    def from_coarse_only_npz(cls, path: str) -> "DMPlan":
        """Inverse of :meth:`to_npz`. Loads a coarse-only ``.npz``."""
        with np.load(path, allow_pickle=False) as data:
            tag = str(data["schema_tag"])
            if tag != "DMPlan-v1":
                raise ValueError(
                    f"unexpected DMPlan schema tag {tag!r}; expected 'DMPlan-v1'"
                )
            return cls(
                dm_pc_cc=data["dm_pc_cc"].astype(np.float64, copy=False),
                n_fine_per_coarse=int(data["n_fine_per_coarse"]),
                t_int_fast_us=float(data["t_int_fast_us"]),
                chgroup_freqs_GHz=data["chgroup_freqs_GHz"].astype(
                    np.float64, copy=False,
                ),
                _delay_native_samples_table=(
                    data["delay_native_samples_table"].astype(np.int64, copy=False)
                ),
            )


def load_dm_plan(npz_path: str) -> DMPlan:
    """Load a :class:`DMPlan` from a canonical ``configs/dm_plan.npz``.

    Convenience top-level entry point matching the brief
    ``load_dm_plan(npz_path)`` contract.
    """
    return DMPlan.from_npz(npz_path)
