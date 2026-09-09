"""Array-common broadband burst detector (core / arm autocorrelation sums).

Why this exists
===============

Every detector in the deployed chain (:mod:`dsart.rfi.sk`,
:mod:`dsart.rfi.bandpass_outlier`, :mod:`dsart.rfi.group_outlier`)
runs *inside a single antenna*, and each needs a reference that the
signal does not also move:

* bandpass-outlier compares a channel against the other channels of
  the same antenna — a broadband burst lifts every channel, so the
  median it is compared against rises with it;
* group-outlier compares an antenna against the other antennas — a
  burst common to the array lifts every antenna, so again the
  reference moves with the signal;
* spectral kurtosis asks whether the voltages are Gaussian *within*
  an accumulation — a burst that fills whole accumulations is
  Gaussian inside them, just louder.

The consequence, measured on ``260812imek sb02`` (offline study,
2026-09-08): its broadband bursts are **0.05 σ** in the cell the
flagger tests (1 antenna, 1 channel, one 134 ms cube) and **13.3 σ**
in the array-summed band power at 2.097 ms — a factor of 270. The
three discriminants above read 3.05 vs 3.09, 38.1 vs 38.9 and 2.98%
vs 2.92% during bursts vs quiet: none of them moves. That is a
structural blind spot, not a threshold that needs lowering.

What this module adds
=====================

One statistic the existing chain does not form: a **short-lived rise
in total power relative to the array's own recent history**.

Scales (see the module-level constants):

``TIME``
    2.097 ms — the ``M = 64`` accumulation :func:`dsart.rfi.compute_autos`
    already builds for SK, so the detector is a reduction over a
    tensor that is *already resident on the GPU*. 64 decisions per
    134.2 ms cube. Going finer would need new moments (a second pass
    over the voltages); going coarser is what the present chain does
    and is exactly what dilutes the signal ~20-fold.

``FREQUENCY``
    Two scales. **Detection** on the full 11.72 MHz sub-band sum,
    which is where the SNR is (13.3 σ integrated against 1.04 σ in a
    single channel). **Decision** on coarse bins of
    :data:`BIN_CHANS_DEFAULT` = 16 channels (0.488 MHz), 24 per node:
    one (82-antenna core, 16-channel, 2.097 ms) cell has ~0.46%
    fractional noise against a ~1.45% burst, so each bin is ~3 σ —
    about the finest binning at which this burst class is still
    individually visible.

Each corr node sees only its own chgroup, so "broadband" here means
broadband *within one 11.72 MHz sub-band*. Cross-node coordination
would need the transport fabric and cannot fit the latency budget.

Groups
======

:func:`build_groups_from_antpos` splits the array on the geometry the
cal blob already carries. DSA-110's core is an offset T: 47 antennas
on an E-W line at constant N (zero spread), 35 on an N-S line at
constant E (zero spread), plus 14 outriggers at station > 102. The
split is stable for any ``arm_tol_m`` between 5 and 50 m.

Measured burst response over the same dump (median / peak
significance of the summed band power at 2.097 ms):

===========  ====  ============  ==========  ====================
group        ants  median sigma  peak sigma  burst % of own power
===========  ====  ============  ==========  ====================
all          96    14.5          39.5        1.42
core         82    15.4          42.8        1.45
EW arm       47    14.6          57.8        1.86
NS arm       35     6.0          31.3        0.81
outriggers   14     2.0          13.8        —
===========  ====  ============  ==========  ====================

The E-W / N-S asymmetry (1.86% vs 0.81%) is a **local vs far-field
discriminant**: a far-field signal illuminates both arms in
proportion to their collecting area, so an asymmetry of 2.3× says the
source is near-field, and roughly where. It is published as a
diagnostic (:attr:`ArrayBurstResult.band_frac` per group) but is
*not* part of the firing decision — see the gate below.

The gate
========

A sample fires when the flag group's band-summed power exceeds
``detect_k`` sigma **and** the excess is present in at least
``occupancy_min`` of the coarse frequency bins. The occupancy term is
what protects real FRBs: across one 11.72 MHz sub-band at 1.4 GHz the
dispersion sweep is ``0.035 · DM`` ms, so within a single 2.097 ms
sample an FRB lights ~60% of the band at DM 100, ~20% at DM 300 and
~6% at DM 1000. A local, dispersion-free burst lights ~100% of the
band, and does fire.

That analytic picture is not the whole story, and the difference
matters. Measured by injecting dispersed pulses into a real dump
(n03, 2026-09-09), the occupancy a sweep actually reaches is higher
than the fraction of band it lights — 0.75 at DM 100, 0.46 at DM 300,
0.29 at DM 1000 — because the wings of a bright sweep still clear the
per-bin threshold. It also **rises with brightness**, so the margin is
not amplitude-independent: at ``occupancy_min = 0.75`` a DM-100 pulse
at +300% band power reached exactly 0.75 and was flagged. The default
is therefore 0.90, clear of every dispersed case measured, while a
dispersion-free burst reaches 1.00 even at +5%.

A genuinely zero-DM broadband event would still fire. Those currently
rail at the ``dm_min = 100`` search floor and are classified
extragalactic by construction, so removing them is arguably a second
win — but it is a real consequence and the operator should know it.

Gain normalisation and dead antennas
====================================

Each antenna is divided by its own running per-(channel, pol) mean
before summing, otherwise a few hot antennas dominate the sum and the
statistic is not the chi-squared it is thresholded as. Dead antennas
must additionally be **excluded**, not merely down-weighted: in the
offline study, including them put the all-96 null at 0.978 ± 0.026
instead of 1.000 ± 0.0013 — a 20 σ error in the very quantity the
threshold is set against.

Cost
====

Everything is a reduction over ``s1[64]`` (``[64, 96, 384, 2]`` fp32
= 18.9 MB, already on the GPU) into ``[64, 5, 24, 2]``, plus one
elementwise gain divide. No new moments, no new pass over the
voltages, and — importantly for the hot path — **no host
synchronisation**: nothing in :meth:`ArrayBurstDetector.detect` calls
``.item()``, ``.any()`` or ``bool()`` on a device tensor. Measured
fleet headroom is ~7 ms at p50 against the 134.218 ms block period,
so this matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import torch

LOG = logging.getLogger("dsart.rfi.array_burst")


# ---------------------------------------------------------------------------
# Scales and defaults
# ---------------------------------------------------------------------------

#: Accumulation depth the detector runs on, in native voltage samples.
#: 64 × 32.768 µs = 2.097 ms. Must be present in the flagger's
#: ``m_values`` (the production config passes ``64,256,1024,4096``).
M_FINE_DEFAULT: Final[int] = 64

#: Channels per coarse decision bin. 16 × 30.5 kHz = 0.488 MHz, giving
#: 24 bins per 384-channel chgroup. See the module docstring for why.
BIN_CHANS_DEFAULT: Final[int] = 16

#: Sigma threshold on the flag group's band-summed power.
DETECT_K_DEFAULT: Final[float] = 6.0

#: Sigma threshold a coarse bin must clear to count as "occupied".
BIN_K_DEFAULT: Final[float] = 2.0

#: Fraction of coarse bins that must be occupied for a sample to fire.
#:
#: Set from measurement, not from the analytic sweep width. Injecting
#: dispersed pulses into a real dump on n03 (2026-09-09) gives a
#: MAXIMUM occupancy of 0.75 at DM 100, 0.46 at DM 300 and 0.29 at
#: DM 1000, against 1.00 for a dispersion-free burst. Those ceilings
#: RISE with brightness, so 0.75 was not safe: a DM-100 pulse at
#: +300% band power reached exactly 0.75 and fired.
OCCUPANCY_MIN_DEFAULT: Final[float] = 0.90

#: EMA time constant for the baseline, in cubes. 224 cubes ≈ 30 s,
#: matching :data:`dsart.common.constants.RFI_BANDPASS_WARMUP_CUBES_DEFAULT`
#: so the two slow references age at the same rate.
EMA_CUBES_DEFAULT: Final[int] = 224

#: Cubes to spend seeding the baseline before any sample may fire.
WARMUP_CUBES_DEFAULT: Final[int] = 32

#: An antenna-pol whose band-mean gain is below this fraction of the
#: array median is treated as dead and dropped from every group.
DEAD_FRAC_DEFAULT: Final[float] = 0.05

#: Half-width, in metres, for "is this core antenna on the E-W line".
#: The real geometry has zero spread, so anything in 5..50 m works.
ARM_TOL_M_DEFAULT: Final[float] = 20.0

#: Highest station number counted as core (matches
#: :mod:`dsart.grid.sparsity_pattern`; 103-116 are the outriggers).
CORE_STATION_MAX: Final[int] = 102

#: Canonical group order. ``core`` is the default flag group: it has
#: the best median significance and excludes the outriggers, which are
#: fewer and noisier.
GROUP_NAMES: Final[tuple[str, ...]] = (
    "all", "core", "ew_arm", "ns_arm", "outriggers",
)

#: 1 / Φ⁻¹(3/4); converts a median-absolute-deviation to a Gaussian σ.
MAD_TO_SIGMA: Final[float] = 1.4826

#: Relative floor on the per-channel gain reference, as a fraction of
#: the antenna-pol's own band-mean gain. Keeps a rolled-off or sick
#: channel from turning into a huge normalisation factor.
_REF_FLOOR_FRAC: Final[float] = 1e-3


# ---------------------------------------------------------------------------
# Antenna grouping
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AntennaGroups:
    """Static membership of each antenna in each summing group.

    Args:
        names: group names, in column order of ``member``.
        member: ``[NANTS, NGROUP]`` float32, 1.0 for membership.
            Deliberately float (not bool) so it drops straight into
            the ``einsum`` without a cast on the hot path.
        sizes: ``[NGROUP]`` float32 nominal membership counts (before
            the live-antenna cut, which is applied per cube).
    """

    names: tuple[str, ...]
    member: torch.Tensor
    sizes: torch.Tensor

    def to(self, device: torch.device | str) -> "AntennaGroups":
        return AntennaGroups(
            names=self.names,
            member=self.member.to(device),
            sizes=self.sizes.to(device),
        )

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(
                f"unknown group {name!r}; have {self.names}"
            ) from None

    @property
    def n_group(self) -> int:
        return len(self.names)


def build_groups_from_antpos(
    antpos_e: Sequence[float] | np.ndarray,
    antpos_n: Sequence[float] | np.ndarray,
    station_numbers: Sequence[int] | np.ndarray | None = None,
    *,
    arm_tol_m: float = ARM_TOL_M_DEFAULT,
    core_station_max: int = CORE_STATION_MAX,
    device: torch.device | str = "cpu",
) -> AntennaGroups:
    """Split the array into ``all`` / ``core`` / ``ew_arm`` / ``ns_arm``
    / ``outriggers`` from the (E, N) positions the cal blob carries.

    Args:
        antpos_e, antpos_n: ``[NANTS]`` local East / North offsets in
            metres, in fada cube order. ``corr_fast_integration``
            already loads both from the beamformer-weights blob via
            :func:`load_antpos_from_cal_blob`, so no new data file is
            needed.
        station_numbers: ``[NANTS]`` DSA-110 station numbers, also in
            cube order (from the sibling cal yaml's ``antenna_order``).
            When ``None`` the core is identified geometrically instead
            — every antenna within ``arm_tol_m`` of either arm axis —
            which is less reliable and is logged as such.
        arm_tol_m: half-width for "on the E-W line". The real DSA-110
            arms have zero coordinate spread, so this is not delicate.
        core_station_max: highest station number counted as core.

    Returns:
        :class:`AntennaGroups` on ``device``.

    Raises:
        ValueError: mismatched lengths, or a degenerate split (an
            empty arm) that would make the arm comparison meaningless.
    """
    e = np.asarray(antpos_e, dtype=np.float64).ravel()
    n = np.asarray(antpos_n, dtype=np.float64).ravel()
    if e.shape != n.shape:
        raise ValueError(
            f"antpos_e {e.shape} and antpos_n {n.shape} differ"
        )
    n_ant = int(e.shape[0])

    if station_numbers is not None:
        st = np.asarray(station_numbers).ravel()
        if st.shape[0] != n_ant:
            raise ValueError(
                f"station_numbers has {st.shape[0]} entries, "
                f"expected {n_ant}"
            )
        core = st <= int(core_station_max)
    else:
        # Geometric fallback: the outriggers sit far off both arm
        # axes. Less reliable than station numbers — the cal yaml is
        # the authority — so say so loudly.
        n_ref_all = float(np.median(n))
        e_ref_all = float(np.median(e))
        core = (np.abs(n - n_ref_all) <= arm_tol_m) | (
            np.abs(e - e_ref_all) <= arm_tol_m
        )
        LOG.warning(
            "build_groups_from_antpos: no station numbers supplied; "
            "falling back to a geometric core mask (%d/%d core). "
            "Pass the cal yaml's antenna_order for the canonical split.",
            int(core.sum()), n_ant,
        )

    if not core.any():
        raise ValueError("core group is empty; check antpos / stations")

    # Within the core the E-W arm is the set at (essentially exactly)
    # the median North coordinate; everything else in the core is the
    # N-S arm. Reference off the CORE only — the outriggers would drag
    # a whole-array median.
    n_ref = float(np.median(n[core]))
    ew = core & (np.abs(n - n_ref) <= arm_tol_m)
    ns = core & ~ew

    if not ew.any() or not ns.any():
        raise ValueError(
            f"degenerate arm split at arm_tol_m={arm_tol_m}: "
            f"ew={int(ew.sum())} ns={int(ns.sum())} core={int(core.sum())}. "
            "The E-W / N-S comparison needs both arms populated."
        )

    masks = {
        "all": np.ones(n_ant, dtype=bool),
        "core": core,
        "ew_arm": ew,
        "ns_arm": ns,
        "outriggers": ~core,
    }
    member = np.stack([masks[g] for g in GROUP_NAMES], axis=1)
    sizes = member.sum(axis=0)

    LOG.info(
        "array-burst groups: %s (N_ref=%.1f m, arm_tol=%.0f m); "
        "EW arm spans E %.0f..%.0f m, NS arm spans N %.0f..%.0f m",
        ", ".join(
            f"{g}={int(s)}" for g, s in zip(GROUP_NAMES, sizes)
        ),
        n_ref, arm_tol_m,
        float(e[ew].min()), float(e[ew].max()),
        float(n[ns].min()), float(n[ns].max()),
    )

    return AntennaGroups(
        names=GROUP_NAMES,
        member=torch.as_tensor(
            member.astype(np.float32), device=device,
        ),
        sizes=torch.as_tensor(
            sizes.astype(np.float32), device=device,
        ),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArrayBurstResult:
    """One cube's worth of array-burst state.

    Every tensor stays on the input device; nothing here has been
    synchronised to the host.

    Args:
        fired: ``[n_acc, NGROUP, NPOL]`` bool — the gate's verdict per
            2.097 ms sample, for every group (not just the flag group,
            so the monitor can show what each arm thought).
        time_chan_mask: ``[n_acc, NCHAN, NPOL]`` bool — the flag
            group's verdict broadcast across the sub-band, ready to
            zero-fill. ``None`` while warming up.
        z: ``[n_acc, NGROUP, NPOL]`` fp32 — band-summed significance.
        band_frac: ``[n_acc, NGROUP, NPOL]`` fp32 — fractional excess
            over the group's own baseline. This is what the E-W / N-S
            comparison is read from.
        occupancy: ``[n_acc, NGROUP, NPOL]`` fp32 in 0..1 — fraction
            of coarse bins above ``bin_k``.
        coarse_z: ``[n_acc, NGROUP, NBIN, NPOL]`` fp32 — per-bin
            significance, so the page can show *where* in the band.
        group_spec: ``[NGROUP, NCHAN, NPOL]`` fp32 — cube-mean
            gain-normalised spectrum per group (≈ 1 in quiet data).
        n_live: ``[NGROUP, NPOL]`` fp32 — antennas actually summed
            after the dead-antenna cut.
        warmup: True while the baseline is still being seeded; no
            sample may fire.
    """

    fired: torch.Tensor
    time_chan_mask: torch.Tensor | None
    z: torch.Tensor
    band_frac: torch.Tensor
    occupancy: torch.Tensor
    coarse_z: torch.Tensor
    group_spec: torch.Tensor
    n_live: torch.Tensor
    warmup: bool


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ArrayBurstDetector:
    """Stateful per-cube detector for array-common broadband bursts.

    Construct once at pipeline start and call :meth:`detect` per cube.
    The state is three exponential moving averages (per-antenna gain,
    per-group baseline, per-bin baseline) plus a cube counter.

    Args:
        groups: :class:`AntennaGroups` for this array.
        n_chan: channels per chgroup (384 in production).
        n_pol: polarisations (2).
        flag_group: which group's verdict drives ``time_chan_mask``.
            Default ``"core"``.
        detect_k: sigma threshold on the band-summed power.
        bin_k: sigma threshold for a coarse bin to count as occupied.
        occupancy_min: fraction of bins that must be occupied.
        bin_chans: channels per coarse bin; must divide ``n_chan``.
        ema_cubes: baseline EMA time constant in cubes.
        warmup_cubes: cubes spent seeding before firing is allowed.
        dead_frac: antenna-pols below this fraction of the median
            band gain are dropped from every group.
        device, dtype: where the state lives.

    Raises:
        ValueError: ``bin_chans`` does not divide ``n_chan``, or a
            threshold is out of range.
    """

    def __init__(
        self,
        groups: AntennaGroups,
        *,
        n_chan: int,
        n_pol: int,
        flag_group: str = "core",
        detect_k: float = DETECT_K_DEFAULT,
        bin_k: float = BIN_K_DEFAULT,
        occupancy_min: float = OCCUPANCY_MIN_DEFAULT,
        bin_chans: int = BIN_CHANS_DEFAULT,
        ema_cubes: int = EMA_CUBES_DEFAULT,
        warmup_cubes: int = WARMUP_CUBES_DEFAULT,
        dead_frac: float = DEAD_FRAC_DEFAULT,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if bin_chans <= 0 or n_chan % bin_chans != 0:
            raise ValueError(
                f"bin_chans={bin_chans} must be a positive divisor of "
                f"n_chan={n_chan}"
            )
        if not 0.0 <= occupancy_min <= 1.0:
            raise ValueError(
                f"occupancy_min={occupancy_min}, expected 0..1"
            )
        if ema_cubes < 1:
            raise ValueError(f"ema_cubes={ema_cubes}, expected >= 1")
        if warmup_cubes < 0:
            raise ValueError(f"warmup_cubes={warmup_cubes}, expected >= 0")

        dev = torch.device(device)
        self._groups = groups.to(dev)
        self._flag_group = str(flag_group)
        self._flag_idx = self._groups.index(self._flag_group)
        self._n_chan = int(n_chan)
        self._n_pol = int(n_pol)
        self._bin_chans = int(bin_chans)
        self._n_bin = self._n_chan // self._bin_chans
        self._detect_k = float(detect_k)
        self._bin_k = float(bin_k)
        self._occupancy_min = float(occupancy_min)
        self._alpha = 1.0 / float(ema_cubes)
        self._warmup_cubes = int(warmup_cubes)
        self._dead_frac = float(dead_frac)
        self._device = dev
        self._dtype = dtype

        n_group = self._groups.n_group
        # EMA state. `_gain` is per-(ant, ch, pol); the two baselines
        # are per-group. All lazily seeded on the first cube so the
        # antenna count is taken from the data rather than assumed.
        self._gain: torch.Tensor | None = None
        self._mu = torch.zeros(
            (n_group, n_pol), dtype=dtype, device=dev,
        )
        self._sd = torch.ones(
            (n_group, n_pol), dtype=dtype, device=dev,
        )
        self._mu_bin = torch.zeros(
            (n_group, self._n_bin, n_pol), dtype=dtype, device=dev,
        )
        self._sd_bin = torch.ones(
            (n_group, self._n_bin, n_pol), dtype=dtype, device=dev,
        )
        self._cubes_seen = 0

    # -- introspection --------------------------------------------------

    @property
    def groups(self) -> AntennaGroups:
        return self._groups

    @property
    def group_names(self) -> tuple[str, ...]:
        return self._groups.names

    @property
    def flag_group(self) -> str:
        return self._flag_group

    @property
    def n_bin(self) -> int:
        return self._n_bin

    @property
    def cubes_seen(self) -> int:
        return self._cubes_seen

    @property
    def in_warmup(self) -> bool:
        return self._cubes_seen < self._warmup_cubes

    def reset(self) -> None:
        """Drop the baselines and re-arm the warmup window."""
        self._gain = None
        self._mu.zero_()
        self._sd.fill_(1.0)
        self._mu_bin.zero_()
        self._sd_bin.fill_(1.0)
        self._cubes_seen = 0

    # -- the hot path ---------------------------------------------------

    def detect(
        self,
        s1_fine: torch.Tensor,
        s1_full: torch.Tensor,
    ) -> ArrayBurstResult:
        """Run the detector on one cube.

        Args:
            s1_fine: ``[n_acc, NANTS, NCHAN, NPOL]`` fp32 — the
                ``M = M_FINE`` entry of :class:`dsart.rfi.AutoSpectra`
                (``autos.s1[64]``), i.e. Σ|E|² per 2.097 ms.
            s1_full: ``[NANTS, NCHAN, NPOL]`` fp32 — the whole-cube
                auto-power (``autos.s1[4096].squeeze(0)``), used as the
                per-antenna gain reference. Must cover exactly the
                same cube, so ``s1_full ≈ s1_fine.sum(0)``.

        Returns:
            :class:`ArrayBurstResult`. During warmup ``fired`` is all
            False and ``time_chan_mask`` is ``None``.

        Raises:
            ValueError: shape mismatch between the two inputs, or
                against the configured channel / pol counts.
        """
        if s1_fine.ndim != 4:
            raise ValueError(
                f"s1_fine must be [n_acc, NANTS, NCHAN, NPOL]; "
                f"got {tuple(s1_fine.shape)}"
            )
        n_acc, n_ant, n_ch, n_pol = s1_fine.shape
        if (n_ch, n_pol) != (self._n_chan, self._n_pol):
            raise ValueError(
                f"s1_fine has (NCHAN, NPOL) = ({n_ch}, {n_pol}); "
                f"detector configured for "
                f"({self._n_chan}, {self._n_pol})"
            )
        if tuple(s1_full.shape) != (n_ant, n_ch, n_pol):
            raise ValueError(
                f"s1_full shape {tuple(s1_full.shape)} != "
                f"{(n_ant, n_ch, n_pol)}"
            )
        if self._groups.member.shape[0] != n_ant:
            raise ValueError(
                f"groups cover {self._groups.member.shape[0]} antennas, "
                f"cube has {n_ant}"
            )

        f32 = self._dtype
        s1f = s1_fine.to(f32)
        s1c = s1_full.to(f32)

        # --- 1. per-antenna gain reference (EMA over cubes) ----------
        # s1_full is the sum over the whole cube; the per-accumulation
        # expectation is that divided by the number of accumulations.
        #
        # THIS CUBE IS NOT FOLDED IN YET. The fold happens at the end
        # of detect(), so the normalisation below divides by the
        # array's history and not by a reference this cube has already
        # lifted. A cube-filling burst would otherwise be scaled down
        # by a factor (1 - alpha) — 0.45% at the default 224-cube time
        # constant, but 50% at ema_cubes=2, and the same trap as the
        # baseline EMA in step 5.
        if self._gain is None:
            self._gain = s1c.clone()          # cube 0 is its own
                                              # reference; warmup
                                              # suppresses firing.
        gain = self._gain

        # Dead-antenna cut. An antenna-pol whose band-mean gain is a
        # negligible fraction of the array median contributes nothing
        # but noise-free zeros, which biases both the mean and the
        # variance of the sum (offline: all-96 null 0.978 ± 0.026 vs
        # 1.000 ± 0.0013 when they are left in).
        band_gain = gain.mean(dim=1)                      # [NANTS, NPOL]
        med_gain = band_gain.median(dim=0, keepdim=True).values
        live = (band_gain > self._dead_frac * med_gain).to(f32)

        # Per-accumulation gain reference. Floor it against the
        # antenna-pol's OWN band mean rather than an absolute epsilon:
        # a relative floor cannot be tripped by a change of scale, and
        # it keeps one rolled-off channel from becoming a 10**6 gain.
        # The floor must itself be strictly positive: a dead antenna
        # has band_gain == 0, so a purely relative floor would leave
        # ref == 0 and turn the live-mask division into 0/0 = NaN,
        # which then poisons every group the antenna belongs to.
        ref_floor = torch.clamp(
            _REF_FLOOR_FRAC * band_gain.unsqueeze(1) / float(n_acc),
            min=torch.finfo(f32).tiny,
        )
        ref = torch.maximum(gain / float(n_acc), ref_floor)

        # --- 2. gain-normalised power, ~1 in quiet data -------------
        # One elementwise pass: the reciprocal reference already
        # carries the dead-antenna cut (live=0 -> inv=0), so dead
        # antennas contribute exact zeros to every sum downstream.
        inv = live.unsqueeze(1) / ref                     # [A, C, P]
        x = s1f * inv.unsqueeze(0)                        # [T, A, C, P]

        # --- 3. group weights, normalised to a MEAN over live ants --
        # w[a, g, p] = member[a, g] * live[a, p] / n_live[g, p]
        w = self._groups.member.unsqueeze(2) * live.unsqueeze(1)
        n_live = w.sum(dim=0)                             # [G, P]
        w = w / torch.clamp(n_live, min=1.0).unsqueeze(0)

        # --- 4. coarse-bin and band sums ----------------------------
        y = x.reshape(
            n_acc, n_ant, self._n_bin, self._bin_chans, n_pol,
        ).mean(dim=3)                                     # [T, A, B, P]
        p_bin = torch.einsum("tabp,agp->tgbp", y, w)      # [T, G, B, P]
        p_band = p_bin.mean(dim=2)                        # [T, G, P]

        # --- 5. robust in-cube location / scale, EMA'd over cubes ---
        # Read the baseline BEFORE folding this cube in, so a burst can
        # never inflate the reference it is measured against.
        mu, sd = self._mu, self._sd
        mu_bin, sd_bin = self._mu_bin, self._sd_bin
        seeding = self._cubes_seen == 0

        med = p_band.median(dim=0).values                 # [G, P]
        mad = (p_band - med).abs().median(dim=0).values * MAD_TO_SIGMA
        med_b = p_bin.median(dim=0).values                # [G, B, P]
        mad_b = (p_bin - med_b).abs().median(dim=0).values * MAD_TO_SIGMA

        # A single-accumulation cube (tests) has zero MAD; floor the
        # scale relative to the level so z stays finite.
        eps = torch.finfo(f32).eps
        mad = torch.clamp(mad, min=eps * med.abs() + torch.finfo(f32).tiny)
        mad_b = torch.clamp(
            mad_b, min=eps * med_b.abs() + torch.finfo(f32).tiny,
        )

        if seeding:
            # Nothing to compare against yet; the cube is its own
            # reference. Firing is suppressed during warmup anyway.
            mu, sd = med, mad
            mu_bin, sd_bin = med_b, mad_b

        # z is computed against the PRE-UPDATE baseline. The EMA fold
        # below is in-place, so doing it first would let a burst move
        # the very reference it is then measured against — a 0.45%
        # error at the default 224-cube time constant, but a 50% one
        # if anybody shortens it.
        z = (p_band - mu) / sd                            # [T, G, P]
        band_frac = p_band / torch.clamp(
            mu.abs(), min=torch.finfo(f32).tiny,
        ) - 1.0
        coarse_z = (p_bin - mu_bin) / sd_bin              # [T, G, B, P]

        # Everything above has materialised its own tensors, so the
        # in-place folds below cannot affect this cube's answer.
        if self._cubes_seen > 0:
            self._gain.mul_(1.0 - self._alpha).add_(s1c, alpha=self._alpha)

        if seeding:
            self._mu = med.clone()
            self._sd = mad.clone()
            self._mu_bin = med_b.clone()
            self._sd_bin = mad_b.clone()
        else:
            self._mu.mul_(1.0 - self._alpha).add_(med, alpha=self._alpha)
            self._sd.mul_(1.0 - self._alpha).add_(mad, alpha=self._alpha)
            self._mu_bin.mul_(1.0 - self._alpha).add_(
                med_b, alpha=self._alpha,
            )
            self._sd_bin.mul_(1.0 - self._alpha).add_(
                mad_b, alpha=self._alpha,
            )

        # --- 6. the gate --------------------------------------------
        occupancy = (coarse_z > self._bin_k).to(f32).mean(dim=2)
        fired = (z > self._detect_k) & (occupancy >= self._occupancy_min)

        warmup = self.in_warmup
        if warmup:
            fired = torch.zeros_like(fired)
            time_chan_mask = None
        else:
            # The flag group's verdict, broadcast across the sub-band.
            # Broadband by construction — the occupancy term is what
            # licenses flagging the whole band rather than part of it.
            time_chan_mask = (
                fired[:, self._flag_idx, :]
                .unsqueeze(1)
                .expand(n_acc, self._n_chan, n_pol)
                .contiguous()
            )

        # --- 7. monitor products ------------------------------------
        # group_spec is the time-mean of the normalised power. Writing
        # that out:
        #
        #     mean_t(x) = mean_t(s1_fine) * inv = (s1_full / n_acc) * inv
        #
        # so it comes straight from s1_full, which is [A, C, P] — 64x
        # smaller than reducing [T, A, C, P] over time. Identical
        # answer, one fewer full pass over the fine moments. Measured
        # on a 2080 Ti this is worth ~0.4 ms of the detector's budget,
        # which matters when the block has ~7 ms of headroom.
        group_spec = torch.einsum(
            "acp,agp->gcp", s1c * inv / float(n_acc), w,
        )                                                 # [G, C, P]

        self._cubes_seen += 1

        return ArrayBurstResult(
            fired=fired,
            time_chan_mask=time_chan_mask,
            z=z,
            band_frac=band_frac,
            occupancy=occupancy,
            coarse_z=coarse_z,
            group_spec=group_spec,
            n_live=n_live,
            warmup=warmup,
        )
