"""Voltage-domain auto-power accumulator (M3 chunk 3c; plan §4.2 step 1).

Consumes voltage tensors from
:func:`dsart.services.slow_corr_kernel.unpack_int4_split` (M2 GEMM
layout ``(NCHAN, NTIMES_PER_PACKET, NPOL, NPACKETS_PER_BLOCK, NANTS)
fp16``) and emits the auto-power moments

* ``S₁_M[N_acc, NANTS, NCHAN, NPOL] = Σ_{t∈M} |E|²``
* ``S₂_M[N_acc, NANTS, NCHAN, NPOL] = Σ_{t∈M} |E|⁴``

for each accumulation depth ``M`` in
``DEFAULT_M_VALUES = (64, 256, 1024, 4096)`` voltage time samples per
accumulation. ``N_acc = TOTAL_NATIVE_T / M`` (e.g. 64 / 16 / 4 / 1 at
the canonical ``TOTAL_NATIVE_T = 4096``).

The on-cube native time axis runs as ``t_native = pkt * NTIMES_PER_PACKET
+ t_sub`` (mirrors the byte-layout convention of the SNAP voltage stream
and bfCorr's `corr_input_copy`). The M-accumulation windows are tiled
inside the ``(NPACKETS, NTIMES_PER_PACKET)`` axis pair so each chunk
spans a contiguous time interval; the SK estimator's per-accumulation
statistics would over-estimate FAR on impulsive RFI if we mixed
non-contiguous samples.

Per :ref:`§3 line 302`, the ``S₁_M`` / ``S₂_M`` contract is float32
``[NANTS, NCHAN, NPOL]`` *per accumulation*; this module returns the
``N_acc``-stacked form so downstream SK gets per-accumulation samples
to OR-fold. For ``M = 4096`` the leading axis collapses to length 1.

Performance (M7.1 op-point soak):
    * baseline (pre-RT-Phase-15): ~113 ms / 134 ms cube on a 2080 Ti
      (n06). Dominated by a 66 ms permute on a 1.2 GB fp32 working
      tensor + 4 redundant per-M reductions on a 600 MB ``pwr`` tensor.
    * RT-Phase-15 (current): ~4 ms / 134 ms cube on a 2080 Ti. We
      compute the base ``M = m_base`` reduction directly on the GEMM
      layout (no 1.2 GB permute), then derive S1/S2 at coarser M's by
      hierarchically summing pairs of base-M entries (``S1[M] =
      Σ_{j} S1_base[k·j..k·(j+1)]`` for ``k = M / m_base``). The full
      base reduction is wrapped in :func:`torch.compile` (mode
      ``default``) which fuses cast + |E|² + reduction into a couple
      of Inductor kernels.

Memory: peak ~600 MB fp32 (one ``pwr`` allocation; ``pwr2`` is folded
inside the reduction). The compile cache is keyed by ``(shape, dtype,
device, m_base, n_pkts, t_per_pkt)`` so test-time shape variations
do not collide with the production canonical-shape cache entry.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final, Sequence

import torch

from dsart.common.constants import NANTS, NCHAN_PER_CHGROUP, NPOL

LOG = logging.getLogger(__name__)

#: Set ``DSART_RFI_AUTOS_COMPILE=0`` to disable :func:`torch.compile`
#: on the autos base reduction (eager fallback). Useful for tests on
#: hosts without an Inductor-compatible CUDA toolchain.
_AUTOS_COMPILE_ENABLED: Final[bool] = (
    os.environ.get("DSART_RFI_AUTOS_COMPILE", "1") != "0"
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Total native time samples per cube (PSRDADA block) at the canonical
#: voltage layout: ``NPACKETS_PER_BLOCK * NTIMES_PER_PACKET = 2048 * 2``.
#: Re-stated locally instead of pulling
#: :data:`dsart.services.slow_corr_kernel.N_TIME_SAMPLES` to avoid a
#: dependency on the slow-corr module from the RFI substrate (the GEMM
#: layout shape is the only contract that crosses this boundary).
TOTAL_NATIVE_T: Final[int] = 4096

#: Default accumulation depths for the SK / autos kernel. Locked by
#: plan §4.2 step 1: ``M ∈ {64, 256, 1024, 4096}`` voltage time samples
#: per accumulation. All four divide :data:`TOTAL_NATIVE_T = 4096`.
DEFAULT_M_VALUES: Final[tuple[int, ...]] = (64, 256, 1024, 4096)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoSpectra:
    """Per-cube auto-power moments at multiple accumulation depths.

    Args:
        s1: dict mapping ``M`` → fp32 tensor of shape
            ``[N_acc(M), NANTS, NCHAN, NPOL]`` carrying ``Σ_t |E|²`` per
            accumulation.
        s2: dict mapping ``M`` → fp32 tensor of same shape carrying
            ``Σ_t |E|⁴`` per accumulation. Same key set as ``s1``.

    All tensors share the device of the input voltages and are
    C-contiguous. ``s1`` / ``s2`` are independently allocated (do not
    alias).
    """

    s1: dict[int, torch.Tensor]
    s2: dict[int, torch.Tensor]


# ---------------------------------------------------------------------------
# Shape validation (cheap; runs only when caller passes ``check=True`` —
# the production hot path skips it)
# ---------------------------------------------------------------------------


def _check_voltage_layout(
    real: torch.Tensor,
    imag: torch.Tensor,
    n_packets: int,
    n_times_per_packet: int,
) -> None:
    if real.shape != imag.shape:
        raise ValueError(
            f"real shape {tuple(real.shape)} != imag shape {tuple(imag.shape)}"
        )
    if real.ndim != 5:
        raise ValueError(
            f"real must be 5-dim (NCHAN, 2t, NPOL, NPACKETS, NANTS); "
            f"got shape {tuple(real.shape)}"
        )
    n_ch, n_t_per_pkt, n_pol, n_pkts, n_ant = real.shape
    # Pin the time / pol axes (these are fixed by SNAP firmware + downstream
    # Stokes-I sums); leave NCHAN / NANTS unconstrained so synthetic tests
    # can pass smaller cubes (production callers always feed
    # NCHAN_PER_CHGROUP=384 + NANTS=96).
    if n_t_per_pkt != n_times_per_packet:
        raise ValueError(
            f"real shape {tuple(real.shape)}: NTIMES_PER_PACKET axis "
            f"must be {n_times_per_packet}, got {n_t_per_pkt}"
        )
    if n_pol != NPOL:
        raise ValueError(
            f"real shape {tuple(real.shape)}: NPOL axis must be "
            f"{NPOL}, got {n_pol}"
        )
    if n_pkts != n_packets:
        raise ValueError(
            f"real shape {tuple(real.shape)}: NPACKETS axis must be "
            f"{n_packets}, got {n_pkts}"
        )
    if n_ch <= 0 or n_ant <= 0:
        raise ValueError(
            f"real shape {tuple(real.shape)}: NCHAN / NANTS axes must "
            f"be positive, got NCHAN={n_ch}, NANTS={n_ant}"
        )
    if real.dtype != imag.dtype:
        raise TypeError(
            f"real dtype {real.dtype} != imag dtype {imag.dtype}"
        )
    if real.device != imag.device:
        raise ValueError(
            f"real device {real.device} != imag device {imag.device}"
        )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _autos_base_reduction_eager(
    real: torch.Tensor,
    imag: torch.Tensor,
    n_ch: int,
    t_per_pkt: int,
    n_pol: int,
    n_acc_base: int,
    pkts_per_acc_base: int,
    n_ant: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Base-M reduction directly on the GEMM layout (no big permute).

    Casts the input voltages to fp32, computes ``|E|²``, then sums over
    the inner-time window ``(t_per_pkt, pkts_per_acc_base)`` to produce
    the base-M S1/S2. Output layout is canonical
    ``(n_acc_base, NANTS, NCHAN, NPOL)`` fp32.

    The permute that lands the output in canonical order operates on a
    *small* tensor (n_acc · ant · ch · pol = O(MB)), unlike the pre-RT-
    Phase-15 implementation which permuted the full ``pwr`` working
    tensor (O(GB)) before reducing.
    """
    # Cast to fp32 before squaring (avoids fp16 |E|² overflow at large
    # M and keeps S2 = |E|⁴ within fp32 range — the unsquared |E|² has
    # max ~64 for 4-bit voltages, so |E|⁴ ≤ 4096 per sample, summed
    # over M ≤ 4096 fits comfortably in fp32).
    real32 = real.to(torch.float32)
    imag32 = imag.to(torch.float32)
    pwr = real32 * real32 + imag32 * imag32           # (NCHAN, 2t, NPOL, NPACKETS, NANTS)

    # Reshape NPACKETS axis into (n_acc_base, pkts_per_acc_base), then
    # fold (t_per_pkt, pkts_per_acc_base) into a single base-M window
    # via two sequential reductions.
    pwr_r = pwr.reshape(
        n_ch, t_per_pkt, n_pol, n_acc_base, pkts_per_acc_base, n_ant,
    )
    s1_base_layout = pwr_r.sum(dim=4).sum(dim=1)      # (NCHAN, NPOL, n_acc_base, NANTS)

    # S2 = sum |E|⁴; compute as (|E|²)² on the same reshape (separate
    # reduction so the multiply lives in the inner kernel scratchpad
    # rather than materializing pwr2 globally).
    pwr2_r = pwr_r * pwr_r
    s2_base_layout = pwr2_r.sum(dim=4).sum(dim=1)

    # Permute the SMALL output tensors to canonical layout.
    s1_base = s1_base_layout.permute(2, 3, 0, 1).contiguous()  # (n_acc_base, NANTS, NCHAN, NPOL)
    s2_base = s2_base_layout.permute(2, 3, 0, 1).contiguous()
    return s1_base, s2_base


#: Per-shape cache of :func:`torch.compile`-wrapped base-M reductions.
#: Key: ``(real_dtype, real_shape, real_device_type, n_acc_base,
#: pkts_per_acc_base)``. Avoids recompiling on every call once a given
#: shape has been seen.
_BASE_REDUCTION_COMPILE_CACHE: "dict[tuple, callable]" = {}


def _get_base_reduction_fn(
    *,
    real: torch.Tensor,
    n_ch: int,
    t_per_pkt: int,
    n_pol: int,
    n_acc_base: int,
    pkts_per_acc_base: int,
    n_ant: int,
):
    """Return an Inductor-compiled base-M reduction for this shape.

    Falls back to the eager kernel when :envvar:`DSART_RFI_AUTOS_COMPILE`
    is ``0`` or when the input is not on a CUDA device (Inductor's CPU
    backend is slower than eager for this workload).
    """
    if (
        not _AUTOS_COMPILE_ENABLED
        or real.device.type != "cuda"
    ):
        return _autos_base_reduction_eager
    key = (
        real.dtype, tuple(real.shape), real.device.type,
        n_acc_base, pkts_per_acc_base,
    )
    fn = _BASE_REDUCTION_COMPILE_CACHE.get(key)
    if fn is None:
        # mode="default" gives best p50 on the 2080 Ti for this kernel
        # (~4 ms) and avoids the cudagraph-output-clobbering issues of
        # mode="reduce-overhead".
        fn = torch.compile(
            _autos_base_reduction_eager, mode="default", dynamic=False,
        )
        _BASE_REDUCTION_COMPILE_CACHE[key] = fn
        LOG.debug(
            "torch.compile autos base reduction: shape=%s dtype=%s "
            "n_acc_base=%d pkts_per_acc_base=%d",
            tuple(real.shape), real.dtype, n_acc_base, pkts_per_acc_base,
        )
    return fn


def compute_autos(
    real: torch.Tensor,
    imag: torch.Tensor,
    *,
    m_values: Sequence[int] = DEFAULT_M_VALUES,
    n_packets: int = 2048,
    n_times_per_packet: int = 2,
    check: bool = True,
) -> AutoSpectra:
    """Compute ``S₁_M`` / ``S₂_M`` from split-real-imag voltages.

    Args:
        real: fp16/fp32 real-component voltages in the M2 GEMM layout
            ``(NCHAN, NTIMES_PER_PACKET, NPOL, NPACKETS_PER_BLOCK,
            NANTS)``.
        imag: same shape/dtype/device as ``real`` carrying the imaginary
            component.
        m_values: accumulation depths. Each must divide
            ``n_packets * n_times_per_packet`` exactly, AND each must
            be an integer multiple of ``min(m_values)`` so that
            hierarchical sums from the base-M reduction land cleanly
            on the larger-M windows. (At the production canonical
            ``m_values=(64, 256, 1024, 4096)`` all M's are powers of
            two ≥ 64; the multiplicity constraint is automatically
            satisfied.)
        n_packets: ``NPACKETS_PER_BLOCK``. Default 2048 (canonical cube).
            Tests can pass smaller blocks for synthetic-input scenarios
            (e.g. 32 packets for FAR tests).
        n_times_per_packet: ``NTIMES_PER_PACKET``. Default 2 (SNAP
            firmware constant).
        check: validate shapes / dtypes / devices. Production hot path
            sets this to ``False``.

    Returns:
        :class:`AutoSpectra` with ``s1[M]`` and ``s2[M]`` of shape
        ``[N_acc, NANTS, NCHAN, NPOL]`` float32 for each M, with
        ``N_acc = (n_packets * n_times_per_packet) // M``.

    Raises:
        ValueError: shape / divisibility mismatch (including the
            multiple-of-``m_base`` constraint above).
        TypeError: dtype / device mismatch.

    Notes:
        Implementation strategy (RT-Phase-15; see module docstring for
        the full performance story):

        1. Sort ``m_values`` ascending and pick ``m_base = m_values[0]``.
        2. Run a single base-M reduction on the GEMM layout that
           produces ``s1_base / s2_base`` in canonical
           ``(n_acc_base, NANTS, NCHAN, NPOL)`` order.
        3. Derive every other M's S1/S2 by reshape-and-sum on the
           leading ``n_acc_base`` axis (a fold of ``M / m_base``
           adjacent base entries per output cell).

        Step 2 is wrapped in :func:`torch.compile` (Inductor, mode
        ``default``) when the input tensor is on CUDA and the env
        var :envvar:`DSART_RFI_AUTOS_COMPILE` is not set to ``0``.
        The compile cache is per-shape so test-time shape variations
        do not evict the production canonical-shape entry.
    """
    if check:
        _check_voltage_layout(real, imag, n_packets, n_times_per_packet)

    total_t = n_packets * n_times_per_packet
    for m in m_values:
        if m <= 0 or total_t % m != 0:
            raise ValueError(
                f"M={m} must be a positive divisor of total native "
                f"time samples ({total_t}); pick m_values from "
                f"divisors of TOTAL_NATIVE_T={TOTAL_NATIVE_T}."
            )
    if not m_values:
        raise ValueError("m_values must be non-empty")

    m_sorted = sorted(int(m) for m in m_values)
    m_base = m_sorted[0]
    for m in m_sorted[1:]:
        if m % m_base != 0:
            raise ValueError(
                f"m_values={m_sorted}: every M must be a multiple of "
                f"the smallest M ({m_base}) so hierarchical sums land "
                f"cleanly on the larger-M windows"
            )
    if m_base % n_times_per_packet != 0:
        raise ValueError(
            f"m_base={m_base} must be a multiple of "
            f"n_times_per_packet={n_times_per_packet} so a base-M "
            f"window can be tiled over an integer number of packets"
        )

    n_ch_in, t_per_pkt_in, n_pol_in, n_pkts_in, n_ant_in = real.shape
    n_acc_base = total_t // m_base
    pkts_per_acc_base = m_base // n_times_per_packet

    base_fn = _get_base_reduction_fn(
        real=real,
        n_ch=n_ch_in, t_per_pkt=t_per_pkt_in, n_pol=n_pol_in,
        n_acc_base=n_acc_base, pkts_per_acc_base=pkts_per_acc_base,
        n_ant=n_ant_in,
    )
    s1_base, s2_base = base_fn(
        real, imag,
        n_ch_in, t_per_pkt_in, n_pol_in,
        n_acc_base, pkts_per_acc_base, n_ant_in,
    )

    s1: dict[int, torch.Tensor] = {m_base: s1_base}
    s2: dict[int, torch.Tensor] = {m_base: s2_base}
    for m in m_sorted[1:]:
        fold = m // m_base
        n_acc = total_t // m
        # Reshape the leading n_acc_base axis into (n_acc, fold) and
        # sum over `fold`. Sum over the leading axis is cheap on the
        # small canonical-layout tensor.
        s1[m] = s1_base.reshape(
            n_acc, fold, n_ant_in, n_ch_in, n_pol_in,
        ).sum(dim=1)
        s2[m] = s2_base.reshape(
            n_acc, fold, n_ant_in, n_ch_in, n_pol_in,
        ).sum(dim=1)

    return AutoSpectra(s1=s1, s2=s2)


def compute_autos_from_complex(
    voltages: torch.Tensor,
    *,
    m_values: Sequence[int] = DEFAULT_M_VALUES,
) -> AutoSpectra:
    """Convenience entry that takes a complex ``(NANTS, NCHAN, NPOL,
    NTIME)`` tensor (instead of the M2 GEMM layout). Used by tests
    that synthesise voltages directly.

    Args:
        voltages: complex64 / complex128 / cfp16 tensor of shape
            ``(NANTS, NCHAN, NPOL, NTIME)``. ``NTIME`` is the native
            time axis (no packet × t_sub split); the caller must pass
            it in natural time order.
        m_values: see :func:`compute_autos`.

    Returns:
        :class:`AutoSpectra` with the same per-M output convention.

    Raises:
        ValueError: shape / divisibility mismatch.
        TypeError: dtype not complex.
    """
    if voltages.ndim != 4:
        raise ValueError(
            f"voltages must be 4-dim (n_ant, n_ch, n_pol, n_time); "
            f"got shape {tuple(voltages.shape)}"
        )
    if not voltages.is_complex():
        raise TypeError(
            f"voltages must be complex; got dtype {voltages.dtype}"
        )
    n_ant, n_ch, n_pol, n_time = voltages.shape
    if n_pol != NPOL:
        raise ValueError(
            f"voltages shape {tuple(voltages.shape)}: NPOL axis "
            f"must be {NPOL}, got {n_pol}"
        )
    if n_ant <= 0:
        raise ValueError(
            f"voltages shape {tuple(voltages.shape)}: NANTS axis must "
            f"be positive, got {n_ant}"
        )
    # Note: this helper accepts arbitrary n_ant / n_ch (tests use
    # smaller dimensions for speed). The GEMM-layout entry
    # ``compute_autos`` similarly accepts arbitrary spatial dimensions
    # via ``_check_voltage_layout`` so synthetic test cubes work in
    # both code paths.

    for m in m_values:
        if m <= 0 or n_time % m != 0:
            raise ValueError(
                f"M={m} must be a positive divisor of NTIME={n_time}"
            )

    pwr = (voltages.real.to(torch.float32) ** 2
           + voltages.imag.to(torch.float32) ** 2)
    pwr2 = pwr * pwr

    s1: dict[int, torch.Tensor] = {}
    s2: dict[int, torch.Tensor] = {}
    for m in m_values:
        n_acc = n_time // m
        s1_m = pwr.reshape(n_ant, n_ch, n_pol, n_acc, m).sum(dim=-1)
        s2_m = pwr2.reshape(n_ant, n_ch, n_pol, n_acc, m).sum(dim=-1)
        s1[m] = s1_m.permute(3, 0, 1, 2).contiguous()
        s2[m] = s2_m.permute(3, 0, 1, 2).contiguous()

    return AutoSpectra(s1=s1, s2=s2)
