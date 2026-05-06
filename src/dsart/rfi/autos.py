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
and bfCorr's `corr_input_copy`). We reshape the GEMM-layout tensor's
``(NPACKETS_PER_BLOCK, NTIMES_PER_PACKET)`` time pair into a single
contiguous ``t_native`` axis so the M-accumulation chunks align with
real time intervals; otherwise the SK estimator's per-accumulation
statistics would mix non-contiguous samples and over-estimate FAR on
impulsive RFI.

Per :ref:`§3 line 302`, the ``S₁_M`` / ``S₂_M`` contract is float32
``[NANTS, NCHAN, NPOL]`` *per accumulation*; this module returns the
``N_acc``-stacked form so downstream SK gets per-accumulation samples
to OR-fold. For ``M = 4096`` the leading axis collapses to length 1.

Performance (informational; plan §4.2 step 1 budget): ~1-2 ms / 134 ms
cube on a Turing 2080 Ti. Memory peak ~2.4 GB fp32 working set
(``pwr`` + ``pwr2`` reshapes); fits in the 11 GB VRAM budget alongside
the M2 GEMM working tensors. Not optimised here; correctness first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch

from dsart.common.constants import NANTS, NCHAN_PER_CHGROUP, NPOL


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
    expected = (
        NCHAN_PER_CHGROUP, n_times_per_packet, NPOL, n_packets, NANTS,
    )
    if tuple(real.shape) != expected:
        raise ValueError(
            f"real shape {tuple(real.shape)} != expected {expected} "
            "(GEMM layout from unpack_int4_split)"
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
            ``n_packets * n_times_per_packet`` exactly. Defaults to
            :data:`DEFAULT_M_VALUES`.
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
        ValueError: shape / divisibility mismatch.
        TypeError: dtype / device mismatch.
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

    # ---- Cast to fp32 BEFORE squaring (avoids fp16 |E|⁴ overflow at
    #      large M and keeps the running sums numerically clean). ----
    real32 = real.to(torch.float32)
    imag32 = imag.to(torch.float32)

    pwr = real32 * real32 + imag32 * imag32  # |E|² fp32
    del real32, imag32

    # ---- Permute (NCHAN, 2t, NPOL, NPACKETS, NANTS)
    #   → (NANTS, NCHAN, NPOL, NPACKETS, 2t) so the trailing (NPACKETS,
    #   2t) pair flattens into the natural ``t_native = pkt*2 + t_sub``
    #   ordering. This is the in-cube time order; required so each
    #   M-chunk corresponds to a contiguous time interval.
    pwr = pwr.permute(4, 0, 2, 3, 1).contiguous()  # (NANTS, NCHAN, NPOL, NPACKETS, 2t)
    pwr_flat = pwr.reshape(NANTS, NCHAN_PER_CHGROUP, NPOL, total_t)
    del pwr

    # |E|⁴ = (|E|²)² — separate buffer so we can sum independently.
    pwr2_flat = pwr_flat * pwr_flat

    s1: dict[int, torch.Tensor] = {}
    s2: dict[int, torch.Tensor] = {}
    for m in m_values:
        n_acc = total_t // m
        s1_m = pwr_flat.reshape(
            NANTS, NCHAN_PER_CHGROUP, NPOL, n_acc, m,
        ).sum(dim=-1)  # (NANTS, NCHAN, NPOL, n_acc) fp32
        s2_m = pwr2_flat.reshape(
            NANTS, NCHAN_PER_CHGROUP, NPOL, n_acc, m,
        ).sum(dim=-1)
        # Move N_acc to the leading axis for downstream consumers
        # (SK iterates per-accumulation along leading dim).
        s1[m] = s1_m.permute(3, 0, 1, 2).contiguous()
        s2[m] = s2_m.permute(3, 0, 1, 2).contiguous()

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
            f"voltages must be 4-dim (NANTS, NCHAN, NPOL, NTIME); "
            f"got shape {tuple(voltages.shape)}"
        )
    if not voltages.is_complex():
        raise TypeError(
            f"voltages must be complex; got dtype {voltages.dtype}"
        )
    n_ant, n_ch, n_pol, n_time = voltages.shape
    # NPOL is hard-coded to 2 throughout downstream RFI / Stokes-I pol
    # collapse; n_ant is left unconstrained so synthetic tests can use
    # smaller antenna counts (production callers always pass NANTS=96).
    if n_pol != NPOL:
        raise ValueError(
            f"voltages shape {tuple(voltages.shape)}: NPOL axis must "
            f"be {NPOL}, got {n_pol}"
        )
    if n_ant <= 0:
        raise ValueError(
            f"voltages shape {tuple(voltages.shape)}: NANTS axis must "
            f"be positive, got {n_ant}"
        )

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
