"""Voltage-domain online injector for the corr-fast pipeline (plan §4.7).

The injector synthesises additive complex contributions to the fast-corr
voltage tensor *before* RFI flagging — so an injected fake source passes
through cal/bandpass/grid/coarse-DM end-to-end and is the gate that the
M6 corr-side acceptance pulls on. Everything is generator-side: the
operator submits a tiny ``InjectionConfig`` (a few hundred bytes JSON,
six fields), and the injector materialises the per-block per-(ant, ch,
t, pol) contribution on-the-fly from cached antpos + the config.

The three cached tables that drive ``apply_block`` are:

  1. Per-antenna phasor table for ``(l, m)``: complex64 ``[NANTS,
     NCHAN_PER_CHGROUP]`` = ``exp(+2π i · ν_ch · (E_a · l + N_a · m
     + U_a · n) / c)``. **POSITIVE sign** in the per-antenna voltage
     exponent — TMS / DSA-110 SNAP voltage convention pinned in
     ``tests/test_cal_loader_dec_phase.py`` (F21 docstring, line ~21)
     and ``tests/test_slow_corr_synth.py``
     ``test_point_source_planar_wave_phase_recovery`` (line 449). The
     downstream visibility ``V_ij = conj(E_i) · E_j`` (F18, with ``i``
     = lower antenna, ``j`` = higher) then evaluates to ``exp(+2π i ·
     ν · (r_j - r_i) · ŝ / c)`` — exactly the F18 + §8.M2-carryover
     line 2227 "DSA-110 SNAP convention V ∝ exp(+2πi · b · ŝ · ν / c)"
     expectation. Test ``test_F22_visibility_phase_matches_lm_target``
     in ``tests/test_online_injector.py`` is the strongest sign-
     convention guard.

  2. Per-channel dispersion delay table for ``dm``: float32
     ``[NCHAN_PER_CHGROUP]`` = ``K_DM_MS_GHZ2_PC · DM · (1/ν_ch² −
     1/ν_top_proc²)`` ms. Positive at lower channels (lower freqs are
     delayed; standard cold-plasma law). Converted to native-sample
     offsets at apply time so the gather indexes the profile vector by
     integer.

  3. Intrinsic profile vector: float32 ``[2 · max_width_samples + 1]``
     centred at index ``max_width_samples``. Profile family selectable
     (``"gaussian"`` | ``"boxcar"``); both unit-area-normalised over
     their support so that ``√(fluence/width)`` is the peak amplitude
     scale.

Per-block math:

  For each native sample ``(pkt, t_sub)`` with absolute native position
  ``n_abs = block_specnum_start * 2 + pkt * 2 + t_sub`` (1 specnum =
  2 native samples), and for each channel ``ch`` with dispersion
  delay ``Δ_ch`` (native samples), the contribution is

      C[ch, t_sub, pol, pkt, ant] = phasor[ant, ch]
                                  · profile[n_abs - peak_native - Δ_ch
                                            + max_width_samples]
                                  · √(fluence_jy_ms / width_samples)

  added to *both* polarisation slots (intrinsic unpolarised source).
  ``peak_native = 2 · apply_at_specnum``.

The injector is **single-pol** (the same value goes into both pol
slots). It is **single-band** (one chgroup per ``OnlineInjector``
instance — matches the §3.6.2 / §4.2 corr-node-per-chgroup deployment
where each ``corr_fast_compute`` service owns one chgroup).

Production etcd integration is layered on top via
``src/dsart/inject/etcd_watcher.py`` (separate module so the injector
itself stays etcd-agnostic and unit-testable). The watcher parses
``/cmd/corr/<n>`` payloads into ``InjectionConfig`` and calls
``add_pending``.

Sign-convention F-item (proposal listed in the chunk-3d final report,
*not* added to ``M3_PLAN_FIXES.md`` per chunk-3d ownership rules): the
parent agent's chunk-3d briefing specifies the per-antenna phasor with
a NEGATIVE sign in the exponent (``exp(-2π i · ...)``); after
``conj(E_i) · E_j`` that produces ``exp(-2π i · ν · (r_j - r_i) · ŝ
/ c)``, which is the COMPLEX CONJUGATE of the F18 / §8.M2-carryover-
line-2227 DSA-110 voltage convention pinned in the slow-corr
synthetic tests + the F21 cal-phase tests. We implement the POSITIVE-
sign convention here so test_F22 (the sign-convention guard the
briefing itself prescribes) passes; the briefing's negative-sign
sentence is the bug.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
import torch

from dsart.common.constants import (
    K_DM_MS_GHZ2_PC,
    NANTS,
    NATIVE_SAMPLE_US,
    NCHAN_PER_CHGROUP,
    NPOL,
    NU_TOP_PROC_GHZ,
    SPEED_OF_LIGHT_M_S,
    freq_GHz,
)
from dsart.services.slow_corr_kernel import (
    NPACKETS_PER_BLOCK,
    NTIMES_PER_PACKET,
)


__all__ = [
    "InjectionConfig",
    "OnlineInjector",
    "PROFILE_FAMILIES",
    "MAX_WIDTH_SAMPLES",
    "build_phasor_table",
    "build_dispersion_delay_table_ms",
    "build_profile_vector",
]


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Supported intrinsic profile families. ``"gaussian"`` is unit-area-
#: normalised by the analytical Gaussian integral over the full
#: ``[-max_width_samples, +max_width_samples]`` window (FWHM = width
#: native samples). ``"boxcar"`` is unit-area-normalised by dividing by
#: ``width``. Adding a new family is one branch in
#: :func:`build_profile_vector`.
PROFILE_FAMILIES: Final[tuple[str, ...]] = ("gaussian", "boxcar")

#: Cap on the half-width (in native samples) of the cached profile
#: vector. Sized to comfortably cover a single fada block (= 4096
#: native samples) for the widest realistic injection. The full vector
#: holds ``2 · MAX_WIDTH_SAMPLES + 1`` floats = ~32 KB float32.
MAX_WIDTH_SAMPLES: Final[int] = 4096


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionConfig:
    """Per-injection generator config (read from etcd val payload).

    Synthesised contribution lives in voltage space at the
    ``apply_at_specnum`` start; the injector adds it to the GEMM-layout
    voltage tensor before RFI. Profile is intrinsic at
    ``NU_TOP_PROC_GHZ``; dispersion is added per-channel via the
    standard cold-plasma law referenced to the same top frequency.

    Attributes
    ----------
    inj_id : str
        Operator-supplied opaque id (e.g. ``"inj_2026_05_05_1234"``).
        Echoed in the per-block injection log dict so trigger-side
        attribution can match captures back to the original command.
    l_rad, m_rad : float
        Direction cosines (rad) of the synthetic source. Bench / test
        defaults use small ``|l|, |m| ≤ 0.1`` — large values risk
        ``n = √(1 - l² - m²)`` going imaginary, which raises in
        :meth:`OnlineInjector.add_pending`.
    dm_pc_cm3 : float
        Dispersion measure (pc / cm³). Sign convention: positive ⇒
        normal cold-plasma dispersion, lower freqs delayed. Negative
        DMs are accepted (some tests use negative DM to invert the
        chirp direction without changing the magnitude).
    fluence_jy_ms : float
        Peak fluence at ``NU_TOP_PROC_GHZ`` (Jy · ms). The peak
        amplitude scale applied to the intrinsic profile is
        ``√(fluence / width_samples)``; the test suite confirms the
        per-(ant, ch) RMS amplitude tracks this scaling.
    width_samples : int
        FWHM width in NATIVE samples (1 native sample = 32.768 µs).
        Must satisfy ``1 ≤ width_samples ≤ MAX_WIDTH_SAMPLES``.
    profile : str
        Intrinsic profile family; one of :data:`PROFILE_FAMILIES`.
    apply_at_specnum : int
        SNAP packet sequence number of the injection peak (top-channel
        time, since dispersion is referenced to ``NU_TOP_PROC_GHZ``).
        Native-sample position of the peak is ``2 · apply_at_specnum``.

    Notes
    -----
    JSON round-trip (etcd payload format): the dataclass serialises
    losslessly via :func:`InjectionConfig.to_json` /
    :func:`InjectionConfig.from_json`. The wire schema is the same six
    fields above with the same names; no ``schema_version`` field is
    included in v1 (operator workflow ships configs alongside the
    `dsart` git SHA, and any breaking schema change accompanies a
    coordinated upgrade).
    """

    inj_id: str
    l_rad: float
    m_rad: float
    dm_pc_cm3: float
    fluence_jy_ms: float
    width_samples: int
    profile: str
    apply_at_specnum: int

    # ---- validation ----

    def __post_init__(self) -> None:
        if not isinstance(self.inj_id, str) or not self.inj_id:
            raise ValueError(f"inj_id must be a non-empty str; got {self.inj_id!r}")
        n2 = self.l_rad * self.l_rad + self.m_rad * self.m_rad
        if n2 >= 1.0:
            raise ValueError(
                f"l_rad={self.l_rad}, m_rad={self.m_rad}: "
                f"l² + m² = {n2:.6f} ≥ 1.0 (n would be imaginary)"
            )
        if not math.isfinite(self.dm_pc_cm3):
            raise ValueError(f"dm_pc_cm3 must be finite; got {self.dm_pc_cm3}")
        if not math.isfinite(self.fluence_jy_ms):
            raise ValueError(
                f"fluence_jy_ms must be finite; got {self.fluence_jy_ms}"
            )
        if not isinstance(self.width_samples, int):
            raise ValueError(
                f"width_samples must be int; got {type(self.width_samples).__name__}"
            )
        if not 1 <= self.width_samples <= MAX_WIDTH_SAMPLES:
            raise ValueError(
                f"width_samples={self.width_samples} outside "
                f"[1, {MAX_WIDTH_SAMPLES}]"
            )
        if self.profile not in PROFILE_FAMILIES:
            raise ValueError(
                f"profile={self.profile!r} not in {PROFILE_FAMILIES}"
            )
        if not isinstance(self.apply_at_specnum, int):
            raise ValueError(
                f"apply_at_specnum must be int; "
                f"got {type(self.apply_at_specnum).__name__}"
            )

    # ---- JSON serde (etcd payload format) ----

    def to_json(self) -> str:
        """Serialise to a JSON string suitable for etcd ``val`` payloads.

        Returns a stable, key-sorted JSON dict with the six
        :class:`InjectionConfig` fields. Floats are emitted at full
        ``repr``-precision (no ``%.6f`` truncation), so
        :meth:`from_json` round-trips bit-exactly.
        """
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str | bytes) -> "InjectionConfig":
        """Parse an etcd-style JSON ``val`` payload into a config.

        Tolerates ``bytes`` input (the etcd client returns ``bytes``).
        Raises :class:`ValueError` on missing / unknown keys; the
        validation in :meth:`__post_init__` then runs the value-level
        checks.
        """
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8")
        try:
            payload = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"InjectionConfig.from_json: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"InjectionConfig.from_json: expected dict, got "
                f"{type(payload).__name__}"
            )
        expected = {
            "inj_id", "l_rad", "m_rad", "dm_pc_cm3", "fluence_jy_ms",
            "width_samples", "profile", "apply_at_specnum",
        }
        missing = expected - payload.keys()
        extra = payload.keys() - expected
        if missing:
            raise ValueError(f"InjectionConfig.from_json missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"InjectionConfig.from_json extra keys: {sorted(extra)}")
        return cls(
            inj_id=str(payload["inj_id"]),
            l_rad=float(payload["l_rad"]),
            m_rad=float(payload["m_rad"]),
            dm_pc_cm3=float(payload["dm_pc_cm3"]),
            fluence_jy_ms=float(payload["fluence_jy_ms"]),
            width_samples=int(payload["width_samples"]),
            profile=str(payload["profile"]),
            apply_at_specnum=int(payload["apply_at_specnum"]),
        )


# ---------------------------------------------------------------------------
# Pure helpers (testable in isolation; no torch / no Injector state)
# ---------------------------------------------------------------------------


def build_phasor_table(
    *,
    l_rad: float,
    m_rad: float,
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    antpos_u: np.ndarray | None = None,
    chgroup: int,
    nchan: int = NCHAN_PER_CHGROUP,
) -> np.ndarray:
    """Per-(ant, ch) complex phasor for a point source at ``(l, m)``.

    Returns ``phasor[a, ch] = exp(+2π i · ν_ch · (E_a · l + N_a · m
    + U_a · n) / c)`` (POSITIVE sign in the exponent — TMS / DSA-110
    voltage convention; see module docstring).

    Parameters
    ----------
    l_rad, m_rad : float
        Direction cosines of the synthetic source.
    antpos_e, antpos_n : np.ndarray
        ``(NANTS,)`` E / N antenna positions (m). Same sign + units
        as :class:`dsart.cal.bf_weights.BfWeights.antpos_{e,n}`.
    antpos_u : np.ndarray, optional
        ``(NANTS,)`` U (up) antenna positions (m). DSA-110 core is
        approximately planar (U_a ≈ 0); pass ``None`` to assume zero.
    chgroup : int
        Corr-node chgroup index 0..15 (selects the per-channel
        frequencies via :func:`dsart.common.constants.freq_GHz`).
    nchan : int
        Channels per chgroup. Default :data:`NCHAN_PER_CHGROUP` = 384.

    Returns
    -------
    np.ndarray
        Shape ``(NANTS, nchan)`` complex64. Each cell has unit
        modulus to numerical precision (``|phasor| = 1.0`` exact in
        complex128 before the cast; the complex64 cast costs ~1e-7
        relative magnitude).
    """
    if antpos_e.shape != (NANTS,) or antpos_n.shape != (NANTS,):
        raise ValueError(
            f"antpos_e shape {antpos_e.shape}, antpos_n shape "
            f"{antpos_n.shape}, expected ({NANTS},) each"
        )
    if antpos_u is None:
        antpos_u_arr = np.zeros(NANTS, dtype=np.float64)
    else:
        if antpos_u.shape != (NANTS,):
            raise ValueError(
                f"antpos_u shape {antpos_u.shape}, expected ({NANTS},)"
            )
        antpos_u_arr = antpos_u.astype(np.float64, copy=False)

    n_dir = math.sqrt(max(0.0, 1.0 - l_rad * l_rad - m_rad * m_rad))
    if n_dir <= 0.0:
        raise ValueError(
            f"l_rad={l_rad}, m_rad={m_rad}: n = sqrt(1 - l² - m²) is "
            f"non-positive ({n_dir})"
        )

    e_a = antpos_e.astype(np.float64, copy=False)               # (NANTS,)
    n_a = antpos_n.astype(np.float64, copy=False)
    u_a = antpos_u_arr

    # Geometric path-difference at each ant (m): r_a · ŝ.
    r_dot_s = e_a * l_rad + n_a * m_rad + u_a * n_dir           # (NANTS,)

    # Frequencies for this chgroup (Hz). Using freq_GHz keeps the
    # injector pinned to the same per-(chgroup, ch) frequency table the
    # rest of the pipeline uses (M1 plan-fix F8 edge convention).
    f_hz = np.asarray(
        [freq_GHz(chgroup, ch) * 1.0e9 for ch in range(nchan)],
        dtype=np.float64,
    )                                                           # (nchan,)

    arg = (
        2.0 * math.pi / SPEED_OF_LIGHT_M_S
        * r_dot_s[:, None] * f_hz[None, :]
    )                                                           # (NANTS, nchan)
    phasor_c128 = np.cos(arg) + 1j * np.sin(arg)
    return phasor_c128.astype(np.complex64)                     # (NANTS, nchan)


def build_dispersion_delay_table_ms(
    *,
    dm_pc_cm3: float,
    chgroup: int,
    nchan: int = NCHAN_PER_CHGROUP,
) -> np.ndarray:
    """Per-channel dispersion delay (ms) referenced to ``NU_TOP_PROC_GHZ``.

    Returns ``Δτ[ch] = K_DM_MS_GHZ2_PC · DM · (1/ν_ch² − 1/ν_top²)``
    in ms, with ``ν_top = NU_TOP_PROC_GHZ`` (= 1.49875 GHz, the top
    edge of the processed band, the same reference the rest of the
    pipeline uses for cross-chgroup time alignment per §3.6.2 stage-2).

    For ``DM > 0`` the bottom of the band lags the top, so this array
    is positive-monotonic with channel index (ch=0 is top → 0 ms;
    ch=383 is bottom → ~3 ms at DM=400 in chgroup 0).

    Parameters
    ----------
    dm_pc_cm3 : float
        DM in pc / cm³. Sign + magnitude both pass through to the
        formula; negative DMs invert the chirp direction.
    chgroup : int
        Corr-node chgroup index 0..15.
    nchan : int
        Channels per chgroup.

    Returns
    -------
    np.ndarray
        Shape ``(nchan,)`` float32 — the array is small (~1.5 KB) so
        the precision loss from the float64 → float32 cast is
        negligible vs the ~32.768 µs native sample period.
    """
    nu_ghz = np.asarray(
        [freq_GHz(chgroup, ch) for ch in range(nchan)],
        dtype=np.float64,
    )                                                           # (nchan,)
    inv_nu2 = 1.0 / (nu_ghz * nu_ghz)
    inv_nu_top2 = 1.0 / (NU_TOP_PROC_GHZ * NU_TOP_PROC_GHZ)
    delay_ms = K_DM_MS_GHZ2_PC * dm_pc_cm3 * (inv_nu2 - inv_nu_top2)
    return delay_ms.astype(np.float32)                          # (nchan,)


def build_profile_vector(
    *,
    width_samples: int,
    profile: str,
    max_width_samples: int = MAX_WIDTH_SAMPLES,
) -> np.ndarray:
    """Centred unit-area intrinsic profile in NATIVE-sample units.

    Returns an array of shape ``(2 · max_width_samples + 1,)`` float32
    centred at index ``max_width_samples`` (so ``profile[centre + k]``
    is the value ``k`` samples after the peak).

    Profile families:
      - ``"gaussian"``: ``exp(-(t/σ)²/2)`` with σ = FWHM / (2√(2 ln 2));
        unit-area-normalised by the analytical Gaussian integral
        ``σ √(2π)``. Sums to ~1.0 over the full window for any
        ``width_samples ≪ max_width_samples`` (the analytical
        normalisation is exact in the limit; the discrete sum errors
        scale as ``erfc(max_width_samples / (σ√2)) / 2`` which is
        ≤ 1e-15 at default support widths).
      - ``"boxcar"``: ``1/width`` over ``[-width/2, +width/2]`` (FWHM
        = full width). Unit-area-normalised by construction.

    Parameters
    ----------
    width_samples : int
        FWHM width in NATIVE samples (1 sample = 32.768 µs).
    profile : str
        One of :data:`PROFILE_FAMILIES`.
    max_width_samples : int
        Half-width of the cached vector. Default
        :data:`MAX_WIDTH_SAMPLES`.

    Returns
    -------
    np.ndarray
        Shape ``(2 · max_width_samples + 1,)`` float32.
    """
    if width_samples < 1 or width_samples > max_width_samples:
        raise ValueError(
            f"width_samples={width_samples} outside [1, {max_width_samples}]"
        )
    if profile not in PROFILE_FAMILIES:
        raise ValueError(f"profile={profile!r} not in {PROFILE_FAMILIES}")

    n = 2 * max_width_samples + 1
    t = np.arange(n, dtype=np.float64) - max_width_samples       # (n,)

    if profile == "gaussian":
        sigma = float(width_samples) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        norm = sigma * math.sqrt(2.0 * math.pi)
        out = np.exp(-(t * t) / (2.0 * sigma * sigma)) / norm
    elif profile == "boxcar":
        out = np.zeros(n, dtype=np.float64)
        half = float(width_samples) / 2.0
        in_box = (t >= -half) & (t < +half)
        out[in_box] = 1.0 / float(width_samples)
    else:  # pragma: no cover — guarded above
        raise ValueError(profile)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# OnlineInjector
# ---------------------------------------------------------------------------


_GEMM_LAYOUT_SHAPE: Final[tuple[int, ...]] = (
    NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
    NPACKETS_PER_BLOCK, NANTS,
)


@dataclass
class _ActiveInjection:
    """Cached per-injection state (built once at ``add_pending`` time)."""

    cfg: InjectionConfig
    phasor_real: torch.Tensor           # (NANTS, NCHAN_PER_CHGROUP) dtype
    phasor_imag: torch.Tensor           # (NANTS, NCHAN_PER_CHGROUP) dtype
    dispersion_offset_samples: torch.Tensor  # (NCHAN_PER_CHGROUP,) int64
    profile: torch.Tensor               # (2 * MAX_WIDTH_SAMPLES + 1,) dtype
    amplitude: float                    # √(fluence / width)
    peak_native: int                    # 2 · apply_at_specnum
    max_width_samples: int

    @property
    def first_native_in_window(self) -> int:
        """Earliest native-sample position with non-zero contribution."""
        return self.peak_native - self.max_width_samples + int(
            self.dispersion_offset_samples.min().item()
        )

    @property
    def last_native_in_window(self) -> int:
        """Latest native-sample position with non-zero contribution."""
        return self.peak_native + self.max_width_samples + int(
            self.dispersion_offset_samples.max().item()
        )


class OnlineInjector:
    """Voltage-domain online injector for one corr-node / chgroup.

    Construct once at service startup with the cal-blob antpos
    (``antpos_e``, ``antpos_n``) and the chgroup index. Per fada-block
    arrival, call :meth:`apply_block` with the block's specnum start
    and the GEMM-layout voltage tensors; in-flight injections (queued
    via :meth:`add_pending`) are added to the voltage in-place.

    Single-pol (intrinsic unpolarised), single-band (one chgroup per
    instance), single-precision (``dtype`` = ``torch.float16`` to
    match the production fast-corr GEMM input; tests pass ``float32``
    for higher-precision phase comparisons).

    Parameters
    ----------
    antpos_e, antpos_n : np.ndarray
        ``(NANTS,)`` antenna positions (m), East / North. Same sign
        and units as :class:`dsart.cal.bf_weights.BfWeights.antpos_{e,n}`
        (read by ``cal_loader``; the injector does *not* depend on
        the U component because DSA-110 is planar — see chunk-3d
        final report for the per-ant U_a magnitude reading on
        a real fixture cal blob).
    chgroup : int
        Corr-node chgroup index 0..15.
    device : torch.device
        Device the per-block voltage tensors live on. The cached
        injection tables are materialised on this device at
        :meth:`add_pending` time.
    dtype : torch.dtype
        Voltage dtype (``torch.float16`` in production,
        ``torch.float32`` for tests). The cached injection tables
        share this dtype so the in-place adds in :meth:`apply_block`
        avoid temporary casts.
    antpos_u : np.ndarray, optional
        ``(NANTS,)`` U component (m). Defaults to zero (DSA-110
        planar-array assumption). Pass an array if the test fixture
        has non-zero U.

    Attributes
    ----------
    pending : dict[str, _ActiveInjection]
        Currently-active injections, keyed by ``inj_id``. Internal;
        treat as read-only from outside the class.

    Notes
    -----
    Memory: each cached injection holds ~32 KB profile + ~9 KB
    dispersion + ~290 KB (NANTS × NCHAN × cplx) phasor (fp16) — total
    ~330 KB / injection. The hot-path :meth:`apply_block` allocates
    one ``[NTIMES_PER_PACKET, NPACKETS_PER_BLOCK, NCHAN] dtype``
    temporary (~12 MB at fp16) per active injection per block.
    """

    def __init__(
        self,
        antpos_e: np.ndarray,
        antpos_n: np.ndarray,
        chgroup: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        antpos_u: np.ndarray | None = None,
    ) -> None:
        if antpos_e.shape != (NANTS,):
            raise ValueError(
                f"antpos_e shape {antpos_e.shape}, expected ({NANTS},)"
            )
        if antpos_n.shape != (NANTS,):
            raise ValueError(
                f"antpos_n shape {antpos_n.shape}, expected ({NANTS},)"
            )
        if not 0 <= chgroup < 16:
            raise ValueError(f"chgroup={chgroup}, expected 0..15")
        # Normalise "cuda" → "cuda:N" the same way slow_corr_kernel does
        # so cached-tensor device equality checks against caller voltage
        # tensors don't spuriously fail.
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.antpos_e = np.ascontiguousarray(antpos_e, dtype=np.float64)
        self.antpos_n = np.ascontiguousarray(antpos_n, dtype=np.float64)
        if antpos_u is None:
            self.antpos_u = np.zeros(NANTS, dtype=np.float64)
        else:
            if antpos_u.shape != (NANTS,):
                raise ValueError(
                    f"antpos_u shape {antpos_u.shape}, expected ({NANTS},)"
                )
            self.antpos_u = np.ascontiguousarray(antpos_u, dtype=np.float64)
        self.chgroup = int(chgroup)
        self.device = dev
        self.dtype = dtype
        self.pending: dict[str, _ActiveInjection] = {}

    # ---- public hot-path API ----

    def add_pending(self, cfg: InjectionConfig) -> None:
        """Queue an injection; build its three cached tables.

        Idempotent on ``inj_id``: a second call with the same id
        replaces the previous entry (the operator workflow allows
        re-issuing an inject command with corrected fields). Logs at
        the etcd-watcher level cite both the old and new ids.

        Raises
        ------
        ValueError
            If the antpos / chgroup geometry produces a degenerate
            phasor table (n ≤ 0, etc.).
        """
        # Pure-numpy table builds (kept as numpy so they're testable
        # without torch in `tests/test_online_injector.py`).
        phasor_c64 = build_phasor_table(
            l_rad=cfg.l_rad,
            m_rad=cfg.m_rad,
            antpos_e=self.antpos_e,
            antpos_n=self.antpos_n,
            antpos_u=self.antpos_u,
            chgroup=self.chgroup,
        )                                                       # (NANTS, NCHAN)
        delay_ms = build_dispersion_delay_table_ms(
            dm_pc_cm3=cfg.dm_pc_cm3,
            chgroup=self.chgroup,
        )                                                       # (NCHAN,) ms
        profile_arr = build_profile_vector(
            width_samples=cfg.width_samples,
            profile=cfg.profile,
            max_width_samples=MAX_WIDTH_SAMPLES,
        )                                                       # (2*MWS+1,)

        # Convert dispersion delay ms → integer NATIVE-sample offsets
        # (1 native sample = NATIVE_SAMPLE_US µs). round() not
        # truncate so ±0.5-sample residuals don't pile up at the band
        # edges; the per-(coarse-DM) corr-side stage-1 already uses
        # the same integer-sample-shift quantisation per §3.6.2 (so a
        # ≤1-sample injector residual is fine — the downstream
        # de-disperser will absorb it).
        delay_samples = np.round(
            delay_ms / (NATIVE_SAMPLE_US * 1.0e-3)
        ).astype(np.int64)                                      # (NCHAN,) int64

        # Materialise on device with the kernel's dtype.
        phasor_real_t = torch.as_tensor(
            phasor_c64.real.astype(np.float32), device=self.device,
        ).to(self.dtype)
        phasor_imag_t = torch.as_tensor(
            phasor_c64.imag.astype(np.float32), device=self.device,
        ).to(self.dtype)
        delay_t = torch.as_tensor(
            delay_samples, device=self.device, dtype=torch.int64,
        )
        profile_t = torch.as_tensor(
            profile_arr, device=self.device,
        ).to(self.dtype)

        amp = math.sqrt(abs(cfg.fluence_jy_ms) / float(cfg.width_samples))
        if cfg.fluence_jy_ms < 0:
            amp = -amp

        active = _ActiveInjection(
            cfg=cfg,
            phasor_real=phasor_real_t,
            phasor_imag=phasor_imag_t,
            dispersion_offset_samples=delay_t,
            profile=profile_t,
            amplitude=amp,
            peak_native=2 * int(cfg.apply_at_specnum),
            max_width_samples=MAX_WIDTH_SAMPLES,
        )
        self.pending[cfg.inj_id] = active

    def apply_block(
        self,
        voltages_real: torch.Tensor,
        voltages_imag: torch.Tensor,
        block_specnum_start: int,
    ) -> dict[str, Any]:
        """Add active injection contributions in-place to a fada block.

        Iterates over :attr:`pending`; for each injection still
        overlapping the block's native-sample window, builds the
        ``[NTIMES_PER_PACKET, NPACKETS_PER_BLOCK, NCHAN]`` per-time
        per-channel envelope (= profile-after-dispersion, evaluated
        at this block's absolute native-sample positions) and adds
        ``phasor[a, ch] · envelope[t, ch] · amplitude`` to both pol
        slots of the voltage tensor in-place.

        Injections whose footprint is entirely past the block (i.e.
        their last contribution sample is before
        ``block_specnum_start * 2``) are auto-purged from
        :attr:`pending` so the dict doesn't grow unboundedly when the
        operator submits long-tailed schedules.

        Parameters
        ----------
        voltages_real, voltages_imag : torch.Tensor
            Real / imag voltages in M2's GEMM layout
            ``(NCHAN_PER_CHGROUP, NTIMES_PER_PACKET, NPOL,
            NPACKETS_PER_BLOCK, NANTS)`` and dtype matching
            :attr:`dtype`. Must be on :attr:`device`. Both tensors
            are mutated in-place via :meth:`torch.Tensor.add_`.
        block_specnum_start : int
            SNAP packet sequence number at the block's first packet.
            Native-sample offset of native-sample ``(pkt, t_sub)``
            within the block is ``pkt * NTIMES_PER_PACKET + t_sub``;
            absolute native position is
            ``block_specnum_start * 2 + pkt * 2 + t_sub`` (1 specnum
            = 2 native samples).

        Returns
        -------
        dict[str, Any]
            Per-block injection log. Keys:
              - ``"block_specnum_start"`` (int) — echo of the input.
              - ``"active_inj_ids"`` (list[str]) — ids that contributed.
              - ``"contributions"`` (list[dict]) — per-active-id
                summary dict with keys
                ``{"inj_id", "rms_real_added", "rms_imag_added",
                "n_native_samples_in_block", "first_native_used",
                "last_native_used"}``. ``rms_*`` is computed across
                the entire ``(ch, t, p, pkt, ant)`` grid of the
                contribution before the in-place add.
              - ``"n_purged"`` (int) — count of pending entries
                purged because their footprint ended before this
                block.
        """
        self._validate_voltage_tensors(voltages_real, voltages_imag)
        block_native_start = int(block_specnum_start) * 2
        block_native_end = block_native_start + (
            NPACKETS_PER_BLOCK * NTIMES_PER_PACKET
        )                                                       # exclusive
        log: dict[str, Any] = {
            "block_specnum_start": int(block_specnum_start),
            "active_inj_ids": [],
            "contributions": [],
            "n_purged": 0,
        }

        # Pre-compute the absolute native-sample position of every
        # (pkt, t_sub) entry: shape (NTIMES_PER_PACKET, NPACKETS_PER_BLOCK).
        # Order matches the GEMM layout's (axis 1, axis 3).
        pkt_idx = torch.arange(
            NPACKETS_PER_BLOCK, device=self.device, dtype=torch.int64,
        )                                                       # (NPACKETS,)
        tsub_idx = torch.arange(
            NTIMES_PER_PACKET, device=self.device, dtype=torch.int64,
        )                                                       # (NTIMES,)
        # native_pos[t, pkt] = block_native_start + pkt * 2 + t_sub
        native_pos = (
            block_native_start
            + pkt_idx[None, :] * NTIMES_PER_PACKET
            + tsub_idx[:, None]
        )                                                       # (NTIMES, NPACKETS)

        purged: list[str] = []
        for inj_id, active in self.pending.items():
            # Footprint test on integer native positions: if no overlap,
            # skip cheaply (no large tensor allocation).
            footprint_first = active.first_native_in_window
            footprint_last = active.last_native_in_window      # inclusive
            if footprint_last < block_native_start:
                # entirely past — purge.
                purged.append(inj_id)
                continue
            if footprint_first >= block_native_end:
                # entirely future — keep, no contribution this block.
                continue

            contrib = self._add_one_injection(
                voltages_real, voltages_imag, native_pos, active,
            )
            if contrib is not None:
                log["active_inj_ids"].append(inj_id)
                log["contributions"].append(contrib)

        for inj_id in purged:
            del self.pending[inj_id]
        log["n_purged"] = len(purged)
        return log

    # ---- internals ----

    def _validate_voltage_tensors(
        self, voltages_real: torch.Tensor, voltages_imag: torch.Tensor,
    ) -> None:
        if voltages_real.shape != _GEMM_LAYOUT_SHAPE:
            raise ValueError(
                f"voltages_real shape {tuple(voltages_real.shape)} != "
                f"GEMM layout {_GEMM_LAYOUT_SHAPE}"
            )
        if voltages_imag.shape != _GEMM_LAYOUT_SHAPE:
            raise ValueError(
                f"voltages_imag shape {tuple(voltages_imag.shape)} != "
                f"GEMM layout {_GEMM_LAYOUT_SHAPE}"
            )
        if voltages_real.dtype != self.dtype:
            raise ValueError(
                f"voltages_real dtype {voltages_real.dtype} != "
                f"injector dtype {self.dtype}"
            )
        if voltages_imag.dtype != self.dtype:
            raise ValueError(
                f"voltages_imag dtype {voltages_imag.dtype} != "
                f"injector dtype {self.dtype}"
            )
        if voltages_real.device != self.device:
            raise ValueError(
                f"voltages_real on {voltages_real.device}, "
                f"injector on {self.device}"
            )
        if voltages_imag.device != self.device:
            raise ValueError(
                f"voltages_imag on {voltages_imag.device}, "
                f"injector on {self.device}"
            )

    def _add_one_injection(
        self,
        voltages_real: torch.Tensor,
        voltages_imag: torch.Tensor,
        native_pos: torch.Tensor,            # (NTIMES, NPACKETS) int64
        active: _ActiveInjection,
    ) -> dict[str, Any] | None:
        """Materialise + add one injection's contribution; return log dict.

        Returns ``None`` if the injection happened to land entirely
        outside the in-block native window (defensive — the caller
        also pre-filters via ``first_native_in_window``).
        """
        # Per-(t, pkt, ch) profile index =
        #   native_pos - peak_native - dispersion_offset[ch] + max_width_samples.
        # Out-of-window indices map to a sentinel slot of value 0 (we
        # use a one-shot guarded gather: clamp to a valid range, then
        # mask out invalid positions to zero before scaling).
        n = active.profile.shape[0]                              # = 2*MWS + 1
        # Build the absolute profile-window position once: offset =
        # native_pos - peak_native; shape (NTIMES, NPACKETS).
        rel_pos = native_pos - active.peak_native               # int64
        # idx[t, pkt, ch] = rel_pos[t, pkt] - delay[ch] + max_width_samples
        idx = (
            rel_pos[:, :, None]
            - active.dispersion_offset_samples[None, None, :]
            + active.max_width_samples
        )                                                       # (T, P, C) int64
        in_window = (idx >= 0) & (idx < n)
        # Clamp + gather; mask the invalid positions to zero post-gather.
        idx_clamped = idx.clamp(min=0, max=n - 1)
        env_flat = active.profile[idx_clamped.reshape(-1)]      # (T*P*C,) dtype
        envelope = env_flat.reshape(idx.shape)                   # (T, P, C) dtype
        # Mask out-of-window cells.
        envelope = envelope * in_window.to(self.dtype)
        # Apply amplitude scaling once on the (T, P, C) envelope (cheap).
        envelope = envelope * float(active.amplitude)

        # Skip cleanly if the envelope is identically zero in this block.
        if not bool(in_window.any()):
            return None

        # Per-(ch, t, p, pkt, ant) contribution =
        #   phasor[ant, ch] * envelope[t, pkt, ch].
        # Re-arrange envelope to (ch, t, pkt) → broadcast against
        # phasor (ch, ant) into the (ch, t, pkt, ant) plane; then
        # broadcast across the NPOL pol slots in the GEMM layout.
        env_ctp = envelope.permute(2, 0, 1).contiguous()         # (C, T, P)
        # phasor (ant, ch) → (ch, ant)
        phasor_real_ca = active.phasor_real.transpose(0, 1).contiguous()  # (C, A)
        phasor_imag_ca = active.phasor_imag.transpose(0, 1).contiguous()
        # contrib_real[ch, t, pkt, ant] =
        #     env_ctp[ch, t, pkt] * phasor_real_ca[ch, ant]
        # Use torch broadcasting + an explicit batched outer-product to
        # keep the per-ch FLOPs in a tight cuBLAS-bound kernel.
        contrib_real = env_ctp[..., None] * phasor_real_ca[:, None, None, :]
        contrib_imag = env_ctp[..., None] * phasor_imag_ca[:, None, None, :]
        # contrib_real now (C, T, P, A); insert NPOL singleton.
        contrib_real_5d = contrib_real.unsqueeze(2)              # (C, T, 1, P, A)
        contrib_imag_5d = contrib_imag.unsqueeze(2)

        # Snapshot the contribution RMS for the log BEFORE the in-place
        # add; expand to the full broadcast shape so the RMS is
        # NPOL-aware (= same value, but the count includes the pol
        # axis to match the voltage tensor's element count).
        rms_real = float(torch.sqrt(
            (contrib_real_5d.to(torch.float32) ** 2).mean()
        ).item())
        rms_imag = float(torch.sqrt(
            (contrib_imag_5d.to(torch.float32) ** 2).mean()
        ).item())

        # In-place add to both pol slots.
        voltages_real.add_(contrib_real_5d.expand(_GEMM_LAYOUT_SHAPE))
        voltages_imag.add_(contrib_imag_5d.expand(_GEMM_LAYOUT_SHAPE))

        n_in = int(in_window.sum().item())
        # Diagnostic: which absolute native positions actually got hit?
        used_pos = native_pos[in_window.any(dim=2)]  # (rows where any ch hit)
        first_native_used = (
            int(used_pos.min().item()) if used_pos.numel() else -1
        )
        last_native_used = (
            int(used_pos.max().item()) if used_pos.numel() else -1
        )
        return {
            "inj_id": active.cfg.inj_id,
            "rms_real_added": rms_real,
            "rms_imag_added": rms_imag,
            "n_native_samples_in_block": n_in,
            "first_native_used": first_native_used,
            "last_native_used": last_native_used,
        }
