"""Spectral Kurtosis (SK) estimator (M3 chunk 3c; plan §4.2 step 2).

Computes per-(ant, ch, pol) per-accumulation Spectral Kurtosis values
from the ``S₁_M`` / ``S₂_M`` auto-power moments, and applies upper /
lower thresholds at a target false-alarm rate (FAR).

Definition:

    SK_M[ant, ch, pol, n_acc] = ((M+1)/(M-1)) · (M · S₂_M / S₁_M² − 1)

For complex Gaussian-distributed voltages (thermal noise), SK has
mean 1 and approximate variance ``4/M`` (standard SK statistic; see
Nita & Gary 2010, MNRAS 406, L60).

**Sign convention** (corrected 2026-09-13; this docstring previously
had it backwards): continuous RFI filling the accumulation depresses
SK **below** 1 — the channel power becomes deterministic, so
``Var(p)/E[p]² → 0`` — while intermittent RFI occupying a fraction
``f < 0.5`` of the accumulation lifts SK **above** 1. Nita & Gary put
it directly: the estimator "is expected to deviate below unity if the
RFI signal is continuous or acting for more than half of the
accumulation time". Both are caught by the two-sided test, so the old
wording never changed what was flagged, only how an operator read it.

The direction of the excursion is a free duty-cycle diagnostic and is
published: :func:`sk_combined_masks` returns the high-side mask
separately, and :class:`dsart.rfi.combine.FlagSourceBit` carries it as
``SK_HIGH`` alongside ``SK``.

We flag where SK is outside the ``[sk_low, sk_high]`` Nita-Gary
confidence interval at the requested two-sided FAR.

Threshold derivation
====================

Under the null (Gaussian noise), SK has known mean 1 and analytic
variance ``4·M·(M-1) / ((M-2)(M+2)(M+3))`` (Nita & Gary 2010 Eq. 4),
**but at small M the distribution is markedly right-skewed** — at
``M = 64`` the empirical skewness γ₁ ≈ 1.1, so a Gaussian-quantile
threshold under-estimates the upper FAR by a factor of ~40× at
``FAR = 1e-4``. Empirically validated at ``M = 64`` on a 524288-cell
thermal-noise sample (h01, 2026-05-05): Gaussian thresholds
``[0.07, 1.93]`` saw FAR = 4.0e-3 instead of the target 1e-4.

For correctness at all four default M's, we compute thresholds by
**Monte-Carlo simulation** of SK under the null and cache the
quantile lookup. The MC uses iid ``|E|² ~ Exp(1)`` (the standard
chi-squared-2 model for unit-variance complex Gaussian voltage
moduli-squared) and computes S₁, S₂ directly without going through
the full autos pipeline — typically ``< 100 ms per M`` at 10⁶ trials.
The cache is keyed on ``(M, far)`` and persists for the lifetime of
the Python interpreter; re-loading the module re-runs the MC.

Operator-supplied ``(sk_low, sk_high)`` overrides bypass the cache
entirely (see :func:`sk_mask` ``sk_low`` / ``sk_high`` arguments)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from fractions import Fraction
from typing import Final

import numpy as np
import torch

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default per-(ant, ch, pol, M) sample two-sided false-alarm rate.
DEFAULT_SK_FAR: Final[float] = 1e-4

#: Number of Monte-Carlo trials per M / FAR threshold solve. 1e6 gives
#: a quantile estimate accurate to ~10 % of the target FAR (i.e. for
#: ``FAR = 1e-4``, the quantile sampling noise is ~1e-5 — fine for our
#: 2× headroom test). Scales linearly in MC time; ~120 ms / M on CPU.
_SK_MC_N_TRIALS: Final[int] = 1_000_000

#: Peak bytes for one Monte-Carlo chunk (see _mc_sk_thresholds).
#: 256 MiB keeps the solve well clear of a busy node's headroom at
#: every M while costing nothing in accuracy — the trials are iid,
#: so chunking changes only the allocation pattern.
_MC_CHUNK_BYTES: Final[int] = 256 * 1024 * 1024

#: Cache for ``(M, far) → (sk_low, sk_high)`` empirical Pearson-IV-
#: equivalent thresholds. Populated lazily by :func:`sk_thresholds`.
_SK_THRESHOLD_CACHE: dict[tuple[int, float], tuple[float, float]] = {}

#: Cache for the analytic Pearson-IV thresholds (R2).
_SK_P4_CACHE: dict[tuple[int, float], tuple[float, float]] = {}


# ---------------------------------------------------------------------------
# Thresholds — Monte Carlo with caching
# ---------------------------------------------------------------------------


def _mc_sk_thresholds(
    m: int,
    far: float,
    *,
    n_trials: int = _SK_MC_N_TRIALS,
    seed: int = 20260505,
) -> tuple[float, float]:
    """Compute ``(sk_low, sk_high)`` empirically via Monte Carlo.

    Generates ``n_trials`` independent SK samples under the null
    (complex Gaussian voltages) by simulating ``|E|² ~ Exp(1)``
    samples directly (faster than going through the full autos
    pipeline by ~10×). Returns the ``far/2`` and ``1 - far/2``
    quantiles of the resulting SK distribution.
    """
    rng = np.random.default_rng(seed)
    # Generate (n_trials, M) iid Exp(1) samples = |E|² for complex
    # Gaussian voltages with unit per-component variance. We synthesise
    # |E|² directly from the chi-squared-2 distribution; this matches
    # the joint distribution of (S1, S2) exactly because S1 and S2 are
    # functions only of the |E|² values, not of the original
    # (real, imag) decomposition.
    #
    # CHUNKED (2026-09-13). The previous one-shot allocation was
    # ``(n_trials, M)`` float64 plus a same-sized temporary for
    # ``e2 * e2``: at M = 4096 that is 30.5 GB + 30.5 GB = ~61 GB
    # peak, on corr nodes with ~60 GB free. It survived only because
    # nothing else was mid-allocation. Chunking bounds the peak at
    # _MC_CHUNK_BYTES regardless of M.
    chunk = max(1, int(_MC_CHUNK_BYTES // (m * 8)))
    sk_parts: list[np.ndarray] = []
    done = 0
    while done < n_trials:
        take = min(chunk, n_trials - done)
        e2 = rng.exponential(scale=1.0, size=(take, m))
        s1_c = e2.sum(axis=-1)
        e2 *= e2                       # in place: no second full temporary
        s2_c = e2.sum(axis=-1)
        del e2
        s1_sq_c = np.maximum(s1_c * s1_c, 1e-30)
        sk_parts.append(
            ((m + 1.0) / (m - 1.0)) * (m * s2_c / s1_sq_c - 1.0)
        )
        done += take
    sk = np.concatenate(sk_parts)
    del sk_parts
    sk_low = float(np.quantile(sk, far / 2.0))
    sk_high = float(np.quantile(sk, 1.0 - far / 2.0))
    return sk_low, sk_high


def sk_thresholds(m: int, far: float = DEFAULT_SK_FAR) -> tuple[float, float]:
    """Two-sided SK thresholds at the given false-alarm rate.

    Computes ``(sk_low, sk_high)`` via Monte-Carlo simulation of SK
    under the null (complex Gaussian voltages with unit per-component
    variance). Results are cached on ``(M, FAR)`` for the lifetime
    of the Python interpreter.

    Args:
        m: accumulation depth (must be ≥ 4).
        far: two-sided false-alarm rate. Default :data:`DEFAULT_SK_FAR`.

    Returns:
        ``(sk_low, sk_high)`` thresholds. Cells with ``SK < sk_low``
        or ``SK > sk_high`` are flagged.

    Raises:
        ValueError: ``m < 4`` or ``far`` outside ``(0, 1)``.
    """
    if m < 4:
        raise ValueError(f"M={m}, expected M >= 4")
    if not 0.0 < far < 1.0:
        raise ValueError(f"far={far}, expected in (0, 1)")
    key = (int(m), float(far))
    if key not in _SK_THRESHOLD_CACHE:
        _SK_THRESHOLD_CACHE[key] = _solve_thresholds(int(m), float(far))
    return _SK_THRESHOLD_CACHE[key]


#: Smallest M at which the Pearson IV fit is trusted for production
#: thresholds. MEASURED, not assumed — see :func:`_solve_thresholds`.
PEARSON4_MIN_M: Final[int] = 256


def _solve_thresholds(m: int, far: float) -> tuple[float, float]:
    """Pearson IV above :data:`PEARSON4_MIN_M`, Monte Carlo below it.

    Pearson IV is a FOUR-MOMENT FIT, not an exact answer. Matching the
    first four moments does not pin a 1e-4 tail, and at small M it
    visibly does not. Realised FAR against an independent 2e6-trial
    null, target 1e-4 (2026-09-13):

    ====  ==========  ==========
    M     Pearson IV  Monte Carlo
    ====  ==========  ==========
    64    5.9e-5      1.13e-4
    256   8.2e-5      9.2e-5
    1024  9.1e-5      1.02e-4
    4096  1.20e-4     1.41e-4
    ====  ==========  ==========

    At M = 64 the type IV fit is **5.9 sigma low** — it would flag
    roughly half the intended rate — while the MC sits within 2 sigma.
    Above M = 256 the null is close enough to Gaussian that the fit
    holds, and that is exactly where the MC is expensive: the M = 4096
    solve is ~60 s at the production 1e6 trials and, before chunking,
    peaked near 61 GB.

    So: MC where it is cheap and the fit fails, Pearson IV where the
    fit holds and the MC is not. Falls back to MC on any Pearson IV
    failure (no scipy, moments outside the type IV region).
    """
    if m >= PEARSON4_MIN_M:
        try:
            return pearson4_sk_thresholds(m, far)
        except Exception as exc:                    # noqa: BLE001
            LOG.warning(
                "Pearson IV SK thresholds unavailable for M=%d far=%g "
                "(%s); falling back to Monte Carlo", m, far, exc,
            )
    return _mc_sk_thresholds(m, far)


def gaussian_sk_thresholds(
    m: int, far: float = DEFAULT_SK_FAR,
) -> tuple[float, float]:
    """Gaussian-approximation SK thresholds (debug / asymptotic only).

    Uses ``SK ~ N(1, σ_SK²)`` with the Nita-Gary variance
    ``σ_SK² = 4·M·(M-1) / ((M-2)·(M+2)·(M+3))``. Underestimates the
    upper-tail FAR at small M by up to ~40× at ``M = 64`` /
    ``FAR = 1e-4`` — production callers should prefer
    :func:`sk_thresholds`.
    """
    if m < 4:
        raise ValueError(f"M={m}, expected M >= 4")
    if not 0.0 < far < 1.0:
        raise ValueError(f"far={far}, expected in (0, 1)")
    z = math.sqrt(2.0) * _erfinv(1.0 - far)
    sigma_sk = math.sqrt(4.0 * m * (m - 1) / ((m - 2) * (m + 2) * (m + 3)))
    return 1.0 - z * sigma_sk, 1.0 + z * sigma_sk


def _erfinv(x: float) -> float:
    """Inverse error function via torch's vectorised erfinv."""
    if not -1.0 < x < 1.0:
        raise ValueError(f"erfinv argument must be in (-1, 1); got {x}")
    return float(torch.erfinv(torch.tensor(x, dtype=torch.float64)))


# ---------------------------------------------------------------------------
# Thresholds — analytic Pearson type IV (R2)
# ---------------------------------------------------------------------------


def _integer_partitions(n: int) -> "list[tuple[int, ...]]":
    """All partitions of ``n`` as non-increasing tuples."""
    if n == 0:
        return [()]
    out: list[tuple[int, ...]] = []

    def rec(rem: int, cap: int, acc: tuple[int, ...]) -> None:
        if rem == 0:
            out.append(acc)
            return
        for part in range(min(rem, cap), 0, -1):
            rec(rem - part, part, acc + (part,))

    rec(n, n, ())
    return out


def sk_raw_moments(m: int, n_max: int = 4) -> "list[Fraction]":
    """Exact raw moments ``E[T^n]``, ``n = 1..n_max``, of
    ``T = S₂/S₁²`` under the null.

    Derivation (exact, not an approximation). With ``p_i`` iid
    ``Exp(1)``, ``S₁ = Σp_i`` is ``Gamma(M, 1)`` and the normalised
    shares ``u_i = p_i/S₁`` are ``Dirichlet(1,...,1)`` **independent of
    S₁**. Hence ``T = Σ u_i²`` depends only on the Dirichlet, and

        E[∏ u_i^{k_i}] = (∏ k_i!) / (M(M+1)···(M+K-1)),  K = Σk_i

    so, grouping the multinomial expansion of ``(Σ u_i²)^n`` by the
    partition ``λ`` of ``n`` (``r`` parts, multiplicities ``m_v``),

        E[T^n] = Σ_λ  falling(M,r)/∏m_v! · n!/∏λ_j! · ∏(2λ_j)!
                 ───────────────────────────────────────────────
                                 rise(M, 2n)

    Computed in exact rational arithmetic, so the moments carry no
    floating-point error at all. Cross-checks: ``E[T] = 2/(M+1)``
    gives ``E[SK] = 1`` identically, and the implied
    ``Var(SK) = 4M²/((M-1)(M+2)(M+3))`` agrees with the Nita & Gary
    variance used by :func:`gaussian_sk_thresholds` to 0.03 % at
    ``M = 64``.
    """
    if m < 2:
        raise ValueError(f"M={m}, expected M >= 2")
    out: list[Fraction] = []
    for n in range(1, n_max + 1):
        total = Fraction(0)
        for lam in _integer_partitions(n):
            r = len(lam)
            if r > m:
                continue
            falling = Fraction(1)
            for j in range(r):
                falling *= (m - j)
            mult = Counter(lam)
            denom_mult = 1
            for v in mult.values():
                denom_mult *= math.factorial(v)
            ways = Fraction(falling, denom_mult)
            multinom = Fraction(
                math.factorial(n),
                math.prod(math.factorial(x) for x in lam),
            )
            dirich = math.prod(math.factorial(2 * x) for x in lam)
            total += ways * multinom * dirich
        rise = Fraction(1)
        for j in range(2 * n):
            rise *= (m + j)
        out.append(total / rise)
    return out


def sk_moments(m: int) -> tuple[float, float, float, float]:
    """``(mean, variance, skewness, kurtosis)`` of SK under the null.

    Exact, from :func:`sk_raw_moments`. ``kurtosis`` is the
    NON-excess ``β₂ = μ₄/μ₂²``. The mean is 1 by construction.
    """
    t1, t2, t3, t4 = sk_raw_moments(m, 4)
    # Central moments of T.
    c2 = t2 - t1 ** 2
    c3 = t3 - 3 * t1 * t2 + 2 * t1 ** 3
    c4 = t4 - 4 * t1 * t3 + 6 * t1 ** 2 * t2 - 3 * t1 ** 4
    # SK = c·(M·T - 1); central moments scale by (c·M)^k.
    scale = Fraction(m + 1, m - 1) * m
    mu2 = c2 * scale ** 2
    mu3 = c3 * scale ** 3
    mu4 = c4 * scale ** 4
    mean = float(Fraction(m + 1, m - 1) * (m * t1 - 1))
    var = float(mu2)
    skew = float(mu3) / var ** 1.5
    kurt = float(mu4) / var ** 2
    return mean, var, skew, kurt


def _pearson4_params(
    mean: float, var: float, skew: float, kurt: float,
) -> tuple[float, float, float, float]:
    """Pearson IV ``(m_p, nu, a, lam)`` from the first four moments.

    Standard parameterisation (Heinrich 2004, "A guide to the Pearson
    type IV distribution", Eqs. 5-8), density

        f(x) ∝ [1 + ((x-lam)/a)²]^(-m_p) · exp(-nu·arctan((x-lam)/a))

    Raises ValueError if the moments do not lie in the type IV region.
    """
    b1 = skew * skew
    denom = 2.0 * kurt - 3.0 * b1 - 6.0
    if denom == 0.0:
        raise ValueError("degenerate moments: 2*beta2 - 3*beta1 - 6 == 0")
    r = 6.0 * (kurt - b1 - 1.0) / denom
    disc = 16.0 * (r - 1.0) - b1 * (r - 2.0) ** 2
    if not (r > 2.0 and disc > 0.0):
        raise ValueError(
            f"moments outside the Pearson IV region (r={r:.4f}, "
            f"disc={disc:.4f})"
        )
    m_p = (r + 2.0) / 2.0
    nu = -r * (r - 2.0) * skew / math.sqrt(disc)
    a = 0.25 * math.sqrt(var * disc)
    lam = mean - 0.25 * (r - 2.0) * math.sqrt(var) * skew
    return m_p, nu, a, lam


def pearson4_sk_thresholds(
    m: int, far: float = DEFAULT_SK_FAR,
) -> tuple[float, float]:
    """Two-sided SK thresholds from the analytic Pearson IV null.

    Nita & Gary (2010, PASP 122, 595) show the first four SK moments
    put the null in the **Pearson type IV** family for ``M >= 24``; all
    four default M's (64, 256, 1024, 4096) clear that bound. This
    routine takes the exact moments from :func:`sk_moments`, solves for
    the type IV parameters, and inverts the CDF by quadrature.

    Why prefer it to :func:`sk_thresholds`' Monte Carlo: at
    ``FAR = 1e-4`` only ~100 of the 1e6 MC trials land in each tail, so
    the MC quantile carries ~10 % sampling noise — a FIXED bias, since
    the MC is seeded, but a bias of unknown sign sitting exactly where
    the threshold lives. The quadrature has no such term.

    The MC path is retained as the cross-check, not replaced: see
    ``tests/test_rfi_flagger.py`` for the agreement assertion.

    Raises:
        ValueError: ``m < 4``, ``far`` outside ``(0, 1)``, or moments
            outside the type IV region.
    """
    if m < 4:
        raise ValueError(f"M={m}, expected M >= 4")
    if not 0.0 < far < 1.0:
        raise ValueError(f"far={far}, expected in (0, 1)")
    key = (int(m), float(far))
    if key in _SK_P4_CACHE:
        return _SK_P4_CACHE[key]

    from scipy import integrate, optimize                     # local import

    mean, var, skew, kurt = sk_moments(int(m))
    m_p, nu, a, lam = _pearson4_params(mean, var, skew, kurt)

    def log_kernel(t: float) -> float:
        return -m_p * math.log1p(t * t) - nu * math.atan(t)

    # Work in the standardised variable t = (x - lam)/a and carry the
    # kernel's log-peak out front so the exponential never overflows.
    t_peak = -nu / (2.0 * m_p)
    log_peak = log_kernel(t_peak)

    def kernel(t: float) -> float:
        return math.exp(log_kernel(t) - log_peak)

    norm, _ = integrate.quad(kernel, -math.inf, math.inf, limit=400)

    def cdf(t: float) -> float:
        val, _ = integrate.quad(kernel, -math.inf, t, limit=400)
        return val / norm

    def solve(target: float) -> float:
        lo, hi = t_peak, t_peak
        step = 1.0
        while cdf(lo) > target:
            lo -= step
            step *= 2.0
        step = 1.0
        while cdf(hi) < target:
            hi += step
            step *= 2.0
        t = optimize.brentq(lambda x: cdf(x) - target, lo, hi, xtol=1e-12)
        return lam + a * t

    out = (solve(far / 2.0), solve(1.0 - far / 2.0))
    _SK_P4_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# SK computation
# ---------------------------------------------------------------------------


def compute_sk(
    s1: torch.Tensor,
    s2: torch.Tensor,
    m: int,
    *,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Compute the SK statistic per (n_acc, ant, ch, pol) cell.

    Args:
        s1: ``Σ_t |E|²`` of shape ``[N_acc, NANTS, NCHAN, NPOL]``
            float32.
        s2: ``Σ_t |E|⁴`` of same shape / dtype as ``s1``.
        m: accumulation depth used to compute ``s1`` / ``s2``.
        eps: clamp on ``s1²`` to avoid 0/0 singularities. Cells with
            ``s1 == 0`` (e.g. masked / dead antennas at this stage)
            return ``SK = 0`` after the clamp; these are not flagged
            by the default thresholds (which are around 1 ± O(σ_SK)).

    Returns:
        SK tensor, same shape and dtype as ``s1``.

    Raises:
        ValueError: ``m < 2`` or shape mismatch.
    """
    if m < 2:
        raise ValueError(f"M={m}, expected M >= 2 for SK formula")
    if s1.shape != s2.shape:
        raise ValueError(
            f"s1 shape {tuple(s1.shape)} != s2 shape {tuple(s2.shape)}"
        )
    s1_sq = s1 * s1
    s1_sq = s1_sq.clamp_min(eps)
    sk = ((m + 1.0) / (m - 1.0)) * (m * s2 / s1_sq - 1.0)
    return sk


def sk_mask(
    s1: torch.Tensor,
    s2: torch.Tensor,
    m: int,
    *,
    far: float = DEFAULT_SK_FAR,
    sk_low: float | None = None,
    sk_high: float | None = None,
) -> torch.Tensor:
    """Boolean SK mask per (n_acc, ant, ch, pol) cell.

    Args:
        s1, s2: see :func:`compute_sk`.
        m: accumulation depth.
        far: two-sided false-alarm rate. Used only when ``sk_low`` /
            ``sk_high`` are both ``None``. Default
            :data:`DEFAULT_SK_FAR`.
        sk_low, sk_high: explicit thresholds. If either is given, both
            must be given and ``far`` is ignored.

    Returns:
        Bool tensor of shape ``[N_acc, NANTS, NCHAN, NPOL]`` —
        True where ``SK < sk_low`` or ``SK > sk_high``.

    Raises:
        ValueError: only one of ``sk_low`` / ``sk_high`` provided.
    """
    if (sk_low is None) ^ (sk_high is None):
        raise ValueError(
            "sk_low and sk_high must both be provided or both be None"
        )
    low_m, high_m = sk_masks(
        s1, s2, m, far=far, sk_low=sk_low, sk_high=sk_high)
    return low_m | high_m


def sk_masks(
    s1: torch.Tensor,
    s2: torch.Tensor,
    m: int,
    *,
    far: float = DEFAULT_SK_FAR,
    sk_low: float | None = None,
    sk_high: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(low_mask, high_mask)`` — the two sides of the SK test.

    ``low_mask`` marks ``SK < sk_low``: a **continuous** emitter that
    fills the accumulation and makes the channel power deterministic.
    ``high_mask`` marks ``SK > sk_high``: an **intermittent** emitter
    occupying less than half the accumulation. See the module
    docstring; the split is the duty-cycle diagnostic, free because
    both comparisons are already computed.

    :func:`sk_mask` is the OR of the two.
    """
    if (sk_low is None) ^ (sk_high is None):
        raise ValueError(
            "sk_low and sk_high must both be provided or both be None"
        )
    if sk_low is None:
        sk_low, sk_high = sk_thresholds(m, far)
    sk = compute_sk(s1, s2, m)
    return sk < sk_low, sk > sk_high


def sk_combined_mask(
    s1_per_m: dict[int, torch.Tensor],
    s2_per_m: dict[int, torch.Tensor],
    *,
    far: float = DEFAULT_SK_FAR,
    overrides: dict[int, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """OR-fold the per-M SK masks into one ``[NANTS, NCHAN, NPOL]``
    bool mask.

    For each M, computes the per-accumulation SK mask, then ORs across
    the leading ``N_acc`` axis (so a cell is flagged if *any*
    accumulation at *any* M trips its threshold). Finally, ORs across
    all M-values into the per-cube mask.

    Args:
        s1_per_m, s2_per_m: dicts mapping ``M`` → per-M ``S₁`` / ``S₂``
            tensors as returned by :func:`dsart.rfi.autos.compute_autos`.
            Both must have the same key set.
        far: per-(ant, ch, pol, M) FAR. Default :data:`DEFAULT_SK_FAR`.
        overrides: optional dict mapping ``M`` → ``(sk_low, sk_high)``
            explicit thresholds. Missing M's fall back to the
            ``far``-derived bounds.

    Returns:
        Bool tensor of shape ``[NANTS, NCHAN, NPOL]``.

    Raises:
        ValueError: empty input or key set mismatch.
    """
    if not s1_per_m:
        raise ValueError("s1_per_m is empty")
    if set(s1_per_m.keys()) != set(s2_per_m.keys()):
        raise ValueError(
            f"s1_per_m keys {sorted(s1_per_m.keys())} != "
            f"s2_per_m keys {sorted(s2_per_m.keys())}"
        )
    overrides = overrides or {}

    any_m, high_m = sk_combined_masks(
        s1_per_m, s2_per_m, far=far, overrides=overrides)
    del high_m
    return any_m


def sk_combined_masks(
    s1_per_m: dict[int, torch.Tensor],
    s2_per_m: dict[int, torch.Tensor],
    *,
    far: float = DEFAULT_SK_FAR,
    overrides: dict[int, tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(any_mask, high_mask)`` OR-folded over all M and N_acc.

    ``any_mask`` is what :func:`sk_combined_mask` returns — a cell
    flagged if any accumulation at any M trips either threshold.
    ``high_mask`` is the subset whose excursion was on the HIGH side
    at some M, i.e. an intermittent emitter. A cell in ``any_mask``
    but not in ``high_mask`` tripped only the low side: a continuous
    carrier.

    A cell can appear in both when different M's disagree — a burst
    that fills the short accumulations but not the long ones. That is
    information, not a contradiction, so no precedence is imposed.
    """
    if not s1_per_m:
        raise ValueError("s1_per_m is empty")
    if set(s1_per_m.keys()) != set(s2_per_m.keys()):
        raise ValueError(
            f"s1_per_m keys {sorted(s1_per_m.keys())} != "
            f"s2_per_m keys {sorted(s2_per_m.keys())}"
        )
    overrides = overrides or {}

    out: torch.Tensor | None = None
    out_high: torch.Tensor | None = None
    for m, s1_m in s1_per_m.items():
        s2_m = s2_per_m[m]
        if m in overrides:
            sk_low, sk_high = overrides[m]
            low_m, high_m = sk_masks(
                s1_m, s2_m, m, sk_low=sk_low, sk_high=sk_high)
        else:
            low_m, high_m = sk_masks(s1_m, s2_m, m, far=far)
        # OR over N_acc (leading axis) → per-(ant, ch, pol).
        mask_m = (low_m | high_m).any(dim=0)
        high_m = high_m.any(dim=0)
        out = mask_m if out is None else (out | mask_m)
        out_high = high_m if out_high is None else (out_high | high_m)
    assert out is not None and out_high is not None
    return out, out_high
