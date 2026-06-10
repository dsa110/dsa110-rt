"""Detector v1 deterministic conv-bank (plan §3.6 / §4.4).

Owns:

  - The ``Detector`` Protocol — the v1-locked detector interface (plan §0
    line 68: "the ``forward()`` interface (input cube shape, output
    candidate record shape) is locked at v1; internal architecture /
    state-dict layout can change in v2").
  - The ``DeterministicDetector`` Module — the v1 conv-bank implementation
    that satisfies the Protocol.
  - The ``boxcar_via_cumsum`` cumsum-difference primitive — the only
    allowed K_dm / K_time consumer per plan §3.6.13
    ``test_detector_conv_flops_cumsum_pin``. ``F.conv1d`` /
    ``F.avg_pool1d`` / ``F.max_pool1d``-with-stride-1-and-kernel>1 along
    the K_dm or K_time axes are forbidden — those would burn ~10× the
    FLOPs for the same output and miss the 1 TFLOPs / cube budget.

Pipeline shape (per cube):

    [T_det, N_fdm, H, W] cube (real fp16, post-Layer-1, edge-masked)
        │
        ├─ for each Kernel k in build_kernel_bank():
        │     1. spatial conv with k.image_kernel  (1×1 delta in v1, D10)
        │     2. centred boxcar over fine_DM axis, width k.k_dm_width
        │        (via boxcar_via_cumsum)
        │     3. running boxcar over time axis, width k.k_time_width
        │        (via boxcar_via_cumsum)
        │     → score tensor c_k[T_det, N_fdm, H, W] fp32
        │
        └─ stack to scores[K, T_det, N_fdm, H, W] fp32

Chunk-2 then divides by Layer-2 ``s_k`` to get SNR, runs per-kernel
local-max NMS + cross-kernel SNR-sort merger, and applies the
canonical-zone emit gate to produce the final ``List[Candidate]``.

For Chunk 1, ``forward()`` returns an empty list (the decoder stub) — the
per-kernel score tensors are exposed via ``_compute_per_kernel_scores()``
so Chunk-1 tests can validate the conv-bank math without depending on the
chunk-2 decoder.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn.functional as F

from ..common.constants import (
    CUBE_CADENCE_S_DEFAULT,
    N_KERNEL_MAX_T_DEFAULT,
    NOISE_LAYER2_N_BURNIN_DEFAULT,
    NOISE_LAYER2_TAU_S_DEFAULT,
)
from ..common.contracts import Candidate, CandidateFlags
from ..noise_norm.layer2 import Layer2State
from .decoder import (
    decode_local_max,
    decode_topk_argmax_lowmem,
    decode_topk_lowmem,
    filter_to_canonical,
)
from .kernels import (
    DEFAULT_DETECTOR_DTYPE,
    Kernel,
    build_kernel_bank,
)

_LOG = logging.getLogger(__name__)
from .merger import (
    DEFAULT_MERGE_RADIUS_FDM,
    DEFAULT_MERGE_RADIUS_LM,
    DEFAULT_MERGE_RADIUS_T,
    MergerConfig,
    merge_across_kernels,
    merge_across_kernels_c1,
)
from .triton_boxcar import boxcar_from_padded_cumsum_triton

__all__ = [
    "Detector",
    "DeterministicDetector",
    "boxcar_via_cumsum",
    "boxcar_from_padded_cumsum",
    "precompute_padded_cumsum",
]


def _with_flags(cand: Candidate, new_flags: int) -> Candidate:
    """Return a copy of ``cand`` with ``flags = new_flags``. The chunk-3
    Layer-2 burn-in path uses this to OR in NOISE_WARMUP without
    mutating the frozen Candidate dataclass."""
    return Candidate(
        l=cand.l,
        m=cand.m,
        dm_fine=cand.dm_fine,
        dm_idx=cand.dm_idx,
        event_specnum=cand.event_specnum,
        width_samples=cand.width_samples,
        kernel_id=cand.kernel_id,
        snr=cand.snr,
        detector_version=cand.detector_version,
        flags=int(new_flags),
        search_node_id=cand.search_node_id,
        gpu_half=cand.gpu_half,
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Detector(Protocol):
    """v1-locked detector interface (plan §0 line 68).

    Implementations may swap internals freely (v2 learned weights, etc.)
    so long as ``forward(cube, validity_mask, sigma_layer1)`` returns a
    ``List[Candidate]`` and ``kernels()`` returns the closed kernel-id
    enum the implementation fires on. Both methods are pure functions of
    their inputs (no per-call state mutation other than the Layer-2 σ_k
    EMA, which is owned by the implementation).

    Args (forward):
        cube: ``[T_det, N_fdm, H, W]`` real cube (Stokes I, post-Layer-1,
            edge-masked). Default dtype is fp16 per §3.6.11 / §3.6.12.
            ``H == W == N_grid``. ``T_det`` is the locked
            ``T_DET_SAMPLES_DEFAULT = 512`` at default ops.
        validity_mask: ``[T_det, N_fdm]`` bool. ``False`` on warmup or
            RFI'd cubes; the detector skips those slices for both forward
            and Layer-2 EMA updates.
        sigma_layer1: ``[T_det, N_fdm]`` float32. The Layer-1 σ used to
            normalise the input cube. Passed through for v2 detectors that
            may want it; v1 ignores it (Layer-1 normalization happens
            upstream in ``noise_norm/layer1.py``).

    Returns:
        ``List[Candidate]`` — final emitted candidates after the per-kernel
        local-max NMS, cross-kernel merge, and canonical-zone emit gate.
        Empty list on warmup cubes or when no candidate exceeds the
        per-kernel threshold.
    """

    def forward(
        self,
        cube: torch.Tensor,
        validity_mask: torch.Tensor,
        sigma_layer1: torch.Tensor,
    ) -> List[Candidate]: ...

    def kernels(self) -> Tuple[str, ...]: ...


# ---------------------------------------------------------------------------
# boxcar_via_cumsum — the only allowed K_dm / K_time consumer
# ---------------------------------------------------------------------------


def boxcar_via_cumsum(
    x: torch.Tensor,
    *,
    axis: int,
    width: int,
    tile_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply an unweighted centred boxcar of integer ``width`` along
    ``axis`` via the cumsum-difference primitive.

    For width = 1 this is a no-op (returns ``x`` unchanged).

    For width > 1, the output at index ``i`` is
    ``sum(x[i - width//2 : i - width//2 + width])`` along ``axis`` —
    i.e. a centred sum-boxcar (NOT a mean — the unweighted-sum form
    matches the §3.6.10 Layer-2 σ_k EMA pin which absorbs the
    ``√width`` amplitude scaling). For odd widths the centre is exact;
    for even widths the centre is biased by half a sample to the LEFT
    (the canonical convention is irrelevant here because v1 K_dm widths
    are odd and v1 K_time widths apply running-boxcar semantics which the
    Layer-2 σ_k EMA absorbs).

    Implementation: pad with zeros, take cumulative sum, take the
    difference of cumsums offset by ``width``, slice back to the original
    shape. This is O(N) FLOPs total regardless of ``width``, vs O(N·width)
    for the naive unfold-and-sum form.

    Plan §3.6.13 ``test_detector_conv_flops_cumsum_pin`` invariants:
      - This is the **only** allowed implementation of K_dm / K_time
        boxcars in ``forward.py`` and ``decoder.py``. The companion
        AST-scan test in ``tests/test_detector_protocol.py`` verifies
        ``F.conv1d`` / ``F.avg_pool1d`` / ``F.max_pool1d``-with-stride-1
        do not appear along the K_dm or K_time axes.
      - Numerical equivalence to ``numpy.lib.stride_tricks.sliding_window_view
        + numpy.sum`` to fp16 rel-err ≤ 1e-3.
      - Output shape == input shape (centred padding).

    Args:
        x: Input tensor. Any shape, any dtype that supports ``cumsum``.
        axis: Axis along which to apply the boxcar.
        width: Boxcar width in samples. Must be ≥ 1.
        tile_size: When set and ``axis != x.ndim - 1`` and ``x.ndim >= 2``,
            process the input in ``tile_size``-column chunks along the
            LAST axis to bound the fp32 cumsum working set. Bit-exact
            equivalent to the untiled output (cumsum is associative
            across tile boundaries that are along an axis DIFFERENT
            from the cumsum axis). Use this on memory-constrained GPUs
            at production geometry (T_det=256, N_fdm=32, N_grid=256):
            untiled, the fp32 cumsum + cat + diff transients peak at
            ~9 GiB; tile_size=64 caps them at ~768 MiB. Default None =
            untiled (current/historical behavior).

    Returns:
        Boxcar-summed tensor of the same shape and dtype as ``x``.
    """
    if width < 1:
        raise ValueError(f"width={width}, expected ≥ 1")
    if width == 1:
        return x

    n = x.shape[axis]
    if width > n:
        raise ValueError(
            f"width={width} exceeds axis-{axis} length {n}; cannot apply "
            f"centred boxcar"
        )

    ndim = x.dim()
    if axis < 0:
        axis = ndim + axis

    if tile_size is not None and ndim >= 2 and axis != ndim - 1:
        last_dim = ndim - 1
        out = torch.empty_like(x)
        for w0 in range(0, x.shape[last_dim], int(tile_size)):
            w1 = min(w0 + int(tile_size), x.shape[last_dim])
            tile_in = x.narrow(last_dim, w0, w1 - w0)
            tile_out = _boxcar_via_cumsum_untiled(
                tile_in, axis=axis, width=width,
            )
            out.narrow(last_dim, w0, w1 - w0).copy_(tile_out)
            del tile_in, tile_out
        return out

    return _boxcar_via_cumsum_untiled(x, axis=axis, width=width)


def precompute_padded_cumsum(
    x: torch.Tensor,
    *,
    axis: int,
    max_width: int,
    accum_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Pre-compute the zero-prepended, max-width-padded cumsum that the
    :func:`boxcar_from_padded_cumsum` shortcut consumes.

    For the v1 detector kernel-bank where every kernel shares a delta
    image + delta DM kernel, the entire bank's per-kernel time-axis
    boxcar can be expressed as ``W`` narrow-subtractions over a SINGLE
    cumsum of the cube (no per-kernel cumsum work). This helper builds
    that one shared cumsum once per pass; ``boxcar_from_padded_cumsum``
    then computes any boxcar with width ``≤ max_width`` for free
    (modulo a single fp32 -> output_dtype subtract and cast).

    Memory model: the returned cumsum is ``fp32`` with shape ``x.shape``
    extended along ``axis`` by ``max_width`` (``max_width // 2`` zero
    pad-left, ``max_width - 1 - max_width // 2`` zero pad-right, ``+1``
    for the prepended-zero row) — for the production geometry
    ``[T_det=256, N_fdm=32, N_grid=256]`` fp16 cube + ``max_width=128``
    that is ``(384 + 1) * 32 * 256 * 256 * 4 B ≈ 3.0 GiB``. This fits
    on an 11 GiB 2080 Ti alongside the fp16 cube (1 GiB) and per-kernel
    fp16 score transient (~1 GiB).

    Args:
        x: Input tensor (any dtype with cumsum support).
        axis: Cumsum axis.
        max_width: Largest boxcar width that will be queried via
            :func:`boxcar_from_padded_cumsum`. The pad geometry is
            sized for this width.
        accum_dtype: Cumsum accumulation dtype. ``None`` -> ``fp32``
            for fp16/bf16 inputs, ``x.dtype`` otherwise.

    Returns:
        ``cs`` tensor of shape ``x.shape`` with axis-``axis`` length
        equal to ``x.shape[axis] + max_width + 1``. ``cs[i] = sum_{j<i}
        padded[j]`` along ``axis``, with the leading ``max_width//2``
        and trailing ``max_width - 1 - max_width//2`` rows zero-padded
        (so off-edge boxcar accesses read partial sums, equivalent to
        the chunk-1 ``F.pad(..., 0)`` boundary).
    """
    if max_width < 1:
        raise ValueError(f"max_width={max_width}, expected ≥ 1")
    n = x.shape[axis]
    if max_width > n:
        raise ValueError(
            f"max_width={max_width} exceeds axis-{axis} length {n}"
        )

    pad_left = max_width // 2
    pad_right = max_width - 1 - pad_left

    if accum_dtype is None:
        accum_dtype = (
            torch.float32
            if x.dtype in (torch.float16, torch.bfloat16)
            else x.dtype
        )

    # Build the padded-cumsum tensor with ZERO transient allocation
    # beyond ``cs`` itself. The chunk-1 ``F.pad + cast + cumsum + cat``
    # recipe blows ~12 GiB transient at production geometry; allocating
    # an intermediate ``cube_accum`` adds another ~2.15 GiB. ``torch.
    # cumsum`` accepts an ``out=`` argument that writes the cumsum
    # directly into a pre-allocated slice of ``cs``, so no additional
    # buffer is needed. Memory profile here:
    #
    #   cs zeros        : (n + max_width, ...) accum_dtype  (~3.0 GiB)
    #   peak (during construction) ............................ ~3.0 GiB
    #
    # ``cs`` itself is the only allocation; the cumsum + right-tail
    # constant-fill both write through views of ``cs``.
    out_shape = list(x.shape)
    out_shape[axis] = n + max_width
    cs = torch.zeros(out_shape, dtype=accum_dtype, device=x.device)
    target = cs.narrow(axis, 1 + pad_left, n)
    torch.cumsum(x, dim=axis, dtype=accum_dtype, out=target)
    if pad_right > 0:
        # Fill the right padding with the final cumulative sum so
        # boxcar windows that overhang the right edge see a constant
        # cs (equivalent to padding the input with zeros on the right).
        last_row = cs.narrow(axis, 1 + pad_left + n - 1, 1)
        cs.narrow(axis, 1 + pad_left + n, pad_right).copy_(last_row)
    return cs


def boxcar_from_padded_cumsum(
    cs: torch.Tensor,
    *,
    axis: int,
    width: int,
    max_width: int,
    n_out: int,
    t_base: int = 0,
    out_dtype: Optional[torch.dtype] = None,
    w_tile_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute a centred boxcar of ``width`` over the original input
    using a cumsum produced by :func:`precompute_padded_cumsum` with
    ``max_width``. ``O(N)`` total work; no cumsum re-computation.

    For ``width = 1`` returns a no-op narrow back to the original
    range — the unit boxcar is identity but we still produce a tensor
    of shape ``[..., n_out, ...]`` for caller convenience.

    Args:
        cs: Output of :func:`precompute_padded_cumsum`.
        axis: Cumsum axis.
        width: Desired boxcar width (``≤ max_width``).
        max_width: ``max_width`` used to build ``cs``.
        n_out: Output length along ``axis`` (typically the original
            input's axis-``axis`` length, e.g. ``T_det``).
        t_base: Start index in the original unpadded axis for the
            returned samples. ``0`` means emit from the first sample;
            positive values allow interior-only scoring windows.
        out_dtype: Optional cast for the returned tensor; ``None``
            keeps ``cs.dtype``.
        w_tile_size: When set and ``axis != cs.ndim - 1`` and ``cs.ndim
            >= 2``, write the output one ``w_tile_size``-column tile
            of the last axis at a time; bounds the fp32 ``high - low``
            transient at ~``w_tile_size × n_out × prefix_axes × 4 B``.
            At production geometry (T=256, F=32, H=W=256) the untiled
            fp32 ``high - low`` is ~2 GiB; ``w_tile_size=64`` caps it
            at ~16 MiB.

    Returns:
        Boxcar-summed tensor with the same shape as ``cs`` except axis
        ``axis`` is ``n_out``.
    """
    if width < 1 or width > max_width:
        raise ValueError(
            f"width={width} not in [1, max_width={max_width}]"
        )
    if t_base < 0:
        raise ValueError(f"t_base must be >= 0; got {t_base}")

    # CUDA fast path (axis=0): fused subtract(+cast) Triton kernel.
    # Fallback to torch path on unsupported dtypes/layouts/devices.
    triton_out = boxcar_from_padded_cumsum_triton(
        cs,
        axis=axis,
        width=width,
        max_width=max_width,
        n_out=n_out,
        t_base=t_base,
        out_dtype=out_dtype,
    )
    if triton_out is not None:
        return triton_out

    pad_left_full = max_width // 2
    pad_left_w = width // 2
    offset = pad_left_full - pad_left_w
    start = t_base + offset
    max_hi = start + width + n_out
    if max_hi > int(cs.shape[axis]):
        raise ValueError(
            "Requested (t_base, n_out, width) window exceeds padded cumsum bounds"
        )

    untiled = (
        w_tile_size is None
        or cs.ndim < 2
        or axis == cs.ndim - 1
    )
    if untiled:
        if width == 1:
            out = cs.narrow(axis, start + 1, n_out) - cs.narrow(axis, start, n_out)
        else:
            high = cs.narrow(axis, start + width, n_out)
            low = cs.narrow(axis, start, n_out)
            out = high - low
        if out_dtype is not None and out.dtype != out_dtype:
            out = out.to(out_dtype)
        return out

    # Tiled along the last axis (W) to bound the fp32 ``high - low``
    # transient. We pre-allocate the output (fp16 in production) and
    # write each tile via a tile-local subtract+cast.
    last_dim = cs.ndim - 1
    out_shape = list(cs.shape)
    out_shape[axis] = n_out
    eff_dtype = out_dtype if out_dtype is not None else cs.dtype
    out = torch.empty(out_shape, dtype=eff_dtype, device=cs.device)
    for w0 in range(0, cs.shape[last_dim], int(w_tile_size)):
        w1 = min(w0 + int(w_tile_size), cs.shape[last_dim])
        cs_tile = cs.narrow(last_dim, w0, w1 - w0)
        if width == 1:
            tile = (
                cs_tile.narrow(axis, start + 1, n_out)
                - cs_tile.narrow(axis, start, n_out)
            )
        else:
            high = cs_tile.narrow(axis, start + width, n_out)
            low = cs_tile.narrow(axis, start, n_out)
            tile = high - low
        if tile.dtype != eff_dtype:
            tile = tile.to(eff_dtype)
        out.narrow(last_dim, w0, w1 - w0).copy_(tile)
        del cs_tile, tile
    return out


def _boxcar_via_cumsum_untiled(
    x: torch.Tensor,
    *,
    axis: int,
    width: int,
) -> torch.Tensor:
    """Untiled core of :func:`boxcar_via_cumsum`. Always executes a single
    pad+cumsum+diff over the full input. Callers wanting bounded fp32
    working-set should use ``boxcar_via_cumsum(..., tile_size=...)``.
    """
    n = x.shape[axis]

    # Half-widths for centred boxcar. For even width the centre is biased
    # left (this matches numpy.cumsum-difference convention; v1 K_dm
    # widths are odd, K_time widths are powers of two but applied as
    # running-sum semantics so the bias is absorbed by Layer-2).
    pad_left = width // 2
    pad_right = width - 1 - pad_left

    # Pad with zeros along ``axis`` only. F.pad expects (last_axis_pad_l,
    # last_axis_pad_r, ..., first_axis_pad_l, first_axis_pad_r) so we
    # construct the pad spec carefully.
    ndim = x.dim()
    pad_spec = [0] * (2 * ndim)
    pad_spec[2 * (ndim - 1 - axis)] = pad_left
    pad_spec[2 * (ndim - 1 - axis) + 1] = pad_right
    x_padded = F.pad(x, pad_spec, mode="constant", value=0.0)

    # Accumulate the cumsum in fp32 even when the input is fp16 — the
    # plan §3.6.13 pin requires fp16 rel-err ≤ 1e-3 vs the fp32 numpy
    # reference, which a pure-fp16 cumsum across the T_det = 512 axis
    # cannot meet (~5e-3 p99 with naive fp16 cumsum). Casting once at
    # entry and once at the boundary is cheap (≤ 2× memory traffic on
    # the cumsum buffers, no extra compute since cumsum is bandwidth-
    # bound) and is what keeps the K_time = 128 boxcar within spec.
    accum_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    cs = torch.cumsum(x_padded.to(accum_dtype), dim=axis)
    zero_shape = list(cs.shape)
    zero_shape[axis] = 1
    zero = torch.zeros(zero_shape, dtype=cs.dtype, device=cs.device)
    cs = torch.cat([zero, cs], dim=axis)

    high = cs.narrow(axis, width, n)
    low = cs.narrow(axis, 0, n)
    out = high - low

    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out


# ---------------------------------------------------------------------------
# DeterministicDetector — v1 conv-bank implementation
# ---------------------------------------------------------------------------


class DeterministicDetector(torch.nn.Module):
    """v1 deterministic conv-bank detector (plan §3.6 / §4.4).

    Constructs the K=128 default kernel bank (D2 lock) at __init__, runs
    the conv-bank over each input cube, and (in chunk 2+) decodes per-kernel
    local maxima → cross-kernel merges → applies the canonical-zone emit
    gate. The Protocol-level ``forward()`` returns a ``List[Candidate]``;
    the per-kernel score tensor is exposed via the private
    ``_compute_per_kernel_scores()`` method for chunk-1 tests + chunk-3
    Layer-2 σ_k EMA consumers.

    Args:
        kernel_bank: Optional pre-built kernel bank (default: full 128-tuple
            from ``build_kernel_bank()``).
        threshold_sigma: Detection threshold in σ (default 8.0 per
            ``config_compute_search.yaml``). Currently consumed only by
            chunk-2 decoder; passed for API stability.
        detector_version: String version stamp written to emitted
            ``Candidate.detector_version`` records. Default ``"v1.M5"``;
            the cube-injection bench's identity-stub mode (M5 DoD §8 line
            2321) overrides this to ``"identity-stub.M5"``.
        device: torch device for the kernel-bank tensors (default: CPU; the
            production service moves the whole module to ``cuda:0``).
        dtype: Image-kernel dtype (default ``torch.float16``).
    """

    def __init__(
        self,
        *,
        kernel_bank: Optional[Tuple[Kernel, ...]] = None,
        threshold_sigma: float = 8.0,
        detector_version: str = "v1.M5",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = DEFAULT_DETECTOR_DTYPE,
        merge_radius_lm: int = DEFAULT_MERGE_RADIUS_LM,
        merge_radius_fdm: int = DEFAULT_MERGE_RADIUS_FDM,
        merge_radius_t: int = DEFAULT_MERGE_RADIUS_T,
        merger_config: Optional[MergerConfig] = None,
        c1_snr_min: Optional[float] = None,
        search_node_id: int = 0,
        gpu_half: int = 0,
        cube_cadence_s: float = CUBE_CADENCE_S_DEFAULT,
        layer2_tau_s: float = NOISE_LAYER2_TAU_S_DEFAULT,
        layer2_n_burnin: int = NOISE_LAYER2_N_BURNIN_DEFAULT,
        n_kernel_max_t: int = N_KERNEL_MAX_T_DEFAULT,
        layer2_state: Optional[Layer2State] = None,
        layer2_seed_unit: bool = True,
        layer2_sigma_floor: float = 0.0,
        layer2_sigma_max_ratio: float = 0.0,
        layer2_clamp_escape_cubes: int = 0,
        layer2_valid_min_fraction: float = 1.0,
        streaming: bool = False,
        streaming_tile_size: int = 64,
        layer2_sigma_max_samples: Optional[int] = 1_000_000,
        streaming_decoder: str = "topk_lowmem",
        streaming_decoder_n_top: int = 64,
        boxcar_accum_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Streaming forward (chunk-8 production refactor): when True,
        # ``forward()`` dispatches to ``_streaming_forward()`` which
        # processes kernels one at a time so the per-kernel score
        # tensor (~1-2 GiB at production geometry T_det=256, N_fdm=32,
        # N_grid=256, fp16) is allocated and freed per kernel instead
        # of held in a [K, T, F, H, W] batched buffer (16 GiB for K=8,
        # 256 GiB for K=128 — won't fit on an 11 GiB 2080 Ti). The
        # production search-compute service (chunk-6) sets streaming=
        # True; the chunk-1/2 unit tests + small-geom benches keep the
        # batched (default False) path so their existing assertions
        # over the K-batched score tensor stay valid.
        self._streaming = bool(streaming)
        self._streaming_tile_size = int(streaming_tile_size)
        # Pre-allocated padded-cumsum scratch for the v1-collapsed-bank
        # amortise fast-path; built lazily on first cube and reused
        # across cubes (saves a ~3 GiB malloc/free per cube).
        self._amortise_cs: Optional[torch.Tensor] = None
        # Layer-2 σ-clip per-kernel subsample-index cache: built lazily
        # on first cube and reused across cubes so the Layer-2 σ
        # estimator's ``randint`` cost is paid once per pipeline
        # lifetime, not K × per-cube. ``(key, idx)`` where ``key``
        # encodes the interior-slab shape + device + max_samples;
        # ``idx`` is a ``[K, max_samples] int64`` cuda tensor of flat
        # indices into the interior slab.
        self._layer2_idx: Optional[torch.Tensor] = None
        self._layer2_idx_key: Optional[Tuple] = None
        # Optional override of the cumsum accumulation dtype for the
        # amortise fast-path. Default ``None`` preserves the chunk-8
        # behaviour (``fp32`` accum when ``cube.dtype`` is ``fp16``/
        # ``bf16``). Setting to ``torch.float16`` opts into the
        # commissioning fast-path where the entire cs buffer is stored
        # in fp16 — halves the memory traffic of ``boxcar_from_padded_
        # cumsum`` (saves ~100 ms / pass at production geometry on a
        # 2080 Ti) at the cost of accumulated fp16 rounding in the
        # cumsum. Empirically (test/precision_fp16_cumsum on n01) the
        # sigma_clipped_std output differs by ≤ 0.02% across all 7
        # K_time widths at T_det=256 / N_fdm=34 / N_grid=256 — 75x
        # below the Layer-2 EMA's intrinsic 1.5% cube-to-cube noise
        # floor (plan §1622). Production search_compute on the 2080 Ti
        # pin sets this to fp16; small-geom unit tests leave it None.
        self._boxcar_accum_dtype: Optional[torch.dtype] = boxcar_accum_dtype
        if (
            boxcar_accum_dtype is not None
            and boxcar_accum_dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                f"boxcar_accum_dtype={boxcar_accum_dtype!r}; expected one of "
                "torch.float16, torch.bfloat16, torch.float32, or None"
            )
        if self._streaming and self._streaming_tile_size < 1:
            raise ValueError(
                f"streaming_tile_size={streaming_tile_size}, expected ≥ 1"
            )
        # Layer-2 σ-clip subsample cap (production knob): the per-kernel
        # interior σ-clipped std at production geometry would otherwise
        # call torch.median on a 2.05 M-cell fp32 tensor, allocating
        # ~8 GiB of sort workspace. With max_samples=1_000_000 (≈ 50 %
        # of the interior at T_det=256 / N_fdm=32 / N_grid=256), the
        # estimator standard error is σ̂ / √(2 N) ≈ 7e-4 σ — well
        # below the EMA cube-to-cube noise floor — and the working set
        # collapses to ~16 MiB. Pass None to disable subsampling
        # (chunk-1 / chunk-3 unit-test path).
        self._layer2_sigma_max_samples = (
            int(layer2_sigma_max_samples)
            if layer2_sigma_max_samples is not None else None
        )
        # Streaming-forward decoder selection. ``decode_local_max`` is
        # the strict 4D local-max NMS but at production geometry its
        # ``permute().reshape()`` + ``F.max_pool1d`` workspace blows
        # past the 11 GiB 2080 Ti budget; ``decode_topk_lowmem`` is a
        # strict subset that uses ``torch.topk`` + per-kernel NMS-radius
        # de-duplication with ~16 MiB workspace. The cross-kernel
        # merger handles cross-kernel deduplication regardless, and the
        # production PerCubePerKernelCap caps emitter dispatch at 4 per
        # kernel per cube, so n_top=64 leaves enough headroom for the
        # lowmem path to match the strict path's emitter output in
        # practice. Pass "local_max" to opt back into the strict NMS
        # (small geometries only).
        if streaming_decoder not in ("local_max", "topk_lowmem"):
            raise ValueError(
                f"streaming_decoder={streaming_decoder!r}; expected "
                f"'local_max' or 'topk_lowmem'"
            )
        self._streaming_decoder = str(streaming_decoder)
        self._streaming_decoder_n_top = int(streaming_decoder_n_top)
        if self._streaming_decoder_n_top < 1:
            raise ValueError(
                f"streaming_decoder_n_top={streaming_decoder_n_top}, expected ≥ 1"
            )
        self._kernel_bank: Tuple[Kernel, ...] = kernel_bank or build_kernel_bank(
            dtype=dtype
        )
        # C1 unification (M7.4): when ``c1_snr_min`` is set, it overrides
        # ``threshold_sigma`` for the per-kernel NMS threshold AND is
        # used as a defensive final-emit floor in
        # ``_apply_c1_emit_floor``. The single knob is the C1 SNR
        # floor; legacy callers may still pass ``threshold_sigma`` alone.
        if c1_snr_min is not None:
            threshold_sigma = float(c1_snr_min)
        self._threshold_sigma = float(threshold_sigma)
        self._c1_snr_min: Optional[float] = (
            float(c1_snr_min) if c1_snr_min is not None else None
        )
        self._detector_version = str(detector_version)
        self._dtype = dtype
        self._merge_radius_lm = int(merge_radius_lm)
        self._merge_radius_fdm = int(merge_radius_fdm)
        self._merge_radius_t = int(merge_radius_t)
        self._merger_config: Optional[MergerConfig] = merger_config
        self._search_node_id = int(search_node_id)
        self._gpu_half = int(gpu_half)
        self._n_kernel_max_t = int(n_kernel_max_t)

        # M7.4 hardening: validity-mask threshold for gating Layer-2
        # σ_k EMA updates. Legacy behaviour required 100% valid cells
        # (``layer2_valid_min_fraction=1.0``); this is too strict in
        # the field where 1–2% of UV cells fail per cube — Layer-2
        # σ_k then never updates and stays at the seed value, which
        # diverges across nodes (see 250924mptq postmortem).
        # Set to e.g. 0.95 to allow updates while still rejecting
        # cubes with significant flag fractions. Per-cube valid
        # fraction is ``validity_mask.sum() / numel``.
        if not (0.0 <= layer2_valid_min_fraction <= 1.0):
            raise ValueError(
                f"layer2_valid_min_fraction={layer2_valid_min_fraction}, "
                f"expected in [0.0, 1.0]"
            )
        self._layer2_valid_min_fraction = float(layer2_valid_min_fraction)

        # Layer-2 σ_k EMA (Chunk 3, plan §3.6.10). On cold start the EMA
        # buffer is seeded to the analytic value for unit-σ Gaussian
        # input — sqrt(k_dm_width × k_time_width) — so the very first
        # cube emitted (before the burn-in completes) divides by a
        # sensible scalar instead of 1.0. Tests / benches that want
        # canonical Welford behaviour from cold can pass
        # layer2_seed_unit=False to start at 1.0.
        if layer2_state is None:
            layer2_state = Layer2State(
                n_kernels=len(self._kernel_bank),
                cube_cadence_s=cube_cadence_s,
                tau_s=layer2_tau_s,
                n_burnin=int(layer2_n_burnin),
                n_kernel_max_t=int(n_kernel_max_t),
                sigma_max_samples=self._layer2_sigma_max_samples,
                sigma_floor=float(layer2_sigma_floor),
                sigma_max_ratio=float(layer2_sigma_max_ratio),
                clamp_escape_cubes=int(layer2_clamp_escape_cubes),
                device=device,
            )
            if layer2_seed_unit:
                seed = torch.tensor(
                    [
                        math.sqrt(k.k_dm_width * k.k_time_width)
                        for k in self._kernel_bank
                    ],
                    dtype=layer2_state._s_k.dtype,
                    device=layer2_state._s_k.device,
                )
                layer2_state._s_k.copy_(seed)
        self._layer2 = layer2_state
        # Mirror current σ_k as a registered buffer so .to(device) /
        # state_dict round-trip the EMA tensor (the Module owns the
        # state via register_buffer; Layer2State.update_and_query
        # writes into the same underlying tensor).
        self.register_buffer(
            "_sigma_k", layer2_state._s_k, persistent=False
        )

        # Move image kernels to ``device`` if given, and register them as
        # buffers so torch.nn.Module's .to() / .cuda() / state_dict
        # plumbing handles them. We register one buffer per unique image
        # kernel rather than per kernel triple (in v1, all 4 image
        # tokens map to the same delta tensor; chunk-2 may grow this).
        self._image_kernel_buffers: dict[str, torch.Tensor] = {}
        seen: dict[int, str] = {}
        for k in self._kernel_bank:
            tensor_id = id(k.image_kernel)
            if tensor_id not in seen:
                buf_name = f"_imgker_{k.image_token}"
                tensor = k.image_kernel
                if device is not None:
                    tensor = tensor.to(device)
                # Cast to declared dtype if necessary (build_kernel_bank
                # may have used a different default).
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                self.register_buffer(buf_name, tensor, persistent=False)
                self._image_kernel_buffers[k.image_token] = buf_name
                seen[tensor_id] = buf_name
            else:
                self._image_kernel_buffers[k.image_token] = seen[tensor_id]

    # -----------------------------------------------------------------
    # Protocol surface
    # -----------------------------------------------------------------

    def kernels(self) -> Tuple[str, ...]:
        """Return the closed kernel-id tuple this detector fires on."""
        return tuple(k.kernel_id for k in self._kernel_bank)

    @property
    def detector_version(self) -> str:
        return self._detector_version

    @property
    def threshold_sigma(self) -> float:
        return self._threshold_sigma

    @property
    def merger_config(self) -> Optional[MergerConfig]:
        """C1 merger geometry (``None`` ⇒ legacy axis-AND merger)."""
        return self._merger_config

    @property
    def c1_snr_min(self) -> Optional[float]:
        """C1 SNR floor (single knob used for NMS threshold + emit gate)."""
        return self._c1_snr_min

    def _merge_per_kernel_cands(
        self, per_kernel_cands: List[Candidate]
    ) -> List[Candidate]:
        """Apply the configured cross-kernel merger to a flat per-kernel
        candidate list. Routes to ``merge_across_kernels_c1`` when a
        :class:`MergerConfig` is wired (M7.4 C1 path) and to the legacy
        axis-AND ``merge_across_kernels`` otherwise.
        """
        if self._merger_config is not None:
            return merge_across_kernels_c1(per_kernel_cands, self._merger_config)
        return merge_across_kernels(
            per_kernel_cands,
            merge_radius_lm=self._merge_radius_lm,
            merge_radius_fdm=self._merge_radius_fdm,
            merge_radius_t=self._merge_radius_t,
        )

    def _apply_c1_emit_floor(
        self, cands: List[Candidate]
    ) -> List[Candidate]:
        """Defensive C1 emit-floor (``c1_snr_min``). The per-kernel
        decoder already drops anything ≤ ``threshold_sigma``; this is a
        belt-and-braces filter so the C1 emitter never ships a sub-SNR
        candidate even if a future decoder change weakens the inner
        threshold. No-op when ``c1_snr_min`` is unset."""
        if self._c1_snr_min is None:
            return cands
        floor = float(self._c1_snr_min)
        return [c for c in cands if float(c.snr) >= floor]

    @property
    def layer2_state(self) -> Layer2State:
        """The Layer-2 σ_k EMA state machine. Exposed for telemetry /
        bench introspection (e.g. asserting ``cube_count`` /
        ``is_warming_up`` / ``s_k`` after a sequence of cubes); the hot
        path consumes it via ``forward()`` directly."""
        return self._layer2

    @property
    def kernel_bank(self) -> Tuple[Kernel, ...]:
        """Return the underlying ``Kernel`` records (not just the ids).

        Chunk-3 Layer-2 σ_k EMA consumers use this to size the ``s_k``
        ring per-kernel; chunk-2 decoder uses it to map kernel index to
        ``Kernel.k_dm_width`` / ``k_time_width`` for the merge-radius
        calculation.
        """
        return self._kernel_bank

    def forward(
        self,
        cube: torch.Tensor,
        validity_mask: torch.Tensor,
        sigma_layer1: torch.Tensor,
        *,
        dm_idx_canonical_lo: Optional[int] = None,
        dm_idx_canonical_hi: Optional[int] = None,
        n_kernel_max_t: Optional[int] = None,
        event_specnum: int = 0,
        fine_to_coarse: Optional[torch.Tensor] = None,
        fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    ) -> List[Candidate]:
        """v1-locked Protocol entrypoint. Returns final emitted Candidates.

        Chunk 2 lands the per-kernel local-max NMS (``decoder.py``) +
        cross-kernel SNR-sort merger (``merger.py``) + canonical-zone
        emit gate. Chunk 3 wires the per-kernel σ_k EMA via
        ``Layer2State`` (registered buffer ``_sigma_k`` mirrors the
        EMA-tracked tensor for state_dict round-trip) and sets
        ``flags.bit3 = noise_warmup`` on every emitted Candidate while
        the EMA is still in burn-in.

        The kwarg-only fields are M5-internal extensions (NOT on the
        Protocol surface — Chunk 6 search_compute owns them and passes
        them in from the cube/DmPlan context):

          - ``dm_idx_canonical_lo`` / ``dm_idx_canonical_hi``: this
            (search_node, gpu_half)'s canonical fine-DM range. If both
            are None, the gate is permissive (no halo drops) — the
            Chunk-2 cube-injection unit tests use this default.
          - ``n_kernel_max_t``: widest time-kernel boxcar width used by
            the time-edge gate. Defaults to the max of the bank's
            k_time_width values.
          - ``event_specnum``: cube-start specnum (stamped onto every
            emitted Candidate as ``event_specnum + t_in_cube``). Default
            0 for unit tests; Chunk-6 search_compute passes the real
            cube start.
          - ``fine_to_coarse`` / ``fine_dm_pc_cm3``: ``DmPlan`` lookup
            tables. If None, decoder uses unit-stub fallbacks (Chunk-2
            unit-test path).

        When ``self._streaming`` is True (production search-compute
        default at production geometry), this dispatches to
        ``_streaming_forward`` which produces semantically equivalent
        candidates with bounded per-kernel memory (chunk-8 production
        refactor); see ``_streaming_forward`` docstring.
        """
        if self._streaming:
            return self._streaming_forward(
                cube, validity_mask, sigma_layer1,
                dm_idx_canonical_lo=dm_idx_canonical_lo,
                dm_idx_canonical_hi=dm_idx_canonical_hi,
                n_kernel_max_t=n_kernel_max_t,
                event_specnum=event_specnum,
                fine_to_coarse=fine_to_coarse,
                fine_dm_pc_cm3=fine_dm_pc_cm3,
            )
        scores = self._compute_per_kernel_scores(cube, validity_mask)

        # Determine cube validity: any (T_det, N_fdm) cell False in the
        # mask flags the cube as suspect. Plan §319 says invalid cubes
        # skip BOTH forward and noise updates; here we still run forward
        # (the candidate log records what the detector saw) but skip
        # the EMA update so a contaminated cube doesn't poison σ_k.
        # M7.4: allow Layer-2 EMA updates when the valid fraction
        # crosses a tunable threshold (default still 100% to preserve
        # legacy behaviour). When relaxed, a small fraction of
        # masked/flagged cells does not prevent σ_k learning, which
        # was empirically required to converge σ_k across nodes on
        # the 250924mptq replay.
        cube_valid = self._compute_cube_valid(validity_mask)
        s_k_tensor, is_warming_up = self._layer2.update_and_query(
            scores=scores, valid=cube_valid,
        )
        # Mirror back into the registered buffer so state_dict / .to()
        # round-trip the latest EMA value.
        self._sigma_k.copy_(s_k_tensor)

        s_k = s_k_tensor.view(-1, 1, 1, 1, 1).to(scores.dtype)
        scores_snr = scores / s_k

        if n_kernel_max_t is None:
            n_kernel_max_t = self._n_kernel_max_t

        warmup_flag = int(CandidateFlags.NOISE_WARMUP) if is_warming_up else 0

        per_kernel_cands: List[Candidate] = []
        for k_idx, kernel in enumerate(self._kernel_bank):
            cands = decode_local_max(
                scores_snr[k_idx],
                threshold=self._threshold_sigma,
                kernel_id=kernel.kernel_id,
                k_dm_width=kernel.k_dm_width,
                k_time_width=kernel.k_time_width,
                detector_version=self._detector_version,
                search_node_id=self._search_node_id,
                gpu_half=self._gpu_half,
                event_specnum=event_specnum,
                fine_to_coarse=fine_to_coarse,
                fine_dm_pc_cm3=fine_dm_pc_cm3,
            )
            if warmup_flag:
                cands = [_with_flags(c, c.flags | warmup_flag) for c in cands]
            per_kernel_cands.extend(cands)

        merged = self._merge_per_kernel_cands(per_kernel_cands)

        if dm_idx_canonical_lo is None or dm_idx_canonical_hi is None:
            return self._apply_c1_emit_floor(merged)
        emit, _dropped = filter_to_canonical(
            merged,
            dm_idx_canonical_lo=dm_idx_canonical_lo,
            dm_idx_canonical_hi=dm_idx_canonical_hi,
            t_det=cube.shape[0],
            n_kernel_max_t=n_kernel_max_t,
            cube_t_offset=event_specnum,
        )
        return self._apply_c1_emit_floor(emit)

    # -----------------------------------------------------------------
    # Internals (chunk-1 testable)
    # -----------------------------------------------------------------

    def _validate_cube(
        self,
        cube: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> None:
        if cube.dim() != 4:
            raise ValueError(
                f"cube.dim()={cube.dim()}, expected 4 "
                f"[T_det, N_fdm, H, W]; got shape {tuple(cube.shape)}"
            )
        if cube.shape[2] != cube.shape[3]:
            raise ValueError(
                f"cube spatial axes must be square (H == W); got "
                f"H={cube.shape[2]}, W={cube.shape[3]}"
            )
        if validity_mask.dim() != 2:
            raise ValueError(
                f"validity_mask.dim()={validity_mask.dim()}, expected 2 "
                f"[T_det, N_fdm]; got shape {tuple(validity_mask.shape)}"
            )
        if validity_mask.shape[0] != cube.shape[0]:
            raise ValueError(
                f"validity_mask.shape[0]={validity_mask.shape[0]} != "
                f"cube.shape[0]={cube.shape[0]} (T_det)"
            )
        if validity_mask.shape[1] != cube.shape[1]:
            raise ValueError(
                f"validity_mask.shape[1]={validity_mask.shape[1]} != "
                f"cube.shape[1]={cube.shape[1]} (N_fdm)"
            )
        if validity_mask.dtype != torch.bool:
            raise TypeError(
                f"validity_mask.dtype={validity_mask.dtype}, expected torch.bool"
            )

    def _compute_cube_valid(self, validity_mask: torch.Tensor) -> bool:
        """Decide whether to use this cube to update Layer-2 σ_k.

        Returns True when at least ``layer2_valid_min_fraction`` of the
        ``(T_det, N_fdm)`` cells are True. Default 1.0 reproduces the
        pre-M7.4 strict ``torch.all`` behaviour bit-for-bit; values
        below 1.0 let σ_k learn from cubes that have a few sparse
        invalid cells (the field-typical regime).
        """
        if self._layer2_valid_min_fraction >= 1.0:
            return bool(torch.all(validity_mask).item())
        n_total = int(validity_mask.numel())
        if n_total <= 0:
            return False
        n_valid = int(validity_mask.sum().item())
        return (n_valid / n_total) >= self._layer2_valid_min_fraction

    def _compute_per_kernel_scores(
        self,
        cube: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the v1 conv bank over one cube; return per-kernel scores.

        Produces a ``[K, T_det, N_fdm, H, W]`` tensor in fp32 (cumsum
        upcasts fp16 internally, and the Layer-2 σ_k EMA wants fp32). For
        each kernel triple ``k = (img, dm, time)``:

          1. **Spatial conv** with ``image_kernel`` (1×1 delta in v1, so
             this is a no-op pass-through; v2 will replace with PSF).
          2. **K_dm centred boxcar** of width ``k.k_dm_width`` along the
             N_fdm axis, via ``boxcar_via_cumsum``.
          3. **K_time centred boxcar** of width ``k.k_time_width`` along
             the T_det axis, via ``boxcar_via_cumsum``.

        Output index ordering matches ``build_kernel_bank()`` iteration:
        ``k = i_img * len(dm) * len(time) + i_dm * len(time) + i_time``.

        Note that the per-cell sum (boxcar with NO ``/width`` mean) gives
        a ~``√(K_dm · K_time)`` amplitude scaling on Gaussian noise, which
        the Layer-2 σ_k EMA (chunk 3) absorbs cleanly. The validity mask is
        not yet applied here; chunk 3 (which owns the σ_k EMA) consumes
        the mask to suppress invalid-cube updates.

        Args:
            cube: ``[T_det, N_fdm, H, W]`` cube tensor.
            validity_mask: ``[T_det, N_fdm]`` bool. Validated here but
                otherwise not consumed in chunk 1 (chunk 3's σ_k EMA does
                consume it).

        Returns:
            ``[K, T_det, N_fdm, H, W]`` per-kernel score tensor in fp32.
        """
        self._validate_cube(cube, validity_mask)

        T_det, N_fdm, H, W = cube.shape  # noqa: N806 (uppercase domain names)
        K = len(self._kernel_bank)  # noqa: N806

        # Promote the cube to fp32 for the conv pipeline so per-kernel
        # scores are accumulated at full precision (the §4.4 noise_norm
        # Layer-2 σ_k EMA wants fp32). ``boxcar_via_cumsum`` itself also
        # internally upcasts fp16/bf16 inputs for the cumsum to honour
        # the §3.6.13 rel-err pin, but doing the promotion once here
        # avoids the cast-cast round-trip across the per-kernel inner
        # loop and keeps the K_dm and K_time boxcars stay-in-fp32.
        cube_f32 = cube.to(torch.float32)

        scores = torch.empty(
            (K, T_det, N_fdm, H, W),
            dtype=torch.float32,
            device=cube.device,
        )

        # Compute scores per-kernel. v1 image kernels are all 1×1 delta,
        # so the spatial conv is a no-op pass-through. We still respect
        # the per-image-kernel buffer (so v2's PSF kernel slots in here
        # without changing the loop structure).
        for k_idx, kernel in enumerate(self._kernel_bank):
            scores[k_idx] = self._compute_score_for_kernel(
                cube_f32, kernel, tile_size=None,
            )

        return scores

    def _compute_score_for_kernel(
        self,
        cube: torch.Tensor,
        kernel: Kernel,
        *,
        tile_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the per-kernel score for ONE kernel triple.

        This is the per-iteration body of
        :meth:`_compute_per_kernel_scores`, factored out so the
        chunk-8 streaming forward can iterate kernel-by-kernel without
        materialising the full ``[K, T_det, N_fdm, H, W]`` batched
        score tensor.

        Args:
            cube: ``[T_det, N_fdm, H, W]`` cube tensor (any float
                dtype). The output is in the same dtype as ``cube``;
                ``boxcar_via_cumsum`` upcasts internally so cumsum
                accuracy is preserved at the §3.6.13 fp16 rel-err pin
                regardless of caller dtype.
            kernel: one of ``self._kernel_bank``.
            tile_size: optional W-axis tile size forwarded to
                ``boxcar_via_cumsum``. Set on memory-constrained GPUs
                at production geometry to bound the fp32 cumsum
                working set (default None = untiled, the historical
                ``_compute_per_kernel_scores`` behaviour).

        Returns:
            Per-kernel score tensor, shape ``[T_det, N_fdm, H, W]``,
            same dtype as ``cube``.
        """
        T_det, N_fdm, H, W = cube.shape  # noqa: N806
        img_buf_name = self._image_kernel_buffers[kernel.image_token]
        img_kernel = getattr(self, img_buf_name)

        if kernel.image_kernel_size == 1:
            # 1×1 delta → no-op; multiply by scalar value (1.0 in v1).
            spatial = cube * float(img_kernel.item())
        else:  # pragma: no cover  (v2 path; not exercised in chunk 1)
            # F.conv2d expects [N, C_in, H, W] with kernel
            # [C_out, C_in, K_h, K_w]. Treat (T_det, N_fdm) as the
            # batch dims and invoke conv2d via reshape.
            x = cube.reshape(T_det * N_fdm, 1, H, W).to(torch.float32)
            w = img_kernel.unsqueeze(0).unsqueeze(0).to(torch.float32)
            pad = kernel.image_kernel_size // 2
            y = F.conv2d(x, w, padding=pad)
            spatial = y.reshape(T_det, N_fdm, H, W).to(cube.dtype)

        if kernel.k_dm_width > 1 and N_fdm >= kernel.k_dm_width:
            dm_summed = boxcar_via_cumsum(
                spatial, axis=1, width=kernel.k_dm_width,
                tile_size=tile_size,
            )
        else:
            dm_summed = spatial

        if kernel.k_time_width > 1 and T_det >= kernel.k_time_width:
            t_summed = boxcar_via_cumsum(
                dm_summed, axis=0, width=kernel.k_time_width,
                tile_size=tile_size,
            )
        else:
            t_summed = dm_summed

        return t_summed

    def _get_or_build_amortise_cs(
        self,
        cube: torch.Tensor,
        *,
        max_width: int,
    ) -> torch.Tensor:
        """Return the pre-allocated padded-cumsum scratch (shape
        ``(T_det + max_width, F, H, W)`` accum_dtype). Allocate on
        first call; reuse across cubes. Caller must populate the
        contents via :meth:`_fill_amortise_cs` before reading.

        The buffer is intentionally NOT zeroed on reuse: the leading
        ``pad_left + 1`` rows are static zeros (they're part of the
        max-width zero-pad and never get overwritten), and the
        trailing ``pad_right`` rows are constant-fill of the new cube's
        total cumsum (rewritten by :meth:`_fill_amortise_cs`). The
        cumsum middle section (``narrow(axis, 1+pad_left, T_det)``)
        is overwritten in-place via ``torch.cumsum(out=...)`` so no
        stale data leaks.
        """
        if self._boxcar_accum_dtype is not None:
            accum_dtype = self._boxcar_accum_dtype
        else:
            accum_dtype = (
                torch.float32
                if cube.dtype in (torch.float16, torch.bfloat16)
                else cube.dtype
            )
        n = cube.shape[0]
        target_shape = (n + max_width,) + tuple(cube.shape[1:])
        if (
            self._amortise_cs is not None
            and self._amortise_cs.shape == target_shape
            and self._amortise_cs.dtype == accum_dtype
            and self._amortise_cs.device == cube.device
        ):
            return self._amortise_cs
        self._amortise_cs = torch.zeros(
            target_shape, dtype=accum_dtype, device=cube.device,
        )
        return self._amortise_cs

    def _fill_amortise_cs(
        self,
        cs: torch.Tensor,
        cube: torch.Tensor,
        *,
        max_width: int,
        axis: int = 0,
    ) -> None:
        """Populate the pre-allocated cs buffer with the new cube's
        zero-padded cumulative sum along ``axis``. See
        :func:`precompute_padded_cumsum` for the layout.
        """
        n = cube.shape[axis]
        pad_left = max_width // 2
        pad_right = max_width - 1 - pad_left
        target = cs.narrow(axis, 1 + pad_left, n)
        torch.cumsum(cube, dim=axis, dtype=cs.dtype, out=target)
        if pad_right > 0:
            last_row = cs.narrow(axis, 1 + pad_left + n - 1, 1)
            cs.narrow(axis, 1 + pad_left + n, pad_right).copy_(last_row)

    def _streaming_forward(
        self,
        cube: torch.Tensor,
        validity_mask: torch.Tensor,
        sigma_layer1: torch.Tensor,
        *,
        dm_idx_canonical_lo: Optional[int] = None,
        dm_idx_canonical_hi: Optional[int] = None,
        n_kernel_max_t: Optional[int] = None,
        event_specnum: int = 0,
        fine_to_coarse: Optional[torch.Tensor] = None,
        fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    ) -> List[Candidate]:
        """Kernel-by-kernel streaming forward (chunk-8 production path).

        Single-pass semantics for production throughput: decode cube N
        with the PREVIOUS Layer-2 EMA state ``s_k_prev``, then update
        Layer-2 at end-of-cube from cube N's per-kernel σ estimates.

          * For each kernel k: compute ``score_k`` once, compute
            ``sigma_this_k`` from ``score_k`` for the Layer-2 update,
            and decode ``snr_k = score_k / s_k_prev[k]``.
          * After all kernels: call
            ``update_and_query(per_kernel_sigma=sigma_this_vec)`` so the
            returned ``s_k`` becomes the divisor state for cube N+1.

        This removes the second full score pass (the dominant detector
        cost at production geometry) while preserving the same
        per-kernel σ estimator and EMA dynamics. The trade-off is a
        one-cube lag in the σ divisor used for thresholding.

        Memory ceiling per kernel:
          * score_k in ``cube.dtype`` (~1 GiB at production fp16) +
          * boxcar transients (~0.7 GiB with ``tile_size=64``;
            ~9 GiB untiled — see ``boxcar_via_cumsum``'s
            ``tile_size=`` docstring) +
          * decode_local_max transients (~5-6 GiB at production fp16
            via the permute().reshape() copy in the time-axis
            ``F.max_pool1d`` step) +
          * persistent: cube + Layer-2 ``s_k`` + image-kernel buffers.
        """
        self._validate_cube(cube, validity_mask)

        K = len(self._kernel_bank)  # noqa: N806
        cube_valid = self._compute_cube_valid(validity_mask)
        tile_size = self._streaming_tile_size

        # Detect the v1-collapsed bank optimisation: every kernel uses
        # a 1×1 delta image kernel + a width-1 (no-op) DM boxcar; the
        # entire bank's per-kernel time-axis boxcar is then a single
        # shared time-axis cumsum + per-kernel narrow-subtract. This
        # amortises K cumsums into 1 per pass — the dominant detector
        # cost at production geometry (T_det=256, N_fdm=32, N_grid=256)
        # because each cumsum is a ~10 GiB fp32 memory-traffic blow.
        # See M5_PLAN_FIXES (D26).
        all_image_delta = all(
            kernel.image_kernel_size == 1
            and float(getattr(kernel, "image_kernel").item()) == 1.0
            for kernel in self._kernel_bank
        )
        all_dm_unit = all(
            kernel.k_dm_width == 1 for kernel in self._kernel_bank
        )
        amortise_time_cumsum = (
            bool(all_image_delta)
            and bool(all_dm_unit)
            and len(self._kernel_bank) > 1
        )
        amortise_max_width = 1
        if amortise_time_cumsum:
            amortise_max_width = max(
                int(kernel.k_time_width) for kernel in self._kernel_bank
            )
            if amortise_max_width > cube.shape[0]:
                # max_width must fit the cumsum axis; fall back to the
                # per-kernel boxcar path for the (rare) tiny-cube case.
                amortise_time_cumsum = False
                amortise_max_width = 1

        # Decode uses the PREVIOUS Layer-2 EMA state. We update the EMA
        # at end-of-cube so the new state applies to cube N+1.
        s_k_decode = self._sigma_k.detach().clone()

        # Per-kernel σ_this for the Layer-2 update is computed AFTER
        # the decode loop in a single batched σ-clip call (saves the
        # 7× python-loop + per-kernel ``.item()`` syncs the chunk-8
        # path used to pay). The per-kernel interior subsamples are
        # accumulated into ``sigma_samples`` as the loop iterates.
        # ``layer2_interior_sigma`` is no longer used on the streaming
        # path, but is still re-exported for non-streaming consumers.
        from ..noise_norm.layer1 import sigma_clipped_std_batched

        # Pre-allocate (or reuse) the amortised-cumsum buffer once
        # per detector lifetime. Avoids re-allocating a ~3 GiB buffer
        # per cube and avoids the caching-allocator fragmentation that
        # forces a per-cube ``empty_cache`` (~150 ms cost).
        if amortise_time_cumsum:
            cs = self._get_or_build_amortise_cs(
                cube, max_width=amortise_max_width,
            )
            self._fill_amortise_cs(
                cs, cube, max_width=amortise_max_width,
            )
        else:
            cs = None

        # Canonical filtering drops edge times in a window set by
        # n_kernel_max_t. Skip scoring those dropped edge bins entirely
        # to avoid wasted detector work.
        if n_kernel_max_t is None:
            n_kernel_max_t = self._n_kernel_max_t
        disable_interior = bool(int(os.environ.get("DSART_DISABLE_INTERIOR_SCORE", "0")))
        t_edge = 0 if disable_interior else max(0, int(n_kernel_max_t) // 2)
        t_base = 0
        n_out_eff = int(cube.shape[0])
        if t_edge > 0 and (2 * t_edge) < int(cube.shape[0]):
            t_base = t_edge
            n_out_eff = int(cube.shape[0]) - (2 * t_edge)
        sigma_n_kernel_max_t = 1 if t_base > 0 else self._layer2.n_kernel_max_t
        decoder_n_top = self._streaming_decoder_n_top
        if t_base > 0 and n_out_eff < int(cube.shape[0]):
            scale = float(cube.shape[0]) / float(n_out_eff)
            decoder_n_top = max(
                self._streaming_decoder_n_top,
                int(math.ceil(self._streaming_decoder_n_top * scale)),
            )
        enable_argmax_decode = bool(
            int(os.environ.get("DSART_ENABLE_ARGMAX_DECODE", "0"))
        )
        use_argmax_decode = (
            self._streaming_decoder == "topk_lowmem"
            and bool(all_image_delta)
            and bool(all_dm_unit)
            and enable_argmax_decode
        )
        argmax_topk = decoder_n_top * K

        # Pre-resolve the interior slab geometry and per-kernel
        # subsample indices for the batched Layer-2 σ. When the
        # caller has interior scoring active (``t_base > 0``), the
        # full ``score_k`` IS the interior slab; otherwise we slice
        # ``[t_sig_lo:t_sig_hi]`` to match the historical
        # ``layer2_interior_sigma`` semantics.
        sig_max = self._layer2_sigma_max_samples
        if t_base > 0:
            t_sig_lo, t_sig_hi = 0, int(n_out_eff)
        else:
            lo = int(sigma_n_kernel_max_t) // 2
            hi = int(cube.shape[0]) - lo
            if hi <= lo:
                # ``sigma_n_kernel_max_t`` wider than the cube (tiny
                # unit-test geometry) → fall back to the whole slab.
                lo, hi = 0, int(cube.shape[0])
            t_sig_lo, t_sig_hi = lo, hi
        interior_t = int(t_sig_hi - t_sig_lo)
        n_per_kernel_interior = (
            interior_t * int(cube.shape[1]) * int(cube.shape[2]) * int(cube.shape[3])
        )
        layer2_subsample_active = (
            sig_max is not None
            and int(sig_max) > 0
            and n_per_kernel_interior > int(sig_max)
        )
        if layer2_subsample_active:
            m_l2 = int(sig_max)
            l2_key = (
                K,
                n_per_kernel_interior,
                m_l2,
                str(cube.device),
            )
            if (
                self._layer2_idx is None
                or self._layer2_idx_key != l2_key
            ):
                gen = torch.Generator(device=cube.device)
                idx_stack = torch.empty(
                    (K, m_l2), dtype=torch.int64, device=cube.device,
                )
                for kk in range(K):
                    gen.manual_seed(int(kk))
                    idx_stack[kk] = torch.randint(
                        0, n_per_kernel_interior, (m_l2,),
                        device=cube.device, generator=gen,
                    )
                self._layer2_idx = idx_stack
                self._layer2_idx_key = l2_key
            sigma_samples = torch.empty(
                (K, m_l2), dtype=torch.float32, device=cube.device,
            )
        else:
            sigma_samples = None  # full interior; batched sigma over per-row flatten

        # Build a stacked container for the non-subsample path. Per-row
        # length is the full interior cell count; for the production
        # 100 k cap this branch is inactive (and would OOM if hit).
        if not layer2_subsample_active:
            sigma_full = torch.empty(
                (K, n_per_kernel_interior),
                dtype=torch.float32,
                device=cube.device,
            )

        per_kernel_cands: List[Candidate] = []
        argmax_snr: Optional[torch.Tensor] = None
        argmax_winner: Optional[torch.Tensor] = None
        # M7.2 fast path: single fused multi-boxcar argmax pass when the
        # bank is K_img=1 / K_dm=1 (i.e., the v1-collapsed bank where
        # ``amortise_time_cumsum`` is true) and we have ≥ 2 kernels. The
        # fused Triton kernel reads the padded cumsum once per output
        # cell and produces (max_snr, winner_kernel_idx) — replacing K
        # cube-sized boxcar passes + K topk calls with one of each.
        fused_done = False
        disable_fused = bool(
            int(os.environ.get("DSART_DISABLE_FUSED_MULTI_BOXCAR", "0"))
        )
        if (
            cs is not None
            and amortise_time_cumsum
            and K >= 2
            and not use_argmax_decode
            and self._streaming_decoder == "topk_lowmem"
            and not disable_fused
            and cube.dtype in (torch.float16, torch.float32)
        ):
            from .triton_boxcar import multi_boxcar_argmax_triton

            pad_left_full = int(amortise_max_width) // 2
            widths_host = [int(k.k_time_width) for k in self._kernel_bank]
            offsets_host = [pad_left_full - (w // 2) for w in widths_host]
            widths_t = torch.tensor(
                widths_host, dtype=torch.int32, device=cube.device,
            )
            offsets_t = torch.tensor(
                offsets_host, dtype=torch.int32, device=cube.device,
            )
            # Compute sigma_inv on-device to avoid the ``[s_k_decode[k]
            # .item() for k in range(K)]`` GPU→CPU sync loop (~K stalls
            # / cube on the search-side hot path; with K=7 that was
            # adding ~10-15 ms / cube).
            s_k_slice = s_k_decode[:K]
            if s_k_slice.device != cube.device:
                s_k_slice = s_k_slice.to(cube.device)
            sigma_inv_t = torch.reciprocal(
                torch.clamp(s_k_slice.to(torch.float32), min=1e-30)
            )

            fused_result = multi_boxcar_argmax_triton(
                cs,
                widths=widths_t,
                offsets=offsets_t,
                sigma_inv=sigma_inv_t,
                n_out=n_out_eff,
                t_base=t_base,
                out_max_dtype=cube.dtype,
            )
            if fused_result is not None:
                max_snr_cube, winner_cube = fused_result

                # Per-kernel σ for the Layer-2 update: gather subsamples
                # from cs at indices ``self._layer2_idx[k]`` (which index
                # a [interior_t × F × H × W] flat layout). For each
                # kernel k, score_k = cs[t+off[k]+w[k]] - cs[t+off[k]];
                # subtract at the random subsample positions.
                if layer2_subsample_active:
                    t_offset = (
                        int(t_base) if t_base > 0 else int(t_sig_lo)
                    )
                    n_f_l = int(cube.shape[1])
                    n_h_l = int(cube.shape[2])
                    n_w_l = int(cube.shape[3])
                    FHW = n_f_l * n_h_l * n_w_l
                    cs_flat = cs.reshape(-1)
                    for k_idx2 in range(K):
                        idx_k = self._layer2_idx[k_idx2]
                        t_in = idx_k // FHW
                        fhw_in = idx_k % FHW
                        low_idx = (
                            (t_in + t_offset + offsets_host[k_idx2]) * FHW
                            + fhw_in
                        )
                        high_idx = (
                            (
                                t_in + t_offset + offsets_host[k_idx2]
                                + widths_host[k_idx2]
                            )
                            * FHW + fhw_in
                        )
                        low_v = cs_flat.index_select(0, low_idx).to(torch.float32)
                        high_v = cs_flat.index_select(0, high_idx).to(torch.float32)
                        sigma_samples[k_idx2] = high_v - low_v
                else:
                    # Full-interior path: materialise per-kernel score
                    # cubes for the sigma estimate (rare; OOMs at prod
                    # geometry so layer2_subsample_active is True there).
                    for k_idx2 in range(K):
                        score_k_full = boxcar_from_padded_cumsum(
                            cs, axis=0,
                            width=int(widths_host[k_idx2]),
                            max_width=amortise_max_width,
                            n_out=n_out_eff,
                            t_base=t_base,
                            out_dtype=torch.float32,
                            w_tile_size=tile_size,
                        )
                        if t_base > 0:
                            interior_view = score_k_full
                        else:
                            interior_view = score_k_full.narrow(
                                0, t_sig_lo, t_sig_hi - t_sig_lo,
                            )
                        sigma_full[k_idx2] = interior_view.reshape(-1)
                        del score_k_full

                # Single global topk + decode on (max_snr, winner_idx).
                per_kernel_cands = decode_topk_argmax_lowmem(
                    max_snr_cube,
                    winner_cube,
                    threshold=self._threshold_sigma,
                    kernel_ids=[k.kernel_id for k in self._kernel_bank],
                    kernel_time_widths=widths_host,
                    merge_radius_lm=self._merge_radius_lm,
                    merge_radius_fdm=self._merge_radius_fdm,
                    merge_radius_t=self._merge_radius_t,
                    detector_version=self._detector_version,
                    search_node_id=self._search_node_id,
                    gpu_half=self._gpu_half,
                    event_specnum=(int(event_specnum) + t_base),
                    fine_to_coarse=fine_to_coarse,
                    fine_dm_pc_cm3=fine_dm_pc_cm3,
                    n_top=decoder_n_top * K,
                )
                # 2026-06-10 wing-decode diagnostic: live injections whose
                # apex lands at cube-tail phases (t≈[192,224)) are decoded
                # only as off-DM b16 "wings" even though the DUMPED cube
                # (staged post-detector from the same tensor) contains a
                # >30σ w4 apex that an offline replica of THIS exact fused
                # path decodes trivially. Log (a) the fused max-SNR cell
                # and (b) the σ-normalised input max over the tail slab so
                # we can tell "detector input lacked the apex at detect
                # time" apart from "decode lost it". Cheap (~1-2 ms) and
                # only syncs/logs when something ≥ log_floor is present.
                try:
                    _log_floor = 12.0
                    _mx_t = max_snr_cube.max()
                    _tail_lo = min(192, int(cube.shape[0]) - 1)
                    _tail_n = max(1, min(32, int(cube.shape[0]) - _tail_lo))
                    _tail_mx_t = cube.narrow(0, _tail_lo, _tail_n).max()
                    _mx = float(_mx_t.item())
                    _tail_mx = float(_tail_mx_t.item())
                    if _mx >= _log_floor or _tail_mx >= _log_floor:
                        _flat_ix = int(torch.argmax(max_snr_cube).item())
                        _F, _H, _W = (
                            int(max_snr_cube.shape[1]),
                            int(max_snr_cube.shape[2]),
                            int(max_snr_cube.shape[3]),
                        )
                        _t = _flat_ix // (_F * _H * _W) + t_base
                        _rem = _flat_ix % (_F * _H * _W)
                        _f = _rem // (_H * _W)
                        _r = (_rem % (_H * _W)) // _W
                        _c = _rem % _W
                        _LOG.info(
                            "fused_decode_diag: specnum=%d max_snr=%.2f "
                            "at(t=%d,f=%d,r=%d,c=%d) tail_input_max=%.2f "
                            "n_cands=%d",
                            int(event_specnum), _mx, _t, _f, _r, _c,
                            _tail_mx, len(per_kernel_cands),
                        )
                except Exception:  # noqa: BLE001 — diag must never kill the hot path
                    pass
                del max_snr_cube, winner_cube
                fused_done = True

        if fused_done:
            kernel_iter: list = []
        else:
            kernel_iter = list(enumerate(self._kernel_bank))
        for k_idx, kernel in kernel_iter:
            if cs is not None:
                score_k = boxcar_from_padded_cumsum(
                    cs,
                    axis=0,
                    width=int(kernel.k_time_width),
                    max_width=amortise_max_width,
                    n_out=n_out_eff,
                    t_base=t_base,
                    out_dtype=cube.dtype,
                    w_tile_size=tile_size,
                )
            else:
                score_k = self._compute_score_for_kernel(
                    cube, kernel, tile_size=tile_size,
                )
            if t_base > 0 and int(score_k.shape[0]) != n_out_eff:
                score_k = score_k.narrow(0, t_base, n_out_eff)
            # Subsample this kernel's interior into the stacked
            # sigma buffer. Math-equivalent to the prior
            # ``layer2_interior_sigma(score_k.unsqueeze(0))`` call but
            # defers the σ-clip iteration loop until after every
            # kernel has contributed (one batched ``nanmedian`` over
            # ``[K, max_samples]`` replaces K sequential medians +
            # ``.item()`` syncs).
            if t_base > 0:
                interior_view = score_k
            else:
                interior_view = score_k.narrow(0, t_sig_lo, t_sig_hi - t_sig_lo)
            flat_interior = interior_view.reshape(-1)
            if layer2_subsample_active:
                # int64 fancy-index gather; upcast on read so the
                # ``sigma_samples`` buffer is already fp32.
                sigma_samples[k_idx] = flat_interior[
                    self._layer2_idx[k_idx]
                ].to(torch.float32)
            else:
                sigma_full[k_idx] = flat_interior.to(torch.float32)

            s_k_scalar = float(s_k_decode[k_idx].item())
            if s_k_scalar == 0.0:
                # Degenerate kernel: should never happen post-Layer-2's
                # zero-replace guard but defend against div-by-zero.
                s_k_scalar = 1.0

            if use_argmax_decode:
                # Argmax decode path still needs an SNR cube to compare
                # cross-kernel; keep the explicit divide on this branch.
                snr_k = score_k / s_k_scalar
                if argmax_snr is None:
                    argmax_snr = snr_k.clone()
                    argmax_winner = torch.full(
                        snr_k.shape,
                        int(k_idx),
                        dtype=torch.int16,
                        device=snr_k.device,
                    )
                else:
                    better = snr_k > argmax_snr
                    argmax_snr = torch.where(better, snr_k, argmax_snr)
                    argmax_winner[better] = int(k_idx)
                del snr_k
            elif self._streaming_decoder == "topk_lowmem":
                # No per-cube SNR materialisation: threshold raw
                # ``score_k`` against ``threshold * s_k``; emitted
                # Candidate.snr is rescaled from raw_top_val / s_k.
                cands = decode_topk_lowmem(
                    score_k,
                    threshold=self._threshold_sigma,
                    kernel_id=kernel.kernel_id,
                    k_dm_width=kernel.k_dm_width,
                    k_time_width=kernel.k_time_width,
                    detector_version=self._detector_version,
                    search_node_id=self._search_node_id,
                    gpu_half=self._gpu_half,
                    event_specnum=(int(event_specnum) + t_base),
                    fine_to_coarse=fine_to_coarse,
                    fine_dm_pc_cm3=fine_dm_pc_cm3,
                    n_top=decoder_n_top,
                    snr_divisor=s_k_scalar,
                )
            else:
                # ``decode_local_max`` builds a ``score > threshold``
                # mask on the raw score; same snr_divisor trick lets
                # it skip the per-cube divide.
                cands = decode_local_max(
                    score_k,
                    threshold=self._threshold_sigma,
                    kernel_id=kernel.kernel_id,
                    k_dm_width=kernel.k_dm_width,
                    k_time_width=kernel.k_time_width,
                    detector_version=self._detector_version,
                    search_node_id=self._search_node_id,
                    gpu_half=self._gpu_half,
                    event_specnum=(int(event_specnum) + t_base),
                    fine_to_coarse=fine_to_coarse,
                    fine_dm_pc_cm3=fine_dm_pc_cm3,
                    snr_divisor=s_k_scalar,
                )
            if not use_argmax_decode:
                per_kernel_cands.extend(cands)
            del score_k

        if use_argmax_decode and argmax_snr is not None and argmax_winner is not None:
            per_kernel_cands = decode_topk_argmax_lowmem(
                argmax_snr,
                argmax_winner,
                threshold=self._threshold_sigma,
                kernel_ids=[kernel.kernel_id for kernel in self._kernel_bank],
                kernel_time_widths=[
                    int(kernel.k_time_width) for kernel in self._kernel_bank
                ],
                merge_radius_lm=self._merge_radius_lm,
                merge_radius_fdm=self._merge_radius_fdm,
                merge_radius_t=self._merge_radius_t,
                detector_version=self._detector_version,
                search_node_id=self._search_node_id,
                gpu_half=self._gpu_half,
                event_specnum=(int(event_specnum) + t_base),
                fine_to_coarse=fine_to_coarse,
                fine_dm_pc_cm3=fine_dm_pc_cm3,
                n_top=argmax_topk,
            )
            del argmax_snr, argmax_winner

        # Single batched σ-clip over the stacked per-kernel samples
        # (replaces K sequential ``layer2_interior_sigma`` calls).
        # ``sigma_samples`` / ``sigma_full`` lives in fp32 already so
        # the batched primitive can skip the upcast.
        sigma_stack = (
            sigma_samples if layer2_subsample_active else sigma_full
        )
        sigma_this_per_kernel = sigma_clipped_std_batched(
            sigma_stack,
            n_sigma=self._layer2.n_sigma,
            n_iterations=self._layer2.n_iterations,
        )

        # End-of-cube Layer-2 update; new s_k applies to the NEXT cube.
        s_k_tensor, is_warming_up = self._layer2.update_and_query(
            per_kernel_sigma=sigma_this_per_kernel, valid=cube_valid,
        )
        self._sigma_k.copy_(s_k_tensor)

        warmup_flag = int(CandidateFlags.NOISE_WARMUP) if is_warming_up else 0
        if warmup_flag:
            per_kernel_cands = [
                _with_flags(c, c.flags | warmup_flag) for c in per_kernel_cands
            ]

        # Note: the amortise cs buffer is owned by the detector and
        # reused across cubes; do not free here.
        if self._merger_config is not None:
            # C1 path (M7.4): re-merge under the new geometry even when
            # the fused/argmax decoders already de-duplicated under
            # legacy radii, so the survivors carry C1 semantics.
            merged = merge_across_kernels_c1(per_kernel_cands, self._merger_config)
        elif use_argmax_decode or fused_done:
            # ``decode_topk_argmax_lowmem`` already produced
            # de-duplicated candidates across the kernel bank — skip the
            # cross-kernel merger.
            merged = per_kernel_cands
        else:
            merged = merge_across_kernels(
                per_kernel_cands,
                merge_radius_lm=self._merge_radius_lm,
                merge_radius_fdm=self._merge_radius_fdm,
                merge_radius_t=self._merge_radius_t,
            )

        if dm_idx_canonical_lo is None or dm_idx_canonical_hi is None:
            return self._apply_c1_emit_floor(merged)
        emit, _dropped = filter_to_canonical(
            merged,
            dm_idx_canonical_lo=dm_idx_canonical_lo,
            dm_idx_canonical_hi=dm_idx_canonical_hi,
            t_det=cube.shape[0],
            n_kernel_max_t=n_kernel_max_t,
            cube_t_offset=event_specnum,
        )
        return self._apply_c1_emit_floor(emit)
