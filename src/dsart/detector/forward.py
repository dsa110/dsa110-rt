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

import math
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
from .decoder import decode_local_max, filter_to_canonical
from .kernels import (
    DEFAULT_DETECTOR_DTYPE,
    Kernel,
    build_kernel_bank,
)
from .merger import (
    DEFAULT_MERGE_RADIUS_FDM,
    DEFAULT_MERGE_RADIUS_LM,
    DEFAULT_MERGE_RADIUS_T,
    merge_across_kernels,
)

__all__ = [
    "Detector",
    "DeterministicDetector",
    "boxcar_via_cumsum",
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
    if axis < 0:
        axis = ndim + axis
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
        search_node_id: int = 0,
        gpu_half: int = 0,
        cube_cadence_s: float = CUBE_CADENCE_S_DEFAULT,
        layer2_tau_s: float = NOISE_LAYER2_TAU_S_DEFAULT,
        layer2_n_burnin: int = NOISE_LAYER2_N_BURNIN_DEFAULT,
        n_kernel_max_t: int = N_KERNEL_MAX_T_DEFAULT,
        layer2_state: Optional[Layer2State] = None,
        layer2_seed_unit: bool = True,
    ) -> None:
        super().__init__()
        self._kernel_bank: Tuple[Kernel, ...] = kernel_bank or build_kernel_bank(
            dtype=dtype
        )
        self._threshold_sigma = float(threshold_sigma)
        self._detector_version = str(detector_version)
        self._dtype = dtype
        self._merge_radius_lm = int(merge_radius_lm)
        self._merge_radius_fdm = int(merge_radius_fdm)
        self._merge_radius_t = int(merge_radius_t)
        self._search_node_id = int(search_node_id)
        self._gpu_half = int(gpu_half)
        self._n_kernel_max_t = int(n_kernel_max_t)

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
        """
        scores = self._compute_per_kernel_scores(cube, validity_mask)

        # Determine cube validity: any (T_det, N_fdm) cell False in the
        # mask flags the cube as suspect. Plan §319 says invalid cubes
        # skip BOTH forward and noise updates; here we still run forward
        # (the candidate log records what the detector saw) but skip
        # the EMA update so a contaminated cube doesn't poison σ_k.
        cube_valid = bool(torch.all(validity_mask).item())
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

        merged = merge_across_kernels(
            per_kernel_cands,
            merge_radius_lm=self._merge_radius_lm,
            merge_radius_fdm=self._merge_radius_fdm,
            merge_radius_t=self._merge_radius_t,
        )

        if dm_idx_canonical_lo is None or dm_idx_canonical_hi is None:
            return merged
        emit, _dropped = filter_to_canonical(
            merged,
            dm_idx_canonical_lo=dm_idx_canonical_lo,
            dm_idx_canonical_hi=dm_idx_canonical_hi,
            t_det=cube.shape[0],
            n_kernel_max_t=n_kernel_max_t,
            cube_t_offset=event_specnum,
        )
        return emit

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
            img_buf_name = self._image_kernel_buffers[kernel.image_token]
            img_kernel = getattr(self, img_buf_name)

            if kernel.image_kernel_size == 1:
                # 1×1 delta → no-op; multiply by scalar value (1.0 in v1).
                spatial = cube_f32 * float(img_kernel.item())
            else:  # pragma: no cover  (v2 path; not exercised in chunk 1)
                # F.conv2d expects [N, C_in, H, W] with kernel [C_out, C_in,
                # K_h, K_w]. Treat (T_det, N_fdm) as the batch dims and
                # invoke conv2d via reshape.
                x = cube_f32.reshape(T_det * N_fdm, 1, H, W)
                w = img_kernel.unsqueeze(0).unsqueeze(0).to(torch.float32)
                pad = kernel.image_kernel_size // 2
                y = F.conv2d(x, w, padding=pad)
                spatial = y.reshape(T_det, N_fdm, H, W)

            # K_dm centred boxcar (axis = 1, the N_fdm axis).
            # ``boxcar_via_cumsum`` is the only allowed K_dm consumer per
            # plan §3.6.13 test_detector_conv_flops_cumsum_pin.
            if kernel.k_dm_width > 1 and N_fdm >= kernel.k_dm_width:
                dm_summed = boxcar_via_cumsum(
                    spatial, axis=1, width=kernel.k_dm_width
                )
            else:
                dm_summed = spatial

            # K_time centred boxcar (axis = 0, the T_det axis).
            if kernel.k_time_width > 1 and T_det >= kernel.k_time_width:
                t_summed = boxcar_via_cumsum(
                    dm_summed, axis=0, width=kernel.k_time_width
                )
            else:
                t_summed = dm_summed

            scores[k_idx] = t_summed

        return scores
