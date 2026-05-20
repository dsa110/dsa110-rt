"""Per-kernel local-maxima decoder + canonical-zone emit gate
(plan §3.6.13 / §4.4 lines 1582-1594; M5 PARALLEL_AGENTS.md §3 Class A).

This module sits between ``DeterministicDetector._compute_per_kernel_scores``
(``forward.py``, Chunk 1) and the cross-kernel merger (``merger.py``,
Chunk 2). It runs three steps:

  1. **Per-kernel local-maxima NMS** (plan §1582-1588):
     - Threshold mask: ``score > θ``.
     - 4D non-max-suppression: a pixel survives iff ``score[t, fdm, l, m]``
       equals the max over a 4D neighborhood with radii
         Δl = Δm = max(2, kernel_psf_radius)  (default 2 — D10 v1 PSF radius is 0)
         Δfdm = k_dm_width // 2 + 1
         Δt   = k_time_width // 2 + 1
     - Implemented as ``F.max_pool3d`` on (fdm, l, m) followed by
       ``F.max_pool1d`` on time; equality test against the original
       score; AND with the threshold mask. ``max_pool*`` here is
       structurally the local-max NMS (single max per cell over a fixed
       neighborhood per kernel triple), **not** a width-by-width sum
       substitute for the cumsum trick — see F9 in M5_PLAN_FIXES.md
       reconciling plan §3.6.13 line 1131 vs §1588.

  2. **Build per-kernel ``Candidate`` list**: each surviving (t, fdm, l, m)
     becomes a ``Candidate`` record with the kernel triple's id, the
     score / s_k value as ``snr``, and ``flags = NONE``. ``dm_fine`` /
     ``dm_idx`` are passed through from the per-cube ``DmPlan`` view; if
     the caller does not supply a ``DmPlan``, the decoder falls back to
     ``dm_fine == fine_dm_idx`` and ``dm_idx == fine_to_coarse[fdm]``
     stub (Chunk-2 unit-test path; Chunk-6 search_compute wires the
     real ``DmPlan``).

  3. **Canonical-zone emit gate** (plan §1590-1594; ``filter_to_canonical``):
     drops candidates outside the canonical zone on either of two axes:
       a. **Fine-DM halo** (``flags.bit5 == HALO_DROPPED``)
       b. **Time-axis edge** (``flags.bit6 == TIME_EDGE_DROPPED``)
     Per plan §1594 the dropped candidates are still **logged** for
     offline analysis; only the trigger emitter is skipped. The decoder
     therefore returns the dropped candidates with their flag bit set
     (the cross-kernel merger consumes the flagged stream and the
     emitter respects the flags).

The decoder is deterministic and stateless (the Layer-2 σ_k EMA is owned
by ``forward.py``; per-kernel NMS only consumes the post-Layer-2 score).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..common.contracts import Candidate, CandidateFlags

__all__ = [
    "decode_local_max",
    "decode_topk_lowmem",
    "decode_topk_argmax_lowmem",
    "filter_to_canonical",
]


# ---------------------------------------------------------------------------
# Per-kernel local-maxima NMS — plan §1582-1588
# ---------------------------------------------------------------------------


def decode_local_max(
    score: torch.Tensor,
    *,
    threshold: float,
    kernel_id: str,
    k_dm_width: int,
    k_time_width: int,
    k_psf_radius: int = 0,
    detector_version: str = "v1.M5",
    search_node_id: int = 0,
    gpu_half: int = 0,
    event_specnum: int = 0,
    fine_to_coarse: Optional[torch.Tensor] = None,
    fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    snr_divisor: float = 1.0,
) -> List[Candidate]:
    """Run per-kernel threshold + 4D local-max NMS and emit Candidates.

    Args:
        score: ``[T_det, N_fdm, H, W]`` SNR-normalised score tensor for
            ONE kernel triple (already divided by ``s_k``). Any dtype
            supporting ``F.max_pool*``; fp32 is recommended (the
            DeterministicDetector hot path produces fp32 scores). Must
            be a single-kernel slice — the caller iterates over the
            kernel bank.
        threshold: detection threshold in σ; e.g. 8.0 per
            ``config_compute_search.yaml``. Cells with ``score ≤ threshold``
            are pruned before NMS.
        kernel_id: ``"k_img:k_dm:k_time"`` opaque id (per
            ``Candidate._check_kernel_id``); recorded on every emitted
            Candidate.
        k_dm_width: this kernel's DM boxcar width (sets the NMS radius
            ``Δfdm = k_dm_width // 2 + 1``).
        k_time_width: this kernel's time boxcar width (sets the NMS
            radius ``Δt = k_time_width // 2 + 1``).
        k_psf_radius: this kernel's image-PSF half-support in cells. v1
            (D10) is 0 (delta kernel) → spatial NMS radius is the
            plan §1585 default of 2; v2 may grow this for matched-filter
            PSFs.
        detector_version: stamped onto every emitted ``Candidate`` per
            §3 line 320-352. Default ``"v1.M5"`` matches
            ``DeterministicDetector.detector_version``.
        search_node_id: stamped onto every emitted Candidate; default 0
            for unit tests (Chunk-6 ``search_compute`` wires the real
            id from etcd).
        gpu_half: 0 or 1 (per-search-node GPU half ownership); same
            stamp pattern.
        event_specnum: stamped onto every emitted Candidate (the cube
            start specnum); default 0 for unit tests.
        fine_to_coarse: optional ``[N_fdm] int64`` mapping each fine-DM
            trial to its coarse-DM index. If None, ``Candidate.dm_idx``
            is set to ``fdm`` (chunk-2 unit-test stub; chunk-6 wires
            the real DmPlan).
        fine_dm_pc_cm3: optional ``[N_fdm] float64`` table of fine-DM
            values in pc cm⁻³. If None, ``Candidate.dm_fine`` is set
            to ``float(fdm)`` (chunk-2 unit-test stub).

    Returns:
        ``List[Candidate]`` — one per local-max cell that exceeds
        ``threshold``. Empty list if no cell exceeds. Candidates are
        emitted with ``flags = CandidateFlags.NONE``; the
        canonical-zone emit gate (``filter_to_canonical``) is a separate
        downstream call that adds HALO_DROPPED / TIME_EDGE_DROPPED bits.
    """
    if score.dim() != 4:
        raise ValueError(
            f"score.dim()={score.dim()}, expected 4 [T_det, N_fdm, H, W]"
        )
    T_det, N_fdm, H, W = score.shape  # noqa: N806

    # NMS radii per plan §1585-1587. PSF radius is 0 in v1 (D10);
    # the plan pin is max(2, kernel_psf_radius) so we use 2 by default.
    delta_l = max(2, int(k_psf_radius))
    delta_m = max(2, int(k_psf_radius))
    delta_fdm = int(k_dm_width) // 2 + 1
    delta_t = int(k_time_width) // 2 + 1

    # Rescale threshold once on the host so we can compare raw scores
    # (skips the per-kernel cube-sized SNR materialisation when the
    # caller passes a raw score + snr_divisor > 1).
    snr_divisor_f = float(snr_divisor)
    if snr_divisor_f <= 0.0:
        raise ValueError(
            f"snr_divisor={snr_divisor_f}, expected > 0"
        )
    inv_snr_divisor = 1.0 / snr_divisor_f
    raw_threshold = float(threshold) * snr_divisor_f

    # Threshold prune. Scores at or below threshold can never produce
    # candidates so we mask them out before max-pool to keep the NMS
    # window from inheriting a sub-threshold "winner" by default.
    above = score > raw_threshold
    if not torch.any(above):
        return []

    # Reshape to [N=1, C=1, D=N_fdm, H, W] for max_pool3d on (fdm, l, m).
    s5 = score.reshape(1, 1, T_det * N_fdm, H, W)
    # max_pool3d collapses the (fdm, l, m) volume — we treat (T, fdm)
    # as a fused depth axis and re-split for the time-axis pool below.
    # Actually we must NOT mix time and fdm in max_pool3d — they have
    # different NMS radii. So do (fdm, l, m) per timestep instead.
    s_per_t = score.reshape(T_det, 1, N_fdm, H, W)
    pooled_spatial = F.max_pool3d(
        s_per_t,
        kernel_size=(2 * delta_fdm + 1, 2 * delta_l + 1, 2 * delta_m + 1),
        stride=1,
        padding=(delta_fdm, delta_l, delta_m),
    )
    pooled_spatial = pooled_spatial.reshape(T_det, N_fdm, H, W)

    # Then collapse the time axis with max_pool1d (kernel = 2·Δt+1,
    # stride=1, padding=Δt). Re-shape so the time axis is the conv axis.
    s_perm = pooled_spatial.permute(1, 2, 3, 0).reshape(N_fdm * H * W, 1, T_det)
    pooled_full = F.max_pool1d(
        s_perm,
        kernel_size=2 * delta_t + 1,
        stride=1,
        padding=delta_t,
    )
    pooled_full = pooled_full.reshape(N_fdm, H, W, T_det).permute(3, 0, 1, 2)

    is_local_max = (score == pooled_full) & above
    if not torch.any(is_local_max):
        return []

    # Materialise local-max indices on CPU so we can package each as a
    # Candidate (Candidate construction is Python-only — fine for the
    # post-NMS handful per cube; the bench-time test_per_kernel_decoder
    # asserts < ~16 / cube at θ=8, well within Python loop budget).
    indices = torch.nonzero(is_local_max, as_tuple=False).cpu().numpy()
    snrs = score[is_local_max].detach().cpu().tolist()

    candidates: List[Candidate] = []
    for (t_idx, fdm_idx, l_idx, m_idx), snr_value in zip(indices, snrs):
        if fine_to_coarse is not None:
            dm_idx = int(fine_to_coarse[fdm_idx].item())
        else:
            dm_idx = int(fdm_idx)
        if fine_dm_pc_cm3 is not None:
            dm_fine = float(fine_dm_pc_cm3[fdm_idx].item())
        else:
            dm_fine = float(fdm_idx)
        candidates.append(
            Candidate(
                l=float(l_idx),
                m=float(m_idx),
                dm_fine=dm_fine,
                dm_idx=dm_idx,
                event_specnum=int(event_specnum) + int(t_idx),
                width_samples=int(k_time_width),
                kernel_id=kernel_id,
                snr=float(snr_value) * inv_snr_divisor,
                detector_version=detector_version,
                flags=int(CandidateFlags.NONE),
                search_node_id=int(search_node_id),
                gpu_half=int(gpu_half),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Memory-bounded peak finder (chunk-8 production at production geometry)
# ---------------------------------------------------------------------------


def decode_topk_lowmem(
    score: torch.Tensor,
    *,
    threshold: float,
    kernel_id: str,
    k_dm_width: int,
    k_time_width: int,
    k_psf_radius: int = 0,
    detector_version: str = "v1.M5",
    search_node_id: int = 0,
    gpu_half: int = 0,
    event_specnum: int = 0,
    fine_to_coarse: Optional[torch.Tensor] = None,
    fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    n_top: int = 64,
    snr_divisor: float = 1.0,
) -> List[Candidate]:
    """Memory-bounded peak finder for one kernel's SNR cube.

    The chunk-2 ``decode_local_max`` is a strict local-max NMS (every
    accepted Candidate is the max of its 4D neighborhood). It runs
    ``F.max_pool3d`` over (fdm, l, m) per timestep + ``F.max_pool1d``
    over time via ``permute().reshape()`` to a contiguous
    ``[N_fdm·H·W, 1, T_det]`` layout. At production geometry (T_det=256,
    N_fdm=32, N_grid=256, fp16) this materialises ~3 GiB of transient
    plus a >2 GiB cuDNN workspace inside ``F.max_pool1d`` — total live
    memory tips past the 11 GiB 2080 Ti budget when stacked behind the
    chunk-8 GpuImager + score buffers.

    ``decode_topk_lowmem`` is a STRICT SUBSET of ``decode_local_max``'s
    output for the same threshold: it lifts the top-N global SNR peaks
    (single fused ``torch.topk`` call, ~16 MiB workspace) then de-
    duplicates within the kernel's NMS radii. Every Candidate emitted
    here would also be emitted by ``decode_local_max`` (same kernel_id,
    same (l, m, dm_idx, t), same SNR); the difference is that
    ``decode_local_max`` may emit additional local maxima beyond the
    top-N — the cross-kernel merger handles cross-kernel deduplication
    regardless, so the merged candidate list is the same when N is
    chosen ≥ the per-kernel post-NMS count (the production
    PerCubePerKernelCap predicate caps the emitter at 4 per kernel
    per cube; n_top=64 is generously above that).

    Args:
        score: ``[T_det, N_fdm, H, W]`` SNR-normalised score tensor.
            Any float dtype (the function does not upcast).
        threshold: SNR threshold; cells with ``snr <= threshold`` are
            never emitted.
        kernel_id / k_dm_width / k_time_width / k_psf_radius:
            forwarded to the per-kernel NMS-radius calculation. Same
            convention as ``decode_local_max``.
        detector_version / search_node_id / gpu_half / event_specnum:
            stamped on emitted Candidates.
        fine_to_coarse / fine_dm_pc_cm3: optional ``DmPlan`` lookup
            tables for ``Candidate.dm_idx`` / ``dm_fine`` respectively.
            Same chunk-2 stub fallback as ``decode_local_max``.
        n_top: number of top-SNR cells to inspect for the NMS-radius
            de-duplication. Default 64. The production
            ``PerCubePerKernelCap`` predicate keeps at most 4 per
            kernel per cube; n_top=64 leaves headroom for a handful
            of competing peaks within the NMS radii to be ordered
            by SNR before the cap.
        snr_divisor: scalar divisor applied to ``score`` to convert
            raw kernel-output values into SNR units (i.e. SNR =
            score / snr_divisor). When > 1, ``score`` is expected to
            be in raw (un-divided) units; ``threshold`` is rescaled
            internally as ``threshold * snr_divisor`` so the
            comparison happens on raw values without materialising a
            divided cube. Emitted ``Candidate.snr`` values are
            rescaled to true SNR units. Default 1.0 (input is already
            SNR-normalised; behaviour identical to the historical
            single-argument form).

    Returns:
        ``List[Candidate]`` ordered by descending SNR. Empty when
        ``score.max() / snr_divisor <= threshold``.
    """
    if score.dim() != 4:
        raise ValueError(
            f"score.dim()={score.dim()}, expected 4 [T_det, N_fdm, H, W]"
        )
    T_det, N_fdm, H, W = score.shape  # noqa: N806

    # Rescale threshold once on the host so the GPU work below
    # operates on raw scores (no per-kernel cube-sized divide).
    snr_divisor_f = float(snr_divisor)
    if snr_divisor_f <= 0.0:
        raise ValueError(
            f"snr_divisor={snr_divisor_f}, expected > 0"
        )
    inv_snr_divisor = 1.0 / snr_divisor_f
    raw_threshold = float(threshold) * snr_divisor_f

    delta_l = max(2, int(k_psf_radius))
    delta_m = max(2, int(k_psf_radius))
    delta_fdm = int(k_dm_width) // 2 + 1
    delta_t = int(k_time_width) // 2 + 1

    flat = score.contiguous().reshape(-1)
    k = int(min(n_top, flat.numel()))
    # Single fused topk: skip the prior .max() early-exit. ``topk(k)``
    # already returns the descending top-k, so if top_vals[0] is below
    # threshold we exit after the same single D->H of <= n_top values.
    top_vals, top_indices = torch.topk(flat, k=k)
    top_vals_np = top_vals.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()

    out: List[Candidate] = []
    accepted: List[Tuple[int, int, int, int]] = []  # (t, fdm, l, m)
    for v, ix in zip(top_vals_np, top_indices_np):
        v_raw = float(v)
        if v_raw <= raw_threshold:
            break
        t = int(ix) // (N_fdm * H * W)
        rem = int(ix) % (N_fdm * H * W)
        f = rem // (H * W)
        rem = rem % (H * W)
        l_ = rem // W
        m_ = rem % W
        suppressed = False
        for at, af, al, am in accepted:
            if (abs(t - at) <= delta_t and abs(f - af) <= delta_fdm
                    and abs(l_ - al) <= delta_l and abs(m_ - am) <= delta_m):
                suppressed = True
                break
        if suppressed:
            continue
        accepted.append((t, f, l_, m_))
        if fine_to_coarse is not None:
            dm_idx = int(fine_to_coarse[f].item())
        else:
            dm_idx = int(f)
        if fine_dm_pc_cm3 is not None:
            dm_fine = float(fine_dm_pc_cm3[f].item())
        else:
            dm_fine = float(f)
        out.append(
            Candidate(
                l=float(l_),
                m=float(m_),
                dm_fine=dm_fine,
                dm_idx=dm_idx,
                event_specnum=int(event_specnum) + int(t),
                width_samples=int(k_time_width),
                kernel_id=kernel_id,
                snr=v_raw * inv_snr_divisor,
                detector_version=detector_version,
                flags=int(CandidateFlags.NONE),
                search_node_id=int(search_node_id),
                gpu_half=int(gpu_half),
            )
        )
    return out


def decode_topk_argmax_lowmem(
    score: torch.Tensor,
    winner_kernel_idx: torch.Tensor,
    *,
    threshold: float,
    kernel_ids: List[str],
    kernel_time_widths: List[int],
    merge_radius_lm: int,
    merge_radius_fdm: int,
    merge_radius_t: int,
    detector_version: str = "v1.M5",
    search_node_id: int = 0,
    gpu_half: int = 0,
    event_specnum: int = 0,
    fine_to_coarse: Optional[torch.Tensor] = None,
    fine_dm_pc_cm3: Optional[torch.Tensor] = None,
    n_top: int = 256,
) -> List[Candidate]:
    """Single-pass low-memory decode on a max-over-kernel SNR cube.

    ``score`` is the per-cell max SNR over the active kernel bank and
    ``winner_kernel_idx`` stores the winning kernel index per cell.
    The function applies one global top-k pass + merge-radius
    suppression and emits Candidates annotated with the winner kernel's
    metadata.
    """
    if score.dim() != 4:
        raise ValueError(
            f"score.dim()={score.dim()}, expected 4 [T_det, N_fdm, H, W]"
        )
    if winner_kernel_idx.shape != score.shape:
        raise ValueError(
            f"winner_kernel_idx.shape={tuple(winner_kernel_idx.shape)} "
            f"!= score.shape={tuple(score.shape)}"
        )
    if merge_radius_lm < 0 or merge_radius_fdm < 0 or merge_radius_t < 0:
        raise ValueError(
            "merge radii must be >= 0; got "
            f"({merge_radius_lm}, {merge_radius_fdm}, {merge_radius_t})"
        )
    if len(kernel_ids) != len(kernel_time_widths):
        raise ValueError(
            "kernel_ids and kernel_time_widths length mismatch: "
            f"{len(kernel_ids)} != {len(kernel_time_widths)}"
        )
    if not kernel_ids:
        return []

    T_det, N_fdm, H, W = score.shape  # noqa: N806
    snr_max_t = score.max()
    if float(snr_max_t.item()) <= threshold:
        return []

    flat = score.contiguous().reshape(-1)
    k = int(min(max(1, n_top), flat.numel()))
    top_vals, top_indices = torch.topk(flat, k=k)
    top_vals_np = top_vals.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()
    winner_flat = winner_kernel_idx.contiguous().reshape(-1)

    out: List[Candidate] = []
    accepted: List[Tuple[int, int, int, int]] = []  # (t, fdm, l, m)
    for v, ix in zip(top_vals_np, top_indices_np):
        v_f = float(v)
        if v_f <= threshold:
            break
        t = int(ix) // (N_fdm * H * W)
        rem = int(ix) % (N_fdm * H * W)
        f = rem // (H * W)
        rem = rem % (H * W)
        l_ = rem // W
        m_ = rem % W

        suppressed = False
        for at, af, al, am in accepted:
            if (
                abs(t - at) <= merge_radius_t
                and abs(f - af) <= merge_radius_fdm
                and abs(l_ - al) <= merge_radius_lm
                and abs(m_ - am) <= merge_radius_lm
            ):
                suppressed = True
                break
        if suppressed:
            continue
        accepted.append((t, f, l_, m_))

        k_idx = int(winner_flat[int(ix)].item())
        if k_idx < 0 or k_idx >= len(kernel_ids):
            continue
        if fine_to_coarse is not None:
            dm_idx = int(fine_to_coarse[f].item())
        else:
            dm_idx = int(f)
        if fine_dm_pc_cm3 is not None:
            dm_fine = float(fine_dm_pc_cm3[f].item())
        else:
            dm_fine = float(f)
        out.append(
            Candidate(
                l=float(l_),
                m=float(m_),
                dm_fine=dm_fine,
                dm_idx=dm_idx,
                event_specnum=int(event_specnum) + int(t),
                width_samples=int(kernel_time_widths[k_idx]),
                kernel_id=kernel_ids[k_idx],
                snr=v_f,
                detector_version=detector_version,
                flags=int(CandidateFlags.NONE),
                search_node_id=int(search_node_id),
                gpu_half=int(gpu_half),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Canonical-zone emit gate — plan §1590-1594
# ---------------------------------------------------------------------------


def filter_to_canonical(
    candidates: List[Candidate],
    *,
    dm_idx_canonical_lo: int,
    dm_idx_canonical_hi: int,
    t_det: int,
    n_kernel_max_t: int,
    cube_t_offset: int = 0,
) -> Tuple[List[Candidate], List[Candidate]]:
    """Apply plan §1590-1594 canonical-zone emit gate to a flat candidate
    list. Returns ``(emit, dropped)`` partition.

    The dropped candidates carry the appropriate flag bit
    (``CandidateFlags.HALO_DROPPED`` or ``CandidateFlags.TIME_EDGE_DROPPED``;
    or both) so the offline log can reconstruct the full pre-gate stream
    per plan §1594. Both the emit-list and the dropped-list are
    returned (the candidate log records both; the trigger emitter only
    consumes ``emit``).

    Args:
        candidates: flat list of ``Candidate`` records (typically the
            cross-kernel-merger output).
        dm_idx_canonical_lo: inclusive low bound of the canonical
            ``dm_idx`` range for this (search_node, gpu_half). Per
            plan §3.2 line 552 / §4.4 line 1591, this comes from the
            ``DmPlan.dm_idx_range_canonical_per_gpu[s, g]`` table.
        dm_idx_canonical_hi: inclusive high bound (same source).
        t_det: cube time depth in samples (= 512 at default ops).
        n_kernel_max_t: widest time-kernel boxcar width (= 128 at
            default ops). The time-edge gate masks out the first
            ``n_kernel_max_t // 2`` and last ``n_kernel_max_t // 2``
            samples per plan §1592.
        cube_t_offset: optional offset from cube-relative ``t_in_cube``
            (= ``Candidate.event_specnum - cube.specnum0``) to the
            cube-relative time index used for the edge check. Default 0
            assumes the candidate's cube-relative ``t_in_cube`` equals
            ``Candidate.event_specnum`` (which is what
            ``decode_local_max`` writes when ``event_specnum=0`` on the
            decode call). Chunk-6 ``search_compute`` passes the real
            cube start specnum and sets this to 0 (the cube-relative
            ``t_in_cube`` is recovered as ``event_specnum -
            cube_specnum_start``; the helper handles the subtraction
            via this kwarg).

    Returns:
        ``(emit, dropped)`` tuple. ``emit`` candidates have NO
        HALO_DROPPED / TIME_EDGE_DROPPED bits set (other flag bits like
        NOISE_WARMUP / RFI_WARMING_UP are passed through unchanged);
        ``dropped`` candidates have the relevant bit(s) set.
    """
    if dm_idx_canonical_hi < dm_idx_canonical_lo:
        raise ValueError(
            f"dm_idx_canonical_hi={dm_idx_canonical_hi} < "
            f"dm_idx_canonical_lo={dm_idx_canonical_lo}"
        )
    if t_det < 1 or n_kernel_max_t < 1:
        raise ValueError(
            f"t_det={t_det} / n_kernel_max_t={n_kernel_max_t} must both be ≥ 1"
        )

    t_edge_lo = n_kernel_max_t // 2
    t_edge_hi = t_det - n_kernel_max_t // 2  # exclusive

    emit: List[Candidate] = []
    dropped: List[Candidate] = []
    for cand in candidates:
        new_flags = int(cand.flags)
        # Halo gate: drop if dm_idx outside canonical range.
        if not (dm_idx_canonical_lo <= cand.dm_idx <= dm_idx_canonical_hi):
            new_flags |= int(CandidateFlags.HALO_DROPPED)
        # Time-edge gate: cube-relative t in [t_edge_lo, t_edge_hi).
        t_in_cube = int(cand.event_specnum) - int(cube_t_offset)
        if not (t_edge_lo <= t_in_cube < t_edge_hi):
            new_flags |= int(CandidateFlags.TIME_EDGE_DROPPED)

        if new_flags == int(cand.flags):
            emit.append(cand)
        else:
            dropped.append(
                Candidate(
                    l=cand.l,
                    m=cand.m,
                    dm_fine=cand.dm_fine,
                    dm_idx=cand.dm_idx,
                    event_specnum=cand.event_specnum,
                    width_samples=cand.width_samples,
                    kernel_id=cand.kernel_id,
                    snr=cand.snr,
                    detector_version=cand.detector_version,
                    flags=new_flags,
                    search_node_id=cand.search_node_id,
                    gpu_half=cand.gpu_half,
                )
            )
    return emit, dropped
