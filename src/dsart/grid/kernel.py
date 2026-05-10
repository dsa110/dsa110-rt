"""GPU fast-vis gridder kernel (M3 chunk 3a; plan §4.2 line 1350).

Pinned by plan §3 line 305 (single-side ``+uv`` Stokes-I uv-grid; no
Hermitian conjugate-mirror cell) + plan §4.2 line 1350 (gridder API +
sparse-COO gather) + ``§8.M2-carryover`` F20 (``np.fft.ifft2`` ``(u, v)``
sign convention).

Pipeline placement (per plan §4.2 streaming pipeline, lines 1283-1346)::

    fast-corr GEMM  →  cal/RFI weight  →  pol-sum (Stokes I)  →
    [stage-1 per-channel coarse-DM]  →  ┌───────────────────────┐
                                        │ gridder (this module) │
                                        └───────────────────────┘
                                                 │
                                                 ▼
                                        sparse-COO value vector
                                          (n_fast_vis, N_filled) cfp32
                                                 │
                                                 ▼
                                  static-sky subtract → quantize → stage-2 FIFO

Inputs
======

A per-(fast-vis tile, baseline, channel) Stokes-I complex visibility
tensor of shape ``(n_fast_vis, NBASE, NCHAN)`` cfp32 — the output of
:func:`dsart.services.corr_fast_kernel.stokes_i_pol_sum` applied to
:meth:`dsart.services.corr_fast_kernel.FastCorrKernel.compute_split`.

Outputs
=======

A sparse-COO value tensor ``(n_fast_vis, N_filled)`` cfp32, one entry
per filled cell of the cached :class:`SparsityPattern`. Per-cell weight
counts (constant per pattern) are exposed via :attr:`cell_weights`
for downstream natural-weighting / noise-norm consumers.

Sign convention (F20)
=====================

The F20 ``(u, v)`` negation is captured **once** at pattern-build time
(see :mod:`dsart.grid.sparsity_pattern`). The gridder here takes the
pattern's pre-negated ``(ix_row, ix_col)`` as ground truth and never
recomputes ``(u, v)``, so the sign convention can never drift between
the two modules.

Performance notes
=================

* The per-(bls, ch) → filled-cell index map is precomputed once at
  :meth:`from_pattern` and cached on the device. Hot path is a single
  ``torch.scatter_add_`` per fast-vis tile (~ms on a 2080 Ti at default
  ops; cf. plan §4.2 step 5 ``grid_single_side_lut`` budget).
* Out-of-grid baselines / autos / outrigger-touching baselines (when a
  ``is_core_baseline_mask`` is supplied) are mapped to the **sentinel
  bin index ``N_filled``** of an oversized scatter buffer, then sliced
  off. This avoids a per-step boolean mask + saves the kernel-launch
  cost of a separate index-validity gate.

References
==========

* Plan §3 line 305 — single-side +uv Stokes-I uv-grid spec.
* Plan §4.2 lines 1283-1346 — streaming pipeline.
* Plan §4.2 line 1350 + G7/G10 — gridder API + accumulator dtype.
* :mod:`dsart.grid.sparsity_pattern` — pattern build (the producer of
  the ``(ix_row, ix_col)`` arrays this module consumes).
* :mod:`dsart.services.corr_fast_kernel` — upstream producer of the
  Stokes-I visibility tensor this module consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from dsart.common.constants import (
    NBASE,
    NCHAN_PER_CHGROUP,
    NU_CHGROUP_TOP_GHZ,
    freq_GHz,
)
from dsart.grid.sparsity_pattern import (
    SPEED_OF_LIGHT_M_PER_S,
    SUPPORTED_KERNEL_SUPPORTS,
    SparsityPattern,
    _per_baseline_uv_meters,
    gaussian_kernel_weights,
)


__all__ = [
    "FastVisGridder",
]


#: Sentinel cell index for (bls, ch) pairs that don't land in any filled
#: cell (out-of-grid / auto / outrigger). The scatter buffer is allocated
#: with one extra element ``[N_filled]`` so writes to the sentinel land
#: in a discarded slot. This is faster than a per-step boolean gate.
_SENTINEL_BIN: Final[int] = -1  # logical; actual index = N_filled


@dataclass
class FastVisGridder:
    """Stateful per-(device, sparsity-pattern) gridder kernel.

    Construct via :meth:`from_pattern`; the bare ``__init__`` is for
    advanced callers that have already pre-computed the cell-index map
    + cell weights themselves (e.g. unit tests with synthetic patterns).

    The kernel is **stateful only insofar as the per-(bls, ch) →
    filled-cell index map is cached** — it has no streaming state
    across calls. Calling :meth:`compute` is equivalent to a pure
    function ``(vis,) → (gridded,)`` once construction is done.

    Attributes
    ----------
    pattern : SparsityPattern
        The sparsity pattern this gridder was built for.
    device : torch.device
        Where the scatter runs.
    cell_index_map : torch.Tensor
        ``(NBASE * NCHAN * K * K,) int64`` mapping flat
        ``(bls, ch, dy, dx)`` tap index to the filled-cell index in
        ``[0, N_filled)``, or to ``N_filled`` (= sentinel) for
        out-of-grid / auto / outrigger taps. ``K = pattern.kernel_support``;
        for K=1 this collapses to ``(NBASE * NCHAN,)`` (one tap per
        (bls, ch)) and is bit-identical to the pre-G7 layout. Inner
        order over the K² taps is row-major
        (``i = (dy + K//2) * K + (dx + K//2)``) so it matches
        ``gaussian_kernel_weights(K).reshape(-1)``.
    cell_weights_cpu : torch.Tensor
        ``(N_filled,) float32`` per-cell **sum of squared per-tap
        Gaussian weights** :math:`\\sum_i w_i^2` accumulated over every
        in-grid (bls, ch, dy, dx) tap that lands in the cell — the
        natural-weighting weight for a tapered grid (cf. plan
        §3.6.5 G10). For K=1 (pillbox) every tap weight is 1.0 so
        this collapses to the per-cell **count** of (bls, ch) hits,
        bit-identical to the pre-G7 cell-weights array. Held on CPU;
        access via :attr:`cell_weights` for the device-resident mirror.
    """

    pattern: SparsityPattern
    device: torch.device
    cell_index_map: torch.Tensor
    cell_weights_cpu: torch.Tensor

    _cell_weights_device: torch.Tensor = field(init=False, repr=False)
    _tap_weights_device: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.device = dev

        kernel_support = int(self.pattern.kernel_support)
        if kernel_support not in SUPPORTED_KERNEL_SUPPORTS:
            raise ValueError(
                f"pattern.kernel_support={kernel_support} not in "
                f"{SUPPORTED_KERNEL_SUPPORTS}; G7 supports K ∈ {{1, 3, 5}}."
            )
        chan_sum_factor = int(getattr(self.pattern, "chan_sum_factor", 1))
        if chan_sum_factor < 1 or NCHAN_PER_CHGROUP % chan_sum_factor != 0:
            raise ValueError(
                f"pattern.chan_sum_factor={chan_sum_factor} invalid "
                f"(must be ≥ 1 and divide NCHAN_PER_CHGROUP="
                f"{NCHAN_PER_CHGROUP})."
            )
        nchan_eff = NCHAN_PER_CHGROUP // chan_sum_factor

        if self.cell_index_map.dtype != torch.int64:
            raise TypeError(
                f"cell_index_map dtype must be int64; got "
                f"{self.cell_index_map.dtype}"
            )
        expected_n = NBASE * nchan_eff * kernel_support * kernel_support
        if self.cell_index_map.shape != (expected_n,):
            raise ValueError(
                f"cell_index_map shape must be ({expected_n},) "
                f"(NBASE * NCHAN_eff * K * K, NCHAN_eff="
                f"{nchan_eff} = NCHAN_PER_CHGROUP // chan_sum_factor="
                f"{chan_sum_factor}, K={kernel_support}); got "
                f"{tuple(self.cell_index_map.shape)}"
            )
        n_filled = self.pattern.n_filled
        # All cell-index values must be in [0, N_filled] (sentinel ==
        # N_filled). torch.minmax skipped — direct .max() is fine for
        # the construction-time sanity check.
        if int(self.cell_index_map.min()) < 0:
            raise ValueError(
                "cell_index_map contains negative values; sentinel must"
                " be N_filled (non-negative)."
            )
        if int(self.cell_index_map.max()) > n_filled:
            raise ValueError(
                f"cell_index_map max {int(self.cell_index_map.max())} > "
                f"N_filled={n_filled}; sentinel index out of bounds."
            )
        if self.cell_weights_cpu.shape != (n_filled,):
            raise ValueError(
                f"cell_weights_cpu shape must be ({n_filled},); got "
                f"{tuple(self.cell_weights_cpu.shape)}"
            )
        if self.cell_weights_cpu.dtype != torch.float32:
            raise TypeError(
                f"cell_weights_cpu dtype must be float32; got "
                f"{self.cell_weights_cpu.dtype}"
            )

        # Migrate index map to the device + cache device-side weights.
        self.cell_index_map = self.cell_index_map.to(self.device)
        self._cell_weights_device = self.cell_weights_cpu.to(self.device)

        # Per-tap Gaussian weights (K*K,) float32 on the device. For
        # K=1 this is exactly ``[1.0]`` so the scatter math collapses
        # to a no-op multiply (bit-identical to pre-G7).
        tap_weights_np = gaussian_kernel_weights(kernel_support).astype(
            np.float32
        ).reshape(-1)
        self._tap_weights_device = torch.from_numpy(tap_weights_np).to(
            self.device
        )

    # ------------------------------------------------------------------
    # Constructor classmethod — the canonical entry point
    # ------------------------------------------------------------------

    @classmethod
    def from_pattern(
        cls,
        pattern: SparsityPattern,
        antpos_e: np.ndarray,
        antpos_n: np.ndarray,
        *,
        is_core_baseline_mask: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ) -> "FastVisGridder":
        """Build a gridder from an existing :class:`SparsityPattern`.

        Re-runs the same per-baseline ``(u, v)`` arithmetic that
        :func:`dsart.grid.sparsity_pattern.build_pattern` ran, then
        looks up each ``(bls, ch, dy, dx)`` tap (K² taps per
        ``(bls, ch)`` for ``K = pattern.kernel_support``) in the
        pattern to produce a flat ``cell_index_map`` of length
        ``NBASE * NCHAN * K * K``. The per-tap Gaussian weights are
        derived deterministically from K via
        :func:`dsart.grid.sparsity_pattern.gaussian_kernel_weights`,
        held on-device for the hot-path scatter and used to compute
        ``cell_weights = Σ w²`` (natural weighting for a tapered grid).
        For K=1 this collapses to one tap per (bls, ch) with weight
        1.0 — bit-identical to the pre-G7 layout. This is the ~ms-cost
        work that plan §4.2 step 5 absorbs into ``cmd: prepare``;
        tests + bench call it on every construction.

        Args:
            pattern: the sparsity pattern. Carries ``chgroup``,
                ``n_grid``, and ``kernel_support`` so we don't
                re-pass them.
            antpos_e, antpos_n: per-antenna (E, N) offsets in metres.
                **Must be identical to what ``build_pattern`` saw**;
                the antpos hash on the pattern is verified against the
                arrays passed here.
            is_core_baseline_mask: optional core-baseline mask. **Must
                be identical** to what ``build_pattern`` saw (otherwise
                the per-(bls, ch) index map may map cells to filled
                rows that don't correspond to that baseline's
                contribution).
            device: where the gridder runs. Defaults to CPU; production
                wires this to ``cuda:0`` per ``configs/numa_topology.yaml``.

        Returns:
            :class:`FastVisGridder` ready for :meth:`compute` calls.
        """
        # ---- Sanity: antpos arrays must match the pattern's hash --------
        from dsart.grid.sparsity_pattern import compute_antpos_hash

        ap_hash = compute_antpos_hash(antpos_e, antpos_n)
        if ap_hash != pattern.antpos_hash:
            raise ValueError(
                f"antpos hash {ap_hash} != pattern.antpos_hash "
                f"{pattern.antpos_hash}; the antpos arrays passed to "
                f"FastVisGridder.from_pattern must match build_pattern."
            )

        chgroup = pattern.chgroup
        n_grid = pattern.n_grid
        n_filled = pattern.n_filled
        kernel_support = int(pattern.kernel_support)
        if kernel_support not in SUPPORTED_KERNEL_SUPPORTS:
            raise ValueError(
                f"pattern.kernel_support={kernel_support} not in "
                f"{SUPPORTED_KERNEL_SUPPORTS}; G7 supports K ∈ {{1, 3, 5}}."
            )
        chan_sum_factor = int(getattr(pattern, "chan_sum_factor", 1))
        if chan_sum_factor < 1 or NCHAN_PER_CHGROUP % chan_sum_factor != 0:
            raise ValueError(
                f"pattern.chan_sum_factor={chan_sum_factor} invalid "
                f"(must be ≥ 1 and divide NCHAN_PER_CHGROUP="
                f"{NCHAN_PER_CHGROUP})."
            )
        nchan_eff = NCHAN_PER_CHGROUP // chan_sum_factor

        # ---- Re-run the geometric build (cheap; same as build_pattern) --
        du_m, dv_m = _per_baseline_uv_meters(
            antpos_e, antpos_n,
            is_core_baseline_mask=is_core_baseline_mask,
        )
        # Build the (NBASE,) → kept-baseline-index map. Negative entries
        # mark autos and outrigger-touching baselines (sentinel).
        nants = antpos_e.shape[0]
        kept_per_bls = np.full(NBASE, -1, dtype=np.int64)
        a_list = np.empty(NBASE, dtype=np.int64)
        b_list = np.empty(NBASE, dtype=np.int64)
        k = 0
        for a in range(nants):
            for b in range(a + 1):
                a_list[k] = a
                b_list[k] = b
                k += 1
        is_cross = a_list != b_list
        keep = is_cross
        if is_core_baseline_mask is not None:
            keep = keep & np.asarray(is_core_baseline_mask, dtype=bool)
        kept_per_bls[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        # Per-channel wavelengths (chgroup-local). F33: when
        # chan_sum_factor > 1, use the band-CENTER frequency of each
        # summed group — must match build_pattern exactly.
        nu_GHz_full = np.asarray(
            [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        if chan_sum_factor == 1:
            nu_GHz = nu_GHz_full
        else:
            nu_GHz = nu_GHz_full.reshape(
                nchan_eff, chan_sum_factor,
            ).mean(axis=1)
        wavelength_m = SPEED_OF_LIGHT_M_PER_S / (nu_GHz * 1e9)        # (NCHAN_eff,)
        u_lam = (du_m[:, None] / wavelength_m[None, :])               # (Nkept, NCHAN)
        v_lam = (dv_m[:, None] / wavelength_m[None, :])
        u_lam = -u_lam                                                # F20
        v_lam = -v_lam

        # F28: read the resolved ``cell_lambda`` straight off the
        # pattern instead of recomputing. ``build_pattern`` either
        # auto-fit it from this chgroup's max baseline-in-λ
        # (legacy) or accepted an external common value (F28); in
        # either case the pattern carries the canonical resolved
        # number, so re-deriving it here would re-introduce the
        # exact divergence between corr and search ends that
        # F28 + the cell_lambda-in-pattern-id field exist to
        # prevent. (Pre-F28 builds default ``cell_lambda`` to 0.0
        # on the dataclass; fall back to recomputing in that case
        # to preserve the legacy contract with any external
        # callers who still construct SparsityPattern by hand.)
        cell_lambda = float(getattr(pattern, "cell_lambda", 0.0))
        if cell_lambda <= 0.0:
            max_baseline_lambda = float(np.max(np.maximum(
                np.abs(u_lam), np.abs(v_lam),
            )))
            cell_lambda = max_baseline_lambda * 2.0 / n_grid
        half = n_grid // 2
        ix_col_kept_center = (
            np.rint(u_lam / cell_lambda).astype(np.int64) + half
        )                                                              # (Nkept, NCHAN)
        ix_row_kept_center = (
            np.rint(v_lam / cell_lambda).astype(np.int64) + half
        )

        # G7: expand each (kept-bls, ch) center into a K×K neighborhood.
        # Inner tap order is row-major over (dy, dx) so the flat index
        # (dy + half_K) * K + (dx + half_K) matches
        # ``gaussian_kernel_weights(K).reshape(-1)``.
        K = kernel_support
        half_kernel = K // 2
        offsets = np.arange(-half_kernel, half_kernel + 1, dtype=np.int64)
        dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
        ix_row_kept_taps = (
            ix_row_kept_center[:, :, None, None]
            + dy_grid[None, None, :, :]
        )                                                              # (Nkept, NCHAN, K, K)
        ix_col_kept_taps = (
            ix_col_kept_center[:, :, None, None]
            + dx_grid[None, None, :, :]
        )
        in_grid_kept_taps = (
            (ix_row_kept_taps >= 0) & (ix_row_kept_taps < n_grid)
            & (ix_col_kept_taps >= 0) & (ix_col_kept_taps < n_grid)
        )

        # Pack pattern (row, col) into a uint32 key. ``pat_keys`` is
        # sorted by virtue of the build_pattern unique() sort, so we
        # can use np.searchsorted instead of a Python dict.
        pat_keys = (
            pattern.ix_row.astype(np.uint32) << 16
        ) | pattern.ix_col.astype(np.uint32)

        # ---- Build the (NBASE, NCHAN_eff, K, K) cell_index map -------------
        cell_idx_taps = np.full(
            (NBASE, nchan_eff, K, K), n_filled, dtype=np.int64
        )

        kept_tap_keys = (
            ix_row_kept_taps.astype(np.uint32) << 16
        ) | ix_col_kept_taps.astype(np.uint32)                         # (Nkept, NCHAN, K, K)
        # Searchsorted only inside the in-grid mask to skip out-of-grid
        # taps (those keys would bind to a wrong cell otherwise because
        # searchsorted returns insertion index).
        in_grid_keys = kept_tap_keys[in_grid_kept_taps]
        positions = np.searchsorted(pat_keys, in_grid_keys)
        # Sanity: every in-grid tap key must match an existing pattern
        # key. ``build_pattern`` unions every K×K-neighborhood cell into
        # the pattern, so an in-grid tap is by construction in the
        # pattern; a mismatch here means the two ends drifted.
        if positions.size:
            if int(positions.max()) >= pat_keys.size:
                raise RuntimeError(
                    "internal: in-grid (bls, ch, dy, dx) tap key not"
                    " present in pattern keys (build_pattern /"
                    " from_pattern geometric drift)."
                )
            if not np.all(pat_keys[positions] == in_grid_keys):
                raise RuntimeError(
                    "internal: in-grid (bls, ch, dy, dx) tap key"
                    " mismatch against pattern keys (build_pattern /"
                    " from_pattern geometric drift)."
                )

        # Scatter the filled-cell indices back into
        # (NBASE, NCHAN_eff, K, K) via the kept→bls map.
        kept_idx_per_bls_inv = np.where(kept_per_bls >= 0)[0]          # (Nkept,)
        cell_idx_kept_taps = np.full(
            (kept_idx_per_bls_inv.size, nchan_eff, K, K),
            n_filled, dtype=np.int64,
        )
        cell_idx_kept_taps[in_grid_kept_taps] = positions.astype(np.int64)
        cell_idx_taps[kept_idx_per_bls_inv, :, :, :] = cell_idx_kept_taps

        # ---- Per-cell weights = Σ w² over in-grid taps -----------------
        # Natural weighting for a tapered grid (plan §3.6.5 G10): the
        # variance-minimising estimator weights samples by their
        # squared gridding kernel coefficient. For K=1 every tap weight
        # is 1.0 so Σ w² collapses to the per-cell sample count.
        tap_weights_2d = gaussian_kernel_weights(K).astype(np.float32)  # (K, K)
        # Broadcast tap weights into (NBASE, NCHAN, K, K) shape so each
        # tap pairs with its weight, then sum-of-squares per filled cell.
        tap_w_per_idx = np.broadcast_to(
            tap_weights_2d[None, None, :, :],
            cell_idx_taps.shape,
        )
        cell_weights = np.zeros(n_filled, dtype=np.float32)
        cell_idx_flat = cell_idx_taps.reshape(-1)
        tap_w_flat_sq = (tap_w_per_idx.reshape(-1) ** 2).astype(np.float32)
        valid_mask = cell_idx_flat < n_filled
        np.add.at(
            cell_weights, cell_idx_flat[valid_mask], tap_w_flat_sq[valid_mask],
        )

        return cls(
            pattern=pattern,
            device=torch.device(device),
            cell_index_map=torch.from_numpy(cell_idx_flat),
            cell_weights_cpu=torch.from_numpy(cell_weights),
        )

    # ------------------------------------------------------------------
    # Hot-path API
    # ------------------------------------------------------------------

    @property
    def cell_weights(self) -> torch.Tensor:
        """``(N_filled,) float32`` per-cell **sum-of-squared per-tap weights**.

        For each filled cell ``c`` returns
        :math:`\\sum_{i\\,:\\,c_i = c}\\,w_i^2`, summed over every
        ``(bls, ch, dy, dx)`` tap whose Gaussian-weighted contribution
        landed in cell ``c`` from any in-grid baseline. This is the
        **natural-weighting** weight for a tapered grid (plan §3.6.5
        G10): the variance-minimising estimator weights samples by
        their squared gridding-kernel coefficient, not by the kernel
        coefficient itself.

        For ``kernel_support = 1`` (pillbox) every per-tap weight is
        ``1.0`` so :math:`\\sum w^2` collapses to the per-cell sample
        count — bit-identical to the pre-G7 ``cell_weights`` array.
        For ``K ∈ {3, 5}`` (G7 Gaussian taper) the central-cell weight
        approaches the count, while corner-tap contributions are
        suppressed quadratically by the Gaussian profile.

        Computed once at :meth:`from_pattern` construction and cached
        on the gridder's device. Constant per pattern (does not depend
        on the input vis), so callers may read this once after
        construction and bind it into a noise-norm pipeline without
        re-fetching every call.

        Plan §4.2 step 5 + §3.6.5 G10 — the search-side detector
        normalises by per-pixel weight when computing Layer-1 noise;
        this is the producer of those weights.
        """
        return self._cell_weights_device

    def compute(self, vis_stokes_i: torch.Tensor) -> torch.Tensor:
        """Grid one batch of fast-vis tiles into the sparse-COO output.

        Args:
            vis_stokes_i: complex visibility tensor of shape
                ``(n_fast_vis, NBASE, NCHAN)`` in cfp32 (= ``complex64``).
                Output of
                :func:`dsart.services.corr_fast_kernel.stokes_i_pol_sum`.
                Out-of-grid / auto / outrigger samples are silently
                discarded by the precomputed cell-index map; callers do
                NOT need to mask the input.

        Returns:
            ``(n_fast_vis, N_filled)`` complex64 — sum of contributions
            from every ``(bls, ch)`` pair that maps to each filled
            cell. **F32 accumulator** per plan §3.6.5 G10 (raw fp16
            accumulation would lose ~5% per cell at the ~10⁴ samples-
            per-cell hit count).
        """
        if not vis_stokes_i.is_complex():
            raise TypeError(
                f"vis_stokes_i must be complex; got {vis_stokes_i.dtype}"
            )
        if vis_stokes_i.ndim != 3:
            raise ValueError(
                f"vis_stokes_i must be 3D (n_fast_vis, NBASE, NCHAN); "
                f"got {vis_stokes_i.ndim}D shape "
                f"{tuple(vis_stokes_i.shape)}"
            )
        n_fv, nb, nch = vis_stokes_i.shape
        if nb != NBASE:
            raise ValueError(f"vis_stokes_i.shape[1]={nb} != NBASE={NBASE}")
        chan_sum_factor = int(getattr(self.pattern, "chan_sum_factor", 1))
        nchan_eff = NCHAN_PER_CHGROUP // chan_sum_factor
        if nch != nchan_eff:
            raise ValueError(
                f"vis_stokes_i.shape[2]={nch} != NCHAN_eff="
                f"{nchan_eff} (NCHAN_PER_CHGROUP // "
                f"chan_sum_factor={chan_sum_factor})"
            )
        if vis_stokes_i.device != self.device:
            raise ValueError(
                f"vis_stokes_i on {vis_stokes_i.device}, gridder on "
                f"{self.device}"
            )

        # Promote to cfp32 for the accumulator (G10). No-op for
        # complex64 inputs; upcasts complex32 / fp16-real-imag inputs.
        vis_cfp32 = vis_stokes_i.to(torch.complex64)

        n_filled = self.pattern.n_filled
        # Flatten (NBASE, NCHAN) → (NBASE * NCHAN); each (bls, ch)
        # sample is replicated K² times and weighted by the per-tap
        # Gaussian coefficients. For K=1 (pillbox) the weight is 1.0
        # so the multiply is exact in fp32 and the scatter math
        # collapses to the pre-G7 single-tap path bit-identically.
        src = vis_cfp32.reshape(n_fv, nb * nch)                        # (n_fv, NBASE*NCHAN)
        # ``self._tap_weights_device`` is a (K*K,) float32 tensor
        # ordered to match the inner (dy, dx) tap order of
        # ``cell_index_map``.
        weights_flat = self._tap_weights_device                        # (K*K,)
        n_taps = int(weights_flat.shape[0])                            # = K*K

        # PyTorch ``scatter_add_`` doesn't support complex on all
        # backends (in particular CPU complex scatter is not vectorised
        # to the same path as float32 scatter). Split into real / imag
        # and scatter each — same answer, portable across CPU / CUDA.
        out_real = torch.zeros(
            (n_fv, n_filled + 1), dtype=torch.float32, device=self.device,
        )
        out_imag = torch.zeros_like(out_real)

        # Per-tap accumulation loop. For each of the K² taps, weight the
        # full (n_fv, NBASE*NCHAN) source tensor by ``weights_flat[t]``
        # and accumulate into the cells indexed by ``cell_index_map[:, t]``
        # using ``Tensor.index_add_`` (1-D index = 1.8 MB int64 at the
        # production op-point) instead of ``scatter_add_`` (would need
        # an (n_fv, NSRC) int64 index = ~3.7 GB at chunk=2 — blows the
        # 2080 Ti budget). For K=1, ``weights_flat[0] == 1.0`` so the
        # multiply is skipped (bit-identical to weighting by 1.0). The
        # K=1 path then collapses to two ``index_add_`` calls (real +
        # imag), no per-tap loop overhead.
        cim = self.cell_index_map.reshape(-1, n_taps)                 # (NBASE*NCHAN_eff, K*K)
        if n_taps == 1:
            # K=1 fast path
            idx_1d = cim[:, 0]                                         # (NSRC,) int64
            src_real_c = src.real.contiguous()
            src_imag_c = src.imag.contiguous()
            out_real.index_add_(1, idx_1d, src_real_c)
            out_imag.index_add_(1, idx_1d, src_imag_c)
            del src_real_c, src_imag_c
        else:
            for t in range(n_taps):
                w = weights_flat[t]
                idx_t = cim[:, t]                                      # (NSRC,) int64
                src_w_real = (src.real * w).contiguous()
                src_w_imag = (src.imag * w).contiguous()
                out_real.index_add_(1, idx_t, src_w_real)
                out_imag.index_add_(1, idx_t, src_w_imag)
                del src_w_real, src_w_imag
        out_buf = torch.complex(out_real, out_imag)

        # Strip the sentinel slot.
        return out_buf[:, :n_filled].contiguous()
