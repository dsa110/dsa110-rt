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
    SparsityPattern,
    _per_baseline_uv_meters,
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
        ``(NBASE * NCHAN,) int64`` mapping flat ``(bls, ch)`` index to
        the filled-cell index in ``[0, N_filled)``, or to ``N_filled``
        (= sentinel) for out-of-grid / auto / outrigger baselines.
    cell_weights_cpu : torch.Tensor
        ``(N_filled,) float32`` count of (bls, ch) hits per filled
        cell. Constant per pattern; computed once at construction.
        Held on CPU; access via :attr:`cell_weights` for the device-
        resident mirror.
    """

    pattern: SparsityPattern
    device: torch.device
    cell_index_map: torch.Tensor
    cell_weights_cpu: torch.Tensor

    _cell_weights_device: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dev = torch.device(self.device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.device = dev

        if self.cell_index_map.dtype != torch.int64:
            raise TypeError(
                f"cell_index_map dtype must be int64; got "
                f"{self.cell_index_map.dtype}"
            )
        expected_n = NBASE * NCHAN_PER_CHGROUP
        if self.cell_index_map.shape != (expected_n,):
            raise ValueError(
                f"cell_index_map shape must be ({expected_n},); got "
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
        looks up each ``(bls, ch)`` cell in the pattern to produce a
        flat ``cell_index_map``. This is the ~ms-cost work that
        plan §4.2 step 5 absorbs into ``cmd: prepare``; tests + bench
        call it on every construction.

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

        # Per-channel wavelengths (chgroup-local).
        nu_GHz = np.asarray(
            [freq_GHz(chgroup, ch) for ch in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        wavelength_m = SPEED_OF_LIGHT_M_PER_S / (nu_GHz * 1e9)        # (NCHAN,)
        u_lam = (du_m[:, None] / wavelength_m[None, :])               # (Nkept, NCHAN)
        v_lam = (dv_m[:, None] / wavelength_m[None, :])
        u_lam = -u_lam                                                # F20
        v_lam = -v_lam

        max_baseline_lambda = float(np.max(np.maximum(
            np.abs(u_lam), np.abs(v_lam),
        )))
        cell_lambda = max_baseline_lambda * 2.0 / n_grid
        half = n_grid // 2
        ix_col_kept = np.rint(u_lam / cell_lambda).astype(np.int64) + half
        ix_row_kept = np.rint(v_lam / cell_lambda).astype(np.int64) + half
        in_grid_kept = (
            (ix_row_kept >= 0) & (ix_row_kept < n_grid)
            & (ix_col_kept >= 0) & (ix_col_kept < n_grid)
        )

        # Pack pattern (row, col) into a uint32 key. Build a dict
        # mapping packed key → filled-cell index.
        pat_keys = (pattern.ix_row.astype(np.uint32) << 16) | pattern.ix_col.astype(np.uint32)
        # `pat_keys` is sorted by virtue of the build_pattern unique()
        # sort, so we can use np.searchsorted instead of a Python dict.

        # ---- Build the (NBASE, NCHAN) cell_index map --------------------
        cell_idx = np.full((NBASE, NCHAN_PER_CHGROUP), n_filled, dtype=np.int64)

        # Index every (kept-bls, ch) into the pattern.
        kept_keys = (
            ix_row_kept.astype(np.uint32) << 16
        ) | ix_col_kept.astype(np.uint32)                              # (Nkept, NCHAN)
        # Searchsorted only inside the in-grid mask to skip out-of-grid
        # baselines (those keys would bind to a wrong cell otherwise
        # because searchsorted returns insertion index).
        in_grid_keys = kept_keys[in_grid_kept]
        positions = np.searchsorted(pat_keys, in_grid_keys)
        # Sanity: every in-grid key must match an existing pattern key.
        if positions.size:
            if int(positions.max()) >= pat_keys.size:
                raise RuntimeError(
                    "internal: in-grid (bls, ch) key not present in"
                    " pattern keys (build_pattern / from_pattern"
                    " geometric drift)."
                )
            if not np.all(pat_keys[positions] == in_grid_keys):
                raise RuntimeError(
                    "internal: in-grid (bls, ch) key mismatch against"
                    " pattern keys (build_pattern / from_pattern"
                    " geometric drift)."
                )

        # Scatter the filled-cell indices back into (NBASE, NCHAN).
        # We need to walk the kept-baseline axis, mapping kept_idx → the
        # full BLS index it came from.
        kept_idx_per_bls_inv = np.where(kept_per_bls >= 0)[0]         # (Nkept,)
        # Build a (Nkept, NCHAN) tensor of filled-cell indices, sentinel
        # in the out-of-grid slots.
        cell_idx_kept = np.full((kept_idx_per_bls_inv.size, NCHAN_PER_CHGROUP),
                                n_filled, dtype=np.int64)
        cell_idx_kept[in_grid_kept] = positions.astype(np.int64)
        # Place rows into cell_idx via the kept→bls map.
        cell_idx[kept_idx_per_bls_inv, :] = cell_idx_kept

        # ---- Per-cell weights = count of (bls, ch) → filled-cell hits ---
        weights = np.zeros(n_filled, dtype=np.float32)
        valid_mask = cell_idx < n_filled
        np.add.at(weights, cell_idx[valid_mask], 1.0)

        return cls(
            pattern=pattern,
            device=torch.device(device),
            cell_index_map=torch.from_numpy(cell_idx.reshape(-1)),
            cell_weights_cpu=torch.from_numpy(weights),
        )

    # ------------------------------------------------------------------
    # Hot-path API
    # ------------------------------------------------------------------

    @property
    def cell_weights(self) -> torch.Tensor:
        """``(N_filled,) float32`` per-cell sample-count for natural weighting.

        Computed once at :meth:`from_pattern` construction and cached on
        the gridder's device. Constant per pattern (does not depend on
        the input vis), so callers may read this once after construction
        and bind it into a noise-norm pipeline without re-fetching every
        call.

        Plan §4.2 step 5 + §3.6.5 G10 — the search-side detector
        normalises by per-pixel sample counts when computing Layer-1
        noise; this is the producer of those counts.
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
        if nch != NCHAN_PER_CHGROUP:
            raise ValueError(
                f"vis_stokes_i.shape[2]={nch} != NCHAN_PER_CHGROUP="
                f"{NCHAN_PER_CHGROUP}"
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
        # Flatten (NBASE, NCHAN) → (NBASE * NCHAN); broadcast index map
        # to all fast-vis tiles. ``scatter_add_`` requires same-shape
        # index + src tensors.
        src = vis_cfp32.reshape(n_fv, nb * nch)                       # (n_fv, NBASE*NCHAN)
        idx = self.cell_index_map.unsqueeze(0).expand(n_fv, -1)       # (n_fv, NBASE*NCHAN)

        # PyTorch ``scatter_add_`` doesn't support complex on all
        # backends (in particular CPU complex scatter is not vectorised
        # to the same path as float32 scatter). Split into real / imag
        # and scatter each — same answer, portable across CPU / CUDA.
        out_real = torch.zeros(
            (n_fv, n_filled + 1), dtype=torch.float32, device=self.device,
        )
        out_imag = torch.zeros_like(out_real)
        out_real.scatter_add_(1, idx, src.real.contiguous())
        out_imag.scatter_add_(1, idx, src.imag.contiguous())
        out_buf = torch.complex(out_real, out_imag)

        # Strip the sentinel slot.
        return out_buf[:, :n_filled].contiguous()
