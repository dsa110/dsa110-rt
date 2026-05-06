"""FastVisGridder acceptance tests (M3 chunk 3a).

Pinned by plan §3 line 305 + §4.2 line 1350.

Coverage
========

* Output shape: ``(n_fast_vis, NBASE, NCHAN)`` cfp32 input →
  ``(n_fast_vis, N_filled)`` cfp32 output.
* Numerical parity against :func:`tools.viz.common.grid_uv_natural`
  (the slow-corr reference, F20-corrected): for one fast-vis tile,
  scattering the gridder's output back into a dense ``(N_grid, N_grid)``
  grid produces the same set of filled cells AND the same per-cell
  sums to ≤ 1e-4 relative.
* Zero vis → zero grid output.
* Auto-correlations are excluded from the gridded output.
* Outrigger-touching baselines are zeroed when ``is_core_baseline_mask``
  is provided.
* Per-cell weight (sample count) is constant per pattern and tracks
  the (bls, ch) fan-in from the cell-index map.

References
==========

* Plan §3 line 305 — single-side +uv Stokes-I uv-grid spec.
* Plan §4.2 line 1350 — gridder + sparse-COO gather.
* :mod:`dsart.grid.kernel` — module under test.
* :mod:`tools.viz.common` — F20 reference imager (Class C; called
  read-only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dsart.common.constants import (
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
    N_GRID_DEFAULT,
    freq_GHz,
)
from dsart.grid.kernel import FastVisGridder
from dsart.grid.sparsity_pattern import (
    SPEED_OF_LIGHT_M_PER_S,
    SparsityPattern,
    build_pattern,
)


# ---------------------------------------------------------------------------
# Synthetic antpos + helpers (mirrors tests/test_sparsity_pattern.py)
# ---------------------------------------------------------------------------


def _synth_antpos(seed: int = 20260505) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _core_baseline_mask(n_core: int = 82) -> np.ndarray:
    nbase = NANTS * (NANTS + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(NANTS):
        for b in range(a + 1):
            mask[k] = (a < n_core) and (b < n_core)
            k += 1
    return mask


def _sparse_4ant_pattern_and_gridder(
    n_grid: int = 64,
    chgroup: int = 0,
    device: str = "cpu",
) -> tuple[
    SparsityPattern, FastVisGridder, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Build a 4-active-ant pattern + gridder for parity tests.

    Returns:
        (pattern, gridder, antpos_e, antpos_n, mask, cell_lambda).
        ``cell_lambda`` is the gridder's internal cell scale, exposed
        so callers can pass the matching ``fov_rad`` to
        :func:`tools.viz.common.grid_uv_natural`.
    """
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    ant_pos = np.array([
        [0.0, 0.0],
        [120.0, 0.0],
        [-50.0, 80.0],
        [200.0, -30.0],
    ], dtype=np.float32)
    e[:4] = ant_pos[:, 0]
    n[:4] = ant_pos[:, 1]

    mask = np.zeros(NANTS * (NANTS + 1) // 2, dtype=bool)
    for a in range(4):
        for b in range(a):
            bls_idx = a * (a + 1) // 2 + b
            mask[bls_idx] = True

    pattern = build_pattern(
        e, n,
        chgroup=chgroup,
        dec_deg=37.234,
        n_grid=n_grid,
        kernel_support=1,
        is_core_baseline_mask=mask,
    )
    gridder = FastVisGridder.from_pattern(
        pattern, e, n,
        is_core_baseline_mask=mask,
        device=device,
    )

    # Recompute cell_lambda (same code path as build_pattern + gridder)
    # so the parity test can pass a matching `fov_rad` to grid_uv_natural.
    nu_GHz = np.asarray(
        [freq_GHz(chgroup, c) for c in range(NCHAN_PER_CHGROUP)],
        dtype=np.float64,
    )
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (nu_GHz * 1e9)
    bls_uv_m = []
    for a in range(4):
        for b in range(a):
            bls_uv_m.append((ant_pos[a, 0] - ant_pos[b, 0],
                             ant_pos[a, 1] - ant_pos[b, 1]))
    bls_uv_m_arr = np.asarray(bls_uv_m, dtype=np.float32)
    u_all = bls_uv_m_arr[:, 0:1] / wavelength_m[None, :]
    v_all = bls_uv_m_arr[:, 1:2] / wavelength_m[None, :]
    max_baseline_lambda = float(np.max(np.maximum(np.abs(u_all), np.abs(v_all))))
    cell_lambda = max_baseline_lambda * 2.0 / n_grid
    return pattern, gridder, e, n, mask, cell_lambda


def _vis_for_4ant_baselines(value: complex = 1.0 + 0.0j,
                            ch_only: int | None = None) -> torch.Tensor:
    """Build a (1, NBASE, NCHAN) cfp32 vis tensor with non-zero entries
    only on the 6 cross-baselines among ants {0, 1, 2, 3}.

    All other baselines are zero (so they won't contribute to the
    gridder output regardless of the cell-index map). If ``ch_only``
    is given, only that channel is non-zero.
    """
    vis = torch.zeros((1, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64)
    bls_idx_list = []
    for a in range(4):
        for b in range(a):
            bls_idx_list.append(a * (a + 1) // 2 + b)
    bls_idx_t = torch.tensor(bls_idx_list, dtype=torch.long)
    if ch_only is None:
        vis[0, bls_idx_t, :] = value
    else:
        vis[0, bls_idx_t, ch_only] = value
    return vis


# ---------------------------------------------------------------------------
# Output shape + dtype
# ---------------------------------------------------------------------------


class TestGridderOutputShape:

    def test_output_shape(self) -> None:
        pat, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        n_fv = 3
        vis = torch.zeros(
            (n_fv, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64
        )
        out = g.compute(vis)
        assert out.shape == (n_fv, pat.n_filled)
        assert out.dtype == torch.complex64

    def test_cell_weights_shape_and_constancy(self) -> None:
        pat, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        w1 = g.cell_weights
        w2 = g.cell_weights
        assert w1.shape == (pat.n_filled,)
        assert w1.dtype == torch.float32
        # Same tensor across calls (no per-call recomputation).
        assert torch.equal(w1, w2)
        # Sum of weights equals the number of (bls, ch) pairs that hit
        # any filled cell — i.e. the count of non-sentinel entries in
        # the cell_index_map.
        n_valid = int((g.cell_index_map < pat.n_filled).sum())
        assert int(w1.sum().item()) == n_valid


# ---------------------------------------------------------------------------
# Parity against grid_uv_natural (F20 reference)
# ---------------------------------------------------------------------------


class TestGridderParityWithGridUvNatural:
    """Per-cell sums match the slow-corr reference imager."""

    def test_single_channel_dense_scatter_matches_grid_uv_natural(self) -> None:
        try:
            from tools.viz.common import grid_uv_natural
        except ImportError:
            pytest.skip("tools.viz.common not importable from test env.")

        n_grid = 64
        ch_test = 23
        pat, g, e, n_, mask, cell_lambda = _sparse_4ant_pattern_and_gridder(
            n_grid=n_grid,
        )

        # Synthetic vis: unit value on all 6 cross-baselines AT ch_test
        # ONLY. Zero elsewhere.
        vis = _vis_for_4ant_baselines(value=1.0 + 0.0j, ch_only=ch_test)
        out = g.compute(vis)                                          # (1, N_filled)

        # Scatter back into a dense (N_grid, N_grid) grid via the
        # pattern.
        dense_us = np.zeros((n_grid, n_grid), dtype=np.complex64)
        rows = pat.ix_row.astype(int)
        cols = pat.ix_col.astype(int)
        np.add.at(
            dense_us, (rows, cols), out.numpy()[0],
        )

        # Reference imager: same antpos arithmetic, F20-corrected.
        # grid_uv_natural integrates over channels, so we feed it a
        # (Nbls, Nfreqs, 1) vis with the same ch_test value and
        # zero on all other channels — its sum-over-channels equals
        # the single-channel value.
        bls_uv_m = []
        for a in range(4):
            for b in range(a):
                bls_uv_m.append((e[a] - e[b], n_[a] - n_[b]))
        bls_uv_m_arr = np.asarray(bls_uv_m, dtype=np.float32)
        uvw_m = np.zeros((6, 3), dtype=np.float32)
        uvw_m[:, 0] = bls_uv_m_arr[:, 0]
        uvw_m[:, 1] = bls_uv_m_arr[:, 1]
        nu_GHz = np.asarray(
            [freq_GHz(0, c) for c in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        vis_ref = np.zeros((6, NCHAN_PER_CHGROUP, 1), dtype=np.complex64)
        vis_ref[:, ch_test, 0] = 1.0 + 0.0j

        fov_rad = 1.0 / cell_lambda
        grid_ref, _w_ref = grid_uv_natural(
            vis_ref, uvw_m, np.asarray(nu_GHz * 1e9, dtype=np.float64),
            n_grid=n_grid, fov_rad=fov_rad, pol=0, drop_autos=True,
        )

        # Filled-cell SET parity.
        gridder_cells = set(map(tuple, np.argwhere(dense_us != 0)))
        ref_cells = set(map(tuple, np.argwhere(grid_ref != 0)))
        assert gridder_cells == ref_cells, (
            f"filled-cell set mismatch:\n  gridder = {sorted(gridder_cells)}\n"
            f"  reference = {sorted(ref_cells)}"
        )
        # Per-cell sum parity to ≤ 1e-4 relative.
        diff = np.abs(dense_us - grid_ref)
        max_abs = float(np.max(diff))
        max_ref = float(np.max(np.abs(grid_ref)))
        rel_err = max_abs / max_ref if max_ref > 0 else 0.0
        assert rel_err < 1e-4, (
            f"per-cell sum mismatch: max abs diff = {max_abs:.3e}, "
            f"max ref = {max_ref:.3e}, rel = {rel_err:.3e}"
        )

    def test_multi_channel_dense_scatter_matches_grid_uv_natural(self) -> None:
        """Same parity gate but with vis on EVERY channel (the
        production scenario; any per-channel cell-index drift would
        show up as a per-cell sum diff)."""
        try:
            from tools.viz.common import grid_uv_natural
        except ImportError:
            pytest.skip("tools.viz.common not importable from test env.")

        n_grid = 64
        pat, g, e, n_, mask, cell_lambda = _sparse_4ant_pattern_and_gridder(
            n_grid=n_grid,
        )
        # Unit vis on all 6 cross-baselines, all 384 channels.
        vis = _vis_for_4ant_baselines(value=1.0 + 0.0j, ch_only=None)
        out = g.compute(vis)
        dense_us = np.zeros((n_grid, n_grid), dtype=np.complex64)
        np.add.at(
            dense_us,
            (pat.ix_row.astype(int), pat.ix_col.astype(int)),
            out.numpy()[0],
        )

        bls_uv_m = []
        for a in range(4):
            for b in range(a):
                bls_uv_m.append((e[a] - e[b], n_[a] - n_[b]))
        bls_uv_m_arr = np.asarray(bls_uv_m, dtype=np.float32)
        uvw_m = np.zeros((6, 3), dtype=np.float32)
        uvw_m[:, 0] = bls_uv_m_arr[:, 0]
        uvw_m[:, 1] = bls_uv_m_arr[:, 1]
        nu_GHz = np.asarray(
            [freq_GHz(0, c) for c in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        vis_ref = np.ones(
            (6, NCHAN_PER_CHGROUP, 1), dtype=np.complex64,
        )
        fov_rad = 1.0 / cell_lambda
        grid_ref, _ = grid_uv_natural(
            vis_ref, uvw_m, np.asarray(nu_GHz * 1e9, dtype=np.float64),
            n_grid=n_grid, fov_rad=fov_rad, pol=0, drop_autos=True,
        )

        # Filled-cell SET parity.
        gridder_cells = set(map(tuple, np.argwhere(dense_us != 0)))
        ref_cells = set(map(tuple, np.argwhere(grid_ref != 0)))
        assert gridder_cells == ref_cells, (
            f"multi-ch filled-cell set mismatch:\n"
            f"  gridder − ref = {sorted(gridder_cells - ref_cells)[:10]}\n"
            f"  ref − gridder = {sorted(ref_cells - gridder_cells)[:10]}"
        )
        diff = np.abs(dense_us - grid_ref)
        max_abs = float(np.max(diff))
        max_ref = float(np.max(np.abs(grid_ref)))
        rel_err = max_abs / max_ref if max_ref > 0 else 0.0
        assert rel_err < 1e-4, (
            f"multi-ch per-cell sum mismatch: max_abs = {max_abs:.3e}, "
            f"max_ref = {max_ref:.3e}, rel = {rel_err:.3e}"
        )


# ---------------------------------------------------------------------------
# Trivial inputs / exclusions
# ---------------------------------------------------------------------------


class TestGridderTrivialAndExclusions:

    def test_zero_vis_yields_zero_grid(self) -> None:
        pat, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        vis = torch.zeros(
            (2, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64
        )
        out = g.compute(vis)
        assert out.shape == (2, pat.n_filled)
        assert torch.all(out == 0)

    def test_autos_excluded(self) -> None:
        """Strong autocorrelations only ⇒ all-zero grid output.

        Builds a vis tensor that is nonzero ONLY on the diagonal
        baselines ``(a, a)`` (bls_idx ``a*(a+1)/2 + a``). Since the
        gridder excludes autos at pattern-build time
        (``_per_baseline_uv_meters`` drops ``a == b``), all those
        contributions land at the sentinel and are discarded.
        """
        pat, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        vis = torch.zeros(
            (1, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64
        )
        # Set strong autos: bls_idx = a*(a+1)/2 + a = (a*(a+3))/2
        for a in range(NANTS):
            bls_idx = a * (a + 1) // 2 + a
            vis[0, bls_idx, :] = 1.0e6 + 1.0e6j
        out = g.compute(vis)
        assert torch.all(out == 0), (
            f"autos leaked into grid: max |out| = "
            f"{float(torch.max(out.abs()).item())}"
        )

    def test_outriggers_excluded_when_mask_present(self) -> None:
        """Strong outrigger-touching baselines + core mask ⇒ zero grid.

        Builds a 96-ant antpos with 82 core ants + 14 outriggers (same
        as :func:`_synth_antpos`) and a core-only mask; sets vis to 0
        on every (a, b) with both ants in the core, and to a large
        value on every (a, b) where at least one ant is an outrigger.
        Since the mask drops outrigger-touching baselines from the
        cell-index map, those contributions land at the sentinel and
        the grid output is zero.
        """
        e, n = _synth_antpos(seed=11)
        mask = _core_baseline_mask(n_core=82)
        pat = build_pattern(
            e, n,
            chgroup=0,
            dec_deg=37.234,
            n_grid=64,                                                # smaller for fast test
            kernel_support=1,
            is_core_baseline_mask=mask,
        )
        g = FastVisGridder.from_pattern(
            pat, e, n, is_core_baseline_mask=mask, device="cpu",
        )

        vis = torch.zeros(
            (1, NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64
        )
        # Touch only outrigger-touching baselines (mask == False).
        outrigger_bls = np.where(~mask)[0]
        # Drop autos from the outrigger-touching set (autos are
        # diagonals, both bits are the same ant).
        nbase = NANTS * (NANTS + 1) // 2
        is_auto = np.zeros(nbase, dtype=bool)
        k = 0
        for a in range(NANTS):
            for b in range(a + 1):
                if a == b:
                    is_auto[k] = True
                k += 1
        outrigger_cross_bls = outrigger_bls[~is_auto[outrigger_bls]]
        for bls_idx in outrigger_cross_bls:
            vis[0, bls_idx, :] = 1.0e3 + 1.0e3j
        out = g.compute(vis)
        assert torch.all(out == 0), (
            f"outriggers leaked into grid: max |out| = "
            f"{float(torch.max(out.abs()).item())}"
        )


# ---------------------------------------------------------------------------
# Construction / antpos sanity
# ---------------------------------------------------------------------------


class TestGridderConstruction:

    def test_from_pattern_rejects_mismatched_antpos(self) -> None:
        e, n = _synth_antpos(seed=1)
        mask = _core_baseline_mask()
        pat = build_pattern(
            e, n, chgroup=0, dec_deg=37.0, n_grid=64,
            is_core_baseline_mask=mask,
        )
        # Perturb antpos → different antpos_hash.
        e_bad = e.copy()
        e_bad[5] += 5.0
        with pytest.raises(ValueError, match="antpos_hash"):
            FastVisGridder.from_pattern(
                pat, e_bad, n, is_core_baseline_mask=mask, device="cpu",
            )

    def test_compute_rejects_wrong_dtype(self) -> None:
        pat, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        vis = torch.zeros(
            (1, NBASE, NCHAN_PER_CHGROUP), dtype=torch.float32
        )
        with pytest.raises(TypeError, match="complex"):
            g.compute(vis)

    def test_compute_rejects_wrong_shape(self) -> None:
        _, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
        vis = torch.zeros(
            (NBASE, NCHAN_PER_CHGROUP), dtype=torch.complex64
        )
        with pytest.raises(ValueError, match="3D"):
            g.compute(vis)

        vis = torch.zeros(
            (1, NBASE, NCHAN_PER_CHGROUP - 1), dtype=torch.complex64
        )
        with pytest.raises(ValueError, match="NCHAN"):
            g.compute(vis)


# ---------------------------------------------------------------------------
# Linearity smoke
# ---------------------------------------------------------------------------


def test_gridder_is_linear_in_vis() -> None:
    """``compute(αv₁ + βv₂) = α·compute(v₁) + β·compute(v₂)`` (exact in cfp32).

    The gridder is a per-(bls, ch) → cell scatter-add, so it is
    perfectly linear. This catches any accidental conditional / mask
    that depends on the input (e.g. a "drop NaN" gate that would
    silently discard contributions).
    """
    _, g, *_ = _sparse_4ant_pattern_and_gridder(n_grid=64)
    rng = np.random.default_rng(seed=7)
    v1_np = (rng.normal(size=(1, NBASE, NCHAN_PER_CHGROUP))
             + 1j * rng.normal(size=(1, NBASE, NCHAN_PER_CHGROUP))).astype(np.complex64)
    v2_np = (rng.normal(size=(1, NBASE, NCHAN_PER_CHGROUP))
             + 1j * rng.normal(size=(1, NBASE, NCHAN_PER_CHGROUP))).astype(np.complex64)
    alpha, beta = 0.7 + 0.2j, -0.3 - 0.5j
    v1 = torch.from_numpy(v1_np)
    v2 = torch.from_numpy(v2_np)
    o1 = g.compute(v1)
    o2 = g.compute(v2)
    o_combined = g.compute(alpha * v1 + beta * v2)
    o_expected = alpha * o1 + beta * o2
    diff = (o_combined - o_expected).abs().max().item()
    # cfp32 round-off bound: ~1e-5 absolute.
    assert diff < 1e-4, f"linearity broken: max abs diff = {diff:.3e}"
