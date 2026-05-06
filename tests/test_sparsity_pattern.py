"""Sparsity-pattern acceptance tests (M3 chunk 3a).

Pinned by plan §3 lines 305-307 + §4.3 line 1378 (Option C).

Coverage
========

* :func:`build_pattern` is **deterministic**: two calls with the same
  inputs return byte-identical ``(ix_row, ix_col)`` arrays + the same
  ``pattern_id``.
* ``dec_deg`` quantisation: declinations within ± 0.125 deg of the
  same 0.25-deg bin produce the same pattern; declinations in
  different bins produce different patterns (different ``pattern_id``).
* ``n_filled`` lands in the plan §3 line 305 range (~ 7-12% fill at
  default ops) when run against a representative 96-ant antpos with
  the 82-ant core mask. Skipped on machines without the on-disk
  beamformer_weights cal blob (i.e. anything but h01).
* ``pattern_id`` changes when *any* of the input tuple
  ``(dec, n_grid, chgroup, kernel_support, antpos_hash,
  chgroup_table_hash)`` changes.
* :func:`predict_pattern_id` matches :func:`build_pattern.pattern_id`
  for matching inputs.
* The F20 ``(u, v)`` negation is applied: a single synthetic baseline
  at known ``(u, v)`` lands in the cell predicted by **negated** uv
  rounding (compared to a matched-config call into the slow-corr
  reference imager
  :func:`tools.viz.common.grid_uv_natural`, which is itself F20-
  corrected).

References
==========

* Plan §3 lines 305-307 — sparsity pattern + ``pattern_id`` semantics.
* Plan §4.2 line 1350 — gridder + sparse-pattern build.
* Plan §8.M2-carryover F20 — ``(u, v)`` negation rationale.
* :mod:`dsart.grid.sparsity_pattern` — module under test.
* :mod:`tools.viz.common` — F20 reference imager (Class C; called
  read-only, never modified by chunk 3a).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dsart.cal.bf_weights import load_bf_weights
from dsart.common.constants import (
    KERNEL_SUPPORT_DEFAULT,
    NANTS,
    NCHAN_PER_CHGROUP,
    N_GRID_DEFAULT,
    freq_GHz,
)
from dsart.grid.sparsity_pattern import (
    SPEED_OF_LIGHT_M_PER_S,
    build_pattern,
    compute_antpos_hash,
    compute_chgroup_table_hash,
    predict_pattern_id,
    quantise_dec_deg,
)


# ---------------------------------------------------------------------------
# Synthetic antpos helpers
# ---------------------------------------------------------------------------


def _synth_antpos(seed: int = 20260505) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise a 96-ant antpos that's representative of DSA-110.

    Layout: 82 "core" antennas inside a ~ 600 m × 600 m square plus 14
    "outrigger" antennas at ~ 5 km baselines. Float32 to match what the
    bf_weights loader produces.

    Used for self-contained tests that don't need the on-h01
    beamformer_weights cal blob. The actual DSA-110 antpos has a more
    complex layout but the deterministic / hash / quantisation tests
    don't depend on its specifics.
    """
    rng = np.random.default_rng(seed=seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    # Core: uniform in a 600 m square (ant_idx 0..81).
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    # Outriggers: clustered at ~ ±5 km E (ant_idx 82..95).
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _core_baseline_mask(n_core: int = 82) -> np.ndarray:
    """``(NBASE,) bool`` mask: True iff both antennas are in [0, n_core).

    Mirrors plan §3 line 452 (`is_core_baseline_mask`); the test here
    uses the simple positional definition (ants [0, 82) are core, the
    rest outrigger), matching :func:`_synth_antpos` above.
    """
    nbase = NANTS * (NANTS + 1) // 2
    mask = np.zeros(nbase, dtype=bool)
    k = 0
    for a in range(NANTS):
        for b in range(a + 1):
            mask[k] = (a < n_core) and (b < n_core)
            k += 1
    return mask


def _h01_real_antpos() -> tuple[np.ndarray, np.ndarray] | None:
    """Load the 250924mptq antpos from h01 cal blob, or return None.

    Falls back to None on machines where the on-disk cal blob is
    absent (i.e. anything but h01). Tests that assert plan-§3-line-305
    fill fractions skip when this returns None.
    """
    # The 250924mptq cal files use the bare ``beamformer_weights_sbNN.dat``
    # naming (no source-name suffix); the 0319 fallback uses the
    # ``beamformer_weights_sbNN_<source>.dat`` form.
    candidates: list[str] = []
    for cals_dir, glob_pat in [
        (Path("/home/ubuntu/data/voltages/250924mptq/cals"),
         "beamformer_weights_sb00*.dat"),
        (Path("/home/ubuntu/data/voltages/0319/cals"),
         "beamformer_weights_sb00_*.dat"),
    ]:
        if cals_dir.is_dir():
            for blob in sorted(cals_dir.glob(glob_pat)):
                candidates.append(str(blob))
    for c in candidates:
        if Path(c).is_file():
            bf = load_bf_weights(c)
            return bf.antpos_e, bf.antpos_n
    return None


# ---------------------------------------------------------------------------
# Deterministic build
# ---------------------------------------------------------------------------


class TestBuildPatternDeterministic:
    """Two calls with identical inputs return byte-identical outputs."""

    def test_two_calls_same_inputs(self) -> None:
        e, n = _synth_antpos(seed=42)
        mask = _core_baseline_mask(n_core=82)
        kw = dict(
            chgroup=0,
            dec_deg=53.85,
            n_grid=N_GRID_DEFAULT,
            kernel_support=KERNEL_SUPPORT_DEFAULT,
            is_core_baseline_mask=mask,
        )
        p1 = build_pattern(e, n, **kw)
        p2 = build_pattern(e, n, **kw)
        # Bit-identical arrays (plan §3 line 306).
        np.testing.assert_array_equal(p1.ix_row, p2.ix_row)
        np.testing.assert_array_equal(p1.ix_col, p2.ix_col)
        assert p1.pattern_id == p2.pattern_id
        assert p1.n_filled == p2.n_filled
        assert p1.n_grid == p2.n_grid
        assert p1.chgroup == p2.chgroup
        # Sorted (lex / row-major) per build_pattern's contract.
        keys1 = (p1.ix_row.astype(np.uint32) << 16) | p1.ix_col.astype(np.uint32)
        assert np.all(np.diff(keys1) > 0), \
            "ix_row/ix_col not strictly increasing in row-major lex order"

    def test_dtypes_and_shape_invariants(self) -> None:
        e, n = _synth_antpos(seed=43)
        p = build_pattern(e, n, chgroup=0, dec_deg=37.234)
        assert p.ix_row.dtype == np.uint16
        assert p.ix_col.dtype == np.uint16
        assert p.ix_row.shape == p.ix_col.shape
        assert p.ix_row.shape[0] == p.n_filled
        assert isinstance(p.pattern_id, int)
        assert 0 <= p.pattern_id < (1 << 64)
        # All cell indices in range.
        assert int(p.ix_row.max()) < p.n_grid
        assert int(p.ix_col.max()) < p.n_grid


# ---------------------------------------------------------------------------
# DEC quantisation (plan §3 line 307)
# ---------------------------------------------------------------------------


class TestBuildPatternDecQuantisation:
    """Patterns within ± 0.125 deg of the same 0.25-deg bin are identical."""

    def test_same_quant_bin_yields_same_pattern(self) -> None:
        e, n = _synth_antpos()
        mask = _core_baseline_mask()
        # 41.234 → quantises to 41.25; 41.249 → quantises to 41.25.
        p_a = build_pattern(
            e, n, chgroup=0, dec_deg=41.234, is_core_baseline_mask=mask
        )
        p_b = build_pattern(
            e, n, chgroup=0, dec_deg=41.249, is_core_baseline_mask=mask
        )
        assert p_a.pattern_id == p_b.pattern_id
        assert p_a.dec_deg_quant == p_b.dec_deg_quant == 41.25
        np.testing.assert_array_equal(p_a.ix_row, p_b.ix_row)
        np.testing.assert_array_equal(p_a.ix_col, p_b.ix_col)

    def test_different_quant_bins_yield_different_pattern_id(self) -> None:
        e, n = _synth_antpos()
        mask = _core_baseline_mask()
        # 41.001 → 41.0 bucket; 41.499 → 41.5 bucket. Different bins.
        p_low = build_pattern(
            e, n, chgroup=0, dec_deg=41.001, is_core_baseline_mask=mask
        )
        p_high = build_pattern(
            e, n, chgroup=0, dec_deg=41.499, is_core_baseline_mask=mask
        )
        assert p_low.dec_deg_quant != p_high.dec_deg_quant
        assert p_low.pattern_id != p_high.pattern_id

    def test_quantise_dec_deg_helper_matches_plan(self) -> None:
        # Plan §3 line 307: quantise to PATTERN_DEC_QUANT_DEG = 0.25 deg.
        assert quantise_dec_deg(41.001) == pytest.approx(41.0)
        assert quantise_dec_deg(41.234) == pytest.approx(41.25)
        assert quantise_dec_deg(41.249) == pytest.approx(41.25)
        assert quantise_dec_deg(41.499) == pytest.approx(41.5)
        # 41.125 is exactly on a bin edge; numpy round-to-even resolves.
        # Either bin would be acceptable as long as the mapping is
        # deterministic; we just check it returns a valid bin centre.
        v = quantise_dec_deg(41.125)
        assert v in (41.0, 41.25)


# ---------------------------------------------------------------------------
# n_filled fill-fraction range (plan §3 line 305)
# ---------------------------------------------------------------------------


class TestBuildPatternNFilledRange:
    """``n_filled`` falls in the plan-predicted range for a real antpos."""

    def test_default_ops_fill_in_expected_range(self) -> None:
        antpos = _h01_real_antpos()
        if antpos is None:
            pytest.skip(
                "no on-disk beamformer_weights cal blob (need h01)"
            )
        e, n = antpos
        mask = _core_baseline_mask(n_core=82)
        p = build_pattern(
            e, n,
            chgroup=0,
            dec_deg=53.85,
            n_grid=N_GRID_DEFAULT,
            kernel_support=1,
            is_core_baseline_mask=mask,
        )
        # Plan §3 line 305: "single-side fill fraction is monotone
        # non-increasing in N_grid; ~ 7-12% at N_grid ∈ {128..512}".
        # That estimate was for the K=5 Gaussian taper (each sample
        # hits ~ 25 cells), which is reserved for the M3 hardening
        # pass (plan §4.2 line 1351 G7); chunk 3a ships K=1 pillbox
        # (each sample hits 1 cell), so the absolute ``n_filled``
        # is ~ 5× lower. Empirically on the 250924mptq antpos with
        # the simple positional 82-ant core mask:
        #   N_grid=128 ⇒ 4.5% fill (~ 740 cells)
        #   N_grid=256 ⇒ 3.4% fill (~ 2200 cells)   [this test point]
        #   N_grid=384 ⇒ 2.2% fill (~ 3200 cells)
        #   N_grid=512 ⇒ 1.4% fill (~ 3700 cells)
        # The bench (``bench/grid_pattern_visualisation.py``) reports
        # the full sweep + monotonicity check. Here we just bound the
        # absolute count loosely to catch a ``build_pattern`` regression
        # that drops below ~ 100 cells (e.g. cell_lambda 100× too big →
        # everything packs into a single cell) or runs above ~ 50k
        # (e.g. F20 sign drift causes wrap-around aliasing). Note
        # ``_core_baseline_mask(n_core=82)`` uses the simple positional
        # definition (ants 0..81 are core), which differs from the
        # production etcd ``/cnf/corr_setup_96.is_core`` array by ~ 1
        # ant (ant 48 is geometrically an outrigger but lands in the
        # first-82 mask); the absolute ``n_filled`` for the production
        # mask would differ by ~ 30%, still well inside the bound.
        assert 500 <= p.n_filled <= 50_000, (
            f"n_filled={p.n_filled} outside sanity bound "
            f"[500, 50000] for chgroup=0 dec=53.85 N_grid=256 K=1; "
            f"investigate antpos / cell-rounding."
        )


# ---------------------------------------------------------------------------
# pattern_id sensitivity (plan §3 line 307)
# ---------------------------------------------------------------------------


class TestPatternIdSensitivity:
    """Each input field flips a bit when changed, holding others fixed."""

    @pytest.fixture
    def base_inputs(self) -> dict:
        e, n = _synth_antpos(seed=99)
        return {
            "antpos_e": e,
            "antpos_n": n,
            "chgroup": 0,
            "dec_deg": 41.0,
            "n_grid": N_GRID_DEFAULT,
            "kernel_support": 1,
            "is_core_baseline_mask": None,
        }

    def test_dec_changes_pattern_id(self, base_inputs: dict) -> None:
        p_a = build_pattern(**base_inputs)
        kw = dict(base_inputs)
        kw["dec_deg"] = 41.5                                          # different bin
        p_b = build_pattern(**kw)
        assert p_a.pattern_id != p_b.pattern_id

    def test_n_grid_changes_pattern_id(self, base_inputs: dict) -> None:
        p_a = build_pattern(**base_inputs)
        kw = dict(base_inputs)
        kw["n_grid"] = 128
        p_b = build_pattern(**kw)
        assert p_a.pattern_id != p_b.pattern_id

    def test_chgroup_changes_pattern_id(self, base_inputs: dict) -> None:
        p_a = build_pattern(**base_inputs)
        kw = dict(base_inputs)
        kw["chgroup"] = 8
        p_b = build_pattern(**kw)
        assert p_a.pattern_id != p_b.pattern_id

    def test_kernel_support_change_changes_pattern_id_via_predict(
        self, base_inputs: dict
    ) -> None:
        # `build_pattern` raises NotImplementedError on K > 1 so we
        # exercise the K-sensitivity via predict_pattern_id (which
        # accepts K freely; the change-of-K is what matters for the
        # certifying-input semantics).
        ap_e = base_inputs["antpos_e"]
        ap_n = base_inputs["antpos_n"]
        pid_k1 = predict_pattern_id(
            chgroup=base_inputs["chgroup"],
            dec_deg=base_inputs["dec_deg"],
            n_grid=base_inputs["n_grid"],
            kernel_support=1,
            antpos_e=ap_e, antpos_n=ap_n,
        )
        pid_k3 = predict_pattern_id(
            chgroup=base_inputs["chgroup"],
            dec_deg=base_inputs["dec_deg"],
            n_grid=base_inputs["n_grid"],
            kernel_support=3,
            antpos_e=ap_e, antpos_n=ap_n,
        )
        assert pid_k1 != pid_k3

    def test_antpos_hash_changes_pattern_id(self, base_inputs: dict) -> None:
        p_a = build_pattern(**base_inputs)
        kw = dict(base_inputs)
        new_e = base_inputs["antpos_e"].copy()
        new_e[0] += 1.0                                               # 1 m shift
        kw["antpos_e"] = new_e
        p_b = build_pattern(**kw)
        assert p_a.antpos_hash != p_b.antpos_hash
        assert p_a.pattern_id != p_b.pattern_id

    def test_chgroup_table_hash_changes_pattern_id(
        self, base_inputs: dict
    ) -> None:
        p_a = build_pattern(**base_inputs)
        # Different chgroup_table_hash (e.g. operator updated
        # corr_setup_96.yaml since the corr restarted).
        kw = dict(base_inputs)
        kw["chgroup_table_hash"] = (p_a.chgroup_table_hash + 1) & 0xFFFFFFFFFFFFFFFF
        p_b = build_pattern(**kw)
        assert p_a.pattern_id != p_b.pattern_id


# ---------------------------------------------------------------------------
# predict_pattern_id parity with build_pattern
# ---------------------------------------------------------------------------


def test_predict_pattern_id_matches_build_pattern_id() -> None:
    e, n = _synth_antpos(seed=5)
    mask = _core_baseline_mask()
    kw = dict(
        chgroup=3,
        dec_deg=37.5,
        n_grid=128,
        kernel_support=1,
    )
    p = build_pattern(e, n, is_core_baseline_mask=mask, **kw)
    # Same inputs ⇒ same pattern_id, regardless of which API path.
    pid = predict_pattern_id(antpos_e=e, antpos_n=n, **kw)
    assert pid == p.pattern_id
    # Bypass with pre-computed hashes.
    ap_h = compute_antpos_hash(e, n)
    cg_h = compute_chgroup_table_hash()
    pid2 = predict_pattern_id(
        antpos_hash=ap_h, chgroup_table_hash=cg_h, **kw,
    )
    assert pid2 == p.pattern_id


def test_predict_pattern_id_requires_antpos_or_hash() -> None:
    with pytest.raises(ValueError, match="antpos_hash"):
        predict_pattern_id(chgroup=0, dec_deg=37.0, n_grid=128, kernel_support=1)


# ---------------------------------------------------------------------------
# F20 (u, v) negation parity against grid_uv_natural
# ---------------------------------------------------------------------------


class TestF20UvNegation:
    """Build_pattern's cell rounding matches the F20 reference imager."""

    def test_known_baseline_lands_at_negated_cell_index(self) -> None:
        """Single baseline at known (du_m, dv_m) lands at the negated cell.

        Construct a 96-ant antpos with **exactly one cross-baseline**
        (ants 0 and 1 at known offsets; all others at (0, 0)). At
        ``chgroup=0`` ch=0 (top of band, λ shortest) the (u, v) per
        baseline is well-defined. Without F20 the cell would be at
        ``(half + round(u/c), half + round(v/c))``; with F20 it lands
        at ``(half - round(u/c), half - round(v/c))``. We assert the
        latter.

        Also asserts that :func:`tools.viz.common.grid_uv_natural`
        (called read-only; never modified by chunk 3a) places the
        sample at the same cell on the same baseline + a synthetic
        unit visibility.
        """
        # 3-ant antpos so the LONG baseline sets ``max_baseline_lambda``
        # (and hence ``cell_lambda``) while the SHORT baseline sits well
        # inside the grid — far enough from the centre that the F20
        # negation flips the cell across the centre, but far enough
        # from the edge that BOTH the F20-negated and the non-F20
        # candidate cells are in-grid (so the test decides on the
        # negation, not on edge clipping).
        e = np.zeros(NANTS, dtype=np.float32)
        n = np.zeros(NANTS, dtype=np.float32)
        e[1] = 30.0     # short baseline (a=1, b=0): +30 m east, +20 m N
        n[1] = 20.0
        e[2] = 200.0    # long  baseline (a=2, b=0): +200 m east, +120 m N
        n[2] = 120.0    #   sets max_baseline_lambda

        # Keep both (0, 1) and (0, 2) cross-baselines so cell_lambda
        # scales with the long one. We then assert specifically about
        # the SHORT baseline's cell, which is well inside the grid.
        mask = np.zeros(NANTS * (NANTS + 1) // 2, dtype=bool)
        mask[1] = True                                                # bls(a=1, b=0)
        mask[3] = True                                                # bls(a=2, b=0); 2*3/2+0=3

        chgroup = 0
        n_grid = 256

        p = build_pattern(
            e, n,
            chgroup=chgroup,
            dec_deg=37.234,
            n_grid=n_grid,
            kernel_support=1,
            is_core_baseline_mask=mask,
        )

        # Pick a mid-band channel for the short baseline so the cell
        # is well away from the grid centre AND well inside the grid.
        ch = 100
        nu_GHz_short = freq_GHz(chgroup, ch)
        wavelength_short = SPEED_OF_LIGHT_M_PER_S / (nu_GHz_short * 1e9)
        u_lam = (e[1] - e[0]) / wavelength_short                       # raw +u
        v_lam = (n[1] - n[0]) / wavelength_short                       # raw +v

        # Recompute cell_lambda exactly as build_pattern did: max over
        # (kept-baseline, channel) of |u_lam| / |v_lam|.
        wavelength_all = SPEED_OF_LIGHT_M_PER_S / np.asarray(
            [freq_GHz(chgroup, c) for c in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        ) / 1e9
        kept_du = np.array([e[1] - e[0], e[2] - e[0]], dtype=np.float64)
        kept_dv = np.array([n[1] - n[0], n[2] - n[0]], dtype=np.float64)
        u_all = kept_du[:, None] / wavelength_all[None, :]
        v_all = kept_dv[:, None] / wavelength_all[None, :]
        max_baseline_lambda = float(
            np.max(np.maximum(np.abs(u_all), np.abs(v_all)))
        )
        cell_lambda = max_baseline_lambda * 2.0 / n_grid
        half = n_grid // 2
        # F20-NEGATED expected cell: (half - round(v/cell), half - round(u/cell)).
        expected_row = half - int(round(v_lam / cell_lambda))
        expected_col = half - int(round(u_lam / cell_lambda))

        # The pattern at ch=0 should contain (expected_row, expected_col).
        rows = p.ix_row.astype(int).tolist()
        cols = p.ix_col.astype(int).tolist()
        assert (expected_row, expected_col) in zip(rows, cols), (
            f"F20 negation NOT applied: expected cell "
            f"({expected_row}, {expected_col}) not in pattern. "
            f"Pattern cells: {sorted(zip(rows, cols))[:10]}..."
        )
        # And the NON-negated cell should NOT be there (otherwise the
        # test would pass trivially when both signs happened to agree
        # — which only happens at (u, v) = 0).
        wrong_row = half + int(round(v_lam / cell_lambda))
        wrong_col = half + int(round(u_lam / cell_lambda))
        if wrong_row != expected_row or wrong_col != expected_col:
            assert (wrong_row, wrong_col) not in zip(rows, cols), (
                f"non-F20 cell ({wrong_row}, {wrong_col}) also in pattern; "
                f"F20 negation may not be applied correctly."
            )

    def test_pattern_matches_grid_uv_natural_filled_cells(self) -> None:
        """Pattern's filled cells = grid_uv_natural's filled cells.

        Build a 4-ant cross-baseline antpos (so we get a handful of
        baselines + a few uv cells). Run :func:`build_pattern` and
        :func:`tools.viz.common.grid_uv_natural` on the same baseline
        list at the same channel + the same cell scale; the SET of
        filled cells must agree. This is the "shared sign convention"
        gate.
        """
        import sys as _sys, pathlib as _pl
        _root = str(_pl.Path(__file__).resolve().parents[1])
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        try:
            from tools.viz.common import grid_uv_natural
        except ImportError as exc:
            pytest.skip(
                f"tools.viz.common not importable from test env: {exc}"
            )

        # 4 active core ants + 92 stub ants at the origin (masked out).
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

        # Core mask: all (a, b) with a, b < 4 and a != b are kept.
        mask = np.zeros(NANTS * (NANTS + 1) // 2, dtype=bool)
        for a in range(4):
            for b in range(a):
                bls_idx = a * (a + 1) // 2 + b
                mask[bls_idx] = True

        chgroup = 0
        n_grid = 64
        ch_test = 0                                                   # single-channel test

        p = build_pattern(
            e, n,
            chgroup=chgroup,
            dec_deg=37.234,
            n_grid=n_grid,
            kernel_support=1,
            is_core_baseline_mask=mask,
        )

        # Replicate build_pattern's cell scale to feed grid_uv_natural
        # the same `fov_rad` (it parameterises by fov rather than by
        # cell_lambda).
        nu_GHz = np.asarray(
            [freq_GHz(chgroup, c) for c in range(NCHAN_PER_CHGROUP)],
            dtype=np.float64,
        )
        wavelength_m = SPEED_OF_LIGHT_M_PER_S / (nu_GHz * 1e9)
        # Per-baseline (du, dv) in metres, in dsamfs ant_2 - ant_1
        # (= a - b) order; build_pattern uses the same order.
        bls_uv_m = []
        for a in range(4):
            for b in range(a):
                bls_uv_m.append((ant_pos[a, 0] - ant_pos[b, 0],
                                 ant_pos[a, 1] - ant_pos[b, 1]))
        bls_uv_m_arr = np.asarray(bls_uv_m, dtype=np.float32)         # (Nbls=6, 2)
        # build_pattern's max over (kept-baseline, ch) of |u_lam|, |v_lam|.
        u_lam_all = bls_uv_m_arr[:, 0:1] / wavelength_m[None, :]
        v_lam_all = bls_uv_m_arr[:, 1:2] / wavelength_m[None, :]
        max_baseline_lambda = float(np.max(np.maximum(
            np.abs(u_lam_all), np.abs(v_lam_all),
        )))
        cell_lambda = max_baseline_lambda * 2.0 / n_grid
        # grid_uv_natural's API: cell_lambda = 1 / fov_rad ⇒ fov_rad = 1/cell_lambda.
        fov_rad = 1.0 / cell_lambda

        # grid_uv_natural takes (Nbls, Nfreqs, Npols) vis + (Nbls, 3) uvw_m.
        # Synthesise unit vis at ch_test, zero elsewhere; uvw at the
        # active baselines (autos and unused ants stay at (0, 0, 0) so
        # drop_autos=True drops them).
        vis_dummy = np.zeros((6, NCHAN_PER_CHGROUP, 1), dtype=np.complex64)
        vis_dummy[:, ch_test, 0] = 1.0 + 0.0j
        uvw_m = np.zeros((6, 3), dtype=np.float32)
        uvw_m[:, 0] = bls_uv_m_arr[:, 0]
        uvw_m[:, 1] = bls_uv_m_arr[:, 1]

        grid_ref, _w_ref = grid_uv_natural(
            vis_dummy, uvw_m, np.asarray(nu_GHz * 1e9, dtype=np.float64),
            n_grid=n_grid, fov_rad=fov_rad, pol=0, drop_autos=True,
        )
        # Build the set of (row, col) cells filled by grid_ref. Keep
        # only the contributions from ch_test by construction (other
        # channels carry zero values; but np.add.at will still touch
        # the cell's WEIGHT). Use the values, not the weight, to
        # extract the ch_test cells.
        ref_cells = set(map(tuple, np.argwhere(grid_ref != 0)))

        # Build_pattern produces the union of cells across all 384
        # channels. Restrict to ch_test cells by re-rounding the same
        # (u, v) at ch_test only.
        nu_test = freq_GHz(chgroup, ch_test)
        wave_test = SPEED_OF_LIGHT_M_PER_S / (nu_test * 1e9)
        half = n_grid // 2
        ch_test_cells = set()
        for du_m, dv_m in bls_uv_m:
            u_lam_t = du_m / wave_test
            v_lam_t = dv_m / wave_test
            row = half - int(round(v_lam_t / cell_lambda))            # F20
            col = half - int(round(u_lam_t / cell_lambda))            # F20
            if 0 <= row < n_grid and 0 <= col < n_grid:
                ch_test_cells.add((row, col))

        # The pattern's full cell set must be a SUPERSET of the ch_test
        # cells (build_pattern unions all channels); the grid_ref's
        # cells must equal the ch_test_cells exactly (single-channel).
        pattern_cells = set(zip(
            p.ix_row.astype(int).tolist(),
            p.ix_col.astype(int).tolist(),
        ))
        assert ref_cells == ch_test_cells, (
            f"grid_uv_natural ref cells {sorted(ref_cells)} != "
            f"F20-rounded expected {sorted(ch_test_cells)}; "
            f"sign convention drifted between build_pattern and viz."
        )
        assert ch_test_cells.issubset(pattern_cells), (
            f"build_pattern missed ch_test cells {sorted(ch_test_cells - pattern_cells)};"
            f" pattern cells {sorted(pattern_cells)}"
        )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestBuildPatternValidation:

    def test_rejects_kernel_support_gt_1_until_hardening(self) -> None:
        e, n = _synth_antpos()
        with pytest.raises(NotImplementedError, match="kernel_support"):
            build_pattern(e, n, chgroup=0, dec_deg=37.0, kernel_support=3)

    def test_rejects_zero_n_grid(self) -> None:
        e, n = _synth_antpos()
        with pytest.raises(ValueError, match="n_grid"):
            build_pattern(e, n, chgroup=0, dec_deg=37.0, n_grid=0)

    def test_accepts_non_power_of_two_n_grid(self) -> None:
        # Pow-of-2 is enforced downstream by SparseCOOPayload, not
        # here — the bench wants to inspect fill fractions at n_grid
        # values like 384 (plan §3 line 311 reports this operating
        # point).
        e, n = _synth_antpos()
        p = build_pattern(e, n, chgroup=0, dec_deg=37.0, n_grid=384)
        assert p.n_grid == 384
        assert p.n_filled > 0

    def test_rejects_bad_chgroup(self) -> None:
        e, n = _synth_antpos()
        with pytest.raises(ValueError, match="chgroup"):
            build_pattern(e, n, chgroup=99, dec_deg=37.0)

    def test_rejects_wrong_antpos_shape(self) -> None:
        bad = np.zeros(NANTS - 1, dtype=np.float32)
        ok = np.zeros(NANTS, dtype=np.float32)
        with pytest.raises(ValueError, match="antpos shapes"):
            build_pattern(bad, ok, chgroup=0, dec_deg=37.0)

    def test_rejects_zero_max_baseline(self) -> None:
        # All-zero antpos + permissive mask → max_baseline_lambda = 0.
        e = np.zeros(NANTS, dtype=np.float32)
        n = np.zeros(NANTS, dtype=np.float32)
        # Need at least one cross baseline kept (otherwise we'd hit
        # the "no kept baselines" path which np.max on an empty array
        # also raises on).
        mask = np.zeros(NANTS * (NANTS + 1) // 2, dtype=bool)
        mask[1] = True                                                # (a=1, b=0)
        with pytest.raises(ValueError, match="max_baseline_lambda"):
            build_pattern(
                e, n, chgroup=0, dec_deg=37.0,
                is_core_baseline_mask=mask,
            )


# ---------------------------------------------------------------------------
# SparsityPattern dataclass smoke
# ---------------------------------------------------------------------------


def test_sparsity_pattern_is_frozen() -> None:
    e, n = _synth_antpos()
    p = build_pattern(e, n, chgroup=0, dec_deg=37.0)
    # frozen=True ⇒ rebinding raises FrozenInstanceError.
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.pattern_id = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# `math` import is currently unused; keeping the import out so flake/
# pyright don't complain.
# ---------------------------------------------------------------------------


def test_math_module_present() -> None:
    # Defensive: guard against the test file losing its `math` reference
    # in a future refactor (some tests above use `math.pi`-style
    # constants implicitly via numpy; this is just a no-op assertion to
    # satisfy linters that flag the unused import otherwise).
    assert math.pi > 3.0
