"""Acceptance tests for ``bench/_corr_fast_replay.py`` (M3 chunks 5+6).

Six smoke tests using synthetic bytes / synthetic antpos so the suite
runs CPU-only on h23 in <1 s and on h01 alongside other M3 work.
The full real-data fixture replay is exercised by the chunk-5 +
chunk-6 benches themselves (gated separately in M3.sh on h01 only).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dsart.common.constants import (
    FADA_BYTES_PER_BLOCK,
    NANTS,
    NCHAN_PER_CHGROUP,
    NPOL,
)
from dsart.services.corr_fast_integration import (
    FastIntegrationConfig,
    IntegrationOutput,
    _build_core_baseline_mask,
)

from bench._corr_fast_replay import (
    accumulate_chgroup_grids,
    compute_chgroup_cell_lambda,
    iterate_voltage_blocks,
    lm_to_pixel,
    pixel_to_lm_radians,
    replay_chgroup,
    sparse_to_dense_grid,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers (mirror tests/test_corr_fast_integration.py)
# ---------------------------------------------------------------------------


SMALL_BLOCK_BYTES: int = 1024  # synthetic block size for iterate tests


def _synth_antpos(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = np.zeros(NANTS, dtype=np.float32)
    n = np.zeros(NANTS, dtype=np.float32)
    e[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    n[:82] = rng.uniform(-300.0, 300.0, size=82).astype(np.float32)
    e[82:] = rng.uniform(-5000.0, 5000.0, size=NANTS - 82).astype(np.float32)
    n[82:] = rng.uniform(-2000.0, 2000.0, size=NANTS - 82).astype(np.float32)
    return e, n


def _make_synth_cfg(**overrides) -> FastIntegrationConfig:
    """No-cal, RFI-off, static-sky-off — mirrors test_corr_fast_integration."""
    base = dict(
        chgroup=0,
        obs_dec_rad=math.radians(53.85),
        n_grid=64,
        kernel_support=1,
        t_int_fast_native=4096,                                      # 1 fast-vis tile per block
        cal_path=None,
        rfi_enabled=False,
        static_sky_disabled=True,
        static_sky_warmup_cubes=0,
    )
    base.update(overrides)
    return FastIntegrationConfig(**base)


# ---------------------------------------------------------------------------
# 1. iterate_voltage_blocks — chunk size + count
# ---------------------------------------------------------------------------


def test_iterate_voltage_blocks_yields_correct_chunk_size(tmp_path) -> None:
    """3-block synthetic file; verify yield count + chunk-size.

    Each yielded chunk MUST be exactly ``block_bytes`` long; the iterator
    drops final partial blocks (with a warning, tested separately) so
    callers always see uniform-sized blocks.
    """
    n_blocks = 3
    payload = b"\xaa" * SMALL_BLOCK_BYTES * n_blocks
    p = tmp_path / "synth.out"
    p.write_bytes(payload)

    blocks = list(iterate_voltage_blocks(p, block_bytes=SMALL_BLOCK_BYTES))
    assert len(blocks) == n_blocks, (
        f"yielded {len(blocks)} blocks, expected {n_blocks}"
    )
    for i, b in enumerate(blocks):
        assert len(b) == SMALL_BLOCK_BYTES, (
            f"block {i} has length {len(b)}, expected {SMALL_BLOCK_BYTES}"
        )


# ---------------------------------------------------------------------------
# 2. iterate_voltage_blocks — partial trailing block dropped + logged
# ---------------------------------------------------------------------------


def test_iterate_voltage_blocks_handles_partial_final_block(tmp_path, caplog) -> None:
    """File length not a multiple of block_bytes: trailing partial block
    is DROPPED and a warning is logged. Mirrors the M2 dumper convention
    that file lengths are always whole-block multiples in production but
    may be truncated by an incomplete dump.
    """
    n_full = 2
    partial = SMALL_BLOCK_BYTES // 3                                 # ~341 bytes
    payload = b"\xff" * (SMALL_BLOCK_BYTES * n_full + partial)
    p = tmp_path / "partial.out"
    p.write_bytes(payload)

    with caplog.at_level("WARNING", logger="bench._corr_fast_replay"):
        blocks = list(iterate_voltage_blocks(p, block_bytes=SMALL_BLOCK_BYTES))
    assert len(blocks) == n_full, (
        f"yielded {len(blocks)} blocks, expected {n_full}"
    )
    assert any(
        "partial bytes" in rec.message
        for rec in caplog.records
    ), "expected partial-block warning was not logged"


# ---------------------------------------------------------------------------
# 3. iterate_voltage_blocks — max_blocks caps iteration
# ---------------------------------------------------------------------------


def test_iterate_voltage_blocks_max_blocks_caps_iteration(tmp_path) -> None:
    """5-block synthetic file with ``max_blocks=2`` → exactly 2 blocks yielded."""
    n_in_file = 5
    payload = b"\x00" * SMALL_BLOCK_BYTES * n_in_file
    p = tmp_path / "five.out"
    p.write_bytes(payload)

    blocks = list(iterate_voltage_blocks(
        p, block_bytes=SMALL_BLOCK_BYTES, max_blocks=2,
    ))
    assert len(blocks) == 2, (
        f"max_blocks=2 yielded {len(blocks)} blocks; expected 2"
    )


# ---------------------------------------------------------------------------
# 4. replay_chgroup — one IntegrationOutput per block
# ---------------------------------------------------------------------------


def test_replay_chgroup_returns_one_output_per_block(tmp_path) -> None:
    """Synth-bytes + synth antpos (no cal) + max_blocks=2 → list of 2 outputs.

    Validates the bench-level loop: build_context once, process_block
    per block, returns ``len(outputs) == max_blocks``.
    """
    n_blocks = 2
    payload = bytes(np.random.default_rng(20260506).integers(
        0, 256, size=FADA_BYTES_PER_BLOCK * n_blocks, dtype=np.uint8,
    ).tobytes())
    p = tmp_path / "real_size.out"
    p.write_bytes(payload)

    cfg = _make_synth_cfg()
    e, n = _synth_antpos()
    ctx, outputs = replay_chgroup(
        p, cal_path=None, cfg=cfg,
        max_blocks=n_blocks, device=torch.device("cpu"),
        antpos_e=e, antpos_n=n,
    )
    assert len(outputs) == n_blocks
    for i, out in enumerate(outputs, start=1):
        assert isinstance(out, IntegrationOutput)
        assert out.block_n == i, f"out[{i-1}].block_n={out.block_n}"
        assert out.gridded_minus_sky is not None
        n_fv, n_filled = out.gridded_minus_sky.shape
        assert n_fv == ctx.kernel.n_fast_vis_per_full_block
        assert n_filled == ctx.gridder.pattern.n_filled
        assert out.gridded_minus_sky.dtype == torch.complex64


# ---------------------------------------------------------------------------
# 5. accumulate_chgroup_grids — concatenates blocks along time-axis
# ---------------------------------------------------------------------------


def test_accumulate_chgroup_grids_sums_correctly() -> None:
    """Synthetic IntegrationOutput list → concat along fv axis.

    Builds 3 fake outputs with ``(2, 5)`` complex grids; verify the
    concatenated tensor has shape ``(6, 5)`` and elementwise matches
    a reference torch.cat.
    """
    n_blocks = 3
    n_fv_per_block = 2
    n_filled = 5
    grids = [
        torch.complex(
            torch.full((n_fv_per_block, n_filled), float(i + 1)),
            torch.full((n_fv_per_block, n_filled), float(-(i + 1))),
        )
        for i in range(n_blocks)
    ]
    outputs = [
        IntegrationOutput(
            gridded_minus_sky=g, rfi=None, n_tx=0, block_n=i + 1,
        )
        for i, g in enumerate(grids)
    ]
    accum = accumulate_chgroup_grids(outputs, n_filled=n_filled)
    expected = torch.cat(grids, dim=0)
    assert accum.shape == (n_blocks * n_fv_per_block, n_filled)
    assert accum.dtype == torch.complex64 or accum.dtype == torch.complex128
    assert torch.equal(accum, expected)


# ---------------------------------------------------------------------------
# 6. replay_chgroup — zero voltages → zero gridded cubes
# ---------------------------------------------------------------------------


def test_replay_chgroup_handles_zero_voltages(tmp_path) -> None:
    """All-zero raw bytes → all-zero gridded outputs.

    Mirrors the chunk-2b ``test_compute_block_zero_voltages_yields_zero_vis``
    pattern. Zero-byte voltages decode to zero (real, imag) per the
    int4 ASR fluff path; the fast-corr GEMM produces zero vis; the
    gridder sums zeros into zero cells. End-to-end, the gridded cube
    must be exactly zero.
    """
    n_blocks = 1
    payload = b"\x00" * FADA_BYTES_PER_BLOCK * n_blocks
    p = tmp_path / "zero.out"
    p.write_bytes(payload)

    cfg = _make_synth_cfg()
    e, n = _synth_antpos()
    _ctx, outputs = replay_chgroup(
        p, cal_path=None, cfg=cfg,
        max_blocks=n_blocks, device=torch.device("cpu"),
        antpos_e=e, antpos_n=n,
    )
    assert len(outputs) == 1
    g = outputs[0].gridded_minus_sky
    assert g is not None
    assert torch.equal(
        g, torch.zeros_like(g),
    ), f"gridded cube has nonzero values on all-zero input; max|g|={g.abs().max().item()}"


# ---------------------------------------------------------------------------
# 7. (bonus, but tagged in same suite) sparse_to_dense + dirty_image round-trip
# ---------------------------------------------------------------------------


def test_sparse_to_dense_and_dirty_image_roundtrip() -> None:
    """sparse → dense (scatter) → iFFT → real image. Pin shape + dtype.

    Not a hard correctness pin (the sign/F20 conventions are pinned by
    chunk-3a tests upstream), just a smoke that the helper composes.
    """
    n_fv, n_grid = 4, 16
    ix_row = np.array([0, 8, 8, 15], dtype=np.uint16)
    ix_col = np.array([0, 8, 9, 15], dtype=np.uint16)
    n_filled = ix_row.size
    sparse = torch.complex(
        torch.arange(n_fv * n_filled, dtype=torch.float32).reshape(n_fv, n_filled),
        torch.zeros((n_fv, n_filled), dtype=torch.float32),
    )

    dense = sparse_to_dense_grid(sparse, ix_row, ix_col, n_grid)
    assert dense.shape == (n_fv, n_grid, n_grid)
    assert dense.dtype == torch.complex64

    # Sanity: the (8, 9) cell of fast-vis tile 0 carries sparse[0, 2] = 2+0j.
    assert dense[0, 8, 9].item() == complex(2.0, 0.0)
    # Cells that were not in the pattern are exactly zero.
    assert dense[0, 1, 1].item() == complex(0.0, 0.0)

    from bench._corr_fast_replay import dirty_image_from_dense_grid
    img = dirty_image_from_dense_grid(dense)
    assert img.shape == (n_fv, n_grid, n_grid)
    assert img.dtype == torch.float32


# ---------------------------------------------------------------------------
# 8. (bonus) compute_chgroup_cell_lambda + lm/pixel round-trip
# ---------------------------------------------------------------------------


def test_chgroup_cell_lambda_and_lm_pixel_roundtrip() -> None:
    """cell_lambda > 0 sanity + lm_to_pixel(pixel_to_lm(p)) == p.

    Validates the bench's (l, m) ↔ pixel utilities used by both chunk-5
    and chunk-6 for the predicted-source overlay and the burst-pixel
    timeseries lookup.
    """
    e, n = _synth_antpos(seed=42)
    n_grid = 256
    cell_lambda = compute_chgroup_cell_lambda(
        e, n, chgroup=0, n_grid=n_grid,
        is_core_baseline_mask=_build_core_baseline_mask(n_core=82),
    )
    assert cell_lambda > 0.0

    # Centre pixel maps to (0, 0); off-centre round-trip
    half = n_grid // 2
    pixel_test_rows = np.array([half, half + 5, half - 10])
    pixel_test_cols = np.array([half, half + 7, half - 3])
    l_rad, m_rad = pixel_to_lm_radians(
        pixel_test_rows, pixel_test_cols,
        n_grid=n_grid, cell_lambda=cell_lambda,
    )
    # Centre pixel ≡ (0, 0).
    assert l_rad[0] == 0.0 and m_rad[0] == 0.0

    rows_back, cols_back = lm_to_pixel(
        l_rad, m_rad, n_grid=n_grid, cell_lambda=cell_lambda,
    )
    np.testing.assert_array_equal(rows_back, pixel_test_rows)
    np.testing.assert_array_equal(cols_back, pixel_test_cols)
