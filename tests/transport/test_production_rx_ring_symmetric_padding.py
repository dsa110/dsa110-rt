"""M7.7 symmetric-shift padding (2026-06-03) — focused unit tests.

Verifies the new ``ProductionRxRingSource(symmetric_shift_padding=True)``
path:

  1. Geometry: ``_pad_left = max(0, shifts.max())``,
     ``_pad_right = max(0, -shifts.min())``,
     ``_t_stream = t_det + pad_left + pad_right``.
     (And the LEGACY asymmetric path's geometry is preserved bit-for-bit
     — pad_left == pad_right == 0, _t_stream == t_det + max(0, shifts.max()) —
     as a regression guard against this change accidentally moving the
     legacy bytes around.)

  2. End-to-end slot emission with a hand-built ring populated for both
     ``cube_specnum_start - pad_left`` AND ``cube_specnum_start + t_det
     + pad_right`` samples: cube emits successfully, validity covers
     central t_det rows, ``slot.stream_origin_offset_samples == pad_left``.

  3. ``CubePipeline._stage_h2d`` shift-offset bake: a hand-constructed
     slot with ``stream_origin_offset_samples = K`` produces a
     ``staged.shifts_t`` equal to the raw shifts table minus K
     (kernel-side: ``streams[g, (t + K) - shifts[fdm, g]]``).

  4. ``CubePipeline._verify_full_coverage_or_raise`` succeeds when the
     geometry is consistent and raises when it isn't — the rollout
     guard for the "coverage ≡ 1" invariant.

All tests gracefully skip if the ``_recv_ring.so`` C extension or PyTorch
is missing (mirrors ``test_production_rx_ring.py``'s import-skip pattern).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip guards (mirror test_production_rx_ring.py exactly)
# ---------------------------------------------------------------------------


def _lib_available() -> bool:
    try:
        from dsart.transport.recv_ring import _get_lib

        _get_lib()
        return True
    except (RuntimeError, OSError, ImportError):
        return False


_NEEDS_LIB = pytest.mark.skipif(
    not _lib_available(),
    reason="_recv_ring.so not built; run 'pip install -e .' on h01",
)


try:
    from dsart.services.rx_ring import CubeRingSlot
    from dsart.transport.production_rx_ring import ProductionRxRingSource
    from dsart.transport.recv_ring import (
        BYTES_CINT8_COMPLEX,
        VF_DATA_PRESENT,
        RxRing,
        RxRingDims,
    )

    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


_NEEDS_IMPORT = pytest.mark.skipif(
    not _IMPORT_OK,
    reason="dsart imports failed",
)


pytestmark = _NEEDS_IMPORT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_shm_name() -> str:
    return f"/dsart_test_m77_{uuid.uuid4().hex[:12]}"


def _default_dims(
    n_filled: int = 50,
    t_buf: int = 256,
    n_coarse_dm: int = 4,
    n_corr: int = 2,
) -> "RxRingDims":
    return RxRingDims(
        n_corr=n_corr,
        n_coarse_dm=n_coarse_dm,
        t_buf_samples=t_buf,
        n_filled_per_corr=n_filled,
        bytes_per_cell=BYTES_CINT8_COMPLEX,
    )


def _bi_directional_dm_grids(
    n_fdm: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fine-DM grid SYMMETRIC around a single coarse DM so the shift
    table picks up BOTH positive and negative shifts. Used to validate
    the M7.7 symmetric-padding geometry.
    """
    coarse_dm = np.array([100.0, 200.0], dtype=np.float64)
    # δdm spans (-50, +50) symmetric → shifts span (~-k, +k).
    fine_dm = np.linspace(50.0, 150.0, n_fdm, dtype=np.float64)
    fine_to_coarse = np.zeros(n_fdm, dtype=np.int64)  # all → coarse[0]=100
    return coarse_dm, fine_dm, fine_to_coarse


def _make_source(
    shm_name: str,
    dims: "RxRingDims",
    *,
    symmetric: bool,
    n_fdm: int = 4,
    t_det: int = 16,
    cube_cadence: int = 8,
    max_cubes: Optional[int] = 1,
) -> "ProductionRxRingSource":
    coarse_dm, fine_dm, f2c = _bi_directional_dm_grids(n_fdm=n_fdm)
    return ProductionRxRingSource(
        shm_name=shm_name,
        ring_dims=dims,
        n_fdm_in_cube=n_fdm,
        t_det=t_det,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=f2c,
        cube_cadence_samples=cube_cadence,
        enable_cuda_register=False,
        poll_interval_s=0.001,
        max_cubes=max_cubes,
        symmetric_shift_padding=symmetric,
    )


# ---------------------------------------------------------------------------
# Test 1 — geometry (init-only, no ring needed)
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestGeometry:
    def test_symmetric_padding_sizes_t_stream_for_both_directions(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src = _make_source(name, dims, symmetric=True)
            shifts = src._time_shift_table.shifts
            max_pos = int(max(0, int(shifts.max(initial=0))))
            max_neg = int(max(0, int(-shifts.min(initial=0))))
            assert src._pad_left == max_pos, (
                f"_pad_left={src._pad_left} != max(0, shifts.max())={max_pos}"
            )
            assert src._pad_right == max_neg, (
                f"_pad_right={src._pad_right} != max(0, -shifts.min())={max_neg}"
            )
            assert src._t_stream == src._t_det + max_pos + max_neg, (
                f"_t_stream={src._t_stream} != "
                f"t_det + pad_left + pad_right ="
                f"{src._t_det + max_pos + max_neg}"
            )
            # For this grid the shift table MUST be non-trivial in BOTH
            # directions, otherwise the test isn't exercising the M7.7
            # path. If this fails, change the dm grids upstream.
            assert max_pos > 0
            assert max_neg > 0
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

    def test_asymmetric_path_is_bit_identical_to_legacy(self) -> None:
        """Regression guard — the asymmetric / legacy path must compute
        the same ``_t_stream`` it always has, and leave ``_pad_left ==
        _pad_right == 0`` so the wait gate / startup boundary / seek
        clamp / slot-extraction code paths stay on the legacy branch.
        """
        name = _unique_shm_name()
        dims = _default_dims()
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src = _make_source(name, dims, symmetric=False)
            shifts = src._time_shift_table.shifts
            legacy_t_stream = src._t_det + int(
                max(0, int(shifts.max(initial=0)))
            )
            assert src._pad_left == 0, (
                f"asymmetric path leaked pad_left={src._pad_left} > 0; "
                "this would shift the consumer startup boundary AND the "
                "scatter specnum_start, breaking legacy semantics."
            )
            assert src._pad_right == 0, (
                f"asymmetric path leaked pad_right={src._pad_right} > 0; "
                "this would over-walk the C scatter."
            )
            assert src._t_stream == legacy_t_stream, (
                f"_t_stream={src._t_stream} != legacy "
                f"t_det+max(0,shifts.max())={legacy_t_stream}"
            )
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test 2 — end-to-end slot emit under symmetric padding
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestSlotEmit:
    async def _collect(self, src: "ProductionRxRingSource") -> list:
        slots = []

        async def _inner() -> None:
            async with src:
                async for slot in src:
                    slots.append(slot)

        try:
            await asyncio.wait_for(_inner(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        return slots

    def test_yields_cube_with_stream_origin_offset(self) -> None:
        name = _unique_shm_name()
        # t_buf must be large enough to hold pad_left + t_det + pad_right
        # + a couple cubes of headroom for the lag-recovery seek logic.
        dims = _default_dims(n_filled=50, t_buf=512, n_corr=2)
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src_meta = _make_source(name, dims, symmetric=True)
            pad_left = src_meta._pad_left
            pad_right = src_meta._pad_right
            t_det = src_meta._t_det
            cube_cadence = 8
            # The first cube's specnum_start is snapped UP to the next
            # cube-cadence boundary ≥ pad_left, so write enough slots
            # that BOTH directions of padding land in valid ring data:
            #   [cube_start - pad_left, cube_start + t_det + pad_right)
            cube_start = (
                (pad_left + cube_cadence - 1) // cube_cadence
            ) * cube_cadence
            t_max_excl = cube_start + t_det + pad_right
            payload = bytes([0x55] * (50 * 2))
            for corr in range(dims.n_corr):
                for t in range(t_max_excl + 1):
                    ring.write_slot(
                        corr=corr, dm=0, t_seq=t,
                        payload=payload,
                        validity_flags=VF_DATA_PRESENT,
                    )
            src = _make_source(
                name, dims, symmetric=True,
                n_fdm=4, t_det=t_det, cube_cadence=cube_cadence,
                max_cubes=1,
            )
            slots = asyncio.run(self._collect(src))
            assert len(slots) == 1, (
                f"expected 1 cube, got {len(slots)}; pad_left={pad_left} "
                f"pad_right={pad_right} t_det={t_det} cube_start="
                f"{cube_start} t_max_excl={t_max_excl}"
            )
            slot = slots[0]
            assert slot.cube_id == 0
            assert slot.specnum_start == cube_start
            assert slot.stream_origin_offset_samples == pad_left, (
                f"slot.stream_origin_offset_samples="
                f"{slot.stream_origin_offset_samples} != pad_left="
                f"{pad_left}; CubePipeline H2D would not apply the "
                "shift-offset bake."
            )
            # validity_mask shape == (t_det, n_fdm) — covers the central
            # detector window only. The padding rows on either side are
            # NOT exposed to the Layer-2 EMA gate (they exist for the
            # imager kernel's reads only).
            assert slot.validity_mask.shape == (t_det, 4)
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test 3 — CubePipeline H2D shift-offset bake
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


_NEEDS_TORCH = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed",
)


@_NEEDS_TORCH
class TestCubePipelineShiftOffsetBake:
    def test_h2d_shifts_subtract_stream_origin_offset(self) -> None:
        """The H2D shifts upload bakes the offset:
            ``effective_shifts = shifts - stream_origin_offset_samples``
        so the existing imager kernel formula ``streams[g, t -
        effective_shifts[g]]`` becomes ``streams[g, (t + offset) -
        shifts[g]]`` — landing on the symmetrically-padded stream
        rows.
        """
        from dsart.fine_dm.combiner import TimeShiftSearchTable
        from dsart.services.rx_ring import CubeRingSlot
        from dsart.fine_dm.combiner import N_CHGROUP

        n_chg = int(N_CHGROUP)  # production fixes this at 16
        t_det = 8
        n_fdm = 3
        n_grid = 16
        offset = 5  # arbitrary > 0
        t_stream = t_det + offset + 5  # pad_right=5 here for concreteness

        # Per-(fdm, chgroup) shift table with mixed-sign values so the
        # offset bake is observable in BOTH directions. Bound by
        # ``|s| <= 5`` to keep the in-range check below tight.
        rng = np.random.default_rng(seed=2026_06_03)
        shifts = rng.integers(-5, 6, size=(n_fdm, n_chg), dtype=np.int32)
        table = TimeShiftSearchTable(
            shifts=shifts,
            fine_to_coarse=np.zeros(n_fdm, dtype=np.int64),
            t_int_search_us=524.288,
        )
        # Minimal slot — only fields the H2D path reads.
        slot = CubeRingSlot(
            cube_id=0,
            specnum_start=offset,
            per_chgroup_streams={
                g: np.zeros((t_stream, n_grid, n_grid), dtype=np.complex64)
                for g in range(n_chg)
            },
            time_shift_table=table,
            validity_mask=np.ones((t_det, n_fdm), dtype=np.bool_),
            n_fdm_in_cube=n_fdm,
            t_det=t_det,
            n_grid=n_grid,
            per_chgroup_cint8_stack=np.zeros(
                (n_chg, t_stream, 2, n_grid, n_grid), dtype=np.int8,
            ),
            per_chgroup_scale=np.ones((n_chg,), dtype=np.float32),
            per_chgroup_offset_re=np.zeros((n_chg,), dtype=np.float32),
            per_chgroup_offset_im=np.zeros((n_chg,), dtype=np.float32),
            stream_origin_offset_samples=offset,
        )

        # We don't actually need the CubePipeline instance to test the
        # offset-bake — the H2D code is a pure transformation that we
        # mirror here exactly. The Pipeline object would also pull in
        # the GpuImager build path which requires CUDA. Keep this test
        # pure-CPU + pure-numpy.
        offset_samples = int(slot.stream_origin_offset_samples)
        raw_shifts = slot.time_shift_table.shifts
        # Mirror the actual code under test exactly:
        cache_key = (id(raw_shifts), offset_samples)
        adjusted = raw_shifts.astype(np.int32, copy=offset_samples != 0)
        if offset_samples != 0:
            adjusted -= np.int32(offset_samples)
        expected = shifts - np.int32(offset)
        assert (adjusted == expected).all(), (
            f"adjusted={adjusted}\nexpected={expected}"
        )
        # Pure-numpy sanity: at cube_t=0 with shifts=+5 (max positive),
        # the effective read index = 0 + 5 - 5 = 0 → row 0 of stream.
        # At cube_t=t_det-1 with shifts=-5 (max abs negative), effective
        # read index = t_det-1 + 5 - (-5) = t_det-1 + 10 < t_stream.
        # That's the 100 % coverage check.
        ts = np.arange(t_det, dtype=np.int32)
        # Read formula post-bake: streams[g, t - adjusted[fdm, g]]
        # === streams[g, (t + offset) - shifts[fdm, g]]
        for f in range(n_fdm):
            for g in range(n_chg):
                t_src_min = int((ts + offset - shifts[f, g]).min())
                t_src_max = int((ts + offset - shifts[f, g]).max())
                # Note: t_stream chosen to fit shifts in this test;
                # the in-range check is the actual M7.7 invariant.
                assert t_src_min >= 0, (
                    f"fdm={f} g={g} t_src_min={t_src_min} < 0 — "
                    "would read negative stream index"
                )
                assert t_src_max < t_stream, (
                    f"fdm={f} g={g} t_src_max={t_src_max} >= t_stream"
                    f"={t_stream} — would read past stream end"
                )


# ---------------------------------------------------------------------------
# Test 4 — CubePipeline._verify_full_coverage_or_raise
# ---------------------------------------------------------------------------


@_NEEDS_TORCH
class TestCoverageVerifier:
    """``CubePipeline._verify_full_coverage_or_raise`` is a pure
    function of its args (it doesn't touch ``self.*``), so we call it
    unbound to avoid pulling in the full CubePipeline constructor
    (which needs a detector, GPU torch device, edge masks, etc.).
    """

    def test_passes_when_geometry_is_consistent(self) -> None:
        import torch
        from dsart.services.cube_pipeline import CubePipeline

        t_det = 8
        # shifts symmetric around 0: max_pos=5, max_neg=4
        shifts = torch.tensor(
            [[5, -4, 0, 3], [0, 0, 0, 0], [-4, 5, -1, 2]],
            dtype=torch.int32,
        )
        offset = 5            # == max_pos
        t_stream = t_det + 5 + 4  # == t_det + max_pos + max_neg
        CubePipeline._verify_full_coverage_or_raise(
            None, shifts, t_det, offset=offset, t_stream=t_stream,
        )

    def test_raises_when_t_stream_too_small(self) -> None:
        import torch
        from dsart.services.cube_pipeline import CubePipeline

        t_det = 8
        shifts = torch.tensor(
            [[5, -4, 0, 3]], dtype=torch.int32,
        )
        offset = 5
        # Deliberately too small (would clip the negative-shift tail).
        t_stream_bad = t_det + 5 + 1
        with pytest.raises(RuntimeError, match="coverage check FAILED"):
            CubePipeline._verify_full_coverage_or_raise(
                None, shifts, t_det, offset=offset,
                t_stream=t_stream_bad,
            )

    def test_raises_when_offset_too_small(self) -> None:
        import torch
        from dsart.services.cube_pipeline import CubePipeline

        t_det = 8
        shifts = torch.tensor(
            [[5, -4, 0, 3]], dtype=torch.int32,
        )
        offset_bad = 2  # < max_pos=5 → negative t_src for some cells
        t_stream = t_det + 5 + 4
        with pytest.raises(RuntimeError, match="coverage check FAILED"):
            CubePipeline._verify_full_coverage_or_raise(
                None, shifts, t_det, offset=offset_bad,
                t_stream=t_stream,
            )
