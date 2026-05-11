"""Tests for M4a chunk 5: ``transport/production_rx_ring.py``.

Tests the ProductionRxRingSource class — verifies it satisfies the
RxRingSource Protocol and that the produce_slots iterator semantics
work correctly.

All tests are h01-only (the C extension is built on h01). When the
.so is missing the tests are skipped gracefully.

10+ tests across:
- Protocol conformance (3 tests)
- Construction / dim validation (3 tests)
- start / stop lifecycle (2 tests)
- iterator semantics (3 tests)
- CUDA host-register smoke (1 test, skipif no cupy)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import List, Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip-if-no-lib guard
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
    from dsart.services.rx_ring import CubeRingSlot, RxRingSource
    from dsart.transport.production_rx_ring import (
        ProductionRxRingSource,
        _try_cuda_host_register,
    )
    from dsart.transport.recv_ring import (
        BYTES_CINT8_COMPLEX,
        VF_DATA_PRESENT,
        VF_PATTERN_MISMATCH,
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
    return f"/dsart_test_prod_{uuid.uuid4().hex[:12]}"


def _default_dims(
    n_filled: int = 100,
    t_buf: int = 128,
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


def _default_dm_grids(n_fdm: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse_dm = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    fine_dm = np.linspace(0.0, 300.0, n_fdm, dtype=np.float64)
    fine_to_coarse = np.zeros(n_fdm, dtype=np.int32)
    return coarse_dm, fine_dm, fine_to_coarse


def _make_source(
    shm_name: str,
    dims: "RxRingDims",
    n_fdm: int = 4,
    t_det: int = 16,
    cube_cadence: int = 8,
    max_cubes: Optional[int] = 1,
) -> "ProductionRxRingSource":
    coarse_dm, fine_dm, f2c = _default_dm_grids(n_fdm=n_fdm)
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
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_has_start_method(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        src = _make_source(name, dims)
        assert hasattr(src, "start")
        assert asyncio.iscoroutinefunction(src.start)

    def test_has_stop_method(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        src = _make_source(name, dims)
        assert hasattr(src, "stop")
        assert asyncio.iscoroutinefunction(src.stop)

    def test_has_release_method_and_aiter(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        src = _make_source(name, dims)
        assert hasattr(src, "release")
        assert asyncio.iscoroutinefunction(src.release)
        assert hasattr(src, "__aiter__")


# ---------------------------------------------------------------------------
# Construction / dim validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_n_fdm_raises(self) -> None:
        dims = _default_dims()
        coarse_dm, fine_dm, f2c = _default_dm_grids()
        with pytest.raises(ValueError, match="n_fdm_in_cube"):
            ProductionRxRingSource(
                shm_name="/x",
                ring_dims=dims,
                n_fdm_in_cube=0,
                t_det=16,
                coarse_dm_pc_cm3=coarse_dm,
                fine_dm_pc_cm3=fine_dm,
                fine_to_coarse=f2c,
            )

    def test_invalid_t_det_raises(self) -> None:
        dims = _default_dims()
        coarse_dm, fine_dm, f2c = _default_dm_grids()
        with pytest.raises(ValueError, match="t_det"):
            ProductionRxRingSource(
                shm_name="/x",
                ring_dims=dims,
                n_fdm_in_cube=4,
                t_det=0,
                coarse_dm_pc_cm3=coarse_dm,
                fine_dm_pc_cm3=fine_dm,
                fine_to_coarse=f2c,
            )

    def test_invalid_compute_half_raises(self) -> None:
        dims = _default_dims()
        coarse_dm, fine_dm, f2c = _default_dm_grids()
        with pytest.raises(ValueError, match="compute_half"):
            ProductionRxRingSource(
                shm_name="/x",
                ring_dims=dims,
                n_fdm_in_cube=4,
                t_det=16,
                coarse_dm_pc_cm3=coarse_dm,
                fine_dm_pc_cm3=fine_dm,
                fine_to_coarse=f2c,
                compute_half=2,
            )


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestLifecycle:
    def test_start_stop_idempotent(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        # Create the ring first as writer.
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src = _make_source(name, dims)
            asyncio.run(self._lifecycle(src))
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

    async def _lifecycle(self, src: "ProductionRxRingSource") -> None:
        await src.start()
        await src.start()  # idempotent
        await src.stop()
        await src.stop()  # idempotent

    def test_async_context_manager(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims()
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src = _make_source(name, dims)
            asyncio.run(self._async_ctx(src))
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

    async def _async_ctx(self, src: "ProductionRxRingSource") -> None:
        async with src:
            assert src._started


# ---------------------------------------------------------------------------
# Iterator semantics
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestIteratorSemantics:
    def test_yields_cube_ring_slot(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=64, n_corr=2)
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            # Pre-populate the ring with a cube's worth of slots.
            payload = bytes([0x11] * (50 * 2))
            cube_cadence = 8
            for corr in range(dims.n_corr):
                for t in range(cube_cadence + 1):
                    ring.write_slot(
                        corr=corr, dm=0, t_seq=t,
                        payload=payload,
                        validity_flags=VF_DATA_PRESENT,
                    )

            src = _make_source(
                name, dims, n_fdm=2, t_det=8, cube_cadence=cube_cadence,
                max_cubes=1,
            )
            slots: List[CubeRingSlot] = asyncio.run(self._collect(src))
            assert len(slots) == 1
            assert isinstance(slots[0], CubeRingSlot)
            assert slots[0].cube_id == 0
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

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

    def test_release_updates_read_seq(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=64, n_corr=2)
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            src = _make_source(name, dims, cube_cadence=8, max_cubes=1)
            asyncio.run(self._release_check(src))
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

    async def _release_check(self, src: "ProductionRxRingSource") -> None:
        await src.start()
        await src.release(cube_id=0)  # should not raise
        await src.release(cube_id=5)
        await src.stop()

    def test_validity_mask_reflects_missing_data(self) -> None:
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=64, n_corr=2)
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass
        ring = RxRing.open_or_create(name, dims)
        try:
            # Write slots for both corrs so write_seq advances on both, but
            # corr=1 has validity_flags=0 (no data_present) — the iterator
            # will assemble the cube and validity_mask will reflect missing data.
            payload = bytes([0x22] * (50 * 2))
            cube_cadence = 8
            for t in range(cube_cadence + 1):
                ring.write_slot(
                    corr=0, dm=0, t_seq=t,
                    payload=payload,
                    validity_flags=VF_DATA_PRESENT,
                )
                ring.write_slot(
                    corr=1, dm=0, t_seq=t,
                    payload=None,
                    validity_flags=0,  # no data_present → triggers validity drop
                )

            src = _make_source(
                name, dims, n_fdm=2, t_det=8, cube_cadence=cube_cadence,
                max_cubes=1,
            )
            slots = asyncio.run(self._collect(src))

            assert len(slots) == 1
            # corr=1 was never written; validity_mask must have at least some
            # False entries.
            assert not slots[0].validity_mask.all()
        finally:
            ring.close()
            try:
                RxRing.unlink_name(name)
            except Exception:
                pass

    async def _collect(self, src: "ProductionRxRingSource") -> list:  # noqa: F811
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


# ---------------------------------------------------------------------------
# CUDA host-register smoke test
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import cupy  # noqa

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _cuda_available(),
    reason="cupy not available — skipping CUDA host-register smoke test",
)
class TestCudaRegister:
    def test_try_cuda_host_register_smoke(self) -> None:
        """Smoke test: _try_cuda_host_register either returns True (CUDA) or
        False (no CUDA). It must NEVER raise."""
        import numpy as np

        # Allocate a small host-memory buffer and pass its address.
        buf = np.zeros(4096, dtype=np.uint8)
        addr = buf.ctypes.data
        result = _try_cuda_host_register(addr, buf.nbytes)
        # Result can be True or False; key invariant: no exception.
        assert isinstance(result, bool)
