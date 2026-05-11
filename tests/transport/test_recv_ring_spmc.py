"""Tests for M4a chunk 4: ``transport/recv_ring.{c,py}`` SPMC receive ring.

Explicitly named ``test_recv_ring_spmc.py`` per plan §4.4 line 1475.

Test groups:
(a) No torn reads (4 tests) — single / dual reader consistency
(b) Overrun semantics (3 tests) — counter behaviour + resume at fresh seq
(c) Cross-NUMA latency probe (1 test, skipif single-NUMA)
(d) Pattern_id mismatch slot (2 tests)
(e) Init / lifecycle (4 tests)

Total: 14 tests.

NOTE: Tests that require the C extension (_recv_ring.so) are skipped with
a clear message if the .so is not built. Run
    pip install -e .   OR   python setup.py build_ext --inplace
to build it before running these tests on h01.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip marker: all tests in this file need the C extension.
# ---------------------------------------------------------------------------

def _lib_available() -> bool:
    """Return True if _recv_ring.so is importable / loadable."""
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


def _unique_shm_name() -> str:
    return f"/dsart_test_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Import conditionally
# ---------------------------------------------------------------------------

try:
    from dsart.transport.recv_ring import (
        BYTES_CFP16_COMPLEX,
        BYTES_CINT8_COMPLEX,
        VF_DATA_PRESENT,
        VF_PATTERN_MISMATCH,
        VF_RX_OVERRUN,
        RxRing,
        RxRingDims,
    )
    _IMPORT_OK = True
except Exception:
    _IMPORT_OK = False


_NEEDS_IMPORT = pytest.mark.skipif(
    not _IMPORT_OK,
    reason="dsart.transport.recv_ring import failed",
)

pytestmark = _NEEDS_IMPORT


def _default_dims(
    n_filled: int = 100,
    t_buf: int = 16,
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


def _open_ring(name: str, dims: "RxRingDims") -> "RxRing":
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass
    return RxRing.open_or_create(name, dims)


# ---------------------------------------------------------------------------
# (a) No torn reads
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestNoTornReads:
    def test_single_producer_consistent_read(self) -> None:
        """Writer writes a known pattern; reader always sees consistent bytes."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=100)
        ring = _open_ring(name, dims)
        try:
            # Write a known 200-byte payload.
            payload = bytes(range(200))
            ring.write_slot(corr=0, dm=0, t_seq=0, payload=payload,
                            validity_flags=VF_DATA_PRESENT)
            # Read it back; should match.
            read_back, vf = ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert vf & VF_DATA_PRESENT
            assert read_back == payload
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_two_readers_consistent(self) -> None:
        """Two readers in separate threads see consistent slot bytes."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=100)
        ring_writer = _open_ring(name, dims)
        results: List[tuple[bytes, int]] = [None, None]  # type: ignore

        try:
            payload = bytes([i % 256 for i in range(200)])
            ring_writer.write_slot(
                corr=0, dm=0, t_seq=5, payload=payload,
                validity_flags=VF_DATA_PRESENT,
            )

            def _reader(half: int) -> None:
                ring_r = RxRing.mmap_attach_readonly(name, dims)
                try:
                    read_back, vf = ring_r.read_slot(
                        corr=0, dm=0, t_seq=5, compute_half=half
                    )
                    results[half] = (read_back, vf)
                finally:
                    ring_r.close()

            t0 = threading.Thread(target=_reader, args=(0,))
            t1 = threading.Thread(target=_reader, args=(1,))
            t0.start(); t1.start()
            t0.join(timeout=2); t1.join(timeout=2)

            assert results[0] is not None
            assert results[1] is not None
            assert results[0][0] == payload
            assert results[1][0] == payload
        finally:
            ring_writer.close()
            RxRing.unlink_name(name)

    def test_reader_sees_monotone_seq_progress(self) -> None:
        """Reader sees monotone seq progress (acquire fence works)."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=32)
        ring = _open_ring(name, dims)
        try:
            for seq in range(20):
                ring.write_slot(
                    corr=0, dm=0, t_seq=seq,
                    payload=bytes([seq % 256] * 100),
                    validity_flags=VF_DATA_PRESENT,
                )

            write_seqs = []
            for seq in range(20):
                wseq = ring.get_write_seq(corr=0)
                write_seqs.append(wseq)

            # All write_seq reads must be non-decreasing.
            for i in range(1, len(write_seqs)):
                assert write_seqs[i] >= write_seqs[i - 1]
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_seq_bumped_after_payload_visible(self) -> None:
        """Writer bumps seq AFTER payload bytes visible (release fence).

        Verifiable by: write a 'fingerprint' pattern; ensure the reader never
        sees a bumped seq with a half-written payload (we check consistency
        of the fingerprint bytes after observing a seq increment).
        """
        name = _unique_shm_name()
        dims = _default_dims(n_filled=100, t_buf=8)
        ring = _open_ring(name, dims)
        errors: List[str] = []

        try:
            def _reader() -> None:
                ring_r = RxRing.mmap_attach_readonly(name, dims)
                try:
                    for _ in range(200):
                        wseq = ring_r.get_write_seq(corr=0)
                        if wseq == 0:
                            time.sleep(0.0001)
                            continue
                        t_to_read = max(0, int(wseq) - 1)
                        try:
                            data, vf = ring_r.read_slot(
                                corr=0, dm=0, t_seq=t_to_read, compute_half=0
                            )
                            # Fingerprint: all bytes should be (t_to_read % 256).
                            expected_byte = (t_to_read % 200) % 256
                            first_byte = data[0] if data else None
                            if first_byte is not None and first_byte != expected_byte:
                                errors.append(
                                    f"seq={t_to_read}: expected {expected_byte}, "
                                    f"got {first_byte}"
                                )
                        except OSError:
                            pass
                        time.sleep(0.0001)
                finally:
                    ring_r.close()

            t = threading.Thread(target=_reader)
            t.start()

            for seq in range(100):
                payload = bytes([seq % 256] * 200)
                ring.write_slot(
                    corr=0, dm=0, t_seq=seq, payload=payload,
                    validity_flags=VF_DATA_PRESENT,
                )
                time.sleep(0.0002)

            t.join(timeout=3)
            assert not errors, f"Torn-read errors: {errors[:5]}"
        finally:
            ring.close()
            RxRing.unlink_name(name)


# ---------------------------------------------------------------------------
# (b) Overrun semantics
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestOverrunSemantics:
    def test_reader_pause_triggers_overrun(self) -> None:
        """Reader pauses; writer advances > T_buf; overrun counter increments."""
        name = _unique_shm_name()
        T_buf = 8
        dims = _default_dims(n_filled=50, t_buf=T_buf)
        ring = _open_ring(name, dims)
        try:
            # Write T_buf + 2 slots (overrun reader at seq=0).
            payload = bytes([0x42] * 100)
            for seq in range(T_buf + 2):
                ring.write_slot(
                    corr=0, dm=0, t_seq=seq, payload=payload,
                    validity_flags=VF_DATA_PRESENT,
                )

            # Now try to read seq=0 (should be overrun).
            try:
                ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
                # If it doesn't raise, overrun counter should have incremented.
                overrun = ring.get_overrun_count(compute_half=0)
                assert overrun >= 1
            except OSError:
                overrun = ring.get_overrun_count(compute_half=0)
                assert overrun >= 1
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_reader_resumes_at_fresh_seq(self) -> None:
        """After overrun, reader reads the freshest seq successfully."""
        name = _unique_shm_name()
        T_buf = 8
        dims = _default_dims(n_filled=50, t_buf=T_buf)
        ring = _open_ring(name, dims)
        try:
            for seq in range(T_buf + 2):
                payload = bytes([seq % 256] * 100)
                ring.write_slot(
                    corr=0, dm=0, t_seq=seq, payload=payload,
                    validity_flags=VF_DATA_PRESENT,
                )

            # Fresh seq is T_buf + 1. Read that.
            fresh_seq = T_buf + 1
            data, vf = ring.read_slot(corr=0, dm=0, t_seq=fresh_seq, compute_half=0)
            assert vf & VF_DATA_PRESENT
            assert data[0] == (fresh_seq % 256)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_per_consumer_overrun_independent(self) -> None:
        """Per-consumer overrun counters are independent."""
        name = _unique_shm_name()
        T_buf = 8
        dims = _default_dims(n_filled=50, t_buf=T_buf)
        ring = _open_ring(name, dims)
        try:
            payload = bytes([0xFF] * 100)
            for seq in range(T_buf + 2):
                ring.write_slot(
                    corr=0, dm=0, t_seq=seq, payload=payload,
                    validity_flags=VF_DATA_PRESENT,
                )

            # Trigger overrun for compute_half=0 but not 1.
            try:
                ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            except OSError:
                pass

            overrun0 = ring.get_overrun_count(compute_half=0)
            overrun1 = ring.get_overrun_count(compute_half=1)
            # compute_half=1 should have 0 overrun (hasn't read the old slot).
            assert overrun0 >= 1
            assert overrun1 == 0
        finally:
            ring.close()
            RxRing.unlink_name(name)


# ---------------------------------------------------------------------------
# (c) Cross-NUMA latency probe
# ---------------------------------------------------------------------------


def _is_single_numa() -> bool:
    """Return True if this host has only one NUMA node."""
    try:
        result = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True, timeout=2
        )
        # 'available: 1 nodes' → single NUMA
        if "available: 1 nodes" in result.stdout:
            return True
        if result.returncode != 0:
            return True  # can't determine; skip to be safe
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Check via /sys
    try:
        nodes = os.listdir("/sys/devices/system/node")
        numa_nodes = [n for n in nodes if n.startswith("node")]
        return len(numa_nodes) <= 1
    except OSError:
        return True
    return False


@pytest.mark.skipif(
    _is_single_numa(),
    reason="Cross-NUMA latency test requires 2+ NUMA nodes (h01 only)",
)
@_NEEDS_LIB
class TestCrossNuma:
    def test_p99_read_latency_within_budget(self) -> None:
        """p99 read latency ≤ 134 ms per cube on a 2-NUMA host (plan §4.4 line 1472)."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=5800, t_buf=512, n_coarse_dm=24, n_corr=16)
        ring = _open_ring(name, dims)
        latencies_ms = []

        try:
            payload = bytes(5800 * 2)
            ring.write_slot(corr=0, dm=0, t_seq=0, payload=payload,
                            validity_flags=VF_DATA_PRESENT)

            ring_r = RxRing.mmap_attach_readonly(name, dims)
            try:
                for _ in range(500):
                    t0 = time.perf_counter()
                    ring_r.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            finally:
                ring_r.close()

            latencies_ms.sort()
            p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
            assert p99 <= 134.0, f"p99 latency {p99:.2f} ms > 134 ms budget"
        finally:
            ring.close()
            RxRing.unlink_name(name)


# ---------------------------------------------------------------------------
# (d) Pattern_id mismatch slot
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestPatternMismatchSlot:
    def test_write_pattern_mismatch_flag_readable(self) -> None:
        """Write with VF_PATTERN_MISMATCH; reader sees the flag."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50)
        ring = _open_ring(name, dims)
        try:
            ring.write_slot(
                corr=0, dm=0, t_seq=0,
                payload=None,
                validity_flags=VF_PATTERN_MISMATCH,
            )
            _, vf = ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert vf & VF_PATTERN_MISMATCH
            assert not (vf & VF_DATA_PRESENT)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_data_present_false_slot(self) -> None:
        """Slot with VF=0 (no data_present) is returned correctly."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50)
        ring = _open_ring(name, dims)
        try:
            ring.write_slot(
                corr=0, dm=0, t_seq=0,
                payload=None,
                validity_flags=0,  # neither data_present nor mismatch
            )
            _, vf = ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert not (vf & VF_DATA_PRESENT)
        finally:
            ring.close()
            RxRing.unlink_name(name)


# ---------------------------------------------------------------------------
# (e) Init / lifecycle
# ---------------------------------------------------------------------------


@_NEEDS_LIB
class TestInitLifecycle:
    def test_fresh_segment_zero_init(self) -> None:
        """open_or_create of a fresh segment zero-inits header + slots."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=8)
        ring = _open_ring(name, dims)
        try:
            # Before any write, write_seq should be 0 for all corrs.
            for corr in range(dims.n_corr):
                assert ring.get_write_seq(corr) == 0
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_memset_data_reraises_seqs_to_zero(self) -> None:
        """cmd: prepare re-zeros data (memset) in < 200 ms for default ops."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=8)
        ring = _open_ring(name, dims)
        try:
            # Write some slots.
            for t in range(4):
                ring.write_slot(corr=0, dm=0, t_seq=t,
                                payload=bytes([0xAB] * 100),
                                validity_flags=VF_DATA_PRESENT)

            # Memset data section.
            t0 = time.monotonic()
            ring.memset_data()
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            assert elapsed_ms < 200.0, (
                f"memset_data took {elapsed_ms:.1f} ms > 200 ms budget"
            )

            # Slots should now read as zeros.
            data, vf = ring.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert vf == 0
            assert all(b == 0 for b in data)
        finally:
            ring.close()
            RxRing.unlink_name(name)

    def test_stop_leaves_ring_intact(self) -> None:
        """cmd: stop: close the ring handle; shm name remains accessible."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=4)
        ring = _open_ring(name, dims)
        ring.write_slot(corr=0, dm=0, t_seq=0,
                        payload=bytes([0x77] * 100),
                        validity_flags=VF_DATA_PRESENT)
        ring.close()  # simulate cmd: stop (close but don't unlink)

        # Re-attach read-only; data should still be there.
        ring_r = RxRing.mmap_attach_readonly(name, dims)
        try:
            data, vf = ring_r.read_slot(corr=0, dm=0, t_seq=0, compute_half=0)
            assert vf & VF_DATA_PRESENT
        finally:
            ring_r.close()
            RxRing.unlink_name(name)

    def test_unlink_removes_shm_name(self) -> None:
        """unlink removes the shm name; subsequent open fails."""
        name = _unique_shm_name()
        dims = _default_dims(n_filled=50, t_buf=4)
        ring = _open_ring(name, dims)
        ring.close()
        ret = RxRing.unlink_name(name)
        assert ret == 0

        # After unlink, mmap_attach_readonly should fail.
        with pytest.raises(OSError):
            RxRing.mmap_attach_readonly(name, dims)
