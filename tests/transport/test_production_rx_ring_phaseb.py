"""M7.2 end-to-end transport test:
    UDP loopback -> recv_epoll (Phase B publish) -> RxRing shm
                 -> ProductionRxRingSource._assemble_cube -> CubeRingSlot

This complements ``test_recv_epoll_ring_phaseb.py`` (which stops at the
ring) by also driving the consumer side of the ring (the M5/M6
``ProductionRxRingSource`` async iterator). It verifies that:

  * A complete cube's worth of UDP frames lands in the ring, the
    consumer's write_seq poll fires, ``_assemble_cube`` returns a
    ``CubeRingSlot``, and the slot's shape contract holds.
  * Per-(corr, dm, t) ring reads happen for EVERY (corr, dm, t) in the
    cube (the ``n_slots_read`` mon counter matches
    ``n_corr × n_coarse_dm × cube_cadence_samples``).
  * ``validity_mask`` is all-True when every slot is data-present, and
    correctly drops to False when frames are dropped on the wire.
  * Multiple cubes flow in order with monotone ``cube_id`` and
    ``specnum_start`` advancing by ``cube_cadence_samples`` each step.

The test uses the same dimensions the M7.2 production path will
(n_corr=4, n_coarse_dm=2, small cube_cadence and t_det so the test
runs in <2 s), and sends frames as fast as the loopback socket can
push them (no rate limit; the goal is correctness, not throughput).
"""
from __future__ import annotations

import asyncio
import socket
import time
import uuid

import numpy as np
import pytest

from dsart.transport.prod_frame import (
    BITS_CINT8_COMPLEX,
    FLAG_LAST_IN_BLOCK,
    FLAG_QUANTIZED,
    ProdFrameHeader,
    pack_frame,
)

recv_epoll = pytest.importorskip("dsart.transport.recv_epoll")
recv_ring = pytest.importorskip("dsart.transport.recv_ring")
production_rx_ring = pytest.importorskip(
    "dsart.transport.production_rx_ring"
)

RxEpoll = recv_epoll.RxEpoll
RxRingDims = recv_ring.RxRingDims
RxRing = recv_ring.RxRing
ProductionRxRingSource = production_rx_ring.ProductionRxRingSource

PID_OK = 0xBADCAFE12345CAFE

# Small dims so the test is quick. Keep cube_cadence_samples and t_det
# small enough that we can saturate the ring in <1 s of loopback sends.
N_CORR = 4
N_COARSE_DM = 2
T_BUF_SAMPLES = 64           # ring depth in samples
N_FILLED = 100
BPC = 2
CUBE_CADENCE = 16            # samples per cube (production uses 256)
T_DET = 12                   # detector window (< cube_cadence)
N_FDM = 4
N_GRID = 16                  # tiny grid for fast alloc

PAYLOAD_BYTES = N_FILLED * BPC


def _hdr(*, seq: int, chgroup: int, dm_idx: int,
         payload_bytes: int = PAYLOAD_BYTES,
         pattern_id: int = PID_OK) -> ProdFrameHeader:
    return ProdFrameHeader(
        seq=seq, specnum=seq * 256,
        chgroup=chgroup, dm_idx=dm_idx,
        frag_idx=0, n_frags=1,
        n_grid=N_GRID, n_filled=N_FILLED,
        pattern_id=pattern_id,
        bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=16,
        scale=1.0, offset=0.0,
        payload_bytes_in_frag=payload_bytes,
        flags=FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK,
    )


def _shm_name(test_name: str) -> str:
    return f"/dsart-prod-rx-{test_name}-{uuid.uuid4().hex[:8]}"


def _make_dm_grids():
    """Minimal coarse/fine DM grids for ``compute_time_shift_search``.
    Matches the convention used in tests/transport/test_production_rx_ring.py:
    every fine-DM maps to coarse cell 0 so δdm = fine_dm − 0 ≥ 0 and
    the shift table is non-negative."""
    coarse_dm = np.linspace(0.0, 300.0, N_COARSE_DM, dtype=np.float64)
    fine_dm = np.linspace(0.0, 100.0, N_FDM, dtype=np.float64)
    fine_to_coarse = np.zeros(N_FDM, dtype=np.int32)
    return coarse_dm, fine_dm, fine_to_coarse


@pytest.fixture
def shm_name(request):
    name = _shm_name(request.node.name.replace("[", "-").replace("]", ""))
    yield name
    try:
        RxRing.unlink_name(name)
    except Exception:
        pass


@pytest.fixture
def producer(shm_name):
    """Start a recv_epoll with a ring attached (singleton RxEpoll)."""
    rx = RxEpoll.open(
        bind_host="127.0.0.1", bind_port=0,
        so_rcvbuf_bytes=8 * 1024 * 1024,
    )
    rx.attach_ring(
        shm_name=shm_name, owner=True,
        n_corr=N_CORR, n_coarse_dm=N_COARSE_DM,
        t_buf_samples=T_BUF_SAMPLES, n_filled=N_FILLED,
        bytes_per_cell=BPC,
    )
    for chg in range(N_CORR):
        rx.set_expected_pattern_id(chg, PID_OK)
    rx.start()
    try:
        yield rx, shm_name
    finally:
        rx.close()


def _send_full_cube(tx_sock: socket.socket, port: int, seq_start: int):
    """Send every (corr, coarse_dm, seq) slot for one cube. Each
    frame has a distinguishable payload (chgroup-encoded constant)."""
    for seq in range(seq_start, seq_start + CUBE_CADENCE):
        for chgroup in range(N_CORR):
            for dm in range(N_COARSE_DM):
                payload = bytes((chgroup * 16 + dm,) * PAYLOAD_BYTES)
                hdr = _hdr(seq=seq, chgroup=chgroup, dm_idx=dm,
                           payload_bytes=len(payload))
                tx_sock.sendto(pack_frame(hdr, payload), ("127.0.0.1", port))


def _wait_for_write_seq(rx_handle: RxRing, target: int,
                        timeout_s: float = 5.0) -> None:
    """Block until every per-corr write_seq has reached ``target``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(rx_handle.get_write_seq(c) >= target for c in range(N_CORR)):
            return
        time.sleep(0.02)
    raise AssertionError(
        "write_seq did not reach target: "
        f"{[rx_handle.get_write_seq(c) for c in range(N_CORR)]} target={target}"
    )


# ---------------------------------------------------------------------------
# A. Single-cube end-to-end
# ---------------------------------------------------------------------------


def test_single_cube_round_trip(producer):
    rx, shm = producer
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        _send_full_cube(tx_sock, rx.port, seq_start=0)
        # Wait for the RX writer to have flushed all CUBE_CADENCE slots
        # per corr (full cube + a small safety margin).
        ring_reader = RxRing.mmap_attach_readonly(
            shm,
            RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES, N_FILLED, BPC),
        )
        try:
            _wait_for_write_seq(ring_reader, CUBE_CADENCE)
        finally:
            ring_reader.close()
    finally:
        tx_sock.close()

    coarse_dm, fine_dm, fine_to_coarse = _make_dm_grids()
    source = ProductionRxRingSource(
        shm_name=shm,
        ring_dims=RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES,
                             N_FILLED, BPC),
        n_fdm_in_cube=N_FDM,
        t_det=T_DET,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        cube_cadence_samples=CUBE_CADENCE,
        n_grid=N_GRID,
        enable_cuda_register=False,
        poll_interval_s=0.005,
        max_cubes=1,
    )

    async def drain_one_cube():
        await source.start()
        try:
            async for slot in source:
                return slot
        finally:
            await source.stop()

    slot = asyncio.run(drain_one_cube())

    assert slot is not None
    assert slot.cube_id == 0
    assert slot.specnum_start == 0
    assert slot.t_det == T_DET
    assert slot.n_fdm_in_cube == N_FDM
    assert slot.n_grid == N_GRID
    # Validity mask shape per CubeRingSlot contract.
    assert slot.validity_mask.shape == (T_DET, N_FDM)
    # Every slot was data-present (we sent everything) -> all-True mask.
    assert slot.validity_mask.all(), (
        f"expected all-True validity, got {slot.validity_mask}"
    )
    # per_chgroup_streams covers every corr in ring_dims.
    assert set(slot.per_chgroup_streams.keys()) == set(range(N_CORR))
    for corr in range(N_CORR):
        stream = slot.per_chgroup_streams[corr]
        # T_stream = t_det + max(time_shift), N_grid square.
        assert stream.shape[1] == N_GRID
        assert stream.shape[2] == N_GRID
        assert stream.shape[0] >= T_DET
        assert stream.dtype == np.complex64
    # M7.2 search-overlap: assemble walks t_det rows per (corr, dm) —
    # cube_cadence is just the cube-emit stride, not the walk length.
    assert source.stats["n_slots_read"] == N_CORR * N_COARSE_DM * T_DET
    assert source.stats["n_overrun"] == 0
    assert source.stats["n_pattern_mismatch"] == 0
    assert source.stats["n_no_data_present"] == 0
    assert source.stats["cubes_emitted"] == 1


# ---------------------------------------------------------------------------
# B. Multi-cube monotone iteration
# ---------------------------------------------------------------------------


def test_two_cubes_yielded_in_order(producer):
    rx, shm = producer
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n_cubes_to_send = 2
    try:
        for k in range(n_cubes_to_send):
            _send_full_cube(tx_sock, rx.port, seq_start=k * CUBE_CADENCE)
        ring_reader = RxRing.mmap_attach_readonly(
            shm,
            RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES, N_FILLED, BPC),
        )
        try:
            _wait_for_write_seq(ring_reader, n_cubes_to_send * CUBE_CADENCE)
        finally:
            ring_reader.close()
    finally:
        tx_sock.close()

    coarse_dm, fine_dm, fine_to_coarse = _make_dm_grids()
    source = ProductionRxRingSource(
        shm_name=shm,
        ring_dims=RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES,
                             N_FILLED, BPC),
        n_fdm_in_cube=N_FDM,
        t_det=T_DET,
        coarse_dm_pc_cm3=coarse_dm,
        fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        cube_cadence_samples=CUBE_CADENCE,
        n_grid=N_GRID,
        enable_cuda_register=False,
        poll_interval_s=0.005,
        max_cubes=n_cubes_to_send,
    )

    async def drain():
        await source.start()
        out = []
        try:
            async for slot in source:
                out.append(slot)
        finally:
            await source.stop()
        return out

    slots = asyncio.run(drain())
    assert len(slots) == n_cubes_to_send
    assert slots[0].cube_id == 0
    assert slots[1].cube_id == 1
    assert slots[1].specnum_start - slots[0].specnum_start == CUBE_CADENCE
    for s in slots:
        assert s.validity_mask.all()
    assert source.stats["cubes_emitted"] == n_cubes_to_send
    # M7.2 search-overlap: each cube walks t_det rows per (corr, dm).
    assert (
        source.stats["n_slots_read"]
        == n_cubes_to_send * N_CORR * N_COARSE_DM * T_DET
    )


# ---------------------------------------------------------------------------
# C. Missing frames -> validity_mask drops
# ---------------------------------------------------------------------------


def test_pattern_mismatch_drops_validity(producer):
    """Inject a wrong-pattern frame at t=3, dm=0, corr=2. The recv_epoll
    Phase B path publishes a VF_PATTERN_MISMATCH stub at that slot; the
    assembler should mark validity_mask[3, :] = False."""
    rx, shm = producer
    PID_BAD = 0xDEADBAD0DEADBAD0
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Send a full cube, but for (corr=2, dm=0, seq=3) substitute a
        # wrong pattern_id so the C path takes the mismatch branch.
        for seq in range(CUBE_CADENCE):
            for chgroup in range(N_CORR):
                for dm in range(N_COARSE_DM):
                    pid = PID_OK
                    if chgroup == 2 and dm == 0 and seq == 3:
                        pid = PID_BAD
                    payload = bytes((chgroup * 16 + dm,) * PAYLOAD_BYTES)
                    hdr = _hdr(seq=seq, chgroup=chgroup, dm_idx=dm,
                               payload_bytes=len(payload), pattern_id=pid)
                    tx_sock.sendto(pack_frame(hdr, payload),
                                   ("127.0.0.1", rx.port))
        ring_reader = RxRing.mmap_attach_readonly(
            shm,
            RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES, N_FILLED, BPC),
        )
        try:
            _wait_for_write_seq(ring_reader, CUBE_CADENCE)
        finally:
            ring_reader.close()
    finally:
        tx_sock.close()

    coarse_dm, fine_dm, fine_to_coarse = _make_dm_grids()
    source = ProductionRxRingSource(
        shm_name=shm,
        ring_dims=RxRingDims(N_CORR, N_COARSE_DM, T_BUF_SAMPLES,
                             N_FILLED, BPC),
        n_fdm_in_cube=N_FDM, t_det=T_DET,
        coarse_dm_pc_cm3=coarse_dm, fine_dm_pc_cm3=fine_dm,
        fine_to_coarse=fine_to_coarse,
        cube_cadence_samples=CUBE_CADENCE, n_grid=N_GRID,
        enable_cuda_register=False, poll_interval_s=0.005,
        max_cubes=1,
    )

    async def drain_one():
        await source.start()
        try:
            async for slot in source:
                return slot
        finally:
            await source.stop()

    slot = asyncio.run(drain_one())
    # t=3 should be False on every FDM (one chgroup is bad => coarse drop).
    assert not slot.validity_mask[3, :].any(), (
        f"expected validity_mask[3] all False, got {slot.validity_mask[3]}"
    )
    # Other in-window t values should still be True.
    other_t = [t for t in range(T_DET) if t != 3]
    for t in other_t:
        assert slot.validity_mask[t, :].all(), (
            f"validity_mask[t={t}] dropped unexpectedly: "
            f"{slot.validity_mask[t]}"
        )
    assert source.stats["n_pattern_mismatch"] >= 1
