"""recv_epoll with 64 streams (corr_fast --n-sub 4).

MAX_CHGROUPS went 16 -> 64 so sub-band stream ids chgroup*4 + s reach
the ring; MAX_FRAGS_PER_PAYLOAD went 16 -> 4 to keep the memset flow
state at its old size. This checks the ids that used to be dropped now
land byte-exact, and that the bounds still reject what they must.
"""
from __future__ import annotations

import socket
import time
import uuid

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
RxEpoll = recv_epoll.RxEpoll
RxRing = recv_ring.RxRing
RxRingDims = recv_ring.RxRingDims
VF_DATA_PRESENT = recv_ring.VF_DATA_PRESENT

DIMS = RxRingDims(n_corr=64, n_coarse_dm=2, t_buf_samples=64,
                  n_filled_per_corr=100, bytes_per_cell=2)
NBYTES = DIMS.n_filled_per_corr * DIMS.bytes_per_cell


def _hdr(*, seq, chgroup, n_frags=1, frag_idx=0, nbytes=NBYTES):
    return ProdFrameHeader(
        seq=seq, specnum=seq * 256, chgroup=chgroup, dm_idx=1,
        frag_idx=frag_idx, n_frags=n_frags, n_grid=256, n_filled=100,
        pattern_id=0x1234 + chgroup, bits_per_cell=BITS_CINT8_COMPLEX,
        t_int_factor=16, scale=1.0, offset=0.0,
        payload_bytes_in_frag=nbytes,
        flags=FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK,
    )


@pytest.fixture
def rx():
    name = f"/dsart-rx-64s-{uuid.uuid4().hex[:8]}"
    inst = RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                        so_rcvbuf_bytes=8 * 1024 * 1024)
    inst.attach_ring(shm_name=name, owner=True, n_corr=DIMS.n_corr,
                     n_coarse_dm=DIMS.n_coarse_dm,
                     t_buf_samples=DIMS.t_buf_samples,
                     n_filled=DIMS.n_filled_per_corr,
                     bytes_per_cell=DIMS.bytes_per_cell)
    inst.start()
    try:
        yield inst, name
    finally:
        inst.close()
        try:
            RxRing.unlink_name(name)
        except Exception:
            pass


def test_stream_ids_above_15_land_in_the_ring(rx):
    inst, name = rx
    sids = (0, 15, 16, 31, 47, 63)
    for sid in sids:
        inst.set_expected_pattern_id(sid, 0x1234 + sid)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payloads = {}
    try:
        for sid in sids:
            payloads[sid] = bytes(((i + 3 * sid) * 7) & 0xFF for i in range(NBYTES))
            sock.sendto(pack_frame(_hdr(seq=0, chgroup=sid), payloads[sid]),
                        ("127.0.0.1", inst.port))
        time.sleep(0.6)
    finally:
        sock.close()
    c = inst.counters()
    assert c.n_committed == len(sids)
    assert c.bad_field_range_count == 0
    assert c.ring_pattern_mismatch_count == 0
    reader = RxRing.mmap_attach_readonly(name, DIMS)
    try:
        for sid in sids:
            got, vf = reader.read_slot(corr=sid, dm=1, t_seq=0, compute_half=0)
            assert vf & VF_DATA_PRESENT
            assert got == payloads[sid], sid
            assert reader.get_write_seq(sid) == 1
        assert reader.get_write_seq(1) == 0
    finally:
        reader.close()


def test_out_of_range_stream_and_frag_count_rejected(rx):
    inst, _ = rx
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # stream 64 is beyond MAX_CHGROUPS
        sock.sendto(pack_frame(_hdr(seq=0, chgroup=64), bytes(NBYTES)),
                    ("127.0.0.1", inst.port))
        # 5 fragments exceeds MAX_FRAGS_PER_PAYLOAD = 4
        sock.sendto(pack_frame(_hdr(seq=0, chgroup=3, n_frags=5, nbytes=40),
                               bytes(40)), ("127.0.0.1", inst.port))
        time.sleep(0.6)
    finally:
        sock.close()
    c = inst.counters()
    assert c.n_committed == 0
    assert c.bad_field_range_count == 2
