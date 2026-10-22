"""recv_epoll v4 mode: UDP -> frame v2 -> ring v4 (localhost, low rate)."""
from __future__ import annotations

import importlib.util
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from dsart.transport import prod_frame, prod_frame2 as f2
from dsart.x2 import quant as q

re_ = pytest.importorskip("dsart.transport.recv_epoll")
rr = pytest.importorskip("dsart.transport.recv_ring")
if not hasattr(re_._find_lib(), "recv_epoll_attach_ring4"):
    pytest.skip("_recv_epoll.so built without ring v4 mode", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec)
sys.modules["wst_util"] = wu
_spec.loader.exec_module(wu)

S_MAIN = float(np.float32(0.3534))
S_HEAVY = float(np.float32(0.1))
BPS = 22


def small_spec(**kw):
    base = dict(dms=(3, 11), t_int_factor=(16, 32), t_buf=(64, 32), n_stream=4, n_sub=2,
                bits_main=8, bits_heavy=16, n_grid=256, s_main=S_MAIN, s_heavy=S_HEAVY,
                stream_n_values=(20,) * 4, stream_n_heavy=(2,) * 4, min_retention_s=0.0,
                reorder_rows=8, hole_timeout_s=0.05)
    base.update(kw)
    return rr.RingV4Spec(**base)


class Rx:
    """RxEpoll in v4 mode on 127.0.0.1 with one port per chgroup."""

    def __init__(self, n_ports=2, n_threads=1, strict=False, spec=None):
        self.name = f"/dsart-x2-test-{uuid.uuid4().hex[:8]}"
        self.rx = re_.RxEpoll.open(bind_host="127.0.0.1", bind_port=0,
                                   so_rcvbuf_bytes=8 << 20)
        self.ports = [self.rx.port]
        for _ in range(n_ports - 1):
            self.ports.append(self.rx.add_port(bind_host="127.0.0.1", bind_port=0,
                                               so_rcvbuf_bytes=8 << 20))
        self.rx.attach_ring4(shm_name=self.name, spec=spec or small_spec(), prefault=False,
                             n_threads=n_threads, strict_ports=strict)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, dg: bytes, chgroup: int) -> None:
        self.tx.sendto(dg, ("127.0.0.1", self.ports[chgroup]))

    def reader(self, access="mmap"):
        return rr.RxRingV4Reader.open(self.name, access=access)

    def close(self):
        self.tx.close()
        self.rx.close()
        rr.unlink_ring_v4(self.name)


@pytest.fixture
def rx1():
    r = Rx()
    yield r
    r.close()


def _wait(pred, timeout=3.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_single_thread_round_trip(rx1):
    assert rx1.rx.ring4_attached and rx1.rx.n_threads4 == 1
    rx1.rx.start()
    for s in range(4):
        for seq in range(0, 16, 2):
            rx1.send(wu.synthetic_frame(stream=s, dm=3, seq_first=seq), chgroup=s // 2)
    r = rx1.reader()
    win = r.wait_window(3, 0, 16, timeout_s=3.0)
    assert win.status == "complete" and (win.validity == rr.RV_PRESENT).all()
    for s in range(4):
        np.testing.assert_array_equal(win.view[s, :, 0], np.arange(16))
    assert rx1.rx.counters().n_committed == 32
    assert r.counters()["writer"]["ok"] == 32
    assert r.heartbeat_age_s() < 1.0                  # the thread sweeps every 10 ms
    r.close()


def test_v1_frames_and_garbage_are_counted_and_dropped(rx1):
    rx1.rx.start()
    v1 = prod_frame.pack_frame(prod_frame.ProdFrameHeader(
        seq=0, specnum=0, chgroup=0, dm_idx=3, frag_idx=0, n_frags=1, n_grid=256, n_filled=1,
        pattern_id=7, bits_per_cell=16, t_int_factor=16, scale=1.0, offset=0.0,
        payload_bytes_in_frag=2, flags=prod_frame.FLAG_QUANTIZED), b"\x00\x00")
    rx1.send(v1, 0)
    rx1.send(b"junk" * 30, 0)
    rx1.send(wu.synthetic_frame(stream=0, dm=9), 0)                 # not owned
    assert _wait(lambda: rx1.rx.counters().n_received == 3)
    r = rx1.reader(access="pread")
    _wait(lambda: r.counters()["writer"]["datagrams"] == 3)
    c = r.counters()["writer"]
    assert (c["v1_frames"], c["bad_version"], c["bad_magic"], c["not_owned"]) == (1, 1, 1, 1)
    c2 = rx1.rx.counters()
    assert c2.bad_version_count == 1 and c2.bad_magic_count == 1 and c2.n_committed == 0
    r.close()


def test_two_threads_own_disjoint_streams():
    rx = Rx(n_ports=2, n_threads=2)
    try:
        assert rx.rx.n_threads4 == 2
        rx.rx.start()
        for s in range(4):
            for seq in range(0, 8, 2):
                rx.send(wu.synthetic_frame(stream=s, dm=11, seq_first=seq, t_int_factor=32), s // 2)
        rx.send(wu.synthetic_frame(stream=3, dm=11, seq_first=8, t_int_factor=32), 0)   # wrong port
        r = rx.reader()
        assert r.wait_window(11, 0, 8, timeout_s=3.0).status == "complete"
        assert _wait(lambda: r.counters()["writer"]["port_mismatch"] == 1)
        assert rx.rx.counters().n_committed == 16
        r.close()
    finally:
        rx.close()


def test_thread_sweep_declares_tail_hole_in_real_time(rx1):
    rx1.rx.start()
    for seq in (0, 2, 6):                                            # rows 4, 5 lost
        rx1.send(wu.synthetic_frame(stream=0, dm=3, seq_first=seq), 0)
    r = rx1.reader()
    # within the 8-row reorder horizon: only the 50 ms gap timeout closes it
    assert _wait(lambda: int(r.heads(3)["complete_through"][0]) == 8, timeout=2.0)
    assert r.window_validity(3, 0, 8)[0].tolist() == [1, 1, 1, 1, 2, 2, 1, 1]
    r.close()


def test_attach_errors():
    rx = re_.RxEpoll.open(bind_host="127.0.0.1", bind_port=0, so_rcvbuf_bytes=1 << 20)
    name = f"/dsart-x2-test-{uuid.uuid4().hex[:8]}"
    try:
        with pytest.raises(RuntimeError):
            rx.attach_ring4(shm_name=name, spec=small_spec(), n_threads=2)   # 1 socket only
        rx.attach_ring4(shm_name=name, spec=small_spec(), prefault=False)
        with pytest.raises(RuntimeError):
            rx.attach_ring4(shm_name=name, spec=small_spec())                 # twice
        with pytest.raises(RuntimeError):
            rx.add_port(bind_host="127.0.0.1", bind_port=0)                   # after v4 attach
        with pytest.raises(RuntimeError):
            rx.attach_ring(shm_name="/dsart-x2-test-v3", owner=True, n_corr=4,
                           n_coarse_dm=1, t_buf_samples=8, n_filled=4, bytes_per_cell=2)
    finally:
        rx.close()
        rr.unlink_ring_v4(name)


def test_sender_to_rx_v4_end_to_end_real_map():
    """MmsgSender -> RxEpoll v4 -> RxRingV4Reader with WS-P's gridded n_sub 2
    map and the Phase 0 row rule: rows decode to the sent codes."""
    sm = pytest.importorskip("dsart.transport.send_mmsg")
    vm = wu.value_map("gridded", 2, 8)
    rule = q.RowRule.for_format(8, 2, warmup_blocks=1)
    streams = list(range(8))                                         # chgroups 0..3
    spec = rr.RingV4Spec(
        dms=(0, 40), t_int_factor=(16, 32), t_buf=(256, 128), reorder_rows=8,
        n_stream=vm.n_streams, n_sub=2, bits_main=8, n_grid=256,
        s_main=float(rule.s_main_f32), s_heavy=float(rule.s_heavy_f32),
        stream_n_values=tuple(int(x) for x in vm.n_values),
        stream_n_heavy=tuple(int(x) for x in vm.n_heavy), min_retention_s=0.0)
    rx = Rx(n_ports=4, n_threads=2, spec=spec)
    tx = sm.MmsgSender([("127.0.0.1", p, 20e6) for p in rx.ports])   # 20 Mb/s per port
    try:
        for s in streams:
            rx.rx.set_expected_pattern_id4(s, int(vm.stream_pattern_ids[s]))
        rx.rx.start()
        tx.start()
        S = 24
        blocks = wu.stream_blocks(vm, streams, rule=rule, n_samples=S, seed=8)
        r = rx.reader(access="pread")
        for d, tif in ((0, 16), (40, 32)):
            seq0 = 1 << 20
            for s, bc in blocks.items():
                tx.set_route(d, s // 2)                              # chgroup's port
                dgs = wu.datagrams_for(vm, bc, stream=s, dm=d, seq_first=seq0,
                                       t_int_factor=tif, rule=rule)
                assert all(len(x) <= 8972 for x in dgs)
                assert tx.submit(dgs) == len(dgs)
                assert tx.flush(5.0)
            win = r.wait_window(d, seq0, S, min_streams=len(streams), grace_s=0.0, timeout_s=5.0)
            assert win.status == "fan_in"
            for s, bc in blocks.items():
                assert (win.validity[s] == rr.RV_PRESENT).all()
                sv = vm.streams[s]
                bps = f2.bytes_per_sample(sv.n_values, sv.n_heavy, 8, 16)
                ch, cm = f2.unpack_samples(np.ascontiguousarray(win.view[s, :, :bps]), S,
                                           sv.n_values, sv.n_heavy, 8, 16)
                np.testing.assert_array_equal(ch, bc.codes_heavy)
                np.testing.assert_array_equal(cm, bc.codes_main)
            assert (win.validity[8:] == rr.RV_MISSING).all()
        c = r.counters()["writer"]
        assert c["pattern_mismatch"] == 0 and c["port_mismatch"] == 0
        assert sum(x["full_drops"] + x["errors"] for x in tx.counters()["per_dest"]) == 0
        r.close()
    finally:
        tx.close()
        rx.close()
