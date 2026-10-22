"""send_mmsg.c sender: routing, pacing, drops, counters (localhost, low rate)."""
from __future__ import annotations

import importlib.util
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sm = pytest.importorskip("dsart.transport.send_mmsg")
try:
    sm._lib()
except RuntimeError as exc:                       # pragma: no cover
    pytest.skip(str(exc), allow_module_level=True)

_spec = importlib.util.spec_from_file_location("wst_util", Path(__file__).with_name("wst_util.py"))
wu = importlib.util.module_from_spec(_spec)
sys.modules["wst_util"] = wu
_spec.loader.exec_module(wu)


def _rx_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2.0)
    return s


def _drain(s, n):
    out = []
    for _ in range(n):
        out.append(s.recv(65536))
    return out


def test_routing_order_and_counters():
    a, b = _rx_socket(), _rx_socket()
    tx = sm.MmsgSender([("127.0.0.1", a.getsockname()[1], 0.0),
                        ("127.0.0.1", b.getsockname()[1], 0.0)], routes={3: 0, 11: 1})
    try:
        dgs = [wu.synthetic_frame(dm=(3 if i % 3 else 11), seq_first=2 * i, k=2) for i in range(60)]
        dgs.append(wu.synthetic_frame(dm=5))                          # unrouted
        dgs.append(b"\x00" * 10)                                      # shorter than a dm_idx
        tx.start()
        assert tx.submit(dgs) == 60
        assert tx.flush(5.0)
        ra = _drain(a, 40)
        rb = _drain(b, 20)
        assert ra == [d for i, d in enumerate(dgs[:60]) if i % 3]     # in order, intact
        assert rb == [d for i, d in enumerate(dgs[:60]) if not i % 3]
        c = tx.counters()
        assert (c["submitted"], c["accepted"], c["unrouted"], c["bad_length"]) == (62, 60, 1, 1)
        assert [d["sent"] for d in c["per_dest"]] == [40, 20]
        assert c["per_dest"][0]["bytes"] == sum(len(d) for d in ra)
        assert all(d["full_drops"] == 0 and d["errors"] == 0 and d["depth"] == 0
                   for d in c["per_dest"])
        assert tx.route(3) == 0 and tx.route(5) == -1
        tx.set_route(5, 1)
        assert tx.submit([dgs[60]]) == 1 and tx.flush(2.0)
        assert b.recv(65536) == dgs[60]
    finally:
        tx.close()
        a.close()
        b.close()


def test_buffer_form_and_jumbo_datagram():
    a = _rx_socket()
    tx = sm.MmsgSender([("127.0.0.1", a.getsockname()[1], 0.0)], routes={0: 0})
    try:
        big = wu.synthetic_frame(dm=0, k=1, n_values=8900, n_heavy=0)       # 72 + 8900 B
        assert len(big) == 8972
        buf, offs, lens = sm.pack_datagrams([big, big[:100]])
        tx.start()
        assert tx.submit((buf, offs, lens)) == 2 and tx.flush(2.0)
        assert a.recv(65536) == big and a.recv(65536) == big[:100]
    finally:
        tx.close()
        a.close()


def test_pacing_holds_the_rate():
    """1 MB/s (8 Mb/s) with a 16 KB bucket: 100 x 1 KB takes >= ~84 ms."""
    a = _rx_socket()
    tx = sm.MmsgSender([("127.0.0.1", a.getsockname()[1], 8e6)], routes={0: 0},
                       burst_bytes=16 << 10)
    try:
        dgs = [wu.synthetic_frame(dm=0, seq_first=2 * i, k=1, n_values=1000 - 72 - 2,
                                  n_heavy=0) for i in range(100)]
        assert all(len(d) == 998 for d in dgs)
        tx.start()
        t0 = time.monotonic()
        assert tx.submit(dgs) == 100
        assert tx.flush(10.0)
        dt = time.monotonic() - t0
        assert 0.07 <= dt <= 2.0, dt
        assert len(_drain(a, 100)) == 100
    finally:
        tx.close()
        a.close()


def test_full_queue_drops_and_bounded_wait():
    a = _rx_socket()
    tx = sm.MmsgSender([("127.0.0.1", a.getsockname()[1], 0.0)], routes={0: 0}, queue_slots=4)
    try:
        dgs = [wu.synthetic_frame(dm=0, seq_first=2 * i) for i in range(10)]
        assert tx.submit(dgs) == 4                               # sender not running yet
        t0 = time.monotonic()
        assert tx.submit(dgs[:1], block_timeout_s=0.1) == 0      # waits, then drops
        assert time.monotonic() - t0 >= 0.09
        c = tx.counters()["per_dest"][0]
        assert c["full_drops"] == 7 and c["max_depth"] == 4 and c["depth"] == 4
        tx.start()
        assert tx.flush(2.0) and len(_drain(a, 4)) == 4
    finally:
        tx.close()
        a.close()


def test_absent_receiver_never_blocks_other_destination():
    a = _rx_socket()
    dead = _rx_socket()
    dead_port = dead.getsockname()[1]
    dead.close()                                                  # nobody listens there
    tx = sm.MmsgSender([("127.0.0.1", dead_port, 0.0), ("127.0.0.1", a.getsockname()[1], 0.0)],
                       routes={1: 0, 2: 1})
    try:
        tx.start()
        dgs = [wu.synthetic_frame(dm=1 + (i % 2), seq_first=2 * i) for i in range(40)]
        assert tx.submit(dgs) == 40 and tx.flush(3.0)
        assert len(_drain(a, 20)) == 20
        c = tx.counters()["per_dest"]
        assert c[1]["sent"] == 20 and c[0]["depth"] == 0
    finally:
        tx.close()
        a.close()


def test_routes_for_owner_halves():
    owner_half = np.arange(47) % 8
    r = sm.routes_for_owner_halves(range(47), owner_half)
    assert r[0] == 0 and r[1] == 0 and r[2] == 1 and r[7] == 3 and r[46] == 3
