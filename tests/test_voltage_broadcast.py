"""Tests for :class:`dsart.coinc.broadcast.VoltageBroadcaster`."""

from __future__ import annotations

import socket
from typing import Dict

import pytest

from dsart.coinc import wire
from dsart.coinc.broadcast import DEFAULT_VOLTAGE_PORT, VoltageBroadcaster


def _listener() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(0.5)
    return s, s.getsockname()[1]


def test_default_port_constant() -> None:
    assert DEFAULT_VOLTAGE_PORT == 11229


def test_empty_hosts_raises() -> None:
    with pytest.raises(ValueError):
        VoltageBroadcaster({})


def test_one_packet_per_corr_node_with_voltage_flag() -> None:
    sock, port = _listener()
    try:
        # Three corr nodes all pointed at the same loopback listener so a
        # single socket sees all three datagrams.
        hosts = {3: "127.0.0.1", 4: "127.0.0.1", 5: "127.0.0.1"}
        bc = VoltageBroadcaster(hosts, port=port)
        try:
            res = bc.broadcast(
                event_name="260610hulw",
                event_specnum=555,
                mjd_target=60800.0,
                trigger_class_id=2,
            )
            assert res == {3: True, 4: True, 5: True}
            seen = 0
            for _ in range(3):
                blob, _addr = sock.recvfrom(4096)
                pkt = wire.decode_c2_trigger(blob)
                assert pkt.flags & wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE
                assert pkt.event_name == "260610hulw"
                assert pkt.event_specnum == 555
                seen += 1
            assert seen == 3
        finally:
            bc.close()
    finally:
        sock.close()


def test_delete_sentinel_specnum_zero() -> None:
    """C3's REJECT delete path sends specnum=0 with the voltage flag."""
    sock, port = _listener()
    try:
        bc = VoltageBroadcaster({7: "127.0.0.1"}, port=port)
        try:
            bc.broadcast(event_name="deadbeef", event_specnum=0,
                         mjd_target=0.0)
            blob, _ = sock.recvfrom(4096)
            pkt = wire.decode_c2_trigger(blob)
            assert pkt.event_specnum == 0
            assert pkt.flags & wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE
        finally:
            bc.close()
    finally:
        sock.close()


class _FailSock:
    def sendto(self, blob: bytes, addr) -> int:
        raise OSError("nope")

    def setsockopt(self, *a, **k) -> None:
        pass

    def close(self) -> None:
        pass


def test_send_failure_records_false() -> None:
    bc = VoltageBroadcaster({3: "127.0.0.1", 4: "127.0.0.1"},
                            port=11229, sock=_FailSock())
    try:
        res: Dict[int, bool] = bc.broadcast(
            event_name="x", event_specnum=1, mjd_target=0.0,
        )
        assert res == {3: False, 4: False}
    finally:
        bc.close()
