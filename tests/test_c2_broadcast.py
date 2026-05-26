"""Tests for :mod:`dsart.coinc.broadcast` (UDP trigger fan-out)."""

from __future__ import annotations

import socket
from typing import Dict, List, Tuple

import pytest

from dsart.coinc import wire
from dsart.coinc.broadcast import GPU_HALVES, TriggerBroadcaster


def _make_listener() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(0.5)
    return s, s.getsockname()[1]


def test_broadcast_sends_to_every_search_node_and_half() -> None:
    """A single broadcast() call must hit every (sid, gpu_half) pair.

    We bind eight listening sockets on 127.0.0.1 (two per search node)
    and verify each receives exactly one well-formed trigger packet.
    """
    listeners: Dict[Tuple[int, int], socket.socket] = {}
    hosts: Dict[int, str] = {}
    # We can't pick the destination port per-(sid, g) because the
    # broadcaster's contract is port_base + g, fixed across sids. So
    # we choose distinct loopback addresses (127.0.0.1, 127.0.0.2, ...)
    # for each search node and a shared port_base; each address can
    # then bind both halves cleanly.
    # Pick a port_base above the OS ephemeral range to avoid collision.
    # We probe upward until we find one where both halves bind on every
    # address; if the harness can't find one, skip.
    port_base = None
    socks: Dict[Tuple[int, int], socket.socket] = {}
    for candidate in range(30000, 30100):
        # Try to bind two halves across 4 sids.
        socks = {}
        try:
            for sid_idx, sid in enumerate((1, 2, 9, 13)):
                addr = f"127.{sid_idx + 1}.0.1"
                for g in GPU_HALVES:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.bind((addr, candidate + g))
                    s.settimeout(0.5)
                    socks[(sid, g)] = s
            port_base = candidate
            break
        except OSError:
            for s in socks.values():
                s.close()
            socks = {}
    if port_base is None:
        pytest.skip("could not bind 8 UDP listeners on loopback aliases")

    try:
        for sid_idx, sid in enumerate((1, 2, 9, 13)):
            hosts[sid] = f"127.{sid_idx + 1}.0.1"
        bcast = TriggerBroadcaster(hosts, port_base=port_base)
        try:
            result = bcast.broadcast(
                event_name="260521abcd",
                event_specnum=987_654_321,
                mjd_target=60781.5,
                trigger_class_id=1,
            )
            assert all(result.values())
            assert set(result.keys()) == {
                (sid, g) for sid in hosts for g in GPU_HALVES
            }
            for (sid, g), sock in socks.items():
                blob, _ = sock.recvfrom(4096)
                assert len(blob) == wire.C2_TRIGGER_PACKET_SIZE
                got = wire.decode_c2_trigger(blob)
                assert got.event_name == "260521abcd"
                assert got.event_specnum == 987_654_321
                assert got.mjd_target == pytest.approx(60781.5)
                assert got.trigger_class_id == 1
        finally:
            bcast.close()
    finally:
        for s in socks.values():
            s.close()


def test_broadcast_destination_port_is_port_base_plus_half() -> None:
    """Single listener bound on (127.0.0.1, port_base+1) receives only
    the gpu_half=1 packet, not the half=0 one."""
    g1_sock, _ = _make_listener()
    g1_port = g1_sock.getsockname()[1]
    # Pick a port_base such that port_base + 1 == g1_port and port_base
    # is unused.
    port_base = g1_port - 1
    try:
        hosts = {1: "127.0.0.1"}
        bcast = TriggerBroadcaster(hosts, port_base=port_base)
        try:
            res = bcast.broadcast(
                event_name="x", event_specnum=1, mjd_target=0.0,
            )
            # Both half=0 and half=1 should have been attempted; half=1
            # MUST land on our socket.
            assert res[(1, 1)] is True
            blob, _ = g1_sock.recvfrom(4096)
            assert wire.decode_c2_trigger(blob).event_specnum == 1
        finally:
            bcast.close()
    finally:
        g1_sock.close()


def test_broadcast_empty_hosts_raises() -> None:
    with pytest.raises(ValueError):
        TriggerBroadcaster({})


def test_broadcast_sendto_failure_records_false() -> None:
    """If the underlying socket raises OSError, the per-dest result is
    False (rather than the exception propagating)."""
    bcast = TriggerBroadcaster(
        {1: "127.0.0.1"}, port_base=11227,
        sock=_AlwaysFailingSock(),
    )
    try:
        res = bcast.broadcast(
            event_name="x", event_specnum=0, mjd_target=0.0,
        )
        assert res == {(1, 0): False, (1, 1): False}
    finally:
        bcast.close()


def test_broadcast_returns_bytewise_correct_packet() -> None:
    """Use broadcast_raw to verify the encoded payload matches the
    direct encoder output."""
    sock, port = _make_listener()
    try:
        bcast = TriggerBroadcaster(
            {1: "127.0.0.1"}, port_base=port - 0,  # half=0 → port = port
        )
        try:
            pkt = wire.C2TriggerPacket(
                event_name="x", event_specnum=42, mjd_target=60781.0,
            )
            expected = wire.encode_c2_trigger(pkt)
            bcast.broadcast_raw(expected)
            got_blob, _ = sock.recvfrom(4096)
            assert got_blob == expected
        finally:
            bcast.close()
    finally:
        sock.close()


class _AlwaysFailingSock:
    """Stub socket that raises OSError on every sendto."""

    def sendto(self, blob: bytes, addr) -> int:
        raise OSError("nope")

    def close(self) -> None:
        pass

    def setsockopt(self, *args, **kwargs) -> None:
        pass
