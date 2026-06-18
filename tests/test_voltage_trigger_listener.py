"""Tests for :mod:`dsart.dump.voltage_trigger_listener` dispatch logic.

Drives ``_on_datagram`` directly with encoded packets (no sockets) so the
routing + counters are deterministic.
"""

from __future__ import annotations

from typing import List, Tuple

from dsart.coinc import wire
from dsart.dump.voltage_trigger_listener import (
    VoltageTriggerListener,
    VoltageTriggerListenerConfig,
)


def _listener(on_dump=None, on_delete=None) -> VoltageTriggerListener:
    return VoltageTriggerListener(
        config=VoltageTriggerListenerConfig(
            bind_host="127.0.0.1", bind_port=11229, cn_id=3, chgroup=0,
        ),
        on_dump=on_dump or (lambda name, spec: True),
        on_delete=on_delete,
    )


def _pkt(*, specnum: int, flags: int, name: str = "260610hulw") -> bytes:
    return wire.encode_c2_trigger(wire.C2TriggerPacket(
        event_name=name, event_specnum=specnum, mjd_target=60800.0,
        flags=flags,
    ))


def test_dump_dispatched_on_voltage_flag() -> None:
    seen: List[Tuple[str, int]] = []
    lis = _listener(on_dump=lambda n, s: (seen.append((n, s)) or True))
    lis._on_datagram(
        _pkt(specnum=777, flags=wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE), None,
    )
    assert seen == [("260610hulw", 777)]
    assert lis.mon["enqueued"] == 1
    assert lis.mon["dump_flagged"] == 1


def test_delete_on_specnum_zero() -> None:
    deletes: List[str] = []
    lis = _listener(on_delete=lambda n: (deletes.append(n) or True))
    lis._on_datagram(
        _pkt(specnum=0, flags=wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE), None,
    )
    assert deletes == ["260610hulw"]
    assert lis.mon["deletes"] == 1
    assert lis.mon["enqueued"] == 0


def test_cube_only_packet_ignored() -> None:
    seen: List[Tuple[str, int]] = []
    lis = _listener(on_dump=lambda n, s: (seen.append((n, s)) or True))
    lis._on_datagram(
        _pkt(specnum=5, flags=wire.C2_TRIGGER_FLAG_DUMP_CUBE), None,
    )
    assert seen == []
    assert lis.mon["not_flagged"] == 1


def test_queue_full_counts() -> None:
    lis = _listener(on_dump=lambda n, s: False)   # queue always full
    lis._on_datagram(
        _pkt(specnum=9, flags=wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE), None,
    )
    assert lis.mon["queue_full"] == 1
    assert lis.mon["enqueued"] == 0


def test_bad_size_and_magic() -> None:
    lis = _listener()
    lis._on_datagram(b"\x00" * 10, None)                 # wrong size
    assert lis.mon["bad_size"] == 1
    bad = bytearray(_pkt(specnum=1, flags=wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE))
    bad[:4] = b"\xde\xad\xbe\xef"                         # corrupt magic
    lis._on_datagram(bytes(bad), None)
    assert lis.mon["bad_magic"] == 1
