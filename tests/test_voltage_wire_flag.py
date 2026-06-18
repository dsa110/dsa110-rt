"""Tests for the DUMP_VOLTAGE wire flag (additive to the locked schema)."""

from __future__ import annotations

from dsart.coinc import wire


def test_dump_voltage_flag_is_distinct_bit() -> None:
    assert wire.C2_TRIGGER_FLAG_DUMP_CUBE == 1 << 0
    assert wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE == 1 << 1
    # Distinct, non-overlapping bits so a packet can carry both.
    assert (
        wire.C2_TRIGGER_FLAG_DUMP_CUBE & wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE
    ) == 0


def test_flags_round_trip_voltage() -> None:
    pkt = wire.C2TriggerPacket(
        event_name="260618paiu",
        event_specnum=123_456_789,
        mjd_target=60781.25,
        trigger_class_id=7,
        flags=wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE,
    )
    got = wire.decode_c2_trigger(wire.encode_c2_trigger(pkt))
    assert got.flags == wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE
    assert got.event_name == "260618paiu"
    assert got.event_specnum == 123_456_789


def test_flags_round_trip_both_bits() -> None:
    both = (
        wire.C2_TRIGGER_FLAG_DUMP_CUBE | wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE
    )
    pkt = wire.C2TriggerPacket(
        event_name="x", event_specnum=1, mjd_target=0.0, flags=both,
    )
    got = wire.decode_c2_trigger(wire.encode_c2_trigger(pkt))
    assert got.flags & wire.C2_TRIGGER_FLAG_DUMP_CUBE
    assert got.flags & wire.C2_TRIGGER_FLAG_DUMP_VOLTAGE


def test_packet_size_unchanged() -> None:
    """The flag is additive — the 64-byte struct must not have grown."""
    assert wire.C2_TRIGGER_PACKET_SIZE == 64
    blob = wire.encode_c2_trigger(
        wire.C2TriggerPacket(event_name="x", event_specnum=0, mjd_target=0.0)
    )
    assert len(blob) == 64
