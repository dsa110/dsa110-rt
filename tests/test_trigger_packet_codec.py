"""Tests for ``dsart.trigger.ndjson_codec`` (M5 chunk 4).

Plan §3 lines 370-398 + §4.4 line 1670 wire format. Coverage:

  * Round-trip: ``decode_packet(encode_packet(p)) == p`` for valid
    packets across schema-edge cases.
  * Round-trip for ``TriggerAck`` both stages (accepted accept=True,
    accepted accept=False with reason+dup_of, completed with
    filterbank_paths).
  * NDJSON splitting: a partial buffer returns (lines, remainder)
    correctly; a buffer with no newline returns ([], buf).
  * Field ordering: ``v`` is first in the JSON object on the wire.
  * None-valued ack fields are dropped from the wire JSON
    (e.g. ``filterbank_paths=None`` doesn't appear).
  * Decoder rejects malformed JSON, non-object JSON, missing schema
    fields.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.common.contracts import TriggerAck, TriggerPacket  # noqa: E402
from dsart.trigger.ndjson_codec import (  # noqa: E402
    decode_ack,
    decode_packet,
    encode_ack,
    encode_packet,
    iter_lines,
    split_ndjson_buffer,
)


def _packet(**overrides) -> TriggerPacket:
    base = dict(
        trigger_id="s2-g1-0000000007",
        search_node_id=2,
        emit_utc_ns=1872345678901234567,
        event_specnum=12345678,
        event_utc_ns=1872345677000000000,
        l=0.012,
        m=-0.034,
        dm_fine=524.6,
        dm_idx=87,
        fine_dm_trial=87,
        width_samples=4,
        kernel_id="psf:d3:b16",
        snr=9.7,
        actions={"voltage_dump": True, "filterbank": True, "n_beams": 1},
        priority="normal",
        src_name="auto_20260430_142511_b3",
    )
    base.update(overrides)
    return TriggerPacket(**base)


def test_packet_round_trip_full() -> None:
    p = _packet()
    wire = encode_packet(p)
    assert wire.endswith(b"\n")
    p2 = decode_packet(wire.rstrip(b"\n"))
    assert p == p2


def test_packet_round_trip_with_pre_post_blocks() -> None:
    p = _packet(n_pre_blocks=15, n_post_blocks=10)
    wire = encode_packet(p)
    p2 = decode_packet(wire.rstrip(b"\n"))
    assert p == p2
    assert p2.n_pre_blocks == 15
    assert p2.n_post_blocks == 10


def test_packet_v_is_first_field_in_json() -> None:
    p = _packet()
    wire = encode_packet(p).rstrip(b"\n").decode("utf-8")
    obj = json.loads(wire)
    # JSON dicts in Python preserve insertion order; the first key
    # should be 'v' per the wire-inspection-ergonomics rule.
    assert next(iter(obj.keys())) == "v"
    assert obj["v"] == 1


def test_packet_n_pre_post_none_dropped_from_wire() -> None:
    """When n_pre_blocks / n_post_blocks are None, they must NOT appear
    on the wire (corr listener uses its config defaults per plan §3
    line 388)."""
    p = _packet()  # defaults: None, None
    wire = encode_packet(p).rstrip(b"\n").decode("utf-8")
    obj = json.loads(wire)
    assert "n_pre_blocks" not in obj
    assert "n_post_blocks" not in obj


def test_packet_decoder_rejects_malformed_json() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        decode_packet(b'{"trigger_id": "x"')


def test_packet_decoder_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        decode_packet(b"[1, 2, 3]")


def test_packet_decoder_rejects_missing_field() -> None:
    """Schema validation is delegated to the M1 dataclass __post_init__;
    a missing required field surfaces as a TypeError from the
    dataclass constructor."""
    with pytest.raises(TypeError):
        decode_packet(b'{"v": 1, "trigger_id": "s0-g0-0000000001"}')


# ---------------------------------------------------------------------------
# TriggerAck round-trip
# ---------------------------------------------------------------------------


def _ack_accepted(accepted: bool = True, reason: str | None = None,
                  dup_of: str | None = None) -> TriggerAck:
    return TriggerAck(
        trigger_id="s2-g1-0000000007",
        stage="accepted",
        ack_utc_ns=1872345679000000000,
        accepted=accepted,
        reason=reason,
        queue_depth=3,
        dup_of=dup_of,
    )


def _ack_completed() -> TriggerAck:
    return TriggerAck(
        trigger_id="s2-g1-0000000007",
        stage="completed",
        ack_utc_ns=1872345679999000000,
        voltage_dump_path="/home/ubuntu/data/fl_12345678.out",
        filterbank_paths=("/home/ubuntu/data/auto_b0.fil",
                          "/home/ubuntu/data/auto_b1.fil"),
        dump_completion_utc_ns=1872345679999000000,
        dump_duration_ms=312,
    )


def test_ack_accepted_round_trip() -> None:
    a = _ack_accepted()
    a2 = decode_ack(encode_ack(a).rstrip(b"\n"))
    assert a == a2


def test_ack_accepted_rejected_with_reason_round_trip() -> None:
    a = _ack_accepted(accepted=False, reason="ratelimit")
    a2 = decode_ack(encode_ack(a).rstrip(b"\n"))
    assert a == a2


def test_ack_dup_round_trip_carries_dup_of() -> None:
    a = _ack_accepted(accepted=False, reason="dup", dup_of="s1-g0-0000000099")
    a2 = decode_ack(encode_ack(a).rstrip(b"\n"))
    assert a == a2
    assert a2.dup_of == "s1-g0-0000000099"


def test_ack_completed_round_trip_with_filterbank_tuple() -> None:
    a = _ack_completed()
    wire = encode_ack(a).rstrip(b"\n")
    a2 = decode_ack(wire)
    assert a == a2
    assert isinstance(a2.filterbank_paths, tuple)


def test_ack_none_fields_dropped_from_wire() -> None:
    """A stage='accepted' ack should not carry stage='completed' fields
    on the wire."""
    a = _ack_accepted()
    wire = encode_ack(a).rstrip(b"\n").decode("utf-8")
    obj = json.loads(wire)
    assert "voltage_dump_path" not in obj
    assert "filterbank_paths" not in obj
    assert "dump_completion_utc_ns" not in obj
    assert "dump_duration_ms" not in obj


def test_ack_v_is_first_field_in_json() -> None:
    a = _ack_accepted()
    wire = encode_ack(a).rstrip(b"\n").decode("utf-8")
    obj = json.loads(wire)
    assert next(iter(obj.keys())) == "v"


# ---------------------------------------------------------------------------
# NDJSON splitting
# ---------------------------------------------------------------------------


def test_split_ndjson_buffer_complete_lines() -> None:
    buf = b'{"a":1}\n{"b":2}\n'
    lines, remainder = split_ndjson_buffer(buf)
    assert lines == [b'{"a":1}', b'{"b":2}']
    assert remainder == b""


def test_split_ndjson_buffer_partial_remainder() -> None:
    buf = b'{"a":1}\n{"b":'
    lines, remainder = split_ndjson_buffer(buf)
    assert lines == [b'{"a":1}']
    assert remainder == b'{"b":'


def test_split_ndjson_buffer_no_newline() -> None:
    buf = b'{"a":1'
    lines, remainder = split_ndjson_buffer(buf)
    assert lines == []
    assert remainder == buf


def test_split_ndjson_buffer_empty() -> None:
    lines, remainder = split_ndjson_buffer(b"")
    assert lines == []
    assert remainder == b""


def test_iter_lines_matches_split_buffer() -> None:
    buf = b'{"a":1}\n{"b":2}\n{"c":3}\n'
    assert list(iter_lines(buf)) == [b'{"a":1}', b'{"b":2}', b'{"c":3}']
