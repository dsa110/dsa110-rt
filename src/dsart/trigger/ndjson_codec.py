"""Newline-delimited JSON codec for ``TriggerPacket`` / ``TriggerAck``
(plan §3 line 370-398; §4.4 lines 1669+).

Wire format: one JSON object per line, UTF-8 encoded, terminated by
``b"\\n"``. The corr listener and the search emitter share this codec
so byte-identical wire layout is enforced by the round-trip tests.

JSON serialisation rules (locked):

  - ``None``-valued fields are **dropped** from the JSON object per
    plan §4.5 line 1718 (this is what the M1 ``TriggerAck.__post_init__``
    expects on stage='accepted' / stage='completed' fields that don't
    apply to the other stage).
  - ``filterbank_paths`` is encoded as a JSON list (the dataclass holds
    a tuple for frozen-friendliness; ``list(t)`` round-trips on decode
    back into a tuple).
  - ``v`` is always the int ``TRIGGER_SCHEMA_VERSION`` (= 1 in v1).
  - Ordering: emitter writes the ``v`` field first to make wire
    inspection easier; the rest of the field order matches the
    dataclass declaration (deterministic). Decoder is order-tolerant.

The codec is **synchronous** (no asyncio); the async emitter / listener
wrap their own bytes-buffer parsing around it.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Iterable, Iterator

from ..common.contracts import TriggerAck, TriggerPacket

__all__ = [
    "encode_packet",
    "encode_ack",
    "decode_packet",
    "decode_ack",
    "iter_lines",
    "split_ndjson_buffer",
]


def _drop_nones(d: dict) -> dict:
    """Strip ``None``-valued fields. Recursively for nested dicts (the
    only nested dict in v1 is ``actions``, which doesn't carry None
    values, but we recurse defensively)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = _drop_nones(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _ordered_dict_for_packet(packet: TriggerPacket) -> dict:
    """Place ``v`` first; preserve dataclass field order otherwise."""
    raw = dataclasses.asdict(packet)
    # Pull ``v`` to the front for wire-inspection ergonomics.
    ordered: dict[str, Any] = {"v": raw.pop("v")}
    ordered.update(raw)
    return _drop_nones(ordered)


def _ordered_dict_for_ack(ack: TriggerAck) -> dict:
    raw = dataclasses.asdict(ack)
    ordered: dict[str, Any] = {"v": raw.pop("v")}
    ordered.update(raw)
    return _drop_nones(ordered)


def encode_packet(packet: TriggerPacket) -> bytes:
    """Serialise a TriggerPacket to a single ``\\n``-terminated UTF-8
    JSON line."""
    return (json.dumps(_ordered_dict_for_packet(packet)) + "\n").encode("utf-8")


def encode_ack(ack: TriggerAck) -> bytes:
    return (json.dumps(_ordered_dict_for_ack(ack)) + "\n").encode("utf-8")


def decode_packet(line: bytes | str) -> TriggerPacket:
    """Parse a single NDJSON line into a TriggerPacket. Raises
    ``ValueError`` (with the offending bytes / parse error) on malformed
    JSON or schema mismatch (the M1 ``__post_init__`` validators enforce
    field semantics under ``DSART_TEST=1``).
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}; line={line[:200]!r}") from e
    if not isinstance(d, dict):
        raise ValueError(f"expected JSON object, got {type(d).__name__}")
    return TriggerPacket(**d)


def decode_ack(line: bytes | str) -> TriggerAck:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}; line={line[:200]!r}") from e
    if not isinstance(d, dict):
        raise ValueError(f"expected JSON object, got {type(d).__name__}")
    if "filterbank_paths" in d and isinstance(d["filterbank_paths"], list):
        d["filterbank_paths"] = tuple(d["filterbank_paths"])
    return TriggerAck(**d)


def iter_lines(buf: bytes) -> Iterator[bytes]:
    """Iterate complete ``\\n``-terminated lines in ``buf``; yields
    individual lines WITHOUT the trailing newline."""
    start = 0
    while True:
        nl = buf.find(b"\n", start)
        if nl < 0:
            return
        yield buf[start:nl]
        start = nl + 1


def split_ndjson_buffer(buf: bytes) -> tuple[list[bytes], bytes]:
    """Split a partial NDJSON byte buffer into (complete_lines, remainder).

    Used by the emitter / listener TCP read paths: when ``recv()``
    returns a partial chunk, append to a per-connection buffer, then
    call this to extract any complete records and keep the (partial)
    last record as the new buffer head.
    """
    lines: list[bytes] = []
    start = 0
    while True:
        nl = buf.find(b"\n", start)
        if nl < 0:
            break
        lines.append(buf[start:nl])
        start = nl + 1
    return lines, buf[start:]
