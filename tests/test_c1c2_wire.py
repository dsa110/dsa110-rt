"""Round-trip + framing tests for the C1 ↔ C2 wire schema.

Covers the byte-level contract in ``docs/c1c2/C1C2_WIRE_SCHEMA.md``.
Sub-agents implementing the C1 emitter or the C2 receiver MUST keep
these tests passing.
"""

from __future__ import annotations

import pytest

from dsart.coinc import wire


# ---------------------------------------------------------------------------
# C1 batch
# ---------------------------------------------------------------------------


def _make_rows(n: int) -> list[wire.C1CandidateRow]:
    return [
        wire.C1CandidateRow(
            snr=8.5e0 + i,
            l_rad=1.2e-4 * (i + 1),
            m_rad=-1.4e-4 * (i + 1),
            l_pix=120 + i,
            m_pix=130 + i,
            dm_pc_cc=375.250 + i,
            dm_idx_global=50 + i,
            fine_dm_idx=i,
            event_specnum=100_000 + 16 * i,
            width_samples=1 << (i % 6),
            kernel_id="unit:d1:b4",
            flags=0,
        )
        for i in range(n)
    ]


def _make_header(n_candidates: int) -> wire.C1BatchHeader:
    return wire.build_header(
        cube_id=7,
        event_specnum_start=100_000,
        mjd_start=60781.123456789012,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=n_candidates,
    )


def test_c1_header_candidate_mjd_round_trips_specnum() -> None:
    header = _make_header(0)
    spn = header.event_specnum_start + 5 * header.sample_period_specnum
    mjd = header.candidate_mjd(spn)
    expected = header.mjd_start + 5 * header.sample_period_us / 1e6 / 86400.0
    assert abs(mjd - expected) < 1e-15
    # zero offset -> mjd_start exactly
    assert header.candidate_mjd(header.event_specnum_start) == header.mjd_start


@pytest.mark.parametrize("n_candidates", [0, 1, 5, 25])
def test_c1_batch_roundtrip(n_candidates: int) -> None:
    rows = _make_rows(n_candidates)
    header = _make_header(n_candidates)
    blob = wire.C1BatchEncoder.encode(header, rows)
    assert blob.startswith(b"# C1 ")
    assert blob.endswith(b"# END\n")

    lines = blob.decode("utf-8").rstrip("\n").split("\n")
    parsed = wire.parse_c1_batch(lines)
    assert parsed.header == header
    assert len(parsed.candidates) == n_candidates
    # exact equality is OK because we control the format strings; the
    # printed precision (%.6e for snr, %.6f for dm, %.9e for lm) is
    # enough that the round-trip preserves the test values bit-exact.
    for original, got in zip(rows, parsed.candidates):
        assert got.l_pix == original.l_pix
        assert got.m_pix == original.m_pix
        assert got.event_specnum == original.event_specnum
        assert got.width_samples == original.width_samples
        assert got.kernel_id == original.kernel_id
        assert got.dm_idx_global == original.dm_idx_global
        assert got.fine_dm_idx == original.fine_dm_idx
        assert got.flags == original.flags
        # floats: tight tolerance, not bit-exact due to printf rounding
        assert abs(got.snr - original.snr) < 1e-5 * abs(original.snr)
        assert abs(got.dm_pc_cc - original.dm_pc_cc) < 1e-5
        assert abs(got.l_rad - original.l_rad) <= 1e-12
        assert abs(got.m_rad - original.m_rad) <= 1e-12


def test_c1_batch_mismatched_count_raises() -> None:
    header = _make_header(3)
    rows = _make_rows(2)
    with pytest.raises(ValueError):
        wire.C1BatchEncoder.encode(header, rows)


def test_c1_parse_unknown_schema_version_raises() -> None:
    line = (
        "# C1 9 1 0 60781.00000000000 16 1048.576000 256 34 1 0 0"
    )
    with pytest.raises(wire.BadBatch, match="schema_version"):
        wire.parse_c1_batch([line, "# END"])


def test_c1_parse_truncated_batch() -> None:
    # Header says 3 candidates but only 1 row + END line
    rows = _make_rows(3)
    header = _make_header(3)
    blob = wire.C1BatchEncoder.encode(header, rows)
    lines = blob.decode("utf-8").rstrip("\n").split("\n")
    truncated = lines[:2] + ["# END"]
    with pytest.raises(wire.BadBatch, match="truncated"):
        wire.parse_c1_batch(truncated)


def test_c1_parse_missing_end_marker() -> None:
    rows = _make_rows(2)
    header = _make_header(2)
    blob = wire.C1BatchEncoder.encode(header, rows)
    lines = blob.decode("utf-8").rstrip("\n").split("\n")
    bad = lines[:-1]  # drop the END line
    with pytest.raises(wire.BadBatch, match="missing END"):
        wire.parse_c1_batch(bad)


def test_c1_parse_bad_end_marker() -> None:
    rows = _make_rows(1)
    header = _make_header(1)
    blob = wire.C1BatchEncoder.encode(header, rows)
    lines = blob.decode("utf-8").rstrip("\n").split("\n")
    lines[-1] = "# NOT_END"
    with pytest.raises(wire.BadBatch, match="END"):
        wire.parse_c1_batch(lines)


def test_c1_parse_bad_field_count_in_row() -> None:
    rows = _make_rows(1)
    header = _make_header(1)
    blob = wire.C1BatchEncoder.encode(header, rows)
    lines = blob.decode("utf-8").rstrip("\n").split("\n")
    # mangle the row by dropping the last field
    lines[1] = " ".join(lines[1].split()[:-1])
    with pytest.raises(wire.BadBatch, match="12 fields"):
        wire.parse_c1_batch(lines)


# ---------------------------------------------------------------------------
# C2 trigger packet
# ---------------------------------------------------------------------------


def test_c2_trigger_roundtrip() -> None:
    pkt = wire.C2TriggerPacket(
        event_name="260521abcd",
        event_specnum=987_654_321,
        mjd_target=60781.5,
        trigger_class_id=1,
        flags=wire.C2_TRIGGER_FLAG_DUMP_CUBE,
    )
    blob = wire.encode_c2_trigger(pkt)
    assert len(blob) == wire.C2_TRIGGER_PACKET_SIZE == 64
    got = wire.decode_c2_trigger(blob)
    assert got == pkt


def test_c2_trigger_magic_is_dsrt_ascii() -> None:
    pkt = wire.C2TriggerPacket(
        event_name="x", event_specnum=0, mjd_target=0.0
    )
    blob = wire.encode_c2_trigger(pkt)
    # Little-endian uint32 == ASCII "DSRT" when read big-endian:
    # first byte 'D', then 'S', 'R', 'T'.
    assert blob[:4] == b"DSRT"


def test_c2_trigger_event_name_too_long_raises() -> None:
    with pytest.raises(ValueError):
        wire.encode_c2_trigger(
            wire.C2TriggerPacket(
                event_name="x" * 17,
                event_specnum=0,
                mjd_target=0.0,
            )
        )


def test_c2_trigger_wrong_size_raises() -> None:
    with pytest.raises(wire.BadBatch, match="wrong size"):
        wire.decode_c2_trigger(b"\x00" * 32)


def test_c2_trigger_bad_magic_raises() -> None:
    pkt = wire.C2TriggerPacket(
        event_name="x", event_specnum=0, mjd_target=0.0
    )
    blob = bytearray(wire.encode_c2_trigger(pkt))
    blob[0] ^= 0xff
    with pytest.raises(wire.BadBatch, match="magic"):
        wire.decode_c2_trigger(bytes(blob))
