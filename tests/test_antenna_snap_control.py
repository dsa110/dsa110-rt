"""Tests for the Control-tab antenna/SNAP surfaces.

The behaviours worth pinning are the ones where getting it wrong sends a bad
command to real hardware:

* an empty antenna box must become the single broadcast key ``/cmd/ant/0``,
  never a fan-out (hwmc has every antenna watching that key);
* an empty SNAP box must fan out, because SNAPs have no broadcast key;
* ``halt`` must be a single-key dict, or hwmc calls ``_halt(val)`` and raises;
* elevation must be bounds-checked here, because hwmc does not check at all;
* ``armed_mjd`` 55000.0 is a sentinel, not a timestamp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools" / "dashboard" / "dsa_monitor")
)

import antenna_snap_control as asc  # noqa: E402


class FakeStore:
    def __init__(self, data: dict | None = None):
        self.writes: list[tuple[str, dict]] = []
        self.data = data or {}

    def put_dict(self, key, payload):
        self.writes.append((key, payload))

    def get_dict(self, key):
        if key not in self.data:
            return None
        val = self.data[key]
        if isinstance(val, Exception):
            raise val
        return val


# --- selection parsing -----------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", []),
        ("   ", []),
        (None, []),
        ("24", [24]),
        ("1,5,7", [1, 5, 7]),
        ("1 5 7", [1, 5, 7]),
        ("7-9", [7, 8, 9]),
        ("3, 7-9, 3", [3, 7, 8, 9]),          # de-duplicated + sorted
    ],
)
def test_parse_antennas_accepts_operator_forms(raw, expected):
    assert asc.parse_antennas(raw) == expected


@pytest.mark.parametrize("raw", ["0", "118", "9-3", "abc", "1,,x", "-4", "1.5"])
def test_parse_antennas_rejects_bad_input(raw):
    with pytest.raises(asc.SelectionError):
        asc.parse_antennas(raw)


@pytest.mark.parametrize("raw", ["0", "33", "40"])
def test_parse_snaps_enforces_1_to_32(raw):
    with pytest.raises(asc.SelectionError):
        asc.parse_snaps(raw)


# --- elevation guard -------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [("0", 0.0), ("143", 143.0), (" 90.5 ", 90.5)])
def test_validate_elevation_accepts_in_range(raw, expected):
    assert asc.validate_elevation(raw) == expected


@pytest.mark.parametrize("raw", ["-1", "143.1", "1000", "", None, "abc", "inf", "nan"])
def test_validate_elevation_rejects_out_of_range_and_junk(raw):
    with pytest.raises(asc.SelectionError):
        asc.validate_elevation(raw)


# --- antenna payload shape -------------------------------------------------

def test_halt_payload_is_single_key():
    # hwmc: `if len(cmd) > 1: args = cmd['val']` -> a 2-key halt would call
    # _halt(val) and blow up.
    assert asc.ant_payload("halt") == {"cmd": "halt"}


def test_move_and_noise_payloads_carry_val():
    assert asc.ant_payload("move", 45.0) == {"cmd": "move", "val": 45.0}
    assert asc.ant_payload("noise_a_on", True) == {"cmd": "noise_a_on", "val": True}


def test_unknown_antenna_verb_refused():
    with pytest.raises(asc.SelectionError):
        asc.ant_payload("selfdestruct", 1)


# --- write fan-out ---------------------------------------------------------

def test_empty_antenna_selection_uses_single_broadcast_key():
    st = FakeStore()
    keys = asc.send_ant_command(st, [], "move", 45.0)
    assert keys == ["/cmd/ant/0"]
    assert st.writes == [("/cmd/ant/0", {"cmd": "move", "val": 45.0})]


def test_named_antennas_write_one_key_each():
    st = FakeStore()
    keys = asc.send_ant_command(st, [7, 3, 7], "noise_b_on", False)
    assert keys == ["/cmd/ant/3", "/cmd/ant/7"]
    assert all(p == {"cmd": "noise_b_on", "val": False} for _, p in st.writes)


def test_empty_snap_selection_fans_out_to_all_32():
    st = FakeStore()
    keys = asc.send_snap_command(st, [], "arm")
    assert len(keys) == 32
    assert keys[0] == "/cmd/snap/1" and keys[-1] == "/cmd/snap/32"
    assert all(p == {"cmd": "arm"} for _, p in st.writes)


def test_snap_verbs_are_restricted_to_the_five_requested():
    assert set(asc.SNAP_VERBS) == {"arm", "progonly", "prong", "set_delay", "level"}
    st = FakeStore()
    for bad in ("prog", "mon", "test"):
        with pytest.raises(asc.SelectionError):
            asc.send_snap_command(st, [1], bad)
    assert st.writes == []


def test_snap_payloads_are_zero_arg():
    st = FakeStore()
    asc.send_snap_command(st, [5], "set_delay")
    assert st.writes == [("/cmd/snap/5", {"cmd": "set_delay"})]


# --- armed_mjd readback ----------------------------------------------------

def test_armed_mjd_sentinel_is_not_treated_as_armed():
    st = FakeStore({"/mon/snap/1/armed_mjd": {"armed_mjd": 55000.0}})
    row = asc.get_snap_armed(st, now_mjd=61252.0)[0]
    assert row["armed"] is False
    assert row["age_s"] is None
    assert "sentinel" in row["reason"]


def test_armed_mjd_age_computed_in_seconds():
    st = FakeStore({"/mon/snap/1/armed_mjd": {"armed_mjd": 61251.5}})
    row = asc.get_snap_armed(st, now_mjd=61252.0)[0]
    assert row["armed"] is True
    assert row["age_s"] == pytest.approx(43200.0)          # half a day


def test_missing_key_distinguished_from_sentinel():
    st = FakeStore({})
    row = asc.get_snap_armed(st, now_mjd=61252.0)[0]
    assert row["armed"] is False
    assert "no key" in row["reason"]


def test_etcd_error_does_not_propagate():
    st = FakeStore({"/mon/snap/1/armed_mjd": RuntimeError("etcd down")})
    rows = asc.get_snap_armed(st, now_mjd=61252.0)
    assert rows[0]["armed"] is False
    assert "etcd error" in rows[0]["reason"]
    assert len(rows) == 32                                  # still one row per SNAP


def test_get_snap_armed_covers_every_snap():
    st = FakeStore({})
    rows = asc.get_snap_armed(st, now_mjd=61252.0)
    assert [r["snap"] for r in rows] == list(range(1, 33))


@pytest.mark.parametrize(
    "age,expected", [(None, "--"), (5, "5s"), (90, "1m 30s"), (7200, "2h 0m"), (90000, "1d 1h")]
)
def test_format_age(age, expected):
    assert asc.format_age(age) == expected
