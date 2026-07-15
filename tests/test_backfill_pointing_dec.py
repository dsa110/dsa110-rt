"""Tests for tools/ops/backfill_pointing_dec.py (manual pointing-dec
backfill into historical Level3 event JSONs).

All fixtures live in tmp_path — the live archive is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCRIPT = REPO_ROOT / "tools" / "ops" / "backfill_pointing_dec.py"

_spec = importlib.util.spec_from_file_location(
    "backfill_pointing_dec", SCRIPT)
bf = importlib.util.module_from_spec(_spec)
sys.modules["backfill_pointing_dec"] = bf
_spec.loader.exec_module(bf)


DEC = 16.273406015527343
MJD_MIN, MJD_MAX = 61230.0, 61236.75
IN_WINDOW_MJD = 61232.5


def _mk_event(
    root: Path, name: str, *,
    mjd: float | None = IN_WINDOW_MJD,
    c2: bool = True,
    pointing_dec: object = "ABSENT",
    filterbank_dec: float | None = None,
    raw_meta: dict | None = None,
) -> Path:
    ev = root / name
    (ev / "Level3").mkdir(parents=True, exist_ok=True)
    if raw_meta is not None:
        meta = raw_meta
    else:
        meta = {"event_name": name, "schema_version": 1}
        if c2:
            c2d: dict = {"snr_max": 12.5, "l_median": 1.5e-3,
                         "m_median": -2.5e-3}
            if mjd is not None:
                c2d["t_peak_mjd"] = mjd
            if pointing_dec != "ABSENT":
                c2d["pointing_dec_deg"] = pointing_dec
            meta["c2"] = c2d
        else:
            meta.update({"ra": 187.7, "dec": 12.4})  # legacy flat schema
    (ev / "Level3" / f"{name}.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    if filterbank_dec is not None:
        (ev / "filterbank").mkdir(parents=True, exist_ok=True)
        (ev / "filterbank" / "filterbank.json").write_text(
            json.dumps({"ok": True, "dec_deg": filterbank_dec}))
    return ev


def _run(root: Path, backup: Path, *extra: str) -> int:
    argv = [
        "--root", str(root),
        "--dec-deg", repr(DEC),
        "--mjd-min", str(MJD_MIN),
        "--mjd-max", str(MJD_MAX),
        "--backup-dir", str(backup),
        "--note", "unit-test backfill",
        *extra,
    ]
    return bf.main(argv)


def _l3(root: Path, name: str) -> dict:
    return json.loads(
        (root / name / "Level3" / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# Apply path
# ---------------------------------------------------------------------------


def test_eligible_event_stamped_with_backup(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710aaaa")
    original = _l3(root, "260710aaaa")

    assert _run(root, backup, "--apply") == 0

    doc = _l3(root, "260710aaaa")
    assert doc["c2"]["pointing_dec_deg"] == DEC
    meta = doc["c2"]["pointing_dec_meta"]
    assert meta["etcd_key"] is None
    assert meta["source"] == "manual_backfill"
    assert meta["note"] == "unit-test backfill"
    assert isinstance(meta["read_unix"], float)
    # Everything else preserved.
    assert doc["event_name"] == "260710aaaa"
    assert doc["c2"]["snr_max"] == original["c2"]["snr_max"]
    # Backup exists and is byte-recoverable to the original.
    bak = json.loads((backup / "260710aaaa.json").read_text())
    assert bak == original
    assert "pointing_dec_deg" not in bak["c2"]


def test_null_pointing_dec_is_eligible(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710bbbb", pointing_dec=None)
    assert _run(root, backup, "--apply") == 0
    assert _l3(root, "260710bbbb")["c2"]["pointing_dec_deg"] == DEC


def test_nonnull_pointing_dec_skipped(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710cccc", pointing_dec=42.0)
    assert _run(root, backup, "--apply") == 0
    assert _l3(root, "260710cccc")["c2"]["pointing_dec_deg"] == 42.0
    assert not (backup / "260710cccc.json").exists()


def test_legacy_no_c2_skipped(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "251201zzzz", c2=False)
    assert _run(root, backup, "--apply") == 0
    doc = _l3(root, "251201zzzz")
    assert "c2" not in doc
    assert not (backup / "251201zzzz.json").exists()


def test_out_of_window_skipped(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260601dddd", mjd=MJD_MIN - 1.0)
    _mk_event(root, "260801eeee", mjd=MJD_MAX + 1.0)
    assert _run(root, backup, "--apply") == 0
    assert "pointing_dec_deg" not in _l3(root, "260601dddd")["c2"]
    assert "pointing_dec_deg" not in _l3(root, "260801eeee")["c2"]


def test_second_apply_run_is_noop(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710ffff")
    assert _run(root, backup, "--apply") == 0
    first = _l3(root, "260710ffff")
    first_bytes = (root / "260710ffff" / "Level3"
                   / "260710ffff.json").read_bytes()
    # Second run: event is already stamped non-null -> skipped, backup
    # untouched, file bytes identical.
    assert _run(root, backup, "--apply") == 0
    assert (root / "260710ffff" / "Level3"
            / "260710ffff.json").read_bytes() == first_bytes
    bak = json.loads((backup / "260710ffff.json").read_text())
    assert "pointing_dec_deg" not in bak["c2"]
    assert _l3(root, "260710ffff") == first


def test_existing_backup_never_overwritten(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710gggg")
    backup.mkdir(parents=True)
    (backup / "260710gggg.json").write_text('{"sentinel": true}')
    assert _run(root, backup, "--apply") == 0
    # Backup preserved verbatim; the event was NOT stamped (we cannot
    # prove the sentinel is a faithful original).
    assert json.loads(
        (backup / "260710gggg.json").read_text()) == {"sentinel": True}
    assert "pointing_dec_deg" not in _l3(root, "260710gggg")["c2"]


def test_output_matches_archive_formatting(tmp_path: Path) -> None:
    """Stamped file re-parses and is exactly json.dumps(indent=2,
    sort_keys=True) + newline — the archive.py write_l3_metadata shape."""
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710hhhh")
    assert _run(root, backup, "--apply") == 0
    text = (root / "260710hhhh" / "Level3" / "260710hhhh.json").read_text()
    doc = json.loads(text)
    assert text == json.dumps(doc, indent=2, sort_keys=True,
                              default=str) + "\n"
    assert doc["c2"]["pointing_dec_deg"] == DEC


# ---------------------------------------------------------------------------
# Consistency guard
# ---------------------------------------------------------------------------


def test_filterbank_mismatch_aborts_no_writes(
    tmp_path: Path, capsys,
) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710iiii")                       # eligible
    _mk_event(root, "260710jjjj", pointing_dec=DEC,     # in-window, stamped,
              filterbank_dec=DEC + 0.5)                 # but fb dec disagrees
    rc = _run(root, backup, "--apply")
    assert rc != 0
    err = capsys.readouterr().err
    assert "260710jjjj" in err and "mismatch" in err
    # NOTHING written: eligible event untouched, no backup dir content.
    assert "pointing_dec_deg" not in _l3(root, "260710iiii")["c2"]
    assert not (backup / "260710iiii.json").exists()


def test_filterbank_match_passes(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710kkkk", filterbank_dec=DEC)
    assert _run(root, backup, "--apply") == 0
    assert _l3(root, "260710kkkk")["c2"]["pointing_dec_deg"] == DEC


def test_guard_runs_in_dry_run_too(tmp_path: Path, capsys) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710llll", filterbank_dec=DEC + 1.0)
    rc = _run(root, backup)  # no --apply
    assert rc != 0
    assert "mismatch" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710mmmm")
    before = (root / "260710mmmm" / "Level3"
              / "260710mmmm.json").read_bytes()
    assert _run(root, backup) == 0   # default: dry run
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "would-stamp" in out
    assert (root / "260710mmmm" / "Level3"
            / "260710mmmm.json").read_bytes() == before
    assert not backup.exists()


def test_summary_counts(tmp_path: Path, capsys) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    _mk_event(root, "260710nnnn")                          # eligible
    _mk_event(root, "260710oooo", pointing_dec=DEC)        # already stamped
    _mk_event(root, "260601pppp", mjd=MJD_MIN - 5.0)       # out of window
    _mk_event(root, "251201qqqq", c2=False)                # legacy
    assert _run(root, backup) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN   eligible: 1" in out
    assert "DRY-RUN   would-stamp: 1" in out
    assert "skipped (already_stamped): 1" in out
    assert "skipped (out_of_window): 1" in out
    assert "skipped (legacy_or_no_c2): 1" in out


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_bad_dec_rejected(tmp_path: Path) -> None:
    root, backup = tmp_path / "cands", tmp_path / "bak"
    root.mkdir()
    rc = bf.main([
        "--root", str(root), "--dec-deg", "123.4",
        "--mjd-min", "61230.0", "--mjd-max", "61236.75",
        "--backup-dir", str(backup), "--note", "x",
    ])
    assert rc != 0
