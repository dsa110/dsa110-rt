"""x2 D1: header-derived sample units downstream (docs/x2/downstream_inventory.md).

Two properties per change: archive schema 1 / pre-schema-2 artefacts give
today's bytes exactly, and schema 2 / a 524.288 us period gives the 2x
values.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dsart.coinc import archive as A
from dsart.coinc.csv_rotator import RollingCsvWriter, concat_recent_hourly
from dsart.coinc.stats import compute_stats
from dsart.coinc.units import (
    T_SEARCH_1X_US,
    native_per_sample,
    period_from_mapping,
    sample_period_us_or_default,
    specnums_per_sample,
    width_ms,
)
from dsart.coinc.window import WindowEntry
from dsart.services import candidate_card as CC

P1 = 1048.576
P2 = 524.288


def _entry(spec: int, period: float, snr: float, w: int = 4,
           dm: float = 500.0) -> WindowEntry:
    return WindowEntry(
        mjd=61236.0 + spec * period / 86400e6, snr=snr, l_rad=0.001,
        m_rad=0.002, l_pix=10, m_pix=12, dm_pc_cc=dm, dm_idx_global=10,
        fine_dm_idx=1, event_specnum=spec, width_samples=w,
        kernel_id="unit:d1:b4", flags=0, search_node_id=1, gpu_half=0,
        cube_id=7, sample_period_us=period,
    )


# ---------------------------------------------------------------- units


def test_units_ratios_both_opoints() -> None:
    assert native_per_sample(P1) == 32 and native_per_sample(P2) == 16
    assert specnums_per_sample(P1) == 16 and specnums_per_sample(P2) == 8
    with pytest.raises(ValueError):
        native_per_sample(1000.0)


def test_period_defaults_and_placeholders() -> None:
    assert sample_period_us_or_default(None) == T_SEARCH_1X_US
    assert sample_period_us_or_default(1.0) == T_SEARCH_1X_US  # heartbeat
    assert sample_period_us_or_default("nan") == T_SEARCH_1X_US
    assert sample_period_us_or_default(P2) == P2
    assert period_from_mapping({}) == T_SEARCH_1X_US
    assert period_from_mapping({"sample_period_us": None,
                                "peak_sample_period_us": P2}) == P2
    assert period_from_mapping({"sample_period_us": P2}) == P2
    assert width_ms(32, P2) == pytest.approx(16.777216)
    assert width_ms(16, P1) == pytest.approx(16.777216)


# ------------------------------------------------------------- archive v1


def test_schema1_rows_and_l3_are_todays() -> None:
    e = _entry(100, P1, 12.0)
    row = A.entry_to_csv_row(e, trigger="x", inj_id="")
    assert tuple(row) == A.C1_WINDOW_CSV_FIELDS
    st = compute_stats([e, _entry(101, P1, 20.0, w=2)])
    srow = A.stats_to_csv_row(st, trigger_class="c", trigger="t")
    assert tuple(srow) == A.C2_CLUSTER_CSV_FIELDS
    l3 = A.stats_to_l3_metadata(
        event_name="260101aaaa", stats=st, trigger_class_name="c",
        trigger_action="dump_all_gpus", holdoff_s=30.0)
    assert l3["schema_version"] == 1
    assert "opoint" not in l3
    for k in ("sample_period_us", "width_median_ms", "dm_peak",
              "width_peak", "width_peak_ms"):
        assert k not in l3["c2"]
    assert A.c1_window_fields(1) == A.C1_WINDOW_CSV_FIELDS
    assert A.c2_cluster_fields(1) == A.C2_CLUSTER_CSV_FIELDS


def test_schema1_files_byte_identical(tmp_path: Path) -> None:
    """The writer at its default schema renders exactly the pre-x2 bytes."""
    members = [_entry(100, P1, 12.0), _entry(101, P1, 20.0, w=2)]
    st = compute_stats(members)
    wr = A.EventArchiveWriter(tmp_path)
    ev = wr.create("260101aaaa")
    p2 = wr.write_c2_cluster_csv(ev, "260101aaaa", st, trigger_class="c",
                                 trigger="260101aaaa")
    p1 = wr.write_c1_window_csv(ev, "260101aaaa", members,
                                trigger="260101aaaa")
    exp2 = A.EventArchiveWriter._render_csv(
        A.C2_CLUSTER_CSV_FIELDS,
        [A._stats_to_csv_row_v1(st, trigger_class="c",
                                trigger="260101aaaa")])
    exp1 = A.EventArchiveWriter._render_csv(
        A.C1_WINDOW_CSV_FIELDS,
        [A._entry_to_csv_row_v1(m, trigger="260101aaaa") for m in members])
    assert p2.read_bytes() == exp2.encode()
    assert p1.read_bytes() == exp1.encode()


# ------------------------------------------------------------- archive v2


def test_schema2_appends_units_at_the_end(tmp_path: Path) -> None:
    members = [_entry(100, P2, 12.0, w=8), _entry(101, P2, 20.0, w=2,
                                                     dm=2500.0)]
    st = compute_stats(members)
    wr = A.EventArchiveWriter(tmp_path, schema=2, opoint="2x",
                              fine_index_resolver=lambda f: (f // 16, 0))
    ev = wr.create("260101bbbb")
    p2 = wr.write_c2_cluster_csv(ev, "260101bbbb", st, trigger_class="c")
    p1 = wr.write_c1_window_csv(ev, "260101bbbb", members)
    rows1 = list(csv.DictReader(p1.open()))
    hdr1 = next(csv.reader(p1.open()))
    assert tuple(hdr1[:len(A.C1_WINDOW_CSV_FIELDS)]) == A.C1_WINDOW_CSV_FIELDS
    assert tuple(hdr1) == A.C1_WINDOW_CSV_FIELDS_V2
    assert float(rows1[0]["sample_period_us"]) == pytest.approx(P2)
    assert float(rows1[0]["width_ms"]) == pytest.approx(8 * P2 / 1e3)
    assert rows1[0]["coarse_idx"] == "0" and rows1[0]["tier"] == "0"
    r2 = next(csv.DictReader(p2.open()))
    assert float(r2["peak_sample_period_us"]) == pytest.approx(P2)
    assert float(r2["dm_peak"]) == pytest.approx(2500.0)
    assert r2["width_peak"] == "2" and r2["opoint"] == "2x"
    assert float(r2["width_median_ms"]) == pytest.approx(5 * P2 / 1e3)

    l3 = A.stats_to_l3_metadata(
        event_name="260101bbbb", stats=st, trigger_class_name="c",
        trigger_action="dump_all_gpus", holdoff_s=30.0, archive_schema=2,
        opoint="2x")
    assert l3["schema_version"] == 2 and l3["opoint"] == "2x"
    c2 = l3["c2"]
    assert c2["sample_period_us"] == pytest.approx(P2)
    assert c2["width_peak_ms"] == pytest.approx(2 * P2 / 1e3)
    assert c2["dm_peak"] == pytest.approx(2500.0)
    json.dumps(l3)  # serialisable


def test_schema2_resolver_failure_leaves_provenance_blank() -> None:
    def bad(_f: int):
        raise KeyError(_f)
    row = A.entry_to_csv_row(_entry(1, P2, 10.0), schema=2,
                             fine_index_resolver=bad)
    assert row["coarse_idx"] == "" and row["tier"] == ""


# ---------------------------------------------------------- rolling CSVs


def test_rolling_csv_rekeys_hour_file_on_schema_change(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 5, 10, tzinfo=timezone.utc)
    w1 = RollingCsvWriter(tmp_path, "c2", A.C2_CLUSTER_CSV_FIELDS)
    st = compute_stats([_entry(1, P1, 12.0)])
    p = w1.append_row(A.stats_to_csv_row(st), now_utc=now)
    before = p.read_text()
    # same schema restart: file untouched
    w1b = RollingCsvWriter(tmp_path, "c2", A.C2_CLUSTER_CSV_FIELDS)
    w1b.append_row(A.stats_to_csv_row(st), now_utc=now)
    assert p.read_text().startswith(before)
    assert p.read_text().count("mjd_peak") == 1
    # schema-2 restart mid-hour: header upgraded, old rows kept
    w2 = RollingCsvWriter(tmp_path, "c2", A.C2_CLUSTER_CSV_FIELDS_V2)
    w2.append_row(A.stats_to_csv_row(st, schema=2, opoint="2x"),
                  now_utc=now)
    rows = list(csv.DictReader(p.open()))
    assert next(csv.reader(p.open())) == list(A.C2_CLUSTER_CSV_FIELDS_V2)
    assert len(rows) == 3
    assert rows[0]["opoint"] == "" and rows[2]["opoint"] == "2x"


def test_concat_same_header_unchanged_and_mixed_rekeyed(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 5, 10, tzinfo=timezone.utc)
    st = compute_stats([_entry(1, P1, 12.0)])
    w1 = RollingCsvWriter(tmp_path, "c2", A.C2_CLUSTER_CSV_FIELDS)
    w1.append_row(A.stats_to_csv_row(st), now_utc=now - timedelta(hours=1))
    w1.append_row(A.stats_to_csv_row(st), now_utc=now)
    out = concat_recent_hourly(tmp_path, "c2", now_utc=now, window_hours=3,
                               out_name="c2_last.csv")
    txt = out.read_text().splitlines()
    assert txt[0].startswith("mjd_peak") and len(txt) == 3
    assert sum(1 for ln in txt if ln.startswith("mjd_peak")) == 1
    # newest hour at schema 2 -> rows re-keyed onto the newest header
    later = now + timedelta(hours=1)
    w2 = RollingCsvWriter(tmp_path, "c2", A.C2_CLUSTER_CSV_FIELDS_V2)
    w2.append_row(A.stats_to_csv_row(st, schema=2, opoint="2x"),
                  now_utc=later)
    out = concat_recent_hourly(tmp_path, "c2", now_utc=later,
                               window_hours=3, out_name="c2_last.csv")
    rows = list(csv.DictReader(out.open()))
    assert next(csv.reader(out.open())) == list(A.C2_CLUSTER_CSV_FIELDS_V2)
    assert len(rows) == 3
    assert [r["opoint"] for r in rows] == ["", "", "2x"]


# ------------------------------------------------------ filterbank, card


def _fb_width(tmp_path: Path, c2row: dict) -> int:
    """Stub toolkit + plot (as tests/test_c3_filterbank.py) and run."""
    import stat

    from dsart.coinc.filterbank import FilterbankConfig, run_for_event
    tmp_path.mkdir(parents=True, exist_ok=True)
    toolkit = tmp_path / "toolkit_stub.sh"
    toolkit.write_text(
        "#!/bin/bash\nout=''\n"
        "while [ $# -gt 0 ]; do\n"
        "  if [ \"$1\" = '-P' ]; then out=$2; fi; shift\ndone\n"
        "dd if=/dev/zero of=$out bs=1k count=1 2>/dev/null\n")
    toolkit.chmod(toolkit.stat().st_mode | stat.S_IEXEC)
    plot = tmp_path / "plot_stub.py"
    plot.write_text("import sys\nout = sys.argv[sys.argv.index('--out') + 1]\n"
                    "open(out, 'wb').write(b'PNG')\n")
    applied = tmp_path / "applied"
    applied.mkdir()
    for sb in range(16):
        (applied / f"beamformer_weights_sb{sb:02d}_2026-07-14T00:00:00.dat"
         ).write_bytes(b"x" * 16)
    cfg = FilterbankConfig(
        enabled=True, toolkit_bin=str(toolkit), plot_script=str(plot),
        core_antennas="all", cal_applied_dir=str(applied), rfi_mode="off",
        timeout_s=30.0)
    name = "260715aaaa"
    ev = tmp_path / name
    (ev / "Level2" / "voltages").mkdir(parents=True)
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level2" / "voltages" / f"{name}_sb00_data.out").write_bytes(b"v")
    rep = run_for_event(cfg, ev, name, c2row, dec_deg=16.0)
    assert rep["ok"] is True, rep
    return int(rep["width_fil_samples"])


def test_filterbank_width_uses_archived_period(tmp_path: Path) -> None:
    base = {"l_median": 0.0, "m_median": 0.0, "dm_median": 300.0,
            "width_median": 4.0, "t_peak_mjd": 61236.5}
    assert _fb_width(tmp_path / "a", dict(base)) == 16          # 1x fallback
    assert _fb_width(tmp_path / "b", dict(base, sample_period_us=P1)) == 16
    assert _fb_width(tmp_path / "c", dict(base, sample_period_us=P2)) == 8


def test_card_width_string() -> None:
    assert CC._width_str(16) == "16.8 ms (16 samp)"
    assert CC._width_str(32, CC._c2_sample_period_ms(
        {"sample_period_us": P2})) == "16.8 ms (32 samp)"
    assert CC._c2_sample_period_ms({"width_median": 3}) is None
    assert CC._c2_sample_period_ms({"sample_period_us": None}) is None
    assert math.isclose(CC._c2_sample_period_ms(
        {"peak_sample_period_us": P1}), P1 / 1e3)
