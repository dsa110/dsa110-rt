"""Pointing-declination provenance: C2 archives ``/mon/array/dec`` into
every event's Level3 ``c2.pointing_dec_deg`` at trigger time, and C3
prefers that archived snapshot over the live etcd key.

Rationale (2026-07-15 data-provenance gap fix): the ``c2.l_median`` /
``c2.m_median`` image offsets are RELATIVE to the array pointing dec, but
the pointing dec lived only in the mutable, history-less etcd key
``/mon/array/dec``. Without a per-event snapshot a historical event's
absolute RA/Dec is unrecoverable, and a C3 re-run days after a re-point
would beamform against a stale pointing.

These tests never contact live etcd — the mon-store is stubbed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc.archive import (                                    # noqa: E402
    EventArchiveWriter,
    stats_to_l3_metadata,
)
from dsart.coinc.stats import ClusterStats                          # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402
from dsart.services.coincidencer import (                           # noqa: E402
    CoincidencerConfig,
    CoincidencerService,
)


# ---------------------------------------------------------------------------
# Test doubles + fixtures
# ---------------------------------------------------------------------------


class _RecordingStore:
    """Mon-store stand-in: serves get_dict from a map and records the
    keys queried so a test can assert the dec key was (not) read.

    ``raise_on`` forces get_dict to raise for a given key (etcd outage).
    """

    def __init__(
        self,
        responses: Optional[Dict[str, Any]] = None,
        *,
        raise_on: Optional[str] = None,
    ) -> None:
        self._responses = dict(responses or {})
        self._raise_on = raise_on
        self.get_calls: List[str] = []
        self.puts: List[tuple[str, dict]] = []

    def get_dict(self, key: str) -> Optional[Any]:
        self.get_calls.append(key)
        if self._raise_on is not None and key == self._raise_on:
            raise RuntimeError("simulated etcd outage")
        return self._responses.get(key)

    def put_dict(self, key: str, value) -> None:
        self.puts.append((key, dict(value)))


class _NoopBroadcaster:
    def broadcast(self, **kwargs) -> Dict[int, bool]:
        return {}

    def close(self) -> None:
        pass


def _stats() -> ClusterStats:
    return ClusterStats(
        n_events=3,
        n_search_nodes=2,
        n_gpu_halves=3,
        snr_max=12.5,
        snr_sum=33.0,
        snr_mean=11.0,
        dm_min=99.0,
        dm_max=101.0,
        dm_median=100.0,
        dm_iqr=1.0,
        l_median=1.5e-3,
        m_median=-2.5e-3,
        lm_diag_rad=2.0e-3,
        width_min=2,
        width_max=8,
        width_median=4.0,
        t_start_mjd=60781.0,
        t_end_mjd=60781.0 + 1.0 / 86400.0,
        t_peak_mjd=60781.0 + 0.5 / 86400.0,
        kernel_ids_distinct=("unit:d1:b4", "unit:d1:b8"),
        peak_event_specnum=42,
    )


def _coinc_config(tmp_path: Path, **overrides) -> CoincidencerConfig:
    crit = tmp_path / "c.yaml"
    crit.write_text(
        "trigger_classes:\n"
        "  - name: log_only\n"
        "    require:\n"
        "      n_events_min: 1\n"
        "    action: log_only\n"
    )
    base: Dict[str, Any] = dict(
        bind_host="127.0.0.1",
        bind_port=0,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=crit,
        name_allocator_offline=True,
    )
    base.update(overrides)
    return CoincidencerConfig(**base)


def _coinc_service(cfg: CoincidencerConfig, store) -> CoincidencerService:
    return CoincidencerService(
        config=cfg, mon_store=store, broadcaster=_NoopBroadcaster(),
    )


def _c3_service(tmp_path: Path, store):
    cfg = c3mod.C3Config(
        archive_root=tmp_path / "candidates",
        rejected_root=tmp_path / "candidates_rejected",
        state_path=tmp_path / "state.json",
        fired_injection_log=None,
        corr_nodes={},
        flag_only=True,
    )
    svc = c3mod.C3Service(cfg, mon_store=store)
    return svc, cfg


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_config_default_pointing_dec_key(tmp_path: Path) -> None:
    cfg = _coinc_config(tmp_path)
    assert cfg.pointing_dec_etcd_key == "/mon/array/dec"


def test_config_from_yaml_reads_pointing_dec_key(tmp_path: Path) -> None:
    import yaml
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "coinc": {
            "bind": {"host": "127.0.0.1", "port": 11500},
            "csv_dir_c1": str(tmp_path / "c1"),
            "csv_dir_c2": str(tmp_path / "c2"),
            "event_archive_root": str(tmp_path / "events"),
            "trigger_criteria_path": str(tmp_path / "c.yaml"),
            "pointing_dec_etcd_key": "/mon/array/dec_custom",
        }
    }))
    cfg = CoincidencerConfig.from_yaml(p)
    assert cfg.pointing_dec_etcd_key == "/mon/array/dec_custom"


# ---------------------------------------------------------------------------
# Part 1 — C2 stamps the pointing dec into Level3 c2 block
# ---------------------------------------------------------------------------


def test_c2_archives_pointing_dec_from_etcd(tmp_path: Path) -> None:
    """(1) Archived JSON carries c2.pointing_dec_deg == the mocked etcd
    value plus a pointing_dec_meta with the key + a plausible read_unix.
    """
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 16.2734}})
    cfg = _coinc_config(tmp_path)
    svc = _coinc_service(cfg, store)

    store.get_calls.clear()  # ignore reads done during service __init__
    dec, meta = svc._read_pointing_dec()
    assert dec == 16.2734
    assert meta["etcd_key"] == "/mon/array/dec"
    assert isinstance(meta["read_unix"], float)
    assert meta["read_unix"] > 1_700_000_000.0  # sane unix epoch seconds
    assert store.get_calls == ["/mon/array/dec"]  # read exactly once

    wr = EventArchiveWriter(cfg.event_archive_root)
    ev = wr.create("260521abcd")
    md = stats_to_l3_metadata(
        event_name="260521abcd",
        stats=_stats(),
        trigger_class_name="bright_frb",
        trigger_action="dump_all_gpus",
        holdoff_s=30.0,
        pointing_dec_deg=dec,
        pointing_dec_meta=meta,
    )
    p = wr.write_l3_metadata(ev, "260521abcd", md)
    parsed = json.loads(p.read_text())
    assert parsed["c2"]["pointing_dec_deg"] == 16.2734
    assert parsed["c2"]["pointing_dec_meta"]["etcd_key"] == "/mon/array/dec"
    assert isinstance(parsed["c2"]["pointing_dec_meta"]["read_unix"], float)


def test_c2_pointing_dec_null_on_etcd_error(tmp_path: Path, caplog) -> None:
    """(2) etcd read raises → pointing_dec_deg is null, archive still
    written, a warning is logged, no exception escapes."""
    store = _RecordingStore(raise_on="/mon/array/dec")
    cfg = _coinc_config(tmp_path)
    svc = _coinc_service(cfg, store)

    with caplog.at_level("WARNING"):
        dec, meta = svc._read_pointing_dec()
    assert dec is None
    assert meta["etcd_key"] == "/mon/array/dec"
    assert isinstance(meta["read_unix"], float)
    assert any("pointing dec" in r.message for r in caplog.records)

    wr = EventArchiveWriter(cfg.event_archive_root)
    ev = wr.create("260521abcd")
    md = stats_to_l3_metadata(
        event_name="260521abcd", stats=_stats(),
        trigger_class_name="log_only", trigger_action="log_only",
        holdoff_s=0.0, pointing_dec_deg=dec, pointing_dec_meta=meta,
    )
    p = wr.write_l3_metadata(ev, "260521abcd", md)
    parsed = json.loads(p.read_text())
    assert parsed["c2"]["pointing_dec_deg"] is None
    assert parsed["c2"]["pointing_dec_meta"]["etcd_key"] == "/mon/array/dec"


@pytest.mark.parametrize("doc", [
    {},                       # missing dec_deg
    {"dec_deg": "not-a-num"},  # non-numeric
    {"dec_deg": None},        # explicit null
    "garbage",                # not even a mapping
])
def test_c2_pointing_dec_null_on_malformed_doc(
    tmp_path: Path, doc: Any,
) -> None:
    """(3) Malformed etcd doc → pointing_dec_deg is null, no exception."""
    store = _RecordingStore({"/mon/array/dec": doc})
    cfg = _coinc_config(tmp_path)
    svc = _coinc_service(cfg, store)
    dec, meta = svc._read_pointing_dec()
    assert dec is None
    assert meta["etcd_key"] == "/mon/array/dec"


@pytest.mark.parametrize("bad_dec", [
    float("nan"),
    float("inf"),
    float("-inf"),
    123.4,   # out of plausibility range: dec must lie in [-90, +90]
    -123.4,
])
def test_c2_pointing_dec_null_on_nonfinite_or_out_of_range(
    tmp_path: Path, bad_dec: float, caplog,
) -> None:
    """(a) NaN/inf/out-of-range dec from etcd → null stamped, WARNING
    logged, archive still written (never a literal NaN/Infinity token
    in the JSON — that would be invalid strict JSON)."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": bad_dec}})
    cfg = _coinc_config(tmp_path)
    svc = _coinc_service(cfg, store)

    with caplog.at_level("WARNING"):
        dec, meta = svc._read_pointing_dec()
    assert dec is None
    assert meta["etcd_key"] == "/mon/array/dec"
    assert any("pointing dec" in r.message for r in caplog.records)

    wr = EventArchiveWriter(cfg.event_archive_root)
    ev = wr.create("260521abcd")
    md = stats_to_l3_metadata(
        event_name="260521abcd", stats=_stats(),
        trigger_class_name="log_only", trigger_action="log_only",
        holdoff_s=0.0, pointing_dec_deg=dec, pointing_dec_meta=meta,
    )
    p = wr.write_l3_metadata(ev, "260521abcd", md)
    text = p.read_text()
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["c2"]["pointing_dec_deg"] is None


def test_c2_l3_metadata_omits_field_gracefully() -> None:
    """Additive contract: with no pointing dec supplied the keys are
    still present as None (consumers must tolerate this)."""
    md = stats_to_l3_metadata(
        event_name="260521abcd", stats=_stats(),
        trigger_class_name="log_only", trigger_action="log_only",
        holdoff_s=0.0,
    )
    assert md["c2"]["pointing_dec_deg"] is None
    assert md["c2"]["pointing_dec_meta"] is None
    assert md["schema_version"] == 1  # additive → not bumped


# ---------------------------------------------------------------------------
# Part 2 — C3 prefers the archived value over the live etcd read
# ---------------------------------------------------------------------------


def test_c3_prefers_archived_pointing_dec(tmp_path: Path) -> None:
    """(4) Level3 c2 HAS pointing_dec_deg → that value is used and the
    live etcd dec key is NOT consulted."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 99.0}})
    svc, _ = _c3_service(tmp_path, store)
    c2row = {"pointing_dec_deg": 16.2734, "l_median": 0.0}
    dec = svc._resolve_pointing_dec("260521abcd", c2row)
    assert dec == 16.2734
    # Live dec key must NOT have been read (archived snapshot wins).
    assert "/mon/array/dec" not in store.get_calls


def test_c3_falls_back_to_live_etcd_when_absent(tmp_path: Path) -> None:
    """(5) Level3 c2 lacks the field → falls back to the live etcd read
    (preserves historical behaviour for pre-fix events)."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 71.601}})
    svc, _ = _c3_service(tmp_path, store)
    dec = svc._resolve_pointing_dec("260521abcd", {"l_median": 0.0})
    assert dec == 71.601
    assert store.get_calls == ["/mon/array/dec"]  # live read happened


def test_c3_falls_back_when_archived_null(tmp_path: Path) -> None:
    """Explicit null archived value (etcd was down at C2 time) still
    falls through to the live read."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 42.5}})
    svc, _ = _c3_service(tmp_path, store)
    dec = svc._resolve_pointing_dec(
        "260521abcd", {"pointing_dec_deg": None})
    assert dec == 42.5
    assert store.get_calls == ["/mon/array/dec"]


def test_c3_non_numeric_archived_falls_back(tmp_path: Path) -> None:
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 33.0}})
    svc, _ = _c3_service(tmp_path, store)
    dec = svc._resolve_pointing_dec(
        "260521abcd", {"pointing_dec_deg": "bad"})
    assert dec == 33.0
    assert store.get_calls == ["/mon/array/dec"]


def test_c3_returns_none_when_neither_source(tmp_path: Path) -> None:
    store = _RecordingStore(raise_on="/mon/array/dec")
    svc, _ = _c3_service(tmp_path, store)
    dec = svc._resolve_pointing_dec("260521abcd", {})
    assert dec is None


_BAD_DECS = [float("nan"), float("inf"), float("-inf"), 123.4, -123.4]


@pytest.mark.parametrize("bad_dec", _BAD_DECS)
def test_c3_nonfinite_or_out_of_range_archived_falls_back(
    tmp_path: Path, bad_dec: float, caplog,
) -> None:
    """(b) NaN/inf/out-of-range ARCHIVED value is corrupt provenance →
    warn + fall back to the live etcd read."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": 33.0}})
    svc, _ = _c3_service(tmp_path, store)
    with caplog.at_level("WARNING"):
        dec = svc._resolve_pointing_dec(
            "260521abcd", {"pointing_dec_deg": bad_dec})
    assert dec == 33.0
    assert store.get_calls == ["/mon/array/dec"]  # live fallback used
    assert any("non-finite or out of range" in r.message
               for r in caplog.records)


@pytest.mark.parametrize("bad_dec", _BAD_DECS)
def test_c3_nonfinite_or_out_of_range_live_read_returns_none(
    tmp_path: Path, bad_dec: float, caplog,
) -> None:
    """(c) NaN/inf/out-of-range LIVE etcd value (no archived value) →
    None (bbproc then beamforms at (l,m)=(0,0)), warning logged."""
    store = _RecordingStore({"/mon/array/dec": {"dec_deg": bad_dec}})
    svc, _ = _c3_service(tmp_path, store)
    with caplog.at_level("WARNING"):
        dec = svc._resolve_pointing_dec("260521abcd", {})
    assert dec is None
    assert any("no pointing dec" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Part 1 + 2 — round trip: C2 writes, C3 reads back bit-identical
# ---------------------------------------------------------------------------


def test_round_trip_c2_to_c3_bit_identical(tmp_path: Path) -> None:
    """(6) The exact float C2 archives is what C3 recovers."""
    dec_in = 16.273401234567
    c2_store = _RecordingStore({"/mon/array/dec": {"dec_deg": dec_in}})
    ccfg = _coinc_config(tmp_path)
    csvc = _coinc_service(ccfg, c2_store)
    dec, meta = csvc._read_pointing_dec()

    wr = EventArchiveWriter(ccfg.event_archive_root)
    ev = wr.create("260521abcd")
    md = stats_to_l3_metadata(
        event_name="260521abcd", stats=_stats(),
        trigger_class_name="log_only", trigger_action="log_only",
        holdoff_s=0.0, pointing_dec_deg=dec, pointing_dec_meta=meta,
    )
    l3_path = wr.write_l3_metadata(ev, "260521abcd", md)

    # C3 side: read the JSON exactly as _do_keep does, then resolve.
    c2row = (json.loads(l3_path.read_text()) or {}).get("c2", {}) or {}
    c3_store = _RecordingStore({"/mon/array/dec": {"dec_deg": 99.0}})
    c3svc, _ = _c3_service(tmp_path, c3_store)
    dec_out = c3svc._resolve_pointing_dec("260521abcd", c2row)

    assert dec_out == dec_in
    assert dec_out.hex() == dec_in.hex()  # bit-identical
    assert "/mon/array/dec" not in c3_store.get_calls  # live not consulted
