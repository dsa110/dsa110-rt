"""Tests for the C2 voltage-dump broadcast gate in the coincidencer.

Covers:
  * DumpsGate(default_enabled=False) fail-CLOSED behaviour (voltages),
  * _maybe_broadcast_voltage gating: injection skip, disabled skip,
    fire-when-enabled, and broadcaster-not-configured no-op.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict

from dsart.services.coincidencer import (
    CoincidencerConfig,
    CoincidencerService,
    DumpsGate,
)
from dsart.services.coincidencer import search_to_snap_specnum


# --- DumpsGate default_enabled -----------------------------------------


def test_dumps_gate_default_open_no_store() -> None:
    g = DumpsGate(None, default_enabled=True)
    assert g.enabled() is True


def test_voltages_gate_default_closed_no_store() -> None:
    g = DumpsGate(None, default_enabled=False)
    assert g.enabled() is False


class _ErrStore:
    def get_dict(self, key: str):
        raise RuntimeError("etcd down")


def test_gate_fails_to_default_on_etcd_error() -> None:
    # Voltages gate must stay CLOSED on a transient etcd error.
    assert DumpsGate(_ErrStore(), default_enabled=False).enabled() is False
    # Cube gate stays OPEN on the same error (fail-open).
    assert DumpsGate(_ErrStore(), default_enabled=True).enabled() is True


# --- _maybe_broadcast_voltage ------------------------------------------


class _NoopBroadcaster:
    def broadcast(self, **kwargs) -> Dict[int, bool]:
        return {}

    def close(self) -> None:
        pass


class _RecordingVoltageBroadcaster:
    def __init__(self) -> None:
        self.calls = []

    def broadcast(self, **kwargs) -> Dict[int, bool]:
        self.calls.append(kwargs)
        return {3: True, 4: True, 5: False}

    def close(self) -> None:
        pass


class _Gate:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self.read_count = 0
        self.fail_count = 0

    def enabled(self) -> bool:
        return self._enabled


def _config(tmp_path: Path) -> CoincidencerConfig:
    crit = tmp_path / "c.yaml"
    crit.write_text(
        "trigger_classes:\n"
        "  - name: log_only\n"
        "    require:\n"
        "      n_events_min: 1\n"
        "    action: log_only\n"
    )
    return CoincidencerConfig(
        bind_host="127.0.0.1",
        bind_port=0,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=crit,
        name_allocator_offline=True,
    )


def _service(tmp_path: Path, *, vbc, vgate) -> CoincidencerService:
    return CoincidencerService(
        config=_config(tmp_path),
        mon_store=None,
        broadcaster=_NoopBroadcaster(),
        voltage_broadcaster=vbc,
        voltages_gate=vgate,
    )


_STATS = SimpleNamespace(
    peak_event_specnum=42, t_peak_mjd=60800.0, snr_max=30.0, dm_median=500.0,
    peak_sample_period_us=1048.576,
)
_TC = SimpleNamespace(name="bright_frb")


def test_voltage_skipped_for_injection(tmp_path: Path) -> None:
    vbc = _RecordingVoltageBroadcaster()
    svc = _service(tmp_path, vbc=vbc, vgate=_Gate(True))
    svc._maybe_broadcast_voltage(
        event_name="ev", stats=_STATS, trigger_class=_TC, is_injection=True,
    )
    assert vbc.calls == []
    assert svc._counters["voltages_skipped_injection"] == 1


def test_voltage_skipped_when_disabled(tmp_path: Path) -> None:
    vbc = _RecordingVoltageBroadcaster()
    svc = _service(tmp_path, vbc=vbc, vgate=_Gate(False))
    svc._maybe_broadcast_voltage(
        event_name="ev", stats=_STATS, trigger_class=_TC, is_injection=False,
    )
    assert vbc.calls == []
    assert svc._counters["voltages_skipped_disabled"] == 1


def test_voltage_fires_when_enabled_and_real(tmp_path: Path) -> None:
    vbc = _RecordingVoltageBroadcaster()
    svc = _service(tmp_path, vbc=vbc, vgate=_Gate(True))
    svc._maybe_broadcast_voltage(
        event_name="ev", stats=_STATS, trigger_class=_TC, is_injection=False,
    )
    assert len(vbc.calls) == 1
    assert vbc.calls[0]["event_name"] == "ev"
    # search-sample specnum converted to SNAP specnums using the peak
    # member's own sample period (2026-07-13 empty-dump bug +
    # VOLTAGE_DUMP_TIMING_FIX.md defect 2).
    assert vbc.calls[0]["event_specnum"] == search_to_snap_specnum(
        42, 1048.576) == 42 * 16
    assert svc._counters["voltages_broadcast"] == 1
    assert svc._counters["voltage_broadcast_ok"] == 2
    assert svc._counters["voltage_broadcast_fail"] == 1


def test_voltage_noop_when_broadcaster_absent(tmp_path: Path) -> None:
    svc = _service(tmp_path, vbc=None, vgate=_Gate(True))
    # Must not raise and must not count anything.
    svc._maybe_broadcast_voltage(
        event_name="ev", stats=_STATS, trigger_class=_TC, is_injection=False,
    )
    assert svc._counters["voltages_broadcast"] == 0
