"""Tests for the C2 coincidencer service's galactic-DM discriminant
plumbing.

We don't spin up a full service (the receiver opens a socket etc.) —
instead we instantiate a CoincidencerService with a stub etcd store and
exercise:

  * config parsing of the new ``gal_dm_*`` block
  * ``_current_gal_dm_max_los`` freshness gating
  * ``_gal_dm_poll_loop`` updating the cache from a stub etcd
  * mon_publish_loop including gal_dm fields in the payload
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import yaml

from dsart.coinc.broadcast import TriggerBroadcaster
from dsart.services.coincidencer import (
    CoincidencerConfig,
    CoincidencerService,
)


class _NoopBroadcaster:
    """In-memory TriggerBroadcaster substitute (no UDP sockets opened).

    The real broadcaster requires a non-empty hosts map; for these
    tests we only care about CoincidencerService construction +
    poll-loop logic, so substituting a noop avoids touching the
    network entirely.
    """

    def broadcast(self, **kwargs) -> Dict[int, bool]:
        return {}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubStore:
    """Stub DsaStore with controllable gal_dm payload."""

    def __init__(self, value: Optional[float] = None) -> None:
        self.value = value
        self.put_count = 0
        self.last_put: Dict[str, Any] = {}

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        if self.value is None:
            return None
        return {"time": 60000.0, "gal_dm": float(self.value)}

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        self.put_count += 1
        self.last_put = dict(value)


def _minimal_config(tmp_path: Path, **overrides) -> CoincidencerConfig:
    """Spin a CoincidencerConfig pointed at tmp_path-scoped dirs.

    Avoids touching /dataz which the bench has no rights to.
    """
    # Spit out a starter criteria file so CriteriaEvaluator can load.
    crit = tmp_path / "c.yaml"
    crit.write_text("""
trigger_classes:
  - name: log_only
    require:
      n_events_min: 1
    action: log_only
""")
    base = dict(
        bind_host="127.0.0.1",
        bind_port=0,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=crit,
        name_allocator_offline=True,
        # The default 30 s would force the poll-loop test into a long
        # wait; use 0.05 s here.
        gal_dm_poll_interval_s=0.05,
    )
    base.update(overrides)
    return CoincidencerConfig(**base)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_config_from_yaml_reads_gal_dm_block(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "coinc": {
            "bind": {"host": "127.0.0.1", "port": 11500},
            "csv_dir_c1": str(tmp_path / "c1"),
            "csv_dir_c2": str(tmp_path / "c2"),
            "event_archive_root": str(tmp_path / "events"),
            "trigger_criteria_path": str(tmp_path / "c.yaml"),
            "gal_dm_etcd_key": "/mon/array/gal_dm",
            "gal_dm_poll_interval_s": 45.0,
            "gal_dm_max_los_override": 137.5,
            "gal_dm_max_age_s": 300.0,
        }
    }))
    cfg = CoincidencerConfig.from_yaml(p)
    assert cfg.gal_dm_etcd_key == "/mon/array/gal_dm"
    assert cfg.gal_dm_poll_interval_s == 45.0
    assert cfg.gal_dm_max_los_override == 137.5
    assert cfg.gal_dm_max_age_s == 300.0


def test_config_from_yaml_defaults_when_block_missing(
    tmp_path: Path,
) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "coinc": {
            "bind": {"host": "127.0.0.1", "port": 11500},
            "csv_dir_c1": str(tmp_path / "c1"),
            "csv_dir_c2": str(tmp_path / "c2"),
            "event_archive_root": str(tmp_path / "events"),
            "trigger_criteria_path": str(tmp_path / "c.yaml"),
        }
    }))
    cfg = CoincidencerConfig.from_yaml(p)
    assert cfg.gal_dm_etcd_key == "/mon/array/gal_dm"
    assert cfg.gal_dm_poll_interval_s == 30.0
    assert cfg.gal_dm_max_los_override is None
    assert cfg.gal_dm_max_age_s == 600.0


# ---------------------------------------------------------------------------
# Override + freshness gating
# ---------------------------------------------------------------------------


def _make_service(
    cfg: CoincidencerConfig,
    stub_store: _StubStore,
) -> CoincidencerService:
    return CoincidencerService(
        config=cfg,
        mon_store=stub_store,
        broadcaster=_NoopBroadcaster(),
    )


def test_override_pins_gal_dm_regardless_of_etcd(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, gal_dm_max_los_override=99.0)
    svc = _make_service(cfg, _StubStore(value=None))
    assert svc._current_gal_dm_max_los() == 99.0


def test_no_poll_yet_returns_none(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    svc = _make_service(cfg, _StubStore(value=None))
    assert svc._current_gal_dm_max_los() is None


def test_stale_value_ages_out_after_max_age(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path, gal_dm_max_age_s=0.1)
    svc = _make_service(cfg, _StubStore(value=42.0))
    # Simulate a successful poll N seconds ago.
    svc._gal_dm_value_pc_cc = 42.0
    import time as _time
    svc._gal_dm_fetched_at_mono = _time.monotonic() - 1.0
    assert svc._current_gal_dm_max_los() is None
    # Fresh poll → still valid.
    svc._gal_dm_fetched_at_mono = _time.monotonic()
    assert svc._current_gal_dm_max_los() == 42.0


# ---------------------------------------------------------------------------
# Poll loop behaviour
# ---------------------------------------------------------------------------


async def _run_poll_loop_briefly(
    svc: CoincidencerService, duration_s: float = 0.25,
) -> None:
    task = asyncio.create_task(svc._gal_dm_poll_loop())
    try:
        await asyncio.sleep(duration_s)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_gal_dm_poll_loop_updates_cache(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    stub = _StubStore(value=137.5)
    svc = _make_service(cfg, stub)
    asyncio.run(_run_poll_loop_briefly(svc))
    assert svc._gal_dm_polls_ok >= 1
    assert svc._gal_dm_value_pc_cc == 137.5
    assert svc._current_gal_dm_max_los() == 137.5


def test_gal_dm_poll_loop_handles_missing_key(tmp_path: Path) -> None:
    cfg = _minimal_config(tmp_path)
    stub = _StubStore(value=None)
    svc = _make_service(cfg, stub)
    asyncio.run(_run_poll_loop_briefly(svc))
    assert svc._gal_dm_polls_fail >= 1
    assert svc._gal_dm_polls_ok == 0
    assert svc._current_gal_dm_max_los() is None


def test_gal_dm_poll_loop_ignores_zero_or_negative(
    tmp_path: Path,
) -> None:
    cfg = _minimal_config(tmp_path)
    stub = _StubStore(value=0.0)
    svc = _make_service(cfg, stub)
    asyncio.run(_run_poll_loop_briefly(svc))
    # Sanity-rejected: 0.0 is not a valid gal_dm.
    assert svc._gal_dm_polls_fail >= 1
    assert svc._gal_dm_value_pc_cc is None


# ---------------------------------------------------------------------------
# mon_publish_loop integration
# ---------------------------------------------------------------------------


async def _run_mon_publish_briefly(
    svc: CoincidencerService, duration_s: float = 0.15,
) -> None:
    task = asyncio.create_task(svc._mon_publish_loop())
    try:
        await asyncio.sleep(duration_s)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_mon_publish_payload_includes_gal_dm(tmp_path: Path) -> None:
    cfg = _minimal_config(
        tmp_path,
        mon_publish_interval_s=0.05,
        gal_dm_max_los_override=200.0,
    )
    stub = _StubStore(value=None)
    svc = _make_service(cfg, stub)
    svc._started_unix = 1.0
    asyncio.run(_run_mon_publish_briefly(svc))
    assert stub.put_count >= 1
    payload = stub.last_put
    assert "gal_dm_max_los_pc_cc" in payload
    assert payload["gal_dm_max_los_pc_cc"] == 200.0
    assert "gal_dm_polls_ok" in payload
    assert "gal_dm_polls_fail" in payload
