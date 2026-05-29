"""Tests for the C2 dumps-enabled runtime gate (M7.4 Phase 6c).

Exercises two surfaces:

1. :class:`dsart.services.coincidencer.DumpsGate` directly — cache
   TTL, default-missing-key (fail-OPEN), etcd-error (fail-OPEN +
   rate-limited WARNING).
2. :class:`dsart.services.coincidencer.CoincidencerService` end-to-end
   through ``_on_batch`` — verifies the suppression path replaces
   ``DUMP`` with ``WOULD-DUMP``, skips the UDP fan-out, rolls back the
   per-class holdoff timer, and leaves the ``log_only`` action
   completely unchanged regardless of the gate state.

The tests deliberately avoid touching the real etcd: a tiny
``_FakeStore`` covers ``get_dict`` for the gate, and a counting
``_RecordingBroadcaster`` stands in for the UDP fan-out.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from dsart.coinc import wire
from dsart.services.coincidencer import (
    DUMPS_ENABLED_KEY,
    CoincidencerConfig,
    CoincidencerService,
    DumpsGate,
)


def asyncio_test(func):
    """Plain-pytest async runner (mirrors test_c2_coincidencer_on_batch)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal DsaStore stand-in: a programmable ``get_dict``.

    ``responses`` maps full etcd keys to the value the next call to
    ``get_dict(key)`` should return. ``raise_on`` (key → Exception
    instance) overrides ``responses`` and forces the matching
    ``get_dict`` call to raise — used to exercise the gate's
    fail-OPEN behaviour on etcd outages.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, Any]] = None,
        raise_on: Optional[Dict[str, Exception]] = None,
    ) -> None:
        self.responses: Dict[str, Any] = dict(responses or {})
        self.raise_on: Dict[str, Exception] = dict(raise_on or {})
        self.gets: List[str] = []
        self.puts: List[tuple] = []

    def get_dict(self, key: str) -> Any:
        self.gets.append(key)
        if key in self.raise_on:
            raise self.raise_on[key]
        return self.responses.get(key)

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        self.puts.append((key, dict(value)))


class _RecordingBroadcaster:
    """Stand-in for :class:`TriggerBroadcaster`. Records every call so
    the suppression tests can assert ``broadcast`` was NOT called.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def broadcast(self, **kwargs) -> Dict[tuple, bool]:
        self.calls.append(dict(kwargs))
        # Pretend two halves × one host responded OK.
        return {(1, 0): True, (1, 1): True}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# DumpsGate unit tests
# ---------------------------------------------------------------------------


class TestDumpsGateDefault:
    def test_default_key_is_canonical(self) -> None:
        # Pinned so the dashboard's copy in dumps_gate.py cannot drift.
        assert DUMPS_ENABLED_KEY == "/cmd/c2/dumps_enabled"

    def test_missing_key_returns_enabled_true(self) -> None:
        store = _FakeStore(responses={})  # key missing
        gate = DumpsGate(store)
        assert gate.enabled() is True
        # The gate must have actually read etcd; subsequent reads
        # within the TTL hit the cache.
        assert store.gets == [DUMPS_ENABLED_KEY]

    def test_explicit_enabled_false_is_honoured(self) -> None:
        store = _FakeStore(
            responses={DUMPS_ENABLED_KEY: {
                "enabled": False, "ts": 1.0, "actor": "ops", "reason": "rfi",
            }},
        )
        gate = DumpsGate(store)
        assert gate.enabled() is False

    def test_explicit_enabled_true_is_honoured(self) -> None:
        store = _FakeStore(
            responses={DUMPS_ENABLED_KEY: {
                "enabled": True, "ts": 2.0, "actor": "ops", "reason": "go",
            }},
        )
        gate = DumpsGate(store)
        assert gate.enabled() is True

    def test_malformed_payload_fails_open(self) -> None:
        # Dict without "enabled" key, list, scalar, all → True.
        for bogus in ({"foo": "bar"}, [1, 2, 3], "garbage", 42, None):
            store = _FakeStore(responses={DUMPS_ENABLED_KEY: bogus})
            assert DumpsGate(store).enabled() is True, repr(bogus)

    def test_none_store_fails_open(self) -> None:
        # Construction with store=None must not raise; enabled() → True.
        gate = DumpsGate(None)
        assert gate.enabled() is True


class TestDumpsGateCacheTTL:
    def test_cache_serves_repeated_reads_within_ttl(self) -> None:
        # Inject a deterministic monotonic clock.
        ticks = [0.0]
        store = _FakeStore(
            responses={DUMPS_ENABLED_KEY: {"enabled": False}},
        )
        gate = DumpsGate(store, cache_ttl_s=0.2, now=lambda: ticks[0])
        assert gate.enabled() is False
        assert store.gets == [DUMPS_ENABLED_KEY]
        # Advance 100 ms — still inside the TTL.
        ticks[0] = 0.1
        assert gate.enabled() is False
        assert store.gets == [DUMPS_ENABLED_KEY], "should not re-read"
        # Right at the TTL boundary: 0.2 - 0.0 = 0.2; (now - cached_at) < ttl
        # is False (the comparison is strict <) → re-read.
        ticks[0] = 0.2
        gate.enabled()
        assert store.gets == [DUMPS_ENABLED_KEY, DUMPS_ENABLED_KEY]

    def test_cache_refreshes_after_ttl_changes_value(self) -> None:
        ticks = [0.0]
        store = _FakeStore(
            responses={DUMPS_ENABLED_KEY: {"enabled": False}},
        )
        gate = DumpsGate(store, cache_ttl_s=0.5, now=lambda: ticks[0])
        assert gate.enabled() is False
        # Operator flips the key.
        store.responses[DUMPS_ENABLED_KEY] = {"enabled": True}
        # Within TTL: still serves False.
        ticks[0] = 0.4
        assert gate.enabled() is False
        # After TTL elapses, the new value is picked up.
        ticks[0] = 0.6
        assert gate.enabled() is True

    def test_invalidate_forces_immediate_reread(self) -> None:
        ticks = [10.0]
        store = _FakeStore(
            responses={DUMPS_ENABLED_KEY: {"enabled": True}},
        )
        gate = DumpsGate(store, cache_ttl_s=999.0, now=lambda: ticks[0])
        assert gate.enabled() is True
        # Flip the key + invalidate — the next enabled() must re-read
        # even though the TTL has not elapsed.
        store.responses[DUMPS_ENABLED_KEY] = {"enabled": False}
        gate.invalidate()
        assert gate.enabled() is False
        assert store.gets == [DUMPS_ENABLED_KEY, DUMPS_ENABLED_KEY]


class TestDumpsGateFailOpen:
    def test_etcd_error_fails_open(self, caplog) -> None:
        store = _FakeStore(
            raise_on={DUMPS_ENABLED_KEY: RuntimeError("etcd down")},
        )
        ticks = [0.0]
        gate = DumpsGate(store, cache_ttl_s=0.0, now=lambda: ticks[0])
        with caplog.at_level(logging.WARNING, logger="dsart.services.coincidencer"):
            # First read: warns + returns True.
            assert gate.enabled() is True
        assert gate.fail_count == 1
        assert any(
            "dumps_gate" in rec.message and "fail-OPEN" in rec.message
            for rec in caplog.records
        )

    def test_etcd_error_warn_is_rate_limited(self, caplog) -> None:
        store = _FakeStore(
            raise_on={DUMPS_ENABLED_KEY: RuntimeError("etcd down")},
        )
        ticks = [0.0]
        # Force cache_ttl_s=0 so every enabled() call re-reads etcd,
        # but rate-limit the WARNING log to 60 s.
        gate = DumpsGate(
            store, cache_ttl_s=0.0, now=lambda: ticks[0],
            warn_rate_limit_s=60.0,
        )
        with caplog.at_level(logging.WARNING, logger="dsart.services.coincidencer"):
            gate.enabled()
            ticks[0] = 30.0                                       # < 60 s
            gate.enabled()
            ticks[0] = 31.0
            gate.enabled()
        # 3 reads attempted, 3 failures, but only ONE WARNING emitted.
        assert gate.fail_count == 3
        warns = [
            rec for rec in caplog.records
            if "dumps_gate" in rec.message and "fail-OPEN" in rec.message
        ]
        assert len(warns) == 1, (
            f"expected 1 rate-limited warn, got {len(warns)}: "
            f"{[r.message for r in warns]}"
        )
        # Now jump past the rate-limit window — a second warning fires.
        ticks[0] = 100.0
        with caplog.at_level(logging.WARNING, logger="dsart.services.coincidencer"):
            gate.enabled()
        warns = [
            rec for rec in caplog.records
            if "dumps_gate" in rec.message and "fail-OPEN" in rec.message
        ]
        assert len(warns) == 2

    def test_get_dict_returns_none_on_outage_is_fail_open(self) -> None:
        # The C2 service wraps DsaStore in _StoreWrapper, whose
        # get_dict returns None on transport failure rather than
        # raising. The gate must treat that the same as missing key.
        store = _FakeStore(responses={DUMPS_ENABLED_KEY: None})
        gate = DumpsGate(store)
        assert gate.enabled() is True


# ---------------------------------------------------------------------------
# CoincidencerService end-to-end (suppression path / holdoff rollback)
# ---------------------------------------------------------------------------


def _criteria_file(
    tmp_path: Path,
    *,
    holdoff_s: float = 30.0,
    include_log_only: bool = True,
) -> Path:
    p = tmp_path / "c.yaml"
    parts = [
        "trigger_classes:",
        "  - name: bright_frb",
        "    require:",
        "      n_events_min: 1",
        "    action: dump_all_gpus",
        f"    holdoff_s: {holdoff_s}",
    ]
    if include_log_only:
        parts.extend([
            "  - name: noise",
            "    require:",
            "      n_events_min: 1",
            "      snr_max_max: -1.0",  # never matches by default
            "    action: log_only",
        ])
    p.write_text("\n".join(parts) + "\n")
    return p


def _batch(
    mjd_start: float,
    n: int,
    *,
    event_specnum_start: int = 0,
    sample_period_us: float = 100_000.0,
    sample_period_specnum: int = 1,
    cube_id: int = 0,
    snr: float = 12.5,
) -> wire.C1Batch:
    header = wire.build_header(
        cube_id=cube_id,
        event_specnum_start=event_specnum_start,
        mjd_start=mjd_start,
        sample_period_specnum=sample_period_specnum,
        sample_period_us=sample_period_us,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=1,
        gpu_half=0,
        n_candidates=n,
    )
    rows = tuple(
        wire.C1CandidateRow(
            snr=snr + 0.1 * i,
            l_rad=0.0, m_rad=0.0, l_pix=0, m_pix=0,
            dm_pc_cc=100.0, dm_idx_global=0, fine_dm_idx=0,
            event_specnum=event_specnum_start + i,
            width_samples=4, kernel_id="unit:d1:b4", flags=0,
        )
        for i in range(n)
    )
    return wire.C1Batch(header=header, candidates=rows)


def _make_service(
    tmp_path: Path,
    *,
    holdoff_s: float = 30.0,
    dumps_enabled: Optional[bool] = True,
) -> tuple[CoincidencerService, _RecordingBroadcaster, _FakeStore]:
    """Build a CoincidencerService with a controlled DumpsGate.

    ``dumps_enabled`` chooses what the fake etcd will report to the
    gate:
      * True  → key present + enabled=True
      * False → key present + enabled=False
      * None  → key MISSING (default-fail-OPEN behaviour)
    """
    cfg = CoincidencerConfig(
        bind_host="127.0.0.1",
        bind_port=0,
        window_s=5.0,
        csv_dir_c1=tmp_path / "c1",
        csv_dir_c2=tmp_path / "c2",
        event_archive_root=tmp_path / "events",
        trigger_criteria_path=_criteria_file(
            tmp_path, holdoff_s=holdoff_s,
        ),
        name_allocator_offline=True,
        gal_dm_poll_interval_s=60.0,
        startup_grace_s=0.0,  # disable startup grace for dumps-gate tests
    )
    if dumps_enabled is None:
        responses: Dict[str, Any] = {}
    else:
        responses = {DUMPS_ENABLED_KEY: {
            "enabled": bool(dumps_enabled),
            "ts": 0.0,
            "actor": "test",
            "reason": "unit",
        }}
    fake_store = _FakeStore(responses=responses)
    gate = DumpsGate(fake_store, cache_ttl_s=0.0)  # bypass cache for tests
    bc = _RecordingBroadcaster()
    svc = CoincidencerService(
        config=cfg,
        mon_store=fake_store,
        broadcaster=bc,
        dumps_gate=gate,
    )
    return svc, bc, fake_store


@asyncio_test
async def test_dumps_enabled_path_broadcasts_normally(tmp_path: Path) -> None:
    """Sanity baseline: dumps_enabled=True → exactly one broadcast call,
    triggers_dump counter incremented, triggers_suppressed stays at 0.
    """
    svc, bc, _ = _make_service(tmp_path, dumps_enabled=True)
    await svc._on_batch(
        _batch(mjd_start=60781.0, n=2, snr=20.0),
        peer_repr="x",
    )
    assert len(bc.calls) == 1
    assert svc._counters["triggers_dump"] == 1
    assert svc._counters["triggers_suppressed"] == 0
    # Archive directory + C2 CSV row must exist on the dumped path.
    events_root = tmp_path / "events"
    assert events_root.exists()
    assert any(events_root.iterdir()), "no event archive directory created"


@asyncio_test
async def test_suppressed_path_writes_archive_but_no_broadcast(
    tmp_path: Path, caplog,
) -> None:
    """Suppression: archive row written, NO broadcast, WOULD-DUMP logged
    instead of DUMP, suppressed counter incremented.
    """
    svc, bc, _ = _make_service(tmp_path, dumps_enabled=False)
    with caplog.at_level(logging.INFO, logger="dsart.services.coincidencer"):
        await svc._on_batch(
            _batch(mjd_start=60781.0, n=2, snr=20.0),
            peer_repr="x",
        )
    # No UDP fan-out happened.
    assert bc.calls == []
    # Counters: suppressed bumped, dump not bumped.
    assert svc._counters["triggers_dump"] == 0
    assert svc._counters["triggers_suppressed"] == 1
    # Archive row still written.
    events_root = tmp_path / "events"
    assert any(events_root.iterdir()), "archive should still be created"
    # Log line is WOULD-DUMP, not DUMP. The exact format is
    # documented to the operator (their grep patterns key off it).
    msgs = [rec.message for rec in caplog.records]
    assert any(
        "WOULD-DUMP" in m and "suppressed=dumps_disabled" in m
        and "class=bright_frb" in m
        for m in msgs
    ), f"WOULD-DUMP log line not found; messages={msgs}"
    assert not any(
        m.startswith("DUMP class=") for m in msgs
    ), "DUMP line should NOT be emitted when suppressed"


@asyncio_test
async def test_holdoff_not_advanced_when_suppressed(tmp_path: Path) -> None:
    """The headline correctness property: with a 30 s holdoff and the
    gate suppressing dumps, a second cluster arriving 0 s later must
    STILL fire the WOULD-DUMP path (because the holdoff was rolled
    back). Without rollback, the second match would be eaten by the
    holdoff and the operator would never see it.
    """
    svc, bc, _ = _make_service(
        tmp_path, holdoff_s=30.0, dumps_enabled=False,
    )
    # Two batches → two distinct clusters with no in-window overlap.
    # Using event_specnum_start to push the rows into non-overlapping
    # (l, m) groups.
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.0 / 86400.0,
            n=1, snr=20.0, event_specnum_start=100,
        ),
        peer_repr="x",
    )
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.01 / 86400.0,
            n=1, snr=20.0, event_specnum_start=200,
        ),
        peer_repr="y",
    )
    # Both attempts must have walked the suppressed path; neither
    # should have hit the broadcaster.
    assert bc.calls == []
    # Both clusters fired WOULD-DUMP → suppressed counter is 2.
    assert svc._counters["triggers_suppressed"] == 2
    # And the holdoff entry for the class is empty (or at least no
    # later than the very first match) so a third match would
    # immediately fire too.
    assert svc._criteria.last_fired_at("bright_frb") is None


@asyncio_test
async def test_holdoff_advances_normally_when_enabled(tmp_path: Path) -> None:
    """Counter-test: with dumps_enabled=True + a long holdoff, a
    rapid-fire pair of matches produces exactly ONE broadcast (the
    second match is eaten by the holdoff). Confirms our suppression
    rollback isn't a side-effect of unrelated changes.
    """
    svc, bc, _ = _make_service(
        tmp_path, holdoff_s=30.0, dumps_enabled=True,
    )
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.0 / 86400.0,
            n=1, snr=20.0, event_specnum_start=100,
        ),
        peer_repr="x",
    )
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.01 / 86400.0,
            n=1, snr=20.0, event_specnum_start=200,
        ),
        peer_repr="y",
    )
    assert len(bc.calls) == 1, (
        f"expected exactly 1 broadcast (holdoff should eat the 2nd); "
        f"got {len(bc.calls)}"
    )
    assert svc._counters["triggers_dump"] == 1
    assert svc._criteria.last_fired_at("bright_frb") is not None


@asyncio_test
async def test_suppression_then_enable_fires_immediately(
    tmp_path: Path,
) -> None:
    """End-to-end of the operator's mental model: while suppressed,
    a real burst is recorded as WOULD-DUMP only; the moment the
    operator flips dumps back on, the very next cluster fires for
    real. The holdoff rollback is what makes "the next trigger"
    actually mean "the next trigger" (not "the next one outside the
    30 s holdoff window left over from the suppressed event").
    """
    svc, bc, fake_store = _make_service(
        tmp_path, holdoff_s=30.0, dumps_enabled=False,
    )
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.0 / 86400.0,
            n=1, snr=20.0, event_specnum_start=100,
        ),
        peer_repr="x",
    )
    # Flip the gate; invalidate the cache (zero TTL already does so
    # but be explicit).
    fake_store.responses[DUMPS_ENABLED_KEY] = {"enabled": True}
    svc._dumps_gate.invalidate()
    await svc._on_batch(
        _batch(
            mjd_start=60781.0 + 0.01 / 86400.0,
            n=1, snr=20.0, event_specnum_start=200,
        ),
        peer_repr="y",
    )
    assert svc._counters["triggers_suppressed"] == 1
    assert svc._counters["triggers_dump"] == 1
    assert len(bc.calls) == 1


@asyncio_test
async def test_log_only_path_unchanged_regardless_of_gate(tmp_path: Path) -> None:
    """The dumps gate must NOT touch the ``log_only`` action — it
    targets only the UDP fan-out + holdoff state of dump_all_gpus
    classes. We override the criteria so the only matching class is
    log_only, then drive both gate states and assert nothing
    differs between them.
    """
    for enabled in (True, False):
        # Fresh service per gate state so counters are independent.
        cfg = CoincidencerConfig(
            bind_host="127.0.0.1",
            bind_port=0,
            window_s=5.0,
            csv_dir_c1=tmp_path / f"c1_{enabled}",
            csv_dir_c2=tmp_path / f"c2_{enabled}",
            event_archive_root=tmp_path / f"events_{enabled}",
            trigger_criteria_path=_log_only_criteria_file(
                tmp_path, suffix=str(enabled),
            ),
            name_allocator_offline=True,
            gal_dm_poll_interval_s=60.0,
            startup_grace_s=0.0,  # disable startup grace for these tests
        )
        store = _FakeStore(responses={
            DUMPS_ENABLED_KEY: {"enabled": enabled},
        })
        gate = DumpsGate(store, cache_ttl_s=0.0)
        bc = _RecordingBroadcaster()
        svc = CoincidencerService(
            config=cfg, mon_store=store,
            broadcaster=bc, dumps_gate=gate,
        )
        await svc._on_batch(
            _batch(mjd_start=60781.0, n=2, snr=20.0),
            peer_repr="x",
        )
        # log_only NEVER broadcasts.
        assert bc.calls == []
        # And the suppressed counter must NOT bump on log_only events
        # regardless of the gate setting.
        assert svc._counters["triggers_suppressed"] == 0
        assert svc._counters["triggers_log_only"] == 1
        assert svc._counters["triggers_dump"] == 0


def _log_only_criteria_file(tmp_path: Path, *, suffix: str) -> Path:
    """Build a criteria file whose only class is log_only."""
    p = tmp_path / f"c_logonly_{suffix}.yaml"
    p.write_text(
        "trigger_classes:\n"
        "  - name: bright_log\n"
        "    require:\n"
        "      n_events_min: 1\n"
        "    action: log_only\n"
        "    holdoff_s: 30.0\n"
    )
    return p
