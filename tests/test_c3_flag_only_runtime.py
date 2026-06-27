"""C3 runtime keep/delete override (``/cmd/c3/flag_only``).

The dashboard's "C3 reject mode" toggle writes ``/cmd/c3/flag_only``;
C3 must read it per event and let it override the configured / CLI
``flag_only`` default, fail-safe in every direction (missing /
malformed / etcd error all keep the configured value, which itself
defaults to the safe ``True``).

These tests drive :class:`dsart.services.c3.C3Service` with a fake
mon-store and a hand-built candidate dir so no etcd / corr nodes are
touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.services import c3 as c3mod                             # noqa: E402


class FakeStore:
    """Mon-store stand-in: serves get_dict from a dict, records puts."""

    def __init__(self, responses: Optional[dict[str, Any]] = None) -> None:
        self._responses = dict(responses or {})
        self.puts: list[tuple[str, dict]] = []

    def put_dict(self, key: str, value) -> None:
        self.puts.append((key, dict(value)))

    def get_dict(self, key: str):
        return self._responses.get(key)


def _make_service(tmp_path: Path, responses=None, *, flag_only_cfg=True):
    cfg = c3mod.C3Config(
        archive_root=tmp_path / "candidates",
        rejected_root=tmp_path / "candidates_rejected",
        state_path=tmp_path / "state.json",
        fired_injection_log=None,
        corr_nodes={},               # no broadcaster constructed
        flag_only=flag_only_cfg,
    )
    svc = c3mod.C3Service(cfg, mon_store=FakeStore(responses or {}))
    return svc, cfg


# ---------------------------------------------------------------------------
# _effective_flag_only
# ---------------------------------------------------------------------------


def test_effective_flag_only_uses_config_when_key_missing(tmp_path) -> None:
    svc, _ = _make_service(tmp_path, responses={}, flag_only_cfg=True)
    assert svc._effective_flag_only() is True
    svc2, _ = _make_service(tmp_path, responses={}, flag_only_cfg=False)
    assert svc2._effective_flag_only() is False


def test_effective_flag_only_override_wins(tmp_path) -> None:
    # config says delete (False), but the etcd override says keep (True).
    svc, _ = _make_service(
        tmp_path,
        responses={c3mod.C3_FLAG_ONLY_KEY: {"flag_only": True}},
        flag_only_cfg=False,
    )
    assert svc._effective_flag_only() is True

    # config says keep (True), override says delete (False).
    svc2, _ = _make_service(
        tmp_path,
        responses={c3mod.C3_FLAG_ONLY_KEY: {"flag_only": False}},
        flag_only_cfg=True,
    )
    assert svc2._effective_flag_only() is False


def test_effective_flag_only_malformed_override_falls_back(tmp_path) -> None:
    for bogus in ({"foo": 1}, [1, 2], "x", 7, None):
        svc, _ = _make_service(
            tmp_path,
            responses={c3mod.C3_FLAG_ONLY_KEY: bogus},
            flag_only_cfg=True,
        )
        assert svc._effective_flag_only() is True, repr(bogus)


# ---------------------------------------------------------------------------
# process_event honours the live override on the REJECT path
# ---------------------------------------------------------------------------


def _seed_reject_event(svc, cfg, name: str = "260101abcd") -> Path:
    """Create a completed-event dir + force a REJECT decision.

    We stub compute_metrics/decide so the test is independent of the
    cube-veto internals; the point under test is the flag_only gating.
    """
    ev_dir = cfg.archive_root / name
    (ev_dir / "Level3").mkdir(parents=True, exist_ok=True)
    (ev_dir / "Level3" / f"{name}.json").write_text("{}")
    (ev_dir / "cubes").mkdir(parents=True, exist_ok=True)
    (ev_dir / "cubes" / "x.npz").write_bytes(b"\x00" * 8)
    return ev_dir


class _RejectDecision:
    action = "REJECT"
    keep = False
    rules_fired = ("R4",)
    notes = "stub"


class _Metrics:
    def __init__(self) -> None:
        self.ok = True


def _patch_veto(monkeypatch, *, is_inj=False):
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: is_inj)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: _Metrics())
    monkeypatch.setattr(c3mod, "decide", lambda *a, **k: _RejectDecision())


def test_reject_flag_only_override_blocks_delete(tmp_path, monkeypatch) -> None:
    # config = delete (flag_only False), but override flips to keep-only:
    # the destructive cleanup must NOT run.
    svc, cfg = _make_service(
        tmp_path,
        responses={c3mod.C3_FLAG_ONLY_KEY: {"flag_only": True}},
        flag_only_cfg=False,
    )
    _patch_veto(monkeypatch)
    name = "260101keep"
    ev_dir = _seed_reject_event(svc, cfg, name)
    rec = svc.process_event(name)
    assert rec["flag_only"] is True
    assert rec["flag_only_source"] == "etcd_override"
    assert rec["reject"] == {"flag_only": True, "no_action_taken": True}
    # The cube is still present (no destructive action taken).
    assert (ev_dir / "cubes" / "x.npz").exists()
    assert svc._counters["rejected_flagged_only"] == 1
    assert svc._counters["rejected"] == 0


def test_reject_override_enables_delete(tmp_path, monkeypatch) -> None:
    # config = keep-only (safe), override flips to delete: the
    # conservative cleanup runs (cube deleted, dir moved aside).
    svc, cfg = _make_service(
        tmp_path,
        responses={c3mod.C3_FLAG_ONLY_KEY: {"flag_only": False}},
        flag_only_cfg=True,
    )
    _patch_veto(monkeypatch)
    name = "260101del"
    ev_dir = _seed_reject_event(svc, cfg, name)
    rec = svc.process_event(name)
    assert rec["flag_only"] is False
    assert rec["flag_only_source"] == "etcd_override"
    # Cube deleted; metadata moved to rejected_root.
    assert not (ev_dir / "cubes" / "x.npz").exists()
    assert (cfg.rejected_root / name / "Level3").exists()
    assert svc._counters["rejected"] == 1
