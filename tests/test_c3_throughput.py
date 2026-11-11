"""C3 keeps up with 2x KEEPs (2026-11-11).

A 2x KEEP took ~23 min inline (11 min of serial rsync + 11 min of two
filterbank runs) and every later event waited behind it, while each
dump's corr staging aged towards the 2 h orphan TTL and the 120 GiB cap.
C3 now processes events oldest first, can pull fragments in parallel,
and with ``filterbank.background`` hands the filterbank to a worker.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc.cal_hdf5_archive import CalHdf5Config               # noqa: E402
from dsart.coinc.cube_veto import CubeMetrics                       # noqa: E402
from dsart.coinc.filterbank import FilterbankConfig                 # noqa: E402
from dsart.coinc.voltage_collect import FragmentPlan                # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class FakeStore:
    def put_dict(self, key: str, value) -> None:
        pass

    def get_dict(self, key: str) -> Optional[Any]:
        return None


class FakeBroadcaster:
    def __init__(self) -> None:
        self.sent: list = []

    def broadcast(self, name, *a) -> None:
        self.sent.append(name)

    def close(self) -> None:
        pass


class FakeSlack:
    def __init__(self) -> None:
        self.followups: list = []

    def post_keep_card(self, name, c2row, ev_dir=None):
        return {"ok": True, "ts": f"ts-{name}"}

    def post_keep_followup(self, name, ev_dir, fb_report, report, thread_ts=None):
        self.followups.append((name, fb_report.get("ok"), report.get("n_fragments_present"), thread_ts))
        return {"ok": True}


def _event(root: Path, name: str, mtime: float) -> Path:
    ev = root / name
    (ev / "Level3").mkdir(parents=True)
    l3 = ev / "Level3" / f"{name}.json"
    l3.write_text(json.dumps({"c2": {"dm_median": 500.0}}))
    os.utime(l3, (mtime, mtime))
    return ev


def _service(tmp: Path, *, background: bool) -> "c3mod.C3Service":
    cfg = c3mod.C3Config(
        archive_root=tmp / "c", rejected_root=tmp / "r", state_path=tmp / "s.json",
        fired_injection_log=None, corr_nodes={}, flag_only=True,
        filterbank=FilterbankConfig(enabled=True, background=background, dec_key=""),
        cal_hdf5=CalHdf5Config(enabled=False))
    svc = c3mod.C3Service(cfg, mon_store=FakeStore())
    svc._broadcaster = FakeBroadcaster()
    svc._slack = FakeSlack()
    return svc


def _patch(monkeypatch, fb_runs: list) -> None:
    metrics = CubeMetrics(ok=True, tz_trig=40.0, tz_apex=40.0, imgz_apex=15.0, img_off_apex=0.0,
                          t_shift=0, dm_shift_trials=0, dm_edge=0)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: metrics)

    def plans(**k):
        out = []
        for cg in range(2):
            dest = k["dest_dir"] / f"{k['event_name']}_sb{cg:02d}_data.out"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"v")                 # already collected
            out.append(FragmentPlan(cg, f"n{cg}:/x", dest))
        return out
    monkeypatch.setattr(c3mod, "plan_fragments", plans)
    monkeypatch.setattr(c3mod, "plan_manifests", lambda **k: [])
    monkeypatch.setattr(c3mod, "run_for_event",
                        lambda cfg, ev_dir, name, c2row, dec_deg=None:
                        fb_runs.append(name) or {"ok": True})


def test_events_are_processed_oldest_first(tmp_path: Path) -> None:
    svc = _service(tmp_path, background=False)
    root = tmp_path / "c"
    _event(root, "261111solj", 1000.0)             # the oldest, last alphabetically
    _event(root, "261111ayxd", 3000.0)
    _event(root, "261111dapz", 2000.0)
    (root / "261111zzzz").mkdir()                  # no Level3 yet: not offered
    assert svc._completed_events() == ["261111solj", "261111dapz", "261111ayxd"]


def test_inline_filterbank_unchanged(tmp_path: Path, monkeypatch) -> None:
    runs: list = []
    _patch(monkeypatch, runs)
    svc = _service(tmp_path, background=False)
    _event(tmp_path / "c", "ev1", 1000.0)
    rec = svc.process_event("ev1")
    assert rec["filterbank"] == {"ok": True} and runs == ["ev1"]
    assert svc._slack.followups == [("ev1", True, 2, "ts-ev1")]


def test_background_filterbank_releases_staging_and_survives_restart(
        tmp_path: Path, monkeypatch) -> None:
    runs: list = []
    _patch(monkeypatch, runs)
    svc = _service(tmp_path, background=True)
    _event(tmp_path / "c", "ev1", 1000.0)
    _event(tmp_path / "c", "ev2", 2000.0)
    rec1 = svc.process_event("ev1")
    rec2 = svc.process_event("ev2")
    # both decided and their staging released without waiting for a filterbank
    assert rec1["action"] == rec2["action"] == "KEEP"
    assert rec1["filterbank"] == {"queued": True} and runs == []
    assert svc._broadcaster.sent == ["ev1", "ev2"]
    assert svc._slack.followups == []
    state = json.loads((tmp_path / "s.json").read_text())
    assert set(state["filterbank_pending"]) == {"ev1", "ev2"}

    # restart: the pending filterbanks come back from the state file
    svc2 = _service(tmp_path, background=True)
    assert set(svc2._fb_pending) == {"ev1", "ev2"}
    svc2._filterbank_one("ev1")
    svc2._filterbank_one("ev2")
    svc2._filterbank_one("ev2")                    # a duplicate queue entry is a no-op
    assert runs == ["ev1", "ev2"]
    assert svc2._slack.followups == [("ev1", True, 2, "ts-ev1"), ("ev2", True, 2, "ts-ev2")]
    assert json.loads((tmp_path / "s.json").read_text())["filterbank_pending"] == {}
    assert svc2._counters["filterbanks_ok"] == 2


def test_running_service_drains_restored_filterbanks(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import time

    runs: list = []
    _patch(monkeypatch, runs)
    svc = _service(tmp_path, background=True)
    _event(tmp_path / "c", "ev1", 1000.0)
    svc.process_event("ev1")
    svc2 = _service(tmp_path, background=True)    # restart with ev1's filterbank pending

    async def drive() -> None:
        task = asyncio.create_task(svc2.run())
        t0 = time.monotonic()
        while (not runs or svc2._fb_pending) and time.monotonic() - t0 < 10.0:
            await asyncio.sleep(0.05)
        svc2._stop.set()
        await task

    asyncio.run(drive())
    assert runs == ["ev1"] and svc2._fb_pending == {}


def test_2x_config_enables_parallel_pulls_and_background_filterbank() -> None:
    cfg = c3mod.C3Config.from_yaml(REPO / "configs" / "opoints" / "2x" / "dsart_search_rt.yaml")
    assert cfg.collect_parallel == 4
    assert cfg.filterbank.background is True
    one = c3mod.C3Config.from_yaml(REPO / "configs" / "dsart_search_rt.yaml")
    assert one.collect_parallel == 1 and one.filterbank.background is False
