"""C3 skips voltage collection only for CONFIRMED injections (2026-11-11).

``event_is_injection``'s fired-log backstop is deliberately loose (any
sky position, +120 s) because it only ever exempts an event from the
veto. On 2026-11-09 it labelled real events at DM 149-1035 as injections
because a probe had fired within two minutes. Skipping collection loses
data when wrong, so it keys on :func:`cube_veto.confirmed_injection_id`:
C2's marker, a C1 ``inj_id``, or a tight DM + sky + time match.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

SRC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc import cube_veto as CV                             # noqa: E402
from dsart.coinc.cube_veto import CubeMetrics                       # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402

T_FIRE = 1_794_240_146.0                 # 261109mvlp's probe
MJD_EV = 40587.0 + (T_FIRE + 4.3) / 86400.0


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


def _fired_log(tmp: Path) -> Path:
    p = tmp / "fired.jsonl"
    p.write_text(json.dumps(dict(inj_id="cal_probe_dm500_w8", fired_at_unix=T_FIRE, dm_pc_cm3=500.0,
                                 l_rad=-0.009, m_rad=-0.009, width_samples=8,
                                 fluence_jy_ms=5e-4)) + "\n")
    return p


def _event(root: Path, name: str, *, dm: float, l: float, m: float, mjd: float = MJD_EV,
           inj_id: str = "", l3: Optional[dict] = None) -> Path:
    ev = root / name
    (ev / "Level2").mkdir(parents=True)
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level3" / f"{name}.json").write_text(json.dumps(l3 or {}))
    with open(ev / "Level2" / f"C1_window_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mjd", "snr", "dm_pc_cc", "l_rad", "m_rad", "inj_id"])
        w.writeheader()
        w.writerow(dict(mjd=mjd, snr=20.0, dm_pc_cc=dm, l_rad=l, m_rad=m, inj_id=inj_id))
    return ev


def test_confirmed_injection_id_sources_and_the_loose_case(tmp_path: Path) -> None:
    log = _fired_log(tmp_path)
    r = tmp_path / "c"
    assert CV.confirmed_injection_id(
        _event(r, "a", dm=900, l=0.0, m=0.0, l3={"injection": {"is_injection": True, "inj_id": "x"}}),
        "a", fired_log_path=log) == "c2:x"
    assert CV.confirmed_injection_id(_event(r, "b", dm=900, l=0, m=0, inj_id="y"), "b",
                                     fired_log_path=log) == "csv:y"
    # 261109mvlp: detected 7% high in DM, same position, 4 s after arming
    assert CV.confirmed_injection_id(_event(r, "c", dm=536.5, l=-0.0089, m=-0.0089), "c",
                                     fired_log_path=log) == "fired:cal_probe_dm500_w8"
    # 261109gmox-like: DM within 10% and 60 s after a probe, but elsewhere in the field
    ev = _event(r, "d", dm=456.6, l=0.0024, m=-0.0085, mjd=MJD_EV + 56 / 86400.0)
    assert CV.event_is_injection(ev, "d", fired_log_path=log)          # the loose backstop
    assert CV.confirmed_injection_id(ev, "d", fired_log_path=log) is None
    # no usable MJD: no time gate, so not confirmed
    assert CV.confirmed_injection_id(_event(r, "e", dm=536.5, l=-0.009, m=-0.009, mjd=0.0), "e",
                                     fired_log_path=log) is None


def _service(tmp: Path, *, collect: bool):
    cfg = c3mod.C3Config(archive_root=tmp / "c", rejected_root=tmp / "r", state_path=tmp / "s.json",
                         fired_injection_log=_fired_log(tmp), corr_nodes={}, flag_only=True,
                         collect_injection_voltages=collect)
    svc = c3mod.C3Service(cfg, mon_store=FakeStore())
    svc._broadcaster = FakeBroadcaster()
    return svc


def _patch(monkeypatch, planned: list) -> None:
    metrics = CubeMetrics(ok=True, tz_trig=40.0, tz_apex=40.0, imgz_apex=15.0, img_off_apex=0.0,
                          t_shift=0, dm_shift_trials=0, dm_edge=0)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: metrics)
    monkeypatch.setattr(c3mod, "plan_fragments", lambda **k: planned.append(k["event_name"]) or [])


def test_confirmed_injection_is_kept_without_collection(tmp_path: Path, monkeypatch) -> None:
    planned: list = []
    _patch(monkeypatch, planned)
    svc = _service(tmp_path, collect=False)
    _event(tmp_path / "c", "inj", dm=536.5, l=-0.0089, m=-0.0089)
    rec = svc.process_event("inj")
    assert rec["action"] == "KEEP" and rec["is_injection"] is True
    assert planned == []                                               # nothing collected
    assert rec["keep"]["skipped"] == "confirmed injection (fired:cal_probe_dm500_w8)"
    assert svc._broadcaster.sent == ["inj"]                           # staging released


def test_loosely_labelled_event_is_still_collected(tmp_path: Path, monkeypatch) -> None:
    planned: list = []
    _patch(monkeypatch, planned)
    svc = _service(tmp_path, collect=False)
    _event(tmp_path / "c", "real", dm=456.6, l=0.0024, m=-0.0085, mjd=MJD_EV + 56 / 86400.0)
    rec = svc.process_event("real")
    assert rec["is_injection"] is True                                 # exempt from the veto...
    assert planned == ["real"] and "skipped" not in rec["keep"]       # ...but its voltages collected


def test_default_config_still_collects_injections(tmp_path: Path, monkeypatch) -> None:
    planned: list = []
    _patch(monkeypatch, planned)
    svc = _service(tmp_path, collect=True)
    _event(tmp_path / "c", "inj", dm=536.5, l=-0.0089, m=-0.0089)
    svc.process_event("inj")
    assert planned == ["inj"]
