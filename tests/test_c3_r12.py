"""C3 R12: the DM-time ridge narrow-band rule in shadow and reject modes (2026-11-12).

Shadow records the features and the would-be verdict and never changes the
decision; reject turns a KEEP R12 fires on into a REJECT (rule
R12_narrowband). Injections are never rejected by R12. Every evaluated event
gets its DM-time plane archived outside the event dir.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc import dmt_ridge as R                              # noqa: E402
from dsart.coinc.cal_hdf5_archive import CalHdf5Config               # noqa: E402
from dsart.coinc.cube_veto import CubeMetrics                       # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FDM = np.linspace(50.0, 1500.0, 300)
T0, T_PK = 10_000, 10_256


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


def _event(root: Path, name: str, *, narrow: bool, seed: int = 0) -> Path:
    """Two halves of cutouts; a 1340 MHz line through the trigger, or a compact burst."""
    ev = root / name
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level3" / f"{name}.json").write_text(json.dumps({"c2": {"dm_median": 500.0}}))
    (ev / "cutouts").mkdir()
    rng = np.random.default_rng(seed)
    pix = rng.normal(size=(512, FDM.size)).astype(np.float32)
    j0 = int(np.argmin(np.abs(FDM - 500.0)))
    if narrow:
        s = float(R.slope_of_freq(1340.0))
        for j, dm in enumerate(FDM):
            t = 256 + int(round(s * (dm - FDM[j0]) / R.T_EXT_MS))
            if 0 <= t < 512:
                pix[t:t + 2, j] += 9.0
    else:
        pix[256:258, j0 - 1:j0 + 2] += 25.0
    for half, cols in (("s1_g0", slice(0, None, 2)), ("s2_g0", slice(1, None, 2))):
        np.savez(ev / "cutouts" / f"cutout_{half}.npz", pix_lc=pix[:, cols], peak_grid=pix[:, cols] + 3.0,
                 fine_dm_pc_cc=FDM[cols].astype(np.float32), t0_ext=np.int64(T0), t_peak_ext=np.int64(T_PK),
                 width_ext_req=np.int32(2), dm_req=np.float32(FDM[j0]),
                 input_valid_frac=np.ones(FDM[cols].size, np.float32))
    return ev


def _service(tmp: Path, *, mode: str = "shadow", enabled: bool = True):
    cfg = c3mod.C3Config(
        archive_root=tmp / "c", rejected_root=tmp / "r", state_path=tmp / "s.json", fired_injection_log=None,
        corr_nodes={}, flag_only=True, cal_hdf5=CalHdf5Config(enabled=False),
        r12=c3mod.R12Config(enabled=enabled, mode=mode, archive_dir=str(tmp / "dmt")))
    svc = c3mod.C3Service(cfg, mon_store=FakeStore())
    svc._broadcaster = FakeBroadcaster()
    return svc


@pytest.fixture
def keep_everything(monkeypatch):
    metrics = CubeMetrics(ok=True, tz_trig=40.0, tz_apex=40.0, imgz_apex=15.0, img_off_apex=0.0,
                          t_shift=0, dm_shift_trials=0, dm_edge=0)
    monkeypatch.setattr(c3mod, "compute_metrics", lambda *a, **k: metrics)
    monkeypatch.setattr(c3mod, "plan_fragments", lambda **k: [])
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: False)


def test_shadow_records_and_archives_but_never_acts(tmp_path: Path, keep_everything) -> None:
    svc = _service(tmp_path)
    _event(tmp_path / "c", "261112narr", narrow=True)
    rec = svc.process_event("261112narr")
    assert rec["action"] == "KEEP" and "R12_narrowband" not in rec["rules_fired"]
    r12 = rec["r12"]
    assert r12["ok"] and r12["would_reject"] and not r12["acted"] and r12["mode"] == "shadow"
    assert r12["features"]["freq_mhz"] == pytest.approx(1340.0, abs=2.0)
    assert r12["features"]["bw_fit_mhz"] <= 6.0
    arch = Path(r12["archive"])
    assert arch == tmp_path / "dmt" / "261112" / "261112narr.npz" and arch.exists()
    with np.load(arch) as z:
        assert z["plane"].dtype == np.float16 and z["plane"].shape == (FDM.size, 512)
        assert json.loads(str(z["r12"]))["would_reject"] is True
    assert svc._counters["r12_would_reject"] == 1 and svc._counters["r12_rejected"] == 0
    assert json.loads((tmp_path / "c" / "261112narr" / "C3_decision.json").read_text())["r12"]["would_reject"]


def test_compact_burst_passes(tmp_path: Path, keep_everything) -> None:
    svc = _service(tmp_path, mode="reject")
    _event(tmp_path / "c", "261112real", narrow=False, seed=1)
    rec = svc.process_event("261112real")
    assert rec["action"] == "KEEP" and rec["r12"]["ok"] and not rec["r12"]["would_reject"]
    assert Path(rec["r12"]["archive"]).exists()                    # archived either way


def test_reject_mode_rejects_a_narrow_band_keep(tmp_path: Path, keep_everything) -> None:
    svc = _service(tmp_path, mode="reject")
    _event(tmp_path / "c", "261112narr", narrow=True)
    rec = svc.process_event("261112narr")
    assert rec["action"] == "REJECT" and rec["rules_fired"] == ["R12_narrowband"]
    assert rec["r12"]["acted"]
    assert rec["notes"].startswith("R12: narrow-band line at 13")
    assert svc._counters["r12_rejected"] == 1


def test_injections_are_never_rejected_by_r12(tmp_path: Path, keep_everything, monkeypatch) -> None:
    monkeypatch.setattr(c3mod, "event_is_injection", lambda *a, **k: True)
    svc = _service(tmp_path, mode="reject")
    _event(tmp_path / "c", "261112inje", narrow=True)
    rec = svc.process_event("261112inje")
    assert rec["action"] == "KEEP" and rec["r12"]["would_reject"] and not rec["r12"]["acted"]
    assert svc._counters["r12_injection_flagged"] == 1


def test_no_cutouts_or_disabled_gives_no_verdict(tmp_path: Path, keep_everything) -> None:
    svc = _service(tmp_path, mode="reject")
    ev = tmp_path / "c" / "261112none"
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level3" / "261112none.json").write_text("{}")
    rec = svc.process_event("261112none")
    assert rec["action"] == "KEEP" and rec["r12"]["ok"] is False and not rec["r12"]["would_reject"]
    assert svc._counters["r12_no_verdict"] == 1
    off = _service(tmp_path / "off", enabled=False)
    _event(tmp_path / "off" / "c", "261112narr", narrow=True)
    assert "r12" not in off.process_event("261112narr")


def test_r12_errors_never_break_the_decision(tmp_path: Path, keep_everything, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("corrupt cutout")
    monkeypatch.setattr(c3mod, "event_ridge", boom)
    svc = _service(tmp_path, mode="reject")
    _event(tmp_path / "c", "261112narr", narrow=True)
    rec = svc.process_event("261112narr")
    assert rec["action"] == "KEEP" and rec["r12"]["ok"] is False and "corrupt" in rec["r12"]["reason"]
    assert svc._counters["r12_errors"] == 1


def test_config(tmp_path: Path) -> None:
    two = c3mod.C3Config.from_yaml(REPO / "configs" / "opoints" / "2x" / "dsart_search_rt.yaml").r12
    assert two.enabled and two.mode == "shadow" and two.bw_max_mhz == 6.0 and two.f_min == 0.45
    assert two.plane == "pix" and two.archive_dir == "/dataz/dsa110/operations/c3/dmt"
    one = c3mod.C3Config.from_yaml(REPO / "configs" / "dsart_search_rt.yaml").r12
    assert not one.enabled
    with pytest.raises(ValueError):
        c3mod.R12Config.from_dict({"mode": "delete"})
    assert c3mod._r12_note({}) == ""
    assert c3mod._r12_note({"ok": False, "reason": "no cutouts"}) == " r12=n/a(no cutouts)"
