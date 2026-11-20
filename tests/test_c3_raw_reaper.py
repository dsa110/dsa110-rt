"""C3 raw reaper: raw/ recreated by late raw uploads after a destructive REJECT (2026-11-20).

C3 deletes raw/ when it REJECTs, but each search half keeps uploading its raw
windows for minutes afterwards and rsync recreates raw/. 2026-11-16..19 left
91 of 303 REJECTs with 278 GiB of raw windows that nothing revisited.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsart.coinc.cal_hdf5_archive import CalHdf5Config               # noqa: E402
from dsart.services import c3 as c3mod                              # noqa: E402


class FakeStore:
    def __init__(self, flag_only: Optional[bool] = None) -> None:
        self.flag_only = flag_only

    def put_dict(self, key: str, value) -> None:
        pass

    def get_dict(self, key: str) -> Optional[Any]:
        if key == c3mod.C3_FLAG_ONLY_KEY and self.flag_only is not None:
            return {"flag_only": self.flag_only}
        return None


def _service(tmp: Path, *, quiet_s: float = 0.0, store: Optional[FakeStore] = None, **kw):
    cfg = c3mod.C3Config(
        archive_root=tmp / "c", rejected_root=tmp / "r", state_path=tmp / "s.json",
        fired_injection_log=None, corr_nodes={}, flag_only=False,
        cal_hdf5=CalHdf5Config(enabled=False), raw_reap_quiet_s=quiet_s, **kw)
    return c3mod.C3Service(cfg, mon_store=store or FakeStore())


def _raw(tmp: Path, name: str, n: int = 2) -> Path:
    raw = tmp / "c" / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (raw / f"raw_s1_g{i}.npz").write_bytes(b"x" * 10)
    return raw


def _event(tmp: Path, name: str) -> Path:
    ev = tmp / "c" / name
    (ev / "Level3").mkdir(parents=True)
    (ev / "Level3" / f"{name}.json").write_text("{}")
    return ev


def test_reject_registers_and_late_raw_is_reaped(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    ev = _event(tmp_path, "261119aaaa")
    _raw(tmp_path, "261119aaaa")
    svc._do_reject("261119aaaa", ev, None, flag_only=False)
    assert not (ev / "raw").exists() and "261119aaaa" in svc._raw_reap
    raw = _raw(tmp_path, "261119aaaa", n=3)          # a half's upload lands afterwards
    svc._reap_raw()
    assert not raw.exists() and svc._counters["raw_reaped"] == 1
    assert "261119aaaa" in svc._raw_reap             # still watched within the horizon
    svc._raw_reap["261119aaaa"] = time.time() - svc._cfg.raw_reap_horizon_s - 1
    svc._reap_raw()
    assert "261119aaaa" not in svc._raw_reap          # gone and past the horizon: dropped


def test_raw_still_landing_is_left_alone(tmp_path: Path) -> None:
    svc = _service(tmp_path, quiet_s=3600.0)
    raw = _raw(tmp_path, "261119bbbb")
    svc._raw_reap["261119bbbb"] = time.time()
    svc._reap_raw()
    assert raw.exists() and svc._counters.get("raw_reaped", 0) == 0


def test_flag_only_mode_deletes_nothing(tmp_path: Path) -> None:
    svc = _service(tmp_path, store=FakeStore(flag_only=True))
    raw = _raw(tmp_path, "261119cccc")
    svc._raw_reap["261119cccc"] = time.time()
    svc._reap_raw()
    assert raw.exists()


def test_flagged_reject_is_not_registered(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    ev = _event(tmp_path, "261119dddd")
    svc._do_reject("261119dddd", ev, None, flag_only=True)
    assert "261119dddd" not in svc._raw_reap


def test_max_per_pass_bounds_a_backlog(tmp_path: Path) -> None:
    svc = _service(tmp_path, raw_reap_max_per_pass=2)
    for i in range(5):
        name = f"26111{9}e{i:03d}"
        _raw(tmp_path, name)
        svc._raw_reap[name] = time.time()
    svc._reap_raw()
    left = [n for n in svc._raw_reap if (tmp_path / "c" / n / "raw").exists()]
    assert svc._counters["raw_reaped"] == 2 and len(left) == 3


def test_seed_backfills_recent_destructive_rejects_only(tmp_path: Path) -> None:
    today = datetime.now(timezone.utc).strftime("%y%m%d")
    old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%y%m%d")
    processed = {f"{today}rejr": "REJECT", f"{today}flag": "REJECT(flagged)",
                 f"{today}keep": "KEEP", f"{old}oldr": "REJECT", f"{today}noraw": "REJECT"}
    (tmp_path / "s.json").write_text(json.dumps({"processed": processed}))
    for name in processed:
        if name != f"{today}noraw":
            _raw(tmp_path, name)
    svc = _service(tmp_path)
    assert set(svc._raw_reap) == {f"{today}rejr"}
    svc._reap_raw()
    assert not (tmp_path / "c" / f"{today}rejr" / "raw").exists()
    assert (tmp_path / "c" / f"{today}keep" / "raw").exists()      # never touched
    assert (tmp_path / "c" / f"{today}flag" / "raw").exists()
    assert f"{today}rejr" not in svc._raw_reap                      # seeded past the horizon


def test_reap_list_survives_a_restart(tmp_path: Path) -> None:
    svc = _service(tmp_path, raw_reap_backfill_days=0)
    ev = _event(tmp_path, "261119ffff")
    svc._do_reject("261119ffff", ev, None, flag_only=False)
    svc._save_state()
    again = _service(tmp_path, raw_reap_backfill_days=0)
    assert "261119ffff" in again._raw_reap
