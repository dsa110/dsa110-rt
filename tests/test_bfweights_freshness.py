"""SEFDs page bf-weights: ages, applied solution, and node consensus.

2026-09-25/26, all measured live:
  * every printed solution age was 3.6 h low -- ``age_hours`` was
    stamped at scan time and the cache then served it frozen;
  * the Update-cals click at 09-25 15:35 distributed
    ``2253+161_2026-09-24T06:35:59`` although
    ``2253+161_2026-09-25T06:32:03`` had been on disk since 06:59 --
    the route chose from the same stale display cache;
  * one distribution copies antennas.out to the nodes ~0.33 s apart, so
    whole-second grouping reported "10 node(s) disagree with the
    6-node consensus" for identical weights.
"""
from __future__ import annotations

import datetime
import os
import sys
import time
from unittest import mock

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSA_MONITOR_DIR = os.path.join(REPO_ROOT, "tools", "dashboard", "dsa_monitor")
for p in (os.path.join(REPO_ROOT, "src"), DSA_MONITOR_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bfweights_update as bu  # noqa: E402
import cal_visibility as cv  # noqa: E402

UTC = datetime.timezone.utc
OLD = "2253+161_2026-09-24T06:35:59"
NEW = "2253+161_2026-09-25T06:32:03"


def _touch(d, descriptor):
    (d / f"beamformer_weights_{descriptor}.yaml").write_text("x: 1\n")


@pytest.fixture()
def gen(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(bu, "_desc_cache", {"ts": 0.0, "by_source": {}})
    return tmp_path


def _unix(isot):
    return datetime.datetime.strptime(isot, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=UTC).timestamp()


def test_age_is_computed_at_read_time_not_scan_time(gen, monkeypatch):
    _touch(gen, OLD)
    t_scan = _unix("2026-09-25T21:08:00")
    monkeypatch.setattr(bu.time, "time", lambda: t_scan)
    bu.fresh_latest_descriptor("2253+161")            # cache filled 21:08
    monkeypatch.setattr(bu.time, "time", lambda: t_scan + 3.65 * 3600)
    got = bu.latest_descriptor("2253+161")            # page read 00:47
    want = round((t_scan + 3.65 * 3600 - _unix("2026-09-24T06:35:59")) / 3600, 1)
    assert got["age_hours"] == want                    # 42.2, not 38.5
    assert bu.latest_descriptors(["2253+161"])["2253+161"]["age_hours"] == want


def test_fresh_lookup_sees_a_solution_newer_than_the_cache(gen):
    _touch(gen, OLD)
    bu.fresh_latest_descriptor("2253+161")
    _touch(gen, NEW)                                   # appears later
    assert bu.latest_descriptor("2253+161")["descriptor"] == OLD   # stale cache
    assert bu.fresh_latest_descriptor("2253+161")["descriptor"] == NEW
    assert bu.latest_descriptor("2253+161")["descriptor"] == NEW   # refreshed


def test_fresh_lookup_raises_when_dir_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "GENERATED_DIR", str(tmp_path / "missing"))
    with pytest.raises(OSError):
        bu.fresh_latest_descriptor("2253+161")


# --- the update route --------------------------------------------------------

@pytest.fixture(scope="module")
def app_module():
    with mock.patch("rfi_store.RFIPoller.start", return_value=None):
        import app  # noqa: F401
        yield app


def test_route_applies_the_fresh_newest_and_refuses_a_stale_page(
        app_module, gen, monkeypatch):
    _touch(gen, OLD)
    bu.fresh_latest_descriptor("2253+161")             # page rendered with OLD
    _touch(gen, NEW)
    monkeypatch.setattr(app_module.sefd_view, "known_sources",
                        lambda: {"2253+161"})
    started = []
    monkeypatch.setattr(
        bu, "start_update",
        lambda d, **kw: started.append(d) or {"job_id": "j1", "descriptor": d})
    client = app_module.app.test_client()
    form = {"source": "2253+161", "confirm": "update_bfweights"}
    # the page showed OLD: refuse, apply nothing
    r = client.post("/control/update_bfweights", data={**form, "descriptor": OLD})
    assert r.status_code == 409 and r.get_json()["latest"] == NEW
    assert started == []
    # confirmed on the newest: applied
    r = client.post("/control/update_bfweights", data={**form, "descriptor": NEW})
    assert r.status_code == 202 and started == [NEW]
    # a client that sends no descriptor still gets the FRESH newest
    started.clear()
    r = client.post("/control/update_bfweights", data=form)
    assert r.status_code == 202 and started == [NEW]


# --- node consensus ----------------------------------------------------------

class FakeEtcd:
    def __init__(self, docs):
        self.docs = docs

    def get_dict(self, key):
        return self.docs.get(key)


def test_sequential_copy_is_one_distribution(tmp_path, monkeypatch):
    applied = tmp_path / "applied"
    applied.mkdir()
    (applied / "beamformer_weights_2026-09-25T15:35:33.yaml").write_text(
        "source: ['2253+161']\ncaltime: [61307.274988425925]\n"
        "weight_files: ['beamformer_weights_sb00_2026-09-25T15:35:25.dat']\n")
    monkeypatch.setattr(cv, "APPLIED_DIR", str(applied))
    t0 = _unix("2026-09-25T15:35:25") + 0.777802      # the live spread
    docs = {}
    for i, cn in enumerate(cv.CORR_NODES):
        t = t0 + 0.33 * i
        docs[cv.CAL_FILE_KEY_TMPL.format(cn=cn.cn_id)] = {
            "path": "antennas.out", "mtime_unix": t,
            "mtime_isot": datetime.datetime.fromtimestamp(t, UTC).isoformat(),
        }
    view = cv.build_pipeline_weights_view(FakeEtcd(docs))
    assert view["disagreeing"] == []
    assert view["consensus_isot"] == "2026-09-25T15:35:25"
    assert view["stale"] is False
    assert view["transit_isot"] == "2026-09-24T06:35:59"
    # a node still on the previous generation IS flagged
    stale_cn = cv.CORR_NODES[3].cn_id
    docs[cv.CAL_FILE_KEY_TMPL.format(cn=stale_cn)] = {
        "path": "antennas.out", "mtime_unix": t0 - 86400,
        "mtime_isot": "2026-09-24T15:35:25+00:00"}
    view = cv.build_pipeline_weights_view(FakeEtcd(docs))
    assert [n["cn_id"] for n in view["disagreeing"]] == [stale_cn]
