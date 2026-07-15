"""Tests for the SEFDs "Pipeline weights" panel (cal_visibility).

Regression anchor (2026-07-15): the panel's "solution:" line used to
come from ``/mon/cal/bfweights``, which the legacy auto-calibration
stack republishes for every transit it solves WITHOUT distributing
anything -- so while the fleet ran the 2253+161 night-transit solution
distributed at 14:13 UT, the card claimed the never-distributed (and
solar-contaminated) 0521+166 17:41 solution and flagged the fleet
"STALE". The panel must trust the ``applied/`` archive (one fleet YAML
per real distribution) and use etcd only as a labelled fallback.
"""

from __future__ import annotations

import datetime
import os
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, ".."))
DSA_MONITOR_DIR = os.path.normpath(
    os.path.join(REPO_ROOT, "tools", "dashboard", "dsa_monitor")
)
if DSA_MONITOR_DIR not in sys.path:
    sys.path.insert(0, DSA_MONITOR_DIR)

import cal_visibility as cv  # noqa: E402

UTC = datetime.timezone.utc

# The real 2026-07-15 numbers (see module docstring).
APPLIED_ISOT = "2026-07-15T14:13:00"       # fleet-YAML filename stamp
DAT_ISOT = "2026-07-15T14:12:57"           # .dat / antennas.out stamp
TRANSIT_MJD = 61236.468831018516           # 2253+161 transit 11:15:07 UT
TRANSIT_ISOT = "2026-07-15T11:15:07"
ETCD_DAT_ISOT = "2026-07-15T17:41:27"      # never-distributed 0521+166
ETCD_TRANSIT_MJD = 61236.73711805556


def _unix(isot: str) -> float:
    return datetime.datetime.strptime(
        isot, "%Y-%m-%dT%H:%M:%S"
    ).replace(tzinfo=UTC).timestamp()


def _write_fleet_yaml(directory, isot, *, source="2253+161",
                      caltime=TRANSIT_MJD, dat_isot=None, text=None):
    p = directory / f"beamformer_weights_{isot}.yaml"
    if text is not None:
        p.write_text(text)
        return p
    doc = {
        "source": [source],
        "caltime": [caltime],
        "weight_files": [
            f"beamformer_weights_sb00_{dat_isot or isot}.dat",
        ],
    }
    p.write_text(yaml.safe_dump(doc))
    return p


def _bfweights_doc():
    """The clobbered /mon/cal/bfweights payload (auto-cal stack)."""
    return {
        "cmd": "update_weights",
        "val": {
            "source": ["0521+166"],
            "caltime": [ETCD_TRANSIT_MJD],
            "weight_files": [
                f"beamformer_weights_sb00_{ETCD_DAT_ISOT}.dat",
            ],
        },
    }


class FakeEtcd:
    """get_dict-only stand-in for app.py's etcd store."""

    def __init__(self, docs=None, raise_always=False):
        self.docs = docs or {}
        self.raise_always = raise_always

    def get_dict(self, key):
        if self.raise_always:
            raise RuntimeError("etcd down")
        return self.docs.get(key)


def _node_docs(mtime_isot, n=None):
    """cal_file docs for every corr node (or the first ``n``)."""
    docs = {}
    nodes = cv.CORR_NODES if n is None else cv.CORR_NODES[:n]
    for cn in nodes:
        docs[cv.CAL_FILE_KEY_TMPL.format(cn=cn.cn_id)] = {
            "path": "/home/ubuntu/proj/.../antennas.out",
            "mtime_isot": mtime_isot + ".349369+00:00",
            "mtime_unix": _unix(mtime_isot) + 0.349369,
        }
    return docs


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    applied = tmp_path / "applied"
    generated = tmp_path / "generated"
    applied.mkdir()
    generated.mkdir()
    monkeypatch.setattr(cv, "APPLIED_DIR", str(applied))
    monkeypatch.setattr(cv, "GENERATED_DIR", str(generated))
    return applied, generated


# ---------------------------------------------------------------------------
# _latest_applied_solution
# ---------------------------------------------------------------------------


def test_latest_applied_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "APPLIED_DIR", str(tmp_path / "nope"))
    assert cv._latest_applied_solution() is None


def test_latest_applied_empty_dir(dirs):
    assert cv._latest_applied_solution() is None


def test_latest_applied_reads_newest_fleet_yaml(dirs):
    applied, _ = dirs
    _write_fleet_yaml(applied, "2026-07-13T21:42:47",
                      dat_isot="2026-07-13T21:42:44")
    _write_fleet_yaml(applied, APPLIED_ISOT, dat_isot=DAT_ISOT)
    sol = cv._latest_applied_solution()
    assert sol == {
        "distributed_isot": DAT_ISOT,      # .dat stamp, not filename stamp
        "source": "2253+161",
        "caltime_mjd": pytest.approx(TRANSIT_MJD),
    }


def test_latest_applied_ignores_per_source_yamls(dirs):
    """A newer per-source solution YAML must not shadow the fleet YAML."""
    applied, _ = dirs
    _write_fleet_yaml(applied, APPLIED_ISOT, dat_isot=DAT_ISOT)
    # Auto-cal wrote a newer per-source YAML (never distributed).
    (applied / f"beamformer_weights_0521+166_{ETCD_DAT_ISOT}.yaml").write_text(
        yaml.safe_dump({"source": ["0521+166"]})
    )
    sol = cv._latest_applied_solution()
    assert sol["source"] == "2253+161"
    assert sol["distributed_isot"] == DAT_ISOT


def test_latest_applied_unreadable_yaml_falls_back_to_filename(dirs):
    applied, _ = dirs
    _write_fleet_yaml(applied, APPLIED_ISOT,
                      text="caltime: !!python/object/apply:numpy.bogus []\n")
    sol = cv._latest_applied_solution()
    assert sol == {
        "distributed_isot": APPLIED_ISOT,  # filename fallback
        "source": None,
        "caltime_mjd": None,
    }


def test_latest_applied_scalar_source_and_caltime(dirs):
    applied, _ = dirs
    p = applied / f"beamformer_weights_{APPLIED_ISOT}.yaml"
    p.write_text(yaml.safe_dump(
        {"source": "2253+161", "caltime": TRANSIT_MJD}
    ))
    sol = cv._latest_applied_solution()
    assert sol["source"] == "2253+161"
    assert sol["caltime_mjd"] == pytest.approx(TRANSIT_MJD)
    assert sol["distributed_isot"] == APPLIED_ISOT   # no weight_files


# ---------------------------------------------------------------------------
# build_pipeline_weights_view: applied/ is ground truth
# ---------------------------------------------------------------------------


def test_view_prefers_applied_over_clobbered_etcd(dirs):
    """THE 2026-07-15 regression: etcd names 0521+166 (never
    distributed); the card must show the applied 2253+161 solution and
    not flag the fleet stale."""
    applied, _ = dirs
    _write_fleet_yaml(applied, APPLIED_ISOT, dat_isot=DAT_ISOT)
    docs = {cv.BFWEIGHTS_KEY: _bfweights_doc()}
    docs.update(_node_docs(DAT_ISOT))
    view = cv.build_pipeline_weights_view(FakeEtcd(docs))
    assert view["solution_provenance"] == "applied_yaml"
    assert view["distributed_source"] == "2253+161"
    assert view["transit_isot"] == TRANSIT_ISOT
    assert view["distributed_isot"] == DAT_ISOT
    assert view["consensus_isot"] == DAT_ISOT
    assert view["stale"] is False
    assert view["disagreeing"] == []
    assert view["n_reported"] == view["n_total"] == len(cv.CORR_NODES)


def test_view_stale_when_nodes_predate_distribution(dirs):
    applied, _ = dirs
    _write_fleet_yaml(applied, APPLIED_ISOT, dat_isot=DAT_ISOT)
    docs = _node_docs("2026-07-13T21:42:44")   # loaded before distribution
    view = cv.build_pipeline_weights_view(FakeEtcd(docs))
    assert view["stale"] is True
    assert view["consensus_isot"] == "2026-07-13T21:42:44"


def test_view_transit_fallback_from_matching_yaml(dirs):
    """Applied YAML without caltime: transit recovered from the fleet
    YAML matching the distributed ISOT (here, itself -- covering old
    payload layouts)."""
    applied, _ = dirs
    p = applied / f"beamformer_weights_{APPLIED_ISOT}.yaml"
    p.write_text(yaml.safe_dump({"source": ["2253+161"]}))
    # A second YAML named by the distributed ISOT carries the caltime.
    _write_fleet_yaml(applied, APPLIED_ISOT + "x")  # non-matching noise
    view = cv.build_pipeline_weights_view(FakeEtcd({}))
    # Fallback searched beamformer_weights_<APPLIED_ISOT>.yaml -- which
    # has no caltime -- so transit stays None (never guesses).
    assert view["distributed_source"] == "2253+161"
    assert view["transit_isot"] is None
    assert view["transit_age_hours"] is None
    assert view["due_for_update"] is False


def test_view_due_for_update_from_old_transit(dirs):
    applied, _ = dirs
    _write_fleet_yaml(applied, "2026-07-10T14:13:00",
                      caltime=61231.4688)      # 5 days old
    view = cv.build_pipeline_weights_view(FakeEtcd({}))
    assert view["due_for_update"] is True
    assert view["transit_age_hours"] > 48


# ---------------------------------------------------------------------------
# build_pipeline_weights_view: etcd fallback (no applied/ visible)
# ---------------------------------------------------------------------------


def test_view_falls_back_to_etcd_when_no_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(cv, "APPLIED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(cv, "GENERATED_DIR", str(tmp_path / "nope2"))
    view = cv.build_pipeline_weights_view(
        FakeEtcd({cv.BFWEIGHTS_KEY: _bfweights_doc()})
    )
    assert view["solution_provenance"] == "etcd"
    assert view["distributed_source"] == "0521+166"
    assert view["distributed_isot"] == ETCD_DAT_ISOT
    assert view["transit_isot"] == "2026-07-15T17:41:27"


def test_view_survives_total_darkness(tmp_path, monkeypatch):
    """No applied dir AND etcd raising: everything None, no exception."""
    monkeypatch.setattr(cv, "APPLIED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(cv, "GENERATED_DIR", str(tmp_path / "nope2"))
    view = cv.build_pipeline_weights_view(FakeEtcd(raise_always=True))
    assert view["solution_provenance"] is None
    assert view["distributed_source"] is None
    assert view["distributed_isot"] is None
    assert view["transit_isot"] is None
    assert view["stale"] is False
    assert view["n_reported"] == 0
    assert view["any_reported"] is False


def test_view_partial_node_reporting_and_disagreement(dirs):
    applied, _ = dirs
    _write_fleet_yaml(applied, APPLIED_ISOT, dat_isot=DAT_ISOT)
    docs = _node_docs(DAT_ISOT, n=10)
    # Two of the reporting nodes still carry the previous vintage.
    for cn in cv.CORR_NODES[:2]:
        docs[cv.CAL_FILE_KEY_TMPL.format(cn=cn.cn_id)] = {
            "path": "x", "mtime_isot": "2026-07-13T21:42:44.1+00:00",
            "mtime_unix": _unix("2026-07-13T21:42:44") + 0.1,
        }
    view = cv.build_pipeline_weights_view(FakeEtcd(docs))
    assert view["n_reported"] == 10
    assert view["consensus_isot"] == DAT_ISOT
    assert len(view["disagreeing"]) == 2
    assert view["stale"] is False
