"""Unit tests for the cal-file mon publish added for pipeline-weights
visibility (SEFDs "Pipeline weights" panel).

Covers:

1. ``_extract_apply_cal_path`` (pure argv-parsing helper) -- both
   two-token (``--apply-cal <path>``) and ``=``-joined
   (``--apply-cal=<path>``) argv shapes, and the "absent" case (search
   nodes / routines with no ``--apply-cal`` flag).
2. ``_stat_cal_file`` -- happy path (real tmp_path file: mtime/size/
   sha256 populated) and the missing-file path (``stat_error`` set,
   never raises).
3. ``RtOrchestrator._publish_cal_file_mon`` end-to-end against a
   minimal orchestrator stub: publishes the expected payload to
   ``<mon_key>/cal_file`` when a ``corr_fast`` routine with
   ``--apply-cal`` was spawned, and is a silent no-op (no publish) when
   there's no ``corr_fast`` routine (search-node case) -- and never
   raises even if the store's ``put_dict`` blows up (must not block
   the ``start`` verb).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dsart.services.dsart_rt import (  # noqa: E402
    PipelineConfig,
    RtOrchestrator,
    _extract_apply_cal_path,
    _stat_cal_file,
)


# ---------------------------------------------------------------------------
# _extract_apply_cal_path
# ---------------------------------------------------------------------------


def test_extract_apply_cal_path_two_token_form():
    argv = ["python", "-u", "-m", "dsart.services.corr_fast_integration",
            "--fada-key", "fada", "--apply-cal", "/path/to/antennas.out",
            "--cal-mode", "phase_only"]
    assert _extract_apply_cal_path(argv) == "/path/to/antennas.out"


def test_extract_apply_cal_path_equals_form():
    argv = ["python", "-m", "x", "--apply-cal=/path/to/antennas.out"]
    assert _extract_apply_cal_path(argv) == "/path/to/antennas.out"


def test_extract_apply_cal_path_absent():
    argv = ["python", "-m", "dsart.services.search_compute", "--cal-blob-path",
            "/path/to/antennas.out"]
    assert _extract_apply_cal_path(argv) is None


def test_extract_apply_cal_path_flag_is_last_token():
    # Malformed argv (flag with no value) must not IndexError.
    argv = ["python", "-m", "x", "--apply-cal"]
    assert _extract_apply_cal_path(argv) is None


# ---------------------------------------------------------------------------
# _stat_cal_file
# ---------------------------------------------------------------------------


def test_stat_cal_file_happy_path(tmp_path):
    f = tmp_path / "antennas.out"
    f.write_bytes(b"weights-blob-contents")
    payload = _stat_cal_file(str(f))
    assert payload["path"] == str(f)
    assert payload["size"] == len(b"weights-blob-contents")
    assert "mtime_unix" in payload and "mtime_isot" in payload
    assert len(payload["sha256_12"]) == 12
    assert "stat_error" not in payload
    assert "hash_error" not in payload


def test_stat_cal_file_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.out"
    payload = _stat_cal_file(str(missing))
    assert payload["path"] == str(missing)
    assert "stat_error" in payload
    assert "mtime_unix" not in payload
    assert "sha256_12" not in payload


# ---------------------------------------------------------------------------
# RtOrchestrator._publish_cal_file_mon
# ---------------------------------------------------------------------------


def _routine(name: str, args: str) -> dict:
    return {"name": name, "cmd": "python", "args": args}


class _FakeStore:
    def __init__(self, *, raise_on_put: bool = False):
        self.published: dict = {}
        self._raise_on_put = raise_on_put

    def put_dict(self, key, value):
        if self._raise_on_put:
            raise RuntimeError("etcd is down")
        self.published[key] = value


def _build_orchestrator(routines: list[dict], *, children: dict,
                         store: _FakeStore) -> RtOrchestrator:
    cfg = PipelineConfig.from_dict({"schema_version": 1, "routines": routines})
    orch = RtOrchestrator.__new__(RtOrchestrator)
    orch._config = cfg
    orch._children = children
    orch._store = store
    orch.mon_key = "/mon/corr_rt/6"
    orch.fqdn = "lxd110h06.pro.pvt"  # _build_argv's hostargs lookup needs this
    orch._substitute = lambda tok, val: tok  # no CN/CHGROUP tokens in these tests
    return orch


def test_publish_cal_file_mon_publishes_expected_payload(tmp_path):
    cal_path = tmp_path / "antennas.out"
    cal_path.write_bytes(b"abc")
    store = _FakeStore()
    orch = _build_orchestrator(
        [_routine("corr_fast",
                   f"-u -m dsart.services.corr_fast_integration "
                   f"--apply-cal {cal_path} --cal-mode phase_only")],
        children={"corr_fast": object()},
        store=store,
    )
    orch._publish_cal_file_mon(val=None)
    assert "/mon/corr_rt/6/cal_file" in store.published
    payload = store.published["/mon/corr_rt/6/cal_file"]
    assert payload["path"] == str(cal_path)
    assert payload["size"] == 3
    assert "spawned_at_unix" in payload
    assert "sha256_12" in payload


def test_publish_cal_file_mon_noop_when_no_corr_fast_routine():
    """Search nodes: no ``corr_fast`` routine at all -- must not
    publish and must not raise."""
    store = _FakeStore()
    orch = _build_orchestrator(
        [_routine("search_compute", "--cal-blob-path /some/path")],
        children={"search_compute": object()},
        store=store,
    )
    orch._publish_cal_file_mon(val=None)
    assert store.published == {}


def test_publish_cal_file_mon_noop_when_corr_fast_not_spawned():
    """corr_fast is configured but wasn't actually spawned this time
    (e.g. ``when`` predicate skipped it) -- self._children won't have
    it; must not publish."""
    store = _FakeStore()
    orch = _build_orchestrator(
        [_routine("corr_fast", "--apply-cal /some/path")],
        children={},  # nothing spawned
        store=store,
    )
    orch._publish_cal_file_mon(val=None)
    assert store.published == {}


def test_publish_cal_file_mon_never_raises_on_store_failure(tmp_path):
    """Best-effort contract: a put_dict failure must be swallowed (and
    logged), never propagated -- this runs inline in ``_verb_start``
    and must not block it."""
    cal_path = tmp_path / "antennas.out"
    cal_path.write_bytes(b"abc")
    store = _FakeStore(raise_on_put=True)
    orch = _build_orchestrator(
        [_routine("corr_fast", f"--apply-cal {cal_path}")],
        children={"corr_fast": object()},
        store=store,
    )
    orch._publish_cal_file_mon(val=None)  # must not raise
