"""Regression tests for ``dsart.coinc.cube_uploader``.

Covers:

  1. ``build_rsync_argv`` produces the exact argv shape the search
     nodes need (``-a --stats --partial``, ``--rsync-path='mkdir -p
     ... && rsync'`` to emulate ``--mkpath`` on rsync < 3.2.3,
     trailing slash on source, ``<dest_host>:<dest_root>/<event>/cubes/``
     destination shape).
  2. ``build_rsync_argv`` honours ``bandwidth_limit_kbps`` (``--bwlimit=``
     inserted) and the override ``rsync_path``.
  3. ``build_rsync_argv`` rejects a bad ``event_name`` (empty / has ``/``).
  4. ``parse_remote_root`` round-trips ``user@host:/path`` and falls
     back when the ``:`` or the path are missing.
  5. ``upload_event_cubes`` calls ``Popen`` with the exact argv from
     ``build_rsync_argv``, with ``stdin=DEVNULL``, ``stderr=STDOUT``,
     ``start_new_session=True``, and a non-inheritable stdout FD.
  6. ``upload_event_cubes`` writes the per-event ``upload.log`` header
     line (timestamp, event, src, dest, argv) before spawning.
  7. ``upload_event_cubes`` returns the ``Popen`` handle the mock built.

Tests deliberately mock ``subprocess.Popen`` via the helper's
``popen`` kwarg so we never exec the real rsync binary. The smoke
test in the cube-uploader bring-up checklist exercises the real
spawn + remote receive separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from dsart.coinc.cube_uploader import (
    DEFAULT_DEST_HOST,
    DEFAULT_DEST_ROOT,
    DEFAULT_RSYNC_OPTS,
    BoundedCubeUploader,
    build_rsync_argv,
    parse_remote_root,
    upload_event_cubes,
)


# ---------------------------------------------------------------------------
# build_rsync_argv
# ---------------------------------------------------------------------------


def test_build_rsync_argv_basic_shape() -> None:
    argv = build_rsync_argv(
        "260527abcd",
        "/home/ubuntu/data/c2/cube_dump/260527abcd",
        "ubuntu@lxd110h23.pro.pvt",
        "/dataz/dsa110/candidates",
    )
    assert argv[0] == "rsync"
    # Defaults are in the published tuple order, preserved.
    for i, opt in enumerate(DEFAULT_RSYNC_OPTS):
        assert argv[1 + i] == opt
    # ``--rsync-path`` follows the defaults and emulates ``--mkpath``.
    rsp = [a for a in argv if a.startswith("--rsync-path=")]
    assert len(rsp) == 1
    assert (
        rsp[0]
        == "--rsync-path=mkdir -p /dataz/dsa110/candidates/260527abcd/cubes && rsync"
    )
    assert argv[-2] == "/home/ubuntu/data/c2/cube_dump/260527abcd/"
    assert (
        argv[-1]
        == "ubuntu@lxd110h23.pro.pvt:/dataz/dsa110/candidates/260527abcd/cubes/"
    )


def test_build_rsync_argv_strips_trailing_dest_slash() -> None:
    # ``dest_root`` with a trailing slash must not produce a double slash.
    argv = build_rsync_argv(
        "evt", "/tmp/src", "ubuntu@h23",
        "/dataz/dsa110/candidates///",
    )
    assert argv[-1] == "ubuntu@h23:/dataz/dsa110/candidates/evt/cubes/"
    rsp = [a for a in argv if a.startswith("--rsync-path=")][0]
    # mkdir target also trims trailing slashes.
    assert "mkdir -p /dataz/dsa110/candidates/evt/cubes" in rsp


def test_build_rsync_argv_bwlimit_and_rsync_path() -> None:
    argv = build_rsync_argv(
        "evt",
        "/tmp/src",
        "ubuntu@h23",
        "/dataz/dsa110/candidates",
        rsync_path="/usr/bin/rsync",
        bandwidth_limit_kbps=12_500,
        extra_opts=("--quiet",),
    )
    assert argv[0] == "/usr/bin/rsync"
    # bwlimit comes after the defaults+rsync-path, before extra_opts+paths.
    assert "--bwlimit=12500" in argv
    assert argv.index("--bwlimit=12500") < argv.index("--quiet")
    assert argv[-2] == "/tmp/src/"
    assert argv[-1] == "ubuntu@h23:/dataz/dsa110/candidates/evt/cubes/"


def test_build_rsync_argv_remote_rsync_bin_override() -> None:
    argv = build_rsync_argv(
        "evt", "/tmp/src", "ubuntu@h23", "/dataz/x",
        remote_rsync_bin="/opt/rsync/bin/rsync",
    )
    rsp = [a for a in argv if a.startswith("--rsync-path=")][0]
    assert rsp.endswith("&& /opt/rsync/bin/rsync")


def test_build_rsync_argv_rejects_bad_event_name() -> None:
    with pytest.raises(ValueError):
        build_rsync_argv("", "/tmp/x", "ubuntu@h23", "/dataz")
    with pytest.raises(ValueError):
        build_rsync_argv("a/b", "/tmp/x", "ubuntu@h23", "/dataz")


# ---------------------------------------------------------------------------
# parse_remote_root
# ---------------------------------------------------------------------------


def test_parse_remote_root_standard_shape() -> None:
    host, root = parse_remote_root("ubuntu@lxd110h23.pro.pvt:/dataz/dsa110/candidates")
    assert host == "ubuntu@lxd110h23.pro.pvt"
    assert root == "/dataz/dsa110/candidates"


def test_parse_remote_root_missing_path_falls_back() -> None:
    host, root = parse_remote_root("ubuntu@h23")
    assert host == "ubuntu@h23"
    assert root == DEFAULT_DEST_ROOT


def test_parse_remote_root_empty_falls_back_to_defaults() -> None:
    host, root = parse_remote_root("")
    assert host == DEFAULT_DEST_HOST
    assert root == DEFAULT_DEST_ROOT


# ---------------------------------------------------------------------------
# upload_event_cubes
# ---------------------------------------------------------------------------


def _make_popen_mock() -> tuple[MagicMock, List[dict]]:
    """Build a ``subprocess.Popen``-compatible mock + capture list."""
    captured: List[dict] = []

    def _fake_popen(argv, **kwargs):
        captured.append({"argv": list(argv), "kwargs": dict(kwargs)})
        m = MagicMock(name="popen_handle")
        m.pid = 12345
        return m

    return MagicMock(side_effect=_fake_popen), captured


def test_upload_event_cubes_invokes_popen_with_expected_argv(tmp_path) -> None:
    src = tmp_path / "260527abcd"
    src.mkdir()
    (src / "cube_s1_g0_42.npz").write_bytes(b"fake")

    popen_mock, captured = _make_popen_mock()
    proc = upload_event_cubes(
        event_name="260527abcd",
        src_dir=src,
        dest_host="ubuntu@lxd110h23.pro.pvt",
        dest_root="/dataz/dsa110/candidates",
        rsync_path="/usr/bin/rsync",
        popen=popen_mock,
    )
    assert proc.pid == 12345
    assert len(captured) == 1
    call = captured[0]
    assert call["argv"][0] == "/usr/bin/rsync"
    assert call["argv"][-2] == f"{src}/"
    assert (
        call["argv"][-1]
        == "ubuntu@lxd110h23.pro.pvt:/dataz/dsa110/candidates/260527abcd/cubes/"
    )
    # Subprocess hygiene: stdin DEVNULL, stderr STDOUT, detached session,
    # stdout is a real FD (Popen dup2's it from us; passed as int FD).
    assert call["kwargs"]["start_new_session"] is True
    assert call["kwargs"]["stdin"].__class__.__name__ in ("int", "_DEVNULL")
    # subprocess.DEVNULL is the sentinel -3 in CPython.
    import subprocess as _sp
    assert call["kwargs"]["stdin"] == _sp.DEVNULL
    assert call["kwargs"]["stderr"] == _sp.STDOUT
    # stdout is an integer FD (the helper os.open's the log).
    assert isinstance(call["kwargs"]["stdout"], int)


def test_upload_event_cubes_writes_log_header(tmp_path) -> None:
    src = tmp_path / "260527xyz"
    src.mkdir()
    popen_mock, _ = _make_popen_mock()
    upload_event_cubes(
        event_name="260527xyz",
        src_dir=src,
        dest_host="ubuntu@lxd110h23.pro.pvt",
        dest_root="/dataz/dsa110/candidates",
        popen=popen_mock,
    )
    log_path = src / "upload.log"
    assert log_path.is_file()
    body = log_path.read_text(encoding="utf-8")
    # Header line: timestamp + event + src + dest + full argv.
    assert "event=260527xyz" in body
    assert str(src) in body
    assert "ubuntu@lxd110h23.pro.pvt:/dataz/dsa110/candidates" in body
    assert "rsync" in body
    # Trailing slash on src AND dest is what makes rsync land files in
    # ``<event>/cubes/`` and not ``<event>/cubes/<event>/``.
    assert f"{src}/" in body
    assert "/260527xyz/cubes/" in body


def test_upload_event_cubes_appends_header_across_calls(tmp_path) -> None:
    src = tmp_path / "evt"
    src.mkdir()
    popen_mock, _ = _make_popen_mock()
    upload_event_cubes(
        "evt", src, "ubuntu@h23", "/dataz", popen=popen_mock,
    )
    upload_event_cubes(
        "evt", src, "ubuntu@h23", "/dataz", popen=popen_mock,
    )
    body = (src / "upload.log").read_text(encoding="utf-8")
    # One header line per invocation.
    header_lines = [
        ln for ln in body.splitlines() if ln.startswith("# ")
    ]
    assert len(header_lines) == 2


def test_upload_event_cubes_honours_bandwidth_limit(tmp_path) -> None:
    src = tmp_path / "evt"
    src.mkdir()
    popen_mock, captured = _make_popen_mock()
    upload_event_cubes(
        "evt",
        src,
        "ubuntu@h23",
        "/dataz",
        bandwidth_limit_kbps=20_000,
        popen=popen_mock,
    )
    assert "--bwlimit=20000" in captured[0]["argv"]


def test_upload_event_cubes_uses_custom_log_path(tmp_path) -> None:
    src = tmp_path / "evt"
    src.mkdir()
    custom_log = tmp_path / "custom" / "upload.log"
    popen_mock, _ = _make_popen_mock()
    upload_event_cubes(
        "evt",
        src,
        "ubuntu@h23",
        "/dataz",
        log_path=custom_log,
        popen=popen_mock,
    )
    assert custom_log.is_file()
    body = custom_log.read_text(encoding="utf-8")
    assert "event=evt" in body
    # The default log next to NPZs must NOT have been created.
    assert not (src / "upload.log").exists()


# ---------------------------------------------------------------------------
# BoundedCubeUploader (T5 — concurrency cap)
# ---------------------------------------------------------------------------


class _FakeProc:
    """Stand-in for subprocess.Popen with an injectable wait()."""

    def __init__(
        self, *, rc: int = 0, on_wait=None, pid: int = 12345,
    ) -> None:
        self.pid = pid
        self._rc = int(rc)
        self._on_wait = on_wait

    def wait(self):
        if self._on_wait is not None:
            self._on_wait()
        return self._rc


def _fake_upload_factory(
    *, hold_event=None, exit_codes=None, log_calls=None,
):
    """Build a fake ``upload_fn`` for BoundedCubeUploader tests.

    ``hold_event`` (threading.Event) — if set, ``proc.wait()`` blocks
        until the event is set so the test can stall the worker and
        observe queue backpressure.
    ``exit_codes`` (list[int] | None) — per-call return codes for
        ``proc.wait()``; defaults to all 0.
    ``log_calls`` (list) — appended-to per spawn for assertions.
    """
    import threading

    counter = {"n": 0}
    log = log_calls if log_calls is not None else []

    def fake_upload(*, event_name, src_dir, dest_host, dest_root, **kwargs):
        log.append((event_name, str(src_dir), dest_host, dest_root))
        if exit_codes is not None:
            rc = exit_codes[counter["n"] % len(exit_codes)]
        else:
            rc = 0
        counter["n"] += 1
        on_wait = None
        if hold_event is not None:
            on_wait = lambda: hold_event.wait(timeout=10.0)
        return _FakeProc(rc=rc, on_wait=on_wait)

    return fake_upload, log


def test_bounded_uploader_serialises_uploads(tmp_path) -> None:
    """T5: with max_concurrent=1 every submitted event runs through
    one worker, in submission order."""
    log: list = []
    fake_upload, _ = _fake_upload_factory(log_calls=log)
    up = BoundedCubeUploader(
        dest_host="ubuntu@h23",
        dest_root="/dataz",
        max_concurrent=1,
        queue_maxsize=4,
        upload_fn=fake_upload,
    )
    up.start()
    try:
        for i in range(3):
            assert up.submit(f"evt{i}", tmp_path / f"evt{i}") is True
        # Drain — give the worker time to process the submitted items.
        for _ in range(50):
            if up.n_uploaded >= 3:
                break
            import time as _t
            _t.sleep(0.05)
        assert up.n_uploaded == 3
        assert up.n_submitted == 3
        assert up.n_dropped_full == 0
        # Submission order preserved.
        assert [e for e, *_ in log] == ["evt0", "evt1", "evt2"]
    finally:
        up.stop()


def test_bounded_uploader_evicts_oldest_on_full_queue(tmp_path) -> None:
    """T5: a full queue evicts the OLDEST pending job in favour of the
    new submission, and bumps ``n_dropped_full``."""
    import threading
    hold = threading.Event()
    log: list = []
    fake_upload, _ = _fake_upload_factory(hold_event=hold, log_calls=log)
    up = BoundedCubeUploader(
        dest_host="ubuntu@h23",
        dest_root="/dataz",
        max_concurrent=1,
        queue_maxsize=2,
        upload_fn=fake_upload,
    )
    up.start()
    try:
        # The first submit gets pulled by the worker immediately and
        # blocks on hold. Subsequent submits sit in the queue (cap=2).
        up.submit("evt-running", tmp_path / "evt-running")
        # Give the worker a moment to dequeue the first item before
        # we fill the queue, so cap=2 is the actual ceiling.
        import time as _t
        for _ in range(50):
            if up.queue_depth == 0:
                break
            _t.sleep(0.02)
        up.submit("evt-A", tmp_path / "evt-A")
        up.submit("evt-B", tmp_path / "evt-B")
        # Now the queue is full ([A, B]); the next submit must evict A.
        up.submit("evt-C", tmp_path / "evt-C")
        assert up.n_dropped_full == 1
        # Release hold and let the worker drain the rest.
        hold.set()
        for _ in range(50):
            if up.n_uploaded >= 3:
                break
            _t.sleep(0.05)
        spawned = [e for e, *_ in log]
        assert "evt-running" in spawned
        assert "evt-B" in spawned
        assert "evt-C" in spawned
        # ``evt-A`` was evicted before being spawned.
        assert "evt-A" not in spawned
    finally:
        hold.set()
        up.stop()


def test_bounded_uploader_counts_failures(tmp_path) -> None:
    """T5: a non-zero rsync rc bumps ``n_failed`` (not n_uploaded)."""
    fake_upload, _ = _fake_upload_factory(exit_codes=[12])
    up = BoundedCubeUploader(
        dest_host="ubuntu@h23",
        dest_root="/dataz",
        max_concurrent=1,
        queue_maxsize=2,
        upload_fn=fake_upload,
    )
    up.start()
    try:
        up.submit("evt", tmp_path / "evt")
        import time as _t
        for _ in range(50):
            if up.n_failed >= 1:
                break
            _t.sleep(0.05)
        assert up.n_failed == 1
        assert up.n_uploaded == 0
    finally:
        up.stop()
