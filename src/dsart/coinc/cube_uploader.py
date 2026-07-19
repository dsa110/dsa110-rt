"""Search-node → h23 cube-uploader (Option A).

Background
----------
For every C2-triggered ``dump_all_gpus`` event, each search node
(``n01``/``n02``/``n09``/``n13``) dumps one NPZ per GPU half into
``<dump_root>/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz``
(default ``dump_root`` = ``/home/ubuntu/data/c2/cube_dump``).

The C2 plotter on ``h23`` (``dsart.coinc.plotter``) reads the cubes
from ``/dataz/dsa110/candidates/<event_name>/cubes/``. Until now
that directory has been populated by hand. This module wires up the
``cube_uploader`` rsync referenced in ``coinc/archive.py`` so the
search nodes push every fresh dump to h23 automatically.

Design
------
* **Option A** (search-node-initiated push). Each ``CubeDumpWriter``
  fires an ``upload_event_cubes`` after every successful ``np.savez``.
  Pros: simple, parallel across the 4 nodes, fully idempotent, runs
  immediately after the bytes are on disk.
* The rsync is spawned as a **detached** subprocess (``Popen`` with
  ``start_new_session=True``) so the search-side hot path NEVER waits
  on the network: ``Popen.__init__`` returns as soon as the fork+exec
  is in flight.
* ``rsync --mkpath`` creates the per-event ``cubes/`` directory on
  ``h23`` if it doesn't exist yet (the plotter creates it eagerly too,
  but the rsync may land before the plotter has touched the event).
* Per-event ``upload.log`` lives next to the NPZs on the search node
  (``<src_dir>/upload.log``). Each invocation appends a header line
  with the timestamp + argv, then the rsync's own stdout/stderr
  (``--stats`` summary, transferred byte count, exit code marker
  written by a shell wrapper). Operators can ``cat`` it after a
  failed event without crawling systemd journal.

The destination is configurable via the ``c1.uploader.remote_root``
yaml knob (see ``configs/dsart_search_rt.yaml``); the defaults below
are used when the helper is called outside the service (e.g. the
smoke-test bench or an operator script).

Returned ``Popen`` is the caller's to inspect if needed; the
production ``CubeDumpWriter`` hook just lets it run detached and
relies on the per-event log for postmortem.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import queue
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Union

__all__ = [
    "DEFAULT_DEST_HOST",
    "DEFAULT_DEST_ROOT",
    "DEFAULT_RSYNC_OPTS",
    "DEFAULT_UPLOAD_QUEUE_MAXSIZE",
    "BoundedCubeUploader",
    "build_rsync_argv",
    "parse_remote_root",
    "upload_event_cubes",
]


_LOG = logging.getLogger("dsart.coinc.cube_uploader")


# Search nodes resolve only the kernel-container alias
# (``lxd110h23.pro.pvt``), NOT the bare ``h23.pro.pvt``. Verified
# 2026-05-27 on n01 via ``getent hosts``. The yaml-level override in
# ``c1.uploader.remote_root`` is the production knob; this constant is
# the fallback for ad-hoc CLI use.
DEFAULT_DEST_HOST: str = "ubuntu@lxd110h23.pro.pvt"
DEFAULT_DEST_ROOT: str = "/dataz/dsa110/candidates"

# ``-a``       archive (preserves perms/times; recursive)
# ``--stats``  emit the transfer summary so ``upload.log`` is useful
# ``--partial`` keep partial transfers if rsync is interrupted (next
#               run resumes instead of re-sending the whole NPZ)
#
# NOTE: ``--mkpath`` (rsync 3.2.3+) would have been the natural choice
# but the lab nodes (n01..n13 + h23, 2026-05) ship rsync 3.1.2. We
# emulate ``--mkpath`` portably via ``--rsync-path="mkdir -p <dest> &&
# rsync"`` — the remote ssh shell runs the mkdir before exec'ing the
# server-side rsync. See :func:`build_rsync_argv`.
DEFAULT_RSYNC_OPTS: tuple[str, ...] = (
    "-a", "--stats", "--partial",
    # The per-event ``upload.log`` lives in the same directory as the
    # NPZs on the search node so operators can grep it locally; it is
    # NOT replicated to h23 (the plotter's glob ``cube_s*_g*_*.npz``
    # would skip it anyway, but excluding keeps the candidate archive
    # clean and avoids two concurrent uploaders from the two GPU halves
    # racing on the same file on the h23 end).
    "--exclude=upload.log",
)


def parse_remote_root(remote_root: str) -> tuple[str, str]:
    """Split ``user@host:/path`` -> ``("user@host", "/path")``.

    The yaml knob ``c1.uploader.remote_root`` uses the standard rsync
    ``user@host:/path`` shape. If the path is omitted (no colon) we
    default to :data:`DEFAULT_DEST_ROOT`.
    """
    if not remote_root:
        return DEFAULT_DEST_HOST, DEFAULT_DEST_ROOT
    if ":" not in remote_root:
        return remote_root, DEFAULT_DEST_ROOT
    host, _, root = remote_root.partition(":")
    if not root:
        root = DEFAULT_DEST_ROOT
    return host, root


def build_rsync_argv(
    event_name: str,
    src_dir: Union[str, Path],
    dest_host: str,
    dest_root: str,
    *,
    rsync_path: str = "rsync",
    remote_rsync_bin: str = "rsync",
    extra_opts: Iterable[str] = (),
    bandwidth_limit_kbps: int = 0,
) -> list[str]:
    """Construct the rsync argv list for one event.

    Shape: ``rsync -a --stats --partial [bwlimit?] --rsync-path='mkdir
    -p <dest_dir> && <remote_rsync_bin>' [extra_opts...] <src>/
    <dest_host>:<dest_root>/<event>/cubes/``.

    The trailing slash on ``src`` is critical — rsync copies the
    directory CONTENTS (the NPZs themselves) into the destination
    ``cubes/`` rather than nesting an extra ``<event>/`` level.

    ``--rsync-path`` portably emulates ``--mkpath`` on rsync < 3.2.3
    (the lab ships 3.1.2 on n01..n13 + h23): the remote ssh shell
    runs ``mkdir -p`` before exec'ing the server-side rsync. The
    destination dir is ``<dest_root>/<event_name>/cubes/``; we shell-
    quote it so an exotic ``dest_root`` can't break the wrapper.
    """
    if not event_name or "/" in event_name:
        raise ValueError(f"bad event_name: {event_name!r}")
    src = str(Path(src_dir))
    if not src.endswith(os.sep):
        src = src + os.sep
    remote_dir = f"{dest_root.rstrip('/')}/{event_name}/cubes"
    dest = f"{dest_host}:{remote_dir}/"
    # --rsync-path runs an UNESCAPED command on the remote shell via
    # ssh; shlex-quote dest_root + event_name so the mkdir never
    # interprets a literal shell metachar.
    rsync_path_arg = (
        f"mkdir -p {shlex.quote(remote_dir)} && {remote_rsync_bin}"
    )
    argv: list[str] = [
        rsync_path,
        *DEFAULT_RSYNC_OPTS,
        f"--rsync-path={rsync_path_arg}",
    ]
    if bandwidth_limit_kbps and int(bandwidth_limit_kbps) > 0:
        argv.append(f"--bwlimit={int(bandwidth_limit_kbps)}")
    argv.extend(str(x) for x in extra_opts)
    argv.append(src)
    argv.append(dest)
    return argv


def upload_event_cubes(
    event_name: str,
    src_dir: Union[str, Path],
    dest_host: str = DEFAULT_DEST_HOST,
    dest_root: str = DEFAULT_DEST_ROOT,
    *,
    log_path: Optional[Union[str, Path]] = None,
    rsync_path: str = "rsync",
    extra_opts: Iterable[str] = (),
    bandwidth_limit_kbps: int = 0,
    popen=subprocess.Popen,
) -> subprocess.Popen:
    """Spawn a detached rsync that pushes ``src_dir/*`` to h23.

    The rsync runs in a NEW session (``setsid``) so the parent
    ``CubeDumpWriter`` thread can exit / be restarted without taking
    the rsync with it. stdout + stderr are appended to the per-event
    ``upload.log`` next to the NPZs (override via ``log_path`` for
    tests).

    Args:
        event_name: per-event archive name (matches the parent dir
            name on the search node and the ``<event>/`` dir on h23).
        src_dir: per-event dump directory on the search node (e.g.
            ``/home/ubuntu/data/c2/cube_dump/<event_name>``). The
            directory's CONTENTS are pushed (rsync trailing-slash
            semantics).
        dest_host: rsync remote endpoint host (``user@host``). Default
            is :data:`DEFAULT_DEST_HOST` (works from search nodes).
        dest_root: top-level remote candidate archive root. Default
            :data:`DEFAULT_DEST_ROOT`; the rsync lands files under
            ``<dest_root>/<event_name>/cubes/``.
        log_path: per-event log file. Default ``<src_dir>/upload.log``.
        rsync_path: ``rsync`` binary (overridable for tests / wrappers).
        extra_opts: extra rsync flags appended after the defaults.
        bandwidth_limit_kbps: maps to ``--bwlimit`` if > 0.
        popen: ``subprocess.Popen`` constructor (overridable so unit
            tests can capture the spawn without exec'ing).

    Returns:
        The spawned ``Popen`` handle. Callers MUST NOT ``.wait()`` on
        the real-time hot path; the writer thread treats it as
        fire-and-forget.
    """
    src = Path(src_dir)
    argv = build_rsync_argv(
        event_name,
        src,
        dest_host,
        dest_root,
        rsync_path=rsync_path,
        extra_opts=extra_opts,
        bandwidth_limit_kbps=bandwidth_limit_kbps,
    )
    log = Path(log_path) if log_path is not None else (src / "upload.log")
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _LOG.warning(
            "cube_uploader: could not mkdir log parent %s: %r",
            log.parent, exc,
        )
    # Append a header line; ``a+`` so a missing file is created.
    header = (
        f"# {_dt.datetime.utcnow().isoformat(timespec='milliseconds')}Z "
        f"event={event_name} src={src} dest={dest_host}:{dest_root} "
        f"argv={shlex.join(argv)}\n"
    )
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(header)
    except OSError as exc:
        _LOG.warning(
            "cube_uploader: could not write log header to %s: %r",
            log, exc,
        )
    # Open in append+binary mode for the subprocess's stdout/stderr;
    # the FD is dup2'd into the child by Popen and we close our copy
    # immediately so the writer thread doesn't hold a stale handle.
    log_fd = os.open(
        str(log),
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o644,
    )
    try:
        proc = popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from rt parent
            close_fds=True,
        )
    finally:
        os.close(log_fd)
    _LOG.info(
        "cube_uploader: spawned rsync pid=%s event=%s src=%s dest=%s:%s",
        getattr(proc, "pid", "?"),
        event_name,
        src,
        dest_host,
        dest_root,
    )
    return proc


# ---------------------------------------------------------------------------
# Bounded uploader (T5 — concurrency cap)
# ---------------------------------------------------------------------------


#: Default backlog the per-process bounded uploader will absorb before
#: dropping the oldest pending job. Sized to comfortably outlast a 60 s
#: dump-rate window at the production cap (``c2.dump_rate_max_per_window:
#: 6``) so a single short flood is queued losslessly; longer floods drop
#: the oldest events (newer ones are more diagnostically useful) so a
#: persistently-noisy source can never grow unbounded RAM/disk pressure.
DEFAULT_UPLOAD_QUEUE_MAXSIZE: int = 8


class BoundedCubeUploader:
    """Per-process upload serializer with bounded backlog (T5, 2026-06-07).

    Background
    ----------

    Pre-2026-06-07 every successful ``CubeDumpWriter._drain_loop`` write
    fired ``upload_event_cubes`` directly, which spawned a *detached*
    rsync via ``subprocess.Popen``. Each search node has 2 GPU halves;
    each half can fire one rsync per cube_dump completion. With C2's
    ``dump_rate_max_per_window: 6 / 60 s`` cap, the worst case is 6
    events × 4 nodes × 2 halves = 48 concurrent rsyncs of ~1.06 GiB each
    (the search-node side of an event archive). 48 simultaneous rsyncs
    over the corr-net starve incoming SNAP traffic, bumping the RFI
    flag fraction → flagged cubes → σ_k anomaly → search-side detector
    freeze (the 2026-06-07 failure mode) — the dump-storm fan-out
    closes the loop into the detector pathology.

    This class fixes that by serializing uploads PER PROCESS through a
    single worker thread bounded by ``max_concurrent`` parallel rsync
    processes (default 1). The worker pops pending jobs, spawns the
    rsync, ``proc.wait()``s for it to finish, and only then pops the
    next. Each search-compute half therefore has at most
    ``max_concurrent`` rsyncs in flight regardless of trigger rate; the
    fleet ceiling collapses from 48 to 4 × 2 × ``max_concurrent``
    concurrent rsyncs (= 8 at the default).

    Submit semantics
    ----------------

    * :meth:`submit` is non-blocking: it puts on the bounded queue with
      ``put_nowait`` and returns ``True`` on success or ``False`` on a
      full queue. On a full-queue submission the OLDEST queued job is
      evicted to make room (so newer / more diagnostically useful
      events win over old ones during a sustained backlog).
    * ``submit`` itself never spawns rsync — that runs on the worker
      thread. The dispatch path (the ``CubeDumpWriter.on_dump_complete``
      callback) returns in microseconds.
    * Counters (``n_submitted``, ``n_uploaded``, ``n_dropped_full``,
      ``n_failed``) are read-only properties, safe to inspect from
      operator threads.

    Lifecycle
    ---------

    Instantiate at service start, call :meth:`start` once, call
    :meth:`submit` from the dispatch thread, call :meth:`stop` at
    shutdown to drain the queue.
    """

    _SENTINEL = object()

    def __init__(
        self,
        *,
        dest_host: str,
        dest_root: str,
        max_concurrent: int = 1,
        queue_maxsize: int = DEFAULT_UPLOAD_QUEUE_MAXSIZE,
        bandwidth_limit_kbps: int = 0,
        rsync_path: str = "rsync",
        extra_opts: Iterable[str] = (),
        log_path_factory: Optional[
            "Any"
        ] = None,
        popen: Any = subprocess.Popen,
        upload_fn: Any = upload_event_cubes,
        thread_name: str = "cube-upload",
        purge_pattern: Optional[str] = None,
    ) -> None:
        """``purge_pattern`` (2026-07-19 disk-full incident): when set,
        the worker deletes the ``src_dir`` files matching this glob
        after a *successful* (rc==0) rsync, and appends an
        ``UPLOAD_OK`` marker line to the per-event ``upload.log`` (the
        marker is what :mod:`dsart.dump.cube_retention` uses to prefer
        already-uploaded dirs when the size cap bites). The set of
        files to purge is snapshotted BEFORE the rsync is spawned so a
        file that lands mid-transfer (and is therefore NOT in the
        rsync's file list) is never deleted. Production wires the
        pattern to THIS HALF's files only
        (``cube_*_g<gpu_half>_*.npz``) so the two halves never delete
        each other's not-yet-uploaded cubes. ``None`` (default)
        preserves the historical keep-everything behaviour."""
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent={max_concurrent}, expected >= 1"
            )
        if queue_maxsize < 1:
            raise ValueError(
                f"queue_maxsize={queue_maxsize}, expected >= 1"
            )
        self.dest_host = str(dest_host)
        self.dest_root = str(dest_root)
        self._max_concurrent = int(max_concurrent)
        self._queue_maxsize = int(queue_maxsize)
        self._bandwidth_limit_kbps = int(bandwidth_limit_kbps)
        self._rsync_path = str(rsync_path)
        self._extra_opts = tuple(extra_opts)
        self._log_path_factory = log_path_factory
        self._popen = popen
        self._upload_fn = upload_fn
        self._queue: queue.Queue = queue.Queue(maxsize=self._queue_maxsize)
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._thread_name = str(thread_name)
        self._purge_pattern = (
            str(purge_pattern) if purge_pattern else None
        )
        self._started = False
        self._stopped = False
        self._n_submitted = 0
        self._n_uploaded = 0
        self._n_dropped_full = 0
        self._n_failed = 0
        self._n_purged_files = 0

    @property
    def n_submitted(self) -> int:
        return self._n_submitted

    @property
    def n_uploaded(self) -> int:
        return self._n_uploaded

    @property
    def n_dropped_full(self) -> int:
        """Submissions that evicted an older queued job (queue was
        full at submit time)."""
        return self._n_dropped_full

    @property
    def n_failed(self) -> int:
        """Worker-side rsync exits with a non-zero return code or a
        ``Popen`` exception."""
        return self._n_failed

    @property
    def n_purged_files(self) -> int:
        """Local NPZs deleted after a verified (rc==0) upload."""
        return self._n_purged_files

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_maxsize(self) -> int:
        return self._queue_maxsize

    def start(self) -> None:
        if self._started:
            return
        for i in range(self._max_concurrent):
            t = threading.Thread(
                target=self._drain_loop,
                name=f"{self._thread_name}-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        self._started = True
        _LOG.info(
            "BoundedCubeUploader started: dest=%s:%s max_concurrent=%d "
            "queue_maxsize=%d",
            self.dest_host, self.dest_root,
            self._max_concurrent, self._queue_maxsize,
        )

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        for _ in self._workers:
            try:
                self._queue.put_nowait(self._SENTINEL)
            except queue.Full:
                # Make room by popping one job — sentinel must land.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(self._SENTINEL)
        for t in self._workers:
            t.join(timeout=10.0)

    def submit(self, event_name: str, src_dir: Union[str, Path]) -> bool:
        """Enqueue an event upload. Non-blocking; on a full queue the
        oldest pending job is evicted to make room and ``False`` is
        returned (the new job IS still queued)."""
        if not self._started:
            raise RuntimeError("BoundedCubeUploader.submit() before start()")
        if self._stopped:
            raise RuntimeError("BoundedCubeUploader.submit() after stop()")
        item = (str(event_name), Path(src_dir))
        with self._lock:
            self._n_submitted += 1
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                # Evict the oldest job + replace it with the new one.
                try:
                    evicted = self._queue.get_nowait()
                    if evicted is not self._SENTINEL:
                        self._n_dropped_full += 1
                        ev_name = (
                            evicted[0] if isinstance(evicted, tuple)
                            else "?"
                        )
                        _LOG.warning(
                            "BoundedCubeUploader: queue full -> evicting "
                            "oldest event=%s in favour of event=%s "
                            "(dest=%s:%s queue_maxsize=%d)",
                            ev_name, event_name,
                            self.dest_host, self.dest_root,
                            self._queue_maxsize,
                        )
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:                                # noqa: BLE001
                    self._n_dropped_full += 1
                    return False
                return False

    def _drain_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                return
            event_name, src_dir = item
            # Snapshot the purge set BEFORE spawning rsync: rsync
            # builds its file list at startup, so anything in this
            # snapshot is transferred by an rc==0 run, and anything
            # that lands later is left alone (it will ride the next
            # submit for this event, or the retention sweeper).
            purge_candidates: list[Path] = []
            if self._purge_pattern:
                try:
                    purge_candidates = sorted(
                        Path(src_dir).glob(self._purge_pattern)
                    )
                except OSError:
                    purge_candidates = []
            try:
                proc = self._upload_fn(
                    event_name=event_name,
                    src_dir=src_dir,
                    dest_host=self.dest_host,
                    dest_root=self.dest_root,
                    rsync_path=self._rsync_path,
                    extra_opts=self._extra_opts,
                    bandwidth_limit_kbps=self._bandwidth_limit_kbps,
                    popen=self._popen,
                )
            except Exception as exc:                              # noqa: BLE001
                self._n_failed += 1
                _LOG.warning(
                    "BoundedCubeUploader: spawn failed for event=%s "
                    "(src=%s dest=%s:%s): %s",
                    event_name, src_dir,
                    self.dest_host, self.dest_root, exc,
                )
                continue
            try:
                rc = proc.wait()
            except Exception as exc:                              # noqa: BLE001
                self._n_failed += 1
                _LOG.warning(
                    "BoundedCubeUploader: wait failed for event=%s "
                    "pid=%s: %s",
                    event_name, getattr(proc, "pid", "?"), exc,
                )
                continue
            if rc != 0:
                self._n_failed += 1
                _LOG.warning(
                    "BoundedCubeUploader: rsync exited rc=%d for event=%s "
                    "pid=%s (see <src>/upload.log)",
                    rc, event_name, getattr(proc, "pid", "?"),
                )
            else:
                self._n_uploaded += 1
                self._purge_after_upload(event_name, src_dir,
                                         purge_candidates)

    def _purge_after_upload(
        self,
        event_name: str,
        src_dir: Union[str, Path],
        purge_candidates: "list[Path]",
    ) -> None:
        """Delete the pre-snapshot files after a verified upload and
        append an ``UPLOAD_OK`` marker to the per-event log.

        2026-07-19: staged cubes were never deleted anywhere, so
        ~2.2 GB/event/node accumulated until all four search nodes hit
        100% disk. Deleting only after an rc==0 rsync keeps the
        archive the single source of truth; the marker line lets the
        retention sweeper distinguish uploaded from stranded dirs.
        """
        if not self._purge_pattern:
            return
        n = 0
        for f in purge_candidates:
            try:
                f.unlink()
                n += 1
            except FileNotFoundError:
                pass  # other-half sweeper / operator raced us: fine
            except OSError as exc:
                _LOG.warning(
                    "BoundedCubeUploader: purge failed for %s: %r", f, exc
                )
        self._n_purged_files += n
        marker = (
            f"# {_dt.datetime.utcnow().isoformat(timespec='milliseconds')}Z "
            f"UPLOAD_OK event={event_name} rc=0 purged={n}\n"
        )
        try:
            with (Path(src_dir) / "upload.log").open(
                "a", encoding="utf-8"
            ) as fh:
                fh.write(marker)
        except OSError as exc:
            _LOG.warning(
                "BoundedCubeUploader: could not append UPLOAD_OK marker "
                "for event=%s: %r", event_name, exc,
            )
        if n:
            _LOG.info(
                "BoundedCubeUploader: purged %d uploaded NPZ(s) for "
                "event=%s", n, event_name,
            )
