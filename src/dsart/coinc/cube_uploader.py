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
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Union

__all__ = [
    "DEFAULT_DEST_HOST",
    "DEFAULT_DEST_ROOT",
    "DEFAULT_RSYNC_OPTS",
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
