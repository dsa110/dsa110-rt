"""Fringe-stopping-table panel for the Control tab.

Lets the operator inspect which per-DEC fringe-stopping cache files
exist on h23 (the master copy) and on each of the 16 corr nodes,
build a new table for an arbitrary DEC in the ``casa38`` conda env
on h23, and rsync that file out to the corr nodes' production cache
dir (``/home/ubuntu/data/fstables/``).

Why this exists
===============

``meridian_fringestop`` (corr-side service) refuses to start when its
per-DEC fringe-stopping cache file is missing or mismatched. Once it
crashes, ``corr_slow`` blocks writing to ``bada`` (which would
normally drain into meridian_fringestop), the multi-reader ``fada``
ring deadlocks against ``corr_slow``'s stall, and ~300 cubes later
``corr_fast`` hangs in ``getNextPage()``. The whole pipeline grinds
to a halt with a cryptic-looking ``ipcbuf_get_next_read: error
decrement FULL`` error far downstream of the actual root cause.

Surfacing fstable presence as a control-tab traffic light next to
"Start fleet" — and giving the operator a one-click way to mint a
new table for a new DEC — eliminates this whole failure mode.

Storage layout
==============

* **h23 (this host) — master copy**: ``<repo>/var/fstables/*.npz``,
  gitignored. The dashboard reads here for the "h23 column" of the
  inventory and writes here when building a new table.
* **corr nodes — production read path**: ``/home/ubuntu/data/fstables/``.
  ``meridian_fringestop`` reads from there (``DEFAULT_FSTABLE_CACHE``
  in ``tools/ops/meridian_fringestop_rt.py``).

Filenames are pinned by
:func:`dsart.ops.meridian_fringestop_rt.cache_table_filename`
(re-exported here via :func:`expected_filename` to keep this module
import-cheap and stdlib-only).

Build environment
=================

``tools/build_fstable_cache.py`` lives in the ``dsa110-rt`` repo but
imports ``dsamfs.fringestopping`` and ``dsacalib`` at runtime —
neither package is installed in the dashboard's ``dsart_h23`` env.
The build sub-process is invoked via the host's pre-existing
``casa38`` env at ``CASA38_PY`` (which the SEFD scanner already
relies on), so the dashboard process itself never imports the casa
stack.

The build is run synchronously (~30 s per DEC) from a Flask route;
that's well under typical browser timeouts, so we don't bother with
job-queue plumbing. If the operator ever wants to build the full
461-DEC bank from the dashboard we'd want background jobs — out of
scope for this panel.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:                                                       # pragma: no cover
    # Reuse the canonical filename helper so the dashboard and the
    # corr-side service agree byte-for-byte on every per-DEC name.
    from sys import path as _sys_path
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    _sys_path.insert(0, str(_REPO_ROOT / "tools"))
    from ops.meridian_fringestop_rt import (              # noqa: E402
        cache_table_filename as _cache_table_filename,
        snap_dec_to_grid as _snap_dec_to_grid,
        DEFAULT_DEC_GRID_STEP as _DEFAULT_DEC_GRID_STEP,
    )
except Exception:                                          # noqa: BLE001
    # Tests can import this module without the sibling tool tree
    # present; fall back to local copies of the two pure helpers.
    _DEFAULT_DEC_GRID_STEP = 0.25

    def _snap_dec_to_grid(dec_deg: float, step_deg: float = _DEFAULT_DEC_GRID_STEP) -> float:
        if step_deg <= 0.0:
            raise ValueError(f"step_deg must be > 0, got {step_deg!r}")
        return round(dec_deg / step_deg) * step_deg

    def _cache_table_filename(dec_deg_grid: float, nant: int, refmjd: float) -> str:
        return (
            f"fringestopping_table_dec_{dec_deg_grid:+08.4f}deg_"
            f"{int(nant)}ant_refmjd{float(refmjd):.6f}.npz"
        )


LOG = logging.getLogger("dsa_monitor.fstable_panel")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Master copy of every fstable on the dashboard host. Gitignored.
H23_MASTER_DIR: Path = Path(__file__).resolve().parents[3] / "var" / "fstables"

#: Production read path on every corr node. Matches
#: :data:`tools.ops.meridian_fringestop_rt.DEFAULT_FSTABLE_CACHE`.
CORR_FSTABLE_DIR: str = "/home/ubuntu/data/fstables"

#: casa38 conda-env python on h23. The build script imports dsamfs +
#: dsacalib at runtime, so we MUST shell out to casa38.
CASA38_PY: str = "/home/ubuntu/anaconda3/envs/casa38/bin/python"

#: Path of ``tools/build_fstable_cache.py`` relative to repo root.
BUILD_SCRIPT_REL: str = "tools/build_fstable_cache.py"

#: Default grid step the runtime snaps DEC to (see
#: ``meridian_fringestop_rt.DEFAULT_DEC_GRID_STEP``). Mirrored here so
#: tests can pin a single source of truth.
DEC_GRID_STEP: float = _DEFAULT_DEC_GRID_STEP

#: Time budget for the casa38 build sub-process. One DEC takes
#: ~4-5 minutes (M2: 263 s for nant=96 + nbls=4656 + nint=24 on h23,
#: 2026-06-03). Cap at 15 min so a hung sub-process still surfaces
#: in the UI as a failure rather than holding the dashboard worker
#: forever, while leaving generous headroom over the typical case.
DEFAULT_BUILD_TIMEOUT_S: float = 900.0

#: Time budget for one rsync to one corr node. The .npz is ~3.5 MB
#: so even on a slow link this is plenty.
DEFAULT_RSYNC_TIMEOUT_S: float = 30.0

#: Time budget for ``ssh host ls`` queries. The inventory fan-out is
#: parallelised; each one is bounded individually.
DEFAULT_INVENTORY_TIMEOUT_S: float = 8.0

#: Number of parallel inventory + deploy threads. 16 corr nodes; one
#: thread per node leaves the dashboard CPU effectively idle.
INVENTORY_PARALLELISM: int = 16

#: Standard ssh args, mirrored from :mod:`fleet_update` so behaviour
#: is consistent with the rest of the dashboard. Always
#: ``BatchMode=yes`` + ``-n`` so no interactive prompts.
_SSH_BASE_ARGS: tuple[str, ...] = (
    "ssh",
    "-o", "ConnectTimeout=5",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-n",
)

#: Regex for the cache filename. Matches what
#: :func:`expected_filename` writes. Captures dec / nant / refmjd.
_FNAME_RE = re.compile(
    r"^fringestopping_table_dec_(?P<dec>[+-]\d+\.\d{4})deg_"
    r"(?P<nant>\d+)ant_refmjd(?P<refmjd>\d+\.\d{6})\.npz$"
)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def snap_dec_to_grid(dec_deg: float, step_deg: float = DEC_GRID_STEP) -> float:
    """Snap ``dec_deg`` to the runtime cache grid. Public re-export."""
    return _snap_dec_to_grid(dec_deg, step_deg)


def expected_filename(dec_deg: float, nant: int, refmjd: float) -> str:
    """The filename meridian_fringestop expects for this (DEC, nant, refmjd).

    ``dec_deg`` is snapped to the grid before formatting so callers
    can pass the raw operator-supplied DEC directly.
    """
    dec_grid = snap_dec_to_grid(dec_deg)
    return _cache_table_filename(dec_grid, nant, refmjd)


def parse_filename(name: str) -> Optional[dict[str, Any]]:
    """Inverse of :func:`expected_filename`. Returns dec/nant/refmjd
    parsed from ``name`` or ``None`` if it doesn't match the scheme.
    """
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    return {
        "dec_deg": float(m.group("dec")),
        "nant": int(m.group("nant")),
        "refmjd": float(m.group("refmjd")),
    }


# ---------------------------------------------------------------------------
# Entry dataclasses (JSON-serialisable shape for the frontend)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FstableEntry:
    """One per-host (or per-h23) row of the inventory.

    ``host`` is ``"h23"`` for the master copy, ``n03.pro.pvt`` etc.
    for the corr nodes. ``filename`` is the on-disk basename.
    """
    host: str
    filename: str
    dec_deg: float
    nant: int
    refmjd: float
    size_bytes: int
    mtime_unix: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# h23-side filesystem helpers
# ---------------------------------------------------------------------------


def ensure_master_dir() -> Path:
    """Create the h23 master fstable dir if it doesn't exist. Returns
    the path. Operator's first build call lands here.
    """
    H23_MASTER_DIR.mkdir(parents=True, exist_ok=True)
    return H23_MASTER_DIR


def list_h23_tables() -> list[FstableEntry]:
    """Enumerate the master-copy ``*.npz`` files on h23.

    Returns an empty list if the master dir doesn't exist yet (the
    operator hasn't built anything). Files whose names don't match
    :func:`parse_filename` are skipped with a warning so a stray
    file in the dir doesn't blow up the page.
    """
    if not H23_MASTER_DIR.exists():
        return []
    out: list[FstableEntry] = []
    for p in sorted(H23_MASTER_DIR.iterdir()):
        if not p.is_file() or not p.name.endswith(".npz"):
            continue
        meta = parse_filename(p.name)
        if meta is None:
            LOG.warning("skipping non-cache file in master dir: %s", p)
            continue
        st = p.stat()
        out.append(FstableEntry(
            host="h23",
            filename=p.name,
            dec_deg=meta["dec_deg"],
            nant=meta["nant"],
            refmjd=meta["refmjd"],
            size_bytes=int(st.st_size),
            mtime_unix=float(st.st_mtime),
        ))
    return out


# ---------------------------------------------------------------------------
# ssh / rsync wrappers (kept thin so tests can monkey-patch subprocess.run)
# ---------------------------------------------------------------------------


def _ssh_run(
    host: str,
    remote_cmd: str,
    *,
    timeout_s: float,
) -> tuple[int, str, str]:
    """``(rc, stdout, stderr)`` from ``ssh host remote_cmd``. Raises
    :class:`subprocess.TimeoutExpired` on timeout.
    """
    args = [*_SSH_BASE_ARGS, host, remote_cmd]
    cp = subprocess.run(                                    # noqa: S603
        args,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return cp.returncode, cp.stdout, cp.stderr


# ---------------------------------------------------------------------------
# Corr-node inventory (parallel ssh ls)
# ---------------------------------------------------------------------------


def list_corr_tables(
    host: str,
    *,
    timeout_s: float = DEFAULT_INVENTORY_TIMEOUT_S,
) -> tuple[bool, list[FstableEntry], Optional[str]]:
    """List ``*.npz`` cache files on one corr node.

    Returns ``(ok, entries, error)``. ``error`` is non-None when the
    ssh fan-out failed (timeout, host unreachable, etc.) — the
    dashboard surfaces that to the operator separately from the
    "no tables yet" case.

    The remote command intentionally uses ``stat -c`` instead of
    a tar/json round-trip so a clean fleet has near-zero per-host
    overhead; one stat per file across at most a few hundred files
    is negligible.
    """
    # `set -o pipefail` so a missing dir fails fast; the find -maxdepth
    # 1 keeps the listing flat (we don't recurse into the cache dir).
    remote_cmd = (
        f"set -o pipefail; "
        f"test -d {CORR_FSTABLE_DIR} || exit 2; "
        f"find {CORR_FSTABLE_DIR} -maxdepth 1 -name '*.npz' -type f "
        f"-printf '%f\\t%s\\t%T@\\n' 2>/dev/null | sort"
    )
    try:
        rc, stdout, stderr = _ssh_run(host, remote_cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return (False, [], f"ssh_timeout after {timeout_s:.0f}s")
    except Exception as exc:                               # noqa: BLE001
        return (False, [], f"ssh_error: {exc!r}")

    if rc == 2:
        return (True, [], None)                            # dir missing → empty
    if rc != 0:
        return (False, [], f"ssh rc={rc} stderr={stderr.strip()!r}")

    entries: list[FstableEntry] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            name, size_s, mtime_s = line.split("\t")
            meta = parse_filename(name)
            if meta is None:
                LOG.warning("%s: skipping non-cache file %s", host, name)
                continue
            entries.append(FstableEntry(
                host=host,
                filename=name,
                dec_deg=meta["dec_deg"],
                nant=meta["nant"],
                refmjd=meta["refmjd"],
                size_bytes=int(size_s),
                mtime_unix=float(mtime_s),
            ))
        except ValueError:
            LOG.warning("%s: skipping malformed line %r", host, line)
            continue
    return (True, entries, None)


def list_fleet_tables(
    corr_hosts: Sequence[str],
    *,
    timeout_s: float = DEFAULT_INVENTORY_TIMEOUT_S,
    max_workers: int = INVENTORY_PARALLELISM,
) -> dict[str, dict[str, Any]]:
    """Fan ``list_corr_tables`` across every corr host. Returns
    ``{host: {ok, entries, error}}`` keyed by host.
    """
    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(corr_hosts)))) as ex:
        futs = {
            ex.submit(list_corr_tables, h, timeout_s=timeout_s): h
            for h in corr_hosts
        }
        for fut in as_completed(futs):
            host = futs[fut]
            try:
                ok, entries, err = fut.result()
            except Exception as exc:                       # noqa: BLE001
                ok, entries, err = (False, [], f"future_error: {exc!r}")
            out[host] = {
                "ok": bool(ok),
                "entries": [e.to_dict() for e in entries],
                "error": err,
            }
    return out


# ---------------------------------------------------------------------------
# Current-DEC status (the traffic light feed)
# ---------------------------------------------------------------------------


def current_dec_status(
    dec_deg: Optional[float],
    nant: int,
    refmjd: float,
    corr_hosts: Sequence[str],
    *,
    fleet_inventory: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compose the traffic-light payload for the current pointing DEC.

    ``dec_deg`` is the operator-supplied / etcd-derived DEC (raw,
    pre-snap). ``nant`` + ``refmjd`` come from the runtime config
    (caller fetches them from etcd / yaml). ``fleet_inventory`` is
    the output of :func:`list_fleet_tables`; pass it in if you
    already have it (saves the second fan-out).

    Returns a dict with::

      {
        "ok": True,
        "dec_deg_raw": 54.5734,
        "dec_deg_grid": 54.5,
        "expected_filename": "fringestopping_table_dec_+54.5000deg_96ant_refmjd58849.000000.npz",
        "h23_has_table": True | False,
        "corr_hosts_with_table": [...],
        "corr_hosts_missing_table": [...],
        "corr_hosts_unreachable": [...],
        "all_ready": True | False,
        "light": "green" | "amber" | "red" | "unknown",
        "message": "..."
      }

    Light semantics (driven by corr-node coverage — that's what
    actually decides whether ``meridian_fringestop`` will crash at
    startup. ``h23`` master is for redeploy convenience and is
    surfaced separately in the detail string but does not affect
    the light):

    * **green**: every reachable corr node has the table → pipeline
      will start cleanly.
    * **amber**: at least one corr node has it but not all → mixed
      fleet, hit Deploy (or Build then Deploy if h23 master is
      also missing).
    * **red**: no reachable corr node has it → ``meridian_fringestop``
      WILL crash at startup; build + deploy before hitting Start.
    * **unknown**: ``dec_deg`` is None (no DEC yet) OR every corr
      node was unreachable (can't tell).
    """
    if dec_deg is None:
        return {
            "ok": True,
            "dec_deg_raw": None,
            "dec_deg_grid": None,
            "expected_filename": None,
            "h23_has_table": False,
            "corr_hosts_with_table": [],
            "corr_hosts_missing_table": [],
            "corr_hosts_unreachable": [],
            "all_ready": False,
            "light": "unknown",
            "message": "no pointing DEC in etcd / form — set obs_dec_deg first",
        }

    dec_grid = snap_dec_to_grid(float(dec_deg))
    expected = expected_filename(dec_grid, nant, refmjd)

    h23_files = {e.filename for e in list_h23_tables()}
    h23_has = expected in h23_files

    if fleet_inventory is None:
        fleet_inventory = list_fleet_tables(corr_hosts)

    have: list[str] = []
    missing: list[str] = []
    unreachable: list[str] = []
    for host in corr_hosts:
        info = fleet_inventory.get(host)
        if info is None or not info.get("ok"):
            unreachable.append(host)
            continue
        names = {e["filename"] for e in info.get("entries", [])}
        if expected in names:
            have.append(host)
        else:
            missing.append(host)

    n_reachable = len(have) + len(missing)
    if n_reachable == 0:
        # Every corr node was unreachable; can't tell either way.
        light = "unknown"
        msg = (
            f"unable to query any corr node ({len(unreachable)} unreachable); "
            f"cannot verify dec={dec_grid:+.4f} fstable presence."
        )
    elif not have:
        # No corr node has it → meridian_fringestop will crash.
        light = "red"
        h23_note = (
            "h23 master also missing — Build then Deploy."
            if not h23_has else
            "h23 has the master — click Deploy."
        )
        msg = (
            f"no corr node has the dec={dec_grid:+.4f} table "
            f"(grid-snapped from {dec_deg:+.4f}); meridian_fringestop "
            f"WILL crash at startup. {h23_note}"
        )
    elif missing or unreachable:
        # Partial coverage — needs a deploy to fully resolve.
        light = "amber"
        h23_note = (
            "h23 master missing — Build then Deploy."
            if not h23_has else
            "click Deploy to fan it out."
        )
        msg = (
            f"dec={dec_grid:+.4f} table present on {len(have)}/"
            f"{n_reachable} corr node(s); "
            f"{len(missing)} missing, {len(unreachable)} unreachable. "
            f"{h23_note}"
        )
    else:
        light = "green"
        h23_note = "" if h23_has else " (h23 master missing — build to keep a redeploy copy)"
        msg = (
            f"dec={dec_grid:+.4f} table present on all {len(have)} "
            f"reachable corr node(s).{h23_note}"
        )

    return {
        "ok": True,
        "dec_deg_raw": float(dec_deg),
        "dec_deg_grid": float(dec_grid),
        "expected_filename": expected,
        "h23_has_table": bool(h23_has),
        "corr_hosts_with_table": have,
        "corr_hosts_missing_table": missing,
        "corr_hosts_unreachable": unreachable,
        # all_ready tracks the traffic light: True iff the pipeline
        # will start cleanly for this DEC (light == "green"). h23
        # master absence is a redeploy concern, not a runtime one.
        "all_ready": bool(light == "green"),
        "light": light,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Build (h23 casa38 sub-process)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    dec_deg_raw: float
    dec_deg_grid: float
    expected_filename: str
    output_path: Optional[str]
    stdout_tail: str
    stderr_tail: str
    elapsed_s: float
    rc: Optional[int]
    error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _tail(s: str, n: int = 4096) -> str:
    if len(s) <= n:
        return s
    return f"…[{len(s) - n} bytes elided]…\n" + s[-n:]


def build_table_for_dec(
    dec_deg: float,
    *,
    nant: int,
    refmjd: float,
    repo_root: Optional[Path] = None,
    casa38_py: str = CASA38_PY,
    output_dir: Optional[Path] = None,
    timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    force: bool = False,
    dec_grid_step: float = DEC_GRID_STEP,
) -> BuildResult:
    """Invoke ``casa38_py tools/build_fstable_cache.py`` for one DEC.

    ``dec_deg`` is the operator's raw value; we snap to the grid
    before passing to the build script so the resulting filename
    matches what meridian_fringestop will look for. ``nant`` and
    ``refmjd`` are NOT passed to the build script — they're sourced
    inside it via ``--from-etcd``; we use them here only to predict
    the output filename.
    """
    t0 = time.monotonic()
    dec_grid = _snap_dec_to_grid(float(dec_deg), dec_grid_step)
    expected = _cache_table_filename(dec_grid, nant, refmjd)
    out_dir = output_dir if output_dir is not None else ensure_master_dir()
    out_path = out_dir / expected

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    build_script = repo_root / BUILD_SCRIPT_REL
    if not build_script.exists():
        return BuildResult(
            ok=False, dec_deg_raw=float(dec_deg), dec_deg_grid=float(dec_grid),
            expected_filename=expected, output_path=str(out_path),
            stdout_tail="", stderr_tail=f"missing build script: {build_script}",
            elapsed_s=0.0, rc=None,
            error=f"build script not found: {build_script}",
        )

    args = [
        casa38_py,
        "-u",
        str(build_script),
        "--from-etcd",
        "--dec-min", f"{dec_grid:.6f}",
        "--dec-max", f"{dec_grid:.6f}",
        "--dec-step", f"{dec_grid_step:.6f}",
        "--output-dir", str(out_dir),
    ]
    if force:
        args.append("--force")

    LOG.info("fstable build: %s", " ".join(args))
    try:
        cp = subprocess.run(                                # noqa: S603
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - t0
        return BuildResult(
            ok=False, dec_deg_raw=float(dec_deg), dec_deg_grid=float(dec_grid),
            expected_filename=expected, output_path=str(out_path),
            stdout_tail=_tail((exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")),
            stderr_tail=_tail((exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")),
            elapsed_s=elapsed, rc=None,
            error=f"build timed out after {timeout_s:.0f}s",
        )
    except Exception as exc:                                # noqa: BLE001
        elapsed = time.monotonic() - t0
        return BuildResult(
            ok=False, dec_deg_raw=float(dec_deg), dec_deg_grid=float(dec_grid),
            expected_filename=expected, output_path=str(out_path),
            stdout_tail="", stderr_tail="",
            elapsed_s=elapsed, rc=None,
            error=f"subprocess error: {exc!r}",
        )

    elapsed = time.monotonic() - t0
    rc = int(cp.returncode)
    produced = out_path.exists()
    ok = bool(rc == 0 and produced)
    return BuildResult(
        ok=ok, dec_deg_raw=float(dec_deg), dec_deg_grid=float(dec_grid),
        expected_filename=expected, output_path=str(out_path),
        stdout_tail=_tail(cp.stdout or ""), stderr_tail=_tail(cp.stderr or ""),
        elapsed_s=elapsed, rc=rc,
        error=None if ok else (
            f"build rc={rc}; file " + ("exists" if produced else "missing")
        ),
    )


# ---------------------------------------------------------------------------
# Deploy (rsync to corr nodes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeployHostResult:
    ok: bool
    host: str
    rc: Optional[int]
    error: Optional[str]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _rsync_one(
    host: str,
    src: Path,
    *,
    timeout_s: float,
) -> DeployHostResult:
    t0 = time.monotonic()
    if shutil.which("rsync") is None:
        return DeployHostResult(
            ok=False, host=host, rc=None,
            error="rsync not in PATH on h23",
            elapsed_s=time.monotonic() - t0,
        )

    args = [
        "rsync",
        "-a",
        "--partial",
        "-e",
        " ".join(_SSH_BASE_ARGS[:1] + tuple(
            a for a in _SSH_BASE_ARGS[1:] if a != "-n"
        )),
        str(src),
        f"{host}:{CORR_FSTABLE_DIR}/",
    ]
    try:
        cp = subprocess.run(                                # noqa: S603
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return DeployHostResult(
            ok=False, host=host, rc=None,
            error=f"rsync timed out after {timeout_s:.0f}s",
            elapsed_s=time.monotonic() - t0,
        )
    except Exception as exc:                                # noqa: BLE001
        return DeployHostResult(
            ok=False, host=host, rc=None,
            error=f"rsync subprocess error: {exc!r}",
            elapsed_s=time.monotonic() - t0,
        )
    rc = int(cp.returncode)
    err = None if rc == 0 else (
        f"rc={rc} stderr={_tail(cp.stderr or '', 512)!r}"
    )
    return DeployHostResult(
        ok=rc == 0, host=host, rc=rc, error=err,
        elapsed_s=time.monotonic() - t0,
    )


def deploy_table_to_fleet(
    filename: str,
    corr_hosts: Sequence[str],
    *,
    timeout_s: float = DEFAULT_RSYNC_TIMEOUT_S,
    max_workers: int = INVENTORY_PARALLELISM,
) -> dict[str, Any]:
    """rsync one cached ``*.npz`` from h23 master to every corr host.

    Returns a JSON-ready dict with the per-host result list and a
    boolean ``all_ok``.
    """
    src = H23_MASTER_DIR / filename
    if not src.is_file():
        return {
            "ok": False,
            "all_ok": False,
            "error": f"h23 master copy missing: {src}",
            "hosts": [],
        }
    parsed = parse_filename(filename)
    if parsed is None:
        return {
            "ok": False,
            "all_ok": False,
            "error": f"filename does not match cache scheme: {filename!r}",
            "hosts": [],
        }

    results: list[DeployHostResult] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(corr_hosts)))) as ex:
        futs = {
            ex.submit(_rsync_one, h, src, timeout_s=timeout_s): h
            for h in corr_hosts
        }
        for fut in as_completed(futs):
            host = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:                        # noqa: BLE001
                results.append(DeployHostResult(
                    ok=False, host=host, rc=None,
                    error=f"future_error: {exc!r}",
                    elapsed_s=0.0,
                ))

    # Stable sort so the UI never flickers.
    results.sort(key=lambda r: r.host)
    return {
        "ok": True,
        "all_ok": all(r.ok for r in results),
        "filename": filename,
        "parsed": parsed,
        "hosts": [r.to_dict() for r in results],
    }


__all__ = [
    "H23_MASTER_DIR",
    "CORR_FSTABLE_DIR",
    "CASA38_PY",
    "BUILD_SCRIPT_REL",
    "DEC_GRID_STEP",
    "DEFAULT_BUILD_TIMEOUT_S",
    "DEFAULT_RSYNC_TIMEOUT_S",
    "DEFAULT_INVENTORY_TIMEOUT_S",
    "FstableEntry",
    "BuildResult",
    "DeployHostResult",
    "snap_dec_to_grid",
    "expected_filename",
    "parse_filename",
    "ensure_master_dir",
    "list_h23_tables",
    "list_corr_tables",
    "list_fleet_tables",
    "current_dec_status",
    "build_table_for_dec",
    "deploy_table_to_fleet",
]
