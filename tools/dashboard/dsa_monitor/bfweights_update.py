"""Beamformer-weights update trigger for the SEFDs page.

Operators refresh the array calibration by running
``update_bfweights.py`` inside the ``calibration23`` LXD container on
h23 (script lives in ``/home/ubuntu/dsa-notebooks``, runs in the
``casa38`` conda env as user ``ubuntu``). The script takes a single
``<SRC>_<ISOT>`` descriptor naming a per-source solution file

    /dataz/dsa110/operations/beamformer_weights/generated/
        beamformer_weights_<SRC>_<ISOT>.yaml

re-averages it, writes a new dated fleet YAML, starts the user-level
``bfweights_copy.service`` to distribute the weights, and publishes
``{"cmd": "update_weights"}`` to ``/mon/cal/bfweights`` for the cal
service to apply. End-to-end it takes ~2-5 min (a fixed 63 s of sleeps
plus averaging I/O).

This module gives the dashboard three pieces:

* :func:`latest_descriptors` — newest available descriptor per SEFD
  calibrator, rendered next to the per-source "Update cals" buttons;
* :func:`start_update` — kick the container run in a daemon thread
  (single-flight: one update at a time, 409 otherwise);
* :func:`job_snapshot` — poll payload for the in-page status banner.

An audit row is appended via ``control_store.audit_log`` (namespace
``cal.update_bfweights``) at start and again at completion, mirroring
the fleet_update.py convention.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dsa_monitor.bfweights_update")

#: h23 view of the container's ``beamformer_dir`` (same filesystem via
#: the /dataz mount; the container sees it as
#: ``/operations/beamformer_weights/generated/``).
GENERATED_DIR = "/dataz/dsa110/operations/beamformer_weights/generated"

# 2026-07-31: the calibration23 LXC container is gone -- the calibration
# services now run as native systemd --user units on h23 (see
# tools/calibration/MIGRATION.md), so `lxc exec calibration23` would now fail
# outright. The update runs locally instead.
#
# The dashboard already runs as `ubuntu` with a live user session (linger on),
# so the XDG_RUNTIME_DIR / DBUS_SESSION_BUS_ADDRESS the container invocation had
# to inject by hand are inherited -- update_bfweights.py can reach
# `systemctl --user` for bfweights_copy.service without them.
UPDATE_WORKDIR = "/home/ubuntu/vikram/dev/dsa110-rt/tools/calibration/scripts"
UPDATE_PYTHON = "/home/ubuntu/anaconda3/envs/casa38/bin/python"
UPDATE_SCRIPT = "update_bfweights.py"

#: The script itself sleeps ~63 s and the averaging step reads/writes
#: multi-GB weight blobs; give the whole run a generous ceiling.
RUN_TIMEOUT_S = 1800

#: Descriptor shape we accept from the filesystem, e.g.
#: ``2253+161_2026-07-08T11:42:38``. Also serves as the anti-injection
#: gate for what gets shell-quoted into the container command line.
_DESCRIPTOR_RE = re.compile(
    r"^(?P<src>[0-9A-Za-z+\-\.]+)_(?P<isot>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})$"
)

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_ACTIVE_JOB_ID: Optional[str] = None


# ---------------------------------------------------------------------------
# Descriptor discovery
# ---------------------------------------------------------------------------


# --- descriptor discovery (cached, non-blocking) -------------------------
#
# GENERATED_DIR lives on /dataz and can hold ~100k weight files; listing
# it can take tens of seconds under I/O load. The /sefds page must never
# block on it, so descriptor discovery is a single cached os.scandir()
# refreshed in the background (single-flight). Requests always return the
# current cache immediately — empty on a cold start, filled once the
# first background scan completes. The prior per-source glob() (a full
# dir scan PER source, on the request thread) hung the page (2026-07-23).
# GENERATED_DIR (~100k files on /dataz) currently takes 30-60 s+ to scan;
# descriptors only change when the cal service writes new weights (~daily),
# so refresh sparingly to keep the background I/O off that degraded dir.
_DESC_TTL_S = 900.0
_desc_cache_lock = threading.Lock()
_desc_cache: Dict[str, Any] = {"ts": 0.0, "by_source": {}}
_desc_refresh_lock = threading.Lock()
_desc_refreshing = False


def _scan_all_descriptors() -> Dict[str, Dict[str, Any]]:
    """One os.scandir() of GENERATED_DIR → newest descriptor per source.

    No per-file stat (age comes from the ISOT in the filename), so the
    cost is a single directory read regardless of source count.
    """
    import datetime
    by_source: Dict[str, Dict[str, Any]] = {}
    with os.scandir(GENERATED_DIR) as it:
        for entry in it:
            name = entry.name
            if not (name.startswith("beamformer_weights_")
                    and name.endswith(".yaml")):
                continue
            descriptor = name[len("beamformer_weights_"):-len(".yaml")]
            m = _DESCRIPTOR_RE.match(descriptor)
            if m is None:
                continue
            src, isot = m.group("src"), m.group("isot")
            cur = by_source.get(src)
            if cur is None or isot > cur["isot"]:
                by_source[src] = {
                    "descriptor": descriptor,
                    "yaml_path": os.path.join(GENERATED_DIR, name),
                    "isot": isot,
                }
    now = time.time()
    for info in by_source.values():
        try:
            dt = datetime.datetime.strptime(
                info["isot"], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            info["age_hours"] = round((now - dt.timestamp()) / 3600.0, 1)
        except ValueError:
            info["age_hours"] = None
    return by_source


def _refresh_descriptors_async() -> None:
    """Kick a single-flight background scan; never blocks the caller."""
    global _desc_refreshing
    with _desc_refresh_lock:
        if _desc_refreshing:
            return
        _desc_refreshing = True

    def _work() -> None:
        global _desc_refreshing
        try:
            data = _scan_all_descriptors()
            with _desc_cache_lock:
                _desc_cache["by_source"] = data
                _desc_cache["ts"] = time.time()
        except Exception:                                  # noqa: BLE001
            LOG.exception("bfweights descriptor scan failed")
        finally:
            with _desc_refresh_lock:
                _desc_refreshing = False

    threading.Thread(
        target=_work, name="bfweights-desc-scan", daemon=True
    ).start()


def _cached_by_source() -> Dict[str, Dict[str, Any]]:
    """Current descriptor cache; triggers a background refresh if stale.
    Returns immediately with whatever is cached (possibly empty)."""
    with _desc_cache_lock:
        age = time.time() - _desc_cache["ts"]
        data = dict(_desc_cache["by_source"])
    if age > _DESC_TTL_S:
        _refresh_descriptors_async()
    return data


def latest_descriptor(source: str) -> Optional[Dict[str, Any]]:
    """Newest ``<SRC>_<ISOT>`` descriptor for one calibrator, or None.

    Served from the cached scan (non-blocking); ``None`` until the first
    background scan of GENERATED_DIR completes.
    """
    return _cached_by_source().get(source)


def latest_descriptors(
    sources: List[str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """:func:`latest_descriptor` for each source, keyed by source."""
    data = _cached_by_source()
    return {s: data.get(s) for s in sources}


# ---------------------------------------------------------------------------
# Container run
# ---------------------------------------------------------------------------


def _build_cmd(descriptor: str, *, dry_run: bool) -> List[str]:
    """The full h23-side argv for one update run.

    Runs directly on h23 in the casa38 env. This was previously wrapped in
    ``lxc exec calibration23 -- sudo -u ubuntu ...``; that container no longer
    exists, so the wrapper is removed rather than retargeted.
    """
    inner = (
        f"cd {UPDATE_WORKDIR} && "
        f"exec {UPDATE_PYTHON} -u {UPDATE_SCRIPT} "
        f"{shlex.quote(descriptor)}"
    )
    if dry_run:
        inner = (
            f"cd {UPDATE_WORKDIR} && ls -l {UPDATE_SCRIPT} && "
            f"{UPDATE_PYTHON} -c 'print(\"casa38 import check\")' && "
            f"echo 'DRY-RUN: would exec {UPDATE_PYTHON} -u "
            f"{UPDATE_SCRIPT} {descriptor}'"
        )
    return ["bash", "-c", inner]


def _audit(store: Any, *, user: str, note: str, val: Dict[str, Any]) -> None:
    """One audit row; failures logged, never raised (fleet_update.py
    convention — an audit hiccup must not block a cal update)."""
    if store is None:
        return
    try:
        from control_store import audit_log
        audit_log(
            store,
            namespace="cal.update_bfweights",
            cn_target="calibration23",
            cmd="update_bfweights",
            val=val,
            user=user,
            note=note,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("update_bfweights: audit_log failed (continuing): %s", exc)


def _worker(job_id: str, descriptor: str, *, dry_run: bool,
            user: str, store: Any) -> None:
    global _ACTIVE_JOB_ID
    cmd = _build_cmd(descriptor, dry_run=dry_run)
    _audit(store, user=user, note="started",
           val={"descriptor": descriptor, "dry_run": dry_run,
                "job_id": job_id})
    ok = False
    output_tail: List[str] = []
    error: Optional[str] = None
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        output_tail = combined.splitlines()[-60:]
        ok = proc.returncode == 0
        if not ok:
            error = f"exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {RUN_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update({
                "done": True,
                "ok": ok,
                "error": error,
                "finished_unix": int(time.time()),
                "output_tail": output_tail,
            })
        _ACTIVE_JOB_ID = None
    _audit(store, user=user,
           note="finished ok" if ok else f"FAILED: {error}",
           val={"descriptor": descriptor, "dry_run": dry_run,
                "job_id": job_id, "ok": ok, "error": error})
    LOG.info("update_bfweights job %s (%s) done ok=%s err=%s",
             job_id, descriptor, ok, error)


def start_update(descriptor: str, *, dry_run: bool, user: str,
                 store: Any = None) -> Dict[str, Any]:
    """Validate + launch one update job. Raises ValueError on a bad
    descriptor, RuntimeError if another update is already running."""
    global _ACTIVE_JOB_ID
    if _DESCRIPTOR_RE.match(descriptor) is None:
        raise ValueError(
            f"descriptor {descriptor!r} does not look like "
            "<SRC>_<yyyy-mm-ddThh:mm:ss>"
        )
    yaml_path = os.path.join(
        GENERATED_DIR, f"beamformer_weights_{descriptor}.yaml"
    )
    if not os.path.isfile(yaml_path):
        raise ValueError(f"no such solution file: {yaml_path}")
    with _JOBS_LOCK:
        if _ACTIVE_JOB_ID is not None:
            active = _JOBS.get(_ACTIVE_JOB_ID, {})
            raise RuntimeError(
                "an update_bfweights run is already in progress "
                f"(job {_ACTIVE_JOB_ID}, descriptor "
                f"{active.get('descriptor')!r}) — wait for it to finish"
            )
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "job_id": job_id,
            "descriptor": descriptor,
            "dry_run": dry_run,
            "user": user,
            "started_unix": int(time.time()),
            "done": False,
            "ok": None,
            "error": None,
            "output_tail": [],
        }
        _ACTIVE_JOB_ID = job_id
    th = threading.Thread(
        target=_worker,
        args=(job_id, descriptor),
        kwargs={"dry_run": dry_run, "user": user, "store": store},
        name=f"update_bfweights_{job_id}",
        daemon=True,
    )
    th.start()
    return {"job_id": job_id, "descriptor": descriptor, "dry_run": dry_run}


def job_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None
