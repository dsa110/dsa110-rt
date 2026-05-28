"""dsa_monitor Control tab: parallel ssh fan-out to update the
per-node ``/home/ubuntu/proj/dsa110-rt`` git checkout on every corr +
search node.

Operator workflow
-----------------

1. Operator visits the Control tab and clicks "Dry run (preview)".
2. The dashboard fans out (``ThreadPoolExecutor``, ``max_workers=8``)
   to every host in :data:`DEFAULT_FLEET_HOSTS`. Per host it captures
   pre-SHA + current branch + uncommitted local changes
   (``git status --porcelain``), runs ``git fetch origin <branch>``,
   and returns ``post_sha == pre_sha`` because we did NOT pull.
3. Operator inspects the per-host SHA table, decides to apply.
4. Operator clicks "Apply update" with ``confirm=update_dsart``. Same
   fan-out, but this time the per-host worker runs
   ``git pull --ff-only origin <branch>`` (or
   ``git reset --hard origin/<branch>`` when ``force=true``) and
   reports the new SHA.

Safety
------

* The default abort rule is **dirty worktree** = abort that host.
  Per-host ``git status --porcelain`` is captured BEFORE any write
  operation; if it is non-empty AND ``force`` is false the host
  short-circuits with ``error="dirty_worktree"``. This is the
  deploy-skew bug (Bug 0 from M7.4_PHASE6_E2E_REPORT.md) made
  visible — we refuse to silently lose ad-hoc per-node changes.
* ``force=true`` overrides dirty-worktree and uses ``git reset
  --hard origin/<branch>``. Use this knowingly.
* ``dry_run=true`` (default) NEVER calls ``git pull`` or
  ``git reset``. Step (1) + ``git fetch`` only. Mocked subprocess
  tests pin this behaviour.

Audit
-----

One audit row is appended at ``/mon/audit/control/<iso_ts>`` per
fan-out with ``cmd="update_dsart"`` and ``val=<summary>``. The summary
carries ``{n_ok, n_failed, n_dirty, n_changed, dry_run, force,
branch}`` so a downstream etcd reader can reconstruct what happened.

The module is stdlib-only (subprocess + concurrent.futures + logging)
so it can ship in any dashboard env without new dependencies.
"""

from __future__ import annotations

import concurrent.futures
import logging
import subprocess
import time
from typing import Any, Iterable

LOG = logging.getLogger("dsa_monitor.fleet_update")


# ---------------------------------------------------------------------------
# Fleet topology — short host names that resolve via /etc/hosts on h23.
# Kept here (not in control_store) because control_store.py is owned by
# the Phase 8 author for the sibling "fleet services" subagent.
# ---------------------------------------------------------------------------

#: Corr hosts (16). Each runs the corr_rt + corr_fast services and
#: keeps a checkout at ``/home/ubuntu/proj/dsa110-rt``.
DEFAULT_CORR_HOSTS: tuple[str, ...] = (
    "n03.pro.pvt", "n04.pro.pvt", "n05.pro.pvt", "n06.pro.pvt",
    "n07.pro.pvt", "n08.pro.pvt", "n10.pro.pvt", "n11.pro.pvt",
    "n12.pro.pvt", "n14.pro.pvt", "n15.pro.pvt", "n16.pro.pvt",
    "n18.pro.pvt", "n19.pro.pvt", "n21.pro.pvt", "n22.pro.pvt",
)

#: Search hosts (4). Each runs search_rt and keeps the same checkout.
DEFAULT_SEARCH_HOSTS: tuple[str, ...] = (
    "n01.pro.pvt", "n02.pro.pvt", "n09.pro.pvt", "n13.pro.pvt",
)

#: All hosts the "update dsart" button targets. Corr first then search.
DEFAULT_FLEET_HOSTS: tuple[str, ...] = (
    *DEFAULT_CORR_HOSTS, *DEFAULT_SEARCH_HOSTS,
)

#: Per-node checkout path. Pinned because the dashboard does not know
#: about per-host ``ops`` env vars; if a node ever moves the checkout
#: somewhere else we will see ``error="ssh_failed: ... No such file ..."``
#: in the per-host report.
REMOTE_CHECKOUT_PATH: str = "/home/ubuntu/proj/dsa110-rt"

#: Per-host ThreadPoolExecutor concurrency. 8 is a comfortable number
#: for 20 hosts on a single h23 — each ssh fan-out costs <1 s of CPU
#: locally and the bottleneck is the remote git operation.
DEFAULT_MAX_WORKERS: int = 8

#: Per-ssh-call timeout (seconds). One ``git fetch`` over the local
#: network should finish well under 5 s, but we leave headroom for the
#: occasional slow proxy. The whole fan-out is bounded by
#: 4 × host_timeout_s / max_workers × n_hosts in the worst case.
DEFAULT_HOST_TIMEOUT_S: float = 30.0

#: Canonical ssh args. ``-n`` so stdin is closed and ssh cannot block
#: on interactive prompts; ``BatchMode=yes`` defeats passphrase
#: prompts; ``StrictHostKeyChecking=no`` because the fleet uses a
#: shared known_hosts that may legitimately change after re-image.
_SSH_BASE_ARGS: tuple[str, ...] = (
    "ssh",
    "-o", "ConnectTimeout=5",
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-n",
)


# ---------------------------------------------------------------------------
# Low-level ssh wrapper
# ---------------------------------------------------------------------------


def _ssh_run(
    host: str,
    remote_cmd: str,
    *,
    timeout_s: float = DEFAULT_HOST_TIMEOUT_S,
) -> tuple[int, str, str]:
    """Run ``remote_cmd`` on ``host`` via ssh. Returns
    ``(returncode, stdout, stderr)``. Raises
    :class:`subprocess.TimeoutExpired` on timeout so the per-host
    worker can convert it into a clean ``ssh_timeout`` error string.
    """
    args = [*_SSH_BASE_ARGS, host, remote_cmd]
    cp = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return cp.returncode, cp.stdout, cp.stderr


def _parse_dirty_files(porcelain_lines: Iterable[str]) -> list[str]:
    """Parse ``git status --porcelain`` output lines into a list of
    dirty file paths.

    Porcelain v1 format is ``XY<space>filename`` where XY are two
    status characters. ``XY`` may be e.g. ``" M"`` (worktree mod),
    ``"M "`` (staged mod), ``"??"`` (untracked), ``"A "`` (added).
    We strip the first 3 chars to recover the path; lines shorter
    than 3 chars or whitespace-only are dropped.
    """
    out: list[str] = []
    for raw in porcelain_lines:
        if not raw or not raw.strip():
            continue
        if len(raw) >= 3:
            out.append(raw[3:].strip())
        else:
            out.append(raw.strip())
    return out


# ---------------------------------------------------------------------------
# Per-host update worker
# ---------------------------------------------------------------------------


def _empty_host_result(host: str, dry_run: bool, force: bool) -> dict[str, Any]:
    """Skeleton dict used as the per-host result, pre-populated with
    None values so the JSON shape is stable across every code path.
    """
    return {
        "ok": False,
        "host": host,
        "branch": None,
        "pre_sha": None,
        "post_sha": None,
        "dirty_files": [],
        "changed": False,
        "error": None,
        "dry_run": bool(dry_run),
        "force": bool(force),
    }


def _per_host_update(
    host: str,
    *,
    dry_run: bool,
    force: bool,
    branch: str | None,
    timeout_s: float = DEFAULT_HOST_TIMEOUT_S,
) -> dict[str, Any]:
    """Update one host's checkout. Returns the per-host result dict.

    Steps:
      1. ssh: get pre-SHA + current branch + porcelain in one call
         (3 git commands joined by ``&&`` so we save 2× ssh setup
         cost per host).
      2. If dirty AND not ``force``: short-circuit with
         ``error="dirty_worktree"``.
      3. ssh: ``git fetch origin <branch>``.
      4. If ``dry_run``: stop. ``post_sha=pre_sha``, ``changed=False``.
      5. ssh: ``git pull --ff-only origin <branch>`` (or
         ``git reset --hard origin/<branch>`` when ``force``).
      6. ssh: ``git rev-parse HEAD`` for the new post-SHA.

    No exception escapes this function — everything turns into an
    ``error`` string on the returned dict so the fan-out can continue.
    """
    res = _empty_host_result(host, dry_run, force)

    # Step 1: pre-SHA + branch + porcelain in one ssh round trip.
    cmd1 = (
        f"cd {REMOTE_CHECKOUT_PATH} && "
        f"git rev-parse HEAD && "
        f"git rev-parse --abbrev-ref HEAD && "
        f"git status --porcelain"
    )
    try:
        rc, out, err = _ssh_run(host, cmd1, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        res["error"] = "ssh_timeout"
        return res
    except Exception as exc:                                       # noqa: BLE001
        res["error"] = f"ssh_exception: {exc!r}"
        return res
    if rc != 0:
        res["error"] = f"ssh_failed: rc={rc} stderr={err.strip()[:200]!r}"
        return res

    lines = out.splitlines()
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        res["error"] = f"unexpected_output: {out!r}"
        return res
    pre_sha = lines[0].strip()
    cur_branch = lines[1].strip()
    porcelain_lines = lines[2:]
    dirty_files = _parse_dirty_files(porcelain_lines)

    target_branch = (branch or cur_branch).strip()
    res["pre_sha"] = pre_sha
    res["branch"] = target_branch
    res["dirty_files"] = dirty_files

    # Step 2: dirty-worktree gate.
    if dirty_files and not force:
        res["error"] = "dirty_worktree"
        return res

    # Step 3: fetch (always — dry_run still fetches so the operator
    # can see what WOULD land).
    fetch_cmd = (
        f"cd {REMOTE_CHECKOUT_PATH} && git fetch origin {target_branch}"
    )
    try:
        rc, out, err = _ssh_run(host, fetch_cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        res["error"] = "ssh_timeout"
        return res
    except Exception as exc:                                       # noqa: BLE001
        res["error"] = f"ssh_exception: {exc!r}"
        return res
    if rc != 0:
        res["error"] = f"git_fetch_failed: rc={rc} stderr={err.strip()[:200]!r}"
        return res

    # Step 4: dry-run path stops here.
    if dry_run:
        res["post_sha"] = pre_sha
        res["changed"] = False
        res["ok"] = True
        return res

    # Step 5: pull or reset.
    if force:
        update_cmd = (
            f"cd {REMOTE_CHECKOUT_PATH} && "
            f"git reset --hard origin/{target_branch}"
        )
    else:
        update_cmd = (
            f"cd {REMOTE_CHECKOUT_PATH} && "
            f"git pull --ff-only origin {target_branch}"
        )
    try:
        rc, out, err = _ssh_run(host, update_cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        res["error"] = "ssh_timeout"
        return res
    except Exception as exc:                                       # noqa: BLE001
        res["error"] = f"ssh_exception: {exc!r}"
        return res
    if rc != 0:
        res["error"] = (
            f"git_update_failed: rc={rc} stderr={err.strip()[:200]!r}"
        )
        return res

    # Step 6: post-SHA.
    post_cmd = f"cd {REMOTE_CHECKOUT_PATH} && git rev-parse HEAD"
    try:
        rc, out, err = _ssh_run(host, post_cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        res["error"] = "ssh_timeout"
        return res
    except Exception as exc:                                       # noqa: BLE001
        res["error"] = f"ssh_exception: {exc!r}"
        return res
    if rc != 0:
        res["error"] = (
            f"ssh_failed_post: rc={rc} stderr={err.strip()[:200]!r}"
        )
        return res

    post_sha_lines = out.strip().splitlines()
    if not post_sha_lines or not post_sha_lines[0].strip():
        res["error"] = f"unexpected_post_output: {out!r}"
        return res
    post_sha = post_sha_lines[0].strip()
    res["post_sha"] = post_sha
    res["changed"] = (post_sha != pre_sha)
    res["ok"] = True
    return res


# ---------------------------------------------------------------------------
# Fan-out + audit
# ---------------------------------------------------------------------------


def _summarise(per_host: list[dict[str, Any]], *, dry_run: bool,
               force: bool, branch: str | None) -> dict[str, Any]:
    """Compute the summary row written to the audit log + returned
    to the UI. All counts are derived from the per-host dicts so any
    drift between worker + summary is impossible.
    """
    return {
        "n_hosts": len(per_host),
        "n_ok": sum(1 for r in per_host if r.get("ok")),
        "n_failed": sum(1 for r in per_host if not r.get("ok")),
        "n_dirty": sum(1 for r in per_host if r.get("dirty_files")),
        "n_changed": sum(1 for r in per_host if r.get("changed")),
        "dry_run": bool(dry_run),
        "force": bool(force),
        "branch": branch,
    }


def update_fleet(
    store: Any,
    *,
    dry_run: bool = True,
    force: bool = False,
    hosts: Iterable[str] | None = None,
    branch: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    host_timeout_s: float = DEFAULT_HOST_TIMEOUT_S,
    user: str | None = None,
) -> dict[str, Any]:
    """Fan out the update over every host in ``hosts``.

    ``store`` is a :class:`control_store.ControlStore` (or any object
    with a :meth:`put_dict` surface) used only to append a single audit
    row at ``/mon/audit/control/<iso_ts>``. Audit failures are
    swallowed so they cannot break the operator-visible response.

    ``hosts`` defaults to :data:`DEFAULT_FLEET_HOSTS`. ``branch=None``
    means "use whichever branch each node is currently on" — the
    typical case. Passing ``branch="m7/c1c2-coincidencer"`` forces
    every node to fetch + pull/reset against that branch (still using
    the host's existing checked-out branch as the HEAD ref).

    Returns the JSON shape:

        {
          "ok": <bool: every per-host ok>,
          "hosts": [<per-host dict>, ...] sorted by host name,
          "summary": {
            "n_hosts": int, "n_ok": int, "n_failed": int,
            "n_dirty": int, "n_changed": int,
            "dry_run": bool, "force": bool, "branch": str|None,
          }
        }
    """
    fleet = list(hosts) if hosts is not None else list(DEFAULT_FLEET_HOSTS)
    per_host: list[dict[str, Any]] = []

    if not fleet:
        summary = _summarise([], dry_run=dry_run, force=force, branch=branch)
        _audit(
            store, summary=summary, ok=True, user=user,
            cn_target="", note="no hosts requested",
        )
        return {"ok": True, "hosts": [], "summary": summary}

    workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="fleet-update",
    ) as ex:
        future_to_host = {
            ex.submit(
                _per_host_update,
                h,
                dry_run=dry_run,
                force=force,
                branch=branch,
                timeout_s=host_timeout_s,
            ): h
            for h in fleet
        }
        for fut in concurrent.futures.as_completed(future_to_host):
            host = future_to_host[fut]
            try:
                res = fut.result()
            except Exception as exc:                               # noqa: BLE001
                # _per_host_update should never raise (it captures all
                # exceptions into res["error"]) — but if it ever does
                # we still surface a clean per-host failure rather
                # than aborting the whole fan-out.
                LOG.exception(
                    "fleet_update: per-host worker for %s raised", host,
                )
                res = _empty_host_result(host, dry_run, force)
                res["error"] = f"worker_exception: {exc!r}"
            per_host.append(res)

    per_host.sort(key=lambda r: r.get("host") or "")
    summary = _summarise(
        per_host, dry_run=dry_run, force=force, branch=branch,
    )
    overall_ok = bool(per_host) and all(r.get("ok") for r in per_host)
    _audit(
        store, summary=summary, ok=overall_ok, user=user,
        cn_target=",".join(fleet),
        note=(
            f"dry_run={dry_run} force={force} "
            f"branch={branch or 'per-host'} "
            f"n_ok={summary['n_ok']} n_failed={summary['n_failed']} "
            f"n_dirty={summary['n_dirty']} n_changed={summary['n_changed']}"
        ),
    )
    return {"ok": overall_ok, "hosts": per_host, "summary": summary}


def _audit(
    store: Any,
    *,
    summary: dict[str, Any],
    ok: bool,
    user: str | None,
    cn_target: str,
    note: str,
) -> None:
    """Write one audit row. Lazy-imports ``control_store.audit_log``
    so this module is independently testable (the unit tests can
    pass a stub ``store`` and skip the import altogether).
    """
    if store is None:
        return
    try:
        from control_store import audit_log
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("fleet_update: control_store import failed: %s", exc)
        return
    try:
        audit_log(
            store,
            namespace="dsa_monitor.fleet_update",
            cn_target=cn_target,
            cmd="update_dsart",
            val=summary,
            ok=bool(ok),
            note=note,
            user=user,
        )
    except Exception as exc:                                       # noqa: BLE001
        # Audit row failures must not break the verb — the operator
        # still gets the per-host SHA report on the wire.
        LOG.warning("fleet_update: audit_log failed (continuing): %s", exc)


__all__ = [
    "DEFAULT_CORR_HOSTS",
    "DEFAULT_SEARCH_HOSTS",
    "DEFAULT_FLEET_HOSTS",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_HOST_TIMEOUT_S",
    "REMOTE_CHECKOUT_PATH",
    "update_fleet",
    "_per_host_update",
    "_ssh_run",
    "_parse_dirty_files",
    "_summarise",
]
