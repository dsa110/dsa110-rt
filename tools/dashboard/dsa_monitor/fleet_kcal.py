"""dsa_monitor Control tab: parallel ssh fan-out to delete the per-node
K-calibration (beamformer-weights) table on every corr node.

Why this lives on the corr nodes, not h23
------------------------------------------

The fast-corr pipeline applies the K-cal table via the
``--apply-cal`` arg of ``dsart.services.corr_fast_integration`` (see
``configs/dsart_pipeline_rt.yaml`` line 368):

    --apply-cal /home/ubuntu/data/voltages/250924mptq/cals/beamformer_weights_CALSB.dat

``CALSB`` is substituted per-node by ``dsart.services.dsart_rt`` to
``sb<chgroup:02d>`` (``_substitute`` / ``_cn_to_chgroup``), so each
corr node applies exactly one blob, named for its own chgroup. The
blobs live under ``/home/ubuntu/data/voltages/<run>/cals/`` on each
corr node — h23 (where this dashboard runs) does not carry that tree,
so the deletion has to fan out over ssh exactly like
:mod:`fleet_update` does for the git checkout.

Operator workflow
-----------------

1. Operator clicks "Preview" — dry-run fan-out reports, per corr node,
   whether its ``beamformer_weights_sb<NN>.dat`` is present.
2. Operator types the confirm word + clicks "Delete K-cal table" — the
   same fan-out ``rm -rf``'s each node's blob and reports
   ``deleted`` / ``not_found`` / ``error`` per host.

``rm -rf`` (rather than ``rm -f``) so a CASA-style K table staged as a
directory is removed just as cleanly as the legacy ``.dat`` file.

Audit
-----

One audit row is appended at ``/mon/audit/control/<iso_ts>`` per
fan-out with ``cmd="delete_kcal"`` and a ``val`` summary carrying
``{n_hosts, n_deleted, n_not_found, n_failed, dry_run, run}``.

The module is stdlib-only (it reuses :func:`fleet_update._ssh_run`) so
it can ship in the dashboard env without new dependencies.
"""

from __future__ import annotations

import concurrent.futures
import logging
import subprocess
from typing import Any, Iterable

from corr_topology import CORR_NODES, CorrNode
from fleet_update import DEFAULT_HOST_TIMEOUT_S, DEFAULT_MAX_WORKERS, _ssh_run

LOG = logging.getLogger("dsa_monitor.fleet_kcal")


#: Voltage-bundle run id whose cal directory the running fleet applies.
#: Pinned to match the ``--apply-cal`` path in
#: ``configs/dsart_pipeline_rt.yaml``; if the operating run ever moves
#: this constant + that YAML must change together.
KCAL_RUN: str = "250924mptq"

#: Directory the per-subband cal blobs are staged in on each corr node.
KCAL_CALS_DIR: str = f"/home/ubuntu/data/voltages/{KCAL_RUN}/cals"


def kcal_filename_for_chgroup(chgroup: int) -> str:
    """``beamformer_weights_sb<NN>.dat`` — the K-cal blob a corr node
    on ``chgroup`` applies (mirrors the ``CALSB`` → ``sb%02d``
    substitution in ``dsart.services.dsart_rt``).
    """
    if not 0 <= chgroup < 16:
        raise ValueError(f"chgroup={chgroup}, expected 0..15")
    return f"beamformer_weights_sb{chgroup:02d}.dat"


def kcal_path_for_chgroup(chgroup: int) -> str:
    """Absolute path of the K-cal table a corr node on ``chgroup`` applies."""
    return f"{KCAL_CALS_DIR}/{kcal_filename_for_chgroup(chgroup)}"


def _empty_host_result(node: CorrNode, dry_run: bool) -> dict[str, Any]:
    """Skeleton per-host result with a stable JSON shape."""
    return {
        "ok": False,
        "host": node.fqdn,
        "cn_id": node.cn_id,
        "chgroup": node.chgroup,
        "path": kcal_path_for_chgroup(node.chgroup),
        "status": None,
        "deleted": False,
        "error": None,
        "dry_run": bool(dry_run),
    }


def _per_host_delete(
    node: CorrNode,
    *,
    dry_run: bool,
    timeout_s: float = DEFAULT_HOST_TIMEOUT_S,
) -> dict[str, Any]:
    """Delete (or, when ``dry_run``, just probe) one node's K-cal table.

    Returns the per-host result dict. A missing blob is reported as
    ``status="not_found"`` with ``ok=True`` (it is not an error to
    delete something that is already gone). No exception escapes — ssh
    failures turn into ``status="error"`` + an ``error`` string so the
    fan-out can continue.
    """
    res = _empty_host_result(node, dry_run)
    path = res["path"]

    if dry_run:
        remote = f'if [ -e {path} ]; then echo EXISTS; else echo NOT_FOUND; fi'
    else:
        remote = (
            f'if [ -e {path} ]; then '
            f'rm -rf {path} && echo DELETED || echo RM_FAILED; '
            f'else echo NOT_FOUND; fi'
        )

    try:
        rc, out, err = _ssh_run(node.fqdn, remote, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        res["status"] = "error"
        res["error"] = "ssh_timeout"
        return res
    except Exception as exc:                                       # noqa: BLE001
        res["status"] = "error"
        res["error"] = f"ssh_exception: {exc!r}"
        return res
    if rc != 0:
        res["status"] = "error"
        res["error"] = f"ssh_failed: rc={rc} stderr={err.strip()[:200]!r}"
        return res

    token = out.strip().splitlines()[-1].strip() if out.strip() else ""
    if token == "DELETED":
        res["status"] = "deleted"
        res["deleted"] = True
        res["ok"] = True
    elif token == "EXISTS":
        res["status"] = "exists"
        res["ok"] = True
    elif token == "NOT_FOUND":
        res["status"] = "not_found"
        res["ok"] = True
    elif token == "RM_FAILED":
        res["status"] = "error"
        res["error"] = "rm_failed (check node permissions / mount)"
    else:
        res["status"] = "error"
        res["error"] = f"unexpected_output: {out!r}"
    return res


def _summarise(
    per_host: list[dict[str, Any]], *, dry_run: bool,
) -> dict[str, Any]:
    """Counts derived from the per-host dicts (no drift possible)."""
    return {
        "n_hosts": len(per_host),
        "n_ok": sum(1 for r in per_host if r.get("ok")),
        "n_deleted": sum(1 for r in per_host if r.get("status") == "deleted"),
        "n_present": sum(1 for r in per_host if r.get("status") == "exists"),
        "n_not_found": sum(1 for r in per_host if r.get("status") == "not_found"),
        "n_failed": sum(1 for r in per_host if not r.get("ok")),
        "dry_run": bool(dry_run),
        "run": KCAL_RUN,
    }


def delete_kcal_fleet(
    store: Any,
    *,
    dry_run: bool = True,
    nodes: Iterable[CorrNode] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    host_timeout_s: float = DEFAULT_HOST_TIMEOUT_S,
    user: str | None = None,
) -> dict[str, Any]:
    """Fan out the K-cal deletion across every corr node.

    ``store`` is a :class:`control_store.ControlStore` (or any object
    with a ``put_dict`` surface) used only to append one audit row.
    Audit failures are swallowed so they cannot break the
    operator-visible response.

    ``nodes`` defaults to :data:`corr_topology.CORR_NODES` (all 16 corr
    nodes); the tests pass a subset.

    Returns the JSON shape:

        {
          "ok": <bool: every per-host ok>,
          "hosts": [<per-host dict>, ...] sorted by host name,
          "summary": {
            "n_hosts", "n_ok", "n_deleted", "n_present",
            "n_not_found", "n_failed", "dry_run", "run",
          }
        }
    """
    fleet = list(nodes) if nodes is not None else list(CORR_NODES)
    per_host: list[dict[str, Any]] = []

    if not fleet:
        summary = _summarise([], dry_run=dry_run)
        _audit(store, summary=summary, ok=True, user=user,
               cn_target="", note="no corr nodes")
        return {"ok": True, "hosts": [], "summary": summary}

    workers = max(1, int(max_workers))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="fleet-kcal",
    ) as ex:
        future_to_node = {
            ex.submit(
                _per_host_delete, n,
                dry_run=dry_run, timeout_s=host_timeout_s,
            ): n
            for n in fleet
        }
        for fut in concurrent.futures.as_completed(future_to_node):
            node = future_to_node[fut]
            try:
                res = fut.result()
            except Exception as exc:                               # noqa: BLE001
                LOG.exception(
                    "fleet_kcal: per-host worker for %s raised", node.fqdn,
                )
                res = _empty_host_result(node, dry_run)
                res["status"] = "error"
                res["error"] = f"worker_exception: {exc!r}"
            per_host.append(res)

    per_host.sort(key=lambda r: r.get("host") or "")
    summary = _summarise(per_host, dry_run=dry_run)
    overall_ok = bool(per_host) and all(r.get("ok") for r in per_host)
    _audit(
        store, summary=summary, ok=overall_ok, user=user,
        cn_target=",".join(n.fqdn for n in fleet),
        note=(
            f"dry_run={dry_run} run={KCAL_RUN} "
            f"n_deleted={summary['n_deleted']} "
            f"n_not_found={summary['n_not_found']} "
            f"n_failed={summary['n_failed']}"
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
    """Write one audit row. Lazy-imports ``control_store.audit_log`` so
    this module is independently testable (the tests can pass a stub
    ``store`` and skip the import altogether).
    """
    if store is None:
        return
    try:
        from control_store import audit_log
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("fleet_kcal: control_store import failed: %s", exc)
        return
    try:
        audit_log(
            store,
            namespace="dsa_monitor.fleet_kcal",
            cn_target=cn_target,
            cmd="delete_kcal",
            val=summary,
            ok=bool(ok),
            note=note,
            user=user,
        )
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("fleet_kcal: audit_log failed (continuing): %s", exc)


__all__ = [
    "KCAL_RUN",
    "KCAL_CALS_DIR",
    "kcal_filename_for_chgroup",
    "kcal_path_for_chgroup",
    "delete_kcal_fleet",
    "_per_host_delete",
    "_summarise",
]
