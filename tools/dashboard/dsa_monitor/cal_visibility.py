"""Pipeline-weights visibility for the SEFDs page (cal-visibility).

The fleet ran on 3-day-stale beamformer weights unnoticed because
nothing published *what a corr node actually loaded* -- fast-corr
(``dsart.services.corr_fast_integration``) reads its ``--apply-cal``
weights blob exactly once at process startup and never reports it
again. This module cross-checks:

* the last DISTRIBUTED solution, published by the calibration23
  ``update_bfweights.py`` script to ``/mon/cal/bfweights``
  (``{"cmd": "update_weights", "val": {"weight_files": [...],
  "source": [...], ...}}`` -- ``weight_files`` entries embed the ISOT,
  e.g. ``beamformer_weights_sb00_2026-07-11T19:17:00.dat``), against

* what each corr node's ``dsart_rt`` orchestrator actually LOADED at
  its last ``start`` verb, published to
  ``/mon/corr_rt/<cn>/cal_file`` (see
  ``dsart.services.dsart_rt.RtOrchestrator._publish_cal_file_mon``).

Older fleet nodes running dsart_rt from before this feature landed
simply never write the ``cal_file`` sub-key -- that's the expected,
graceful "not reported by pipeline (needs restart with new dsart
code)" state, not an error.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from corr_topology import CORR_NODES

CAL_FILE_KEY_TMPL = "/mon/corr_rt/{cn}/cal_file"
BFWEIGHTS_KEY = "/mon/cal/bfweights"

_ISOT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _distributed_isot(bfweights_doc: Optional[Any]) -> Optional[str]:
    """Best-effort ISOT of the last-distributed solution.

    Doesn't assume a fixed schema-version key exists -- just pulls the
    ISOT embedded in the first ``weight_files`` entry, which is the
    one field that's been stable across the legacy x-engine and
    dsart-rt eras.
    """
    if not isinstance(bfweights_doc, dict):
        return None
    val = bfweights_doc.get("val")
    if not isinstance(val, dict):
        return None
    files = val.get("weight_files")
    if isinstance(files, list) and files:
        m = _ISOT_RE.search(str(files[0]))
        if m:
            return m.group(0)
    return None


def _distributed_source(bfweights_doc: Optional[Any]) -> Optional[str]:
    if not isinstance(bfweights_doc, dict):
        return None
    val = bfweights_doc.get("val")
    if not isinstance(val, dict):
        return None
    src = val.get("source")
    if isinstance(src, list) and src:
        return src[0]
    if isinstance(src, str):
        return src
    return None


def build_pipeline_weights_view(etcd_store: Any) -> Dict[str, Any]:
    """One-shot summary dict for the SEFDs page "Pipeline weights" panel.

    ``etcd_store`` needs only a ``.get_dict(key) -> dict | None``
    method (matches app.py's module-level ``etcd_store`` /
    ``_LazyEtcd``). Never raises -- any etcd hiccup for any one key
    collapses to that key's "unreported" state so a partially-down
    etcd doesn't blank the whole SEFDs page.

    Returned shape::

        {
          "distributed_isot": str | None,
          "distributed_source": str | None,
          "n_total": int,            # 16 corr nodes
          "n_reported": int,         # nodes with a cal_file mon key
          "any_reported": bool,
          "consensus_isot": str | None,   # majority mtime_isot among reporting nodes
          "stale": bool,             # consensus_isot < distributed_isot
          "disagreeing": [ {cn_id, host, mtime_isot, ...}, ... ],
          "nodes": [ {cn_id, host, reported, mtime_isot, path, ...}, ... ],
        }
    """
    try:
        bfweights_doc = etcd_store.get_dict(BFWEIGHTS_KEY)
    except Exception:  # noqa: BLE001
        bfweights_doc = None
    distributed_isot = _distributed_isot(bfweights_doc)
    distributed_source = _distributed_source(bfweights_doc)

    nodes: List[Dict[str, Any]] = []
    for cn in CORR_NODES:
        entry: Dict[str, Any] = {"cn_id": cn.cn_id, "host": cn.host}
        try:
            doc = etcd_store.get_dict(CAL_FILE_KEY_TMPL.format(cn=cn.cn_id))
        except Exception as exc:  # noqa: BLE001
            doc = None
            entry["read_error"] = str(exc)
        if not isinstance(doc, dict):
            entry["reported"] = False
        else:
            entry["reported"] = True
            entry["path"] = doc.get("path")
            entry["mtime_isot"] = doc.get("mtime_isot")
            entry["mtime_unix"] = doc.get("mtime_unix")
            entry["sha256_12"] = doc.get("sha256_12")
            entry["stat_error"] = doc.get("stat_error")
            entry["hash_error"] = doc.get("hash_error")
            entry["spawned_at_unix"] = doc.get("spawned_at_unix")
        nodes.append(entry)

    reported_nodes = [n for n in nodes if n.get("reported")]
    n_reported = len(reported_nodes)

    consensus_isot: Optional[str] = None
    disagreeing: List[Dict[str, Any]] = []
    if reported_nodes:
        counts = Counter(n.get("mtime_isot") for n in reported_nodes)
        consensus_isot, _ = counts.most_common(1)[0]
        if len(counts) > 1:
            disagreeing = [
                n for n in reported_nodes
                if n.get("mtime_isot") != consensus_isot
            ]

    # ISOT strings (``YYYY-MM-DDTHH:MM:SS``) sort lexically == chronologically.
    stale = bool(
        consensus_isot is not None
        and distributed_isot is not None
        and consensus_isot < distributed_isot
    )

    return {
        "distributed_isot": distributed_isot,
        "distributed_source": distributed_source,
        "n_total": len(nodes),
        "n_reported": n_reported,
        "any_reported": n_reported > 0,
        "consensus_isot": consensus_isot,
        "stale": stale,
        "disagreeing": disagreeing,
        "nodes": nodes,
    }
