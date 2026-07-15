"""Pipeline-weights visibility for the SEFDs page (cal-visibility).

The fleet ran on 3-day-stale beamformer weights unnoticed because
nothing published *what a corr node actually loaded* -- fast-corr
(``dsart.services.corr_fast_integration``) reads its ``--apply-cal``
weights blob exactly once at process startup and never reports it
again. This module cross-checks:

* the last DISTRIBUTED solution, read from the newest fleet YAML in
  the ``applied/`` archive that ``update_bfweights.py`` writes at
  distribution time (``beamformer_weights_<ISOT>.yaml`` with
  ``source``/``caltime``/``weight_files``; the ``weight_files``
  entries embed the exact ``.dat`` ISOT the corr nodes' rsynced
  ``antennas.out`` carries, e.g.
  ``beamformer_weights_sb00_2026-07-11T19:17:00.dat``), against

* what each corr node's ``dsart_rt`` orchestrator actually LOADED at
  its last ``start`` verb, published to
  ``/mon/corr_rt/<cn>/cal_file`` (see
  ``dsart.services.dsart_rt.RtOrchestrator._publish_cal_file_mon``).

The ``/mon/cal/bfweights`` etcd key is only a FALLBACK for hosts that
can't see the ``applied/`` archive: the legacy auto-calibration stack
republishes that key for every calibrator transit it solves (writing
to ``generated/`` only, distributing nothing), so hours after a real
distribution the key routinely names a newer, never-distributed
solution -- e.g. 2026-07-15, where the fleet ran the 2253+161 night
transit distributed at 14:13 UT while the key claimed the (solar
contaminated) 0521+166 17:41 transit solved at 18:43.

Older fleet nodes running dsart_rt from before this feature landed
simply never write the ``cal_file`` sub-key -- that's the expected,
graceful "not reported by pipeline (needs restart with new dsart
code)" state, not an error.
"""

from __future__ import annotations

import datetime
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from corr_topology import CORR_NODES

CAL_FILE_KEY_TMPL = "/mon/corr_rt/{cn}/cal_file"
BFWEIGHTS_KEY = "/mon/cal/bfweights"

#: h23 view of the fleet-YAML directories (same layout as
#: bfweights_update.py's GENERATED_DIR) -- used only as a fallback to
#: recover ``caltime`` (the calibrator TRANSIT epoch, MJD) for older
#: ``/mon/cal/bfweights`` payloads that don't embed it directly.
GENERATED_DIR = "/dataz/dsa110/operations/beamformer_weights/generated"
APPLIED_DIR = "/dataz/dsa110/operations/beamformer_weights/applied"

#: MJD of the Unix epoch (1970-01-01T00:00:00 UTC).
_MJD_UNIX_EPOCH = 40587

_ISOT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

#: A fleet YAML in ``applied/`` (no calibrator name in the filename --
#: per-source solution YAMLs like ``beamformer_weights_0521+166_*.yaml``
#: deliberately don't match).
_FLEET_YAML_RE = re.compile(
    r"^beamformer_weights_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.yaml$"
)


def _age_hours_from_isot(isot: Optional[str]) -> Optional[float]:
    """Hours since ``isot`` (a ``YYYY-MM-DDTHH:MM:SS`` UTC timestamp).

    Same UTC-naive-parse convention as ``bfweights_update.py``'s
    ``latest_descriptor`` age calc. Never raises -- an unparseable or
    missing timestamp just means no age badge is shown.
    """
    if not isot:
        return None
    try:
        dt = datetime.datetime.strptime(
            isot, "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=datetime.timezone.utc)
        return round((time.time() - dt.timestamp()) / 3600.0, 1)
    except (ValueError, TypeError):
        return None


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


def _mjd_to_isot(mjd: Any) -> Optional[str]:
    """MJD (float) -> ``YYYY-MM-DDTHH:MM:SS`` UTC ISOT, or None."""
    try:
        unix = (float(mjd) - _MJD_UNIX_EPOCH) * 86400.0
        dt = datetime.datetime.utcfromtimestamp(unix)
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _caltime_mjd(doc: Optional[Any]) -> Optional[float]:
    """Pull a ``caltime`` (MJD, possibly a 1-element list) out of a
    bfweights ``val`` dict or a fleet YAML dict -- both use the same
    key name."""
    if not isinstance(doc, dict):
        return None
    ct = doc.get("caltime")
    if isinstance(ct, list) and ct:
        ct = ct[0]
    if isinstance(ct, (int, float)):
        return float(ct)
    return None


def _transit_isot_fallback(distributed_isot: Optional[str]) -> Optional[str]:
    """Best-effort recovery of the transit ISOT from the on-disk fleet
    YAML matching ``distributed_isot``, for bfweights payloads that
    don't carry ``caltime`` directly. Never raises; missing/unreadable
    files or a missing PyYAML just mean no transit line is shown.
    """
    if not distributed_isot:
        return None
    try:
        import yaml  # local import -- optional dependency for this fallback only
    except Exception:  # noqa: BLE001
        return None
    for directory in (APPLIED_DIR, GENERATED_DIR):
        path = os.path.join(
            directory, f"beamformer_weights_{distributed_isot}.yaml"
        )
        try:
            with open(path, "r") as f:
                doc = yaml.safe_load(f)
        except Exception:  # noqa: BLE001
            continue
        mjd = _caltime_mjd(doc)
        if mjd is not None:
            return _mjd_to_isot(mjd)
    return None


def _latest_applied_solution(
    applied_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The last actually-distributed solution, from the ``applied/``
    archive (ground truth: ``update_bfweights.py`` writes one fleet
    YAML there per distribution, and ONLY per distribution).

    Returns ``{"distributed_isot", "source", "caltime_mjd"}`` for the
    newest fleet YAML, or None when the directory is absent/empty
    (dev hosts without /dataz -> caller falls back to etcd).
    ``distributed_isot`` prefers the ISOT embedded in the YAML's first
    ``weight_files`` entry -- that is the mtime the rsynced
    ``antennas.out`` carries on the corr nodes, so the stale
    comparison against the per-node ``cal_file`` keys stays exact (the
    YAML filename itself is written a few seconds later). ``source``
    and ``caltime_mjd`` are None if the YAML is unreadable. Never
    raises.
    """
    if applied_dir is None:
        applied_dir = APPLIED_DIR
    try:
        names = os.listdir(applied_dir)
    except OSError:
        return None
    latest: Optional[str] = None
    for n in names:
        m = _FLEET_YAML_RE.match(n)
        # ISOT strings sort lexically == chronologically.
        if m and (latest is None or m.group(1) > latest):
            latest = m.group(1)
    if latest is None:
        return None
    out: Dict[str, Any] = {
        "distributed_isot": latest, "source": None, "caltime_mjd": None,
    }
    try:
        import yaml  # local import -- same optionality as the fallback above
        path = os.path.join(applied_dir, f"beamformer_weights_{latest}.yaml")
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
    except Exception:  # noqa: BLE001
        return out
    if not isinstance(doc, dict):
        return out
    src = doc.get("source")
    if isinstance(src, list) and src:
        src = src[0]
    if isinstance(src, str):
        out["source"] = src
    out["caltime_mjd"] = _caltime_mjd(doc)
    files = doc.get("weight_files")
    if isinstance(files, list) and files:
        m = _ISOT_RE.search(str(files[0]))
        if m:
            out["distributed_isot"] = m.group(0)
    return out


def _mtime_isot_sec(entry: Dict[str, Any]) -> Optional[str]:
    """Whole-second-precision ``YYYY-MM-DDTHH:MM:SS`` UTC ISOT for a
    per-node ``cal_file`` entry, used for consensus grouping/display.

    rsync stamps each corr node's ``antennas.out`` with microsecond-level
    jitter (e.g. ``...:00.487950`` vs ``...:00.491950``) even when every
    node loaded the identical-vintage weights file, so grouping raw
    ``mtime_isot`` values at full precision spuriously reports
    near-universal disagreement. Truncating to whole seconds absorbs
    that jitter while still flagging genuinely different weight
    generations, which in practice differ by minutes or hours.

    Prefers ``mtime_unix`` (exact, tz-aware via ``datetime.utcfromtimestamp``,
    matching the producer's ``tz=utc`` convention in dsart_rt.py's
    ``_stat_cal_file``); falls back to truncating the ``mtime_isot``
    string to its first 19 characters (``YYYY-MM-DDTHH:MM:SS``) when
    ``mtime_unix`` is missing or unparseable. Never raises.
    """
    mtime_unix = entry.get("mtime_unix")
    if mtime_unix is not None:
        try:
            return datetime.datetime.utcfromtimestamp(
                float(mtime_unix)
            ).strftime("%Y-%m-%dT%H:%M:%S")
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    mtime_isot = entry.get("mtime_isot")
    if isinstance(mtime_isot, str) and len(mtime_isot) >= 19:
        return mtime_isot[:19]
    return None


def _transit_isot(
    bfweights_doc: Optional[Any], distributed_isot: Optional[str]
) -> Optional[str]:
    """ISOT of the calibrator TRANSIT that produced the distributed
    solution (the ``caltime`` MJD field) -- distinct from
    ``distributed_isot``, which is when the fleet YAML was written.
    Prefers the value embedded in the etcd payload; falls back to
    reading it off the matching fleet YAML on disk.
    """
    val = bfweights_doc.get("val") if isinstance(bfweights_doc, dict) else None
    mjd = _caltime_mjd(val) if isinstance(val, dict) else None
    if mjd is not None:
        return _mjd_to_isot(mjd)
    return _transit_isot_fallback(distributed_isot)


def build_pipeline_weights_view(etcd_store: Any) -> Dict[str, Any]:
    """One-shot summary dict for the SEFDs page "Pipeline weights" panel.

    ``etcd_store`` needs only a ``.get_dict(key) -> dict | None``
    method (matches app.py's module-level ``etcd_store`` /
    ``_LazyEtcd``). Never raises -- any etcd hiccup for any one key
    collapses to that key's "unreported" state so a partially-down
    etcd doesn't blank the whole SEFDs page.

    Returned shape::

        {
          "solution_provenance": "applied_yaml" | "etcd" | None,
                                     # where the solution identity came from:
                                     # the applied/ archive (ground truth) or
                                     # the /mon/cal/bfweights fallback (may name
                                     # a never-distributed auto-cal solution)
          "transit_isot": str | None,        # calibrator transit that produced the solution (caltime, UTC)
          "transit_age_hours": float | None,  # hours since transit_isot (UTC) -- PRIMARY staleness clock
          "due_for_update": bool,            # transit_age_hours > 48 (expected cadence ~2 days)
          "distributed_isot": str | None,    # .dat ISOT of the distributed weights (secondary clock)
          "distributed_age_hours": float | None,  # hours since distributed_isot (UTC)
          "distributed_source": str | None,
          "n_total": int,            # 16 corr nodes
          "n_reported": int,         # nodes with a cal_file mon key
          "any_reported": bool,
          "consensus_isot": str | None,   # majority mtime_isot among reporting nodes,
                                          # whole-second precision (see _mtime_isot_sec)
          "consensus_age_hours": float | None,    # hours since consensus_isot (UTC)
          "stale": bool,             # consensus_isot < distributed_isot
          "disagreeing": [ {cn_id, host, mtime_isot, mtime_isot_sec, ...}, ... ],
          "nodes": [ {cn_id, host, reported, mtime_isot, mtime_isot_sec, path, ...}, ... ],
        }

    Node ``mtime_isot`` retains the raw, full-microsecond value reported
    by the pipeline; ``mtime_isot_sec`` is the whole-second-truncated
    value used for consensus grouping, disagreement detection, the
    stale-vs-``distributed_isot`` comparison, and the displayed
    ``consensus_isot`` -- rsync jitter means two nodes with identical
    weights can differ by a few hundred microseconds in ``mtime_isot``,
    which would otherwise show up as spurious disagreement.
    """
    applied = _latest_applied_solution()
    if applied is not None:
        # Ground truth: the applied/ archive gains a fleet YAML only
        # when update_bfweights actually distributed weights.
        solution_provenance: Optional[str] = "applied_yaml"
        distributed_isot = applied["distributed_isot"]
        distributed_source = applied["source"]
        mjd = applied["caltime_mjd"]
        transit_isot = (
            _mjd_to_isot(mjd) if mjd is not None
            else _transit_isot_fallback(distributed_isot)
        )
    else:
        # No applied/ archive visible (dev host): fall back to the
        # /mon/cal/bfweights key -- which the auto-cal stack clobbers
        # per solved transit, so it may name a never-distributed
        # solution (see module docstring).
        try:
            bfweights_doc = etcd_store.get_dict(BFWEIGHTS_KEY)
        except Exception:  # noqa: BLE001
            bfweights_doc = None
        solution_provenance = (
            "etcd" if isinstance(bfweights_doc, dict) else None
        )
        distributed_isot = _distributed_isot(bfweights_doc)
        distributed_source = _distributed_source(bfweights_doc)
        transit_isot = _transit_isot(bfweights_doc, distributed_isot)
    transit_age_hours = _age_hours_from_isot(transit_isot)

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
        if entry.get("reported"):
            entry["mtime_isot_sec"] = _mtime_isot_sec(entry)
        nodes.append(entry)

    reported_nodes = [n for n in nodes if n.get("reported")]
    n_reported = len(reported_nodes)

    consensus_isot: Optional[str] = None
    disagreeing: List[Dict[str, Any]] = []
    if reported_nodes:
        counts = Counter(n.get("mtime_isot_sec") for n in reported_nodes)
        consensus_isot, _ = counts.most_common(1)[0]
        if len(counts) > 1:
            disagreeing = [
                n for n in reported_nodes
                if n.get("mtime_isot_sec") != consensus_isot
            ]

    # ISOT strings (``YYYY-MM-DDTHH:MM:SS``) sort lexically == chronologically.
    stale = bool(
        consensus_isot is not None
        and distributed_isot is not None
        and consensus_isot < distributed_isot
    )

    return {
        "solution_provenance": solution_provenance,
        "transit_isot": transit_isot,
        "transit_age_hours": transit_age_hours,
        "due_for_update": bool(
            transit_age_hours is not None and transit_age_hours > 48
        ),
        "distributed_isot": distributed_isot,
        "distributed_age_hours": _age_hours_from_isot(distributed_isot),
        "distributed_source": distributed_source,
        "n_total": len(nodes),
        "n_reported": n_reported,
        "any_reported": n_reported > 0,
        "consensus_isot": consensus_isot,
        "consensus_age_hours": _age_hours_from_isot(consensus_isot),
        "stale": stale,
        "disagreeing": disagreeing,
        "nodes": nodes,
    }
