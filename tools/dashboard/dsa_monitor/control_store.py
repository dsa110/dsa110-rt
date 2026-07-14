"""dsa_monitor Phase 8: shared etcd write surface for the Control tab.

Wraps :class:`dsautils.dsa_store.DsaStore.put_dict` with the same
``/cmd/{namespace}/{cn}`` payload shape used by
``tools/ops/_m75_phaseB_16x4_launch.sh`` and the canonical
``tools/ops/dsart-rt`` operator CLI. Three layers:

1. :class:`ControlStore` — thin DsaStore wrapper that exposes
   :meth:`put_dict` and :meth:`get_dict`. Used by the dashboard's
   Control tab as the only place that does etcd *writes* (read-only
   `_LazyEtcd` is kept untouched for the antennas tab).
2. Verb senders — :func:`send_verb`, :func:`broadcast_corr`,
   :func:`fanout_search` mirror the ops-script idioms one-for-one so
   the dashboard's behaviour is bit-identical to what the operator
   gets from the shell.
3. :func:`compute_arm_seq` — fleet-wide ARM_SEQ calculation that
   walks the 32 capture mon-keys (``/mon/corr_rt/<cn>/capture/<port>``)
   for ``last_seq_no`` and returns ``max(last_seq_no) + margin``. The
   default margin (30 000 specnums ≈ 2 s) matches the Phase A
   single-cn pattern; bump to 60 000 for the full fleet to absorb
   the extra fan-out + clock skew. Returns ``None`` if no captures
   are reporting.
4. :func:`audit_log` — appends a one-shot audit row at
   ``/mon/audit/control/<iso-ts>`` per verb. Cheap, visible in any
   etcd tool, and the only proof-of-control we have until a real
   authentication layer goes in (Phase 8 ships read-only-by-network
   per the inventory).

The module is intentionally tiny and stateless so it can be imported
by both the Flask app and the unit tests without dragging in any
HTTP / Jinja machinery.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

LOG = logging.getLogger("dsa_monitor.control_store")


# ---------------------------------------------------------------------------
# Fleet topology — mirrors corr_topology + the four search hosts. Kept
# local so test code can import this module without going through the
# full RFI-poller stack.
# ---------------------------------------------------------------------------

#: Default ARM_SEQ headroom (in specnums). One specnum = 32.768 µs;
#: 30 000 specnums ≈ 0.98 s. Phase A (single cn) uses this; Phase B
#: (16 corr fleet) bumps to 60 000.
DEFAULT_ARM_SEQ_MARGIN: int = 30_000

#: Per-cn UDP capture ports (matches
#: ``configs/dsart_pipeline_rt.yaml`` ``capture.udp_port`` for
#: NUMA-0 / NUMA-1 capture services).
CAPTURE_UDP_PORTS: tuple[int, ...] = (4011, 4012)

#: Corr-node cn_id list (chgroup-ordered; cn 9, 13, 17, 20 skip
#: because those hosts run search_rt or are unallocated).
CORR_CN_IDS: tuple[int, ...] = (
    3, 4, 5, 6, 7, 8,
    10, 11, 12,
    14, 15, 16,
    18, 19,
    21, 22,
)

#: Search-node cn_id list. n09 (cn=9) and n13 (cn=13) host search_rt
#: services in the Phase B fleet; n01 (cn=1) and n02 (cn=2) hold the
#: other two search halves.
SEARCH_CN_IDS: tuple[int, ...] = (1, 2, 9, 13)

#: Canonical broadcast cn for corr_rt. Any orchestrator with a
#: broadcast watch picks up ``/cmd/corr_rt/0`` in addition to its
#: per-cn key — one PUT fans out to the whole fleet without
#: SSH-per-host.
CORR_BROADCAST_CN: int = 0


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlAction:
    """One audit row for the Control tab. Persisted at
    ``/mon/audit/control/<iso_ts>`` so an operator can replay the
    sequence of verbs in any etcd reader.
    """

    iso_ts: str
    user: str
    host: str
    namespace: str
    cn_target: str                       # "0" (broadcast), "1,2,9,13" (fanout), "3" (single)
    cmd: str
    val: Any
    ok: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iso_ts": self.iso_ts,
            "user": self.user,
            "host": self.host,
            "namespace": self.namespace,
            "cn_target": self.cn_target,
            "cmd": self.cmd,
            "val": self.val,
            "ok": bool(self.ok),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Etcd write surface
# ---------------------------------------------------------------------------


class ControlStore:
    """Lazy DsaStore wrapper with the *write* surface for the Control
    tab. Construction is idempotent and import-time-safe; the actual
    etcd handle is only built on first call so the dashboard still
    boots if etcd is briefly down.
    """

    def __init__(self) -> None:
        self._store: Any = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._store is not None:
                return
            from dsautils.dsa_store import DsaStore   # local import → no import-time cost
            self._store = DsaStore()

    def put_dict(self, key: str, payload: dict[str, Any]) -> None:
        self._ensure()
        self._store.put_dict(key, payload)

    def get_dict(self, key: str) -> Any:
        self._ensure()
        return self._store.get_dict(key)


# ---------------------------------------------------------------------------
# Verb senders — keep the payload shape bit-identical to the ops scripts
# ---------------------------------------------------------------------------


def _verb_payload(cmd: str, val: Any) -> dict[str, Any]:
    """Build the ``{"cmd", "val"}`` dict the orchestrator's
    :meth:`_on_command_event` parser expects.

    ``val`` is forwarded as-is (ints for ARM_SEQ / declination,
    ``None`` for verbs that don't carry a value). Any complex types
    must already be JSON-serialisable; we don't try to be smart.
    """
    return {"cmd": str(cmd), "val": val}


def send_verb(
    store: ControlStore,
    *,
    namespace: str,
    cn: int,
    cmd: str,
    val: Any,
) -> str:
    """Send one verb to ``/cmd/{namespace}/{cn}``. Returns the etcd
    key written (caller logs it). Raises whatever :meth:`put_dict`
    raises so the Flask error handler can surface it.
    """
    if namespace not in {"corr_rt", "search_rt"}:
        raise ValueError(
            f"namespace={namespace!r} unexpected (only corr_rt / search_rt)"
        )
    key = f"/cmd/{namespace}/{int(cn)}"
    store.put_dict(key, _verb_payload(cmd, val))
    LOG.info(
        "send_verb: key=%s cmd=%s val=%r", key, cmd, val,
    )
    return key


def broadcast_corr(
    store: ControlStore,
    *,
    cmd: str,
    val: Any,
) -> str:
    """Broadcast one verb to all 16 corr orchestrators via the
    ``/cmd/corr_rt/0`` key. Every running orchestrator's broadcast
    watch picks it up in addition to its per-cn key.
    """
    return send_verb(
        store, namespace="corr_rt", cn=CORR_BROADCAST_CN, cmd=cmd, val=val,
    )


def fanout_search(
    store: ControlStore,
    *,
    cmd: str,
    val: Any,
    cn_ids: Iterable[int] = SEARCH_CN_IDS,
) -> list[str]:
    """Send the same verb to each of the search cn_ids individually.

    There is no broadcast key on the search side (search_rt's
    ``broadcast_key`` is the same as its ``cmd_key`` by config), so
    we just do N puts. Returns the list of etcd keys written, in
    order.
    """
    written: list[str] = []
    for cn in cn_ids:
        written.append(
            send_verb(
                store, namespace="search_rt", cn=int(cn), cmd=cmd, val=val,
            )
        )
    return written


def fanout_corr(
    store: ControlStore,
    *,
    cmd: str,
    val: Any,
    cn_ids: Iterable[int] = CORR_CN_IDS,
) -> list[str]:
    """Send the same verb to each corr cn_id individually (non-broadcast
    path). Useful when an orchestrator missed the broadcast and the
    operator wants per-node re-issue, or for safety on the *stop*
    path where we want every node to clear independently.
    """
    written: list[str] = []
    for cn in cn_ids:
        written.append(
            send_verb(
                store, namespace="corr_rt", cn=int(cn), cmd=cmd, val=val,
            )
        )
    return written


# ---------------------------------------------------------------------------
# ARM_SEQ computation
# ---------------------------------------------------------------------------


def _capture_mon_keys(
    cn_ids: Iterable[int] = CORR_CN_IDS,
    ports: Iterable[int] = CAPTURE_UDP_PORTS,
) -> list[str]:
    """All 32 ``/mon/corr_rt/<cn>/capture/<port>`` keys."""
    return [
        f"/mon/corr_rt/{cn}/capture/{p}" for cn in cn_ids for p in ports
    ]


def compute_arm_seq(
    store: ControlStore,
    *,
    margin: int = DEFAULT_ARM_SEQ_MARGIN,
    cn_ids: Iterable[int] = CORR_CN_IDS,
    ports: Iterable[int] = CAPTURE_UDP_PORTS,
) -> dict[str, Any]:
    """Compute the next ARM_SEQ from fleet-wide capture state.

    Walks 32 capture mon-keys, reads ``last_seq_no`` from each, and
    returns ``max(last_seq_no) + margin`` along with a tiny
    diagnostic that lists which nodes were polled, which ones
    replied, and what the underlying max came from. ``arm_seq`` is
    ``None`` if no capture is reporting (the caller should refuse to
    broadcast).

    Phase A operators use ``margin=30000`` for single-cn; Phase B
    fleet launches use ``margin=60000`` to absorb the wider clock
    skew (per ``_m75_phaseB_16x4_launch.sh`` Stage 5).
    """
    out: dict[str, Any] = {
        "arm_seq": None,
        "margin": int(margin),
        "max_last_seq_no": None,
        "max_source": None,
        "polled": [],
        "answered": [],
        "missing": [],
    }
    keys = _capture_mon_keys(cn_ids=cn_ids, ports=ports)
    out["polled"] = list(keys)
    max_seq: int | None = None
    max_src: str | None = None
    for k in keys:
        try:
            d = store.get_dict(k)
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("compute_arm_seq get_dict(%s) failed: %s", k, exc)
            out["missing"].append(k)
            continue
        if not isinstance(d, dict):
            out["missing"].append(k)
            continue
        seq = d.get("last_seq_no")
        if not isinstance(seq, int):
            out["missing"].append(k)
            continue
        out["answered"].append(k)
        if max_seq is None or seq > max_seq:
            max_seq = seq
            max_src = k
    if max_seq is None:
        return out
    out["max_last_seq_no"] = max_seq
    out["max_source"] = max_src
    out["arm_seq"] = max_seq + int(margin)
    return out


# ---------------------------------------------------------------------------
# System state (the Control-tab banner)
# ---------------------------------------------------------------------------
#
# One rolled-up lifecycle state for the whole fleet, so the operator
# knows -- at a glance -- whether it is safe to arm. The crucial
# distinction is PREPARING vs PREPARED: arming (utc_start) during the
# corr_fast kernel warmup floods fada to 70/70 and drops packets on sky
# (see corr_fast_integration._warmup_pipeline_jit). The orchestrator
# ``state`` flips to "running" *before* warmup finishes, so we rely on
# the per-process ``corr_fast_ready`` record (pid-matched to the live
# routine) for the "warmed, safe to arm" signal.

_STATE_STALE_S: float = 25.0  # heartbeat older than this == not reporting

# Machine-state -> (display label, css class, one-line meaning).
SYSTEM_STATES: dict[str, dict[str, str]] = {
    "offline":   {"label": "OFFLINE",   "meaning": "orchestrators not reachable"},
    "ready":     {"label": "READY",     "meaning": "services up, pipeline stopped"},
    "preparing": {"label": "PREPARING", "meaning": "pipeline warming up — do NOT arm yet"},
    "prepared":  {"label": "PREPARED",  "meaning": "warmed — safe to arm (utc_start)"},
    "observing": {"label": "OBSERVING", "meaning": "armed — capture writing voltages"},
}


def _is_fresh(payload: Any, now_mjd: float, stale_s: float = _STATE_STALE_S) -> bool:
    """True if a /mon heartbeat dict carries a ``time_mjd`` within
    ``stale_s`` of now. A stale or missing stamp == not reporting."""
    if not isinstance(payload, dict):
        return False
    try:
        age_s = (now_mjd - float(payload["time_mjd"])) * 86400.0
    except (KeyError, TypeError, ValueError):
        return False
    return -5.0 <= age_s <= stale_s


def compute_system_state(
    store: ControlStore,
    *,
    corr_cn_ids: Iterable[int] = CORR_CN_IDS,
    search_cn_ids: Iterable[int] = SEARCH_CN_IDS,
    ports: Iterable[int] = CAPTURE_UDP_PORTS,
) -> dict[str, Any]:
    """Roll the fleet up into one lifecycle state for the Control banner.

    Returns ``{state, label, meaning, detail, safe_to_arm, counts}``.
    Precedence: offline > observing > preparing > prepared > ready.
    Best-effort: any etcd error degrades that node to "not reporting"
    rather than raising.
    """
    import time as _time

    now_mjd = _time.time() / 86400.0 + 40587.0
    corr_cn_ids = list(corr_cn_ids)
    search_cn_ids = list(search_cn_ids)

    def _get(key: str) -> Any:
        try:
            return store.get_dict(key)
        except Exception as exc:                                   # noqa: BLE001
            LOG.debug("system_state get_dict(%s) failed: %s", key, exc)
            return None

    # ---- reachability ---------------------------------------------------
    corr_mon = {cn: _get(f"/mon/corr_rt/{cn}") for cn in corr_cn_ids}
    reachable = {
        cn: d for cn, d in corr_mon.items() if _is_fresh(d, now_mjd)
    }
    n_search_up = sum(
        1 for cn in search_cn_ids
        if _is_fresh(_get(f"/mon/search_rt/{cn}"), now_mjd)
    )

    counts: dict[str, Any] = {
        "corr_total": len(corr_cn_ids),
        "corr_up": len(reachable),
        "search_total": len(search_cn_ids),
        "search_up": n_search_up,
    }

    def _result(state: str, detail: str, *, safe: bool = False) -> dict[str, Any]:
        meta = SYSTEM_STATES[state]
        return {
            "state": state,
            "label": meta["label"],
            "meaning": meta["meaning"],
            "detail": detail,
            "safe_to_arm": safe,
            "counts": counts,
        }

    if not reachable and n_search_up == 0:
        return _result(
            "offline",
            "No corr/search orchestrators are reporting to etcd. "
            "Start the orchestrators (or run the fleet launch).",
        )

    # ---- running vs stopped --------------------------------------------
    running = {
        cn: d for cn, d in reachable.items()
        if str(d.get("state")) in ("running", "starting")
    }
    if not running:
        return _result(
            "ready",
            f"{len(reachable)}/{len(corr_cn_ids)} corr orchestrators up, "
            f"pipeline stopped. Send start to spin up the pipeline.",
        )

    # ---- observing? (any capture actively writing voltages) ------------
    n_writing = 0
    for cn in running:
        for p in ports:
            c = _get(f"/mon/corr_rt/{cn}/capture/{p}")
            if isinstance(c, dict) and str(c.get("arm_state", "")).upper() == "WRITING":
                n_writing += 1
    counts["captures_writing"] = n_writing
    if n_writing > 0:
        return _result(
            "observing",
            f"{n_writing} capture(s) writing voltages; "
            f"{len(running)}/{len(corr_cn_ids)} corr nodes running"
            + ("" if len(running) == len(corr_cn_ids)
               else f" ({len(corr_cn_ids) - len(running)} not running)"),
        )

    # ---- preparing vs prepared (corr_fast kernel warmup) ---------------
    warmed: list[int] = []
    warming: list[int] = []
    for cn, d in running.items():
        live_pid = ((d.get("routines") or {}).get("corr_fast") or {}).get("pid")
        rd = _get(f"/mon/corr_rt/{cn}/corr_fast_ready")
        is_warm = bool(
            isinstance(rd, dict)
            and rd.get("ready") is True
            and live_pid is not None
            and rd.get("pid") == live_pid
        )
        (warmed if is_warm else warming).append(cn)
    counts["corr_running"] = len(running)
    counts["corr_warmed"] = len(warmed)

    # ---- SPL fringe-stopper fstable staged? (2026-07-14) ----------------
    # On an fstable-cache miss dsamfs regenerates the table INSIDE
    # run_fringestopping (~30 s, sometimes longer), which happens AFTER
    # the SPL wrapper's heartbeat already publishes ready=True. The
    # wrapper therefore also publishes ``fstable_staged`` (does the
    # staged/regenerated table exist in the working dir yet); hold the
    # banner at PREPARING until every SPL-running node has it. Nodes
    # with SPL gated off (routine not spawned) are skipped. Heartbeats
    # from wrappers predating the field are trusted on freshness+pid
    # alone (fstable_staged missing != not staged).
    spl_pending: list[int] = []
    for cn, d in running.items():
        spl_rt = (d.get("routines") or {}).get("meridian_fringestop_spl")
        if not (isinstance(spl_rt, dict) and spl_rt.get("alive")):
            continue
        rd = _get(f"/mon/corr_rt/{cn}/meridian_spl_ready")
        if not (
            isinstance(rd, dict)
            and _is_fresh(rd, now_mjd)
            and rd.get("pid") == spl_rt.get("pid")
        ):
            spl_pending.append(cn)
            continue
        if rd.get("fstable_staged") is False:
            spl_pending.append(cn)
    counts["spl_pending"] = len(spl_pending)

    if warming or spl_pending:
        bits = []
        if warming:
            bits.append(
                f"{len(warmed)}/{len(running)} corr_fast warmed "
                f"(warming: {sorted(warming)})"
            )
        if spl_pending:
            bits.append(
                f"SPL fstable staging/regen pending on {sorted(spl_pending)}"
            )
        return _result(
            "preparing",
            "; ".join(bits) + " — wait before arming.",
        )
    return _result(
        "prepared",
        f"all {len(running)} running corr_fast warmed — safe to arm "
        "(utc_start)."
        + ("" if len(running) == len(corr_cn_ids)
           else f" Note: {len(corr_cn_ids) - len(running)} corr node(s) "
                "not running."),
        safe=True,
    )


# ---------------------------------------------------------------------------
# M7.4 Phase 6c: corr_fast service-start epoch arm for injection
# ---------------------------------------------------------------------------
#
# ``compute_arm_seq`` above derives the *SNAP-header* epoch (the
# ``last_seq_no`` published by capture_control). That epoch is the
# right one for ``utc_start``, but the corr_fast hot path's
# ``OnlineInjector.apply_block`` consumes the *service-start* epoch
# (``block_n × NPACKETS_PER_BLOCK``). The two differ by ~3 orders of
# magnitude (Bug 1 in M7.4_PHASE6_E2E_REPORT.md). Until M7.2.8 lifts
# corr_fast onto the SNAP-header epoch, we publish the service-start
# epoch from corr_fast via :mod:`dsart.services.corr_fast_mon` and
# consume it here.
# ---------------------------------------------------------------------------

#: Default margin in *blocks* (NPACKETS_PER_BLOCK = 2048 specnums each)
#: ahead of the fleet's current ``block_n`` to schedule the injection.
#:
#: 2026-06-10 root-cause fix (16 → 32 blocks + staleness
#: extrapolation, see below): the corr_fast mon-point is only
#: published every 16 blocks (~2.1 s), and the old margin was ALSO
#: 16 blocks (~2.1 s) — so the *effective* lead time was
#: ``2.1 s − (publish-cycle staleness ∈ [0, 2.1 s]) ≈ 0–2.1 s``:
#: pure luck of the publish phase. When it landed near zero, the
#: command reached the corr nodes after their pipelines had already
#: passed ``apply_at`` and ``OnlineInjector.apply_block`` silently
#: dropped every channel whose dispersed window was in the past —
#: only the band-bottom chgroups (largest dispersion delay, e.g.
#: +280 ms at DM 500, +1.4 s at DM 2500) still contributed.
#: Confirmed in events 260610snoe/mamv: the dumped cubes show a
#: thin constant-width DM-time track whose slope implies an
#: emitting frequency of 1.313 GHz = the band bottom, i.e. a
#: partial-band injection. 32 blocks (~4.3 s) + the staleness
#: extrapolation in :func:`compute_inject_apply_at` makes the
#: effective lead deterministic (~4.3 s regardless of publish
#: phase), still well inside the matcher TTL (60 s) and specnum
#: gate (±4096 search samples ≈ ±4.3 s, measured from apply_at
#: itself so the margin does not eat into it).
DEFAULT_INJECT_MARGIN_BLOCKS: int = 32

#: Hard per-axis reject bound for injection ``l_rad``/``m_rad``: image
#: half-FoV = 256 px × 0.0002181661565 rad/px / 2 (n_grid=256,
#: ``configs/dsart_search_rt.yaml``); beyond this the FFT image aliases
#: the source to the conjugate position (confirmed live 2026-07-11:
#: injected m=+0.04 was detected at m=-0.0159 = 0.04-0.05585). The
#: primary-beam half-power radius is ~0.025 rad, so operators should
#: stay within |l|,|m| <= 0.02 rad in practice.
INJECT_LM_MAX_RAD: float = 0.0279

#: Local copy of ``corr_fast_integration.NPACKETS_PER_BLOCK``. Pinned
#: here so the dashboard doesn't need to import torch via dsart.
#: Drift would be caught by the
#: :func:`tests/test_dsa_monitor_control_store.py::TestComputeInjectApplyAt`
#: assertions.
NPACKETS_PER_BLOCK: int = 2048

#: Wall-clock seconds per specnum (one specnum = one SNAP packet
#: sequence number = 2 native samples = 2 × 32.768 µs). Used to
#: extrapolate stale corr_fast mon-points to the live stream
#: position. One block = 2048 specnums ≈ 134.2 ms (matches the
#: observed production ``last_block_ms`` ≈ 120–137 ms).
_SPECNUM_SECONDS: float = 65.536e-6


def _corr_fast_mon_key(chgroup: int) -> str:
    """Canonical etcd key the corr_fast service-start mon-publisher
    writes to. Duplicated from
    :func:`dsart.services.corr_fast_mon.build_corr_fast_mon_key`
    because the dashboard env does not import dsart.
    """
    return f"/mon/corr_rt/{int(chgroup)}/corr_fast"


def compute_inject_apply_at(
    store: ControlStore,
    *,
    margin_blocks: int = DEFAULT_INJECT_MARGIN_BLOCKS,
    chgroups: Iterable[int] = tuple(range(16)),
    max_age_s: float = 10.0,
) -> dict[str, Any]:
    """Compute ``apply_at_specnum`` in the corr_fast service-start
    epoch.

    Walks ``/mon/corr_rt/<cn>/corr_fast`` for each requested chgroup,
    reads ``block_specnum_start`` + ``ts_mono``, drops entries older
    than ``max_age_s`` (stale publishers from a previous routine), and
    returns ``max(extrapolated block_specnum_start) + margin_blocks *
    NPACKETS_PER_BLOCK``.

    We pick the *max* (not min) so even the latest-running corr_fast
    has caught up to the target block by the time the inject fires.
    The margin then absorbs the per-chgroup clock skew across the
    fleet (typically a handful of blocks at steady state, more during
    startup).

    2026-06-10 staleness extrapolation: the publisher only PUTs every
    16 blocks (~2.1 s), so the raw ``block_specnum_start`` lags the
    live pipeline by 0–2.1 s depending on where we land in the
    publish cycle. We advance each publisher's value by
    ``(now − ts_wall_unix) / SPECNUM_SECONDS`` specnums (the stream
    runs in real time: one specnum = 2 native samples = 65.536 µs)
    before taking the max, which makes the *effective* lead time
    deterministic. Without this, an unlucky publish phase silently
    consumed the entire old 16-block margin and the corr nodes
    received commands whose ``apply_at`` was already in the past —
    the band-top channels were then dropped on the floor (partial-
    band injection, see ``DEFAULT_INJECT_MARGIN_BLOCKS`` docstring).
    The extrapolation diagnostic is returned per-answer in
    ``extrapolated_specnums`` (max over answered publishers).

    Returns a dict matching :func:`compute_arm_seq`'s shape:

    .. code-block:: python

        {
            "apply_at_specnum": int | None,
            "margin_blocks": int,
            "npackets_per_block": int,
            "max_block_specnum_start": int | None,
            "max_source": str | None,
            "max_block_n": int | None,
            "polled": list[str],
            "answered": list[str],
            "missing": list[str],
            "stale": list[str],
        }

    ``apply_at_specnum`` is ``None`` iff no corr_fast publishers are
    answering — the caller (UI / API) should refuse to broadcast.

    Implementation note: the per-publisher ``ts_mono`` field is the
    publisher's local monotonic clock — comparing it to *our* wall
    clock is nonsense across hosts. We compare ``ts_wall_unix`` (UTC
    wall-clock) against the current wall-clock and reject anything
    older than ``max_age_s``. NTP keeps the cluster within ~ms of
    each other so a 10 s threshold is safe.
    """
    out: dict[str, Any] = {
        "apply_at_specnum": None,
        "margin_blocks": int(margin_blocks),
        "npackets_per_block": int(NPACKETS_PER_BLOCK),
        "max_block_specnum_start": None,
        "max_source": None,
        "max_block_n": None,
        "extrapolated_specnums": 0,
        "polled": [],
        "answered": [],
        "missing": [],
        "stale": [],
    }
    cg_list = list(chgroups)
    keys = [_corr_fast_mon_key(cg) for cg in cg_list]
    out["polled"] = list(keys)
    now_wall = time.time()
    max_bss: int | None = None
    max_block_n: int | None = None
    max_src: str | None = None
    max_extrap: int = 0
    for k in keys:
        try:
            d = store.get_dict(k)
        except Exception as exc:                              # noqa: BLE001
            LOG.warning(
                "compute_inject_apply_at get_dict(%s) failed: %s", k, exc,
            )
            out["missing"].append(k)
            continue
        if not isinstance(d, dict):
            out["missing"].append(k)
            continue
        bss = d.get("block_specnum_start")
        if not isinstance(bss, int):
            out["missing"].append(k)
            continue
        extrap = 0
        ts_wall = d.get("ts_wall_unix")
        if isinstance(ts_wall, (int, float)):
            age_s = float(now_wall - float(ts_wall))
            if age_s > max_age_s:
                out["stale"].append(k)
                continue
            # Advance to the publisher's *live* position: the stream
            # runs in real time at one specnum per SPECNUM_SECONDS,
            # and the mon-point is only refreshed every ~16 blocks
            # (~2.1 s). Round up to whole blocks so we never target a
            # block boundary that has already started.
            if age_s > 0.0:
                blocks_elapsed = int(
                    age_s / (_SPECNUM_SECONDS * NPACKETS_PER_BLOCK)
                ) + 1
                extrap = blocks_elapsed * NPACKETS_PER_BLOCK
        out["answered"].append(k)
        live_bss = int(bss) + extrap
        if max_bss is None or live_bss > max_bss:
            max_bss = live_bss
            max_src = k
            max_extrap = extrap
            blk_n = d.get("block_n")
            if isinstance(blk_n, int):
                max_block_n = int(blk_n)
    if max_bss is None:
        return out
    out["max_block_specnum_start"] = max_bss
    out["max_source"] = max_src
    out["max_block_n"] = max_block_n
    out["extrapolated_specnums"] = max_extrap
    out["apply_at_specnum"] = (
        max_bss + int(margin_blocks) * int(NPACKETS_PER_BLOCK)
    )
    return out


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M7.4 Phase 6 runtime injection
#
# The corr-fast service (when launched with ``--inject-watch``) opens
# a DsaStore watch on ``/cmd/dsart/corr/<chgroup>/inject``. The Control
# tab pushes ``{cmd: "inject", val: <InjectionConfig dict>}`` payloads
# to those keys; the watch dispatches into
# :meth:`OnlineInjector.add_pending` on the corr node, queuing the
# injection at the configured ``apply_at_specnum``.
#
# We write one inject payload per chgroup — fan-out is 16 PUTs for a
# fleet-wide injection, 1 PUT for a single-chgroup smoke test. Each PUT
# is independent (no broadcast key) which makes per-chgroup targeting
# trivial (handy for the Phase-6c recovery benchmark where we want to
# probe one node in isolation before going fleet-wide).
# ---------------------------------------------------------------------------


#: Default chgroups to broadcast injections to. Matches every
#: corr-fast service in the Phase B fleet (one service per chgroup).
DEFAULT_INJECT_CHGROUPS: tuple[int, ...] = tuple(range(16))


def _inject_key(chgroup: int) -> str:
    """Canonical per-chgroup runtime-injection etcd key.

    Mirrors :func:`dsart.inject.runtime_watch.build_runtime_inject_key`
    — duplicated here so the dashboard can build the URL without
    importing dsart (the dashboard ships on h23 where the dsart src
    tree may not be on PYTHONPATH).
    """
    return f"/cmd/dsart/corr/{int(chgroup)}/inject"


#: Local copies of the constants in
#: :mod:`dsart.inject.online`. We don't import the module here
#: because it pulls in torch — which is not installed in the h23
#: dashboard env. The numeric defaults below are pinned to the
#: production values; if they ever change in
#: ``src/dsart/inject/online.py`` the
#: :func:`tests/test_dsa_monitor_control_store.py::TestInjectKey`
#: assertions will catch the drift.
_PROFILE_FAMILIES: tuple[str, ...] = ("gaussian", "boxcar")
_MAX_WIDTH_SAMPLES: int = 4096
_INJECT_REQUIRED_KEYS: tuple[str, ...] = (
    "inj_id", "l_rad", "m_rad", "dm_pc_cm3",
    "fluence_jy_ms", "width_samples", "profile", "apply_at_specnum",
)


def _validate_inject_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the shape and values of an injection payload.

    Mirrors :meth:`dsart.inject.online.InjectionConfig.__post_init__`
    exactly, but uses only stdlib so the dashboard env does not
    require torch/numpy. The receive side
    (``RuntimeInjectWatch.handle_inject_payload``) re-validates via
    the real :class:`InjectionConfig` constructor, so the two
    independent checks must agree on every rule. Any drift is
    surfaced by the unit tests in
    ``tests/test_dsa_monitor_control_store.py``.

    Raises :class:`ValueError` on any validation failure (missing
    keys, out-of-range, unknown profile, etc.). On success returns
    a *normalised* dict with int/float coercion applied.
    """
    import math

    if not isinstance(payload, dict):
        raise ValueError(
            f"inject payload must be a dict; got {type(payload).__name__}"
        )
    missing = set(_INJECT_REQUIRED_KEYS) - payload.keys()
    extra = payload.keys() - set(_INJECT_REQUIRED_KEYS)
    if missing:
        raise ValueError(f"inject payload missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"inject payload has unknown keys: {sorted(extra)}")

    inj_id = payload["inj_id"]
    if not isinstance(inj_id, str) or not inj_id:
        raise ValueError(f"inj_id must be a non-empty str; got {inj_id!r}")

    try:
        l_rad = float(payload["l_rad"])
        m_rad = float(payload["m_rad"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"l_rad/m_rad must be floats: {exc}") from exc
    n2 = l_rad * l_rad + m_rad * m_rad
    if n2 >= 1.0:
        raise ValueError(
            f"l_rad={l_rad}, m_rad={m_rad}: l² + m² = {n2:.6f} ≥ 1.0 "
            f"(n would be imaginary)"
        )
    if abs(l_rad) > INJECT_LM_MAX_RAD or abs(m_rad) > INJECT_LM_MAX_RAD:
        raise ValueError(
            f"l_rad/m_rad must be within +/-{INJECT_LM_MAX_RAD} rad "
            f"(imaged half-FoV); beyond that the source aliases to the "
            f"wrong sky position. Recommended |l|,|m| <= 0.02 rad "
            f"(primary-beam half-power ~0.025 rad)."
        )

    try:
        dm = float(payload["dm_pc_cm3"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dm_pc_cm3 must be a float: {exc}") from exc
    if not math.isfinite(dm):
        raise ValueError(f"dm_pc_cm3 must be finite; got {dm}")

    try:
        fluence = float(payload["fluence_jy_ms"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fluence_jy_ms must be a float: {exc}") from exc
    if not math.isfinite(fluence):
        raise ValueError(f"fluence_jy_ms must be finite; got {fluence}")

    width = payload["width_samples"]
    # InjectionConfig.__post_init__ requires width_samples to be an
    # int (not bool, not float). We coerce here so callers can pass
    # numeric form-field strings, then re-check the type.
    try:
        width_int = int(width)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"width_samples must be an int: {exc}") from exc
    if isinstance(width, bool):
        raise ValueError(
            f"width_samples must be int; got bool ({width!r})"
        )
    if not 1 <= width_int <= _MAX_WIDTH_SAMPLES:
        raise ValueError(
            f"width_samples={width_int} outside [1, {_MAX_WIDTH_SAMPLES}]"
        )

    profile = payload["profile"]
    if profile not in _PROFILE_FAMILIES:
        raise ValueError(
            f"profile={profile!r} not in {_PROFILE_FAMILIES}"
        )

    try:
        apply_at = int(payload["apply_at_specnum"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"apply_at_specnum must be an int: {exc}"
        ) from exc

    return {
        "inj_id": str(inj_id),
        "l_rad": float(l_rad),
        "m_rad": float(m_rad),
        "dm_pc_cm3": float(dm),
        "fluence_jy_ms": float(fluence),
        "width_samples": int(width_int),
        "profile": str(profile),
        "apply_at_specnum": int(apply_at),
    }


def send_inject(
    store: ControlStore,
    *,
    inj_id: str,
    l_rad: float,
    m_rad: float,
    dm_pc_cm3: float,
    fluence_jy_ms: float,
    width_samples: int,
    profile: str,
    apply_at_specnum: int,
    chgroups: Iterable[int] = DEFAULT_INJECT_CHGROUPS,
    user: str | None = None,
) -> dict[str, Any]:
    """Push an injection to every requested chgroup.

    Validates the payload via :func:`_validate_inject_payload` before
    any write, then PUTs ``{cmd: "inject", val: <cfg>}`` to each
    chgroup's runtime-inject key. A single audit row summarises the
    fan-out (so the audit table doesn't get spammed by 16 nearly-
    identical rows for a fleet-wide injection).

    Returns a JSON-ready dict for the Flask layer:

        {
          "ok": true,
          "cmd": "inject",
          "val": <validated cfg dict>,
          "chgroups": [0, 1, ..., 15],
          "keys": ["/cmd/dsart/corr/0/inject", ...]
        }

    Raises any validation failure as a :class:`ValueError` so the
    Flask route surfaces a 400; etcd PUT failures (post-validation)
    bubble through ``put_dict`` and are caught by
    :func:`_control_json_or_error` upstream.
    """
    raw_cfg = {
        "inj_id": inj_id,
        "l_rad": float(l_rad),
        "m_rad": float(m_rad),
        "dm_pc_cm3": float(dm_pc_cm3),
        "fluence_jy_ms": float(fluence_jy_ms),
        "width_samples": int(width_samples),
        "profile": str(profile),
        "apply_at_specnum": int(apply_at_specnum),
    }
    cfg = _validate_inject_payload(raw_cfg)

    payload = {"cmd": "inject", "val": cfg}
    cg_list = [int(c) for c in chgroups]
    keys: list[str] = []
    for cg in cg_list:
        key = _inject_key(cg)
        store.put_dict(key, payload)
        keys.append(key)
        LOG.info("send_inject: PUT %s val.inj_id=%s", key, cfg["inj_id"])

    audit_log(
        store,
        namespace="dsart.inject",
        cn_target=",".join(str(c) for c in cg_list),
        cmd="inject",
        val=cfg,
        ok=True,
        note=(
            f"chgroups={len(cg_list)} apply_at_specnum={cfg['apply_at_specnum']}"
        ),
        user=user,
    )
    return {
        "ok": True,
        "cmd": "inject",
        "val": cfg,
        "chgroups": cg_list,
        "keys": keys,
    }


def control_inject_pulse(
    store: ControlStore,
    *,
    inj_id: str,
    l_rad: float,
    m_rad: float,
    dm_pc_cm3: float,
    fluence_jy_ms: float,
    width_samples: int,
    profile: str,
    apply_at_specnum: int | None,
    margin_blocks: int = DEFAULT_INJECT_MARGIN_BLOCKS,
    chgroups: Iterable[int] = DEFAULT_INJECT_CHGROUPS,
    user: str | None = None,
    publish_active_registry: bool = True,
    inject_label_ttl_s: float = 60.0,
    target_snr: float | None = None,
) -> dict[str, Any]:
    """High-level helper for the Control tab.

    Same surface as :func:`send_inject` but ``apply_at_specnum`` may
    be ``None`` — in which case we compute it via
    :func:`compute_inject_apply_at` (corr_fast service-start epoch
    ``max(block_specnum_start) + margin_blocks × NPACKETS_PER_BLOCK``).

    Why not :func:`compute_arm_seq` (which mirrors ``utc_start``)? The
    corr_fast hot path's ``OnlineInjector.apply_block`` consumes the
    *service-start* epoch (``block_n × NPACKETS_PER_BLOCK`` ≈ 10⁷
    after 17 min) while ``last_seq_no`` runs in the *SNAP-header*
    epoch (≈ 10¹⁰ after months of uptime). Until M7.2.8 lifts
    corr_fast onto the SNAP-header epoch, the inject path needs the
    publisher set up by ``dsart.services.corr_fast_mon`` (active
    whenever corr_fast was launched with ``--inject-watch``).

    Returns the dict :func:`send_inject` returns, plus an extra
    ``arm_info`` key describing how ``apply_at_specnum`` was derived
    (handy for the UI to confirm what the auto-pick produced).
    """
    if apply_at_specnum is None:
        info = compute_inject_apply_at(
            store, margin_blocks=margin_blocks, chgroups=chgroups,
        )
        if info["apply_at_specnum"] is None:
            audit_log(
                store, namespace="dsart.inject",
                cn_target=",".join(str(c) for c in chgroups),
                cmd="inject", val=None, ok=False,
                note=(
                    f"refused (auto-arm): no corr_fast publishers "
                    f"answering ({len(info['missing'])} missing, "
                    f"{len(info['answered'])} answered, "
                    f"{len(info['stale'])} stale)"
                ),
                user=user,
            )
            return {
                "ok": False,
                "cmd": "inject",
                "error": (
                    "apply_at_specnum=null requested but no "
                    "/mon/corr_rt/<cn>/corr_fast publishers are "
                    "answering — is the fleet up with --inject-watch "
                    "enabled? Either start the corr_fast routines "
                    "first or pass apply_at_specnum explicitly."
                ),
                "info": info,
            }
        derived_specnum = int(info["apply_at_specnum"])
    else:
        info = None
        derived_specnum = int(apply_at_specnum)

    out = send_inject(
        store,
        inj_id=inj_id,
        l_rad=l_rad,
        m_rad=m_rad,
        dm_pc_cm3=dm_pc_cm3,
        fluence_jy_ms=fluence_jy_ms,
        width_samples=width_samples,
        profile=profile,
        apply_at_specnum=derived_specnum,
        chgroups=chgroups,
        user=user,
    )
    out["arm_info"] = info
    out["auto_arm"] = apply_at_specnum is None

    # M7.4 Phase 6c.A: also publish the active-injection registry row
    # so the C2 coincidencer's InjectionMatcher can pair incoming C1
    # candidates with this injection (labels them in the C1 CSV and
    # publishes /mon/dsart/inject/matches/<inj_id> for the dashboard
    # calibration loop to consume).
    if publish_active_registry and out.get("ok"):
        try:
            import inject_calibration as _ic   # local; avoid import cycle
            _ic.publish_active_inject(
                store,
                inj_id=inj_id,
                dm_pc_cm3=float(dm_pc_cm3),
                l_rad=float(l_rad),
                m_rad=float(m_rad),
                width_samples=int(width_samples),
                fluence_jy_ms=float(fluence_jy_ms),
                apply_at_specnum=int(derived_specnum),
                ttl_s=float(inject_label_ttl_s),
                fired_by=str(user or "anon"),
                target_snr=(
                    float(target_snr) if target_snr is not None else None
                ),
            )
            out["active_inject_key"] = (
                _ic.build_active_inject_key(inj_id)
            )
        except Exception as exc:  # noqa: BLE001
            # Don't fail the verb just because the registry PUT
            # failed — the underlying inject already succeeded.
            LOG.warning(
                "publish_active_inject(%s) failed: %s", inj_id, exc,
            )
            out["active_inject_warning"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# M7.4 Phase 8 v2: fleet service status + restart-all
# ---------------------------------------------------------------------------
#
# Thin wrappers around :mod:`fleet_services` that audit the call to
# ``/mon/audit/control/<iso_ts>`` and translate the result dict into
# the same JSON-ready shape the verb-senders above use. The heavy
# lifting (ssh fanout, lxc exec, ThreadPoolExecutor) lives in
# fleet_services so it can be unit-tested in isolation by mocking
# ``subprocess.run`` / ``subprocess.Popen``.


def fleet_service_status(
    store: "ControlStore",
    *,
    user: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Query the status of every entry in
    :data:`fleet_services.SERVICE_INVENTORY`.

    Returns the dict :func:`fleet_services.query_all_services_status`
    returns, with one additional ``audit_iso_ts`` field for the
    audit-row pointer. The audit row is logged with
    ``cmd="services_status"`` and ``ok=True`` iff no rows came back
    as ``unreachable``/``unknown``.
    """
    import fleet_services as _fleet
    kwargs: dict[str, Any] = {}
    if timeout_s is not None:
        kwargs["timeout"] = float(timeout_s)
    res = _fleet.query_all_services_status(**kwargs)
    ok = res["n_unreachable"] == 0 and res["n_unknown"] == 0
    action = audit_log(
        store, namespace="services",
        cn_target="fleet", cmd="services_status",
        val={
            "n_active": res["n_active"],
            "n_inactive": res["n_inactive"],
            "n_failed": res["n_failed"],
            "n_unreachable": res["n_unreachable"],
            "n_unknown": res["n_unknown"],
        },
        ok=ok,
        note=f"{len(res['rows'])} entries polled",
        user=user,
    )
    res["audit_iso_ts"] = action.iso_ts
    return res


def fleet_restart_all(
    store: "ControlStore",
    *,
    user: str | None = None,
    dry_run: bool = False,
    self_restart_delay_s: float = 2.0,
) -> dict[str, Any]:
    """Run the "Restart everything (except lxd110h20)" sequence.

    Wraps :func:`fleet_services.restart_all_services`. Bridges the
    etcd ``stop`` broadcast (step 1) into the existing
    :func:`control_stop_fleet` helper by passing it in as
    ``stop_broadcast`` (which keeps :mod:`fleet_services` free of
    any dsautils import).

    Always writes one audit row at the end summarising per-host
    success / failure.
    """
    import fleet_services as _fleet

    def _stop_bcast() -> dict[str, Any]:
        return control_stop_fleet(store, user=user)

    summary = _fleet.restart_all_services(
        dry_run=bool(dry_run),
        stop_broadcast=None if dry_run else _stop_bcast,
        self_restart_delay_s=float(self_restart_delay_s),
    )

    # Per-host pass/fail rollup is the most useful audit val — the
    # operator can scan it to see which corr nodes failed without
    # diffing 20 sub-dicts.
    host_summary: dict[str, bool] = {
        host: all(step["ok"] for step in steps.values())
        for host, steps in summary.get("host_results", {}).items()
    }
    action = audit_log(
        store, namespace="services",
        cn_target="fleet",
        cmd="restart_all",
        val=host_summary,
        ok=bool(summary.get("ok")),
        note=(
            f"dry_run={dry_run} elapsed_s={summary.get('elapsed_s')} "
            f"hosts={len(host_summary)}"
        ),
        user=user,
    )
    summary["audit_iso_ts"] = action.iso_ts
    return summary


def _iso_ts_utc() -> str:
    """ISO-8601 UTC stamp with microseconds, used as audit-row key."""
    return _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ",
    )


def audit_log(
    store: ControlStore,
    *,
    namespace: str,
    cn_target: str,
    cmd: str,
    val: Any,
    ok: bool,
    note: str = "",
    user: str | None = None,
    host: str | None = None,
) -> ControlAction:
    """Write a single audit row at ``/mon/audit/control/<iso_ts>``.

    The dashboard records every Control-tab verb attempt (successful
    or not) so a downstream `dsa-store` reader can replay the
    sequence. We *don't* aggregate into a journal-style list because
    the per-row TTL / retention is then up to etcd policy, and
    snapshotting is one ``get_dict_recursive`` call.

    The row carries the local hostname + (optionally) the
    ``DSA_MONITOR_USER`` env value if the dashboard is wrapped by a
    reverse-proxy that injects an authenticated user header — for v1
    this is just a label.
    """
    action = ControlAction(
        iso_ts=_iso_ts_utc(),
        user=str(user or os.environ.get("DSA_MONITOR_USER", "unknown")),
        host=str(host or socket.gethostname()),
        namespace=str(namespace),
        cn_target=str(cn_target),
        cmd=str(cmd),
        val=val,
        ok=bool(ok),
        note=str(note),
    )
    key = f"/mon/audit/control/{action.iso_ts}"
    try:
        store.put_dict(key, action.to_dict())
    except Exception as exc:                                       # noqa: BLE001
        # Audit failure must not break the verb — log + continue.
        LOG.error("audit_log put_dict(%s) failed: %s", key, exc)
    return action


# ---------------------------------------------------------------------------
# High-level orchestration helpers — what the Flask POST handlers call
# ---------------------------------------------------------------------------


def control_start_fleet(
    store: ControlStore,
    *,
    obs_dec_deg: float | None = None,
    user: str | None = None,
) -> dict[str, Any]:
    """Start the corr fleet via broadcast + each search node via
    fanout. Returns a dict the Flask layer can serialise to JSON.

    ``obs_dec_deg`` is the declination passed as the ``start`` verb's
    ``val``. ``None`` means "let the orchestrator resolve from
    ``/mon/array/dec``" (M7.4 ``CUSTOMDEC`` fallback).
    """
    val = float(obs_dec_deg) if obs_dec_deg is not None else None
    corr_key = broadcast_corr(store, cmd="start", val=val)
    search_keys = fanout_search(store, cmd="start", val=val)
    audit_log(
        store, namespace="corr_rt+search_rt",
        cn_target="0+1,2,9,13",
        cmd="start", val=val, ok=True,
        note=f"corr={corr_key} search={search_keys}",
        user=user,
    )
    return {
        "ok": True,
        "cmd": "start",
        "val": val,
        "corr_broadcast_key": corr_key,
        "search_fanout_keys": search_keys,
    }


def control_stop_fleet(
    store: ControlStore,
    *,
    fanout_corr_too: bool = True,
    user: str | None = None,
) -> dict[str, Any]:
    """Stop the corr fleet + every search node. Belt-and-braces:
    broadcast on the corr side AND optionally fanout to every
    individual corr cn (default True) so a node that missed the
    broadcast also gets the stop.
    """
    corr_keys: list[str] = []
    corr_broadcast = broadcast_corr(store, cmd="stop", val=None)
    corr_keys.append(corr_broadcast)
    if fanout_corr_too:
        corr_keys.extend(fanout_corr(store, cmd="stop", val=None))
    search_keys = fanout_search(store, cmd="stop", val=None)
    audit_log(
        store, namespace="corr_rt+search_rt",
        cn_target="0+all+1,2,9,13" if fanout_corr_too else "0+1,2,9,13",
        cmd="stop", val=None, ok=True,
        note=f"corr={len(corr_keys)} search={len(search_keys)}",
        user=user,
    )
    return {
        "ok": True,
        "cmd": "stop",
        "corr_keys": corr_keys,
        "search_fanout_keys": search_keys,
    }


def control_utc_start_now(
    store: ControlStore,
    *,
    margin: int = DEFAULT_ARM_SEQ_MARGIN,
    user: str | None = None,
) -> dict[str, Any]:
    """Compute ARM_SEQ from current capture state + broadcast
    ``utc_start`` to all corr orchestrators.

    Returns a JSON-ready dict with the computed ARM_SEQ, the source
    of the max last_seq_no, the answered/missing capture-keys count,
    and the etcd broadcast key written. Refuses to broadcast (sets
    ``ok=False``) if no captures are answering.
    """
    info = compute_arm_seq(store, margin=margin)
    if info["arm_seq"] is None:
        audit_log(
            store, namespace="corr_rt",
            cn_target="0", cmd="utc_start", val=None, ok=False,
            note=(
                f"refused: no captures answering "
                f"({len(info['missing'])} missing, "
                f"{len(info['answered'])} answered)"
            ),
            user=user,
        )
        return {
            "ok": False,
            "cmd": "utc_start",
            "error": (
                "no captures answering — refusing to broadcast a "
                "blind utc_start. Inspect "
                "/mon/corr_rt/<cn>/capture/<port>.last_seq_no first."
            ),
            "info": info,
        }
    arm_seq = int(info["arm_seq"])
    key = broadcast_corr(store, cmd="utc_start", val=arm_seq)
    audit_log(
        store, namespace="corr_rt",
        cn_target="0", cmd="utc_start", val=arm_seq, ok=True,
        note=(
            f"arm_seq={arm_seq} (= max {info['max_last_seq_no']} + "
            f"{info['margin']} from {info['max_source']!r})"
        ),
        user=user,
    )
    return {
        "ok": True,
        "cmd": "utc_start",
        "val": arm_seq,
        "info": info,
        "corr_broadcast_key": key,
    }


def control_utc_stop_now(
    store: ControlStore,
    *,
    user: str | None = None,
) -> dict[str, Any]:
    """Broadcast ``utc_stop`` with the current ARM_SEQ + 0 margin
    (i.e. "stop as soon as possible"). The orchestrator's
    ``_verb_utc_stop`` interprets a 0 / null ``val`` as "stop
    immediately"; here we still try to compute a sane ARM_SEQ so the
    capture sidecar logs the requested stop point.
    """
    info = compute_arm_seq(store, margin=0)
    val = int(info["arm_seq"]) if info["arm_seq"] is not None else 0
    key = broadcast_corr(store, cmd="utc_stop", val=val)
    audit_log(
        store, namespace="corr_rt",
        cn_target="0", cmd="utc_stop", val=val, ok=True,
        note=(
            f"arm_seq={val} (= max {info['max_last_seq_no']} + 0)"
            if info["arm_seq"] is not None else
            "no captures answering; sent val=0"
        ),
        user=user,
    )
    return {
        "ok": True,
        "cmd": "utc_stop",
        "val": val,
        "info": info,
        "corr_broadcast_key": key,
    }


# ---------------------------------------------------------------------------
# Audit log read-side (Control tab's "recent actions" panel)
# ---------------------------------------------------------------------------


def list_recent_audit(
    store: ControlStore,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent audit rows.

    Uses the raw etcd3 client exposed by ``DsaStore.get_etcd()``
    (the vendored ``dsautils.dsa_store.DsaStore`` does not expose
    a recursive get of its own). The etcd values are JSON dicts so
    we ``json.loads`` each one and normalise to the
    :class:`ControlAction` shape the template expects. Failures at
    any step (etcd timeout, bad JSON) are swallowed and logged so
    the audit panel never breaks the dashboard.
    """
    store._ensure()
    es = store._store
    rows: list[dict[str, Any]] = []
    try:
        client = es.get_etcd()
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("list_recent_audit: DsaStore.get_etcd() failed: %s", exc)
        return rows
    try:
        pairs = list(client.get_prefix("/mon/audit/control/"))
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("list_recent_audit: get_prefix failed: %s", exc)
        return rows
    for value, _meta in pairs:
        try:
            if isinstance(value, (bytes, bytearray)):
                payload = json.loads(value.decode("utf-8"))
            elif isinstance(value, str):
                payload = json.loads(value)
            elif isinstance(value, dict):
                payload = value
            else:
                continue
        except Exception as exc:                                   # noqa: BLE001
            LOG.warning("list_recent_audit: bad row JSON: %s", exc)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    rows.sort(key=lambda r: r.get("iso_ts", ""), reverse=True)
    return rows[: int(limit)]


# ---------------------------------------------------------------------------
# M7.4 Phase 8 v3: search / C2 quick-recovery helpers
#
# These address two recurring "things are stuck" patterns that came out
# of the Phase 6c soaks:
#
# * ``bounce_search`` cycles ``stop`` → ``start`` on one or more search
#   cn_ids. Repairs the common "c1_emit queue full, total_dropped
#   climbs but no reconnect" wedge where the TCP socket to C2 looks
#   connected but the send thread is blocked.
#
# * ``restart_c2_service_local`` is a local ``systemctl --user restart
#   dsart_c2.service`` on h23 — the dashboard host. Used when C2 is
#   itself wedged (rare but happens after some etcd hiccups).
#
# Both write a single audit row on success/failure. They never raise:
# the caller gets ``{ok: bool, ...}`` and the Flask layer surfaces it
# as JSON. Tests mock ``time.sleep`` + ``subprocess.run`` so the unit
# tests stay sub-second.
# ---------------------------------------------------------------------------


#: Pause between the ``stop`` and ``start`` fanouts when bouncing
#: search. 2 s is enough for the search orchestrator to tear down its
#: child processes and for the c1_emit TCP socket to enter TIME_WAIT
#: cleanly. The matching ``test_bounce_search_waits_between_stop_and_start``
#: assertion pins this default.
DEFAULT_BOUNCE_SLEEP_S: float = 2.0

#: Local dsart_c2 service unit. Used by
#: :func:`restart_c2_service_local` and
#: :func:`c2_journal_tail_local`. Kept here (rather than imported from
#: services_inventory) so the unit tests can patch it cheaply.
C2_SERVICE_UNIT: str = "dsart_c2.service"

#: The h23 systemd ``--user`` units that make up the dsa110-rt C2/C3
#: stack — the same set ``tools/c2/install.sh`` installs + enables.
#: Restarted in this order by :func:`restart_h23_services_local`.
#: The dashboard's own unit (``dsa_monitor.service``) is deliberately
#: NOT here: restarting it from inside a request would kill the very
#: process serving the restart.
H23_DSART_UNITS: tuple[str, ...] = (
    "dsart_c2.service",
    "dsart_c3.service",
    "hiplot_c1.service",
    "hiplot_c2.service",
)


def bounce_search(
    store: ControlStore,
    *,
    cn_ids: Iterable[int] | None = None,
    sleep_s: float = DEFAULT_BOUNCE_SLEEP_S,
    user: str | None = None,
    sleep_fn: Any = None,
) -> dict[str, Any]:
    """Cycle ``stop`` → wait ``sleep_s`` → ``start`` across the given
    search cn_ids (default: every search cn).

    Returns a JSON-ready dict matching the other control_store verbs:

        {
          "ok": bool,
          "cmd": "bounce_search",
          "cn_ids": [1, 2, 9, 13],
          "stop_keys":  ["/cmd/search_rt/1", ...],
          "start_keys": ["/cmd/search_rt/1", ...],
          "sleep_s": 2.0,
        }

    The start verb is sent with ``val=None`` — the search orchestrator
    falls back to ``/mon/array/dec`` per the M7.4 CUSTOMDEC contract,
    so a bounce does not require the operator to re-supply the
    declination. Audit row carries the list of cn_ids touched.
    """
    cn_list: list[int] = (
        [int(c) for c in cn_ids] if cn_ids is not None
        else list(SEARCH_CN_IDS)
    )
    if not cn_list:
        raise ValueError("bounce_search requires at least one cn_id")
    sleep_s = float(sleep_s)
    if sleep_s < 0.0:
        raise ValueError(f"sleep_s={sleep_s} must be ≥ 0")

    _sleep = sleep_fn if sleep_fn is not None else time.sleep

    stop_keys = fanout_search(store, cmd="stop", val=None, cn_ids=cn_list)
    _sleep(sleep_s)
    start_keys = fanout_search(store, cmd="start", val=None, cn_ids=cn_list)

    audit_log(
        store, namespace="search_rt",
        cn_target=",".join(str(c) for c in cn_list),
        cmd="bounce_search", val={"cn_ids": cn_list, "sleep_s": sleep_s},
        ok=True,
        note=f"stop+sleep({sleep_s:.1f}s)+start on cn_ids={cn_list}",
        user=user,
    )
    return {
        "ok": True,
        "cmd": "bounce_search",
        "cn_ids": cn_list,
        "stop_keys": stop_keys,
        "start_keys": start_keys,
        "sleep_s": sleep_s,
    }


def restart_c2_service_local(
    store: ControlStore,
    *,
    unit: str = C2_SERVICE_UNIT,
    timeout_s: float = 10.0,
    user: str | None = None,
) -> dict[str, Any]:
    """Run ``systemctl --user restart <unit>`` on the local host.

    The dsa_monitor process runs on h23 as the same user that owns
    ``dsart_c2.service``, so no ssh is required. Returns
    ``{ok, rc, stdout, stderr, err, elapsed_s, unit}``. Never raises;
    a missing systemctl binary or a non-zero exit shows up as
    ``ok=False`` with a descriptive ``err``.
    """
    argv = ["systemctl", "--user", "restart", unit]
    started = time.monotonic()
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s,
        )
        ok = cp.returncode == 0
        result = {
            "ok": ok,
            "rc": cp.returncode,
            "stdout": cp.stdout or "",
            "stderr": cp.stderr or "",
            "err": "" if ok else f"rc={cp.returncode}",
            "elapsed_s": round(time.monotonic() - started, 3),
            "unit": unit,
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            "ok": False,
            "rc": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "err": f"timeout after {timeout_s:.1f}s",
            "elapsed_s": round(time.monotonic() - started, 3),
            "unit": unit,
        }
    except (FileNotFoundError, OSError) as exc:
        result = {
            "ok": False,
            "rc": None,
            "stdout": "",
            "stderr": "",
            "err": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
            "unit": unit,
        }

    audit_log(
        store, namespace="services",
        cn_target="local",
        cmd="restart_c2",
        val={"unit": unit, "elapsed_s": result["elapsed_s"]},
        ok=bool(result["ok"]),
        note=(
            f"systemctl --user restart {unit} → "
            f"rc={result.get('rc')} elapsed_s={result['elapsed_s']}"
            + (f" err={result['err']!r}" if result.get("err") else "")
        ),
        user=user,
    )
    return result


def _systemctl_user_restart(unit: str, *, timeout_s: float) -> dict[str, Any]:
    """Run ``systemctl --user restart <unit>`` once on the local host.

    Returns ``{unit, ok, rc, stdout, stderr, err, elapsed_s}``. Never
    raises — a missing systemctl binary, a timeout, or a non-zero exit
    all surface as ``ok=False`` with a descriptive ``err``.
    """
    argv = ["systemctl", "--user", "restart", unit]
    started = time.monotonic()
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s,
        )
        ok = cp.returncode == 0
        return {
            "unit": unit,
            "ok": ok,
            "rc": cp.returncode,
            "stdout": cp.stdout or "",
            "stderr": cp.stderr or "",
            "err": "" if ok else f"rc={cp.returncode}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "unit": unit, "ok": False, "rc": None,
            "stdout": exc.stdout or "", "stderr": exc.stderr or "",
            "err": f"timeout after {timeout_s:.1f}s",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except (FileNotFoundError, OSError) as exc:
        return {
            "unit": unit, "ok": False, "rc": None,
            "stdout": "", "stderr": "",
            "err": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }


def restart_h23_services_local(
    store: ControlStore,
    *,
    units: Iterable[str] = H23_DSART_UNITS,
    timeout_s: float = 30.0,
    user: str | None = None,
) -> dict[str, Any]:
    """Restart every h23 dsa110-rt ``systemctl --user`` unit, in order.

    Walks :data:`H23_DSART_UNITS` (dsart_c2, dsart_c3, hiplot_c1,
    hiplot_c2) and runs a local ``systemctl --user restart`` on each.
    The dashboard runs on h23 as the same user that owns those units, so
    no ssh is required. The dashboard's own unit is deliberately NOT in
    the list (it would kill this request mid-flight).

    Returns ``{ok, cmd, results, elapsed_s}`` where ``results`` is the
    per-unit list from :func:`_systemctl_user_restart`. ``ok`` is True
    iff every unit restarted cleanly. One summary audit row is written;
    individual unit failures do NOT abort the rest (best-effort: a wedged
    hiplot must not stop C2/C3 from cycling). Never raises.
    """
    unit_list = [str(u) for u in units]
    started = time.monotonic()
    results = [
        _systemctl_user_restart(u, timeout_s=timeout_s) for u in unit_list
    ]
    ok = all(r["ok"] for r in results) and bool(results)
    elapsed_s = round(time.monotonic() - started, 3)
    n_ok = sum(1 for r in results if r["ok"])
    audit_log(
        store, namespace="services",
        cn_target="h23",
        cmd="restart_h23_services",
        val={r["unit"]: bool(r["ok"]) for r in results},
        ok=ok,
        note=(
            f"{n_ok}/{len(results)} h23 units restarted; "
            f"elapsed_s={elapsed_s}"
            + (
                " failed: " + ",".join(
                    r["unit"] for r in results if not r["ok"]
                )
                if n_ok != len(results) else ""
            )
        ),
        user=user,
    )
    return {
        "ok": ok,
        "cmd": "restart_h23_services",
        "results": results,
        "elapsed_s": elapsed_s,
    }


# ---------------------------------------------------------------------------
# Read-side helpers — for the Control tab's live "what's happening
# right now" mini-panels. Local subprocess + local etcd reads only.
# ---------------------------------------------------------------------------


#: Default etcd key for the C2 mon snapshot. Mirrors
#: ``dsart.services.coincidencer._mon_publish_loop``'s publish target.
C2_MON_KEY: str = "/mon/c2/h23"

#: Regex that matches each meaningful per-cluster decision line in the
#: dsart_c2 journal. Kept here so tests can pin it (an upstream
#: refactor that changes the log shape will fail the matching test).
C2_DECISION_RE: re.Pattern[str] = re.compile(
    r"(FIRE class|WOULD-DUMP|LOG class|inject_match)",
)


def c2_mon_snapshot(store: ControlStore) -> dict[str, Any]:
    """Read ``/mon/c2/h23`` and return a stripped-down view suited for
    a small live panel.

    Returns:

        {
          "ok": bool,
          "key": "/mon/c2/h23",
          "raw": <full dict or None>,
          "counters": <counters dict or {}>,
          "dumps_enabled": bool | None,
          "inject_match": {
            "active_count", "rows_checked", "matches",
            "evicted_expired", "last_refresh_age_s", "active",
          } or {},
        }

    A missing key returns ``ok=False`` with ``raw=None`` — the UI
    treats that as "C2 hasn't published yet" rather than a hard error.
    """
    try:
        raw = store.get_dict(C2_MON_KEY)
    except Exception as exc:                                       # noqa: BLE001
        LOG.warning("c2_mon_snapshot: get_dict failed: %s", exc)
        return {
            "ok": False, "key": C2_MON_KEY, "raw": None,
            "counters": {}, "dumps_enabled": None, "inject_match": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(raw, dict):
        return {
            "ok": False, "key": C2_MON_KEY, "raw": None,
            "counters": {}, "dumps_enabled": None, "inject_match": {},
        }
    counters = raw.get("counters") if isinstance(raw.get("counters"), dict) else {}
    im_raw = raw.get("inject_match") if isinstance(raw.get("inject_match"), dict) else {}
    last_refresh_unix = im_raw.get("last_refresh_unix")
    if isinstance(last_refresh_unix, (int, float)):
        last_refresh_age_s = float(time.time() - float(last_refresh_unix))
    else:
        last_refresh_age_s = None
    im_view = {
        "active_count": im_raw.get("active_count"),
        "rows_checked": im_raw.get("rows_checked"),
        "matches": im_raw.get("matches"),
        "evicted_expired": im_raw.get("evicted_expired"),
        "last_refresh_age_s": last_refresh_age_s,
        "active": im_raw.get("active") if isinstance(im_raw.get("active"), list) else [],
    }
    return {
        "ok": True,
        "key": C2_MON_KEY,
        "raw": raw,
        "counters": counters,
        "dumps_enabled": raw.get("dumps_enabled"),
        "last_event_name": raw.get("last_event_name"),
        "last_trigger_class": raw.get("last_trigger_class"),
        "inject_match": im_view,
    }


def c2_journal_tail_local(
    *,
    limit: int = 50,
    since: str = "10 min ago",
    unit: str = C2_SERVICE_UNIT,
    grep_re: re.Pattern[str] | None = C2_DECISION_RE,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Return the last ``limit`` decision lines from the C2 user
    journal, optionally filtered by ``grep_re`` (default: FIRE /
    WOULD-DUMP / LOG class / inject_match lines).

    Output dict:

        {
          "ok": bool,
          "unit": "dsart_c2.service",
          "since": "...",
          "lines": ["<line>", ...],   # newest last
          "raw_lines_scanned": int,
          "err": str,                  # empty on success
        }

    No stdin is consumed; safe to call from any thread. Errors
    (missing journalctl, permission, timeout) come back as
    ``ok=False`` with an explanatory ``err`` — they NEVER raise.
    """
    argv = [
        "journalctl",
        "--user", "-u", unit,
        "--no-pager",
        "--since", str(since),
        "-n", str(int(limit * 4 if grep_re is not None else limit)),
    ]
    started = time.monotonic()
    try:
        cp = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False, "unit": unit, "since": since,
            "lines": [], "raw_lines_scanned": 0,
            "err": f"timeout after {timeout_s:.1f}s",
            "elapsed_s": round(time.monotonic() - started, 3),
            "stdout": exc.stdout or "", "stderr": exc.stderr or "",
        }
    except (FileNotFoundError, OSError) as exc:
        return {
            "ok": False, "unit": unit, "since": since,
            "lines": [], "raw_lines_scanned": 0,
            "err": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    if cp.returncode != 0:
        return {
            "ok": False, "unit": unit, "since": since,
            "lines": [], "raw_lines_scanned": 0,
            "err": f"journalctl rc={cp.returncode} stderr={cp.stderr.strip()[:200]!r}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    raw_lines = (cp.stdout or "").splitlines()
    if grep_re is None:
        kept = raw_lines
    else:
        kept = [ln for ln in raw_lines if grep_re.search(ln)]
    return {
        "ok": True,
        "unit": unit,
        "since": since,
        "lines": kept[-int(limit):],
        "raw_lines_scanned": len(raw_lines),
        "err": "",
        "elapsed_s": round(time.monotonic() - started, 3),
    }
