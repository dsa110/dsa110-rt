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
import socket
import threading
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
    margin: int = DEFAULT_ARM_SEQ_MARGIN,
    chgroups: Iterable[int] = DEFAULT_INJECT_CHGROUPS,
    user: str | None = None,
) -> dict[str, Any]:
    """High-level helper for the Control tab.

    Same surface as :func:`send_inject` but ``apply_at_specnum``
    may be ``None`` — in which case we compute it as
    ``max(last_seq_no across captures) + margin`` (the same
    computation :func:`compute_arm_seq` uses for ``utc_start``).
    This is the operator's "inject as soon as possible" path: the
    pulse lands ``margin`` specnums (~32.768 µs each) after the
    current capture front-edge, which gives the corr-fast pipeline
    enough headroom to receive + queue the injection before the
    target block arrives.

    Returns the dict :func:`send_inject` returns, plus an extra
    ``arm_info`` key describing how ``apply_at_specnum`` was derived
    (handy for the UI to confirm what the auto-pick produced).
    """
    if apply_at_specnum is None:
        info = compute_arm_seq(store, margin=margin)
        if info["arm_seq"] is None:
            audit_log(
                store, namespace="dsart.inject",
                cn_target=",".join(str(c) for c in chgroups),
                cmd="inject", val=None, ok=False,
                note=(
                    f"refused (auto-arm): no captures answering "
                    f"({len(info['missing'])} missing, "
                    f"{len(info['answered'])} answered)"
                ),
                user=user,
            )
            return {
                "ok": False,
                "cmd": "inject",
                "error": (
                    "apply_at_specnum=null requested but no captures "
                    "are answering last_seq_no — cannot derive a safe "
                    "specnum. Either start the capture fleet first or "
                    "pass apply_at_specnum explicitly."
                ),
                "info": info,
            }
        derived_specnum = int(info["arm_seq"])
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
    return out


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
