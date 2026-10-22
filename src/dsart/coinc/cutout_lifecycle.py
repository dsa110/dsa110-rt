"""C2-side cutout lifecycle (x2 SPEC §9, docs/x2/downstream_inventory.md §3.5).

Pieces the coincidencer composes when ``coinc.cutout.enabled`` is true:

* :class:`CutoutConfig`   -- the ``coinc.cutout`` block (defaults = off).
* :func:`expected_halves` -- the halves that must answer, from the plan's
  owner map (never a literal 8).
* :func:`build_request`   -- a :class:`~dsart.coinc.wire.CutoutRequest`
  from the fired cluster's peak member.
* :class:`CutoutTracker`  -- per-event request / ack / resend / status
  bookkeeping; lossy UDP status is advisory, the per-half MANIFEST on h23
  is the completion signal.
* :class:`CutoutStatusProtocol` -- asyncio UDP endpoint for status datagrams.
* pending-state persistence (``Level2/products_pending.json``) so a C2
  restart resumes in-flight events instead of losing them, and the Level3
  ``products`` block.

An event is never deleted here: an incomplete set is FINALISED with its
missing halves recorded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import wire
from .stats import ClusterStats
from .window import WindowEntry

__all__ = [
    "CutoutConfig",
    "PARTIAL_POLICY_DISCARD",
    "PARTIAL_POLICY_ARCHIVE",
    "expected_halves",
    "sid_of_node",
    "peak_member",
    "build_request",
    "HalfState",
    "CutoutTracker",
    "CutoutStatusProtocol",
    "products_block",
    "write_pending",
    "read_pending",
    "PENDING_FILE",
]

_LOG = logging.getLogger("dsart.coinc.cutout_lifecycle")

PARTIAL_POLICY_DISCARD = "discard"
PARTIAL_POLICY_ARCHIVE = "archive"
_PARTIAL_POLICIES = (PARTIAL_POLICY_DISCARD, PARTIAL_POLICY_ARCHIVE)

#: Persisted pending state, under the event's Level2/ (C3 ignores it).
PENDING_FILE = "products_pending.json"

#: Search nodes in owner-half order (SPEC §2: half h -> node h // 2).
_DEFAULT_NODES: Tuple[str, ...] = ("n01", "n02", "n09", "n13")


@dataclass(frozen=True)
class CutoutConfig:
    """``coinc.cutout`` (SPEC §9 defaults). ``enabled: false`` = 1x path."""

    enabled: bool = False
    status_bind_host: str = "0.0.0.0"
    status_port: int = wire.CUTOUT_STATUS_PORT_DEFAULT
    rows: int = 512
    raw_pre_ms: float = 500.0
    raw_post_ms: float = 500.0
    raw_for_injections: bool = False
    stamp_half_px: int = 16
    n_stamp_t: int = 64
    stamp_dm_halfwidth_pc_cc: float = 150.0
    deadline_s: int = 15
    ack_timeout_s: float = 2.0
    max_resends: int = 3
    wait_s: float = 120.0
    late_accept_s: float = 3600.0
    plan_path: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls, d: Optional[Mapping[str, Any]], *, default_enabled: bool = False,
    ) -> "CutoutConfig":
        d = dict(d or {})
        plan = d.get("plan_path")
        return cls(
            enabled=bool(d.get("enabled", default_enabled)),
            status_bind_host=str(d.get("status_bind_host", "0.0.0.0")),
            status_port=int(d.get("status_port", wire.CUTOUT_STATUS_PORT_DEFAULT)),
            rows=int(d.get("rows", 512)),
            raw_pre_ms=float(d.get("raw_pre_ms", 500.0)),
            raw_post_ms=float(d.get("raw_post_ms", 500.0)),
            raw_for_injections=bool(d.get("raw_for_injections", False)),
            stamp_half_px=int(d.get("stamp_half_px", 16)),
            n_stamp_t=int(d.get("n_stamp_t", 64)),
            stamp_dm_halfwidth_pc_cc=float(d.get("stamp_dm_halfwidth_pc_cc", 150.0)),
            deadline_s=int(d.get("deadline_s", 15)),
            ack_timeout_s=float(d.get("ack_timeout_s", 2.0)),
            max_resends=int(d.get("max_resends", 3)),
            wait_s=float(d.get("wait_s", 120.0)),
            late_accept_s=float(d.get("late_accept_s", 3600.0)),
            plan_path=Path(plan) if plan else None,
        )


def check_partial_policy(policy: str) -> str:
    p = str(policy or PARTIAL_POLICY_DISCARD)
    if p not in _PARTIAL_POLICIES:
        raise ValueError(f"partial_event_policy={policy!r}, expected one of {_PARTIAL_POLICIES}")
    return p


def sid_of_node(node: str) -> int:
    """``"n13"`` -> 13 (the C1 ``search_node_id`` / dump-host key)."""
    return int(str(node).lstrip("nN"))


def expected_halves(plan: Any, nodes: Sequence[str] = _DEFAULT_NODES) -> List[Tuple[int, int]]:
    """``[(sid, g)]`` of every half that owns >= 1 coarse DM in ``plan``.

    ``plan`` is a TieredDmPlan (``owner_half`` per coarse DM, half h = node
    ``h // 2``, GPU ``h % 2``). For a plan without ``owner_half`` (v2) every
    half of ``nodes`` is expected (one coarse DM per half by construction).
    """
    owner = getattr(plan, "owner_half", None)
    if owner is None:
        halves = range(2 * len(nodes))
    else:
        halves = sorted({int(h) for h in list(owner)})
    return [(sid_of_node(nodes[h // 2]), h % 2) for h in halves]


def peak_member(members: Sequence[WindowEntry]) -> WindowEntry:
    """The max-SNR member, first on ties -- the same member
    ``compute_stats`` takes for ``peak_*`` (``np.argmax``)."""
    best = members[0]
    for m in members[1:]:
        if m.snr > best.snr:
            best = m
    return best


def _ext_rows(ms: float, period_us: float) -> int:
    return max(0, int(round(float(ms) * 1000.0 / float(period_us))))


def build_request(
    *,
    event_name: str,
    request_id: int,
    stats: ClusterStats,
    members: Sequence[WindowEntry],
    cfg: CutoutConfig,
    trigger_class_name: str,
    is_injection: bool,
    plan: Any = None,
    plan_id: int = 0,
) -> wire.CutoutRequest:
    """The request for a fired cluster, centred on its PEAK member."""
    pk = peak_member(members)
    period = float(getattr(pk, "sample_period_us", 0.0) or 0.0)
    if not (math.isfinite(period) and period > 0.0):
        period = float(stats.peak_sample_period_us or 0.0) or 524.288
    t0, n_t = wire.default_cutout_window(int(pk.event_specnum), int(cfg.rows))
    fine_g = int(pk.dm_idx_global)
    coarse = -1
    if plan is not None:
        f2c = getattr(plan, "fine_to_coarse", None)
        if f2c is not None and 0 <= fine_g < len(f2c):
            coarse = int(f2c[fine_g])
        else:
            fine_g = -1
    flags = wire.CUTOUT_FLAG_CUTOUT
    if is_injection:
        flags |= wire.CUTOUT_FLAG_INJECTION
    if not is_injection or cfg.raw_for_injections:
        flags |= wire.CUTOUT_FLAG_RAW_ARCHIVE
    req = wire.CutoutRequest(
        event_name=event_name, request_id=int(request_id) & 0xFFFFFFFF,
        t_peak_ext=int(pk.event_specnum), t0_ext=t0, n_t=n_t,
        raw_pre_ext=_ext_rows(cfg.raw_pre_ms, period),
        raw_post_ext=_ext_rows(cfg.raw_post_ms, period),
        width_ext=max(1, int(pk.width_samples)),
        dm_peak_pc_cc=float(pk.dm_pc_cc), snr_peak=float(pk.snr),
        l_rad=float(pk.l_rad), m_rad=float(pk.m_rad),
        l_pix=int(pk.l_pix), m_pix=int(pk.m_pix),
        fine_global_peak=fine_g, coarse_global_peak=coarse,
        stamp_half_px=int(cfg.stamp_half_px), n_stamp_t=int(cfg.n_stamp_t),
        stamp_dm_halfwidth_pc_cc=float(cfg.stamp_dm_halfwidth_pc_cc),
        mjd_peak=float(pk.mjd), plan_id=int(plan_id),
        deadline_s=int(cfg.deadline_s),
        trigger_class_id=wire.class_id_for(trigger_class_name),
        flags=flags,
    )
    req.validate()
    return req


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


@dataclass
class HalfState:
    """What C2 knows about one half for one request."""

    sends: int = 0
    last_send_mono: float = 0.0
    acked: bool = False
    state: str = "requested"
    reason: str = ""
    valid_frac: float = float("nan")
    latency_ms: int = 0
    cutout_bytes: int = 0
    raw_bytes: int = 0


@dataclass
class _Tracked:
    request: wire.CutoutRequest
    halves: Dict[Tuple[int, int], HalfState]
    started_mono: float


class CutoutTracker:
    """Request / ack / resend bookkeeping for in-flight cutout requests.

    ``send`` is ``(blob, halves) -> {(sid, g): bool}``
    (:meth:`TriggerBroadcaster.send_to`). Single-threaded (C2 event loop).
    """

    def __init__(
        self,
        send: Callable[[bytes, Iterable[Tuple[int, int]]], Dict[Tuple[int, int], bool]],
        cfg: CutoutConfig,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send = send
        self._cfg = cfg
        self._now = now
        self._events: Dict[str, _Tracked] = {}
        self.counters: Dict[str, int] = {
            "cutout_requests_sent": 0,
            "cutout_resends": 0,
            "cutout_send_fail": 0,
            "cutout_status_rx": 0,
            "cutout_status_bad": 0,
            "cutout_status_unknown_event": 0,
        }

    def start(self, req: wire.CutoutRequest, halves: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], bool]:
        t = self._now()
        tr = _Tracked(req, {(int(s), int(g)): HalfState() for s, g in halves}, t)
        self._events[req.event_name] = tr
        return self._send_to(tr, list(tr.halves), resend=False)

    def _send_to(self, tr: _Tracked, halves: List[Tuple[int, int]], *, resend: bool) -> Dict[Tuple[int, int], bool]:
        req = tr.request
        if resend:
            req = wire.CutoutRequest(**{**_req_fields(req), "flags": req.flags | wire.CUTOUT_FLAG_RESEND})
        blob = wire.encode_cutout_request(req)
        res = self._send(blob, halves)
        t = self._now()
        for sg, ok in res.items():
            hs = tr.halves.get(sg)
            if hs is None:
                continue
            hs.sends += 1
            hs.last_send_mono = t
            if not ok:
                self.counters["cutout_send_fail"] += 1
        key = "cutout_resends" if resend else "cutout_requests_sent"
        self.counters[key] += len(res)
        return res

    def service_resends(self) -> int:
        """Resend to halves with no ACK after ``ack_timeout_s`` (at most
        ``max_resends`` times). Returns the number of datagrams resent."""
        now = self._now()
        n = 0
        for tr in self._events.values():
            due = [sg for sg, hs in tr.halves.items()
                   if not hs.acked and hs.sends <= self._cfg.max_resends
                   and now - hs.last_send_mono >= self._cfg.ack_timeout_s]
            if due:
                n += len(self._send_to(tr, due, resend=True))
        return n

    def on_status(self, st: wire.CutoutStatus) -> bool:
        """Record a status datagram; False if it matches no tracked request."""
        self.counters["cutout_status_rx"] += 1
        tr = self._events.get(st.event_name)
        if tr is None or int(tr.request.request_id) != int(st.request_id):
            self.counters["cutout_status_unknown_event"] += 1
            return False
        hs = tr.halves.get((int(st.search_node_id), int(st.gpu_half)))
        if hs is None:
            self.counters["cutout_status_unknown_event"] += 1
            return False
        hs.acked = True
        # A late ACCEPTED never overwrites a terminal state.
        if st.is_terminal or hs.state in ("requested",):
            hs.state = st.state_name
            hs.reason = st.reason_name
        hs.valid_frac = float(st.valid_frac)
        hs.latency_ms = int(st.latency_ms)
        hs.cutout_bytes = max(hs.cutout_bytes, int(st.cutout_bytes))
        hs.raw_bytes = max(hs.raw_bytes, int(st.raw_bytes))
        return True

    def on_datagram(self, data: bytes) -> None:
        if wire.packet_kind(data) != "cutout_status":
            self.counters["cutout_status_bad"] += 1
            return
        try:
            st = wire.decode_cutout_status(data)
        except wire.BadBatch:
            self.counters["cutout_status_bad"] += 1
            return
        self.on_status(st)

    def get(self, event_name: str) -> Optional[_Tracked]:
        return self._events.get(event_name)

    def half_states(self, event_name: str) -> Dict[Tuple[int, int], HalfState]:
        tr = self._events.get(event_name)
        return dict(tr.halves) if tr else {}

    def all_terminal(self, event_name: str) -> bool:
        tr = self._events.get(event_name)
        return bool(tr) and all(hs.state in ("done", "partial", "failed")
                                for hs in tr.halves.values())

    def forget(self, event_name: str) -> None:
        self._events.pop(event_name, None)

    @property
    def n_pending(self) -> int:
        return len(self._events)


def _req_fields(req: wire.CutoutRequest) -> Dict[str, Any]:
    return {f: getattr(req, f) for f in req.__dataclass_fields__}


class CutoutStatusProtocol(asyncio.DatagramProtocol):
    """asyncio endpoint feeding :meth:`CutoutTracker.on_datagram`."""

    def __init__(self, tracker: CutoutTracker) -> None:
        self._tracker = tracker

    def datagram_received(self, data: bytes, addr) -> None:  # noqa: ANN001
        try:
            self._tracker.on_datagram(bytes(data))
        except Exception:  # noqa: BLE001 - never kill the endpoint
            _LOG.exception("cutout status handling failed (from %s)", addr)

    def error_received(self, exc: Exception) -> None:
        _LOG.warning("cutout status endpoint error: %r", exc)


# ---------------------------------------------------------------------------
# Level3 products block + persisted pending state
# ---------------------------------------------------------------------------


def _label(sg: Tuple[int, int]) -> str:
    return f"s{int(sg[0])}g{int(sg[1])}"


def products_block(
    *,
    kind: str,
    expected: Sequence[Tuple[int, int]],
    present: Iterable[Tuple[int, int]],
    half_states: Optional[Mapping[Tuple[int, int], HalfState]] = None,
    opoint: str = "",
    plan_id: int = 0,
    request: Optional[wire.CutoutRequest] = None,
    raw: Optional[Mapping[str, Any]] = None,
    finalised_unix: Optional[float] = None,
) -> Dict[str, Any]:
    """The Level3 ``products`` block (SPEC §9). Missing halves carry the
    reason of their last status, or ``no_status``/``no_product``."""
    exp = sorted({(int(s), int(g)) for s, g in expected})
    pres = sorted({(int(s), int(g)) for s, g in present})
    states = dict(half_states or {})
    missing: Dict[str, str] = {}
    for sg in exp:
        if sg in pres:
            continue
        hs = states.get(sg)
        if hs is None:
            missing[_label(sg)] = "no_product"
        elif not hs.acked:
            missing[_label(sg)] = "no_status"
        else:
            missing[_label(sg)] = f"{hs.state}:{hs.reason}" if hs.reason else hs.state
    out: Dict[str, Any] = {
        "kind": kind,
        "opoint": opoint,
        "plan_id": f"0x{int(plan_id):016x}",
        "expected": [_label(sg) for sg in exp],
        "present": [_label(sg) for sg in pres],
        "missing": missing,
        "status": "complete" if exp and not missing else "incomplete",
        "finalised_unix": float(finalised_unix if finalised_unix is not None else time.time()),
        "late": [],
    }
    if request is not None:
        out["request"] = {
            "request_id": int(request.request_id), "t_peak_ext": int(request.t_peak_ext),
            "t0_ext": int(request.t0_ext), "n_t": int(request.n_t),
            "raw_pre_ext": int(request.raw_pre_ext), "raw_post_ext": int(request.raw_post_ext),
            "flags": int(request.flags),
        }
    if raw is not None:
        out["raw"] = dict(raw)
    return out


def _atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_pending(event_dir: Path, doc: Mapping[str, Any]) -> Path:
    p = Path(event_dir) / "Level2" / PENDING_FILE
    _atomic_json(p, doc)
    return p


def read_pending(event_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(event_dir) / "Level2" / PENDING_FILE
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
