"""dsa_monitor Phase 6c: one-shot fleet-wide cube dump trigger.

Operator workflow
-----------------

1. Operator hits the "Dump Now" button on the Control tab during a
   persistent on-sky RFI event. The dashboard POSTs to
   ``/control/dump_now`` with ``confirm=dump_now``.
2. The Flask route calls :func:`fleet_dump_now`, which:

   a. Resolves the 8 search-half UDP destinations
      (``(sid, gpu_half, host, port)``) from
      ``configs/dsart_search_rt.yaml`` (``coinc.dump_broadcast``).
   b. Picks an ``event_specnum`` a few blocks behind the fleet's
      latest published ``block_specnum_start`` so the synthetic
      trigger lands squarely in the middle of every cube ring's
      retention window (not ``too_early`` and not ``too_late``).
   c. Encodes one ``C2TriggerPacket`` with a synthetic
      ``dumpnow_<8base32>`` event name via the canonical
      ``src.dsart.coinc.wire.encode_c2_trigger`` so the byte layout
      is bit-identical to what the production C2 service ships.
   d. UDP-sends the same 64-byte blob to every ``(host, port)`` pair.
      The sender callable is injected so tests can fake the network.

3. The result (``per_half`` map, ``pass_count``, ``fail_count``,
   ``event_name``, ``event_specnum``, ``latency_ms``) is returned to
   the UI as JSON.

This module deliberately performs NO production-data plumbing. The
search nodes already listen for C2 trigger packets via
:class:`dsart.dump.c2_trigger_listener.C2TriggerListener` bound at
``(c1.dump_listener.bind_host, c1.dump_listener.base_port +
gpu_half)``. We just send them an extra, synthetic packet.

The module is stdlib + ``yaml`` only at import time. The
``dsart.coinc.wire`` import is lazy (inside :func:`fleet_dump_now`)
so the dashboard env, which historically does not always have
``dsart`` on ``PYTHONPATH``, can at least import this module to
surface a clear error message rather than crashing on import.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

LOG = logging.getLogger("dsa_monitor.cube_dump_now")


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

#: Repo-root-relative path to the search-rt yaml. Computed from this
#: module's ``__file__`` so the dashboard finds the canonical yaml in
#: the deployed checkout (``/home/ubuntu/proj/dsa110-rt`` on h23).
_HERE: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH: str = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "configs", "dsart_search_rt.yaml")
)

#: Default broadcast port base (``c1.dump_listener.base_port`` /
#: ``coinc.dump_broadcast.port_base``). The actual destination port
#: per search-half is ``base_port + gpu_half``.
DEFAULT_PORT_BASE: int = 11227

#: The two halves served by each search node (``cuda:0`` and
#: ``cuda:1`` of the per-host RTX 2080 Ti pair).
GPU_HALVES: tuple[int, ...] = (0, 1)

#: Default per-call sendto timeout (seconds). UDP sendto rarely
#: actually blocks; the timeout is mostly here to keep an unreachable
#: host from hanging the whole fan-out indefinitely.
DEFAULT_TIMEOUT_S: float = 2.0

#: Mirrors :data:`dsart.services.corr_fast_mon.NPACKETS_PER_BLOCK_DEFAULT`
#: + ``control_store.NPACKETS_PER_BLOCK`` so the dashboard does not
#: pull torch (which is not on the h23 env) just to read one constant.
#: Drift between the two copies is caught by
#: :func:`tests/test_dsa_monitor_control_store.py::test_npackets_per_block_constant_matches_corr_fast`.
NPACKETS_PER_BLOCK: int = 2048

#: How many ``NPACKETS_PER_BLOCK``-sized blocks to subtract from the
#: fleet's latest ``max(block_specnum_start)`` so the synthetic
#: trigger lands in the middle of every ring's retention window. At
#: production geometry (cube_ring_depth=16, 7.45 cubes/s, cube cadence
#: 128 samples × sample_period_specnum 16 = 2048 specnums/cube)
#: subtracting 4 blocks = 8192 specnums ≈ 4 cubes places us well
#: inside the [oldest, newest) retention window for every gpu_half
#: regardless of which one is the lagger.
#:
#: NOTE — this is the *legacy* corr_fast fallback path's lookback.
#: The primary path (search-ring mon-key, see
#: :func:`_resolve_event_specnum_from_search_ring`) uses
#: :data:`DEFAULT_LOOKBACK_CUBES` instead so the units are
#: ring-natural.
DEFAULT_LOOKBACK_BLOCKS: int = 4

#: How many cubes behind the fleet-wide *min-newest* cube to target
#: when using the search-ring mon-key path. Cube_ring_depth is 16 in
#: production, so 4 cubes behind the laggard's newest still leaves
#: 12 cubes of margin before the oldest slot rolls over. Keeping the
#: target a few cubes behind the leading edge also guarantees the
#: leader has finished writing the slot we hit (avoids
#: half-populated cubes that show up as ``too_early`` even when the
#: specnum domain is right).
DEFAULT_LOOKBACK_CUBES: int = 4

#: Default sender timeout for tests + cold-boot safeguarding. The
#: corr_fast publisher writes ``/mon/corr_rt/<cn>/corr_fast`` every
#: 16 blocks (~2 s) so any wall-clock entry older than this is
#: treated as stale (NTP-bounded across hosts). Search-ring keys
#: refresh at the same cadence (every cube_progress log line ≈ 1.3 s
#: at 7.45 cubes/s).
DEFAULT_MAX_PUBLISHER_AGE_S: float = 10.0

#: Event-name prefix for synthetic dumps. The C2TriggerPacket
#: ``event_name`` field is 16 ASCII bytes — ``dumpnow_`` (8 chars) +
#: 8 base-32 characters = 16 characters exactly. Operators can grep
#: for ``dumpnow_*`` in the search-node logs / cube_dump_root to
#: distinguish these from production C2-triggered events.
EVENT_NAME_PREFIX: str = "dumpnow_"
EVENT_NAME_RANDOM_LEN: int = 8

#: RFC 4648 base-32 alphabet (uppercase, no padding). Restricting to
#: this set keeps the event name pure-ASCII and filesystem-safe so
#: the search-side ``${dump_root}/<event_name>/`` directory creation
#: never tries to escape into a subdirectory.
_BASE32_ALPHABET: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FleetDumpNowResult:
    """One-shot fleet dump-now result, JSON-serialisable via
    :meth:`to_dict`.

    ``per_half`` keys are ``(search_node_id, gpu_half)`` tuples and
    values are status strings — ``"ok"`` for a successful sendto,
    ``"timeout"`` / ``"oserror: <repr>"`` / ``"exception: <repr>"`` /
    ``"skipped"`` otherwise.

    ``error`` is non-``None`` only on the cold-boot path where no
    corr_fast mon-key was answering and no caller-provided
    ``target_specnum`` was supplied. The Flask layer translates that
    into HTTP 503.
    """

    ok: bool
    event_name: str
    event_specnum: int | None
    per_half: dict[tuple[int, int], str]
    pass_count: int
    fail_count: int
    latency_ms: float
    bind_hosts: tuple[tuple[int, int, str, int], ...] = ()
    arm_info: dict[str, Any] | None = None
    error: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; ``per_half`` keys are flattened to
        ``"s<sid>_g<g>"`` strings because tuple keys do not survive
        ``json.dumps``.
        """
        return {
            "ok": bool(self.ok),
            "event_name": str(self.event_name),
            "event_specnum": (
                None if self.event_specnum is None else int(self.event_specnum)
            ),
            "per_half": {
                f"s{int(sid)}_g{int(g)}": str(status)
                for (sid, g), status in sorted(
                    self.per_half.items(),
                    key=lambda kv: (int(kv[0][0]), int(kv[0][1])),
                )
            },
            "pass_count": int(self.pass_count),
            "fail_count": int(self.fail_count),
            "latency_ms": float(self.latency_ms),
            "bind_hosts": [
                {
                    "sid": int(sid), "gpu_half": int(g),
                    "host": str(host), "port": int(port),
                }
                for (sid, g, host, port) in self.bind_hosts
            ],
            "arm_info": self.arm_info,
            "error": self.error,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Bind-list resolution from yaml
# ---------------------------------------------------------------------------


def _load_yaml(yaml_path: str) -> dict[str, Any]:
    """Read + ``yaml.safe_load`` a config. Returns ``{}`` if the
    file is missing / unreadable so the caller can fall back to a
    hardcoded bind list without dying on file IO.
    """
    try:
        import yaml                                          # local import
    except Exception as exc:                                 # noqa: BLE001
        LOG.warning("cube_dump_now: pyyaml not importable: %s", exc)
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except FileNotFoundError:
        LOG.warning(
            "cube_dump_now: yaml not found at %s; bind list will be empty",
            yaml_path,
        )
        return {}
    except Exception as exc:                                 # noqa: BLE001
        LOG.warning(
            "cube_dump_now: yaml.safe_load(%s) failed: %s", yaml_path, exc,
        )
        return {}
    return doc if isinstance(doc, dict) else {}


def resolve_bind_hosts(
    yaml_path: str | None = None,
) -> list[tuple[int, int, str, int]]:
    """Read ``configs/dsart_search_rt.yaml`` and return one
    ``(sid, gpu_half, host, port)`` tuple per search-half.

    Looks up ``coinc.dump_broadcast.hosts`` (the production C2-side
    fan-out map, ``{sid: ipv4}``) + ``coinc.dump_broadcast.port_base``.
    The port for each gpu_half is ``port_base + gpu_half`` per
    :class:`dsart.dump.c2_trigger_listener.C2TriggerListenerConfig`.
    Returns an empty list if the yaml is missing or has no host map —
    the caller surfaces that as ``fail_count=0, pass_count=0``.
    """
    path = str(yaml_path or DEFAULT_YAML_PATH)
    doc = _load_yaml(path)
    coinc = doc.get("coinc", {}) if isinstance(doc, dict) else {}
    if not isinstance(coinc, dict):
        coinc = {}
    dump = coinc.get("dump_broadcast", {}) or {}
    if not isinstance(dump, dict):
        dump = {}
    hosts_raw = dump.get("hosts", {}) or {}
    if not isinstance(hosts_raw, dict):
        hosts_raw = {}
    try:
        port_base = int(dump.get("port_base", DEFAULT_PORT_BASE))
    except (TypeError, ValueError):
        port_base = DEFAULT_PORT_BASE

    out: list[tuple[int, int, str, int]] = []
    # Sort by integer sid so the per_half dict has a stable order in
    # the UI / JSON output regardless of yaml key ordering.
    items: list[tuple[int, str]] = []
    for sid_str, ip in hosts_raw.items():
        try:
            sid = int(sid_str)
        except (TypeError, ValueError):
            LOG.warning(
                "cube_dump_now: skipping non-int sid in yaml: %r", sid_str,
            )
            continue
        if not isinstance(ip, str):
            LOG.warning(
                "cube_dump_now: skipping non-str ip in yaml: sid=%d ip=%r",
                sid, ip,
            )
            continue
        items.append((sid, str(ip)))
    items.sort()
    for sid, host in items:
        for g in GPU_HALVES:
            out.append((sid, int(g), host, port_base + int(g)))
    return out


# ---------------------------------------------------------------------------
# Event-name generation
# ---------------------------------------------------------------------------


def _gen_event_name(rng: random.Random | None = None) -> str:
    """Produce a ``dumpnow_<8 base-32 chars>`` event name. The
    randomness is RFC4648 base-32 (no padding) so the result is
    pure-ASCII, case-stable, and filesystem-safe.
    """
    r = rng or random.SystemRandom()
    suffix = "".join(
        r.choice(_BASE32_ALPHABET) for _ in range(EVENT_NAME_RANDOM_LEN)
    )
    return f"{EVENT_NAME_PREFIX}{suffix}"


# ---------------------------------------------------------------------------
# Event-specnum derivation
# ---------------------------------------------------------------------------


def _corr_fast_mon_key(chgroup: int) -> str:
    """Mirror :func:`control_store._corr_fast_mon_key` / the canonical
    ``dsart.services.corr_fast_mon.build_corr_fast_mon_key`` so we
    don't take a dsart import here.
    """
    return f"/mon/corr_rt/{int(chgroup)}/corr_fast"


def _search_ring_mon_key(sid: int, gpu_half: int) -> str:
    """Mirror :func:`dsart.services.search_ring_mon.build_search_ring_mon_key`.

    Avoids taking a dsart import here so the dashboard env without
    PYTHONPATH=src can still import this module.
    """
    return f"/mon/search/{int(sid)}/{int(gpu_half)}/ring"


def _resolve_event_specnum_from_search_ring(
    store: Any,
    *,
    targets: Sequence[tuple[int, int]],
    lookback_cubes: int = DEFAULT_LOOKBACK_CUBES,
    max_age_s: float = DEFAULT_MAX_PUBLISHER_AGE_S,
) -> tuple[int | None, dict[str, Any]]:
    """Primary auto-pick path: poll ``/mon/search/<sid>/<g>/ring``
    for every search-half in the fleet, pick
    ``min(newest_event_specnum_start)`` (= the *laggard's* newest
    cube, so every faster half also has it in its ring), and
    subtract ``lookback_cubes × t_det × sample_period_specnum`` to
    land squarely inside every half's [oldest, newest) retention
    window.

    Returns ``(event_specnum, arm_info)``. ``event_specnum`` is
    ``None`` iff no fresh search-ring mon-key answered — the caller
    falls back to the legacy corr_fast path (or returns 503).

    Why ``min(newest)`` and not ``max(newest)`` like the corr_fast
    path? The corr_fast path picks the leader because the search
    ring keys retain *fewer* cubes than corr_fast publishes blocks
    (depth=16 vs hundreds of blocks), so we want to err toward the
    laggard. ``cube_ring_depth=16`` gives every half ~16 × 1/7.45 ≈
    2.1 s of retention, and the publishers refresh every ~1.3 s,
    so a target = ``min_newest - 4 cubes`` lands ~half a second
    behind the leader of every half — comfortably inside everyone's
    window.
    """
    info: dict[str, Any] = {
        "source": "search_ring",
        "min_newest_event_specnum_start": None,
        "min_newest_sid_g": None,
        "newest_t_det": None,
        "newest_sample_period_specnum": None,
        "lookback_cubes": int(lookback_cubes),
        "polled": [],
        "answered": [],
        "missing": [],
        "stale": [],
        "empty": [],
    }
    if not targets:
        return None, info
    now_wall = time.time()
    min_newest: int | None = None
    min_src: tuple[int, int] | None = None
    t_det_ref: int | None = None
    spp_ref: int | None = None
    for sid, g in targets:
        key = _search_ring_mon_key(int(sid), int(g))
        info["polled"].append(key)
        try:
            d = store.get_dict(key)
        except Exception as exc:                              # noqa: BLE001
            LOG.warning(
                "cube_dump_now (search_ring): get_dict(%s) failed: %s",
                key, exc,
            )
            info["missing"].append(key)
            continue
        if not isinstance(d, dict):
            info["missing"].append(key)
            continue
        ts_wall = d.get("ts_wall_unix")
        if isinstance(ts_wall, (int, float)):
            age_s = float(now_wall - float(ts_wall))
            if age_s > max_age_s:
                info["stale"].append(key)
                continue
        n_committed = d.get("n_committed")
        newest = d.get("newest_event_specnum_start")
        # A primed ring publishes newest as an int; an empty ring
        # publishes None as a sentinel. Either way: missing newest
        # means we can't use this half for the min.
        if (
            not isinstance(n_committed, int)
            or n_committed <= 0
            or not isinstance(newest, int)
        ):
            info["empty"].append(key)
            continue
        info["answered"].append(key)
        if min_newest is None or newest < min_newest:
            min_newest = int(newest)
            min_src = (int(sid), int(g))
            td = d.get("t_det")
            sp = d.get("sample_period_specnum")
            if isinstance(td, int) and td > 0:
                t_det_ref = int(td)
            if isinstance(sp, int) and sp > 0:
                spp_ref = int(sp)
    if min_newest is None or t_det_ref is None or spp_ref is None:
        return None, info
    info["min_newest_event_specnum_start"] = min_newest
    info["min_newest_sid_g"] = (
        list(min_src) if min_src is not None else None
    )
    info["newest_t_det"] = t_det_ref
    info["newest_sample_period_specnum"] = spp_ref
    # Subtract a few cubes behind the laggard's newest. At depth=16
    # and lookback=4, the target lands ~0.5 s behind the leader of
    # every half — well inside every [oldest, newest) window.
    cube_span = int(t_det_ref) * int(spp_ref)
    target = min_newest - int(lookback_cubes) * cube_span
    if target < 0:
        target = 0
    return int(target), info


def _resolve_event_specnum(
    store: Any,
    *,
    chgroups: Iterable[int] = tuple(range(16)),
    lookback_blocks: int = DEFAULT_LOOKBACK_BLOCKS,
    max_age_s: float = DEFAULT_MAX_PUBLISHER_AGE_S,
    npackets_per_block: int = NPACKETS_PER_BLOCK,
) -> tuple[int | None, dict[str, Any]]:
    """Walk ``/mon/corr_rt/<cn>/corr_fast`` keys, pick
    ``max(block_specnum_start)``, subtract ``lookback_blocks ×
    npackets_per_block`` to land squarely inside the cube_ring's
    retention window.

    Returns ``(event_specnum, arm_info)``. ``event_specnum`` is
    ``None`` iff no fresh corr_fast publisher answered — the caller
    surfaces that as the 503-equivalent error.

    ``arm_info`` carries the answered / missing / stale lists so the
    operator can see *why* the auto-pick produced what it did — same
    diagnostic shape as :func:`control_store.compute_inject_apply_at`.
    """
    info: dict[str, Any] = {
        "max_block_specnum_start": None,
        "max_block_n": None,
        "max_source": None,
        "lookback_blocks": int(lookback_blocks),
        "npackets_per_block": int(npackets_per_block),
        "polled": [],
        "answered": [],
        "missing": [],
        "stale": [],
    }
    keys = [_corr_fast_mon_key(c) for c in chgroups]
    info["polled"] = list(keys)
    now_wall = time.time()
    max_bss: int | None = None
    max_blk: int | None = None
    max_src: str | None = None
    for k in keys:
        try:
            d = store.get_dict(k)
        except Exception as exc:                              # noqa: BLE001
            LOG.warning(
                "cube_dump_now: get_dict(%s) failed: %s", k, exc,
            )
            info["missing"].append(k)
            continue
        if not isinstance(d, dict):
            info["missing"].append(k)
            continue
        bss = d.get("block_specnum_start")
        if not isinstance(bss, int):
            info["missing"].append(k)
            continue
        ts_wall = d.get("ts_wall_unix")
        if isinstance(ts_wall, (int, float)):
            age_s = float(now_wall - float(ts_wall))
            if age_s > max_age_s:
                info["stale"].append(k)
                continue
        info["answered"].append(k)
        if max_bss is None or bss > max_bss:
            max_bss = int(bss)
            max_src = k
            blk_n = d.get("block_n")
            if isinstance(blk_n, int):
                max_blk = int(blk_n)
    if max_bss is None:
        return None, info
    info["max_block_specnum_start"] = max_bss
    info["max_block_n"] = max_blk
    info["max_source"] = max_src
    # Subtract a few blocks so the synthetic trigger lands a few
    # cubes BEHIND the latest cube_ring slot. find_cube_for_specnum
    # walks newest → oldest and matches the freshest cube whose
    # [start, start + t_det * sample_period_specnum) window contains
    # event_specnum; landing a few blocks behind the leading edge
    # guarantees the leader has finished writing the slot we hit.
    target = max_bss - int(lookback_blocks) * int(npackets_per_block)
    if target < 0:
        target = 0
    return int(target), info


# ---------------------------------------------------------------------------
# UDP sender protocol
# ---------------------------------------------------------------------------


def default_sender(
    data: bytes,
    addr: tuple[str, int],
    timeout_s: float,
) -> None:
    """Send one UDP datagram. Raises :class:`OSError` /
    :class:`socket.timeout` on failure.

    A per-call socket is intentional: UDP sendto is non-blocking,
    socket construction is cheap, and per-call sockets give us a
    clean per-destination timeout without state shared across the
    8 sends. Tests inject a fake to bypass this entirely.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP sendto doesn't really block on the wire, but in the
        # extreme case where the routing layer can't resolve the
        # destination, sendto can raise after a per-OS retry budget.
        # settimeout caps that.
        sock.settimeout(float(timeout_s))
        sock.sendto(data, addr)
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fleet_dump_now(
    store: Any,
    *,
    event_name: str | None = None,
    target_specnum: int | None = None,
    bind_hosts: Sequence[tuple[int, int, str, int]] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sender: Callable[..., None] | None = None,
    yaml_path: str | None = None,
    lookback_blocks: int = DEFAULT_LOOKBACK_BLOCKS,
    chgroups: Iterable[int] = tuple(range(16)),
    rng: random.Random | None = None,
    user: str | None = None,
) -> FleetDumpNowResult:
    """Broadcast a one-shot synthetic ``C2TriggerPacket`` to every
    search-half in the fleet.

    Parameters
    ----------
    store
        Object with a :meth:`get_dict` surface (a
        :class:`control_store.ControlStore` in production, a fake in
        tests). Used only to read ``/mon/corr_rt/<cn>/corr_fast`` for
        the auto-pick of ``event_specnum``. Ignored entirely when
        ``target_specnum`` is supplied by the caller.
    event_name
        If ``None``, auto-generated as ``dumpnow_<8 base-32 chars>``.
        Truncated / padded to the 16-byte C2TriggerPacket field by
        the encoder.
    target_specnum
        If ``None``, derived from
        :func:`_resolve_event_specnum`. If no corr_fast mon-key is
        answering AND no caller-provided ``target_specnum``, the
        result carries an ``error`` string and a non-broadcast
        ``per_half`` (the caller surfaces this as HTTP 503).
    bind_hosts
        If ``None``, derived from :func:`resolve_bind_hosts`. Each
        entry is ``(search_node_id, gpu_half, host, port)`` and gets
        ONE datagram in this fan-out. Pass ``[]`` to short-circuit
        the broadcast (used by tests for the empty-fleet path).
    timeout_s
        Per-sendto timeout. Default 2 s.
    sender
        Override for the UDP sendto. Tests inject a fake that
        records every (data, addr) tuple. Production uses the
        default :func:`default_sender`.
    yaml_path
        Override for the config-yaml location. Defaults to the
        canonical ``configs/dsart_search_rt.yaml`` next to the repo.
    lookback_blocks
        Number of ``NPACKETS_PER_BLOCK``-sized blocks to subtract
        from the fleet's latest ``block_specnum_start`` to position
        the synthetic trigger inside the cube_ring retention window.
    chgroups
        Which corr chgroups to poll for the auto-pick. Default
        ``range(16)`` matches the production fleet topology.
    rng
        Override for the event-name RNG. Tests pin this for
        reproducibility.
    user
        Optional operator label (best-effort; logged only).

    Returns
    -------
    FleetDumpNowResult
        ``ok`` is True iff every requested half ACKed (``fail_count
        == 0``) AND we actually had a list of destinations to send
        to. ``per_half`` maps ``(sid, gpu_half) -> status`` for every
        attempted destination; ``"ok"`` for success, ``"timeout"`` /
        ``"oserror: ..."`` / ``"exception: ..."`` for failures.
    """
    t0 = time.monotonic()

    # Resolve destinations -------------------------------------------------
    if bind_hosts is None:
        resolved = resolve_bind_hosts(yaml_path=yaml_path)
    else:
        resolved = [
            (int(sid), int(g), str(host), int(port))
            for (sid, g, host, port) in bind_hosts
        ]
    resolved_tuple: tuple[tuple[int, int, str, int], ...] = tuple(resolved)

    # Resolve event_name ---------------------------------------------------
    ev_name = str(event_name) if event_name else _gen_event_name(rng=rng)

    # Resolve event_specnum -----------------------------------------------
    arm_info: dict[str, Any] | None = None
    if target_specnum is None:
        # PRIMARY path (M7.4 Phase 6c): poll the per-half
        # search-ring mon-keys, which speak the *search-side*
        # specnum domain — the same one the cube_ring keys cubes
        # by. This fixes the 2026-05-28 ``too_early`` miss seen
        # when the dump trigger used corr_fast's service-start
        # epoch.
        sr_targets = [
            (int(sid), int(g)) for (sid, g, _, _) in resolved_tuple
        ]
        sr_derived, sr_info = _resolve_event_specnum_from_search_ring(
            store, targets=sr_targets,
            lookback_cubes=DEFAULT_LOOKBACK_CUBES,
        )
        if sr_derived is not None:
            ev_specnum = int(sr_derived)
            arm_info = {**sr_info, "source_used": "search_ring"}
        else:
            # FALLBACK path: legacy corr_fast block_specnum_start.
            # Kept around so a half-deployed fleet (search_ring_mon
            # not yet rolled out) still has *some* auto-arm signal
            # rather than refusing to dump. The note + arm_info
            # surface "fell back to corr_fast" so the operator
            # sees what happened.
            cf_derived, cf_info = _resolve_event_specnum(
                store, chgroups=chgroups,
                lookback_blocks=lookback_blocks,
            )
            cf_info_full = dict(cf_info)
            cf_info_full["search_ring_attempt"] = sr_info
            cf_info_full["source_used"] = (
                "corr_fast" if cf_derived is not None else None
            )
            if cf_derived is None:
                elapsed_ms = round((time.monotonic() - t0) * 1e3, 3)
                LOG.warning(
                    "cube_dump_now: refusing to broadcast (no "
                    "search-ring AND no corr_fast mon-keys "
                    "answering); arm_info=%s", cf_info_full,
                )
                return FleetDumpNowResult(
                    ok=False,
                    event_name=ev_name,
                    event_specnum=None,
                    per_half={},
                    pass_count=0,
                    fail_count=0,
                    latency_ms=elapsed_ms,
                    bind_hosts=resolved_tuple,
                    arm_info=cf_info_full,
                    error="no corr_fast mon-keys; is the fleet up?",
                    note=(
                        f"polled={len(cf_info_full['polled'])} "
                        f"answered={len(cf_info_full['answered'])} "
                        f"missing={len(cf_info_full['missing'])} "
                        f"stale={len(cf_info_full['stale'])} "
                        f"search_ring_answered="
                        f"{len(sr_info['answered'])}"
                    ),
                )
            LOG.warning(
                "cube_dump_now: search-ring mon-keys unavailable "
                "(answered=%d missing=%d stale=%d empty=%d); falling "
                "back to corr_fast block_specnum_start. NOTE: this "
                "uses the wrong specnum domain on M7.4 search and "
                "will likely miss; redeploy search_compute with "
                "SearchRingMonPublisher.",
                len(sr_info["answered"]), len(sr_info["missing"]),
                len(sr_info["stale"]), len(sr_info["empty"]),
            )
            ev_specnum = int(cf_derived)
            arm_info = cf_info_full
    else:
        ev_specnum = int(target_specnum)

    # Build the 64-byte packet (single blob, broadcast to N) --------------
    blob = _encode_packet(ev_name, ev_specnum)

    # Empty bind list = nothing to do. Return a non-error result that
    # still surfaces the choice so the operator can see what went
    # wrong upstream (yaml missing / no hosts configured).
    if not resolved_tuple:
        elapsed_ms = round((time.monotonic() - t0) * 1e3, 3)
        LOG.warning(
            "cube_dump_now: no bind destinations resolved; nothing sent",
        )
        return FleetDumpNowResult(
            ok=True,
            event_name=ev_name,
            event_specnum=ev_specnum,
            per_half={},
            pass_count=0,
            fail_count=0,
            latency_ms=elapsed_ms,
            bind_hosts=(),
            arm_info=arm_info,
            error=None,
            note="no bind destinations resolved (empty fleet)",
        )

    # Fan-out -------------------------------------------------------------
    snd = sender or default_sender
    per_half: dict[tuple[int, int], str] = {}
    pass_n = 0
    fail_n = 0
    for sid, g, host, port in resolved_tuple:
        key = (int(sid), int(g))
        try:
            snd(blob, (str(host), int(port)), float(timeout_s))
        except socket.timeout as exc:
            per_half[key] = f"timeout: {exc!r}"
            fail_n += 1
            LOG.warning(
                "cube_dump_now: sendto (sid=%d g=%d) %s:%d timed out: %r",
                sid, g, host, port, exc,
            )
        except TimeoutError as exc:
            per_half[key] = f"timeout: {exc!r}"
            fail_n += 1
            LOG.warning(
                "cube_dump_now: sendto (sid=%d g=%d) %s:%d timed out: %r",
                sid, g, host, port, exc,
            )
        except OSError as exc:
            per_half[key] = f"oserror: {exc!r}"
            fail_n += 1
            LOG.warning(
                "cube_dump_now: sendto (sid=%d g=%d) %s:%d failed: %r",
                sid, g, host, port, exc,
            )
        except Exception as exc:                              # noqa: BLE001
            per_half[key] = f"exception: {exc!r}"
            fail_n += 1
            LOG.exception(
                "cube_dump_now: sendto (sid=%d g=%d) %s:%d unexpected error",
                sid, g, host, port,
            )
        else:
            per_half[key] = "ok"
            pass_n += 1

    elapsed_ms = round((time.monotonic() - t0) * 1e3, 3)
    LOG.info(
        "cube_dump_now: event=%s specnum=%d sent=%d ok=%d fail=%d "
        "user=%s elapsed_ms=%.3f",
        ev_name, ev_specnum, len(resolved_tuple), pass_n, fail_n,
        user, elapsed_ms,
    )
    return FleetDumpNowResult(
        ok=(fail_n == 0),
        event_name=ev_name,
        event_specnum=ev_specnum,
        per_half=per_half,
        pass_count=pass_n,
        fail_count=fail_n,
        latency_ms=elapsed_ms,
        bind_hosts=resolved_tuple,
        arm_info=arm_info,
    )


# ---------------------------------------------------------------------------
# Packet encoding — lazy dsart import so the dashboard env can still
# import this module to surface a clean error message.
# ---------------------------------------------------------------------------


def _encode_packet(event_name: str, event_specnum: int) -> bytes:
    """Encode the 64-byte ``C2TriggerPacket`` blob.

    The encoder lives in :mod:`dsart.coinc.wire`. We import it lazily
    so a dashboard env where ``dsart`` is not on ``PYTHONPATH`` still
    imports this module cleanly — the caller gets a :class:`RuntimeError`
    only when actually trying to dump.
    """
    try:
        from dsart.coinc.wire import (                       # local import
            C2TriggerPacket,
            encode_c2_trigger,
        )
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(
            "cube_dump_now: cannot import dsart.coinc.wire "
            "(is PYTHONPATH=src set?); "
            f"original: {exc!r}"
        ) from exc
    pkt = C2TriggerPacket(
        event_name=str(event_name),
        event_specnum=int(event_specnum),
        # mjd_target is informational on the search side (the listener
        # keys off event_specnum). 0.0 is a sentinel that won't be
        # confused with a real on-sky candidate.
        mjd_target=0.0,
    )
    return encode_c2_trigger(pkt)


__all__ = [
    "DEFAULT_LOOKBACK_BLOCKS",
    "DEFAULT_LOOKBACK_CUBES",
    "DEFAULT_MAX_PUBLISHER_AGE_S",
    "DEFAULT_PORT_BASE",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_YAML_PATH",
    "EVENT_NAME_PREFIX",
    "EVENT_NAME_RANDOM_LEN",
    "GPU_HALVES",
    "FleetDumpNowResult",
    "NPACKETS_PER_BLOCK",
    "default_sender",
    "fleet_dump_now",
    "resolve_bind_hosts",
]
