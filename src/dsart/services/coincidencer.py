"""C2 (h23 coincidencer) service entry.

Top-level async orchestrator that wires the
:mod:`dsart.coinc` library into one long-running service. Lifted into
``dsart.services.coincidencer`` so systemd's ExecStart can target it
as ``python -u -m dsart.services.coincidencer``.

Lifecycle (matches docs/c1c2/C1C2_DESIGN.md §3):

  1. Parse CLI args, load YAML.
  2. Bind the TCP receiver on ``coinc.bind`` (default 0.0.0.0:11500).
  3. Start two RollingCsvWriter instances (C1 per-row, C2 per-cluster).
  4. Spawn the PlotWorker (ThreadPoolExecutor).
  5. Per accepted C1 batch: push to TimeWindow + CoincidenceGraph,
     evaluate each touched component, on ``dump_all_gpus``:
       - allocate a name (EventNameAllocator);
       - append the hourly C2 CSV row (unconditional audit trail);
       - UDP-broadcast the trigger to the 8 C1 listeners;
       - wait for the cube set. ONLY once all
         ``plot_expected_cube_count`` cubes have landed (2026-06-09):
         create the per-event archive directory, write
         Level2/C2_<name>.csv + Level2/C1_window_<name>.csv +
         Level3/<name>.json, and dispatch the plot job. If the
         deadline (``plot_cube_wait_s``) expires with an incomplete
         set, the event is discarded and any partial cubes/ dir is
         deleted — dumps disabled / rate-capped / under-delivered
         events never leave a directory in candidates/.
  6. Hourly housekeeping: CSV rotation + retention enforcement +
     stale-pending-plot reaper.
  7. SIGHUP → ``CriteriaEvaluator.force_reload``.
  8. SIGTERM / SIGINT → graceful shutdown.
  9. Mon-points export to etcd at ``/mon/c2/h23`` every 5 s.

The service is single-threaded asyncio (plus the plotter
ThreadPoolExecutor); per-connection work serialises through one loop.

CLI mirrors ``search_compute.py`` for operator muscle memory:

  python -u -m dsart.services.coincidencer
      --config /path/to/dsart_search_rt.yaml
      --criteria /path/to/c2_trigger_criteria.yaml
      [--log-level INFO]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import yaml

from ..coinc import wire
from ..coinc.archive import (
    C1_WINDOW_CSV_FIELDS,
    C2_CLUSTER_CSV_FIELDS,
    EventArchiveWriter,
    entry_to_csv_row,
    stats_to_csv_row,
    stats_to_l3_metadata,
)
from ..coinc.broadcast import TriggerBroadcaster
from ..coinc.components import CoincidenceGraph
from ..coinc.criteria import CriteriaEvaluator
from ..coinc.csv_rotator import RollingCsvWriter, concat_recent_hourly
from ..coinc.inject_match import (
    DEFAULT_REGISTRY_REFRESH_S as INJECT_REGISTRY_REFRESH_S,
    InjectionMatcher,
)
from ..coinc.names import EventNameAllocator
from ..coinc.plotter import PlotWorker, enqueue_event
from ..coinc.receiver import C1BatchReceiver
from ..coinc.stats import ClusterStats, compute_stats
from ..coinc.window import TimeWindow, WindowEntry

__all__ = [
    "CoincidencerConfig",
    "CoincidencerService",
    "DumpsGate",
    "DUMPS_ENABLED_KEY",
    "main",
]


_LOG = logging.getLogger("dsart.services.coincidencer")


# ---------------------------------------------------------------------------
# Dumps-enabled etcd gate (M7.4 Phase 6c)
# ---------------------------------------------------------------------------
#
# Operator-controlled runtime kill-switch for UDP dump triggers. The
# dashboard's Control tab writes ``/cmd/c2/dumps_enabled`` as
# ``{"enabled": bool, "ts": float, "actor": str, "reason": str}``; the
# coincidencer consults this key on every classification through the
# small in-process cache implemented below. When ``enabled=False`` C2
# still runs the criteria + writes the per-event archive rows, but
# skips the UDP fan-out into ``broadcast.broadcast_triggers`` AND does
# NOT advance the per-class holdoff timer — so flipping dumps back on
# makes the next match fire immediately rather than being eaten by a
# stale holdoff.
#
# Default (missing key) is fail-OPEN: ``enabled=True``. Any etcd error
# (outage, dropped connection, malformed payload) is also treated as
# enabled — we never want a transient etcd hiccup to silently suppress
# dumps in production. Errors are logged at WARNING rate-limited to
# 1/min so the operator sees the problem in the dashboard without
# flooding the journal.
#
# See ``docs/c1c2/C1C2_DESIGN.md`` §6c for the operational contract.
# ---------------------------------------------------------------------------


#: etcd key the dashboard writes; the coincidencer polls.
DUMPS_ENABLED_KEY: str = "/cmd/c2/dumps_enabled"

#: Default cache TTL: 200 ms. Bigger than a single batch's eval time
#: (microseconds) so back-to-back classifications share the same etcd
#: read, but small enough that an operator flip is honoured within
#: ~one ``_housekeep_loop`` tick (~1 Hz).
DEFAULT_DUMPS_CACHE_TTL_S: float = 0.2


class DumpsGate:
    """Polled etcd cache for the dump-enabled runtime gate.

    Construction is cheap and side-effect free; the first call to
    :meth:`enabled` reads etcd. Subsequent calls within ``cache_ttl_s``
    serve the cached answer.

    All failure modes are fail-OPEN (returns True + logs WARNING on
    the first failure of a window). The coincidencer never sees an
    exception out of :meth:`enabled`.

    Parameters
    ----------
    store:
        A duck-typed DsaStore-like object exposing ``get_dict(key)``.
        ``None`` is allowed (e.g. the dsa_store import failed on
        boot); :meth:`enabled` then permanently returns True.
    key:
        etcd key to poll; defaults to :data:`DUMPS_ENABLED_KEY`.
    cache_ttl_s:
        Maximum age of the cached value before the next
        :meth:`enabled` triggers a re-read.
    now:
        Monotonic clock function; injected by tests to drive the TTL
        deterministically.
    warn_rate_limit_s:
        Minimum interval between WARNING logs on repeated etcd failure
        (default 60 s).
    """

    def __init__(
        self,
        store: Optional[Any],
        *,
        key: str = DUMPS_ENABLED_KEY,
        cache_ttl_s: float = DEFAULT_DUMPS_CACHE_TTL_S,
        now: Optional[Callable[[], float]] = None,
        warn_rate_limit_s: float = 60.0,
    ) -> None:
        self._store = store
        self._key = str(key)
        self._ttl = float(cache_ttl_s)
        self._now = now if now is not None else time.monotonic
        self._warn_rate_limit_s = float(warn_rate_limit_s)
        # Start fail-OPEN so the very first eval (before the first
        # refresh) cannot accidentally suppress.
        self._cached_value: bool = True
        self._cached_at: float = -math.inf
        self._last_warn_at: float = -math.inf
        self._read_count: int = 0
        self._fail_count: int = 0

    @property
    def key(self) -> str:
        return self._key

    @property
    def cache_ttl_s(self) -> float:
        return self._ttl

    @property
    def read_count(self) -> int:
        return self._read_count

    @property
    def fail_count(self) -> int:
        return self._fail_count

    def enabled(self) -> bool:
        """Return the current dump-enabled state (cached)."""
        now = self._now()
        if (now - self._cached_at) < self._ttl:
            return self._cached_value
        self._refresh(now=now)
        return self._cached_value

    def invalidate(self) -> None:
        """Force the next :meth:`enabled` call to re-read etcd."""
        self._cached_at = -math.inf

    def _refresh(self, *, now: float) -> None:
        self._read_count += 1
        if self._store is None:
            self._cached_value = True
            self._cached_at = now
            return
        try:
            doc = self._store.get_dict(self._key)
        except Exception as exc:  # noqa: BLE001
            self._fail_count += 1
            self._cached_value = True              # fail-OPEN
            self._cached_at = now
            if (now - self._last_warn_at) > self._warn_rate_limit_s:
                _LOG.warning(
                    "dumps_gate: etcd read of %s failed (%s); "
                    "fail-OPEN (dumps stay enabled)",
                    self._key, exc,
                )
                self._last_warn_at = now
            return
        if isinstance(doc, Mapping) and "enabled" in doc:
            self._cached_value = bool(doc["enabled"])
        else:
            # Missing / malformed payload — fail-OPEN.
            self._cached_value = True
        self._cached_at = now


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoincidencerConfig:
    bind_host: str = "0.0.0.0"
    bind_port: int = 11500

    window_s: float = 5.0
    csv_retention_hours: int = 48
    csv_dir_c1: Path = Path("/dataz/dsa110/operations/C1/cluster_output")
    csv_dir_c2: Path = Path("/dataz/dsa110/operations/C2/cluster_output")
    #: M7.6: write the per-candidate C1 hiplot CSV. Default off — the C1
    #: row volume is large and rarely useful (every received candidate,
    #: including late arrivals). C2 cluster rows + the rolling 24h C2 view
    #: are the operator-facing hiplot surfaces.
    write_c1_csv: bool = False
    #: M7.6: hours of C2 cluster history to expose as a single rolling
    #: ``c2_last24h.csv`` (concatenation of the recent hourly C2 CSVs) so
    #: the C2 hiplot always shows ~the last day in one experiment. 0
    #: disables the rolling file.
    c2_rolling_window_hours: int = 24
    event_archive_root: Path = Path("/dataz/dsa110/candidates")
    trigger_criteria_path: Path = Path(
        "/home/ubuntu/vikram/dev/dsa110-rt/configs/c2_trigger_criteria.yaml"
    )

    dump_broadcast_port_base: int = 11227
    dump_broadcast_hosts: Mapping[int, str] = field(default_factory=dict)

    plotter_n_workers: int = 2
    plotter_per_event_timeout_s: float = 30.0
    plotter_use_process_pool: bool = True
    """2026-05-30 resilience: render plots in a separate PROCESS, not a
    thread. Plotting touches ~855 MB cubes + matplotlib (both GIL-heavy);
    in a ThreadPoolExecutor that work serialises against the single
    asyncio event loop in THIS process and can starve the C1 receiver —
    the observed failure mode where a dump storm (one noisy high-DM half
    matching ``bright_pulsar`` every 5 s) wedged the loop for ~34 min and
    flatlined C1->C2. A ProcessPoolExecutor isolates that CPU/IO so the
    receiver always drains its sockets. Set false to fall back to threads
    (e.g. in restricted sandboxes where fork/spawn is unavailable)."""

    # 2026-05-30 dump-rate cap: bound how often a dump-class trigger can
    # fire a full 8-GPU broadcast, regardless of candidate volume, so a
    # single persistently-noisy source cannot drive a cube-dump storm
    # (the proximate cause of the C2 stall). 0 disables the cap.
    dump_rate_max_per_window: int = 6
    dump_rate_window_s: float = 60.0

    # Event-name allocator config.
    etcd_lastname_key: str = "/mon/corr/1/trigger"
    event_pkg_path: Optional[Path] = Path(
        "/home/ubuntu/proj/dsa110-shell/dsa110-event"
    )
    name_allocator_offline: bool = False

    # Plot dispatcher (how long to wait for cubes before plotting).
    plot_cube_wait_s: float = 60.0
    plot_dispatch_poll_s: float = 5.0
    plot_expected_cube_count: int = 8

    # Mon-points export.
    mon_etcd_key: str = "/mon/c2/h23"
    mon_publish_interval_s: float = 5.0

    # Galactic-DM discriminant (added 2026-05-27).
    # The C2 service polls /mon/array/gal_dm (written by
    # declination.service on h23 from NE2001) every
    # gal_dm_poll_interval_s seconds and threads the value through
    # compute_stats so trigger classes can gate on the
    # ``dm_galactic_fraction_*`` predicates. Set
    # ``gal_dm_max_los_override`` to a positive number to pin the
    # value (useful for offline replay tests or when the etcd
    # pointing path is dark).
    gal_dm_etcd_key: str = "/mon/array/gal_dm"
    gal_dm_poll_interval_s: float = 30.0
    gal_dm_max_los_override: Optional[float] = None
    gal_dm_max_age_s: float = 600.0

    # Startup trigger-grace window (M7.4 Phase 8c, 2026-05-29). For this
    # many seconds after the FIRST C1 batch arrives, C2 evaluates
    # coincidences as usual but SUPPRESSES the trigger action (no UDP
    # dump fan-out, no log_only record, no plot) and counts the would-be
    # triggers in ``triggers_startup_grace``. Rationale: the corr-side
    # RFI bandpass-outlier excisor is bypassed during its cold-start
    # warmup (``RFI_BANDPASS_WARMUP_CUBES_DEFAULT ≈ 1118 cubes ≈ 5·τ_B ≈
    # 150 s`` at the canonical τ_B=30 s). During that window persistent
    # narrowband RFI leaks into the gridded cubes and the search detector
    # fires on it — but the warmup state is collapsed to a boolean
    # validity mask at the search RX boundary, so those candidates reach
    # C2 UNFLAGGED (NOISE_WARMUP only covers the search-side Layer-2 σ_k
    # burn-in, ~4 s). This produced the observed ~3-minute post-restart
    # burst (437 rows / 150 false log_only triggers in the first 3 min,
    # then ~0/min steady state, with rows_warmup_drop=0). The grace
    # window is keyed on the first CANDIDATE-bearing batch (not C2 start,
    # and explicitly NOT the continuous 0-row heartbeats idle halves now
    # send) so it tracks the fleet's synchronized utc_start RFI warmup,
    # and is set ONCE (never re-armed on reconnects). Default 180 s >
    # the ~150 s RFI warmup. Set to 0 to disable.
    startup_grace_s: float = 180.0

    @classmethod
    def from_yaml(cls, path: Path, *, override: Optional[Mapping[str, Any]] = None
                 ) -> "CoincidencerConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        coinc = doc.get("coinc", {}) or {}
        if override:
            coinc = {**coinc, **override}
        bind = coinc.get("bind", {}) or {}
        dump = coinc.get("dump_broadcast", {}) or {}
        plotter = coinc.get("plotter", {}) or {}
        hosts_raw = dump.get("hosts", {}) or {}
        hosts: Dict[int, str] = {int(k): str(v) for k, v in hosts_raw.items()}
        return cls(
            bind_host=str(bind.get("host", "0.0.0.0")),
            bind_port=int(bind.get("port", 11500)),
            window_s=float(coinc.get("window_s", 5.0)),
            csv_retention_hours=int(coinc.get("csv_retention_hours", 48)),
            csv_dir_c1=Path(coinc.get(
                "csv_dir_c1",
                "/dataz/dsa110/operations/C1/cluster_output",
            )),
            csv_dir_c2=Path(coinc.get(
                "csv_dir_c2",
                "/dataz/dsa110/operations/C2/cluster_output",
            )),
            write_c1_csv=bool(coinc.get("write_c1_csv", False)),
            c2_rolling_window_hours=int(
                coinc.get("c2_rolling_window_hours", 24)
            ),
            event_archive_root=Path(coinc.get(
                "event_archive_root", "/dataz/dsa110/candidates",
            )),
            trigger_criteria_path=Path(coinc.get(
                "trigger_criteria_path",
                "/home/ubuntu/vikram/dev/dsa110-rt/configs/c2_trigger_criteria.yaml",
            )),
            dump_broadcast_port_base=int(dump.get("port_base", 11227)),
            dump_broadcast_hosts=hosts,
            plotter_n_workers=int(plotter.get("n_workers", 2)),
            plotter_per_event_timeout_s=float(
                plotter.get("per_event_timeout_s", 30.0),
            ),
            plotter_use_process_pool=bool(
                plotter.get("use_process_pool", True),
            ),
            dump_rate_max_per_window=int(
                coinc.get("dump_rate_max_per_window", 6),
            ),
            dump_rate_window_s=float(
                coinc.get("dump_rate_window_s", 60.0),
            ),
            etcd_lastname_key=str(coinc.get(
                "etcd_lastname_key", "/mon/corr/1/trigger",
            )),
            event_pkg_path=(
                Path(coinc["event_pkg_path"])
                if coinc.get("event_pkg_path") else None
            ),
            name_allocator_offline=bool(
                coinc.get("name_allocator_offline", False),
            ),
            plot_cube_wait_s=float(coinc.get("plot_cube_wait_s", 60.0)),
            plot_dispatch_poll_s=float(
                coinc.get("plot_dispatch_poll_s", 5.0),
            ),
            plot_expected_cube_count=int(
                coinc.get("plot_expected_cube_count", 8),
            ),
            mon_etcd_key=str(coinc.get("mon_etcd_key", "/mon/c2/h23")),
            mon_publish_interval_s=float(
                coinc.get("mon_publish_interval_s", 5.0),
            ),
            gal_dm_etcd_key=str(coinc.get(
                "gal_dm_etcd_key", "/mon/array/gal_dm",
            )),
            gal_dm_poll_interval_s=float(
                coinc.get("gal_dm_poll_interval_s", 30.0),
            ),
            gal_dm_max_los_override=(
                float(coinc["gal_dm_max_los_override"])
                if coinc.get("gal_dm_max_los_override") is not None
                else None
            ),
            gal_dm_max_age_s=float(
                coinc.get("gal_dm_max_age_s", 600.0),
            ),
            startup_grace_s=float(coinc.get("startup_grace_s", 180.0)),
        )


# ---------------------------------------------------------------------------
# Mon-points etcd wrapper (mockable, exactly like other dsart services)
# ---------------------------------------------------------------------------


class _StoreWrapper:
    """Thin DsaStore wrapper; falls back to a no-op when etcd is gone."""

    def __init__(self, mock: Optional[Any] = None) -> None:
        if mock is not None:
            self._store = mock
            self._available = True
            return
        try:
            from dsautils.dsa_store import DsaStore  # noqa: WPS433
            self._store = DsaStore()
            self._available = True
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "DsaStore unavailable (%s); mon-points export disabled",
                exc,
            )
            self._store = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._store is not None

    def put_dict(self, key: str, value: Mapping[str, Any]) -> None:
        if not self._available or self._store is None:
            return
        try:
            self._store.put_dict(key, dict(value))
        except Exception:  # noqa: BLE001
            _LOG.exception("etcd put_dict(%s) failed", key)

    def get_dict(self, key: str) -> Optional[Mapping[str, Any]]:
        """Best-effort etcd read; returns None on outage / missing key.

        Used by the galactic-DM poll loop (which expects to fail
        silently when /mon/array/gal_dm hasn't been written yet on a
        cold boot).
        """
        if not self._available or self._store is None:
            return None
        try:
            return self._store.get_dict(key)
        except Exception:  # noqa: BLE001
            _LOG.exception("etcd get_dict(%s) failed", key)
            return None

    def get_etcd(self) -> Any:
        """Passthrough to the underlying ``DsaStore.get_etcd()`` so
        consumers needing the raw ``etcd3`` client (e.g.
        :class:`dsart.coinc.inject_match.InjectionMatcher` walking the
        ``/cnf/inject/active/`` prefix) can use it.

        Raises ``RuntimeError`` when etcd is unavailable so the caller
        can fall back to a degraded mode (the matcher counts this as
        a refresh failure).
        """
        if not self._available or self._store is None:
            raise RuntimeError("etcd unavailable")
        return self._store.get_etcd()


# ---------------------------------------------------------------------------
# Pending plot bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _PendingPlot:
    """One in-flight ``dump_all_gpus`` event awaiting its cube set.

    2026-06-09: this now carries everything needed to write the
    per-event archive directory AFTER the full cube set lands, so
    ``/dataz/dsa110/candidates/<name>/`` is only ever created for
    events whose cubes all arrived (no dirs at all when dumping is
    off / rate-capped / the fleet under-delivers).
    """

    event_name: str
    submitted_at_monotonic: float
    stats: ClusterStats
    members: Tuple[WindowEntry, ...]
    trigger_class_name: str = ""
    trigger_action: str = ""
    trigger_holdoff_s: float = 0.0
    #: id(member) → inj_id labels, computed at fire time (the inject
    #: registry is time-sensitive; matching later would miss).
    member_inj_ids: Dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CoincidencerService:
    """C2 orchestrator. Construct, then call :meth:`run`."""

    #: Seconds after an incomplete-event discard before the partial
    #: dir is checked (and removed) once more — covers straggler cube
    #: rsyncs that land just after the deadline cleanup.
    _DISCARD_RESWEEP_S: float = 300.0

    def __init__(
        self,
        config: CoincidencerConfig,
        *,
        mon_store: Optional[Any] = None,
        name_allocator: Optional[EventNameAllocator] = None,
        broadcaster: Optional[TriggerBroadcaster] = None,
        dumps_gate: Optional[DumpsGate] = None,
        inject_matcher: Optional[InjectionMatcher] = None,
    ) -> None:
        self._config = config

        self._window = TimeWindow(window_s=config.window_s)
        self._graph = CoincidenceGraph()
        self._criteria = CriteriaEvaluator(config.trigger_criteria_path)

        # M7.6: the per-candidate C1 hiplot CSV is opt-in (large + rarely
        # useful). Default off — see ``CoincidencerConfig.write_c1_csv``.
        self._c1_csv: Optional[RollingCsvWriter] = (
            RollingCsvWriter(
                config.csv_dir_c1, "c1", C1_WINDOW_CSV_FIELDS,
                retention_hours=config.csv_retention_hours,
            )
            if config.write_c1_csv
            else None
        )
        self._c2_csv = RollingCsvWriter(
            config.csv_dir_c2, "c2", C2_CLUSTER_CSV_FIELDS,
            retention_hours=config.csv_retention_hours,
        )

        self._archive = EventArchiveWriter(config.event_archive_root)

        self._broadcaster = broadcaster or TriggerBroadcaster(
            config.dump_broadcast_hosts,
            port_base=config.dump_broadcast_port_base,
        )

        self._allocator = name_allocator or EventNameAllocator(
            etcd_key=config.etcd_lastname_key,
            event_pkg_path=config.event_pkg_path,
            offline=config.name_allocator_offline,
        )

        self._plot_worker = PlotWorker(
            max_workers=config.plotter_n_workers,
            per_event_timeout_s=config.plotter_per_event_timeout_s,
            use_process_pool=config.plotter_use_process_pool,
        )
        self._pending_plots: Dict[str, _PendingPlot] = {}
        # Names discarded for an incomplete cube set, re-swept once
        # more after _DISCARD_RESWEEP_S in case a straggler rsync
        # re-created the partial dir after the deadline cleanup.
        self._discarded_events: Dict[str, float] = {}
        # 2026-05-30 dump-rate cap: monotonic timestamps of recent dump
        # broadcasts, trimmed to the rolling window on each check.
        self._dump_fire_times: List[float] = []

        self._mon_store = _StoreWrapper(mock=mon_store)

        # M7.4 Phase 6c: operator-controlled dump kill-switch. Wired
        # against the same DsaStore wrapper that drives the mon-points
        # export so a single etcd outage has a single failure mode
        # (the wrapper's get_dict() returns None on outage; the gate
        # then fail-OPENs by design).
        self._dumps_gate: DumpsGate = (
            dumps_gate if dumps_gate is not None
            else DumpsGate(self._mon_store)
        )

        # M7.4 Phase 6c.A: C1-row ↔ active-injection matcher. The
        # matcher reads /cnf/inject/active/<inj_id> from etcd via the
        # same DsaStore the rest of the service uses; on a row that
        # matches an active injection it (a) publishes the running
        # best to /mon/dsart/inject/matches/<inj_id> for the
        # dashboard's bootstrap-SNR calibration to consume, and (b)
        # stamps the C1 CSV's inj_id column so injection candidates
        # are visibly labelled.
        # DIAGNOSTIC (2026-05-31): allow widening the match tolerances
        # via env so an operator can discover WHERE an injection images
        # (and whether it images at all) without redeploying. Defaults
        # fall back to the production constants in inject_match.
        _lm_tol_env = os.environ.get("DSART_INJECT_LM_TOL_RAD")
        _dm_tol_env = os.environ.get("DSART_INJECT_DM_TOL_FRAC")
        _max_snr_env = os.environ.get("DSART_INJECT_MAX_OBSERVED_SNR")
        _min_snr_env = os.environ.get("DSART_INJECT_MIN_OBSERVED_SNR")
        _matcher_kwargs: dict = {
            "store": self._mon_store,
            "refresh_s": INJECT_REGISTRY_REFRESH_S,
        }
        if _lm_tol_env:
            _matcher_kwargs["lm_tol_rad"] = float(_lm_tol_env)
        if _dm_tol_env:
            _matcher_kwargs["dm_tol_frac"] = float(_dm_tol_env)
        if _max_snr_env:
            _matcher_kwargs["max_observed_snr"] = float(_max_snr_env)
        if _min_snr_env:
            _matcher_kwargs["min_observed_snr"] = float(_min_snr_env)
        self._inject_matcher: InjectionMatcher = (
            inject_matcher if inject_matcher is not None
            else InjectionMatcher(**_matcher_kwargs)
        )

        self._receiver = C1BatchReceiver(
            host=config.bind_host,
            port=config.bind_port,
            on_batch=self._on_batch,
        )

        # Bookkeeping / mon-points.
        self._counters: Dict[str, int] = {
            "rows_in": 0,
            "rows_late_drop": 0,
            # M7.4 Phase 8 (2026-05-28): rows dropped because the
            # search-side Layer-2 σ_k EMA was still in burn-in
            # (CandidateFlags.NOISE_WARMUP). Exposed in the mon-dict
            # for Grafana so the operator can see the warmup-filter
            # activity during the first ~30 cubes after each search
            # restart and after the first ~50 s of every cold start
            # (slow ramp from 0.96 -> 7.55 cubes/s in search_compute
            # gives ~30 cubes spread over ~30 s).
            "rows_warmup_drop": 0,
            "components_evaluated": 0,
            "triggers_dump": 0,
            "triggers_suppressed": 0,
            # 2026-05-30: dump broadcasts skipped by the rolling-window
            # dump-rate cap (CoincidencerConfig.dump_rate_max_per_window).
            # A persistent rise here means a noisy source is hammering a
            # dump class -- check the C1 DM-smearing-floor filter / criteria.
            "dumps_rate_capped": 0,
            # M7.4 Phase 8c (2026-05-29): would-be triggers suppressed
            # because they fired inside the startup grace window (corr
            # RFI bandpass warmup leaks RFI -> false candidates that
            # reach C2 unflagged). See CoincidencerConfig.startup_grace_s.
            "triggers_startup_grace": 0,
            "triggers_log_only": 0,
            "broadcast_send_ok": 0,
            "broadcast_send_fail": 0,
            "plots_dispatched": 0,
            # 2026-06-09: per-event archive dirs are written only when
            # the FULL cube set lands. events_archived counts dirs
            # materialised; events_incomplete_discarded counts dump
            # events dropped (and partial cubes deleted) because fewer
            # than plot_expected_cube_count cubes arrived in
            # plot_cube_wait_s.
            "events_archived": 0,
            "events_incomplete_discarded": 0,
            "csv_rotations": 0,
            "csv_removed": 0,
        }
        self._last_event_name: Optional[str] = None
        self._last_trigger_class: Optional[str] = None
        self._last_event_mjd: Optional[float] = None
        # Monotonic time of the first C1 batch ever received; anchors the
        # startup grace window (see CoincidencerConfig.startup_grace_s).
        # None until the first batch arrives.
        self._first_batch_mono: Optional[float] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._tasks: List[asyncio.Task] = []
        self._started_unix: float = 0.0

        # Galactic-DM discriminant cache (refreshed by
        # _gal_dm_poll_loop). ``_gal_dm_value_pc_cc`` is None until the
        # first successful poll OR an operator override is in place;
        # ``_gal_dm_fetched_at_mono`` is used to age out a stale value
        # (declination.service hiccup → gal_dm_max_age_s elapsed → fall
        # back to None and emit nan into ClusterStats).
        if config.gal_dm_max_los_override is not None:
            self._gal_dm_value_pc_cc: Optional[float] = float(
                config.gal_dm_max_los_override,
            )
            self._gal_dm_fetched_at_mono: Optional[float] = float("inf")
        else:
            self._gal_dm_value_pc_cc = None
            self._gal_dm_fetched_at_mono = None
        self._gal_dm_polls_ok = 0
        self._gal_dm_polls_fail = 0

    # ----- public API ---------------------------------------------------

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._install_signal_handlers(loop)
        self._started_unix = time.time()

        # 2026-06-10: pre-expire the startup grace window when the corr
        # fleet has already been streaming longer than the grace. The
        # grace exists to mask the corr-side RFI bandpass warmup after a
        # FLEET start — but it was keyed on "first batch after C2
        # start", so a C2-only restart (code deploy, plotter bounce)
        # re-armed it against a long-warm fleet and silently suppressed
        # the first 180 s of genuine triggers (observed 2026-06-10
        # 05:09: operator injections produced WOULD-FIRE/suppressed).
        self._maybe_pre_expire_startup_grace()

        await self._receiver.start()
        self._tasks.append(
            asyncio.create_task(self._receiver.serve_forever(),
                                name="c2-receiver"),
        )
        self._tasks.append(
            asyncio.create_task(self._housekeep_loop(),
                                name="c2-housekeep"),
        )
        self._tasks.append(
            asyncio.create_task(self._plot_dispatcher_loop(),
                                name="c2-plot-dispatcher"),
        )
        self._tasks.append(
            asyncio.create_task(self._mon_publish_loop(),
                                name="c2-mon-publish"),
        )
        # Galactic-DM poll loop: refreshes /mon/array/gal_dm so each
        # cluster is tagged with dm_galactic_fraction for the
        # gal/extragal C2 discriminant. No-op when the operator pinned
        # an override.
        if self._config.gal_dm_max_los_override is None:
            self._tasks.append(
                asyncio.create_task(self._gal_dm_poll_loop(),
                                    name="c2-gal-dm-poll"),
            )

        _LOG.info(
            "coincidencer up: bind=%s:%d window=%.1fs criteria=%s "
            "archive_root=%s csv_dir_c1=%s csv_dir_c2=%s",
            self._config.bind_host, self._config.bind_port,
            self._config.window_s, self._config.trigger_criteria_path,
            self._config.event_archive_root,
            self._config.csv_dir_c1, self._config.csv_dir_c2,
        )

        rc = 0
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()
        return rc

    async def stop(self) -> None:
        _LOG.info("coincidencer shutdown begin")
        for t in self._tasks:
            t.cancel()
        await self._receiver.stop()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        try:
            self._broadcaster.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._plot_worker.shutdown(wait=True)
        except Exception:  # noqa: BLE001
            pass
        _LOG.info("coincidencer shutdown done")

    # ----- batch handler ------------------------------------------------

    async def _on_batch(self, batch: wire.C1Batch, peer_repr: str) -> None:
        # M7.4 Phase 6c.A: refresh the active-injection registry on
        # every batch (heartbeat or non-empty); the matcher's
        # refresh_if_due gates on its own cadence so this is a cheap
        # no-op when the cadence hasn't elapsed.
        self._inject_matcher.refresh_if_due()
        if batch.header.n_candidates == 0:
            # Empty heartbeat; nothing to do for the graph but record.
            # NOTE: heartbeats are emitted continuously by idle C1 halves
            # (including before utc_start), so they must NOT anchor the
            # startup grace window — otherwise the window would open at
            # connect time and expire before the post-utc_start RFI-warmup
            # candidate burst it is meant to absorb.
            return
        # Anchor the startup grace window on the first CANDIDATE-bearing
        # batch we ever see. Keying on first-candidate (rather than C2 start
        # or first heartbeat) tracks the fleet's synchronized utc_start RFI
        # warmup even if C2 was launched well before the search nodes. Set
        # exactly once.
        if self._first_batch_mono is None:
            self._first_batch_mono = time.monotonic()

        # M7.4 Phase 8 (2026-05-28): drop noise-warmup-flagged candidate
        # rows before they hit the window/graph. The search-side
        # Layer-2 σ_k EMA is in burn-in for the first N_burnin cubes
        # (default 30 cubes = ~4 s at production cadence); during this
        # window σ_k is computed by a Welford running mean that has not
        # yet absorbed enough samples to be a reliable estimator. The
        # detector sets ``flags & NOISE_WARMUP`` on every Candidate
        # emitted while ``Layer2State.is_warming_up`` is True (see
        # ``detector/forward.py``); we filter on that bit here so the
        # coincidencer's window count, graph membership, MJD-key gate,
        # triggers_log_only counter, and per-event archives all reflect
        # ONLY post-burn-in (statistically valid) candidates. Without
        # this filter the cold-start burst at 2026-05-28T21:29:30Z
        # produced 407 rows_in over 50 s (peak 107 rows/s vs ~0/s
        # steady state) and 25 false log_only triggers/s — exactly
        # matching the L2 burn-in (30 cubes) × the slow-start
        # (~0.96 cubes/s during the first ~30 cubes) = ~30 s of
        # rolling false-positive rate per search half. Dropped rows
        # are counted in self._counters["rows_warmup_drop"] for the
        # /mon/c2 monitor surface so the operator can see warmup
        # filtering activity. The full row list is still streamed
        # into the rolling C1 CSV further down (faithful wire log of
        # everything received); only the trigger pipeline skips them.
        # Per the wire schema the bit set is
        # ``CandidateFlags.NOISE_WARMUP = 1 << 3 = 8``.
        warmup_mask = 1 << 3  # CandidateFlags.NOISE_WARMUP -- see contracts.py
        all_candidates = tuple(batch.candidates)
        candidates_post_warmup = tuple(
            r for r in all_candidates if (int(r.flags) & warmup_mask) == 0
        )
        warmup_dropped = len(all_candidates) - len(candidates_post_warmup)
        if warmup_dropped:
            self._counters["rows_warmup_drop"] = (
                self._counters.get("rows_warmup_drop", 0) + warmup_dropped
            )
        if not candidates_post_warmup:
            # All rows were warmup-flagged; nothing to graph. Still
            # bump rows_in by zero (no-op) and skip the rest of the
            # heavy graph/trigger path.
            return

        new_entries = self._window.add(batch.header, candidates_post_warmup)
        aged = self._window.aged_out()
        # Split aged into "really aged" entries (were already in the window
        # at the start of this batch) and "late arrivals" (rows that came in
        # with mjd already below the window cutoff and were inserted then
        # immediately popped by the same age-out pass). Without this split
        # we'd graph.add the late arrivals (since they appear in new_entries)
        # which makes the components graph diverge from the window: the
        # window correctly drops them but the graph would carry them
        # forever, producing the graph_size >> window_size leak observed
        # in /mon/c2/h23. The leak is triggered any time the search side
        # restarts with mjd_at_specnum_0=0 (so a fresh session's batches
        # arrive with mjd values older than the C2 window's anchor) — but
        # the contract belongs here regardless: graph membership must
        # mirror window membership.
        new_id_to_entry = {id(e): e for e in new_entries}
        late_arrivals_ids = {id(e) for e in aged if id(e) in new_id_to_entry}
        really_aged = [e for e in aged if id(e) not in new_id_to_entry]
        if really_aged:
            self._graph.remove_many(really_aged)
        survivors = [e for e in new_entries if id(e) not in late_arrivals_ids]
        for e in survivors:
            self._graph.add(e)
        self._counters["rows_in"] += len(new_entries)
        if late_arrivals_ids:
            self._counters["rows_late_drop"] = (
                self._counters.get("rows_late_drop", 0)
                + len(late_arrivals_ids)
            )

        # Hot-reload criteria if the file mtime changed under us.
        self._criteria.reload_if_changed()

        # M7.4 Phase 6c.A: run each new C1 row through the matcher
        # against the active-injection registry (already refreshed at
        # batch entry). Match hits update the per-id best, publish to
        # /mon/dsart/inject/matches/<id>, and produce an ``inj_id``
        # label we stamp into both the rolling hiplot CSV and (later,
        # on a dump_all_gpus trigger) the per-event C1 window CSV.
        inj_id_by_entry_id: Dict[int, str] = {}
        for e in new_entries:
            label = self._inject_matcher.try_match(
                snr=e.snr,
                dm_pc_cc=e.dm_pc_cc,
                l_rad=e.l_rad,
                m_rad=e.m_rad,
                width_samples=e.width_samples,
                event_specnum=e.event_specnum,
                kernel_id=e.kernel_id,
                search_node_id=e.search_node_id,
                gpu_half=e.gpu_half,
                mjd=e.mjd,
            )
            if label is not None:
                inj_id_by_entry_id[id(e)] = label.inj_id

        # Write the per-row C1 hiplot CSV for *every* received candidate
        # (independent of triggering) — including late arrivals, so the
        # csv is a faithful record of what hit the wire even when those
        # rows didn't make it into the in-window graph.
        now_utc = datetime.now(timezone.utc)
        if self._c1_csv is not None:
            for e in new_entries:
                self._c1_csv.append_row(
                    entry_to_csv_row(
                        e, trigger="",
                        inj_id=inj_id_by_entry_id.get(id(e), ""),
                    ),
                    now_utc=now_utc,
                )

        # Walk only the components touched by this batch. ``survivors`` are
        # the new_entries that actually made it into the graph; passing
        # late arrivals here would be harmless (components_touched skips
        # ids not in the graph) but ``survivors`` is the clearer contract.
        gal_dm = self._current_gal_dm_max_los()
        touched = self._graph.components_touched(survivors)
        for comp_id in touched:
            members = self._graph.component_members(comp_id)
            if not members:
                continue
            stats = compute_stats(members, gal_dm_max_los=gal_dm)
            self._counters["components_evaluated"] += 1
            # Snapshot per-class holdoff state BEFORE evaluate (which
            # mutates ``_last_fired_at[name] = now`` on every match).
            # If the resulting trigger is suppressed by the dumps gate
            # we restore that one entry so the next eval still sees a
            # cold (or pre-suppression) holdoff and fires immediately
            # when the operator flips dumps back on.
            holdoff_snapshot = self._criteria.last_fired_at_snapshot()
            tc = self._criteria.evaluate(stats)
            if tc is None:
                continue
            prev_holdoff = holdoff_snapshot.get(tc.name)
            await self._fire(
                stats, members, tc, prev_holdoff=prev_holdoff,
            )

    def _in_startup_grace(self) -> bool:
        """True while inside the post-first-batch startup grace window
        (see CoincidencerConfig.startup_grace_s). Disabled when
        startup_grace_s <= 0 or before the first batch arrives."""
        grace = self._config.startup_grace_s
        if grace <= 0 or self._first_batch_mono is None:
            return False
        return (time.monotonic() - self._first_batch_mono) < grace

    def _maybe_pre_expire_startup_grace(self) -> None:
        """Disarm the startup grace when the corr fleet is already warm.

        The grace window masks candidates produced while the corr-side
        RFI bandpass excisor is in cold-start warmup (~150 s after a
        fleet utc_start). That warmup is a property of the CORR fleet,
        not of C2 — so when C2 alone restarts mid-run we must not
        suppress trigger actions for another 180 s. We read the
        fleet's ``/mon/corr_rt/<cn>/corr_fast`` heartbeat (published
        every ~2 s): ``block_n × NPACKETS_PER_BLOCK × 65.536 µs`` is
        the corr service uptime. If the freshest publisher has been up
        longer than the grace, pre-expire the window by back-dating
        the first-batch anchor. Best-effort: any etcd problem leaves
        the conservative (armed) behaviour in place.
        """
        grace = float(self._config.startup_grace_s)
        if grace <= 0 or not self._mon_store.available:
            return
        specnum_seconds = 65.536e-6
        now_wall = time.time()
        best_uptime_s: Optional[float] = None
        for cn in range(16):
            doc = self._mon_store.get_dict(f"/mon/corr_rt/{cn}/corr_fast")
            if not isinstance(doc, Mapping):
                continue
            ts_wall = doc.get("ts_wall_unix")
            if not isinstance(ts_wall, (int, float)):
                continue
            # Freshness window: the publisher PUTs every ~2.1 s, so a
            # LIVE fleet is never more than a few seconds stale. Keep
            # this tight (≪ the restart_all pkill→C2-restart gap):
            # with a generous 30 s window the first deploy of this fix
            # mis-classified a FLEET restart as "fleet warm" because
            # C2 came up within 30 s of the old corr processes dying,
            # and their last mon-points still carried hours of uptime
            # (observed 2026-06-10 06:37: warmup-junk dump 260610ejnr
            # fired on the very first post-utc_start batch).
            if (now_wall - float(ts_wall)) > 8.0:
                continue  # stale publisher from a previous run
            bss = doc.get("block_specnum_start")
            if not isinstance(bss, int):
                continue
            uptime_s = float(bss) * specnum_seconds
            if best_uptime_s is None or uptime_s > best_uptime_s:
                best_uptime_s = uptime_s
        if best_uptime_s is not None and best_uptime_s > grace:
            self._first_batch_mono = time.monotonic() - grace
            _LOG.info(
                "startup grace pre-expired: corr fleet has been "
                "streaming for ~%.0f s (> grace %.0f s) — C2-only "
                "restart detected, triggers will NOT be suppressed",
                best_uptime_s, grace,
            )
        elif best_uptime_s is not None:
            _LOG.info(
                "startup grace stays armed: corr fleet uptime ~%.0f s "
                "<= grace %.0f s (fresh fleet start)",
                best_uptime_s, grace,
            )

    def _dump_rate_exceeded(self) -> bool:
        """True if firing another dump broadcast now would exceed the
        configured rolling-window cap. Trims the timestamp buffer to the
        window as a side effect. A cap of ``<= 0`` disables (always
        False)."""
        cap = self._config.dump_rate_max_per_window
        if cap <= 0:
            return False
        window = float(self._config.dump_rate_window_s)
        now = time.monotonic()
        self._dump_fire_times = [
            t for t in self._dump_fire_times if (now - t) < window
        ]
        return len(self._dump_fire_times) >= cap

    def _record_dump_fire(self) -> None:
        """Record a dump broadcast against the rate-cap window."""
        self._dump_fire_times.append(time.monotonic())

    async def _fire(
        self,
        stats: ClusterStats,
        members: List[WindowEntry],
        trigger_class,
        *,
        prev_holdoff: Optional[float] = None,
    ) -> None:
        """Dispatch on the trigger-class action.

        ``prev_holdoff`` is the per-class ``last_fired_at`` snapshot
        captured before :meth:`CriteriaEvaluator.evaluate` advanced it.
        When ``action == "dump_all_gpus"`` and
        :attr:`_dumps_gate` reports ``enabled=False`` we use it to
        roll the timer back to its pre-evaluate value, so the very
        next match after the operator flips dumps on fires
        immediately rather than being eaten by a stale holdoff.
        """
        action = trigger_class.action
        now_utc = datetime.now(timezone.utc)

        # M7.4 Phase 8c: startup grace window. During the corr-side RFI
        # bandpass warmup (~150 s) un-excised RFI leaks into the cubes and
        # the search detector fires on it, producing a burst of false
        # coincident triggers that reach C2 UNFLAGGED. Suppress the
        # trigger ACTION here (no dump fan-out / log_only record / plot /
        # event-dir), count it, and roll back the per-class holdoff so the
        # first genuine trigger after the window fires immediately.
        if self._in_startup_grace():
            self._counters["triggers_startup_grace"] += 1
            if prev_holdoff is not None:
                self._criteria.restore_last_fired_at(
                    trigger_class.name, prev_holdoff,
                )
            elapsed = (
                time.monotonic() - self._first_batch_mono
                if self._first_batch_mono is not None else 0.0
            )
            _LOG.info(
                "WOULD-FIRE class=%s action=%s n=%d snr_max=%.2f "
                "dm_med=%.2f suppressed=startup_grace (%.0fs/%.0fs)",
                trigger_class.name, action, stats.n_events,
                stats.snr_max, stats.dm_median,
                elapsed, self._config.startup_grace_s,
            )
            return

        if action == "dump_all_gpus":
            dumps_enabled = self._dumps_gate.enabled()
            event_name = self._allocator.allocate(stats.t_peak_mjd)
            self._last_event_name = event_name
            self._last_trigger_class = trigger_class.name
            self._last_event_mjd = stats.t_peak_mjd

            # The hourly C2 CSV (csv_dir_c2, OUTSIDE the candidates
            # tree) is the unconditional audit trail. The per-event
            # directory under /dataz/dsa110/candidates is written ONLY
            # once the full cube set has landed (2026-06-09 operator
            # request): suppressed / rate-capped / under-delivered
            # events never leave a directory behind.
            self._c2_csv.append_row(
                stats_to_csv_row(
                    stats, trigger_class=trigger_class.name,
                    trigger=event_name,
                ),
                now_utc=now_utc,
            )

            if not dumps_enabled:
                # Suppressed path: NO UDP fan-out, NO event dir, NO
                # plot scheduling (there will be no cubes), and roll
                # back the per-class holdoff so the next genuine
                # trigger after dumps re-enable fires immediately.
                self._counters["triggers_suppressed"] += 1
                self._criteria.restore_last_fired_at(
                    trigger_class.name, prev_holdoff,
                )
                _LOG.info(
                    "WOULD-DUMP class=%s name=%s n=%d snr_max=%.2f "
                    "dm_med=%.2f suppressed=dumps_disabled",
                    trigger_class.name, event_name, stats.n_events,
                    stats.snr_max, stats.dm_median,
                )
                return

            # 2026-05-30 dump-rate cap: even with dumps enabled, bound how
            # many full-fleet dump broadcasts fire per rolling window so a
            # persistently-noisy source cannot drive a cube-dump storm (the
            # proximate cause of the C2 stall). The hourly C2 CSV row above
            # is still written; we skip the UDP fan-out + plot scheduling
            # (and hence the event dir — no cubes, no dir).
            # Holdoff stays advanced so normal per-class backoff applies.
            if self._dump_rate_exceeded():
                self._counters["dumps_rate_capped"] += 1
                _LOG.warning(
                    "DUMP RATE-CAPPED class=%s name=%s n=%d snr_max=%.2f "
                    "dm_med=%.2f (>%d dumps / %.0fs) -- skipping broadcast",
                    trigger_class.name, event_name, stats.n_events,
                    stats.snr_max, stats.dm_median,
                    self._config.dump_rate_max_per_window,
                    self._config.dump_rate_window_s,
                )
                return
            self._record_dump_fire()

            result = self._broadcaster.broadcast(
                event_name=event_name,
                event_specnum=stats.peak_event_specnum,
                mjd_target=stats.t_peak_mjd,
                trigger_class_id=hash(trigger_class.name) & 0xFFFF,
            )
            ok = sum(1 for v in result.values() if v)
            fail = len(result) - ok
            self._counters["broadcast_send_ok"] += ok
            self._counters["broadcast_send_fail"] += fail
            self._counters["triggers_dump"] += 1

            # M7.4 Phase 6c.A: tag injection-matching members NOW (the
            # inject registry is time-sensitive; matching at archive
            # time would miss). matched_inj_id is side-effect-free and
            # walks the same registry the batch-time try_match consults.
            member_inj_ids: Dict[int, str] = {}
            for m in members:
                label = self._inject_matcher.matched_inj_id(
                    dm_pc_cc=m.dm_pc_cc,
                    l_rad=m.l_rad,
                    m_rad=m.m_rad,
                    width_samples=m.width_samples,
                    snr=m.snr,
                    event_specnum=m.event_specnum,
                    mjd=m.mjd,
                )
                if label is not None:
                    member_inj_ids[id(m)] = label

            # Schedule the event; the dispatcher loop watches for the
            # FULL cube set to arrive (or the deadline). The per-event
            # archive directory + CSVs + L3 metadata are written there,
            # only on a complete cube set.
            self._pending_plots[event_name] = _PendingPlot(
                event_name=event_name,
                submitted_at_monotonic=time.monotonic(),
                stats=stats,
                members=tuple(members),
                trigger_class_name=trigger_class.name,
                trigger_action=trigger_class.action,
                trigger_holdoff_s=trigger_class.holdoff_s,
                member_inj_ids=member_inj_ids,
            )
            _LOG.info(
                "DUMP class=%s name=%s n=%d snr_max=%.2f dm_med=%.2f "
                "broadcast=%d/%d",
                trigger_class.name, event_name, stats.n_events,
                stats.snr_max, stats.dm_median, ok, len(result),
            )
        elif action == "log_only":
            self._c2_csv.append_row(
                stats_to_csv_row(
                    stats, trigger_class=trigger_class.name,
                    trigger="",
                ),
                now_utc=now_utc,
            )
            self._counters["triggers_log_only"] += 1
            _LOG.info(
                "LOG class=%s n=%d snr_max=%.2f dm_med=%.2f",
                trigger_class.name, stats.n_events, stats.snr_max,
                stats.dm_median,
            )
        else:
            _LOG.warning(
                "unknown trigger action %r for class %s; skipping",
                action, trigger_class.name,
            )

    # ----- background loops --------------------------------------------

    async def _housekeep_loop(self) -> None:
        """Hourly CSV rotation check + retention enforcement."""
        try:
            while True:
                await asyncio.sleep(60.0)
                now_utc = datetime.now(timezone.utc)
                rotated = int(self._c2_csv.maybe_rotate(now_utc))
                removed = self._c2_csv.housekeep(now_utc)
                if self._c1_csv is not None:
                    rotated += int(self._c1_csv.maybe_rotate(now_utc))
                    removed += self._c1_csv.housekeep(now_utc)
                self._counters["csv_rotations"] += rotated
                self._counters["csv_removed"] += removed
                # M7.6: refresh the rolling ~24h C2 view (single file the
                # C2 hiplot can pin) by concatenating the recent hourly
                # C2 CSVs. Best-effort; never let it sink the loop.
                win = int(self._config.c2_rolling_window_hours)
                if win > 0:
                    try:
                        concat_recent_hourly(
                            self._config.csv_dir_c2, "c2",
                            now_utc=now_utc, window_hours=win,
                            out_name="c2_last24h.csv",
                        )
                    except Exception:                          # noqa: BLE001
                        _LOG.warning(
                            "c2 rolling-24h concat failed", exc_info=True,
                        )
        except asyncio.CancelledError:
            return

    def _write_event_archive(self, pp: _PendingPlot) -> Path:
        """Materialise ``<archive_root>/<event_name>/`` (dir tree +
        Level2 CSVs + Level3 metadata) for a completed cube set.

        Deferred from fire time (2026-06-09) so the directory only
        ever exists for events whose full cube set landed. The
        ``cubes/`` subdir typically already exists — the search-node
        uploaders rsync straight into it — and ``create()`` is
        idempotent over it.
        """
        name = pp.event_name
        ev_dir = self._archive.create(name)
        self._archive.write_c2_cluster_csv(
            ev_dir, name, pp.stats,
            trigger_class=pp.trigger_class_name, trigger=name,
        )
        self._archive.write_c1_window_csv(
            ev_dir, name, list(pp.members), trigger=name,
            inj_ids=pp.member_inj_ids,
        )
        self._archive.write_l3_metadata(
            ev_dir, name,
            stats_to_l3_metadata(
                event_name=name,
                stats=pp.stats,
                trigger_class_name=pp.trigger_class_name,
                trigger_action=pp.trigger_action,
                holdoff_s=pp.trigger_holdoff_s,
            ),
        )
        return ev_dir

    def _discard_partial_event_dir(self, name: str) -> None:
        """Remove a partial ``<archive_root>/<name>/`` left behind by
        cube rsyncs for an event that never completed. Validated +
        best-effort; never raises."""
        if not name or "/" in name or name.startswith("."):
            return
        ev_dir = self._config.event_archive_root / name
        if not ev_dir.is_dir():
            return
        try:
            shutil.rmtree(ev_dir)
            _LOG.info("removed partial event dir %s", ev_dir)
        except OSError as exc:
            _LOG.warning("could not remove partial dir %s: %s", ev_dir, exc)

    async def _plot_dispatcher_loop(self) -> None:
        """Periodically scan pending events.

        * Full cube set arrived → write the per-event archive dir +
          CSVs + L3 metadata, then dispatch the plot job.
        * Deadline expired with an incomplete set → discard: no
          archive dir is written and any partial ``cubes/`` rsynced
          by the search nodes is deleted (2026-06-09: a burst dir in
          /dataz/dsa110/candidates exists iff ALL cubes were dumped).

        A straggler rsync can re-create a partial dir moments after
        the deadline discard; ``_discarded_events`` re-sweeps each
        discarded name once more after ``_DISCARD_RESWEEP_S``.
        """
        try:
            while True:
                await asyncio.sleep(self._config.plot_dispatch_poll_s)
                self._scan_pending_events(time.monotonic())
        except asyncio.CancelledError:
            return

    def _scan_pending_events(self, now_mono: float) -> None:
        """One dispatcher pass (factored out of the loop for tests)."""
        # Re-sweep straggler partial dirs from earlier discards.
        for name, t_disc in list(self._discarded_events.items()):
            if now_mono - t_disc >= self._DISCARD_RESWEEP_S:
                self._discard_partial_event_dir(name)
                del self._discarded_events[name]

        if not self._pending_plots:
            return
        ready: List[str] = []
        expired: List[Tuple[str, int]] = []
        for name, pp in list(self._pending_plots.items()):
            cubes_dir = (
                self._config.event_archive_root / name / "cubes"
            )
            n_cubes = 0
            if cubes_dir.is_dir():
                n_cubes = sum(
                    1 for p in cubes_dir.glob("cube_s*_g*_*.npz")
                    if p.is_file()
                )
            age = now_mono - pp.submitted_at_monotonic
            if n_cubes >= self._config.plot_expected_cube_count:
                ready.append(name)
            elif age >= self._config.plot_cube_wait_s:
                expired.append((name, n_cubes))
        for name in ready:
            pp = self._pending_plots.pop(name)
            try:
                self._write_event_archive(pp)
                self._counters["events_archived"] += 1
            except Exception:                              # noqa: BLE001
                _LOG.exception(
                    "event archive write failed for %s "
                    "(plot still dispatched)", name,
                )
            enqueue_event(
                self._plot_worker, name,
                self._config.event_archive_root,
                stats=pp.stats, members=list(pp.members),
            )
            self._counters["plots_dispatched"] += 1
            _LOG.info(
                "plot dispatched for %s (waited %.1fs)",
                name, now_mono - pp.submitted_at_monotonic,
            )
        for name, n_cubes in expired:
            self._pending_plots.pop(name, None)
            self._counters["events_incomplete_discarded"] += 1
            _LOG.warning(
                "DISCARD %s: %d/%d cubes after %.0fs -- no "
                "event dir written; partial cubes removed",
                name, n_cubes,
                self._config.plot_expected_cube_count,
                self._config.plot_cube_wait_s,
            )
            self._discard_partial_event_dir(name)
            self._discarded_events[name] = now_mono

    async def _mon_publish_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._config.mon_publish_interval_s)
                gal_dm = self._current_gal_dm_max_los()
                payload: Dict[str, Any] = {
                    "ts_unix": time.time(),
                    "uptime_s": time.time() - self._started_unix,
                    "window_size": len(self._window),
                    "graph_size": len(self._graph),
                    "pending_plots": len(self._pending_plots),
                    "last_event_name": self._last_event_name,
                    "last_trigger_class": self._last_trigger_class,
                    "last_event_mjd": self._last_event_mjd,
                    "receiver": self._receiver.counters.snapshot(),
                    "counters": dict(self._counters),
                    "gal_dm_max_los_pc_cc": (
                        float(gal_dm) if gal_dm is not None else None
                    ),
                    "gal_dm_polls_ok": self._gal_dm_polls_ok,
                    "gal_dm_polls_fail": self._gal_dm_polls_fail,
                    "dumps_enabled": bool(self._dumps_gate.enabled()),
                    "dumps_gate_reads": int(self._dumps_gate.read_count),
                    "dumps_gate_fails": int(self._dumps_gate.fail_count),
                    "inject_match": self._inject_matcher.snapshot(),
                }
                self._mon_store.put_dict(
                    self._config.mon_etcd_key, payload,
                )
        except asyncio.CancelledError:
            return

    # ----- galactic-DM poller ------------------------------------------

    def _current_gal_dm_max_los(self) -> Optional[float]:
        """Return the freshest gal_dm_max_los value (pc/cc), or None.

        Falls back to None when the cached poll value is older than
        ``gal_dm_max_age_s``; in that case ClusterStats records nan and
        criteria predicates that gate on the fraction don't match.
        Operator-pinned overrides bypass the age check entirely.
        """
        if self._config.gal_dm_max_los_override is not None:
            return float(self._config.gal_dm_max_los_override)
        if (
            self._gal_dm_value_pc_cc is None
            or self._gal_dm_fetched_at_mono is None
        ):
            return None
        age = time.monotonic() - self._gal_dm_fetched_at_mono
        if age > self._config.gal_dm_max_age_s:
            return None
        return self._gal_dm_value_pc_cc

    async def _gal_dm_poll_loop(self) -> None:
        """Refresh ``/mon/array/gal_dm`` periodically.

        On every successful poll, replaces the in-memory value. Failures
        (etcd outage, missing key, malformed payload) just bump the
        ``gal_dm_polls_fail`` counter — the existing cached value is
        kept until it ages out via ``_current_gal_dm_max_los``. The
        very first poll runs immediately so the operator can confirm
        the wiring on startup.
        """
        if not self._mon_store.available:
            _LOG.warning(
                "gal_dm poll loop: DsaStore unavailable; "
                "dm_galactic_fraction will stay nan",
            )
            return
        key = self._config.gal_dm_etcd_key
        interval = float(self._config.gal_dm_poll_interval_s)
        try:
            while True:
                doc = self._mon_store.get_dict(key)
                ok = False
                if doc is not None and isinstance(doc, Mapping):
                    val = doc.get("gal_dm")
                    try:
                        v = float(val) if val is not None else None
                    except (TypeError, ValueError):
                        v = None
                    if v is not None and math.isfinite(v) and v > 0.0:
                        self._gal_dm_value_pc_cc = v
                        self._gal_dm_fetched_at_mono = time.monotonic()
                        self._gal_dm_polls_ok += 1
                        ok = True
                if not ok:
                    self._gal_dm_polls_fail += 1
                    _LOG.debug(
                        "gal_dm poll: bad/missing value at %s (doc=%r)",
                        key, doc,
                    )
                # First poll is logged at INFO so the operator sees the
                # wiring on startup; subsequent polls are DEBUG.
                if self._gal_dm_polls_ok == 1:
                    _LOG.info(
                        "gal_dm: first successful poll, value=%.3f pc/cc",
                        self._gal_dm_value_pc_cc,
                    )
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    # ----- signals ------------------------------------------------------

    def _install_signal_handlers(
        self, loop: asyncio.AbstractEventLoop,
    ) -> None:
        def _on_term() -> None:
            _LOG.info("SIGTERM/SIGINT received; stopping")
            if self._stop_event is not None:
                self._stop_event.set()

        def _on_hup() -> None:
            _LOG.info("SIGHUP received; reloading criteria")
            try:
                self._criteria.force_reload()
            except Exception:  # noqa: BLE001
                _LOG.exception("criteria reload failed")

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_term)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            loop.add_signal_handler(signal.SIGHUP, _on_hup)
        except (NotImplementedError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=Path(
            "/home/ubuntu/vikram/dev/dsa110-rt/configs/dsart_search_rt.yaml",
        ),
        help="YAML config; reads the 'coinc:' top-level block.",
    )
    p.add_argument(
        "--criteria", type=Path, default=None,
        help="Optional override for coinc.trigger_criteria_path.",
    )
    p.add_argument(
        "--bind-host", type=str, default=None,
        help="Override coinc.bind.host.",
    )
    p.add_argument(
        "--bind-port", type=int, default=None,
        help="Override coinc.bind.port.",
    )
    p.add_argument(
        "--archive-root", type=Path, default=None,
        help="Override coinc.event_archive_root.",
    )
    p.add_argument(
        "--name-allocator-offline", action="store_true",
        help="Force the FallbackAllocator (no etcd / event.names).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


async def _run_async(args: argparse.Namespace) -> int:
    override: Dict[str, Any] = {}
    if args.bind_host or args.bind_port:
        bind = {}
        if args.bind_host:
            bind["host"] = args.bind_host
        if args.bind_port:
            bind["port"] = args.bind_port
        override["bind"] = bind
    if args.archive_root is not None:
        override["event_archive_root"] = str(args.archive_root)
    if args.criteria is not None:
        override["trigger_criteria_path"] = str(args.criteria)
    if args.name_allocator_offline:
        override["name_allocator_offline"] = True

    cfg = CoincidencerConfig.from_yaml(args.config, override=override)
    svc = CoincidencerService(cfg)
    return await svc.run()


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
