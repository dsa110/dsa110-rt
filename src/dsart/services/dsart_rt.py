"""dsart-rt control-plane orchestrator (M7 chunk 0).

One Python process per corr or search node. Reads its pipeline config
from etcd (``/cnf/pipeline_rt`` for corr instances, ``/cnf/search_rt``
for search instances), watches ``/cmd/corr_rt/<n>`` for operator verbs
(``start`` / ``stop`` / ``utc_start`` / ``utc_stop``), fork-execs the
configured worker processes, and publishes a per-node heartbeat +
monitoring dict to ``/mon/corr_rt/<n>`` (or ``/mon/search_rt/<n>``).

This is the dsart-rt analog of the legacy ``corr.py`` (see
``REALTIME_FRB_SEARCH.md`` §8) — same systemd-managed singleton-per-node
shape, same etcd verb surface, but it talks to the dsart-rt key
namespace so the new control plane can run side-by-side with the
legacy ``corr.py`` without contending for the same etcd keys.

The orchestrator is intentionally state-light: it does NOT cache
pipeline output, NOT read the data stream, NOT touch PSRDADA buffers
other than via ``dada_db`` create/destroy on ``start`` / ``stop``. Its
job is process lifecycle + heartbeat. Per-routine telemetry beyond
"alive y/n" is gathered by the routine's own logging — dsart_rt just
parrots ``dada_dbmetric`` snapshots for the configured buffers into
the mon-dict so an operator polling etcd sees a single point-in-time
truth.

CLI::

    python -m dsart.services.dsart_rt -in pipeline_rt -cn 6 \\
                                       [--config-key /cnf/pipeline_rt] \\
                                       [--mon-cadence-s 2.0] \\
                                       [--namespace-prefix corr_rt]

Verbs (JSON ``{"cmd": "<verb>", "val": <any>}`` posted to the cmd key):

* ``start`` (val: optional float = observing dec in degrees, for
  ``CUSTOMDEC`` substitution in routine args) — create buffers, fork
  routines.
* ``stop`` — kill routines, destroy buffers.
* ``utc_start`` (val: int = first specnum to record) — UDP-poke each
  capture process on ``127.0.0.1:11223`` and ``:11224`` with the
  legacy ``UTC_START-<seq>`` string. Persisted to ``/mon/snap/1/utc_start``.
* ``utc_stop`` (val: int) — same shape, ``UTC_STOP-<seq>``.

Heartbeat is written every ``--mon-cadence-s`` seconds to
``/mon/service/<namespace>/<n>``. Full mon-dict (routine PIDs + buffer
dada_dbmetric stats + uptime) goes to ``/mon/<namespace>/<n>`` on the
same cadence.

M7 stagedown:

* M7.0 (this module): ``start`` and ``stop`` work end-to-end with the
  ``captures.mode: junkdb`` routine set. ``utc_*`` verbs implemented
  (no-op UDP-poke; sends the legacy string format). Other legacy
  verbs (``record`` / ``trigger`` / ``ctrltrigger``) accepted-and-
  logged but not acted on — they get wired up in M7.6 once we need
  on-sky triggers.
* M7.1 — wire the real ``dsaX_merge`` + ``corr_slow_compute`` +
  ``corr_fast_integration`` routines in ``dsart_pipeline_rt.yaml``.
* M7.2 — wire the real ``search_rx`` + ``search_compute`` routines
  in ``dsart_search_rt.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from astropy.time import Time
    _HAVE_ASTROPY = True
except ImportError:  # pragma: no cover — dev hosts without astropy
    _HAVE_ASTROPY = False

from dsautils.dsa_store import DsaStore

LOG = logging.getLogger("dsart.services.dsart_rt")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BufferSpec:
    """One PSRDADA buffer to create on ``start`` / destroy on ``stop``.

    Mirrors legacy /cnf/pipeline schema. ``k`` is the 4-char key
    alias (PSRDADA hex key); ``b`` bytes per block; ``n`` num blocks;
    ``c`` cluster id (page-locking); ``r`` num readers (multi-reader
    rings).
    """
    k: str
    b: int
    n: int
    c: int = 1
    r: int = 1

    def dada_db_cmd(self) -> list[str]:
        # Legacy invocation: `dada_db -k KEY -b BYTES -n N -l -p [-c C] [-r R]`.
        # -l = lock pages (avoid swap thrash on big rings).
        # -p = enable page-aligned allocator.
        cmd = ["dada_db", "-k", self.k,
               "-b", str(self.b), "-n", str(self.n),
               "-l", "-p"]
        if self.c:
            cmd += ["-c", str(self.c)]
        if self.r and self.r > 1:
            cmd += ["-r", str(self.r)]
        return cmd

    def dada_db_destroy_cmd(self) -> list[str]:
        return ["dada_db", "-d", "-k", self.k]


@dataclass(frozen=True, slots=True)
class RoutineSpec:
    """One worker subprocess managed by dsart_rt.

    ``cmd`` is the executable + any pre-fixed wrapper (``taskset``,
    ``nice``, …). ``args`` is the remaining argv as a single string
    (legacy convention; we ``shlex.split`` it). ``hostargs`` is an
    optional per-FQDN-prefix arg-string appended at spawn time
    (matches the legacy /cnf/pipeline ``hostargs`` field). ``when``
    is a simple ``"<dotted.path> == <value>"`` predicate against the
    raw config dict — used by the ``captures.mode`` switch (see
    dsart_pipeline_rt.yaml).
    """
    name: str
    cmd: str
    args: str = ""
    hostargs: dict[str, str] = field(default_factory=dict)
    when: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    buffers: tuple[BufferSpec, ...]
    routines: tuple[RoutineSpec, ...]
    raw: dict[str, Any]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        bufs = tuple(
            BufferSpec(
                k=b["k"],
                b=int(b["b"]),
                n=int(b["n"]),
                c=int(b.get("c", 1)),
                r=int(b.get("r", 1)),
            )
            for b in (raw.get("buffers") or [])
        )
        routines = tuple(
            RoutineSpec(
                name=r["name"],
                cmd=r["cmd"],
                args=r.get("args", ""),
                hostargs=dict(r.get("hostargs") or {}),
                when=r.get("when"),
            )
            for r in (raw.get("routines") or [])
        )
        return cls(
            buffers=bufs,
            routines=routines,
            raw=raw,
            schema_version=int(raw.get("schema_version", 1)),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate_when(expr: str, raw_cfg: dict[str, Any]) -> bool:
    """Tiny predicate evaluator for ``RoutineSpec.when``.

    Supports only ``"<dotted.path> == <literal>"`` where literal is a
    bare word (e.g. ``junkdb``, ``real``) or a quoted string. No
    operators beyond ``==``; keep the language small to keep etcd
    YAMLs predictable.
    """
    parts = expr.split("==")
    if len(parts) != 2:
        LOG.warning("unparseable `when` predicate %r; defaulting to False", expr)
        return False
    lhs_path = parts[0].strip()
    rhs = parts[1].strip().strip("'\"")
    node: Any = raw_cfg
    for seg in lhs_path.split("."):
        if not isinstance(node, dict) or seg not in node:
            return False
        node = node[seg]
    return str(node) == rhs


def _mjd_now() -> float:
    if _HAVE_ASTROPY:
        return float(Time.now().mjd)
    # Fallback: UNIX -> MJD via the standard offset (40587.0 = MJD on 1970-01-01).
    return 40587.0 + time.time() / 86400.0


def _dada_dbmetric(key: str) -> dict[str, Any]:
    """Read full / clear / written / read counters for one PSRDADA ring.

    Returns an empty dict on any failure (buffer doesn't exist, binary
    missing, parse fail). Don't let mon-dict publish crash on a
    transient buffer-create race.

    The on-disk dada_dbmetric ships in two output flavours: older PSRDADA
    builds print positional CSV ``<nbufs>,<nfull>,<nclear>,<nclear>,...``,
    newer ones print ``key=value`` tokens. We accept both: parse k=v
    when present, fall back to storing the raw CSV under ``"raw_csv"``.
    Mon-dict consumers (M7.1+ soak validators, web UI) can refine the
    interpretation as needed — the legacy column meanings live in the
    PSRDADA dada_db source (search REALTIME_FRB_SEARCH.md §8 +
    ``ipcio_query`` for the canonical interpretation).
    """
    try:
        proc = subprocess.run(
            ["dada_dbmetric", "-k", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    out = (proc.stdout or "").strip()
    if not out:
        return {}
    counters: dict[str, Any] = {}
    for tok in out.replace(",", " ").split():
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        try:
            counters[k.strip()] = int(v)
        except ValueError:
            continue
    if not counters and "," in out:
        # Positional CSV. Trim trailing newline / whitespace; expose
        # both the raw line and a parsed list for consumer convenience.
        fields = [f.strip() for f in out.splitlines()[0].split(",")]
        counters = {"raw_csv": out.splitlines()[0], "fields": fields}
    return counters


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RtOrchestrator:
    """One-per-node systemd-managed pipeline orchestrator."""

    # Verbs that we accept-and-act-on. The remaining legacy verbs
    # (record / trigger / ctrltrigger / inject) are accepted but
    # currently logged-only — they get wired in M7.6.
    _ACTIVE_VERBS = ("start", "stop", "utc_start", "utc_stop")
    _LOGGED_VERBS = ("record", "trigger", "ctrltrigger", "inject",
                     "reload_cal", "reload_flagants")

    def __init__(
        self,
        *,
        instance: str,
        cn_id: int,
        namespace: str,
        config_key: str,
        cmd_key: str,
        broadcast_key: str,
        mon_key: str,
        heartbeat_key: str,
        mon_cadence_s: float = 2.0,
        utc_udp_ports: tuple[int, int] = (11223, 11224),
    ):
        self.instance = instance
        self.cn_id = cn_id
        self.namespace = namespace
        self.config_key = config_key
        self.cmd_key = cmd_key
        self.broadcast_key = broadcast_key
        self.mon_key = mon_key
        self.heartbeat_key = heartbeat_key
        self.mon_cadence_s = mon_cadence_s
        self.utc_udp_ports = utc_udp_ports
        self.fqdn = socket.gethostname()

        self._store = DsaStore()
        self._config: Optional[PipelineConfig] = None
        self._children: dict[str, subprocess.Popen[bytes]] = {}
        self._state = "stopped"  # "stopped" / "starting" / "running" / "stopping"
        self._start_time: Optional[float] = None
        self._last_verb: Optional[tuple[str, Any, float]] = None
        self._stop_evt = threading.Event()
        self._watch_ids: list[int] = []
        self._lock = threading.RLock()

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> int:
        """Block until SIGINT/SIGTERM. Returns 0 on clean exit."""
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        self._reload_config()
        self._install_watches()

        LOG.info("dsart-rt up: instance=%s cn=%d cmd=%s mon=%s",
                 self.instance, self.cn_id, self.cmd_key, self.mon_key)
        try:
            while not self._stop_evt.is_set():
                self._reap_dead_children()
                try:
                    self._publish_mon()
                except Exception:  # noqa: BLE001
                    LOG.exception("mon publish failed (continuing)")
                self._stop_evt.wait(self.mon_cadence_s)
        finally:
            self._on_shutdown()
        return 0

    def _install_watches(self) -> None:
        self._watch_ids.append(
            self._store.add_watch(self.cmd_key, self._on_command_event)
        )
        if self.broadcast_key and self.broadcast_key != self.cmd_key:
            self._watch_ids.append(
                self._store.add_watch(self.broadcast_key, self._on_command_event)
            )

    def _on_signal(self, signum: int, frame: object) -> None:
        LOG.info("got signal %d; shutting down", signum)
        self._stop_evt.set()

    def _on_shutdown(self) -> None:
        for wid in self._watch_ids:
            try:
                self._store.cancel(wid)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            if self._state == "running":
                LOG.info("auto-stop on shutdown (had active routines)")
                try:
                    self._verb_stop(None)
                except Exception:  # noqa: BLE001
                    LOG.exception("auto-stop on shutdown failed")
        # Final heartbeat update so consumers see we left cleanly.
        try:
            self._publish_heartbeat(extra={"final": True})
        except Exception:  # noqa: BLE001
            pass

    # ---- config -------------------------------------------------------

    def _reload_config(self) -> None:
        raw = self._store.get_dict(self.config_key)
        if not raw:
            raise RuntimeError(
                f"etcd key {self.config_key!r} is empty / unset — "
                f"run tools/ops/push_dsart_to_etcd.py before starting."
            )
        cfg = PipelineConfig.from_dict(raw)
        LOG.info("loaded config: %d buffers, %d routines, schema=%d",
                 len(cfg.buffers), len(cfg.routines), cfg.schema_version)
        self._config = cfg

    # ---- command dispatch --------------------------------------------

    def _on_command_event(self, event: Any) -> None:
        """``dsautils`` add_watch callback shape.

        dsautils gives us either the parsed dict (when parse_func is
        the default JSON parser) or the raw event object. We accept
        both shapes defensively because the dsautils API is a little
        slippery across versions.
        """
        try:
            payload = self._extract_payload(event)
        except Exception:  # noqa: BLE001
            LOG.exception("could not extract verb payload from %r", event)
            return
        if payload is None:
            return
        verb = str(payload.get("cmd", "")).strip()
        val = payload.get("val")
        self._last_verb = (verb, val, time.time())
        if verb in self._ACTIVE_VERBS:
            try:
                getattr(self, f"_verb_{verb}")(val)
            except Exception:  # noqa: BLE001
                LOG.exception("verb %s failed (val=%r)", verb, val)
        elif verb in self._LOGGED_VERBS:
            LOG.info("verb %s accepted-and-logged (val=%r); "
                     "no-op until later M7 chunk", verb, val)
        else:
            LOG.warning("unknown verb %r (payload=%r); ignoring", verb, payload)

    @staticmethod
    def _extract_payload(event: Any) -> Optional[dict[str, Any]]:
        if event is None:
            return None
        if isinstance(event, dict):
            return event
        if isinstance(event, (bytes, str)):
            return json.loads(event)
        # dsautils delivers etcd3.events.PutEvent in some configs; pull `.value`.
        val = getattr(event, "value", None)
        if isinstance(val, (bytes, str)):
            return json.loads(val)
        return None

    # ---- verbs --------------------------------------------------------

    def _verb_start(self, val: Any) -> None:
        with self._lock:
            if self._state == "running":
                LOG.info("verb start: already running; ignoring")
                return
            self._state = "starting"
            self._reload_config()
            assert self._config is not None
            self._create_buffers(self._config.buffers)
            self._spawn_routines(self._config.routines, val)
            self._start_time = time.time()
            self._state = "running"
            LOG.info("verb start: %d routines spawned", len(self._children))

    def _verb_stop(self, val: Any) -> None:
        with self._lock:
            if self._state == "stopped":
                LOG.info("verb stop: already stopped; ignoring")
                return
            self._state = "stopping"
            self._kill_routines()
            if self._config is not None:
                self._destroy_buffers(self._config.buffers)
            self._start_time = None
            self._state = "stopped"
            LOG.info("verb stop: clean")

    def _verb_utc_start(self, val: Any) -> None:
        seq = int(val) if val is not None else 0
        self._send_utc_udp(f"UTC_START-{seq}")
        # Persist the trigger sequence into the mon namespace so other
        # consumers (legacy snap monitor, web UIs) can read it. Mirrors
        # the legacy corr.py behaviour of writing /mon/snap/1/utc_start.
        try:
            self._store.put_dict(f"/mon/snap/1/utc_start_rt", {"val": seq})
        except Exception:  # noqa: BLE001
            LOG.exception("could not publish utc_start mirror to etcd")

    def _verb_utc_stop(self, val: Any) -> None:
        seq = int(val) if val is not None else 0
        self._send_utc_udp(f"UTC_STOP-{seq}")

    # ---- buffer ops ---------------------------------------------------

    def _create_buffers(self, buffers: tuple[BufferSpec, ...]) -> None:
        for b in buffers:
            cmd = b.dada_db_cmd()
            LOG.info("create buffer %s: %s", b.k, " ".join(cmd))
            rc = subprocess.run(cmd, check=False).returncode
            if rc != 0:
                # Buffer may already exist from a previous unclean stop;
                # destroy + recreate. If that fails too, surface the error.
                LOG.warning("dada_db create for %s rc=%d; trying destroy+recreate",
                            b.k, rc)
                subprocess.run(b.dada_db_destroy_cmd(), check=False)
                rc2 = subprocess.run(cmd, check=False).returncode
                if rc2 != 0:
                    raise RuntimeError(
                        f"dada_db create for {b.k} failed twice (rc={rc}, then rc={rc2})"
                    )

    def _destroy_buffers(self, buffers: tuple[BufferSpec, ...]) -> None:
        # Tear down in reverse order: readers first (corr_slow / corr_fast
        # already killed by _kill_routines), then writers, then captures.
        for b in reversed(buffers):
            cmd = b.dada_db_destroy_cmd()
            LOG.info("destroy buffer %s: %s", b.k, " ".join(cmd))
            subprocess.run(cmd, check=False)

    # ---- routine spawn ------------------------------------------------

    def _select_routines(
        self, routines: tuple[RoutineSpec, ...]
    ) -> tuple[RoutineSpec, ...]:
        assert self._config is not None
        return tuple(
            r for r in routines
            if r.when is None or _evaluate_when(r.when, self._config.raw)
        )

    def _spawn_routines(
        self, routines: tuple[RoutineSpec, ...], val: Any
    ) -> None:
        active = self._select_routines(routines)
        if not active:
            LOG.info("no routines to spawn (check captures.mode in config)")
            return
        for r in active:
            argv = self._build_argv(r, val)
            log_path = self._routine_log_path(r.name)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            LOG.info("spawn %s -> log=%s; argv=%s",
                     r.name, log_path, " ".join(shlex.quote(a) for a in argv))
            # Use shell=False; capture all output to per-routine log.
            log_fh = open(log_path, "ab", buffering=0)
            try:
                proc = subprocess.Popen(  # noqa: S603 — argv is operator-controlled
                    argv,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                log_fh.close()
                LOG.error("spawn %s failed: %s (cmd not found: %r)",
                          r.name, exc, argv[0])
                continue
            self._children[r.name] = proc

    def _build_argv(self, r: RoutineSpec, val: Any) -> list[str]:
        cmd_parts = shlex.split(r.cmd)
        arg_parts = shlex.split(r.args) if r.args else []
        host_extra = []
        # hostargs lookup: try the full FQDN first, then the bare short
        # hostname (legacy /cnf/pipeline uses 'lxd110h01' style; new
        # configs may use 'n06' or 'n06.pro.pvt').
        for key in (self.fqdn, self.fqdn.split(".")[0]):
            if key in r.hostargs:
                host_extra = shlex.split(r.hostargs[key])
                break
        argv = cmd_parts + arg_parts + host_extra
        # Legacy substitution: CUSTOMDEC -> verb val (operator-supplied
        # dec in degrees on `start`). Also CHGROUP -> cn_id-derived
        # chgroup index (per dsart-rt chgroup convention).
        argv = [self._substitute(tok, val) for tok in argv]
        return argv

    def _substitute(self, token: str, val: Any) -> str:
        if "CUSTOMDEC" in token and val is not None:
            token = token.replace("CUSTOMDEC", str(val))
        if "CHGROUP" in token:
            token = token.replace("CHGROUP", str(self._cn_to_chgroup()))
        if "CN" in token:
            token = token.replace("CN", str(self.cn_id))
        return token

    def _cn_to_chgroup(self) -> int:
        # corr-node IDs are 3..22 (skipping 17 + 20). Chgroups 0..15.
        # See REALTIME_FRB_SEARCH.md §1 + tools/dod/corner_turn.sh
        # CORR_NODES for the canonical order.
        corr_nodes = (3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22)
        if self.cn_id in corr_nodes:
            return corr_nodes.index(self.cn_id)
        return 0

    def _routine_log_path(self, name: str) -> str:
        log_root = os.environ.get("DSART_RT_LOG_DIR",
                                  os.path.expanduser("~/tmp/dsart-rt"))
        return os.path.join(log_root, f"{self.namespace}-{self.cn_id}-{name}.log")

    # ---- routine kill -------------------------------------------------

    def _kill_routines(self) -> None:
        for name, proc in list(self._children.items()):
            if proc.poll() is not None:
                continue
            try:
                # start_new_session=True at spawn -> killpg targets the
                # whole subprocess group, so shell-launched grandchildren
                # (e.g. taskset wrappers) get caught too.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        # Wait briefly for clean shutdown, then SIGKILL stragglers.
        deadline = time.monotonic() + 5.0
        for name, proc in list(self._children.items()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                LOG.warning("routine %s did not exit on SIGTERM; SIGKILL", name)
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    LOG.error("routine %s still alive after SIGKILL", name)
        self._children.clear()

    def _reap_dead_children(self) -> None:
        """Notice if a routine exited on its own; log RC."""
        for name in list(self._children):
            proc = self._children[name]
            rc = proc.poll()
            if rc is not None:
                LOG.warning("routine %s exited rc=%d", name, rc)
                del self._children[name]

    # ---- utc UDP poke -------------------------------------------------

    def _send_utc_udp(self, payload: str) -> None:
        # Legacy convention (REALTIME_FRB_SEARCH.md §8): UTC_*-<seq>
        # blast to 127.0.0.1:11223 AND :11224 (one port per capture).
        # In junkdb mode there's no listener; the send is still safe
        # (UDP -> ICMP unreachable is silently dropped).
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for port in self.utc_udp_ports:
                sock.sendto(payload.encode("ascii"), ("127.0.0.1", port))
        finally:
            sock.close()
        LOG.info("UTC poke %r -> 127.0.0.1:%s", payload, self.utc_udp_ports)

    # ---- mon publish --------------------------------------------------

    def _publish_mon(self) -> None:
        self._publish_heartbeat()
        self._publish_mon_dict()

    def _publish_heartbeat(self, *, extra: Optional[dict[str, Any]] = None) -> None:
        beat: dict[str, Any] = {
            "cadence": self.mon_cadence_s,
            "time_mjd": _mjd_now(),
            "state": self._state,
        }
        if extra:
            beat.update(extra)
        self._store.put_dict(self.heartbeat_key, beat)

    def _publish_mon_dict(self) -> None:
        # IMPORTANT: a `start` verb on the production buffer set holds
        # the lock for ~17 s while `dada_db -l` mlocks ~22 GiB of pages
        # (the fada ring alone is 288 MiB * 70 ~= 20 GiB and pinning
        # takes ~10 s). If mon-publish blocks waiting for the lock, the
        # main loop also blocks, which silences the heartbeat and makes
        # the orchestrator look dead to consumers. Acquire with a short
        # timeout and skip this cycle if a verb is in flight; heartbeat
        # is already lock-free (see _publish_heartbeat) so consumers
        # still see we're alive.
        if not self._lock.acquire(timeout=0.5):
            LOG.debug("mon-dict publish skipped: verb in flight")
            return
        try:
            children_snap = {
                name: {"pid": p.pid, "alive": p.poll() is None}
                for name, p in self._children.items()
            }
            buffers_snap = {}
            if self._config is not None:
                for b in self._config.buffers:
                    buffers_snap[b.k] = {
                        "key": b.k,
                        "metric": _dada_dbmetric(b.k),
                    }
            uptime = (time.time() - self._start_time
                      if self._start_time is not None else 0.0)
            last_verb = None
            if self._last_verb is not None:
                v, val, ts = self._last_verb
                last_verb = {"verb": v, "val": val,
                             "age_s": round(time.time() - ts, 3)}
            mon = {
                "cadence": self.mon_cadence_s,
                "time_mjd": _mjd_now(),
                "instance": self.instance,
                "cn": self.cn_id,
                "host": self.fqdn,
                "state": self._state,
                "uptime_s": round(uptime, 3),
                "routines": children_snap,
                "buffers": buffers_snap,
                "last_verb": last_verb,
            }
        finally:
            self._lock.release()
        self._store.put_dict(self.mon_key, mon)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_orchestrator(args: argparse.Namespace) -> RtOrchestrator:
    # Namespace prefix per Q15 (m7/control-plane): corr instances live
    # under /cmd/corr_rt + /mon/corr_rt; search instances under
    # /cmd/search_rt + /mon/search_rt. Stays disjoint from legacy
    # /cmd/corr + /mon/corr so the two control planes can coexist.
    if args.instance == "pipeline_rt":
        namespace = "corr_rt"
    elif args.instance == "search_rt":
        namespace = "search_rt"
    else:
        raise SystemExit(f"unknown -in/--instance value {args.instance!r}")
    config_key = args.config_key or f"/cnf/{args.instance}"
    cmd_key = f"/cmd/{namespace}/{args.cn_id}"
    broadcast_key = f"/cmd/{namespace}/0"
    mon_key = f"/mon/{namespace}/{args.cn_id}"
    heartbeat_key = f"/mon/service/{namespace}/{args.cn_id}"
    return RtOrchestrator(
        instance=args.instance,
        cn_id=args.cn_id,
        namespace=namespace,
        config_key=config_key,
        cmd_key=cmd_key,
        broadcast_key=broadcast_key,
        mon_key=mon_key,
        heartbeat_key=heartbeat_key,
        mon_cadence_s=args.mon_cadence_s,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-in", "--instance", required=True,
                   choices=("pipeline_rt", "search_rt"),
                   help="role: pipeline_rt = corr-node orchestration; "
                        "search_rt = search-node orchestration.")
    p.add_argument("-cn", "--cn-id", required=True, type=int,
                   help="node id (corr-node short ID 3..22 for "
                        "pipeline_rt; search-node short ID 1/2/9/13 "
                        "for search_rt). Matches REALTIME_FRB_SEARCH.md §1.")
    p.add_argument("--config-key", default=None,
                   help="override etcd config key (default: /cnf/<instance>)")
    p.add_argument("--mon-cadence-s", type=float, default=2.0,
                   help="seconds between heartbeat + mon-dict publish (default: 2.0)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    orch = _build_orchestrator(args)
    return orch.start()


if __name__ == "__main__":
    sys.exit(main())
