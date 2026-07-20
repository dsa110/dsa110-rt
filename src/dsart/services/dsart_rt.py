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
  legacy ``UTC_START-<seq>`` string. Persists the SNAP-wall specnum to
  ``/mon/snap/1/utc_start_rt``; also refreshes the legacy keys
  ``/mon/snap/1/armed_mjd`` (= now MJD) and ``/mon/snap/1/utc_start``
  (= 0) so the slow-vis writer (``dsamfs``) anchors UVH5 time tags on
  the current wall clock instead of the long-frozen pre-M7-cutover
  values.
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
import datetime
import hashlib
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

# ---------------------------------------------------------------------------
# SNAP PPS arm anchoring (2026-07-20)
# ---------------------------------------------------------------------------
#: Wire sequence tick (one native time sample) and the 35-bit wrap.
#: The SNAP counter is zeroed on a GPS-locked PPS edge at the SNAP arm
#: and the wire seq field wraps every 2^35 ticks = 13.0312 days.
SNAP_SEQ_TICK_US: float = 32.768
SNAP_SEQ_WRAP: int = 1 << 35

#: /mon/snap/<N>/armed_mjd keys written by the SNAP arm cycle hold the
#: PPS epoch (MJD of seq 0). N=1 is EXCLUDED: that key is the legacy
#: singleton this service overwrites as the capture-arm record.
SNAP_PPS_EPOCH_IDS = tuple(range(2, 18))
#: Minimum number of per-SNAP epochs that must agree for the PPS path.
SNAP_PPS_MIN_AGREE: int = 3
#: Agreement tolerance between per-SNAP epochs (10 ms in days).
SNAP_PPS_TOL_DAYS: float = 10e-3 / 86400.0
#: Sanity window: a PPS-derived armed_mjd farther than this from the
#: wall clock means a stale/incoherent epoch — fall back to the latch.
SNAP_PPS_SANITY_S: float = 600.0


def compute_pps_armed_mjd(
    pps_epoch_mjd: float, arm_seq: int, now_mjd: float,
) -> "tuple[float, int]":
    """MJD at which the SNAP counter reaches ``arm_seq`` (wrap-aware).

    The SNAP wire sequence is zeroed on the GPS PPS recorded at
    ``pps_epoch_mjd`` and ticks at 32.768 µs, wrapping every 2^35
    ticks. ``arm_seq`` (the UTC_START value) is defined modulo the
    wrap, so the absolute arm time is
    ``epoch + (arm_seq + k·2^35)·tick`` with the wrap count ``k``
    chosen so the result lands nearest the wall clock ``now_mjd``
    (the verb is always issued within seconds of the arm moment,
    vastly inside the ±6.5-day wrap ambiguity).

    Returns ``(armed_mjd, k)``.
    """
    tick_days = SNAP_SEQ_TICK_US * 1e-6 / 86400.0
    wrap_days = SNAP_SEQ_WRAP * tick_days
    base = pps_epoch_mjd + (arm_seq % SNAP_SEQ_WRAP) * tick_days
    k = int(round((now_mjd - base) / wrap_days))
    if k < 0:
        k = 0
    return base + k * wrap_days, k

from dsautils.dsa_store import DsaStore

LOG = logging.getLogger("dsart.services.dsart_rt")

# ---------------------------------------------------------------------------
# Spectral-line (SPL) mode
# ---------------------------------------------------------------------------
#: etcd key the dashboard's spectral-line panel writes and the
#: orchestrator reads at ``start`` to decide, per sub-band, whether to
#: spawn the SPL second fringe-stopper (``meridian_fringestop_spl``) or
#: the no-op ``bada_null_drain`` as the constant second ``bada`` reader.
#: Schema::
#:
#:   {"version": 1, "ts": <unix>, "actor": <str>,
#:    "subbands": {"<chgroup 0..15>": {"enabled": bool,
#:                                     "integration_s": float,
#:                                     "nfreq_int": int}}}
#:
#: A missing key / missing sub-band entry ⇒ SPL disabled (fail-SAFE to
#: the production single-fringe-stop topology).
SPECTRAL_LINE_KEY: str = "/cnf/spectral_line"

#: Slow-vis block cadence (s) — corr_slow writes one ``bada`` block this
#: often (see configs/config_corr.yaml). Used only for the human-facing
#: default integration time; the casa38 wrapper recomputes nint from the
#: authoritative /cnf/corr tsamp at runtime.
_SPL_TSAMP_S: float = 0.134217728

#: Legacy production slow-vis integration (nint=96 in /cnf/fringe) → the
#: default SPL integration time so an operator who only changes the
#: channelisation keeps the same 12.88 s cadence (and re-uses the
#: production fringe-stopping-table cache → no slow regen).
_SPL_DEFAULT_NINT: int = 96
_SPL_DEFAULT_INTEGRATION_S: float = _SPL_DEFAULT_NINT * _SPL_TSAMP_S
_SPL_DEFAULT_NFREQ_INT: int = 1


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
    # Per-routine env overlay. Merged on top of os.environ at spawn
    # time so multi-process CUDA workers can each get their own
    # CUDA_VISIBLE_DEVICES mask (without that mask, two children
    # initializing different CUDA devices on the same node hit a
    # "Triton Error [CUDA]: context is destroyed" race; see the M7.1
    # report for the failure mode).
    env: dict[str, str] = field(default_factory=dict)
    # M7.2 (2026-05-19) warmup-aware spawn ordering: this routine
    # will not be spawned until ALL paths listed in ``gate_on_paths``
    # exist on the local filesystem. Used to delay capture routines
    # (dada_junkdb) until corr_fast / corr_slow have finished their
    # multi-second Python + GPU + Triton import cold start (they
    # touch a sentinel right before entering their main loop). The
    # orchestrator deletes any pre-existing sentinels at the start
    # of ``_verb_start`` so a re-start doesn't see stale files.
    gate_on_paths: tuple[str, ...] = field(default_factory=tuple)
    # 2026-06-09 startup-stagger: when > 0, sleep this many seconds
    # AFTER spawning this routine before spawning the NEXT one in the
    # same wave. Used to keep two CUDA workers on the same node from
    # simultaneously racing the kernel for ~17 GiB of mlock'd pinned
    # host pages (each search_compute half does its own cudaHostAlloc
    # storm during pipeline build, and on the 93 GiB nodes the
    # combined startup transient OOM-killed n09/n13 halves on
    # 2026-06-09 even though each half's steady state fit fine). 5 s
    # is comfortably longer than a single half's pinned-pool warmup
    # without meaningfully delaying the "start" verb (search_compute
    # cold-start is ~2.5 min total to first cube anyway).
    spawn_delay_s: float = 0.0
    # Spectral-line (SPL) gate. ``None`` → routine is unconditional
    # (default). ``"on"``  → spawn ONLY when this node's sub-band has
    # spectral-line mode enabled in /cnf/spectral_line. ``"off"`` →
    # spawn ONLY when SPL is disabled for this node. This is how the
    # ``bada`` ring keeps a constant reader count of r=2: with SPL on
    # the second reader is ``meridian_fringestop_spl`` (a finer-
    # channelisation second fringe-stopper); with SPL off the second
    # reader is ``bada_null_drain`` (a no-op drain) so corr_slow's
    # multi-reader back-pressure chain never stalls waiting on an
    # absent reader. See configs/dsart_pipeline_rt.yaml + the
    # spectral-line panel in the dashboard Control tab.
    spl_gate: Optional[str] = None


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
                env={str(k): str(v) for k, v in (r.get("env") or {}).items()},
                gate_on_paths=tuple(
                    str(p) for p in (r.get("gate_on_paths") or [])
                ),
                spawn_delay_s=float(r.get("spawn_delay_s", 0.0)),
                spl_gate=(
                    str(r["spl_gate"]).strip().lower()
                    if r.get("spl_gate") is not None
                    else None
                ),
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

    Supports ``"<dotted.path> == <literal>"`` and the negated form
    ``"<dotted.path> != <literal>"`` where literal is a bare word
    (e.g. ``junkdb``, ``real``, ``synth_fada``) or a quoted string. No
    operators beyond ``==`` / ``!=``; keep the language small to keep
    etcd YAMLs predictable.

    Note: split on the 2-char operators first; we check ``!=`` before
    ``==`` so the ``==`` branch isn't accidentally taken when the
    predicate uses ``!=`` (since ``"a != b".split("==")`` returns the
    whole string).
    """
    negate = False
    if "!=" in expr:
        parts = expr.split("!=")
        negate = True
    else:
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
    eq = str(node) == rhs
    return (not eq) if negate else eq


def _mjd_now() -> float:
    if _HAVE_ASTROPY:
        return float(Time.now().mjd)
    # Fallback: UNIX -> MJD via the standard offset (40587.0 = MJD on 1970-01-01).
    return 40587.0 + time.time() / 86400.0


def _extract_apply_cal_path(argv: list[str]) -> Optional[str]:
    """Pull the ``--apply-cal <path>`` (or ``--apply-cal=<path>``) value
    out of a routine's built argv, or ``None`` if the flag isn't
    present (e.g. search-node routines, which don't apply a cal blob
    directly). Pure / side-effect-free so it's cheap to unit test
    independently of the rest of the orchestrator.
    """
    for i, tok in enumerate(argv):
        if tok == "--apply-cal":
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--apply-cal="):
            return tok.split("=", 1)[1]
    return None


def _stat_cal_file(path: str) -> dict[str, Any]:
    """Best-effort stat + hash of a cal-weights file for the
    ``<mon_key>/cal_file`` mon publish (see :meth:`RtOrchestrator.
    _publish_cal_file_mon`). Never raises: a stat/hash failure is
    reported in the returned dict (``stat_error`` / ``hash_error``)
    rather than propagated, since this must never block the ``start``
    verb.
    """
    payload: dict[str, Any] = {"path": path}
    try:
        st = os.stat(path)
    except OSError as exc:
        payload["stat_error"] = str(exc)
        return payload
    payload["mtime_unix"] = st.st_mtime
    payload["mtime_isot"] = datetime.datetime.fromtimestamp(
        st.st_mtime, tz=datetime.timezone.utc
    ).isoformat()
    payload["size"] = st.st_size
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        payload["sha256_12"] = h.hexdigest()[:12]
    except OSError as exc:
        payload["hash_error"] = str(exc)
    return payload


#: Legacy PSRDADA ``dada_dbmetric`` CSV column order (`/usr/local/bin/dada_dbmetric`
#: build that ships on the dsa110 corr nodes). Maps positional CSV
#: indices to canonical field names so M7.4 Phase 7 Grafana panels have a
#: stable schema regardless of which CLI flavour the node ships.
#:
#: See ``/home/ubuntu/vishnu/dev/dsa110-xengine/scripts/corr.py:166-186``
#: (``get_buf_info``) for the legacy correlator's interpretation of the
#: same CSV.
_DBMETRIC_CSV_FIELDS: tuple[str, ...] = (
    "nbufs",        # total blocks in the ring
    "nfull",        # blocks currently containing unread data
    "nclear",       # blocks that are written + cleared (legacy "n_cleared")
    "n_written",    # cumulative blocks written since ring open
    "n_read",       # cumulative blocks read since ring open
)


def _normalise_dbmetric(counters: dict[str, Any]) -> dict[str, Any]:
    """Add derived fields + legacy-name aliases to a raw dbmetric dict.

    Operationally the single most useful number is ``free_blocks`` =
    ``nbufs - nfull``; older snapshot scripts also look for ``free`` /
    ``full`` (no ``n``-prefix) so we alias those in too. We never
    overwrite an existing key.
    """
    if not counters:
        return counters
    try:
        nbufs = int(counters["nbufs"])
        nfull = int(counters["nfull"])
        counters.setdefault("free_blocks", nbufs - nfull)
        counters.setdefault("free", nbufs - nfull)
        counters.setdefault("full", nfull)
    except (KeyError, TypeError, ValueError):
        pass
    return counters


def _dada_dbmetric(key: str) -> dict[str, Any]:
    """Read full / clear / written / read counters for one PSRDADA ring.

    Returns an empty dict on a true failure (buffer doesn't exist,
    binary missing, parse fail). Don't let mon-dict publish crash on a
    transient buffer-create race.

    The on-disk ``dada_dbmetric`` ships in two output flavours: older
    PSRDADA builds print positional CSV
    ``<nbufs>,<nfull>,<nclear>,<n_written>,<n_read>``, newer ones print
    ``key=value`` tokens. We accept both: parse k=v when present, fall
    back to mapping positional CSV via :data:`_DBMETRIC_CSV_FIELDS`. The
    output dict always has the canonical names so downstream consumers
    (the influx pusher's ``corr_rt_buffer`` measurement, Grafana panels,
    snapshot scripts) get stable field names. We also derive
    ``free_blocks = nbufs - nfull`` (and the ``free`` / ``full`` aliases
    for the legacy snapshot scripts).

    The binary is looked up via two paths so the orchestrator works
    both from a login shell with PATH set and from a systemd service
    unit with the default minimal PATH: ``dada_dbmetric`` (PATH) and
    ``/usr/local/bin/dada_dbmetric`` (DSA-110 cluster canonical install
    location).

    On a parse / binary failure we still return a non-empty dict carrying
    a ``_error`` field so Grafana can distinguish "buffer momentarily
    locked / racy" from "binary missing"; the influx pusher's
    ``make_buffer_points`` skips dicts whose only field is ``_error``
    (no numeric fields ⇒ no time-series point).
    """
    bin_candidates = ("dada_dbmetric", "/usr/local/bin/dada_dbmetric")
    last_err: str | None = None
    proc: subprocess.CompletedProcess[str] | None = None
    for binary in bin_candidates:
        try:
            proc = subprocess.run(
                [binary, "-k", key],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            break
        except FileNotFoundError as e:
            last_err = f"FileNotFoundError({binary}): {e}"
            continue
        except subprocess.TimeoutExpired:
            return {"_error": "dada_dbmetric timeout after 2.0s"}
    if proc is None:
        return {"_error": last_err or "dada_dbmetric binary not found"}
    # Critical: the dsa110-cluster ``/usr/local/bin/dada_dbmetric`` build
    # writes the metric line to **STDERR**, not STDOUT (only the
    # ``ipc_alloc/ipcsync_get/ipcbuf_connect`` error chain on a missing
    # ring goes to stderr too, but those are well-formed prefixed lines
    # we can filter out). Newer builds write the same line to STDOUT.
    # Coalesce both streams and let the parser figure it out — this is
    # the fix for M7.4 Phase 7 ``metric: {}`` everywhere on the live
    # fleet.
    raw_out = (proc.stdout or "")
    raw_err = (proc.stderr or "")
    combined_lines: list[str] = []
    for line in (raw_out + raw_err).splitlines():
        s = line.strip()
        if not s:
            continue
        # Filter ipc_alloc / ipcsync_get / ipcbuf_connect error chain
        # so they don't get misread as a CSV row of nonsense.
        if s.startswith(("ipc_alloc:", "ipcsync_get:", "ipcbuf_connect:")):
            continue
        combined_lines.append(s)
    if not combined_lines:
        err_first = (raw_err.strip().splitlines() or [""])[0]
        return {"_error": err_first or "empty stdout+stderr"}
    out = combined_lines[0]
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
        # Positional CSV. Map by index to canonical names so the
        # downstream Grafana panels have a stable schema (legacy
        # builds don't emit k=v).
        csv_line = out.splitlines()[0]
        raw_fields = [f.strip() for f in csv_line.split(",")]
        for i, name in enumerate(_DBMETRIC_CSV_FIELDS):
            if i >= len(raw_fields):
                break
            try:
                counters[name] = int(raw_fields[i])
            except ValueError:
                continue
        if any(isinstance(v, int) for v in counters.values()):
            counters["raw_csv"] = csv_line
        else:
            counters = {"_error": f"unparseable: {csv_line!r}"}
    if not counters:
        # Output was non-empty but neither k=v nor CSV-numbers. Surface
        # the first line as a diagnostic so an operator can see what
        # the binary actually produced (e.g. "could not parse key from
        # xxxx" on a typoed key).
        counters = {"_error": out[:200]}
    return _normalise_dbmetric(counters)


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
        # operator-integration: strict observation time cap. ``_armed_at``
        # is the wall-clock moment this process last saw UTC_START; the
        # watchdog (see _check_obs_watchdog) auto-stops once the elapsed
        # time exceeds ``max_obs_seconds`` from /cmd/operator/control.
        # Scoped to this process's lifetime so it bounds session/agent
        # observations without surprising-stopping pre-existing recording.
        self._armed_at: Optional[float] = None
        self._obs_cap_cache: tuple[float, int] = (0.0, 0)
        self._stop_evt = threading.Event()
        self._watch_ids: list[int] = []
        self._lock = threading.RLock()
        # Per-node spectral-line state, refreshed from /cnf/spectral_line
        # at each ``start``. Default = SPL disabled so routine selection
        # + token substitution have a sane value even before the first
        # _load_spl_cfg() (e.g. mon publish racing a cold start).
        self._spl: dict[str, Any] = {
            "enabled": False,
            "integration_s": _SPL_DEFAULT_INTEGRATION_S,
            "nfreq_int": _SPL_DEFAULT_NFREQ_INT,
        }

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
                try:
                    self._check_obs_watchdog()
                except Exception:  # noqa: BLE001
                    LOG.exception("obs watchdog tick failed (continuing)")
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

    def _load_spl_cfg(self) -> None:
        """Refresh this node's spectral-line state from etcd.

        Reads :data:`SPECTRAL_LINE_KEY` and resolves the entry for THIS
        node's chgroup (sub-band). On any miss / malformed value /
        transport error we fall back to SPL DISABLED — the production
        single-fringe-stop topology — so a cold or partial etcd never
        accidentally turns on the second fringe-stopper.
        """
        chgroup = self._cn_to_chgroup()
        spl = {
            "enabled": False,
            "integration_s": _SPL_DEFAULT_INTEGRATION_S,
            "nfreq_int": _SPL_DEFAULT_NFREQ_INT,
        }
        try:
            raw = self._store.get_dict(SPECTRAL_LINE_KEY)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("spectral-line config read failed (%s); SPL disabled", exc)
            self._spl = spl
            return
        if isinstance(raw, dict):
            subbands = raw.get("subbands") or {}
            sub = subbands.get(str(chgroup))
            if isinstance(sub, dict):
                spl["enabled"] = bool(sub.get("enabled", False))
                try:
                    spl["integration_s"] = float(
                        sub.get("integration_s", _SPL_DEFAULT_INTEGRATION_S)
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    spl["nfreq_int"] = int(
                        sub.get("nfreq_int", _SPL_DEFAULT_NFREQ_INT)
                    )
                except (TypeError, ValueError):
                    pass
        self._spl = spl
        LOG.info(
            "spectral-line: chgroup=%d enabled=%s integration_s=%.6g nfreq_int=%d",
            chgroup, spl["enabled"], spl["integration_s"], spl["nfreq_int"],
        )

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
            # If the operator didn't supply an observing-dec value with
            # the verb (val is None / missing), fall back to the
            # canonical declination service at /mon/array/dec.dec_deg
            # (same key the meridian-fringestop pipeline reads via
            # dsa110-meridian-fs/dsamfs/utils.get_pointing_declination).
            # Lets routine startup proceed without an out-of-band --dec.
            if val is None:
                val = self._resolve_dec_from_etcd()
            self._state = "starting"
            self._reload_config()
            self._load_spl_cfg()
            assert self._config is not None
            self._create_buffers(self._config.buffers)
            self._spawn_routines(self._config.routines, val)
            self._publish_cal_file_mon(val)
            self._start_time = time.time()
            self._state = "running"
            LOG.info("verb start: %d routines spawned", len(self._children))

    def _publish_cal_file_mon(self, val: Any) -> None:
        """Best-effort: publish which cal-weights file the ``corr_fast``
        routine actually loaded on THIS start.

        ``corr_fast_integration`` reads ``--apply-cal <path>`` exactly
        once at process startup (see its module docstring) and
        publishes nothing about it, so an operator has no way to tell
        whether a node is running stale beamformer weights short of
        SSH-ing in and checking the file mtime by hand (this bit the
        fleet for 3 days — see the SEFDs "Pipeline weights" panel).
        We stat+hash the file right after spawn and publish to
        ``<mon_key>/cal_file`` (e.g. ``/mon/corr_rt/6/cal_file``) so the
        dashboard can cross-check it against the last distributed
        solution (``/mon/cal/bfweights``).

        Schema published::

            {"path": str, "mtime_unix": float, "mtime_isot": str,
             "size": int, "sha256_12": str (first 12 hex chars),
             "spawned_at_unix": float}

        (``stat_error`` / ``hash_error`` instead of the corresponding
        fields on a failure.) Search nodes (no ``corr_fast`` routine)
        and any stat/hash failure are both handled without raising —
        this must never block the ``start`` verb.
        """
        assert self._config is not None
        routine = next(
            (r for r in self._config.routines if r.name == "corr_fast"),
            None,
        )
        if routine is None or "corr_fast" not in self._children:
            LOG.debug(
                "cal-file mon: no corr_fast routine on this node "
                "(e.g. search node); skipping"
            )
            return
        try:
            argv = self._build_argv(routine, val)
            path = _extract_apply_cal_path(argv)
            if path is None:
                LOG.debug(
                    "cal-file mon: corr_fast argv has no --apply-cal "
                    "flag; nothing to publish"
                )
                return
            payload = _stat_cal_file(path)
            payload["spawned_at_unix"] = time.time()
            self._store.put_dict(f"{self.mon_key}/cal_file", payload)
            LOG.info("cal-file mon published to %s/cal_file: %s",
                      self.mon_key, payload)
        except Exception as exc:  # noqa: BLE001 — best-effort, never
                                   # block the start verb on this.
            LOG.warning("cal-file mon publish failed (non-fatal): %s", exc)

    def _resolve_dec_from_etcd(self) -> Optional[float]:
        """Read /mon/array/dec.dec_deg from etcd as a CUSTOMDEC fallback.

        Returns None if the key is missing / malformed / unreachable;
        the caller treats that as "no substitution" and routines that
        reference CUSTOMDEC will fail to start with the literal token
        (deliberately loud — better than silently using a stale or zero
        declination).
        """
        try:
            d = self._store.get_dict("/mon/array/dec")
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "verb start: /mon/array/dec read failed: %s; "
                "CUSTOMDEC unresolved", exc,
            )
            return None
        if not isinstance(d, dict) or "dec_deg" not in d:
            LOG.warning(
                "verb start: /mon/array/dec has no dec_deg field "
                "(got %r); CUSTOMDEC unresolved", d,
            )
            return None
        try:
            v = float(d["dec_deg"])
        except (TypeError, ValueError) as exc:
            LOG.warning(
                "verb start: /mon/array/dec.dec_deg not a float "
                "(%r): %s; CUSTOMDEC unresolved", d.get("dec_deg"), exc,
            )
            return None
        LOG.info(
            "verb start: CUSTOMDEC fallback from "
            "/mon/array/dec.dec_deg = %.4f (mtime_mjd=%s)",
            v, d.get("time", "?"),
        )
        return v

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
            self._armed_at = None  # operator-integration: disarm watchdog
            self._state = "stopped"
            LOG.info("verb stop: clean")

    def _read_snap_pps_epoch(self) -> Optional[float]:
        """The SNAP GPS-PPS arm epoch (MJD of wire-seq 0), or None.

        Reads the per-SNAP ``/mon/snap/<N>/armed_mjd`` keys written by
        the SNAP arm cycle (N ≥ 2 — N=1 is the legacy singleton this
        service itself overwrites below) and returns their consensus
        when at least ``SNAP_PPS_MIN_AGREE`` agree to
        ``SNAP_PPS_TOL_DAYS``. Best-effort: any etcd trouble → None
        (the caller falls back to the wall latch).
        """
        vals: list[float] = []
        for n in SNAP_PPS_EPOCH_IDS:
            try:
                doc = self._store.get_dict(f"/mon/snap/{n}/armed_mjd")
            except Exception:                          # noqa: BLE001
                continue
            if isinstance(doc, dict):
                v = doc.get("armed_mjd")
                if isinstance(v, (int, float)) and 50000.0 < v < 80000.0:
                    vals.append(float(v))
        if len(vals) < SNAP_PPS_MIN_AGREE:
            return None
        vals.sort()
        med = vals[len(vals) // 2]
        agree = [v for v in vals if abs(v - med) <= SNAP_PPS_TOL_DAYS]
        if len(agree) < SNAP_PPS_MIN_AGREE:
            LOG.warning(
                "SNAP PPS epochs disagree (spread %.3f s over %d snaps); "
                "not using PPS anchoring",
                (max(vals) - min(vals)) * 86400.0, len(vals),
            )
            return None
        return med

    def _verb_utc_start(self, val: Any) -> None:
        seq = int(val) if val is not None else 0
        self._send_utc_udp(f"UTC_START-{seq}")
        # operator-integration: arm the observation-time watchdog.
        self._armed_at = time.time()
        # Persist the trigger sequence into the mon namespace.
        #
        # Two keyspaces, two consumer populations:
        #
        # 1) /mon/snap/1/utc_start_rt -- the dsa110-rt key, value is the
        #    SNAP-wall specnum at which capture was armed to begin
        #    recording.  Read by the modern dashboard / Influx pusher.
        #
        # 2) /mon/snap/1/armed_mjd + /mon/snap/1/utc_start -- the legacy
        #    keys read by dsamfs.utils.get_time (slow-vis UVH5 writer,
        #    docs/overview Section 8) and dsacalib.config (K-cal time
        #    stamps). The legacy SNAP arm script that used to keep these
        #    fresh was retired at the M7 cutover, leaving both keys
        #    frozen at their last-pre-cutover values; with no bridge the
        #    slow-vis archive grows wall-clock-wrong UVH5 files anchored
        #    on the stale armed_mjd. Their formula is
        #        get_time() = armed_mjd + utc_start * 4 * 8.192e-6 / 86400
        #    so writing utc_start = 0 makes armed_mjd itself the anchor.
        #    2026-07-20: armed_mjd is now derived from the SNAP GPS-PPS
        #    arm epoch + arm_seq x 32.768 us (wrap-aware) when the
        #    per-SNAP epochs are available -- GPS-grade and exactly the
        #    moment capture begins -- with the previous wall latch as
        #    the recorded fallback (see the anchor block below; the
        #    "source" field in the arm record says which one you got).
        try:
            self._store.put_dict(f"/mon/snap/1/utc_start_rt", {"val": seq})
        except Exception:  # noqa: BLE001
            LOG.exception("could not publish utc_start mirror to etcd")
        # ---- capture-arm time anchor (2026-07-20 PPS upgrade) ---------
        # Preferred: derive armed_mjd from the SNAP GPS-PPS arm epoch
        # (per-SNAP /mon/snap/<N>/armed_mjd) + arm_seq × 32.768 µs,
        # wrap-aware. This is exact: the SNAP counter is zeroed on a
        # known GPS PPS edge, so the moment it reaches `seq` is known
        # to the PPS. The previous wall latch (armed_mjd = Time.now()
        # at this verb) was NTP-grade and systematically EARLY by the
        # arm margin (~1.8 s measured 2026-07-20) because capture
        # only begins when the counter reaches seq. Fallback: the wall
        # latch, with provenance recorded either way in the arm record
        # so downstream consumers (search anchor, voltage manifests,
        # dsamfs bridge — all read "armed_mjd") can tell which they
        # got via the "source" field.
        now_mjd = time.time() / 86400.0 + 40587.0
        pps_epoch = self._read_snap_pps_epoch()
        armed_mjd: Optional[float] = None
        wrap_k = 0
        source = "wall_latch"
        if pps_epoch is not None:
            cand, wrap_k = compute_pps_armed_mjd(pps_epoch, seq, now_mjd)
            if abs(cand - now_mjd) * 86400.0 <= SNAP_PPS_SANITY_S:
                armed_mjd = cand
                source = "pps_epoch"
                LOG.info(
                    "utc_start: PPS-anchored armed_mjd=%.8f "
                    "(epoch=%.8f arm_seq=%d wrap_k=%d; wall-latch "
                    "would be %+.2f s off)",
                    armed_mjd, pps_epoch, seq, wrap_k,
                    (now_mjd - armed_mjd) * 86400.0,
                )
            else:
                LOG.warning(
                    "utc_start: PPS-derived armed_mjd %.8f is %+.1f s "
                    "from the wall clock — stale/incoherent SNAP epoch? "
                    "Falling back to the wall latch",
                    cand, (cand - now_mjd) * 86400.0,
                )
        else:
            LOG.warning(
                "utc_start: no SNAP PPS epoch consensus in etcd; "
                "using the wall-latch armed_mjd (NTP-grade, ~2 s early)"
            )
        if armed_mjd is None:
            armed_mjd = now_mjd
        try:
            self._store.put_dict(
                f"/mon/snap/1/armed_mjd",
                {
                    "armed_mjd": float(armed_mjd),
                    # Provenance (2026-07-20): how this anchor was made.
                    "source": source,
                    "pps_epoch_mjd": pps_epoch,
                    "arm_seq": int(seq),
                    "wrap_k": int(wrap_k),
                    "seq_tick_us": SNAP_SEQ_TICK_US,
                    "verb_wall_mjd": float(now_mjd),
                },
            )
            self._store.put_dict(
                f"/mon/snap/1/utc_start", {"utc_start": 0}
            )
        except Exception:  # noqa: BLE001
            LOG.exception(
                "could not refresh legacy /mon/snap/1/armed_mjd + "
                "/mon/snap/1/utc_start (slow-vis anchor); dsamfs may "
                "continue writing stale time tags"
            )

    def _verb_utc_stop(self, val: Any) -> None:
        seq = int(val) if val is not None else 0
        self._send_utc_udp(f"UTC_STOP-{seq}")
        # operator-integration: disarm the observation-time watchdog.
        self._armed_at = None

    # ---- operator-integration: observation-time watchdog --------------

    def _operator_max_obs_s(self) -> int:
        """Cached read of the human-set hard cap on observation length.

        Reads ``/cmd/operator/control.max_obs_seconds`` at most every 15 s
        so the 2 s mon loop never hammers etcd. Returns 0 (no cap) on any
        problem or when unset — fail-open so a transient etcd hiccup can
        never auto-stop a healthy observation.
        """
        now = time.time()
        ts, cached = self._obs_cap_cache
        if now - ts < 15.0:
            return cached
        cap = 0
        try:
            doc = self._store.get_dict("/cmd/operator/control")
            if isinstance(doc, dict):
                cap = max(0, int(doc.get("max_obs_seconds") or 0))
        except Exception:  # noqa: BLE001
            cap = cached  # keep last known value on a read error
        self._obs_cap_cache = (now, cap)
        return cap

    def _check_obs_watchdog(self) -> None:
        """Auto-``utc_stop`` once an armed observation exceeds the cap.

        Independent of the dsa110-operator agent: even a runaway or
        crashed agent cannot exceed this limit because enforcement lives
        here, in the orchestrator. No-op unless this process armed
        recording (``_armed_at``) and a positive cap is configured.
        """
        if self._armed_at is None:
            return
        cap = self._operator_max_obs_s()
        if cap <= 0:
            return
        elapsed = time.time() - self._armed_at
        if elapsed < cap:
            return
        LOG.warning(
            "OBS WATCHDOG: elapsed %.0fs >= cap %ds -> auto UTC_STOP "
            "(set via /cmd/operator/control max_obs_seconds)", elapsed, cap)
        try:
            self._verb_utc_stop(0)  # also clears _armed_at
        except Exception:  # noqa: BLE001
            LOG.exception("obs watchdog: auto utc_stop failed")
            self._armed_at = None
        try:
            self._store.put_dict(
                f"/mon/operator/watchdog/{self.instance}",
                {"event": "auto_utc_stop", "elapsed_s": round(elapsed, 1),
                 "cap_s": cap, "ts": time.time(), "host": self.fqdn})
        except Exception:  # noqa: BLE001
            pass

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
        spl_on = bool(self._spl.get("enabled"))
        out: list[RoutineSpec] = []
        for r in routines:
            if r.when is not None and not _evaluate_when(r.when, self._config.raw):
                continue
            # Spectral-line gate: keep exactly one of the two bada
            # second-reader routines depending on this node's SPL state.
            if r.spl_gate == "on" and not spl_on:
                continue
            if r.spl_gate == "off" and spl_on:
                continue
            out.append(r)
        return tuple(out)

    # ------------------------------------------------------------------
    # M7.2 (2026-05-19) warmup-aware spawn ordering
    # ------------------------------------------------------------------
    # Default timeout the orchestrator will wait on sentinel paths
    # before opening the gate anyway. corr_fast's cold start runs ~90 s
    # on a 2080Ti including Triton module imports; 240 s gives 2.5×
    # headroom. Override via DSART_RT_GATE_TIMEOUT_S if a host is slow.
    _GATE_DEFAULT_TIMEOUT_S = 240.0

    def _spawn_routines(
        self, routines: tuple[RoutineSpec, ...], val: Any
    ) -> None:
        active = self._select_routines(routines)
        if not active:
            LOG.info("no routines to spawn (check captures.mode in config)")
            return

        # Partition into two waves: routines that don't gate on
        # anything spawn first (producers + consumers of the compute
        # path); routines that DO gate spawn second (capture sources).
        # Sentinel cleanup happens BEFORE wave 1 so stale files from a
        # crashed prior run don't open the gate prematurely.
        # gate_on_paths go through _substitute() so CN / CHGROUP /
        # CALSB / CUSTOMDEC tokens resolve consistently with the argv
        # tokens (per-node sentinel files avoid cross-node collisions
        # when multiple corr_rt instances share /tmp via NFS).
        wave1 = tuple(r for r in active if not r.gate_on_paths)
        wave2 = tuple(r for r in active if r.gate_on_paths)
        all_gate_paths: tuple[str, ...] = tuple({
            self._substitute(p, val)
            for r in wave2 for p in r.gate_on_paths
        })
        if all_gate_paths:
            for p in all_gate_paths:
                try:
                    os.remove(p)
                    LOG.info("removed stale sentinel %s", p)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    LOG.warning("could not remove stale sentinel %s: %s", p, e)

        # Wave 1: spawn ungated routines (compute path). Honors
        # per-routine ``spawn_delay_s`` so two CUDA workers can be
        # staggered to avoid a combined pinned-host-pool startup
        # transient that would OOM the node (2026-06-09 n09 / n13
        # search_compute halves).
        wave1_last = len(wave1) - 1
        for idx, r in enumerate(wave1):
            self._spawn_one_routine(r, val)
            if idx < wave1_last and r.spawn_delay_s > 0.0:
                LOG.info(
                    "spawn-stagger: sleeping %.1fs after %s before "
                    "next routine",
                    r.spawn_delay_s, r.name,
                )
                if self._stop_evt.wait(r.spawn_delay_s):
                    LOG.info(
                        "spawn-stagger: interrupted by stop signal "
                        "(remaining wave-1 routines NOT spawned)"
                    )
                    return
        if wave2:
            LOG.info(
                "wave-1 spawned (%d routines); waiting for sentinels: %s",
                len(wave1), list(all_gate_paths),
            )
            # Wait for sentinels. Capture routines (wave 2) stay gated
            # until all listed paths exist. If the gate times out we
            # open it anyway and log a warning — better to run with a
            # warmup transient than to deadlock the start verb.
            timeout_s = float(
                os.environ.get(
                    "DSART_RT_GATE_TIMEOUT_S",
                    self._GATE_DEFAULT_TIMEOUT_S,
                )
            )
            self._wait_for_sentinels(all_gate_paths, timeout_s=timeout_s)
            # Wave 2: spawn gated routines (capture path).
            for r in wave2:
                self._spawn_one_routine(r, val)

    def _wait_for_sentinels(
        self,
        paths: tuple[str, ...],
        *,
        timeout_s: float,
    ) -> None:
        """Block until every path in ``paths`` exists, or ``timeout_s``
        elapses. Polls at 0.5 s. Honors ``self._stop_evt`` so SIGTERM
        during the wait short-circuits cleanly.
        """
        t_start = time.monotonic()
        pending = list(paths)
        while pending:
            if self._stop_evt.is_set():
                LOG.info("gate wait interrupted by stop signal")
                return
            still_pending = [p for p in pending if not os.path.exists(p)]
            if not still_pending:
                elapsed = time.monotonic() - t_start
                LOG.info("all gate sentinels present after %.1fs", elapsed)
                return
            pending = still_pending
            if (time.monotonic() - t_start) > timeout_s:
                LOG.warning(
                    "gate timeout: %.1fs elapsed, still missing %s; "
                    "opening gate anyway (capture routines may see a "
                    "warmup transient on consumers)",
                    timeout_s, pending,
                )
                return
            self._stop_evt.wait(0.5)

    def _spawn_one_routine(self, r: RoutineSpec, val: Any) -> None:
        argv = self._build_argv(r, val)
        log_path = self._routine_log_path(r.name)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Per-routine env overlay on top of os.environ. Substitution
        # (CHGROUP / CALSB / CN / CUSTOMDEC) is applied to env
        # VALUES too, so e.g. CUDA_VISIBLE_DEVICES could use CN.
        env = None
        if r.env:
            env = dict(os.environ)
            for k, v in r.env.items():
                env[k] = self._substitute(str(v), val)
            env_log = ", ".join(f"{k}={env[k]!r}" for k in r.env)
        else:
            env_log = ""
        gate_log = (
            f"; gate_on_paths={list(r.gate_on_paths)}"
            if r.gate_on_paths else ""
        )
        LOG.info(
            "spawn %s -> log=%s; argv=%s%s%s",
            r.name, log_path, " ".join(shlex.quote(a) for a in argv),
            f"; env_overlay={{{env_log}}}" if env_log else "",
            gate_log,
        )
        # Use shell=False; capture all output to per-routine log.
        log_fh = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(  # noqa: S603 — argv is operator-controlled
                argv,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except FileNotFoundError as exc:
            log_fh.close()
            LOG.error("spawn %s failed: %s (cmd not found: %r)",
                      r.name, exc, argv[0])
            return
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
        # Spectral-line tokens (substituted before CN/CHGROUP since the
        # token names contain neither substring): the SPL second
        # fringe-stopper's per-sub-band integration time + channel
        # averaging come from /cnf/spectral_line (loaded at start).
        if "SPL_INTEGRATION_S" in token:
            token = token.replace(
                "SPL_INTEGRATION_S", repr(float(self._spl["integration_s"]))
            )
        if "SPL_NFREQ_INT" in token:
            token = token.replace(
                "SPL_NFREQ_INT", str(int(self._spl["nfreq_int"]))
            )
        # Legacy substitutions (match corr.py CUSTOMDEC semantics):
        if "CUSTOMDEC" in token and val is not None:
            token = token.replace("CUSTOMDEC", str(val))
        # Per-node substitutions (dsart-rt convention):
        #   CHGROUP -> int chgroup index 0..15
        #   CALSB   -> "sb%02d" (e.g. n06 -> "sb03"), for cal-blob paths
        #              shared across all 16 corr nodes via one YAML.
        #   CN      -> cn_id (int)
        if "CHGROUP" in token:
            token = token.replace("CHGROUP", str(self._cn_to_chgroup()))
        if "CALSB" in token:
            token = token.replace("CALSB", f"sb{self._cn_to_chgroup():02d}")
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
        """Notice if a routine exited on its own; log RC.

        Runs on the main loop while ``_verb_start`` / ``_verb_stop``
        mutate ``self._children`` from the etcd watch thread. Take the
        same ``self._lock`` they hold and pop defensively so a verb that
        clears/respawns routines between the snapshot and the removal can
        no longer crash the orchestrator with ``KeyError`` (the recurring
        ``KeyError: 'search_compute_0'`` that silently killed the n02/n13
        orchestrators — they were ``setsid nohup`` so never restarted).
        """
        with self._lock:
            for name in list(self._children):
                proc = self._children.get(name)
                if proc is None:
                    continue
                rc = proc.poll()
                if rc is not None:
                    LOG.warning("routine %s exited rc=%d", name, rc)
                    self._children.pop(name, None)

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
                "spectral_line": dict(self._spl),
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
