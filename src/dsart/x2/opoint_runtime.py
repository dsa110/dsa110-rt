"""Op-point runtime: which configuration set is active and where it lives.

Frozen interface: docs/x2/SPEC.md §10. One code version runs every op-point;
switching is configuration only.

Op-points
    ``1x``        today's search, run by the unified code in 1x-compat mode;
    ``1x-legacy`` today's search exactly as deployed before x2 (retired later);
    ``2x``        the 2x time-resolution search.

Repo layout (per op-point set: corr pipeline, search pipeline, C2 criteria)
    configs/dsart_pipeline_rt.yaml, dsart_search_rt.yaml, c2_trigger_criteria.yaml
        the base (today's) set, byte-identical to production;
    configs/opoints/<op>/<same file names>
        an op-point's own file. ``1x`` and ``1x-legacy`` fall back per file to
        the base set when they have no own file; ``2x`` never falls back (a
        missing 2x file must fail, not run 1x under a 2x label).

etcd
    /cnf/opoints/<op>/{pipeline_rt,search_rt}  sets pushed by
        ``tools/ops/push_dsart_to_etcd.py --opoint <op> --write`` (never live);
    /cnf/opoints/<op>/meta                     provenance of that push;
    /cnf/opoint                                the active op-point record
        ``{"opoint": "<op>", ...}``; absent = ``1x``;
    /cnf/pipeline_rt, /cnf/search_rt           live; written only by the
        web-UI Start control when it switches op-point (fleet stopped);
    /cnf/opoints/_previous/*                   one-deep backup of the live
        keys taken by each switch.

Services
    h23 services (C2, C3, inject bot) call :func:`select_service_config` with
    the ``--config`` path their systemd unit passes (the base-set path) and
    their ``--opoint`` argument (default ``auto`` = /cnf/opoint, read once at
    startup; switching restarts them). Node services get ``--opoint <op>``
    explicitly in their op-point's etcd routine args.

Placeholders
    Values not yet decided for an op-point are written as
    ``X2TODO:<owner>:<what>`` tokens (they make a service fail at argument
    parsing rather than run with a 1x value), and section-level reviews still
    owed are listed under a top-level ``x2_todo:`` key. :func:`validate_set`
    reports both; the Start control refuses to switch to a set that has any.

Stdlib + PyYAML only (imported by the dashboard and the push tool).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

LOG = logging.getLogger("dsart.x2.opoint_runtime")

OPOINTS: tuple[str, ...] = ("1x", "1x-legacy", "2x")
DEFAULT_OPOINT: str = "1x"
#: Op-points that fall back to the base config file when they have no own file.
BASE_FALLBACK: frozenset[str] = frozenset({"1x", "1x-legacy"})

ACTIVE_KEY: str = "/cnf/opoint"
SETS_PREFIX: str = "/cnf/opoints"
BACKUP_PREFIX: str = "/cnf/opoints/_previous"
INSTANCES: tuple[str, ...] = ("pipeline_rt", "search_rt")
LIVE_KEYS: dict[str, str] = {i: f"/cnf/{i}" for i in INSTANCES}

#: config kind -> file name (same name in the base dir and in each op-point dir).
CONFIG_FILES: dict[str, str] = {
    "pipeline_rt": "dsart_pipeline_rt.yaml",
    "search_rt": "dsart_search_rt.yaml",
    "c2_criteria": "c2_trigger_criteria.yaml",
}
_FILE_TO_KIND: dict[str, str] = {v: k for k, v in CONFIG_FILES.items()}

STAMP_KEY: str = "opoint"
TODO_LIST_KEY: str = "x2_todo"
#: Informational list (does NOT block a switch): values carried over from 1x
#: that are to be recalibrated on 2x data at CP2.
RECAL_LIST_KEY: str = "x2_cp2_recalibrate"

#: Injection K (S/N calibration) lives under this etcd prefix (dashboard
#: ``inject_calibration.CALIBRATION_PREFIX``). K measured at one time
#: resolution is wrong at the other, so a switch between K families parks the
#: live K of the old family and restores the parked K of the new one.
K_PREFIX: str = "/cnf/inject/snr_calibration/"
K_FAMILY: dict[str, str] = {"1x": "1x", "1x-legacy": "1x", "2x": "2x"}
K_PARK_PREFIX: str = "/cnf/opoints/_kpark"


def k_park_prefix(family: str) -> str:
    if family not in set(K_FAMILY.values()):
        raise ValueError(f"unknown K family {family!r}")
    return f"{K_PARK_PREFIX}/{family}/"


#: Every time-unit parameter, explicit per op-point (SPEC §8/§10). The code
#: defaults are 2x (``common/constants.py`` T_INT_SEARCH_US_DEFAULT = 524.288),
#: so a set that omits one of these silently runs 2x units: validate_set flags
#: a missing or wrong value as a problem. ``("args", routine, flag)`` = the last
#: occurrence of ``flag`` in that routine's args; ``("key", dotted.path)`` = a
#: YAML value. ``cube_sample_period_specnum`` (search_compute) is derived from
#: --t-int-search-us (16 at 1x, 8 at 2x); ``cube.sample_period_specnum`` is only
#: the merger's fallback and stays 16 with it. The merger keeps 16 at 2x on
#: purpose: it holds the merge window constant in wall-clock time
#: (detector/merger.py; downstream inventory finding 16).
_TU_1X: dict[tuple, float] = {
    ("pipeline_rt", "args", "corr_fast", "--t-int-fast-native"): 32,
    ("pipeline_rt", "args", "corr_fast", "--chan-sum-factor"): 8,
    ("search_rt", "args", "search_compute_0", "--t-int-search-us"): 1048.576,
    ("search_rt", "args", "search_compute_1", "--t-int-search-us"): 1048.576,
    ("search_rt", "args", "search_compute_0", "--chan-sum-factor"): 8,
    ("search_rt", "args", "search_compute_1", "--chan-sum-factor"): 8,
    ("search_rt", "key", "c1.merger.sample_period_specnum"): 16,
    ("search_rt", "key", "c1.max_c1c2_width_samples"): 16,
    ("search_rt", "key", "cube.sample_period_specnum"): 16,
}
_TU_2X: dict[tuple, float] = {
    ("pipeline_rt", "args", "corr_fast", "--t-int-fast-native"): 8,
    ("pipeline_rt", "args", "corr_fast", "--chan-sum-factor"): 2,
    ("search_rt", "args", "search_compute_0", "--t-int-search-us"): 524.288,
    ("search_rt", "args", "search_compute_1", "--t-int-search-us"): 524.288,
    ("search_rt", "args", "search_compute_0", "--chan-sum-factor"): 2,
    ("search_rt", "args", "search_compute_1", "--chan-sum-factor"): 2,
    ("search_rt", "key", "c1.merger.sample_period_specnum"): 16,
    ("search_rt", "key", "c1.max_c1c2_width_samples"): 32,
    ("search_rt", "key", "cube.sample_period_specnum"): 16,
}
TIME_UNITS: dict[str, dict[tuple, float]] = {"1x": _TU_1X, "1x-legacy": _TU_1X, "2x": _TU_2X}
PLACEHOLDER_RE = re.compile(r"X2TODO:[A-Za-z0-9-]+:[A-Za-z0-9_./=,-]+")
#: Any path that names an op-point-managed config file.
_CONFIG_REF_RE = re.compile(
    r"configs/(?:opoints/([A-Za-z0-9.-]+)/)?"
    r"(dsart_pipeline_rt\.yaml|dsart_search_rt\.yaml|c2_trigger_criteria\.yaml)"
)

#: Where the last successful etcd resolution is cached (h23 services survive an
#: etcd outage at restart on the op-point they last saw instead of 1x).
CACHE_ENV: str = "DSART_OPOINT_CACHE"
_DEFAULT_CACHE = Path.home() / ".local" / "state" / "dsart" / "opoint_last.json"


class OpPointError(RuntimeError):
    """An op-point name, record or set is invalid."""


# ---------------------------------------------------------------------------
# names and keys
# ---------------------------------------------------------------------------


def validate_name(op: str) -> str:
    if op not in OPOINTS:
        raise OpPointError(f"unknown op-point {op!r} (valid: {', '.join(OPOINTS)})")
    return op


def set_key(op: str, instance: str) -> str:
    validate_name(op)
    if instance not in INSTANCES and instance != "meta":
        raise OpPointError(f"unknown instance {instance!r}")
    return f"{SETS_PREFIX}/{op}/{instance}"


def meta_key(op: str) -> str:
    return set_key(op, "meta")


def backup_key(name: str) -> str:
    return f"{BACKUP_PREFIX}/{name}"


def is_live_key(key: str) -> bool:
    """True for keys the running system reads (never written by the push tool)."""
    return key in LIVE_KEYS.values() or key == ACTIVE_KEY


def parse_active_record(raw: Any) -> str | None:
    """Op-point name from a /cnf/opoint value (``None`` = unset)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        name = raw.strip()
    elif isinstance(raw, Mapping):
        name = raw.get(STAMP_KEY)
        if name is None:
            return None
        name = str(name).strip()
    else:
        raise OpPointError(f"{ACTIVE_KEY} holds an unrecognised value {raw!r}")
    return validate_name(name)


def config_stamp(cfg: Any) -> str | None:
    """The ``opoint`` stamp of a config dict (``None`` = unstamped base set)."""
    if isinstance(cfg, Mapping):
        v = cfg.get(STAMP_KEY)
        return None if v is None else str(v)
    return None


def stamp_matches(cfg: Any, op: str) -> bool:
    """An unstamped config is today's base set: it matches 1x and 1x-legacy."""
    s = config_stamp(cfg)
    return (s is None and op in BASE_FALLBACK) or s == op


def stamp(cfg: Mapping[str, Any], op: str) -> dict[str, Any]:
    """Copy of ``cfg`` stamped ``opoint: op``; refuses a conflicting stamp."""
    validate_name(op)
    have = config_stamp(cfg)
    if have is not None and have != op:
        raise OpPointError(f"config is stamped opoint={have!r}, not {op!r}")
    out = dict(cfg)
    out[STAMP_KEY] = op
    return out


# ---------------------------------------------------------------------------
# active op-point resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveOpPoint:
    """Result of :func:`resolve_active_opoint`.

    ``source``: ``cli`` (explicit --opoint), ``etcd`` (/cnf/opoint), ``etcd-unset``
    (key absent -> default), ``cache`` (etcd unreachable -> last seen),
    ``default`` (etcd unreachable, no cache). Services log and publish it.
    """

    name: str
    source: str
    record: Any = None
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"opoint": self.name, "source": self.source, "warning": self.warning}


def _cache_path(cache_path: str | os.PathLike | None) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else _DEFAULT_CACHE


def _write_cache(path: Path, name: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"opoint": name, "unix": time.time()}))
        tmp.replace(path)
    except OSError as exc:
        LOG.warning("op-point cache write %s failed: %s", path, exc)


def _read_cache(path: Path) -> str | None:
    try:
        return validate_name(json.loads(path.read_text())["opoint"])
    except (OSError, ValueError, KeyError, TypeError, OpPointError):
        return None


def _default_store() -> Any:
    from dsautils.dsa_store import DsaStore  # local: optional dependency

    return DsaStore()


def resolve_active_opoint(
    store: Any = None,
    *,
    cli: str | None = None,
    retries: int = 3,
    retry_sleep_s: float = 2.0,
    cache_path: str | os.PathLike | None = None,
    use_cache: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ActiveOpPoint:
    """Resolve the op-point a service should run.

    ``cli`` other than ``None``/``"auto"`` wins. Otherwise /cnf/opoint is read
    (``retries`` attempts); absent = :data:`DEFAULT_OPOINT`. If etcd cannot be
    read the last successfully resolved value (cache file) is used, then the
    default, with a warning. An *invalid* value in etcd raises: that is an
    operator error, not something to paper over.
    """
    if cli not in (None, "", "auto"):
        return ActiveOpPoint(validate_name(str(cli)), "cli")
    cpath = _cache_path(cache_path)
    raw: Any = None
    err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            if store is None:
                store = _default_store()
            raw = store.get_dict(ACTIVE_KEY)
            err = None
            break
        except Exception as exc:  # noqa: BLE001 - etcd/gRPC errors vary
            err = exc
            LOG.warning("read %s failed (attempt %d): %s", ACTIVE_KEY, attempt + 1, exc)
            if attempt + 1 < retries:
                sleep_fn(retry_sleep_s)
    if err is None:
        name = parse_active_record(raw)
        if use_cache:
            _write_cache(cpath, name or DEFAULT_OPOINT)
        if name is None:
            return ActiveOpPoint(DEFAULT_OPOINT, "etcd-unset", raw)
        return ActiveOpPoint(name, "etcd", raw)
    cached = _read_cache(cpath) if use_cache else None
    if cached is not None:
        return ActiveOpPoint(cached, "cache", None,
                             f"etcd unreachable ({err}); using last seen op-point {cached}")
    return ActiveOpPoint(DEFAULT_OPOINT, "default", None,
                         f"etcd unreachable ({err}) and no cache; defaulting to {DEFAULT_OPOINT}")


# ---------------------------------------------------------------------------
# config files
# ---------------------------------------------------------------------------


def config_path(kind: str, op: str, configs_dir: str | os.PathLike) -> Path:
    """Path of config ``kind`` for op-point ``op`` under ``configs_dir``."""
    validate_name(op)
    if kind not in CONFIG_FILES:
        raise OpPointError(f"unknown config kind {kind!r} (valid: {sorted(CONFIG_FILES)})")
    base = Path(configs_dir)
    own = base / "opoints" / op / CONFIG_FILES[kind]
    if own.is_file():
        return own
    if op in BASE_FALLBACK:
        return base / CONFIG_FILES[kind]
    raise FileNotFoundError(f"op-point {op} has no {kind} config: {own} is missing")


def map_config_path(path: str | os.PathLike, op: str) -> Path:
    """Map a config path (as a systemd unit passes it) to op-point ``op``.

    ``.../configs/dsart_search_rt.yaml`` -> ``.../configs/opoints/2x/dsart_search_rt.yaml``
    for 2x; a path already inside ``configs/opoints/<x>/`` is re-based. Files
    that are not op-point managed are returned unchanged.
    """
    p = Path(path)
    kind = _FILE_TO_KIND.get(p.name)
    if kind is None:
        return p
    base = p.parent
    if base.parent.name == "opoints":
        base = base.parent.parent
    return config_path(kind, op, base)


def add_opoint_argument(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add the standard ``--opoint`` option (SPEC §10)."""
    parser.add_argument(
        "--opoint", default=default, choices=("auto",) + OPOINTS,
        help="op-point (default %(default)s = read /cnf/opoint once at startup)",
    )


def select_service_config(
    config: str | os.PathLike,
    *,
    cli_opoint: str | None = "auto",
    store: Any = None,
    **resolve_kw: Any,
) -> tuple[ActiveOpPoint, Path]:
    """One call for an h23 service: resolve the op-point and its config file.

    ``config`` is the path the service was given (``--config``); the returned
    path is that file's equivalent in the active op-point's set.
    """
    active = resolve_active_opoint(store, cli=cli_opoint, **resolve_kw)
    if active.warning:
        LOG.warning("op-point: %s", active.warning)
    path = map_config_path(config, active.name)
    LOG.info("op-point %s (source %s): config %s", active.name, active.source, path)
    return active, path


def load_yaml(path: str | os.PathLike) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# validation (shared by the push tool and the Start control)
# ---------------------------------------------------------------------------


def _walk_strings(obj: Any, path: str = ""):
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            tag = v.get("name") if isinstance(v, Mapping) and "name" in v else i
            yield from _walk_strings(v, f"{path}[{tag}]")
    elif isinstance(obj, str):
        yield path, obj


def find_placeholders(cfg: Any) -> list[tuple[str, str]]:
    """``(path, token)`` for every X2TODO token and every ``x2_todo`` entry."""
    out: list[tuple[str, str]] = []
    if isinstance(cfg, Mapping):
        todo = cfg.get(TODO_LIST_KEY)
        if todo:
            items = todo if isinstance(todo, list) else [todo]
            for i, item in enumerate(items):
                out.append((f"{TODO_LIST_KEY}[{i}]", str(item)))
    for path, s in _walk_strings(cfg):
        if path.startswith(TODO_LIST_KEY):
            continue
        for m in PLACEHOLDER_RE.finditer(s):
            out.append((path, m.group(0)))
    return out


def recalibration_notes(cfg: Any) -> list[str]:
    """Entries of the informational ``x2_cp2_recalibrate`` list (never blocking)."""
    if not isinstance(cfg, Mapping):
        return []
    v = cfg.get(RECAL_LIST_KEY) or []
    return [str(x) for x in (v if isinstance(v, list) else [v])]


def find_cross_refs(cfg: Any, op: str) -> list[tuple[str, str]]:
    """References to another op-point's config files inside ``cfg``.

    An op-point with its own directory must reference only its own files
    (``configs/opoints/<op>/...``); a base-fallback op-point (1x, 1x-legacy)
    may also reference the base set.
    """
    validate_name(op)
    bad: list[tuple[str, str]] = []
    for path, s in _walk_strings(cfg):
        for m in _CONFIG_REF_RE.finditer(s):
            ref_op = m.group(1)
            ok = ref_op == op or (ref_op is None and op in BASE_FALLBACK)
            if not ok:
                bad.append((path, m.group(0)))
    return bad


def routine_arg(cfg: Mapping[str, Any], routine: str, flag: str) -> str | None:
    """Value of the LAST ``flag`` in ``routine``'s args (argparse semantics)."""
    import shlex

    for r in cfg.get("routines") or []:
        if isinstance(r, Mapping) and r.get("name") == routine:
            try:
                toks = shlex.split(str(r.get("args") or ""))
            except ValueError:
                toks = str(r.get("args") or "").split()
            val = None
            for i, t in enumerate(toks):
                if t == flag and i + 1 < len(toks):
                    val = toks[i + 1]
                elif t.startswith(flag + "="):
                    val = t.split("=", 1)[1]
            return val
    return None


def _dotted(cfg: Any, path: str) -> Any:
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_time_units(cfgs: Mapping[str, Any], op: str) -> list[str]:
    """Problems with the op-point's explicit time-unit parameters (:data:`TIME_UNITS`)."""
    problems: list[str] = []
    for spec, want in TIME_UNITS[validate_name(op)].items():
        inst = spec[0]
        cfg = cfgs.get(inst)
        if not isinstance(cfg, Mapping):
            continue  # reported as missing elsewhere
        if spec[1] == "args":
            where = f"{inst}: routines[{spec[2]}] {spec[3]}"
            got = routine_arg(cfg, spec[2], spec[3])
        else:
            where = f"{inst}: {spec[2]}"
            got = _dotted(cfg, spec[2])
        if got is None:
            problems.append(f"{where} not set explicitly (want {want:g}; code defaults are 2x)")
            continue
        if PLACEHOLDER_RE.fullmatch(str(got)):
            continue  # an unfilled placeholder is reported as a placeholder
        try:
            ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            problems.append(f"{where} = {got!r}, op-point {op} needs {want:g}")
    return problems


@dataclass
class SetReport:
    """Validation result for one op-point set (``ok`` = switchable)."""

    opoint: str
    problems: list[str] = field(default_factory=list)
    placeholders: list[tuple[str, str, str]] = field(default_factory=list)  # (instance, path, token)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.placeholders

    def summary(self, limit: int = 12) -> str:
        bits = list(self.problems)
        if self.placeholders:
            shown = ", ".join(f"{i}:{t}" for i, _p, t in self.placeholders[:limit])
            more = len(self.placeholders) - limit
            bits.append(
                f"{len(self.placeholders)} placeholder(s) unfilled: {shown}"
                + (f" (+{more} more)" if more > 0 else "")
            )
        return "; ".join(bits) if bits else "ok"


def validate_config(cfg: Any, op: str, instance: str) -> SetReport:
    rep = SetReport(op)
    if not isinstance(cfg, Mapping):
        rep.problems.append(f"{instance}: not a mapping ({type(cfg).__name__})")
        return rep
    if instance in INSTANCES and not isinstance(cfg.get("routines"), list):
        rep.problems.append(f"{instance}: no 'routines' list")
    if instance in INSTANCES:
        s = config_stamp(cfg)
        if s is not None and s != op:
            rep.problems.append(f"{instance}: stamped opoint={s!r}, expected {op!r}")
    for path, ref in find_cross_refs(cfg, op):
        rep.problems.append(f"{instance}: {path} references another op-point's file {ref}")
    for path, tok in find_placeholders(cfg):
        rep.placeholders.append((instance, path, tok))
    return rep


def validate_set(cfgs: Mapping[str, Any], op: str, *, require_stamp: bool = False,
                 check_units: bool = True) -> SetReport:
    """Validate a set ``{instance: cfg}`` (instances, plus optional c2_criteria).

    ``check_units`` also requires every :data:`TIME_UNITS` parameter to be set
    explicitly to the op-point's value.
    """
    validate_name(op)
    rep = SetReport(op)
    if check_units:
        rep.problems.extend(check_time_units(cfgs, op))
    for inst in INSTANCES:
        cfg = cfgs.get(inst)
        if cfg is None:
            rep.problems.append(f"{inst}: missing")
            continue
        if require_stamp and config_stamp(cfg) != op:
            rep.problems.append(
                f"{inst}: stamp {config_stamp(cfg)!r} != {op!r} (push it with "
                f"push_dsart_to_etcd.py --opoint {op})")
        sub = validate_config(cfg, op, inst)
        rep.problems.extend(sub.problems)
        rep.placeholders.extend(sub.placeholders)
    if "c2_criteria" in cfgs:
        sub = validate_config(cfgs["c2_criteria"], op, "c2_criteria")
        rep.problems.extend(sub.problems)
        rep.placeholders.extend(sub.placeholders)
    return rep


def load_repo_set(op: str, configs_dir: str | os.PathLike) -> dict[str, Any]:
    """``{kind: parsed yaml}`` of op-point ``op``'s files in the repo."""
    return {kind: load_yaml(config_path(kind, op, configs_dir)) for kind in CONFIG_FILES}
