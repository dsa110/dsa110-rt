"""Op-point runtime: which configuration is active, where it lives, how it is built.

Frozen interface: docs/x2/SPEC.md §10. One code version runs every op-point;
switching is configuration only. Physics parameters come from
:mod:`dsart.x2.opoint` (SPEC §1); this module adds the configuration layer.

Selections (what the operator picks; what etcd sets are keyed by)
    ``1x``          today's search, run by the unified code in 1x-compat mode;
    ``1x-legacy``   today's search exactly as deployed before x2 (retired later);
    ``2x-<kind>-ns<n>-cint<b>``  a 2x corner-turn format = ``OpPoint.opoint_id``
                    (e.g. ``2x-gridded-ns2-cint4``; the six CP1 rows are
                    ``dsart.x2.opoint.FORMATS_2X``). Both 1x selections have
                    ``opoint_id`` ``1x``.

Config sets (repo files; three per set: corr pipeline, search pipeline, C2 criteria)
    ``configs/<file>``                 the base (today's) set, byte-identical to
                                       production; used by 1x and 1x-legacy;
    ``configs/opoints/<set>/<file>``   a set's own file. ``1x`` / ``1x-legacy``
                                       fall back per file to the base; ``2x``
                                       never falls back. The 2x files are a
                                       TEMPLATE for every 2x format: routine args
                                       may use ``${var}`` (see
                                       :func:`render_context`), rendered from the
                                       op-point and the v3 plan at push time.

etcd
    /cnf/opoints/<selection>/{pipeline_rt,search_rt}  rendered sets, written by
        ``tools/ops/push_dsart_to_etcd.py --opoint <selection> [--plan ...] --write``;
    /cnf/opoints/<selection>/meta                     provenance of that push;
    /cnf/opoint                                       the active record
        ``{"opoint": <opoint_id>, "selection": <sel>, "config_set": <set>,
        "plan_id": "0x..."|None, "plan_path": ...}``; absent = selection 1x;
    /cnf/pipeline_rt, /cnf/search_rt                  live; written only by the
        web-UI Start control when it switches (fleet stopped);
    /cnf/opoints/_previous/*, /cnf/opoints/_kpark/<family>/*  switch backups
        and parked injection K.

A rendered config carries top-level ``opoint`` (op-point id), ``opoint_selection``
and, for 2x, ``plan_id`` / ``plan_path``. An unstamped config is today's base set
(it matches the 1x selections).

Placeholders
    Values not decided yet are ``X2TODO:<owner>:<what>`` tokens (they fail
    argument parsing) or entries of a top-level ``x2_todo:`` list; values carried
    over from 1x for recalibration at CP2 are listed under ``x2_cp2_recalibrate:``
    (informational). :func:`validate_set` reports problems and placeholders; the
    push refuses placeholders without --allow-placeholders, the Start control
    always refuses them.

Stdlib + PyYAML here; :func:`render_context` imports numpy only when given a plan.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from dsart.x2.opoint import FORMATS_2X, OPPOINT_1X, OpPoint, opoint_from_id

LOG = logging.getLogger("dsart.x2.opoint_runtime")

CONFIG_SETS: tuple[str, ...] = ("1x", "1x-legacy", "2x")
DEFAULT_SELECTION: str = "1x"
#: Config sets that fall back to the base config file when they have no own file.
BASE_FALLBACK: frozenset[str] = frozenset({"1x", "1x-legacy"})
#: Selections offered by the Start control: both 1x sets, then the CP1 format rows.
UI_SELECTIONS: tuple[str, ...] = ("1x", "1x-legacy") + tuple(sorted(FORMATS_2X))

ACTIVE_KEY: str = "/cnf/opoint"
SETS_PREFIX: str = "/cnf/opoints"
BACKUP_PREFIX: str = "/cnf/opoints/_previous"
INSTANCES: tuple[str, ...] = ("pipeline_rt", "search_rt")
LIVE_KEYS: dict[str, str] = {i: f"/cnf/{i}" for i in INSTANCES}

#: config kind -> file name (same name in the base dir and in each set dir).
CONFIG_FILES: dict[str, str] = {
    "pipeline_rt": "dsart_pipeline_rt.yaml",
    "search_rt": "dsart_search_rt.yaml",
    "c2_criteria": "c2_trigger_criteria.yaml",
}
_FILE_TO_KIND: dict[str, str] = {v: k for k, v in CONFIG_FILES.items()}

#: Top-level keys of a rendered config.
STAMP_KEY: str = "opoint"               # op-point id (template: the config set)
SELECTION_KEY: str = "opoint_selection"
PLAN_ID_KEY: str = "plan_id"
PLAN_PATH_KEY: str = "plan_path"
TODO_LIST_KEY: str = "x2_todo"
RECAL_LIST_KEY: str = "x2_cp2_recalibrate"

PLACEHOLDER_RE = re.compile(r"X2TODO:[A-Za-z0-9-]+:[A-Za-z0-9_./=,-]+")
TEMPLATE_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
_CONFIG_REF_RE = re.compile(
    r"configs/(?:opoints/([A-Za-z0-9.-]+)/)?"
    r"(dsart_pipeline_rt\.yaml|dsart_search_rt\.yaml|c2_trigger_criteria\.yaml)"
)

#: Injection K (S/N calibration) lives under this etcd prefix (dashboard
#: ``inject_calibration.CALIBRATION_PREFIX``). K measured at one time
#: resolution is wrong at the other, so a switch between K families parks the
#: live K of the old family and restores the parked K of the new one. The
#: family is the op-point name (1x | 2x): the 2x formats differ by < 1% S/N.
K_PREFIX: str = "/cnf/inject/snr_calibration/"
K_PARK_PREFIX: str = "/cnf/opoints/_kpark"
K_FAMILIES: tuple[str, ...] = ("1x", "2x")

#: Where the last successful etcd resolution is cached (h23 services survive an
#: etcd outage at restart on the op-point they last saw instead of 1x).
CACHE_ENV: str = "DSART_OPOINT_CACHE"
_DEFAULT_CACHE = Path.home() / ".local" / "state" / "dsart" / "opoint_last.json"


class OpPointError(RuntimeError):
    """A selection, record or set is invalid."""


# ---------------------------------------------------------------------------
# selections, sets, keys
# ---------------------------------------------------------------------------


def validate_selection(sel: str) -> str:
    """``1x``, ``1x-legacy`` or a valid 2x op-point id (round-trips)."""
    sel = str(sel).strip()
    if sel in BASE_FALLBACK:
        return sel
    try:
        op = opoint_from_id(sel)
    except (ValueError, TypeError) as exc:
        raise OpPointError(
            f"unknown op-point {sel!r} (1x, 1x-legacy or 2x-<kind>-ns<n>-cint<b>, "
            f"e.g. {sorted(FORMATS_2X)[0]}): {exc}") from None
    if op.name != "2x" or op.opoint_id != sel:
        raise OpPointError(f"op-point {sel!r} does not round-trip (got {op.opoint_id!r})")
    return sel


def config_set_of(sel: str) -> str:
    sel = validate_selection(sel)
    return sel if sel in BASE_FALLBACK else "2x"


def op_of(sel: str) -> OpPoint:
    sel = validate_selection(sel)
    return OPPOINT_1X if sel in BASE_FALLBACK else opoint_from_id(sel)


def k_family_of(sel: str) -> str:
    return op_of(sel).name


def k_park_prefix(family: str) -> str:
    if family not in K_FAMILIES:
        raise ValueError(f"unknown K family {family!r}")
    return f"{K_PARK_PREFIX}/{family}/"


def set_key(sel: str, instance: str) -> str:
    validate_selection(sel)
    if instance not in INSTANCES and instance != "meta":
        raise OpPointError(f"unknown instance {instance!r}")
    return f"{SETS_PREFIX}/{sel}/{instance}"


def meta_key(sel: str) -> str:
    return set_key(sel, "meta")


def backup_key(name: str) -> str:
    return f"{BACKUP_PREFIX}/{name}"


def is_live_key(key: str) -> bool:
    """True for keys the running system reads (never written by the push tool)."""
    return key in LIVE_KEYS.values() or key == ACTIVE_KEY


def make_active_record(sel: str, *, plan_id: str | None = None,
                       plan_path: str | None = None, **extra: Any) -> dict[str, Any]:
    sel = validate_selection(sel)
    rec = {STAMP_KEY: op_of(sel).opoint_id, "selection": sel,
           "config_set": config_set_of(sel), PLAN_ID_KEY: plan_id, PLAN_PATH_KEY: plan_path}
    rec.update(extra)
    return rec


def parse_active_record(raw: Any) -> str | None:
    """Selection from a /cnf/opoint value (``None`` = unset)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return validate_selection(raw)
    if not isinstance(raw, Mapping):
        raise OpPointError(f"{ACTIVE_KEY} holds an unrecognised value {raw!r}")
    sel = raw.get("selection")
    oid = raw.get(STAMP_KEY)
    if sel is None and oid is None:
        return None
    if sel is None:
        oid = str(oid)
        sel = oid if oid.startswith("2x") else str(raw.get("config_set") or DEFAULT_SELECTION)
    sel = validate_selection(sel)
    if oid is not None and op_of(sel).opoint_id != str(oid):
        raise OpPointError(f"{ACTIVE_KEY}: selection {sel!r} has op-point id "
                           f"{op_of(sel).opoint_id!r}, record says {oid!r}")
    return sel


def rendered_selection(cfg: Any) -> str | None:
    """The selection a config was rendered for (``None`` = unstamped base set)."""
    if isinstance(cfg, Mapping) and cfg.get(SELECTION_KEY) is not None:
        return str(cfg[SELECTION_KEY])
    return None


def stamp_matches(cfg: Any, sel: str) -> bool:
    """Does a LIVE config belong to selection ``sel``? Unstamped = base set = 1x selections."""
    s = rendered_selection(cfg)
    return (s is None and sel in BASE_FALLBACK) or s == sel


# ---------------------------------------------------------------------------
# active op-point resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveOpPoint:
    """Result of :func:`resolve_active_opoint`.

    ``source``: ``cli`` (explicit --opoint), ``etcd`` (/cnf/opoint), ``etcd-unset``
    (absent -> 1x), ``cache`` (etcd unreachable -> last seen), ``default`` (etcd
    unreachable, no cache). Services log and publish :meth:`to_dict`.
    """

    selection: str
    source: str
    record: Any = None
    warning: str = ""

    @property
    def opoint(self) -> OpPoint:
        return op_of(self.selection)

    @property
    def opoint_id(self) -> str:
        return self.opoint.opoint_id

    @property
    def config_set(self) -> str:
        return config_set_of(self.selection)

    @property
    def plan_id(self) -> str | None:
        return self.record.get(PLAN_ID_KEY) if isinstance(self.record, Mapping) else None

    @property
    def plan_path(self) -> str | None:
        return self.record.get(PLAN_PATH_KEY) if isinstance(self.record, Mapping) else None

    def to_dict(self) -> dict[str, Any]:
        return {"opoint": self.opoint_id, "selection": self.selection,
                "config_set": self.config_set, "plan_id": self.plan_id,
                "source": self.source, "warning": self.warning}


def _cache_path(cache_path: str | os.PathLike | None) -> Path:
    if cache_path is not None:
        return Path(cache_path)
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else _DEFAULT_CACHE


def _write_cache(path: Path, sel: str, record: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"selection": sel, "record": record, "unix": time.time()}))
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        LOG.warning("op-point cache write %s failed: %s", path, exc)


def _read_cache(path: Path) -> tuple[str, Any] | None:
    try:
        d = json.loads(path.read_text())
        return validate_selection(d["selection"]), d.get("record")
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
    """Resolve the selection a service should run.

    ``cli`` other than ``None``/``"auto"`` wins. Otherwise /cnf/opoint is read
    (``retries`` attempts); absent = 1x. If etcd cannot be read the last
    successfully resolved value (cache file) is used, then 1x, with a warning.
    An *invalid* record raises: that is an operator error, not something to
    paper over.
    """
    if cli not in (None, "", "auto"):
        return ActiveOpPoint(validate_selection(str(cli)), "cli")
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
        sel = parse_active_record(raw)
        if use_cache:
            _write_cache(cpath, sel or DEFAULT_SELECTION, raw)
        if sel is None:
            return ActiveOpPoint(DEFAULT_SELECTION, "etcd-unset", raw)
        return ActiveOpPoint(sel, "etcd", raw)
    cached = _read_cache(cpath) if use_cache else None
    if cached is not None:
        return ActiveOpPoint(cached[0], "cache", cached[1],
                             f"etcd unreachable ({err}); using last seen op-point {cached[0]}")
    return ActiveOpPoint(DEFAULT_SELECTION, "default", None,
                         f"etcd unreachable ({err}) and no cache; defaulting to {DEFAULT_SELECTION}")


# ---------------------------------------------------------------------------
# config files
# ---------------------------------------------------------------------------


def config_path(kind: str, sel: str, configs_dir: str | os.PathLike) -> Path:
    """Path of config ``kind`` for selection (or config set) ``sel``."""
    cset = sel if sel in CONFIG_SETS else config_set_of(sel)
    if kind not in CONFIG_FILES:
        raise OpPointError(f"unknown config kind {kind!r} (valid: {sorted(CONFIG_FILES)})")
    base = Path(configs_dir)
    own = base / "opoints" / cset / CONFIG_FILES[kind]
    if own.is_file():
        return own
    if cset in BASE_FALLBACK:
        return base / CONFIG_FILES[kind]
    raise FileNotFoundError(f"config set {cset} has no {kind} config: {own} is missing")


def map_config_path(path: str | os.PathLike, sel: str) -> Path:
    """Map a config path (as a systemd unit passes it) to selection ``sel``.

    ``.../configs/dsart_search_rt.yaml`` -> ``.../configs/opoints/2x/dsart_search_rt.yaml``
    for any 2x selection; a path already inside ``configs/opoints/<x>/`` is
    re-based. Files that are not op-point managed are returned unchanged.
    """
    p = Path(path)
    kind = _FILE_TO_KIND.get(p.name)
    if kind is None:
        return p
    base = p.parent
    if base.parent.name == "opoints":
        base = base.parent.parent
    return config_path(kind, sel, base)


def _selection_arg(value: str) -> str:
    if value == "auto":
        return value
    try:
        return validate_selection(value)
    except OpPointError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_opoint_argument(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add the standard ``--opoint`` option (SPEC §10): ``auto`` or a selection."""
    parser.add_argument(
        "--opoint", default=default, type=_selection_arg,
        help="op-point: auto (default: read /cnf/opoint once at startup), 1x, 1x-legacy "
             "or a 2x id such as 2x-gridded-ns2-cint4",
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
    path is that file's equivalent in the active selection's config set.
    """
    active = resolve_active_opoint(store, cli=cli_opoint, **resolve_kw)
    if active.warning:
        LOG.warning("op-point: %s", active.warning)
    path = map_config_path(config, active.selection)
    LOG.info("op-point %s (%s, source %s, plan %s): config %s", active.opoint_id,
             active.selection, active.source, active.plan_id, path)
    return active, path


def load_yaml(path: str | os.PathLike) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# rendering (2x template -> concrete set for one format + plan)
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def render_context(sel: str, *, plan: Any = None, plan_path: str | None = None) -> dict[str, str]:
    """Template variables for selection ``sel`` (strings).

    From the op-point: ``opoint_id``, ``n_sub``, ``n_streams`` (16 x n_sub),
    ``value_kind``, ``bits_main``, ``bits_heavy``, ``t_int_fast_native``,
    ``chan_sum_factor``, ``t_ext_us``, ``t_ext_native``, ``stage1_q_native``,
    ``n_grid``. From a v3 plan (2x): ``plan_path``, ``plan_id`` (0x%016x),
    ``n_coarse``, ``n_fine``, ``max_fine_dm``; per half h (0..7)
    ``owned_coarse_csv_h<h>`` (global coarse d, ascending), ``n_owned_h<h>``,
    ``n_owned_A_h<h>`` / ``n_owned_B_h<h>`` (per tier), ``n_fine_h<h>``; per
    search node X (n01, n02, n09, n13) ``owned_coarse_csv_X`` / ``n_owned_X``
    (union of its two halves).
    """
    op = op_of(sel)
    ctx = {
        "opoint_id": op.opoint_id, "selection": validate_selection(sel),
        "n_sub": str(op.n_sub), "n_streams": str(op.n_streams),
        "value_kind": op.value_kind, "bits_main": str(op.bits_main),
        "bits_heavy": str(op.bits_heavy), "t_int_fast_native": str(op.t_int_fast_native),
        "chan_sum_factor": str(op.chan_sum_factor), "t_ext_us": _fmt(op.t_ext_us),
        "t_ext_native": str(op.t_ext_native), "stage1_q_native": str(op.stage1_q_native),
        "n_grid": str(op.n_grid),
    }
    if plan is None:
        return ctx
    import numpy as np

    from dsart.x2.plan import SEARCH_NODES

    if plan_path is None:
        raise OpPointError("render_context: plan_path is required with a plan")
    ctx.update(plan_path=str(plan_path), plan_id=f"0x{int(plan.plan_id):016x}",
               n_coarse=str(int(plan.n_coarse)), n_fine=str(int(plan.n_fine)),
               max_fine_dm=_fmt(float(np.max(plan.fine_dm))))
    tiers = [str(t) for t in plan.tier_name]
    for h in range(int(plan.n_halves)):
        ds = [int(d) for d in plan.dms_of_half(h)]
        ctx[f"owned_coarse_csv_h{h}"] = ",".join(str(d) for d in ds)
        ctx[f"n_owned_h{h}"] = str(len(ds))
        ctx[f"n_fine_h{h}"] = str(int(np.sum(plan.n_fdm[ds]))) if ds else "0"
        for i, t in enumerate(tiers):
            ctx[f"n_owned_{t}_h{h}"] = str(sum(1 for d in ds if int(plan.coarse_tier[d]) == i))
    n_gpu = int(plan.n_halves) // len(SEARCH_NODES)
    for i, node in enumerate(SEARCH_NODES):
        ds = sorted(int(d) for h in range(i * n_gpu, (i + 1) * n_gpu) for d in plan.dms_of_half(h))
        ctx[f"owned_coarse_csv_{node}"] = ",".join(str(d) for d in ds)
        ctx[f"n_owned_{node}"] = str(len(ds))
    return ctx


def check_plan_for(sel: str, plan: Any) -> list[str]:
    """Problems if ``plan`` is not THE v3 plan of selection ``sel``'s op-point.

    Compares the plan_id with a rebuild from the op-point (plan_id ignores git
    sha and build time; value kind and bits do not change the plan).
    """
    from dsart.x2.plan import build_tiered_plan_for_opoint

    op = op_of(sel)
    if op.plan_version != 3:
        return [f"{sel} uses a v{op.plan_version} plan; no tiered plan expected"]
    want = build_tiered_plan_for_opoint(op)
    if int(plan.plan_id) != int(want.plan_id):
        return [f"plan_id 0x{int(plan.plan_id):016x} is not the {sel} plan "
                f"(rebuild gives 0x{int(want.plan_id):016x}; n_sub {plan.n_sub} vs {op.n_sub})"]
    return []


def _render_str(s: str, ctx: Mapping[str, str], missing: set[str]) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) in ctx:
            return ctx[m.group(1)]
        missing.add(m.group(1))
        return m.group(0)

    return TEMPLATE_VAR_RE.sub(sub, s)


def render(cfg: Mapping[str, Any], sel: str, ctx: Mapping[str, str],
           *, instance: str | None = None) -> dict[str, Any]:
    """Concrete config for selection ``sel`` from a set file (template or base).

    Substitutes ``${var}`` in routine ``args`` and ``hostargs`` (the only place
    variables are allowed: h23 services read the other blocks straight from
    the file), stamps ``opoint`` / ``opoint_selection`` (+ ``plan_id`` /
    ``plan_path`` when the context has them). Raises :class:`OpPointError` on
    an unknown variable, a variable outside routines, or a template stamped
    for another config set.
    """
    sel = validate_selection(sel)
    cset = config_set_of(sel)
    have = cfg.get(STAMP_KEY)
    if have is not None and str(have) not in (cset, op_of(sel).opoint_id):
        raise OpPointError(f"{instance or 'config'} is stamped opoint={have!r}, "
                           f"not config set {cset!r}")
    bad = [p for p, s in _walk_strings(cfg) if TEMPLATE_VAR_RE.search(s)
           and not re.match(r"routines\[[^\]]+\]\.(args|hostargs\.[^.]+)$", p)]
    if bad:
        raise OpPointError(f"{instance or 'config'}: template variables outside routine "
                           f"args/hostargs at {bad[:5]}")
    out = copy.deepcopy(dict(cfg))
    missing: set[str] = set()
    for r in out.get("routines") or []:
        if not isinstance(r, dict):
            continue
        if isinstance(r.get("args"), str):
            r["args"] = _render_str(r["args"], ctx, missing)
        if isinstance(r.get("hostargs"), dict):
            r["hostargs"] = {k: _render_str(v, ctx, missing) if isinstance(v, str) else v
                             for k, v in r["hostargs"].items()}
    if missing:
        raise OpPointError(f"{instance or 'config'}: unknown template variable(s) "
                           f"{sorted(missing)} for {sel}")
    out[STAMP_KEY] = op_of(sel).opoint_id
    out[SELECTION_KEY] = sel
    if "plan_id" in ctx:
        out[PLAN_ID_KEY] = ctx["plan_id"]
        out[PLAN_PATH_KEY] = ctx["plan_path"]
    return out


# ---------------------------------------------------------------------------
# validation (shared by the push tool and the Start control)
# ---------------------------------------------------------------------------


def time_units_for(op: OpPoint) -> dict[tuple, float]:
    """Every time-unit parameter a set must pass EXPLICITLY (SPEC §8/§10).

    The code defaults are 2x (``common/constants.py`` T_INT_SEARCH_US_DEFAULT
    = 524.288), so a set omitting one of these silently runs 2x units.
    ``("args", routine, flag)`` = the last ``flag`` in that routine's args;
    ``("key", dotted.path)`` = a YAML value. ``cube_sample_period_specnum``
    (search_compute) is derived from --t-int-search-us. The merger keeps 16 =
    the 1x sample in specnums at every op-point: it holds the merge window
    constant in wall-clock time (detector/merger.py; inventory finding 16);
    ``cube.sample_period_specnum`` is only the merger's fallback and stays 16
    with it. The C1 width cap is 512 native samples (16.8 ms) at every op-point.
    """
    merger_spp = OPPOINT_1X.t_ext_native // 2
    units: dict[tuple, float] = {
        ("pipeline_rt", "args", "corr_fast", "--t-int-fast-native"): op.t_int_fast_native,
        ("pipeline_rt", "args", "corr_fast", "--chan-sum-factor"): op.chan_sum_factor,
        ("search_rt", "key", "c1.merger.sample_period_specnum"): merger_spp,
        ("search_rt", "key", "c1.max_c1c2_width_samples"): 512 // op.t_ext_native,
        ("search_rt", "key", "cube.sample_period_specnum"): merger_spp,
    }
    for half in ("search_compute_0", "search_compute_1"):
        units[("search_rt", "args", half, "--t-int-search-us")] = op.t_ext_us
        units[("search_rt", "args", half, "--chan-sum-factor")] = op.chan_sum_factor
    return units


def routine_arg(cfg: Mapping[str, Any], routine: str, flag: str) -> str | None:
    """Value of the LAST ``flag`` in ``routine``'s args (argparse semantics)."""
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


def check_time_units(cfgs: Mapping[str, Any], sel: str) -> list[str]:
    """Problems with selection ``sel``'s explicit time-unit parameters."""
    op = op_of(sel)
    problems: list[str] = []
    for spec, want in time_units_for(op).items():
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
            problems.append(f"{where} not set explicitly (want {_fmt(want)}; code defaults are 2x)")
            continue
        if PLACEHOLDER_RE.fullmatch(str(got)):
            continue  # reported as a placeholder
        try:
            ok = abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            problems.append(f"{where} = {got!r}, op-point {op.opoint_id} needs {_fmt(want)}")
    return problems


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
        if path.startswith(TODO_LIST_KEY) or path.startswith(RECAL_LIST_KEY):
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


def find_cross_refs(cfg: Any, sel: str) -> list[tuple[str, str]]:
    """References to another config set's files inside ``cfg``.

    A set with its own directory (2x) must reference only its own files
    (``configs/opoints/2x/...``); a base-fallback set (1x, 1x-legacy) may also
    reference the base set.
    """
    cset = sel if sel in CONFIG_SETS else config_set_of(sel)
    bad: list[tuple[str, str]] = []
    for path, s in _walk_strings(cfg):
        for m in _CONFIG_REF_RE.finditer(s):
            ref = m.group(1)
            if not (ref == cset or (ref is None and cset in BASE_FALLBACK)):
                bad.append((path, m.group(0)))
    return bad


@dataclass
class SetReport:
    """Validation result for one selection's set (``ok`` = switchable)."""

    selection: str
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


def validate_config(cfg: Any, sel: str, instance: str, *, rendered: bool = True) -> SetReport:
    """One config. ``rendered`` = a concrete (pushed/live) config: it must carry
    the selection stamp and no template variables."""
    rep = SetReport(sel)
    if not isinstance(cfg, Mapping):
        rep.problems.append(f"{instance}: not a mapping ({type(cfg).__name__})")
        return rep
    if instance in INSTANCES and not isinstance(cfg.get("routines"), list):
        rep.problems.append(f"{instance}: no 'routines' list")
    if instance in INSTANCES and rendered:
        s = rendered_selection(cfg)
        if s != sel:
            rep.problems.append(f"{instance}: rendered for {s!r}, expected {sel!r}")
        left = sorted({m.group(0) for _p, v in _walk_strings(cfg)
                       for m in TEMPLATE_VAR_RE.finditer(v)})
        if left:
            rep.problems.append(f"{instance}: unrendered template variables {left[:6]}")
    for path, ref in find_cross_refs(cfg, sel):
        rep.problems.append(f"{instance}: {path} references another config set's file {ref}")
    for path, tok in find_placeholders(cfg):
        rep.placeholders.append((instance, path, tok))
    return rep


def validate_set(cfgs: Mapping[str, Any], sel: str, *, rendered: bool = True,
                 check_units: bool = True) -> SetReport:
    """Validate ``{instance: cfg}`` (pipeline_rt, search_rt, optional c2_criteria).

    ``rendered``: the configs are concrete (stamped for ``sel``, no ``${var}``);
    ``check_units``: every :func:`time_units_for` parameter explicit and right.
    """
    validate_selection(sel)
    rep = SetReport(sel)
    if check_units:
        rep.problems.extend(check_time_units(cfgs, sel))
    for inst in INSTANCES:
        cfg = cfgs.get(inst)
        if cfg is None:
            rep.problems.append(f"{inst}: missing")
            continue
        sub = validate_config(cfg, sel, inst, rendered=rendered)
        rep.problems.extend(sub.problems)
        rep.placeholders.extend(sub.placeholders)
    if "c2_criteria" in cfgs:
        sub = validate_config(cfgs["c2_criteria"], sel, "c2_criteria", rendered=False)
        rep.problems.extend(sub.problems)
        rep.placeholders.extend(sub.placeholders)
    return rep


def load_repo_set(sel: str, configs_dir: str | os.PathLike) -> dict[str, Any]:
    """``{kind: parsed yaml}`` of selection ``sel``'s files (unrendered)."""
    return {kind: load_yaml(config_path(kind, sel, configs_dir)) for kind in CONFIG_FILES}


def render_repo_set(sel: str, configs_dir: str | os.PathLike, *, plan: Any = None,
                    plan_path: str | None = None) -> dict[str, Any]:
    """Repo set rendered for ``sel`` (pipeline_rt, search_rt) + the C2 criteria file."""
    raw = load_repo_set(sel, configs_dir)
    ctx = render_context(sel, plan=plan, plan_path=plan_path)
    out = {inst: render(raw[inst], sel, ctx, instance=inst) for inst in INSTANCES}
    out["c2_criteria"] = raw["c2_criteria"]
    return out
