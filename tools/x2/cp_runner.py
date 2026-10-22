#!/usr/bin/env python3
"""x2 checkpoint runner: one command per track, JSON results, safe abort to "stopped in 1x".

    cp_runner.py run cp1 --track pre                 # before the window: read-only (+ guarded CPU tests)
    cp_runner.py run cp1 --track all  [--dry-run]    # window -> corr || search -> fleet -> restore
    cp_runner.py run cp1 --track corr                # or one track per terminal
    cp_runner.py abort cp1 [--dry-run]               # teardown every track, verify stopped-in-1x
    cp_runner.py check <name> ...                    # helpers the steps call (etcd / log checks)

The plan (steps, commands, budgets, gates) is in ``cp_runner_plans.py``; the runbook is
``docs/x2/CP1_RUNBOOK.md``. Everything runs from h23; node commands go over ssh.

Safety rules (2026-10-22 incident: an mmap reader of n13's RX ring cost 41.5 s of production data):
- a step may run while production observes ONLY if it is read-only (``Step.touches`` empty) or runs
  under ``tools/x2/prod_guard.sh`` on a search node; corr-node work happens only in the window;
- before EVERY step that touches GPUs, shm, NICs, services or a host setting, the runner asserts that
  production is STOPPED: etcd READY with no search orchestrator running (``check fleet-stopped``), and
  on each node of the step no production pipeline process (one without this run's marker) and no
  unmarked GPU process. A failed assertion aborts: teardown of what the runner started, then the
  restore checks; the runner never stops or starts production itself (the user does, in the web UI);
- every process the runner starts on a node carries ``DSART_X2_CP=<tag>`` in its environment; the
  teardown kills exactly those processes (and test shm named ``/dev/shm/dsart-x2-*``);
- a step whose script does not exist yet (another workstream's bench) is a PLACEHOLDER: printed in a
  dry run, a failed prerequisite (= abort) in a real run.

Bench contract (WS-C / WS-S / WS-T scripts the steps call): exit 0 = every gate passed, 1 = a gate
failed (the JSON names it), 2 = error; write results under the ``--out`` directory; name test shm
``dsart-x2-*``; tear down everything they create (``teardown`` subcommand).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[2]
_TOOLS_X2 = Path(__file__).resolve().parent
if str(_TOOLS_X2) not in sys.path:
    sys.path.insert(0, str(_TOOLS_X2))

PY_H23 = "/home/ubuntu/anaconda3/envs/dsart_h23/bin/python"
PY_NODE = "/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python"
SEARCH_NODES = ("n01", "n02", "n09", "n13")
CORR_NODES = ("n03", "n04", "n05", "n06", "n07", "n08", "n10", "n11", "n12", "n14", "n15",
              "n16", "n18", "n19", "n21", "n22")
#: Production pipeline processes (bracketed so pgrep over ssh does not match its own wrapper;
#: the orchestrator dsart_rt itself stays up while stopped and is NOT in this list).
PROD_PROC_RE = ("dsart[.]services[.](corr_fast_integration|corr_slow|search_compute|search_rx|"
                "voltage_retention)|dsart_capture_manythrea[d]|dsaX_merg[e]|meridian_fringesto[p]")
_VAR_RE = re.compile(r"@\{([A-Za-z0-9_]+)\}")
STEP_OK = ("passed", "dry-run", "skipped")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def subst(s: str, ctx: dict[str, str], *, strict: bool) -> str:
    """Replace ``@{name}``. ``strict`` raises on an unknown name; otherwise it prints ``<name?>``."""
    def rep(m: re.Match) -> str:
        k = m.group(1)
        if k in ctx and ctx[k] is not None:
            return str(ctx[k])
        if strict:
            raise KeyError(f"unset runner variable @{{{k}}}")
        return f"<{k}: set at run time>"

    return _VAR_RE.sub(rep, s)


# ---------------------------------------------------------------------------
# plan objects
# ---------------------------------------------------------------------------


@dataclass
class Command:
    """One shell command. ``host`` None = h23 (local). ``mark``: run with DSART_X2_CP=<tag>."""

    sh: str
    host: str | None = None
    timeout_s: int = 600
    mark: bool = True
    guard: bool = False      #: wrap in tools/x2/prod_guard.sh (search nodes; from the scratch copy)
    group: str | None = None  #: parallel steps: commands of one group run in order (default: host)


@dataclass
class Step:
    id: str
    track: str
    title: str
    budget_min: float
    commands: list[Command] = field(default_factory=list)
    gate: str = ""
    #: what the step touches: any of gpu, shm, nic, services, host. Non-empty => production must be
    #: STOPPED (asserted right before the step). Empty = read-only, safe while observing.
    touches: tuple[str, ...] = ()
    confirm: bool = False            #: needs an explicit yes (interactive or --yes <id>)
    manual: str = ""                 #: instructions for the user; the runner waits for "done"
    choose_format: bool = False      #: the S1 sign-off step: sets @{format}
    requires: tuple[str, ...] = ()   #: repo paths ("path" or "path::regex") that must exist
    owner: str = ""                  #: workstream that provides the required script
    in_abort: bool = True            #: restore steps: also run during an abort
    optional: bool = False           #: failure is recorded but does not abort
    parallel: bool = False           #: run command groups (per node) concurrently
    guard_hosts: tuple[str, ...] = ()  #: nodes a local driver touches (for the stopped assertion)

    def hosts(self) -> list[str]:
        return sorted({c.host for c in self.commands if c.host} | set(self.guard_hosts))


@dataclass
class Plan:
    name: str
    steps: list[Step]
    #: execution order for --track all: groups run one after another, tracks in a group in parallel
    groups: list[list[str]]
    teardown: dict[str, list[Step]]
    restore_track: str = "restore"
    before_window: tuple[str, ...] = ("pre",)

    def track(self, name: str) -> list[Step]:
        return [s for s in self.steps if s.track == name]

    def tracks(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            if s.track not in out:
                out.append(s.track)
        return out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


class Abort(Exception):
    """A step failed (or a guard refused): stop and restore."""


class Runner:
    def __init__(
        self,
        plan: Plan,
        ctx: dict[str, str],
        *,
        dry_run: bool = False,
        yes: Iterable[str] = (),
        only: Iterable[str] = (),
        start_from: str | None = None,
        results_dir: Path | None = None,
        run_fn: Callable[..., Any] = subprocess.run,
        input_fn: Callable[[str], str] = input,
        out: Callable[[str], None] = print,
        interactive: bool | None = None,
        repo: Path = REPO,
    ) -> None:
        self.plan, self.ctx, self.dry = plan, dict(ctx), dry_run
        self.yes, self.only, self.start_from = set(yes), set(only), start_from
        self.results_dir = results_dir
        self.run_fn, self.input_fn, self.out = run_fn, input_fn, out
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.repo = repo
        self.records: list[dict[str, Any]] = []
        self.started_tracks: list[str] = []
        self._lock = threading.Lock()
        self._abort = threading.Event()
        self.status = "dry-run" if dry_run else "running"
        self.started_utc = utc_now()

    # -- output / results ---------------------------------------------------------------------
    def _save(self) -> None:
        if self.dry or self.results_dir is None:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        doc = {"checkpoint": self.plan.name, "status": self.status, "started_utc": self.started_utc,
               "updated_utc": utc_now(), "ctx": self.ctx, "tracks": self.started_tracks,
               "steps": self.records}
        tmp = self.results_dir / "results.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=str))
        tmp.replace(self.results_dir / "results.json")

    def _wrap(self, c: Command, *, strict: bool) -> list[str]:
        sh = subst(c.sh, self.ctx, strict=strict)
        tag = self.ctx.get("tag", "x2cp")
        if c.host is None:
            body = f"export DSART_X2_CP={tag}; {sh}" if c.mark else sh
            return ["bash", "-c", body]
        inner = f"timeout -k 10 {int(c.timeout_s)} bash -c {shlex.quote(sh)}"
        if c.guard:
            scr = subst("@{scratch}", self.ctx, strict=strict)
            inner = f"cd {scr} && bash tools/x2/prod_guard.sh {inner}"
        if c.mark:
            inner = f"export DSART_X2_CP={tag}; {inner}"
        host = subst(c.host, self.ctx, strict=strict)
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{host}.pro.pvt", inner]

    def describe(self, c: Command) -> str:
        where = subst(c.host, self.ctx, strict=False) if c.host else "h23"
        extra = " +prod_guard" if c.guard else ""
        return f"({where}, timeout {c.timeout_s}s{extra}) {subst(c.sh, self.ctx, strict=False)}"

    def _exec(self, c: Command, log: Path | None) -> dict[str, Any]:
        argv = self._wrap(c, strict=True)
        t0 = time.monotonic()
        rec: dict[str, Any] = {"host": subst(c.host, self.ctx, strict=True) if c.host else "h23",
                               "cmd": subst(c.sh, self.ctx, strict=True)}
        try:
            cp = self.run_fn(argv, capture_output=True, text=True,
                             timeout=c.timeout_s + (30 if c.host else 0))
            rec.update(rc=cp.returncode, timed_out=False, stdout=cp.stdout or "", stderr=cp.stderr or "")
        except subprocess.TimeoutExpired as exc:
            rec.update(rc=None, timed_out=True, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or ""))
        except OSError as exc:
            rec.update(rc=None, timed_out=False, stdout="", stderr=f"{type(exc).__name__}: {exc}")
        rec["elapsed_s"] = round(time.monotonic() - t0, 1)
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as fh:
                fh.write(f"### {utc_now()} {rec['host']}: {rec['cmd']}\n{rec['stdout']}{rec['stderr']}"
                         f"### rc={rec['rc']} timed_out={rec['timed_out']} {rec['elapsed_s']}s\n")
        if "PRODUCTION_GUARD" in rec["stdout"] + rec["stderr"]:
            rec["guard_tripped"] = True
        rec["stdout_tail"] = rec.pop("stdout")[-2000:]
        rec["stderr_tail"] = rec.pop("stderr")[-2000:]
        return rec

    def _missing_prereqs(self, step: Step) -> list[str]:
        miss = []
        for req in step.requires:
            path, _, rx = req.partition("::")
            p = self.repo / path
            if not p.is_file():
                miss.append(path)
            elif rx and not re.search(rx, p.read_text(errors="replace")):
                miss.append(f"{path} (no match for /{rx}/)")
        return miss

    def guard_commands(self, step: Step) -> list[Command]:
        """The stopped-production assertion run right before a touching step."""
        if not step.touches:
            return []
        tag = self.ctx.get("tag", "x2cp")
        cmds = [Command("@{py_h23} @{repo}/tools/x2/cp_runner.py check fleet-stopped",
                        None, 60, mark=False)]
        node_check = (
            f"for p in $(pgrep -f '{PROD_PROC_RE}'); do "
            f"tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx 'DSART_X2_CP={tag}' "
            f"|| {{ echo \"production process $p: $(tr '\\0' ' ' < /proc/$p/cmdline | cut -c1-120)\"; "
            f"exit 1; }}; done")
        if "gpu" in step.touches:
            node_check += (
                "; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
                f"tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qx 'DSART_X2_CP={tag}' "
                "|| { echo \"GPU in use by unmarked pid $p\"; exit 1; }; done")
        for h in step.hosts():
            cmds.append(Command(node_check, h, 60, mark=False))
        return cmds

    def _ask(self, prompt: str, step_id: str) -> str:
        if step_id in self.yes:
            return "yes"
        if not self.interactive:
            return ""
        return self.input_fn(prompt).strip()

    def _record(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(rec)
            self._save()

    def run_step(self, step: Step, *, phase: str = "main") -> dict[str, Any]:
        rec: dict[str, Any] = {"id": step.id, "track": step.track, "phase": phase,
                               "title": step.title, "gate": step.gate, "started_utc": utc_now(),
                               "commands": [], "guard": []}
        log = None if self.results_dir is None else self.results_dir / "logs" / f"{step.id}.log"
        missing = self._missing_prereqs(step)
        if self.dry:
            rec["status"] = "dry-run"
            self._print_step(step, missing)
            self.records.append(rec)
            return rec
        self.out(f"--> [{step.id}] {step.title}")
        if missing:
            rec.update(status="missing-prereq", error=f"not available yet ({step.owner}): {missing}")
            self._record(rec)
            return rec
        if step.confirm:
            ans = self._ask(f"[{step.id}] {step.title}\n    {step.gate}\n    type '{step.id}' to run: ",
                            step.id)
            if ans not in ("yes", step.id):
                rec.update(status="declined", error="not confirmed")
                self._record(rec)
                return rec
        for g in self.guard_commands(step):
            r = self._exec(g, log)
            rec["guard"].append(r)
            if r["rc"] != 0:
                rec.update(status="refused-guard",
                           error="production is not stopped: " + (r["stdout_tail"] + r["stderr_tail"])[-300:])
                self._record(rec)
                return rec
        if step.manual:
            self.out(f"    USER ACTION: {subst(step.manual, self.ctx, strict=False)}")
            ans = self._ask("    type 'done' when finished (anything else aborts): ", step.id)
            if ans not in ("done", "yes"):
                rec.update(status="declined", error="user action not confirmed")
                self._record(rec)
                return rec
        status = self._run_commands(step, rec, log)
        if status == "passed" and step.choose_format:
            for c in rec["commands"]:
                self.out(c["stdout_tail"])
            fmt = self.ctx.get("format")
            if not fmt:
                fmt = self._ask("    S1 sign-off: type the 2x format id to use (e.g. 2x-gridded-ns2-cint4): ",
                                "__never__")
            try:
                from dsart.x2.opoint import opoint_from_id

                op = opoint_from_id(fmt)
                if op.name != "2x" or op.opoint_id != fmt:
                    raise ValueError(fmt)
                self.ctx["format"] = fmt
                rec["format"] = fmt
                self.out(f"    format = {fmt}")
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                rec["error"] = f"no valid 2x format chosen ({exc})"
        rec["status"] = status
        rec["finished_utc"] = utc_now()
        self._record(rec)
        self.out(f"    {status} ({sum(c['elapsed_s'] for c in rec['commands']):.0f}s)")
        return rec

    def _run_commands(self, step: Step, rec: dict[str, Any], log: Path | None) -> str:
        """Run the step's commands (in order, or per group in parallel); return the status."""
        def run_seq(cmds: list[Command], sink: list[dict[str, Any]]) -> str:
            for c in cmds:
                r = self._exec(c, log)
                sink.append(r)
                if r.get("guard_tripped"):
                    return "guard-tripped"
                if r["rc"] != 0:
                    return "failed"
            return "passed"

        if not step.parallel:
            return run_seq(step.commands, rec["commands"])
        groups: dict[str, list[Command]] = {}
        for c in step.commands:
            groups.setdefault(c.group or c.host or "h23", []).append(c)
        sinks: dict[str, list[dict[str, Any]]] = {g: [] for g in groups}
        stats: dict[str, str] = {}
        th = [threading.Thread(target=lambda g=g: stats.__setitem__(g, run_seq(groups[g], sinks[g])))
              for g in groups]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for g in groups:
            rec["commands"].extend(sinks[g])
        bad = {g: s for g, s in stats.items() if s != "passed"}
        if bad:
            rec["error"] = f"failed on {sorted(bad)}"
            return "guard-tripped" if "guard-tripped" in bad.values() else "failed"
        return "passed"

    def _print_step(self, step: Step, missing: list[str]) -> None:
        tags = []
        if step.touches:
            tags.append("touches " + "/".join(step.touches))
        if step.confirm:
            tags.append("CONFIRM")
        if step.manual:
            tags.append("USER ACTION")
        if step.choose_format:
            tags.append("S1 SIGN-OFF")
        if missing:
            tags.append(f"PLACEHOLDER ({step.owner}): {', '.join(missing)} not written yet")
        if step.optional:
            tags.append("optional")
        if step.parallel:
            tags.append("parallel per node")
        tmax = max([c.timeout_s for c in step.commands] or [0])
        self.out(f"[{step.id}] {step.title}  ({step.budget_min:g} min"
                 + (f", timeout {tmax}s" if tmax else "") + ")"
                 + (f"  [{'; '.join(tags)}]" if tags else ""))
        if step.gate:
            self.out(f"    gate: {subst(step.gate, self.ctx, strict=False)}")
        for g in self.guard_commands(step):
            self.out(f"    GUARD $ {self.describe(g)}")
        if step.manual:
            self.out(f"    USER: {subst(step.manual, self.ctx, strict=False)}")
        for c in step.commands:
            self.out(f"    $ {self.describe(c)}")

    def _selected(self, step: Step) -> bool:
        return not self.only or step.id in self.only

    def run_track(self, track: str) -> bool:
        steps = [s for s in self.plan.track(track) if self._selected(s)]
        if self.start_from:
            ids = [s.id for s in steps]
            if self.start_from in ids:
                steps = steps[ids.index(self.start_from):]
        if self.dry:
            budget = sum(s.budget_min for s in self.plan.track(track))
            self.out(f"\n=== track {track}  (budget {budget:g} min)")
        with self._lock:
            self.started_tracks.append(track)
        for s in steps:
            if self._abort.is_set():
                self._record({"id": s.id, "track": track, "status": "skipped",
                              "error": "abort requested by another track"})
                return False
            rec = self.run_step(s)
            if rec["status"] not in STEP_OK:
                if s.optional:
                    continue
                self._abort.set()
                return False
        return True

    def run_teardown(self, tracks: Iterable[str]) -> None:
        for t in tracks:
            steps = self.plan.teardown.get(t, [])
            if self.dry and steps:
                self.out(f"\n=== teardown {t} (on failure/abort)")
            for s in steps:
                self.run_step(s, phase="teardown")

    def run_restore(self, *, aborting: bool) -> bool:
        steps = self.plan.track(self.plan.restore_track)
        if aborting:
            steps = [s for s in steps if s.in_abort and not s.manual and not s.touches]
        if self.dry:
            self.out(f"\n=== {'abort restore' if aborting else 'track ' + self.plan.restore_track}")
        ok = True
        for s in steps:
            rec = self.run_step(s, phase="abort-restore" if aborting else "main")
            ok = ok and rec["status"] in STEP_OK
            if not ok and not aborting:
                break
        return ok

    def abort(self, reason: str) -> None:
        self.out(f"\n*** ABORT: {reason}\n*** teardown, then verify the fleet is stopped in 1x")
        self.status = "aborting"
        started = [t for t in self.started_tracks if t in self.plan.teardown] or \
            [t for t in self.plan.teardown]
        self.run_teardown(started)
        ok = self.run_restore(aborting=True)
        self.status = "aborted" + ("" if ok else " (restore checks FAILED: see results.json)")
        self._save()
        self.out(f"*** {self.status}. The fleet must be left STOPPED in 1x; start it from the web UI "
                 "(Start, op-point 1x) when the restore checks pass.")

    def run(self, tracks: list[str]) -> int:
        if self.dry:
            self.out(f"{self.plan.name.upper()} runner -- DRY RUN: every command, nothing executed")
            self.out("context: " + ", ".join(f"{k}={v}" for k, v in sorted(self.ctx.items())
                                               if k not in ("repo",)))
        try:
            if tracks == ["all"]:
                if self.dry:
                    for t in self.plan.before_window:
                        self.out(f"\n### BEFORE THE WINDOW (cp_runner.py run {self.plan.name} --track {t})")
                        self.run_track(t)
                    self.out("\n### THE WINDOW (cp_runner.py run "
                             f"{self.plan.name} --track all)")
                for group in self.plan.groups:
                    if group == [self.plan.restore_track]:
                        if self.dry:
                            self.run_teardown(self.plan.teardown)
                        if not self.run_restore(aborting=False):
                            raise Abort("restore checks failed")
                        continue
                    if len(group) == 1 or self.dry:
                        oks = [self.run_track(t) for t in group]
                    else:
                        res: dict[str, bool] = {}
                        th = [threading.Thread(target=lambda t=t: res.__setitem__(t, self.run_track(t)))
                              for t in group]
                        for x in th:
                            x.start()
                        for x in th:
                            x.join()
                        oks = [res.get(t, False) for t in group]
                    if not all(oks):
                        raise Abort(f"track(s) {[t for t, ok in zip(group, oks) if not ok]} failed")
                    if not self.dry:
                        self.run_teardown(group)
            else:
                for t in tracks:
                    if t == self.plan.restore_track:
                        if not self.run_restore(aborting=False):
                            raise Abort("restore checks failed")
                    elif not self.run_track(t):
                        raise Abort(f"track {t} failed")
                    elif not self.dry:
                        self.run_teardown([t])
        except (Abort, KeyboardInterrupt) as exc:
            if self.dry:
                return 0
            reason = str(exc) or "interrupted"
            if tracks != ["all"] and all(t in self.plan.before_window for t in tracks):
                self.status = f"failed: {reason}"     # nothing outside read-only work was done
                self._save()
                self.out(f"*** {self.status}")
                return 1
            if tracks != ["all"] and len(tracks) == 1 and tracks[0] != self.plan.restore_track:
                self.run_teardown(tracks)
                self.status = f"failed: {reason}"
                self._save()
                self.out(f"*** {self.status}; track {tracks[0]} torn down. Run "
                         f"'cp_runner.py abort {self.plan.name}' to restore the whole fleet.")
                return 1
            self.abort(reason)
            return 1
        if self.dry:
            self._print_totals()
            return 0
        self.status = "passed"
        self._save()
        return 0

    def _print_totals(self) -> None:
        b = {t: sum(s.budget_min for s in self.plan.track(t)) for t in self.plan.tracks()}
        crit = 0.0
        for group in self.plan.groups:
            crit += max(b.get(t, 0.0) for t in group)
        self.out("\n=== budgets (min): " + ", ".join(f"{t} {v:g}" for t, v in b.items()))
        self.out(f"=== window critical path {crit:g} min = {crit / 60:.1f} h "
                 f"(+ contingency; tracks in a group run in parallel: {self.plan.groups})")
        n_ph = sum(1 for s in self.plan.steps if self._missing_prereqs(s))
        self.out(f"=== {n_ph} step(s) are placeholders for scripts other workstreams still owe")


# ---------------------------------------------------------------------------
# checks (called by plan steps; also unit-tested with fakes)
# ---------------------------------------------------------------------------


def _dashboard_modules():
    dash = REPO / "tools" / "dashboard" / "dsa_monitor"
    for p in (str(REPO / "src"), str(dash)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import control_store
    import opoint_switch

    return control_store, opoint_switch


def _real_store():
    control_store, _ = _dashboard_modules()
    return control_store.ControlStore()


def check_fleet_stopped(store: Any) -> tuple[bool, str]:
    _, osw = _dashboard_modules()
    r = osw.check_fleet_stopped(store)
    if r["ok"]:
        c = r["system_state"].get("counts", {})
        return True, f"READY: corr {c.get('corr_up')}/{c.get('corr_total')} up, " \
                     f"search {c.get('search_up')}/{c.get('search_total')} up, nothing running"
    return False, r["error"]


def check_fleet_state(store: Any, want: str, *, require_all: bool = False) -> tuple[bool, str]:
    control_store, _ = _dashboard_modules()
    st = control_store.compute_system_state(store)
    c = st.get("counts", {})
    ok = st.get("state") == want
    if ok and require_all:
        ok = c.get("corr_up") == c.get("corr_total") and c.get("search_up") == c.get("search_total")
    return ok, f"{st.get('label')} ({st.get('detail')}) counts={c}"


SNAPSHOT_KEYS = ("/cnf/pipeline_rt", "/cnf/search_rt", "/cnf/opoint")


def etcd_snapshot(store: Any) -> dict[str, Any]:
    from dsart.x2 import config_diff as cdiff
    from dsart.x2 import opoint_runtime as opr

    snap: dict[str, Any] = {"utc": utc_now(), "keys": {}}
    for k in SNAPSHOT_KEYS:
        v = store.get_dict(k)
        snap["keys"][k] = None if v is None else cdiff.canonical_sha256(v)
    k = store.get_prefix_dicts(opr.K_PREFIX) if hasattr(store, "get_prefix_dicts") else {}
    snap["keys"][opr.K_PREFIX] = cdiff.canonical_sha256(k)
    snap["active"] = opr.parse_active_record(store.get_dict(opr.ACTIVE_KEY)) or opr.DEFAULT_SELECTION
    return snap


def verify_snapshot(store: Any, snap: dict[str, Any]) -> tuple[bool, list[str]]:
    now = etcd_snapshot(store)
    bad = [k for k, v in snap["keys"].items() if now["keys"].get(k) != v]
    if now["active"] not in ("1x", "1x-legacy") or now["active"] != snap.get("active", now["active"]):
        bad.append(f"/cnf/opoint is {now['active']} (was {snap.get('active')})")
    return not bad, bad


_PROGRESS_RE = re.compile(r"\(([0-9.]+) cubes/s last")
_SCATTER_RE = re.compile(r"scatter=([0-9.]+)/([0-9.]+) us")


def summarize_search_health(lines: Iterable[str], *, min_rate: float = 4.9,
                            max_scatter_ms: float = 90.0) -> dict[str, Any]:
    """Incident signature over search_compute log lines (2026-10-22 n13): lag-recovery seeks, two
    consecutive cube rates below ``min_rate`` (cadence 4.97/s), scatter mean above ``max_scatter_ms``
    (normal ~55 ms; 110-135 ms in the incident). Lines may be prefixed "<file>: " (tail -F headers)."""
    seeks = 0
    rates: list[float] = []
    scat: list[float] = []
    low_run = 0
    low_twice = False
    for ln in lines:
        if "lag-recovery seek" in ln:
            seeks += 1
        if "cube_progress" not in ln:
            continue
        m = _PROGRESS_RE.search(ln)
        if m:
            r = float(m.group(1))
            rates.append(r)
            low_run = low_run + 1 if r < min_rate else 0
            low_twice = low_twice or low_run >= 2
        m = _SCATTER_RE.search(ln)
        if m:
            scat.append(float(m.group(1)) / 1000.0)
    problems = []
    if seeks:
        problems.append(f"{seeks} lag-recovery seek(s)")
    if low_twice:
        problems.append(f"cube rate below {min_rate}/s twice in a row (min {min(rates):.2f})")
    if scat and max(scat) > max_scatter_ms:
        problems.append(f"scatter mean up to {max(scat):.0f} ms (> {max_scatter_ms:.0f})")
    if not rates:
        problems.append("no cube_progress lines")
    return {"ok": not problems, "problems": problems, "n_progress": len(rates), "seeks": seeks,
            "rate_min": min(rates) if rates else None,
            "rate_median": sorted(rates)[len(rates) // 2] if rates else None,
            "scatter_ms_max": max(scat) if scat else None,
            "scatter_ms_median": sorted(scat)[len(scat) // 2] if scat else None}


def search_health(nodes: Iterable[str], *, seconds: int, settle_s: int = 0,
                  min_rate: float = 4.9, max_scatter_ms: float = 90.0,
                  run_fn: Callable[..., Any] = subprocess.run,
                  sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Watch every search_compute log on ``nodes`` for ``seconds`` (READ-ONLY: tail -F on the logs)
    and summarise the incident signature per half."""
    if settle_s:
        sleep(settle_s)
    out: dict[str, Any] = {}
    threads = []

    def one(node: str) -> None:
        cmd = (f"for f in /tmp/dsart-rt-children/*search_compute_*.log; do :; done; "
               f"timeout {int(seconds)} tail -n0 -F /tmp/dsart-rt-children/*search_compute_*.log "
               "2>/dev/null; true")
        try:
            cp = run_fn(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{node}.pro.pvt", cmd],
                        capture_output=True, text=True, timeout=seconds + 60)
            text = cp.stdout or ""
        except (subprocess.SubprocessError, OSError) as exc:
            out[node] = {"ok": False, "problems": [f"ssh failed: {exc}"]}
            return
        halves: dict[str, list[str]] = {}
        cur = "?"
        for ln in text.splitlines():
            m = re.match(r"==> (.*) <==", ln)
            if m:
                cur = Path(m.group(1)).name
                continue
            halves.setdefault(cur, []).append(ln)
        res = {h: summarize_search_health(v, min_rate=min_rate, max_scatter_ms=max_scatter_ms)
               for h, v in halves.items()}
        if not res:
            res = {"?": summarize_search_health([], min_rate=min_rate)}
        out[node] = {"ok": all(r["ok"] for r in res.values()), "halves": res}

    for n in nodes:
        t = threading.Thread(target=one, args=(n,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return {"ok": all(v.get("ok") for v in out.values()), "nodes": out}


NIC_COLS = ("t", "rx_bytes_phy", "rx_packets_phy", "rx_discards_phy", "rx_out_of_buffer",
            "rx_crc_errors_phy", "tx_pause_ctrl_phy", "udp_rcvbuf_errors", "udp_in_errors",
            "ip_reasm_fails")


def nic_rates(text: str) -> dict[str, Any]:
    """Rates over a cp_runner_nic_sample.sh capture (first -> last sample)."""
    iface = None
    rows = []
    for ln in text.splitlines():
        if ln.startswith("# iface"):
            iface = ln.split()[2]
            continue
        parts = ln.split()
        if len(parts) >= len(NIC_COLS):
            try:
                rows.append([float(x) for x in parts[:len(NIC_COLS)]])
            except ValueError:
                continue
    if len(rows) < 2:
        return {"iface": iface, "ok": False, "error": "fewer than 2 samples"}
    a, b = rows[0], rows[-1]
    d = {k: b[i] - a[i] for i, k in enumerate(NIC_COLS)}
    dt = max(d["t"], 1e-9)
    pk = max(d["rx_packets_phy"], 1.0)
    return {"iface": iface, "ok": True, "seconds": round(dt, 1),
            "gbps": d["rx_bytes_phy"] * 8 / dt / 1e9, "pps": d["rx_packets_phy"] / dt,
            "discard_frac": d["rx_discards_phy"] / pk, "oob_frac": d["rx_out_of_buffer"] / pk,
            "crc_frac": d["rx_crc_errors_phy"] / pk, "pause_per_s": d["tx_pause_ctrl_phy"] / dt,
            "udp_rcvbuf_err_per_s": d["udp_rcvbuf_errors"] / dt,
            "reasm_fails_per_s": d["ip_reasm_fails"] / dt}


def nic_gate(base_text: str, test_text: str, *, slack: float = 1e-5) -> dict[str, Any]:
    """Relative NIC gate: discard + out-of-buffer fraction under test <= the 1x baseline + slack."""
    b, t = nic_rates(base_text), nic_rates(test_text)
    if not (b.get("ok") and t.get("ok")):
        return {"ok": False, "error": b.get("error") or t.get("error"), "base": b, "test": t}
    lb = b["discard_frac"] + b["oob_frac"]
    lt = t["discard_frac"] + t["oob_frac"]
    return {"ok": lt <= lb + slack, "loss_base": lb, "loss_test": lt, "base": b, "test": t}


def format_table(capacity: dict[str, Any] | None) -> list[str]:
    """The S1 decision table: Phase 0 numbers + the measured clean ingress (if a sweep result exists)."""
    from cp_runner_plans import FORMAT_TABLE

    lines = ["format                      total-vs-model  need/node(Gb/s, <=80%)  fits"]
    clean = None
    if capacity:
        per = capacity.get("clean_gbps_per_node") or {}
        clean = min(per.values()) if per else capacity.get("clean_gbps_min")
    best = None
    for fid, row in FORMAT_TABLE.items():
        fits = "?" if clean is None else ("yes" if row["need_gbps"] <= clean else "no")
        lines.append(f"{fid:<28}{row['total']:>8.3f}{row['need_gbps']:>20.1f}      {fits}")
        if fits == "yes":
            key = (row["total"], row["kind"] == "gridded")
            if best is None or key > best[0]:
                best = (key, fid)
    if clean is not None:
        lines.append(f"measured clean ingress (min over search nodes): {clean:.1f} Gb/s")
        lines.append("suggested (highest total that fits; gridded on ties): "
                     + (best[1] if best else "NONE fits -> escalation E4"))
    lines.append("also check the search GPU-duty aim (<= 0.6, gate 0.75) from the CP1 search track")
    return lines


def _check_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="cp_runner.py check")
    sub = p.add_subparsers(dest="what", required=True)
    sub.add_parser("fleet-stopped")
    s = sub.add_parser("fleet-state")
    s.add_argument("--want", required=True, choices=("ready", "observing", "prepared"))
    s.add_argument("--require-all", action="store_true")
    s.add_argument("--wait-s", type=float, default=0.0)
    s = sub.add_parser("snapshot")
    s.add_argument("--out", required=True)
    s = sub.add_parser("verify-snapshot")
    s.add_argument("--snapshot", required=True)
    s = sub.add_parser("search-health")
    s.add_argument("--nodes", default=",".join(SEARCH_NODES))
    s.add_argument("--seconds", type=int, default=600)
    s.add_argument("--settle-s", type=int, default=0)
    s.add_argument("--min-rate", type=float, default=4.9)
    s.add_argument("--max-scatter-ms", type=float, default=90.0)
    s.add_argument("--out", default=None)
    s = sub.add_parser("format-table")
    s.add_argument("--capacity", default=None)
    s = sub.add_parser("nic-gate")
    s.add_argument("--baseline-dir", required=True)
    s.add_argument("--test-dir", required=True)
    s.add_argument("--nodes", required=True)
    s.add_argument("--slack", type=float, default=1e-5)
    s.add_argument("--out", default=None)
    a = p.parse_args(argv)

    if a.what == "nic-gate":
        res: dict[str, Any] = {}
        for n in a.nodes.split(","):
            try:
                res[n] = nic_gate(Path(a.baseline_dir, f"{n}.txt").read_text(),
                                  Path(a.test_dir, f"{n}.txt").read_text(), slack=a.slack)
            except OSError as exc:
                res[n] = {"ok": False, "error": str(exc)}
        ok = all(v["ok"] for v in res.values())
        text = json.dumps({"ok": ok, "nodes": res}, indent=1, default=str)
        if a.out:
            Path(a.out).write_text(text)
        print(text)
        return 0 if ok else 1

    if a.what == "format-table":
        cap = None
        if a.capacity and os.path.isfile(a.capacity):
            cap = json.loads(Path(a.capacity).read_text())
        print("\n".join(format_table(cap)))
        return 0
    if a.what == "search-health":
        res = search_health(a.nodes.split(","), seconds=a.seconds, settle_s=a.settle_s,
                            min_rate=a.min_rate, max_scatter_ms=a.max_scatter_ms)
        text = json.dumps(res, indent=1)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text)
        print(text)
        return 0 if res["ok"] else 1
    store = _real_store()
    if a.what == "fleet-stopped":
        ok, msg = check_fleet_stopped(store)
    elif a.what == "fleet-state":
        deadline = time.monotonic() + a.wait_s
        while True:
            ok, msg = check_fleet_state(store, a.want, require_all=a.require_all)
            if ok or time.monotonic() >= deadline:
                break
            time.sleep(10)
    elif a.what == "snapshot":
        snap = etcd_snapshot(store)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(snap, indent=1))
        ok, msg = True, f"snapshot -> {a.out}: active {snap['active']}"
    else:
        ok, bad = verify_snapshot(store, json.loads(Path(a.snapshot).read_text()))
        msg = "etcd matches the pre-window snapshot" if ok else f"differs: {bad}"
    print(("OK " if ok else "FAIL ") + msg)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _git_short(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "nosha"
    except (OSError, subprocess.SubprocessError):
        return "nosha"


def build_ctx(a: argparse.Namespace) -> dict[str, str]:
    tag = a.tag or f"{a.checkpoint}_{_git_short(REPO)}"
    root = Path(a.results_root).expanduser()
    return {
        "checkpoint": a.checkpoint, "tag": tag, "repo": str(REPO),
        "scratch": f"/home/ubuntu/x2_scratch/{tag}", "state": str(root / tag),
        "py_h23": PY_H23, "py": PY_NODE, "pyflakes": "/home/ubuntu/anaconda3/envs/casa38/bin/pyflakes",
        "corr_node": a.corr_node, "search_node": a.search_node,
        "live_format": a.live_format, "format": a.format or None,
        "corr_nodes": " ".join(CORR_NODES), "search_nodes": " ".join(SEARCH_NODES),
        "corr_nodes_csv": ",".join(CORR_NODES), "search_nodes_csv": ",".join(SEARCH_NODES),
        "all_nodes": " ".join(CORR_NODES + SEARCH_NODES),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":
        return _check_main(argv[1:])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("run", "abort", "list"))
    p.add_argument("checkpoint", choices=("cp1",))
    p.add_argument("--track", default="all",
                   help="pre | window | corr | search | fleet | restore | all (default all)")
    p.add_argument("--dry-run", action="store_true", help="print every command, execute nothing")
    p.add_argument("--yes", action="append", default=[], metavar="STEP",
                   help="pre-approve a CONFIRM / USER ACTION step (repeatable)")
    p.add_argument("--only", action="append", default=[], metavar="STEP")
    p.add_argument("--from", dest="start_from", default=None, metavar="STEP")
    p.add_argument("--corr-node", default="n04")
    p.add_argument("--search-node", default="n09")
    p.add_argument("--live-format", default="2x-gridded-ns2-cint4",
                   help="2x format for the single-node live/engine steps (heaviest gridded by default)")
    p.add_argument("--format", default="", help="pre-chosen S1 format (else asked at fleet.format_choice)")
    p.add_argument("--tag", default="")
    p.add_argument("--results-root", default="~/x2_checkpoints")
    p.add_argument("--plan-json", default=None, help="also write the plan as JSON")
    a = p.parse_args(argv)

    from cp_runner_plans import build_plan

    ctx = build_ctx(a)
    plan = build_plan(a.checkpoint)
    if a.plan_json:
        Path(a.plan_json).write_text(json.dumps(
            {"name": plan.name, "groups": plan.groups, "steps": [asdict(s) for s in plan.steps],
             "teardown": {t: [asdict(s) for s in v] for t, v in plan.teardown.items()}}, indent=1))
    dry = a.dry_run or a.action == "list"
    results = None if dry else Path(ctx["state"]) / f"{a.action}_{a.track}_{utc_now().replace(':', '')}"
    r = Runner(plan, ctx, dry_run=dry, yes=a.yes, only=a.only, start_from=a.start_from,
               results_dir=results)
    if a.action == "abort":
        if dry:
            r.out(f"{plan.name.upper()} abort -- DRY RUN")
            r.started_tracks = list(plan.teardown)
            r.run_teardown(plan.teardown)
            r.run_restore(aborting=True)
            return 0
        r.started_tracks = list(plan.teardown)
        r.abort("operator requested abort")
        return 0
    tracks = [t.strip() for t in a.track.split(",") if t.strip()]
    valid = set(plan.tracks()) | {"all"}
    bad = [t for t in tracks if t not in valid]
    if bad:
        p.error(f"unknown track(s) {bad}; valid: {sorted(valid)}")
    if results:
        r.out(f"results: {results}/results.json")
    return r.run(tracks)


if __name__ == "__main__":
    sys.exit(main())
