"""C3 — voltage-dump collector + cube-veto service (h23).

The dsa110-rt replacement for the legacy ``dsa110-T3`` voltage handler.
C3 watches the candidate archive C2 materialises and, for each completed
event:

  1. runs the cube-morphology veto (:mod:`dsart.coinc.cube_veto`),
     ALWAYS keeping injections and anything ambiguous;
  2. KEEP   → collects the 16 per-corr staged voltage fragments into
     ``<cand>/Level2/voltages/`` (:mod:`dsart.coinc.voltage_collect`),
     writes a collection report, and (optionally) frees the corr staging;
  3. REJECT → conservative cleanup: tell the corr nodes to delete their
     staged voltages, delete the bulky ``cubes/*.npz``, and MOVE the
     event's metadata + plots to ``candidates_rejected/<event>/`` with a
     full ``REJECT.json`` audit. Never an outright ``rm -rf``.

Rollout safety: ``flag_only`` (default **True**) makes C3 log every
decision + write the audit sidecar but skip all destructive REJECT
actions — the report's "flag-first for a few days, confirm 0 real losses,
then enable auto-delete" plan. Voltage collection (non-destructive) still
runs in flag-only mode.

See ``docs/voltage_dumps/VOLTAGE_DUMP_C3_DESIGN.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from ..coinc.broadcast import DEFAULT_VOLTAGE_PORT, VoltageBroadcaster
from ..coinc.cube_veto import (
    CubeVetoThresholds,
    compute_metrics,
    decide,
    event_is_injection,
)
from ..coinc.filterbank import FilterbankConfig, run_for_event
from ..coinc.voltage_collect import (
    CorrNode,
    collect_fragments,
    collection_done,
    plan_fragments,
    rsync_pull,
)

LOG = logging.getLogger("dsart.services.c3")

__all__ = ["C3Config", "C3Service", "C3_FLAG_ONLY_KEY", "main"]

#: Runtime override key for the keep/delete mode, written by the
#: dashboard Control tab ("C3 reject mode" panel) as
#: ``{"flag_only": bool, "ts": float, "actor": str, "reason": str}``.
#: When the key is present C3 uses its ``flag_only`` value on every
#: event; when it is absent (or malformed, or etcd is unreachable) C3
#: falls back to the configured / CLI default (which itself defaults to
#: the safe ``flag_only=True``). This makes the destructive REJECT path
#: a deliberate, audited, fail-safe flip rather than a config redeploy.
#: Mirrored by ``tools/dashboard/dsa_monitor/voltage_controls.py``;
#: drift is caught by ``tests/test_voltage_controls.py``.
C3_FLAG_ONLY_KEY: str = "/cmd/c3/flag_only"


@dataclass(frozen=True)
class C3Config:
    archive_root: Path = Path("/dataz/dsa110/candidates")
    rejected_root: Path = Path("/dataz/dsa110/candidates_rejected")
    state_path: Path = Path("/dataz/dsa110/operations/c3/c3_state.json")
    # Durable, never-expiring fired-injection log (written by the
    # dashboard's publish_active_inject). Used as a registry-independent
    # backstop so C3 exempts injections the live C2 matcher missed at
    # fire time. None disables the coincidence fallback (L3 marker + CSV
    # inj_id checks still run).
    fired_injection_log: Optional[Path] = Path(
        "/dataz/dsa110/operations/inject/fired_injections.jsonl"
    )
    staging_dir: str = "/home/ubuntu/data/voltage_staging"
    corr_nodes: Mapping[int, CorrNode] = field(default_factory=dict)
    voltage_port: int = DEFAULT_VOLTAGE_PORT
    expected_cube_count: int = 8
    scan_interval_s: float = 10.0
    collect_timeout_s: float = 1800.0
    collect_poll_s: float = 15.0
    collect_min_fragments: int = 8       # legacy MIN_CT
    # Give-up window when NO fragment shows up at all (event was never
    # dumped — e.g. voltages disabled, or trigger predates the corr
    # retention window). Bounds the per-event wait so the scan loop can
    # never wedge on an uncollectable KEEP. Once >=1 fragment lands the
    # full collect_timeout_s / collect_min_fragments logic takes over.
    collect_no_show_grace_s: float = 120.0
    rsync_bwlimit_kbps: int = 0
    cleanup_staging_after_keep: bool = True
    flag_only: bool = True               # conservative default
    mon_key: str = "/mon/c3/h23"
    mon_interval_s: float = 5.0
    veto: CubeVetoThresholds = field(default_factory=CubeVetoThresholds)
    # M8+bbproc: coherent filterbank + inspection plot per KEEP event
    # (dsa110-bbproc toolkit; runs synchronously in the scan loop so
    # filterbank jobs are strictly serialized).
    filterbank: FilterbankConfig = field(default_factory=FilterbankConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "C3Config":
        with Path(path).open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        c3 = doc.get("c3", {}) or {}
        corr_raw = c3.get("corr_nodes", {}) or {}
        corr: Dict[int, CorrNode] = {}
        for chg, spec in corr_raw.items():
            corr[int(chg)] = CorrNode(
                chgroup=int(chg),
                ssh_host=str(spec["ssh_host"]),
                udp_ip=str(spec["udp_ip"]),
            )
        vt = c3.get("veto", {}) or {}
        th = CubeVetoThresholds(
            r1_img_off_px=float(vt.get("r1_img_off_px", 40.0)),
            r2_t_shift_samples=int(vt.get("r2_t_shift_samples", 30)),
            r3_dm_shift_trials=int(vt.get("r3_dm_shift_trials", 8)),
            r4_time_prom_sigma=float(vt.get("r4_time_prom_sigma", 4.0)),
            r4_image_prom_sigma=float(vt.get("r4_image_prom_sigma", 4.0)),
            r5_dm_shift_trials=int(vt.get("r5_dm_shift_trials", 4)),
            r10_tz_trig_sigma=float(vt.get("r10_tz_trig_sigma", 10.0)),
        )
        fired_log_raw = c3.get(
            "fired_injection_log",
            "/dataz/dsa110/operations/inject/fired_injections.jsonl",
        )
        return cls(
            archive_root=Path(c3.get("archive_root", "/dataz/dsa110/candidates")),
            rejected_root=Path(c3.get(
                "rejected_root", "/dataz/dsa110/candidates_rejected")),
            state_path=Path(c3.get(
                "state_path", "/dataz/dsa110/operations/c3/c3_state.json")),
            fired_injection_log=(
                Path(fired_log_raw) if fired_log_raw else None
            ),
            staging_dir=str(c3.get(
                "staging_dir", "/home/ubuntu/data/voltage_staging")),
            corr_nodes=corr,
            voltage_port=int(c3.get("voltage_port", DEFAULT_VOLTAGE_PORT)),
            expected_cube_count=int(c3.get("expected_cube_count", 8)),
            scan_interval_s=float(c3.get("scan_interval_s", 10.0)),
            collect_timeout_s=float(c3.get("collect_timeout_s", 1800.0)),
            collect_poll_s=float(c3.get("collect_poll_s", 15.0)),
            collect_min_fragments=int(c3.get("collect_min_fragments", 8)),
            collect_no_show_grace_s=float(
                c3.get("collect_no_show_grace_s", 120.0)),
            rsync_bwlimit_kbps=int(c3.get("rsync_bwlimit_kbps", 0)),
            cleanup_staging_after_keep=bool(
                c3.get("cleanup_staging_after_keep", True)),
            flag_only=bool(c3.get("flag_only", True)),
            mon_key=str(c3.get("mon_key", "/mon/c3/h23")),
            mon_interval_s=float(c3.get("mon_interval_s", 5.0)),
            veto=th,
            filterbank=FilterbankConfig.from_dict(c3.get("filterbank")),
        )


class _StoreWrapper:
    def __init__(self, mock: Optional[Any] = None) -> None:
        if mock is not None:
            self._store, self._available = mock, True
            return
        try:
            from dsautils.dsa_store import DsaStore  # noqa: WPS433
            self._store, self._available = DsaStore(), True
        except Exception as exc:  # noqa: BLE001
            LOG.warning("DsaStore unavailable (%s); mon export disabled", exc)
            self._store, self._available = None, False

    def put_dict(self, key: str, value: Mapping[str, Any]) -> None:
        if not self._available or self._store is None:
            return
        try:
            self._store.put_dict(key, dict(value))
        except Exception:  # noqa: BLE001
            LOG.exception("etcd put_dict(%s) failed", key)

    def get_dict(self, key: str) -> Optional[Any]:
        if not self._available or self._store is None:
            return None
        try:
            return self._store.get_dict(key)
        except Exception:  # noqa: BLE001
            LOG.exception("etcd get_dict(%s) failed", key)
            return None


class C3Service:
    def __init__(
        self,
        config: C3Config,
        *,
        mon_store: Optional[Any] = None,
        broadcaster: Optional[VoltageBroadcaster] = None,
    ) -> None:
        self._cfg = config
        self._mon_store = _StoreWrapper(mock=mon_store)
        self._broadcaster: Optional[VoltageBroadcaster] = broadcaster
        if self._broadcaster is None and config.corr_nodes:
            self._broadcaster = VoltageBroadcaster(
                {chg: n.udp_ip for chg, n in config.corr_nodes.items()},
                port=config.voltage_port,
            )
        self._processed: Dict[str, str] = {}   # event_name -> action
        self._stop = asyncio.Event()
        self._counters: Dict[str, int] = {
            "scanned": 0,
            "kept": 0,
            "rejected": 0,
            "rejected_flagged_only": 0,
            "veto_errors": 0,
            "fragments_collected": 0,
            "filterbanks_ok": 0,
            "filterbanks_failed": 0,
        }
        self._load_state()

    # ----- state -------------------------------------------------------

    def _load_state(self) -> None:
        p = self._cfg.state_path
        if p.is_file():
            try:
                with p.open("r") as fh:
                    self._processed = dict(json.load(fh).get("processed", {}))
            except (OSError, ValueError):
                LOG.warning("could not read C3 state %s; starting fresh", p)

    def _save_state(self) -> None:
        p = self._cfg.state_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with tmp.open("w") as fh:
                json.dump({"processed": self._processed}, fh, indent=2)
            tmp.replace(p)
        except OSError as exc:
            LOG.warning("could not write C3 state %s: %s", p, exc)

    # ----- discovery ---------------------------------------------------

    def _completed_events(self) -> List[str]:
        """Event names with a Level3 JSON (C2 writes it last, only on a
        complete cube set) that C3 has not processed yet."""
        root = self._cfg.archive_root
        out: List[str] = []
        if not root.is_dir():
            return out
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            if name in self._processed:
                continue
            if (d / "Level3" / f"{name}.json").is_file():
                out.append(name)
        return out

    # ----- runtime mode ------------------------------------------------

    def _effective_flag_only(self) -> bool:
        """Resolve the live keep/delete mode.

        Returns the dashboard's ``/cmd/c3/flag_only`` override when that
        key is present and well-formed, else the configured / CLI default
        (:attr:`C3Config.flag_only`). Fail-safe in every direction: a
        missing key, a malformed payload, or an etcd read error all keep
        the configured value — and since that defaults to ``True``, the
        destructive REJECT path stays OFF unless an operator has
        deliberately (and auditably) flipped the key to ``flag_only:
        false``.
        """
        doc = self._mon_store.get_dict(C3_FLAG_ONLY_KEY)
        if isinstance(doc, Mapping) and "flag_only" in doc:
            return bool(doc["flag_only"])
        return self._cfg.flag_only

    # ----- per-event processing ----------------------------------------

    def process_event(self, name: str) -> Dict[str, Any]:
        ev_dir = self._cfg.archive_root / name
        self._counters["scanned"] += 1
        override = self._mon_store.get_dict(C3_FLAG_ONLY_KEY)
        if isinstance(override, Mapping) and "flag_only" in override:
            flag_only = bool(override["flag_only"])
            flag_only_source = "etcd_override"
        else:
            flag_only = self._cfg.flag_only
            flag_only_source = "config"
        is_inj = event_is_injection(
            ev_dir, name, fired_log_path=self._cfg.fired_injection_log,
        )
        metrics = compute_metrics(ev_dir, name)
        if not metrics.ok:
            self._counters["veto_errors"] += 1
        decision = decide(metrics, is_injection=is_inj, thresholds=self._cfg.veto)
        rec: Dict[str, Any] = {
            "event_name": name,
            "decided_at_unix": time.time(),
            "is_injection": is_inj,
            "action": decision.action,
            "rules_fired": list(decision.rules_fired),
            "notes": decision.notes,
            "flag_only": flag_only,
            "flag_only_source": flag_only_source,
            "metrics": metrics.__dict__,
        }
        if decision.keep:
            rec.update(self._do_keep(name, ev_dir))
            self._counters["kept"] += 1
        else:
            rec.update(self._do_reject(name, ev_dir, decision, flag_only))
        # Always drop an audit sidecar in the (still-present or moved) dir.
        self._write_audit(name, ev_dir, rec)
        self._processed[name] = decision.action + (
            "(flagged)" if (not decision.keep and flag_only) else ""
        )
        self._save_state()
        LOG.info(
            "C3 %s name=%s inj=%s rules=%s%s",
            decision.action, name, is_inj, list(decision.rules_fired),
            " [flag-only]" if (not decision.keep and flag_only) else "",
        )
        return rec

    def _do_keep(self, name: str, ev_dir: Path) -> Dict[str, Any]:
        """Collect voltage fragments (non-destructive)."""
        dest = ev_dir / "Level2" / "voltages"
        plans = plan_fragments(
            event_name=name,
            corr_nodes=self._cfg.corr_nodes,
            staging_dir=self._cfg.staging_dir,
            dest_dir=dest,
        )
        n_total = len(plans)
        present: Dict[int, bool] = {}
        n_present = 0
        t0 = time.monotonic()
        if n_total:
            while not self._stop.is_set():
                present, n_present = collect_fragments(
                    plans,
                    pull_fn=lambda r, d: rsync_pull(
                        r, d, bwlimit_kbps=self._cfg.rsync_bwlimit_kbps),
                )
                elapsed = time.monotonic() - t0
                if collection_done(
                    n_present=n_present, n_total=n_total,
                    min_fragments=self._cfg.collect_min_fragments,
                    elapsed_s=elapsed, timeout_s=self._cfg.collect_timeout_s,
                    no_show_grace_s=self._cfg.collect_no_show_grace_s,
                ):
                    break
                time.sleep(self._cfg.collect_poll_s)
        self._counters["fragments_collected"] += n_present
        report = {
            "n_fragments_present": n_present,
            "n_fragments_total": n_total,
            "present_by_chgroup": {str(k): v for k, v in present.items()},
        }
        try:
            with (ev_dir / "Level3" / f"{name}_voltages.json").open("w") as fh:
                json.dump(report, fh, indent=2)
        except OSError as exc:
            LOG.warning("could not write voltage report for %s: %s", name, exc)

        # bbproc coherent filterbank + inspection plot (synchronous, so
        # events queue one after another; best-effort — a filterbank
        # failure never affects the KEEP).
        fb_report: Dict[str, Any] = {"skipped": "disabled"}
        if self._cfg.filterbank.enabled and n_present > 0:
            c2row: Dict[str, Any] = {}
            l3p = ev_dir / "Level3" / f"{name}.json"
            try:
                with l3p.open("r") as fh:
                    c2row = (json.load(fh) or {}).get("c2", {}) or {}
            except (OSError, json.JSONDecodeError) as exc:
                LOG.warning("filterbank %s: could not read %s: %s — "
                            "beamforming at (l,m)=(0,0)", name, l3p, exc)
            dec_deg = None
            if self._cfg.filterbank.dec_key:
                try:
                    dd = self._mon_store.get_dict(
                        self._cfg.filterbank.dec_key) or {}
                    dec_deg = float(dd["dec_deg"])
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "filterbank %s: no pointing dec from %s (%s) — "
                        "the F21 DEC fringe-stop will be skipped and the "
                        "beamform will be decorrelated", name,
                        self._cfg.filterbank.dec_key, exc)
            fb_report = run_for_event(
                self._cfg.filterbank, ev_dir, name, c2row, dec_deg=dec_deg)
            if fb_report.get("ok"):
                self._counters["filterbanks_ok"] += 1
            else:
                self._counters["filterbanks_failed"] += 1

        if (
            n_present > 0
            and self._cfg.cleanup_staging_after_keep
            and self._broadcaster is not None
        ):
            # Voltages are safely in the candidate dir → free corr NVMe.
            self._broadcaster.broadcast(name, 0, 0.0)
        return {"keep": report, "filterbank": fb_report}

    def _do_reject(
        self, name: str, ev_dir: Path, decision, flag_only: bool,
    ) -> Dict[str, Any]:
        """Conservative cleanup (gated by the live ``flag_only`` mode)."""
        if flag_only:
            self._counters["rejected_flagged_only"] += 1
            return {"reject": {"flag_only": True, "no_action_taken": True}}

        self._counters["rejected"] += 1
        actions: Dict[str, Any] = {}

        # 1. tell corr nodes to delete staged voltages
        if self._broadcaster is not None:
            actions["delete_broadcast"] = self._broadcaster.broadcast(
                name, 0, 0.0)

        # 2. delete bulky cubes
        n_cubes_deleted = 0
        cubes_dir = ev_dir / "cubes"
        if cubes_dir.is_dir():
            for cf in cubes_dir.glob("*.npz"):
                try:
                    cf.unlink()
                    n_cubes_deleted += 1
                except OSError as exc:
                    LOG.warning("could not delete cube %s: %s", cf, exc)
        actions["cubes_deleted"] = n_cubes_deleted

        # 3. move metadata + plots aside (never rm the dir)
        rej_dir = self._cfg.rejected_root / name
        try:
            rej_dir.mkdir(parents=True, exist_ok=True)
            for sub in ("Level2", "Level3"):
                src = ev_dir / sub
                if src.exists():
                    shutil.move(str(src), str(rej_dir / sub))
            # leave the (now cube-less / voltage-less) original dir as a
            # tombstone; remove only if it is empty of bulky data.
            actions["moved_to"] = str(rej_dir)
            self._write_audit(name, rej_dir, {"event_name": name})
        except OSError as exc:
            LOG.warning("reject move failed for %s: %s", name, exc)
            actions["move_error"] = str(exc)
        return {"reject": actions}

    def _write_audit(self, name: str, where: Path, rec: Dict[str, Any]) -> None:
        try:
            where.mkdir(parents=True, exist_ok=True)
            fname = "C3_decision.json" if rec.get("action") else "REJECT.json"
            with (where / fname).open("w") as fh:
                json.dump(rec, fh, indent=2, default=str)
        except OSError as exc:
            LOG.warning("could not write C3 audit for %s: %s", name, exc)

    # ----- loops -------------------------------------------------------

    async def _scan_loop(self) -> None:
        try:
            while not self._stop.is_set():
                for name in self._completed_events():
                    if self._stop.is_set():
                        break
                    try:
                        await asyncio.to_thread(self.process_event, name)
                    except Exception:  # noqa: BLE001
                        LOG.exception("C3 process_event(%s) failed", name)
                        self._processed[name] = "ERROR"
                        self._save_state()
                await asyncio.sleep(self._cfg.scan_interval_s)
        except asyncio.CancelledError:
            return

    async def _mon_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._mon_store.put_dict(self._cfg.mon_key, {
                    "ts_unix": time.time(),
                    "counters": dict(self._counters),
                    "n_processed": len(self._processed),
                    "flag_only": self._effective_flag_only(),
                    "flag_only_config": self._cfg.flag_only,
                    "n_corr_nodes": len(self._cfg.corr_nodes),
                })
                await asyncio.sleep(self._cfg.mon_interval_s)
        except asyncio.CancelledError:
            return

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        LOG.info(
            "C3 up: archive=%s rejected=%s flag_only=%s corr_nodes=%d",
            self._cfg.archive_root, self._cfg.rejected_root,
            self._cfg.flag_only, len(self._cfg.corr_nodes),
        )
        tasks = [
            asyncio.create_task(self._scan_loop(), name="c3-scan"),
            asyncio.create_task(self._mon_loop(), name="c3-mon"),
        ]
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._broadcaster is not None:
                self._broadcaster.close()
        return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=Path(
            "/home/ubuntu/vikram/dev/dsa110-rt/configs/dsart_search_rt.yaml"),
        help="YAML config; reads the top-level 'c3:' block.",
    )
    p.add_argument("--flag-only", dest="flag_only", action="store_true",
                   default=None, help="Force flag-only (no destructive REJECT).")
    p.add_argument("--no-flag-only", dest="flag_only", action="store_false",
                   help="Enable destructive REJECT cleanup.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = C3Config.from_yaml(args.config)
    if args.flag_only is not None:
        cfg = C3Config(**{**cfg.__dict__, "flag_only": bool(args.flag_only)})
    svc = C3Service(cfg)
    return asyncio.run(svc.run())


if __name__ == "__main__":
    sys.exit(main())
