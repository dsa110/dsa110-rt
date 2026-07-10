# Research packet 02 — Operating the system: verbs, injections, voltage dumps, dashboard, monitoring

Citations file:line, paths relative to `/home/ubuntu/vikram/dev/dsa110-rt` unless noted.

## 1. Orchestrator (dsart_rt) — keys and verbs

- Config keys read at startup: `/cnf/pipeline_rt` (corr) / `/cnf/search_rt` (search), pushed by `tools/ops/push_dsart_to_etcd.py` from `configs/dsart_pipeline_rt.yaml` / `configs/dsart_search_rt.yaml` (dsart_rt.py:5-8,593-599). Config loads once per `start` — a push takes effect at next start.
- Command keys: `/cmd/corr_rt/<n>`, `/cmd/search_rt/<n>` + broadcast `/cmd/<ns>/0` (dsart_rt.py:1296-1305,559-566). Payload: `{"cmd": "<verb>", "val": <any>}`.
- Verbs (dsart_rt.py:32-47,474-476):
  - `start` (val = observing dec deg for CUSTOMDEC, or None → resolve from `/mon/array/dec.dec_deg`): state→starting, reload config, load SPL cfg, create buffers (`dada_db -k KEY -b B -n N -l -p [-c C] [-r R]`; destroy+recreate on failure — dsart_rt.py:146-161,890-905), spawn routines in two waves (wave 1 compute ungated; wave 2 capture gated on sentinel files `gate_on_paths`, timeout 240 s / DSART_RT_GATE_TIMEOUT_S) (dsart_rt.py:944-1047,707-755).
  - `stop`: SIGTERM process group, 5 s grace, SIGKILL; destroy buffers reverse order (dsart_rt.py:1155-1182).
  - `utc_start` (val = first specnum): UDP `UTC_START-<seq>` to 127.0.0.1:11223 & :11224; arms observation watchdog; writes `/mon/snap/1/utc_start_rt = {"val": seq}`, refreshes `/mon/snap/1/armed_mjd` = now, `/mon/snap/1/utc_start` = 0 (dsart_rt.py:771-824,1207-1218).
  - `utc_stop` (val=int): `UTC_STOP-<seq>`; disarms watchdog.
  - Accepted-but-logged no-ops: `record`, `trigger`, `ctrltrigger`, `inject`, `reload_cal`, `reload_flagants` (dsart_rt.py:475-476,674-676).
- Token substitution in routine argv/env: CUSTOMDEC, CHGROUP (0..15), CALSB (sb%02d), CN, SPL_INTEGRATION_S/SPL_NFREQ_INT from `/cnf/spectral_line` (dsart_rt.py:1110-1137).
- Observation-time watchdog (dsart_rt.py:834-886): reads `/cmd/operator/control.max_obs_seconds` (15 s cache); elapsed ≥ cap → auto `utc_stop(0)`, logs `/mon/operator/watchdog/<instance>`. Enforced in orchestrator, independent of the operator agent.
- Heartbeats: `/mon/service/<ns>/<n>` = {cadence, time_mjd, state}; full mon dict `/mon/<ns>/<n>` = {instance, cn, host, state, uptime_s, routines{name:{pid,alive}}, buffers{key:{metric: dada_dbmetric}}, last_verb, spectral_line} every --mon-cadence-s (default 2 s) (dsart_rt.py:1220-1283).
- CLI: `python -m dsart.services.dsart_rt -in pipeline_rt|search_rt -cn <id> [--config-key] [--mon-cadence-s 2.0] [--namespace-prefix]` (dsart_rt.py:1320-1347).

## 2. Operator tooling

- `tools/ops/push_dsart_to_etcd.py --instance all|pipeline_rt|search_rt [--dry-run]`.
- `tools/ops/dsart-rt` — canonical operator CLI (bash):
  - `dsart-rt services {install|up|down|restart|status} [--corr LIST] [--search LIST]` — ssh `systemctl --user ... dsart-rt` fleet-wide over `${host}.pro.pvt`. Default fleet: corr = n03 n04 n05 n06 n07 n08 n10 n11 n12 n14 n15 n16 n18 n19 n21 n22; search = n01 n02 n09 n13.
  - `dsart-rt pipeline {start|stop|status} [--dec D]` — start sends {"cmd":"start","val":dec} to every node; status reads /mon.
  - `dsart-rt verb send VERB [--val V] [--corr LIST] [--search LIST]` — e.g. `dsart-rt verb send utc_start --val 1234567 --corr n06`.
  - `dsart-rt mon show`; `dsart-rt push-config`.
- Dev script `tools/ops/_m72_send_verbs.py` (raw etcd PUTs, OBS_DEC=53.85, CORR_CNS=(3,4,5,6,7,8,10,11,12,14,15,16,18,19,21,22)).
- In practice fleet ops go through the **dsa_monitor dashboard** (tools/dashboard/dsa_monitor/app.py, 2663 lines). POST routes (all with typed confirmation + reason, audited to `/mon/audit/control/...`): /control/start (obs_dec_deg), /control/stop (confirm=stop), /control/utc_start (margin, computes ARM_SEQ from capture last_seq_no), /control/utc_stop, /control/restart_all (confirm=restart_all, async 202+job_id, poll /control/system_state), /control/bounce_search (cn_ids), GET /control/system_state (fleet traffic-light).
- `dsa110-operator/src/dsa_operator/control/verbs.py` mirrors all of it as named Plans (start_fleet, stop_fleet, utc_start, utc_stop, restart_all, bounce_search, point_array, build_fstable, deploy_fstable, set_spectral_line, fire_injection, inject_calibrate, set_dumps_enabled, dump_now, delete_snr_cal, update_fleet_code, set_policy) — single source of truth for the dashboard POSTs.

## 3. Injections — operator procedure

- Dashboard Control tab panel `#panel-inject` (templates/control.html:896-1006): form inj_id, dm_pc_cm3, target_snr OR fluence_jy_ms, width_samples (1..4096), profile (gaussian|boxcar), l_rad, m_rad, optional apply_at_specnum, margin_blocks, chgroups → POST /control/inject (app.py:891-1027). target_snr requires a stored per-DM-bucket calibration (HTTP 412: "run /control/inject_calibrate first").
- control_inject_pulse (dsa_monitor/control_store.py:931-1052): auto-arm apply_at = max(block_specnum_start over /mon/corr_rt/<cn>/corr_fast publishers) + margin_blocks × NPACKETS_PER_BLOCK (default margin 16). Fails clearly if no corr_fast with --inject-watch responds. Fan-out per chgroup (default all 16):
  - key `/cmd/dsart/corr/<chgroup>/inject`, payload `{"cmd":"inject","val":{inj_id, l_rad, m_rad, dm_pc_cm3, fluence_jy_ms, width_samples, profile, apply_at_specnum}}` (control_store.py:889-908,724-732).
- Receive: `src/dsart/inject/etcd_watcher.py` watches `/cmd/dsart/corr/<n>/` prefix inside corr_fast → `OnlineInjector.add_pending(cfg)` (inject/online.py).
- Active registry `/cnf/inject/active/<inj_id>` = {inj_id, dm_pc_cm3, l_rad, m_rad, width_samples, fluence_jy_ms, apply_at_specnum, fired_at_unix, ttl_s (60), fired_by, target_snr?} (dsa_monitor/inject_calibration.py:423-461; dsart.coinc.inject_match.ACTIVE_INJECT_PREFIX). Durable JSONL: $DSART_FIRED_INJECTION_LOG default `/dataz/dsa110/operations/inject/fired_injections.jsonl` (dsart.coinc.inject_log) — C3 recognizes injections after TTL expiry.
- Matching: C2 InjectionMatcher polls active keys, pairs C1 candidates, publishes `/mon/dsart/inject/matches/<inj_id>` = {best: {observed_snr, observed_event_specnum, matched_at_unix, K_inferred, observed_l_rad, observed_m_rad, observed_width_samples}, n_matches}.
- SNR calibration: POST /control/inject_calibrate (app.py:1030-1174) — health-gated laddered probe; K = observed_snr × sqrt(width_samples) / fluence_jy_ms (linear model); stored at `/cnf/inject/snr_calibration/<bucket>`, bucket = dm{round(dm/50)*50:04d} (e.g. dm0500). snr_to_fluence = target_snr × sqrt(width)/K. Constants: DEFAULT_CALIBRATION_FLUENCE 7e-4 Jy·ms, width 4 native samples, MAX_PROBE_FLUENCE 1e-3 (fp16-cuFFT overflow cliff), SATURATION_OBSERVED_SNR 240 (detector clips ±250σ), ladder ×(1,2,4) with 60 s steps (σ_k EMA recovery). Health pre-flight: corr_fast heartbeats <30 s, search compute heartbeats, c1_metering_active==0. GET /control/inject_calibrations; POST /control/delete_snr_cal (confirm=delete_snr_cal).

## 4. Voltage dumps

- Gate 1 `/cmd/c2/dumps_enabled` (dsa_monitor/dumps_gate.py; coincidencer.py:132 DUMPS_ENABLED_KEY): {"enabled": bool, "ts", "actor", "reason"}; **fail-OPEN** (missing ⇒ enabled). Controls C2 cube-dump broadcasts vs "WOULD-DUMP" logging. Route GET/POST /control/dumps_enabled (app.py:~1521), confirm enable|suppress, reason mandatory. ~200 ms cache in C2.
- Gate 2 `/cmd/c2/voltages_enabled` (dsa_monitor/voltage_controls.py; VOLTAGES_ENABLED_KEY): same shape; **fail-CLOSED** (missing ⇒ disabled — cold etcd never fills NVMe). Gates C2 DUMP_VOLTAGE UDP broadcast to 16 corr nodes. POST /control/voltages_enabled (app.py:~1659), confirm word enable.
- C3 mode `/cmd/c3/flag_only` (voltage_controls.py:293-330; services/c3.py): flag_only=True (default) = KEEP-ONLY (collect + log veto, no deletion); False enables conservative REJECT (delete staged voltages + cubes/*.npz, MOVE metadata to candidates_rejected/<event>/, never rm -rf). Panel #panel-c3-mode, confirm word delete.
- Trigger criteria `configs/c2_trigger_criteria.yaml`, hot-reload on mtime/SIGHUP; ordered first-match-wins trigger_classes; require predicates (snr_max_min, dm_median_min/max_pc_cc, dm_iqr_max_pc_cc, width_median_max_samples, lm_diag_max_rad, dm_galactic_fraction_min/max, n_events_min, n_search_nodes_min); action dump_all_gpus | log_only; holdoff_s. Classes: bright_frb_extragalactic (dm_median ≥ 0.75×NE2001 max-LOS, snr_max ≥ 12, dm 115–2700 → dump_all_gpus), bright_galactic (snr ≥ 15, dm ≥ 100 → dump), bright_pulsar (train → log_only), log_only fallback.
- Disk paths (services/c3.py:76-145): archive `/dataz/dsa110/candidates`, rejected `/dataz/dsa110/candidates_rejected`, state `/dataz/dsa110/operations/c3/c3_state.json`, staging (per corr node NVMe) `/home/ubuntu/data/voltage_staging`. Event layout: `<event>/Level2/voltages/` (fragments), `<event>/Level3/<event>.json` (C2 manifest = arrival sentinel C3 polls), `<event>_voltages.json` after collection.
- Manual DUMP_VOLTAGE test (scratch/M8_E2E_VOLTAGE_DUMP_TEST_20260709.md, PASSED 2026-07-09): from h23 use `dsart.coinc.broadcast.VoltageBroadcaster(hosts from /cnf/search_rt c3.corr_nodes).broadcast(name≤16B, target_block*2048, mjd)`; delete sentinel = same with event_specnum=0. Single node ~24 s, fragment 23×301,989,888 B = 6,945,767,424 B, window [target−8, target+14]. Fleet: 16/16 in 25 s; C3 picked up within one 10 s scan; 104 GiB rsync ~11 min. Caveats: voltages_enabled stayed FALSE (synthetic-only; live C2 `_maybe_broadcast_voltage` and REJECT delete path never exercised); manifest mjd_target always 0.0 (known cosmetic bug); dsart_c3.service unit shows inactive while process runs (unreconciled).

## 5. Dashboard Control tab panels

Start fleet (obs_dec_deg) / Arm (utc_start, margin) / Disarm / Stop (confirm=stop) / Restart-all (confirm=restart_all, dry_run) / Update fleet code (branch, force) / fstable build+deploy / C2 restart + activity + candidates + decision log / restart h23 services / fleet services table (fleet_services.py SERVICE_INVENTORY) / signal injection + SNR calibration / Dump Now (confirm=dump_now) / Dumps Enabled / Voltage Dumps Enabled / C3 Reject Mode / Operator Agent Authority (#panel-operator → `/cmd/operator/control`: agents_enabled master lockout, executor_email pin, max_obs_seconds cap; confirm word operator) / Spectral-line SPL per-subband table → `/cnf/spectral_line` (confirm spectral_line; applies only at next restart_all+start).

Authority model (dsa110-operator/src/dsa_operator/control/authority.py): `/cmd/operator/control` written ONLY by dashboard; agent can read, never write (agent writes only `/operator/` and `/cmd/ant/`); absent key = fail-open (enabled/unpinned/uncapped).

## 6. Monitoring workflow

Influx pusher tools/dashboard/dsart_rt_to_influx/pusher.py maps: /mon/service/corr_rt/<cn> → corr_rt_heartbeat; /mon/corr_rt/<cn> → corr_rt_routine + corr_rt_buffer (dada_dbmetric); .../capture/<port> → corr_rt_capture; .../rfi → corr_rt_rfi; .../meridian_ready → corr_rt_meridian; search equivalents (search_rt_heartbeat/routine/compute[c1_metering_active]/noise/dump/rx/cands); /mon/c2/<host> → c2_service (dumps_enabled) + c2_receiver + c2_inject_match. Also /mon/audit/control/<ns>/<unix_ms> (every dashboard action), /mon/operator/watchdog/<instance>, /mon/array/dec, /mon/array/gal_dm (NE2001 max-LOS, used by C2 criteria), /mon/snap/1/{armed_mjd, utc_start, utc_start_rt}. The workspace `obs-status` skill gives a one-shot health report from dashboard + InfluxDB.

## Gaps flagged (not guessed)
- /control/dump_now, restart_c2, restart_h23, fstables/*, update_dsart route bodies not traced in detail.
- C2 `_maybe_broadcast_voltage` exact sequence in services/coincidencer.py not read line-by-line.
- C3 veto/collection state machine not traced line-by-line (docs/voltage_dumps/VOLTAGE_DUMP_C3_DESIGN.md and docs/c1c2/C1C2_DESIGN.md have design detail).
- Live C2-triggered voltage dumps with voltages_enabled ON never exercised in production — guide must warn.
- dsa110-operator agent execution internals (observing/session.py, control/engine.py) not read.
