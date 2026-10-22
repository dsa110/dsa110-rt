# WS-D config requests for WS-O

**Status:** living list. It is updated with each WS-D task.

**What this is:** the keys WS-D code reads. Every key is optional and defaults to today's behaviour, so
**leaving a key out of a 1x section keeps 1x bit-identical**. WS-O owns `configs/*.yaml` and puts these
keys in the op-point sections (SPEC §10).

**Flags that need sign-off:** keys marked **1x≠** change 1x behaviour if they are set in a 1x section.
Do not set them there until the user signs off (they are the CP1 sign-off items).

## `configs/dsart_search_rt.yaml` → `coinc:` (read by `services/coincidencer.py`)

| key | default (= today) | 2x value | task | notes |
|---|---|---|---|---|
| `archive_schema` | `1` | `2` | D1 | 2 appends unit and provenance columns to the C1 and C2 CSVs and to Level3 `c2` (`sample_period_us`, `width_ms`, `dm_peak`, `width_peak`, `peak_sample_period_us`, `peak_event_specnum`, `width_median_ms`, `opoint`), and sets Level3 `schema_version` 2. A schema change mid-hour re-keys the rolling CSV for that hour. |
| `opoint` | `""` | `"2x"` | D1 | Op-point label recorded in CSV and Level3 at schema 2. It also selects the criteria file's `opoints.<id or family>` section, and the YAML's `coinc_opoints.<id or family>` overlay. The `--opoint` CLI argument overrides it. |
| `cutout.enabled` | `false` | `true` | D6 | Turns on the cutout lifecycle (SPEC §9). C2 sends `CutoutRequest` to `dump_broadcast.hosts` × halves on `port_base+g` instead of the DSRT cube trigger. The event dir and Level2 CSVs are written at fire time, with `Level2/products_pending.json`. C2 finalises on distinct per-half manifests, or at `wait_s`. It never deletes an event. |
| `cutout.plan_path` | none | the deployed tiered plan `.npz` | D6 | Supplies the expected halves (owner map), `plan_id` and the fine→(coarse, tier) resolver. If the plan is absent, C2 expects every dump host × 2 halves and sends `plan_id` 0. **WS-S: please treat `plan_id` 0 as "no check".** |
| `cutout.status_bind_host` / `cutout.status_port` | `0.0.0.0` / `11501` | same | D6 | UDP endpoint for the halves' `CutoutStatus`. The search side sends to `c1.c2_endpoint.host:11501`. |
| `cutout.rows`, `raw_pre_ms`, `raw_post_ms`, `stamp_half_px`, `n_stamp_t`, `stamp_dm_halfwidth_pc_cc`, `deadline_s` | 512, 500, 500, 16, 64, 150, 15 | same | D6 | Request contents (SPEC §9). |
| `cutout.raw_for_injections` | `false` | `false` | D6 | Archive raw windows for injections too. |
| `cutout.ack_timeout_s`, `max_resends`, `wait_s`, `late_accept_s` | 2.0, 3, 120, 3600 | same | D6 | Resend and finalisation timing. |
| `partial_event_policy` | `discard` | `archive` | D6 | **1x≠.** `archive` keeps an event whose cube set is incomplete, with Level3 `products.status=incomplete`, instead of the `rmtree`. In cutout mode C2 always archives. |
| `fallback_sample_period_us` | `1048.576` | `524.288` | D6 | Voltage-dump specnum fallback when a peak header is unusable: ×16 at 1x, ×8 at 2x. |
| `dedupe_fired_clusters` | `false` | `true` (after sign-off) | D6 | **1x≠.** Never dump the same cluster twice through class-holdoff fall-through. |

| `inject_specnum_tol_s` | none (4096 samples) | `4.294967296` | D10 | Inject-matcher time gate in seconds, applied with each row's own period. |

**At a 2x op-point the 2x behaviours default on.** C2 resolves the op-point at startup through
`select_service_config` (SPEC §10). At 2x the following defaults switch on; an explicit key still
overrides:

| key | 2x default |
|---|---|
| `cutout.enabled` | `true` |
| `archive_schema` | `2` |
| `partial_event_policy` | `archive` |
| `fallback_sample_period_us` | `524.288` |
| `inject_specnum_tol_s` | `4.294967296` |

The 2x set therefore needs only what it already has: `partial_event_policy` and `cutout.{wait_s,status_port}`.
If `cutout.plan_path` is absent, C2 uses `/cnf/opoint.plan_path`. C2 refuses to start if the plan's
`plan_id` differs from `/cnf/opoint.plan_id`.

**The WS-D `x2_todo` entries in `configs/opoints/2x/dsart_search_rt.yaml` are resolved by the code**
(D6, D7 and D10 are merged on `x2/ws-downstream`). WS-O can delete these three entries:
- "WS-D D6 coinc";
- "WS-D c3";
- "WS-D D10 inject_bot". The bot takes the factor and cap from the op-point and the `c1` block, the
  owner map from `plan_path`, and the K family from `/cnf/opoint`.

The bot's `width_choices` / `dm_choices` values are WS-O's to set. Recommended:
- `width_choices`: add 16 native (one T_ext sample).
- `dm_choices`: e.g. `[100, 300, 800, 1400, 2000, 2500]`. Keep them away from 1575-1625, where the 50 pc/cc
  K bucket straddles the tier A/B boundary.

## `configs/opoints/2x/dsart_search_rt.yaml` → `inject_bot:`

| key | default (= today) | 2x value | notes |
|---|---|---|---|
| `metering_gate` | `any` | `owner` (after sign-off) | **1x≠ / sign-off.** `owner` means only the halves that own the shot's DM can block it (owner map from `plan_path`). |
| `plan_path` | `""` | from `/cnf/opoint` | Needed only for `metering_gate: owner`. Filled automatically from the active record. |
| `state_path` | today's file | `~/.dsa_monitor/inject_bot_state_2x.json` (already set by WS-O) | Per-family K-doctor and round-robin state. |

## `configs/opoints/2x/dsart_pipeline_rt.yaml`

- `X2TODO:WS-D:n-pre-for-dm3000` → **15**.
  - The full-band sweep at DM 3000 is 1.698 s. With the +1-block event-placement margin, the minimum is
    ceil(1.698 / 0.134218) + 1 = 14 blocks.
  - 14 blocks leaves 47 ms of margin; 15 gives 0.18 s.
  - `tests/test_voltage_dump_containment.py` will need to read the v3 plan's max fine DM
    (`fine_dm` is a flat array in v3 too, so it works as is).

## `configs/c2_trigger_criteria.yaml` (read by `coinc/criteria.py`)

- A new optional top-level `opoints:` mapping, keyed by op-point id or family (`2x`), each holding its
  own `trigger_classes:` list. C2 selects the section by `coinc.opoint` and keeps it across hot reloads.
  Without an `opoints` key the file loads exactly as today.
- New require keys `width_median_max_ms` / `width_median_min_ms`. They use the peak member's sample
  period and fail closed if it is unknown. The 2x section should use these keys in place of
  `width_median_*_samples`, and set `dm_median_max_pc_cc` ≥ 3000 and the DM-50 floor margin
  (inventory §2.3).

## `configs/dsart_search_rt.yaml` → `c3:` (read by `services/c3.py`)

| key | default (= today) | 2x value | task | notes |
|---|---|---|---|---|
| `veto.r2_t_shift_ms` | absent | `31.5` | D7 | R2 in ms; applies only to cutout events, whose metrics carry `t_ext_us`. 1x cube events keep `r2_t_shift_samples`. R3/R5/R10 keep their 1x values until they are recalibrated from the shadow veto on CP2 injections. |

C3 itself needs no op-point switch: it reads cutouts when an event has `cutouts/`, and cubes otherwise.
A non-flag-only REJECT now also deletes `<event>/raw/` and keeps `cutouts/`. The Slack readiness gate
skips the cube count for cutout events.
