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

## `configs/c2_trigger_criteria.yaml` (read by `coinc/criteria.py`)

- A new optional top-level `opoints:` mapping, keyed by op-point id or family (`2x`), each holding its
  own `trigger_classes:` list. C2 selects the section by `coinc.opoint` and keeps it across hot reloads.
  Without an `opoints` key the file loads exactly as today.
- New require keys `width_median_max_ms` / `width_median_min_ms`. They use the peak member's sample
  period and fail closed if it is unknown. The 2x section should use these keys in place of
  `width_median_*_samples`, and set `dm_median_max_pc_cc` ≥ 3000 and the DM-50 floor margin
  (inventory §2.3).
