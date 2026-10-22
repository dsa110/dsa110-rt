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
| `opoint` | `""` | `"2x"` | D1 | Op-point label recorded in CSV and Level3 at schema 2. |
