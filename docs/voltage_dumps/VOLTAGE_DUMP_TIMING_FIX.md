# Voltage dump timing — defect report and fix specification

Status: REVIEWED 2026-07-15 (code audit; no fix applied yet).
Audience: implementing agent. Everything needed is pinned here; do not
re-derive conventions from the plan or legacy docs — where this report and
older prose disagree, this report reflects the code as deployed.

## 1. Definitions (exact, use these units everywhere)

| Term | Definition |
|---|---|
| **specnum** | SNAP packet-sequence unit = **65.536 µs** (= 2 native samples of 32.768 µs). `dsart.common.constants.SPECNUM_PERIOD_US`. |
| **block** | One fada PSRDADA page = **2048 specnums = 0.134217728 s** (`BLOCK_SAMPLES_SPECNUM`, `BLOCK_DURATION_S`). |
| **block_n** | Per-process fada page counter, first page = 1. `block_specnum_start = block_n * 2048`. |
| **search sample** | Detector sample period. **Production = 1048.576 µs = 16 specnums** (`--t-int-search-us 1048.576` in `configs/dsart_search_rt.yaml:221,309`). NOT the 524.288 µs value in `constants.T_INT_SEARCH_US_DEFAULT`. |
| **C1 `event_specnum`** | Candidate time in **search-sample units** (absolute, block_n-anchored). |
| **ν_top, ν_bot** | Processed band edges: 1.498750 GHz, 1.311281 GHz (`NU_TOP_PROC_GHZ`, `NU_BOT_PROC_GHZ`). |
| **τ_sweep(DM)** | Full-band dispersion sweep = `4.148808e-3 × DM × (1/1.311281² − 1/1.498750²)` s = **5.65916e-4 × DM seconds** (1.5014 s at DM 2653). |

**Time reference of every candidate (verified):** the corr-side stage-2
FIFO delays each chgroup so all 16 align to pulse arrival at **ν_bot**
(deployed plan `/home/ubuntu/data/dm_plans/dm_plan_N8_dmmin100_tol1.6_v2.npz`:
`time_shift_corr_stage2[15,:] = 0`, chgroup 0 delayed most), and transport
frames are labeled with the emission-time block counter
(`corr_fast_integration.py` `_TransportTxAdapter` / `_AsyncTransportTxAdapter`,
`specnum=int(block_n)`). Therefore **`event_specnum` = pulse arrival time at
the BOTTOM of the band (1.311 GHz)**. The raw voltages of the burst lie
entirely at or BEFORE this time; chgroup g's earliest signal is
`Δτ(ν_g_top → ν_bot, DM)` before it (1.5014 s for chgroup 0 at DM 2653).

## 2. The trigger→dump chain (as deployed, all correct except noted)

1. Search candidate → C1 row (`event_specnum` in search samples).
2. C2 coincidencer clusters rows; peak-SNR member becomes
   `ClusterStats.peak_event_specnum` (`coinc/stats.py:155`).
3. `coincidencer.py:1303` converts to corr units:
   `native_specnum = peak_event_specnum * T_INT_FACTOR_DEFAULT` (×16)
   → **DEFECT 2** (right number today, wrong constant; see §4).
4. UDP broadcast to 16 corr nodes, port 11229 (`coinc/broadcast.py`).
5. `voltage_retention` (per corr node) computes
   `target_block = native_specnum // 2048` and stages blocks
   `[target_block − n_pre, target_block + n_post]`
   (`dump/voltage_ring.py::window_block_range`,
   `services/voltage_retention.py::write_window_to_staging`).
6. Deployed window: `--n-pre 8 --n-post 14 --retention-s 15.0`
   (`configs/dsart_pipeline_rt.yaml:562-578`) → **DEFECT 1**.

## 3. DEFECT 1 (primary): dump window points the wrong way

The window guarantees only `8 × 0.134218 = 1.074 s` before the event time,
but the event time is at ν_bot and the burst extends up to
`τ_sweep(DM)` **before** it. The 14 post-blocks (1.879 s) buy almost
nothing (signal after t_bot is only pulse width ≤ ~17 ms + per-chgroup
smear ≤ ~78 ms).

Containment at the deployed search range (fine-DM max = 2652.98):

- DM ≤ 1898: fully contained on all nodes.
- DM 1898–2135: chgroup 0 clipped depending on where the event falls in
  its block.
- DM 2653 (worst case, event at start of target block): chgroups 0–5
  (1499–1440 MHz) partially or fully missing; shortfall 0.43 s.

History note: `n_pre=8 / n_post=14` matches the sealed plan's DEDISP
convention where the trigger label was to be converted to ν_top before
dispatch. That conversion lived in the retired original-M6 trigger
workstream (`common/constants.py:325-333`) and was never re-implemented in
the C2 path. The window shape survived; the label reference did not.

### Required fix (Option A — recommended, config-only)

In `configs/dsart_pipeline_rt.yaml` `voltage_retention` routine args, swap
the asymmetry:

```
--n-pre 14      (was 8)
--n-post 8      (was 14)
```

Also update the argparse defaults and their help strings in
`services/voltage_retention.py` (`--n-pre` default 8 → 14, `--n-post`
default 14 → 8) so defaults match production, and fix the stale help text.

Derivation the implementer must preserve (add as a comment at the yaml
args): with the event label at ν_bot,

```
n_pre  ≥ ceil(τ_sweep(DM_max_fine) / 0.134217728) + 1     # +1 = block quantization
n_post ≥ ceil((width_max + intra_chgroup_smear) / 0.134217728) + margin
```

At DM_max_fine = 2652.98: `ceil(1.5014/0.134218)+1 = 13`, so `n_pre=14`
gives one spare block. `n_post=8` (1.074 s) is ≥ 10× the physical need and
keeps the total window at 23 blocks = 6.47 GiB/node (unchanged RAM/disk;
retention ring is 112 blocks, so 14 pre-blocks is comfortably inside it).

**If the deployed DM plan's fine-DM max ever changes, n_pre must be
re-derived.** Add a guard: a check (preflight script or unit test, see §5)
that loads the deployed dm-plan `.npz` (`fine_dm.max()`), reads the
deployed yaml `--n-pre`, and asserts the inequality above.

### Option B (alternative — do NOT combine with Option A)

Keep `n_pre=8 / n_post=14` and instead convert the label to ν_top at the
broadcast point (`coincidencer._maybe_broadcast_voltage`):

```
native_specnum_top = native_specnum − round(τ_sweep(dm) / 65.536e-6)
# with dm = stats.dm_median, τ_sweep in seconds, result in specnums
```

This restores the plan's original convention but adds a DM-dependent code
path and changes the meaning of `event_specnum` in the staged manifests
(`target_block_n` would then be the ν_top block). **Applying both A and B
double-compensates and clips the post side.** Pick exactly one; A is
preferred (simpler, no semantic change to manifests, robust to DM up to
~3300 at n_pre=14).

## 4. DEFECT 2: unit-conversion constant is semantically wrong (latent 2× bug)

`coincidencer.py:1297-1303` multiplies C1 search-sample specnums by
`T_INT_FACTOR_DEFAULT` (=16). That constant is defined
(`common/constants.py:512`) as *native samples per search sample at the
524.288 µs op-point*. The conversion needs **specnums per search sample**
= `t_int_search_us / 65.536`, which is 16 only because production runs
t_int_search = 1048.576 µs. If t_int_search ever reverts to 524.288 µs the
dump target lands 2× in the future and every dump stages 0 blocks (mirror
image of the 2026-07-13 incident).

### Required fix

The peak cluster member already carries the authoritative per-batch value:
`WindowEntry.sample_period_us` (`coinc/window.py`), sourced from the C1
batch header. In `_maybe_broadcast_voltage` (or in `ClusterStats`
construction, threading a `peak_sample_period_us` field through
`stats.py`):

```python
factor = round(peak_sample_period_us / SPECNUM_PERIOD_US)   # 65.536
native_specnum = int(peak_event_specnum) * factor
```

Guards required:
- `assert abs(factor * SPECNUM_PERIOD_US - peak_sample_period_us) < 1e-6`
  (integer-factor sanity; log + fall back to config-derived factor on
  failure rather than raising — this path must never kill the cube dump).
- Never use a heartbeat header's placeholder (`sample_period_us=1.0`,
  `c1_emit.py:690`); peak members are real rows so this cannot occur, but
  guard `peak_sample_period_us > 100.0` anyway.
- Remove the `T_INT_FACTOR_DEFAULT` import from `coincidencer.py` so the
  wrong constant cannot be reintroduced silently.

## 5. Tests (all new, required with the fix)

1. `tests/test_voltage_dump_containment.py::test_window_contains_full_sweep`
   — pure arithmetic: for the deployed dm plan
   (`configs/dm_plans/…` or the path in `dsart_pipeline_rt.yaml`
   `--dm-plan-path`), parse `--n-pre`/`--n-post` out of
   `configs/dsart_pipeline_rt.yaml` and assert
   `n_pre * BLOCK_DURATION_S ≥ τ_sweep(fine_dm.max()) + BLOCK_DURATION_S`
   and `n_post * BLOCK_DURATION_S ≥ 0.2`.
2. `::test_c1_to_native_specnum_conversion` — build a WindowEntry with
   `sample_period_us=1048.576`, `event_specnum=S`; assert the broadcast
   value is `S*16`; repeat with `524.288` → `S*8` (this is the case the
   current code gets wrong; it must fail before the fix and pass after).
3. `::test_dump_window_reference_frequency` — end-to-end arithmetic
   fixture: given a synthetic burst at DM 2653 with ν_bot arrival specnum
   T, assert every chgroup's earliest-signal specnum
   `T − τ(ν_g_top→ν_bot)/65.536e-6` falls inside
   `[(T//2048 − n_pre)*2048, (T//2048 + n_post + 1)*2048)`.

## 6. Explicit non-goals of this fix (known, tracked separately)

- `block_n` provenance: page counters, not SNAP packet counters; a mid-run
  restart of `voltage_retention` or `corr_fast` on any node silently
  desyncs dump targeting fleet-wide. Real fix is the deferred M7.2.8
  corner-turn (specnum from packet headers). Do not attempt here.
- Stale duplicate knobs `n_pre_blocks: 10 / n_post_blocks: 5` in
  `configs/config_compute_corr.yaml` and `configs/operating_points.yaml`
  are read by nothing in the voltage path; deleting them is optional
  cleanup, not part of this fix.
- Orphaned staging (events with no C2 archive dir never get a C3 verdict
  and their staging is never cleaned) — separate GC concern.

## 7. Rollout / verification

1. Land code + tests; run the new tests plus
   `pytest tests/ -k "voltage or retention"` on a corr or search node
   (NOT h23 — no usable GPU env there; these particular tests are
   CPU-only but keep the convention).
2. Config push: `tools/ops/push_dsart_to_etcd.py` after editing
   `dsart_pipeline_rt.yaml` (voltage_retention reads CLI args at spawn, so
   corr `dsart_rt` restart — or at minimum a voltage_retention routine
   restart — is required; note the restart resets the ring, losing the
   15 s retention history and, per the non-goal above, must happen while
   capture is NOT armed, or block_n desyncs).
3. Live check: fire a test trigger (`260715test`-style) and confirm each
   node's staged manifest shows `n_pre=14`, `n_post=8`,
   `n_blocks_written=23`, `n_blocks_dropped=0`.
