# M6_PLAN_FIXES.md — locked decisions for M6 (search-node clustering + cube-dump)

This file is the working scratchpad for M6. Locked decisions accumulate
here as D-items (numbered) and as F-items (plan.md follow-ups). At
chunk-9 hardening this file is folded into the user-managed
`dsa110-rt_revamp_*.plan.md` and then `git rm`'d (mirrors the M5
hardening protocol — see M5 history through commits 61f7c4d → 45f6d23).

## Background — M6 architectural pivot (2026-05-07 user-locked)

M5 shipped a search-node detector pipeline that emitted **voltage
triggers** over persistent TCP to the corr nodes. M6's original scope
in plan.md was to harden that emitter path (new ring buffer for the
voltage trigger emitter, fast filterbank former, TCP trigger listener,
trigger dumper).

The user pivoted M6 to a different operational model:

  1. **No detector-emitted voltage triggers.** Voltage trigger handling
     is delegated to the existing `dsa110-xengine` framework
     (`dsaX_trigger` / `dsaX_store` / `fada` merged-voltage PSRDADA
     buffer). The M6 detector internally self-triggers for cube dumps;
     no fan-out to corr.
  2. **HDBSCAN clustering of detector candidates** — sidelobe-driven
     multiple-detection on bright bursts is folded into a single cluster
     labelled with the highest-S/N candidate. Clustering runs in a
     ThreadPool worker per (search_node, gpu_half) so the next cube
     keeps streaming.
  3. **Conditional cube dumping (NPZ)** — on a configurable
     bright-pulse predicate (per-cluster), or on an external UDP
     trigger (any-datagram → next cube). Writer thread + bounded queue
     (maxsize=4) so the real-time hot path is non-blocking.
  4. **ASCII candidate + cluster logs** — hourly-rotated, space-separated
     ASCII files (one T1-equivalent for per-candidate, one
     T2-equivalent for per-cluster). No JSON.
  5. **Original M6 deferred items** (voltage trigger ring, voltage
     trigger emitter, fast filterbank former, TCP trigger listener,
     trigger dumper) → moved to a new `§M-defer` section in plan.md
     (chunk 8 fold).

This pivot was confirmed in answers to the chunk-0 clarifying questions
on 2026-05-07. Recorded below.

## Locked decisions (D-items)

### D1 — T1 + T2 ASCII schemas drop `utc_iso`, `detector_version`, `flags`

The user's chunk-0 reply 2 read literally as: "remove utc_iso,
detector_version, flagsutc_iso, and detector_version, flags from T2".
The "flagsutc_iso" segment is interpreted as a typo concatenating
"flags" + "utc_iso" — i.e. the user is removing the same three fields
(`utc_iso`, `detector_version`, `flags`) from BOTH the T1 (per-candidate)
and T2 (per-cluster) ASCII schemas. Conservative reading; trivial to
revise if the user wanted T1 to keep them.

Final T1 columns (per-candidate row):

```
mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc  fine_dm_idx
t_in_cube  width_samples  snr  kernel_id  cl  is_cluster_peak
search_node_id  gpu_half
```

Final T2 columns (per-cluster row, peak candidate's properties):

```
mjd  event_specnum  l_rad  m_rad  l_pix  m_pix  dm_fine_pc_cc  fine_dm_idx
t_in_cube  width_samples  snr  kernel_id  cluster_id  cntc  cntb_lm  cntb_dm
cube_dump_triggered  search_node_id  gpu_half
```

`cl` = cluster id (-1 for unclustered noise). `is_cluster_peak`
= 1 for the highest-S/N candidate of its cluster. `cntc` = total
candidates in cluster. `cntb_lm` = number of unique (l_pix, m_pix)
cells. `cntb_dm` = number of unique fine_dm_idx trials.
`cube_dump_triggered` = 1 if this cluster's peak fired the auto-trigger
predicate (so the operator can correlate disk-side dumps with cluster
rows). `mjd` is double-precision MJD computed from `event_specnum`
through the on-host specnum→UTC table (chunks 1+2 will pull this from
`dsart.common.mjd` if it exists, else through
`utc_block_start_ns / 86400e9 + 40587`).

Header line: single `#`-prefixed comment, with column names matching
the schema verbatim, written once when the file is first opened (or
re-opened on hourly rotation).

### D2 — Hourly rotation, single file per (search_node, gpu_half) process

Rotation interval: every UTC hour, on the hour. File name template:

```
${LOG_ROOT}/cands_T1_s${search_node_id}_g${gpu_half}_${YYYYMMDD}_${HH}.txt
${LOG_ROOT}/cands_T2_s${search_node_id}_g${gpu_half}_${YYYYMMDD}_${HH}.txt
```

`${LOG_ROOT}` defaults to `${REPO_ROOT}/bench/reports/M6/cands_log/`
in dev; production overrides via `DSART_M6_CANDS_LOG_DIR` env var.

`fcntl.flock(LOCK_EX)` on each row write so concurrent writers (e.g. a
future second GPU-half process) can append safely. The hot-path
overhead of one flock per row is acceptable (~1 µs uncontended on
ext4); we choose it over a buffered-batch design because clusters are
small (median 1-3 rows; tail likely <50) and the operator wants
immediate visibility.

### D3 — Clustering features: real-units always, both options for input

Per chunk-0 reply 6: real-unit features (radians for l/m, pc cm⁻³ for
DM, seconds for time, samples for width) are always written to the T1
and T2 logs. The clusterer itself accepts either `int` indices (l_pix,
m_pix, fine_dm_idx, t_in_cube, width_samples) or real units, gated by
`features.py::FeatureMode = {"int", "real"}`. Default mode is `int`
(matches T2's DBSCAN convention from the cjl/dev `cluster_heimdall.py`
reference; cheap to switch in benches).

The clustering distance is `cityblock` (manhattan) per the T2 reference
hyperparameters. Per chunk-0 reply 8: the proposed weights
(`log2_width × 4`, `idm × 1`, `itime × 1`, `l_pix × 1`, `m_pix × 1`)
"sound right" — locked.

### D4 — HDBSCAN, with explicit fallback to DBSCAN if benchmarking fails

Per chunk-0 reply 7: HDBSCAN, but bench at the chunk-6 throughput gate.
The fallback gate (D5 below) defines the bail-out condition.

`min_cluster_size = 2` (matches T2's `min_samples = 2`; preserves the
"≥2 candidates anywhere" cluster behaviour). `min_samples = 1`
(HDBSCAN-specific, controls how conservative the noise label is; 1
labels everything as part of a cluster vs noise more aggressively than
the default of `min_cluster_size`). `cluster_selection_epsilon = 10.0`
(matches T2's `eps = 10` cityblock distance). `metric = "manhattan"`
(equivalent to cityblock; HDBSCAN spells it `manhattan`).

### D5 — Fallback to sklearn DBSCAN if HDBSCAN p99 > 50ms at production load

`bench/clusterer_throughput.py` (chunk 6) measures HDBSCAN cluster
latency on a synthetic 1000-candidate workload (the upper end of what
the M5 detector emits per cube at saturated FAR). If p99 > 50 ms (~38%
of the 134 ms cube cadence — too high to safely pipeline behind the
next cube's detector forward), `cluster.forward.cluster_candidates`
falls back to `sklearn.cluster.DBSCAN(eps=10, min_samples=2,
metric="cityblock")` and the M6.sh status JSON records
`clusterer_backend = "dbscan_fallback"`.

### D6 — Per-cube clustering with documented limitation

Per chunk-0 reply 10: cluster per-cube (not across cubes / time
windows). The known limitation: a burst that straddles a cube boundary
will produce two clusters (one per cube), each potentially with its own
auto-trigger fire. The operator can re-cluster offline against the T1
log if cross-cube grouping is needed. Documented in
`cluster/forward.py` docstring; flagged here for potential lift in M7
or beyond.

### D7 — Cube-dump NPZ schema + path + maxsize=4 queue

Per chunk-0 replies 12-15:

  * Auto-trigger fires on the **current cube** (the cluster that
    crossed the predicate is dispatched immediately after clustering;
    the cube tensor is still in scope on GPU).
  * UDP-trigger fires on the **next cube** (the listener queues a
    one-shot "dump next cube" flag; the per-cube driver consumes it
    after pipeline.process and before releasing the slot).
  * Format: NPZ with `cube` (`[T_det, N_fine_DM, N_grid, N_grid]
    float16`, real-valued, post-Layer-1 normalised), `mjd_start`,
    `event_specnum_start`, `t_det`, `n_fdm_in_cube`, `n_grid`,
    `cluster_record` (asdict of the auto-trigger cluster, or
    `{"trigger_source": "udp", ...}` for UDP), `search_node_id`,
    `gpu_half`. Loadable via `numpy.load(path)`.
  * Path: `${DUMP_ROOT}/cube_s${sid}_g${g}_${specnum_start}.npz`.
    `${DUMP_ROOT}` defaults to `${REPO_ROOT}/bench/reports/M6/cube_dump/`;
    production overrides via `DSART_M6_CUBE_DUMP_DIR`.
  * Writer thread: single `concurrent.futures.ThreadPoolExecutor(max_workers=1)`
    fed by a `queue.Queue(maxsize=4)`. Backpressure: when the queue is
    full, `put_nowait` raises and the dispatch path logs
    `"cube_dump_dropped: queue full (cube_id=%d)"` at WARNING. Per
    chunk-0 reply 21: dropped dumps are logged.

### D8 — Auto-trigger predicate config (essential subset)

Per chunk-0 reply 16, the auto-trigger predicate gates a cluster on:

  * `min_snr` (peak candidate SNR threshold; default 10.0σ, configurable)
  * `dm_fine_min_pc_cc` / `dm_fine_max_pc_cc` (None ⇒ no bound)
  * `width_samples_max` (None ⇒ no bound; defaults None)
  * `min_cntc` (cluster cardinality floor; default 1, i.e. accept
    singletons)
  * `holdoff_ms` between dumps (per-process; default 5000 ms — bursts
    are rare and we don't want a sidelobe run to fire dozens of NPZs)

A cluster fires the dump if ALL conditions hold. The cluster's
`cube_dump_triggered` field is set to 1 in the T2 log row.

### D9 — UDP listener: configurable port, any-datagram triggers

Per chunk-0 reply 17:

  * Default port 11227. The legacy `dsaX_filTrigger_twoInput` (which
    used to bind 11227 on search nodes) is removed in this M6 path, so
    the port is free.
  * Configurable via `--udp-trigger-port` CLI on the search-compute
    service entry.
  * Any datagram on the port triggers a one-shot "dump next cube" flag.
    The flag is consumed and reset on the next cube; queued requests
    don't accumulate (one datagram = one dump-next, regardless of
    payload).
  * Bind to `127.0.0.1` by default (the trigger source is on-host:
    operator scripts, T2/T3 future re-clusterers); configurable via
    `--udp-trigger-host` CLI.

### D10 — Plan-renaming: M6 = "Search-node clustering + cube dump"

Per chunk-0 reply 18(c). The full title in plan.md §M6 becomes:

> **M6 — Search-node clustering + cube dump (HDBSCAN clusterer + auto/UDP-triggered NPZ dumps; corr-side voltage triggers via dsa110-xengine).**

Original M6 deferred items (voltage trigger ring, voltage trigger
emitter, fast filterbank former, TCP trigger listener, trigger dumper)
move to a new plan §M-defer block (chunk 8 fold).

### D11 — h01 isolation envelope shared with M5

Per PARALLEL_AGENTS.md §4: M6 is incremental on top of M5. The
isolation envelope is reused:

  * `DSART_BUFFER_KEY_PREFIX=m5` (no rebinding; M3 is on `m3`, M5/M6 on
    `m5`)
  * `DSART_ETCD_NAMESPACE_PREFIX=m5`
  * `CUDA_VISIBLE_DEVICES=1` (M5 + M6 → GPU 1; M3 → GPU 0)
  * `flock /var/lock/dsart-m6.lock` (per-milestone lock; same writable
    fallback to `${HOME}/.dsart-m6.lock` as M5)

The M6.sh + M6_preflight.sh both set these.

### D12 — Specnum is unchanged; UDP datagrams carry no specnum

Per chunk-0 reply 4 + 17: the UDP trigger has NO specnum / coordinates;
it just dumps the next cube. The auto-trigger NPZ stamps the cluster's
peak `event_specnum`; the UDP NPZ stamps `specnum_start` of the next
cube.

## Plan.md follow-ups (F-items)

Folded into plan.md at chunk-9 hardening. F-items below are
plan-anchor-targeted edits, not standalone documents.

### F1 — Rewrite plan §M6 in full

Replace the deferred-items list under §M6 with the new D10 title and a
4-bullet body:

  1. HDBSCAN clusterer (per-cube, `cityblock`, `min_cluster_size=2`,
     `cluster_selection_epsilon=10.0`, `metric="manhattan"`); fallback
     to sklearn DBSCAN if HDBSCAN p99 > 50ms at the chunk-6 bench
     (`eps=10`, `min_samples=2`, `metric="cityblock"`).
  2. T1 (per-candidate) + T2 (per-cluster) hourly-rotated ASCII logs
     under `${LOG_ROOT}` with the D1 schemas.
  3. NPZ cube-dump on bright-pulse cluster predicate (D8) or UDP
     trigger (D9). Writer thread + maxsize=4 queue (D7).
  4. UDP trigger listener on configurable port (default 11227), any-
     datagram = dump next cube. Bind 127.0.0.1 by default.

### F2 — Add §M-defer block listing the deferred original-M6 items

Adjacent to §M6 in plan.md, add a new §M-defer block:

  > **M-defer — items deferred out of M6 (2026-05-07 user-pivoted).**
  >
  > Voltage trigger ring buffer (per-search-node), voltage trigger
  > emitter (TCP fan-out, predicate chain, holdoff state machine),
  > fast filterbank former (per-trigger), TCP trigger listener (corr
  > side), trigger dumper. Voltage trigger handling delegated to the
  > existing `dsa110-xengine` framework (`dsaX_trigger`, `dsaX_store`,
  > `fada` merged-voltage PSRDADA buffer). M6 detector self-triggers
  > internally for cube dumps; corr-side voltage triggers handled by
  > the legacy framework.

### F3 — §3 contract additions

Add two new contracts to plan §3 (next to `Candidate`):

  * `ClusterRecord` (frozen dataclass; outputs of
    `cluster.forward.cluster_candidates`). Fields: `cluster_id`,
    `cntc`, `cntb_lm`, `cntb_dm`, `peak_candidate_idx` (index into
    the per-cube candidate list), and the peak's `(l_rad, m_rad,
    l_pix, m_pix, dm_fine_pc_cc, fine_dm_idx, t_in_cube, width_samples,
    snr, kernel_id, event_specnum)` flattened in.
  * `CubeDumpManifest` (frozen dataclass; sidecar metadata for each
    NPZ dump). Fields: `cube_id`, `event_specnum_start`, `mjd_start`,
    `t_det`, `n_fdm_in_cube`, `n_grid`, `trigger_source` ∈
    `{"auto", "udp"}`, `cluster_record` (None for UDP), `npz_path`,
    `search_node_id`, `gpu_half`.

### F4 — §3.3 ASCII log schemas

Insert a new §3.3.x subsection documenting the T1 + T2 schemas (D1
columns + header convention). Cross-reference dsa110-T2's
`cluster_heimdall.py` as the inspiration but note our `mjd` is
double-precision and our distance metric matches T2's `cityblock`.

### F5 — §8 DoD additions for M6

Pin three new benches in plan §8:

  * `bench/clusterer_throughput.py` — HDBSCAN p99 at production
    candidate rates (D5 fallback gate).
  * `bench/cube_dump_e2e.py` — auto + UDP triggers end-to-end on h01
    against the 250924mptq fixture (re-uses the chunk-7 voltage
    fixture infrastructure).
  * `tools/viz/cluster_check.py` — operator-facing render of T1 + T2
    log rows for a given (search_node, gpu_half, hour) plus per-NPZ
    cube butterfly. Operator-approval gate #3 (after M5's two).

## Open follow-ups (post-chunk-0)

  * Tune HDBSCAN `cluster_selection_epsilon` against the chunk-7
    250924mptq burst sidelobe pattern. The T2 reference's `eps=10`
    cityblock is a starting point; the actual sidelobe spread on h01
    geometry may want tighter (5-7) or looser (15-20).
  * Decide whether the UDP listener should support a JSON-payload mode
    (override default specnum / cluster metadata) in a later
    milestone. Out-of-scope for M6.
  * Dead-data-contract sweep (chunk 9 hardening): `TriggerPacket`,
    `TriggerAck`, `TRIGGER_OPERATOR_SEARCH_NODE_ID` in
    `src/dsart/common/contracts.py` (plus the corresponding
    `tests/test_contracts.py` cases) are unused now that
    `dsart/trigger/` is gone. Chunk 0 leaves them as dead-but-tested
    code (low-risk; not imported by any live path) and removes them in
    the chunk-9 sweep along with the §3 lines 355-372 plan entry. F-item
    candidate (`F6 — drop legacy TriggerPacket / TriggerAck contracts`).
