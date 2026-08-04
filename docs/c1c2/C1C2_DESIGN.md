# C1 / C2 — search-side candidate stage and h23 coincidencer

Status: design locked 2026-05-21 (user direction).  Implementation in
progress on branch `m7/c1c2-coincidencer`.

This is the **single design source of truth** for the C1 stage (running
on each search node inside `SearchComputeService`) and the C2 stage
(running once on h23 as `dsart_c2.service`).  Sub-agent implementation
tasks reference it.  When a design question comes up, update this
document first and link the relevant `M7.x` plan-fixes entry.


## 1. Architecture summary

```
                      n01 search_compute_0 ─┐
                      n01 search_compute_1  │
                      n02 search_compute_0  │      TCP, persistent
                      n02 search_compute_1  ├── ──────────────────►  h23:11500
                      n09 search_compute_0  │       (C1 emit)        dsart_c2.service
                      n09 search_compute_1  │
                      n13 search_compute_0  │
                      n13 search_compute_1 ─┘
                                                          │
                                                          ▼
                                              connected-components
                                              (time-only edges,
                                              |Δt| ≤ (w_i + w_j) / 2)
                                                          │
                                       ┌───────────────────┼───────────────────┐
                                       ▼                   ▼                   ▼
                          rolling C1 / C2 CSVs    /dataz/dsa110/candidates/  C2 → C1 dump
                          (hourly rotation,        <name>/{Level2, Level3,    trigger UDP
                          48 h retention,          cubes, plots, ...}        broadcast
                          hiplot input)             (event archive)              │
                                                          │                     ▼
                                                          ▼              search_compute halves
                                                  dashboard "Burst       lookup specnum in
                                                  candidates" tab        N=8 CPU cube ring,
                                                                         dump NPZ → rsync
                                                                         to h23 archive
```

The system replaces the legacy T2 + tasktrigger + per-search-node
DBSCAN clusterer entirely.  Legacy services are shut down at bring-up
(see `M7.4 — legacy shutdown` below).


## 2. C1 stage (search node, per cube)

### 2.1 Where it lives

In-process inside `SearchComputeService` (`src/dsart/services/search_compute.py`).
Per (search node, gpu_half) there is exactly one process, so eight C1
emitters total across the four search nodes.

### 2.2 Detector threshold

Single knob: `c1.snr_min` (default 9.0).  This is applied as the per-
kernel detector NMS threshold inside `DeterministicDetector`, so weaker
peaks never survive NMS in the first place (no extra work in the hot
path) and the same number gates the C1 emit boundary defensively.

### 2.3 Cross-kernel merger (new geometry)

The legacy axis-AND box merger is replaced.  New suppression rule, for
candidates `c` (later in the SNR-sorted scan) and `s` (already-accepted
survivor):

```
dt_max          = c1.merger.t_frac * 0.5 * (c.width_samples + s.width_samples)
in_t            = |c.event_specnum - s.event_specnum|       ≤ dt_max
in_fdm          = |c.dm_idx        - s.dm_idx|              ≤ c1.merger.dm_max_trials
in_lm_or_cross  = (|c.l - s.l|     ≤ c1.merger.lm_max_cells)  OR
                  (|c.m - s.m|     ≤ c1.merger.lm_max_cells)

suppress c iff   in_t AND in_fdm AND in_lm_or_cross
```

Defaults (locked 2026-05-21):

```
c1:
  merger:
    lm_max_cells: 3          # OR semantics on l, m for cross-shaped PSF of the DSA core
    dm_max_trials: 2         # |Δfdm|
    t_frac: 1.0              # |Δt| ≤ t_frac * (w_i + w_j) / 2
                             # t_frac = 1.0 is the half-window edge predicate
                             # the user specified; <1.0 only useful for stress tests.
```

Note: the OR over (l, m) is intentional — a real burst spread along
either the EW or NS arm of the cross-shaped DSA core PSF leaves
survivors at small Δl or small Δm respectively, and we want to merge
those into a single peak.

### 2.4 Per-node clustering

REMOVED from the hot path.  The HDBSCAN / DBSCAN clusterer in
`src/dsart/cluster/` is kept as a library for offline analysis but is
**not** instantiated by `SearchComputeService`.  The `ClusterRecord`
contract is kept (the C2 path constructs them on h23).

### 2.5 CPU-side cube retention ring (N=8)

Each `SearchComputeService` half holds a circular buffer of the last
`c1.cube_ring_depth` cube tensors (default 8) in pinned host memory:

```
~/buffer/(search_node, gpu_half):
    ring[0..N-1] = (cube_id, event_specnum_start, mjd_start, t_det,
                    n_fdm, n_grid, cube_fp16_host)
    write_pos = (write_pos + 1) % N
```

The host-side mirror is populated asynchronously from the GPU on a
side stream after the detector returns (the GPU pipeline does not
block on it).  Existing implementation has a pinned host buffer already
populated by `cube_pipeline._stage_h2d`; we extend it so the host buffer
is **kept** for N cubes rather than reused after one cube.

At production geometry (192 × 34 × 256 × 256 × 2 B ≈ 3.2 GB / cube)
this is ~26 GB pinned host RAM per half × 2 halves = ~52 GB / search
node.  Within budget on the 256 GB search nodes.  We surface
`c1.cube_ring_depth` as a YAML knob so we can throttle in M7.4
shake-down.

### 2.6 C1 emit (TCP → h23)

Per `(search_node_id, gpu_half)` an asyncio task maintains one
persistent TCP connection to `c1.c2_endpoint` (default
`h23:11500`).  On every cube, the per-cube driver hands the survivor
list to the emitter via a non-blocking `queue.put_nowait` (depth
`c1.emit_queue_depth`, default 16).  Queue-full drops are counted as
`c1_emit_dropped` in the mon-points; the per-cube driver never blocks
on emit.

Wire format is **ASCII**, candsfile-style.  See `C1C2_WIRE_SCHEMA.md`
for the byte-level contract.

### 2.7 Dump-trigger listener (replaces UdpTriggerListener)

The current UDP listener (one-shot "dump next cube" flag, opaque
payload, bound 127.0.0.1:11227) is replaced by `C2TriggerListener`
bound on the search-net interface (10.41.0.x: see
`configs/dsart_search_rt.yaml::hostargs`) port 11227.

Payload (binary, fixed layout — see wire schema):

```
struct C2TriggerPacket {
    uint32_t  magic;             // 'DSRT' = 0x44535254
    uint16_t  version;           // 1
    uint16_t  trigger_class_id;  // operator-defined; opaque to C1
    char[16]  event_name;        // ASCII, e.g. "260521abcd"
    int64_t   event_specnum;     // cube anchor specnum
    double    mjd_target;        // event MJD (for sanity)
    uint32_t  flags;             // bit 0 = dump_cube
    uint32_t  reserved;          // 0
}
```

On receive, the listener looks up the cube whose
`specnum_start <= event_specnum < specnum_start + cube_cadence_samples *
sample_period_specnum` in the N=8 ring.  If found, submit a
`CubeDumpWriter` job with path
`${dump_root}/<event_name>/cube_s<sid>_g<g>_<event_specnum>.npz`.  If
not found (already aged out), log a `c1_trigger_too_late` warning and
increment the corresponding mon-point counter.


## 3. C2 stage (h23, fan-in coincidencer)

### 3.1 Service layout

- Code: `src/dsart/services/coincidencer.py` (entrypoint) + `src/dsart/coinc/{...}` (library).
- Systemd: `systemd/dsart_c2.service` (a user unit, installed into
  `~/.config/systemd/user/` by `tools/c2/install.sh`).
- Conda env: `casa38` (same as legacy T2 — gives us `event.names`,
  `dsautils`, `slack_sdk`).

### 3.2 Receiver

Async TCP server bound `0.0.0.0:11500`.  Accepts up to 8 long-lived
client connections (one per `(s, g)`).  Each connection's read loop
parses cube-batches as defined in the wire schema and pushes the new
candidate rows onto the rolling window.

### 3.3 Rolling-window connected-components

The window holds all candidates within
`coinc.window_s` seconds of the latest received candidate (default
5.0 s).  On every new batch:

1. Append new candidates.
2. Build a graph with nodes = candidates in the window, edges =
   `|t_i - t_j| ≤ (w_i + w_j) / 2` in seconds.
3. Compute connected components (union-find on the new edges only —
   the rest of the graph is incremental).
4. For each component touched by the new batch, compute the cluster
   statistics (see `ClusterStats` schema below) and run the trigger
   evaluator.
5. Age out candidates older than the window.

Statistics are time-only-clustered; DM / position / width contribute
to the cluster's *characterisation* (mean / IQR / max-spread) but
**not** to the connectivity edge.  This is the user's explicit
direction: cluster across DMs / positions / widths, then characterise.

### 3.4 Trigger criteria

YAML file `configs/c2_trigger_criteria.yaml`, hot-reloadable on
SIGHUP.  Each entry has a `require` block (all conditions ANDed) and
an `action`.  Default initial criteria (subject to operator tuning in
M7.4 shake-down):

```yaml
trigger_classes:
  - name: bright_frb
    require:
      snr_max_min: 12.0
      n_events_min: 3
      n_search_nodes_min: 2
      dm_median_min_pc_cc: 50
      dm_median_max_pc_cc: 4000
      dm_iqr_max_pc_cc: 30
      width_median_max_samples: 32
      lm_diag_max_rad: 5.0e-3
    action: dump_all_gpus
    holdoff_s: 30.0

  - name: bright_pulsar
    require:
      snr_max_min: 10.0
      n_events_min: 5
      dm_iqr_max_pc_cc: 2
    action: dump_all_gpus
    holdoff_s: 5.0

  - name: log_only
    require:
      n_events_min: 1
    action: log_only
```

Actions:

- `dump_all_gpus`: allocate an event name via
  `event.names.increment_name(mjd_peak, lastname=get_lastname())`,
  set up `/dataz/dsa110/candidates/<name>/`, emit a `C2TriggerPacket`
  to each of the 8 `(s, g)` C1 listeners with the assigned name +
  `event_specnum`, write the C2 row, queue the plot job.
- `log_only`: write the C2 row, no name allocated, no broadcast.

### 3.5 Event archive layout

```
/dataz/dsa110/candidates/<name>/
├── Level2/
│   ├── C2_<name>.csv           # the cluster's per-candidate rows (T1-equiv)
│   ├── C1_window_<name>.csv    # all C1 candidates in the time window around it
│   └── plots/
│       ├── dm_time_<name>.png
│       ├── image_peak_<name>.png
│       ├── lightcurve_<name>.png
│       └── kernel_snrs_<name>.png
├── Level3/
│   └── <name>.json             # trigger metadata, matches old shape so the
│                               # archive consumer can keep its conventions
├── cubes/
│   └── cube_s<sid>_g<g>_<event_specnum>.npz   # 8 NPZs, one per (s, g)
├── voltages/                   # filled later by corr-side voltage dump (M-defer)
├── filterbank/                 # filled later
└── calibration/                # symlink to fixture cal/ for replays
```

The legacy tasktrigger.py role (watching for new `cluster_output_<name>.json`
and assembling per-event directories) is folded into C2 directly — C2
owns the event directory creation and population.

### 3.6 Hiplot-viewable rolling CSVs

Two output directories, each populated by C2:

```
/dataz/dsa110/operations/C1/cluster_output/
  c1_${YYYYMMDD}_${HH}.csv      # hourly rotation, 48-hour retention
/dataz/dsa110/operations/C2/cluster_output/
  c2_${YYYYMMDD}_${HH}.csv      # hourly rotation, 48-hour retention
```

Retention is enforced inside C2 via a background `housekeeping` task
that deletes files older than `coinc.csv_retention_hours = 48` on each
rotation boundary.

Schemas:

- **C1 hiplot CSV** (one row per received candidate):

  ```
  mjd, event_specnum, snr, dm_pc_cc, dm_idx_global, fine_dm_idx,
  l_rad, m_rad, l_pix, m_pix, width_samples, kernel_id, flags,
  search_node_id, gpu_half, cube_id, trigger
  ```

  `trigger` = event name if this candidate ended up in a
  `dump_all_gpus` cluster, else `""`.

- **C2 hiplot CSV** (one row per coincidenced cluster):

  ```
  mjd_peak, snr_max, snr_sum, snr_mean, n_events, n_search_nodes, n_gpu_halves,
  dm_median, dm_iqr, dm_min, dm_max,
  l_median, m_median, lm_diag_rad,
  width_median, width_min, width_max,
  t_span_s, t_start_mjd, t_end_mjd, kernel_ids_distinct,
  trigger_class, trigger
  ```

### 3.7 Plots

After cube NPZs from a triggered event land in
`/dataz/dsa110/candidates/<name>/cubes/`, the C2 plot worker
generates four PNGs into `Level2/plots/`:

1. **`dm_time_<name>.png`** — concatenate the 8 cubes along DM,
   max-project over (l, m) in a small window around the cluster peak,
   render DM × time waterfall.
2. **`image_peak_<name>.png`** — image plane at peak (DM, t).
3. **`lightcurve_<name>.png`** — time series at peak (l, m, DM).
4. **`kernel_snrs_<name>.png`** — bar plot of cluster SNR by kernel_id.

Worker is a `concurrent.futures.ThreadPoolExecutor(max_workers=2)` so
plots don't block the receive loop.  Per-event plot timeout 30 s.

**Matching the detector's boxcar (2026-08-02).** Panels 1-3 place
themselves at the burst's cube time, which the plotter has to relocate
because the dumped NPZ carries no usable first-sample anchor (see the
note below). That relocation used to argmax the *raw* per-sample
series while the detector thresholds a boxcar match-filtered one, so
the panels landed several samples off a wide burst (3 samples on
260801rmep at `width_samples=16`, 7 on 260801bdga) and the re-measured
significance understated it. The DM light curve is now convolved with
the candidate's own `width_samples` boxcar (normalised `1/sqrt(w)`)
before the argmax, `dm_time` renders the smoothed cube, `image_peak`
averages the plane over the same boxcar, and every title prints the
detector's SNR alongside the cube re-measurement. Panels 1 and 3 also
work from the **detected pixel** rather than the image max, whose
extreme-value distribution over 65 536 pixels is not a point-source
sigma scale. `DSART_PLOTTER_SMOOTH=0` reverts to the raw behaviour.

**Known gap — no in-cube time anchor.**
`dump/c2_trigger_listener._build_manifest` writes
`event_specnum_start = packet.event_specnum` (the trigger specnum,
which is also the NPZ filename key) rather than the retained cube's
sample-0 specnum, while `mjd_start` keeps the cube's real sample-0
MJD. The two anchors in every archived NPZ therefore disagree and
`(event_specnum - event_specnum_start) / sample_period_specnum` is 0
for every event, so the detector's own time index cannot be recovered
offline. Fixing it means ADDING a field (`cube_specnum_start` /
`cube_mjd_start`); repurposing `event_specnum_start` would change the
NPZ filenames that C3 and the dashboard glob on.


## 4. C2 → C1 dump broadcast

UDP packets to each of the 8 listeners on the search-net interface.
We use UDP (not TCP) because the message is fixed-size, one-shot, and
broadcasts may have to fan out within tens of milliseconds.  Lost
packets are tolerable (the operator can re-trigger).

Listener bind addresses derived from `dsart_search_rt.yaml::hostargs`:

```
n01 → 10.41.0.205:11227   (search-net interface, lxd110h01 / n01)
n02 → 10.41.0.222:11227
n09 → 10.41.0.253:11227
n13 → 10.41.0.238:11227
```

`(search_node, gpu_half)` is selected by source-port: C2 sends to
`(host, 11227)` for `gpu_half=0` and `(host, 11228)` for `gpu_half=1`.


## 5. Dependency graph + parallel sub-agent work

```
                    ┌──────────────────────────────────────────┐
                    │ C1C2_WIRE_SCHEMA.md  (locked first)      │
                    │ c2_trigger_criteria.yaml  (locked first) │
                    └──────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
   │ subagent-A: C1 path  │  │ subagent-B: C2 svc   │  │ subagent-C: glue    │
   │  - merger geometry   │  │  - receiver          │  │  - dashboard tab    │
   │  - cube ring         │  │  - coincidencer      │  │  - hiplot units     │
   │  - emit task         │  │  - trigger emitter   │  │  - legacy shutdown  │
   │  - trigger listener  │  │  - archive layout    │  │  - M7.4 readiness   │
   │  - strip clusterer   │  │  - csv rotation      │  │    doc              │
   │  - configs + tests   │  │  - 4-panel plotter   │  │                     │
   │                      │  │  - systemd unit      │  │                     │
   └──────────────────────┘  └──────────────────────┘  └─────────────────────┘
                                       │
                                       ▼
                              integrate + lock M7.4
                              fixture-replay readiness
```

## 6. M7.4 legacy shutdown (operator runbook lands in `docs/c1c2/M7.4_BRINGUP.md`)

Before launching the new stack:

1. Stop legacy T2: in `calibration23`, `systemctl --user stop run_T2*`
   (or equivalent ExecStart wrapper).  Verify no listener on
   `10.42.0.90:{12345..12348, 13345..13348}`.
2. Stop legacy hiplot: on h23, `systemctl --user stop hiplot.service`
   (port 5007 freed).
3. Stop legacy tasktrigger: on h23,
   `systemctl --user stop tasktrigger.service`.
4. Snapshot the legacy `cluster_output/` dir into
   `/dataz/dsa110/operations/T2/cluster_output.legacy/` so we don't
   stomp on it.
5. Start the new units in this order:
   `dsart_c2.service` → `hiplot_c1.service` → `hiplot_c2.service`
   → restart search-node `dsart-rt search_rt` so the C1 emitters
   re-connect.

A `tools/c2/legacy_shutdown.sh` script automates 1–4 with idempotent
checks.

## 6b. Zero-filled cube edges — root cause (2026-08-04)

Dumped cubes intermittently show a hard zero block at the end, most
visibly at low DM: `260803wsxt`'s DM×time waterfall goes dead from
**t=192** in its lowest-DM half (`s1g0`) and from **t≈205** in its
highest. This is not cosmetic — it is the mechanism behind a family of
end-of-cube false positives, and it corrupts the C3 veto's own metrics.

**It is not the inter-cube overlap.** `t_det=256`,
`cube_cadence_samples=192` ⇒ 64 samples of designed overlap, and 192
coincides with the boundary, which is misleading. The M7.7 wait gate
*does* cover the trailing padding:

```python
target_seq = (last_cube_seq_boundary + t_det + pad_right) * n_active_dms_per_corr
```

and the scatter buffers *are* sized `t_stream`, not `t_det` (the
"we deliberately size at T_det" comment above
`_scatter_cint8_buf` is stale, pre-M7.7).

**The cause is the fan-in gate emitting incomplete cubes.** The gate
passes once `fan_in_min_corrs` corrs reach `target_seq` — production
runs `--fan-in-min-corrs 15`, so a cube is emitted with **15 of 16**
corrs present. The absent corr's chgroup rows stay zero in the scatter
buffer. The fine-DM combiner then sums 16 chgroups at per-`(fdm,
chgroup)` time shifts, so a missing chgroup depresses
`n_chg_contrib(t, fdm)` in a pattern set by the shift table —
concentrated at the cube's time edges and varying with DM trial, which
is exactly the observed ragged, DM-dependent tail. Under a persistently
late corr (the known ~8–11 % hash-dependent corr→search fabric loss)
this is chronic rather than occasional.

Two downstream layers then fail to correct it:

* **Layer-1 coverage correction** divides by expected coverage. Under
  symmetric padding (enabled in production) it is designed to be a
  no-op because `n_chg_contrib ≡ 16` — an assumption a genuinely
  missing corr violates, so the short cells are never corrected.
* **`validity_mask` cannot express it.** It is `[t_det, n_fdm]` but is
  *broadcast* from a per-`t` vector, so "this DM trial's edge is short a
  chgroup" is unrepresentable. Worse, the detector never masks the data
  with it at all — `_compute_per_kernel_scores`' own docstring says the
  mask "is validated here but not used to mask the data"; it only gates
  the Layer-2 σ_k EMA, and `layer2_valid_min_fraction` was relaxed from
  1.0 so the EMA keeps learning off partly-zero cubes.

The boxcar bank therefore convolves straight across the zero step and
produces an edge response at the end of the cube.

**Real fixes, in order of value:**

1. **Make the coverage correction honour actual coverage.** Compute
   `n_chg_contrib(t, fdm)` from which corrs were really present this
   cube — the RX already knows, it has per-corr `wseq` at gate time —
   and divide by that instead of assuming 16. This makes a 15/16 cube
   statistically correct rather than subtly wrong at the edges.
2. **Make `validity_mask` per-`(t, fdm)`** and have the detector exclude
   invalid cells from the boxcar, not just from the EMA gate.
3. **Operational lever:** `--fan-in-min-corrs 16` removes the artefact
   outright, at the cost of dropping a cube whenever any corr is late.
   Not obviously the right trade, but it is one config token and it
   makes the artefact vanish, so it is the cheapest way to confirm this
   diagnosis on sky.
4. Fix the underlying fabric loss so 16/16 is the norm.

Until (1)/(2) land, `dsart/coinc/cube_veto.py` computes every statistic
on `live_span()` so at least the *offline* adjudication is not corrupted
by the zeros; that is a mitigation, not a fix.

## 7. References (in-tree)

- Detector + merger code: `src/dsart/detector/{forward.py, merger.py,
  decoder.py}` (pre-C1).
- Candidate + ClusterRecord contracts: `src/dsart/common/contracts.py`.
- Legacy T2 reference: `calibration23:/home/ubuntu/vishnu_test/dev/dsa110-T2/`.
- Name generator: `dsa110-event/event/names.py` (h23
  `~/proj/dsa110-shell/dsa110-event/`).
- Dashboard: `tools/dashboard/dsa_monitor/app.py` (Flask, port 5778).
- Plan entry to update once C1/C2 is merged: `dsa110-rt_revamp_7b1d2669.plan.md`
  §M7.4 bullet (currently "DISCUSS FIRST" gated).
