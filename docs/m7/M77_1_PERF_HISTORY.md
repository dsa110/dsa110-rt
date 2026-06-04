# Search-side perf timeline — May 6 to June 4, 2026

Reconstructed for the question "we thought this was under budget, right?
What's been happening in the production system?"

## Headline

| Date | Source | Geometry | Result | Notes |
|---|---|---|---|---|
| May 6 | `bench/imager_only_gpu` | T=192 N_fdm=32 N=256 | **12.62 cubes/s** (74 ms imager) | Imager-only, no L1/detector. Headline of `bench/imager_only_gpu_results.md`. cuda:1. |
| May 20 (M7.2/M7.4) | Full 16×1 fleet | prod | **7.55 cubes/s** sustained | First time real-time was met. `--include-coarse-offset-in-search-shifts` ON. T_stream ≈ 1600. |
| May 20 (M7.2/M7.4) | Full 16×4 fleet | prod | **7.451 cubes/s** sustained | Same, scaled to 4 search nodes. |
| May 28 | M7.4 gate (full real-SNAP soak) | prod | **7.39–7.56 cubes/s** sustained per half | All 8 halves PASS at the 7.45 cubes/s target. |
| Jun 4 (Phase A.2 bench) | n01 cuda:0 bench | prod | **5.77 cubes/s** (173 ms GPU) | GPU SM busy time = sustained throughput floor. |
| Jun 4 (live observation, per prior session) | live fleet | prod | **~6.4 cubes/s** (156 ms / cube) | Slightly better than bench, possibly due to real-wire cint8 sparsity. |

**Bottom line: production has regressed ~15–22% vs the M7.4 gate (May 28)
even though we expected M7.7 + Option A to be neutral-or-better.**

## What changed since the M7.4 gate (May 28)

Diff of `configs/dsart_search_rt.yaml` between the M7.4 PASS commit
(7153ac6) and HEAD has effectively only one switch:

```
-      --include-coarse-offset-in-search-shifts
+      --symmetric-shift-padding
```

(plus ring-buffer sizing for memory pressure: `--t-buf-samples 32768
→ 8192`, `cube_ring_depth 24 → 8 → 16`; cal-blob path; CUSTOMDEC.)

Code-side, the relevant compute-touching commits since M7.4 gate:

1. **`131ca41` M7.4.2 Layer-1 coverage correction** — adds a per-`(t,
   fdm)` coverage compute + broadcast divide before σ-estimation. The
   commit message and the inline rationale estimated this cost the
   `~96 ms/cube` L1 regression. Phase A.2 GPU-event measurement shows
   it actually adds only ~1.4 ms (cf. M7.7 OFF vs ON: L1 = 14 ms vs 3
   ms = 11 ms delta, almost all from the fused-L1 mask update
   replacing the broadcast divide, NOT from removing coverage
   correction).
2. **`7c6ee9a` Option A enable** — moves coarse-DM dedispersion to the
   corr side. Search-side shifts shrink from ±1432 to ±83 samples;
   T_stream shrinks 1624 → 275. SHOULD be neutral or slightly faster
   (combine memory traffic per cube is independent of T_stream:
   17 × t_det × n_fdm × N² bytes regardless).
3. **`ef6ffd5` M7.7 symmetric padding** — pads stream by max(0, shifts.
   max()) + max(0, -shifts.min()) so the imager kernel has in-range
   source rows everywhere. T_stream grows 275 → 358. Saves ~9 ms /
   cube via fused-L1 (measured A/B above).

Net expected change M7.4 → today: −10 ms / cube (faster). Net
observed: **+30 to +40 ms / cube (slower)**.

## What's unexplained

The combine kernel's GPU SM time is **~64 ms at T_det=192 / N_fdm=34
/ N_grid=256 today**, vs **~33 ms at T_det=192 / N_fdm=32 / N_grid=
256 on May 6 (imager-only headline)**. Same kernel
(`fused_dequant_combine_per_fdm_cf16`), nearly the same geometry
(+6% N_fdm work), but the kernel is **~80% slower**.

Per-cube combine memory traffic = 17 × t_det × n_fdm × N² bytes
(read 16 chgroups + write 1, all cint8 → cfp16) = 27 GiB / cube.
At 500 GB/s HBM peak achievable on a 2080 Ti → 54 ms theoretical
floor. May 6 measurement (33 ms) is *below* the simple roofline →
implies it benefited from cache reuse (T_stream there was ~600 with
the larger shifts → bigger working set but more sequential reads).
Today's measurement (64 ms) is *above* roofline → cache thrash from
something we haven't isolated.

**Hypotheses to investigate (none confirmed):**

* **GPU memory pool fragmentation**: the cube_ring_depth went
  through 24 → 8 → 16; the imager workspace alloc patterns may
  interact poorly. (Easy to test: a fresh `python` process should
  reset.)
* **Implicit kernel-launch fence** from the M7.4.1 GPU-scatter
  (commit a98bda8): the dense cint8 buffer is written on the main
  stream then consumed by the combine kernel. If the scatter doesn't
  write the entire buffer cleanly, the combine reads might miss in
  L2 cache for the lookahead rows.
* **Compiler regression**: the kernel goes through NVRTC; cupy
  version updates between May 6 and now could change codegen.
* **Different GPU**: May 6 ran on cuda:1; our gate runs on cuda:0.
  Need to A/B by flipping `CUDA_VISIBLE_DEVICES=1`.

## What's *not* the cause

* M7.7 / symmetric padding — A/B shows it saves 9 ms / cube vs OFF;
  it's a net positive at the production op-point.
* Option A — moved coarse-DM work corr-side; should be neutral or
  faster for the search.
* M7.4.2 coverage correction — measured 1.4 ms / cube, not 96 ms.
* Detector kernel set — k_time and accum dtype haven't changed.
* L1 sample cap — still 10 000 since M7.2 (line was 100k → 10k
  *before* the M7.2 gate).

## Reading guide for the discrepancy

If the live fleet currently shows 6.4 cubes/s and the bench shows
5.77 cubes/s, the gap (~17 ms / cube) is most likely:

* **Bench uses `--prequantise`** — one pre-built cint8 buffer reused
  every cube. Real wire delivers sparse cint8 (n_filled ≤ 5000 cells
  per (corr, t)); the GPU dense buffer post-scatter has many zero
  rows. The combine kernel reads those zeros but the cache pressure
  is lower → real wire could plausibly be faster.
* **Bench is single-half**, live has the other half running
  concurrently on the second GPU. They share PCIe / memory controller
  / ~something. Could be ±5 ms either way.
* **Bench's NVRTC kernel cache state** could differ.

So the bench's 5.77 cubes/s is a CONSERVATIVE estimate; live is
~6.4. Both are below the 7.45 target.

## What to do about it

Phase A.2 found no quick perf win. The committed instrumentation
(`bench/preflight/search_speed_gate.py`) now correctly reports
sustained throughput so we won't be misled by the wall-clock again.

Larger options that could close the gap (all out of A.2 scope):

1. **Triage the combine 33→64 ms regression** — first confirm on
   cuda:1 (does the regression persist? if not, GPU-specific). If
   it does, bisect the commits between May 6 and today on the imager
   path. Most actionable.
2. **Detector overlap** — run detector on a separate stream so it
   overlaps with the next cube's imager (the ping-pong output buffer
   in `GpuImager` already supports this geometry). Highest reward
   but biggest code change.
3. **Cut detector kernel set** — drop from 7 to 4 time-kernels;
   gains ~25 ms; operator decides on the C1 sensitivity tradeoff.
4. **Move to a faster GPU** — out of band.

Recommend (1) as the first follow-up, paired with re-running the
M7.4 gate to confirm whether the production system has actually
regressed or whether the M7.4 numbers were transient (note the M7.4
report measured "last 1.3 s window" steady-state, not long-term
average).
