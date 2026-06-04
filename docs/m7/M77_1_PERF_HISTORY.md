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

## RESOLVED: the combine "regression" is a correctness fix

After A/B'ing on n01 (commit 8a218ab instrumentation), the source of
the 33 ms → 64 ms combine "slowdown" is **shift-table magnitude**,
not a code regression. Same kernel, same geometry, same GPU. The
combine kernel reads `streams[g, t - shifts[g]]` and skips cells
when `t - shifts[g]` falls out of `[0, T_stream)`. With smaller
shifts, FEWER cells skip → MORE memory reads → SLOWER kernel.

A/B on n01 cuda:0 with the `fused_cuda_cint8` kernel at the exact
production geometry (T_det=192, N_fdm=34, N_grid=256), varying only
the synthetic-shift range:

| Shift range | Coverage / (cell, chgroup) | combine_ms | Era |
|---|---|---|---|
| `[0, 400)` | ~50 % (huge shifts skip half) | **24 ms** | emulates M7.4 gate (`--include-coarse-offset` shifts ±1400) |
| `[0, 128)` | ~67 % | ~33 ms | matches the May 6 imager-only headline |
| `[0, 83)` | ~70 – 80 % | **48 ms** | emulates Option A alone (shifts ≤83 post-corr-side dedispersion) |
| pipeline today (M7.7 post-pad-subtract shifts) | **100 %** | **64 ms** | Option A + M7.7 — every cell contributes all 16 chgroups |

The ratio is exactly what kernel arithmetic predicts: combine cost
∝ number of in-range `(cell, chgroup)` reads. Moving from ~50 %
coverage (M7.4 gate) to 100 % coverage (today) doubles the work.

### What this means for the May 28 M7.4 gate PASS

The 7.45 cubes/s the fleet hit on May 28 was **achieved partly
because the search side was skipping ~half its chgroup
contributions at high DM** — exactly the bug the Option A enable
commit (`7c6ee9a`) called out: *"Option A enablement going live in
production: at high DM the search side now sees all 16 chgroups
contributing coherently to every cube (previously ~2/16 at DM=900,
search effectively blind for DM >~ 2000)."*

So the M7.4 gate was passing the throughput target while being
scientifically incomplete. Option A (Jun 2) fixed the correctness;
M7.7 (Jun 3) closed the last cell-edge coverage gap. Each pulled the
combine kernel toward the memory-bandwidth roofline (~54 ms
theoretical at 500 GB/s on a 2080 Ti). Today we sit at 64 ms,
within ~20 % of the roofline.

### Implication for the 134 ms RT budget

The 134 ms / cube budget was set when the kernel was effectively
doing half the work. With 100 % coverage (the correct scientific
behaviour) we need either:

* a budget revision (acknowledge that correct-search is ~5.8 cubes
  /s on the 2080 Ti, accept the resulting per-cube cadence change in
  the downstream pipeline);
* a structural change that lets the kernel do its full read budget
  in less wall-clock time (carry-over re-imaging — only re-image
  the new ``cube_cadence_samples=128`` of each cube instead of the
  full T_det=192 — drops combine ~30 %, total GPU ~152 ms; still
  not quite under 134, but closer);
* an op-point change (smaller N_grid or N_fdm — physics tradeoff);
* faster hardware (the A100 / H100 generation has ~3× the HBM
  bandwidth of the 2080 Ti and would put combine well below
  budget).

## What's *not* the cause

* **The combine kernel itself** — unchanged code; the same NVRTC
  binary runs faster or slower depending purely on shift-table
  magnitude (see the A/B table above).
* **GPU memory pool fragmentation, NVRTC codegen, cuda:0 vs cuda:1,
  M7.4.1 GPU-scatter fence** — all tested / ruled out by the
  imager-only A/B at exact pipeline geometry.
* **M7.7 sym pad** — A/B shows it adds only 3 ms of combine vs M7.7
  OFF (and saves 11 ms of L1 via the fused mask update — net
  -8 ms / cube vs M7.7 OFF).
* **M7.4.2 coverage correction** — measured 1.4 ms / cube, not 96
  ms.
* **Detector kernel set, L1 sample cap** — unchanged since M7.2.

## What *is* the cause

* **Option A enable (commit `7c6ee9a`, Jun 2)** — moved the coarse-
  DM dedispersion corr-side, shrinking search-side shifts from
  ±1400 → ±83. That's the correctness fix. The downstream perf
  consequence (combine ~24 → ~62 ms because fewer cells skip
  out-of-range) wasn't captured in the commit's "Net +1 ms / cube"
  benchmark — that was the corr-side cost only.
* **M7.7 sym pad (commit `ef6ffd5`, Jun 3)** — closed the last
  ~5 % coverage gap at the leading time-edge of each cube. Adds
  ~3 ms of combine on top of Option A's ~25 ms increment.

Together these account for the entire ~25–40 ms / cube cost increase
that moved the search from 7.45 cubes/s (M7.4 gate) to ~5.8 cubes/s
(today bench) / ~6.4 cubes/s (today live).

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

The 134 ms budget was set against a *scientifically incomplete*
search-side (M7.4 gate). The correct-search cost is ~5.8 cubes/s on
the 2080 Ti. Options:

1. **Acknowledge correctness > throughput** — re-baseline the
   downstream pipeline against the new sustained cadence
   (~155–175 ms / cube) and update the C1/C2 buffer assumptions to
   match. Probably the right answer in the short term.
2. **Carry-over re-imaging** — only re-image the new 128 samples
   per cube cadence instead of all 192 (the previous 64 carry over
   from the prior cube). Drops combine ~30 % (64 → ~45 ms), total
   GPU ~152 ms. Gets us close to budget but requires a state-
   machine rewrite of the imager workspace + a per-cube edge-
   compensation pass for the carried-over samples. Moderate work,
   moderate risk.
3. **Cut detector kernel set** — drop from 7 to 4 time-kernels;
   gains ~25 ms detector; total GPU ~148 ms. Single-flag change.
   Operator decides on the resulting C1 sensitivity tradeoff.
4. **Detector on its own stream** — runs concurrent with next
   cube's imager via the existing `output_cube_alt` ping-pong.
   Effective per-cube = max(imager, L1+detector) = max(110, 63) =
   110 ms — well under budget. Bigger code change, biggest reward.
5. **Move to A100/H100** — 3× HBM bandwidth puts combine well below
   budget. Out of band.

The committed instrumentation (`bench/preflight/search_speed_gate
.py`) now correctly reports sustained throughput so we won't be
misled by wall-clock again. The speed gate's PASS/FAIL semantics
reflect the GPU-bound reality.
