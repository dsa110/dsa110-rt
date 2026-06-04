# M7.7.1 Phase A.2 — Search-side GPU bottleneck findings

**Date:** 2026-06-04
**Hardware:** n01, RTX 2080 Ti (cuda:0), CUDA 12.x
**Op-point:** production yaml (`configs/dsart_search_rt.yaml`):
`n_grid=256 n_fdm=34 t_det=192 cube_dtype=fp16 image_backend=gpu`
detector-streaming + tile_size=256 + n_top=24 + boxcar_accum=fp16
M7.7 symmetric-shift-padding ON, fused-L1 on, narrow `--pipeline-overlap`

## TL;DR

The M7.7 search-side pipeline is **GPU-bound at ~173 ms / cube** (sustained
5.78 cubes/s) at the production op-point. The 134 ms RT budget (7.46
cubes/s) **cannot be met** without one of: a deeper redesign (detector
on a second stream / GPU), an op-point change (smaller N_grid / N_fdm),
or faster hardware. The previous fix-loop reading of `total_pipeline
p50 = 137 ms` from the bench was misleading because the bench's
pipeline-overlap masks the GPU drain time — the real bottleneck is
genuine main-stream serial GPU work, not CPU sync overhead.

## Method

Added cuda-Event-based per-substage GPU timing on the main stream
(`process_h2d_prefetched`) and inside `GpuImager.process_cube`'s
per-batch inner loop. The speed gate displays both wall-clock and
GPU SM busy time per cube; PASS/FAIL is gated on
`sustained_ms = max(wall p50, total_gpu_ms)`. See commit `8a218ab`.

## Numbers (60 cubes, 55 warmed, M7.7 ON, owner_idx=0)

| stage | wall p50 (ms) | GPU SM (ms) | notes |
|---|---|---|---|
| build_cube | 140.6 | — | misleading wall; see notes |
| layer1_norm | 100.5 | 2.9 | wall is mostly CPU sync waiting for imager |
| detector_forward | 34.5 | 60.4 | wall is just CPU launch; GPU keeps running |
| imager (total) | — | 110.1 | dominant single contributor |
| &nbsp;&nbsp;combine | — | 64.5 | **at memory-BW roofline** (~67 ms theoretical) |
| &nbsp;&nbsp;fft | — | 33.9 | cuFFT-cfp16 ifft2 + fftshift |
| &nbsp;&nbsp;mask | — | 8.4 | edge-mask multiply + permute + writeback |
| **total_pipeline (wall)** | **137.8** | — | per-iter CPU wall |
| **total_gpu_ms** | — | **173.3** | genuine main-stream GPU work / cube |
| **sustained** | **= max(wall, gpu) = 173.3 ms / cube = 5.78 cubes/s** ||

The 36 ms gap between wall-p50 (138) and GPU (173) is the
`--pipeline-overlap` mask: cube N+1's CPU work (queueing prefetch,
fetching next slot) overlaps cube N's GPU drain. Real production
cadence tracks total_gpu_ms.

## A/B with M7.7 OFF (60 cubes, owner_idx=0)

| stage | M7.7 OFF | M7.7 ON | Δ ON−OFF |
|---|---|---|---|
| GPU imager_ms | 109.4 | 110.6 | +1.2 |
| GPU layer1_ms | 13.9 | 2.8 | **−11.0** |
| GPU detector_ms | 59.5 | 60.3 | +0.8 |
| GPU total | 182.8 | 173.7 | −9.2 |

M7.7 saves ~9 ms / cube — entirely from the fused-imager L1 mask
update replacing the broadcast divide. The 100% coverage benefit is
real (no coverage-correction needed) but the perf savings are modest.

## Failed perf experiments

1. **`DSART_IMAGER_FFT_BATCH=17`** (2 batches vs default 3): no change
   (FFT plan cached).
2. **`DSART_IMAGER_FFT_BATCH=34`** (single batch): OOMs the 11 GiB
   2080 Ti at the production geometry.
3. **`--full-prefetch`** (imager on its own stream concurrent with
   L1+detector): **worse** — wall p50 = 160 ms (vs 138 ms narrow
   overlap). Confirms the chunk-8d narrow-overlap design choice: the
   combine kernel and the detector both saturate memory bandwidth, so
   concurrent execution costs more than it saves.
4. **owner_idx sensitivity** (0 / 3 / 7): combine varies only
   ~±0.5 ms across DM owners (shift-table differences are too small
   to matter).

## Why combine is hard to speed up

The fused dequant+combine kernel reads `[N_chg=16, T_stream=349,
2, N_grid=256, N_grid=256] int8` once per fdm = 17× slab volume /
cube ≈ 27 GiB / cube. At the 2080 Ti's ~500 GB/s achievable
bandwidth → ~54 ms theoretical floor; measured 64 ms. The kernel is
within ~20% of the memory-bandwidth roofline. The
`fused_combine_cuda.py` docstring (chunk-8 D21) corroborates: their
production headline was 9.79 cubes/s ≈ 102 ms / cube
**imager-only**, matching our 110 ms imager today.

Where can savings come from?

* **Cross-fdm cint8 read sharing**: the kernel currently re-reads
  the cint8 streams for every fdm trial. Sharing across multiple fdm
  in a single pass would reduce the 17× slab volume factor; would
  require a kernel rewrite.
* **Lower-precision combine accumulator**: int8 → fp16 directly
  (skipping the int32 register accumulation) may enable tensor-core
  use; would require a kernel rewrite and validation that fp16 accum
  is numerically safe for N_chg=16.

Both are larger changes outside Phase A.2 scope.

## Wider-overlap options that could fit the budget

The 173 ms GPU total decomposes as imager (110) + L1 (3) + detector
(60). If these could be parallelised without SM contention, the
per-cube cost would drop to `max(imager, L1+detector) = max(110, 63)
= 110 ms` — comfortably under budget.

The bench's `--full-prefetch` path attempts this on a single GPU
and regresses (SM contention). Two paths that would NOT regress:

1. **Detector on a second GPU half**. Each search node has 2 GPUs;
   the current production design uses GPU 0 + GPU 1 to own different
   coarse-DM ranges. Splitting the per-half pipeline so the detector
   runs on the *other* GPU's spare cycles would require coordination
   but should give the parallelism without contention.
2. **Cut the detector kernel set**. The streaming detector at
   `boxcar_accum=fp16, tile_size=256, n_top=24` reports 60 ms GPU
   for 7 time-kernels. Halving the number of time-kernels would
   halve the detector cost; the operator decides whether that's
   acceptable for the C1 false-alarm budget.

## Operator interpretation of `bench/preflight/search_speed_gate.py`

* **PASS**: `sustained_ms ≤ budget_ms`. Safe to fleet-push from a
  search-side perf standpoint.
* **FAIL** with `wall p50 ≤ budget < total_gpu_ms`: the bench's
  pipeline-overlap is hiding a GPU-bound regression. Production will
  not keep up at this op-point. Focus on the dominant GPU substage
  (printed below the table).
* **FAIL** with `wall p50 > budget AND total_gpu_ms > budget`: both
  CPU iter and GPU work are over budget. Drill into both.
* **FAIL** with `wall p50 > budget AND total_gpu_ms ≤ budget`: the
  CPU is the bottleneck (rare; check for `.cpu()` / `.item()` syncs
  in py-spy).
