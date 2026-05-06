# M5 imager-only GPU throughput (chunk-6c follow-up)

This file captures the headline numbers from `bench/imager_only_gpu.py`,
which measures the GPU-cuFFT-cfp16 path for the production imager
(combiner + 2D iFFT + edge mask, no detector). Run on h01 GPU 1
(NVIDIA RTX 2080 Ti, 11 GiB), `dsa110-rt` conda env, against
synthetic cint8 streams pre-staged on GPU. The chunk-6a `image/imager.py`
numpy/CPU path is a placeholder per its own docstring (lines 18-24);
this bench measures the production target.

## Setup

- Device: cuda:1 (`CUDA_VISIBLE_DEVICES=1`, M3 isolation)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Bench: 30 cubes per geometry, cuFFT plan pre-warmed, single
  pre-staged 1.25 GiB cint8 input buffer reused per cube (host-side
  numpy.random input gen excluded from the cubes/s rollup so we
  measure the imager work and not the bench-only host overhead).
- Wall: ~36 s / sweep cell at production geometry (most of which is
  bench-only host-side input gen; per-cube imager work is well under
  1 s).
- Bench reports tree (h01): `bench/reports/20260506T172631Z/imager_gpu/M5/`

## Headline (T_det sweep at fixed N_fdm=32 / N_grid=256, cfp16)

| T_det | scatter (ms) | combine (ms) | ifft2 (ms) | mask (ms) | total p50 (ms) | total p99 (ms) | cubes/s |
|-------|--------------|--------------|------------|-----------|----------------|----------------|---------|
| 128   |         23.7 |         97.4 |       22.0 |       5.7 |          148.9 |          159.1 |    6.69 |
| 192   |         29.7 |        145.4 |       32.5 |       8.3 |          215.9 |          228.7 |    4.62 |
| 256   |         35.7 |        193.3 |       43.1 |      10.9 |          282.9 |          297.7 |    3.52 |
| 384   |         47.6 |        289.0 |       64.2 |      15.9 |          416.7 |          436.3 |    2.39 |
| 512   |         59.5 |        385.0 |       85.2 |      21.0 |          550.7 |          575.2 |    1.81 |

## cfp16 vs cfp32 sanity (T_det=256, fixed N_fdm=32 / N_grid=256)

| dtype | scatter | combine | ifft2 | mask | total p50 | cubes/s |
|-------|---------|---------|-------|------|-----------|---------|
| cfp16 |    35.7 |   193.3 |  43.1 | 10.9 |     282.9 |    3.52 |
| cfp32 |    32.9 |   385.0 |  85.6 | 20.7 |     524.4 |    1.90 |

cfp16 is exactly 2× faster than cfp32 across `combine`, `ifft2`,
`mask` — every stage is memory-bandwidth-bound and scales linearly
with operand size. (Scatter is unchanged because the bench's
host→device copy + cint8→fp32 dequant produces fp32 either way.)

## Pipeline cost breakdown

At all geometries, the **combine** step dominates at ~70% of cube
time. It is the 16-chgroup index-shifted sum into the per-fdm uv
slab: for each of 32 fdm trials, accumulate 16 contiguous-time
slices of cfp16 streams via in-place `add_`. Total memory traffic is
`N_fdm × N_chgroup × T_det × N_grid² × 4 bytes` cfp16 reads + writes.

At T_det=256: 32 × 16 × 256 × 256² × 4 B ≈ 8.6 GB / cube. Observed
193 ms ≈ 43 GB/s effective vs ~600 GB/s peak HBM bandwidth on the
RTX 2080 Ti. **~14× headroom for a fused gather+reduce CUDA kernel**
that reads each chgroup's stream once into the per-fdm output. This
is the single biggest remaining win on the imager side.

cuFFT-cfp16 is well-tuned: at T_det=512 the 16384 (32 × 512) 256²
FFT2s take 86 ms ≈ 5.2 µs / FFT2.

The **scatter** step (host→device + chunked cint8 → cfp16 dequant
via fp32 → cf64 → cf32 cast) costs ~60 ms at production geometry.
In production this is replaced by direct cfp16 input from M3 over
UCX (the M3→M5 IPC step is "scatter" equivalent in the production
pipeline; here the bench includes the host→device transfer, which
is a worst-case proxy).

## Sensitivity to GPU memory layout

| variant | combine (ms) | total (ms) | notes |
|---------|--------------|------------|-------|
| `torch.stack(slices).sum()` | 401.8 | 567.4 | initial vectorisation; allocates 1 GiB tmp/fdm |
| `uv.zero_(); for g: uv.add_(slice)` | 385.0 | 550.7 | current; saves the stack alloc but same memory traffic |
| **(future) fused gather+reduce kernel** | ~70 (theoretical) | ~210 | single read of 16 chgroups + reduce, near peak HBM |

## Implications for the M5 throughput budget

Plan §8 line 2317 mandates ≥ 8 cubes/s at p99. With the current
bench numbers and the chunk-6c-α v1-bank collapse to k_img=unit;
k_dm=d1 (D18, 8 kernels — detector cost drops by 16× to ~3 ms / cube
at T_det=128), **the imager is the binding constraint for the
8 cubes/s target** at the user-pinned production geometry
(N_fdm=32, N_grid=256):

| T_det | imager (ms) | det@8kernels (ms) | total (ms) | cubes/s | hits 8/s? |
|-------|-------------|-------------------|------------|---------|-----------|
| 128   | 149         | ~3                | ~152       | 6.6     | **no** (close) |
| 192   | 216         | ~5                | ~221       | 4.5     | no |
| 256   | 283         | ~6                | ~289       | 3.5     | no |
| 512   | 551         | ~13               | ~564       | 1.8     | no |

To hit 8 cubes/s we need either:
1. A fused combine kernel (~5× win on the imager → all geometries land < 125 ms).
2. T_det as small as the science permits (128 is closest;
   need detector + RX-ring sensitivity analysis to confirm).
3. A combination of both.

Path forward (per operator, 2026-05-06): port the production
combiner+imager at `image/imager.py` to this GPU path during M5
hardening (chunk 8); track the fused-combine kernel as a
performance follow-up.
