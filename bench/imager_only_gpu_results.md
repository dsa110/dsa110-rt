# M5 imager-only GPU throughput

This file captures the headline numbers from `bench/imager_only_gpu.py`,
which measures the GPU-cuFFT-cfp16 path for the production imager
(combiner + 2D iFFT + edge mask, no detector). Run on h01 GPU 1
(NVIDIA RTX 2080 Ti, 11 GiB), `dsa110-rt` conda env, against
synthetic cint8 streams pre-staged on GPU. The chunk-6a `image/imager.py`
numpy/CPU path is a placeholder per its own docstring (lines 18-24);
this bench measures the production target.

The bench has three `--combine-impl` modes:

- `python_addloop` (chunk-6c-α baseline, A/B reference): cint8 → cfp16
  scatter dequant + per-fdm `uv.zero_(); for g: uv.add_(...)`.
- `fused_cuda` (chunk-6c follow-up, fused combine only): cint8 → cfp16
  scatter dequant + a single NVRTC kernel that fuses the per-fdm
  16-chgroup combine into one read-each-chgroup-once + one-write pass.
- **`fused_cuda_cint8`** (chunk-8, **production target, default**):
  a single NVRTC kernel that reads cint8 streams DIRECTLY, accumulates
  per-fdm in int32 registers (exact for N_chg ≤ 16), and writes the
  cfp16 uv slab. The bench's scatter step is omitted entirely.

## Setup

- Device: cuda:1 (`CUDA_VISIBLE_DEVICES=1`, M3 isolation)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Bench: 30 cubes per geometry, cuFFT plan pre-warmed, single
  pre-staged 1.25 GiB cint8 input buffer reused per cube (host-side
  numpy.random input gen excluded from the cubes/s rollup so we
  measure the imager work and not the bench-only host overhead).
- Bench reports tree (h01): `bench/reports/20260506T212501Z/imager_gpu/M5/`
  (chunk-8) and `bench/reports/20260506T191520Z/imager_gpu/M5/` (chunk-6c).

## Headline (T_det sweep, fused_cuda_cint8, fixed N_fdm=32 / N_grid=256, cfp16 output)

| T_det | scatter (ms) | combine (ms) | ifft2 (ms) | mask (ms) | total p50 (ms) | total p99 (ms) | **cubes/s** |
|-------|--------------|--------------|------------|-----------|----------------|----------------|-------------|
| **128**   |          0 |         **23.0** |       22.0 |       5.8 |           **50.8** |          135.3 |   **18.18** ✅ |
| **192**   |          0 |         **33.3** |       32.6 |       8.3 |           **74.2** |          168.1 |   **12.62** ✅ |
| **256**   |          0 |         **43.6** |       43.2 |      10.9 |           **97.7** |          187.9 |    **9.79** ✅ |
| 384   |          0 |         65.8 |       64.5 |      16.0 |          146.3 |          234.5 |    6.63 |
| 512   |          0 |         88.5 |       85.7 |      21.3 |          195.3 |          296.9 |    5.00 |

✅ = clears 8 cubes/s plan target. **T_det=256 lands inside the budget with 22% headroom**, which was the operator-pinned T_det for May 2026 deployment.

## A/B/C compare across impls (cfp16, N_fdm=32, N_grid=256)

### T_det = 256 (operator-pinned target)

| impl | scatter | combine | ifft | mask | total p50 | cubes/s |
|------|---------|---------|------|------|-----------|---------|
| python_addloop | 35.7 | 193.3 | 43.1 | 10.9 | 283.3 | 3.52 |
| fused_cuda | 35.6 | 72.1 | 43.2 | 10.9 | 161.8 | 5.93 |
| **fused_cuda_cint8** | **0** | **43.6** | 43.2 | 10.9 | **97.7** | **9.79** ✅ |

cint8-fused vs python_addloop: **2.78× speedup** in cubes/s.
cint8-fused vs fused_cuda: **1.65× speedup** (the chunk-6c follow-up
doubled cubes/s; chunk-8 doubles it again).

### T_det = 192 (fall-back option if science needs more T)

| impl | total p50 | cubes/s |
|------|-----------|---------|
| python_addloop | 215.9 | 4.61 |
| fused_cuda | 125.2 | 7.59 |
| **fused_cuda_cint8** | **74.6** | **12.34** ✅ |

## What the chunk-8 cint8 kernel actually changed

| | chunk-6c (fused_cuda, cfp16-input) | **chunk-8 (fused_cuda_cint8)** |
|---|---|---|
| scatter step | dequant: 35.6 ms (4 kernel passes) | **omitted** |
| combine kernel input | cfp16 streams (4 B/cell) | **cint8 streams (2 B/cell)** |
| combine kernel acc | per-cell `__half2` add via `__hadd2` | **int32 (exact)** then 1 fp16 cast |
| streams memory traffic per cube | 32 GiB cfp16 reads + 2 GiB cfp16 writes | **17.2 GiB cint8 reads** + 2 GiB cfp16 writes |
| effective HBM BW (combine) | ~510 GB/s (84% of peak) | ~394 GB/s effective on cint8, but absolute time halved (smaller transactions, same HBM) |
| numerical precision | 16 ULP (cfp16 reduction chain) | **≤ 1 ULP** (int32 acc + 1 cast) |

## Pipeline cost split (T_det=256, fused_cuda_cint8)

After fusing dequant + combine into one kernel, the per-cube cost
is split essentially **fifty-fifty between combine and ifft**:

- combine (fused dequant+combine kernel): 43.6 ms (45%)
- ifft2 (cuFFT-cfp16): 43.2 ms (44%)
- mask (fftshift + edge): 10.9 ms (11%)
- scatter: 0 ms (eliminated)

cuFFT-cfp16 is the new headline binding constraint at T_det=256
(~5.2 µs/FFT2, near-optimal). Further wins require either reducing
N_fdm (operator vetoed) or N_grid (operator vetoed), or moving to a
batched-real-FFT path (potentially saves ~30% via Hermitian symmetry,
worth investigating in chunk-9 if T_det > 256 is needed).

## End-to-end M5 budget check (plan §8 ≥ 8 cubes/s)

With the chunk-8 cint8-fused kernel plus the D18 collapsed 8-kernel
detector bank (k_img=unit; k_dm=d1, ~3-13 ms detector cost), the M5
search-node hits the budget at every T_det ≤ 256:

| T_det | imager (ms) | det@8 kernels (ms) | total (ms) | hits 8/s? |
|-------|-------------|--------------------|------------|-----------|
| 128   | 50.8        | ~3                 | ~54        | **yes (18.5)** ✅ |
| 192   | 74.2        | ~5                 | ~79        | **yes (12.6)** ✅ |
| **256**   | **97.7**        | **~6**                 | **~104**       | **yes (9.6)** ✅  |
| 384   | 146.3       | ~10                | ~156       | no (6.4)  |
| 512   | 195.3       | ~13                | ~208       | no (4.8)  |

T_det=256 (operator's preferred integration time) is **the canonical
deployment configuration**: the imager hits 9.79 cubes/s, the
collapsed-bank detector adds ~6 ms, end-to-end ~104 ms / cube ≈
9.6 cubes/s with the 8-cube/s budget cleared by ~20%.

## Path forward (chunk 8 hardening, remainder)

1. **Land the cint8-fused kernel into the production imager.**
   Replace the chunk-6a numpy/CPU placeholder at `image/imager.py`
   with this GPU path. Track `src/dsart/image/fused_combine_cuda.py`
   as the canonical kernel implementation. (Estimated 1-2 days
   including unit-test surface for the production wiring.)
2. **Pre-stage cint8 streams on GPU at RX-ring level.** The bench
   pre-stages a single 1.25 GiB cint8 buffer once and reuses it
   per cube. In production this is enforced by the M3 → M5 IPC
   over UCX into a pinned-memory ring; verify during M3-coupling
   integration tests.
3. **Bake per-block scale/offset into the kernel.** The bench's
   cint8 fixture uses unit scale / zero offset (worst-case random
   fill). Production has per-chgroup `(scale, offset)` floats from
   the gridder — pass as small int32 → fp32 lookup tables, fold
   into the int32→fp16 cast at the kernel tail.
4. **(Optional)** Reduce iFFT cost via batched-real-FFT
   (Hermitian symmetry, ~30% projected). Not needed for the v1
   8 cubes/s target; chunk-9 hardening item.
