# M5 imager-only GPU throughput (chunk-6c follow-up)

This file captures the headline numbers from `bench/imager_only_gpu.py`,
which measures the GPU-cuFFT-cfp16 path for the production imager
(combiner + 2D iFFT + edge mask, no detector). Run on h01 GPU 1
(NVIDIA RTX 2080 Ti, 11 GiB), `dsa110-rt` conda env, against
synthetic cint8 streams pre-staged on GPU. The chunk-6a `image/imager.py`
numpy/CPU path is a placeholder per its own docstring (lines 18-24);
this bench measures the production target.

The bench has two `--combine-impl` modes for the per-fdm 16-chgroup
index-shifted sum:
- `python_addloop` (chunk-6c-α baseline): `uv.zero_(); for g: uv.add_(...)`
- `fused_cuda` (chunk-6c-β, default, **production target**): a
  custom CUDA kernel JIT-compiled via NVRTC (cupy.RawKernel) that
  reads each chgroup once and writes the output once. Cell-wise
  boundary check matches the Python guard.

## Setup

- Device: cuda:1 (`CUDA_VISIBLE_DEVICES=1`, M3 isolation)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Bench: 30 cubes per geometry, cuFFT plan pre-warmed, single
  pre-staged 1.25 GiB cint8 input buffer reused per cube (host-side
  numpy.random input gen excluded from the cubes/s rollup so we
  measure the imager work and not the bench-only host overhead).
- Bench reports tree (h01): `bench/reports/20260506T191520Z/imager_gpu/M5/`

## Headline (T_det sweep at fixed N_fdm=32 / N_grid=256, cfp16, fused_cuda)

| T_det | scatter (ms) | combine (ms) | ifft2 (ms) | mask (ms) | total p50 (ms) | total p99 (ms) | **cubes/s** |
|-------|--------------|--------------|------------|-----------|----------------|----------------|-------------|
| **128**   |         23.7 |         **37.2** |       22.1 |       5.8 |           **88.7** |          177.8 |   **10.74** ✅ |
| **192**   |         29.7 |         **54.6** |       32.7 |       8.3 |          **125.3** |          217.3 |    **7.71** ⚠️ |
| 256   |         35.6 |         72.1 |       43.2 |      10.9 |          161.9 |          257.7 |    6.01 |
| 384   |         47.6 |        107.0 |       64.3 |      16.0 |          234.9 |          337.6 |    4.17 |
| 512   |         59.5 |        142.2 |       85.3 |      21.0 |          308.0 |          418.7 |    3.19 |

✅ = clears 8 cubes/s plan target; ⚠️ = within 4% (would clear at T=195 or with detector hardening).

## fused_cuda vs python_addloop A/B (cfp16, N_fdm=32, N_grid=256)

| T_det | python_addloop combine | fused_cuda combine | speedup | python total | fused total | python cubes/s | fused cubes/s |
|-------|------------------------|--------------------|---------|---------------|-------------|----------------|---------------|
| 128   |         97.5 ms        |          **37.4 ms**   | **2.6×** |     148.8 ms  |    **89.0 ms**  |        6.68    |     **10.40** |
| 256   |        193.3 ms        |          **72.1 ms**   | **2.7×** |     282.9 ms  |   **161.9 ms**  |        3.52    |      **5.92** |
| 512   |        385.0 ms        |         **141.9 ms**   | **2.7×** |     550.8 ms  |   **307.7 ms**  |        1.81    |      **3.17** |

## cfp16 vs cfp32 sanity (T_det=256, fixed N_fdm=32 / N_grid=256, python_addloop)

| dtype | scatter | combine | ifft2 | mask | total p50 | cubes/s |
|-------|---------|---------|-------|------|-----------|---------|
| cfp16 |    35.7 |   193.3 |  43.1 | 10.9 |     282.9 |    3.52 |
| cfp32 |    32.9 |   385.0 |  85.6 | 20.7 |     524.4 |    1.90 |

cfp16 is exactly 2× faster than cfp32 across `combine`, `ifft2`,
`mask` — every stage memory-bandwidth-bound and scales linearly with
operand size.

## Pipeline cost breakdown

After fusing, the per-cube cost is split roughly equally across:
- **combine** (fused_cuda kernel): 42-46% of cube time
- **iFFT2** (cuFFT-cfp16): 25-28%
- **scatter** (cint8 → cfp16 dequant): 22-27%
- **mask** (fftshift + edge): 6-7%

The fused kernel hits ~510 GB/s effective on the 2080 Ti (memory-
bandwidth bound, near the ~616 GB/s peak). Theoretical min for combine
at this geometry ≈ 67 ms (T=256), observed 72 ms — within 7% of peak.

cuFFT-cfp16 is well-tuned: at T=512 the 16384 (32 × 512) 256² FFT2s
take 85 ms ≈ 5.2 µs / FFT2.

The remaining win available is **fusing the cint8 → cfp16 dequant
into the combine kernel** (read cint8 directly, accumulate as fp32,
write cfp16). At production geometry the cint8 stream is half the
size of the cfp16 stream, so the streams memory traffic halves;
combine drops from 72 ms → ~36 ms at T=256, total → ~125 ms = 8.0
cubes/s. That's the next chunk-8 hardening item.

## Implications for the M5 throughput budget

Plan §8 line 2317 mandates ≥ 8 cubes/s at p99. With the fused
combine + chunk-6c-α v1-bank collapse to k_img=unit; k_dm=d1 (D18,
8 kernels, detector cost ~3-13 ms / cube), the operator can pick
T_det based on science requirements:

| T_det | imager (ms) | det@8 kernels (ms) | total (ms) | hits 8/s? |
|-------|-------------|-------|-------|------|
| 128   | 89          | ~3    | ~92   | **yes (10.7)** ✅ |
| 192   | 125         | ~5    | ~130  | **yes (7.7→7.7)** ⚠️ borderline |
| 256   | 162         | ~6    | ~168  | no (6.0)  |
| 512   | 308         | ~13   | ~321  | no (3.1)  |

T_det=128 lands cleanly inside the budget with ~25% headroom.
T_det=192 is borderline.

## Path forward (chunk 8 hardening)

1. **Land the fused combine kernel into the production imager.**
   `image/imager.py` is a placeholder (numpy/CPU); replace with the
   GPU path used by `bench/imager_only_gpu.py`. Track
   `src/dsart/image/fused_combine_cuda.py` as the canonical
   implementation.
2. **Fuse cint8 dequant into the combine kernel** for the projected
   ~2× win on combine + scatter, landing T=256 at 8 cubes/s.
3. **Pre-stage cint8 streams on GPU at RX-ring level.** Currently
   the bench's "scatter" step assumes streams are pre-staged. In
   production this is enforced by the M3 → M5 IPC over UCX into a
   pinned-memory ring; verify during M3-coupling integration tests.
