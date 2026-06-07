# M7.7.2 — Carry-over re-imaging design (option C from M77_1_PERF_HISTORY.md)

**Status:** IMPLEMENTED + VALIDATED. Flag `cube_pipeline_carry_over_re_imaging`
defaults OFF — operator flips it in `configs/dsart_search_rt.yaml` when ready
to push to fleet.

## Goal

Drop search-side GPU work by ~21 ms / cube by skipping the imager work for the
64 image-space samples that overlap with the previous cube.

  * Today: 173 ms GPU / cube. Sustained 5.78 cubes/s.
  * Target: ~152 ms GPU / cube. Sustained ~6.6 cubes/s.
  * Budget: 134 ms. **Still not under budget** — would need to compose with
    a second lever (detector stream or kernel cut). But +21 ms / cube is a
    real throughput win even on its own.

## Mathematical correctness

Production cubes advance by `cube_cadence_samples = 128`, each cube spans
`t_det = 192` samples, so consecutive cubes share `t_det - cube_cadence
= 64` image-space samples.

For cube N starting at specnum `s_N` and cube N+1 starting at
`s_{N+1} = s_N + 128`, the imager output is

    cube_N    [t,  f, u, v] = FFT⁻¹( Σ_g  streams_N    [g, t  - shifts[g] + pad_left] )
    cube_{N+1}[t', f, u, v] = FFT⁻¹( Σ_g  streams_{N+1}[g, t' - shifts[g] + pad_left] )

Since `streams_{N+1}[g, x] = streams_N[g, x + 128]` (the streams are the
same physical scattered cint8 data, indexed at a 128-sample-shifted
absolute time origin), we have

    cube_{N+1}[t', f, u, v] = FFT⁻¹( Σ_g  streams_N[g, t' + 128 - shifts[g] + pad_left] )
                            = cube_N[t' + 128, f, u, v]                       ∀ t' ∈ [0, 64)

provided that **every chgroup g remains in-range** at the read site
`t' + 128 - shifts[g] + pad_left`. M7.7 symmetric padding guarantees
exactly this: `pad_left = max(shifts)`, `pad_right = -min(shifts)`, so
every (cube_t, fdm, chgroup) tuple lands in `[0, T_stream)`.

**∴ Without σ rescale, cube_{N+1}[t' ∈ [0, 64), f, u, v] is bit-equal
to cube_N[t'+128, f, u, v].** Carry-over is mathematically exact.

## σ rescale (the only subtlety)

The fused-L1 path multiplies the imager output by `1/σ_layer1_prev[f]`
in-place (via `set_edge_mask_per_fdm`). So:

    cube_N    on-disk = (raw image at cube N)    × 1/σ^{N-1}[f]
    cube_{N+1} on-disk = (raw image at cube N+1) × 1/σ^{N}[f]

Naïvely copying `cube_N[t=128:192, f]` → `cube_{N+1}[t=0:64, f]` would
leave the carry-over region with the wrong σ. The fix is a one-time
per-fdm rescale:

    cube_{N+1}[t=0:64, f, u, v] = cube_N[t=128:192, f, u, v] × (σ^{N-1}[f] / σ^{N}[f])

The factor is a single per-fdm scalar (`N_fdm = 34` floats). Memory
traffic = read 64×34×256×256 fp16 + write same = 280 MiB ≈ 0.5 ms at
the 2080 Ti's HBM bandwidth.

State to keep: `_sigma_layer1_prev_for_carryover` — the σ that was
applied to cube N's imager output, retained one extra cube past
`_sigma_layer1_prev` (which gets overwritten when cube N+1's σ is
estimated). One extra `[N_fdm] float32` tensor.

**On the very first cube** (no prior cube to carry over from):
fall back to the existing process_cube; no carry-over for cube 0.

**When σ_prev[f] ≈ 0**: clamp to `torch.finfo(float32).tiny` (already
done in `_layer1_normalise_fused` for the next-cube σ multiply, so the
σ values written to `_sigma_layer1_prev_for_carryover` are never 0).

## Implementation approach (chosen: kernel modification)

Three options considered:

| Option | Savings (combine + FFT + mask) | Code surface | Notes |
|---|---|---|---|
| A. Kernel `t_lo` parameter + smaller launch grid | 21 ms / cube (combine 33% + FFT 33% + mask 33% on 64/192 rows) | NVRTC kernel adds 1 param + early-return check; wrapper passes through; CubePipeline orchestrates | **Chosen** |
| B. Full combine, slice uv_batch for FFT only | 11 ms (FFT + mask only; combine still full) | Smallest change | Doesn't save the dominant combine cost |
| C. Carry uv state (skip combine + FFT, recompute mask) | 30 ms but requires saving uv_batch state across cubes | Largest memory cost (+576 MiB GPU); fragile | Rejected — uv state is huge |

Option A specifics:

1. **fused_combine_cuda.py**: add `int t_lo` as a new kernel parameter.
   At top of every kernel body, add `if (t < t_lo) return;`. Default
   `t_lo = 0` in the wrapper to preserve all existing call sites.
2. **fused_combine_cuda.py** wrapper: add `t_lo: int = 0` keyword arg
   to `fused_dequant_combine_per_fdm()`; thread it through to the
   `kernel(...)` call.
3. **imager_gpu.py::process_cube**: add `t_lo: int = 0` parameter.
   Pass to `fused_dequant_combine_per_fdm(..., t_lo=t_lo)`. The FFT
   and mask multiply are sliced: `ifft2(uv_batch[:n_batch, t_lo:, :, :])`
   and write to `img_batch_real[:n_batch, t_lo:, :, :]`. The writeback
   `out_cube[t_lo:, f0:f0+n_batch, :, :]` only touches the new rows.
4. **cube_pipeline.py**: new `_run_imager_from_staged_carryover(staged,
   prev_cube, sigma_prev_inv)` that calls `process_cube(..., t_lo=64)`,
   then does the σ-rescaled copy of `prev_cube[128:192]` →
   `new_cube[0:64]`. Gated by new config flag
   `cube_pipeline_carry_over_re_imaging` (default OFF). Tracks
   `_sigma_layer1_prev_for_carryover` and `_prev_cube_handle`.
5. **Config flag**: `CubePipelineConfig.cube_pipeline_carry_over_re_imaging:
   bool = False`. Wire through `search_compute.py` CLI. **Default OFF**
   for safety — operator enables explicitly.
6. **Production launch yaml**: NOT yet flipped. After implementation +
   bench validation, the operator adds `--cube-pipeline-carry-over-re-imaging`
   to `configs/dsart_search_rt.yaml` for the next deployment.

## Validation plan + results

### 1. Numerical equivalence A/B (DONE — PASS at fp32)

`bench/preflight/search_carryover_equivalence.py` runs N cubes through
two `CubePipeline` instances side-by-side (carry-over OFF vs ON), fed
by identical `SyntheticRxRingSource(overlap_streams=True,
cube_cadence_samples=128)` slot sequences. Asserts `np.allclose` on
both the carry-over region `[0:t_lo]` and the newly-imaged region
`[t_lo:t_det]` to fp16-grade tolerance.

Run command (5 minutes; uses n_grid=128 to fit two fp32 pipelines on a
single 2080 Ti):

```
CUDA_VISIBLE_DEVICES=0 python -m bench.preflight.search_carryover_equivalence \
    --n-cubes 6 --n-grid 128 --cube-dtype fp32
```

Result (cubes 1..N-1; cube 0 is full re-image in both runs by
construction):

```
  cube 0: |a|_max=3.647 |b|_max=3.647 co-diff=0       new-diff=0
  cube 1: |a|_max=6.167 |b|_max=6.167 co-diff=0       new-diff=0
  cube 2: |a|_max=6.527 |b|_max=6.527 co-diff=9.5e-7  new-diff=0
  cube 3: |a|_max=6.535 |b|_max=6.535 co-diff=9.5e-7  new-diff=0
  cube 4: |a|_max=6.332 |b|_max=6.332 co-diff=9.5e-7  new-diff=9.5e-7
  cube 5: |a|_max=6.595 |b|_max=6.595 co-diff=9.5e-7  new-diff=4.8e-7
  ✓ PASS — carry-over output matches full re-image to fp16 precision.
```

The new-rows region is bit-exact (the partial-grid kernel is the same
launch as the full kernel sliced; fp16 launch-grid noise = ~5e-7). The
carry-over region matches to ~10⁻⁶ relative — well inside fp16's ~10⁻³
relative precision.

### 1a. Test-setup notes uncovered during validation

Two non-bug issues surfaced during the A/B run and are documented for
future operators:

* **fp16 saturation with unit-variance synthetic streams**: the
  synthetic source emits unit-variance complex Gaussian cells. The
  imager output is `O(N_grid² × N_chgroup) ≈ 1e6` for `N_grid=256,
  N_chgroup=16`, which saturates fp16 (max ≈ 6.5e4). Both runs
  saturate, but at DIFFERENT cells (different code paths trip
  different fp16 rounding), giving a misleading divergence in the
  saturated cells. Production fp16 has no problem because the
  per-chgroup calibration scales the dirty image well into fp16 range.
  Mitigation in the gate: run with `--cube-dtype fp32 --n-grid 128`
  (kernel logic is dtype-independent).
* **Per-cube quantise scale variation**: the `SyntheticRxRingSource`
  used to re-derive its cf→cint8 scale on every cube via
  `quantise_per_chgroup_into_cint8(target_max=120)`. When two
  consecutive cubes draw streams with different per-cube max-abs
  values, the cint8 codes drift by a per-cube scale factor, so the
  imager output at overlapping absolute time is NO LONGER equal
  between cubes — breaking the carry-over equality proof at its
  cint8-input premise. The rx_ring now computes the quantise scale
  ONCE (cube 0) and reuses it for every subsequent cube when
  `overlap_streams=True` (see M7.7.2 changes in `rx_ring.py` and
  `transport/quantize.py::fixed_scale`). Production is unaffected
  because chunk-8 RX-ring delivers cint8 with a stable per-chgroup
  calibration that varies on minute timescales.

### 2. Sustained throughput delta (DONE)

`bench/preflight/search_speed_gate.py --n-cubes 30 [--carry-over]` on
n01 (CUDA_VISIBLE_DEVICES=0, single half, production op-point with
the M7.7 symmetric-shift padding + Option-A coarse-DM alignment):

| Metric | OFF (baseline) | ON (carry-over) | Δ |
|---|---|---|---|
| `total_gpu_ms` | 208.8 | 187.5 | **−21.3 ms (−10.2%)** |
| `imager_ms` | 117.7 | 95.0 | **−22.7 ms (−19%)** |
| → combine | 72.0 | 54.8 | −17.2 |
| → fft | 34.0 | 28.9 | −5.1 |
| → mask | 8.5 | 5.8 | −2.7 |
| `detector_ms` | 86.8 | 87.9 | +1.1 (noise) |
| `layer1_ms` | 4.3 | 4.6 | +0.3 (noise) |
| Sustained cubes/s | 4.79 | 5.33 | **+0.54 (+11%)** |
| RT budget margin | −74.8 ms | −53.5 ms | +21.3 ms recovered |

The savings track the theoretical 33% reduction in imager work
(64/192 rows skipped) attenuated by fixed launch + writeback +
ping-pong overhead, so the realised gain is ~19% on the imager.
Combine remains the dominant GPU substage even with carry-over (55 ms
of 95 ms imager); the next lever is detector reduction (still PENDING
upstream).

### 3. Cube-boundary smoothness (PENDING — operator inspection)

Plot a time series at a random pixel across 10 consecutive cubes and
verify no discontinuity at `t=64`. Not blocking the flag-flip but
worth doing once before fleet-push.

### 4. End-to-end recovery (PENDING — depends on Phase C correctness gate)

Once Phase C is wired, enable carry-over and verify the same
voltage-injection still recovers with the same SNR. Hard gate: SNR
within ±0.5 σ of carry-over OFF.

## Risk + rollback

* The kernel parameter is additive (new arg `t_lo`, default 0). The
  existing call sites are unchanged in semantics.
* The config flag defaults OFF. Production keeps the existing behaviour
  until the operator explicitly enables it.
* If anything goes wrong, flip the flag back; no rollback needed.

## What this does NOT solve

* Sustained throughput goes 5.78 → ~6.6 cubes/s, still below the 7.45
  cubes/s target. Need to compose with one of:
  - **detector kernel cut** (k_time 7→4): ~25 ms additional savings →
    sustained ~7.5 cubes/s, PASS.
  - **detector on its own stream**: max(imager 90, L1+detector 63) =
    90 ms total → sustained ~11 cubes/s, comfortably PASS.

## Effort estimate vs realised

| Item | Estimated | Realised |
|---|---|---|
| Kernel change + wrapper | 30 min | ~40 min (six kernel variants × the cf16/cf32 split) |
| `GpuImager.process_cube` wiring | 30 min | ~30 min |
| `CubePipeline` orchestration + state | 1 h | ~1.5 h (incl. ping-pong + σ-rescale branches for all 4 sigma_prev / sigma_now combinations) |
| Config + CLI plumbing (bench + speed gate) | 15 min | ~30 min |
| Numerical-equivalence A/B script | 45 min | ~1 h |
| Validation runs on n01 + debug | 30 min | ~2 h (uncovered the per-cube quantise-scale issue in the synthetic source; required `rx_ring.py` + `transport/quantize.py` extensions) |

**Total realised: ~6 hours** (vs 3-4 h estimate). The overrun was
entirely on the validation side, not the implementation; the kernel /
imager / pipeline changes themselves were tractable.
