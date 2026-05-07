# M5 hardening rollup — fold into plan.md

**Status**: ready for operator review (2026-05-07).
**Source**: `M5_PLAN_FIXES.md` (D1–D28 + F1–F13).
**Target**: `dsa110-rt_revamp_7b1d2669.plan.md` (operator-managed canonical).

This document is the chunk-8 hardening artifact — every F-item from
`M5_PLAN_FIXES.md` rewritten as a concrete plan-edit instruction (what to
change, where, and the exact replacement prose) plus every D-item compressed to
a one-line canonical statement suitable for a "M5 locked decisions" appendix.

After folding into plan.md, drop both `M5_PLAN_FIXES.md` and this file in a
single hardening commit; M5 will roll from `complete (approved)` to
`complete (hardened)` automatically (`tools/dod/M5.sh::status_emit` checks
`plan_fixes_tracker_present`).

The order below matches the order each item should be applied in plan.md (top of
file → bottom of file). Both files (`M5_PLAN_FIXES.md` + this rollup) carry the
full back-reference detail — the prose below is the **terse plan form** only.

---

## 1. Plan edits — F-items

### F8 (apply first; renames downstream refs)

**§3 line 94** (detector subpackage decomposition).
Replace `interface.py / v1_deterministic.py / decoder.py / noise_norm.py`
with: `forward.py / decoder.py / merger.py / kernels.py / noise_norm/{layer1,
layer2}.py`. Detector Protocol + the v1 deterministic conv-bank Module live
in `forward.py`; cross-kernel SNR-sort + 4D merge-radius suppression splits out
into `merger.py`; kernel-bank construction is `kernels.py`; noise norm is a
two-file directory split (per-layer testability). `configs/config_compute_search.yaml::detector_class`
points at `dsart.detector.forward.DeterministicDetector`.

### F1

**§3.1 (contracts)** — add `CubeInjectionConfig` frozen contract entry.
Fields `(l_pix: int, m_pix: int, fine_dm_idx: int, t_in_cube: int, snr: float,
profile: str, width_samples: int)`. Serialiser to/from yaml/npz library.
Cross-reference `tools/build_injection_library.py` schema.

### F7

**§8 line 2314** — clarify the v1 / v2 detector-image-kernel split.
Append parenthetical: "(v1: delta kernels for all four image-kernel slots
`('unit', 'psf', 'psf_shift_lm', 'psf_shift_l')`; v2 introduces matched-filter
PSFs when the production `search_compute` consumes pre-imager sparse-COO
tensors)".

### F2

**§8 line 2329** — clarify the cube_injection cube dtype.
Append parenthetical to "`[T_det, N_fine_DM, N_grid, N_grid] float32`": "(real-
valued post-imager dirty image; σ = 1 per cell after Layer-1 normalisation)".

### F11 (paired with F2)

**§8 line 2329** — clarify the cube_injection σ_layer1 input to the detector.
Append: "(the cube is already in post-Layer-1 normalised units; the per-cube
σ-clipped global scalar from `noise_norm/layer1.py` is unity by construction
for cube_injection cubes — Layer-1 is a no-op on already-unit-σ data)."

### F10

**§8 line 2329** — disambiguate (l, m) units in the cube_injection schema.
After "`(l, m, fine_dm_idx, t_in_cube, snr, profile, width_samples)`" insert:
"(`l` and `m` here are integer cube-pixel indices `l_pix, m_pix ∈ [0, N_grid)`
with `(N_grid//2, N_grid//2)` = the phase center — distinct from the sky-cosine
(l, m) used in §3.6.5 / §4.4. A `CubeInjectionConfig.from_lm_radians(...)`
classmethod converts when needed.)"

### F5

**§8 line 2329** — pin the MockTriggerListener port for the cube_injection bench.
After "MockTriggerListener on `127.0.0.1`" insert: "`:11227` (same listener and
port as the §8 line 2328 trigger_emitter_wiring bench)".

### F12

**§8 line 2329** — pin the cube_injection bench → viz contract.
Append at end of paragraph: "The bench produces three artifacts consumed by
`tools/viz/search_detector_check.py --mode cube_injection`:
(a) `injection_log.ndjson` — one record per (snr, width) cell with fields
`{kind: 'injection', injected: {…}, n_trials, n_recovered, recovered_snrs,
matched_kernel_id, score_per_kernel_at_match: {kernel_id → snr},
cube_geometry: {T_det, N_fdm, N_grid}}`;
(b) `noise_only_log.ndjson` — one record per noise-only cube with `{kind:
'noise_only', cube_id, candidate_snrs, cube_geometry}`;
(c) `summary.json` — config snapshot + cells + far table."

### F3

**§8 line 2329** — pin the FAR effective cell count for the analytic Gaussian-tail check.
After "FAR on the noise-only background within [0.5×, 2.0×] of analytic at θ=8"
insert footnote:
"Analytic per-cube expected count at threshold θ:
`N_eff_per_cube_per_kernel ≈ (T_det · N_fine_DM · N_grid²) / (K_img · K_dm · K_time)`,
summed over the 128 kernel triples. Pinned in `noise_norm/` docstrings + the
FAR-check helper."

### F13 (paired with F3)

**§8 line 2329** — relax the literal θ=8 FAR check to a curve-shape check.
Replace "FAR on the noise-only background within [0.5×, 2.0×] of analytic at θ=8
across 30 s of synthetic cubes" with:
"The bench produces the empirical-vs-analytic FAR curve over θ ∈ {6, 7, 8, 9, 10}.
The operator inspects the *shape* of the falloff (it should track the analytic
Gaussian tail across the observable range). The {0.5×, 2.0×} bound is asserted
at the lowest θ with `N_expected ≥ 10` (typically θ=6 at the bench's grid sizes;
θ=8 itself is sub-1-event at bench-scale geometries and is unobservable)."

### F9

**§3.6.13 line 1131** — narrow the forbidden ops list for `decoder.py`.
Replace the blanket forbid with: "`{F.conv1d, F.avg_pool1d}` are forbidden in
`decoder.py` (they are sum substitutes that would burn 10× the FLOPs of the
cumsum-difference form). `F.max_pool1d` and `F.max_pool3d` are PERMITTED in
`decoder.py` (per §1588: `F.max_pool3d` on (fdm, l, m) followed by `F.max_pool1d`
on time is the prescribed local-max NMS implementation). In `forward.py` /
`kernels.py` (where the conv bank lives) all three remain forbidden."

### F4

**§8 lines 2316-2329** — pin which DoD benches gate M5 closure under M3 ∥ M5 parallel development.
Add note after the bench list: "Per `PARALLEL_AGENTS.md` §1, only `voltage_
fixture_search.py` (line 2330) depends on M3. The other three benches (`search_
node_throughput`, `noise_norm_calibration`, `trigger_emitter_wiring`) plus the
new `cube_injection_detector.py` (line 2329) are M5-internal and run on h01
alone. M5 lands all four non-fixture benches before M3 / M4a finish; M5 closure
waits only on the voltage-fixture gate operator sign-off. `M5.sh`'s status JSON
tracks `chunks_complete / total_chunks` so M3 / Ops can see the partial-progress
state."

### F6 (RESOLVED 2026-05-06; pin captured-NPZ schema in plan)

**§8 lines 2291 + 2330** — pin the M3 → M5 captured-NPZ contract.
Per-chgroup `.npz` files at `/home/ubuntu/data/m5_fixtures/<run_id>/chgroupNN.npz`
carry F26 sparse-COO (`vis_cube_sparse: complex64 [N_DM=1, n_fv_total, N_filled]`
+ `ix_row, ix_col: uint16 [N_filled]` + `pattern_id, n_grid, n_filled,
dec_deg_quant, kernel_support, antpos_hash, chgroup_table_hash`) + antenna
provenance (`antpos_e, antpos_n, is_core_baseline_mask`) + scalar config
(`chgroup, t_int_fast_native, t_int_fast_us, n_fv_total, n_blocks_processed,
cell_lambda, phi_lat_ovro_deg, obs_dec_deg`) + T2 truth (`src_kind, src_name,
src_{ra,dec}_deg, src_mjd_trigger, src_dm_pc_cc, src_t2_snr` — NaN for continuum)
+ provenance (`run_id, cal_path, voltage_path, git_sha, utc_iso`). Cross-chgroup
`manifest.json` carries `{milestone, purpose, run_id, src_kind, src_name,
src_truth, obs_dec_deg, t_int_fast_native, t_int_fast_us, n_chgroups, chgroups,
per_chgroup, git_sha, utc_iso, n_baselines, phi_lat_ovro_deg}`. M5 consumer is
`src/dsart/transport/captured_npz.py` (`load_captured_run`, `stack_dense_streams`).
Producer is `bench/m3_emit_m5_fixtures.py` (M3 worktree).

---

## 2. Locked-decisions appendix — D-items

Insert as a new appendix (suggested title: **"Appendix M5.D — locked decisions
2026-05-05 → 2026-05-07"**). One row per D-item; back-references in
`M5_PLAN_FIXES.md` carry the full implementation history if needed.

| ID | One-line statement |
|----|--------------------|
| D1 | `bench/cube_injection_detector.py` operates on post-imaging `[T_det, N_fine_DM, N_grid, N_grid] float32` cubes synthesised by `inject/cube_injection.py`; bypasses every upstream stage (no corr, no transport, no RX ring, no fine-DM combiner, no imager). |
| D2 | Default detector kernel-bank shape: 4 image × 4 DM × 8 time = **128 kernel triples** per `DETECTOR_IMAGE_KERNELS` / `DETECTOR_DM_KERNELS` / `DETECTOR_TIME_KERNELS` in `src/dsart/common/constants.py`. Kernel id schema `"k_img:k_dm:k_time"` is enforced by `Candidate._check_kernel_id`. |
| D3 | h01 test-isolation envelope: `DSART_BUFFER_KEY_PREFIX=m5` (so `fada → fa5a`, `bada → ba5a`, `dada → da5a`); `CUDA_VISIBLE_DEVICES=1`; `DSART_ETCD_NAMESPACE_PREFIX=m5`; lockfile `/var/lock/dsart-m5.lock`; status JSON `~/dsart-m5-status.json`; reports under `bench/reports/<UTC>/<run_id>/M5/`; operator-approval marker `bench/reports/M5/m_operator_approved.yaml`. Wired into `tools/dod/M5.sh`. |
| D4 | M5 voltage-fixture gate fixture: `/home/ubuntu/data/voltages/250924mptq/` — burst, DM ≈ 404.7 pc cm⁻³, RA = 307.78°, Dec = 53.85°, MJD ≈ 60942.172, T2 SNR ≈ 30. Identical to the M3 sub-DoD fixture (plan §8 lines 2286 / 2291). |
| D5 | M5-owned viz: `tools/viz/search_helpers.py` + `tools/viz/search_detector_check.py`. Edits to the M3-owned `tools/viz/common.py` from M5 only via PR ack'd by M3. |
| D6 | Internal imager `(l, m)` sign convention inherited from M2/F20: negate `(u, v)` once at the table-build site so `np.fft.ifft2` (positive-exponent iFFT) lands TMS-canonical `(+l, +m)`. Document with `# F20` comments. |
| D7 | Operator-approval marker scheme reuses M2's D11 verbatim. Single yaml at `bench/reports/M5/m_operator_approved.yaml` covers BOTH M5 operator gates (cube-injection synthetic gate with `voltage_run_id="cube_injection"` AND voltage-fixture gate with `voltage_run_id="<burst_run_id>"`). M5.sh stamp logic mirrors M2.sh (`complete (needs operator approval)` → `complete (approved)` → `complete (hardened)`). |
| D8 | Cube-injection synthetic noise: cube tensor is `float32` real-valued (post-imager dirty image); thermal noise is iid Gaussian σ=1 per cell so SNR readouts are direct (pixel value / 1.0). Injection amplitudes injected at the requested SNR directly. Cube does NOT carry complex visibilities. |
| D9 | MockTriggerListener port: `127.0.0.1:11227` (per plan §8 line 2328); shared by both the trigger_emitter_wiring bench and the cube_injection_detector bench. |
| D10 | v1 detector image kernels (`unit`, `psf`, `psf_shift_lm`, `psf_shift_l`) ALL ship as 1×1 delta-function stubs (`# TODO(v2)` in `kernels.py`). PSF-aware matched filters are a v2 deliverable when the production `search_compute` (Chunk 6) consumes pre-imager sparse-COO tensors and the detector applies the PSF as a matched filter. |
| D11 | (RETIRED Chunk 3.) Layer-2 σ_k placeholder during chunks 1-2 was the analytic per-kernel constant `s_k = sqrt(k_dm_width × k_time_width)`; replaced by the Layer-2 EMA in Chunk 3. |
| D12 | v1 cube-injection profile is **boxcar** only: temporal pulse of duration `width_samples` at peak amplitude `snr / sqrt(width_samples)` per cell on top of unit-σ Gaussian thermal noise (D8). Other profile values raise `NotImplementedError`. |
| D13 | Layer-2 EMA cold-start seed: `Layer2State._s_k` initialised to `1.0` per kernel (canonical Welford); `DeterministicDetector(layer2_seed_unit=True)` (default) seeds the EMA to `sqrt(k_dm_width × k_time_width)` so the first cube divides by a sensible scalar before Welford burn-in completes. Tests / benches that need raw cold-start pass `layer2_seed_unit=False`. |
| D14 | Layer-2 invalid-cube semantics: an invalid cube (any `validity_mask` cell False) skips both `Detector.forward()`'s noise-update wiring and the EMA `cube_count` increment (so burn-in stays correct in calendar-cube terms). Forward still runs the conv-bank + decoder + merger (the candidate log records what the detector saw); per-(t, fdm) masking is plumbed through to the EMA via a future `validity_mask_per_kernel` argument. |
| D15 | `bench/cube_injection_detector.py` predicate-chain knobs deliberately RELAX the production trigger-emit chain so the FAR sub-check sees the full noise-only emit distribution: `SnrThreshold = 5.0σ` (vs production 8.0), `PerCubePerKernelCap = 128` (vs 4), `PerCubeTotalCap = 1024` (vs 16), `RateLimitTokenBucket(rate=1e6/s, burst=1e6)`. Holdoff stays at production 50 ms. Production `services/search_compute.py` reads from `configs/config_compute_search.yaml` and is unaffected. |
| D16 | Cube-injection bench recovery match-window: an emitted Candidate matches an injection when `(Δl ≤ 2, Δm ≤ 2, Δfdm ≤ max(2, k_dm/2+1), Δt ≤ max(64, k_time/2+1))`. lm absorbs the §1585 NMS spatial radius; fdm grows with K_dm; time grows with K_time and absorbs the §1592 time-edge gate. Helper at `bench._kernel_match_radius`. |
| D17 | Bank-mask CLI for both detector benches (`cube_injection_detector.py` and `search_node_throughput.py`): `--bank-mask "k_img=<tokens>;k_dm=<tokens>;k_time=<tokens>"` (parser shared in `bench/_bank_mask.py`). Each axis defaults to `*` (= keep all). Operators sweep `K_img × K_dm × K_time` at fixed cube geometry to characterise the perf-vs-recovery Pareto. The throughput bench also takes `--n-grid`. |
| D18 | Empirical Chunk 6c finding: at v1 with delta image kernels (D10) and on-grid (fine_dm_idx-aligned) boxcar injections, the per-cell `snr_ratio_mean` is identical across `{full, k_img=unit, k_dm=d1, k_img=unit;k_dm=d1}` to 4 decimal places. Production v1 bank can be safely collapsed to `k_img=unit; k_dm=d1` (8 kernels) for on-grid pulses; K_time stays full. v2 (matched-filter PSFs + DM-smeared injections) re-opens the question. |
| D19 | Imager-only GPU throughput baseline (cuFFT-cfp16, h01 GPU 1, RTX 2080 Ti, python_addloop combiner): T_det=512→1.81; T_det=384→2.39; T_det=256→3.52; T_det=192→4.62; T_det=128→6.69 cubes/s. Per-cube cost dominated by the 16-chgroup index-shifted sum (~70%) at ~510 GB/s effective vs ~616 GB/s peak HBM (~3× headroom for a fused single-pass kernel). cfp32 vs cfp16 measured 524 vs 283 ms / cube at T=256 (exact 2× memory traffic). |
| D20 | Chunk 6c follow-up: cupy-NVRTC fused per-fdm combine kernel at `src/dsart/image/fused_combine_cuda.py` (NVRTC chosen over `torch.utils.cpp_extension.load_inline` because h01 sits in a host-gcc/nvcc compatibility sandwich). Memory traffic drops from 49× to 17× slab volume; 2.7× combine speedup. **Plan §8's 8-cubes/s target met at T_det=128 (10.74 cubes/s)**; T_det=192 borderline (7.71); T_det=256 / 384 / 512 → 6.01 / 3.91 / 3.19 cubes/s. |
| D21 | Chunk 8 fused dequant+combine (cint8-input variant): `fused_dequant_combine_per_fdm_cint8_to_cf{16,32}` reads cint8 streams in M3-native split-plane re/im layout, accumulates per-fdm in int32 registers (exact for N_chg ≤ 16), single fp16 cast at end. **Plan §8's 8-cubes/s target met at T_det=256 (9.79 cubes/s, the operator-pinned v1 deployment integration time)**; T_det=192 → 12.62; T_det=128 → 18.18 cubes/s. Wrapper `src/dsart/image/imager_gpu.py::GpuImager` (workspace pre-alloc + schema validation + `build_default_gpu_imager`). End-to-end 10.60 cubes/s (94.3 ms/cube) at T_det=256 / N_fdm=32 / N_grid=256 / cfp16. Strictly more accurate than chunk-6c reference (int32 acc is exact). |
| D22 | M3 → M5 captured-NPZ loader (Chunk 7, F6 closure): `src/dsart/transport/captured_npz.py` (read-only). `load_captured_run` returns `Dict[chgroup_idx, CapturedChgroup]` + `CapturedManifest`. `CapturedChgroup.scatter_dense` materialises F26 sparse-COO back to dense `[N_DM=1, n_fv_total, N_grid, N_grid] complex64`. `stack_dense_streams` assembles the production-shape `[16, n_fv_total, N_grid, N_grid] complex64` stack with zero-fill + `valid_mask`. Two h01 fixtures verified end-to-end: `0319` continuum (15/16 chgroups; sb12 known M2 gap) and `250924mptq` burst (16/16 chgroups; T2 truth populated). |
| D23 | Chunk 7 captured-mode end-to-end gate (PASS, 250924mptq): global peak at `(t=253, fdm=24, l=142, m=198)`, max/std SNR 10.75 σ; recovered DM 408.71 pc/cc vs labelled 404.69 (+0.97% offset). Spatial consistency 31/32 fdm trials within 2 px of the global peak. Recovered (l, m) is off-boresight by 71 pixels — expected and realistic per operator. PASS gate replaces the original boresight-distance gate with a per-fdm spatial-consistency gate: `--recovery-snr` (default 8) AND `≥ --recovery-consistency` (default 0.5) of fdm trials' top peaks within `--recovery-pix-tol` (default 2) pixels of the global peak's (l, m). Negative-control sanity check: a deliberately wrong DM range still picks the burst at the right (l, m) — DM-discrimination signal lives in the `dm_curve_at_peak_lm` block, not in cube_max alone. |
| D24 | (SUPERSEDED by D25.) Chunk 7 detector hardening end-to-end (250924mptq) measured 17.89 σ at K_time=8 with 1.64× MF boost — but the boost was a sign-aliasing artifact of the broken D25 sign convention. Post-D25 the matched-filter envelope scales coherently to 57.50 σ at K=128. |
| D25 | **§3.6.3 lock-in.** Fused-combine CUDA kernels (`fused_combine_per_fdm_cf{16,32}` + `fused_dequant_combine_per_fdm_cint8_to_cf{16,32}`) shipped initially with `out[t] = sum_g streams[g, t + shifts[g]]` (PLUS) but `compute_time_shift_search` builds its shift table for the §3.6.3 / `combine_chgroups` MINUS convention `out[t] = sum_g streams[g, t - shifts[g]]`. **Fix**: invert kernel index expressions to `t - s` with bounded `0 ≤ t_src < t_stream` (4 kernel sites + 2 Python references + 2 docstrings). Two new lock-in tests at `tests/test_fused_combine_cuda.py::test_fused_*_matches_combine_chgroups` plant a synthetic dispersed pulse and assert coherent peak lands at cube-time `t_15` with magnitude `N_chgroup`. Headline impact on 250924mptq: max/std SNR 10.75 σ → **50.19 σ** (4.7× boost) at the recovered burst; per-K_time scales to **57.50 σ at K=128** as expected. |
| D26 | Chunk-9 production-readiness throughput pass on `bench/search_node_throughput.py --image-backend gpu` lands four orthogonal fixes: (a) streaming per-chgroup cf32→cint8 quantiser at `transport.quantize.quantise_per_chgroup_into_cint8` (~80 MiB/chgroup vs ~13 GiB transient); (b) chunk-8b RX-ring emulation via `CubeRingSlot.per_chgroup_cint8_stack` + `SyntheticRxRingSource(pre_quantise=True)` (build_cube 4185 ms → 218 ms); (c) Layer-1 σ-clip subsample knob `Layer1State(max_samples=1_000_000)` (Layer-1 norm 782 ms → 92 ms); (d) v1-collapsed-bank cumsum-once amortisation (`detector.forward.{precompute_padded_cumsum, boxcar_from_padded_cumsum, _get_or_build_amortise_cs}`) with W-tiled narrow subtract (detector forward 1074 ms → 595 ms). Headline end-to-end at T_det=256 / N_fdm=32 / N_grid=256 / fp16 / collapsed bank / 30-cube sustained: **0.68 cubes/s** (build 219 ms · l1 92 ms · detector 595 ms · emit 2 ms · total p50 907 ms / p99 1430 ms · 173 cands/cube). Detector remains binding at 65% wall-clock; per-kernel σ_clipped_std + topk + Layer-2 EMA is the v2 optimisation surface. Sharp edge fixed: `precompute_padded_cumsum`'s in-place `cumsum_` after `x.to(dtype)` corrupted callers when dtypes matched (because `.to(dtype)` returns self) — fixed via `torch.cumsum(x, dim, dtype=accum_dtype, out=target)`. |
| D27 | Chunk 7 closure: `bench/voltage_fixture_search.py --mode captured --detector-sweep` delegates to `bench.captured_burst_detector._bench_main` and folds the resulting `detector.json` into the canonical M5 closure-gate envelope. Gate semantics: is_burst fixture → PASS iff burst_match present AND matched_snr ≥ threshold AND MF boost ≥ 1.0; negative-control fixture → PASS iff 0 candidates. Operator-approval marker references this bench's `voltage_run_id` (= manifest.run_id). Headline 250924mptq: **PASS** at SNR 20.81 (b1 15.88, MF K_time=4 boost 1.31×); recovered DM 397.42 vs labelled 404.69 (residual −1.80%); 13 post-merge candidates (6 burst-consistent + 7 off-burst persistent-source/RFI). Defaults: `--detector-t-det 256` (D26 v1 deployment), `--detector-n-fdm 32`, `--detector-shift-offset 125` (places the 250924mptq burst at cube_t ≈ 128), `--detector-coarse-dm 0.0` (250924mptq is empirically NOT pre-stage-2 dedispersed per D23 caveat). NMS radii pulled directly from `dsart.detector.merger.DEFAULT_MERGE_RADIUS_*`. |
| D28 | Chunk 8(c) — production-ready dirty-image output in physical visibility units. New NVRTC kernels `fused_dequant_scale_offset_combine_per_fdm_cint8_to_cf{16,32}` apply `z[g] = scale[g] * cint8[g] + offset[g]` per chgroup with fp32 accumulation. The unit-scale int32-acc fast path (D21) is preserved as the default — bit-exact when `scales` / `offsets_re` / `offsets_im` are unset. Plumbed through three layers: `fused_dequant_combine_per_fdm` kwargs → `GpuImager.process_cube` kwargs → `CubePipeline._build_cube_gpu` (chunk-8b production path: M3 ships per-chgroup calibration alongside the cint8 stack). Bench fallback path computes `1 / quantise_global_scale` and broadcasts it across all 16 chgroups so bench output is also in physical units. New `CubePipelineConfig.bake_quantise_scale: bool = True` toggles the bake-in. Schema additions: `CubeRingSlot.per_chgroup_{scale, offset_re, offset_im}` (`Optional[np.ndarray]` length-N_chg float32). 20 new tests (13 in `test_fused_combine_cuda.py` + 3 in `test_imager_gpu.py` + 4 in `test_cube_pipeline_gpu.py`); full M5 suite **486/486 pass on h01**. No perf regression. |

---

## 3. Post-fold checklist

After the operator folds the above into plan.md and re-locks:

1. Delete `M5_PLAN_FIXES.md` (`git rm M5_PLAN_FIXES.md`).
2. Delete `M5_HARDENING_ROLLUP.md` (this file) (`git rm M5_HARDENING_ROLLUP.md`).
3. Commit with message `docs(M5): hardening — fold M5_PLAN_FIXES into plan.md, retire trackers`.
4. Re-run `tools/dod/M5.sh status` on h01; the status JSON's `plan_fixes_tracker_present` flips to `false` and the stage label rolls from `complete (approved)` to `complete (hardened)`.

---

## 4. Outstanding post-M5 follow-ups (not blocking M5 closure)

Both items below are explicitly post-M5-ship and tracked in the M5 todo list:

- **ch8b RX-ring integration test** — verify M3 → M5 RX-ring delivers cint8
  streams pre-staged on GPU. Depends on M3 chunk-9 RX-coupling integration test
  landing on real GPU staging. The M5 schema (`CubeRingSlot.per_chgroup_cint8_stack`
  + `per_chgroup_scale` + `per_chgroup_offset_*`, see D26 / D28) is already in
  place to receive the M3 hand-off.

- **ch8c real-calibration bake-in** — production now consumes per-chgroup
  `(scale, offset)` from `CubeRingSlot` (D28); the `bench/search_node_throughput.py`
  prequantise path uses the global `1 / quantise_scale` broadcast which is
  pessimistic vs M3's per-chgroup calibration. When M3 emits real per-chgroup
  calibration alongside the cint8 streams, the bench can switch to it and the
  end-to-end throughput numbers in D26 may shift.
