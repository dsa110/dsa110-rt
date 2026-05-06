# M3 plan fixes + decisions tracker

This file mirrors the M2 pattern (see git log for `M2_PLAN_FIXES.md`):
F-items are *fixes* (corrections to plan.md or implementation choices that
deviate from the plan as written and need to be folded back), D-items are
*decisions* locked during implementation. Both accumulate here during M3
development and are folded into `dsa110-rt_revamp_7b1d2669.plan.md` during
M3 hardening (the final M3 chunk), at which point this file is deleted.

For binding M2 context that M3 inherits verbatim — and the F18/F20/D8/D11/
D13/D14/D15/D16/D17/D18 items in particular — see plan.md `§8.M2-carryover`
(lines ~2198-2261). Do not duplicate M2 context here.

For parallel-agent file-ownership / branch-model / h01-test-isolation
conventions, see `PARALLEL_AGENTS.md`.

---

## F-items (fixes — corrections / additions to plan.md)

### F21 — fast-corr cal-apply must fold the bfCorr DEC-only phase

**Status**: IMPLEMENTED in chunk 1 (`chunk_1_cal_apply_with_F21_dec_phase`,
2026-05-05). Lives in `src/dsart/cal/cal_loader.py` (new, M3-owned;
peer to M2's slow-corr cal pipeline in
`corr_slow_compute._build_cal_tensors`). Four acceptance pytests in
`tests/test_cal_loader_dec_phase.py` pin the sign convention against
`dsaX_bfCorr.cu::populate_weights_matrix` central-beam (iArm==1, bm=127)
to ≤ 1e-11 rad in fp64; `M3.sh` chunk_1 STEP gates on those pytests.

**Implementation notes** (carryover for M3 hardening / plan §4.2 fold):
- `compute_dec_phase` returns complex128 (full fp64) so the F21 fold
  composes losslessly with the cal-blob multiply; the cplx64 → fp16
  cast is deferred to `make_cal_broadcast_tensors` (single point of
  precision loss for the cal tensor, matching M2's slow-corr path).
- TMS Eq. 3.19 sign convention pinned: per-antenna voltage phase
  `+2π f sin(δ_src − φ_lat) N_a / c`; F21 cal weight has the
  OPPOSITE sign `−2π f sin(δ_obs − φ_lat) N_a / c` (cancels the
  geometric phase when δ_obs = δ_src). Module docstring derives
  both from first principles + bfCorr CU-code line numbers.
- F21.4 (bfCorr round-trip) does NOT use the bfCorr literal `37.23`
  for φ_lat — it uses M2's full-precision `PHI_LAT_OVRO_RAD =
  math.radians(37.234)` on both sides, so the test pins the formula
  STRUCTURE / SIGN, not the literal. (M1 plan-fix F10 already
  retired the 4-digit truncation in production code.)

**Problem**: plan.md §4.2 ("cal + bandpass flatten from antennas.out
(legacy binary format ...) cal_mode={phase_only,full} flag, hot-reload via
cmd: reload_cal") describes the cal apply but does not specify the
DEC-only fringe-stop phase that legacy `dsaX_bfCorr.cu::populate_weights_matrix`
folds into the per-(ant, ch, pol) cal weight before the GEMM.

The slow-corr (M2) does not need this fold because `meridian_fringestop`
in casa38 applies fringe-stopping (HA + DEC) downstream, after the slow
GEMM, before UVH5 write. The fast-corr has no such downstream step — it
goes straight from GEMM → grid → iFFT, so the visibility phase **must**
be referenced to the source direction at gridding time, otherwise the
source falls outside the iFFT FoV.

For DSA-110 specifically (meridian-pointing array, HA=0 by construction
for any single-pulse trigger dump), only the DEC component is needed.

**Fix**: extend `cal_loader` (M3, peer to M2's `cal_loader.py` for the
slow path) to populate the per-(ant, ch, pol) GPU cal tensor as

```
cal[a, ch, pol] = bandpass[a, ch, pol] · gain[a, pol]
                 · exp(−2π i f[ch] · sin(δ_obs − φ_lat) · N_a / c)
```

where:
- `f[ch]` = channel center frequency (Hz)
- `δ_obs` = observing dec (rad), supplied via etcd `cmd: prepare` (prod) or
  CLI flag (bench)
- `φ_lat` = OVRO latitude = 37.234° (already a constant in
  `src/dsart/common/constants.py`)
- `N_a` = N-S antenna position (m), from the M2-validated antpos table
- `c` = speed of light

The HA and E-W components are zero (HA=0; E-W cancels at the meridian),
matching `dsaX_bfCorr.cu::populate_weights_matrix` for the central beam
(`bm = 127`, lines 1082-1085 of `dsa110-xengine/src/dsaX_bfCorr.cu`).

**Sign convention**: matches bfCorr's `iArm == 1` branch literally —
`afac = -2π f / c · sin(theta)` with `theta = -(π/180) · (φ_lat − δ_obs)`,
i.e. `θ = δ_obs − φ_lat` in rad. The cal weight is then `cos(afac · N_a) +
i · sin(afac · N_a)`. Consumers of `V_ij = E_i^* · E_j` (per F18) get the
right phase sign for free because the cal weight pre-multiplies E_i.

**M3 plan.md changes (during hardening)**:
- §4.2 cal subsection: add a new paragraph specifying the DEC-only fold
  and the formula above; cross-ref `dsaX_bfCorr.cu` lines.
- §3 add: `obs_dec_deg` to the `cmd: prepare` etcd-key schema.
- §8 M3 DoD: add the `bench/cal_reload.py` test for `δ_obs` → cal phase
  hot-reload (cycle through 3 different obs_dec values, confirm cal
  tensor swaps atomically and the source position in a synthetic-fixture
  image moves to the predicted `(l, m)` for each).

**Tests (chunk 2)**:
- `tests/test_cal_loader_dec_phase.py`:
  - F21.1: synthetic point source at (HA=0, dec=δ_obs) is at (l, m)≈(0, 0)
    in the iFFT image after cal-apply with `δ_obs = δ_src` (within ≤ 1
    grid cell at default `N_grid = 256`).
  - F21.2: same source, with `δ_obs ≠ δ_src` by Δδ, lands at (0, sin(Δδ))
    in the image (within ≤ 1 grid cell).
  - F21.3: two synthetic point sources at `(0, ±0.05)` rad relative to
    `δ_obs` are correctly resolved (no axis flip vs F20).
  - F21.4: bfCorr round-trip — given the same antpos + cal blob + δ_obs,
    M3's cal tensor matches bfCorr's combined weight tensor element-wise
    (`atol=1e-6` in fp32; `atol=2e-3` in fp16) — pinned by reading
    `dsaX_bfCorr.cu::populate_weights_matrix` byte-equivalent.

### F22 — voltage-injector sign convention pinned to F18

**Status**: IMPLEMENTED in chunk 3d (`tests/test_online_injector.py
::test_F22_visibility_phase_matches_lm_target`, 2026-05-06). Lives in
`src/dsart/inject/online.py`. The chunk-3d brief stated a `−sign`
per-antenna phasor; the implementation uses `+sign` so that
`conj(E_lower) · E_higher` (F18 / V_ab convention) lands at the
`+2π i ν (b · ŝ) / c` phase that the F21 acceptance suite already
pinned. Plan §4.2 step 5 (online injection) inherits this convention;
the M3 hardening pass will replace the briefing's `−sign` sentence
with a forward reference to F18 + the F22 acceptance test.

### F23 — exact Nita-Gary chi-squared SK thresholds

**Status**: IMPLEMENTED in chunk 3c (commit `25c77ac`, 2026-05-06).
Lives in `src/dsart/rfi/sk.py` — `sk_thresholds(M, far)` now returns
Monte-Carlo-derived `(sk_low, sk_high)` from a 1M-trial simulation of
`|E|² ~ Exp(1)` per `(M, far)` pair, cached on first call (~120 ms /
M on CPU). The empirical thresholds are equivalent to a Pearson Type
IV moment-matched CDF lookup (the proper Nita-Gary 2010 prescription)
within MC sampling noise. The Gaussian asymptotic form remains
available as `gaussian_sk_thresholds(M, far)` for diagnostic
inspection / asymptotic-large-M sanity checks.

**Background** (the bug this fixes): the Gaussian asymptotic SK
distribution (`SK ~ N(1, 4·(M-1)/((M+2)·(M+3)))`) under-estimates
the upper-tail mass at the lowest accumulation depth `M = 64`. The
SK distribution at M=64 has γ_1 ≈ 1.1 (markedly right-skewed), and
the Gaussian quantile under-estimated the upper-tail FAR by ~40× at
FAR=1e-4 — measured 4.0e-3 false-flag rate against a 2e-4 target on
524k thermal-noise cells (per `test_sk_thermal_noise_far`).

`test_sk_thermal_noise_far` now asserts the canonical 2× FAR bound
across all M (no per-M tolerance multipliers). Backwards-compat:
`gaussian_sk_thresholds(...)` is exported alongside `sk_thresholds`
for any downstream consumer that needs the asymptotic form (none in
production).

### F24 — coarse-DM time shifts are stored in NATIVE samples (not bin units)

**Status**: IMPLEMENTED in chunk 3b (`tests/test_coarse_dm.py
::test_F24_coarse_dm_uses_native_t_axis`, 2026-05-06). Lives in
`src/dsart/coarse_dm/dm_plan.py`. The per-`(chgroup, channel, dm)`
delay table is in NATIVE samples (32.768 µs each, M2 D15);
fast-vis-bin-unit shifts are derived once at apply time via
`round(delay_native / t_int_fast_native)`. This composes losslessly
with the canonical `contracts.DmPlan.time_shift_corr_stage2` (also
native units) without compound rounding.

**Rationale**: storing shifts in fast-vis bin units would couple the
coarse-DM plan to the runtime `t_int_fast_native` knob (default 8 →
262.144 µs cadence; burst-test override 32 → 1048.576 µs). Native
samples decouple them; the lookup table is fixed-once at plan build,
the bin reduction happens at integration time per the active
`t_int_fast_native`.

### F25 — coarse-DM integration shape: vis-domain stage-1 shifts (not post-grid)

**Status**: DOCUMENTED (architectural reconciliation deferred to chunk
6 / 9 integration). Plan §4.2 step 4 specifies stage-1 per-channel
INTEGER-sample time shifts on the visibility tensor `(n_fv, NBASE,
NCHAN)` BEFORE the gridder, with one gridder call per coarse-DM
trial — i.e. the per-trial coarse-DM cube is produced by `grid(time_
shift_corr_stage1(vis, dm_idx))`, not by post-grid manipulation.

Three primitives are involved:

1. **`coarse_dm.dedisp.coarse_dedisp(image_cube, plan, ...)`** —
   chunk-3b's pure-math primitive on `[T_fast, NCHAN, N_grid,
   N_grid]` cubes. Pinned by 18 acceptance tests; bench
   `coarse_dm_recovery.py` confirms exact `(t, l, m)` recovery on
   synthetic burst.

2. **`corr_fast_integration.CoarseDMStage` Protocol** — chunk-4's
   plug-in surface. Currently typed against POST-grid sparse-COO
   `(n_fv, N_filled)` and returns `(N_DM, n_fv, N_filled)`. The
   default `NoOpCoarseDM` returns `gridded.unsqueeze(0)` — a
   no-op-with-trivial-DM-axis adapter that is correct for the
   `N_DM == 1` case but does NOT exercise the production stage-1
   shift path.

3. **`time_shift_corr_stage1` (planned, not yet authored)** — vis-
   domain integer-sample shift. Per plan §4.2 step 4, this is what
   chunk-4 should call before `gridder.compute()` per coarse-DM
   trial. The `DMPlan.delay_native_samples()` lookup from chunk 3b
   provides the data; the chunk-4 integration loop is the missing
   wiring.

**Reconciliation plan** (chunk 6 + chunk 9 work):

- Chunk 6 (`voltage_fixture_burst_250924mptq`) runs with
  `--single-dm` (custom 1-cell DM plan per the user's burst-test
  config), so the no-op CoarseDMStage stub is correct as-is. No
  chunk-4 refactor needed for chunk 6.
- Chunk 9 (`dod_orchestrator_completion`) is where the production
  multi-DM-trial integration wires together. At that point chunk 4
  needs:
  - `process_block` restructured to loop over `dm_idx in plan
    .coarse_indices` after Stokes-I sum: `vis_shifted =
    apply_stage1_shifts(vis_stokes_i, plan, dm_idx)` →
    `gridded[dm_idx] = gridder.compute(vis_shifted)` → static-sky
    EMA per-trial → stage2 fifo per-trial → transport per-trial.
  - `CoarseDMStage` Protocol re-typed to take vis_stokes_i + plan +
    chgroup, return `dict[int, torch.Tensor]` of per-trial shifted
    visibilities. The chunk-3b `coarse_dedisp` primitive remains the
    math reference; the integration loop is the production path.
- This is a 6-9 chunk-of-effort refactor in chunk 9 — tracked as
  D-decision in the chunk-9 brief.

**Why this is safe to defer**: today's stub passes the no-DM case
through correctly, and chunk-4's integration tests pin the
orchestrator's correctness (RFI zero-fill, static-sky, plug-in stage
wiring) for the `N_DM == 1` case which is what chunk 5 (continuum)
+ chunk 6 (single-DM burst) need. The vis-domain stage-1 shifts only
matter once the search starts asking for `N_DM > 1` trials, and
that's chunk-9 material.

### F27 — core/outrigger discrimination must be radius-based, not positional

**Status**: IMPLEMENTED in chunk 7 (2026-05-06; user-spotted bug in the
chunk-3a grid-pattern footprint plot). Lives in
`src/dsart/grid/sparsity_pattern.py::core_baseline_mask_from_antpos`
(new public helper, Class C — corr ↔ search shared); 8 acceptance
pytests in `tests/test_sparsity_pattern.py::TestCoreBaselineMask
FromAntpos` pin the no-regression-on-synthetic-antpos behavior + the
real-h01-cal-blob-differs-from-positional behavior + edge cases
(both-spec error, n_core out-of-range, shape mismatch).

**The bug**: the legacy positional helpers in
`tests/test_sparsity_pattern.py::_core_baseline_mask`,
`tests/test_fast_vis_gridder.py::_core_baseline_mask`,
`bench/grid_pattern_visualisation.py::_core_baseline_mask`, and
`src/dsart/services/corr_fast_integration.py::_build_core_baseline
_mask` all defined "core antennas" as `ant_idx in [0, n_core)`. This
matched the synthetic antpos in the test files (where `_synth_antpos`
explicitly places ants 0..81 in a tight core box) but is **wrong** on
real DSA-110 cal-blob antpos:

| ant_idx | (e_m, n_m)         | r_m   | classification        |
|---------|--------------------|-------|-----------------------|
| 47      | (9.0, 440.8)       | 441   | core                  |
| **48**  | **(-985, -216)**   | **1008** | **OUTRIGGER (positional helper kept it as core)** |
| 82      | (9.0, 432.2)       | 432   | core                  |
| **83**  | **(197.9, -374.1)**| **423** | **CORE (positional helper rejected it as outrigger)** |
| 84-95   | r > 627            | -     | outriggers (correct)  |

The positional mask leaked outrigger-touching baselines into the
gridder (visible as stray fills in the outer uv-plane of
`bench/reports/<UTC>/grid-pattern-bench/M3-grid-pattern/footprint
_chgroup0_dec53p85_ngrid256.png`) AND dropped real core baselines
(missing fills in the core).

**The fix**: `core_baseline_mask_from_antpos(antpos_e, antpos_n, *,
n_core=82 | r_core_m=500.0)` selects the core by physical radius:
either pick the ``n_core`` smallest-radius antennas or apply a
``r_core_m`` (m) cut. The DSA-110 antpos has a clean gap between
the largest core baseline (~441 m) and the smallest outrigger
(~627 m); both the count-based and radius-based specs agree on the
canonical 82-ant core.

**Production**: production code reads `is_core` from etcd
`/cnf/corr_setup_96` (plan §3 line 446) — not affected by this bug.
The chunk-4 `_load_antpos_from_cal_blob` fallback path now uses the
new radius-based helper, matching what production etcd should
return.

**Test files NOT updated**: `tests/test_sparsity_pattern.py::_core_
baseline_mask` and `tests/test_fast_vis_gridder.py::_core_baseline_
mask` still use the positional definition — they're paired with
`_synth_antpos` which places core ants at indices [0, 82), so
positional is correct for those tests' synthetic data. Adding the
new `TestCoreBaselineMaskFromAntpos` class pins the antpos-based
helper without disturbing the legacy positional tests.

---

## D-items (decisions — locked during M3 implementation)

### D-coarse-dm-A — Convention A vs B for DMPlan delay reference

**Status**: LOCKED in chunk 3b. Documented in `src/dsart/coarse_dm/
dm_plan.py` module docstring + this entry.

The chunk-3b `DMPlan.delay_native_samples(g, ch, dm)` uses
**Convention A**: per-chgroup, the reference channel is each
chgroup's own TOP channel, so `delay_native_samples(g, ch=0, dm) ≡
0` for all `g, dm` (channel 0 = top frequency in chgroup `g`). The
canonical `contracts.DmPlan.time_shift_corr_stage1` table (M2
schema) uses **Convention B**: reference is each chgroup's BOT
channel.

Both conventions are equivalent up to a per-`(g, dm)` constant
offset of `max_delay_in_chgroup`. The chunk-3b convention forces
output `t' ∈ [0, T_fast - max_Δ)` which keeps the dedispersed cube
positionally aligned with the input cube's first sample (cleaner
test semantics + matches the chunk-3b briefing's
`test_dm_plan_delay_zero_at_top_freq` pin). Convention B is needed
downstream for stage-2 inter-chgroup alignment to `ν_bot_proc`.

The chunk-4 integration glue (per F25) is responsible for
translating between conventions when wiring into the production
stage-2 path (chunk 9). Both conventions live in the codebase:
canonical `DmPlan` (Class C, contracts) keeps Convention B; chunk-3b
slim `DMPlan` (Class A, owned by `coarse_dm/dm_plan.py`) keeps
Convention A.

---

## Cross-cutting notes for M3 hardening

When folding this file into plan.md (M3 chunk 11):

1. F21 → §4.2 cal subsection + §8 M3 DoD `cal_reload` line.
2. Append per-chunk implementation notes to plan.md `§8.M3-carryover`
   (new subsection between §8 M3 and §8 M4a, mirrors `§8.M2-carryover`).
3. Delete this file in the same commit; M3.sh detects the absence and
   stamps `complete (hardened)`.
