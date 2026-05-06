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

---

## D-items (decisions — locked during M3 implementation)

(none yet)

---

## Cross-cutting notes for M3 hardening

When folding this file into plan.md (M3 chunk 11):

1. F21 → §4.2 cal subsection + §8 M3 DoD `cal_reload` line.
2. Append per-chunk implementation notes to plan.md `§8.M3-carryover`
   (new subsection between §8 M3 and §8 M4a, mirrors `§8.M2-carryover`).
3. Delete this file in the same commit; M3.sh detects the absence and
   stamps `complete (hardened)`.
