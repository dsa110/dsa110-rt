# Phase 0 study 5: time filters (matched-filter banks) at the 2x op-point

**Recommendation:** replace today's 7-boxcar ladder with **`b3x`**, an 8-filter bank. It has boxcars
of 1 and 2 samples, the triangle [1,2,1], and quadratic B-splines (three cascaded running sums) at the
nominal widths 4 to 64. In the model it never loses to the boxcar ladder for Gaussian or scattered
pulses at any DM or width. For a 1 ms Gaussian it gains **x1.043**. Over DM 50-3000 and widths
0.1-16 ms it gains **x1.048 (Gaussian) and x1.026 (scattered)** with white noise, and **x1.038 / x1.017**
after the slow image residual measured on live cubes. Top-hat pulses that exactly match a boxcar width
lose up to 4.5% at 8-16 ms, but averaged over widths they still gain x1.019. The bank needs **148 rows**
of detector history (today 63; the plan estimated ~170) and **36 adds per pixel-sample** (today 12).
Its false-alarm rate is lower than the boxcar's.

Scripts are in `docs/x2/phase0/scripts/tf_*.py`; results are in `docs/x2/phase0/results/time_filters*.{json,npz}`.

## Method
- **Model.** `tf_snr_model.py` is a copy of `docs/timeres_study/snr2x/snr_model.py`; only the worktree
  path changed. It adds a generic filter engine:
  - filters are normalised by ||h||_2, so white-noise sigma is 1;
  - every lag where the filter fully overlaps the series is evaluated;
  - the best filter is taken per arrival phase, then the mean over 16 phases;
  - the best of the 8 nearest DM trials is kept, as `snr_binned` / `evaluate()['total']` do.

  The boxcar bank reproduces the original model to 2e-16 (`tf_validate.py`).
- **Op-point.** `proposed(q1=262.144, n_sub=2)`. Tier A covers DM 50-1600 with 0.524 ms samples; tier B
  covers 1600-3000 with 1.049 ms samples. Filters are in the samples of each tier.
- **Grids.**
  - Main grid: 738 DMs, the exact `run_snr2x` prop_q026 set (step 4, jitter seed 12), with
    `run_snr2x.WIDTHS_MS`.
  - Dense grid: 31 widths at 4 per octave, 0.088-16 ms, on 60 DMs. This catches the dips between bank
    widths.
  - Unsmeared scan (`tf_scallop.py`): the filter-limited efficiency on its own, versus W/T.
- **Pulse shapes.** Gaussian (FWHM W), top-hat (width W), and a one-sided exponential scattering tail
  (tau = W).
- **Means.** "Mean" is the mean over DMs 50-3000 and widths 0.1-16 ms (8 widths) of S/N / ideal. The
  ratio is to the boxcar ladder in the same run, as a ratio of means.

## The banks
Widths are nominal (FWHM, in tier samples). "Lengths" are the cascaded running-sum lengths.
||h||_2 is for the un-normalised integer cascade, which is the sigma_k seed. Support minus 1 gives the
detector history rows.

| bank | w=1 | w=2 | w=4 | w=8 | w=16 | w=32 | w=64 | history rows | adds/output (bank) |
|---|---|---|---|---|---|---|---|---|---|
| boxcar (today) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 63 | 12 |
| ||h||_2 | 1 | 1.414 | 2 | 2.828 | 4 | 5.657 | 8 | | |
| B2 triangle (a,a) | delta | (2,2) | (4,4) | (8,8) | (16,16) | (32,32) | (64,64) | 126 | 24 |
| support / ||h||_2 | 1 / 1 | 3 / 2.449 | 7 / 6.633 | 15 / 18.55 | 31 / 52.31 | 63 / 147.8 | 127 / 418.1 | | |
| B3 quadratic | delta | (1,2,2)=[1,2,1] | (3,3,3) | (6,6,7) | (12,13,13) | (25,25,26) | (50,50,51) | 148 | 34 |
| achieved FWHM | 1 | 2 | 3.67 | 8.00 | 16.09 | 32.14 | 63.82 | | |
| support / ||h||_2 | 1 / 1 | 3 / 2.449 | 7 / 11.87 | 17 / 74.55 | 36 / 423.0 | 74 / 2395 | 149 / 13329 | | |
| Gaussian, FWHM w, ±3 sigma | 5 taps | 7 | 13 | 23 | 43 | 83 | 165 | 164 | 678 (FIR) |
| **b3x** (8 filters) | box 1 | box 2 **and** [1,2,1] | B3 | B3 | B3 | B3 | B3 | **148** | **36** |

- **How B2 is defined.** A trapezoid (a < w, w) also has FWHM w, so FWHM alone does not fix the shape.
  B2 is therefore the equal-length triangle.
- **How B3 is defined.** B3 uses near-equal integer lengths (a or a+1). These are chosen to match the
  FWHM, using FWHM = 1.268 a for three equal stages.
- **B-splines at w ≤ 2.**
  - At w = 1, both B2 and B3 collapse to a delta, the same as the boxcar.
  - At w = 2, the closest FWHM match is the triangle [1,2,1]. This covers 3 samples where the boxcar
    [1,1] covers 2.
  - [1,1] is better for pulses split across two samples (≤ 0.5 ms). [1,2,1] is better for about 1 ms
    at tier A.
  - `b3x` therefore carries both. The pure B3 bank loses 2.3% to the boxcar at 0.1 ms (table below);
    `b3h` (boxcar at w ≤ 2, B3 above) gives up 1.3% of B3's S/N at 1 ms.

## Recovered S/N (fraction of an ideal matched filter)
| bank | Gaussian mean (x box) | Gaussian @1 ms | top-hat mean | top-hat @1 ms | top-hat dense | exp mean | exp @1 ms | points < box |
|---|---|---|---|---|---|---|---|---|
| boxcar | 0.7376 | 0.7935 | 0.7002 | 0.7165 | - | 0.7112 | 0.7550 | - |
| B2 | 0.7652 (x1.037) | x1.029 | x0.973 | x1.008 | x1.005 | x1.013 | x1.026 | 33% (Gauss) |
| B3 | 0.7661 (x1.039) | x1.030 | x0.986 | x1.008 | x1.008 | x1.019 | x1.036 | 33% |
| Gaussian | 0.7683 (x1.042) | x1.034 | x0.984 | x1.010 | x1.008 | x1.022 | x1.036 | 33% |
| b3h | 0.7664 (x1.039) | x1.017 | x0.988 | x1.002 | x1.010 | x1.018 | x1.026 | 0% |
| **b3x** | **0.7727 (x1.048)** | **0.8275 (x1.043)** | **x0.9965** | **x1.030** | **x1.019** | **x1.026** | **x1.042** | **0% (Gauss, exp); 32% (top-hat)** |

**Per tier, b3x.**

| tier | Gaussian | top-hat | exp |
|---|---|---|---|
| A | x1.050 | x0.993 | x1.025 |
| B | x1.045 | x1.001 | x1.027 |

**Dense grid (mean over widths, worst width), b3x.**

| shape | mean | worst width |
|---|---|---|
| Gaussian | x1.044 | x1.007 at 0.11 ms |
| exp | x1.025 | x1.000 |
| top-hat | x1.019 | x0.958 at 16 ms |

**Per width, Gaussian (S/N / ideal).**

| width (ms) | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| boxcar | 0.230 | 0.324 | 0.452 | 0.662 | 0.794 | 0.877 | 0.921 | 0.934 | 0.937 |
| B3 | 0.224 | 0.316 | 0.442 | 0.659 | 0.817 | 0.925 | 0.978 | 0.995 | 0.998 |
| b3x | 0.231 | 0.326 | 0.455 | 0.677 | 0.828 | 0.926 | 0.978 | 0.995 | 0.998 |

### Reading the table
- **Below 0.5 ms the filter does not matter.** The loss there is sampling and dedispersion smearing,
  and every bank is a delta or [1,1] at those widths.
- **For resolved Gaussian pulses** the boxcar's ceiling is 0.937. B3 reaches 0.998.
- **Top-hats in the main grid favour the boxcar artificially.** `WIDTHS_MS` 2-16 sit within 5% of the
  tier-A boxcar widths (0.524 x 2^k ms), so the boxcar is exactly matched there.
- **Between widths the boxcar scallops badly.** On the unsmeared scan for top-hats at W/T 4-64:

  | bank | mean | min | ripple over an octave |
  |---|---|---|---|
  | boxcar | 0.916 | 0.841 | 14.3% |
  | B3 | 0.928 | 0.888 | 3.6% |

  So the dense-grid average favours b3x (x1.019) even for top-hats.
- **Unsmeared Gaussians at W/T 4-64:** boxcar 0.929 (min 0.902), B3 0.986 (min 0.953), Gaussian bank
  0.988 (min 0.966).
- **B2 is dominated by B3.** It is equal for Gaussians and worse for top-hat and exp, at two-thirds of
  the stages.
- **The Gaussian bank adds only +0.1-0.3% over B3**, for about 20x the arithmetic (FIR).

## Red noise (live cubes are not white in time)
- **What the live cubes show.** The Layer-2 sigma_k on 32 live cubes grows faster than sqrt(w): 1.19x at
  b64. Fitting a white term plus a slow component gives c = 0.70% of the per-sample variance with
  tau = 432 samples (~450 ms).
- **Why it hurts B3 more.** A B3 with the same FWHM has longer support. Its equivalent noise length,
  (Σh)²/Σh², is 1.43x the boxcar's at w = 64, so it integrates more of the slow residual.
- **Extra noise of B3 over the boxcar**, mapped to 2x samples:

  | residual share | w=8 | w=16 | w=32 | w=64 |
  |---|---|---|---|---|
  | kept (conservative) | 1.011 | 1.022 | 1.039 | 1.062 |
  | halved (white variance per sample doubles at 2x) | 1.006 | 1.011 | 1.021 | 1.037 |

- **Realised b3x gain with the conservative case:**

  | shape | white noise | with red noise | per-width detail |
  |---|---|---|---|
  | Gaussian | x1.048 | **x1.038** | 1 ms x1.043 unchanged; 16 ms x1.034 |
  | exp | x1.026 | x1.017 | |
  | top-hat | x0.9965 | x0.987 | |

- **What this means for operations.** Layer-2 measures the real sigma_k, so reported S/N stays correct.
  Only the gain at ≥ 8 ms shrinks. Re-measure sigma_k(b3x)/||h||_2 on recorded 2x cubes (CP2 soak);
  `tf_rednoise.py` does the arithmetic.

## Trials factor
White noise, filters normalised:

| bank | adjacent-width output correlation | upcrossing rate at 9.55 sigma (union over bank) | equivalent threshold shift | Monte Carlo bank-max upcrossings at 4.0 / 4.5 sigma |
|---|---|---|---|---|
| boxcar | 0.707 (all pairs) | 1 | 0 | 1 / 1 |
| B3 | 0.82-0.89 | x0.638 | -0.047 sigma | x0.69 / x0.66 |
| b3x | 0.71-0.89 | x0.827 | -0.020 sigma | x0.81 / x0.79 |
| Gaussian | 0.86-0.89 | x0.818 | -0.021 sigma | x0.67 / x0.64 |

Smoother wide filters decorrelate more slowly, so the bank has fewer effectively independent trials. The
DM and pixel trials do not change. **There is no threshold penalty.** At the C1 floor of 9.55 the
quiet-sky false-alarm rate should fall by about 20%. Keep 9.55 and confirm the quiet-sky C1 rate on sky,
as the plan already requires.

## Implementation notes for WS-S (K3b) and WS-D
- **History and memory.**
  - History is 148 rows (B3-64 support 149), plus the hop. The ring should be sized to 149, not ~170.
  - A streaming cascade needs per-stage rings: 50 + 50 + 51 rows for w = 64.
  - Alternatively, a shared triple prefix sum with 8-tap differences avoids those rings. It grows as t³,
    so it needs fp64 or per-chunk re-basing.
- **Accumulator.**
  - The integer cascade sums reach Σh = 127,500 for B3-64. With a 250 sigma clip, outputs reach 3.2e7.
  - fp16 accumulation (today's `--detector-boxcar-accum-dtype fp16`) cannot hold this. Use fp32, ideally
    as moving averages (divide by each stage length), so values stay O(clip).
  - Re-seed the running sums every cube to bound fp32 drift.
- **sigma_k seeds** are ||h||_2 from the table, or ||h||_2/Σh if the filter outputs moving averages.
  - The plan's rule "seed = ||h||_2" holds only for a sum-normalised implementation.
  - Layer-2 then measures the red-noise inflation above (up to 1.27x at w = 64).
- **Width identity.**
  - b3x has two filters at nominal width 2: [1,1] and [1,2,1]. The decoder and merger must carry a filter
    id, not just a width.
  - Report widths as FWHM in ms (w x T).
  - Recalibrate injection K per filter (memory: K is width-dependent).
  - The width cap (32 samples) is unaffected.
- **Cost.**
  - 36 adds per (pixel, sample) versus 12 today: 3 stages for each of the 5 B3 filters, 2 for [1,2,1],
    2 for box-2.
  - This is consistent with the plan's +0.3-0.5 us/image allowance for B-splines.
  - Adopt only if CP1 measures K3b within the duty aim; the plan's gain/cost rule.

## Contradictions with the plan
- **History rows.** The plan says "~170 B3 history rows". FWHM-matched B3 needs 148; a ±3 sigma Gaussian
  needs 164.
- **Bank definition.** The plan says "B-splines ... with the same 7 widths". A pure 7-filter B3 bank
  loses up to 2.3% to the boxcar at ≤ 0.2 ms, because it degenerates to [1,2,1] at w = 2. The
  recommended bank keeps boxcars at w = 1, 2 and adds [1,2,1], which makes 8 filters.
- **Adoption rule.** "Adopt whenever the modelled gain is positive" holds for Gaussian and scattered
  pulses at every point. For top-hat pulses that exactly match a boxcar width at 8-16 ms the gain is
  negative (down to x0.955), although it is positive averaged over widths.
