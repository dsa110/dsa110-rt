# Phase 0, study 4: imaging response of per-unique-baseline entries vs gridded cells

Scripts: `scripts/img_response.py` (all numbers below), `scripts/img_noise.py` (noise Monte Carlo),
geometry from `scripts/valuemaps.py`. Results: `results/imaging_per_baseline.json`,
`results/imaging_per_baseline_map.npz` (17 x 17 field map), `results/imaging_per_baseline_noise.json`.
Runtime about 3 min on one core of h23.

**Re-run on WS-P's exact classes.** The first run used the model's 1773 rint-binned classes. After WS-P
measured 1748 exact classes, everything was re-run on them. Every number below is unchanged at the
4th decimal; the only visible change is B4 vs A astrometry, max 0.0051 px instead of 0.0052.

## What is compared
The setup is the 2x op-point: csf 2 (16 chgroups x 192 summed channels), 3321 core cross-baselines,
N = 10,202,112 contributions per image, and the production gridding convention (`build_pattern`,
tee45): u = -dE/lambda, v = -dN/lambda, no dec projection, a 17.905-lambda cell (45" pixel), a
256 grid, fftshift, and a real image (equivalent to the Hermitian half-spectrum `irfft2`).

| scheme | placement |
|---|---|
| **A**: gridded cells (today's semantics) | each (baseline, summed channel) at its own nearest cell |
| **B**: per-baseline, n_sub 2 / 4 | each exact 3-D redundancy class (5 cm; **1748 classes from WS-P's `baseline_grid.redundancy_classes` @ 193b89c**) summed over its members and the sub-band's channels (natural weighting = SUM). The sum is placed at the nearest cell of the class member-mean uv at the sub-band CENTRE frequency. 55,936 / 111,872 entries |
| **Bx** | the same entries at their EXACT (unrounded) uv, evaluated by DFT. This isolates bandwidth smearing |
| ideal | every (baseline, channel) at its exact uv. The peak is exactly 1 at the true position |

**Source model.** A unit point source sits at apparent image offset (x, y) pixels, with
V_bc = exp(2 pi i (dE l + dN m) f_c / c). The planar-w fringe stop, the dec foreshortening of m and
the (n-1) term are per-(baseline, channel) phases. A and B sum the identical V_bc, so these terms
cancel from the comparison. The only term not modelled is the core's height scatter, which is
below 1 cm (an error of less than 1 cm in a baseline).

**Primary-beam weighting.** The weighting is beam^2: a Gaussian with FWHM 3.2 deg, evaluated at
the true sky offset (m_true = y pix / cos(dec - lat)), over a 17 x 17 grid spanning +-120 px
(+-1.5 deg).

## 1. Point-source S/N, B relative to A
The noise per pixel is identical in A and B (section 4), so the S/N ratio equals the ratio of the
continuous peaks.

| position (px) | A / ideal | B2 / A | B4 / A | Bx2 | Bx4 | bw_smear formula (cosz), n_sub 2 / 4 |
|---|---|---|---|---|---|---|
| centre (0,0) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1 / 1 |
| half-way EW (64,0) | 0.9174 | 0.9976 | 0.9993 | 0.9974 | 0.9994 | 0.9974 / 0.9994 |
| half-way NS (0,64) | 0.9321 | 0.9944 | 0.9985 | 0.9940 | 0.9985 | 0.9959 / 0.9990 |
| edge EW (120,0) | 0.7313 | 0.9919 | 0.9979 | 0.9911 | 0.9978 | 0.9911 / 0.9978 |
| edge NS (0,120) | 0.7791 | 0.9828 | 0.9955 | 0.9791 | 0.9947 | 0.9857 / 0.9964 |
| corner (120,120) | 0.5624 | 0.9751 | 0.9934 | 0.9710 | 0.9926 | 0.9775 / 0.9943 |
| anti-corner (120,-120) | 0.5624 | 0.9742 | 0.9932 | 0.9698 | 0.9923 | 0.9764 / 0.9940 |
| **field, beam^2-weighted** | 0.8797 | **0.9938** | **0.9984** | 0.9931 | 0.9983 | (bw_smear.json: 0.9937 / 0.9984) |
| field, minimum | 0.5624 | 0.9741 | 0.9932 | 0.9698 | 0.9923 | |
| field, uniform mean | | 0.9901 | 0.9974 | | | |

**B relative to A.** B/A tracks Bx, the bandwidth smearing alone, to within 0.4% everywhere. Rounding
the cell once per entry, instead of once per channel, adds no loss on average (beam-weighted B/A is
0.07% *better* than Bx).

**Comparison with the plan.** The plan's imaging column is confirmed:

| | plan | measured here |
|---|---|---|
| per-baseline n_sub 4: beam-weighted; corner | 0.998; 0.994 | 0.9984; 0.9932 |
| per-baseline n_sub 2: beam-weighted; corner | 0.994; 0.974 | 0.9938; 0.9741 |

**Comparison with bw_smear.py.** bw_smear.py projected v by cos z (0.825), which production does
not do. It therefore understates the N-S smear at a given image pixel by about 1/cos z. For
example, at edge NS with n_sub 2 the smear loss is 2.1% here against 1.4% in bw_smear.py. The
beam-weighted numbers agree anyway, because the narrower true-sky beam along m offsets this.

**Common to both schemes.** The A/ideal column is today's pillbox (nearest-cell) taper: 0.88
beam-weighted and 0.56 at the corner. This is the audit's imaging term, present in both A and B
and in today's production. It is not part of the format choice.

**Pixel phase.** The max-pixel value for a source half a pixel off the pixel centres is the same in
both schemes: 0.693 at the centre for A, B2 and B4.

## 2. Astrometry
The offset is the continuous-peak position (grid search followed by Newton refinement) minus the
truth, over all 289 field positions.

| | max | rms | positions > 0.1 px |
|---|---|---|---|
| A vs truth (today's pillbox rounding) | 0.068 px (3.1") | 0.037 px | 0 |
| B2 vs truth | 0.073 px | 0.040 px | 0 |
| B4 vs truth | 0.073 px | 0.038 px | 0 |
| **B2 vs A** | **0.0066 px (0.30")** | 0.0040 px | 0 |
| **B4 vs A** | **0.0052 px (0.23")** | 0.0024 px | 0 |
| Bx (exact placement) vs truth | 0.000 | 0.000 | 0 |

- Nothing exceeds 0.1 px.
- The per-baseline format moves positions by 1/10 of the offset today's gridding already has.
- A 3x3 parabolic fit on the pixel grid has its own bias of up to 0.08 px, and it is the same in
  A and B (for example, corner +0.5 px: A -0.084, B4 -0.086).

## 3. Dirty beam
The main-lobe FWHM and first minimum are measured along x (E-W) and y (N-S). The sidelobes are
taken outside the ellipse defined by A's first-minimum radii, and are relative to the source's own
peak. The last column is the maximum of |B - A| over the image, divided by A's peak.

| source at | scheme | FWHM x / y (px) | first min x / y | peak sidelobe | most negative | rms sidelobe | max\|B-A\| / A peak |
|---|---|---|---|---|---|---|---|
| centre | A | 2.499 / 1.595 (112" / 72") | 3.23 / 1.58 | 0.421 | -0.018 | 0.0311 | - |
| centre | B2 | 2.499 / 1.595 | 3.23 / 1.58 | 0.421 | -0.020 | 0.0311 | 0.0058 |
| centre | B4 | 2.499 / 1.595 | 3.23 / 1.58 | 0.421 | -0.019 | 0.0311 | 0.0029 |
| edge NS | A | 2.375 / 1.764 | 2.81 / 1.58 | 0.501 | -0.025 | 0.0372 | - |
| edge NS | B2 | 2.371 / 1.801 | 2.80 / 1.59 | 0.506 | -0.029 | 0.0378 | 0.0175 |
| edge NS | B4 | 2.374 / 1.774 | 2.80 / 1.58 | 0.501 | -0.026 | 0.0373 | 0.0046 |
| corner | A | 2.490 / 1.805 | 2.81 / 1.61 | 0.462 | -0.034 | 0.0424 | - |
| corner | B2 | 2.512 / 1.838 | 2.87 / 1.63 | 0.465 | -0.037 | 0.0433 | 0.0252 |
| corner | B4 | 2.496 / 1.813 | 2.82 / 1.62 | 0.462 | -0.035 | 0.0427 | 0.0069 |

Other positions, max|B - A| / A peak:

| position | B2 | B4 |
|---|---|---|
| half-way EW | 0.0071 | 0.0043 |
| half-way NS | 0.0096 | 0.0031 |
| edge EW | 0.0105 | 0.0053 |

- **At the phase centre** the beam is unchanged in shape and sidelobe level. The 0.3-0.6% pixel
  differences come only from where the contributions land.
- **Off-centre, n_sub 2** broadens the main lobe radially by at most 0.9% (x) and 2.1% (y) at the
  edge or corner, with sidelobes within 0.005 of A. n_sub 4 stays within 0.6%.
- **Where the difference image comes from.** It is dominated by the peak deficit already counted
  in section 1. No new sidelobe structure appears.

## 4. Natural weighting / noise
Analytically, every pixel of the real dirty image has variance sum_k n_k / N^2 = 1/N, wherever the
contributions are placed. B's entries carry SUMs (n_e = multiplicity x channels), and
sum n_e = N = 10,202,112 for both n_sub, the same as A. No contribution falls off the grid in
either scheme.

Monte Carlo over 10 realisations gives the spatial rms x sqrt(N), mean +- per-realisation spread:

| scheme | rms x sqrt(N) |
|---|---|
| A | 1.0074 +- 0.014 |
| B2 | 1.0041 +- 0.015 |
| B4 | 1.0056 +- 0.016 |

These are identical within 0.3% (standard error 0.5%). The noise per unit point-source signal is
the same, so the section 1 peak ratios are the S/N ratios. The noise remains correlated by each
scheme's own beam (section 3 shows the beams are equal).

## 5. Per-entry phase correction
**Not needed, and it would not help.**

- **What a correction cannot touch.** B's own phase is already exact at the sub-band centre: the
  channel sum about f_ctr is real (Dirichlet), and the member-mean uv is used. With exact placement
  (Bx) the peak sits on the truth at 0.000 px.
- **What it would have to fix.** The only offsets are from nearest-cell rounding. They are
  0.07 px in A, which is today's production, and B adds at most 0.007 px.
- **Why a phase correction does not work.** A rounding error dk moves the phase by
  2 pi dk . x / N, which depends on position. One phase per entry can cancel it at a single
  position only, and it is already zero at the phase centre.
- **What would work, and its cost.** Removing rounding effects needs a gridding kernel instead of
  a pillbox. For example, `kernel_support` 3 (already supported by `build_pattern`) means 9 scatter
  taps per entry: about 9x the K1 combine reads, a different taper, and the loss of 1x-compat
  imaging. That cost is not justified by an effect 10x below the 0.1 px flag.

## Verdict
- **n_sub 4 per-baseline.** Imaging is equivalent to gridded cells: S/N 0.998 beam-weighted and
  0.993 at the worst corner. Astrometry changes by at most 0.005 px (0.23"). The beam is
  unchanged. Accept.
- **n_sub 2 per-baseline.** S/N is 0.994 beam-weighted and 0.974 at the corner. This is
  bandwidth smearing (11.7/2 MHz sub-bands): radial, symmetric, with no astrometric bias (at most
  0.007 px, 0.30"). The main lobe broadens by up to 2% at the field edge. Acceptable only if the
  network forces it: it is the 0.6% (mean) / 2.6% (corner) imaging cost already in the plan's table
  for this row.
- **Both n_sub.** Natural weighting is preserved exactly (identical noise per unit signal). No
  per-entry correction is required.
