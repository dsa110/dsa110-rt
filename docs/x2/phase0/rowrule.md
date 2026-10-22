# Phase 0, study 1: the row rule (SPEC section 4)

Scripts: `scripts/valuemaps.py` (contribution counts), `scripts/rr_lib.py` (quantisers, exact
Gaussian moments), `rr_static.py`, `rr_bright.py`, `rr_track.py`, `rr_hot.py`, `rr_load.py`,
`rr_fold.py`. Results: `results/valuemaps.json`, `results/rowrule_{static,bright,track,hot,load,fold}.json`.
All modelling is on CPU (h23, nice 10). Recorded 1x RX data (including a 07:16/19:46 storm window) do
not exist yet. The `tools/x2/rx_recorder.py` replay check stays open for WS-T/I1.

## 0. Real contribution counts

These use the production gridding convention: `sparsity_pattern.build_pattern`, tee45 cell 17.905
lambda, F20 negation, a single-sided grid, and **no declination projection** (the timeres model used
`v cos z`). The core array has 82 antennas and 3321 baselines. There are 1773 exact 3-D classes
(5 cm), of which 1650 are singletons; the largest multiplicity is 41.

| format | streams | values/stream (mean, max) | N per image | n max | heavy at N/1e4: values/stream, weight | heavy at N/3e4: values/stream, weight |
|---|---|---|---|---|---|---|
| today (csf 8, n_sub 4, gridded) | 64 | 2134, 2243 | 2.55 M | 492 | (8-bit) | (8-bit) |
| gridded n_sub 2 | 32 | 2544, 2759 | 10.20 M | 3936 | 74, 42.4% | 103, 48.5% |
| gridded n_sub 4 | 64 | 2159, 2267 | 10.20 M | 1968 | 24.5, 20.7% | 90, 47.4% |
| per-baseline n_sub 2 | 32 | 1773 | 10.20 M | 3936 | 71, 40.1% | 114, 49.6% |
| per-baseline n_sub 4 | 64 | 1773 | 10.20 M | 1968 | 23, 19.5% | 99, 47.4% |

The heavy fractions at N/1e4 reproduce the plan: 3.1%/43.5%, 1.2%/21.7%, 71 entries/40%, and
23/19.5%. Gridded value counts are 2.2-2.5% higher than the plan's 2483/2112. The reason is that
production gridding has no `cos z` projection and no Hermitian fold. That also makes the gridded
load independent of declination, which removes the plan's "<= 2% with pointing declination" caveat.
A conjugate fold of k and -k would save 1.5-1.7% of gridded values (`rr_fold.py`). That saving is
optional and is not assumed below.

## 1. Quantiser and steps

The efficiency eta is exact, computed from the normal CDF over the quantiser bins.

| quantiser | step / sigma_hat | eta | top level / sigma_hat | eta at scale x0.7 | eta at scale x1.4 |
|---|---|---|---|---|---|
| mid-rise 4-bit (SPEC v0.1) | 0.3352 | 0.99421 | 2.51 | 0.98784 | 0.99092 |
| **mid-tread 4-bit** (-7..7) | **0.3534** | 0.99353 | 2.47 | 0.98685 | 0.98992 |
| mid-rise 5-bit (SPEC v0.1) | 0.1881 | 0.99825 | 2.92 | 0.99466 | 0.99712 |
| **mid-tread 5-bit** (-15..15) | **0.1932** | 0.99815 | 2.90 | 0.99445 | 0.99696 |
| mid-tread 8-bit (heavy) | s_h (below) | 0.99996 at its optimum, >= 0.99978 at s_h | 127 s_h | | |

The SPEC draft steps (0.336 and 0.191, mid-rise) are confirmed as the mid-rise optima. **This study
recommends mid-tread** (section 5): it costs 0.07% at 4 bits and 0.01% at 5 bits, and it leaves the
code -2^(b-1) free to use as a "no data" sentinel.

For the heavy step, s_h = 2 x 250 x sqrt(n_max/N) / 127. That gives **0.0773 for n_sub 2 and 0.0547
for n_sub 4** (clips at 9.8 and 6.9 sigma_hat, 2x the heaviest value's S/N at a 250-sigma image).

## 2. Effective quantisation efficiency per format

These are point-source image S/N relative to unquantised natural weighting. Main values are
mid-tread at s_b and heavy values are cint8 at s_h. The analytic result was checked by Monte Carlo
(gridded n_sub 2, 4-bit): 0.99602 MC against 0.99617 analytic.

| format | 4-bit, heavy N/1e4 | 4-bit, heavy N/3e4 | 5-bit, heavy N/1e4 | 5-bit, heavy N/3e4 |
|---|---|---|---|---|
| gridded n_sub 2 | 0.99617 | 0.99656 | 0.99883 | 0.99893 |
| gridded n_sub 4 | 0.99485 | 0.99654 | 0.99851 | 0.99897 |
| per-baseline n_sub 2 | 0.99603 | 0.99662 | 0.99879 | 0.99895 |
| per-baseline n_sub 4 | 0.99477 | 0.99655 | 0.99849 | 0.99897 |

The tracking cost at tau 3 s (section 3) is at most 0.04% more.

**Today's 1x wire costs 1.2%, not 0.05%.** That is per-row max-abs cint8 with real n, Monte Carlo
over all 64 streams: eta = 0.9880. The 8-bit quantisation itself costs 0.02%. The other 1.2% comes
from the row-to-row jitter of the amax/127 weights (0.9882 with no quantisation at all), because
codes are summed without the scale. The 2026-09 audit's 0.05% counted quantisation only. **Every 2x
format is therefore better on the wire than today**, by +0.6% to +1.0%.

## 3. Tracking sigma_hat: tau, drifts, flags, start-up

The setup is an EMA of a per-block robust estimate (134.2 ms blocks, estimator jitter 0.1%), with
64 independent streams. The total is the time-mean of eta(sigma_hat/sigma) times the cross-stream
weighting efficiency (relative to today's 1/sigma semantics). 4-bit mid-tread, static eta = 0.99353.

| scenario | tau 1 s | **tau 3 s** | tau 10 s | tau 30 s | tau 100 s |
|---|---|---|---|---|---|
| drift +-15% / 10 min | 0.99353 | 0.99353 | 0.99347 | 0.99299 | 0.99041 |
| drift +-30% / 5 min | 0.99352 | 0.99344 | 0.99247 | 0.98650 | 0.97412 |
| random walk 10% rms / 2 min | 0.99351 | 0.99347 | 0.99332 | 0.99292 | 0.99171 |
| channel flags 0 <-> 30% (dwell 60 s) | 0.99341 | 0.99316 | 0.99249 | 0.99144 | 0.99034 |
| gain steps x1.3 (~5 min) | 0.99349 | 0.99340 | 0.99306 | 0.99209 | 0.98893 |
| start-up, EMA from block 0 (8 warm-up blocks at 5x) | 0.99300 | 0.99183 | 0.98773 (min 0.892) | 0.97602 | 0.94519 |
| start-up, EMA initialised at the first post-warm-up block | 0.99353 | 0.99353 | 0.99353 | | |

- **Recommendation: tau = 3 s (22 blocks).** Every scenario stays within 0.04% of static. A single
  anomalous block moves sigma_hat by at most 4.4% of its excess, and a 1-sample burst changes at
  most 1/256 of a block's data, so the weights are not burst-dependent. The SPEC's 10 s costs up to
  0.1%. That is harmless, but there is no reason to choose it.
- **The "exclude |y| > 6 sigma_hat_prev" rule changes nothing** in any scenario, because the median
  already has a 50% breakdown point. It is also a deadlock risk: if sigma_hat is ever initialised
  near zero (an all-zero block), every value is excluded. Replace it with three rules. Skip the
  update when fewer than 50% of a block's values are non-zero (flagged or zero-filled). Initialise
  sigma_hat from the first block after the static-sky warm-up (`StaticSkyMean.warmup_cubes` = 8).
  Floor sigma_hat at a small positive value.
- Cheap estimator, an implementation note for WS-C/T: the packer already produces codes. The
  per-(stream, DM) histogram of |code| from the previous block gives the robust scale for free.
  For example, take the fraction p of |code| <= 1, then sigma = 1.5 s / Phi^-1((1 + p)/2). This is
  bounded by construction, so no median sort over 1e6 elements per (stream, DM) is needed.

## 4. Hot values

The test puts one value of a row at H sigma: a narrow RFI line on one short baseline, or a stuck
cell. The table gives the point-source efficiency of the **other** values in that row.

| case | H = 0 | H = 10 | H = 50 | H = 1000 |
|---|---|---|---|---|
| today, 8-bit max-abs, hot = heaviest cell (n 492) | 0.988 | 0.989 | 0.857 | **0.000** (row crushed) |
| today, 8-bit max-abs, hot = median cell (n 12) | 0.988 | 0.989 | 0.995 | 0.598 |
| 4-bit max-abs, today's n | 0.903 | 0.578 / 0.903 | **0.000** / 0.605 | 0.000 |
| 2x rule, 4-bit, gridded n_sub 2 (any hot value, any H) | 0.9958 | 0.9958 | 0.9958 | 0.9958 |
| 2x rule, 4-bit, per-baseline n_sub 4 | 0.9938 | 0.9938 | 0.9938 | 0.9938 |

Under the 2x rule a hot value is clipped at its own top level, and nothing else in the row changes.
Its image artefact is bounded at clip x sqrt(n)/sqrt(N) per sample:
- a main value (n 90): **0.007 sigma**;
- the heaviest value: 0.19 sigma (n_sub 2) or 0.10 sigma (n_sub 4).

The max-abs fragility the plan quoted (0.80 at 4 bits) is worse with real n: 0.90 with no hot cell
at all, and 0 when the hot cell is heavy.

## 5. Mid-rise has no zero level, so exact zeros become a constant fake source

A value whose sources are all flagged is exactly 0. A mid-rise code maps it to (+0.5 + 0.5i) step,
every sample, coherently, and nothing downstream removes a constant: the static sky is subtracted
on the corr before quantisation. The table gives the resulting offset at the phase centre with
mid-rise 4-bit. A b64 boxcar turns c sigma per sample into 6.9 c sigma (sigma_k = 1.163 x 8).

| format | one whole stream zero (flagged sub-band, zero-coded fill) | one antenna flagged: mean (max) |
|---|---|---|
| gridded n_sub 2 | 1.19 sigma -> b64 8.2 sigma | 0.76 (0.93) sigma -> 5.3 (6.4) |
| gridded n_sub 4 | 0.80 sigma -> 5.5 | 1.02 (1.22) -> 7.0 (8.4) |
| per-baseline n_sub 2 | 1.07 sigma -> 7.4 | 0.66 (0.81) -> 4.6 (5.5) |
| per-baseline n_sub 4 | 0.76 sigma -> 5.2 | 0.94 (1.14) -> 6.4 (7.8) |

Two flagged antennas reach the C1 floor (9.55) in every cube. 13 antenna-pols are flagged in every
cube today. **Use mid-tread** (today's cint8 is mid-tread: `np.round`, `tx.py:863`), which decodes
an exact zero to 0. Two further rules apply:
- the owner's fan-in zero-fill must be 0.0 in DECODED space, never a code;
- code -2^(b-1) is reserved as a "no data" sentinel that decodes to 0.

## 6. Noise surges and the per-row guard

Today every (stream, DM, sample) row is renormalised by its own amax. A broadband noise surge (noise
x F in a row, from RFI, power transients or saturation) therefore leaves the image noise of that
sample unchanged.

A slow sigma_hat cannot do that. The surging values fill the levels up to the clip, so the image
noise of that sample inflates. The table gives the expected maximum pixel over one sample of the
cube (256 x 256 x 17 fine DMs); unit noise gives 4.90.

| surge | 4-bit, no guard: inflation, max pixel | with guard G = 1.5 |
|---|---|---|
| F 1.5, all streams | 1.38, 6.8 sigma | 1.38, 6.8 |
| F 2, all streams | 1.63, 8.0 sigma | 1.00, 4.9 |
| F 3, all streams | 1.91, **9.4 sigma** | 1.00, 4.9 |
| F 5, all streams | 2.15, **10.5 sigma** | 1.00, 4.9 |
| F >= 10, all streams | 2.3-2.5, **11-12 sigma** | 1.00, 4.9 |
| F 5, 25% of streams | 1.38, 6.8 sigma | 1.00, 4.9 |

Without a guard, every broadband surge F >= 3 puts junk at the C1 floor in 1-sample filters. This
is exactly the class of change that made the 2026-07-22 storm. **Recommended guard:** when a row's
own robust scale exceeds G = 1.5 sigma_hat, code that row (main and heavy values) with its own
scale. The owner still decodes in sigma_hat units with no row factor, so such a row is weighted
1/s_row. That is today's semantics, applied only to anomalous rows.

The bright-burst Monte Carlo (section 7) shows the guard first acts on a real point source between
500 and 700 sigma per sample, which is 2-3x above the detector rail. A cheap statistic for the
guard is the fraction of saturated codes in the row: 2.2% for normal noise, 12.6% at F 1.5, 25% at
F 2. The guard re-encodes only the rows that trip. WS-T must calibrate the threshold so that no
point source below 500 sigma per sample trips it.

## 7. Bright response of the wire

L(rho) is the per-sample image S/N out divided by S/N in, normalised to the small-signal efficiency;
1 means a linear wire. Rows marked "centre" put all the signal in Re (a calibrated source at the
phase centre, the worst case). Rows marked "random" have random phases (off-centre). The 2x rows
are exact Gaussian moments; today's rows are Monte Carlo of per-row max-abs cint8 over 32 streams.

| case | 35 | 50 | 100 | 150 | 250 | 500 | 1000 |
|---|---|---|---|---|---|---|---|
| **today (1x), centre** | **0.967** | **0.934** | **0.797** | **0.670** | **0.494** | 0.287 | 0.150 |
| today (1x), random | 0.962 | 0.929 | 0.799 | 0.682 | 0.513 | 0.301 | 0.159 |
| gridded n_sub 2, 4-bit, heavy N/1e4, centre | 1.000 | 1.000 | 0.999 | 0.997 | 0.989 | 0.953 | 0.732 |
| gridded n_sub 2, 4-bit, heavy N/3e4, centre | 1.000 | 1.000 | 0.999 | 0.999 | **0.996** | 0.977 | 0.773 |
| gridded n_sub 4, 4-bit, heavy N/1e4, centre | 1.000 | 0.999 | 0.997 | 0.991 | 0.972 | 0.881 | 0.674 |
| gridded n_sub 4, 4-bit, heavy N/3e4, centre | 1.000 | 1.000 | 1.000 | 0.999 | **0.997** | 0.985 | 0.815 |
| gridded n_sub 4, 5-bit, heavy N/3e4, centre | 1.000 | 1.000 | 1.000 | 1.000 | **0.999** | 0.991 | 0.841 |
| per-baseline n_sub 2, 4-bit, heavy N/3e4, centre | 1.000 | 1.000 | 0.999 | 0.998 | **0.995** | 0.974 | 0.766 |
| per-baseline n_sub 4, 4-bit, heavy N/1e4, centre | 1.000 | 0.999 | 0.997 | 0.992 | 0.973 | 0.882 | 0.667 |
| per-baseline n_sub 4, 4-bit, heavy N/3e4, centre | 1.000 | 1.000 | 1.000 | 0.999 | **0.997** | 0.983 | 0.818 |
| per-baseline n_sub 4, 5-bit, heavy N/3e4, centre | 1.000 | 1.000 | 1.000 | 1.000 | **0.999** | 0.991 | 0.846 |

- rho is the S/N carried by ONE row (one stream sample at one coarse DM, summed over the 16 x
  n_sub streams). A burst whose signal is spread over several samples of a stream gets less
  compression at the same detected S/N. The spread can come from intrinsic width, or from the
  intra-sub-band sweep between the coarse and the fine DM (1-5 samples at 1x). The curves are
  therefore the worst case per detected S/N, and are exact for unresolved bursts at a coarse DM.
- **Heavy threshold N/3e4, not N/1e4.** At N/1e4 the heaviest main value sits exactly at the 4-bit
  top level at 250 sigma, and the many near-threshold values compress the n_sub 4 formats by 2.7-2.8%
  at the rail. At N/3e4 every format is within 0.5% (4-bit) or 0.2% (5-bit) up to 250 sigma per
  sample. The load cost is +0.5 to +1.4 Gb/s in total (section 8).
- **Today's curve is not the plan's "linear to ~35 sigma, rail ~250 x sqrt(k)".** That curve
  described the period before 2026-09-23, when the fp16 FFT overflowed at ~35 sigma and the overflow
  value was clipped to 250.
  - The wire model is validated: it predicts today's raw image sigma at 0.0674 against 0.064
    measured live (`amp_regime.json`).
  - With the 1/256 prescale (live since 2026-09-23), today's per-row max-abs wire compresses bright
    single-sample signals itself: -3% at 35 sigma, -7% at 50, -20% at 100, -51% at 250.
  - Single-sample output saturates at **~150 sigma per sample** (rho x L is about 150 from 1000 to
    20000 sigma).
  - So today's effective rail is ~150 x sqrt(k). Every sample of a k-sample burst saturates on its
    own, so a point-source burst never reaches the 250 clip.
  - The 2x rule is linear to the 250 clip, so 2x reports bright events **higher** than today, by up
    to 2x at 250 sigma per sample. For the bright C2 classes (snr_max_min 35, the >= 50 sigma
    peak-member rule) the difference is 3-7% at the threshold.
  - This is a change of regime in exactly the direction 3b43118 warns about. It is listed in
    `SUMMARY.md` as a sign-off item, together with the C1/C2 replay check it needs.

## 8. Format table (refresh of the plan's)

Columns:
- dedispersion: snr2x `prop_ns4`/`prop_q026` mean over DM 50-3000. 0.9978 at 1 ms, 0.9986 over 0.1-16 ms.
- imaging: study 4, beam^2-weighted, with the corner value in brackets.
- quantisation: this study, mid-tread, heavy N/3e4, x 0.9996 tracking at tau 3 s.
- load: real value counts, frame-v2 72 B header + 28 B IP/UDP + 38 B Ethernet per datagram,
  k = floor(8900 / bytes per sample).
- per node: interleaved ownership d mod 8, worst node; 80% rule.

| format | dedisp | imaging | quant | **total vs model** | load Gb/s (N/3e4; N/1e4) | worst node Gb/s | **clean ingress needed / node** | datagrams/s worst node |
|---|---|---|---|---|---|---|---|---|
| gridded n_sub 2, cint4 | 1.000 | 1.000 | 0.9962 | **0.996** | 51.9; 51.4 | 13.15 | **16.4** | 203k |
| gridded n_sub 2, cint5 | 1.000 | 1.000 | 0.9985 | **0.9985** | 64.2; 63.7 | 16.24 | **20.3** | 305k |
| gridded n_sub 4, cint5 | 0.998 | 1.000 | 0.9986 | **0.997** | 54.9; 54.0 | 13.73 | **17.2** | 203k |
| gridded n_sub 4, cint4 | 0.998 | 1.000 | 0.9962 | **0.994** | 44.8; 43.4 | 11.19 | **14.0** | 191k |
| per-baseline n_sub 4, cint5 | 0.998 | 0.998 (0.993) | 0.9986 | **0.995** | 45.6; 44.5 | 11.41 | **14.3** | 203k |
| per-baseline n_sub 4, cint4 | 0.998 | 0.998 (0.993) | 0.9962 | **0.993** | 37.2; 35.8 | 9.31 | **11.6** | 153k |
| per-baseline n_sub 2, cint5 | 1.000 | 0.994 (0.974) | 0.9985 | **0.992** | 45.3; 44.7 | 11.46 | **14.3** | 203k |
| per-baseline n_sub 2, cint4 | 1.000 | 0.994 (0.974) | 0.9962 | **0.990** | 37.1; 36.2 | 9.38 | **11.7** | 153k |
| (today's 1x wire, for reference) | | | 0.988 | | | | | |

Changes against the plan's table:
- gridded loads are +2-3%, because of production gridding (no fold, no projection);
- the heavy threshold N/3e4 adds 0.5-1.4 Gb/s;
- n_sub 4 dedispersion is 0.998 rather than 0.997.

The ranking and the selection rule are unchanged. Every row beats today's wire (0.988), so the
network-forced cost relative to **today** is negative: +0.2% to +1.0%.

## 9. Proposed SPEC section 4 text (for WS-T to freeze)

1. Normalise: y = x / sqrt(n_static), unchanged.
2. Scale sigma_hat per (stream, coarse DM):
   - an EMA with tau = 3 s of a robust per-block estimate (from the code histogram, or
     1.4826 x median|re, im|);
   - no |y| > 6 sigma_hat exclusion;
   - skip a block with fewer than 50% non-zero values;
   - initialise from the first post-warm-up block;
   - positive floor.
3. Main values: **mid-tread**, step s_4 = 0.3534 or s_5 = 0.1932 times sigma_hat,
   code = clip(rint(y/step), -(2^(b-1)-1), 2^(b-1)-1). Code -2^(b-1) means "no data" and decodes
   to 0.
4. Heavy values: **n >= N_image / 3e4**, cint8 mid-tread, s_h = 2 x 250 x sqrt(n_max/N)/127
   (0.0773 for n_sub 2, 0.0547 for n_sub 4).
5. Row guard: when a row's own robust scale exceeds 1.5 sigma_hat (or the equivalent saturated-code
   fraction, calibrated so that no point source below 500 sigma per sample trips it), code that row
   with its own scale. The header carries the applied per-sample scale for monitoring.
6. Owner: v = code x s_b (or s_h) x sqrt(n). No row factor. Zero-fill is 0.0 in decoded space.
