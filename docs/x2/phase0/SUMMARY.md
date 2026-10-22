# Phase 0 summary: decisions for CP1 (S1/S2) and contradictions with the plan

Studies (branch `x2/ws-phase0`, merged into `x2/main`): `rowrule.md` (1), `amplitude_regime.md` (2),
`cfp16.md` (3), `imaging_per_baseline.md` (4), `time_filters.md` (5). Every value count comes from WS-P's
`build_value_map` (193b89c) on its fixture. Gridded maps agree with the model value for value; the
per-baseline numbers use the 1748 exact redundancy classes. `scripts/run_all.sh` re-runs everything.

## Decisions (recommendations; the user signs off at CP1)
| item | recommendation | key numbers |
|---|---|---|
| S1 format | Selection rule unchanged. "Total vs model" with the S2 choices below, and the clean ingress per node needed at 80% load. | gridded ns2 cint4 0.996 / 16.4 Gb/s; gridded ns4 cint5 0.997 / 17.2; gridded ns4 cint4 0.994 / 14.0; baseline ns4 cint5 0.995 / 14.0; baseline ns4 cint4 0.993 / 11.4; baseline ns2 cint4 0.990 / 11.5 |
| S2 quantiser | **Mid-tread**: s4 = 0.3534, s5 = 0.1932. Code -2^(b-1) means "no data". | Mid-rise turns flagged (exactly zero) values into a constant fake source: 0.7-1.2 sigma per flagged antenna or zeroed stream, 5-8 sigma in b64; two flagged antennas reach the C1 floor every cube. Mid-tread costs 0.07% (4-bit) / 0.01% (5-bit). |
| S2 heavy values | Threshold **3e-5 N** (n >= 306). cint8 with s_h 0.0773 (ns2) / 0.0547 (ns4). | The wire stays linear within 0.5% up to 250 sigma per sample. At 1e-4 N, the ns4 formats compress 2.8-2.9% at the rail, which breaks the plan's own "clip >= 2x the rail" requirement. Costs +0.2-1.4 Gb/s. |
| S2 sigma_hat tracking | **tau = 3 s**. Drop the "\|y\| > 6 sigma_hat" exclusion. Skip blocks < 50% non-zero. Initialise after the static-sky warm-up. | Within 0.04% of static under +-30% drifts, flag toggles and gain steps. An EMA started during the warm-up costs up to 11%. |
| S2 per-row guard (new) | A row whose own robust scale is > 1.5 sigma_hat is coded at that scale. | Without it, broadband noise surges (x3 or more) put 9.4-12 sigma of junk into 1-sample filters, the class of change behind the 07-22 storm. It acts on point sources only above 500-700 sigma per sample. WS-T calibrates the trip so it never fires on normal noise. |
| S2 image units | Output scale c = 1/sqrt(sum n V): the raw image is in sigma (1.00). | Layer-1 floor 0.078; Layer-2 floor 0.5 (post-Layer-1 units, never rescaled); clip 250; fp32 accumulation; guards 240/30 unchanged. A cuFFT-fp16 fallback would need prescale 1/256 (overflow at 20.6 sigma without it). |
| S2 cfp16 corr storage | Accept. | S/N factor 1 - 2e-8, no bias, 16x fp16 headroom (phase-only cal; amplitude cal needs \|g\| < 4). |
| per-baseline imaging | ns4 is equivalent to gridded; ns2 only if the network forces it. | ns4 0.998 beam-weighted / 0.993 corner, +0.005 px; ns2 0.994 / 0.974. No phase correction needed. |
| B-splines (S2e) | Adopt the **b3x** bank, subject to CP1 timing: box1, box2, [1,2,1], then B3 at widths 4-64. | x1.048 for Gaussian bursts and x1.026 scattered (x1.043 at 1 ms); x1.038 / x1.017 after live red noise. 148 history rows, 36 adds, 0.83x the false-alarm rate. Needs fp32, a filter id and per-filter K. |

## Contradictions with the plan / SPEC
1. **Bright response.** The plan's "today's curve: linear to ~35 sigma, rail 250 sqrt(k)" describes the
   regime before 09-23. With the prescale on, today's per-row max-abs wire compresses bright signals
   itself: 0.93 at 50 sigma, 0.80 at 100, 0.49 at 250, saturating near 150 sigma per sample, so the 250
   clip is never reached. The model reproduces the live raw image sigma (0.0674 predicted, 0.064
   measured). 2x is linear up to the clip, so bright events will read up to 2x higher than today. That is
   a regime change in the direction 3b43118 warns about. Needs the user's decision (ledger S3) and a
   C1/C2 bright-class replay before CP2.
2. **Today's wire costs 1.2% of S/N** (jitter of the amax/127 row weights), not the audit's 0.05%. Every
   2x format is 0.2-1.0% better on the wire, so the expected x1.20 becomes about x1.21.
3. **SPEC section 4 changes:** mid-rise -> mid-tread, tau 10 s -> 3 s, heavy 1e-4 -> 3e-5, plus the
   per-row guard.
4. The 07-20 storm regime differed from today's stable one mainly in a Layer-2 floor scaled by 1/256,
   which in effect disabled it. This is the likely cause of the storm (an inference, not a measurement).
5. Gridded loads are 2-3% above the plan and do not depend on declination. Gridded ns2 cint5 needs
   305k datagrams/s on the worst node, above the plan's 153-204k.
6. Time filters: the bank needs 8 filters (not "the same 7 widths") and 148 history rows (not ~170).
   Top-hat bursts that exactly match a boxcar width at 8-16 ms lose up to 4.5%.

## Still open
- Checking the row rule and guard against 1x RX recordings that include a storm window. No recording
  exists: the first recorder attempt (07:25 UTC 10-22) disturbed production and was stopped (see
  `signoff_ledger.md`).
- Re-measuring the b3x Layer-2 noise levels on recorded 2x cubes (CP1).
