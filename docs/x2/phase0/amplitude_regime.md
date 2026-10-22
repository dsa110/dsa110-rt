# Phase 0, study 2: today's amplitude regime, re-expressed for the 2x image units

Script: `scripts/amp_regime.py` -> `results/amp_regime.json`. The wire curves come from study 1
(`results/rowrule_bright.json`). Code locations are on `origin/x2/main` (= main @ 79b6878).

## 1. Today's chain (live since 2026-09-23)

| # | stage | what it does to amplitudes | where |
|---|---|---|---|
| 1 | corr TX | each (stream, DM, sample) row: `code = clip(round(x * 127/amax_row), -128, 127)` (mid-tread) | `transport/tx.py:858-866` (`_encode_rows`) |
| 2 | search combine | codes summed **without** the per-row scale; Hermitian half-spectrum `G_h = 0.5 (G + conj G(-k))`, cfp16 | `image/imager_gpu.py:312` (compact = unit scale only), `:549-566` |
| 3 | uv prescale | `uv *= 1/256` before the FFT | `imager_gpu.py:393, 579-580`; `--imager-uv-prescale 0.00390625` at `configs/dsart_search_rt.yaml:313, 389` |
| 4 | FFT | cuFFT cfp16 `irfft2`, torch "backward" 1/N^2 -> `real(ifft2(G))` | `imager_gpu.py:590` |
| 5 | edge mask x 1/sigma_L1(prev) (fused), inf -> +-60000 | | `cube_pipeline.py:2287-2354`, `_clamp_inf_to_finite :107` |
| 6 | Layer-1 | per-fdm sigma-clipped std over [T, H, W] (edge-mask cells excluded), clamped below at `layer1_sigma_floor x prescale` | `noise_norm/layer1.py:590-591`; floor `yaml:802` (5e-3) scaled at `services/search_compute.py:702-716` |
| 7 | detector input clip | `cube.clamp_(+-250)` in post-L1 sigma units | `cube_pipeline.py:2356`; `input_clip_sigma: 250.0` `yaml:873` |
| 8 | boxcars | unweighted sums (k_img unit, k_dm d1, b1..b64), **fp16 accumulation**; start-up guard that 250 x 1 x 64 = 16000 < 65504 | `--detector-boxcar-accum-dtype fp16` `yaml:292, 392`; `cube_pipeline.py:815-852` |
| 9 | Layer-2 | sigma_k EMA (tau 30 s), floor **0.5**, upward ratio clamp 4.0, escape after 64 cubes | `noise_norm/layer2.py:366-433`; `yaml:829, 842, 856` |
| 10 | C1 | snr_min 9.55 (matched with the edge-mask sigma fix) | `yaml:456` |
| 11 | injection guards | `SATURATION_OBSERVED_SNR = 240` (K not stored at or above it), `IMAGER_SAFE_OBSERVED_SNR = 30` (probe aim) | `tools/dashboard/dsa_monitor/inject_calibration.py:1021, 1012`; check at `:795` |

The search detector `detector/forward.py` does not clip. The 250 clip lives in `cube_pipeline`.

### Today's numbers

| quantity | value | source |
|---|---|---|
| raw image sigma, un-prescaled | 0.064 live; **0.0674 model** (per-row amax, real n) | yaml:85-90 comment; this study |
| raw image sigma, prescaled (what Layer-1 sees) | 2.5e-4 | |
| Layer-1 floor, raw | 5e-3 x 1/256 = 1.953e-5 = **0.078 sigma_ss** | |
| fp16 FFT linear range | ~4000 sigma (prescaled); 15.6 sigma without prescale | 65504 / (sigma_raw N^2) |
| boxcar fp16 worst sum | 250 x 64 = 16000 | |
| sigma_k at b1 / b64 | ~1.0 / 1.163 x 8 (audit, live) | |
| **wire saturation, one sample** | **~150 sigma per sample**; L = 0.97 at 35, 0.93 at 50, 0.80 at 100, 0.49 at 250 | study 1, section 7 |
| detection rail | ~150 x sqrt(k) from the wire. The 250 clip is not reached by a point source. | study 1 |

The model raw sigma (0.0674) agrees with the live value (0.064) to 5%. That validates the per-row
max-abs wire model behind the compression numbers.

### The regime history, and which knob actually matters

| period | prescale | L1 floor (raw) | L2 floor | clip | accum | outcome |
|---|---|---|---|---|---|---|
| before 07-16 | 1 | 5e-3 | 0.5 | 250 | fp16 | stable; fp16 FFT overflow at ~35 sigma -> 60000 -> rail 250 sqrt(k) |
| 07-20 19:00 (3a5b8db) | 1/256 | 5e-3 (unscaled, binding) | 0.5 | 1000 | fp32 | L1 floor 42x too high: C1 silent; storm begins |
| 07-20 22:41 (f0eccd9) | 1/256 | 1.95e-5 | **1.95e-3** | 300 | fp16 | 60-480 bright_frb_extragalactic per day (median 13 sigma) |
| 07-22 revert (3b43118) | 1 | 5e-3 | 0.5 | 250 | fp16 | storm gone |
| **since 09-23 (35110f5 + 80e7b66)** | 1/256 | 5e-3 x 1/256 (in code) | **0.5** | 250 | fp16 | stable, no storm |

The stable live regime differs from the storm regime in two ways. First, the Layer-2 floor was 0.5,
not 0.5/256. Second, the clip was 250, not 300. A clip change at 250-300 sigma cannot make 13-sigma
candidates.

Layer-2 works on the Layer-1-normalised cube, so its floor is in **post-L1 sigma units**, which are
invariant to any image scale. Dividing it by 256 disabled the protection it gives against sigma_k
depression by sparse-valid cubes. Our inference is that this removal, not the prescale, is the
storm's most likely cause. The live record since 09-23 supports this. **Rule for 2x: re-derive only
the Layer-1 floor (raw image units); never scale the Layer-2 floor with the image units.**

## 2. 2x image units

The owner decodes v = code x s_b (heavy: x s_h) x sqrt(n), in sigma_hat units. The FFT uses fp32
butterflies. Each value contributes per-component variance n_v V_v, where V_v = E[Q^2] of its
quantiser (0.99-1.01). So:

- **Option A (the plan's "1/N^2")**: sigma_img = sqrt(sum_v n_v V_v) / 256^2 = **0.0486** (all
  formats within 0.3%).
- **Option B (recommended)**: output scale c_img = 1 / sqrt(sum_v n_v V_v). This is one static float
  per pattern, computed with the CSR, so **sigma_img = 1.00**. Under antenna flags (zero values) it
  becomes sqrt(active weight fraction). Channel flags are absorbed by sigma_hat.

Option B puts the raw image directly in sigma units. The Layer-1 sigma then becomes a monitor that
should sit at ~1.0, the analogue of the "s_k_min ~ 1.0 after any prescale change" check. No floor
then depends on an empirical steady-state value. The cost is the same as option A: one multiply,
already present in K3a.

### Equivalent constants

| knob | today | 2x, option B (recommended) | 2x, option A | change at |
|---|---|---|---|---|
| uv prescale | 1/256 | none (fp32 butterflies). **1/256 if the plain cuFFT-fp16 fallback is ever used**: without it fp16 overflows at **20.6 sigma**; with it at 5270 sigma | same | K3a; `imager_gpu.py` if reused |
| image output scale | 1/N^2 | c_img = 1/sqrt(sum n V) = 3.14e-4 | 1/N^2 | K3a (`src/dsart/x2/search/*`) |
| steady sigma_img | 2.5e-4 | 1.00 | 0.0486 | |
| Layer-1 floor (raw units) | 1.953e-5 = 0.078 sigma_ss | **0.078** | 3.80e-3 | `noise.layer1_sigma_floor` + the scaling at `search_compute.py:702`. 2x: floor = 0.078 x sigma_pred, with sigma_pred from the pattern (1.0 in B), not via `imager_uv_prescale` |
| Layer-2 floor | 0.5 (sums of unit-sigma samples) | **0.5** for boxcars. For a weighted kernel h_k replacing boxcar width w_k: 0.5 x \|\|h_k\|\|_2 / sqrt(w_k), the same fraction of the seed as today | same | `detector.layer2_sigma_floor` (per-kernel in 2x) |
| Layer-2 seed | sqrt(w) implicit | \|\|h_k\|\|_2 (plan) | same | |
| sigma_max_ratio / escape / tau | 4.0 / 64 cubes / 30 s | 4.0 / 64 cubes (per-DM state, 201.3 ms cycle = 12.9 s as today) / 30 s | same | |
| input clip | 250 sigma per sample, post-L1 | **250** | 250 | `detector.input_clip_sigma` -> K3b |
| boxcar accumulation | fp16 (guard 16000 < 65504) | fp32 (plan K3b); the fp16 guard becomes irrelevant. Keep it for any fp16 fallback: 250 x max support < 65504 needs support <= 262 (B3 ~170 OK) | | `cube_pipeline.py:815` |
| fp16 image ring | | max 6.5e4 sigma, ULP 1e-3 sigma at 1 sigma | max 1.3e6 sigma, ULP 6e-4 sigma | |
| saturation guard | 240 | **240** (0.96 x the smallest rail, b1 = 250) | | `inject_calibration.py:1021` |
| probe aim | 30 | 30 (keep; the 2x wire is linear far beyond) | | `inject_calibration.py:1012` |
| C1 floor | 9.55 | 9.55 (needs the edge-mask-excluding sigma, as today) | | `yaml:456` |

Rail per filter (2x): R_k = 250 x \|\|h_k\|\|_1 / (\|\|h_k\|\|_2 x c_k), with c_k = sigma_k/\|\|h_k\|\|_2
(1.0-1.16 live for boxcars):
- boxcar w: 250 sqrt(w);
- triangle (two boxcars of m): 250 sqrt(1.5 m) for m >> 1;
- B3 (three boxcars of m): about 250 x 1.36 sqrt(m).
`amp_regime.json` has the exact values for m = 2..32.

## 3. What cannot be made identical (sign-off)

1. **The bright end of the wire.** Today's per-row max-abs compresses a single sample by -20% at 100
   sigma and saturates at ~150 sigma per sample, so today's effective rail is ~150 sqrt(k) at
   1.05 ms. The 2x rule (study 1) is linear to the clip, so the 2x rail is 250 sqrt(k) at 0.524 ms.
   For a burst of width W >= 1 ms the rail is 2.4x higher in S/N than today, and below 35 sigma per
   sample the two agree within 3%. This is the plan's intended "rail 250 sqrt(k)". It means bright
   RFI and bright FRBs read higher than today.
   - Mitigations already in the design: the per-row guard (study 1, section 6) restores today's
     per-row normalisation for noise surges, and the 250 clip is kept.
   - Check needed before CP2: replay recorded or synthetic bright-RFI rows through both rules and
     compare the C1 rate above 35 and above 50 sigma (bright_galactic and peak-member rules). That
     is the 3b43118 lesson.
2. **+1.2% on all S/N** from removing today's row-weight jitter (study 1, section 2). Injection K
   must be wiped and recalibrated (plan) anyway.
3. Layer-1 and Layer-2 run per DM. The estimators are unchanged, and the applied sigma travels with
   the cube (the `test_layer1_fused_overlap.py` class).

## 4. Test recipe (bright ladder, three levels)

**CPU, I1, before CP1.** Use synthetic full-rate streams through `ref/corr2x_ref` -> frame v2 ->
owner CPU path. Inject point sources at per-sample S/N rho in {10, 20, 35, 50, 100, 150, 250, 400,
700, 1000}, widths k in {1, 2, 4, 16, 64} samples (0.524 ms), at three positions (centre,
half-way, corner). Pass criteria:
- the wire L(rho) matches `rowrule_bright.json` within 0.5%;
- detected S/N = min(rho_tot x L, R_k) within 2%, with R_k = 250 sqrt(k)/c_k for boxcars;
- Layer-1 sigma = 1.00 +- 0.02 (option B), and s_k(b1) = 1.00 +- 0.05;
- no non-finite values at 1e4 sigma;
- a broadband noise surge F = 2, 3, 10 on one sample gives max pixel <= 6.8 sigma with the guard on
  (>= 9.4 with it off, which proves the test can fail).

**1x reference curve.** Run the same ladder through the 1x CPU twin (per-row max-abs). The predicted
curve is L = 0.93 at 50, 0.80 at 100, 0.49 at 250, saturating at ~150 per sample. This curve, not
"linear to ~35, rail 250 sqrt(k)", is today's.

**On sky, CP2 E.** Fire the bright injection ladder at target 20/35/50/100/200 sigma for w = 1, 4
and 16 (native-equivalent), in 1x and then in 2x. Pass criteria:
- the 2x/1x ratio follows 1/L_today(rho/sqrt(k)), for example x1.25 at 100 sigma per sample;
- the 2x rail sits at 250 sqrt(k);
- the guard counters stay at zero on injections.

After any change to image units: Layer-1 sigma ~1 (option B), s_k_min ~1.0, and n_clamped_high ~0.
