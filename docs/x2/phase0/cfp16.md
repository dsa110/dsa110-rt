# Phase 0, study 3: cfp16 storage of the corr GEMM output before the stage-1 sums

Script: `scripts/cfp16.py` -> `results/cfp16.json`.

**What is stored.** In the planned 2x corr, the fused GEMM accumulates each element in fp32. An
element is the sum of K = 32 products: 2 pols x csf 2 channels x 8 native samples, over 4-bit SNAP
voltages with the phase-only calibration applied on load. Each element is written once as cfp16,
per (baseline, csf-2 channel, 0.262 ms sample). Stage-1 then sums those elements in fp32 over the
value's n sources and over 2 or 4 samples.

Today the 1x kernel writes cfp32 after the same fp32 accumulation (`services/corr_fast_kernel.py`,
the Triton no-cat path). The cfp16 store is therefore the only new rounding.

**Simulation.**
- Gaussian voltages at SNAP level sigma_v = 1.5, 2.5 and 4 counts per component, saturated to the
  4-bit range [-8, 7].
- A common point source (correlation 0.02), with instrumental phases that the on-load calibration
  removes. This makes the elements non-integer, as on the GPU.
- Exact fp64 elements, compared with the same elements after fp32 -> fp16 rounding (round to
  nearest even).
- Values formed as plain sums of n = 1, 96 and 3936 elements.

| sigma_v (counts) | rounding rms / noise rms | mean bias / sigma (re, im) | S/N fp16 / fp32 at n = 1, 96, 3936 | max element |
|---|---|---|---|---|
| 1.5 | 2.09e-4 | +1.7e-7, +1.3e-7 | 1.0000000, 1.0000005, 0.99999 | 104 |
| 2.5 | 2.08e-4 | +2.7e-7, -0.5e-7 | 1.0000015, 1.000007, 1.00002 | 283 |
| 4.0 | 2.09e-4 | +3.9e-7, +1.5e-7 | 1.0000027, 1.000005, 1.00004 | 645 |

- **S/N.** The fp16 rounding adds independent noise at 2.1e-4 of the element noise, which gives an
  S/N factor of 1/sqrt(1 + 4.4e-8) = **1 - 2.2e-8**. Each value sums independently rounded elements,
  so the ratio does not change with n. The n = 3936 ratios scatter by +-4e-5, which is Monte Carlo
  noise.
- **Bias.** Below 4e-7 sigma per element, consistent with zero (round-to-nearest-even is
  unbiased). Even if it added coherently over n = 3936 elements it would stay below 3e-5 sigma per
  value.
- **Headroom.** The largest possible element, every voltage at full scale |v| = 8 sqrt 2, is
  32 x 128 = 4096. That is 16x below the fp16 maximum, so saturating RFI cannot overflow.
  - fp16 relative precision is constant above the normal minimum (6.1e-5). Typical elements are
    tens of counts^2, so subnormals do not occur.
  - **Condition:** this holds only while the calibrated operands stay in count units. An
    amplitude calibration on load must keep |g|^2 x 4096 < 65504 (|g| < 4) and |g| above ~1e-2.
    Otherwise apply a fixed per-pattern pre-scale. Phase-only calibration, as deployed, is safe.
- **1x compatibility.** cfp16 storage is not bit-identical to today's cfp32. It differs at 2e-4 of
  the noise per element. 1x-compat mode must keep the cfp32 path, which the plan already covers
  ("1x mode keeps separate kernels if not bit-identical"). In 2x this is a sign-off item (S2,
  "cfp16 corr storage") with a measured cost of zero.

**Verdict: accept cfp16 storage.** The S/N cost is below 1e-7 and there is no measurable bias.
