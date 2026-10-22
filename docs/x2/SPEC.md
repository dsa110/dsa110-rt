# x2 — shared specification for the 2x time-resolution search

Status: **v0.1 (2026-10-22)**, owner: integration (main session). Implements the approved plan
(`/home/ubuntu/.claude-vikram/plans/velvety-jingling-crescent.md`, rev 4). Sections marked
**[FREEZE: WS-x]** are drafts that workstream x finalises in a PR to this file before implementing
against them; everything else is fixed. Change requests go through integration.

## 0. Ground rules (every workstream)
- Branches: long-running `x2/main`; each workstream works on `x2/ws-<name>` cut from `origin/x2/main`,
  pushes it, and integration merges into `x2/main`. Never commit to `main`.
- **No GPU execution outside observing interruptions.** CPU runs set `CUDA_VISIBLE_DEVICES=""`.
- **Tests run on a search node, never on h23** (CLAUDE.md), via `tools/x2/node_cpu_tests.sh` (scratch
  copy of the node checkout, nice 19, ionice idle, one core, single-threaded BLAS, per-file timeout).
  Pure modelling scripts (no repo tests) may run on h23 at nice 10.
- GPU tests are written now and marked `@pytest.mark.x2_gpu`; checkpoint runners execute them.
  Every GPU kernel ships a CPU twin (Triton under `TRITON_INTERPRET=1`, CUDA via a numpy twin of its
  index/layout logic) that runs in CPU CI.
- **1x is untouched until O2.** New behaviour is reached only through new modules or explicit 2x
  parameters; at today's parameters every changed module must be bit-identical to production
  (1x-compat taps, plan §Correctness 1). Existing tests must keep passing.
- Production is never touched outside a checkpoint: no etcd writes to live keys, no service restarts,
  no writes into `/home/ubuntu/proj/dsa110-rt` on nodes or the h23 live checkout.
- New code lives under `src/dsart/x2/` unless it extends an existing module; new tests under
  `tests/x2/`. Match surrounding style; pyflakes-clean.

## 1. Op-point parameters (`src/dsart/x2/opoint.py`) [FROZEN v1 by WS-P, 2026-10-22]
`OpPoint` is a frozen dataclass (pure python, no numpy import; validated in `__post_init__`). API:
`OPPOINT_1X`; `opoint_2x(value_kind="gridded", n_sub=2, bits_main=8) -> OpPoint` (alias `OPPOINT_2X`);
`FORMATS_2X: dict[str, OpPoint]` (the six CP1 format-table rows, keyed by id); `opoint_from_id(id)`;
`TierSpec(name, dm_lo, dm_hi, t_native)` with `.t_us`; constants `VALUE_KINDS`,
`STAGE2_CORR_FIFO_MERGED`, `STAGE2_SEARCH_SINGLE_RINT`.

| field | type | 1x | 2x |
|---|---|---|---|
| `name` | str | "1x" | "2x" |
| `t_int_fast_native` | int | 32 | 8 |
| `chan_sum_factor` | int | 8 | 2 |
| `n_sub` | int | 4 | 2 or 4 (format) |
| `stage1_q_native` | int | 32 | 8 |
| `stage2_mode` | str | "corr_fifo_merged" | "search_single_rint" |
| `value_kind` | str | "gridded" | "gridded" or "baselines" (format) |
| `bits_main` | int, bits per complex value | 16 (cint8) | 8 / 10 / 12 (cint4/5/6; format) |
| `bits_heavy` | int | 0 (no heavy class) | 16 (cint8) |
| `t_ext_native` | int | 32 (1048.576 us) | 16 (524.288 us) |
| `plan_version` | int | 2 (DmPlan v2 file) | 3 (`TieredDmPlan`) |
| `tiers` | tuple[TierSpec] | () | (A 50-1600 @ 16 native, B 1600-3000 @ 32 native) |
| `dm_tol`, `dm_ti_us` | float or None | None | 1.25, 40.0 |
| `coarse_density`, `coarse_density_t_ref_native` | float or None, int | None, 32 | 0.0206, 32 |
| `normalise_values` | bool | False | True (value / sqrt(n) on the wire, x sqrt(n) at the owner) |
| `heavy_frac` | float or None | None | 1e-4 (heavy iff n >= heavy_frac x N_image) |
| `redundancy_tol_m` | float | 0.05 | 0.05 |
| `n_grid`, `cell_lambda_mode` | int, str | 256, "tee45" | 256, "tee45" |
| `n_halves` | int | 8 | 8 |

Derived properties: `nchan_eff` (48 or 192), `n_streams` (16 x n_sub), `t_fast_us`, `stage1_q_us`,
`t_ext_us`, `chan_width_khz`, `opoint_id` ("1x", or "2x-<kind>-ns<n_sub>-cint<bits_main/2>", e.g.
"2x-gridded-ns2-cint4"; the string published to etcd). Value kind and bits do not change the DM plan.
The 2x tables reproduce `docs/timeres_study/snr2x/schemes.py::proposed(q1=262.144, n_sub=...)` bit for bit
(coarse, fine, f2c, stage-1 bins, stage-2 bins, applied delays; `tests/x2/test_plan_oracle.py`).

## 2. Tiered DM plan v3 (`src/dsart/x2/plan.py`, `tools/build_dm_plan_tiers.py`) [FROZEN v1 by WS-P, 2026-10-22]
`TieredDmPlan` (metadata `schema = "dsart.x2.TieredDmPlan"`, `version = 3`) lives in
`src/dsart/x2/plan.py`. It is torch-free, so it loads on h23. `DmPlan` v2 (`common/contracts.py`) and
`coarse_dm/dm_plan.py` are unchanged. v2 plans keep loading through them; tests build a v2 plan with
`tools/build_dm_plan_explicit.py` and load it.

**Definitions** (constants from `dsart.common.constants`; K = `K_DM_MS_GHZ2_PC`, T_nat = 32.768 us):
- **Fine grid per tier.** `gen_dm_list(dm_lo, dm_hi, T_tier_us, dnu = csf x 250/8192 MHz, nchans = 6144 / csf,
  F_CTR_MHZ, ti_us = 40, tol = 1.25)`. This is a verbatim port of `model/physics.py`. Values above `dm_hi + 1e-9`
  are dropped. `F_CTR_MHZ` = 1405.0152587890625 is built exactly like `physics.py`: top 1498.75, bottom =
  top - 6143 dnu.
- **Coarse grid per tier.** `n_c = max(1, ceil(0.0206 x (dm_hi - dm_lo) x (T_ref / T_tier) / n_sub))` with
  T_ref = 32 native. The tier has `n_c` uniform buckets over [dm_lo, dm_hi], and coarse = the bucket centre.
  `fine_to_coarse` maps each fine DM to the bucket containing it (`searchsorted(edges, fine, "right") - 1`,
  clipped).
- **Stage 1 (corr).** `a1[g,k] = 1e3 K (nu[g,k]^-2 - nu_ref[g, k // per]^-2)` us per pc/cc. Here `nu` =
  band-centre summed-channel frequencies (`summed_freqs_GHz(csf)`, bit-identical to
  `build_summed_chgroup_freq_table_GHz`), `nu_ref` = the first summed channel of the sub-band, and
  `per = nchan_eff / n_sub`. Then
  `stage1_bins[g,k,d] = rint(rint(a1[g,k] x coarse_dm[d] / T_nat) / q1_native)`, which is >= 0 and is 0 on every
  sub-band's first channel. The corr ADVANCES summed channel k by `stage1_bins x q1_native` native samples
  (`out[t] = vis_k[t + shift]`), which aligns it to the arrival at the sub-band reference. This equals
  `DMPlan.delay_bins_per_subband` at the 2x fast sample (test-pinned).
- **Stage 2 (search).** `a2[g,s] = 1e3 K (NU_BOT_PROC_GHZ^-2 - nu_ref[g,s]^-2)` and
  `stage2_bins[f, g x n_sub + s] = rint(a2[g,s] x fine_dm[f] / T_tier_us)`. There is one rounding, at the
  tier sample. The array is signed int32 by contract and >= 0 in this plan. The search reads
  `stream[t - stage2_bins[f, stream]]`, which aligns every stream to the arrival at nu_bot_proc (today's read
  convention).
- **Global indices.**
  - Coarse d: tier A 0..nA-1, then tier B.
  - **Global fine index f**: plan order, tier A fines then tier B fines (= coarse d ascending, fines ascending
    within d). `fine_dm` is strictly increasing.
  - Owner half of d = `d % 8`. Half h is search node `SEARCH_NODES[h // 2]`
    (`("n01", "n02", "n09", "n13")`), GPU `h % 2`.
- **plan_id** (u64). A blake2b-64 (person `dsart-x2-plan`) over the scalar inputs and every array below,
  hashing name, dtype, shape and little-endian bytes, in table order. It is independent of git sha and
  build time. SPEC §9 carries it.

**Fields** (read-only ndarrays). Shapes: T tiers (2), C coarse (47 at n_sub 2, 24 at n_sub 4), F fine
(758), G = 16, K = 192 summed channels, S = 16 n_sub streams, N = n_sub.

| name | dtype | shape | meaning |
|---|---|---|---|
| `tier_name` | <U8 | T | "A", "B" |
| `tier_dm_lo`, `tier_dm_hi` | f8 | T | 50/1600, 1600/3000 |
| `tier_t_native`, `tier_q1_native` | i4 | T | 16/32; 8/8 |
| `tier_coarse_offsets`, `tier_fine_offsets` | i4 | T+1 | global index range of each tier |
| `coarse_dm` | f8 | C | coarse DM (bucket centre) |
| `coarse_tier`, `coarse_t_native` | i4 | C | tier index; stream sample of d (16 or 32 native) |
| `fine_offset` | i4 | C+1 | CSR: fines of d = `fine_offset[d]:fine_offset[d+1]` |
| `n_fdm` | i4 | C | fines per coarse DM (VARIES) |
| `owner_half` | i4 | C | `d % 8` |
| `stage2_max`, `stage2_min` | i4 | C | extreme stage-2 shift over d's fines x all streams (tier samples; ring sizing) |
| `fine_dm` | f8 | F | fine DM, global index |
| `fine_to_coarse` | i4 | F | global coarse d |
| `fine_local_idx` | i4 | F | index within d's bucket (`f - fine_offset[d]`; the C1 `fine_dm_idx`) |
| `fine_tier` | i4 | F | tier of f |
| `stage1_bins` | i4 | G, K, C | see above (q1 units) |
| `stage2_bins` | i4 | F, S | see above (tier samples) |
| `summed_freq_GHz`, `stage1_coeff_us` | f8 | G, K | nu; a1 |
| `subband_ref_GHz`, `stage2_coeff_us` | f8 | G, N | nu_ref; a2 |

Scalars: `n_sub`, `chan_sum_factor`, `tol`, `ti_us`, `f_ctr_MHz`, `coarse_density`,
`coarse_density_t_ref_native`, `n_halves`, `metadata` (dict). Property: `plan_id`.

**API.**

Plan properties and methods:
- `n_tiers`, `n_coarse`, `n_fine`, `n_streams`, `nchan_eff`.
- `fines_of(d) -> slice`; `fines_of_half(h) -> i4[]` (global fines of half h, in global (d, fine) order);
  `dms_of_half(h) -> i4[]`.
- `stage2_for_coarse(d) -> i4[n_fdm[d], S]`; `t_us(d)`; `owner_node(d) -> "n01".."n13"`; `owner_gpu(d)`.
- `tier_dict(i)`: the oracle form.
- `validate()`: structural; runs on construction.
- `verify_rebuild()`: rebuilds from `metadata["inputs"]` and requires bit-identical arrays.
- `to_npz(path)`; `TieredDmPlan.from_npz(path, verify=False)`; `summary() -> dict`.

Module functions:
- `build_tiered_plan(*, n_sub, chan_sum_factor=2, tiers=(("A", 50, 1600, 16, 8), ("B", 1600, 3000, 32, 8)),
  tol=1.25, ti_us=40.0, coarse_density=0.0206, coarse_density_t_ref_native=32, opoint_id=None,
  git_sha=None, build_utc_ns=None)`.
- `build_tiered_plan_for_opoint(op, **kw)`.
- `load_any_plan(path)`: returns a v2 `DmPlan` or a `TieredDmPlan`.
- `plan_metadata_version(path)`.
- `gen_dm_list`, `tier_grid`, `summed_freqs_GHz`, `subband_ref_freqs_GHz`, `stage1_coeff_us`, `stage2_coeff_us`.
- Constants: `F_CTR_MHZ`, `SEARCH_NODES`, `N_HALVES`, `PLAN_SCHEMA`, `PLAN_VERSION`.

CLI: `tools/build_dm_plan_tiers.py --opoint <id> | --n-sub {2,4} --out <npz> [--git-sha]`. It writes the plan,
reloads it with `verify=True`, and prints the summary JSON.

**npz** (`allow_pickle=False`): every array above, plus `metadata` (JSON). The metadata holds `schema`, `version`,
`git_sha`, `build_utc_ns`, `build_utc`, `opoint_id`, `inputs{n_sub, chan_sum_factor, tol, ti_us, coarse_density,
coarse_density_t_ref_native, tiers[{name, dm_lo, dm_hi, t_native, q1_native}]}`, `constants{...}`,
`search_nodes`, `oracle` and `conventions`. A second key, `scalars`, is JSON too. `from_npz` refuses any other
schema or version, including the v1 `configs/dm_plan.npz` and v2 files. Note that `*_v3.npz` file NAMES under
`/home/ubuntu/data/dm_plans` are v2-schema plans; the `schema` key disambiguates them.

**Values.**

| | n_sub 2 | n_sub 4 |
|---|---|---|
| coarse A + B | 32 + 15 | 16 + 8 |
| n_fdm A / B | 15-18 / 15-17 | 30-35 / 28-30 |
| per half | 4 A + 2 B (half 7: 4 A + 1 B) | 2 A + 1 B |

Common to both n_sub:
- F = 526 + 232. Tier A fines run 50-1597.44; tier B fines run 1600-2996.47.
- `stage2_max` <= 1724 tier-A samples (904 ms) and <= 1617 tier-B samples (1696 ms).

## 3. Value maps (`src/dsart/grid/baseline_grid.py`) [FROZEN v1 by WS-P, 2026-10-22]
**Builders.**
- `build_value_map(antpos_e, antpos_n, *, antpos_u=None, is_core_baseline_mask, kind, n_sub, chan_sum_factor,
  dec_deg, cell_lambda, n_grid=256, heavy_frac=1e-4, normalise=True, redundancy_tol_m=0.05) -> ValueMap`.
- `build_value_map_for_opoint(op, antpos_e, antpos_n, *, antpos_u=None, is_core_baseline_mask, dec_deg,
  cell_lambda=None)`. With `cell_lambda=None` it uses `cell_lambda_for_pixel_arcsec(45, n_grid)` (tee45).
- `OPPOINT_1X` gives the 1x-compat map: gridded, n_sub 4, csf 8, `heavy_frac=None`, `normalise=False`.

Other functions:
- `redundancy_classes(antpos_e, antpos_n, antpos_u=None, *, is_core_baseline_mask, tol_m=0.05) -> RedundancyClasses`
- `kept_core_baselines(mask, nants=96) -> (bl, a, b)`
- `value_map_pattern_id(**build keywords) -> int`
- `heavy_threshold(n_image, heavy_frac) -> int`
- `build_owner_csr(streams, n_grid, normalise) -> OwnerCsr`

The builder is numpy-only. Importing the module pulls in torch via `dsart/grid/__init__.py`.

**Production inputs.**
- `antpos_e/n`: the float32 E/N header of the cal blob (`load_antpos_from_cal_blob`).
- `is_core_baseline_mask`: `core_baseline_mask_from_station_numbers` of the `configs/corr_setup_96.yaml`
  `antenna_order` (station <= 102; 82 antennas, 3321 cross-baselines in xGPU order a > b).
- `dec_deg`: the pointing DEC. It enters only `pattern_id` (0.25 deg bins), as today.
- `cell_lambda`: tee45.
- `antpos_u = None` (zeros). The station table gives every core station the same nominal elevation
  (1182.6 m). The residual ENU curvature is <= 8.5 mm across core baselines, and the 3-D classes built with
  it are identical to the 2-D classes (tested). A surveyed height table, if one appears, enters through
  `antpos_u` and `pattern_id`. Corr and search must pass identical inputs.

**Kinds.**
- **gridded.** A value is one uv cell. A stream's cells are exactly
  `build_pattern(..., kernel_support=1, sub_band=(s, n_sub))`, i.e. `build_subband_patterns`. Sources are
  every in-grid (core cross-baseline, summed channel of the sub-band), mapped with `FastVisGridder.from_pattern`
  arithmetic: `u = -dE/lambda`, `v = -dN/lambda` (F20), `col = rint(u / cell_lambda) + n_grid/2`, and row likewise.
  This is lambda-uniform, with NO DEC projection and no w (today's convention). `class_id = -1`.
  `legacy_pattern_id` = today's sub-band `pattern_id`.
- **baselines.** A value is one exact redundancy class.
  - Classes are single-linkage at `tol_m` on 3-D vectors in either orientation, and must have diameter <= tol
    (otherwise the builder raises).
  - Class ids are ordered by first member (lowest kept index). The canonical orientation is the first member's
    natural vector `pos[a] - pos[b]`. `bl_conj[i]` marks members whose natural vector is the reverse; their
    visibilities are conjugated before summing.
  - The value is the sum over members x the sub-band's summed channels.
  - The owner cell is the class's mean (E, N) at the sub-band CENTRE frequency (the mean of its summed-channel
    frequencies), with the same cell formula. Off-grid (class, stream) pairs are dropped and counted
    (0 with tee45).
  - `class_id` = the class.

**Per value.**
- `n` (`n_contrib`): the number of (baseline, summed channel) sources: gridded = in-grid sources of the
  cell; baselines = multiplicity x channels per sub-band.
- Owner cell `(cell_row, cell_col)`, uint16, in today's single-side `[n_grid, n_grid]` layout
  (`SparsityPattern.ix_row/ix_col`; LUT form `row x n_grid + col`, `ValueMap.cell_flat(stream)`).
- `heavy` = `n >= heavy_threshold`, where `heavy_threshold = ceil(heavy_frac x N_image)` in exact rationals and
  N_image = the sum of n over all values of all streams (10,202,112 at 2x, giving a threshold of 1021).

**Wire order** within a stream: heavy values first, then main values. Each group is in (cell key
`(row << 16) | col`, class id) order. With no heavy class (1x) this is today's pattern order.

**Data classes** (frozen; identity equality):

`ValueMap`:
- Scalars: `kind`, `n_sub`, `chan_sum_factor`, `n_grid`, `cell_lambda`, `dec_deg_quant`, `normalise`,
  `heavy_frac`, `heavy_threshold` (int or None), `n_image`, `redundancy_tol_m`, `pattern_id` (u64),
  `stream_pattern_ids` (tuple of u64; payload + stream index), `n_dropped_offgrid`.
- Members: `streams` (tuple[StreamValues], index = stream id g x n_sub + s), `owner` (OwnerCsr), `classes`
  (RedundancyClasses or None), `kept_bl` (i4[3321]).
- Methods: `n_streams`, `nchan_eff`, `n_values` (i4[S]), `n_heavy` (i4[S]), `cell_flat(stream)`,
  `corr_csr(stream) -> CorrCsr`, `corr_csr_chgroup(g) -> (CorrCsr, value_offsets[n_sub+1])`, `summary()`.

`StreamValues`:
- `stream`, `chgroup`, `sub_band`, `cell_row`/`cell_col` u2[V], `n_contrib` i4[V], `heavy` bool[V],
  `class_id` i4[V], `legacy_pattern_id`.
- Properties: `n_values`, `n_heavy`, `sqrt_n` and `inv_sqrt_n` (f4[V]).

`CorrCsr` (corr side, one stream; value v <- sources `ptr[v]:ptr[v+1]`):
- `ptr` i4[V+1], `bl` i4 (xGPU baseline index 0..4655), `ch` i4 (summed channel within the chgroup), and
  `conj` bool.
- Sources are in ascending `bl x nchan_eff + ch` order: the fixed fp32 summation order.
- Built on demand and deterministic. `ptr` diff == `n_contrib`.

`OwnerCsr` is the owner side, in the Hermitian half-spectrum of today's real-FFT combine
(`build_sparse_cell_csr`).
- Fields:
  - `cell_uw` i4[A] = `row x (n_grid/2+1) + col`, ascending.
  - `cell_ptr` i4[A+1].
  - Per entry: `ent_stream` i4, `ent_value` i4 (index into the stream's wire order), `ent_conj` u1, and
    `ent_weight` f4 (= sqrt(n) if `normalise`, else 1).
  - Entries within a cell are ordered by (stream, conj, value).
- Folding: a value at full cell (r, c) contributes
  - a direct entry at (r, c) if c <= n_grid/2;
  - a conj entry at ((n_grid-r) % n_grid, (n_grid-c) % n_grid) if that column is <= n_grid/2.
  Columns 0 and n_grid/2 get both.
- The consumer writes `G_h[cell] = 0.5 x sum(weight x (conj ? conj(x) : x))`. Then
  `irfft2(G_h, s=(N,N)) == Re(ifft2(G_full))`.
- The 0.5 and the fftshift checkerboard are kernel constants and are not in the CSR. One cell can have
  more than 64 entries in 2x.

`RedundancyClasses`:
- `tol_m`.
- Per kept baseline (i4[n_kept] unless noted): `kept_bl`, `ant_a`, `ant_b`, `bl_class`, and `bl_conj` (bool).
- `class_ptr` i4[n_class+1] and `class_members` i4[n_kept] (grouped, ascending).
- `class_uvw_m` f8[n_class, 3] (canonical mean E, N, U) and `multiplicity` i4.
- `max_diameter_m` and `min_gap_m`.

**pattern_id** = blake2b-64 (person `dsart-x2-vmap`) over:
- schema version 1, kind, csf, n_sub, n_grid, normalise, n_ants;
- cell_lambda, `quantise_dec_deg(dec)`, redundancy tol, heavy_frac (-1 if None), all as f8;
- the chgroup-table hash;
- antpos E/N/U as f8, and the core mask as u1.

**Natural weighting.** A value is the sum of its sources / sqrt(n) (corr) and is re-weighted by sqrt(n) at the
owner. For a phase-centre point source every (baseline, channel) then contributes weight 1, and the image
peak is N_image / N^2 for both kinds (tested).

**1x-compat** (tested). The `OPPOINT_1X` map equals today's production structures:
- Cells, order and legacy ids: `build_subband_patterns`.
- Corr-side source set per cell: `build_cell_csr(build_subband_layout(...).cell_index_map)`. Our
  within-cell order is fixed; today's `torch.argsort` order is unspecified.
- Owner side: `build_sparse_cell_csr(build_inverse_lut(...))`, entry for entry.

**Measured** (production antpos, tee45, DEC 71.63; plan-rev-4 model values in parentheses):

| map | values / stream | heavy / stream | heavy share of values / natural weight |
|---|---|---|---|
| gridded, n_sub 2 | 2318-2759 (mean 2544) | 70-78 | 2.9% / 42.4% (3.1% / 43.5%) |
| gridded, n_sub 4 | 2043-2267 (mean 2159) | 21-25 | 1.13% / 20.7% (1.2% / 21.7%) |
| baselines, n_sub 2 | 1748 | 80 (71) | 4.6% / 46.3% (40%) |
| baselines, n_sub 4 | 1748 | 26 (23) | 1.5% / 22.1% (19.5%) |
| 1x | 2032-2243 (mean 2134) | none | — |

**Redundancy: 3321 baselines form 1748 exact classes, not the model's 1773.**
- The largest multiplicity is 41.
- Within a class, members are at most 11.7 mm apart. Different classes are at least 2.88 m apart, so any
  tolerance from 1.2 cm to 2.8 m gives the same 1748 classes.
- The model's `rint(B / 0.05 m)` binning splits 1556 pairs that are at most 1.2 cm apart across bin edges.
  Random bin offsets give 1779-1802 classes.
- Every class the model finds lies inside one exact class (tested).
- Per-baseline formats therefore carry 1748 values, about 1.4% fewer than the load table assumed, with more
  of them heavy.
- The gridded counts differ slightly from the model because production gridding has no DEC projection. The
  model's `nfilled.py` used `v x cos(z)`.

## 4. Quantisation ("row rule") [FREEZE: WS-T + Phase 0]
Corr side, per (stream, coarse DM, sample) row of values x_v (complex):
1. normalise: y_v = x_v / sqrt(n_v) (static n from §3) — equal noise per value;
2. sigma_hat per (stream, coarse DM): slow EMA (tau ~ 10 s) of a robust estimate (1.4826 x median|re,im|
   over the block, after excluding |y| > 6 sigma_hat_prev);
3. main values: step_m = s_b x sigma_hat (s_4 = 0.336, s_5 = 0.191 — Phase 0 confirms), code =
   clip(floor(y / step_m), -2^(b-1), 2^(b-1)-1) per component (mid-rise);
4. heavy values: cint8, step_h = s_h x sigma_hat with s_h chosen so the clip (128 x step_h) is >= 2x the
   value's S/N at today's 250-sigma image rail (Phase 0 sets s_h; ~0.1);
5. owner decode in sigma_hat units: v = (code + 0.5) x s_b (main) or x s_h (heavy), then x sqrt(n_v)
   when gridding — natural weighting of what was summed, 1/sigma_hat weighting across streams (today's
   per-row 1/max-abs semantics without the burst-dependent compression). step values travel in the
   header for monitoring only.
1x-compat keeps v1 exactly: per-row max-abs cint8, codes summed without the scale.

## 5. Frame v2 (`src/dsart/transport/prod_frame2.py`) [FREEZE: WS-T]
Same magic, `version = 2`, 72-byte header (little-endian, packed): magic u32 | version u16 | flags u16
(v1 bits + bit5 heavy-present) | seq_first u64 (stream-sample index of the first sample, per (stream,
dm), tier clock) | t0_native u64 (absolute native-sample index of the first sample — same epoch as
C2's `search_to_snap_specnum`) | stream_id u16 (g x n_sub + s) | dm_idx u16 (global coarse d) | k u8
(samples in datagram) | n_sub u8 | bits_main u8 (8/10/12) | bits_heavy u8 (16 or 0) | n_grid u16 |
t_int_factor u8 (native samples per stream sample: 16 or 32) | value_kind u8 (0 cells, 1 baselines) |
n_values u32 | pattern_id u64 | n_heavy u32 | step_main f32 | step_heavy f32 | payload_bytes u32 |
reserved u8[8]. Payload = k samples, each: n_heavy x (int8 re, int8 im), then the main values as a
packed little-endian bit stream (re then im per value), padded to a byte. k = floor(8900 / bytes per
sample). v1 frames are unchanged and still accepted.

## 6. Ring v4 and fan-in [FREEZE: WS-T]
Owned DMs only (<= 16 per node), per-DM cadence and t_buf, write_seq per (stream, local DM), layout
[dm][stream][t][bytes] so a (dm, stream) window is contiguous for DMA, per-DM fan-in (>= 15/16 of
streams or a deadline with the missing streams flagged and zero-filled), per-DM stall watchdog,
retention >= 20 s, overrun counters per DM. Readers never write into the shared header (the
`read_slot` overrun-counter trap). API consumed by WS-S: open(shm), dms(), wait_window(dm, t_first,
n_rows) -> (view, validity), counters().

## 7. Search engine contract [FREEZE: WS-S]
Per half: owned DMs from the plan; per-DM cubes (hop 384 tier-A / 192 tier-B samples, cycle 201.3 ms
aligned to absolute specnum); amplitude regime identical to today in sigma units (`input_clip_sigma`
250, Layer-1/Layer-2 floors re-derived for the new image units — Phase 0, saturation guard 240);
per-DM Layer-1 (applied sigma travels with the cube) and Layer-2; candidates carry global DM index,
fine DM value, width in external samples, tier, (l,m). Today's image orientation/fftshift/edge mask
unchanged. On-demand cutout service per half (schema from WS-D, §9).

## 8. External clock
Downstream of the search engine the only sample unit is T_ext = 524.288 us (16 native samples):
`event_specnum` = absolute native index // 16, widths in T_ext samples (tier-B widths are even), C1
batch header `sample_period_us = 524.288` in 2x. Every "samples" parameter downstream is audited
(WS-D inventory) and either converted to ms or doubled.

## 9. Cutouts and raw windows [FROZEN v1 by WS-D, 2026-10-22]
Full text, rationale and consumer coverage: `docs/x2/downstream_inventory.md` §3. Summary of the frozen
interface:

**Units and indices.**
- **Rows.** A row is the absolute T_ext index `native // native_per_ext`: 16 at 2x, 32 at 1x. Native
  index = 2 × SNAP specnum.
- **Time reference.** Times are arrival times at `NU_BOT_PROC_GHZ` after dedispersion at the row's
  fine DM. C1 `event_specnum`, the request, the cutout and Level3 all share this index.
- **Tier B.** Values occupy rows 2k and 2k+1. `t0_ext` is even. Statistics use only rows with
  `r % t_step_ext[f] == 0`.
- **Widths.** Integer T_ext samples. A FWHM in ms, if adopted, is a separate field.
- **Global fine index** (WS-P must define it in `TieredDmPlan`): plan order, tier A then tier B. In 2x it
  is carried by the C1 `dm_idx_global`; today that field is the half-local fine index. The C1
  `fine_dm_idx` is the index within its coarse bucket. Owner half = `d % 8`.

**Request (C2 → every half).**
- `CutoutRequest`: 128 B little-endian, `struct "<IHH16sIHHqqIIIHHffffihhhHfdQH14s"`.
- Fields, in order: magic `b"DSCQ"`, version 1, flags, event_name, request_id, trigger_class_id
  (crc32 & 0xFFFF), reserved, t_peak_ext, t0_ext, n_t, raw_pre_ext, raw_post_ext, width_ext,
  stamp_half_px, dm_peak, snr_peak, l_rad, m_rad, fine_global_peak, coarse_global_peak, l_pix, m_pix,
  n_stamp_t, stamp_dm_halfwidth, mjd_peak, plan_id, deadline_s, reserved.
- Flags: bit0 CUTOUT, bit1 RAW_ARCHIVE (cleared for injections by default), bit2 INJECTION, bit3 MANUAL
  (Dump Now: `l_pix = −1`, `dm < 0`), bit4 RESEND.
- Sent over UDP to the existing `coinc.dump_broadcast.hosts[sid]` : `port_base + g` (11227 / 11228).
  Old 1x listeners drop it on size. A 2x half ignores and counts 64-B DSRT v1 packets.
- Defaults: n_t 512 (268 ms), raw ±954 rows (±0.5 s), stamps S = 16 / 64 rows / ±150 pc/cc,
  deadline 15 s.

**Status (half → C2).**
- `CutoutStatus`: 64 B, `struct "<IHBB16sIBBBBfIQIh10s"`. Magic `b"DSCS"`.
- State: ACCEPTED, then exactly one of DONE / PARTIAL / FAILED. Reason codes are in §3.3 of the
  inventory.
- Sent over UDP to `c1.c2_endpoint.host:11501` (`coinc.cutout.status_port`).
- Lossy. Used only for resends (2 s, at most 3) and early completion.

**Half behaviour.**
- Queue depth 8.
- Raw snapshot from ring v4 first: wait for per-DM frontier coverage, up to `deadline_s`, then zero-fill
  and flag. A start that has already rotated out is TOO_LATE.
- Then re-image every owned fine over `[t0, t0+n_t)` on the low-priority stream, from the snapshot,
  with the Layer-1 sigma applied at detection.
- Write atomically `cutouts/cutout_s<sid>_g<g>.npz` and then `.json`.
- Upload the cutout first, then the raw window (bwlimit). Each half purges its own files after rc==0,
  with a per-product `UPLOAD_OK`.

**Cutout file** (`dsart.cutout` v1). Uncompressed NPZ, one per half per event, always produced.
- F = the half's fines in global (d, fine) order. D = the half's owned coarse DMs.
- Images are Layer-1-normalised, the same as today's `cube`.

| group | keys |
|---|---|
| scalars | provenance (opoint, plan_id, pattern_id, code_version); clock (`t_ext_us`, `native_per_ext`, `sample_period_specnum`, `t0_ext`, `n_t`, `mjd_t0`, `mjd_clock`); request echo; image geometry (`n_grid`, `cell_l_rad`, `cell_m_rad`, `l0_rad`, `m0_rad`); `status`, `status_json` |
| fine axis [F] | `fine_dm_pc_cc`, `fine_global_idx`, `coarse_global_idx`, `fine_local_idx`, `tier`, `t_step_ext`, `input_valid_frac` |
| coarse axis [D] | `coarse_owned`, `fine_offset[D+1]`, `row_valid[n_t, D]`, `apex_t`, `apex_f`, `apex_image[D, n_grid, n_grid]` |
| planes [n_t, F] | `peak_grid` (full-field width-1 max, today's definition), `peak_pix`, `pix_lc` (value at the request pixel), `sigma_applied` |
| images | `has_peak_image`, `peak_image`, `peak_image_w` (mean over the width window at the peak fine), `stamp_fine`, `stamp_t0_ext`, `stamp[n_stamp_t, F_s, 2S+1, 2S+1]` |
| optional | `kernel_ids`, `sigma_k[D, K]` |

- The apex follows C3's definition exactly: `nanargmax` over the coarse block, all rows.
- Size ≈ 2-4 MB per half.

**Raw window archive.** One directory per half: `raw_s<sid>_g<g>/` holding `raw_d<DDD>.npz` per owned
coarse DM, plus `meta.json` and `RAW_OK.json`.
- `raw_d<DDD>.npz` holds the ring v4 bytes verbatim, `codes[n_stream, n_rows, bytes_per_sample]`, along
  with step fields, `valid`, per-stream row-0 seq / t0_native, and the frame scalars.
- Value maps are referenced by `pattern_id` and uploaded once to `operations/x2/patterns/`.
- Size ≈ corner-turn load × window: 4.5-6.6 GB per event at 1.0 s depending on format.
- Rebuilt into cubes by `tools/x2/reimage_raw_window.py`.

**C2 lifecycle.**
- Expected halves come from the plan's owner map. C2 counts distinct per-half manifests, not files.
- At `coinc.cutout.wait_s` (120 s) it finalises Level3 v2 with a `products` block listing
  expected / present / missing (with reasons), and `status` complete or incomplete.
- The pending set is persisted. Late manifests are recorded.
- **An event is never deleted.** The 1x `rmtree` stays behind `coinc.partial_event_policy`
  (default `discard`) until the user signs off.
- C3, the plotter, Slack and the dashboard read products through one adapter, which also reads 1x
  cubes bit-identically.

## 10. Op-point switching [FROZEN v1 by WS-O, 2026-10-22]
Code: `src/dsart/x2/opoint_runtime.py`, `src/dsart/x2/config_diff.py`, `tools/ops/push_dsart_to_etcd.py`,
`tools/dashboard/dsa_monitor/opoint_switch.py`, and the Start control (`app.py /control/start`,
`control_store.control_start_fleet`). Tests: `tests/x2/test_opoint_runtime.py`, `test_push_opoint.py`,
`test_opoint_switch.py`.

**Selections.** A selection is what the operator picks and what etcd sets are keyed by:
- `1x`: today's search, run by the unified code in 1x-compat mode;
- `1x-legacy`: today's search as deployed before x2 (until it is retired);
- a 2x op-point id `2x-<kind>-ns<n>-cint<b>` (§1 `OpPoint.opoint_id`). The UI offers the six CP1
  rows of `FORMATS_2X`.

Both 1x selections have op-point id `1x`. `opoint_runtime.validate_selection` / `op_of` /
`config_set_of` map a selection to its `OpPoint` and its config set.

**Config sets (repo).**
- `configs/{dsart_pipeline_rt,dsart_search_rt,c2_trigger_criteria}.yaml` is the base set. It stays
  byte-identical to production and is used by `1x` and `1x-legacy`.
- `configs/opoints/<set>/<same names>` holds a set's own files. `1x` and `1x-legacy` fall back to
  the base per file. `2x` never falls back.
- `configs/opoints/2x/*` is a template for every 2x format:
  - routine `args`/`hostargs` may use `${var}`. `render_context(sel, plan, plan_path)` supplies:
    - from the op-point: `opoint_id`, `n_sub`, `n_streams` (16 x n_sub), `value_kind`,
      `bits_main`, `bits_heavy`, `t_int_fast_native`, `chan_sum_factor`, `t_ext_us`,
      `t_ext_native`, `stage1_q_native`, `n_grid`;
    - from the v3 plan: `plan_path`, `plan_id` (0x%016x), `n_coarse`, `n_fine`, `max_fine_dm`;
    - per half h: `owned_coarse_csv_h<h>`, `n_owned_h<h>`, `n_owned_A_h<h>`, `n_owned_B_h<h>`,
      `n_fine_h<h>`;
    - per search node X: `owned_coarse_csv_X`, `n_owned_X`.
  - Variables are allowed only there, because the h23 services read every other block straight
    from the file.
  - Paths to op-point-managed files must point into the set's own directory.
  - Values not decided yet are `X2TODO:<owner>:<what>` tokens (they fail argument parsing) or
    entries of a top-level `x2_todo:` list. Both block a switch.
  - `x2_cp2_recalibrate:` lists 1x values to recalibrate at CP2. It does not block.
- **Time units are explicit per op-point.** The code defaults are 2x. `time_units_for(op)`
  requires, with the op-point's value:
  - corr_fast `--t-int-fast-native` and `--chan-sum-factor`;
  - both search halves' `--t-int-search-us` (= `t_ext_us`) and `--chan-sum-factor`;
  - `c1.max_c1c2_width_samples` (= 512 native: 16 | 32);
  - `c1.merger.sample_period_specnum` (= 16 at every op-point: wall-clock merge window);
  - `cube.sample_period_specnum` (16).

  Validation fails if any of these is missing or wrong.

**etcd.**
- `/cnf/opoints/<sel>/{pipeline_rt,search_rt}`: rendered sets. Each carries top-level `opoint`
  (the op-point id) and `opoint_selection`; 2x sets also carry `plan_id` and `plan_path`.
- `/cnf/opoints/<sel>/meta`: provenance (`opoint`, `selection`, `config_set`, `plan_id`,
  `plan_path`, `pushed_utc`, `user`, `host`, `source` yaml|live, per-file sha256, `git_sha`,
  `placeholders`).
- `/cnf/opoint`: the active record `{opoint: <op-point id>, selection, config_set, plan_id,
  plan_path, set_utc, user, host, previous, sha256, set_meta}`. Absent means selection `1x`.
- `/cnf/opoints/_previous/{pipeline_rt,search_rt,opoint}`: a one-deep backup written by each switch.
- `/cnf/opoints/_kpark/<1x|2x>/<bucket>`: injection K parked per K family.

Writers:
- `push_dsart_to_etcd.py --opoint <sel>` writes `/cnf/opoints/<sel>/*` only. It is a dry run unless
  `--write`, prints a structural diff, and validates the set. A 2x selection needs `--plan <v3
  npz>`, whose `plan_id` must equal a rebuild from the op-point. `--seed-from-live` snapshots
  production into a 1x set.
- The Start control is the only writer of `/cnf/pipeline_rt`, `/cnf/search_rt` and `/cnf/opoint`
  for switches.
- The legacy `push_dsart_to_etcd.py` without `--opoint` is unchanged.

**Services.**
- **h23 services** (C2, C3, inject bot):
  - `--opoint {auto | <selection>}`, default `auto` (`opoint_runtime.add_opoint_argument`);
  - at startup they call `select_service_config(<--config path>, cli_opoint=...)`. This reads
    `/cnf/opoint` once (3 tries; on an etcd outage, the last-seen value from
    `~/.local/state/dsart/opoint_last.json`, then 1x with a warning). It returns the
    `ActiveOpPoint` (`selection`, `opoint_id`, `config_set`, `plan_id`, `plan_path`, `source`) and
    maps the unit's base-set path to the set's file;
  - the op-point is fixed for the process lifetime: a switch restarts them;
  - C2 criteria hot reload re-reads the same (set's) criteria file;
  - every service publishes `opoint` (the op-point id string, inventory §4), `opoint_selection` and
    `plan_id` in its mon payload.
- **Node services** get `--opoint <op-point id>` in their rendered routine args. Absent means
  today's (1x-legacy) behaviour.
- **Plan check.** Every service that loads the v3 plan must compare `plan.plan_id` with the set's
  `plan_id` (template variable `${plan_id}`, or `/cnf/opoint.plan_id`) and refuse to run on a
  mismatch.

**Switch (web-UI Start with an op-point selection).**

A Start with the active selection and no "re-apply" is today's start. It first checks that the
live configs carry the active selection's stamp; an unstamped config counts as a 1x selection.

Otherwise `opoint_switch.switch_opoint` runs these steps:
1. Fleet READY (`compute_system_state`) and no search orchestrator running.
2. `systemctl --user` reachable. The dashboard is the systemd --user unit `dsa_monitor.service`
   under the same manager; `SystemdUser` fills `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` from
   `/run/user/<uid>` when missing.
3. The stored set is valid: rendered for the selection, explicit time units, no placeholders, no
   cross-set references, same plan in both instances.
4. The h23 files the services will read are present and clean, and the inject bot state file is
   not shared across K families.
5. The h23 copy of the plan (if present) has the same `plan_id`. A different plan refuses the
   switch; a missing plan is a warning.
6. Backup of the live keys.
7. `dsart_inject_bot` is stopped.
8. One atomic etcd write (transaction) of the two live configs, `/cnf/opoint`, and the K moves.
   Between K families (1x | 2x), the live `/cnf/inject/snr_calibration/*` is parked and the new
   family's parked K is restored. The bot's own K memory is per op-point because the 2x set has
   its own `inject_bot.state_path`.
9. A read-back of the write.
10. Restart, in order: `dsart_c2`, `dsart_c3`, `dsart_inject_bot`, `hiplot_c1`, `hiplot_c2`.
11. A wait until each unit is active/running on a new MainPID that stays stable for 8 s (90 s
    timeout).
12. Node housekeeping, then the start broadcast.

A failure after step 7 writes every touched key back to its exact previous value (absent keys are
deleted) and restarts the units again. The Start is then refused (HTTP 409) with the reason.
Every step is audit-logged under namespace `opoint`.

Without `PYTHONHASHSEED` in `dsart_c2`, the switch warns. The drop-in
`systemd/dsart_c2.service.d/x2-pythonhashseed.conf` fixes this.

The banner shows the active selection, its plan_id, and whether the live configs match.
`GET /control/opoint` reports which stored sets are switchable.

## 11. Ownership (single owner per file; others request changes through the owner)
- WS-P: `src/dsart/x2/opoint.py`, `src/dsart/x2/plan.py`, `src/dsart/grid/baseline_grid.py`,
  `tools/build_dm_plan_tiers.py`, `tests/x2/test_plan*.py`, `tests/x2/test_baseline_grid*.py`,
  `tests/fixtures/x2/dsa110_antpos.npz` (cal-blob E/N, corr_setup_96 station order, station-table U).
  `src/dsart/x2/__init__.py` and `tests/x2/__init__.py` are empty package markers: anyone may add
  them, nobody puts code in them.
- WS-T: `src/dsart/transport/prod_frame2.py`, `send_mmsg.c` + wrapper, `recv_epoll.c`, `recv_ring.{c,h,py}`,
  `production_rx_ring.py`, `services/search_rx.py`, `bench/fake_corr_to_search.py`, `tools/x2/rx_recorder.py`.
- WS-C: `services/corr_fast_integration.py`, `src/dsart/x2/corr/*`, `src/dsart/x2/ref/corr2x_ref.py`.
- WS-S: `services/search_compute.py`, `services/cube_pipeline.py`, `src/dsart/x2/search/*` (incl. the
  cutout producer `cutout_service.py`), `image/*`, `detector/*`, `noise_norm/*`, `fine_dm/combiner.py`,
  `src/dsart/x2/ref/search2x_ref.py`, `services/search_compute_mon.py`, `services/search_ring_mon.py`.
- WS-D: `services/c1_emit.py`, `services/coincidencer.py`, `coinc/*`, `services/c3.py`, `inject/*`,
  `dump/*`, `services/slack_notify.py`, `services/candidate_card.py`, `src/dsart/x2/cutout/*`,
  `tools/inject/*`, `tools/viz/*`, `tools/c2/*`, `tools/ops/prune_candidate_products.py`,
  `tools/dashboard/**` except the Start control, grafana, pusher.
- Integration: `common/contracts.py`, `docs/x2/SPEC.md` (sections are frozen by their owners; edits
  between freezes go through integration), `tools/x2/node_cpu_tests.sh`, merges into `x2/main`.
- WS-O: `configs/*.yaml`, `tools/ops/push_dsart_to_etcd.py`, dashboard Start control
  (`app.py /control/start`, `control_store.control_start_fleet`), `tools/x2/cp*_runner*`, runbooks.
- Phase 0: `docs/x2/phase0/**`, `docs/timeres_study/model/*` additions.
