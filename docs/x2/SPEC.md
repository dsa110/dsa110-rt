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

## 1. Op-point parameters (`src/dsart/x2/opoint.py`) [FREEZE: WS-P]
One frozen dataclass `OpPoint` with presets `OPPOINT_1X` (today) and `OPPOINT_2X(fmt)`:

| field | 1x | 2x |
|---|---|---|
| t_int_fast_native | 32 | 8 |
| chan_sum_factor | 8 | 2 |
| n_sub | 4 | 2 or 4 (format) |
| tiers | one: DM plan v2 file | A: DM 50-1600, T = 16 native (524.288 us); B: 1600-3000, T = 32 native |
| stage-1 quantum (native) | 32 | 8 |
| stage-2 | corr FIFO + merged search rounding | corr FIFO off; one rint at the fine DM, search side |
| value kind | gridded cells | gridded cells or unique baselines (format) |
| bits main / heavy | cint8 (16) / none | cint4/5 (8/10) / cint8 (16) |
| external sample | 1048.576 us | 524.288 us |
| fine grid | v2 plan | `gen_dm_list` tol 1.25, ti 40 us, csf-2 channels, f_ctr = band centre |
| coarse count | 8 | ceil(0.0206 x dDM x (T1/T) / n_sub) per tier, uniform-width buckets, coarse = bucket centre |
The 2x tables must reproduce `docs/timeres_study/snr2x/schemes.py::proposed(q1=262.144, n_sub=...)`
exactly (coarse, fine, f2c, stage-1 bins, stage-2 bins) — that file is the oracle.

## 2. Tiered DM plan v3 (`src/dsart/x2/plan.py`, `tools/build_dm_plan_tiers.py`) [FREEZE: WS-P]
New class `TieredDmPlan` (metadata `version = 3`); `DmPlan` v2 is not modified and keeps loading.
- Per tier: `T_native`, `coarse_dm[c]`, `fine_dm[f]`, `fine_to_coarse[f]` (variable fines per
  coarse: n_sub 2 -> 15-18, n_sub 4 -> 28-35), `q1_native`.
- Global coarse index d: tier A coarse 0..nA-1 then tier B. Owner half of d = `d % 8`
  (half h -> search node index h // 2, gpu h % 2 in `N_SEARCH` order n01, n02, n09, n13).
- `stage1_bins[g, k, d]`: per chgroup g, summed channel k, global coarse d =
  rint(rint(a1[g,k] x DM_c / T_nat) / q1_native), a1 relative to the sub-band reference = first summed
  channel of the sub-band (`DMPlan.delay_bins_per_subband` semantics, band-centre summed frequencies
  from `build_summed_chgroup_freq_table_GHz(csf)`).
- `stage2_bins[f, stream]` (signed int32; stream = g x n_sub + s) = rint(a2[g,s] x DM_f / T_tier),
  a2 = delay from the sub-band reference to nu_bot_proc. One rounding, applied at the search side.
- Derived per-DM quantities the search needs: n_fdm per DM, max positive/negative stage-2 shift per DM
  (ring sizing), owner map.
- npz round-trip with explicit schema version, git sha, build UTC, inputs; validators.

## 3. Value maps (`src/dsart/grid/baseline_grid.py`) [FREEZE: WS-P]
One function builds, per (chgroup, sub-band) stream, the ordered list of values and their sources,
for either kind:
- **gridded cells**: sources = (baseline, summed channel) pairs of core cross-baselines (station
  number <= 102; `core_baseline_mask_from_station_numbers`), each mapped to its nearest uv cell at that
  channel's frequency exactly as `grid/sparsity_pattern.build_pattern` does today (same cell_lambda,
  sign/flip convention, DEC projection). A value = one cell.
- **unique baselines**: exact 3-D redundancy classes (positions agree within 5 cm incl. height), with
  per-member conjugation flags; a value = one class summed over the sub-band's channels; the owner
  places it at the cell nearest its uv at the sub-band CENTRE frequency.
- Per value: static contribution count `n` (number of (baseline, summed channel) sources), and the
  owner-side cell (row, col) in today's image layout and conventions.
- **Heavy values**: n >= N_image / 1e4 (N_image = total contributions per image over all streams);
  ordered FIRST in each stream; count `n_heavy`. Then main values in cell order.
- CSR forms: corr side (value <- sources, fixed source order) and owner side (cell <- (stream, value,
  conj, weight)), plus `pattern_id` = blake2b-64 over every input (kind, antpos, heights, chgroup table,
  csf, n_sub, n_grid, cell_lambda, quantised dec, tolerances, heavy threshold).
- For kind = gridded, n_sub 4, csf 8 the map must equal today's `build_subband_patterns` /
  `build_cell_csr` cell set and order (1x-compat check).

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

## 10. Op-point switching [FREEZE: WS-O]
etcd `/cnf/opoints/{1x,1x-legacy,2x}/{pipeline_rt,search_rt}`, `/cnf/opoint`; per-op-point sections in
`configs/dsart_pipeline_rt.yaml`, `dsart_search_rt.yaml`, `c2_trigger_criteria.yaml`; services take
`--opoint`. The web-UI Start control switches only when stopped (plan §Op-point switching).

## 11. Ownership (single owner per file; others request changes through the owner)
- WS-P: `src/dsart/x2/opoint.py`, `src/dsart/x2/plan.py`, `src/dsart/grid/baseline_grid.py`,
  `tools/build_dm_plan_tiers.py`, `tests/x2/test_plan*.py`, `tests/x2/test_baseline_grid*.py`.
- WS-T: `src/dsart/transport/prod_frame2.py`, `send_mmsg.c` + wrapper, `recv_epoll.c`, `recv_ring.{c,h,py}`,
  `production_rx_ring.py`, `services/search_rx.py`, `bench/fake_corr_to_search.py`, `tools/x2/rx_recorder.py`.
- WS-C: `services/corr_fast_integration.py`, `src/dsart/x2/corr/*`, `src/dsart/x2/ref/corr2x_ref.py`.
- WS-S: `services/search_compute.py`, `services/cube_pipeline.py`, `src/dsart/x2/search/*`,
  `image/*`, `detector/*`, `noise_norm/*`, `fine_dm/combiner.py`, `src/dsart/x2/ref/search2x_ref.py`.
- WS-D: `services/c1_emit.py`, `services/coincidencer.py`, `coinc/*`, `services/c3.py`, `inject/*`,
  `tools/dashboard/**` except the Start control, grafana, pusher.
- WS-O: `configs/*.yaml`, `tools/ops/push_dsart_to_etcd.py`, dashboard Start control
  (`app.py /control/start`, `control_store.control_start_fleet`), `tools/x2/cp*_runner*`, runbooks.
- Phase 0: `docs/x2/phase0/**`, `docs/timeres_study/model/*` additions.
