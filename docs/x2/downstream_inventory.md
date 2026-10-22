# x2 downstream inventory, cutout interface and WS-D task plan

Status: **v1.0 (2026-10-22)**, owner WS-D. Inventory phase only: no code was changed. Every reference
is `file:line` on `origin/x2/main` @ 99f332f, whose code is identical to `origin/main`. Paths without a
prefix are relative to `src/dsart/`; others start with `configs/`, `tools/`, `tests/`, `systemd/` or
`docs/`. "op" = `dsa110-operator` (separate repo, read-only). Plan: `velvety-jingling-crescent.md` rev 4;
spec: `docs/x2/SPEC.md` (this branch freezes SPEC §9 from §3 below).

Classification used throughout:
- **2x** — correct today, must change (or be parameterised) for 2x.
- **both** — a latent defect today; the fix is needed in both op-points.
- **1x=** — the 2x change can be made with 1x behaviour bit-identical (config/header-derived).
- **1x≠** — fixing it changes 1x behaviour; needs the user's sign-off before O2 (SPEC §0).

Units used below: native sample 32.768 µs; SNAP specnum 65.536 µs = 2 native
(`common/constants.py:24-28`); 1x search sample 1048.576 µs = 32 native = 16 specnums; 2x external
sample T_ext = 524.288 µs = 16 native = 8 specnums. **Two different "16"s exist:** today's
`sample_period_specnum` (specnums per sample, becomes 8) and 2x's native-per-sample (was 32). Several
modules below use one where the other is meant.

---------------------------------------------------------------------------------------------------

## 0. Main findings (things the plan does not cover, or covers incorrectly)

1. **C2 counts files, not halves.** It waits for `plot_expected_cube_count` (8) files matching
   `cube_s*_g*_*.npz` (`services/coincidencer.py:2153-2163`). When the deadline passes it
   `rmtree`s the event (`:2101-2114,2184-2195`) and re-sweeps the directory after 300 s (`:791,2140-2143`).
   - The filename carries a specnum (`dump/cube_dump.py:800`, `dump/proactive_stager.py:235`), so two
     files from one half can reach "8" while another half is missing.
   - C2 writes Level3 only on completion, and C3 finds events only by Level3 (`services/c3.py:295-312`).
     A discarded event is therefore invisible to C3, Slack and the dashboard.
   - The literal 8 is repeated in C3 (`c3.py:104,172`, dead config, never read), Slack
     (`services/slack_notify.py:87,377`, which globs `*.npz` rather than `cube_s*`), the dashboard
     (`tools/dashboard/dsa_monitor/cands_panel_funcs.py:522`, `templates/bursts.html:397`,
     `templates/burst_event.html:344`) and the inject bot (`inject/bot.py:1090`).
   - §3.5 replaces all of these with one expected-halves set taken from the plan's owner map.
2. **`dm_idx_global` is not global.**
   - `CubePipeline` never passes `fine_to_coarse` to the detector (`services/cube_pipeline.py:2455-2461`).
     The decoder therefore falls back to `dm_idx = fdm` (`detector/decoder.py:202-205`).
   - So the C1 column `dm_idx_global` (`services/c1_emit.py:448`) is the half-local fine index 0..24.
     It is identical on all 8 halves, contradicting `docs/c1c2/C1C2_WIRE_SCHEMA.md` §1.3.
   - C2 only archives it today (`coinc/archive.py:73`). In 2x the cutout and the C2 → half request must
     use a real global fine index (§3.1). (**both**, **1x≠** only if C2 starts using it.)
3. **One coarse DM per half is hard-wired in the C1 stage.**
   - `_select_dm_owner_half` keeps exactly one coarse DM and checks K = N_fine/8
     (`services/search_compute.py:2365-2404`).
   - `_geom_from_slot` labels every cube with the prefix `fine_dm_pc_cc_full[:n_fdm]` (`:1371-1378`).
     With 3-6 coarse DMs per half, every cube after the first would carry wrong DM values. Those
     wrong values would feed the noise-colour filter, the DM-floor filter and C1.
   - The retention ring rebuilds whenever `n_fdm` or `t_det` changes (`:1658-1684`). That would
     happen on almost every per-DM cube.
   - (WS-S scope, listed because downstream depends on it.)
4. **C1 metering, the escape and the governor are per cube, not per cycle.**
   - `_submit_c1_batch` runs once per cube (`search_compute.py:1541-1542`). Cap 8 per cube
     (`configs/dsart_search_rt.yaml:589`), escape per block 1 (`:571`).
   - With N per-DM cubes per cycle, the fleet row ceiling rises N-fold (today ≈318 rows/s).
   - Governor tokens go to whichever DM cube of a cycle is processed first, not to the brightest.
     A bright wide burst spanning several same-half DMs can spend its tokens without shipping its
     peak DM (`c1_emit.py:120-142`).
   - Required: pool the cycle's candidates before width cap, governor and metering. This is
     identical to today when N=1 (**1x=**).
5. **The noise-colour de-rate costs sensitivity at the top of today's DM range.**
   - `_DEDISP_CHAN_SUM_FACTOR = 8` (`search_compute.py:266`) with strength 4.0 (`yaml:631`).
     The smear length L exceeds 1 sample above DM ≈ 1398. At the deployed plan's top (DM 1496), L
     is about 1.07.
   - So every width ≥ 2 candidate there has its S/N divided by 1.13 (w=2) to 1.26 (w=64). I checked
     the formula at `search_compute.py:337-394` by hand.
   - The rule was calibrated for DM ≥ 2300 floods under the old plan. In 2x (csf 2, T_ext units)
     L < 1 below DM ≈ 2800, so it becomes a no-op. (**both**, **1x≠**; outside 2x scope.)
   - Reported to the user, not changed.
6. **The C2 matcher will reject 2x injections at DM ≥ ~2000.**
   - The time gate is ±4096 search samples with 16 specnums per sample (`coinc/inject_match.py:201,207`).
     Neither is configurable from C2 (`coincidencer.py:932-953`).
   - At T_ext, ±4096 samples is ±2.15 s, the window that already rejected DM-2500 probes on
     2026-06-10. Fix: a tolerance in seconds (4.294967296 s reproduces 1x exactly) plus the row's
     period (**1x=**).
   - If the tier-B `event_specnum` were left in 1.05 ms units, the residual would fall through to
     the 28 s wall-clock fallback, so a units bug would stay hidden.
7. **The C2 width gates are raw sample integers.**
   - `width_median_max_samples` 16/8/32 (`configs/c2_trigger_criteria.yaml:61,104,127`) is compared
     to `width_samples` with no unit (`coinc/criteria.py:157-161,217-218`). At 2x the same file would
     halve every width cut.
   - Converting 1x values to ms (16 × 1.048576 = 16.777216) can flip the w=16 boundary by one
     float ulp. So 1x keeps `_samples`, and 2x uses new `_ms` predicates computed per member
     (**1x=**).
8. **Duplicate dumps (both).**
   - The class holdoff makes `evaluate` fall through to the next matching class (`criteria.py:391-404`).
   - `bright_galactic` has `dm_galactic_fraction_max: 1.0` (`c2_trigger_criteria.yaml:129`), while
     the header comment says < 0.75. So a ≥35σ burst with DM/gal_dm in [0.75, 1] is dumped a second
     time as `bright_galactic` while `bright_frb_extragalactic` is in holdoff.
   - The faint class got an 11.99 cap for exactly this reason (`:98`). 2x makes it more likely,
     because there are ~3-6x more batches per burst and the DM-50 floor brings the [0.75G, G] band
     into reach.
   - Fix: per-component "already fired" memory (**1x≠**).
9. **C2's rate guards count evaluations.** `cluster_rate` counts once per touched component per batch
   (`coincidencer.py:1506-1509`). At 2x, 3-6x more batches reach `cluster_rate_max: 400`
   (`yaml:1012`) that much sooner, and while it is tripped no dump fires. Several other C2 costs
   also scale with rows/s × batches/s:
   - O(N) graph adds (`coinc/components.py:134-138,251-258`);
   - full rebuilds on age-out (`:160-210`);
   - a synchronous 24 h CSV concat on the event loop (`coincidencer.py:2003-2010`).

   A replay throughput test at 2x rates is required (plan §Downstream already asks for it).
   `ingest_queue_depth: 512` (`yaml:905`) holds about 13 s today, not the "100 s" in its comment,
   and about 2 s at 2x.
10. **The trigger listener ignores `flags`; any 64-byte DSRT packet to 11227/11228 is a cube dump**
    (`dump/c2_trigger_listener.py:376-409,430`). Bumping the trigger `version` would break both the
    search listener and the corr voltage listener (`wire.py:375-378`; `dump/voltage_trigger_listener.py:149-156`).
    → The cutout request is a NEW 128-byte packet with its own magic (§3.2): old listeners drop it on
    size, and the voltage path is untouched.
    `trigger_class_id = hash(name) & 0xFFFF` (`coincidencer.py:1659,1904`) changes on every C2
    restart (PYTHONHASHSEED not set, `systemd/dsart_c2.service`); use crc32.
11. **No sample period anywhere in the archive.**
    - Level3 `c2`, the C2 CSV and the C1-window CSV (`coinc/archive.py:68-117,373-426`) all omit it.
      `WindowEntry` drops it (`coinc/window.py:34-56`), although `stats.peak_sample_period_us` exists
      (`coinc/stats.py:168-170`).
    - Three readers hard-code 1048.576:
      - the C3 filterbank (`coinc/filterbank.py:52,249`);
      - the candidate card (`services/candidate_card.py:70,294`);
      - plotter defaults (`coinc/plotter.py:799`).
    - HiPlot shows raw sample widths. Adding a CSV column mid-hour corrupts the rolling CSVs
      (`coinc/csv_rotator.py:186-187,98-102`): append new columns at the end and roll the file on any
      schema change.
12. **The plotter mislabels the DM axis today and assumes contiguous ownership.**
    - Row→DM comes from the DM plan via `_owner_coarse_idx = owners_half0[sid] + g`
      (`coinc/plotter.py:243-260,308-323`). Only the first routine's `--coarse-dm-owners-half-0` is
      parsed.
    - When the bucket check fails, the fallback is `np.arange(n_fdm)`, still labelled "dispersion
      measure (pc cm⁻³)" (`:784-789,1613-1627,1677-1704`). This is a latent bug (**both**).
    - Rows are stacked in (sid, g) order, assumed to be ascending DM (`:1222,1571-1588`). All halves
      are cropped to a common T with no time anchor (`:1574-1577`).
    - The cutout therefore carries a per-row DM and time anchor, and the plotter sorts and regrids.
13. **Monitoring is mostly log-only.**
    - Missing from etcd:
      - search stage timings and cubes/s (`search_compute.py:2054-2075`);
      - the RX counters (`services/search_rx.py:214-245`; the pusher deliberately drops `/mon/search_rt/<cn>/rx`,
        `tools/dashboard/dsart_rt_to_influx/pusher.py:1661-1673`, pinned by a test);
      - TX metrics (`transport/tx.py:154,463-479` publishes to a nonexistent emitter).
    - Corr block time is in etcd but not Influx. It is a single wall sample, not p50/p99, and is
      only published when `--inject-watch` is passed (`services/corr_fast_integration.py:4059-4079`).
    - `mon_snapshot()` has no caller (`search_compute.py:1084-1117`), so the `c1_cands_dropped_color`
      mon-point the yaml tells operators to tune against does not exist.
    - The committed Grafana JSON is older than `build_dashboard.py` (missing the 10-05 governor
      panels).
    - The h20 pusher is outside `update_dsart` and outside the dashboard service inventory.
14. **The search ring's mon `_end` field is still the 16x units bug** (`services/search_ring_mon.py:180-184`;
    `tests/test_search_ring_mon.py:158-159` pins it). It is a fifth instance after the four fixed at
    `search_compute.py:2184-2198`. Dump Now does not read `_end` today (**both**, mon-only).
15. **The 2x defaults are already in the code.** `T_INT_SEARCH_US_DEFAULT = 524.288`
    (`common/constants.py:523`), with an inconsistent `cube_sample_period_specnum = 16`
    (`search_compute.py:578-579`). 1x is correct only because argv passes `--t-int-search-us 1048.576`
    (`yaml:282,374`). Op-point configs must set every one of these explicitly.
16. **The C1 merger window deliberately uses the units "bug".** It multiplies by
    `merger.sample_period_specnum` (16), which makes it 16x wider (`detector/merger.py:166-177`; `yaml:540`).
    Keeping 16 in 2x keeps the window constant in wall-clock time. Setting it to 8 "for consistency"
    would halve it. Pin it with a test.
17. **Injection K has no op-point provenance, and a wipe has three parts.**
    - etcd `/cnf/inject/snr_calibration/`.
    - Bot state `k_last_good` / `k_calibration_dec` (`inject/bot.py:700,893`). If this survives,
      the K doctor writes 1x K back after any suspicious shot (`:905-933`). The ×3 recal guard rejects
      legitimate 2x K changes (`:757-782`).
    - `fleet_kcal.py` is beamformer weights, not injection K; it must not be used for this.
    - Bucket `dm1600` straddles tiers A and B.
18. **The `op` probe is broken in both op-points.** It never sends `inj_id`, and `/control/inject`
    returns 400 without it (op `control/verbs.py:119-133`; `tools/dashboard/dsa_monitor/app.py:1661-1662`).
    Its default probe (DM 100, width 1 native, target 30 at l=m=0) also hits the search floor, a
    missing K bucket, the >30 guard and the sidereal veto (op `monitor/injection.py:28-35`).
19. **SPEC gaps.**
    - §11 assigns no owner for `dump/*`, `services/search_compute_mon.py`,
      `services/search_ring_mon.py`, `common/contracts.py`, `services/slack_notify.py`,
      `services/candidate_card.py`, `tools/inject/*`, `tools/viz/*` or `tools/c2/*`. Proposal in §5.
    - SPEC §5's `t0_native` says "same epoch as `search_to_snap_specnum`". That function returns
      **specnums** (65.536 µs), not native samples, so the units differ by 2.
    - SPEC §8 is consistent with today's C1 `event_specnum` (native//32 at 1x, native//16 at 2x).

---------------------------------------------------------------------------------------------------

## 1. Consumers of cube data today

### 1.1 The producer side (what exists)

| piece | where | what it does |
|---|---|---|
| retained cube | `services/cube_pipeline.py:446-482,517-563,665-700` | Pinned-host ring of `cube_norm` = **Layer-1-normalised** images `[t_det, n_fdm, n_grid, n_grid]` fp16 (`cube_pipeline.py:2486,2675,2723`), depth 12 (`yaml:507`, ≈2.4 s at 201 ms/cube) |
| lookup | `cube_pipeline.py:485-514` | `find_cube_for_specnum`: newest cube with `start ≤ s < start+t_det` (search samples), time only |
| trigger listener | `dump/c2_trigger_listener.py` | Bind `bind_host:(11227+g)` (`:84-110`); 64-B DSRT v1 (`:379-405`). On a hit, writes `<dump_root>/<event>/cube_s<sid>_g<g>_<trigger_specnum>.npz` (`:665-704`). `too_early` requests are parked for up to 120 s (`:430-443,489-630`; `yaml:675`). On `too_late`, claims the proactive stage (`:455-481`). Never checks `flags` |
| writer | `dump/cube_dump.py:695-793` | Atomic tmp + fsync + `os.replace` (`:744-788`). `peak_grid = cube.max(axis=(2,3))` → **(t_det, n_fdm)** (`:729-732`; the docstring says the opposite order). Queue 4 with udp>auto displacement (`:276-279,482-543`) |
| NPZ keys | `cube_dump.py:747-784` | `cube`, `peak_grid`, `mjd_start`, `event_specnum_start` (= the **trigger** specnum on C2 dumps, `c2_trigger_listener.py:687-691`), `cube_specnum_start` (true row-0 anchor, −1 if absent), `cube_mjd_start`, `sample_period_specnum` (16), `t_det` (256), `n_fdm_in_cube` (25), `n_grid` (256), `cluster_record` (JSON or "null"), `trigger_source`, `search_node_id`, `gpu_half`. **Missing:** `sample_period_us`, fine-DM values, coarse index, tier |
| proactive stager | `dump/proactive_stager.py:133-400` | Stages bright cubes (≥50σ, `yaml:735-760`) under `pending_g<g>_<specnum>/` (`:232`), keyed by specnum only (`:231-235,302,336-338`) |
| uploader | `coinc/cube_uploader.py:120-170,293-615` | rsyncs the whole event dir to `<root>/<event>/cubes/` (dest hard-coded at `:152`), excluding `upload.log` (`:95-100`). Each half purges its own `cube_*_g<g>_*.npz` after rc==0 (`:520-545,585-615`) and writes a dir-level `UPLOAD_OK` |
| retention sweeper | `dump/cube_retention.py:60-230` | Tiers: tmp 2 h, age 96 h, size 150→120 GB (`yaml:707-713`). "Uploaded" means **any** `UPLOAD_OK` line in the shared dir-level `upload.log` (`:92-123`). One half's marker therefore marks the other half's not-yet-uploaded cube as uploaded, and it is sacrificed first (**both**, minor) |

### 1.2 Consumers

| consumer (file:line) | finds files by | reads | shapes / derived | what a 2x product must supply |
|---|---|---|---|---|
| **C2 completion gate** `coincidencer.py:2137-2195` | `cubes/cube_s*_g*_*.npz` count ≥ 8 | nothing inside | counts files, not halves | Expected-halves set, per-half manifest, no discard (§3.5) |
| **C3 veto metrics** `coinc/cube_veto.py:620-683` | `cubes/cube_s*_g*_*.npz` (`:638`); sid/g from the filename (`:645-651`) | `peak_grid`, `t_det`, `n_fdm_in_cube` (`:589-596`; accepts both orientations); `cube[t,fdm]` via a hand-made mmap of the stored `cube.npy` (`:423-454,665-670`) | Trigger half's `peak_grid` (T, F) → see the R-rule rows below | Per-coarse-DM `peak_grid`, its fine axis, apex image, row validity |
| ↳ trigger row `cube_veto.py:599-617,626-636` | `Level2/C1_window_<ev>.csv` max-SNR row | `search_node_id`, `gpu_half`, `fine_dm_idx`, `l_pix`, `m_pix`, `snr`, `dm_pc_cc`, `width_samples` | `fine_dm_idx` indexes cube axis 1 directly | C1 row → (half, coarse d, fine-in-d) mapping (§3.1) |
| ↳ R1 image offset `:231,500-518` | apex image (t_apex, f_apex) of the **trigger half** | full 256² plane | argmax pixel distance to C1 (l,m) > 40 px | `apex_image[d]` at the apex defined exactly as C3 computes it |
| ↳ R2 time shift `:234,488-497` | trigger `peak_grid` | — | \|argmax over (t,f)ᵗ − argmax of the trigger-DM light curve\| > 30 samples ("31 ms") | rows in a known period; threshold in ms |
| ↳ R3 / R5 DM shift / edge `:237,245-249,495-498` | trigger `peak_grid` | — | `dm_shift > 8` trials; apex on fine bin 0 or F−1 **of the cube** (= the bucket edge) and > 4 trials | per-bucket fine axis; trials or pc/cc |
| ↳ R4 prominence `:240-243` | light curve + apex image | — | robust z < 4 for both | as R1 + R2 |
| ↳ R10 `tz_trig` `:283,492` | trigger-DM light curve over the whole cube | — | robust z of `peak_grid[:, f_trig]` ≥ 8 (≈256 rows; width-biased, memory note) | ≥ today's span (268 ms) of width-1 full-field max; independent rows per tier |
| ↳ R11 (off) `:286-316,384-420` | every half | — | lag-1 autocorrelation along time + frac\|z\|>5 over the live span | independent rows per tier (duplicated tier-B rows would fake ac1 ≈ 0.5) |
| ↳ global apex, dead tail `:520-555,345-369` | all halves | — | recorded only (`g_apex_cube_same` not a rule); `live_span` heuristic for the zero overlap tail | `row_valid` replaces the heuristic |
| **plotter** `coinc/plotter.py:748-838` | `cube_s*_g*_*.npz` regex (`:184-186,764`) | `cube` (mmap; ZIP_STORED only, `:646-715`), `mjd_start`, `sample_period_us` (default 1048.576, never used, `:799`), `cube_mjd_start`, `cube_specnum_start`, `sample_period_specnum` (`:807-824`). Not `peak_grid` (re-reduces about 8×855 MB, `:1197-1232`) | — | — |
| ↳ DM-time panel `:1213-1232,1542-1792` | all halves | `cube.max(axis=(2,3))` | Stacked (sid,g), cropped to the common T; row→DM from the plan (`:272-323`); robust row-normalised | **All DMs from all halves**: full-field max per (t, fine), per-row DM, per-tier period and anchor |
| ↳ burst selection `:1172-1194,1235-1335` | C1 members / CSV | — | `t = event_specnum − cube_specnum_start` (`:979-1052`); pixel series `cube[:, f, l, m]` (`:1055-1074`) | Pixel series at (l,m) for every fine; anchor in C1 units |
| ↳ image panel `:1795-1871` | burst half | `cube[t0:t1, f]` mean over the width window, full 256² | Reticle in full-grid pixels, circle r = 8 px | Full plane at the peak fine, averaged over the width window |
| ↳ light curve `:1874-1945` | burst half | pixel series + full-field max at f | Boxcar, robust z, x in samples | `pix_lc` + `peak_grid` at the peak fine |
| ↳ regenerate `:1998-2042` | `cubes/` | as above | — | Must keep the 1x reader |
| **candidate card** `services/candidate_card.py` | reads no NPZ; 4 PNGs `Level2/plots/{dm_time,image_peak,kernel_snrs,lightcurve}_<ev>.png` (`:75-80,515-519`) + Level3 `c2` | — | `width_median × 1.048576 ms` (`:70,283-295`) | PNG names unchanged; Level3 period |
| **Slack readiness** `services/slack_notify.py:353-396` | 4 PNGs + `len(cubes/*.npz) ≥ 8` (`:87,377-380`) | — | — | Product status from Level3 |
| **dashboard** `cands_panel_funcs.py:257,522,574-579`; `app.py:989,1072,3081`; `bursts.html:397`; `burst_event.html:229,344-352` | counts and lists `cubes/*.npz`; serves PNGs | none | "n/8" | Product status from Level3 |
| **Dump Now** `tools/dashboard/dsa_monitor/cube_dump_now.py` | sends 8 v1 triggers to `hosts × (0,1)` (`:74,236-290`); target = fleet min `newest_event_specnum_start + t_det//2` from `/mon/search/<sid>/<g>/ring` (`:400-474`); the fallback uses corr specnums where search samples are expected (`:752-800`, **both**, says it "will likely miss") | — | — | A no-candidate cutout request; a raw-ring window mon key |
| **viz / dev** `tools/viz/m6_cube_dump_verifier.py:128-166`; `tools/viz/cube_burst_explorer.ipynb`; `bench/preflight/_render_*.py` | `cube_*.npz` | full 4-D, strict shape check | 49×49 stamps over time; pixel DM-time across 8 cubes | Retire or port to cutouts / the offline re-imager |
| **localisation** (`dsa110-voltage-imaging`, `dsavim/metadata.py:126-200`) | Level3 + `filterbank.json` + voltage manifests | **no cubes** | `c2.t_peak_mjd`, `l_median`, `m_median`, `dm_median`, `snr_max` | Keep the `t_peak_mjd` reference (arrival at the band bottom, 1.311 GHz) for both tiers |
| **C3 reject / pruner** `c3.py:632-642`; `tools/ops/prune_candidate_products.py:82` | delete `cubes/*.npz` | — | — | Delete `raw/`, keep `cutouts/` |

### 1.3 Requirements on the replacement product (derived from 1.2)

- R-a. One product per half per event, always (every half owns DMs in both n_sub choices: 3 or 5-6
  coarse, 79-101 fines per half; computed from `snr2x/schemes.py::proposed`).
- R-b. Full-field (all n_grid² pixels) width-1 max per (row, fine) for **every** owned fine DM, in
  the same Layer-1-normalised units as today's `peak_grid`. This drives R2/R3/R5/R10/R11, the DM-time
  panel and the global apex.
- R-c. Per-fine DM value, global fine index, coarse index, tier, and per-tier period. Rows on one
  absolute T_ext grid, with explicit validity.
- R-d. The pixel light curve at the candidate (l,m) for every fine, on every half.
- R-e. Full 256² planes: the apex plane per coarse DM (R1/R4), and the peak-fine plane averaged over
  the burst width (image panel).
- R-f. A window of at least today's 268 ms around the event, enough rows for robust statistics,
  covering the widest width (32 T_ext = 16.8 ms) several times.
- R-g. Metadata: event, half, request echo, (l,m) geometry, clock, the applied normalisation, and
  status.
- R-h. The 1x path keeps reading cubes. Consumers go through one adapter (§3.9).

---------------------------------------------------------------------------------------------------

## 2. Assumption audit (sample period, samples-as-units, 8 halves / 1 DM per half, n_fdm, DM range, csf, streams, cadence)

One line per item: `file:line` — assumption → required 2x change [class].

### 2.1 C1 stage (`services/search_compute.py` downstream of the detector, `services/c1_emit.py`, detector merger/decoder)

- `search_compute.py:266` `_DEDISP_CHAN_SUM_FACTOR=8`, never wired from `--chan-sum-factor`. → Take
  csf from the plan (2). Compute L in the units the boxcar was summed in: tier-B internal 1.05 ms
  vs T_ext differ ×2 (at DM 3000: 0.54 vs 1.07) [2x, 1x=]. Also finding 5 [both, 1x≠].
- `search_compute.py:273,302,342,403` `t_search_us` defaults 524.288; `:578-579` 524.288 paired with
  spp 16; `common/constants.py:523`. → Set explicitly per op-point; add an assertion that period/65.536
  == spp [both, 1x=].
- `search_compute.py:1371-1378` `_geom_from_slot` prefix slice; `:724-736` whole-half DM table to the
  detector; `:2982-2989,3255`. → Per-DM geometry from the plan [2x].
- `search_compute.py:2365-2404,3468-3479,2356,3172` single owner, K = N_fine/8, `--n-coarse-dm 8`,
  `owned_coarse_dm` int. → Owned list d%8==h, per-DM n_fdm from the plan (no `--n-fdm`) [2x].
- `search_compute.py:1393-1454` `mjd_start = anchor + specnum_start × cube_sample_period_us`, one
  period per process → tier-B `specnum_start` must be in T_ext [2x]. Known +5 s latch bug
  (memory: localisation): 2x should derive MJD from the capture epoch + native index [both, 1x≠].
- `search_compute.py:1541-1542,1725-1908` per-cube filter chain (noise colour → DM floor → width
  cap/escape → governor → metering → batch). → Per-cycle pooling (finding 4) [2x, 1x=].
- `search_compute.py:1794` width cap compares `width_samples` to 16 (`yaml:545`). → 32 T_ext;
  tier-B widths doubled first [2x, 1x=].
- `search_compute.py:1892-1903` C1 header: `sample_period_specnum` 16→8, `sample_period_us`
  1048.576→524.288 (SPEC §8), `n_fdm_in_cube` per cube [2x, 1x=].
- `search_compute.py:120-123` `_METER_WINDOW_BLOCKS=16` (comment "≈2.1 s"; really 3.2 s, 0.5-1.1 s at
  2x); `:2234` status_every 10 cubes. → Time-based (2 s) windows/publishes [2x; 1x cadence 2.01→2.0 s
  mon-only].
- `search_compute.py:1555-1568` proactive-stager arming per cube. → Disable in 2x, or key by (specnum,
  d) (§3.8) [2x].
- `search_compute.py:1241-1259` placeholder ring with depth 0. Every trigger would park 120 s, then
  count `too_early`. → Swap in the cutout listener in 2x [2x].
- `search_compute.py:1658-1684` ring rebuild on geometry change → depth 0 in 2x [2x].
- `search_compute.py:1084-1117` `mon_snapshot()` has no caller. → Publish `c1_cands_dropped_{color,
  dmfloor,meter}`, stager and trigger rescue counters [both, additive].
- `search_compute.py:312,482,1387,2904,3162-3166,3395-3403` comments/help: 1048.576, "37 cubes ≈ 5 s"
  [docs].
- `c1_emit.py:120-142` WideEscapeGovernor: brightest-first only within a block, tokens on wall
  clock → cycle-level admit [2x, 1x=].
- `c1_emit.py:434-440,448` `fine_dm_idx` = argmin over geometry; `dm_idx_global` = decoder `dm_idx`
  (= local fdm, finding 2). → Global fine index from the plan [both].
- `c1_emit.py:216-227,264-265,288-292,322-372` docstrings: "≥32", 64-sample overlap, 7.45 cubes/s,
  480 rows/s, DM 2330/2538 at 1048.576 [docs].
- `detector/decoder.py:202-205,216-217,375-376,523-524` `event_specnum = start + t`,
  `width = k_time_width`. → Tier-B ×2 to T_ext immediately after decode [2x].
- `detector/kernels.py:264`, `decoder.py:217` `kernel_id "unit:d1:bW"`. Tier B reports bW with
  width 2W, and `s_k_<kernel>` noise fields collide between tiers. → Add the tier to kernel_id or a
  separate field [2x].
- `detector/merger.py:27-39,166-177` window × `sample_period_specnum` (finding 16) [2x: keep 16];
  `:184-187` DM test on `dm_idx` (local, per cube) and `dm_max_trials: 2` (`yaml:538`). 2x fine
  spacing is 2.8-3.2 pc/cc in tier A (vs ~7) → retune in pc/cc. Note `injections-blocked-by-c1-metering`
  already suspects this knob [2x].
- `configs/dsart_search_rt.yaml:874` `detector.cube_cadence_s` is commented out, so Layer-2 τ uses
  134 ms instead of 201 ms (τ ≈ 45 s, not 30). `:856` "64 cubes ≈ 8.6 s" is really 12.9 s. Per-DM
  state in 2x (SPEC §7) [both, WS-S].
- `common/contracts.py:539-542,810-818` docstring formulas that divide by spp (16x error if
  followed) [docs, both].
- Tests: `tests/test_c1_metering.py:294-312,362-406,457-478`; `test_c1_merger.py:85,226-240`;
  `test_c1_emit.py:104-108,194,235-241`; `test_wide_escape_governor.py:30-101`;
  `test_search_compute_service.py:341-342,512-525`; `test_proactive_stager.py:136,200-202,241-244`;
  `test_search_ring_mon.py:158-159`. No test covers `_select_dm_owner_half`.

### 2.2 C1/C2 wire (`coinc/wire.py`) and C1→C2 transport

- `wire.py:44,264,246,296` schema 1; header exactly 11 fields, rows exactly 12, strict version.
  → No C1 schema change in 2x. The unit lives in the header `sample_period_us` (SPEC §8). Tier,
  coarse d and global fine come from the plan via `dm_idx_global` once it is truly global
  [2x, 1x=].
- `wire.py:80-90,118-130` docstrings hard-code 1048.576 and "factor 32" [docs].
- `wire.py:48-62,324-386` 64-B trigger `<IHH16sqdII16s>`; reserved u32 + pad[16] unused; the search
  listener ignores flags (finding 10). → New `CutoutRequest`/`CutoutStatus` codecs in this module (§3.2)
  [2x].
- `coinc/broadcast.py:47,133-145` `GPU_HALVES=(0,1)`, port_base+g, single `sendto`, no ack →
  ack/resend for cutout requests [2x].
- `docs/c1c2/C1C2_WIRE_SCHEMA.md:162-166` (§2.1 stale lookup), `:177-189` (§2.2 names a nonexistent
  `dsart_cube_uploader.service`) [docs].
- Tests: `tests/test_c1c2_wire.py:47-70,127,189`; `test_c2_broadcast.py:45,71-72,131`.

### 2.3 C2 (`services/coincidencer.py`, `coinc/*`)

- `coincidencer.py:733,749-765` `_SPECNUM_FACTOR_FALLBACK=16`. A malformed 2x period stages voltages at
  2x the specnum. → Derive from the op-point or skip the voltage dump [both, 1x=].
- `coincidencer.py:1652-1653` voltage specnum from the peak period: correct at 2x.
- `coincidencer.py:1552-1578` startup grace reads `/mon/corr_rt/{0..15}/corr_fast` (chgroup-indexed,
  SNAP units): op-point-free. `:429` 180 s tied to the corr RFI warmup in cubes → re-check [2x].
- `coincidencer.py:1402-1431`, `yaml:896` `window_s: 5.0`. At 2x, DM-3000 rows lag the anchor by
  ~1.7 s sweep + per-DM cube latency. Verify, or high-DM rows < 30σ land in `rows_late_drop` [2x].
- `coincidencer.py:1461-1472,1918-1926` matcher calls pass no period (finding 6) [2x, 1x=].
- `coincidencer.py:1506-1509` evaluation-rate counting (finding 9) [2x; fix is 1x≠].
- `coincidencer.py:1758,1792-1798` name allocated and C2 row `trigger=<name>` written **before** the
  vetoes, gate and rate cap; more names consumed at 2x [both, cosmetic].
- `coincidencer.py:1900-1905` dump broadcast = 64-B trigger with `peak_event_specnum` only → cutout
  request with t, DM, (l,m), width (§3.2) [2x].
- `coincidencer.py:380,598-600,2101-2195,791` the 8-file gate, rmtree, re-sweep (finding 1). → §3.5
  [2x; 1x≠ behind `partial_event_policy`].
- `coincidencer.py:2003-2010` synchronous 24 h concat on the loop [both, perf at 2x].
- `coincidencer.py:2398-2434,486`; `criteria.py:294` no `--opoint`; reads the YAML file
  (`systemd/dsart_c2.service:12`), top-level `coinc:` / `trigger_classes:` only. → Op-point section
  selection, consistent on hot reload (SPEC §10) [2x].
- `coinc/components.py:53-56` time edge `|dt| ≤ (w_a P_a + w_b P_b)/2`. Per-row period, so correct, but
  narrower in seconds at 2x for w=1. Narrow bright bursts split into more clusters across DM, which
  feeds the DM-comb veto [2x].
- `coinc/stats.py:112-115,161-163,174` widths raw; `width_median` over mixed tiers is meaningless
  unless rows carry T_ext widths (they will, SPEC §8). `:147-148` `n_search_nodes` / `n_gpu_halves`:
  at 2x bright bursts span more halves (interleaved). All classes use `n_search_nodes_min: 1` today;
  a ≥ 2 criterion would become a DM-breadth filter (trap) [2x].
- `coinc/criteria.py:157-161,217-218`; `c2_trigger_criteria.yaml:61,104,127` widths 16/8/32 samples
  (finding 7) → `width_median_max_ms` [2x, 1x=].
- `c2_trigger_criteria.yaml:52,101` `dm_median_min 115` (= floor 100 + 15). At 2x the floor is 50 and
  the zero-DM rail sits at 50, so a rail at 50 with gal_dm < 66.7 reads as extragalactic (memory
  `zero-dm-rail-at-search-floor`) → floor + margin from the plan [2x].
- `c2_trigger_criteria.yaml:59,102` `dm_median_max 2700` → ≥ 3000 [2x]. `:125` `bright_galactic`
  floor 100 [2x].
- `c2_trigger_criteria.yaml:60,103,126,138` `dm_iqr` caps in pc/cc, tuned to ~7 pc/cc fine spacing →
  retune on 2x replays [2x].
- `c2_trigger_criteria.yaml:129` vs `:7-8` duplicate-dump hole (finding 8) [both, 1x≠]. `:115` "64-cube
  dump" stale.
- Hot reload: mtime check once per non-empty batch (`coincidencer.py:1451`; not per cluster as
  `criteria.py:61-63` says) + SIGHUP force (`:2375-2390`, `criteria.py:362-387`). A bad file keeps
  the last good set; a bad file at start raises (`coincidencer.py:809`). Holdoff state is keyed by
  class name and survives reloads. C2 reads the h23 checkout live: **editing the file is a deploy**.
- `yaml:1019-1023` DM comb (3 clusters, 300 pc/cc span, 2 s, 90″) with no S/N exemption
  (`coincidencer.py:1826`; `coinc/veto.py:97-113`) → retune with 2x injections (risk: vetoing real
  bright bursts) [2x].
- `yaml:1012` `cluster_rate_max 400`, `:998-999` dump rate 6/60 s (physical, fine), `:905` ingest
  depth 512 [2x].
- `coinc/veto.py:8` "8 search nodes" [docs]; sidereal (l,m) 90″ assumes 45″ pixels; unchanged image →
  fine.
- `coinc/archive.py:68-87,91-117,373-426` CSV/Level3 lack period, `dm_peak`, `width_peak`,
  `peak_event_specnum` (CSV), op-point → append (§3.5) [2x, 1x≠ only if 1x gets the columns; keep 1x
  files unchanged].
- `coinc/archive.py:53-61,240-263` `create()` makes `cubes/`, `voltages/` (unused; C3 uses
  `Level2/voltages/`). 2x: `cutouts/`, `raw/`.
- `tools/c2/sanity_check.py:43` hard-codes spp 16, 1048.576, n_fdm 34, width 4; its `event_specnum =
  time.time()` vs MJD 60781 sets a fresh C2's window anchor wrongly [both, tool].
- Tests pinning: `test_c2_criteria_faint.py:17,43-65,95-96` and `test_c2_criteria_bright.py:19,42-72`
  load the **production** criteria file; `test_c2_criteria.py:278-284`; `test_dumps_gate.py:623-630,660-719`
  (discard + re-sweep); `test_coincidencer_voltage_gate.py:111-114,149-150`;
  `test_voltage_dump_containment.py:78-86,109-111`; fixtures with 1048.576/16/34:
  `test_c1c2_wire.py:47-70`, `test_c2_receiver.py:179`, `test_c2_stats.py:44`,
  `test_c2_archive.py:42`, `test_c2_window.py:21`, `test_c2_components.py:18`,
  `test_c2_trigger_listener_retry.py:92-93`, `test_c2_proactive_claim.py:67-68`.

### 2.4 C3 (`services/c3.py`, `coinc/cube_veto.py`, `coinc/filterbank.py`, `services/slack_notify.py`)

- `cube_veto.py:62` `r2_t_shift_samples 30` ("31 ms") → `r2_t_shift_ms 31.5` (60 T_ext) [2x, 1x=].
- `cube_veto.py:63,66` R3/R5 in fine trials. At 2x the bucket holds 15-35 trials of 2.8-3.2 (A) /
  5.8-6.3 (B) pc/cc vs 25 × ~7 today → keep trials per bucket, recalibrate on 2x injections
  (shadow veto, `c3.py:350-365`, gives the evidence) [2x].
- `cube_veto.py:86` R10 `tz_trig ≥ 8`: robust z over ~256 rows of 1.05 ms. At 2x the trigger light
  curve has 512 tier-A rows and narrow bursts have higher per-sample S/N → recalibrate; compute on
  independent rows only [2x].
- `cube_veto.py:105-117` R11 (off), `r11_min_width_samples 8` → ms; ac1 on independent rows [2x].
- `cube_veto.py:345-369` `live_span` (64-row zero overlap) → `row_valid` [2x].
- `cube_veto.py:495` `dm_edge` = cube edge = coarse-bucket edge. At 2x use the bucket of the trigger's
  coarse d (same semantics) [2x].
- `cube_veto.py:589-596,638-651,665-670` file discovery/IO on `cubes/` → adapter (§3.9) [2x, 1x=].
- `c3.py:104,172` `expected_cube_count` dead config; `c3.py:295-312` discovery by Level3 (finding 1);
  `c3.py:632-642` REJECT deletes `cubes/*.npz` → in 2x delete `raw/`, keep `cutouts/` [2x].
- `c3.py:148-152` only R1-R5/R10 thresholds are plumbed from YAML (`yaml:1127-1142`); R11 is not [both,
  minor].
- `filterbank.py:52,248-250` `T_INT_SEARCH_US = 1048.576` × `width_median` → Level3 period [2x, 1x=].
  `:180-181` `N_PRE_BLOCKS 14` fallback only (manifest authoritative).
- `tests/test_voltage_dump_containment.py:43-75` reads `--n-pre/--n-post/--dm-plan-path` from
  `dsart_pipeline_rt.yaml` and `d["fine_dm"].max()` from the plan npz. DM-3000 sweep 1.698 s vs
  n_pre 14 → passes (14 × 0.134 = 1.879 ≥ 1.832 s, 47 ms margin). The test must read the v3 tiered
  plan's max fine DM (key layout from WS-P) [2x].
- `slack_notify.py:87,353-396` `expected_cubes 8`, globs `*.npz` → product status from Level3 [2x].
- Tests: `test_cube_veto_decide.py:24-25,305-332` (256×34, 192 live), `test_c3_filterbank.py:105`
  (1048.576), `test_c3_slack_notify.py:904,950-953`.

### 2.5 Injection (`inject/*`, `coinc/inject_match.py`, dashboard, op)

- `inject/online.py:409-450,772-774,816` delay model: 4.148808·DM·(ν⁻²−ν_top⁻²) per **native** channel,
  rounded to native samples; no csf/t_int/n_sub/stage-1/2 model. → **No change**: the search's own 2x
  rounding is exercised. Fix the stale comment `:765-771`. Caveat: no intra-native-channel smear
  (0.18 ms at DM 1600, 0.34 ms at DM 3000, band bottom), so narrow high-DM injections read slightly
  optimistic at 2x [2x, note].
- `inject/bot.py:117` `NATIVE_PER_SEARCH_SAMPLE=32`, `:123` `MAX_C1C2_WIDTH_SEARCH_SAMPLES=16`: module
  constants, not drift-tested against `yaml:545`. The native cap is 512 at both op-points (16×32 = 32×16).
  Changing one alone breaks it → derive both from the `c1` block the bot already loads
  (`systemd/dsart_inject_bot.service:14`, `bot.py:426`) [2x, 1x=].
- `bot.py:154`, `yaml:1251` width choices (4..512): no 16-native shot (1 T_ext) → add 16 in 2x [2x].
- `bot.py:275,365`, `yaml:1227` `dm_choices` → to 3000, avoid 1575-1625 (bucket `dm1600` straddles
  tiers, `dashboard/inject_calibration.py:103,226-227`) [2x].
- `bot.py:158-160,575-628`; `inject_calibration.py:1111-1185` ANY-half metering gate, 8 halves, two
  copies; no owner-map reader (owner map is only argv + plan). → Owner map from the plan; gate on the
  owner halves of the shot's DM behind a flag (default ANY = today) [2x, 1x≠ if the flag is flipped].
- `bot.py:700,757-782,871-933,893` K provenance / recal guard / K doctor (finding 17). → op-point-tagged
  bucket keys in 2x (`dm0500@w0004#2x`-style; bare keys stay 1x), wipe state file + etcd prefix
  [2x, 1x=].
- `inject_calibration.py:123,988,997,1012`, `templates/control.html:889-890` empirical 1x fluences and
  `IMAGER_SAFE_OBSERVED_SNR 30` → re-measure in 2x (sub-sample bursts gain √2 S/N); `:1021` 240 tracks
  the 250 clip, unchanged [2x].
- `inject_calibration.py:773-785`; `inject_match.py:1097-1108` first published match wins; best =
  closest (l,m) then S/N, no owner/DM proximity. At 2x, neighbour halves report the same burst, so K can
  come from a weaker neighbour → prefer the owner half of the nearest fine DM, settle ~2 cycles [2x].
- `inject_match.py:155` DM tol 5% (2.5 pc/cc at DM 50 vs 2.8 step) → check; `:201,207,242,993-996` time
  gate (finding 6); `:236` wall tol 28 s → re-measure [2x].
- `tools/inject/analyse_campaign.py:118-129,222-228` text hard-codes 32.768/1048.576 [2x, text].
- `control_store.py:617-619,629` comment ±4096 ≈ 4.3 s; `INJECT_LM_MAX_RAD 0.0279` from n_grid/pixel
  scale (unchanged image) [none]. `templates/control.html:960` shows the margin at 32.768 µs per specnum
  (should be 65.536) [both, display].
- Stale docs: "K width-independent" (`inject_calibration.py:32-36,312-314`, `app.py:1622,1824`); √
  model (`inject_match.py:10-12,87-96`); "never writes etcd" (`bot.py:52-54`, `yaml:1197`).
- op: `monitor/injection.py:28-35,78-84,125-126`, `control/verbs.py:119-133`, `config/policy.yaml:69-73,
  126-140` (finding 18) [both]; `tools/readonly.py:28-29,331-336` 8 halves (invariant).
- Tests: `test_inject_bot.py:98,692-706,724,735-740,753,794-802`; `test_inject_match.py:460-462,699-830`;
  `test_dsa_monitor_inject_calibration.py:712,748`; op `tests/test_executors.py:70-74`,
  `tests/test_control_engine.py:174`.

### 2.6 Monitoring, HiPlot, Grafana, dashboard health, operator

- Pusher `tools/dashboard/dsart_rt_to_influx/pusher.py:180-185` `COARSE_DM_OWNER` (1x map), tag
  `coarse_dm` single-valued (`:467-476`, search_rt_* measurements) → owners from the plan; per-DM
  measurement (§4) [2x, 1x=].
- `pusher.py:97-104` prefixes (no `/mon/c3/`), `:1300-1303` C2 allow-list drops `cluster_rate`,
  `ingest_queue_*`, `voltages_*`; `:1661-1673` drops `/mon/search_rt/<cn>/rx` (pinned by
  `tests/test_dsart_rt_to_influx_pusher.py:851-870`); `:1675-1678` corr_fast key unknown; `:111`
  `meridian_spl_ready` dropped; `:93` schema 1 [both].
- Index trap: `/mon/corr_rt/<chgroup>/corr_fast` (`corr_fast_mon.py:65`) vs `/mon/corr_rt/<cn>/*`
  everywhere else. Ingesting corr_fast with `_host_for_cn` (`pusher.py:210-219`) would mis-tag hosts →
  add `cn_id`/`chgroup` fields [both].
- Grafana `tools/dashboard/dsart_rt_to_influx/grafana/build_dashboard.py:355-370` (8 halves,
  1.048576e-3, stride 192, nominal 4.96707), `:465-487` ingest ×1.048576e-3, `:550-575` noise
  agreement across 8 identical halves, `:592,605-608,611,735-755,986,1256-1269,1371,1392,1580,1666-1668,
  1682-1686` literals; no corr block-time or search stage panels; committed JSON stale (no 10-05
  governor panels) → op-point-aware builder; nominal per half = n_dm(h)/0.201326592 s [2x, 1x=
  (JSON identical at 1x)].
- HiPlot `systemd/hiplot_c1.service`, `hiplot_c2.service`: stock `hiplot` over the CSV dirs (ports
  5017/5027), no repo code; columns from `archive.py:68-117` → append `width_ms`, `sample_period_us`,
  `tier`, `opoint` [2x].
- `services/search_compute_mon.py:24`, `search_ring_mon.py:57` docstrings 7.45 cubes/s;
  `search_ring_mon.py:180-184` `_end` bug (finding 14) [both].
- `corr_fast_integration.py:4322,4602-4637,4651-4652,4059-4079,4304-4316` block time single sample,
  p50/p99 only at exit (unbounded list), publisher gated on `--inject-watch`, stage profiler
  log-only and refused with `--pipeliner-3s` [both; WS-C].
- Dashboard: `control_store.py:69-80`, `services_inventory.py:110-117,206-221` (no
  `dsart_rt_to_influx`, no `hiplot_c1`), `corr_topology.py:32-38`, `fleet_update.py:67-82` (no h20),
  `cube_dump_now.py:74,91-92,120` hard-coded topology/cadence; `control_store.py:56-57` says 1 specnum =
  32.768 µs (wrong; `:637-643` uses 65.536) [both].
- `/control/system_state` (`app.py:1415-1433`, `control_store.py:398-605`) never checks that search
  halves process data; banner shows counts only (`control.html:2890-2896`) → op-point banner + RT health
  page (§4) [2x].
- op: `tools/readonly.py:83-84` heartbeat alive = any non-empty dict (dsart writes without lease,
  `dsart_rt.py:1463`) so dead nodes stay alive; `monitor/health.py:28-34,128-228,543` search health
  never fed into `evaluate_health`; hard-coded 16/4 [both].
- Tests: `test_dsart_rt_to_influx_pusher.py:851-870`, `test_search_ring_mon.py:158-161`,
  `test_cube_dump_now.py:168,177,357,370,391,515,563,657,660,717-799` (8 halves, t_det 192 — stale vs
  256), `test_fleet_services.py:144,154,751`, `test_c2_plotter.py:41,122-141,180,542-546`,
  `test_candidate_card.py:429-437` ("16.8 ms (16 samp)"), `test_c2_cands_panel.py:64-65,115`.

### 2.7 Visualisation

- `coinc/plotter.py:207-210,243-323` plan path + contiguous owners (finding 12) [2x; fallback label
  bug both]; `:1571-1588,1574-1577,1680-1691` stack order = DM order, common-T crop, linear ticks [2x];
  `:1705,1922` x-axis in samples → ms [2x, 1x≠ for PNGs unless gated to the cutout path];
  `:930-939,1819-1826` width in cube samples; `:1838-1843` reticle radius 8 px (unchanged image) [2x];
  `:106-112` stale distrust of `peak_grid` (writer is consistent, `cube_dump.py:729`) — use it [both, perf].
- `services/candidate_card.py:61-70,283-295,355` width × 1.048576 → Level3 period [2x, 1x=].
- `tools/viz/m6_cube_dump_verifier.py:163-166` strict 4-D shape; notebook stale plan
  (`dm_plan_N8_dmmin100_tol1.6_v2`) [dev].

---------------------------------------------------------------------------------------------------

## 3. Cutout interface (frozen; copied to SPEC §9)

### 3.1 Units, clock and DM indexing

- **Rows.** A cutout row r is the absolute T_ext index `t0_ext + r`, with `t_ext = native // native_per_ext`.
  - `native_per_ext` is 16 at 2x and 32 at 1x. `t_ext_us` is 524.288 / 1048.576.
  - The native epoch is the capture specnum epoch: native index = 2 × SNAP specnum.
  - C1 `event_specnum`, the request, the cutout and Level3 all use this one index.
- **Time reference.** A row's time is the arrival time at `NU_BOT_PROC_GHZ` (1.31128 GHz) after
  dedispersion at that row's fine DM. This is the same as today's event label and the Level3
  `t_peak_mjd` convention localisation relies on.
- **Tier B.** A tier-B sample covers rows 2k and 2k+1 (cycles are aligned to absolute specnum,
  SPEC §7). `t0_ext` is always even. Tier-B values are written to both rows. **The independent samples
  of fine f are rows r with r % t_step_ext[f] == 0.** Statistics (robust z, ac1, frac_z5) must use
  only those rows.
- **Widths.** Widths are integers in T_ext samples; tier B is even. A FWHM in ms, if B-splines are
  adopted, travels as a separate float field and never replaces the integer.
- **Global fine index.** Fines are numbered in plan order: tier A ascending, then tier B ascending
  (WS-P must define this index in `TieredDmPlan`). The C1 row's `dm_idx_global` must carry it in 2x;
  today it is local (finding 2). The C1 row's `fine_dm_idx` is the index within its coarse bucket.
  Coarse d is global (tier A first); the owner half is `d % 8`, and half h is node `N_SEARCH[h//2]`,
  GPU `h%2`.
- **MJD.** `mjd_t0` is the MJD of row 0 on the same clock as the C1 batch `mjd_start`, so Level3 and
  cutout times agree. `mjd_clock` names the source. Recommendation (WS-S): derive it from the capture
  epoch plus the native index, which removes the latched-clock offset of several seconds.

### 3.2 Request: C2 → every half (`coinc/wire.py`, new `CutoutRequest`)

- **Transport.** UDP to `(coinc.dump_broadcast.hosts[sid], coinc.dump_broadcast.port_base + g)`. These
  are the same keys, hosts and ports as today's DSRT trigger: 11227 / 11228 on the search NIC, bound by
  `c1.dump_listener.{bind_host,base_port}`.
- **Packet.** Exactly 128 bytes, little-endian, packed, `struct "<IHH16sIHHqqIIIHHffffihhhHfdQH14s"`.
  Old 1x listeners drop it on size (`c2_trigger_listener.py:379`); the corr voltage listener never
  receives it.

| off | type | field | notes |
|---|---|---|---|
| 0 | u32 | magic | `b"DSCQ"` = 0x51435344 |
| 4 | u16 | version | 1 |
| 6 | u16 | flags | bit0 CUTOUT, bit1 RAW_ARCHIVE, bit2 INJECTION, bit3 MANUAL (Dump Now), bit4 RESEND |
| 8 | 16s | event_name | NUL-padded ASCII |
| 24 | u32 | request_id | C2-monotonic; (event_name, request_id) is the idempotency key |
| 28 | u16 | trigger_class_id | crc32(class name) & 0xFFFF (stable) |
| 30 | u16 | reserved0 | 0 |
| 32 | i64 | t_peak_ext | peak member `event_specnum` (§3.1) |
| 40 | i64 | t0_ext | cutout row 0; even; default `t_peak − rows/2` rounded down to even |
| 48 | u32 | n_t | cutout rows; even; default 512 (268.4 ms) |
| 52 | u32 | raw_pre_ext | raw-window rows before t_peak (default 954 ≈ 0.5 s) |
| 56 | u32 | raw_post_ext | rows after (default 954) |
| 60 | u16 | width_ext | peak member width (T_ext) |
| 62 | u16 | stamp_half_px | S; stamps are (2S+1)²; default 16; 0 = none |
| 64 | f32 | dm_peak_pc_cc | peak member DM; < 0 = none (MANUAL) |
| 68 | f32 | snr_peak | |
| 72 | f32 | l_rad | |
| 76 | f32 | m_rad | |
| 80 | i32 | fine_global_peak | −1 = unknown |
| 84 | i16 | coarse_global_peak | −1 = unknown |
| 86 | i16 | l_pix | −1 = no candidate pixel (MANUAL) |
| 88 | i16 | m_pix | |
| 90 | u16 | n_stamp_t | stamp rows centred on t_peak; default 64 |
| 92 | f32 | stamp_dm_halfwidth_pc_cc | stamps for fines with \|DM − dm_peak\| ≤ this; default 150 |
| 96 | f64 | mjd_peak | informational |
| 104 | u64 | plan_id | blake2b-64 of the TieredDmPlan; mismatch → FAILED(PLAN_MISMATCH) |
| 112 | u16 | deadline_s | the half's maximum wait for the window to be complete; default 15 |
| 114 | 14s | reserved1 | zero |

- **Recipients.** C2 sends one request to every half in the expected set (§3.5), with the same
  request_id.
- **Raw archive.** RAW_ARCHIVE is set for sky events. It is cleared for INJECTION events by default
  (`coinc.cutout.raw_for_injections: false`), because injections are roughly half of all dumps and are
  re-creatable.
- **Ignored packets.** A 2x half that receives a v1 DSRT DUMP_CUBE packet counts it
  (`c2_trigger_v1_ignored`) and ignores it, because its units are ambiguous.

### 3.3 Status: each half → C2 (`coinc/wire.py`, new `CutoutStatus`)

- **Transport.** UDP to `(c1.c2_endpoint.host, coinc.cutout.status_port = 11501)`. C2 binds
  `coinc.cutout.status_bind` (default 0.0.0.0:11501).
- **Packet.** 64 bytes, `struct "<IHBB16sIBBBBfIQIh10s"`.

| off | type | field |
|---|---|---|
| 0 | u32 | magic `b"DSCS"` = 0x53435344 |
| 4 | u16 | version 1 |
| 6 | u8 | state: 1 ACCEPTED, 2 DONE, 3 PARTIAL, 4 FAILED |
| 7 | u8 | reason: 0 OK, 1 TOO_LATE, 2 WINDOW_TIMEOUT, 3 QUEUE_FULL, 4 PLAN_MISMATCH, 5 BAD_REQUEST, 6 GPU_ERROR, 7 IO_ERROR, 8 UPLOAD_FAILED, 9 SHUTTING_DOWN |
| 8 | 16s | event_name |
| 24 | u32 | request_id |
| 28 | u8 ×4 | search_node_id, gpu_half, n_dm_owned, n_dm_ok |
| 32 | f32 | valid_frac (valid (stream,row) inputs over the window) |
| 36 | u32 | cutout_bytes |
| 40 | u64 | raw_bytes |
| 48 | u32 | latency_ms (request receipt → this state) |
| 52 | i16 | upload_rc (cutout upload; −1 n/a) |
| 54 | 10s | reserved |

- **When it is sent.** ACCEPTED on receipt. Then exactly one of DONE / PARTIAL / FAILED, after the
  cutout upload returns.
- **Duplicates.** A duplicate request (RESEND, or a repeated key) re-sends the current state and does no
  new work.
- **Not authoritative.** Status is lossy and is used only for resends, early completion and logs. **The
  authoritative signal is the per-half manifest file on h23 (§3.5).**

### 3.4 Half-side behaviour and timeouts (WS-S implements; WS-D supplies codecs, writer and uploader)

1. **Validate.** Check size, magic, version, plan_id, `n_t ≤ 4096` and raw window ≤ 4 s.
2. **Enqueue.** Queue depth `c1.cutout.queue_depth` (default 8); full → FAILED/QUEUE_FULL. MANUAL
   requests are served after triggered ones. Send ACCEPTED.
3. **Raw snapshot (if RAW_ARCHIVE).**
   - For each owned coarse d, wait until the ring's per-d frontier covers every (stream, d) sample that
     contributes to any of d's fines at any row of `[t_peak − raw_pre, t_peak + raw_post)`, including
     the stage-2 extents. Stop waiting at `deadline_s` (window incomplete → zero-filled, `valid=0`).
   - Stream the slabs from the shm ring to `raw_s<sid>_g<g>/raw_d<DDD>.npz` (layout in §3.8).
   - If a needed start sample has already rotated out, that d is TOO_LATE; write what exists.
   - Budget: the request arrives ≤ ~10 s after the burst (measured C2 round trip 7.8-9 s, memory
     2026-07-21). Ring retention ≥ 20 s (plan) leaves about 8 s for the write (0.55-0.82 GB per half,
     ~1 s on NVMe).
4. **Cutout.**
   - Re-image `[t0, t0+n_t)` for every owned fine on the low-priority CUDA stream, one job at a time.
     Read from the raw snapshot when one was written (so the cutout can be reproduced offline from the
     archive), else directly from the ring.
   - Normalise with the Layer-1 sigma that was applied at detection time. The half keeps a ≥ 30 s
     history of the per-(cycle, fine) applied sigma.
   - Cost: ≈ 40-50k images per half ≈ 0.1-0.17 s GPU at 2.4-3.5 µs per image. At the dump cap of
     6/min that is ≤ 1.7% duty.
5. **Write and upload.**
   - Write `cutouts/cutout_s<sid>_g<g>.npz`, then `cutouts/cutout_s<sid>_g<g>.json` (the manifest),
     both atomically (tmp + fsync + replace).
   - Upload the cutout first (its own uploader, no bandwidth limit).
   - Then upload the raw window (second uploader, `bwlimit` default 50 MB/s per half, so that 8
     concurrent ~0.8 GB uploads do not saturate h23 ingress or corr-net; T5 2026-06-07 lesson).
   - Each half purges only its own files after rc==0, and writes `UPLOAD_OK` into the per-product
     subdirectory's `upload.log`.
6. **Report.** Send DONE / PARTIAL / FAILED. PARTIAL means some d was TOO_LATE, timed out or had
   `valid_frac < 1`.

### 3.5 C2-side event lifecycle (no event is ever discarded)

1. **Fire (criteria and vetoes unchanged).**
   - Create `<archive>/<event>/` with `Level2/C2_<ev>.csv` and `Level2/C1_window_<ev>.csv` immediately,
     so a C2 crash cannot lose the record.
   - Record the pending event in `<archive>/<event>/Level2/products_pending.json`. It is persisted so
     that a C2 restart resumes; today the pending set lives only in memory.
2. **Request.**
   - The expected set is the halves that own ≥ 1 coarse DM in the deployed plan's owner map, i.e. all
     8. It comes from the plan, not from a literal.
   - Send requests to all expected halves. Resend (RESEND flag, same request_id) to any half with no
     ACCEPTED after `ack_timeout_s` (2.0), up to `max_resends` (3).
3. **Collect.**
   - Poll `<archive>/<event>/cutouts/cutout_s*_g*.json` every `plot_dispatch_poll_s` (5 s).
   - Count **distinct (sid, g) manifests**, not files.
   - The event is complete when every expected half has a manifest. At `coinc.cutout.wait_s` (default
     120 s; cutouts are ~5-10 MB, so expect < 30 s) it is finalised as **incomplete** with whatever has
     arrived.
4. **Finalise.**
   - Write `Level3/<ev>.json` with `schema_version: 2` and a `products` block:
     - `kind` ("cutouts" | "cubes"), `opoint`, `plan_id`;
     - `expected`, `present` and `missing` halves (each missing half with its reason from the last
       status, or "no_status");
     - `status` ("complete" | "incomplete"), `finalised_unix`;
     - `raw {requested, expected, present, bytes}`.
   - Add Level3 `c2.` fields: `sample_period_us`, `dm_peak`, `width_peak`, `width_median_ms`,
     `peak_event_specnum`, `peak_fine_global_idx`, `peak_coarse_idx`, `peak_tier`.
   - Queue the plot job (placeholders for missing DM ranges). C3 then sees the event as today.
5. **Late arrivals.**
   - A manifest that lands after finalisation within `late_accept_s` (3600) is appended to
     `products.late`. Plots can be regenerated; C3 is not re-run.
   - Raw windows are never waited for. They are tracked in `products.raw` by a later sweep that reads
     `raw/raw_s*_g*/RAW_OK.json`.
6. **Never `rmtree`.** The `events_incomplete_discarded` counter stays (for 1x Grafana continuity)
   and new counters are added (§4).
   - **1x:** guarded by `coinc.partial_event_policy: discard|archive`. 1x keeps `discard` (bit-identical)
     until the user approves switching it at O2. That would fix the ~7% sky-candidate loss in 1x too.
7. **CSV columns.**
   - Appended at the end of both CSVs: C1 window `sample_period_us, coarse_idx, tier`; C2
     `dm_peak, width_peak, peak_sample_period_us, peak_event_specnum, width_median_ms, opoint`.
   - The rolling CSV starts a new file whenever its header changes.
   - All of these are 2x-only, so 1x files are unchanged.

### 3.6 Cutout file: `cutout_s<sid>_g<g>.npz` (schema `dsart.cutout` v1)

**File format and the per-file axes.**
- Uncompressed `np.savez` (ZIP_STORED, so arrays can be memory-mapped like today's cubes),
  `allow_pickle=False`-safe, and written atomically.
- F = Σ n_fdm(d) over the half's owned coarse DMs, ordered by global d and then by fine; 79-101 at 2x.
- D = the number of owned coarse DMs: 3 for n_sub 4, 5-6 for n_sub 2.
- **Units:** "image units" means Layer-1-normalised image intensity, exactly as in today's `cube`.

**A. Scalars (0-d arrays).**

| key | dtype | content |
|---|---|---|
| `schema`, `schema_version` | U, i32 | `"dsart.cutout"`, 1 |
| `event_name`, `request_id`, `request_flags` | U, i64, i32 | echo |
| `opoint`, `plan_id`, `pattern_id`, `code_version` | U, u64, u64, U | provenance |
| `search_node_id`, `gpu_half`, `half_index` | i32 | `half_index = 2·node_order + g` (0..7) |
| `t_ext_us`, `native_per_ext`, `sample_period_specnum` | f64, i32, i32 | 524.288 / 16 / 8 (1x: 1048.576 / 32 / 16) |
| `t0_ext`, `n_t`, `mjd_t0`, `mjd_clock` | i64, i32, f64, U | §3.1 |
| `t_peak_ext`, `width_ext_req`, `dm_req`, `fine_global_req`, `coarse_global_req`, `l_pix_req`, `m_pix_req` | i64, i32, f32, i32, i32, i32, i32 | echo of the request |
| `n_grid`, `cell_l_rad`, `cell_m_rad`, `l0_rad`, `m0_rad` | i32, f64 ×4 | `CubeGeometry` pixel→rad (`common/contracts.py:549-554`) |
| `status`, `status_json` | U, U | complete/partial/failed; JSON with per-d state, reasons, latencies, valid fractions |

**B. Fine axis (F).**

| key | dtype | content |
|---|---|---|
| `fine_dm_pc_cc` | f32[F] | DM of each fine |
| `fine_global_idx` | i32[F] | plan-global fine index (= C1 `dm_idx_global` in 2x) |
| `coarse_global_idx`, `fine_local_idx` | i16[F] | coarse d; index within d's bucket (= C1 `fine_dm_idx` in 2x) |
| `tier`, `t_step_ext` | u8[F] | 0 = A, 1 = B; 1 or 2 |
| `input_valid_frac` | f32[F] | fraction of contributing (stream, sample) inputs present |

**C. Coarse axis (D).**

| key | dtype | content |
|---|---|---|
| `coarse_owned` | i16[D] | global d ascending |
| `fine_offset` | i32[D+1] | the fines of `coarse_owned[k]` are `[fine_offset[k], fine_offset[k+1])` |
| `row_valid` | u8[n_t, D] | 1 when every input of that row was present (replaces `live_span`) |
| `apex_t`, `apex_f` | i32[D] | apex of each coarse block: `np.unravel_index(np.nanargmax(peak_grid[:, fo[k]:fo[k+1]]))` over ALL rows, i.e. **exactly C3's definition** (`cube_veto.py:488`); `apex_f` is local to the block |
| `apex_image` | f16[D, n_grid, n_grid] | full image at (apex_t[k], apex_f[k]) → R1 / R4 |

**D. Time × fine planes (n_t × F).**

| key | dtype | content |
|---|---|---|
| `peak_grid` | f32[n_t, F] | max over all n_grid² pixels of the width-1 image; **identical definition to today's NPZ `peak_grid`** (`cube_dump.py:729-730`) |
| `peak_pix` | i32[n_t, F] | flat pixel index `l·n_grid + m` of that maximum |
| `pix_lc` | f32[n_t, F] | image value at (l_pix_req, m_pix_req); zeros if l_pix_req < 0 |
| `sigma_applied` | f32[n_t, F] | Layer-1 sigma applied to each row (image units = raw / sigma) |

**E. Images near the candidate.**

| key | dtype | content |
|---|---|---|
| `has_peak_image` | u8 | 1 on the half that owns `coarse_global_req`, or the half's own apex coarse block for MANUAL |
| `peak_image` | f16[n_grid, n_grid] | full plane at (t_peak, fine_global_req) |
| `peak_image_w` | f16[n_grid, n_grid] | mean over rows `[t_peak − ⌊w/2⌋, t_peak − ⌊w/2⌋ + w)` at fine_global_req, w = width_ext_req (plotter image panel, `plotter.py:1819-1826`) |
| `stamp_fine` | i32[F_s] | indices into the fine axis of the fines within ±`stamp_dm_halfwidth` of dm_req |
| `stamp_t0_ext` | i64 | |
| `stamp` | f16[n_stamp_t, F_s, 2S+1, 2S+1] | centred on (l_pix_req, m_pix_req), clipped at the field edge (out-of-field pixels = NaN) |

**F. Optional detector context (filled when available, otherwise absent).**

| key | dtype | content |
|---|---|---|
| `kernel_ids` | U[K] | |
| `sigma_k` | f32[D, K] | Layer-2 sigma per coarse block and kernel, at t_peak |

**Size per half** (n_t 512, F ≤ 101, D ≤ 6, S 16, n_stamp_t 64):
- the four planes ≈ 0.83 MB;
- apex images ≤ 0.79 MB;
- peak images 0.26 MB (owner half only);
- stamps ≤ 64 × F_s × 1089 × 2 B. Across the event F_s totals ≈ 100 fines for ±150 pc/cc in tier A,
  i.e. ≈ 14 MB per event in total.

That makes about 2-4 MB per half and ≈ 20-30 MB per event, against 6.7 GB of cubes today.

**Sidecar `cutout_s<sid>_g<g>.json`:** every scalar from A, plus per-d status, file size, blake2b of the
NPZ, `t_written_unix`, the uploader version, and the same for the raw window if one was written.

**Dump Now / MANUAL** (`l_pix < 0`, `dm_peak < 0`):
- `pix_lc` and the stamps are omitted (zero-size arrays);
- the peak image comes from the half's own highest apex;
- `t_peak` is chosen from the new `/mon/search_rt/<cn>/cutout/<g>` raw-ring window (§4).

### 3.7 Coverage: every consumer in §1.2 against the cutout

| consumer | fields used | 1x→2x semantic change |
|---|---|---|
| C2 completion | `*.json` manifests, `status` | Counts halves, not files; never discards |
| C3 R2/R3/R5/R10 | `peak_grid[:, fo[k]:fo[k+1]]` for k = trigger coarse, `fine_local_idx`, `row_valid`, `t_step_ext` | Trigger plane = trigger coarse bucket (15-35 trials vs 25); independent rows; thresholds R2 in ms, R3/R5/R10 recalibrated (shadow veto) |
| C3 R1/R4 | `apex_t`, `apex_f`, `apex_image[k]` | C3 checks that its own apex equals `apex_t`/`apex_f`; on a mismatch it falls back to image metrics = 0 (fail-safe KEEP, as today) |
| C3 R11 / global apex / dead tail | `peak_grid` per coarse, `row_valid` | Per coarse d instead of per half |
| C3 trigger row | C1-window CSV `dm_idx_global` → `fine_global_idx`; `coarse_idx` column | Global index, so half and coarse are recovered from the plan |
| plotter DM-time | every half's `peak_grid`, `fine_dm_pc_cc`, `t_step_ext`, `t0_ext`, `t_ext_us` | Rows sorted by DM across halves; physical-time axis (ms); dashed lines at coarse/tier boundaries |
| plotter light curve | `pix_lc[:, f_peak]`, `peak_grid[:, f_peak]` | ms axis; boxcar width in T_ext |
| plotter image panel | `peak_image_w` (+ `l_pix_req`, `m_pix_req`, `n_grid`) | Unchanged geometry |
| plotter burst selection | `fine_global_idx`, `t0_ext` | `t = event_specnum − t0_ext` |
| card, Slack, dashboard counts | Level3 `products`, `c2.sample_period_us`, `width_median_ms` | "cutouts n/8"; ms widths |
| Dump Now | MANUAL request + `/mon/.../cutout/<g>` window | Per-half selection possible |
| notebook / analysis | `pix_lc` (pixel DM-time across all halves), `stamp`; anything else comes from the offline re-imager on the raw window | — |
| localisation | Level3 only | `t_peak_mjd` reference unchanged |

### 3.8 Raw-stream window archive

**Layout on the half and on h23:**

```
<dump_root>/<event>/raw_s<sid>_g<g>/           ->  <archive>/<event>/raw/raw_s<sid>_g<g>/
    raw_d<DDD>.npz      one per owned coarse DM (uncompressed)
    meta.json           request, plan_id, pattern_id, op-point, stage-2 extents per d,
                        sigma_applied history covering the window, code sha, per-file size + blake2b
    RAW_OK.json         written last, after every slab file is complete
    upload.log          per-half, per-product UPLOAD_OK marker (fixes the shared-marker ambiguity in §1.1)
<dump_root>/<event>/cutouts/cutout_s<sid>_g<g>.{npz,json}  ->  <archive>/<event>/cutouts/
```

**`raw_d<DDD>.npz` contents.** The ring v4 bytes are stored verbatim, layout `[stream][t][bytes]` (SPEC §6):

| key | dtype | content |
|---|---|---|
| `codes` | u8[n_stream, n_rows_d, bytes_per_sample] | Heavy int8 pairs, then the packed main codes, exactly as in the frame v2 payload (SPEC §5) |
| `step_main`, `step_heavy` | f32[n_stream, n_rows_d] | Per-sample scale fields (monitoring) |
| `valid` | u8[n_stream, n_rows_d] | Fan-in validity / zero-fill flag |
| `row0_seq`, `row0_t0_native` | i64[n_stream] | First archived sample per stream, in the tier clock and in absolute native samples |
| scalars | | `coarse_global_idx`, `tier`, `t_int_factor`, `bits_main`, `bits_heavy`, `n_values`, `n_heavy`, `value_kind`, `n_sub`, `n_grid`, `pattern_id` |

**Value maps.** These (SPEC §3) are deterministic functions of `pattern_id`. Each half uploads its
pattern file once, to `/dataz/dsa110/operations/x2/patterns/<pattern_id>.npz`, idempotently at start.
They are not copied per event.

**Offline re-imaging.** `tools/x2/reimage_raw_window.py` (WS-D) is built on
`x2/ref/search2x_ref.py` (WS-S) and rebuilds a cube or cutout from `raw/` + plan + pattern. A CI test
checks that it reproduces the half's cutout from the archived window, within the fp32/fp16
tolerances of SPEC §Correctness.

**Sizes.** An event's raw size ≈ fleet corner-turn load × window / 8. The default window is ±0.5 s
(1.0 s), which contains every possible placement of today's 268 ms cube around the event.

| format (plan table) | load Gb/s | per event (1.0 s) | per half | /dataz at 20 sky dumps/day |
|---|---|---|---|---|
| gridded n_sub 2 cint4 | 50.4 | 6.3 GB | 0.79 GB | 126 GB/day |
| gridded n_sub 4 cint5 | 52.9 | 6.6 GB | 0.83 GB | 132 GB/day |
| gridded n_sub 4 cint4 | 42.5 | 5.3 GB | 0.66 GB | 106 GB/day |
| per-baseline n_sub 4 cint5 | 44.7 | 5.6 GB | 0.70 GB | 112 GB/day |
| per-baseline n_sub 2 cint4 | 36.3 | 4.5 GB | 0.57 GB | 91 GB/day |
| per-baseline n_sub 4 cint4 | 35.9 | 4.5 GB | 0.56 GB | 90 GB/day |

How these compare:
- Today's cubes are 8 × 839 MB = 6.7 GB per event, about 134 GB/day at 20 dumps/day, including
  injections.
- The plan's "6-10 GB/event" corresponds to a window of about 1.0-1.5 s.
- Injections are excluded from the raw archive by default (`raw_for_injections: false`).
- The per-node staging peak is 1.1-1.7 GB per event. The existing 150→120 GB cap (`yaml:707-713`) holds
  70-100 un-uploaded events.

**Purge and retention.**
- *Search node, after upload.* Each half purges its own product subdirectory after rc==0, then writes
  `UPLOAD_OK`. This reuses `coinc/cube_uploader.py` with a new `dest_subdir` argument; the default
  `"cubes"` keeps 1x identical.
- *Search node, sweeper.* `dump/cube_retention.py` keeps its tiers (tmp 2 h, age 96 h, size 150→120 GB,
  uploaded first). It treats each per-half product subdirectory as one sweep entry, so markers are never
  shared.
- *Proactive stager.* Disabled in 2x: ring retention of 20 s or more covers the round trip. As an
  option, a "proactive raw snapshot" at ≥ 50σ can be keyed by (specnum, d) if `too_late` cutouts are ever
  observed.
- *h23, C3.* A non-flag-only REJECT deletes `raw/` and keeps `cutouts/` (a few MB, needed to re-plot and
  audit), as today's plots.
- *h23, pruner.* `tools/ops/prune_candidate_products.py:82` `DEFAULT_PRODUCTS` adds `raw`. It already
  ages out bulk products after 14 days with label protection.

### 3.9 1x compatibility

- **1x keeps the cube path unchanged** (plan: the retention ring, stager and uploader retire only in 2x).
- **One adapter for consumers.** `x2/cutout/products.py::load_event_products(event_dir)` returns
  per-(half, coarse) `PeakPlane` records. From `cutouts/` it reads the schema above. From `cubes/`, one
  coarse block per half:
  - `peak_grid` from the NPZ;
  - `apex_image` lazily through the existing mmap (`cube_veto.py:423-454`);
  - `row_valid` from `live_span`;
  - `t_step` = 1;
  - DMs from the plan, falling back to indices exactly as today.
- **Consumers that use the adapter:** C3 metrics, the plotter, Slack readiness, the dashboard counts
  and the C2 completion gate.
- **Equality tests.** On archived 1x events, the C3 `CubeMetrics` produced through the adapter must equal
  today's field by field, and the 1x PNGs produced through the adapter must be byte-identical.
- **Cutouts in 1x** (optional, later). The same schema with `t_ext_us` 1048.576 would remove 1x's
  `too_late` class and the 6.7 GB per event. This is not part of x2.

---------------------------------------------------------------------------------------------------

## 4. Monitoring: keys, measurements and panels to add

Rules:
- Additive only: every 1x key, field, measurement and tag keeps its name and meaning.
- Every payload gains `opoint` (string).
- Publishing becomes time-based (every 2 s) instead of every N cubes.
- Keys marked (WS-x) are published by that workstream; WS-D adds the pusher and Grafana side.

**etcd (new or extended)**

| key | fields | publisher |
|---|---|---|
| `/mon/search_rt/<cn>/compute/<g>` (+) | `opoint`, `coarse_dms` (CSV string), `n_dm_owned`, `cubes_per_s`, `cubes_per_s_nominal`, `cycle_ms_p50`, `cycle_ms_p99`, `stage_ms_{k0,k1,k2,k3a,l1,k3b,topk}_p50/_p99`, `gpu_duty`, `host_ms_p99`, `ring_lag_ms_max`, `c1_cands_dropped_{color,dmfloor,meter}_total`, `c1_queue_depth`, `c1_batches_{sent,dropped}_total`, `pattern_id`, `pattern_build_ok`, `plan_id` | WS-S |
| `/mon/search_rt/<cn>/dm/<d>` (new, one per owned coarse DM) | `tier`, `gpu_half`, `n_fdm`, `cubes_total`, `cubes_per_s`, `ring_lag_ms`, `ring_overrun_total`, `fanin_streams_present`, `fanin_streams_expected`, `fanin_deadline_fills_total`, `s_k_{median,p95,max}`, `layer1_sigma_{median,min}`, `clamp_streak_max`, `c1_cands_per_cycle_mean`, `c1_metering_active` | WS-S |
| `/mon/search_rt/<cn>/cutout/<g>` (new) | `raw_ring_oldest_ext`, `raw_ring_newest_ext` (per-DM min/max frontier), `cutout_requests_total`, `cutout_accepted_total`, `cutout_done_total`, `cutout_partial_total`, `cutout_failed_total`, `cutout_failed_{too_late,window_timeout,queue_full,plan_mismatch,gpu,io,upload}_total`, `cutout_queue_depth`, `cutout_last_latency_ms`, `cutout_gpu_ms_last`, `raw_bytes_total`, `raw_upload_ok_total`, `raw_upload_failed_total`, `purged_files_total`, `dump_root_used_gb`, `retention_swept_total`, `c2_trigger_v1_ignored` | WS-D (helpers) / WS-S (wiring) |
| `/mon/search_rt/<cn>/rx` (activate the planned key) | `ring_version`, `rx_gbps`, `rx_pps`, `frames_total`, `loss_frac`, `bad_{magic,length,pattern}_total`, `zerofill_total`, `fanin_complete_frac` + per-stream loss as `loss_cg<NN>_s<S>` | WS-T |
| `/mon/corr_rt/<chgroup>/corr_fast` (+) | `cn_id`, `chgroup`, `opoint`, `block_ms_p50/_p99/_max` (window), `block_budget_ms`, `stage_ms_<stage>_p50/_p99`, `tx_gbps`, `tx_pps`, `tx_drops_total`, `row_sigma_hat_{min,median,max}`, `clip_frac`, `heavy_clip_frac`, `pattern_id`, `pattern_build_ok` — published **without** needing `--inject-watch` | WS-C |
| `/mon/c2/h23` (+) | `opoint`, `sample_period_us_ext`, `cutout_requests_sent_total`, `cutout_resends_total`, `cutout_status_rx_total`, `events_complete_total`, `events_incomplete_total`, `events_finalised_late_total`, `halves_missing_total`, `missing_<sid>_<g>_total`, `cutout_wait_s_last`, `pending_events` | WS-D |
| `/mon/c3/h23` (+) | `opoint`, `veto_input_kind` (cubes/cutouts), `apex_mismatch_total`, `raw_deleted_total` | WS-D |
| `/mon/service/influx_pusher/h20` (new) | `n_points_written`, `n_writes_failed`, `n_route_errors`, `git_sha`, `ts_unix` | WS-D |

**Influx (pusher):**
- New measurements, all tagged with `opoint`:
  - `search_rt_dm` (tags cn_id, host, gpu_half, coarse_dm, tier);
  - `search_rt_cutout` (tags cn_id, host, gpu_half);
  - `search_rt_rx` (cn_id, host) and `search_rt_rx_stream` (chgroup, stream);
  - `corr_rt_corr_fast` (cn_id, host, chgroup — corr block time finally in Influx);
  - `c3_service`.
- `c2_service` gains the new counters generically (`pusher.py:1264-1341`) and the allow-list adds the
  already-published `cluster_rate`, `ingest_queue_*` and `voltages_*`.
- `COARSE_DM_OWNER` (`pusher.py:180-185`) is replaced by the payload's `coarse_dms` field, which works
  for both op-points.
- The 1x `coarse_dm` tag value stays identical.
- Flip `tests/test_dsart_rt_to_influx_pusher.py:851-870`.

**Grafana** (`build_dashboard.py --opoint`):
- *Rewritten for the op-point:* the search tile row, nominal `n_dm(h)/0.201326592 s`, and the ingest
  scale `T_ext`. The 1x JSON stays byte-identical.
- *New rows:*
  - corr block time p50/p99 against the 134.218 ms budget, with 85% and 100% lines (plan gate);
  - search GPU duty, with 0.6 and 0.75 lines;
  - per-stage µs/image;
  - ring lag;
  - per-DM fan-in, with a 99.9% line;
  - RX loss per stream, with a 1e-4 line;
  - TX Gb/s and pps;
  - row-scale / clip statistics;
  - cutout latency and failures;
  - C2 events complete vs incomplete;
  - missing halves.
- Regenerate and commit the JSON.
- Posting needs `GRAFANA_AUTH` (operator).

**HiPlot:** comes free with the appended CSV columns (`width_ms`, `sample_period_us`, `tier`, `opoint`).

**Dashboard** (WS-D, except Start which is WS-O):
- op-point banner comparing every `opoint` field with `/cnf/opoint`;
- per-half DM ownership table (from the plan);
- RT health page against the plan's gates;
- Dump Now per half (MANUAL requests);
- burst pages reading Level3 `products` ("cutouts 8/8", missing reasons);
- `dsart_rt_to_influx` and `hiplot_c1` added to `services_inventory.py`.

**Operator (separate repo):**
- staleness check on `time_mjd`;
- feed search health into `evaluate_health`;
- per-DM search health;
- probe sends `inj_id` and a varied (l,m), at a DM inside the searched range.

---------------------------------------------------------------------------------------------------

## 5. Implementation task list (WS-D), ordered

**Ownership proposal for the SPEC §11 gaps (integration decides):**
- **WS-D:** `dump/*`, `services/slack_notify.py`, `services/candidate_card.py`, `tools/inject/*`,
  `tools/viz/*`, `tools/c2/*`, `tools/ops/prune_candidate_products.py`, and new `src/dsart/x2/cutout/*`.
- **WS-S:** `services/search_compute_mon.py`, `services/search_ring_mon.py`, and the cutout
  producer `src/dsart/x2/search/cutout_service.py`.
- **Integration:** `common/contracts.py`.

Every task ships CPU tests run through `tools/x2/node_cpu_tests.sh`, and keeps 1x bit-identical
unless it is marked 1x≠ (those go behind a flag, default off, until the user signs off).

| # | task | files (owner) | depends on | 1x |
|---|---|---|---|---|
| D1 | **Units helpers + header-derived periods.** Replace every hard-coded 1048.576 / 32 / 16 downstream (filterbank, card, plotter defaults, bot constants from the `c1` config, `analyse_campaign` text). Add Level3 `c2.sample_period_us` / `dm_peak` / `width_peak` / `width_median_ms` and the appended CSV columns (2x-gated) with rolling-CSV header roll | `coinc/{filterbank,archive,csv_rotator,stats,window}.py`, `services/candidate_card.py`, `inject/bot.py` (WS-D) | — | 1x= |
| D2 | **Wire codecs.** `CutoutRequest`/`CutoutStatus` encode/decode + fuzz/round-trip tests; crc32 class id for the new packets | `coinc/wire.py` | — | 1x= (new types only) |
| D3 | **Cutout schema module.** Writer, reader, validator, `load_event_products` adapter (1x cubes and 2x cutouts); synthetic fixtures; 1x adapter equality on archived events (read-only copies) | `x2/cutout/{schema,products}.py` (WS-D) | WS-P plan indices | 1x= |
| D4 | **Raw-window writer/reader + offline re-imager tool.** Uploader `dest_subdir` + per-product purge; sweeper per-subdir entries; pruner `raw` | `x2/cutout/raw_archive.py`, `coinc/cube_uploader.py`, `dump/cube_retention.py`, `tools/x2/reimage_raw_window.py`, `tools/ops/prune_candidate_products.py` | WS-T ring v4 API, WS-S ref | 1x= |
| D5 | **Search-side cutout service wiring** (with WS-S): listener on 11227+g (both packet types), job queue, raw snapshot, low-priority re-imaging call, status sender, mon key `/cutout/<g>` | `x2/search/cutout_service.py` (WS-S) using D2-D4 | WS-S engine, D2-D4 | 2x-only path |
| D6 | **C2 lifecycle.** Expected halves from the plan, request/ack/resend, status receiver, manifest collection, persisted pending set, Level3 v2 `products`, never-discard (`partial_event_policy`), late arrivals; `--opoint` / section selection incl. hot reload; `_ms` width predicates; `_SPECNUM_FACTOR_FALLBACK` from the op-point; per-component fired memory (flagged); C2 2x-rate replay benchmark (graph, CSV concat off-loop) | `services/coincidencer.py`, `coinc/{criteria,components,veto,broadcast,archive}.py`, `configs/c2_trigger_criteria.yaml` 2x section (via WS-O) | D1, D2 | 1x= except flagged items |
| D7 | **C3 on cutouts.** Adapter-based metrics, per-coarse trigger plane, independent-row statistics, R2 in ms, apex check, 2x thresholds section (initially = 1x values, recalibrated from the shadow veto on CP2 injections), REJECT deletes `raw/`, Slack readiness from Level3; `test_voltage_dump_containment` reads the v3 plan | `services/c3.py`, `coinc/cube_veto.py`, `services/slack_notify.py`, tests | D3, D6 | 1x= |
| D8 | **Plotter + card on cutouts.** DM-sorted multi-half DM-time with tier boundaries, ms axes, DM to 3000, missing-half placeholders; 1x path untouched (byte-identical PNG test); fix the DM-index label fallback (flagged) | `coinc/plotter.py`, `services/candidate_card.py` | D3 | 1x= |
| D9 | **C1 path (c1_emit side).** Cycle-pooled width cap / governor / metering helpers, width cap in T_ext, tier-aware `kernel_id`, global `dm_idx_global`; the `search_compute._submit_c1_batch` integration is WS-S | `services/c1_emit.py` (+ WS-S) | WS-P | 1x= |
| D10 | **Injection.** Seconds-based matcher gate + per-row period; owner map from the plan; owner-scoped health gate (flag); op-point-tagged K keys + wipe procedure (etcd prefix + bot state); width 16 / dm_choices to 3000 avoiding 1600; best-match prefers the owner of the nearest fine; re-measure fluence constants at CP2 | `coinc/inject_match.py`, `inject/bot.py`, `tools/dashboard/dsa_monitor/{inject_calibration,app}.py` | D1 | 1x= (gate flag 1x≠) |
| D11 | **Monitoring.** Pusher measurements + owner from payload, C3 prefix, pusher self-health; Grafana `--opoint` builder + JSON; ring `_end` fix (with WS-S); dashboard banner, ownership, RT health page, Dump Now per half (MANUAL), product counts; services inventory | `tools/dashboard/**`, `grafana` | WS-S/T/C keys | 1x= |
| D12 | **Operator repo.** Probe fix, staleness, search health, per-DM health (separate repo; via the user) | dsa110-operator | D11 | both |
| D13 | **Downstream I1/I2 participation.** Replay archived 1x events and recorded C1 streams through 1x mode (decisions identical), 2x synthetic C1 streams through C2 (rate, clustering, DM comb), the cutout round trip on CPU (synthetic ring → cutout → C2 test instance → C3 → plots) | `tests/x2/test_downstream_*.py` | D1-D11 | — |

**Sign-off items to raise at a checkpoint (1x≠):**
- `partial_event_policy: archive` in 1x;
- the per-component duplicate-dump fix;
- `cluster_rate` counting distinct clusters;
- the noise-colour strength at DM 1400-1500 (finding 5);
- the owner-scoped inject health gate;
- deriving the MJD from the capture epoch.
