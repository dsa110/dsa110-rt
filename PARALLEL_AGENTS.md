# Parallel-agent coordination — M3 ∥ M5

**Scope**: this document is the binding coordination protocol for running M3
(corr-node fast-vis path, plan §8 line 2262) and M5 (search-node detector
pipeline, plan §8 line 2312) in parallel, with two independent agents both
pushing to `dsa110-rt` and both running tests on `h01`.

**Audience**: any agent (or human) picking up M3 or M5. Read this in full
before opening a feature branch. Briefing entry points are listed in §6.

This document is **operative through M3 + M5 hardening** and is **retired
during M6 hardening** (M6 deps M3 + M5 both done, so by M6 the parallelism
question is moot).

---

## 1. Why M3 ∥ M5 works

- Plan dep graph: M3 deps {M1, M2}; M5 deps {M1, M4a}. M5 does **not** dep M3.
- `bench/cube_injection_detector.py` + `inject/cube_injection.py` (plan §8
  line 2329) lets M5 develop the detector + decoder + emitter against
  synthetic cubes, **bypassing every upstream stage** (no corr, no transport,
  no RX ring, no fine-DM combiner, no imager).
- GPU-isolated on h01: M3 → GPU 0, M5 → GPU 1 (per `configs/numa_topology.yaml`
  h01 service-pinning). No device contention for parallel test runs.
- Process-isolation hard constraint (§0): every service is its own systemd
  unit + own buffers + own GPU + own ports. The two milestones touch
  disjoint runtime surfaces.

The only M3 → M5 coupling point is `bench/voltage_fixture_search.py` (M5
operator gate, §8 line 2330), which needs `dsart-corr-fast@01` working.
M5 develops up to that point against `cube_injection_detector.py` and only
fires the voltage-fixture gate once M3 has emitted a captured transport-TX
`.npz` set.

---

## 2. Branch / merge model

- `main` is canonical. **No direct commits to `main`** during M3 ∥ M5
  parallel work.
- Each agent owns a long-running branch:
  - **M3 agent → `m3/main`**
  - **M5 agent → `m5/main`**
- Per-feature branches roll up: `m3/rfi`, `m3/grid`, `m5/imager`,
  `m5/detector`, etc. → fast-forward into the agent's long-running branch.
- **PRs from `m{3,5}/main` → `main`**: opened ~daily. Rebase the
  long-running branch on `origin/main` before each merge. The first agent
  to merge sets the baseline; the second rebases.
- Per-milestone fix tracker file lives at the repo root for the duration of
  the milestone, exactly like M2 used `M2_PLAN_FIXES.md`:
  - **M3** uses `M3_PLAN_FIXES.md` (created at M3 kickoff, retired in M3
    hardening).
  - **M5** uses `M5_PLAN_FIXES.md` (same pattern).

Each agent owns its own preflight + DoD scripts + status JSON; see §4.

---

## 3. File ownership

Three classes of files. Class A and B never need coordination; Class C does.

### Class A — single-owner-per-milestone (no conflicts)

**M3 owns:**

- `src/dsart/services/corr_fast_compute.py`
- `src/dsart/rfi/*` (autos, sk, bandpass_outlier, group_outlier, sum_threshold,
  flagants_loader)
- `src/dsart/cal/antennas_out.py` (legacy binary cal-blob loader)
- `src/dsart/grid/*` (kernel, sparsity_pattern, sparse_pattern; **see Class C
  for sparsity_pattern.py — both ends import**)
- `src/dsart/coarse_dm/*` (dedisp, stage2_fifo)
- `src/dsart/inject/online.py` (voltage-domain online injector)
- `src/dsart/services/static_sky_subtract.py` (or grid/static_sky.py)
- `bench/{fast_path_throughput,fast_isolation,static_sky_subtract,
  rfi_calibration,rfi_warmup,cal_reload,
  voltage_fixture_fast_corr_continuum,voltage_fixture_fast_corr_burst}.py`
- `tests/test_{rfi,grid,coarse_dm,corr_fast}*.py`,
  `tests/fast_corr/*`
- `tools/dod/M3_preflight.sh`, `tools/dod/M3.sh`
- `configs/config_corr.yaml`
- `M3_PLAN_FIXES.md`

**M5 owns:**

- `src/dsart/services/search_compute.py`
- `src/dsart/fine_dm/combiner.py`
- `src/dsart/image/imager.py`
- `src/dsart/detector/{forward,decoder,merger,kernels}.py`
- `src/dsart/inject/cube_injection.py` (post-imaging detector unit-test injector)
- `src/dsart/noise_norm/*` (Layer-1 + Layer-2)
- `src/dsart/trigger/emitter.py`
- `bench/{search_node_throughput,noise_norm_calibration,
  trigger_emitter_wiring,cube_injection_detector,voltage_fixture_search}.py`
- `tools/viz/search_detector_check.py`
- `tools/viz/search_helpers.py` *(new file; see Class C below)*
- `tests/test_{detector,fine_dm,imager,noise_norm}*.py`
- `tools/dod/M5_preflight.sh`, `tools/dod/M5.sh`
- `configs/config_compute_search.yaml`
- `M5_PLAN_FIXES.md`

### Class B — append-only joint files (cheap to merge)

Both agents append; Git auto-merges non-overlapping insertions.

- `src/dsart/common/contracts.py` — pre-allocated bit budget for
  `CandidateFlags`:
  - **M3 owns bits 0-2** (Layer-3 / off-zenith-rejection — currently reserved
    in plan §3.1; M3 may consume these for fast-corr-side rejection flags).
  - **M5 owns bits 3-6** (already populated: NOISE_WARMUP, RFI_WARMING_UP,
    HALO_DROPPED, TIME_EDGE_DROPPED).
  - Bits 7+ : unallocated, ask before consuming.
  Each agent only edits its own dataclasses + bit definitions.
- `src/dsart/common/constants.py` — both append; no overlap expected.
- `tests/test_contracts.py`, `tests/test_numerical_conventions.py` — both
  append.
- `dsa110-rt_revamp_7b1d2669.plan.md` — each agent edits its **own milestone
  section + its own `Mn-carryover` subsection** only. Don't touch the other
  agent's section.

### Class C — genuine shared files (need explicit ownership)

| file | owner | non-owner protocol |
|---|---|---|
| `src/dsart/grid/sparsity_pattern.py` | **M3** (corr-side `cmd: prepare` build) | M5 imports the read-only API; if M5 needs a function added, opens a small PR + M3 acks. Plan §4.3 Option C: "both ends import this; corr-side and search-side rebuild patterns locally at `cmd: prepare`." |
| `tools/viz/common.py` | **M3** (already touched by M2 — F20 fix) | M5 puts its viz helpers in **`tools/viz/search_helpers.py`** (new file). Edits to `common.py` from M5 only via PR. |
| `src/dsart/cal/bf_weights.py` | **M3** (production cal-apply path) | Already 221 lines from M2 (slow-corr cal-loader). M3 extends for the fast-corr `cal_loader`. M5 does not import. |
| `tools/build_dm_plan.py` | **M3** (M1 deliverable; both ends consume `dm_plan.npz`) | M5 only consumes the npz output via `DmPlan.from_npz()`. If M5 needs a new field, PR + M3 acks. |

If you need to edit a Class C file you don't own, **open a small PR first**;
the owner reviews + merges, then you rebase your branch on the updated
ownership-file state.

---

## 4. h01 test-conflict isolation

Five contention surfaces, all solvable with cheap conventions baked into the
DoD scripts.

### 4.1 PSRDADA buffer keys

M2 used `fada` / `bada` for the slow-corr smoke. M3 and M5 each get their
own per-milestone prefix so two parallel test runs never share a key.

| env var | M3 default | M5 default | production |
|---|---|---|---|
| `DSART_BUFFER_KEY_PREFIX` | `m3` | `m5` | unset → canonical `f`/`b`/`d` |
| → `fada` becomes | `fa3a` | `fa5a` | `fada` |
| → `bada` becomes | `ba3a` | `ba5a` | `bada` |
| → `dada`/`dadc` becomes | `da3a`/`da3c` | `da5a`/`da5c` | `dada`/`dadc` |

`tools/dod/M3.sh` exports `DSART_BUFFER_KEY_PREFIX=m3`; `M5.sh` exports
`m5`. Service bring-up scripts (`bench/replay_voltage_dump.py`,
`corr_fast_compute.py` startup, search-rx bring-up) read this env at
startup and substitute the third character of each canonical 4-char key.

### 4.2 GPU pinning

| milestone | GPU | rationale |
|---|---|---|
| **M3** | `CUDA_VISIBLE_DEVICES=0` | matches production `dsart-corr-fast@01` pinning (`configs/numa_topology.yaml::lxd110h01.service_pinning.dsart-corr-fast@01.cuda_device = 0`). |
| **M5** | `CUDA_VISIBLE_DEVICES=1` | uses the sole-occupant `dsart-search-compute@01-1` pinning. The `@01-0` instance shares GPU 0 with corr-fast/slow in production; for parallel dev we use only `@01-1`. |

Both DoD scripts pin via env at the top.

### 4.3 Filesystem & report dirs

Already separated by design — no contention:

| artifact | M3 path | M5 path |
|---|---|---|
| status JSON | `~/dsart-m3-status.json` | `~/dsart-m5-status.json` |
| bench reports | `bench/reports/<UTC>/<run_id>/M3/...` | `bench/reports/<UTC>/<run_id>/M5/...` |
| operator-approval marker | `bench/reports/M3/m_operator_approved.yaml` | `bench/reports/M5/m_operator_approved.yaml` |

Read-only consumers (`/home/ubuntu/data/voltages/`, `/home/ubuntu/data/fstables/`,
`/home/ubuntu/data/cal/`) are shared with no write contention.

### 4.4 Concurrent DoD invocation guard

Each DoD script `flock`s its own per-milestone lockfile at the top:

```bash
exec {LOCKFD}>/var/lock/dsart-m3.lock || exit 1
flock -n "$LOCKFD" || { echo "another M3 run in progress"; exit 1; }
```

Per-milestone lock — M3 and M5 do **not** share a lock and are free to run
simultaneously.

### 4.5 Network ports (M5 + M4a)

M5's loopback search-RX binds `127.0.0.1:9000+chgroup` (per plan §4.3).
M3 doesn't bind ports. No collision.

### 4.6 etcd test-key namespace

`cmd: prepare` paths under `/cnf/dsart/...` are namespaced per agent via
`DSART_ETCD_NAMESPACE_PREFIX`:

| env var | M3 default | M5 default | production |
|---|---|---|---|
| `DSART_ETCD_NAMESPACE_PREFIX` | `m3` | `m5` | unset → `dsart` |
| → key prefix becomes | `/cnf/dsart-m3/...` | `/cnf/dsart-m5/...` | `/cnf/dsart/...` |

Wired into `src/dsart/common/config_loader.py` (M1).

---

## 5. Voltage-fixture conventions on h01

The plan (§3.3 / §4.7) talks about `/home/ubuntu/data/voltage_fixtures/<run_id>/`
as the canonical fixture root, but in practice **M2 wired benches directly
to `/home/ubuntu/data/voltages/<run_id>/`** because that's where the legacy
DSA-110 dumper deposits files. M3 and M5 follow the same convention — the
plan's `voltage_fixtures/` reference is flagged for retirement during M3
hardening.

Canonical dump layout (mirrors 0319 for both continuum and burst fixtures):

```
/home/ubuntu/data/voltages/<run_id>/
├── voltages/
│   ├── <run_id>_sb00_data.out
│   ├── ...
│   ├── <run_id>_sb15_data.out
│   └── T2_<run_id>.json     ← {ra, dec, mjds, specnum, dm, snr, ...} legacy
└── cals/
    ├── beamformer_weights_sb<NN>_<source>.dat (one per SB)
    ├── beamformer_weights_<source>.yaml
    └── flagants.dat                              ← single shared file across SBs
```

**No on-disk `manifest.yaml` is required.** Benches synthesise the manifest
in-memory from `T2_<run_id>.json` (same pattern as
`bench/run_0319_pipeline.py` lines 463-471: `{ra → ra_deg, dec → dec_deg,
mjds → mjd, specnum → utc_start_specnum}`). For burst fixtures, the
additional `T2.dm` field is the burst's known DM; the burst position is
`T2.{ra, dec}`; the burst arrival time is `T2.mjds` (approximated as
dump-start MJD for ≤32 s of LST drift, which is negligible for `(l, m)`
acceptance per F19/D18 from the M2 carryover).

**Active fixtures**:

- `0319/` — continuum fixture (3C-class compact source 0319+415,
  `T2_0319bbb.json`). M2 acceptance fixture; M3 reuses for the continuum
  imager check (§8 line 2282).
- `250924mptq/` — burst fixture (DM≈404.7 pc/cc, RA=307.78°, Dec=53.85°,
  MJD=60942.172, SNR≈30, `T2_250924mptq.json` carries `dm`/`ra`/`dec`/
  `mjds`/`specnum`). Used for the M3 burst sub-DoD (§8 line 2286), the M3
  16-chgroup alignment preview (§8 line 2291), and the M5 voltage-fixture
  end-to-end gate (§8 line 2330).

### 5.1 Burst-fixture test config recipe (250924mptq)

To run the M3 fast-corr / coarse-DM / gridder pipeline against the burst
fixture using **production code paths only** (no test-bespoke modes):

1. **Integration time**: `--t-int-fast-us 1048.576` (= 4× native 262.144 µs).
   The native goal stays 262.144 µs; this just exercises the existing
   configurable knob (plan §3 line 524) for a coarser, easier-to-inspect
   fast cadence (~128 frames per fada block, ~1920 frames per dump).
2. **DM plan**: a custom single-cell `dm_plan_burst_250924mptq.npz`
   generated by the existing M1 `tools/build_dm_plan.py` with `dm_min ≈ 404`,
   `dm_max ≈ 406` so the coarse-DM grid resolves to one cell at ~405 pc/cc.
   Path passed via `--dm-plan-path`. Reduces captured transport `.npz` size
   ~24× and skips DM trials we don't need.
3. **DEC phasing**: comes from the cal-blob load + the new
   **F21 fast-corr cal-apply DEC-only phase fold** (see §7 below). The
   bench supplies `obs_dec` via CLI; in production it comes from etcd
   `cmd: prepare`.
4. **Bench**: `bench/voltage_fixture_fast_corr_burst.py --voltage-run-id
   250924mptq --t-int-fast-us 1048.576 --dm-plan-path
   configs/dm_plan_burst_250924mptq.npz --obs-dec 53.848986`. Captures
   transport-TX `.npz` per chgroup to
   `bench/reports/<UTC>/250924mptq/M3/transport_dumps/`.
5. **Viz**: `python -m tools.viz.corr_imager_dedisperser_check --mode burst
   --voltage-run-id 250924mptq --chgroup all --out
   bench/reports/<UTC>/250924mptq/M3-burst/`. The tool's existing burst-mode
   panels (synthetic filterbank, coarse-DM sweep, static-sky before/after)
   plus a fast-cadence dirty-image animation at the single coarse-DM cell
   (the latter as a small extension to the existing tool; useful for any
   future burst-fixture inspection in production, not test-bespoke).

This same recipe (with the appropriate `<run_id>`) services any future burst
fixture; no test-only code paths are introduced anywhere in the corr-side
service tree.

---

## 6. Briefing entry points

Each agent reads, in this order, before opening any feature branch:

1. `dsa110-rt_revamp_7b1d2669.plan.md` **§8.M2-carryover** subsection (lines
   2198-2261) — locked decisions D1-D18 and implementation-level gotchas
   F1-F20 from M2. **Binding context for both M3 and M5** (especially the
   F18 PyTorch row-major vs cuBLAS column-major upper-tri-gather index swap,
   the F20 `np.fft.ifft2` `(u, v)` negation for TMS-canonical `(l, m)` axes,
   the D13/D14 conda-env split, the D15/D16 voltage-layout + int4-fluff
   optimisations, the F8/D7 PSRDADA replay writer, and the D8/F12 cfp32
   bada dtype).
2. `dsa110-rt_revamp_7b1d2669.plan.md` **§8 M3** (lines 2262-2294) for M3
   agents; **§8 M5** (lines 2312-2341) for M5 agents.
3. **This document** (`PARALLEL_AGENTS.md`) for branch / file-ownership /
   h01-test-isolation conventions.

That's the full briefing. No chat-relayed context is needed.

---

## 7. M3-specific notes carried into the substrate

These are M3 design decisions identified during substrate authoring (during
the M3 kickoff). They land in `M3_PLAN_FIXES.md` as F-items as M3 progresses
and are folded into plan.md during M3 hardening.

- **F21 (proposed)**: fast-corr cal-apply folds a per-(ant, ch, pol)
  DEC-only phase exp(−2π i f sin(δ_obs − φ_lat) · N_a / c) into the cal
  weight before the GEMM, mirroring `dsaX_bfCorr.cu::populate_weights_matrix`
  central-beam (`bm=127`) computation (lines 1082-1085, with the
  beam-offset cross-track term zeroed). Without this fold a source 16.6°
  off zenith (e.g. the 250924mptq burst at Dec=53.85° vs OVRO lat 37.234°)
  lives outside the iFFT FoV at any reasonable `N_grid`. The slow-corr
  doesn't need this because `meridian_fringestop` does it downstream in
  casa38; the fast-corr has no such downstream step.
  - `obs_dec` is supplied via etcd `cmd: prepare` in production, CLI in
    benches.
  - `N_a` (per-antenna N-S position) comes from the M2-validated antpos
    table.
  - `φ_lat = 37.234°` (OVRO) — already a constant in `common/constants.py`.

(More F-items will accumulate here / in `M3_PLAN_FIXES.md` as M3 lands.)

---

## 8. Adding a new milestone-parallel agent

If a third agent later needs to run in parallel (M4a is the natural
candidate during late M3 / early M5 dev):

1. Pick an unused single-letter prefix (`m4a` → `f4a` / `b4a` / `d4a` keys,
   `~/dsart-m4a-status.json`, `bench/reports/<UTC>/<run_id>/M4a/...`).
2. Pick the unused GPU if any, OR live with shared-GPU testing (M4a is
   transport-only, low GPU usage; could share GPU 0 with M3 or GPU 1 with
   M5 with negligible contention).
3. Create `tools/dod/M{n}.sh` + `M{n}_preflight.sh` from the M3
   templates.
4. Add a row to the §3 ownership tables with the agent's owned modules.
5. Update §4 buffer-prefix table.
6. Open the agent's long-running branch `m4a/main` off `main`.
