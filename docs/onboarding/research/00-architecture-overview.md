# Research packet 00 — Workspace map, architecture, audit, web UIs

Condensed findings from the initial read-only study (2026-07-10). All paths relative to `/home/ubuntu/vikram/dev` unless absolute.

## Workspace map

`~/vikram/dev` is a workspace (not a git repo): canonical repo `dsa110-rt/` (branch `main`, package `dsart`) + per-milestone git worktrees (`dsa110-rt-m5` = `m5/main` detector, `dsa110-rt-m6` = `m6/main` clustering, `dsa110-rt-m4a-tx` = `m4a/tx-prod-header` transport, `dsa110-rt-operint` = `operator-integration`, `dsa110-rt-rfi-flagger` = `m3/rfi-flagger`) + separate repo `dsa110-operator/` (agent console, laptop-side, SSH-to-h23 only) + docs (`REALTIME_FRB_SEARCH.md` legacy ground truth; `dsa110-rt_revamp_7b1d2669.plan.md` milestone plan; dev-plan narrative; critical review) + artifact dirs (`m3-*`, `m4b-deploy`, `m7-deploy`, `cal_plots`, `_inspect`, `scratch`). Parallel-agent protocol: `PARALLEL_AGENTS.md`. Milestones M0–M3, M5, M6 hardened; M8 live voltage-dump E2E passed 2026-07-09 (`scratch/M8_E2E_VOLTAGE_DUMP_TEST_20260709.md`).

## dsart architecture

Invariants: one service = one systemd unit = one process = own buffers/GPU/ports. Control plane = etcd (never RPC). Data plane = PSRDADA shm rings (corr) / POSIX-shm ring (search); reader counts physics-pinned (missing reader stalls the writer). Best narrative doc: `dsa110-rt/docs/overview/dsa110-rt-overview.pdf` (.tex).

**Corr node (16, one per chgroup):**
- Capture: C binary `dsart_capture_manythread` (recvmmsg), SNAP UDP ports 4011 (pair A → `dada`) / 4012 (pair B → `eada`); each ring 150,994,944 B × 20 blocks (~2.7 s).
- `dsaX_merge` → `fada` (301,989,888 B × 70 ≈ 9.4 s, r=3): readers = corr_slow, corr_fast, voltage_retention (in-RAM VoltageRing, single-writer seqlock so disk dumps can't back-pressure capture).
- Slow path (GPU 0): `corr_slow_compute` → `bada` (28,606,464 B × 300, r=2→now sole reader meridian_fringestop) → UVH5 for calibration. Deliberately uncalibrated visibilities.
- Fast path (GPU 1), `corr_fast_integration` (plan §4.2 steps): int4 unpack → autos → RFI flag (SK + bandpass outlier + group outlier + sum-threshold + static flagants, OR-combined into mask + flag-source bit tags) → zero-fill flagged (ant,ch,pol) → cal/bandpass/DEC-fringe weights (`cal/cal_loader.py`, legacy beamformer_weights blobs) → GEMM fast visibilities → Stokes-I → GPU gridder (`grid/kernel.py`, sparse-COO pattern `grid/sparsity_pattern.py`) → static-sky subtract (StaticSkyMean causal 8-block mean) → coarse dedispersion (stage-1 per-channel integer shifts before grid; `coarse_dm/dedisp.py` sums channels per DM trial) → stage-2 FIFO → int8 quantize → transport TX (UDP ProdFrames, 72-byte header).

**Transport:** search RX binds ports 6625+chgroup (16 sockets). C extensions: `transport/_recv_ring` (POSIX-shm SPMC ring, recv_ring.c) and `_recv_epoll` (epoll + recvmmsg drain, header parse, per-(corr,dm) reorder window). Ring name `/dsart-rxring-<cn_id>`, production ~9.8 GiB (t-buf-samples 8192).

**Search node (4: n01/n02/n09/n13 = lxd110h01/h02/h09/h13; 2 GPUs each):** `search_rx` reassembles from all 16 corr nodes → shm ring. Two `search_compute` halves (one per GPU, disjoint coarse-DM owners; per-host overrides so fleet covers all trials). Per cube: fine-DM combine (`fine_dm/combiner.py`, per-chgroup fine time shifts, fused sparse-scatter-sum) → imager (`image/imager_gpu.py`: fused dequant+combine NVRTC kernel, int32 accumulate, cuFFT irfft2 cfp16, fftshift, edge mask → fp16 `[T_det, N_fdm, N, N]`) → noise_norm layer1 (σ-clipped global scalar per cube/fine-DM, 5-cube burn-in) + layer2 (per-detector-kernel σ_k EMA τ≈30 s) → detector (`detector/forward.py` boxcar-via-cumsum, conv1d forbidden for FLOP budget ~1 TFLOP/cube; decoder NMS; merger 4D suppression) → clusterer (`cluster/forward.py` HDBSCAN cityblock min_cluster_size=2 ε=10, DBSCAN fallback) → cands logger (T1/T2 hourly ASCII) → C1 emit (`services/c1_emit.py`, persistent TCP to h23:11500, 8 sockets fleet-wide; schema `docs/c1c2/C1C2_WIRE_SCHEMA.md`).

**h23:** C2 coincidencer (`services/coincidencer.py`, `coinc/`): rolling MJD window, union-find components (edge iff |t_i−t_j| ≤ (w_i+w_j)/2), YAML criteria hot-reload, holdoff; on dump trigger: event name, CSV audit, UDP broadcast to 8 C1 listeners → corr nodes stage retained voltage. C3 (`services/c3.py`): polls candidate archive, cube_veto morphology veto (robust z = (max−median)/(1.4826·MAD); always keeps injections/ambiguous), KEEP → collect 16 fragments to `<event>/Level2/voltages/`, REJECT → conservative cleanup to `candidates_rejected/`; default `flag_only=True`.

**Control plane:** `services/dsart_rt.py` per node (`-in pipeline_rt|search_rt -cn <id>`), reads `/cnf/pipeline_rt` / `/cnf/search_rt`, watches `/cmd/{corr_rt,search_rt}/<cn>` + broadcast `<ns>/0`, verbs start/stop/utc_start/utc_stop, heartbeats + dada_dbmetric → `/mon/...`. Mirrors legacy corr.py surface; disjoint key namespace so both run side-by-side.

**Slow-vis:** feeds calibration (SEFDs, beamformer weights) via meridian_fringestop UVH5; separate from trigger path, same capture front-end.

**Testing without telescope — three injection altitudes:** (1) cube-domain `inject/cube_injection.py` straight into detector (primary detector gate; enabled M5 ∥ M3 dev); (2) voltage-domain `inject/online.py` — dispersed, cal-phasored per-pol envelope `.add_()`ed into voltages_real/imag BEFORE RFI → full end-to-end; armed via etcd, matched post-hoc; (3) replay real voltage dumps into `fada` (`bench/replay_voltage_dump.py`, e.g. run 250924mptq) or `dada_junkdb` noise. ~131 test files; ~50 bench drivers; DoD gates `tools/dod/M<n>.sh` (spin up GPUs/services — approval needed to run).

## Audit verdict (claims vs code): GENUINE

- Dispersion constant 4.148808 ms·GHz²·pc⁻¹cm³ consistent (`common/constants.py:163`, `inject/online.py:441`, `coinc/cube_veto.py:210`; ν_top = 1.49875 GHz).
- Boxcar SNR √(N) normalization correct (`detector/forward.py:744`), real Welford/EMA σ.
- No injection backchannel: detector blind; matching post-hoc by tolerance (`coinc/inject_match.py`, specnum gate ±2048 added to fix cross-attribution — documented at inject_match.py:47-58).
- DoD scripts run real pytests + numeric gates; M7_0.sh spawns orchestrator, asserts state transitions.
- Preserved FAIL artifact: `m3-burst-correctness/summary.json` `"stage": "FAIL"` (8-cell pixel offset) with git_sha/UTC provenance. Later l/m re-centering fix commits (3b2526d, cf13ca0, bdbed71).
- Honest git history; ~4,970 asserts / 133 test files; tight tolerances; only benign mock (tx.time clock).
- Weak spots (disclosed): `tools/dod/M3.sh:311` continuum bench `|| true` soft gate; headline 5.84 Gb/s zero-loss 1-h soak (m7-deploy/M7-2-REPORT.md) is LOOPBACK (labeled); `services/capture_supervisor.py` NotImplementedError stub (dead code, never wired).
- CORRECTION vs early audit: `transport/frame.py` FastVisFrame computes a REAL zlib.crc32; the production ProdFrame (`transport/prod_frame.py:49-60`) deliberately has NO CRC by design (pattern_id + seq reorder + n_filled instead). Do not call it a "placeholder bug".

## Web UIs / port forwarding

| UI | Host:port | Notes |
|---|---|---|
| dsa_monitor dashboard | h23:5778 | Flask, systemd `dsa_monitor.service`, `DSA_MONITOR_PORT=5778`, bind 0.0.0.0. Main observatory UI + Control tab. `ssh -L 5778:localhost:5778 h23` |
| Grafana | lxd110h20.pro.pvt:3000 | grafana-server.service on h20; read-only tier "never restarted". `ssh -L 3000:lxd110h20.pro.pvt:3000 h23` |
| InfluxDB 1.x | lxd110h20.pro.pvt:8086 | db=dsa110; fed by dsart_rt_to_influx.service on h20. No web page — /ping→204, query via /query?db=dsa110&q=... DNS aliases: grafanaservice.pro.pvt = influxdbservice.pro.pvt = lxd110h20 = 10.42.0.249. |
| dsa110-operator console | laptop 127.0.0.1:8787 | `python -m dsa_operator.web.app` via scripts/laptop.sh; needs tunnels 12379→etcdv3service.pro.pvt:2379 and 15778→h23 localhost:5778 (transport/ssh_tunnel.py). |
| hiplot | h23:5027 | calibration23 LXC container on h23; hiplot.service; confirmed listening. |
| etcd | etcdv3service.pro.pvt:2379 | API not UI. |

SSH topology (user's laptop config): laptop → `ovro` (ssh.ovro.caltech.edu) → `dsa110maas` (dsa110maas.ovro.pvt) → `h23` (lxd110h23.pro.pvt) → corr/search nodes (`lxd110hNN.pro.pvt`, ProxyJump h23). Gotcha we hit live: ssh ControlMaster multiplexing keeps old LocalForwards alive; change forwards with `ssh -O cancel -L ...` / `ssh -O forward -L ...` or `ssh -O exit` to restart the master. `LocalForward 8086 localhost:8086` on dsa110maas was wrong (nothing on maas); correct target `influxdbservice.pro.pvt:8086`.

Legacy topology (REALTIME_FRB_SEARCH.md): 16 corr hosts = h03,04,05,06,07,08,10,11,12,14,15,16,18,19,21,22; 4 search = h01,02,09,13 (beam slices 0-127/128-255/256-383/384-511 in legacy). h23 = head/dev + C2/C3 + dashboard. Block cadence 0.134217728 s.
