# Research packet 03 — Included vs not included; legacy coexistence; caveats

Sources: dsa110-rt/docs/overview/dsa110-rt-overview.tex (2026-06-10 — predates M8 test and SPL landing; doc-lag flagged), dsa110-rt_revamp_7b1d2669.plan.md, dsa110-rt_dev_plan_narrative.md, dsa110-rt_critical_review.md, REALTIME_FRB_SEARCH.md, scratch/M8_E2E_VOLTAGE_DUMP_TEST_20260709.md.

## INCLUDED — implemented and validated

- Fast-vis search path end-to-end: 10-step corr_fast hot path (overview.tex:602-766); M7.7 op-point ~166 ms/cube vs 201.3 ms budget (~17% headroom) (tex:114-126,221-226); M7.4 gate PASS (docs/M7.4_GATE_REPORT.md:13-24: 30-min soak, zero kernel/corr drops, 128/128 corr + 12/12 search routines alive, RFI ~8% flag fraction, C2 clean).
- Slow-vis UVH5/cal path: corr_slow → bada (byte-identical legacy contract) → meridian_fringestop (unmodified casa38 dsamfs) → UVH5 + /cmd/cal notify (tex:592-600,2670-2739). Sole bada reader since M7.4 Phase 9 (2026-05-30); drains removed (tex:585-591,2729-2732).
- Injections: overhauled — √flux voltage convention, linear S/N model K·F/√W, per-DM buckets, saturation-aware fluence ladder, probe rotation, pre-flight health check, C2 specnum-proximity gate, 32-block auto-arm margin (tex:159-170,1854-2398); injection hook post-cal since M7.4 Phase 8 (tex:635-651).
- C1/C2/C3: C1/C2 fleet gate PASSED 2026-05-28 (tex:3211-3212); C2 stacked-waterfall plotter (tex:178-181); C3 live in flag_only=true (safe default; REJECT untested) (docs/voltage_dumps/VOLTAGE_DUMP_C3_DESIGN.md:254-258; M8 report:12-16,55-58).
- Voltage dumps: M8 live test PASSED 2026-07-09 (scratch/M8_E2E...md): single node ~24 s byte-exact, fleet 16/16 in 25 s zero drops, C3 fail-open KEEP path exercised, staging cleanup sentinel worked. Caveats: mjd_target=0.0 cosmetic bug (fix identified not applied); C2 live _maybe_broadcast_voltage + cube-veto REJECT path NOT exercised (voltages_enabled default false); dsart_c3.service unit-vs-process mismatch unreconciled.
- Dashboard: dsa_monitor Control tab (5-state lifecycle banner, fleet recovery ops), injection panel, SPL panel, C2 metering (tex:2536,3226-3232); Influx/Grafana pusher dsartRtMpV1 89-panel generator on lxd110h20 (tex:3118-3180).
- RFI chain (flagants, SK, bandpass, group, SumThreshold) (tex:773-824), active at ~8% in gate.
- Static-sky: StaticSkyMean causal 8-block sliding mean (replaced EMA) (tex:138-144,715-745).
- M4b real 40GbE: pair-rate PASSED 2026-05-15 at 1.752 Gb/s/pair; fleet 16→4 corner-turn at 28 Gb/s aggregate PASS after ipfrag_high_thresh sysctl fix (plan.md:42,45). Loopback→real-fabric transition COMPLETE (dev_plan_narrative.md:158-166,323-327).

## PARTIAL / NOT YET / DEFERRED

- SPL spectral-line mode: implemented in source (dsart_rt.py:94-220 _load_spl_cfg; dashboard spectral_line_gate.py) — etcd-gated /cnf/spectral_line per-chgroup toggle spawning meridian_fringestop_spl (finer channelization, *_sb<NN>_spl.hdf5) instead of bada_null_drain. Fail-safe default disabled; takes effect only on next restart_all+start (spectral_line_gate.py:22-24,107-109). Tests exist. Postdates 2026-06-10 doc freeze; UNKNOWN if exercised on-sky.
- M4a ProdFrame (72 B production header, prod_frame.py:1-60) coexists with older 32 B FastVisFrame (frame.py) — "both co-exist during M4a… becoming the canonical wire form" (prod_frame.py:6-9). ProdFrame deliberately has NO CRC (integrity via pattern_id + seq reorder + n_filled) — design choice, not omission. NOTE: frame.py's FastVisFrame computes a REAL zlib.crc32 (frame.py:195-304) — do not call it a stub.
- capture_supervisor.py: 9-line NotImplementedError("M2/M3/M5") stub, referenced by nothing (dead code).
- Still-pending list as of 2026-06-10 (tex:3274-3311): (1) Phase 6c RFI-zero-fill total-power bias + DM-recovery-bias study; (2) slow-vis archive replay / re-deriving calibration23 K-cal from post-2026-05-30 UVH5 (earlier files have stale time anchor); (3) control verbs prepare, record, inject (orchestrator-side), reload_cal, reload_flagants, ctrltrigger, trigger unimplemented; consumers not yet migrated off legacy /mon/corr/<n>; (4) Detector v2 (learned) — swap mechanism proven (M5 IdentityDetector), rollout deferred to M-defer (plan.md:3108).
- Also deferred: carry-over re-imaging implemented+validated but OFF in production (tex:122-123); Layer-3 per-fine-DM EMA deferred in v1 (tex:3644); Detector-v2 roadmap Phases I–VI (PSF-matched kernels, PSF-aware merger, TriggerCondition extensions, learned detector, dedispersion-domain RFI features, priors) = forward-looking only (tex:3509-3727); runtime set_detector_mode + atomic rollback not in v1 (plan.md:2866); fl_<specnum>.out collision handling deferred (critical_review.md:289,343).
- Fast path NOT fringe-stopped per element by design (<0.01% S/N loss DM=100; ~6% worst-case) (tex:706-713).

## Legacy KEPT vs DELETED

- KEPT: meridian_fringestop (casa38 dsamfs, unmodified, wrapped by tools/ops/meridian_fringestop_rt.py) — sole bada reader; bada byte-for-byte contract ([4656,384,2] complex64 @ 134.218 ms) preserved so meridian/calibration23/H5 archive/operator UIs at /mon/corr/<n> keep working "until the M7.6 cutover" (tex:2670-2739,3825-3833).
- Legacy dsaX_hella (beamformed single-pulse search, 2/node, → coincidencer/T2 at 10.42.0.90:12345) is the legacy production search (RFS.md:475-538,69,142). **UNKNOWN whether Hella still runs in production today or has been retired — no doc states a retirement date. Flag as open question.**
- DELETED: dsaX_nsfrb (+ gen_nsfrb_fstable.py, caba buffer, dada_dbnull -k caba) (RFS.md:4,42-44,405-417,646,825-833); bada/dada drain stand-ins (M7.4 Phase 9).

## Coexistence / cutover

- Same etcd cluster + hosts, disjoint namespaces: "/cnf/pipeline_rt, /cmd/corr_rt/<n>, /mon/corr_rt/<n>… so legacy corr.py can keep running through M7 without contention (Q15)" (plan.md:2658). Conda envs side-by-side: casa38 (py3.8, legacy/meridian) + dsa110-rt (py3.11) (plan.md:167,2475).
- Cutover staged (M7.6): mon-key rename /mon/{corr,search}_rt → /mon/{corr,search} ("key-rename only"); gate = pulsar single-pulse detection in-beam OR side-by-side trigger-rate parity vs legacy corr.py OR a real FRB; operator sign-off (plan.md:2822). As of 2026-06-10 doc, formal cutover NOT confirmed complete (tension: M7.6 deliverables listed as shipped, tex:3226-3245, but "until the M7.6 cutover" implies pending). **Current status unknown — check dashboard/obs-status.**

## Caveats / limitations

- GPU driver: PyTorch 2.x cu118 needs driver ≥520.61; h01 was 455.23/CUDA 11.1 (silent failure at first tensor.cuda(); no forward-compat on 2080 Ti) — pre-M0 bump on h01; other 17 nodes deferred to M-defer (dev_plan_narrative.md:714-725; plan.md:27,61).
- Python: dsa110-rt env pins py3.11 (envs/dsa110-rt.yml:10); legacy casa38 py3.8; psrdada-python rebuild risk R11 resolved pre-M0 (plan.md:3118). Both ABI .so builds on disk (cpython-38 + cpython-311).
- Phase A (single-host h01, loopback) vs Phase B (fleet, real 40GbE) — historical, resolved (dev_plan_narrative.md:158-166).
- RFI/false positives: SK FAR pinned 1e-4 per (ant,ch,pol,M) (tex:794-797); Layer-1 coverage correction kills ~50000× synth-Gaussian false alarms (tex:3224-3225); gate flag fraction ~8%; "structured false-positives near bright sources" noted for a FUTURE TriggerCondition extension (tex:3722).
- DM plan v2 has quantified S/N loss vs DM (Monte-Carlo, tex §6) — bounded, known.

## Open unknowns to state explicitly in the guide

1. Is legacy Hella still the production search alongside dsart today? Unknown.
2. Formal M7.6 cutover (mon-key rename + parity sign-off) completion status. Unknown.
3. SPL exercised on-sky? Unknown.
4. mjd_target=0.0 fix landed? Unknown.
5. dsart_c3.service unit-vs-process mismatch resolution. Unknown.
