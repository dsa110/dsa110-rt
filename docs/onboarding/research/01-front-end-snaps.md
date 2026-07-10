# Research packet 01 — Analog/digital front end: antennas, SNAPs, channelization, UDP fabric, arming

Sources: `REALTIME_FRB_SEARCH.md` (legacy dsa110-xengine reference) and `dsa110-rt/docs/overview/dsa110-rt-overview.tex` (dsart, 2026-06-10 edition). Citations file:line.

## 1. SNAP boards

- SNAP = FPGA "F-engine" that digitizes/channelizes antenna voltages, streams UDP. Diagram: `SNAP F-engine (96 ant, 4-bit, 2-pol)` — overview.tex:248.
- `NSNAPS = 32` system-wide (dsaX_def.h) — REALTIME_FRB_SEARCH.md:237. Each SNAP carries 3 antennas (`ant_id/3` → SNAP slot) — RFS.md:291. 32×3 = 96 = NANTS (RFS.md:236).
- SNAP UDP packet payload: [3 ants, 384 chans, 2 times, 2 pols, 4-bit complex] = 4608 B — RFS.md:258-260.
- Each corr node runs two capture processes; each ingests 16 SNAPs (half of 32) = 48 antennas ("SNAP pair", NANTS_PER_SNAP_PAIR=48) — RFS.md:261-264; dsa110-rt/configs/config_corr.yaml:8-12.
- Pair A = UDP 4011 → `dada`; pair B = UDP 4012 → `eada` — RFS.md:61-63,275-286; configs/dsart_pipeline_rt.yaml:230,251; naming cap_a_real/cap_b_real overview.tex:2456,3912.
- SNAP → dsart_capture over 40 GbE; capture NUMA-pinned to GPU socket — overview.tex:266,568.
- **GAP (do not invent):** physical location of SNAPs (at antenna vs central hut) is not documented anywhere in these docs. No firmware/board-model details.

## 2. Channelization plan

- Band: f0 = 1.53 GHz top, bw = 250 MHz, nchan = 8192 native, nchan_spw = 384/corr node, npol = 2 — configs/corr_setup_96.yaml:199,220,224-227,236; overview.tex:316-318.
- Native Δν = 250 MHz/8192 = 30.517578125 kHz — overview.tex:318.
- tsamp block = 0.134217728 s = 2048 packets × 65.536 µs (legacy framing) ≡ 4096 native samples × 32.768 µs (new framing). Native sample = 32.768 µs; 1 specnum = 2 native samples = 65.536 µs — overview.tex:199-205,334-337; corr_setup_96.yaml:236.
- 16 chgroups × 384 channels; only 6144 contiguous channels (1024–7167) processed, 2048 edge channels dropped — overview.tex:1249-1272; full per-host ch0 table in configs/chgroup_assignments.yaml:1-70 (chgroup 0 = corr00/h03 ch0=1024 fch_top=1498.75 MHz … chgroup 15 = corr16/h22 ch0=6784 fch_top=1322.96875 MHz); same table RFS.md:49-54.
- Processed band: ν_top = 1.49875 GHz, ν_bot = 1.311280517578125 GHz, BW ≈ 187.47 MHz (small ~10-15 kHz doc-to-doc arithmetic discrepancy, flagged as a "pin" RFS.md:1269-1272; overview.tex:319-327).
- Fast path does 8× channel-sum → 48 chans/chgroup, 768 fleet-wide, Δν_eff = 244.14 kHz — overview.tex:667-671; legacy same via dsaX_dbnic/bfCorr RFS.md:249,459-461.
- 4-bit complex voltages: nibble-packed, low=real high=imag, signed two's complement, ×0.05 scale to fp16 range — RFS.md:922-948; overview.tex:522-527,610-612.
- NANTS=96, NBASE=96×97/2=4656 — RFS.md:241; overview.tex:360-364.

## 3. UDP fabric

- Networks: 10.41.x.x corr-net (SNAP↔corr, corr↔search); 10.42.x.x coincidencer/monitor net — RFS.md:59-70.
- SNAP→corr: UDP 4011/4012; control ports 11223 (cap A)/11224 (cap B) for UTC_START/UTC_STOP pokes — RFS.md:284-289; src/dsart/capture/README.md:49-52; configs/config_corr.yaml:64.
- Corr→search: ports 6625+chgroup; search host data IPs {10.41.0.205,.222,.253,.238} = h01/h02/h09/h13 — RFS.md:65-66,435-437; configs/dsart_search_rt.yaml:92-99.
- Per-corr-node data IP table (identical for both capture ports): configs/dsart_pipeline_rt.yaml:232-247 (lxd110h03=10.41.0.224 … lxd110h22=10.41.0.233); doubles as voltage-dump broadcast address (dsart_search_rt.yaml:680-695).
- NUMA: capture A NUMA0/`dada`, B NUMA1/`eada` (config_corr.yaml:44,50); full core map configs/numa_topology.yaml:1-124 (corr: NUMA0 cpus 0-9,20-29 = slow GPU; NUMA1 cpus 10-19,30-39 = fast GPU; SNAP NICs eth_snap0/1 NUMA0; search NIC eth_search40 MTU 9000 NUMA1).

## 4. utc_start / arming / specnum

- specnum: legacy 44-bit SNAP packet sequence number (seq_no); verb string `UTC_START-<seq>` is a misnomer (not a UTC time) — RFS.md:1244-1247 (dsaX_capture_manythread.c:400-410,567). New doc: 1 specnum = 2 native samples = 65.536 µs — overview.tex:334-337. (16-bit vs 44-bit mentions refer to different things; caveat, don't conflate.)
- Legacy arm gate: `if (seq_no >= UTC_START - 50 && UTC_START != 10000) begin ingest` — RFS.md:1238-1241.
- dsart: deterministic arming to a common specnum fleet-wide; legacy auto-arm (+30000 on first packet) only behind `DSART_CAPTURE_LEGACY_AUTOARM=1` (default OFF) — src/dsart/capture/README.md:15,58.
- Command flow: etcd verb utc_start/utc_stop → UDP `UTC_START-<seq>` to 127.0.0.1:11223 and :11224 — RFS.md:582-584; capture/README.md:50-52; verbs list overview.tex:2884-2887.
- SNAP wall-clock etcd "arm trio" (overview.tex:4389-4422): `/mon/snap/1/utc_start_rt` (new, armed specnum), `/mon/snap/1/armed_mjd` (= now MJD), `/mon/snap/1/utc_start` pinned to 0 (unit mismatch: legacy anchor formula armed_mjd + utc_start*4*8.192e-6/86400 expects native samples; writing raw specnum would be wrong; setting 0 makes anchor = "now") — overview.tex:2802-2830; bug found+fixed 2026-06-02 commit e6ee7cd, matters for meridian_fringestop/dsamfs UVH5 time anchor.
- Trigger-side specnum string `<specnum>-<src_name>-` UDP 11227 — RFS.md:386-394,1125-1141.

## 5. Antennas / layout / pointing

- "The DSA-110 is a 110-element radio interferometer at OVRO operating in the 1.31–1.50 GHz band" — overview.tex:193-194. But pipeline processes 96 "online antennas" (corr_setup_96.yaml:222-223; RFS.md:236). **GAP: 110-vs-96 never reconciled in docs.** Hint (inference only): dsamfs `outrigger_delays: {110:222, 113:1394, 114:3514, 115:5070}` ns — RFS.md:708.
- antenna_order / snap_antenna_order maps: corr_setup_96.yaml:4-197 (index 0→ant 1 … 47→102, 48→116, … 95→115); hard-coded again in dsaX_merge.c ao1/ao2 — RFS.md:311-314,1193-1210.
- Pointing: declination-only drift-scan / meridian-transit. Gridder (u,v) valid at HA=0, v_m = b_n·cos(φ_lat − δ0) — overview.tex:695-707. `/mon/array/dec` (dec_deg) sole live pointing state, written by declination pipeline/service on h23 — overview.tex:4172-4173; RFS.md:609. UVH5 phase_type=drift — RFS.md:770-772. Fast path deliberately NOT fringe-stopped (<0.01% S/N loss at DM=100, ~6% worst-case top of DM range) — overview.tex:706-714,2858-2860.
- Legacy `-g <obs_dec_deg>` via CUSTOMDEC substitution — RFS.md:369-370.
- Antenna positions: local East/North meters in antennas.out; ECEF→EN rotation done by out-of-repo tool — RFS.md:986-1049.

## Gaps to flag (do not invent)
1. SNAP physical location undocumented.
2. 110 vs 96 antennas unreconciled (outriggers inferred, not stated).
3. specnum bit-width: 16 vs 44-bit mentions are different quantities; caveat.
4. No SNAP firmware/vendor, dish size, or numeric OVRO coordinates in these docs.
