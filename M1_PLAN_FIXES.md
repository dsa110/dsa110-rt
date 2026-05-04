# M1 plan-fix checklist

Captured at the start of M1 from the explore-subagent fan-out (2026-05-04). These are
internal inconsistencies / typos / under-specifications in
`/home/ubuntu/vikram/dev/dsa110-rt_revamp_7b1d2669.plan.md` that the M1 author resolved
in code via locked decisions (see "Locked decisions" below). All plan edits are
**deferred to the M1 hardening pass at the end** (Chunk 6) so the M1 critical path is
not blocked by sequential plan-edit + scp + relock cycles.

## Plan-fix items (apply during M1 hardening)

| ID | Section | Fix | Source |
|----|---------|-----|--------|
| F1 | §3 line 310 | Drop `utc_block_start_ns` from `SparseCOOPayload` prose. The §4.3 72-byte layout (lines 1360-1394) is authoritative and does not carry it. | Agent A cross-check |
| F2 | §4.3 line 1384 | `bits_per_cell` enum is `{8, 16}`, not the comment's `"16 or 32"`. Operational quantize is `cint8` (8 bits, §4.2 line 1296). cfp16 debug is 16 bits. | Agent A; D7 lock |
| F3 | §4.3 lines 1382-1383 | Header struct comment for `pattern_id` is abbreviated to 4 inputs `(chgroup, dec, n_grid, K_support)`. Mirror §3 line 307's full 6-input list (adds `antpos_hash`, `chgroup_table_hash`). | Agent A |
| F4 | §3 line 358 vs §4.4 line 1691 | `trigger_id` format is inconsistent: §3 example `s2-000123456` vs §4.4 `s2-g1-000123`. Pin the §4.4 form `s<sid>-g<gh>-<counter>` since per-GPU emitters need to disambiguate; update §3 example to match. | Agent A |
| F5 | §3 lines 320-352 | `Candidate.flags` bit table missing. Bits 3-6 are mentioned piecewise in §4.2/§4.4; bits 0-2 are silent. Add canonical table (see D5 lock below). | Agent A |
| F6 | §3.5 line 649 | Typo: says `Candidate.event_utc_ns` but `Candidate` removed it in O-7. Should be `TriggerPacket.event_utc_ns`. | Agent A |
| F7 | §3.1 line 405 | Path `dsa110-xengine/scripts/config_corr.yaml` does not exist on h23. Real path: `/home/ubuntu/proj/dsa110-shell/dsa110-cnf/config_dsa96_corr.yaml`. The `dsaX_dbnic.hostargs` mapping is in this file. | Agent C |
| F8 | §3.1 line ~418 | Clarify `ν_bot_proc = 1.311265 GHz` convention: it's the band-bottom *centre-of-channel* (`f0 - 7167.5·Δν`), not the lower edge of channel 7167 (which gives `1.311280 GHz`, off by half a channel = 15.26 kHz). | Agent C |
| F9 | §3.1 line ~418 | Note explicitly that `BW_proc = 187.485 MHz` follows from `ν_top_proc - ν_bot_proc` with the F8 centre convention, **NOT** `N_chan_proc · Δν = 187.500 MHz`. The 15-kHz difference matters for sample-shift-table reproducibility. | Agent C |
| F10 | §3.1 line ~446 | `PHI_LAT_OVRO_RAD = 0.64980` is a 5-digit truncation. `math.radians(37.234) = 0.6498508034`. Use full precision in code (D4) and update the plan literal. | Agent C |
| F11 | §3.6.1 lines 715, 723 | `Δτ_ms(1.311265, 1.49875, 3000) = 1697.8 ms` (computed from pinned values + standard formula), NOT `1699.5 ms`. Fix value + adjust `test_dispersion_delay` tolerance window in §3.6.13 to `1697.8 ± 0.5 ms`. | Agent C |

## Locked decisions (M1 author's resolutions)

These are the values M1 code uses. Plan §3/§3.2/§3.5 will be updated in Chunk 6 to
reflect them where the plan was ambiguous.

| ID | Decision | Locked value | Rationale |
|----|----------|--------------|-----------|
| D1 | `Voltages` tensor rank | `[2048, 96, 384, 2, 2]` (5-axis, legacy) | Matches legacy `dsaX_merge.c` output and `fada` block layout. A collapsed `[4096, …, 2]` view requires a real transpose (~5-10 ms / block on GPU HBM), buying nothing — downstream consumers (GEMM, voltage ring, dumper) all expect the 5-axis form. |
| D2 | `Voltages` dtype | accept `int8` or `fp16` | Dataclass is for in-process tensors; int4-on-wire is a transport detail (§3 line 297-298). |
| D3 | NSFRB `gen_dm` extras (trim trials > nsamps; prepend DM=0) | drop both | Plan §3.2 step 1 starts the list with `[dm_min]`, so DM=0 is included when `--dm-min=0`. Trim-by-nsamps is a buffer-sizing concern that's decoupled in our architecture. |
| D4 | `PHI_LAT_OVRO_RAD` precision | `0.6498508034` (full `math.radians(37.234)`) | The pinned `0.64980` is documentation truncation, not the geodetic value. |
| D5 | `Candidate.flags` bit table | bit 0: reserved · bit 1: reserved · bit 2: reserved · bit 3: `noise_warmup` (Layer-2 burn-in) · bit 4: `rfi_warming_up` · bit 5: `halo_dropped` · bit 6: `time_edge_dropped` · bits 7-31: reserved | Matches piecewise mentions in plan §4.2/§4.4. Encoded as `enum.IntFlag` for ergonomics. |
| D6 | `DmPlan.metadata.git_sha` | `dsa110-rt` repo HEAD at build time (`git rev-parse HEAD`) | Tracks "which build of build_dm_plan.py made this .npz" — what most consumers care about. |
| D7 | `SparseCOOPayload.bits_per_cell` | `8` for cint8, `16` for cfp16; enum `{8, 16}` | Matches operational quantize values (§4.2 line 1296). Plan comment "16 or 32" is wrong. |

## Provenance

- **Agent A** — M1 contract dataclass inventory subagent (5aea... 2026-05-04 18:11Z)
- **Agent B** — Legacy DM code survey subagent (5aea... 2026-05-04 18:11Z)
- **Agent C** — Numerical constants cross-check subagent (5aea... 2026-05-04 18:12Z)
- **User decisions** — Q1-Q9 (defaults accepted), D1-D7 (D1 user-revised, others default), P1 (default, plan-fixes batched into Chunk 6).

This file is committed at the start of M1 and consumed by Chunk 6 (hardening). It is
deleted after the plan is patched and re-locked — its job is to prevent the F1-F11
fixes from being lost between chunks.
