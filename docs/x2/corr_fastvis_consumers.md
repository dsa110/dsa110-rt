# Consumers of corr_fast's fast visibilities (x2 WS-C, plan §Architecture)

Done before dropping non-core baselines from the 2x fast GEMM. Source: the
code on `x2/main` (2026-10-22), `corr_fast_integration.run`/`process_block`
and every importer of their outputs.

| consumer | reads | 1x today | 2x (`--opoint 2x-*`) |
|---|---|---|---|
| Corner-turn TX (`AsyncTransportTx` → search) | stage-1 + static-sky cube, core cells only | gridded cells of core cross-baselines (the gridder already drops autos and outriggers) | `Corr2xPipeline` → row-rule packer → v2 sender (WS-T); values = cells or redundancy classes of core cross-baselines |
| Static-sky snapshot (`sky_export.SkySnapshotExporter` → h23 `/sky/ingest`, `sky_monitor.py`) | `StaticSkyMean.running_mean_for(0)` + the cells' `(ix_row, ix_col)` | slot 0 (lowest coarse DM) of the gridded cube, sub-band patterns concatenated; h23 `np.add.at`s duplicate cells | tier-A `BatchedStaticSky` slot 0 through `Corr2xPipeline.sky_static` (scaled x2 = 32/16 natives per sample, so the Sky tab's amplitude scale is unchanged); cells from `Corr2xPipeline.sky_cells()` = each value's owner cell from the value map. Per-baseline values land on their class's owner cell at the sub-band centre frequency; h23's `np.add.at` already sums duplicates, so the Sky tab needs no change (natural weighting is preserved: a class value is the sum over its members) |
| RFI flagger / array-burst detector / RFI monitor (`rfi_window`, shm → `/arraysum`, RFI page) | the VOLTAGES (`flag_block(real_v, imag_v)`, `s1_full`) | unchanged | unchanged (same flagger on the same unpacked voltages) |
| On-disk blocks (`--blocks-output-mode`) | `IntegrationOutput.gridded_minus_sky` | `none` in production | not produced in 2x (`gridded_minus_sky=None`) |
| `corr_fast_compute` (M3 spine) | its own GEMM | not deployed | not affected |
| corr_slow / voltage_retention | `fada` directly (own readers) | unchanged | unchanged (reader count unchanged) |

No consumer reads autocorrelations or outrigger baselines from the fast
path: autos for RFI come from the voltages (`s1_full`), and outriggers were
already excluded by the core mask before gridding. Dropping them from the
2x GEMM (82 of 96 antennas, 3321 cross-baselines) therefore removes no
consumer.
