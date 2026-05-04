# dada_junkdb runbook (M0 Layer-1 plumbing)

PSRDADA buffer sizes and keys match `configs/config_corr.yaml` (§6.1):

| Buffer | Key   | bytes/block | num_blocks |
|--------|-------|------------:|-----------:|
| dada   | dada  |  94_371_840 |          8 |
| eada   | eada  | 402_653_184 |          4 |
| fada   | fada  |  51_118_080 |          4 |
| bada   | bada  |   1_996_800 |         40 |

Native **block cadence** on the voltage path is **134.218 ms** per §3.1 (4096 native samples per block). Target sustained byte rate into `dada` is therefore:

`94_371_840 B / 0.134218 s ≈ 703 MB/s ≈ 1124 MiB/s` (order-of-magnitude cross-check for `-r` tuning).

## 1. Calibrate what `-r` means on your build

`dada_junkdb -r` is documented as **MB/s** in REALTIME_FRB_SEARCH.md §15.14, but some builds differ (§4.7).

1. Ensure the ring exists (after `cmd: prepare` / `dada_db` create path used on the host).
2. Run a **short** fill at a deliberately low rate, e.g.  
   `dada_junkdb -k dada -r 64 -t 5`
3. In parallel or immediately after, inspect fill / timing with  
   `dada_dbmetric -k dada` (and related `dada_dbmonitor` if available).
4. Compare observed block cadence to **134.218 ms**. Scale `-r` proportionally until `dada_dbmetric` reports full blocks at native cadence.

Record the verified interpretation (MB/s vs Mbit/s) and the working `-r` value **here in ops notes on the host** once calibrated.

## 2. Feed `dada` and `eada` at full native rate

After calibration:

1. Start **two** junk writers (or staggered single writer if your ops pattern serialises half-rings), one bound to `dada` and one to `eada`, each with the calibrated `-r` matching **its** `bytes_per_block / 0.134218 s`.
2. Keep `dada_dbmetric -k dada` / `-k eada` running (or poll periodically) while soaking.

Full-rate soak belongs on **h01** with real `dada_db` rings — not on macOS dev laptops.

## 3. Verify block cadence

Use `dada_dbmetric -k <key>`:

- Full blocks should advance at **~7.45 Hz** (1 / 0.134218 s) per ring when paced at native cadence.
- Compare idle vs loaded states; stalls usually mean `-r` too low, ring not created, or wrong `-r` unit interpretation — return to §1.

## 4. NIC sysctl tunings

Large UDP RX rings (search path) assume §6.1 sysctl headroom. Apply on corr/search hosts with:

`bash tools/ops/sysctl.sh`

(requires NOPASSWD sudoers entries from §6.2.)
