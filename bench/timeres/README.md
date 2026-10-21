# bench/timeres — limited benchmarks for the 4x time-resolution study

Companion to `docs/timeres_study/DSA110_timeres_study.pdf`. Each script firms
up one number the report had to extrapolate. **Run only in an observing gap**:
the GPU scripts refuse to start while any process holds a GPU (`--force`
overrides — don't). Nothing here touches etcd, PSRDADA buffers or the
production services.

| script | where | time | measures | decides |
| --- | --- | --- | --- | --- |
| `b0_selftest.py` | any node GPU | 10 s | prototype Triton kernels vs torch reference | whether b2/b3 "chunk"/e2e rows are valid |
| `b1_fft.py` | search node GPU | ~1 min | irfft2 µs/image at 256², 256×128 (FoV/2), 192², 128²; fp16/fp32 | imaging floor; value of the FoV/2 trade |
| `b2_detector.py` | search node GPU | ~2 min | boxcar detector µs/image: read floor / production per-voxel kernel / prototype chunk-max; K=7 (b64) and K=9 (b256) | detector cost at 262 µs |
| `b3_owner_e2e.py` | search node GPU **and** a corr node GPU | ~3 min | one imaging owner end to end: combine 128 sub-band streams → irfft2 → detector; **max fine DMs one GPU sustains in real time** | the headline feasibility test |
| `b4_xengine.py` | corr node GPU | ~2 min | GEMM rates for the slow product, the fast product at t_int 32/16/8/4, the dedispersing-correlator shape, int8 vs fp16 peak, pinned H2D/D2H | whether one corr GPU can carry slow+fast (freeing the other) |
| `b5_fabric.sh` | from h23 (drives 15 corr + 7 owners) | ~6 min | corner-turn ingress at 7.4 / 12 / 14.8 / 20 Gb/s per owner, incl. corr owners with SNAP background | whether corr nodes can be imaging owners on their PCIe-x4 NICs |
| `b5b_corr_nic.sh` | from h23 | ~9 min | corr-node ingress beside SNAP at 2/4/8 Gb/s, read from the ConnectX counters (`nic_sampler.sh`; the corr `_recv_epoll` is a stale py3.8 build) | corr headroom (measured 2026-10-21: zero, ~17 Gb/s ceiling) |
| `b5c_search_knee.sh` | from h23 | ~9 min | search-node ingress at 6/9/11 Gb/s with NIC counters | search knee (measured: 11 clean, 13 loses 3.7%) |
| `b7_stage1.py` | corr node GPU | ~1 min | stage-1 dedispersion (combine kernel) for the 4x coarse-DM shapes | corr-side 4x budget |

`b5b_analyse.py <dir> ["nodes"]` turns either fabric run into per-window
delivered rate and discard fractions.  Results of the 2026-10-21 gap are in
`docs/timeres_study/bench_results/`.

## How to run (gap of ~30 min)

```bash
# from h23, at this worktree / branch checkout
rsync -a bench/timeres/ n01.pro.pvt:timeres_bench_src/
rsync -a bench/timeres/ n04.pro.pvt:timeres_bench_src/
ssh n01.pro.pvt 'cd timeres_bench_src && bash run_suite.sh 0 1'   # search node, both GPUs
ssh n04.pro.pvt 'cd timeres_bench_src && bash run_suite.sh 0 1'   # corr node, both GPUs
bash bench/timeres/b5_fabric.sh                                    # fabric, from h23
```

`run_suite.sh` writes `~/timeres_bench/<UTC>_<host>/{results.jsonl,summary.txt}`
on the node. The pipeline must be stopped on the nodes being benchmarked; the
fabric test needs every corr + search node idle.

## What the numbers must show for the recommended design (report §5)

* `b3` best `n_fdm_max_realtime` ≥ **44** at 256×128 (DM ≤ 1500, native channels,
  24 owners) — ≥ 78 for DM ≤ 3000. 20–44 means a further ~2× of kernel work
  or tol 1.25 → 1.4 (−23 % trials).
* `b2` chunk ≤ ~1.5 µs/image (256²); prod row = today's design at the new shape.
* `b4`: slow ≤ 10 ms + fast(t_int 8) ≲ 25 ms per 134 ms block on one GPU; an
  int8/fp16 ratio ≥ 2.5 makes an IMMA dedispersing correlator worth building.
* `b5` S1 loss < 1e-4 on the corr owners with the SNAPs streaming → corr GPUs can
  own buckets at cint4 without re-seating NICs; otherwise move the ConnectX-4s
  to x8 slots first (all nodes train at Gen3 x4 today).

The prototype kernels (`kernels.py`) are deliberately simple; they bound what
a production rewrite would reach rather than being one.
