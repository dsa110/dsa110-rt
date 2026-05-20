# dsart capture binary

This directory hosts the dsart-vendored SNAP-UDP -> PSRDADA capture
binary used by the M7.5+ on-sky stages of the dsa110-rt pipeline.

## Provenance

Vendored from `dsa110-xengine/src/dsaX_capture_manythread.c` on
**2026-05-20**. Upstream repo is read-only / out-of-tree for the
real-time pipeline; vendoring brings the binary under our CI +
fleet-deploy lifecycle.

| File | Upstream | dsart changes |
|---|---|---|
| `dsart_capture_manythread.c` | `dsaX_capture_manythread.c` | (1) `recvmmsg(64)` batch receive in `recv_thread`; (2) deterministic arming (legacy auto-arm gated behind `DSART_CAPTURE_LEGACY_AUTOARM=1`); (3) explicit `SO_RCVBUF=256 MiB` + read-back; (4) POSIX-shm mon publisher; (5) bug-fixes (`return EXIT_FAILURE` from `void` thread function -> `pthread_exit(NULL)`; `memset(buf, 0, sizeof(buf))` where `buf` is a `char *` -> proper `bufsize`). |
| `dsart_capture_manythread.h` | `dsaX_capture_manythread.h` | renamed types (`dsaX_*` -> `dsart_*`); pruned to symbols actually used. |
| `dsart_capture_def.h` | subset of `dsaX_def.h` | only the SNAP wire-format + topology constants this binary references. |
| `dsart_capture_mon.{h,c}` | (new) | POSIX-shm mon publisher consumed by the Python sidecar `control.py`. |
| `Makefile` | adapted from `dsa110-xengine/src/Makefile` | trimmed to the libs we actually link (`-lpsrdada -lm -lpthread -lrt`); legacy pulled in `-ldedisp -lxgpu -lcudart ...` for unrelated sibling targets in the shared file. |

## Build

```bash
# Standalone (for ad-hoc bench)
cd src/dsart/capture && make

# Via the dsart build_ext (this is what _sync_fleet.sh uses)
python setup.py build_ext --inplace
```

The build needs `libpsrdada` headers under `/usr/local/include/` and
the static lib `/usr/local/lib/libpsrdada.a`. Both are part of the
dsa110-psrdada install on n01..n22.

## Monitoring surface

The binary publishes counters to a POSIX-shm segment named
`/dsart-capture-<UDP_data_port>` (e.g. `/dev/shm/dsart-capture-4011`).
The Python sidecar `dsart.services.capture_control` mmaps this shm
and pushes the counters into etcd under
`/mon/corr_rt/<n>/capture/<port>` at the standard 2 s mon-publisher
cadence.

See `dsart_capture_mon.h` for the wire-pinned shm struct layout and
`src/dsart/services/capture_control.py` for the mon-key map.

## Operator surface

The legacy `UTC_START-<seq>` / `UTC_STOP-<seq>` UDP control messages
are unchanged. dsart-rt routes the etcd `utc_start` / `utc_stop`
verbs through `dsart_rt._send_utc_udp` to the binary's control
thread on ports 11223 (cap_a / dada) and 11224 (cap_b / eada).

## Environment overrides

| Env var | Default | Purpose |
|---|---|---|
| `DSART_CAPTURE_LEGACY_AUTOARM` | `0` | Restore the legacy "auto-arm UTC_START to seq_no+30000 on first packet" behaviour. Default OFF; required for fleet operation where every node must arm to the *same* specnum. |
| `DSART_CAPTURE_SO_RCVBUF` | `268435456` (256 MiB) | Override the `SO_RCVBUF` target. The kernel halves and clamps to `net.core.rmem_max`; the effective granted value is exposed via the mon shm. |
