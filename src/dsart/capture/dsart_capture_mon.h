/* dsart_capture_mon.h --- POSIX-shm monitoring publisher for the
 * dsart capture binary.
 *
 * NEW in dsart (not in legacy dsaX_capture_manythread.c). The legacy
 * binary publishes performance only to syslog; we additionally expose
 * a fixed-shape, atomic-counter shm segment that the Python sidecar
 * (`src/dsart/capture/control.py`) reads at the standard 2 s
 * mon-publisher cadence and pushes into etcd under
 * `/mon/corr_rt/<n>/capture/<port>`.
 *
 * Why shm rather than calling into the etcd Python client directly:
 *   - Keeps the C binary dependency-clean (no libdsa_store, no
 *     etcd3 client linkage; the binary is reusable in isolation).
 *   - The 4 KiB shm page costs us nothing; atomic updates are
 *     lock-free (no recv-thread blocked on a mutex with the publisher).
 *   - Operator can inspect the binary's state without an etcd round
 *     trip by mmaping the shm directly.
 *
 * Shm layout is binary-pinned by struct definitions below. ABI bump
 * requires bumping DSART_CAPTURE_MON_VERSION. Python sidecar
 * (src/dsart/capture/control.py::_MON_VERSION) MUST track this.
 */
#ifndef __DSART_CAPTURE_MON_H
#define __DSART_CAPTURE_MON_H

#include <stdatomic.h>
#include <stdint.h>
#include <sys/types.h>

#define DSART_CAPTURE_MON_VERSION 1u
#define DSART_CAPTURE_MON_MAGIC   0xCA77A1E1u  /* "CAPTAIN-1" */
#define DSART_CAPTURE_MON_SHM_FMT "/dsart-capture-%d"  /* %d = UDP data port */

/* arm_state enum -- matches Python sidecar's interpretation. */
typedef enum {
    DSART_ARM_WAITING_FOR_ARM = 0,
    DSART_ARM_ARMED           = 1,
    DSART_ARM_WRITING         = 2,
    DSART_ARM_STOPPED         = 3,
} dsart_arm_state_t;

/* Fixed shm record. ALL writes from the C binary MUST go through
 * atomic operations (`atomic_store_explicit` or `atomic_fetch_add_explicit`)
 * with at least `memory_order_release`; the sidecar reads with
 * `memory_order_acquire`. The layout has no natural-alignment
 * padding because the 6 uint32 fields total 24 B, which is
 * 8-byte aligned, so the uint64 sequence starts cleanly.
 *
 * Footprint: 208 bytes total (6*4 + 18*8 + 5*8). Fits comfortably
 * in 4 cache lines on x86-64. The Python sidecar pins this in
 * dsart/capture/mon_shm.py::_MON_STRUCT_BYTES.
 */
typedef struct {
    _Atomic uint32_t magic;            /* DSART_CAPTURE_MON_MAGIC */
    _Atomic uint32_t version;          /* DSART_CAPTURE_MON_VERSION */
    _Atomic uint32_t udp_port;         /* data port (4011 / 4012) */
    _Atomic uint32_t control_port;     /* control port (11223 / 11224) */
    _Atomic uint32_t socket_rcvbuf_bytes;  /* getsockopt(SO_RCVBUF) effective value */
    _Atomic uint32_t arm_state;        /* dsart_arm_state_t */

    /* Time / arming. */
    _Atomic uint64_t pid;              /* getpid() of the capture binary */
    _Atomic uint64_t startup_utc_ns;   /* clock_gettime(REALTIME) at startup */
    _Atomic uint64_t last_update_utc_ns; /* updated every ~100 ms by stats_thread */
    _Atomic uint64_t utc_start_specnum; /* armed value (echo of etcd utc_start) */
    _Atomic uint64_t utc_stop_specnum;  /* armed value (echo of etcd utc_stop) */
    _Atomic uint64_t last_seq_no;       /* most-recent specnum seen on the wire */

    /* Cumulative counters since binary startup. */
    _Atomic uint64_t n_recv_packets;    /* total UDP packets received */
    _Atomic uint64_t n_recv_bytes;      /* total UDP payload bytes received */
    _Atomic uint64_t n_dropped_payload; /* tracked by binary (block gaps + too-late) */
    _Atomic uint64_t n_dropped_kernel;  /* SO_RCVBUF / NIC drops (from /proc/net/udp) */
    _Atomic uint64_t n_seq_skipped;     /* gaps in seq_no detected at the recv side */
    _Atomic uint64_t n_too_late;        /* packets that arrived after their block closed */
    _Atomic uint64_t n_wrong_size;      /* recvfrom returned != UDP_PAYLOAD */
    _Atomic uint64_t n_recv_errors;     /* recvmmsg / recvfrom returned -1 with !EAGAIN */
    _Atomic uint64_t n_block_writes;    /* full blocks written to dada/eada */

    /* Rolling-window rates (re-derived by stats_thread every second). */
    _Atomic uint64_t rate_gbps_milli;   /* rx rate in mGb/s (= Gb/s * 1000) */
    _Atomic uint64_t rate_drop_milli;   /* drop rate in mB/s (= bytes/s * 1000) */
    _Atomic uint64_t rate_kernel_drop_pps; /* kernel drops per second */

    /* Reserved for forward-compat schema bumps. Zero-init at startup. */
    _Atomic uint64_t _reserved[5];
} dsart_capture_mon_t;

/* Initialise the shm publisher for ``udp_port``. Creates (or attaches
 * to) the segment "/dsart-capture-<udp_port>", sizes it, and returns
 * a pointer ready to be written by the binary's atomic operations.
 *
 * On failure, logs to syslog and returns NULL; the binary should
 * continue running (mon-only failure is non-fatal). The orchestrator
 * will notice the missing shm via the sidecar's heartbeat / staleness
 * check.
 *
 * The shm is unlinked at process exit via atexit() so an unclean
 * restart does not inherit stale counters.
 */
dsart_capture_mon_t *dsart_capture_mon_open(int udp_port, int control_port);

/* Helper: stamp last_update_utc_ns. Called by stats_thread every
 * ~100 ms (the new tighter loop -- legacy ticked at 1 s). */
void dsart_capture_mon_tick(dsart_capture_mon_t *mon);

/* Helper: atomically transition arm_state. The C binary calls this
 * on UTC_START / UTC_STOP / first-write transitions. */
void dsart_capture_mon_set_arm(dsart_capture_mon_t *mon,
                               dsart_arm_state_t state);

#endif  /* __DSART_CAPTURE_MON_H */
