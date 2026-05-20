/* dsart_capture_manythread.c --- SNAP-UDP -> PSRDADA capture binary.
 *
 * Vendored from dsa110-xengine/src/dsaX_capture_manythread.c on
 * 2026-05-20 (legacy provenance pinned in src/dsart/capture/README.md).
 *
 * dsart improvements over the upstream legacy ("Section X" refs are
 * to the dsa110-rt revamp plan doc):
 *
 *   1. recvmmsg(64) batch receive (replaces per-packet recvfrom),
 *      amortising the syscall cost ~64x.
 *   2. Deterministic arming: drop the legacy "if UTC_START==0 then
 *      seq_no+30000" auto-arm; require an explicit etcd-driven
 *      `UTC_START-<seq>` message before writing to dada/eada. The
 *      DSART_CAPTURE_LEGACY_AUTOARM=1 env var restores the legacy
 *      auto-arm behaviour for offline / single-node testing.
 *   3. Explicit SO_RCVBUF (256 MiB target; reads back the granted
 *      value and reports it via the mon shm so the sidecar can
 *      gate on sysctl regressions at startup).
 *   4. POSIX-shm mon publisher (`dsart_capture_mon.h`); the Python
 *      sidecar in `src/dsart/capture/control.py` snapshots this at
 *      the standard 2 s mon-publisher cadence into
 *      /mon/corr_rt/<n>/capture/<port>/ keys.
 *
 * Bug fixes during the vendor:
 *   - `return EXIT_FAILURE;` inside a `void recv_thread(void *arg)`
 *     replaced with `pthread_exit(NULL);` (UB in the legacy).
 *   - `memset(buffer,'\0',sizeof(buffer))` where `buffer` is a `char*`
 *     replaced with a proper `bufsize` constant. Legacy zero-cleared
 *     only 8 bytes of a 1024-byte buffer, which can confuse the
 *     strtok parsing if a previous longer message contained dashes.
 *
 * Wire format (UDP packet layout, 8 B header + 4608 B payload) is
 * unchanged. SNAP firmware is an explicit non-goal of this plan.
 */
#ifndef __USE_GNU
#define __USE_GNU
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <stdatomic.h>
#include <signal.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <syslog.h>
#include <errno.h>

#include <sock.h>
#include <tmutil.h>
#include <dada_client.h>
#include <dada_def.h>
#include <dada_hdu.h>
#include <ipcio.h>
#include <ipcbuf.h>
#include <dada_affinity.h>
#include <ascii_header.h>

#include "dsart_capture_manythread.h"
#include "dsart_capture_def.h"
#include "dsart_capture_mon.h"

/* ---- recvmmsg batching --------------------------------------------- */
/* Each recv thread does one recvmmsg() syscall for VLEN packets at a
 * time. 64 is the same value our search-side recv_epoll.c uses; it
 * hits the kernel's sweet spot of "amortised syscall cost negligible
 * vs scheduler-noise tolerance for one-by-one packet processing." */
#define DSART_RECVMMSG_VLEN 64

/* ---- global state -------------------------------------------------- */
int      dPort, cPort;
/* DSART: volatile so the worker threads pick up the signal-handler
 * store without -O2's value-caching optimisations latching the
 * loaded value in a register. (Legacy upstream missed this, but
 * got away with it because the syslog() calls in its loops act as
 * compiler barriers; we removed several of those when adding the
 * recvmmsg batching path, so the volatile guard is now load-bearing.) */
volatile sig_atomic_t quit_threads = 0;
char     STATE[20];
uint64_t UTC_START = 10000;   /* sentinel: 10000 = "no arm yet" */
uint64_t UTC_STOP  = 40000000000ULL;
int      MONITOR = 0;
char     iP[100];
int      DEBUG = 0;
int      writeBlock = 0;
const int nth  = 4;
const int nwth = 2;
int      cores[16] = {10,12,11,13,30,31,32,33};
int      write_cores[8] = {14,15,34,35};
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int blockStatus[64];
volatile int skipBlock = 0;
volatile int skipping = 0;
volatile int lWriteBlock = 0;
volatile int write_ct = 0;
volatile uint64_t last_seq = 0;
volatile int skipct = 0;
volatile uint64_t block_count = 0;
volatile uint64_t block_start_byte = 0, block_end_byte = 0;
volatile unsigned capture_started = 0;
volatile char *wblock;

/* DSART: shared shm mon publisher pointer (NULL = mon failed to open;
 * binary continues, sidecar reports the gap via heartbeat). */
static dsart_capture_mon_t *MON = NULL;

/* DSART: env-var escape hatch to restore the legacy auto-arm. The
 * autoarm sets UTC_START to seq_no+30000 on first packet receipt, so
 * the binary will start writing without an external arming message.
 * Useful for single-node tests; in fleet operation we want every
 * node armed to the SAME specnum, so default is OFF. */
static int LEGACY_AUTOARM = 0;

/* DSART: SIGTERM/SIGINT handler -- sets quit_threads = 1 so the main
 * loop falls through to the join+cleanup path, which in turn runs
 * the atexit handler that unlinks /dev/shm/dsart-capture-<port>.
 * The legacy upstream relied on default SIGTERM termination (kernel
 * kills the process without running atexit), which left stale shm
 * segments around. */
static void dsart_on_signal(int sig)
{
    quit_threads = 1;
    /* async-signal-safe write to stderr so the operator sees the
     * shutdown trigger even if syslog is buffered. Avoid syslog()
     * here -- it is NOT async-signal-safe. */
    static const char msg[] = "[capture] caught signal; shutting down\n";
    (void)!write(2, msg, sizeof(msg) - 1);
    (void)sig;
}

void usage(void)
{
    fprintf(stdout,
        "dsart_capture_manythread [options]\n"
        " -c core   bind process to CPU core [no default]\n"
        " -j IP to listen on for data packets [no default]\n"
        " -i IP to listen on for control commands [no default]\n"
        " -p PORT for data\n"
        " -q PORT for control\n"
        " -f filename of template dada header [no default]\n"
        " -o out_key [default CAPTURE_BLOCK_KEY (=0x0000dada)]\n"
        " -k out_key (alias for -o; matches dsart-rt YAML convention)\n"
        " -d send debug messages to syslog\n"
        " -g chgroup [default 0]\n"
        " -h print usage\n"
        "\n"
        "Environment overrides:\n"
        "  DSART_CAPTURE_LEGACY_AUTOARM=1 restore the legacy "
        "self-arm-on-first-packet behaviour. Default OFF for fleet\n"
        "  operation -- write to dada/eada blocks does not start "
        "until the orchestrator sends a UTC_START-<seq> message.\n"
        "  DSART_CAPTURE_SO_RCVBUF=<bytes> override the SO_RCVBUF "
        "target (default 256 MiB).\n");
}

static void dsart_dbgpu_cleanup(dada_hdu_t *out)
{
    if (dada_hdu_unlock_write(out) < 0) {
        syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
    dada_hdu_destroy(out);
}

/* ---- socket setup -------------------------------------------------- */
/* DSART: target SO_RCVBUF size. The legacy binary used 64 MiB; our
 * sysctls expose 256 MiB via net.core.rmem_max. Explicitly request
 * the larger size and read back the granted value (the kernel halves
 * the request once and clamps to rmem_max; we report the effective
 * value via mon so the sidecar can flag a sysctl regression). */
#define DSART_TARGET_SO_RCVBUF_BYTES (256 * 1024 * 1024)

static dsart_sock_t *dsart_make_sock(dsart_udpdb_t *ctx)
{
    syslog(LOG_INFO, "dsart_make_sock(): preparing sock structure");
    dsart_sock_t *b = (dsart_sock_t *)malloc(sizeof(dsart_sock_t));
    assert(b != NULL);
    b->bufsz = sizeof(char) * UDP_PAYLOAD;
    b->buf = (char *)malloc(b->bufsz);
    assert(b->buf != NULL);
    b->have_packet = 0;
    b->fd = 0;

    syslog(LOG_INFO, "prepare: creating udp socket on %s:%d", ctx->interface, dPort);
    b->fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    assert(b->fd >= 0);

    int one = 1;
    setsockopt(b->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &one, sizeof(one));

    struct sockaddr_in udp_sock;
    memset(&udp_sock, 0, sizeof(udp_sock));
    udp_sock.sin_family = AF_INET;
    udp_sock.sin_port = htons(dPort);
    udp_sock.sin_addr.s_addr = inet_addr(ctx->interface);

    if (bind(b->fd, (struct sockaddr *)&udp_sock, sizeof(udp_sock)) == -1) {
        syslog(LOG_ERR, "prepare: failed to bind to socket on %s:%d: %s",
               ctx->interface, dPort, strerror(errno));
        free(b->buf);
        free(b);
        return NULL;
    }

    /* DSART: explicit SO_RCVBUF + read-back. */
    int target = DSART_TARGET_SO_RCVBUF_BYTES;
    const char *env = getenv("DSART_CAPTURE_SO_RCVBUF");
    if (env != NULL) {
        long v = strtol(env, NULL, 10);
        if (v > 0) target = (int)v;
    }
    if (setsockopt(b->fd, SOL_SOCKET, SO_RCVBUF, &target, sizeof(target)) != 0) {
        syslog(LOG_WARNING,
               "prepare: setsockopt(SO_RCVBUF=%d) failed: %s (continuing)",
               target, strerror(errno));
    }
    int granted = 0;
    socklen_t glen = sizeof(granted);
    if (getsockopt(b->fd, SOL_SOCKET, SO_RCVBUF, &granted, &glen) == 0) {
        syslog(LOG_INFO,
               "prepare: SO_RCVBUF target=%d granted=%d (kernel halves+clamps)",
               target, granted);
        if (MON != NULL && ctx->thread_id == 0) {
            atomic_store_explicit(&MON->socket_rcvbuf_bytes,
                                  (uint32_t)granted,
                                  memory_order_release);
        }
    }

    syslog(LOG_INFO, "prepare: setting non_block");
    sock_nonblock(b->fd);

    syslog(LOG_INFO, "prepare: clearing packets at socket");
    (void)dada_sock_clear_buffered_packets(b->fd, UDP_PAYLOAD);

    for (int i = 0; i < 64; i++) blockStatus[i] = 0;

    return b;
}

static void dsart_free_sock(dsart_sock_t *b)
{
    if (!b) return;
    b->fd = 0;
    b->bufsz = 0;
    b->have_packet = 0;
    if (b->buf) free(b->buf);
    b->buf = 0;
    free(b);
}

/* ---- PSRDADA block management (unchanged from legacy) --------------- */
static int dsart_open_buffer(dsart_write_t *ctx)
{
    if (ctx->block_open) {
        syslog(LOG_ERR, "open_buffer: buffer already opened");
        return -1;
    }
    uint64_t block_id = 0;
    wblock = ipcio_open_block_write(ctx->hdu->data_block, &block_id);
    if (!wblock) {
        syslog(LOG_ERR, "open_buffer: ipcio_open_block_write failed");
        return -1;
    }
    ctx->block_open = 1;
    return 0;
}

static int dsart_close_buffer(dsart_write_t *ctx, uint64_t bytes_written, unsigned eod)
{
    if (!ctx->block_open) {
        syslog(LOG_ERR, "close_buffer: buffer already closed");
        return -1;
    }
    if ((bytes_written != 1) && (bytes_written != ctx->hdu_bufsz)) {
        syslog((eod ? LOG_INFO : LOG_WARNING),
               "close_buffer: bytes_written[%" PRIu64 "] != hdu_bufsz[%" PRIu64 "]",
               bytes_written, ctx->hdu_bufsz);
    }
    if (eod) {
        if (ipcio_update_block_write(ctx->hdu->data_block, bytes_written) < 0) {
            syslog(LOG_ERR, "close_buffer: ipcio_update_block_write failed");
            return -1;
        }
    } else {
        if (ipcio_close_block_write(ctx->hdu->data_block, bytes_written) < 0) {
            syslog(LOG_ERR, "close_buffer: ipcio_close_block_write failed");
            return -1;
        }
    }
    wblock = 0;
    ctx->block_open = 0;
    return 0;
}

static int dsart_new_buffer(dsart_write_t *ctx)
{
    if (dsart_close_buffer(ctx, ctx->hdu_bufsz, 0) < 0) {
        syslog(LOG_ERR, "new_buffer: dsart_close_buffer failed");
        return -1;
    }
    if (dsart_open_buffer(ctx) < 0) {
        syslog(LOG_ERR, "new_buffer: dsart_open_buffer failed");
        return -1;
    }
    return 0;
}

static void dsart_udpdb_increment(dsart_udpdb_t *ctx)
{
    writeBlock++;
    block_start_byte = block_end_byte + UDP_DATA;
    block_end_byte = block_start_byte + (ctx->packets_per_buffer - 1) * UDP_DATA;
    block_count = 0;
}

/* ---- /proc/net/udp parser for kernel drop counter ------------------ */
/* DSART: stats_thread polls this once per second. Returns the
 * cumulative "drops" column from the /proc/net/udp row whose
 * local_address ends in :<port>. Falls back to 0 on parse failure.
 *
 * /proc/net/udp format:
 *  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops
 *  0:  0100007F:0FAB 00000000:0000 07 00000000:00000000 00:00000000 00000000  1000        0 12345 2 ffff...    42
 *
 * Column 12 (0-indexed) is "drops". The `:<port>` we match in column 1
 * is a hex 4-digit field, so we render the target port as hex with
 * leading zeros.
 */
static uint64_t read_kernel_drops_for_port(int port)
{
    FILE *f = fopen("/proc/net/udp", "r");
    if (!f) return 0ULL;

    char line[512];
    /* Skip header. */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0ULL;
    }

    char hex_port[8];
    snprintf(hex_port, sizeof(hex_port), ":%04X", port);

    uint64_t total = 0;
    while (fgets(line, sizeof(line), f)) {
        /* Quick filter: line must contain ":<hex_port> " somewhere
         * near the start (in the local_address column). */
        if (!strstr(line, hex_port)) continue;
        /* Parse: skip column 0 (sl:); column 1 = local_address; we
         * want column 12 (drops). */
        char *p = line;
        int field = 0;
        char *tok;
        while ((tok = strtok_r(p, " \t\n", &p)) != NULL) {
            if (field == 12) {
                total += (uint64_t)strtoull(tok, NULL, 10);
                break;
            }
            field++;
        }
    }
    fclose(f);
    return total;
}

/* ---- THREADS ------------------------------------------------------- */

/* STATS THREAD --- ticks every ~100 ms; updates mon shm; logs CAPSTATS
 * once per second (preserves the legacy syslog operator-visible line). */
void stats_thread(void *arg)
{
    dsart_stats_t *ctx = (dsart_stats_t *)arg;
    uint64_t b_rcv_total = 0;
    uint64_t b_drp_total = 0;
    uint64_t kernel_drops_last = 0;
    int tick = 0;
    int target_ticks_per_logline = 10;  /* 10 * 100 ms = 1 s */

    syslog(LOG_INFO, "stats_thread: starting (100 ms cadence)");
    sleep(2);  /* let the recv threads come up */
    syslog(LOG_INFO, "stats_thread: running");

    while (!quit_threads) {
        uint64_t b_rcv_curr = ctx->bytes->received;
        uint64_t b_drp_curr = ctx->bytes->dropped;
        uint64_t pkt_drp = ctx->packets->dropped;

        /* DSART: update mon shm.
         *
         * NOTE: n_recv_packets / n_recv_bytes are *not* set here --
         * the recv_thread increments them atomically per packet for
         * a low-latency live counter. The legacy block-complete
         * counters (`ctx->bytes->received`, `ctx->packets->received`)
         * are used only for the CAPSTATS syslog line below.
         *
         * n_dropped_payload is here (it's a block-complete sum), and
         * last_seq_no / n_seq_skipped / n_block_writes are stamped
         * here too because they're cheap to read once per 100 ms.
         */
        if (MON != NULL) {
            atomic_store_explicit(&MON->n_dropped_payload, pkt_drp,
                                  memory_order_release);
            atomic_store_explicit(&MON->n_seq_skipped, (uint64_t)skipct,
                                  memory_order_release);
            atomic_store_explicit(&MON->n_block_writes, (uint64_t)writeBlock,
                                  memory_order_release);
            /* NOTE: last_seq_no is stamped by recv_thread on every
             * packet -- see the atomic_store there. We deliberately
             * don't stamp it here because (a) the recv side already
             * stamps it at packet rate, and (b) the legacy `last_seq`
             * volatile is 0-initialised, so reading it here would
             * race with recv_thread's first-packet update during
             * the first stats_thread tick. */
            dsart_capture_mon_tick(MON);
        }

        if (tick % target_ticks_per_logline == 0) {
            uint64_t b_rcv_1sec = b_rcv_curr - b_rcv_total;
            uint64_t b_drp_1sec = b_drp_curr - b_drp_total;
            b_rcv_total = b_rcv_curr;
            b_drp_total = b_drp_curr;

            double mb_drp_ps = (double)b_drp_1sec / 1000000.0;
            double gb_rcv_ps = (double)b_rcv_1sec * 8.0 / 1000000000.0;

            /* DSART: kernel drop counter (cumulative -> per-second). */
            uint64_t kernel_now = read_kernel_drops_for_port(dPort);
            uint64_t kernel_1sec = (kernel_now >= kernel_drops_last)
                                   ? (kernel_now - kernel_drops_last) : 0;
            kernel_drops_last = kernel_now;
            if (MON != NULL) {
                atomic_store_explicit(&MON->n_dropped_kernel, kernel_now,
                                      memory_order_release);
                atomic_store_explicit(&MON->rate_kernel_drop_pps, kernel_1sec,
                                      memory_order_release);
                atomic_store_explicit(&MON->rate_gbps_milli,
                                      (uint64_t)(gb_rcv_ps * 1000.0),
                                      memory_order_release);
                atomic_store_explicit(&MON->rate_drop_milli,
                                      (uint64_t)(mb_drp_ps * 1000.0),
                                      memory_order_release);
            }

            syslog(LOG_NOTICE,
                   "CAPSTATS %6.3f [Gb/s], D %4.1f [MB/s], "
                   "D %" PRIu64 " pkts, last_seq=%" PRIu64 ", "
                   "skipped %d, kernel_drops=%" PRIu64 " (+%" PRIu64 "/s)",
                   gb_rcv_ps, mb_drp_ps, pkt_drp, last_seq, skipct,
                   kernel_now, kernel_1sec);
        }

        usleep(100 * 1000);  /* 100 ms */
        tick++;
    }
}

/* CONTROL THREAD --- receives UTC_START / UTC_STOP / MONITOR messages
 * on a UDP socket. Each message is a single string of the form
 * "UTC_START-<seq>" / "UTC_STOP-<seq>" / "MONITOR-<n>".
 *
 * The orchestrator (dsart_rt.py::_verb_utc_start) sends these by
 * routing the legacy `utc_start` etcd verb to UDP 127.0.0.1:<cPort>.
 * Format is unchanged from upstream. */
void control_thread(void *arg)
{
    (void)arg;
    syslog(LOG_INFO, "control_thread: starting");

    int port = cPort;
    char sport[10];
    snprintf(sport, sizeof(sport), "%d", port);

    const int bufsize = 1024;
    char *buffer = (char *)malloc(sizeof(char) * bufsize);
    memset(buffer, '\0', (size_t)bufsize);

    struct addrinfo hints;
    struct addrinfo *res = 0;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    if (getaddrinfo(iP, sport, &hints, &res) != 0) {
        syslog(LOG_ERR, "control_thread: getaddrinfo(%s:%s) failed", iP, sport);
        free(buffer);
        pthread_exit(NULL);
    }

    int fd = -1;
    struct sockaddr_storage src_addr;
    socklen_t src_addr_len = sizeof(src_addr);
    char *endptr;

    syslog(LOG_INFO, "control_thread: created socket on port %d", port);

    while (!quit_threads) {
        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) {
            syslog(LOG_ERR, "control_thread: socket() failed: %s",
                   strerror(errno));
            sleep(1);
            continue;
        }
        if (bind(fd, res->ai_addr, res->ai_addrlen) < 0) {
            syslog(LOG_ERR, "control_thread: bind(%d) failed: %s",
                   port, strerror(errno));
            close(fd);
            sleep(1);
            continue;
        }
        /* DSART: SO_RCVTIMEO so the thread wakes up periodically and
         * gets a chance to observe quit_threads on shutdown. Without
         * this the join() hangs forever on SIGTERM. 500 ms is a
         * good balance between shutdown latency and idle wakeups. */
        struct timeval rcv_to;
        rcv_to.tv_sec  = 0;
        rcv_to.tv_usec = 500 * 1000;
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                       &rcv_to, sizeof(rcv_to)) != 0) {
            syslog(LOG_WARNING,
                   "control_thread: setsockopt(SO_RCVTIMEO) failed: %s",
                   strerror(errno));
        }
        /* DSART: full bufsize zero, not sizeof(char*) = 8. */
        memset(buffer, '\0', (size_t)bufsize);
        ssize_t ct = recvfrom(fd, buffer, (size_t)(bufsize - 1), 0,
                              (struct sockaddr *)&src_addr, &src_addr_len);
        if (ct < 0) {
            int errsv = errno;
            close(fd);
            if (errsv == EAGAIN || errsv == EWOULDBLOCK) {
                continue;  /* idle timeout; loop to re-check quit_threads */
            }
            syslog(LOG_WARNING, "control_thread: recvfrom failed: %s",
                   strerror(errsv));
            continue;
        }
        buffer[ct] = '\0';

        syslog(LOG_INFO, "control_thread: received buffer string %s", buffer);

        char *rest = buffer;
        char *cmd = strtok_r(rest, "-", &rest);
        char *val = strtok_r(rest, "-", &rest);
        if (cmd == NULL || val == NULL) {
            syslog(LOG_WARNING, "control_thread: malformed message %s", buffer);
            close(fd);
            continue;
        }
        syslog(LOG_INFO,
               "control_thread: split into COMMAND %s, VALUE %s", cmd, val);

        if (strcmp(cmd, "UTC_START") == 0) {
            UTC_START = strtoull(val, &endptr, 0);
            if (MON != NULL) {
                atomic_store_explicit(&MON->utc_start_specnum, UTC_START,
                                      memory_order_release);
                dsart_capture_mon_set_arm(MON, DSART_ARM_ARMED);
            }
            syslog(LOG_NOTICE,
                   "control_thread: ARMED UTC_START=%" PRIu64, UTC_START);
        }
        if (strcmp(cmd, "UTC_STOP") == 0) {
            UTC_STOP = strtoull(val, &endptr, 0);
            if (MON != NULL) {
                atomic_store_explicit(&MON->utc_stop_specnum, UTC_STOP,
                                      memory_order_release);
            }
            syslog(LOG_NOTICE,
                   "control_thread: ARMED UTC_STOP=%" PRIu64, UTC_STOP);
        }
        close(fd);
    }

    free(buffer);
    if (res) freeaddrinfo(res);
    syslog(LOG_INFO, "control_thread: exiting");
    pthread_exit(NULL);
}

/* RECV THREAD --- recvmmsg(VLEN) -> per-packet processing. */
void recv_thread(void *arg)
{
    dsart_udpdb_t *udpdb = (dsart_udpdb_t *)arg;
    int thread_id = udpdb->thread_id;

    /* set affinity (legacy) */
    const pthread_t pid = pthread_self();
    int core_id = (dPort == 4011)
                  ? cores[thread_id]
                  : cores[thread_id + nth];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (set_result != 0)
        syslog(LOG_ERR, "recv_thread %d: setaffinity_np fail core_id=%d",
               thread_id, core_id);

    dsart_sock_t *sock = dsart_make_sock(udpdb);
    if (sock == NULL) {
        syslog(LOG_ERR, "recv_thread %d: dsart_make_sock failed -- exiting",
               thread_id);
        pthread_exit(NULL);
    }

    /* SNAP antenna-id lookup table (legacy convention). */
    uint64_t ant_lookup[100];
    for (int i = 0; i < 100; i++) ant_lookup[i] = 0;
    for (int i = 0; i < NSNAPS / 2; i++) {
        for (int j = 0; j < 2; j++) {
            uint64_t vv = (uint64_t)((i * 2 + j) * 3);
            ant_lookup[vv] = (uint64_t)i;
        }
    }

    /* DSART: recvmmsg batch buffers. */
    struct mmsghdr msgs[DSART_RECVMMSG_VLEN];
    struct iovec   iovs[DSART_RECVMMSG_VLEN];
    char *bufs = (char *)malloc((size_t)DSART_RECVMMSG_VLEN * UDP_PAYLOAD);
    assert(bufs != NULL);
    for (int i = 0; i < DSART_RECVMMSG_VLEN; i++) {
        iovs[i].iov_base = bufs + (size_t)i * UDP_PAYLOAD;
        iovs[i].iov_len  = UDP_PAYLOAD;
        memset(&msgs[i].msg_hdr, 0, sizeof(msgs[i].msg_hdr));
        msgs[i].msg_hdr.msg_iov    = &iovs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    /* per-packet processing state (kept across batches). */
    uint64_t tpack = 0;
    uint64_t act_seq_no = 0;
    uint64_t block_seq_no = 0;
    uint64_t seq_no = 0, prev_seq_no = 0;
    uint64_t ant_id = 0, aid;
    int64_t  byte_offset = 0;
    uint64_t seq_byte = 0;
    /* "saved" out-of-order packets near block edges. */
    unsigned int temp_idx = 0;
    unsigned int temp_max = 500;
    char **temp_buffers = (char **)malloc(sizeof(char *) * temp_max);
    for (unsigned int i = 0; i < temp_max; i++)
        temp_buffers[i] = (char *)malloc(sizeof(char) * UDP_DATA);
    uint64_t *temp_seq_byte = (uint64_t *)malloc(sizeof(uint64_t) * temp_max);
    uint64_t timeouts = 0;
    int canWrite = 0;
    int mod_WB;
    int local_seq_skipped = 0;

    while (!quit_threads) {

        /* DSART: batch-receive. Returns the number of packets received,
         * or -1 with errno indicating why. */
        int got_n = recvmmsg(sock->fd, msgs, DSART_RECVMMSG_VLEN, 0, NULL);

        if (got_n < 0) {
            int errsv = errno;
            if (errsv == EAGAIN || errsv == EWOULDBLOCK) {
                if (capture_started) timeouts++;
                continue;
            }
            /* DSART: don't bail out of a void thread function. */
            syslog(LOG_WARNING,
                   "recv_thread %d: recvmmsg failed: %s",
                   thread_id, strerror(errsv));
            if (MON != NULL)
                atomic_fetch_add_explicit(&MON->n_recv_errors, 1,
                                          memory_order_release);
            continue;
        }
        timeouts = 0;

        for (int p = 0; p < got_n; p++) {
            unsigned int got = (unsigned int)msgs[p].msg_len;
            if (got != UDP_PAYLOAD) {
                if (MON != NULL)
                    atomic_fetch_add_explicit(&MON->n_wrong_size, 1,
                                              memory_order_release);
                syslog(LOG_NOTICE,
                       "recv_thread %d: short packet (%u bytes, expected %d)",
                       thread_id, got, UDP_PAYLOAD);
                continue;
            }

            /* DSART: per-packet live counters. The legacy block-
             * complete counters (n_recv_packets / n_recv_bytes via
             * stats_thread snapshot) lag by up to one full block
             * (~2048 packets, ~130 ms), which is too coarse for the
             * operator-facing live-monitor view. These atomic adds
             * cost ~5ns each = ~0.04% CPU overhead at production
             * rates (17 k pps / port). */
            if (MON != NULL) {
                atomic_fetch_add_explicit(&MON->n_recv_packets, 1,
                                          memory_order_release);
                atomic_fetch_add_explicit(&MON->n_recv_bytes,
                                          (uint64_t)UDP_DATA,
                                          memory_order_release);
            }

            unsigned char *pktbuf =
                (unsigned char *)(bufs + (size_t)p * UDP_PAYLOAD);

            /* decode packet header (64 bits)
             *  35 bits seq_no | 13 bits ch_id | 16 bits ant ID
             * (legacy bit shuffle preserved verbatim) */
            seq_no = 0;
            seq_no |= (((uint64_t)(pktbuf[4]) & 224) >> 5) & 7;
            seq_no |= (((uint64_t)(pktbuf[3])) << 3)  & 2040;
            seq_no |= (((uint64_t)(pktbuf[2])) << 11) & 522240;
            seq_no |= (((uint64_t)(pktbuf[1])) << 19) & 133693440;
            seq_no |= (((uint64_t)(pktbuf[0])) << 27) & 34225520640ULL;
            ant_id = 0;
            ant_id |= (unsigned char)(pktbuf[6]) << 8;
            ant_id |= (unsigned char)(pktbuf[7]);
            aid = ant_lookup[(int)ant_id];

            /* DSART: only auto-arm when explicitly enabled via env var.
             * Default (LEGACY_AUTOARM=0) requires an explicit
             * UTC_START-<seq> control message before writing begins. */
            if (LEGACY_AUTOARM && UTC_START == 10000) {
                UTC_START = seq_no + 30000;
                if (MON != NULL) {
                    atomic_store_explicit(&MON->utc_start_specnum,
                                          UTC_START,
                                          memory_order_release);
                    dsart_capture_mon_set_arm(MON, DSART_ARM_ARMED);
                }
                syslog(LOG_NOTICE,
                       "recv_thread %d: LEGACY_AUTOARM engaged, "
                       "UTC_START=%" PRIu64, thread_id, UTC_START);
            }

            act_seq_no = seq_no * NSNAPS / 4 + aid;
            block_seq_no = UTC_START * NSNAPS / 4;

            /* shared last_seq + seq-skip detector */
            pthread_mutex_lock(&mutex);
            if (last_seq > 0 && seq_no > last_seq + 1) {
                local_seq_skipped += (int)(seq_no - last_seq - 1);
            }
            prev_seq_no = last_seq;
            last_seq = seq_no;
            pthread_mutex_unlock(&mutex);
            (void)prev_seq_no;

            /* DSART: stamp last_seq_no into mon shm here, not in the
             * stats_thread, so the operator's "arm against latest
             * specnum" workflow never races the stats_thread's 2 s
             * startup sleep. Without this, the first sample of
             * last_seq_no via the shm reads the initial 0 even
             * though the recv threads have been draining packets
             * for hundreds of ms -- the operator computes
             * `UTC_START = 0 + offset`, which is far in the past,
             * and the binary spends 5 s catching block_start_byte
             * up to actual wire position via the temp_buffers
             * max-overflow path. (Observed live on n06: 2465 fake
             * block-completes before steady state.)
             *
             * Atomic store cost: ~1 ns / packet. At production
             * 244 k pps / port this is ~0.024% CPU per thread. */
            if (MON != NULL) {
                atomic_store_explicit(&MON->last_seq_no, seq_no,
                                      memory_order_release);
            }

            /* start-condition gate (unchanged from legacy other than
             * the deterministic-arm tightening on UTC_START sentinel) */
            if (canWrite == 0) {
                if (UTC_START != 10000 && seq_no >= UTC_START - 50) {
                    canWrite = 1;
                }
            }
            if (canWrite == 0) continue;

            /* threadsafe start of capture (unchanged) */
            pthread_mutex_lock(&mutex);
            if (!capture_started) {
                block_start_byte = block_seq_no * UDP_DATA;
                block_end_byte = (block_start_byte + udpdb->hdu_bufsz) - UDP_DATA;
                capture_started = 1;
                if (MON != NULL)
                    dsart_capture_mon_set_arm(MON, DSART_ARM_WRITING);
                syslog(LOG_INFO,
                       "recv_thread %d: START [%" PRIu64 " - %" PRIu64 "]",
                       thread_id, block_start_byte, block_end_byte);
            }
            pthread_mutex_unlock(&mutex);

            if (capture_started) {
                seq_byte = act_seq_no * UDP_DATA;
                tpack++;

                if ((seq_byte <= block_end_byte) && (seq_byte >= block_start_byte)) {
                    byte_offset = seq_byte - block_start_byte;
                    mod_WB = writeBlock % 64;
                    memcpy(udpdb->tblock + byte_offset + mod_WB * udpdb->hdu_bufsz,
                           pktbuf + UDP_HEADER, UDP_DATA);
                    pthread_mutex_lock(&mutex);
                    block_count++;
                    pthread_mutex_unlock(&mutex);
                } else if (seq_byte > block_end_byte) {
                    if (temp_idx < temp_max) {
                        memcpy(temp_buffers[temp_idx],
                               pktbuf + UDP_HEADER, UDP_DATA);
                        temp_seq_byte[temp_idx] = seq_byte;
                        temp_idx++;
                    }
                } else {
                    if (MON != NULL)
                        atomic_fetch_add_explicit(&MON->n_too_late, 1,
                                                  memory_order_release);
                }
            }

            /* threadsafe end of block (unchanged from legacy) */
            pthread_mutex_lock(&mutex);
            if ((block_count >= udpdb->packets_per_buffer)
                || (temp_idx >= temp_max)) {
                syslog(LOG_INFO,
                       "BLOCK COMPLETE thread_id=%d seq_no=%" PRIu64
                       " ant_id=%" PRIu64 " block_count=%" PRIu64
                       " temp_idx=%u writeBlock=%d",
                       thread_id, seq_no, ant_id, block_count,
                       temp_idx, writeBlock);

                if (blockStatus[writeBlock % 64] > 0)
                    blockStatus[writeBlock % 64] += 1;
                else
                    blockStatus[writeBlock % 64] = 1;

                uint64_t dropped = udpdb->packets_per_buffer - block_count;
                udpdb->packets->received += block_count;
                udpdb->bytes->received += block_count * UDP_DATA;
                if (dropped) {
                    udpdb->packets->dropped += dropped;
                    udpdb->bytes->dropped += dropped * UDP_DATA;
                }

                dsart_udpdb_increment(udpdb);
                tpack = 0;

                /* DSART: capture the local_seq_skipped accumulator. */
                if (local_seq_skipped > 0) {
                    skipct += local_seq_skipped;
                    local_seq_skipped = 0;
                }

                for (unsigned int i = 0; i < temp_idx; i++) {
                    seq_byte = temp_seq_byte[i];
                    byte_offset = seq_byte - block_start_byte;
                    if (byte_offset < (int64_t)udpdb->hdu_bufsz && byte_offset >= 0) {
                        mod_WB = writeBlock % 64;
                        memcpy(udpdb->tblock + byte_offset
                               + mod_WB * udpdb->hdu_bufsz,
                               temp_buffers[i], UDP_DATA);
                        block_count++;
                    }
                }
                temp_idx = 0;
            }
            pthread_mutex_unlock(&mutex);

            /* opportunistic drain of saved temp queue (unchanged) */
            if (temp_idx > 0
                && temp_seq_byte[0] >= block_start_byte
                && temp_seq_byte[0] <= block_end_byte) {
                tpack = 0;
                for (unsigned int i = 0; i < temp_idx; i++) {
                    seq_byte = temp_seq_byte[i];
                    byte_offset = seq_byte - block_start_byte;
                    if (byte_offset < (int64_t)udpdb->hdu_bufsz && byte_offset >= 0) {
                        mod_WB = writeBlock % 64;
                        memcpy(udpdb->tblock + byte_offset
                               + mod_WB * udpdb->hdu_bufsz,
                               temp_buffers[i], UDP_DATA);
                        pthread_mutex_lock(&mutex);
                        block_count++;
                        pthread_mutex_unlock(&mutex);
                    }
                }
                temp_idx = 0;
            }
        }
    }

    /* cleanup */
    dsart_free_sock(sock);
    for (unsigned int i = 0; i < temp_max; i++) free(temp_buffers[i]);
    free(temp_buffers);
    free(temp_seq_byte);
    free(bufs);
    pthread_exit(NULL);
}

/* WRITE THREAD --- unchanged from legacy. */
void write_thread(void *arg)
{
    dsart_write_t *udpdb = (dsart_write_t *)arg;
    int thread_id = udpdb->thread_id;

    const pthread_t pid = pthread_self();
    int core_id = (dPort == 4011)
                  ? write_cores[thread_id]
                  : write_cores[thread_id + nwth];
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int set_result = pthread_setaffinity_np(pid, sizeof(cpu_set_t), &cpuset);
    if (set_result != 0)
        syslog(LOG_ERR, "write_thread %d: setaffinity_np fail core_id=%d",
               thread_id, core_id);

    int mod_WB = 0;
    int a;
    while (!quit_threads) {
        mod_WB = lWriteBlock % 64;
        while (blockStatus[mod_WB] == 0) {
            a = 1;
            if (quit_threads) {
                (void)a;
                pthread_exit(NULL);
            }
        }
        (void)a;

        memcpy((void *)wblock + thread_id * udpdb->hdu_bufsz / nwth,
               udpdb->tblock + mod_WB * udpdb->hdu_bufsz
                              + thread_id * udpdb->hdu_bufsz / nwth,
               udpdb->hdu_bufsz / nwth);

        pthread_mutex_lock(&mutex);
        write_ct++;
        pthread_mutex_unlock(&mutex);

        if (thread_id > 0) {
            while (write_ct != 0) a = 1;
        } else {
            while (write_ct < nwth) a = 1;
            if (dsart_new_buffer(udpdb) < 0) {
                syslog(LOG_ERR, "write_thread: dsart_new_buffer failed");
                pthread_exit(NULL);
            }
            syslog(LOG_INFO,
                   "write_thread %d: written block %d", thread_id, lWriteBlock);
            lWriteBlock++;

            skipct = 0;
            for (int i = 0; i < 64; i++) skipct += blockStatus[i];
            blockStatus[mod_WB] -= 1;
            write_ct = 0;
        }
    }
    pthread_exit(NULL);
}

/* ---- MAIN ---------------------------------------------------------- */
int main(int argc, char *argv[])
{
    openlog("dsart_capture_manythread",
            LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
    syslog(LOG_NOTICE, "dsart_capture_manythread started by uid=%d", getuid());

    /* DSART: env-driven escape hatch + mon initialisation deferred
     * until we parse -p / -q (mon segment is keyed by data port). */
    const char *autoarm_env = getenv("DSART_CAPTURE_LEGACY_AUTOARM");
    LEGACY_AUTOARM = (autoarm_env != NULL && autoarm_env[0] == '1') ? 1 : 0;

    /* DSART: install graceful-shutdown handlers so atexit() runs and
     * /dev/shm/dsart-capture-<port> is unlinked on SIGTERM/SIGINT. */
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = dsart_on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;  /* deliberately NOT SA_RESTART, so blocking
                            syscalls return early with EINTR */
        sigaction(SIGTERM, &sa, NULL);
        sigaction(SIGINT,  &sa, NULL);
        /* Ignore SIGPIPE on the off-chance a write thread fires one
         * during cleanup; lets us not abort on a broken pipe. */
        signal(SIGPIPE, SIG_IGN);
    }

    dada_hdu_t *hdu_out = 0;
    key_t out_key = CAPTURE_BLOCK_KEY;

    int core = -1;
    int chgroup = 0;
    int arg;
    char dada_fnam[200] = {0};
    char iface[100] = {0};
    int  have_dada_fnam = 0;

    while ((arg = getopt(argc, argv, "c:j:i:f:o:k:g:p:q:dh")) != -1) {
        switch (arg) {
        case 'o':
        case 'k': {
            unsigned int u = 0;
            if (sscanf(optarg, "%x", &u) != 1) {
                syslog(LOG_ERR, "could not parse hex key from %s", optarg);
                return EXIT_FAILURE;
            }
            out_key = (key_t)u;
            break;
        }
        case 'i':
            strncpy(iP, optarg, sizeof(iP) - 1);
            break;
        case 'g':
            chgroup = atoi(optarg);
            break;
        case 'j':
            strncpy(iface, optarg, sizeof(iface) - 1);
            break;
        case 'c':
            core = atoi(optarg);
            break;
        case 'p':
            dPort = atoi(optarg);
            break;
        case 'q':
            cPort = atoi(optarg);
            break;
        case 'f':
            strncpy(dada_fnam, optarg, sizeof(dada_fnam) - 1);
            have_dada_fnam = 1;
            break;
        case 'd':
            DEBUG = 1;
            break;
        case 'h':
            usage();
            return EXIT_SUCCESS;
        default:
            usage();
            return EXIT_FAILURE;
        }
    }
    (void)chgroup;

    if (dPort <= 0) {
        syslog(LOG_ERR, "must specify -p <data port>");
        usage();
        return EXIT_FAILURE;
    }
    if (cPort <= 0) {
        /* DSART: derive control port from data port using the legacy
         * convention if -q is unset (4011 -> 11223, 4012 -> 11224). */
        cPort = (dPort == 4011) ? 11223
              : (dPort == 4012) ? 11224
              : 11223;
        syslog(LOG_INFO,
               "control port not specified; defaulting to %d (data port %d)",
               cPort, dPort);
    }
    if (iP[0] == '\0') {
        strncpy(iP, "127.0.0.1", sizeof(iP) - 1);
    }
    if (iface[0] == '\0') {
        strncpy(iface, iP, sizeof(iface) - 1);
    }

    /* DSART: open mon shm now that we know dPort. */
    MON = dsart_capture_mon_open(dPort, cPort);
    if (MON != NULL) {
        syslog(LOG_INFO, "mon shm online: /dsart-capture-%d", dPort);
    } else {
        syslog(LOG_WARNING,
               "mon shm unavailable; binary continues but operator "
               "monitoring will see staleness from the sidecar");
    }

    /* default header file -- the legacy correlator_header_dsaX.txt is
     * shipped under tests/fixtures/headers/correlator_header_dsaX.txt
     * for unit-test smokes; for production the orchestrator passes
     * the absolute path via -f. */
    if (!have_dada_fnam) {
        strncpy(dada_fnam,
                "/home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/"
                "correlator_header_dsaX.txt",
                sizeof(dada_fnam) - 1);
    }

    /* start control thread (handles UTC_START / UTC_STOP). */
    int rval;
    pthread_t control_thread_id;
    rval = pthread_create(&control_thread_id, 0,
                          (void *(*)(void *))control_thread, NULL);
    if (rval != 0) {
        syslog(LOG_ERR, "Error creating control_thread: %s", strerror(rval));
        return -1;
    }
    syslog(LOG_NOTICE,
           "Created control thread, listening on %s:%d", iP, cPort);

    if (core >= 0) {
        if (dada_bind_thread_to_core(core) < 0)
            syslog(LOG_ERR, "failed to bind main thread to core %d", core);
        else
            syslog(LOG_NOTICE, "bound main thread to core %d", core);
    }

    hdu_out = dada_hdu_create(NULL);
    dada_hdu_set_key(hdu_out, out_key);
    if (dada_hdu_connect(hdu_out) < 0) {
        syslog(LOG_ERR, "could not connect to output dada buffer 0x%x",
               (unsigned)out_key);
        return EXIT_FAILURE;
    }
    if (dada_hdu_lock_write(hdu_out) < 0) {
        dsart_dbgpu_cleanup(hdu_out);
        syslog(LOG_ERR, "could not lock write on output dada buffer");
        return EXIT_FAILURE;
    }
    syslog(LOG_INFO, "opened connection to output DB key=0x%x",
           (unsigned)out_key);

    /* PSRDADA header */
    char *hout = (char *)malloc(sizeof(char) * 4096);
    if (fileread(dada_fnam, hout, 4096) < 0) {
        syslog(LOG_ERR, "could not read ASCII header from %s", dada_fnam);
        free(hout);
        return EXIT_FAILURE;
    }
    char *header_out = ipcbuf_get_next_write(hdu_out->header_block);
    if (!header_out) {
        syslog(LOG_ERR, "could not get next header block [output]");
        dsart_dbgpu_cleanup(hdu_out);
        free(hout);
        return EXIT_FAILURE;
    }
    memcpy(header_out, hout, 4096);
    if (ipcbuf_mark_filled(hdu_out->header_block, 4096) < 0) {
        syslog(LOG_ERR, "could not mark header block filled [output]");
        dsart_dbgpu_cleanup(hdu_out);
        free(hout);
        return EXIT_FAILURE;
    }
    free(hout);

    snprintf(STATE, sizeof(STATE), "LISTEN");
    syslog(LOG_INFO,
           "marked output header block as filled -- now in LISTEN state");

    /* allocate the shared structs */
    dsart_udpdb_t udpdb[nth];
    dsart_stats_t stats;
    dsart_write_t writey[nwth];

    uint64_t bufsz = ipcbuf_get_bufsz((ipcbuf_t *)hdu_out->data_block);
    char *tblock = (char *)malloc(sizeof(char) * bufsz * 64);
    assert(tblock != NULL);
    stats_t *packets = init_stats_t();
    stats_t *bytes   = init_stats_t();
    reset_stats_t(packets);
    reset_stats_t(bytes);

    stats.packets = packets;
    stats.bytes = bytes;

    for (int i = 0; i < nwth; i++) {
        writey[i].hdu = hdu_out;
        writey[i].hdu_bufsz = bufsz;
        writey[i].block_open = 0;
        writey[i].tblock = tblock;
        writey[i].thread_id = i;
    }
    dsart_open_buffer(&writey[0]);

    for (int i = 0; i < nth; i++) {
        udpdb[i].packets = packets;
        udpdb[i].bytes = bytes;
        udpdb[i].tblock = tblock;
        udpdb[i].port = dPort;
        udpdb[i].interface = strdup(iface);
        udpdb[i].hdu_bufsz = bufsz;
        udpdb[i].packets_per_buffer = bufsz / UDP_DATA;
        udpdb[i].num_inputs = NSNAPS;
        udpdb[i].verbose = 0;
        udpdb[i].rcv_sleeps = 0;
        udpdb[i].thread_id = i;
    }

    pthread_t stats_thread_id;
    rval = pthread_create(&stats_thread_id, 0,
                          (void *(*)(void *))stats_thread, (void *)&stats);
    if (rval != 0) {
        syslog(LOG_ERR, "Error creating stats_thread: %s", strerror(rval));
        return -1;
    }

    pthread_t recv_thread_id[nth];
    for (int i = 0; i < nth; i++) {
        rval = pthread_create(&recv_thread_id[i], 0,
                              (void *(*)(void *))recv_thread,
                              (void *)&udpdb[i]);
        if (rval != 0) {
            syslog(LOG_ERR, "Error creating recv_thread %d: %s",
                   i, strerror(rval));
            return -1;
        }
    }

    pthread_t write_thread_id[nwth];
    for (int i = 0; i < nwth; i++) {
        rval = pthread_create(&write_thread_id[i], 0,
                              (void *(*)(void *))write_thread,
                              (void *)&writey[i]);
        if (rval != 0) {
            syslog(LOG_ERR, "Error creating write_thread %d: %s",
                   i, strerror(rval));
            return -1;
        }
    }

    syslog(LOG_NOTICE,
           "dsart_capture_manythread up: data=%s:%d ctl=%s:%d "
           "out_key=0x%x autoarm=%d", iface, dPort, iP, cPort,
           (unsigned)out_key, LEGACY_AUTOARM);

    while (!quit_threads) {
        sleep(1);
    }

    syslog(LOG_INFO, "joining all threads");
    quit_threads = 1;
    void *result = 0;
    pthread_join(control_thread_id, &result);
    pthread_join(stats_thread_id, &result);
    for (int i = 0; i < nth; i++)  pthread_join(recv_thread_id[i],  &result);
    for (int i = 0; i < nwth; i++) pthread_join(write_thread_id[i], &result);

    if (MON != NULL) dsart_capture_mon_set_arm(MON, DSART_ARM_STOPPED);

    free(tblock);
    dsart_dbgpu_cleanup(hdu_out);
    closelog();
    return EXIT_SUCCESS;
}
