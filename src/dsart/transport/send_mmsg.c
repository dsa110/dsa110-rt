/*
 * send_mmsg.c — paced multi-destination UDP sender for the 2x corner turn
 * (x2 WS-T; docs/x2/SPEC.md §5; plan "pack + D2H + TX").
 *
 * The corr's GPU packer produces whole frame-v2 datagrams per block. Python
 * hands them over in ONE call (txm_submit: contiguous buffer + offsets +
 * lengths); this module copies each datagram into the queue of its
 * destination and a dedicated pthread drains the queues with sendmmsg(2).
 * No Python is on the per-datagram path and ctypes releases the GIL for
 * every call, so the Python thread is never blocked by the network.
 *
 * Routing is EXPLICIT: a datagram's destination comes from a dm -> dest
 * table indexed by the frame's u16 dm_idx (byte offset 26 in v1 and v2
 * headers). Unrouted DMs are dropped and counted (never "sent somewhere").
 *
 * Pacing is per destination: a token bucket of ``rate_bps`` bits/s of UDP
 * payload (datagram bytes; IP/UDP/Ethernet overhead not included) and depth
 * ``burst_bytes``; rate 0 = unpaced. A slow / absent receiver never blocks
 * another destination: each destination has its own non-blocking socket and
 * queue. A full queue drops at submit (counted) after an optional bounded
 * wait (``block_timeout_s``), mirroring the M4b "drop, don't block" rule.
 *
 * Threading: one producer thread may call txm_submit; one internal sender
 * thread consumes. Counters are atomics readable at any time.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define TXM_MAX_DEST   64
#define TXM_MAX_BATCH  1024
#define TXM_N_GLOBAL   8
#define TXM_N_DEST_CTR 10

typedef struct txm_config {
    uint32_t n_dest;
    uint32_t batch;          /* datagrams per sendmmsg, 1..1024 */
    uint32_t queue_slots;    /* datagrams queued per destination */
    uint32_t slot_bytes;     /* max datagram size accepted */
    uint64_t burst_bytes;    /* token-bucket depth per destination */
    int32_t  so_sndbuf;      /* 0 = kernel default */
    int32_t  cpu;            /* sender thread affinity; -1 = none */
    uint32_t dm_offset;      /* byte offset of the u16 dm_idx (26) */
    uint32_t _pad;
    char     bind_host[64];  /* source address; "" = any */
} txm_config_t;

typedef struct txm_dest {
    char     host[64];
    uint16_t port;
    uint16_t _pad[3];
    double   rate_bps;       /* bits/s of datagram bytes; 0 = unpaced */
} txm_dest_t;

typedef struct txm_queue {
    int                fd;
    struct sockaddr_in addr;
    uint8_t           *slots;
    uint32_t          *lens;
    _Atomic uint64_t   head;        /* consumer */
    _Atomic uint64_t   tail;        /* producer */
    _Atomic double     rate_Bps;    /* bytes/s (txm_set_rate may change it) */
    double             tokens;      /* consumer only */
    uint64_t           last_ns;     /* consumer only */
    /* counters: enqueued, sent, bytes, full_drops, errors, eagain,
     * refused, max_depth, depth(now: computed), last_errno */
    _Atomic uint64_t   enq, sent, bytes, full_drops, errors, eagain, refused, max_depth;
    _Atomic int        last_errno;
} txm_queue_t;

typedef struct txm {
    txm_config_t      cfg;
    txm_queue_t      *q;
    _Atomic int16_t   route[65536];
    pthread_t         th;
    int               started;
    _Atomic int       run;
    _Atomic uint64_t  drain_until_ns;
    /* global counters: submitted, unrouted, bad_length, sendmmsg_calls,
     * idle_sleeps, accepted, 0, 0 */
    _Atomic uint64_t  submitted, unrouted, bad_length, calls, sleeps, accepted;
    struct mmsghdr   *mm;
    struct iovec     *iov;
} txm_t;

static uint64_t
txm_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static void
txm_sleep_ns(uint64_t ns)
{
    struct timespec ts = { (time_t)(ns / 1000000000u), (long)(ns % 1000000000u) };
    nanosleep(&ts, NULL);
}

void txm_destroy(txm_t *t);

txm_t *
txm_create(const txm_config_t *cfg, const txm_dest_t *dests, char *err, size_t errlen)
{
#define EM(...) do { if (err && errlen) snprintf(err, errlen, __VA_ARGS__); } while (0)
    if (!cfg || !dests) { EM("NULL config"); return NULL; }
    if (cfg->n_dest < 1 || cfg->n_dest > TXM_MAX_DEST) {
        EM("n_dest=%u not in [1, %d]", cfg->n_dest, TXM_MAX_DEST); return NULL;
    }
    if (cfg->batch < 1 || cfg->batch > TXM_MAX_BATCH) {
        EM("batch=%u not in [1, %d]", cfg->batch, TXM_MAX_BATCH); return NULL;
    }
    if (cfg->queue_slots < 2 || cfg->slot_bytes < cfg->dm_offset + 2 || cfg->slot_bytes > 65507) {
        EM("queue_slots=%u / slot_bytes=%u invalid", cfg->queue_slots, cfg->slot_bytes);
        return NULL;
    }
    txm_t *t = (txm_t *)calloc(1, sizeof(txm_t));
    if (!t) { EM("out of memory"); return NULL; }
    t->cfg = *cfg;
    for (int i = 0; i < 65536; i++) atomic_store_explicit(&t->route[i], -1, memory_order_relaxed);
    t->q = (txm_queue_t *)calloc(cfg->n_dest, sizeof(txm_queue_t));
    t->mm = (struct mmsghdr *)calloc(cfg->batch, sizeof(struct mmsghdr));
    t->iov = (struct iovec *)calloc(cfg->batch, sizeof(struct iovec));
    if (!t->q || !t->mm || !t->iov) { EM("out of memory"); txm_destroy(t); return NULL; }
    for (uint32_t d = 0; d < cfg->n_dest; d++) t->q[d].fd = -1;
    for (uint32_t d = 0; d < cfg->n_dest; d++) {
        txm_queue_t *q = &t->q[d];
        q->slots = (uint8_t *)malloc((size_t)cfg->queue_slots * cfg->slot_bytes);
        q->lens = (uint32_t *)calloc(cfg->queue_slots, sizeof(uint32_t));
        if (!q->slots || !q->lens) { EM("out of memory (queues)"); txm_destroy(t); return NULL; }
        memset(&q->addr, 0, sizeof(q->addr));
        q->addr.sin_family = AF_INET;
        q->addr.sin_port = htons(dests[d].port);
        if (inet_pton(AF_INET, dests[d].host, &q->addr.sin_addr) != 1) {
            EM("dest %u: bad IPv4 address %.63s", d, dests[d].host); txm_destroy(t); return NULL;
        }
        atomic_store_explicit(&q->rate_Bps, dests[d].rate_bps > 0 ? dests[d].rate_bps / 8.0 : 0.0,
                              memory_order_relaxed);
        q->tokens = (double)cfg->burst_bytes;
        q->fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (q->fd < 0) { EM("socket: %s", strerror(errno)); txm_destroy(t); return NULL; }
        if (cfg->so_sndbuf > 0)
            setsockopt(q->fd, SOL_SOCKET, SO_SNDBUF, &cfg->so_sndbuf, sizeof(cfg->so_sndbuf));
        if (cfg->bind_host[0]) {
            struct sockaddr_in src;
            memset(&src, 0, sizeof(src));
            src.sin_family = AF_INET;
            if (inet_pton(AF_INET, cfg->bind_host, &src.sin_addr) != 1
                || bind(q->fd, (struct sockaddr *)&src, sizeof(src)) != 0) {
                EM("bind %.63s: %s", cfg->bind_host, strerror(errno)); txm_destroy(t); return NULL;
            }
        }
    }
    return t;
#undef EM
}

int
txm_set_route(txm_t *t, uint32_t dm, int32_t dest)
{
    if (!t || dm > 0xFFFF || dest >= (int32_t)t->cfg.n_dest || dest < -1) return -1;
    atomic_store_explicit(&t->route[dm], (int16_t)dest, memory_order_release);
    return 0;
}

/* Change one destination's pacing rate (bits/s; 0 = unpaced) at any time. */
int
txm_set_rate(txm_t *t, uint32_t dest, double rate_bps)
{
    if (!t || dest >= t->cfg.n_dest) return -1;
    atomic_store_explicit(&t->q[dest].rate_Bps, rate_bps > 0 ? rate_bps / 8.0 : 0.0,
                          memory_order_relaxed);
    return 0;
}

int32_t
txm_get_route(txm_t *t, uint32_t dm)
{
    if (!t || dm > 0xFFFF) return -2;
    return atomic_load_explicit(&t->route[dm], memory_order_acquire);
}

/* Queue datagrams; returns how many were accepted. block_timeout_s > 0 waits
 * (bounded) for queue space; otherwise a full queue drops immediately. */
int64_t
txm_submit(txm_t *t, const uint8_t *buf, const uint64_t *offsets,
           const uint32_t *lengths, uint32_t n, double block_timeout_s)
{
    if (!t || (!buf && n)) return -1;
    const uint32_t S = t->cfg.queue_slots, SB = t->cfg.slot_bytes;
    const uint64_t deadline = block_timeout_s > 0
        ? txm_now_ns() + (uint64_t)(block_timeout_s * 1e9) : 0;
    int64_t accepted = 0;
    atomic_fetch_add_explicit(&t->submitted, n, memory_order_relaxed);
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *dg = buf + offsets[i];
        const uint32_t len = lengths[i];
        if (len > SB || len < t->cfg.dm_offset + 2) {
            atomic_fetch_add_explicit(&t->bad_length, 1, memory_order_relaxed);
            continue;
        }
        uint16_t dm;
        memcpy(&dm, dg + t->cfg.dm_offset, 2);
        const int16_t d = atomic_load_explicit(&t->route[dm], memory_order_acquire);
        if (d < 0) {
            atomic_fetch_add_explicit(&t->unrouted, 1, memory_order_relaxed);
            continue;
        }
        txm_queue_t *q = &t->q[d];
        const uint64_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
        uint64_t depth = tail - atomic_load_explicit(&q->head, memory_order_acquire);
        while (depth >= S && deadline && txm_now_ns() < deadline) {
            txm_sleep_ns(50000);
            depth = tail - atomic_load_explicit(&q->head, memory_order_acquire);
        }
        if (depth >= S) {
            atomic_fetch_add_explicit(&q->full_drops, 1, memory_order_relaxed);
            continue;
        }
        memcpy(q->slots + (size_t)(tail % S) * SB, dg, len);
        q->lens[tail % S] = len;
        atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
        atomic_fetch_add_explicit(&q->enq, 1, memory_order_relaxed);
        if (depth + 1 > atomic_load_explicit(&q->max_depth, memory_order_relaxed))
            atomic_store_explicit(&q->max_depth, depth + 1, memory_order_relaxed);
        accepted++;
    }
    atomic_fetch_add_explicit(&t->accepted, (uint64_t)accepted, memory_order_relaxed);
    return accepted;
}

static int
txm_all_empty(txm_t *t)
{
    for (uint32_t d = 0; d < t->cfg.n_dest; d++)
        if (atomic_load_explicit(&t->q[d].tail, memory_order_acquire)
            != atomic_load_explicit(&t->q[d].head, memory_order_acquire))
            return 0;
    return 1;
}

static void *
txm_main(void *arg)
{
    txm_t *t = (txm_t *)arg;
    const txm_config_t *c = &t->cfg;
    const uint32_t S = c->queue_slots, SB = c->slot_bytes;
    uint32_t rot = 0;
    if (c->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(c->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    for (;;) {
        const int running = atomic_load_explicit(&t->run, memory_order_acquire);
        if (!running) {
            const uint64_t until = atomic_load_explicit(&t->drain_until_ns, memory_order_acquire);
            if (txm_all_empty(t) || txm_now_ns() >= until) break;
        }
        const uint64_t now = txm_now_ns();
        uint64_t wait_ns = 1000000;              /* idle poll: 1 ms */
        int any = 0;
        for (uint32_t k = 0; k < c->n_dest; k++) {
            const uint32_t d = (rot + k) % c->n_dest;
            txm_queue_t *q = &t->q[d];
            const uint64_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
            const uint64_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);
            const uint64_t avail = tail - head;
            if (!avail) continue;
            const double rate = atomic_load_explicit(&q->rate_Bps, memory_order_relaxed);
            if (rate > 0) {
                if (q->last_ns) {
                    q->tokens += rate * (double)(now - q->last_ns) * 1e-9;
                    if (q->tokens > (double)c->burst_bytes) q->tokens = (double)c->burst_bytes;
                }
                q->last_ns = now;
            }
            uint32_t m = 0;
            double budget = q->tokens;
            uint64_t bytes = 0;
            while (m < c->batch && m < avail) {
                const uint64_t idx = (head + m) % S;
                const uint32_t len = q->lens[idx];
                if (rate > 0 && budget < (double)len) {
                    const double need_s = ((double)len - budget) / rate;
                    const uint64_t w = (uint64_t)(need_s * 1e9) + 1;
                    if (w < wait_ns) wait_ns = w;
                    break;
                }
                budget -= (double)len;
                t->iov[m].iov_base = q->slots + (size_t)idx * SB;
                t->iov[m].iov_len = len;
                memset(&t->mm[m], 0, sizeof(t->mm[m]));
                t->mm[m].msg_hdr.msg_iov = &t->iov[m];
                t->mm[m].msg_hdr.msg_iovlen = 1;
                t->mm[m].msg_hdr.msg_name = &q->addr;
                t->mm[m].msg_hdr.msg_namelen = sizeof(q->addr);
                bytes += len;
                m++;
            }
            if (!m) continue;
            atomic_fetch_add_explicit(&t->calls, 1, memory_order_relaxed);
            const int r = sendmmsg(q->fd, t->mm, m, 0);
            if (r < 0) {
                const int e = errno;
                if (e == EAGAIN || e == EWOULDBLOCK || e == ENOBUFS || e == EINTR) {
                    atomic_fetch_add_explicit(&q->eagain, 1, memory_order_relaxed);
                    if (wait_ns > 50000) wait_ns = 50000;
                } else if (e == ECONNREFUSED) {
                    atomic_fetch_add_explicit(&q->refused, 1, memory_order_relaxed);
                } else {
                    /* Unsendable datagram: drop it so the queue keeps moving. */
                    atomic_fetch_add_explicit(&q->errors, 1, memory_order_relaxed);
                    atomic_store_explicit(&q->last_errno, e, memory_order_relaxed);
                    atomic_store_explicit(&q->head, head + 1, memory_order_release);
                }
                continue;
            }
            uint64_t sent_bytes = 0;
            for (int i = 0; i < r; i++) sent_bytes += t->iov[i].iov_len;
            if (rate > 0) q->tokens -= (double)sent_bytes;
            atomic_store_explicit(&q->head, head + (uint64_t)r, memory_order_release);
            atomic_fetch_add_explicit(&q->sent, (uint64_t)r, memory_order_relaxed);
            atomic_fetch_add_explicit(&q->bytes, sent_bytes, memory_order_relaxed);
            any = 1;
            (void)bytes;
        }
        rot++;
        if (!any) {
            atomic_fetch_add_explicit(&t->sleeps, 1, memory_order_relaxed);
            if (wait_ns < 20000) sched_yield();
            else txm_sleep_ns(wait_ns > 1000000 ? 1000000 : wait_ns);
        }
    }
    return NULL;
}

int
txm_start(txm_t *t)
{
    if (!t) return -1;
    if (t->started) return 0;
    atomic_store_explicit(&t->run, 1, memory_order_release);
    if (pthread_create(&t->th, NULL, txm_main, t) != 0) {
        atomic_store_explicit(&t->run, 0, memory_order_release);
        return -2;
    }
    t->started = 1;
    return 0;
}

/* Stop the sender; keep sending queued datagrams for up to drain_timeout_s. */
int
txm_stop(txm_t *t, double drain_timeout_s)
{
    if (!t) return -1;
    if (!t->started) return 0;
    atomic_store_explicit(&t->drain_until_ns,
                          txm_now_ns() + (uint64_t)(drain_timeout_s > 0 ? drain_timeout_s * 1e9 : 0),
                          memory_order_release);
    atomic_store_explicit(&t->run, 0, memory_order_release);
    pthread_join(t->th, NULL);
    t->started = 0;
    return 0;
}

/* Wait (bounded) until every queue is empty. 0 = drained, -1 = timeout. */
int
txm_flush(txm_t *t, double timeout_s)
{
    if (!t) return -1;
    const uint64_t until = txm_now_ns() + (uint64_t)(timeout_s * 1e9);
    while (!txm_all_empty(t)) {
        if (txm_now_ns() >= until) return -1;
        txm_sleep_ns(200000);
    }
    return 0;
}

/* out_global[8]: submitted, accepted, unrouted, bad_length, sendmmsg_calls,
 * idle_sleeps, 0, 0. out_dest[n_dest][10]: enqueued, sent, bytes,
 * full_drops, errors, eagain, refused, max_depth, depth, last_errno. */
void
txm_counters(txm_t *t, uint64_t *out_global, uint64_t *out_dest)
{
    if (!t) return;
    if (out_global) {
        out_global[0] = atomic_load_explicit(&t->submitted, memory_order_relaxed);
        out_global[1] = atomic_load_explicit(&t->accepted, memory_order_relaxed);
        out_global[2] = atomic_load_explicit(&t->unrouted, memory_order_relaxed);
        out_global[3] = atomic_load_explicit(&t->bad_length, memory_order_relaxed);
        out_global[4] = atomic_load_explicit(&t->calls, memory_order_relaxed);
        out_global[5] = atomic_load_explicit(&t->sleeps, memory_order_relaxed);
        out_global[6] = 0;
        out_global[7] = 0;
    }
    if (out_dest) {
        for (uint32_t d = 0; d < t->cfg.n_dest; d++) {
            txm_queue_t *q = &t->q[d];
            uint64_t *o = out_dest + (size_t)d * TXM_N_DEST_CTR;
            o[0] = atomic_load_explicit(&q->enq, memory_order_relaxed);
            o[1] = atomic_load_explicit(&q->sent, memory_order_relaxed);
            o[2] = atomic_load_explicit(&q->bytes, memory_order_relaxed);
            o[3] = atomic_load_explicit(&q->full_drops, memory_order_relaxed);
            o[4] = atomic_load_explicit(&q->errors, memory_order_relaxed);
            o[5] = atomic_load_explicit(&q->eagain, memory_order_relaxed);
            o[6] = atomic_load_explicit(&q->refused, memory_order_relaxed);
            o[7] = atomic_load_explicit(&q->max_depth, memory_order_relaxed);
            o[8] = atomic_load_explicit(&q->tail, memory_order_acquire)
                   - atomic_load_explicit(&q->head, memory_order_acquire);
            o[9] = (uint64_t)(int64_t)atomic_load_explicit(&q->last_errno, memory_order_relaxed);
        }
    }
}

void
txm_destroy(txm_t *t)
{
    if (!t) return;
    if (t->started) txm_stop(t, 0.0);
    if (t->q) {
        for (uint32_t d = 0; d < t->cfg.n_dest; d++) {
            if (t->q[d].fd >= 0) close(t->q[d].fd);
            free(t->q[d].slots);
            free(t->q[d].lens);
        }
        free(t->q);
    }
    free(t->mm);
    free(t->iov);
    free(t);
}

/* ABI cross-check for the Python wrapper. */
int
txm_abi(uint64_t *out, uint32_t n)
{
    const uint64_t v[] = { sizeof(txm_config_t), sizeof(txm_dest_t), TXM_N_GLOBAL,
                           TXM_N_DEST_CTR, TXM_MAX_DEST };
    const uint32_t m = (uint32_t)(sizeof(v) / sizeof(v[0]));
    for (uint32_t i = 0; out && i < n && i < m; i++) out[i] = v[i];
    return (int)m;
}
