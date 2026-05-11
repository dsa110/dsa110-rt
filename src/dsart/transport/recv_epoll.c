/*
 * recv_epoll.c — C epoll receive loop for M4a prod-frame UDP RX (chunk 6).
 *
 * Replaces the Python `_RxLoop` + `TransportRxProd.ingest_datagram` hot path
 * with a single-pthread C loop that:
 *
 *   1. Owns the UDP socket (bound + SO_RCVBUF set).
 *   2. Drains via `recvmmsg(64)` (one syscall per batch).
 *   3. Parses the 72-byte ProdFrame header in C (memcpy + field reads;
 *      no memory allocation on the hot path).
 *   4. Validates magic, version, length, flag set, and field ranges
 *      per :mod:`dsart.transport.prod_frame`.
 *   5. Checks per-chgroup ``pattern_id`` against an expected table; on
 *      mismatch, increments the mon-key counter and drops.
 *   6. Maintains per-(chgroup, dm_idx) reorder windows (depth W=4) using
 *      a fixed-size struct array. State machine mirrors
 *      :class:`dsart.transport.rx._ReorderWindow`.
 *   7. On full reassembly: increments ``n_committed`` (and optionally
 *      copies the raw quantised payload to a chunk-4 SPMC shm ring;
 *      Phase B — not in this MVP).
 *   8. On window-slide with missing seqs: increments
 *      ``window_slide_zerofill_count`` and zero-fills the dropped slot.
 *
 * All exposed counters are ``_Atomic uint64_t`` so Python ctypes can read
 * them without locks; the C loop bumps them with ``__ATOMIC_RELAXED``
 * (counters do not order other accesses) except for ``write_seq`` style
 * counters which use ``__ATOMIC_RELEASE`` (none in this MVP).
 *
 * Build: compiled via setuptools as ``_recv_epoll`` extension. See
 * ``setup.py``.
 *
 * Threading model:
 *   - One pthread owns the epoll loop. Started by ``recv_epoll_start``.
 *   - Python may call:
 *       - ``recv_epoll_open`` (main thread, before start)
 *       - ``recv_epoll_set_expected_pid`` (any thread; atomic-set)
 *       - ``recv_epoll_start`` (main thread; idempotent)
 *       - ``recv_epoll_stop`` (main thread; signals via atomic flag,
 *         then joins)
 *       - ``recv_epoll_close`` (main thread; after stop)
 *       - any counter getter (any thread; atomic-load)
 *   - No locks on the hot path. All inter-thread state is via
 *     ``_Atomic`` fields.
 *
 * Performance target: 100k+ datagrams/sec sustained on h01 loopback,
 * corresponding to ~7 Gb/s aggregate per-search ingress at the §9
 * default op-point (2-fragment payloads at jumbo MTU). See
 * ``docs/m4a/prod_rate_findings.md`` for measured Python ceilings.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>


/* -----------------------------------------------------------------------
 * Wire-protocol constants — mirror src/dsart/transport/prod_frame.py
 * Drift here = silent loss; keep in sync via the C ↔ Python tests in
 * tests/transport/test_recv_epoll.py.
 * ---------------------------------------------------------------------- */
#define PROD_FRAME_MAGIC        UINT32_C(0xD5A1107E)
#define PROD_FRAME_VERSION      UINT16_C(1)
#define PROD_FRAME_HEADER_BYTES 72

#define FLAG_QUANTIZED          (1u << 0)
#define FLAG_LAST_IN_BLOCK      (1u << 1)
#define FLAG_RESERVED_BIT2      (1u << 2)
#define FLAG_NOISE_WARMUP       (1u << 3)
#define FLAG_RFI_WARMING_UP     (1u << 4)
#define FLAG_DEFINED_V1_MASK    (FLAG_QUANTIZED | FLAG_LAST_IN_BLOCK \
                                 | FLAG_RESERVED_BIT2 | FLAG_NOISE_WARMUP \
                                 | FLAG_RFI_WARMING_UP)

#define BITS_CINT8_COMPLEX      16
#define BITS_CFP16_COMPLEX      32

/* Topology bounds — sized for plan §11 line 2654 (16 corr × 24 coarse DMs). */
#define MAX_CHGROUPS            16
#define MAX_DMS                 32
#define WINDOW_DEPTH            4
#define MAX_FRAGS_PER_PAYLOAD   16
#define MAX_FRAG_PAYLOAD_BYTES  9000  /* generous; jumbo MTU caps at 8964 */

/* recvmmsg batch size. Larger = fewer syscalls; 64 is a good loopback default. */
#define RECV_BATCH              64
#define RECV_BUF_BYTES          (PROD_FRAME_HEADER_BYTES + MAX_FRAG_PAYLOAD_BYTES)


/* -----------------------------------------------------------------------
 * Header parsing — packed struct matching prod_frame.py layout
 * Format: <I H H Q Q H H H H H H I Q B B H f f I 8s
 * ---------------------------------------------------------------------- */
typedef struct __attribute__((packed)) prod_hdr {
    uint32_t magic;                  /* 0 */
    uint16_t version;                /* 4 */
    uint16_t flags;                  /* 6 */
    uint64_t seq;                    /* 8 */
    uint64_t specnum;                /* 16 */
    uint16_t chgroup;                /* 24 */
    uint16_t dm_idx;                 /* 26 */
    uint16_t frag_idx;               /* 28 */
    uint16_t n_frags;                /* 30 */
    uint16_t n_grid;                 /* 32 */
    uint16_t reserved0;              /* 34 */
    uint32_t n_filled;               /* 36 */
    uint64_t pattern_id;             /* 40 */
    uint8_t  bits_per_cell;          /* 48 */
    uint8_t  t_int_factor;           /* 49 */
    uint16_t reserved1;              /* 50 */
    float    scale;                  /* 52 */
    float    offset;                 /* 56 */
    uint32_t payload_bytes_in_frag;  /* 60 */
    uint8_t  reserved2[8];           /* 64..71 */
} prod_hdr_t;
_Static_assert(sizeof(prod_hdr_t) == PROD_FRAME_HEADER_BYTES,
               "prod_hdr_t must be 72 bytes");


/* -----------------------------------------------------------------------
 * Reorder window state — per (chgroup, dm_idx)
 *
 * Fixed-size storage: depth W=4 slots, each carrying up to
 * MAX_FRAGS_PER_PAYLOAD fragment buffers of up to MAX_FRAG_PAYLOAD_BYTES.
 * Total per-flow footprint:
 *   16 (chgroups) × 32 (dms) × 4 (W) × 16 (frags) × 9000 (bytes)
 *   = 256 MiB worst-case
 * In practice we'll use 16 × 24 × 4 × 2 × 9000 ≈ 27 MiB. Memory is the
 * tradeoff for zero allocation on the hot path.
 * ---------------------------------------------------------------------- */
typedef struct slot {
    uint64_t seq;
    uint16_t n_frags_expected;
    uint32_t fragments_bitmap;       /* bit i set => frag i received */
    bool     occupied;
    bool     committed;
    /* Header from first-arriving fragment (carries scale/offset/n_filled). */
    prod_hdr_t hdr;
    bool     hdr_set;
    /* Per-fragment storage. frag_bytes[i] = length of frag i payload. */
    uint8_t  frag_data[MAX_FRAGS_PER_PAYLOAD][MAX_FRAG_PAYLOAD_BYTES];
    uint32_t frag_bytes[MAX_FRAGS_PER_PAYLOAD];
} slot_t;

typedef struct flow {
    slot_t   slots[WINDOW_DEPTH];
    uint64_t head_seq;
    bool     head_seq_set;
} flow_t;


/* -----------------------------------------------------------------------
 * Global state — singleton C loop per Python process. The MVP only
 * supports one loop at a time, which matches production (one search-node
 * process owns one listening socket).
 * ---------------------------------------------------------------------- */
typedef struct rx_epoll {
    int       sockfd;
    int       epfd;
    pthread_t thread;
    bool      thread_started;

    /* Loop-control flag. Set by main thread; checked by hot path. */
    _Atomic int run_flag;

    /* Expected pattern_id per chgroup. -1 (== UINT64_MAX) sentinel means
     * "no expectation; accept everything". Updated atomically from
     * Python at cmd:prepare time. */
    _Atomic uint64_t expected_pid[MAX_CHGROUPS];
    _Atomic int      expected_pid_set[MAX_CHGROUPS];  /* 0 == unset */

    /* Per-flow reorder windows. Not in any atomic — only the loop pthread
     * mutates these. */
    flow_t flows[MAX_CHGROUPS][MAX_DMS];

    /* Atomic counters (RELAXED-ordered; readable from Python). */
    _Atomic uint64_t n_received;
    _Atomic uint64_t n_committed;
    _Atomic uint64_t bad_magic_count;
    _Atomic uint64_t bad_version_count;
    _Atomic uint64_t bad_length_count;
    _Atomic uint64_t bad_field_range_count;
    _Atomic uint64_t reserved_bit_count;
    _Atomic uint64_t pattern_mismatch_count;
    _Atomic uint64_t window_slide_zerofill_count;
    _Atomic uint64_t out_of_order_drop_count;
    _Atomic uint64_t bytes_received_total;

    /* Pre-allocated recvmmsg buffers. */
    struct mmsghdr mmsg[RECV_BATCH];
    struct iovec   iovs[RECV_BATCH];
    uint8_t        bufs[RECV_BATCH][RECV_BUF_BYTES];
} rx_epoll_t;

static rx_epoll_t g_state;
static bool g_state_init = false;


/* -----------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static inline int valid_bits_per_cell(uint8_t b) {
    return b == BITS_CINT8_COMPLEX || b == BITS_CFP16_COMPLEX;
}

static inline int valid_t_int_factor(uint8_t t) {
    return t == 1 || t == 4 || t == 8 || t == 16 || t == 32 || t == 64 || t == 128;
}

/* Commit one fully-reassembled payload. Updates n_committed. The MVP
 * does NOT write to the shm ring — that's Phase B. */
static inline void commit_slot(rx_epoll_t *s, slot_t *slot, uint32_t corr,
                               uint32_t dm) {
    (void)slot; (void)corr; (void)dm;
    atomic_fetch_add_explicit(&s->n_committed, 1, memory_order_relaxed);
}

/* Zero-fill a slot that's sliding out without all frags. */
static inline void zerofill_slot(rx_epoll_t *s) {
    atomic_fetch_add_explicit(&s->window_slide_zerofill_count, 1,
                              memory_order_relaxed);
}

static void slot_reset(slot_t *slot) {
    slot->seq = 0;
    slot->n_frags_expected = 0;
    slot->fragments_bitmap = 0;
    slot->occupied = false;
    slot->committed = false;
    slot->hdr_set = false;
    /* Don't zero frag_data — we'll overwrite on next use. */
}

/* Mirror of _ReorderWindow._slide_to in rx.py. */
static void window_slide_to(rx_epoll_t *s, flow_t *flow,
                            uint32_t corr, uint32_t dm,
                            uint64_t new_head) {
    (void)corr; (void)dm;
    for (uint64_t seq = flow->head_seq; seq < new_head; seq++) {
        slot_t *slot = &flow->slots[seq % WINDOW_DEPTH];
        if (slot->occupied && slot->seq == seq) {
            if (!slot->committed) {
                zerofill_slot(s);
            }
            slot_reset(slot);
        } else {
            /* Never received any fragment for this seq. */
            zerofill_slot(s);
        }
    }
    flow->head_seq = new_head;
}

static inline uint64_t window_tail(const flow_t *flow) {
    return flow->head_seq + WINDOW_DEPTH - 1;
}

/* Mirror of _ReorderWindow.ingest_fragment in rx.py. */
static void ingest_fragment(rx_epoll_t *s, const prod_hdr_t *hdr,
                            const uint8_t *payload, size_t payload_len) {
    uint32_t corr = hdr->chgroup;
    uint32_t dm = hdr->dm_idx;
    if (corr >= MAX_CHGROUPS || dm >= MAX_DMS) {
        atomic_fetch_add_explicit(&s->bad_field_range_count, 1,
                                  memory_order_relaxed);
        return;
    }
    flow_t *flow = &s->flows[corr][dm];
    uint64_t seq = hdr->seq;
    uint16_t frag_idx = hdr->frag_idx;
    uint16_t n_frags = hdr->n_frags;

    if (n_frags == 0 || n_frags > MAX_FRAGS_PER_PAYLOAD) {
        atomic_fetch_add_explicit(&s->bad_field_range_count, 1,
                                  memory_order_relaxed);
        return;
    }
    if (frag_idx >= n_frags) {
        atomic_fetch_add_explicit(&s->bad_field_range_count, 1,
                                  memory_order_relaxed);
        return;
    }

    if (!flow->head_seq_set) {
        flow->head_seq = seq;
        flow->head_seq_set = true;
    }

    if (seq < flow->head_seq) {
        atomic_fetch_add_explicit(&s->out_of_order_drop_count, 1,
                                  memory_order_relaxed);
        return;
    }

    uint64_t tail = window_tail(flow);
    if (seq > tail) {
        uint64_t new_head = seq - WINDOW_DEPTH + 1;
        window_slide_to(s, flow, corr, dm, new_head);
    }

    slot_t *slot = &flow->slots[seq % WINDOW_DEPTH];
    if (!slot->occupied) {
        slot->seq = seq;
        slot->n_frags_expected = n_frags;
        slot->occupied = true;
        slot->hdr = *hdr;
        slot->hdr_set = true;
    }
    /* Store fragment if not already. */
    if (frag_idx < 32) {
        uint32_t mask = 1u << frag_idx;
        if (!(slot->fragments_bitmap & mask)) {
            slot->fragments_bitmap |= mask;
            size_t copy = payload_len > MAX_FRAG_PAYLOAD_BYTES
                              ? MAX_FRAG_PAYLOAD_BYTES
                              : payload_len;
            memcpy(slot->frag_data[frag_idx], payload, copy);
            slot->frag_bytes[frag_idx] = (uint32_t)copy;
        }
    }

    /* Check completion. */
    uint32_t expected_mask = (n_frags >= 32) ? 0xFFFFFFFFu
                                             : ((1u << n_frags) - 1u);
    if ((slot->fragments_bitmap & expected_mask) == expected_mask
        && !slot->committed) {
        commit_slot(s, slot, corr, dm);
        slot->committed = true;
    }
}


/* -----------------------------------------------------------------------
 * Header validation — mirrors prod_frame.unpack_frame
 * ---------------------------------------------------------------------- */
static int validate_and_unpack_header(rx_epoll_t *s, const uint8_t *buf,
                                      size_t buf_len, prod_hdr_t *out_hdr,
                                      const uint8_t **out_payload,
                                      size_t *out_payload_len) {
    if (buf_len < PROD_FRAME_HEADER_BYTES) {
        atomic_fetch_add_explicit(&s->bad_length_count, 1,
                                  memory_order_relaxed);
        return -1;
    }
    /* Copy via memcpy to handle alignment; modern x86 doesn't need it
     * for packed-struct loads, but it's free and safer. */
    memcpy(out_hdr, buf, PROD_FRAME_HEADER_BYTES);

    if (out_hdr->magic != PROD_FRAME_MAGIC) {
        atomic_fetch_add_explicit(&s->bad_magic_count, 1,
                                  memory_order_relaxed);
        return -1;
    }
    if (out_hdr->version != PROD_FRAME_VERSION) {
        atomic_fetch_add_explicit(&s->bad_version_count, 1,
                                  memory_order_relaxed);
        return -1;
    }

    /* Payload-length consistency check. */
    size_t payload_len_actual = buf_len - PROD_FRAME_HEADER_BYTES;
    if ((size_t)out_hdr->payload_bytes_in_frag != payload_len_actual) {
        atomic_fetch_add_explicit(&s->bad_length_count, 1,
                                  memory_order_relaxed);
        return -1;
    }

    /* Field-range validation. */
    if (!valid_bits_per_cell(out_hdr->bits_per_cell)
        || !valid_t_int_factor(out_hdr->t_int_factor)) {
        atomic_fetch_add_explicit(&s->bad_field_range_count, 1,
                                  memory_order_relaxed);
        return -1;
    }
    /* Reject reserved-bit2 set. */
    if (out_hdr->flags & FLAG_RESERVED_BIT2) {
        atomic_fetch_add_explicit(&s->reserved_bit_count, 1,
                                  memory_order_relaxed);
        return -1;
    }
    /* Reject any undefined v1 flag bits. */
    if (out_hdr->flags & ~FLAG_DEFINED_V1_MASK) {
        atomic_fetch_add_explicit(&s->bad_field_range_count, 1,
                                  memory_order_relaxed);
        return -1;
    }

    *out_payload = buf + PROD_FRAME_HEADER_BYTES;
    *out_payload_len = payload_len_actual;
    return 0;
}


/* -----------------------------------------------------------------------
 * Hot path — recvmmsg → validate → reorder window
 * ---------------------------------------------------------------------- */
static void drain_one_batch(rx_epoll_t *s) {
    int n = recvmmsg(s->sockfd, s->mmsg, RECV_BATCH, MSG_DONTWAIT, NULL);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        /* Non-recoverable: just bail; caller will retry on next epoll wakeup. */
        return;
    }
    for (int i = 0; i < n; i++) {
        unsigned msg_len = s->mmsg[i].msg_len;
        if (msg_len == 0) continue;
        atomic_fetch_add_explicit(&s->bytes_received_total, msg_len,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&s->n_received, 1, memory_order_relaxed);

        prod_hdr_t hdr;
        const uint8_t *payload;
        size_t payload_len;
        if (validate_and_unpack_header(s, s->bufs[i], msg_len, &hdr,
                                       &payload, &payload_len) != 0) {
            continue;
        }
        /* pattern_id check. */
        if (hdr.chgroup < MAX_CHGROUPS
            && atomic_load_explicit(&s->expected_pid_set[hdr.chgroup],
                                    memory_order_relaxed)) {
            uint64_t expected =
                atomic_load_explicit(&s->expected_pid[hdr.chgroup],
                                     memory_order_relaxed);
            if (hdr.pattern_id != expected) {
                atomic_fetch_add_explicit(&s->pattern_mismatch_count, 1,
                                          memory_order_relaxed);
                continue;
            }
        }
        ingest_fragment(s, &hdr, payload, payload_len);
    }
}

static void *epoll_thread_main(void *arg) {
    rx_epoll_t *s = (rx_epoll_t *)arg;
    struct epoll_event ev;
    while (atomic_load_explicit(&s->run_flag, memory_order_acquire)) {
        int n = epoll_wait(s->epfd, &ev, 1, 100 /* ms */);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        /* Drain everything available; multiple recvmmsg passes per wakeup
         * because at 7 Gb/s an EPOLLIN can deliver many batches before
         * the next epoll_wait. */
        if (n > 0 && (ev.events & EPOLLIN)) {
            for (int i = 0; i < 64; i++) {
                drain_one_batch(s);
            }
        }
    }
    return NULL;
}


/* -----------------------------------------------------------------------
 * Public API (called from Python via ctypes)
 * ---------------------------------------------------------------------- */

int recv_epoll_open(const char *bind_host, uint16_t bind_port,
                    int so_rcvbuf_bytes, uint16_t *out_actual_port) {
    if (g_state_init) {
        return -1;  /* singleton already open */
    }
    memset(&g_state, 0, sizeof(g_state));
    rx_epoll_t *s = &g_state;

    s->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (s->sockfd < 0) return -2;

    int one = 1;
    setsockopt(s->sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (so_rcvbuf_bytes > 0) {
        /* SO_RCVBUF doubles internally; we set what the user asked for and
         * ignore failures (it'll cap at /proc/sys/net/core/rmem_max). */
        setsockopt(s->sockfd, SOL_SOCKET, SO_RCVBUF, &so_rcvbuf_bytes,
                   sizeof(so_rcvbuf_bytes));
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(bind_port);
    if (bind_host == NULL || bind_host[0] == '\0') {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, bind_host, &addr.sin_addr) != 1) {
        close(s->sockfd);
        return -3;
    }
    if (bind(s->sockfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(s->sockfd);
        return -4;
    }
    /* Read back the actual port (in case user passed 0). */
    socklen_t alen = sizeof(addr);
    if (getsockname(s->sockfd, (struct sockaddr *)&addr, &alen) == 0) {
        if (out_actual_port) *out_actual_port = ntohs(addr.sin_port);
    }

    s->epfd = epoll_create1(EPOLL_CLOEXEC);
    if (s->epfd < 0) {
        close(s->sockfd);
        return -5;
    }
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = s->sockfd;
    if (epoll_ctl(s->epfd, EPOLL_CTL_ADD, s->sockfd, &ev) < 0) {
        close(s->epfd);
        close(s->sockfd);
        return -6;
    }

    /* Initialise recvmmsg buffer ring. */
    for (int i = 0; i < RECV_BATCH; i++) {
        s->iovs[i].iov_base = s->bufs[i];
        s->iovs[i].iov_len = RECV_BUF_BYTES;
        s->mmsg[i].msg_hdr.msg_iov = &s->iovs[i];
        s->mmsg[i].msg_hdr.msg_iovlen = 1;
        s->mmsg[i].msg_hdr.msg_name = NULL;
        s->mmsg[i].msg_hdr.msg_namelen = 0;
        s->mmsg[i].msg_hdr.msg_control = NULL;
        s->mmsg[i].msg_hdr.msg_controllen = 0;
        s->mmsg[i].msg_hdr.msg_flags = 0;
        s->mmsg[i].msg_len = 0;
    }

    g_state_init = true;
    return 0;
}

int recv_epoll_set_expected_pid(uint32_t chgroup, uint64_t pattern_id) {
    if (!g_state_init) return -1;
    if (chgroup >= MAX_CHGROUPS) return -2;
    atomic_store_explicit(&g_state.expected_pid[chgroup], pattern_id,
                          memory_order_relaxed);
    atomic_store_explicit(&g_state.expected_pid_set[chgroup], 1,
                          memory_order_relaxed);
    return 0;
}

int recv_epoll_clear_expected_pid(uint32_t chgroup) {
    if (!g_state_init) return -1;
    if (chgroup >= MAX_CHGROUPS) return -2;
    atomic_store_explicit(&g_state.expected_pid_set[chgroup], 0,
                          memory_order_relaxed);
    return 0;
}

int recv_epoll_start(void) {
    if (!g_state_init) return -1;
    if (g_state.thread_started) return 0;
    atomic_store_explicit(&g_state.run_flag, 1, memory_order_release);
    int rc = pthread_create(&g_state.thread, NULL, epoll_thread_main,
                            &g_state);
    if (rc != 0) {
        atomic_store_explicit(&g_state.run_flag, 0, memory_order_release);
        return -2;
    }
    g_state.thread_started = true;
    return 0;
}

int recv_epoll_stop(void) {
    if (!g_state_init) return -1;
    if (!g_state.thread_started) return 0;
    atomic_store_explicit(&g_state.run_flag, 0, memory_order_release);
    pthread_join(g_state.thread, NULL);
    g_state.thread_started = false;
    return 0;
}

int recv_epoll_close(void) {
    if (!g_state_init) return -1;
    if (g_state.thread_started) recv_epoll_stop();
    if (g_state.epfd >= 0) close(g_state.epfd);
    if (g_state.sockfd >= 0) close(g_state.sockfd);
    g_state_init = false;
    return 0;
}


/* -----------------------------------------------------------------------
 * Counter getters
 * ---------------------------------------------------------------------- */
#define COUNTER_GETTER(NAME) \
uint64_t recv_epoll_get_##NAME(void) { \
    if (!g_state_init) return 0; \
    return atomic_load_explicit(&g_state.NAME, memory_order_relaxed); \
}

COUNTER_GETTER(n_received)
COUNTER_GETTER(n_committed)
COUNTER_GETTER(bad_magic_count)
COUNTER_GETTER(bad_version_count)
COUNTER_GETTER(bad_length_count)
COUNTER_GETTER(bad_field_range_count)
COUNTER_GETTER(reserved_bit_count)
COUNTER_GETTER(pattern_mismatch_count)
COUNTER_GETTER(window_slide_zerofill_count)
COUNTER_GETTER(out_of_order_drop_count)
COUNTER_GETTER(bytes_received_total)
#undef COUNTER_GETTER
