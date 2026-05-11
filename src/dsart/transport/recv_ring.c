/*
 * recv_ring.c — POSIX-shm SPMC sparse receive ring (M4a chunk 4).
 *
 * Implements the CONC-1 contract from plan §4.4 lines 1463-1475.
 *
 * Layout overview
 * ===============
 *
 * The shm segment contains:
 *   [rx_ring_header (4 KiB padded)] [COO slot data]
 *
 * COO slot data is logically:
 *   [N_corr, N_coarse_DM, T_buf_samples, N_filled_per_corr * bytes_per_cell]
 *   plus per-slot validity_flags uint16 immediately after each cell block.
 *
 * Slot layout per (corr, dm, t):
 *   offset 0           : payload (N_filled * bytes_per_cell bytes)
 *   offset N_filled*bpc: uint16 validity_flags
 *   total slot size    : N_filled * bytes_per_cell + 2 (rounded up to 8 B alignment)
 *
 * Atomic write protocol (CONC-1 §4.4 line 1469):
 *   1. Write payload bytes.
 *   2. Store validity_flags to slot header (via atomic store release).
 *   3. Increment write_seq_per_corr[corr] with atomic store release.
 *
 * Compute reads write_seq_per_corr[corr] with atomic load acquire before
 * reading payload bytes, ensuring the compiler + CPU see the payload write
 * before the seq increment.
 *
 * SPMC: 1 writer (RX), N_compute=2 readers. RX never reads
 * read_seq_per_compute[] (plan §4.4 line 1471).
 *
 * Build: compiled via setuptools as _recv_ring extension (pyproject.toml).
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Try POSIX shm_open / shm_unlink from <sys/mman.h> (Linux). */
#ifndef SHM_NAME_MAX
#define SHM_NAME_MAX 255
#endif

/* -----------------------------------------------------------------------
 * Constants matching the plan contract
 * ---------------------------------------------------------------------- */
#define RX_RING_MAGIC    UINT32_C(0xD5A1107E)  /* "DSA110 7E" */
#define RX_RING_VERSION  UINT32_C(1)
#define N_COMPUTE        2                      /* plan §4.4 line 1471 */
#define HEADER_SIZE      4096                   /* 4 KiB header block */
#define CACHE_LINE       64

/* Validity-flags bit positions (plan §4.4 line 1470). */
#define VF_DATA_PRESENT     (1 << 0)
#define VF_PATTERN_MISMATCH (1 << 1)
#define VF_RESERVED_B2      (1 << 2)
#define VF_RESERVED_B3      (1 << 3)
#define VF_RX_OVERRUN       (1 << 4)
#define VF_RFI_WARMING_UP   (1 << 5)

/* -----------------------------------------------------------------------
 * rx_ring_header — 4 KiB block at shm base
 *
 * All uint64 fields are 8-byte aligned and accessed via __atomic_*_n
 * (release/acquire) per the CONC-1 contract.
 * ---------------------------------------------------------------------- */
typedef struct __attribute__((packed)) rx_ring_header {
    /* Magic + version + dimensions */
    uint32_t magic;                      /* 0xD5A1107E */
    uint32_t version;                    /* 1 */
    uint32_t n_corr;                     /* = 16 */
    uint32_t n_coarse_dm;
    uint32_t t_buf_samples;
    uint32_t n_filled_per_corr;
    uint32_t bytes_per_cell;             /* 2 (cint8 cplx) or 4 (cfp16 cplx) */
    uint32_t _pad0;                      /* align next field to 8 B */

    /* Per-corr write sequence (atomic; release on write) */
    uint64_t write_seq_per_corr[16];     /* offset 32 */

    /* Per-compute-half read sequence (read by compute; NOT read by RX) */
    uint64_t read_seq_per_compute[N_COMPUTE]; /* offset 32 + 16*8 = 160 */

    /* Per-corr wrap counters */
    uint64_t wrap_counter_per_corr[16]; /* offset 176 */

    /* Per-consumer overrun counters [N_compute] */
    uint64_t overrun_count_per_compute[N_COMPUTE]; /* offset 304 */

    /* Slot stride in bytes (per (dm, t) step for a given corr) */
    uint64_t slot_stride_bytes;          /* offset 320 */

    /* Total data section offset from shm base */
    uint64_t data_offset;               /* offset 328 */

    /* Padding to fill out 4 KiB */
    uint8_t _pad1[HEADER_SIZE - 336];
} rx_ring_header_t;

_Static_assert(sizeof(rx_ring_header_t) == HEADER_SIZE,
               "rx_ring_header_t size must be exactly 4096 bytes");

/* -----------------------------------------------------------------------
 * rx_ring — opaque handle
 * ---------------------------------------------------------------------- */
typedef struct rx_ring {
    rx_ring_header_t *hdr;    /* points to shm base */
    void             *data;   /* points to hdr + HEADER_SIZE */
    size_t            shm_size;
    int               shm_fd;
    char              shm_name[SHM_NAME_MAX + 1];
    int               owner;  /* 1 = created; 0 = attached read-only */
} rx_ring_t;

/* -----------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

static inline size_t
_slot_stride(uint32_t n_filled, uint32_t bytes_per_cell)
{
    /* payload + uint16 validity_flags, padded to 8-byte boundary */
    size_t raw = (size_t)n_filled * bytes_per_cell + sizeof(uint16_t);
    return (raw + 7) & ~(size_t)7;
}

static inline size_t
_ring_data_size(uint32_t n_corr, uint32_t n_coarse_dm,
                uint32_t t_buf, uint32_t n_filled, uint32_t bpc)
{
    return (size_t)n_corr * n_coarse_dm * t_buf * _slot_stride(n_filled, bpc);
}

static inline void *
_slot_ptr(rx_ring_t *ring, uint32_t corr, uint32_t dm, uint32_t t)
{
    rx_ring_header_t *h = ring->hdr;
    uint32_t t_idx = t % h->t_buf_samples;
    size_t offset = ((size_t)corr * h->n_coarse_dm * h->t_buf_samples
                     + (size_t)dm  * h->t_buf_samples
                     + (size_t)t_idx) * h->slot_stride_bytes;
    return (uint8_t *)ring->data + offset;
}

static inline uint16_t *
_vf_ptr(rx_ring_t *ring, uint32_t corr, uint32_t dm, uint32_t t)
{
    uint8_t *slot = (uint8_t *)_slot_ptr(ring, corr, dm, t);
    return (uint16_t *)(slot + (size_t)ring->hdr->n_filled_per_corr
                                * ring->hdr->bytes_per_cell);
}

/* -----------------------------------------------------------------------
 * Public API: create / attach / destroy
 * ---------------------------------------------------------------------- */

/*
 * rx_ring_open_or_create — create (owner=1) or attach (owner=0).
 *
 * If owner=1, initialises the header + zeroes the data section.
 * If owner=0, maps read-only and validates magic/version/dims.
 *
 * Returns: pointer to a heap-allocated rx_ring_t on success, NULL on error.
 * Writes a human-readable error message to errbuf (if non-NULL).
 */
rx_ring_t *
rx_ring_open_or_create(
    const char *name,
    uint32_t    n_corr,
    uint32_t    n_coarse_dm,
    uint32_t    t_buf_samples,
    uint32_t    n_filled_per_corr,
    uint32_t    bytes_per_cell,  /* 2 or 4 */
    int         owner,           /* 1 = create/writer; 0 = attach/reader */
    char       *errbuf,
    size_t      errbuf_len
)
{
#define ERRF(fmt, ...) do { \
    if (errbuf && errbuf_len > 0) \
        snprintf(errbuf, errbuf_len, fmt, ##__VA_ARGS__); \
} while (0)

    if (!name || name[0] == '\0') {
        ERRF("rx_ring_open_or_create: name is empty");
        return NULL;
    }
    if (bytes_per_cell != 2 && bytes_per_cell != 4) {
        ERRF("bytes_per_cell must be 2 or 4, got %u", bytes_per_cell);
        return NULL;
    }

    rx_ring_t *ring = (rx_ring_t *)calloc(1, sizeof(rx_ring_t));
    if (!ring) {
        ERRF("out of memory");
        return NULL;
    }

    strncpy(ring->shm_name, name, SHM_NAME_MAX);
    ring->shm_name[SHM_NAME_MAX] = '\0';
    ring->owner = owner;

    size_t data_size = _ring_data_size(
        n_corr, n_coarse_dm, t_buf_samples, n_filled_per_corr, bytes_per_cell
    );
    ring->shm_size = HEADER_SIZE + data_size;

    int oflags = owner ? (O_CREAT | O_RDWR) : O_RDONLY;
    mode_t mode = owner ? 0660 : 0;

    ring->shm_fd = shm_open(name, oflags, mode);
    if (ring->shm_fd < 0) {
        ERRF("shm_open(%s): %s", name, strerror(errno));
        free(ring);
        return NULL;
    }

    if (owner) {
        if (ftruncate(ring->shm_fd, (off_t)ring->shm_size) < 0) {
            ERRF("ftruncate: %s", strerror(errno));
            close(ring->shm_fd);
            free(ring);
            return NULL;
        }
    }

    int prot = owner ? (PROT_READ | PROT_WRITE) : PROT_READ;
    int flags = MAP_SHARED;

    void *base = mmap(NULL, ring->shm_size, prot, flags, ring->shm_fd, 0);
    if (base == MAP_FAILED) {
        ERRF("mmap: %s", strerror(errno));
        close(ring->shm_fd);
        free(ring);
        return NULL;
    }

    ring->hdr  = (rx_ring_header_t *)base;
    ring->data = (uint8_t *)base + HEADER_SIZE;

    if (owner) {
        /* Zero-init header + data. */
        memset(base, 0, ring->shm_size);

        /* Fill header. */
        ring->hdr->magic             = RX_RING_MAGIC;
        ring->hdr->version           = RX_RING_VERSION;
        ring->hdr->n_corr            = n_corr;
        ring->hdr->n_coarse_dm       = n_coarse_dm;
        ring->hdr->t_buf_samples     = t_buf_samples;
        ring->hdr->n_filled_per_corr = n_filled_per_corr;
        ring->hdr->bytes_per_cell    = bytes_per_cell;
        ring->hdr->slot_stride_bytes = (uint64_t)_slot_stride(
            n_filled_per_corr, bytes_per_cell
        );
        ring->hdr->data_offset       = (uint64_t)HEADER_SIZE;

        /* Ensure header writes are visible before any slot writes. */
        atomic_thread_fence(memory_order_release);
    } else {
        /* Validate existing header. */
        uint32_t magic = __atomic_load_n(&ring->hdr->magic, __ATOMIC_ACQUIRE);
        if (magic != RX_RING_MAGIC) {
            ERRF("bad magic: 0x%08x", magic);
            munmap(base, ring->shm_size);
            close(ring->shm_fd);
            free(ring);
            return NULL;
        }
        uint32_t ver = ring->hdr->version;
        if (ver != RX_RING_VERSION) {
            ERRF("bad version: %u", ver);
            munmap(base, ring->shm_size);
            close(ring->shm_fd);
            free(ring);
            return NULL;
        }
    }

    return ring;
#undef ERRF
}


/*
 * rx_ring_close — unmap the shm segment; close fd. Does NOT unlink.
 */
void
rx_ring_close(rx_ring_t *ring)
{
    if (!ring) return;
    if (ring->hdr)
        munmap(ring->hdr, ring->shm_size);
    if (ring->shm_fd >= 0)
        close(ring->shm_fd);
    free(ring);
}


/*
 * rx_ring_unlink — remove the shm name from the namespace.
 * Call after rx_ring_close.
 */
int
rx_ring_unlink(const char *name)
{
    return shm_unlink(name);
}


/* -----------------------------------------------------------------------
 * Public API: write slot (RX side)
 * ---------------------------------------------------------------------- */

/*
 * rx_ring_write_slot — write one assembled COO slot.
 *
 * CONC-1 atomic write protocol:
 *   1. Copy payload bytes to slot.
 *   2. Atomic store (release) validity_flags.
 *   3. Atomic increment (release) write_seq_per_corr[corr].
 *
 * t_seq is the absolute sequence number; we index by t_seq % T_buf.
 */
int
rx_ring_write_slot(
    rx_ring_t  *ring,
    uint32_t    corr,
    uint32_t    dm,
    uint64_t    t_seq,         /* absolute sequence; slot = t_seq % T_buf */
    const void *payload,
    size_t      payload_bytes,
    uint16_t    validity_flags
)
{
    if (!ring || !ring->hdr) return -1;
    rx_ring_header_t *h = ring->hdr;
    if (corr >= h->n_corr || dm >= h->n_coarse_dm) return -1;

    uint32_t t_idx = (uint32_t)(t_seq % h->t_buf_samples);
    void *slot = _slot_ptr(ring, corr, dm, t_idx);
    uint16_t *vfp = _vf_ptr(ring, corr, dm, t_idx);

    size_t expected = (size_t)h->n_filled_per_corr * h->bytes_per_cell;
    if (payload && payload_bytes > 0) {
        size_t copy_bytes = payload_bytes < expected ? payload_bytes : expected;
        memcpy(slot, payload, copy_bytes);
        if (copy_bytes < expected)
            memset((uint8_t *)slot + copy_bytes, 0, expected - copy_bytes);
    } else {
        memset(slot, 0, expected);
    }

    /* Step 2: atomic store validity_flags (release). */
    __atomic_store_n(vfp, validity_flags, __ATOMIC_RELEASE);

    /* Step 3: atomic increment write_seq_per_corr (release). */
    __atomic_fetch_add(&h->write_seq_per_corr[corr], 1, __ATOMIC_RELEASE);

    /* Advance wrap counter if we've wrapped around T_buf. */
    if (t_idx == 0 && t_seq > 0)
        __atomic_fetch_add(&h->wrap_counter_per_corr[corr], 1, __ATOMIC_RELEASE);

    return 0;
}


/* -----------------------------------------------------------------------
 * Public API: read slot (compute side)
 * ---------------------------------------------------------------------- */

/*
 * rx_ring_read_slot — read one COO slot and its validity_flags.
 *
 * Uses acquire semantics on read_seq_check (the write_seq_per_corr value)
 * to ensure payload bytes are visible before reading them.
 *
 * out_payload: caller-provided buffer, at least n_filled * bytes_per_cell bytes.
 * out_validity: filled with the slot's validity_flags.
 *
 * Returns 0 on success, -1 on error.
 *
 * Compute callers MUST call rx_ring_update_read_seq after consuming the
 * slot to advance their own read_seq_per_compute[compute_half].
 */
int
rx_ring_read_slot(
    rx_ring_t *ring,
    uint32_t   corr,
    uint32_t   dm,
    uint64_t   t_seq,
    uint32_t   compute_half,   /* 0 or 1 */
    void      *out_payload,
    size_t     out_payload_bytes,
    uint16_t  *out_validity
)
{
    if (!ring || !ring->hdr) return -1;
    rx_ring_header_t *h = ring->hdr;
    if (corr >= h->n_corr || dm >= h->n_coarse_dm) return -1;
    if (compute_half >= N_COMPUTE) return -1;

    /* Acquire load on write_seq to synchronise with the writer's release
     * store (plan §4.4 line 1469). */
    uint64_t wseq = __atomic_load_n(&h->write_seq_per_corr[corr], __ATOMIC_ACQUIRE);

    /* Check for overrun: if write_seq advanced past t_seq + T_buf, the slot
     * was overwritten. */
    if (wseq > t_seq + h->t_buf_samples) {
        /* Overrun: bump counter for this compute half. */
        __atomic_fetch_add(
            &h->overrun_count_per_compute[compute_half], 1, __ATOMIC_RELAXED
        );
        if (out_validity) *out_validity = VF_RX_OVERRUN;
        return -1;
    }

    uint32_t t_idx = (uint32_t)(t_seq % h->t_buf_samples);
    void *slot = _slot_ptr(ring, corr, dm, t_idx);
    uint16_t *vfp = _vf_ptr(ring, corr, dm, t_idx);

    /* Read validity_flags with acquire. */
    uint16_t vf = __atomic_load_n(vfp, __ATOMIC_ACQUIRE);
    if (out_validity) *out_validity = vf;

    if (out_payload && out_payload_bytes > 0) {
        size_t slot_data_bytes = (size_t)h->n_filled_per_corr * h->bytes_per_cell;
        size_t copy_bytes =
            out_payload_bytes < slot_data_bytes ? out_payload_bytes : slot_data_bytes;
        memcpy(out_payload, slot, copy_bytes);
    }

    return 0;
}


/*
 * rx_ring_update_read_seq — advance a compute reader's read sequence.
 */
void
rx_ring_update_read_seq(
    rx_ring_t *ring,
    uint32_t   compute_half,
    uint64_t   new_read_seq
)
{
    if (!ring || !ring->hdr) return;
    if (compute_half >= N_COMPUTE) return;
    __atomic_store_n(
        &ring->hdr->read_seq_per_compute[compute_half],
        new_read_seq,
        __ATOMIC_RELEASE
    );
}


/*
 * rx_ring_get_write_seq — read current write_seq_per_corr[corr] (acquire).
 */
uint64_t
rx_ring_get_write_seq(rx_ring_t *ring, uint32_t corr)
{
    if (!ring || !ring->hdr || corr >= ring->hdr->n_corr) return 0;
    return __atomic_load_n(&ring->hdr->write_seq_per_corr[corr], __ATOMIC_ACQUIRE);
}


/*
 * rx_ring_get_overrun_count — per-compute-half overrun counter.
 */
uint64_t
rx_ring_get_overrun_count(rx_ring_t *ring, uint32_t compute_half)
{
    if (!ring || !ring->hdr || compute_half >= N_COMPUTE) return 0;
    return __atomic_load_n(
        &ring->hdr->overrun_count_per_compute[compute_half], __ATOMIC_ACQUIRE
    );
}


/*
 * rx_ring_memset_data — zero-fill entire data section (cmd: prepare).
 * Must only be called by the RX (owner) process.
 */
void
rx_ring_memset_data(rx_ring_t *ring)
{
    if (!ring || !ring->hdr || !ring->owner) return;
    memset(ring->data, 0, ring->shm_size - HEADER_SIZE);
    atomic_thread_fence(memory_order_release);
}


/*
 * rx_ring_get_dims — expose dimensions to callers (useful for Python ctypes).
 */
void
rx_ring_get_dims(
    rx_ring_t *ring,
    uint32_t  *out_n_corr,
    uint32_t  *out_n_coarse_dm,
    uint32_t  *out_t_buf_samples,
    uint32_t  *out_n_filled_per_corr,
    uint32_t  *out_bytes_per_cell,
    uint64_t  *out_slot_stride_bytes,
    size_t    *out_shm_size
)
{
    if (!ring || !ring->hdr) return;
    if (out_n_corr)            *out_n_corr            = ring->hdr->n_corr;
    if (out_n_coarse_dm)       *out_n_coarse_dm       = ring->hdr->n_coarse_dm;
    if (out_t_buf_samples)     *out_t_buf_samples     = ring->hdr->t_buf_samples;
    if (out_n_filled_per_corr) *out_n_filled_per_corr = ring->hdr->n_filled_per_corr;
    if (out_bytes_per_cell)    *out_bytes_per_cell    = ring->hdr->bytes_per_cell;
    if (out_slot_stride_bytes) *out_slot_stride_bytes = ring->hdr->slot_stride_bytes;
    if (out_shm_size)          *out_shm_size          = ring->shm_size;
}
