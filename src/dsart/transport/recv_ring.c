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
 *   plus per-slot scale/offset/validity sidecars immediately after each
 *   cell block.
 *
 * Slot layout per (corr, dm, t)  [VERSION 2, M7.4 — adds scale/offset]:
 *   offset 0              : payload (N_filled * bytes_per_cell bytes)
 *   offset N_filled*bpc   : float32 scale  (dequant: x = scale*q + offset_re/im)
 *   offset N_filled*bpc+4 : float32 offset (shared re/im — wire convention,
 *                                           symmetric cint8 → offset == 0)
 *   offset N_filled*bpc+8 : uint16 validity_flags
 *   total slot size       : N_filled*bpc + 10  (rounded up to 8 B alignment
 *                                               → +2 B padding worst case)
 *
 *   M7.4 motivation: the corr-side TX computes per-(cube, dm, t_idx) scale
 *   over the FILLED cells only (tx.py::_compute_scale_offset; symmetric
 *   cint8 ⇒ offset == 0). The search-side dense-scatter MUST dequantise
 *   each ring slot with the SAME scale that was used to encode it,
 *   otherwise per-slot dynamic range collapses. We persist the scale and
 *   offset into the ring slot as a sidecar so the search consumer
 *   (rx_ring_assemble_dense_block) can produce a per-(corr, t) dequant
 *   sidecar for the GPU imager. Prior to v2 the slot only carried the
 *   payload bytes + uint16 vf; M7.4_PLAN_FIXES.md and the comment block
 *   above `rx_ring_assemble_validity_block` (this file) called out the
 *   gap explicitly.
 *
 * Atomic write protocol (CONC-1 §4.4 line 1469; v2 amend):
 *   1. Write payload bytes.
 *   2. Store scale + offset (plain stores — atomicity is established by
 *      the validity_flags release store that follows; the consumer
 *      ALWAYS acquires on validity before reading scale/offset).
 *   3. Store validity_flags to slot header via atomic store release.
 *   4. Increment write_seq_per_corr[corr] with atomic store release.
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
/* M7.4 amend (2026-05-20): bump RX_RING_VERSION to 2. The v2 slot layout
 * stores per-slot scale (f32) + offset (f32) immediately after the
 * payload bytes and before the uint16 validity_flags. Old (v1) attached
 * consumers will fail the version check at attach time — by design;
 * mixed-version producers/consumers cannot share the same shm. The
 * orchestrator-level cleanup (_m72_*_cleanup.sh) clears stale shm
 * segments on every fleet redeploy. */
#define RX_RING_VERSION  UINT32_C(2)
#define N_COMPUTE        2                      /* plan §4.4 line 1471 */
#define HEADER_SIZE      4096                   /* 4 KiB header block */
#define CACHE_LINE       64

/* Sidecar byte sizes per slot (v2). Wire convention: symmetric cint8 →
 * offset == 0 for both re and im; we still ship offset on the wire (one
 * f32 in the ProdFrameHeader, mirrored here) so the dequant rule is
 * uniformly ``x = scale * q + offset`` regardless of quantiser choice. */
#define SCALE_OFFSET_BYTES (2u * sizeof(float))    /* 8 */
#define VF_BYTES           sizeof(uint16_t)        /* 2 */

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
    /* payload + f32 scale + f32 offset + uint16 validity_flags,
     * padded to 8-byte boundary (v2 layout). */
    size_t raw = (size_t)n_filled * bytes_per_cell
                 + SCALE_OFFSET_BYTES
                 + VF_BYTES;
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

static inline size_t
_payload_bytes(const rx_ring_header_t *h)
{
    return (size_t)h->n_filled_per_corr * h->bytes_per_cell;
}

static inline float *
_scale_ptr(rx_ring_t *ring, uint32_t corr, uint32_t dm, uint32_t t)
{
    uint8_t *slot = (uint8_t *)_slot_ptr(ring, corr, dm, t);
    return (float *)(slot + _payload_bytes(ring->hdr));
}

static inline float *
_offset_ptr(rx_ring_t *ring, uint32_t corr, uint32_t dm, uint32_t t)
{
    uint8_t *slot = (uint8_t *)_slot_ptr(ring, corr, dm, t);
    return (float *)(slot + _payload_bytes(ring->hdr) + sizeof(float));
}

static inline uint16_t *
_vf_ptr(rx_ring_t *ring, uint32_t corr, uint32_t dm, uint32_t t)
{
    uint8_t *slot = (uint8_t *)_slot_ptr(ring, corr, dm, t);
    return (uint16_t *)(slot
                        + _payload_bytes(ring->hdr)
                        + SCALE_OFFSET_BYTES);
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

    /* CONSUMER WRITE PATHS:
     *   rx_ring_read_slot bumps overrun_count_per_compute[half] on overrun
     *   rx_ring_update_read_seq writes read_seq_per_compute[half] on every
     *   cube release.
     * Neither field is writable through PROT_READ — a "read-only" attach
     * therefore segfaults on the first overrun/release. We open the fd
     * O_RDWR and map PROT_READ|PROT_WRITE on both sides. The OS-level
     * isolation between writer and consumer is intentionally weak: the
     * SPMC contract is enforced by the C atomic protocol, not by
     * mmap protection bits.
     */
    int oflags = owner ? (O_CREAT | O_RDWR) : O_RDWR;
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

    int prot = PROT_READ | PROT_WRITE;
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
 * rx_ring_write_slot — write one assembled COO slot (v2 layout, M7.4).
 *
 * CONC-1 atomic write protocol (v2 amend):
 *   1. Copy payload bytes to slot.
 *   2. Store per-slot scale + offset (plain stores; the release-store on
 *      validity_flags below establishes happens-before for the consumer's
 *      acquire load of validity).
 *   3. Atomic store (release) validity_flags.
 *   4. Atomic increment (release) write_seq_per_corr[corr].
 *
 * t_seq is the absolute sequence number; we index by t_seq % T_buf.
 *
 * Args:
 *   ring, corr, dm, t_seq, payload, payload_bytes, validity_flags — same
 *       as the v1 API.
 *   scale, offset                                                 — per-slot
 *       dequant parameters from the ProdFrame header (tx.py::
 *       _compute_scale_offset). The consumer's dequant rule is
 *       ``cell_f32 = scale * cint8 + offset``; for the standard
 *       symmetric cint8 path ``offset == 0``. Pass 0.0f / 0.0f for the
 *       legacy "no dequant info" case (validity drops to "stale" on the
 *       consumer side because dequant would be ambiguous).
 */
int
rx_ring_write_slot(
    rx_ring_t  *ring,
    uint32_t    corr,
    uint32_t    dm,
    uint64_t    t_seq,         /* absolute sequence; slot = t_seq % T_buf */
    const void *payload,
    size_t      payload_bytes,
    float       scale,
    float       offset,
    uint16_t    validity_flags
)
{
    if (!ring || !ring->hdr) return -1;
    rx_ring_header_t *h = ring->hdr;
    if (corr >= h->n_corr || dm >= h->n_coarse_dm) return -1;

    uint32_t t_idx = (uint32_t)(t_seq % h->t_buf_samples);
    void *slot = _slot_ptr(ring, corr, dm, t_idx);
    float *scp = _scale_ptr(ring, corr, dm, t_idx);
    float *offp = _offset_ptr(ring, corr, dm, t_idx);
    uint16_t *vfp = _vf_ptr(ring, corr, dm, t_idx);

    size_t expected = _payload_bytes(h);
    if (payload && payload_bytes > 0) {
        size_t copy_bytes = payload_bytes < expected ? payload_bytes : expected;
        memcpy(slot, payload, copy_bytes);
        if (copy_bytes < expected)
            memset((uint8_t *)slot + copy_bytes, 0, expected - copy_bytes);
    } else {
        memset(slot, 0, expected);
    }

    /* Step 2: persist per-slot dequant parameters. Plain stores; the
     * RELEASE store on validity_flags below sequences this BEFORE any
     * consumer acquires the slot. */
    *scp = scale;
    *offp = offset;

    /* Step 3: atomic store validity_flags (release). */
    __atomic_store_n(vfp, validity_flags, __ATOMIC_RELEASE);

    /* Step 4: atomic increment write_seq_per_corr (release). */
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
 * rx_ring_assemble_validity_block — batched validity-walk over a cube's
 * detector window.
 *
 * Replaces the ~16K Python-level rx_ring_read_slot calls per cube that
 * ProductionRxRingSource._assemble_cube used to do (M7.2 hot path —
 * search_compute observed 0.12 cubes/s on n01 with the Python loop vs
 * the 7.45 cubes/s production target). We do NOT memcpy payloads: the
 * M7.2 bring-up consumer hands a pre-allocated zero-filled per-chgroup
 * streams cache to the downstream CubePipeline (synthetic TX ships
 * all-zero payloads), so only the per-slot vf flags are needed to
 * compute the validity_mask. M7.4 will add a companion
 * rx_ring_assemble_dense_block that ALSO scatters quantised cells into
 * the dense per-chgroup grid using a pattern_id-keyed LUT.
 *
 * Semantics MIRROR ProductionRxRingSource._assemble_cube exactly:
 *
 *   For each (corr, dm, t) with bit dm set in coarse_dm_mask, t in
 *   [0, t_det):
 *     - If wseq_per_corr[corr] has advanced past t_abs + t_buf_samples
 *       the slot was lapped by the writer -> n_overrun++, bump
 *       overrun_count_per_compute[half], mark this t-row invalid.
 *     - Else read vf with __ATOMIC_ACQUIRE. VF_RX_OVERRUN ⇒ n_overrun;
 *       VF_PATTERN_MISMATCH ⇒ n_pattern_mismatch; missing
 *       VF_DATA_PRESENT ⇒ n_no_data_present. Any of those marks the
 *       t-row invalid.
 *
 *   The Python loop is conservative: ANY bad slot at ANY (corr, dm) for
 *   a given t invalidates the WHOLE [t, :] row. We preserve that
 *   semantic by returning a single uint8 per t — the caller broadcasts
 *   across the N_fdm dimension. (Per-coarse-dm validity tracking
 *   becomes interesting in M7.4 when each search_compute half processes
 *   only its owned coarse_dm.)
 *
 * Search-overlap geometry (M7.2 perf gate):
 *
 *   cube_cadence_samples is the stride BETWEEN cube emits — how many
 *   "new" samples enter the cube per emit. t_det is the WINDOW the
 *   detector consumes per cube. Production runs at
 *   cube_cadence_samples=128, t_det=192 ⇒ 64-sample search overlap
 *   that covers any pulse up to b_max=64 samples wide straddling a
 *   cube boundary. The wseq wait in _iter() must already gate on
 *   t_det samples having been written, so this walk is guaranteed
 *   to see in-window slots regardless of which cube initially wrote
 *   them.
 *
 * Cost model (n_corr=16, n_coarse_dm=8, t_det=192, all 8 dms in the
 * mask): ~25K atomic acquire loads on the vf bytes + 16 atomic
 * acquires on write_seq_per_corr ≈ 75 μs per cube on a 2080Ti host —
 * still four orders of magnitude faster than the per-slot Python loop.
 *
 * Args:
 *   ring                 attached SPMC ring handle.
 *   specnum_start        absolute t for slot 0 of the cube's window.
 *   cube_cadence_samples reserved for future per-cube book-keeping;
 *                        the walk uses ``t_det`` exclusively.
 *   t_det                detector window length (samples walked).
 *   compute_half         0 or 1, for the per-half overrun bump.
 *   coarse_dm_mask       bit i set ⇒ include dm=i (e.g. 0xFF = all 8).
 *   out_validity_per_t   uint8[t_det]; initialised to 1 on entry by
 *                        the helper, cleared to 0 per t-row where any
 *                        slot is bad.
 *   out_n_overrun        if non-NULL, += per-cube overrun count.
 *   out_n_pattern_mismatch  if non-NULL, += per-cube pattern-mismatch count.
 *   out_n_no_data_present   if non-NULL, += per-cube no-data-present count.
 *
 * Returns:
 *   0 on success;
 *  -1 if ring/out_validity_per_t is NULL or compute_half ≥ N_COMPUTE.
 */
int
rx_ring_assemble_validity_block(
    rx_ring_t *ring,
    uint64_t   specnum_start,
    uint32_t   cube_cadence_samples,
    uint32_t   t_det,
    uint32_t   compute_half,
    uint32_t   coarse_dm_mask,
    uint32_t   n_active_dms_per_corr,
    uint8_t   *out_validity_per_t,
    uint64_t  *out_n_overrun,
    uint64_t  *out_n_pattern_mismatch,
    uint64_t  *out_n_no_data_present
)
{
    if (!ring || !ring->hdr || !out_validity_per_t) return -1;
    if (compute_half >= N_COMPUTE) return -1;
    /* cube_cadence_samples is currently unused inside the walk (the
     * search-overlap geometry uses ``t_det`` for the walk and the
     * cadence drives the wseq wait in the Python iterator). Reserve
     * the parameter for future per-cube book-keeping (e.g. statistics
     * tagged with "samples-since-last-emit"). */
    (void)cube_cadence_samples;

    rx_ring_header_t *h = ring->hdr;
    const uint32_t n_corr        = h->n_corr;
    const uint32_t n_coarse_dm   = h->n_coarse_dm;
    const uint64_t t_buf         = h->t_buf_samples;
    const uint64_t slot_stride   = h->slot_stride_bytes;
    /* v2 slot layout: ``[payload | scale f32 | offset f32 | vf u16]``.
     * The vf bytes sit AFTER the 8-byte scale/offset sidecar. M7.4
     * amend — pre-M7.4 this was just ``n_filled * bpc``. */
    const size_t   vf_off_in_slot =
        (size_t)h->n_filled_per_corr * h->bytes_per_cell
        + SCALE_OFFSET_BYTES;
    uint8_t *data_base = (uint8_t *)ring->data;

    /* Initialise t_det rows to "all valid"; callers do not need to
     * pre-clear. The output buffer must be sized to at least t_det
     * bytes (the Python wrapper enforces this). */
    memset(out_validity_per_t, 1, (size_t)t_det);

    uint64_t n_overrun = 0;
    uint64_t n_pattern_mismatch = 0;
    uint64_t n_no_data_present = 0;

    for (uint32_t corr = 0; corr < n_corr; corr++) {
        /* One acquire-load per (corr) — write_seq is monotone, so a
         * single read covers the whole t-loop. A later writer advance
         * is a normal sliding-race window that the next cube assembly
         * will catch on its own wseq probe. */
        uint64_t wseq = __atomic_load_n(
            &h->write_seq_per_corr[corr], __ATOMIC_ACQUIRE
        );

        for (uint32_t dm = 0; dm < n_coarse_dm; dm++) {
            if (!((coarse_dm_mask >> dm) & 1u)) continue;

            /* Base pointer to the (corr, dm) sub-block of the ring;
             * each t indexes by slot_stride and wraps mod t_buf. */
            uint8_t *col_base = data_base
                + ((size_t)corr * n_coarse_dm + dm) * t_buf * slot_stride;

            /* M7.4 amend (2026-05-26): wseq is in slot-write units
             * (1 increment per rx_ring_write_slot call), but t_abs is
             * in sample units. With n_active_dms_per_corr writes per
             * sample, wseq grows ``n_active_dms_per_corr`` × faster
             * than t_abs; comparing them directly trips overrun for
             * any t_abs > t_buf / (n_active_dms_per_corr - 1) which
             * collapses validity to 0 for the entire run. Scale the
             * lap threshold accordingly. */
            const uint64_t n_active_safe =
                n_active_dms_per_corr > 0 ? n_active_dms_per_corr : 1;
            const uint64_t wseq_lap_threshold =
                (t_buf + (uint64_t)t_det) * n_active_safe;
            for (uint32_t t = 0; t < t_det; t++) {
                const uint64_t t_abs = specnum_start + (uint64_t)t;
                int bad = 0;

                if (wseq > t_abs * n_active_safe + wseq_lap_threshold) {
                    /* Slot was lapped by the writer (or never written
                     * within the ring window). Mirror
                     * rx_ring_read_slot's overrun semantics: bump the
                     * per-half overrun counter so the writer's
                     * back-pressure math stays consistent. */
                    n_overrun++;
                    __atomic_fetch_add(
                        &h->overrun_count_per_compute[compute_half],
                        1, __ATOMIC_RELAXED
                    );
                    bad = 1;
                } else {
                    const size_t t_idx = (size_t)(t_abs % t_buf);
                    /* vf is at (col_base + t_idx*stride + vf_off_in_slot)
                     * where vf_off_in_slot already includes the 8-byte
                     * scale/offset sidecar (v2 amend). */
                    uint16_t *vfp = (uint16_t *)(
                        col_base + t_idx * slot_stride + vf_off_in_slot
                    );
                    uint16_t vf = __atomic_load_n(vfp, __ATOMIC_ACQUIRE);
                    if (vf & VF_RX_OVERRUN) {
                        n_overrun++;
                        bad = 1;
                    } else if (vf & VF_PATTERN_MISMATCH) {
                        n_pattern_mismatch++;
                        bad = 1;
                    } else if (!(vf & VF_DATA_PRESENT)) {
                        n_no_data_present++;
                        bad = 1;
                    }
                }

                if (bad) {
                    /* t is bounded by the enclosing loop's `t < t_det`. */
                    out_validity_per_t[t] = 0;
                }
            }
        }
    }

    if (out_n_overrun)          *out_n_overrun          += n_overrun;
    if (out_n_pattern_mismatch) *out_n_pattern_mismatch += n_pattern_mismatch;
    if (out_n_no_data_present)  *out_n_no_data_present  += n_no_data_present;

    return 0;
}


/* -----------------------------------------------------------------------
 * rx_ring_assemble_dense_block (M7.4)
 *
 * Walks (corr, owned_dm, t) ring slots for the cube's t_det window and
 * scatters each slot's COO cint8 payload into a dense per-(corr, t)
 * plane via a caller-supplied linear-index LUT. Also captures the
 * per-slot scale + offset sidecar into per-(corr, t) sidecar arrays
 * for the GPU dequant kernel.
 *
 * Memory contract
 * ---------------
 * Inputs:
 *   - ring                 attached SPMC ring (v2 layout)
 *   - specnum_start        absolute t for slot 0 of the cube window
 *   - t_det                detector window length (rows walked)
 *   - n_grid               edge size of the dense grid (n_grid × n_grid)
 *   - n_filled_per_corr    int32[n_corr] — exact wire N_filled per corr
 *                          (each corr can have a different sparsity pattern
 *                          giving a different filled count; we walk only
 *                          the first n_filled_per_corr[corr] LUT entries)
 *   - linear_lut_strided   int32[n_corr * lut_stride] — per-corr LUT;
 *                          entry k for corr c is
 *                          ``linear_lut_strided[c*lut_stride + k]`` and
 *                          gives the flat ``ix_row*n_grid + ix_col``
 *                          target index in the dense [n_grid, n_grid]
 *                          plane. Padding entries (>= n_filled_per_corr[c])
 *                          are ignored.
 *   - lut_stride           leading dim of the LUT (≥ max n_filled).
 *   - owned_dm             coarse-DM index to scatter (one per call —
 *                          search compute halves own ONE coarse_dm each).
 *   - compute_half         0 or 1 (overrun counter bump).
 *
 * Outputs (caller-allocated, all stored as contiguous numpy arrays):
 *   - out_cint8            int8[n_corr * out_t_stride * 2 * n_grid * n_grid]
 *                          dense plane stack. The function ONLY writes
 *                          rows ``t in [0, t_det)`` AND only zeros
 *                          those same rows on entry (so the caller can
 *                          pre-allocate a larger T_stream-sized buffer
 *                          whose tail rows [t_det, T_stream) hold
 *                          carry-over from a previous cube — or stay
 *                          zero on cold start, which is what the
 *                          search-side currently uses). ``out_t_stride``
 *                          gives the leading T axis size of the dense
 *                          buffer so the corr-axis stride math is
 *                          correct even when t_stride > t_det.
 *                          Plane index 0 = re, 1 = im (split-plane;
 *                          matches the GPU kernel's expectations).
 *   - out_scale_per_t      float32[n_corr * out_t_stride]; per-(corr, t)
 *                          scale read from the slot's sidecar.
 *   - out_offset_re_per_t  float32[n_corr * out_t_stride]; per-(corr, t)
 *                          offset for the re plane. The wire ships ONE
 *                          offset per slot (symmetric cint8 → 0); we
 *                          duplicate it into the re/im outputs so the
 *                          GPU dequant kernel signature stays uniform.
 *   - out_offset_im_per_t  float32[n_corr * out_t_stride].
 *   - out_validity_per_t   uint8[t_det] (mirrors assemble_validity_block;
 *                          ANY (corr, t) bad slot ⇒ t-row invalid).
 *   - out_n_overrun        if non-NULL, += per-cube overrun count.
 *   - out_n_pattern_mismatch  if non-NULL, += per-cube pattern-mismatch count.
 *   - out_n_no_data_present   if non-NULL, += per-cube no-data-present count.
 *
 * Returns 0 on success; -1 on invalid args.
 *
 * Cost model (n_corr=16, t_det=192, n_filled≈3179, bpc=2):
 *   ~16 * 192 = 3072 atomic acquires on vf (one per (corr, t)) +
 *   ~3072 × 3179 byte stores (~10 MB) into the dense out_cint8 buffer.
 *   On a Skylake host this measures ~3-5 ms per cube — small compared
 *   to the 134 ms cube budget. The scatter is intentionally CPU-side
 *   (no GPU H2D dependency); the dense buffer is then passed to
 *   cube_pipeline._stage_h2d which performs ONE async H2D into the
 *   pinned GPU buffer.
 */
int
rx_ring_assemble_dense_block(
    rx_ring_t      *ring,
    uint64_t        specnum_start,
    uint32_t        t_det,
    uint32_t        out_t_stride,
    uint32_t        n_grid,
    uint32_t        owned_dm,
    uint32_t        compute_half,
    uint32_t        n_active_dms_per_corr,
    const int32_t  *n_filled_per_corr,
    const int32_t  *linear_lut_strided,
    uint32_t        lut_stride,
    int8_t         *out_cint8,
    float          *out_scale_per_t,
    float          *out_offset_re_per_t,
    float          *out_offset_im_per_t,
    uint8_t        *out_validity_per_t,
    uint64_t       *out_n_overrun,
    uint64_t       *out_n_pattern_mismatch,
    uint64_t       *out_n_no_data_present
)
{
    if (!ring || !ring->hdr) return -1;
    if (!n_filled_per_corr || !linear_lut_strided) return -1;
    if (!out_cint8 || !out_scale_per_t) return -1;
    if (!out_offset_re_per_t || !out_offset_im_per_t) return -1;
    if (!out_validity_per_t) return -1;
    if (compute_half >= N_COMPUTE) return -1;
    /* ``out_t_stride`` is the size of the T axis of the dense output
     * buffer the caller pre-allocated; we write rows [0, t_det) and
     * leave the tail rows [t_det, out_t_stride) untouched (so the
     * caller can carry T_stream lookahead from previous cubes). It
     * MUST be >= t_det. */
    if (out_t_stride < t_det) return -1;

    rx_ring_header_t *h = ring->hdr;
    const uint32_t n_corr        = h->n_corr;
    const uint32_t n_coarse_dm   = h->n_coarse_dm;
    const uint64_t t_buf         = h->t_buf_samples;
    const uint64_t slot_stride   = h->slot_stride_bytes;
    const uint32_t bpc           = h->bytes_per_cell;
    const size_t   payload_bytes = (size_t)h->n_filled_per_corr * bpc;
    uint8_t       *data_base     = (uint8_t *)ring->data;

    if (owned_dm >= n_coarse_dm) return -1;
    /* The wire layout is ``cint8 [re, im, re, im, ...]`` — see
     * tx.py::_encode_payload and rx.py::dequantise_cint8. Each cell is
     * exactly 2 bytes (re int8 + im int8). M7.4 v1 only supports the
     * cint8 path; the cfp16 wire variant would need a separate scatter
     * kernel (different element type, different dequant). */
    if (bpc != 2) return -1;

    const size_t n_grid_sq = (size_t)n_grid * n_grid;
    const size_t plane_bytes = n_grid_sq;            /* int8 */
    const size_t t_stride = 2 * n_grid_sq;
    const size_t corr_stride = (size_t)out_t_stride * t_stride;

    /* Initialise outputs. Dense cint8 MUST be cleared in the rows we
     * will read/write ([0, t_det)) because we may skip slots (bad
     * validity, overrun). Rows [t_det, out_t_stride) are NOT touched
     * — the caller owns their content (typically zeros from
     * fill(0) before each cube; or carry-over from a previous cube
     * if future M7.4 work wants lookahead). Scale / offset arrays
     * default to (0, 0, 0) — the GPU kernel treats zero scale as
     * "skip this (corr, t) contribution" so downstream sums stay
     * correct. */
    /* M7.4 perf (2026-05-27): considered parallelising the per-corr
     * memset + scatter with ``#pragma omp parallel for``, but the
     * extra threads SIGSEGV when ctypes calls in from CPython on
     * search nodes that own ``coarse_dm[7]`` (n13). Sticking with
     * single-threaded scatter for now; the dense memset is the
     * primary cost (~25 MiB/corr × 16 = 400 MiB of zero stores at
     * ~20 GiB/s → ~20 ms / cube). */
    for (uint32_t corr = 0; corr < n_corr; corr++) {
        int8_t *dense_corr = out_cint8 + (size_t)corr * corr_stride;
        memset(dense_corr, 0, (size_t)t_det * t_stride * sizeof(int8_t));
        memset(out_scale_per_t     + (size_t)corr * out_t_stride, 0,
               (size_t)t_det * sizeof(float));
        memset(out_offset_re_per_t + (size_t)corr * out_t_stride, 0,
               (size_t)t_det * sizeof(float));
        memset(out_offset_im_per_t + (size_t)corr * out_t_stride, 0,
               (size_t)t_det * sizeof(float));
    }
    memset(out_validity_per_t,  1, (size_t)t_det);

    uint64_t n_overrun = 0;
    uint64_t n_pattern_mismatch = 0;
    uint64_t n_no_data_present = 0;

    for (uint32_t corr = 0; corr < n_corr; corr++) {
        /* One acquire-load per corr — write_seq is monotone (see
         * assemble_validity_block for full rationale). */
        uint64_t wseq = __atomic_load_n(
            &h->write_seq_per_corr[corr], __ATOMIC_ACQUIRE
        );

        const int32_t nfilled_c = n_filled_per_corr[corr];
        if (nfilled_c < 0) continue;  /* corr is intentionally silent */
        const int32_t  *lut_c = linear_lut_strided
                                + (size_t)corr * lut_stride;

        /* Base pointer to the (corr, owned_dm) sub-block of the ring. */
        uint8_t *col_base = data_base
            + ((size_t)corr * n_coarse_dm + owned_dm) * t_buf * slot_stride;

        /* Dense output base for this corr (with out_t_stride leading
         * dim — matches the caller's T_stream-sized buffer). */
        int8_t *dense_corr = out_cint8 + (size_t)corr * corr_stride;
        float  *scale_corr  = out_scale_per_t
                              + (size_t)corr * out_t_stride;
        float  *offre_corr  = out_offset_re_per_t
                              + (size_t)corr * out_t_stride;
        float  *offim_corr  = out_offset_im_per_t
                              + (size_t)corr * out_t_stride;

        /* M7.4 amend (2026-05-26): wseq is in slot-write units; t_abs
         * is in sample units. See rx_ring_assemble_validity_block for
         * the full rationale. Scale the lap threshold by
         * ``n_active_dms_per_corr``. */
        const uint64_t n_active_safe =
            n_active_dms_per_corr > 0 ? n_active_dms_per_corr : 1;
        const uint64_t wseq_lap_threshold =
            (t_buf + (uint64_t)t_det) * n_active_safe;
        for (uint32_t t = 0; t < t_det; t++) {
            const uint64_t t_abs = specnum_start + (uint64_t)t;

            if (wseq > t_abs * n_active_safe + wseq_lap_threshold) {
                /* Slot was lapped. Same overrun semantics as
                 * rx_ring_read_slot / assemble_validity_block. */
                n_overrun++;
                __atomic_fetch_add(
                    &h->overrun_count_per_compute[compute_half],
                    1, __ATOMIC_RELAXED
                );
                out_validity_per_t[t] = 0;
                /* Leave dense plane + scale = 0. */
                continue;
            }

            const size_t t_idx = (size_t)(t_abs % t_buf);
            uint8_t *slot_base = col_base + t_idx * slot_stride;

            /* Acquire-load on vf gates the reads of payload + scale +
             * offset (the writer release-stored vf AFTER those writes). */
            uint16_t *vfp = (uint16_t *)(slot_base
                                         + payload_bytes
                                         + SCALE_OFFSET_BYTES);
            uint16_t vf = __atomic_load_n(vfp, __ATOMIC_ACQUIRE);

            int bad = 0;
            if (vf & VF_RX_OVERRUN) {
                n_overrun++; bad = 1;
            } else if (vf & VF_PATTERN_MISMATCH) {
                n_pattern_mismatch++; bad = 1;
            } else if (!(vf & VF_DATA_PRESENT)) {
                n_no_data_present++; bad = 1;
            }

            if (bad) {
                out_validity_per_t[t] = 0;
                continue;  /* dense plane + scale stay 0 */
            }

            /* Good slot — scatter payload + record dequant params. */
            const float scale = *(float *)(slot_base + payload_bytes);
            const float off   = *(float *)(slot_base + payload_bytes
                                          + sizeof(float));
            scale_corr[t] = scale;
            offre_corr[t] = off;
            offim_corr[t] = off;

            /* Walk filled cells: payload is [re_0, im_0, re_1, im_1, ...]
             * (see rx.py::dequantise_cint8, n_filled × 2 int8 row-major).
             * We scatter into the [re plane, im plane] split-plane GPU
             * layout via lut_c[k] = ix_row * n_grid + ix_col. */
            int8_t *re_plane = dense_corr + (size_t)t * t_stride;
            int8_t *im_plane = re_plane + plane_bytes;
            const int8_t *src = (const int8_t *)slot_base;

            for (int32_t k = 0; k < nfilled_c; k++) {
                const int32_t lin = lut_c[k];
                if ((uint32_t)lin >= n_grid_sq) continue;  /* defensive */
                re_plane[lin] = src[2 * k];
                im_plane[lin] = src[2 * k + 1];
            }
        }
    }

    if (out_n_overrun)          *out_n_overrun          += n_overrun;
    if (out_n_pattern_mismatch) *out_n_pattern_mismatch += n_pattern_mismatch;
    if (out_n_no_data_present)  *out_n_no_data_present  += n_no_data_present;

    return 0;
}


/*
 * rx_ring_assemble_compact_block (M7.4.1)
 *
 * Compact-payload variant of rx_ring_assemble_dense_block. Walks the
 * same (corr, owned_dm, t in [0, t_det)) slot set, performs the same
 * vf-bit validation, and writes the SAME per-(corr, t) sidecar arrays
 * (scale, offset_re, offset_im) and the validity bitmap. Diverges only
 * in the payload destination: instead of scattering cells into a
 * [N_corr, out_t_stride, 2, N_grid, N_grid] dense int8 plane via a
 * per-corr LUT, it memcpy's the raw slot's [n_filled_max * 2] cint8
 * wire bytes into a compact [N_corr, t_det, n_filled_max * 2] buffer.
 *
 * Memory model
 * ------------
 * - The slot's "payload" is the leading ``n_filled_max * bpc`` bytes of
 *   the slot. With ``bpc=2`` (cint8 path) that's exactly ``n_filled_max
 *   * 2`` bytes laid out as ``[re_0, im_0, re_1, im_1, …]`` — see
 *   tx.py::_encode_payload + rx.py::dequantise_cint8.
 * - We blindly memcpy ALL ``n_filled_max * 2`` bytes for valid slots
 *   even though only the leading ``n_filled_per_corr[corr] * 2`` of
 *   them carry data; the tail is wire-padding the producer left at
 *   zero (slot allocation is np.zeros at startup; producer touches
 *   only the [0, n_filled_per_corr[corr]) prefix). The GPU scatter
 *   kernel ignores entries beyond ``n_filled_per_corr[corr]`` via the
 *   per-corr LUT length, so reading the wire-zero tail is harmless.
 *   Net win: ONE memcpy per slot instead of a per-cell loop. On a
 *   Skylake host this is ~3 GiB/s ⇒ for 3072 slots × 10 KiB =
 *   30 MiB ⇒ ~10 ms (vs the dense path's 38 ms memset + 30 ms scatter
 *   = ~68 ms; ~3× CPU speedup on the source side alone).
 * - The compact buffer is zeroed in [0, t_det) rows on entry so
 *   invalid slots land at zero — the GPU scatter kernel skips
 *   contribution when ``scale==0`` so this is consistent with the
 *   dense path's "invalid → dense plane stays zero" semantics.
 *
 * Returns 0 on success; -1 on invalid args.
 */
int
rx_ring_assemble_compact_block(
    rx_ring_t      *ring,
    uint64_t        specnum_start,
    uint32_t        t_det,
    uint32_t        sidecar_t_stride,
    uint32_t        owned_dm,
    uint32_t        compute_half,
    uint32_t        n_active_dms_per_corr,
    const int32_t  *n_filled_per_corr,
    int8_t         *out_cells_packed,
    uint32_t        n_filled_max,
    float          *out_scale_per_t,
    float          *out_offset_re_per_t,
    float          *out_offset_im_per_t,
    uint8_t        *out_validity_per_t,
    uint64_t       *out_n_overrun,
    uint64_t       *out_n_pattern_mismatch,
    uint64_t       *out_n_no_data_present
)
{
    if (!ring || !ring->hdr) return -1;
    if (!n_filled_per_corr) return -1;
    if (!out_cells_packed) return -1;
    if (!out_scale_per_t || !out_offset_re_per_t || !out_offset_im_per_t)
        return -1;
    if (!out_validity_per_t) return -1;
    if (compute_half >= N_COMPUTE) return -1;
    /* Sidecars must be >= t_det wide so the imager's per-(g, t) scale
     * lookup (with stride = T_stream) lands in-bounds even when the
     * GPU dense plane uses a T_stream > t_det lookahead. */
    if (sidecar_t_stride < t_det) return -1;

    rx_ring_header_t *h = ring->hdr;
    const uint32_t n_corr        = h->n_corr;
    const uint32_t n_coarse_dm   = h->n_coarse_dm;
    const uint64_t t_buf         = h->t_buf_samples;
    const uint64_t slot_stride   = h->slot_stride_bytes;
    const uint32_t bpc           = h->bytes_per_cell;
    const size_t   payload_bytes = (size_t)h->n_filled_per_corr * bpc;
    uint8_t       *data_base     = (uint8_t *)ring->data;

    if (owned_dm >= n_coarse_dm) return -1;
    if (bpc != 2) return -1;
    /* n_filled_max must match the wire-side header EXACTLY: we memcpy
     * a fixed payload_bytes per slot and the compact buffer's row
     * stride is sized at n_filled_max*2. A mismatch would either
     * overflow the destination or skip cells. */
    if (n_filled_max != h->n_filled_per_corr) return -1;

    const size_t row_bytes = (size_t)n_filled_max * 2;  /* bpc=2 enforced */

    /* Zero the compact buffer (full [0, t_det) rows) + sidecars (full
     * [0, sidecar_t_stride) so the [t_det, T_stream) tail lookups by
     * the imager kernel see scale==0 = skip) + validity. */
    memset(out_cells_packed, 0, (size_t)n_corr * t_det * row_bytes);
    memset(out_scale_per_t,
           0, (size_t)n_corr * sidecar_t_stride * sizeof(float));
    memset(out_offset_re_per_t,
           0, (size_t)n_corr * sidecar_t_stride * sizeof(float));
    memset(out_offset_im_per_t,
           0, (size_t)n_corr * sidecar_t_stride * sizeof(float));
    memset(out_validity_per_t,  1, (size_t)t_det);

    uint64_t n_overrun = 0;
    uint64_t n_pattern_mismatch = 0;
    uint64_t n_no_data_present = 0;

    for (uint32_t corr = 0; corr < n_corr; corr++) {
        uint64_t wseq = __atomic_load_n(
            &h->write_seq_per_corr[corr], __ATOMIC_ACQUIRE
        );
        const int32_t nfilled_c = n_filled_per_corr[corr];
        if (nfilled_c < 0) continue;  /* intentionally silent */

        /* Base pointer to the (corr, owned_dm) sub-block of the ring. */
        uint8_t *col_base = data_base
            + ((size_t)corr * n_coarse_dm + owned_dm) * t_buf * slot_stride;

        int8_t *cells_corr = out_cells_packed
            + (size_t)corr * (size_t)t_det * row_bytes;
        float  *scale_corr = out_scale_per_t     + (size_t)corr * sidecar_t_stride;
        float  *offre_corr = out_offset_re_per_t + (size_t)corr * sidecar_t_stride;
        float  *offim_corr = out_offset_im_per_t + (size_t)corr * sidecar_t_stride;

        const uint64_t n_active_safe =
            n_active_dms_per_corr > 0 ? n_active_dms_per_corr : 1;
        const uint64_t wseq_lap_threshold =
            (t_buf + (uint64_t)t_det) * n_active_safe;

        for (uint32_t t = 0; t < t_det; t++) {
            const uint64_t t_abs = specnum_start + (uint64_t)t;

            if (wseq > t_abs * n_active_safe + wseq_lap_threshold) {
                n_overrun++;
                __atomic_fetch_add(
                    &h->overrun_count_per_compute[compute_half],
                    1, __ATOMIC_RELAXED
                );
                out_validity_per_t[t] = 0;
                continue;  /* leave compact row + scale = 0 */
            }

            const size_t t_idx = (size_t)(t_abs % t_buf);
            uint8_t *slot_base = col_base + t_idx * slot_stride;

            uint16_t *vfp = (uint16_t *)(slot_base
                                         + payload_bytes
                                         + SCALE_OFFSET_BYTES);
            uint16_t vf = __atomic_load_n(vfp, __ATOMIC_ACQUIRE);

            int bad = 0;
            if (vf & VF_RX_OVERRUN) {
                n_overrun++; bad = 1;
            } else if (vf & VF_PATTERN_MISMATCH) {
                n_pattern_mismatch++; bad = 1;
            } else if (!(vf & VF_DATA_PRESENT)) {
                n_no_data_present++; bad = 1;
            }

            if (bad) {
                out_validity_per_t[t] = 0;
                continue;
            }

            const float scale = *(float *)(slot_base + payload_bytes);
            const float off   = *(float *)(slot_base + payload_bytes
                                          + sizeof(float));
            scale_corr[t] = scale;
            offre_corr[t] = off;
            offim_corr[t] = off;

            /* COMPACT MEMCPY: ALL n_filled_max*2 bytes go to the dest
             * row. The tail beyond n_filled_per_corr[corr]*2 is
             * wire-zero (producer never writes there); reading it is
             * a benign small-memory waste and the GPU scatter ignores
             * those slots via the per-corr LUT length. */
            memcpy(cells_corr + (size_t)t * row_bytes,
                   slot_base,
                   row_bytes);
        }
    }

    if (out_n_overrun)          *out_n_overrun          += n_overrun;
    if (out_n_pattern_mismatch) *out_n_pattern_mismatch += n_pattern_mismatch;
    if (out_n_no_data_present)  *out_n_no_data_present  += n_no_data_present;

    return 0;
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
