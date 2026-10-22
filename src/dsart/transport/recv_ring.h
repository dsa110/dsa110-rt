/*
 * recv_ring.h — public C API for the M4a chunk-4 POSIX-shm SPMC sparse
 * receive ring. Source of truth: recv_ring.c (this file just lifts the
 * extern-public surface so other in-tree C extensions can link against
 * the same writer/reader semantics without forking the implementation).
 *
 * Two consumers in-tree today:
 *   - _recv_ring.so   — pure ring extension (Python wrapper uses ctypes
 *                       to drive create / attach / write / read directly).
 *   - _recv_epoll.so  — UDP epoll receive loop (M4a chunk 6); the M7.2
 *                       Phase B work uses ``rx_ring_open_or_create`` +
 *                       ``rx_ring_write_slot`` to publish reassembled
 *                       cube slots into the same shm contract.
 *
 * Both .so files compile against the same recv_ring.c so they end up
 * with their own static copy of the symbols. The shared state lives in
 * POSIX shm (named by the caller), so the two extensions cooperate via
 * the kernel and not via any cross-.so symbol resolution.
 *
 * Wire / atomic protocol: see recv_ring.c top-of-file and plan §4.4
 * lines 1463-1475 (CONC-1 release/acquire ordering).
 */

#ifndef DSART_TRANSPORT_RECV_RING_H_
#define DSART_TRANSPORT_RECV_RING_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Validity-flag bit positions (mirror recv_ring.c). Producers OR these
 * together when calling rx_ring_write_slot. */
#define RX_RING_VF_DATA_PRESENT     (1u << 0)
#define RX_RING_VF_PATTERN_MISMATCH (1u << 1)
#define RX_RING_VF_RESERVED_B2      (1u << 2)
#define RX_RING_VF_RESERVED_B3      (1u << 3)
#define RX_RING_VF_RX_OVERRUN       (1u << 4)
#define RX_RING_VF_RFI_WARMING_UP   (1u << 5)

/* Opaque handle. */
typedef struct rx_ring rx_ring_t;

/* Lifecycle. */
rx_ring_t *
rx_ring_open_or_create(
    const char *name,
    uint32_t    n_corr,
    uint32_t    n_coarse_dm,
    uint32_t    t_buf_samples,
    uint32_t    n_filled_per_corr,
    uint32_t    bytes_per_cell,
    int         owner,            /* 1 = create/writer; 0 = attach/reader */
    char       *errbuf,
    size_t      errbuf_len
);

void rx_ring_close(rx_ring_t *ring);

int  rx_ring_unlink(const char *name);

/* Producer (RX side). M7.4 amend (v2 slot layout): per-slot scale + offset
 * are now persisted alongside the payload bytes so the search-side
 * scatter (rx_ring_assemble_dense_block) can dequantise correctly.
 * Producers that don't have dequant info (e.g. zerofill / pattern-
 * mismatch stubs) should pass scale=0.0f, offset=0.0f — the assembler
 * treats zero scale as "skip dequant contribution" so the dense
 * accumulator stays correct. */
int
rx_ring_write_slot(
    rx_ring_t  *ring,
    uint32_t    corr,
    uint32_t    dm,
    uint64_t    t_seq,
    const void *payload,
    size_t      payload_bytes,
    float       scale,
    float       offset,
    uint16_t    validity_flags
);

/* Consumer (compute side). */
int
rx_ring_read_slot(
    rx_ring_t *ring,
    uint32_t   corr,
    uint32_t   dm,
    uint64_t   t_seq,
    uint32_t   compute_half,
    void      *out_payload,
    size_t     out_payload_bytes,
    uint16_t  *out_validity
);

void
rx_ring_update_read_seq(rx_ring_t *ring, uint32_t compute_half,
                        uint64_t new_read_seq);

uint64_t rx_ring_get_write_seq(rx_ring_t *ring, uint32_t corr);

uint64_t rx_ring_get_overrun_count(rx_ring_t *ring, uint32_t compute_half);

void rx_ring_memset_data(rx_ring_t *ring);

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
);

/* M7.4: batched dense-scatter walker. Replaces the per-cube Python
 * ``rx_ring_read_slot`` × N_corr × T_det loop AND the COO→dense scatter
 * that would otherwise live in pure Python. See recv_ring.c for the
 * full memory contract. */
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
);

/* M7.4.1 GPU-scatter (2026-05-27): like ``rx_ring_assemble_dense_block``
 * but writes the compact COO wire payload [N_corr, T_det, n_filled_max*2]
 * cint8 instead of the dense per-(corr, t) ``[2, N_grid, N_grid]`` plane.
 * The dense scatter then happens GPU-side from this compact buffer,
 * eliminating the 565 MiB CPU memset + the 565 MiB H2D in favour of a
 * ~30 MiB H2D (n_filled=5000, N_corr=16, T_det=192 → 30 MiB compact vs
 * 565 MiB dense, ~19× reduction). The compact buffer carries the same
 * wire bytes (re_0, im_0, re_1, im_1, …) the slot already has, so the
 * C body collapses to ``memcpy(slot.payload, compact[corr, t, :], n*2)``
 * per slot — no per-cell loop, no per-corr LUT touch on CPU. The LUT
 * lives GPU-side permanently after a one-time H2D at startup.
 *
 * Same validation + skipping semantics as ``rx_ring_assemble_dense_block``;
 * invalid slots leave the compact row at zero (the GPU kernel skips
 * scatter when scale==0 → the dense plane stays at the memset-0 baseline
 * the GPU re-applies each cube). */
int
rx_ring_assemble_compact_block(
    rx_ring_t      *ring,
    uint64_t        specnum_start,
    uint32_t        t_det,
    uint32_t        sidecar_t_stride,         /* T axis of the sidecars (>= t_det) */
    uint32_t        owned_dm,
    uint32_t        compute_half,
    uint32_t        n_active_dms_per_corr,
    const int32_t  *n_filled_per_corr,        /* [n_corr] */
    int8_t         *out_cells_packed,         /* [n_corr * t_det * n_filled_max * 2] */
    uint32_t        n_filled_max,             /* wire ``n_filled_per_corr`` from header */
    float          *out_scale_per_t,          /* [n_corr * sidecar_t_stride] */
    float          *out_offset_re_per_t,      /* [n_corr * sidecar_t_stride] */
    float          *out_offset_im_per_t,      /* [n_corr * sidecar_t_stride] */
    uint8_t        *out_validity_per_t,       /* [t_det] */
    uint64_t       *out_n_overrun,
    uint64_t       *out_n_pattern_mismatch,
    uint64_t       *out_n_no_data_present
);

/* =======================================================================
 * Ring v4 (x2 2x time resolution; docs/x2/SPEC.md §6). Independent of the
 * v3 ring above (different shm segment, magic shared, version 4). Owned
 * DMs only; layout [dm][stream][t][row_bytes] with a per-row u64 tag.
 * Writer: the RX (recv_epoll v4 mode, or rx4_ingest from Python tests).
 * Readers map the segment PROT_READ and use the const-pointer helpers
 * below; they never write any byte of the segment.
 * ===================================================================== */

#define RX4_MAX_DM          16
#define RX4_MAX_STREAM      64

/* Per-row validity returned to readers (uint8 per (stream, row)). */
#define RX4_RV_PRESENT          (1u << 0)  /* data present and intact */
#define RX4_RV_HOLE             (1u << 1)  /* declared lost by the RX; row zeroed */
#define RX4_RV_PATTERN_MISMATCH (1u << 2)  /* frame pattern_id mismatch; row zeroed */
#define RX4_RV_MISSING          (1u << 3)  /* not written (yet); ring bytes stale */
#define RX4_RV_LAPPED           (1u << 4)  /* overwritten by a newer lap */
#define RX4_RV_RFI_WARMING_UP   (1u << 5)  /* present; frame flag bit4 */
#define RX4_RV_NOISE_WARMUP     (1u << 6)  /* present; frame flag bit3 */

/* rx4_ingest return codes (also the counter index of the outcome). */
enum rx4_counter {
    RX4_C_DATAGRAMS = 0,     /* every call */
    RX4_C_OK,                /* rows placed from a data frame */
    RX4_C_BAD_LENGTH,
    RX4_C_BAD_MAGIC,
    RX4_C_BAD_VERSION,       /* includes v1 frames */
    RX4_C_BAD_FLAGS,
    RX4_C_BAD_FIELD,
    RX4_C_BAD_TIME,          /* t0_native != seq_first * t_int_factor */
    RX4_C_NOT_OWNED,         /* dm_idx not in this ring */
    RX4_C_TIER_MISMATCH,     /* t_int_factor != the DM's tier */
    RX4_C_FORMAT_MISMATCH,   /* bits / kind / n_sub / n_grid / mode / scales / row size */
    RX4_C_STREAM_MISMATCH,   /* n_values / n_heavy != the stream's map */
    RX4_C_PATTERN_MISMATCH,  /* rows placed as zero-filled mismatch rows */
    RX4_C_V1_FRAMES,         /* subset of BAD_VERSION */
    RX4_C_PORT_MISMATCH,     /* stream arrived on another chgroup's port */
    RX4_C_BYTES,
    RX4_C_SWEEPS,
    RX4_C_N
};

typedef struct rx4_config {
    uint32_t n_dm;            /* 1..16 owned coarse DMs */
    uint32_t n_stream;        /* 1..64 */
    uint32_t n_sub;           /* 1, 2, 4 */
    uint32_t row_bytes;       /* row stride, multiple of 8, >= every stream's bytes/sample */
    uint32_t bits_main;       /* 8, 10, 12 */
    uint32_t bits_heavy;      /* 16 (or 0: no stream has heavy values) */
    uint32_t value_kind;      /* 0 cells, 1 baselines */
    uint32_t n_grid;
    uint32_t quant_mode;      /* 0 mid-tread, 1 mid-rise (frame flag bit6) */
    float    s_main;          /* decode steps; checked bit-exactly iff check_scales */
    float    s_heavy;
    uint32_t check_scales;
    uint64_t hole_timeout_ns; /* sweep declares holes after this long without progress */
    uint32_t dm_global[RX4_MAX_DM];
    uint32_t t_int_factor[RX4_MAX_DM];
    uint32_t t_buf[RX4_MAX_DM];        /* rows per stream (retention) */
    uint32_t reorder_rows[RX4_MAX_DM]; /* rows behind max_end before a gap is a hole */
    uint32_t stream_n_values[RX4_MAX_STREAM];  /* 0 = unchecked */
    uint32_t stream_n_heavy[RX4_MAX_STREAM];
} rx4_config_t;

typedef struct rx4_ring rx4_ring_t;

/* Writer (owner). rx4_create unlinks any old segment of that name and
 * creates a fresh one; prefault != 0 touches every page up front. */
rx4_ring_t *rx4_create(const char *name, const rx4_config_t *cfg, int prefault,
                       char *errbuf, size_t errbuf_len);
void        rx4_close(rx4_ring_t *r);           /* state -> closed, unmap */
int         rx4_unlink(const char *name);
void       *rx4_base(rx4_ring_t *r);
uint64_t    rx4_size(rx4_ring_t *r);
int         rx4_ingest(rx4_ring_t *r, const uint8_t *buf, size_t len,
                       uint64_t now_ns, uint32_t port_chgroup /* UINT32_MAX = any */);
void        rx4_sweep(rx4_ring_t *r, uint64_t now_ns, uint64_t stream_mask);
int         rx4_set_expected_pid(rx4_ring_t *r, uint32_t stream, uint64_t pid);
int         rx4_clear_expected_pid(rx4_ring_t *r, uint32_t stream);
uint64_t    rx4_now_ns(void);                   /* CLOCK_MONOTONIC */

/* Readers: pure functions of a (read-only) mapping base. */
int rx4_abi(uint64_t *out, uint32_t n);
int rx4_scan(const void *base, uint32_t dm_local, uint64_t *out_complete_through,
             uint64_t *out_max_end, uint64_t *out_last_progress_ns);
int rx4_window_validity(const void *base, uint32_t dm_local, uint64_t t_first,
                        uint32_t n_rows, uint8_t *out_validity, uint64_t *out_counts);
int rx4_copy_window(const void *base, uint32_t dm_local, uint64_t t_first,
                    uint32_t n_rows, uint8_t *out_rows, float *out_steps,
                    uint8_t *out_validity, uint64_t *out_counts);
int rx4_recheck(const void *base, uint32_t dm_local, uint64_t t_first,
                uint32_t n_rows, uint8_t *validity, uint64_t *out_n_changed);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* DSART_TRANSPORT_RECV_RING_H_ */
