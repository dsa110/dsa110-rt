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

/* Producer (RX side). */
int
rx_ring_write_slot(
    rx_ring_t  *ring,
    uint32_t    corr,
    uint32_t    dm,
    uint64_t    t_seq,
    const void *payload,
    size_t      payload_bytes,
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

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* DSART_TRANSPORT_RECV_RING_H_ */
