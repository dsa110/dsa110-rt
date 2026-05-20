/* dsart_capture_manythread.h --- public structs for the dsart capture binary.
 *
 * Vendored from dsa110-xengine/src/dsaX_capture_manythread.h on 2026-05-20
 * (commit pin in src/dsart/capture/README.md).
 *
 * Trimmed to only the symbols actually used by dsart_capture_manythread.c
 * after the dsart-side improvements landed (recvmmsg + deterministic
 * arm + SO_RCVBUF + mon shm publisher). Wire-format constants live in
 * dsart_capture_def.h.
 */

#ifndef __DSART_CAPTURE_MANYTHREAD_H
#define __DSART_CAPTURE_MANYTHREAD_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>
#include <inttypes.h>
#include <sys/types.h>

#include <futils.h>
#include <dada_hdu.h>
#include <multilog.h>
#include <ipcio.h>
#include <ascii_header.h>
#include <dada_udp.h>

#include "dsart_capture_def.h"

/* Per-socket receive buffer. One per recv thread. */
typedef struct {
    int    fd;            /* fd of the bound UDP socket */
    size_t bufsz;         /* size of socket buffer (= UDP_PAYLOAD) */
    char  *buf;           /* the socket buffer (legacy single-packet path) */
    int    have_packet;
    size_t got;
} dsart_sock_t;

/* Per-write-thread context (memcpys from the staging tblock into the
 * PSRDADA hdu and advances the block). */
typedef struct {
    dada_hdu_t *hdu;
    uint64_t    hdu_bufsz;
    unsigned    block_open;
    char       *block;
    char       *tblock;
    int         thread_id;
} dsart_write_t;

/* Aggregated stats for the per-second stats_thread + shm publisher. */
typedef struct {
    stats_t  *packets;
    stats_t  *bytes;
    uint64_t *last_seq;
} dsart_stats_t;

/* Per-recv-thread context. */
typedef struct {
    multilog_t  *log;
    int          verbose;

    int          port;            /* UDP data port (4011 / 4012) */
    int          control_port;    /* UDP control port (legacy 11223 / 11224) */
    char        *interface;       /* bind IP for the data socket */

    unsigned int num_inputs;      /* NSNAPS */

    uint64_t    *block_start_byte;
    uint64_t    *block_end_byte;
    uint64_t    *block_count;
    uint64_t     hdu_bufsz;
    char        *tblock;          /* staging area shared across recv threads */

    unsigned    *capture_started;
    uint64_t     packets_per_buffer;

    stats_t *packets;
    stats_t *bytes;
    uint64_t rcv_sleeps;

    uint64_t  *last_seq;
    struct timeval timeout;
    int        thread_id;
} dsart_udpdb_t;

void stats_thread(void *arg);
void control_thread(void *arg);

#endif  /* __DSART_CAPTURE_MANYTHREAD_H */
