/* dsart_capture_def.h --- minimal subset of the legacy dsaX_def.h
 *
 * Vendored from dsa110-xengine/src/dsaX_def.h on 2026-05-20.
 *
 * The legacy header pulls in xGPU, dedisp, beamformer, and filterbank
 * sizing constants that this binary doesn't use. We only keep the
 * symbols that dsart_capture_manythread.c actually references, so the
 * build does not need to drag in unrelated dependencies. The legacy
 * upstream is the source of truth for the *values* here -- if any of
 * these change in dsaX_def.h the vendored copy must follow.
 *
 * Provenance: dsa110-xengine commit pinned in dsa110-rt
 * src/dsart/capture/README.md.
 *
 * DO NOT edit the constant values without a coordinated update to
 * the SNAP firmware-emitted UDP wire format. Wire format is a
 * non-goal of this plan (see dsart plan doc Section 0 + Section 3).
 */
#ifndef __DSART_CAPTURE_DEF_H
#define __DSART_CAPTURE_DEF_H

#include <dada_def.h>  /* libpsrdada: provides DADA_DEFAULT_BLOCK_KEY, key_t etc. */

/* PSRDADA key for the SNAP-UDP -> dada/eada writer. The orchestrator
 * picks dada vs eada at spawn time via -k <key>. */
#define CAPTURE_BLOCK_KEY 0x0000dada

/* UDP wire format (unchanged from legacy SNAP firmware) */
#define UDP_HEADER   8              /* size of header/sequence number    */
#define UDP_DATA     4608           /* obs bytes per packet              */
#define UDP_PAYLOAD  4616           /* header + datasize                 */

/* SNAP topology: 32 SNAPs per capture pair (16 per UDP port).
 * Each SNAP packet carries 3 antennas; 32 SNAPs * 3 ants = 96 ants
 * in the merged voltage tensor. */
#define NSNAPS 32

#endif  /* __DSART_CAPTURE_DEF_H */
