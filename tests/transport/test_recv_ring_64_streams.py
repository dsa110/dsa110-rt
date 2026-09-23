"""RxRing with 64 streams (16 chgroups x n_sub=4).

The v2 header sized its per-corr write_seq / wrap_counter arrays [16], so
writing stream >= 16 overwrote the read_seq / overrun / slot_stride /
data_offset fields and write_slot segfaulted around stream 48. v3 sizes
them for RX_RING_MAX_CORR and every stream must round-trip independently.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from dsart.transport.recv_ring import (
    RX_RING_MAX_CORR,
    VF_DATA_PRESENT,
    RxRing,
    RxRingDims,
)

N_CORR = 64
N_FILLED = 24
T_BUF = 32


@pytest.fixture
def ring():
    name = f"/dsart-test-rx64-{uuid.uuid4().hex[:8]}"
    r = RxRing.open_or_create(
        name, RxRingDims(n_corr=N_CORR, n_coarse_dm=8, t_buf_samples=T_BUF,
                         n_filled_per_corr=N_FILLED, bytes_per_cell=2))
    yield r
    r.close()
    RxRing.unlink_name(name)


def _payload(c, t):
    return (np.arange(N_FILLED * 2, dtype=np.int16) + 7 * c + t
            ).astype(np.int8).tobytes()


def test_every_stream_round_trips(ring):
    n_t = 5
    for c in range(N_CORR):
        for t in range(n_t):
            ring.write_slot(corr=c, dm=3, t_seq=t, payload=_payload(c, t),
                            validity_flags=VF_DATA_PRESENT,
                            scale=0.5 + c, offset=0.0)
    for c in range(N_CORR):
        assert ring.get_write_seq(c) == n_t
    nfp = np.full(N_CORR, N_FILLED, dtype=np.int32)
    out = np.zeros((N_CORR, n_t, N_FILLED * 2), dtype=np.int8)
    res = ring.assemble_compact_block(
        specnum_start=0, t_det=n_t, owned_dm=3, n_filled_per_corr=nfp,
        n_filled_max=N_FILLED, sidecar_t_stride=n_t, compute_half=0,
        out_cells_packed=out)
    for c in (0, 15, 16, 47, 48, 63):
        for t in range(n_t):
            np.testing.assert_array_equal(
                out[c, t], np.frombuffer(_payload(c, t), dtype=np.int8))
    np.testing.assert_allclose(res[1][:, 0], 0.5 + np.arange(N_CORR))
    assert bool(np.all(res[4]))
    assert res[5] == 0 and res[7] == 0


def test_n_corr_above_max_is_refused():
    with pytest.raises(ValueError, match="n_corr"):
        RxRingDims(n_corr=RX_RING_MAX_CORR + 1, n_coarse_dm=8,
                   t_buf_samples=T_BUF, n_filled_per_corr=N_FILLED)
