"""Manifest absolute-time provenance (2026-07-16).

``write_window_to_staging`` records the capture arm anchor
(``armed_mjd``, ``utc_start_specnum``) and the derived
``block_mjd_first`` so downstream consumers (C3, dsavim) can compute
sample times on the slow-vis time base without trusting the C2 label
(which historically ran late by the search pipeline's first-cube fill
latency).
"""

from __future__ import annotations

import json

import numpy as np

from dsart.common.constants import BLOCK_DURATION_S
from dsart.dump.voltage_ring import VoltageRing
from dsart.services.voltage_retention import write_window_to_staging


def _block(nbytes: int, fill: int) -> np.ndarray:
    return np.full(nbytes, fill % 256, dtype=np.uint8)


def _write(tmp_path, **kwargs):
    bpb = 8
    ring = VoltageRing(n_blocks=4, bytes_per_block=bpb)
    for b in range(3):
        ring.store(b, _block(bpb, b + 1))
    return write_window_to_staging(
        ring=ring,
        event_name="testev",
        event_specnum=1 * 2048,  # target block 1
        cn_id=3,
        chgroup=0,
        staging_dir=tmp_path,
        n_pre=1,
        n_post=1,
        **kwargs,
    )


def test_manifest_records_arm_anchor(tmp_path) -> None:
    armed = 61236.5
    m = _write(
        tmp_path, mjd_target=61236.6, armed_mjd=armed,
        utc_start_specnum=21111787046,
    )
    assert m["armed_mjd"] == armed
    assert m["utc_start_specnum"] == 21111787046
    # block_n counts from 1 for the first armed block
    expected = armed + (m["block_n_first"] - 1) * BLOCK_DURATION_S / 86400.0
    assert abs(m["block_mjd_first"] - expected) < 1e-12
    # round-trips through the JSON sidecar
    doc = json.loads((tmp_path / "testev_sb00.json").read_text())
    assert doc["armed_mjd"] == armed
    assert doc["block_mjd_first"] == m["block_mjd_first"]


def test_manifest_anchor_absent_is_none(tmp_path) -> None:
    m = _write(tmp_path, mjd_target=61236.6)
    assert m["armed_mjd"] is None
    assert m["utc_start_specnum"] is None
    assert m["block_mjd_first"] is None
