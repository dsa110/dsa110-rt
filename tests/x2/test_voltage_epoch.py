"""voltage_retention.capture_relative_specnum: 2x C2 triggers carry SNAP-epoch specnums."""
import pytest

from dsart.common.constants import BLOCK_SAMPLES_SPECNUM
from dsart.dump.voltage_ring import specnum_to_block_n
from dsart.services.voltage_retention import capture_relative_specnum


def test_capture_epoch_passes_through():
    assert capture_relative_specnum(123456, epoch="capture", utc_start_specnum=None) == 123456


def test_snap_epoch_lands_on_the_ring_block():
    utc = 7_140_393_504                                   # G2 production arm, 2026-11-09
    # page 1 (block_n 1) starts at the arm: an event there is block 1, one block later block 2
    assert specnum_to_block_n(capture_relative_specnum(utc, epoch="snap", utc_start_specnum=utc)) == 1
    ev = utc + 1800 * BLOCK_SAMPLES_SPECNUM + 100
    assert specnum_to_block_n(capture_relative_specnum(ev, epoch="snap", utc_start_specnum=utc)) == 1801
    # the failure it fixes: the absolute specnum maps ~3.5e6 blocks away from a ~2000-block ring
    assert specnum_to_block_n(ev) > 3_000_000


def test_snap_epoch_refuses_what_it_cannot_place():
    with pytest.raises(ValueError, match="utc_start_specnum unknown"):
        capture_relative_specnum(10**9, epoch="snap", utc_start_specnum=None)
    with pytest.raises(ValueError, match="precedes the arm"):
        capture_relative_specnum(100, epoch="snap", utc_start_specnum=10**6)
    with pytest.raises(ValueError, match="'capture' or 'snap'"):
        capture_relative_specnum(100, epoch="native", utc_start_specnum=0)
