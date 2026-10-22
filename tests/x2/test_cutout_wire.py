"""x2 D2: CutoutRequest / CutoutStatus codecs (SPEC §9 frozen layout)."""

from __future__ import annotations

import random
import struct

import pytest

from dsart.coinc import wire as W


def _req(**kw) -> W.CutoutRequest:
    base = dict(
        event_name="261022abcd", request_id=7, t_peak_ext=123_456_791,
        t0_ext=123_456_534, n_t=512, raw_pre_ext=954, raw_post_ext=954,
        width_ext=4, dm_peak_pc_cc=812.5, snr_peak=14.25, l_rad=0.0123,
        m_rad=-0.0045, l_pix=140, m_pix=97, fine_global_peak=301,
        coarse_global_peak=18, stamp_half_px=16, n_stamp_t=64,
        stamp_dm_halfwidth_pc_cc=150.0, mjd_peak=61335.123456789,
        plan_id=0xDEADBEEFCAFEF00D, deadline_s=15,
        trigger_class_id=W.class_id_for("bright_frb_extragalactic"),
        flags=W.CUTOUT_FLAG_CUTOUT | W.CUTOUT_FLAG_RAW_ARCHIVE,
    )
    base.update(kw)
    return W.CutoutRequest(**base)


def test_frozen_layout_offsets() -> None:
    """SPEC §9: the byte layout is frozen; pin sizes and a few offsets."""
    assert W.CUTOUT_REQUEST_SIZE == 128 and W.CUTOUT_STATUS_SIZE == 64
    assert W.CUTOUT_REQUEST_MAGIC == int.from_bytes(b"DSCQ", "little")
    assert W.CUTOUT_STATUS_MAGIC == int.from_bytes(b"DSCS", "little")
    b = W.encode_cutout_request(_req())
    assert len(b) == 128 and b[:4] == b"DSCQ"
    assert struct.unpack_from("<H", b, 4)[0] == 1
    assert b[8:18] == b"261022abcd"
    assert struct.unpack_from("<q", b, 32)[0] == 123_456_791      # t_peak
    assert struct.unpack_from("<q", b, 40)[0] == 123_456_534      # t0
    assert struct.unpack_from("<f", b, 64)[0] == pytest.approx(812.5)
    assert struct.unpack_from("<h", b, 86)[0] == 140             # l_pix
    assert struct.unpack_from("<Q", b, 104)[0] == 0xDEADBEEFCAFEF00D
    assert b[114:] == b"\x00" * 14


def test_request_round_trip() -> None:
    r = _req()
    d = W.decode_cutout_request(W.encode_cutout_request(r))
    assert d.event_name == r.event_name and d.key == r.key
    assert (d.t_peak_ext, d.t0_ext, d.n_t) == (r.t_peak_ext, r.t0_ext, r.n_t)
    assert d.dm_peak_pc_cc == pytest.approx(r.dm_peak_pc_cc)
    assert d.l_rad == pytest.approx(r.l_rad, rel=1e-6)
    assert d.mjd_peak == r.mjd_peak and d.plan_id == r.plan_id
    assert d.flags == r.flags and d.wants_raw and not d.is_manual
    assert d.has_pixel


def test_manual_request() -> None:
    r = _req(flags=W.CUTOUT_FLAG_CUTOUT | W.CUTOUT_FLAG_MANUAL, l_pix=-1,
             m_pix=-1, dm_peak_pc_cc=-1.0, fine_global_peak=-1,
             coarse_global_peak=-1)
    d = W.decode_cutout_request(W.encode_cutout_request(r))
    assert d.is_manual and not d.has_pixel and not d.wants_raw


@pytest.mark.parametrize("kw", [
    dict(n_t=511), dict(t0_ext=123_456_535), dict(t_peak_ext=1),
    dict(l_pix=-1), dict(event_name="x" * 17), dict(event_name=""),
    dict(flags=1 << 9), dict(width_ext=-1),
])
def test_request_validation(kw) -> None:
    with pytest.raises(W.BadBatch):
        W.encode_cutout_request(_req(**kw))


def test_request_decode_rejects_bad_bytes() -> None:
    b = bytearray(W.encode_cutout_request(_req()))
    with pytest.raises(W.BadBatch):
        W.decode_cutout_request(bytes(b[:64]))
    bad = bytearray(b)
    bad[0] ^= 0xFF
    with pytest.raises(W.BadBatch):
        W.decode_cutout_request(bytes(bad))
    bad = bytearray(b)
    struct.pack_into("<H", bad, 4, 2)
    with pytest.raises(W.BadBatch):
        W.decode_cutout_request(bytes(bad))


def test_status_round_trip_and_names() -> None:
    s = W.CutoutStatus(
        event_name="261022abcd", request_id=7, state=W.CUTOUT_STATE_PARTIAL,
        search_node_id=13, gpu_half=1, reason=W.CUTOUT_REASON_TOO_LATE,
        n_dm_owned=6, n_dm_ok=5, valid_frac=0.83, cutout_bytes=3_100_000,
        raw_bytes=812_000_000, latency_ms=6400, upload_rc=0)
    b = W.encode_cutout_status(s)
    assert len(b) == 64 and b[:4] == b"DSCS"
    d = W.decode_cutout_status(b)
    assert d.state_name == "partial" and d.reason_name == "too_late"
    assert (d.search_node_id, d.gpu_half, d.n_dm_owned, d.n_dm_ok) == \
        (13, 1, 6, 5)
    assert d.valid_frac == pytest.approx(0.83) and d.raw_bytes == 812_000_000
    assert d.is_terminal
    assert not W.decode_cutout_status(W.encode_cutout_status(
        W.CutoutStatus("261022abcd", 7, W.CUTOUT_STATE_ACCEPTED, 1, 0)
    )).is_terminal
    bad = bytearray(b)
    bad[6] = 9
    with pytest.raises(W.BadBatch):
        W.decode_cutout_status(bytes(bad))


def test_packets_never_confused_with_dsrt_trigger() -> None:
    """A 1x listener checks size==64 and the DSRT magic before dumping a
    cube (it ignores flags), so the new packets must fail both checks on
    the request and the magic on the status."""
    req = W.encode_cutout_request(_req())
    st = W.encode_cutout_status(
        W.CutoutStatus("261022abcd", 7, W.CUTOUT_STATE_DONE, 1, 0))
    trig = W.encode_c2_trigger(W.C2TriggerPacket("261022abcd", 5, 61335.0))
    assert len(req) != W.C2_TRIGGER_PACKET_SIZE
    assert int.from_bytes(st[:4], "little") != W.C2_TRIGGER_MAGIC
    with pytest.raises(W.BadBatch):
        W.decode_c2_trigger(st)
    assert W.packet_kind(req) == "cutout_request"
    assert W.packet_kind(st) == "cutout_status"
    assert W.packet_kind(trig) == "dsrt_trigger"
    assert W.packet_kind(b"xx") == "unknown"
    assert W.packet_kind(b"\x00" * 128) == "unknown"


def test_fuzz_never_raises_other_than_badbatch() -> None:
    rng = random.Random(1234)
    for n in (64, 128):
        for _ in range(500):
            buf = bytes(rng.getrandbits(8) for _ in range(n))
            for fn in (W.decode_cutout_request, W.decode_cutout_status):
                try:
                    fn(buf)
                except W.BadBatch:
                    pass


def test_class_id_is_stable_and_default_window() -> None:
    assert W.class_id_for("bright_frb_extragalactic") == \
        W.class_id_for("bright_frb_extragalactic")
    assert 0 <= W.class_id_for("x") <= 0xFFFF
    for tp in (1001, 1000, 7, 6):
        t0, n = W.default_cutout_window(tp, 512)
        assert t0 % 2 == 0 and n == 512 and t0 <= tp < t0 + n
    assert W.default_cutout_window(100, 511)[1] == 512
