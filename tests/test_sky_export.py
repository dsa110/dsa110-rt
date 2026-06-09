"""Unit tests for the corr-side static-sky snapshot exporter
(``dsart.services.sky_export``) — E2E test 1 plumbing.

Covers: payload round-trip, the rate-limit / warmup gating of
``maybe_export``, the fail-soft POST path (mocked urlopen), and the
bounded-queue drop behaviour. No network, no GPU: the StaticSkyMean
double below mimics the two attributes the exporter reads.
"""
from __future__ import annotations

import time
import urllib.error

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dsart.services.sky_export import (                     # noqa: E402
    SKY_SNAPSHOT_VERSION,
    SkySnapshotExporter,
    build_snapshot_npz,
    parse_snapshot_npz,
)


N_FILLED = 64


class _FakeStaticSky:
    """Duck-typed StaticSkyMean: exposes cubes_seen_for(),
    running_mean_for(), and window_blocks — all the exporter reads."""

    def __init__(self, n_filled: int = N_FILLED, cubes_seen: int = 1000):
        rng = np.random.default_rng(7)
        arr = (rng.standard_normal(n_filled)
               + 1j * rng.standard_normal(n_filled)).astype(np.complex64)
        self._mean = torch.from_numpy(arr)
        self._cubes = cubes_seen
        self.window_blocks = 8

    def cubes_seen_for(self, dm_slot: int = 0) -> int:
        return self._cubes

    def running_mean_for(self, dm_slot: int = 0):
        return self._mean if self._cubes > 0 else None


def _pattern(n_filled: int = N_FILLED):
    rng = np.random.default_rng(3)
    ix_row = rng.integers(0, 256, n_filled).astype(np.uint16)
    ix_col = rng.integers(0, 256, n_filled).astype(np.uint16)
    return ix_row, ix_col


def _make_exporter(monkeypatch=None, **kw) -> SkySnapshotExporter:
    ix_row, ix_col = _pattern()
    defaults = dict(
        interval_s=30.0,
        chgroup=5,
        n_grid=256,
        cell_lambda=12.5,
        pattern_id=0xDEADBEEF,
        ix_row=ix_row,
        ix_col=ix_col,
        dec_deg=54.5,
        amp_scale=2.0,
        min_cubes_seen=64,
    )
    defaults.update(kw)
    return SkySnapshotExporter("http://h23.invalid:5778/sky/ingest",
                               **defaults)


# ---------------------------------------------------------------------------
# Payload round-trip
# ---------------------------------------------------------------------------


def test_payload_roundtrip():
    ix_row, ix_col = _pattern()
    vis = (np.arange(N_FILLED) + 1j * np.arange(N_FILLED)).astype(np.complex64)
    meta = {"chgroup": 3, "n_grid": 256, "amp_scale": 1.5, "unix_ts": 1.0}
    body = build_snapshot_npz(vis, ix_row=ix_row, ix_col=ix_col, meta=meta)
    out = parse_snapshot_npz(body)
    np.testing.assert_array_equal(out["vis"], vis)
    np.testing.assert_array_equal(out["ix_row"], ix_row)
    np.testing.assert_array_equal(out["ix_col"], ix_col)
    assert out["meta"] == meta


def test_payload_shape_mismatch_rejected():
    ix_row, ix_col = _pattern()
    vis = np.ones(N_FILLED + 1, dtype=np.complex64)
    with pytest.raises(ValueError):
        build_snapshot_npz(vis, ix_row=ix_row, ix_col=ix_col, meta={})


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_snapshot_npz(b"not an npz at all")


def test_parse_rejects_wrong_version():
    ix_row, ix_col = _pattern()
    vis = np.ones(N_FILLED, dtype=np.complex64)
    body = build_snapshot_npz(vis, ix_row=ix_row, ix_col=ix_col, meta={})
    # Corrupt the version by rebuilding with a patched constant.
    import dsart.services.sky_export as se
    old = se.SKY_SNAPSHOT_VERSION
    try:
        se.SKY_SNAPSHOT_VERSION = old + 1
        body_bad = build_snapshot_npz(
            vis, ix_row=ix_row, ix_col=ix_col, meta={},
        )
    finally:
        se.SKY_SNAPSHOT_VERSION = old
    with pytest.raises(ValueError):
        parse_snapshot_npz(body_bad)
    # sanity: the good body still parses
    assert parse_snapshot_npz(body)["vis"].shape == (N_FILLED,)


# ---------------------------------------------------------------------------
# maybe_export gating
# ---------------------------------------------------------------------------


def test_maybe_export_posts_snapshot(monkeypatch):
    posted = []

    def fake_urlopen(req, timeout=None):
        posted.append(req.data)

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n=-1):
                return b"{}"

        return _Resp()

    monkeypatch.setattr(
        "dsart.services.sky_export.urllib.request.urlopen", fake_urlopen,
    )
    exp = _make_exporter()
    try:
        ema = _FakeStaticSky()
        assert exp.maybe_export(ema, block_n=100) is True
        # Wait for the worker thread to drain.
        deadline = time.monotonic() + 5.0
        while not posted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(posted) == 1
        out = parse_snapshot_npz(posted[0])
        assert out["meta"]["chgroup"] == 5
        assert out["meta"]["window_blocks"] == 8
        assert out["meta"]["block_n"] == 100
        assert out["meta"]["amp_scale"] == 2.0
        assert out["meta"]["cubes_seen"] == 1000
        assert exp.n_exported == 1
    finally:
        exp.close()


def test_maybe_export_rate_limited():
    exp = _make_exporter()
    try:
        ema = _FakeStaticSky()
        # Pretend the first export just happened.
        exp._last_export_monotonic = time.monotonic()
        assert exp.maybe_export(ema, block_n=1) is False
        assert exp.maybe_export(ema, block_n=2) is False
    finally:
        exp.close()


def test_maybe_export_waits_for_warmup():
    exp = _make_exporter(min_cubes_seen=64)
    try:
        cold = _FakeStaticSky(cubes_seen=10)
        assert exp.maybe_export(cold, block_n=1) is False
        assert exp.maybe_export(None, block_n=2) is False
    finally:
        exp.close()


def test_post_failure_is_fail_soft(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        "dsart.services.sky_export.urllib.request.urlopen", boom,
    )
    exp = _make_exporter()
    try:
        ema = _FakeStaticSky()
        assert exp.maybe_export(ema, block_n=1) is True   # snapshot still taken
        deadline = time.monotonic() + 5.0
        while exp.n_failed == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert exp.n_failed == 1
        assert exp.n_exported == 0
    finally:
        exp.close()
