"""Unit tests for ``dsart.services.corr_fast_mon``.

Covers the M7.4 Phase 6c corr_fast service-start mon-key publisher
that fixes Bug 1 from ``docs/M7.4_PHASE6_E2E_REPORT.md`` (the Control
tab's "Send injection" form couldn't derive a usable
``apply_at_specnum`` because the SNAP-header and service-start epochs
differ by 3 orders of magnitude).

The publisher's contract:

* ``build_corr_fast_mon_key(chgroup)`` returns
  ``/mon/corr_rt/<chgroup>/corr_fast``.
* :meth:`CorrFastMonPublisher.publish` writes a dict containing at
  least ``block_n``, ``block_specnum_start``, ``ts_wall_unix``,
  ``npackets_per_block`` to that key — the dashboard's
  ``compute_inject_apply_at`` requires those four.
* etcd errors are logged once and swallowed; the hot loop never
  raises into corr_fast.
* Lazy import of ``dsautils`` so unit tests don't need it.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, os.pardir, "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from dsart.services.corr_fast_mon import (             # noqa: E402
    NPACKETS_PER_BLOCK_DEFAULT,
    CorrFastMonPublisher,
    build_corr_fast_mon_key,
)


class FakeStore:
    """Mimics ``DsaStore.put_dict`` for the publisher tests."""

    def __init__(self, *, raise_first: bool = False):
        self.puts: list[tuple[str, dict[str, Any]]] = []
        self._raise_first = bool(raise_first)
        self._n_puts = 0

    def put_dict(self, key: str, val: dict[str, Any]) -> None:
        self._n_puts += 1
        if self._raise_first and self._n_puts == 1:
            raise RuntimeError("simulated etcd hiccup")
        self.puts.append((key, dict(val)))


# -------------------------------------------------------------------
# build_corr_fast_mon_key
# -------------------------------------------------------------------


def test_build_corr_fast_mon_key_zero():
    assert build_corr_fast_mon_key(0) == "/mon/corr_rt/0/corr_fast"


def test_build_corr_fast_mon_key_fifteen():
    assert build_corr_fast_mon_key(15) == "/mon/corr_rt/15/corr_fast"


def test_build_corr_fast_mon_key_coerces_to_int():
    """A stringly-typed chgroup must still produce the canonical key."""
    assert (
        build_corr_fast_mon_key("3")           # type: ignore[arg-type]
        == "/mon/corr_rt/3/corr_fast"
    )


# -------------------------------------------------------------------
# CorrFastMonPublisher.publish — happy path
# -------------------------------------------------------------------


def test_publish_writes_block_specnum_start():
    fake = FakeStore()
    pub = CorrFastMonPublisher(chgroup=0, store=fake)
    ok = pub.publish(
        block_n=1234, n_processed=1230, n_drop=0, n_tx=1225,
        last_block_ms=125.4,
    )
    assert ok is True
    assert pub.n_published == 1
    assert pub.n_errors == 0
    [(key, payload)] = fake.puts
    assert key == "/mon/corr_rt/0/corr_fast"
    assert payload["block_n"] == 1234
    assert payload["block_specnum_start"] == 1234 * NPACKETS_PER_BLOCK_DEFAULT
    assert payload["npackets_per_block"] == NPACKETS_PER_BLOCK_DEFAULT
    assert payload["n_processed"] == 1230
    assert payload["n_drop"] == 0
    assert payload["n_tx"] == 1225
    assert payload["last_block_ms"] == pytest.approx(125.4)
    # wall + mono timestamps present and reasonable.
    assert isinstance(payload["ts_wall_unix"], float)
    assert isinstance(payload["ts_mono"], float)
    assert payload["ts_wall_unix"] > 1_700_000_000   # > 2023
    assert payload["ts_mono"] > 0


def test_publish_custom_npackets_per_block():
    """The constant is repeated locally so it must be honoured if a
    caller passes a different value (e.g. for the M7.2.8 corner-turn
    work that may change the packet packing)."""
    fake = FakeStore()
    pub = CorrFastMonPublisher(
        chgroup=3, npackets_per_block=4096, store=fake,
    )
    pub.publish(block_n=10)
    [(_, payload)] = fake.puts
    assert payload["block_specnum_start"] == 10 * 4096
    assert payload["npackets_per_block"] == 4096


def test_publish_multiple_increments_n_published():
    fake = FakeStore()
    pub = CorrFastMonPublisher(chgroup=0, store=fake)
    for n in (16, 32, 48, 64):
        pub.publish(block_n=n)
    assert pub.n_published == 4
    assert pub.n_errors == 0
    assert [p["block_n"] for _, p in fake.puts] == [16, 32, 48, 64]


def test_publish_without_last_block_ms_omits_field():
    fake = FakeStore()
    pub = CorrFastMonPublisher(chgroup=0, store=fake)
    pub.publish(block_n=10)
    [(_, payload)] = fake.puts
    assert "last_block_ms" not in payload


def test_publish_extra_merges_in():
    fake = FakeStore()
    pub = CorrFastMonPublisher(chgroup=0, store=fake)
    pub.publish(block_n=10, extra={"warmup_done": True, "label": "v1"})
    [(_, payload)] = fake.puts
    assert payload["warmup_done"] is True
    assert payload["label"] == "v1"


# -------------------------------------------------------------------
# CorrFastMonPublisher.publish — error handling
# -------------------------------------------------------------------


def test_publish_swallows_etcd_error(caplog):
    fake = FakeStore(raise_first=True)
    pub = CorrFastMonPublisher(chgroup=0, store=fake)
    with caplog.at_level(logging.WARNING):
        ok = pub.publish(block_n=10)
    assert ok is False
    assert pub.n_published == 0
    assert pub.n_errors == 1
    # The first error gets a WARNING line.
    msgs = " ".join(r.message for r in caplog.records)
    assert "first put_dict" in msgs or "put_dict" in msgs.lower()


def test_publish_second_error_silent(caplog):
    """After the first warning, subsequent errors are silent — we
    just bump n_errors. The dashboard surfaces the count separately
    so a stuck publisher is still visible."""

    class AlwaysRaiseStore:
        def put_dict(self, key, val):
            raise RuntimeError("etcd dead")

    pub = CorrFastMonPublisher(chgroup=0, store=AlwaysRaiseStore())
    with caplog.at_level(logging.WARNING):
        pub.publish(block_n=10)
        pre = len([r for r in caplog.records if r.levelno >= logging.WARNING])
        pub.publish(block_n=20)
        pub.publish(block_n=30)
        post = len([r for r in caplog.records if r.levelno >= logging.WARNING])
    # First call logs once; subsequent calls do NOT add new warnings.
    assert post == pre
    assert pub.n_errors == 3
    assert pub.n_published == 0


def test_publish_lazy_dsautils_unavailable(caplog, monkeypatch):
    """If dsautils isn't importable (the common dev-host case),
    publish must return False, log once, and never raise."""

    # Block the dsautils import even when it's installed.
    monkeypatch.setitem(sys.modules, "dsautils", None)

    pub = CorrFastMonPublisher(chgroup=0, store=None)
    with caplog.at_level(logging.WARNING):
        ok = pub.publish(block_n=10)
    assert ok is False
    assert pub.n_errors == 1
    msgs = " ".join(r.message for r in caplog.records)
    assert "dsautils" in msgs


# -------------------------------------------------------------------
# Thread-safety pin: concurrent publishes share the lock
# -------------------------------------------------------------------


def test_concurrent_publish_thread_safe():
    """Two threads hitting publish() simultaneously must not corrupt
    the shared counters / store-handle. We just sanity-check that
    both calls land puts and the total count matches."""

    import threading

    fake = FakeStore()
    pub = CorrFastMonPublisher(chgroup=0, store=fake)

    barrier = threading.Barrier(8)
    n_per_thread = 50

    def worker(start: int):
        barrier.wait()
        for i in range(n_per_thread):
            pub.publish(block_n=start * n_per_thread + i)

    threads = [
        threading.Thread(target=worker, args=(t,)) for t in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pub.n_published == 8 * n_per_thread
    assert len(fake.puts) == 8 * n_per_thread
