"""Tests for :mod:`dsart.coinc.names` (event-name allocator).

These tests use the offline fallback path; the h23 etcd + event.names
path is exercised at integration time, not here.
"""

from __future__ import annotations

import re

import pytest

from dsart.coinc.names import (
    DEFAULT_EVENT_PKG_PATH,
    EventNameAllocator,
    FallbackAllocator,
    _mjd_to_date_yymmdd,
)


def test_mjd_to_date_yymmdd_matches_known() -> None:
    # MJD 60781.0 = 2025-04-16 (UTC). Verify against datetime math directly.
    from datetime import datetime, timedelta
    expected = datetime(1970, 1, 1) + timedelta(days=60781 - 40587)
    assert _mjd_to_date_yymmdd(60781.0) == (
        f"{expected.year % 100:02d}{expected.month:02d}{expected.day:02d}"
    )
    # MJD 59580.0 = 2022-01-01 UTC
    assert _mjd_to_date_yymmdd(59580.0) == "220101"


def test_fallback_allocator_yields_uniquish_names() -> None:
    a = FallbackAllocator()
    n1 = a.allocate(60781.0)
    n2 = a.allocate(60781.0)
    assert n1 != n2  # the loop guarantees a different suffix from lastname
    assert re.fullmatch(r"\d{6}[a-z]{4}", n1)
    assert re.fullmatch(r"\d{6}[a-z]{4}", n2)
    # YYMMDD prefix consistent across allocations from the same MJD
    assert n1[:6] == n2[:6]
    assert n1[:6] == _mjd_to_date_yymmdd(60781.0)


def test_fallback_reset_clears_lastname() -> None:
    a = FallbackAllocator()
    a.allocate(60781.0)
    a.reset()
    assert a.lastname is None


def test_event_name_allocator_offline_uses_fallback() -> None:
    """offline=True must skip both event.names and etcd entirely."""
    alloc = EventNameAllocator(offline=True)
    n = alloc.allocate(60781.0)
    assert re.fullmatch(r"\d{6}[a-z]{4}", n)
    assert not alloc.is_online


def test_event_name_allocator_falls_back_when_event_pkg_missing(
    tmp_path,
) -> None:
    """Pointing at a non-existent event-pkg directory should fall back."""
    alloc = EventNameAllocator(event_pkg_path=tmp_path / "no_such_dir")
    # Even in non-offline mode, missing event.names import → fallback
    name = alloc.allocate(60781.0)
    assert re.fullmatch(r"\d{6}[a-z]{4}", name)


def test_event_name_allocator_default_event_pkg_path() -> None:
    """The default event-pkg path constant points at the h23 location."""
    assert str(DEFAULT_EVENT_PKG_PATH).endswith("dsa110-event")
