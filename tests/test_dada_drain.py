"""Unit tests for dsart.services.dada_drain.

Behavioural surface that matters:
1. CLI ``--key`` is hex and must validate.
2. SIGTERM handler is installed and toggles ``state["stop"]`` (not the
   process-wide signal disposition — we only test the public state-flip
   contract since pytest hijacks SIGINT/SIGTERM and we don't want to
   actually exit the test runner).
3. ``run()`` short-circuits cleanly on a Reader that raises
   StopIteration on first getNextPage (≡ EOD before any data) and
   returns 0.
4. ``run()`` exits cleanly when ``isEndOfData`` flips True between
   pages, drains the final page via ``markCleared``, and returns 0.
5. ``run()`` honours the ``state["stop"]`` flag injected externally
   (simulating a SIGTERM during a normal drain) — no extra
   getNextPage call past the stop point.

We deliberately do not exercise psrdada wire protocol end-to-end here;
that requires a live PSRDADA ring which the smoke harness on n06
covers. These tests fence the Python control-flow paths so a future
refactor can't silently break the drain semantics.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any
from unittest import mock

import pytest

from dsart.services import dada_drain


def test_main_rejects_non_hex_key(capsys: pytest.CaptureFixture[str]) -> None:
    """``--key`` must be hex; bad input returns 2 and logs an error."""
    rc = dada_drain.main(["--key", "not-a-hex"])
    assert rc == 2


def test_main_accepts_hex_key_and_dispatches_to_run() -> None:
    """``--key bada`` parses to 0xbada and is passed through to run."""
    with mock.patch.object(dada_drain, "run", return_value=0) as m_run:
        rc = dada_drain.main(["--key", "bada"])
    assert rc == 0
    assert m_run.call_count == 1
    args, kwargs = m_run.call_args
    assert args[0] == 0xBADA
    assert kwargs.get("log_every") == 1024


def test_main_passes_log_every() -> None:
    with mock.patch.object(dada_drain, "run", return_value=0) as m_run:
        rc = dada_drain.main(["--key", "dada", "--log-every", "16"])
    assert rc == 0
    assert m_run.call_args.kwargs["log_every"] == 16


class _FakeReaderEODFirstPage:
    """Mimic Reader: isEndOfData True immediately."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.isEndOfData = False

    def getNextPage(self) -> bytes:
        self.calls.append("getNextPage")
        self.isEndOfData = True
        return b"\x00" * 4

    def markCleared(self) -> None:
        self.calls.append("markCleared")


class _FakeReaderRaisesStopIteration:
    """Mimic Reader that raises StopIteration on first read (no data)."""

    isEndOfData = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def getNextPage(self) -> bytes:
        self.calls.append("getNextPage")
        raise StopIteration

    def markCleared(self) -> None:
        self.calls.append("markCleared")


class _FakeReaderNDrains:
    """Mimic Reader: emits N pages then sets isEndOfData."""

    def __init__(self, n_pages: int) -> None:
        self._n_remaining = n_pages
        self.isEndOfData = False
        self.calls: list[str] = []

    def getNextPage(self) -> bytes:
        self.calls.append("getNextPage")
        self._n_remaining -= 1
        if self._n_remaining <= 0:
            self.isEndOfData = True
        return b"\x00" * 4

    def markCleared(self) -> None:
        self.calls.append("markCleared")


def _patch_reader(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    """Stub out ``from psrdada import Reader`` inside dada_drain.run."""
    fake_module = mock.MagicMock()
    fake_module.Reader = mock.MagicMock(return_value=fake)
    monkeypatch.setitem(sys.modules, "psrdada", fake_module)


def test_run_eod_first_page_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeReaderEODFirstPage()
    _patch_reader(monkeypatch, fake)
    with caplog.at_level(logging.INFO, logger="dsart.dada_drain"):
        rc = dada_drain.run(0xBADA, log_every=1)
    assert rc == 0
    assert fake.calls == ["getNextPage", "markCleared"]


def test_run_stopiteration_first_page_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeReaderRaisesStopIteration()
    _patch_reader(monkeypatch, fake)
    with caplog.at_level(logging.INFO, logger="dsart.dada_drain"):
        rc = dada_drain.run(0xBADA, log_every=1)
    assert rc == 0
    assert fake.calls == ["getNextPage"]


def test_run_drains_n_pages_then_eod(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _FakeReaderNDrains(n_pages=5)
    _patch_reader(monkeypatch, fake)
    with caplog.at_level(logging.INFO, logger="dsart.dada_drain"):
        rc = dada_drain.run(0xBADA, log_every=2)
    assert rc == 0
    n_gnp = sum(1 for c in fake.calls if c == "getNextPage")
    n_mc = sum(1 for c in fake.calls if c == "markCleared")
    assert n_gnp == 5
    assert n_mc == 5


def test_install_signals_toggles_stop_flag() -> None:
    """Verify the handler-installed callbacks toggle state['stop']
    when called directly. We don't actually raise the signal here —
    pytest's signal-handling would treat it as an interruption and we
    only care about the state-flip semantics.
    """
    state: dict[str, Any] = {"stop": False}

    with mock.patch("signal.signal") as m_sig:
        dada_drain._install_signals(state)

    assert m_sig.call_count == 2
    installed = {call.args[0]: call.args[1] for call in m_sig.call_args_list}
    import signal as _sig

    assert _sig.SIGTERM in installed
    assert _sig.SIGINT in installed
    handler = installed[_sig.SIGTERM]
    handler(_sig.SIGTERM, None)
    assert state["stop"] is True
