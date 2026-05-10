"""Tests for ``dsart.dump.udp_listener`` (M6 chunk 4 / D9 / D12).

Covers the asyncio UDP listener that arms a one-shot "dump next cube"
flag on receipt of any datagram. Lifecycle, payload-agnosticism, the
one-shot semantics, counters, port-in-use error path, and reusability
are all gated here.

Senders are plain blocking ``socket.SOCK_DGRAM`` calls (cheap, doesn't
need an asyncio sender). Datagrams to ``127.0.0.1`` are essentially
in-kernel: the ``_wait_for_datagrams`` helper polls
``n_datagrams_received`` with a short timeout to bridge the asyncio
recv-callback delivery latency.
"""

from __future__ import annotations

import asyncio
import functools
import os
import socket
from typing import Optional

import pytest

os.environ.setdefault("DSART_TEST", "1")

from dsart.dump.udp_listener import (  # noqa: E402
    UdpTriggerListener,
    UdpTriggerListenerConfig,
)


# ---------------------------------------------------------------------------
# Custom asyncio-test decorator (no pytest-asyncio).
# ---------------------------------------------------------------------------


def asyncio_test(func):
    """Run an async coroutine inside a fresh event loop per test."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send_udp(payload: bytes, host: str, port: int) -> None:
    """Send a single UDP datagram via a synchronous socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, (host, port))
    finally:
        sock.close()


async def _wait_for_datagrams(
    listener: UdpTriggerListener,
    expected: int,
    timeout: float = 2.0,
) -> None:
    """Poll until ``n_datagrams_received`` reaches ``expected``.

    Raises ``asyncio.TimeoutError`` if the count doesn't reach
    ``expected`` within ``timeout`` seconds. The poll interval is
    short enough that the test is cheap on success but won't busy-loop.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while listener.n_datagrams_received < expected:
        if loop.time() > deadline:
            raise asyncio.TimeoutError(
                f"datagram-count timeout: got "
                f"{listener.n_datagrams_received} of {expected}"
            )
        await asyncio.sleep(0.005)


async def _start_ephemeral(host: str = "127.0.0.1") -> UdpTriggerListener:
    listener = UdpTriggerListener(UdpTriggerListenerConfig(host=host, port=0))
    await listener.start()
    return listener


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


def test_config_defaults_match_d9() -> None:
    """D9: default port 11227, default bind 127.0.0.1."""
    cfg = UdpTriggerListenerConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11227


def test_config_is_frozen() -> None:
    cfg = UdpTriggerListenerConfig()
    with pytest.raises(Exception):
        cfg.port = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


@asyncio_test
async def test_happy_path_single_datagram_consumed_once() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        assert port > 0

        assert listener.consume_dump_next_cube_flag() is False
        assert listener.n_datagrams_received == 0
        assert listener.n_triggers_consumed == 0

        _send_udp(b"go", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)

        assert listener.consume_dump_next_cube_flag() is True
        assert listener.consume_dump_next_cube_flag() is False
        assert listener.consume_dump_next_cube_flag() is False

        assert listener.n_datagrams_received == 1
        assert listener.n_triggers_consumed == 1
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 2. Multiple datagrams between consumes => one True (one-shot)
# ---------------------------------------------------------------------------


@asyncio_test
async def test_multiple_datagrams_between_consumes_is_one_shot() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        for i in range(5):
            _send_udp(f"d{i}".encode(), "127.0.0.1", port)
        await _wait_for_datagrams(listener, 5)

        # Five datagrams in => exactly ONE True out (D9 one-shot lock).
        assert listener.consume_dump_next_cube_flag() is True
        assert listener.consume_dump_next_cube_flag() is False
        assert listener.consume_dump_next_cube_flag() is False

        assert listener.n_datagrams_received == 5
        assert listener.n_triggers_consumed == 1
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 3. Datagrams across multiple consume cycles
# ---------------------------------------------------------------------------


@asyncio_test
async def test_datagrams_across_consume_cycles() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port

        _send_udp(b"a", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)
        assert listener.consume_dump_next_cube_flag() is True

        _send_udp(b"b", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 2)
        assert listener.consume_dump_next_cube_flag() is True

        assert listener.consume_dump_next_cube_flag() is False

        assert listener.n_datagrams_received == 2
        assert listener.n_triggers_consumed == 2
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 4. Counters
# ---------------------------------------------------------------------------


@asyncio_test
async def test_counters_track_independently() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port

        # Three datagrams in, but only one consume cycle => True once.
        _send_udp(b"x", "127.0.0.1", port)
        _send_udp(b"y", "127.0.0.1", port)
        _send_udp(b"z", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 3)

        assert listener.consume_dump_next_cube_flag() is True
        # Two more consumes that return False must NOT bump n_triggers_consumed.
        assert listener.consume_dump_next_cube_flag() is False
        assert listener.consume_dump_next_cube_flag() is False
        assert listener.n_triggers_consumed == 1
        assert listener.n_datagrams_received == 3

        # Another datagram + consume.
        _send_udp(b"w", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 4)
        assert listener.consume_dump_next_cube_flag() is True
        assert listener.n_triggers_consumed == 2
        assert listener.n_datagrams_received == 4
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 5. Payload-agnostic
# ---------------------------------------------------------------------------


@asyncio_test
async def test_empty_payload_triggers_flag() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        _send_udp(b"", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)
        assert listener.consume_dump_next_cube_flag() is True
    finally:
        await listener.stop()


@asyncio_test
async def test_random_1k_payload_triggers_flag() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        payload = os.urandom(1024)
        _send_udp(payload, "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)
        assert listener.consume_dump_next_cube_flag() is True
        assert listener.n_datagrams_received == 1
    finally:
        await listener.stop()


@asyncio_test
async def test_literal_trigger_payload_triggers_flag() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        _send_udp(b"trigger", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)
        assert listener.consume_dump_next_cube_flag() is True
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 6. start() raises OSError if the port is already in use
# ---------------------------------------------------------------------------


@asyncio_test
async def test_start_raises_oserror_if_port_taken() -> None:
    # Park a UDP socket on an OS-chosen port, then try to start the
    # listener on the same port. The bind must fail.
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        blocker.bind(("127.0.0.1", 0))
        port = blocker.getsockname()[1]

        listener = UdpTriggerListener(
            UdpTriggerListenerConfig(host="127.0.0.1", port=port)
        )
        with pytest.raises(OSError):
            await listener.start()

        # Listener must not have ended up partially started.
        assert listener.is_running is False
        assert listener.bound_port == 0
    finally:
        blocker.close()


# ---------------------------------------------------------------------------
# 7. stop() prevents new datagrams from being counted
# ---------------------------------------------------------------------------


@asyncio_test
async def test_stop_prevents_new_datagrams_counted() -> None:
    listener = await _start_ephemeral()
    port = listener.bound_port

    _send_udp(b"pre", "127.0.0.1", port)
    await _wait_for_datagrams(listener, 1)

    await listener.stop()
    # After stop, the listener is no longer bound. Sending to the same
    # (now closed) port becomes either a no-op (no listener) or an ICMP
    # unreachable to the sender — but our listener cannot receive
    # anything since its socket is closed and its callbacks are detached.
    n_before = listener.n_datagrams_received
    for _ in range(10):
        # Use a try/except: depending on kernel timing the send may
        # raise (ECONNREFUSED) on the next sendto after the prior
        # ICMP-unreachable. That's not a test failure; we just need to
        # confirm the listener counter doesn't move.
        try:
            _send_udp(b"after", "127.0.0.1", port)
        except OSError:
            pass
    await asyncio.sleep(0.05)
    assert listener.n_datagrams_received == n_before
    assert listener.is_running is False


# ---------------------------------------------------------------------------
# 8. Concurrent senders
# ---------------------------------------------------------------------------


@asyncio_test
async def test_concurrent_senders_one_consume() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        n_senders = 50

        async def _sender(i: int) -> None:
            # sendto on a blocking socket from inside an asyncio task is
            # fine here: localhost UDP sends don't block in practice.
            _send_udp(f"task-{i}".encode(), "127.0.0.1", port)

        await asyncio.gather(*(_sender(i) for i in range(n_senders)))
        await _wait_for_datagrams(listener, n_senders, timeout=5.0)

        assert listener.n_datagrams_received == n_senders

        assert listener.consume_dump_next_cube_flag() is True
        assert listener.consume_dump_next_cube_flag() is False
        assert listener.n_triggers_consumed == 1
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 9. start() + stop() + start() — listener is reusable
# ---------------------------------------------------------------------------


@asyncio_test
async def test_start_stop_start_is_supported() -> None:
    listener = UdpTriggerListener(
        UdpTriggerListenerConfig(host="127.0.0.1", port=0)
    )
    await listener.start()
    port_first = listener.bound_port
    assert port_first > 0
    _send_udp(b"first", "127.0.0.1", port_first)
    await _wait_for_datagrams(listener, 1)
    assert listener.consume_dump_next_cube_flag() is True

    await listener.stop()
    assert listener.is_running is False

    # Restart on a fresh ephemeral port. Counters survive restart per
    # the spec ("monotonic, total"); the bound_port may be a different
    # ephemeral.
    await listener.start()
    port_second = listener.bound_port
    assert port_second > 0

    _send_udp(b"second", "127.0.0.1", port_second)
    await _wait_for_datagrams(listener, 2)
    assert listener.consume_dump_next_cube_flag() is True
    assert listener.n_datagrams_received == 2
    assert listener.n_triggers_consumed == 2
    await listener.stop()


@asyncio_test
async def test_double_start_raises() -> None:
    listener = await _start_ephemeral()
    try:
        with pytest.raises(RuntimeError):
            await listener.start()
    finally:
        await listener.stop()


@asyncio_test
async def test_double_stop_is_idempotent() -> None:
    listener = await _start_ephemeral()
    await listener.stop()
    # Second stop must not raise.
    await listener.stop()
    assert listener.is_running is False


# ---------------------------------------------------------------------------
# 10. bound_port introspection
# ---------------------------------------------------------------------------


@asyncio_test
async def test_bound_port_matches_send_target() -> None:
    listener = await _start_ephemeral()
    try:
        port = listener.bound_port
        assert port > 0

        _send_udp(b"x", "127.0.0.1", port)
        await _wait_for_datagrams(listener, 1)
        # If bound_port disagreed with the actual bind, the datagram
        # would never have arrived and _wait_for_datagrams would have
        # timed out.
        assert listener.n_datagrams_received == 1
    finally:
        await listener.stop()


def test_bound_port_zero_before_start() -> None:
    listener = UdpTriggerListener(
        UdpTriggerListenerConfig(host="127.0.0.1", port=12345)
    )
    assert listener.bound_port == 0
    assert listener.is_running is False


@asyncio_test
async def test_fixed_port_round_trip() -> None:
    # Pick a port deterministically by binding ephemeral then closing.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    chosen_port = probe.getsockname()[1]
    probe.close()

    listener = UdpTriggerListener(
        UdpTriggerListenerConfig(host="127.0.0.1", port=chosen_port)
    )
    # Race window: between probe.close() and create_datagram_endpoint
    # bind, another process could grab the port. Tolerate that.
    try:
        await listener.start()
    except OSError:  # pragma: no cover — race between probe + bind
        pytest.skip("ephemeral port reused mid-test; harmless race")
        return
    try:
        assert listener.bound_port == chosen_port
        _send_udp(b"hi", "127.0.0.1", chosen_port)
        await _wait_for_datagrams(listener, 1)
        assert listener.consume_dump_next_cube_flag() is True
    finally:
        await listener.stop()
