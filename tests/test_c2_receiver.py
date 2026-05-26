"""End-to-end tests for :mod:`dsart.coinc.receiver` (async TCP fan-in).

These tests use plain ``asyncio.run`` rather than pytest-asyncio so
they're portable across the dsa110-rt env (which doesn't ship
pytest-asyncio).
"""

from __future__ import annotations

import asyncio
import socket
from typing import List, Tuple

import pytest

from dsart.coinc import wire
from dsart.coinc.receiver import C1BatchReceiver


def _make_header(
    n: int, search_node_id: int = 1, gpu_half: int = 0, cube_id: int = 0,
) -> wire.C1BatchHeader:
    return wire.build_header(
        cube_id=cube_id,
        event_specnum_start=100_000,
        mjd_start=60781.123456789,
        sample_period_specnum=1,
        sample_period_us=1048.576,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=search_node_id,
        gpu_half=gpu_half,
        n_candidates=n,
    )


def _make_rows(n: int) -> list[wire.C1CandidateRow]:
    return [
        wire.C1CandidateRow(
            snr=10.0 + i,
            l_rad=1.2e-4 * (i + 1),
            m_rad=-1.4e-4 * (i + 1),
            l_pix=120 + i,
            m_pix=130 + i,
            dm_pc_cc=375.25 + i,
            dm_idx_global=50 + i,
            fine_dm_idx=i,
            event_specnum=100_000 + i,
            width_samples=1 << (i % 6),
            kernel_id="unit:d1:b4",
            flags=0,
        )
        for i in range(n)
    ]


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _await_counter(rx: C1BatchReceiver, name: str, target: int,
                         timeout_s: float = 2.0) -> None:
    end = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < end:
        if getattr(rx.counters, name) >= target:
            return
        await asyncio.sleep(0.01)


def test_receiver_accepts_and_parses_round_trip() -> None:
    async def main() -> None:
        received: List[Tuple[wire.C1Batch, str]] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append((batch, peer))

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            rows = _make_rows(3)
            header = _make_header(3, search_node_id=1, gpu_half=0, cube_id=7)
            blob = wire.C1BatchEncoder.encode(header, rows)
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(blob)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await _await_counter(rx, "batches_ok", 1)
            assert rx.counters.batches_ok == 1
            assert len(received) == 1
            assert received[0][0].header.cube_id == 7
            assert len(received[0][0].candidates) == 3
        finally:
            await rx.stop()

    asyncio.run(main())


def test_receiver_handles_back_to_back_batches() -> None:
    async def main() -> None:
        received: List[wire.C1Batch] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append(batch)

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            body = b"".join(
                wire.C1BatchEncoder.encode(
                    _make_header(2, cube_id=i), _make_rows(2),
                )
                for i in range(3)
            )
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(body)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await _await_counter(rx, "batches_ok", 3)
            assert rx.counters.batches_ok == 3
            assert [b.header.cube_id for b in received] == [0, 1, 2]
        finally:
            await rx.stop()

    asyncio.run(main())


def test_receiver_counts_torn_batch_on_eof_midstream() -> None:
    async def main() -> None:
        received: List[wire.C1Batch] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append(batch)

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            rows = _make_rows(3)
            header = _make_header(3)
            full = wire.C1BatchEncoder.encode(header, rows).decode()
            lines = full.split("\n")
            # Header + 1 row, no END.
            partial = "\n".join(lines[:2]) + "\n"
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(partial.encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await _await_counter(rx, "torn_batch", 1)
            assert rx.counters.batches_ok == 0
            assert rx.counters.torn_batch >= 1
            assert received == []
        finally:
            await rx.stop()

    asyncio.run(main())


def test_receiver_counts_bad_schema() -> None:
    async def main() -> None:
        received: List[wire.C1Batch] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append(batch)

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            body = (
                "# C1 9 0 0 60781.00000000000 16 1048.576000 256 34 1 0 0\n"
                "# END\n"
            ).encode()
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(body)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await _await_counter(rx, "bad_schema", 1)
            assert rx.counters.bad_schema >= 1
            assert received == []
        finally:
            await rx.stop()

    asyncio.run(main())


def test_receiver_recovers_from_new_header_mid_batch() -> None:
    """If a new '# C1 ' header arrives before the previous '# END',
    the prior batch is treated as torn and the new one parsed."""
    async def main() -> None:
        received: List[wire.C1Batch] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append(batch)

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            rows = _make_rows(2)
            header_t = _make_header(2, cube_id=42)
            full = wire.C1BatchEncoder.encode(header_t, rows).decode()
            lines = full.split("\n")
            truncated = "\n".join(lines[:2]) + "\n"  # header + 1 row only
            complete = wire.C1BatchEncoder.encode(
                _make_header(1, cube_id=43), _make_rows(1),
            )
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(truncated.encode() + complete)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            await _await_counter(rx, "batches_ok", 1)
            assert rx.counters.batches_ok == 1
            assert rx.counters.torn_batch >= 1
            assert received[0].header.cube_id == 43
        finally:
            await rx.stop()

    asyncio.run(main())


def test_receiver_supports_multiple_concurrent_connections() -> None:
    async def main() -> None:
        received: List[Tuple[int, int]] = []

        async def on_batch(batch: wire.C1Batch, peer: str) -> None:
            received.append(
                (batch.header.search_node_id, batch.header.gpu_half)
            )

        port = _pick_free_port()
        rx = C1BatchReceiver(host="127.0.0.1", port=port, on_batch=on_batch)
        await rx.start()
        try:
            async def client(sid: int, g: int) -> None:
                _reader, writer = await asyncio.open_connection(
                    "127.0.0.1", port,
                )
                blob = wire.C1BatchEncoder.encode(
                    _make_header(1, search_node_id=sid, gpu_half=g),
                    _make_rows(1),
                )
                writer.write(blob)
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            await asyncio.gather(
                client(1, 0), client(1, 1),
                client(2, 0), client(2, 1),
            )
            await _await_counter(rx, "batches_ok", 4)
            assert rx.counters.batches_ok == 4
            assert sorted(received) == [(1, 0), (1, 1), (2, 0), (2, 1)]
        finally:
            await rx.stop()

    asyncio.run(main())
