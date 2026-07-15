"""Corr-node voltage retention + dump service.

One long-running service per corr node (spawned by ``dsart_rt`` like the
other corr routines). It owns the operator-chosen "dedicated isolated
retention ring fed by a 3rd ``fada`` reader" so a slow voltage dump can
never back-pressure capture:

  * **reader thread** — attaches ``fada`` as the 3rd PSRDADA reader, and
    on every block does nothing but ``memcpy`` it into a RAM
    :class:`~dsart.dump.voltage_ring.VoltageRing` slot and
    ``markCleared()`` the page. ~30–50 ms / 134 ms block.
  * **dump worker thread** — consumes dump/delete requests, extracts the
    ~3 s window out of the ring, and streams it to local NVMe staging
    (``<event>_sb<NN>_data.out`` raw + ``.json`` manifest). All slow I/O
    lives here, downstream of the ring.
  * **UDP listener** (asyncio) — decodes C2 voltage triggers and enqueues
    requests (see :mod:`dsart.dump.voltage_trigger_listener`).

C3 (h23) later pulls the staged ``.out`` files (KEEP) or sends a delete
sentinel (REJECT). See ``docs/voltage_dumps/VOLTAGE_DUMP_C3_DESIGN.md``.

Block-number alignment: the reader counts ``block_n`` starting at 1 for
its first ``fada`` page, matching ``corr_fast_integration`` (whose first
page is ``block_n = 1`` ⇒ ``block_specnum_start = block_n * 2048``). Both
readers attach before the operator arms capture, so they see the same
first armed block and a C2 ``event_specnum`` maps to the same ``block_n``
on every node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..common.constants import FADA_BYTES_PER_BLOCK
from ..dump.voltage_ring import VoltageRing, specnum_to_block_n
from ..dump.voltage_trigger_listener import (
    VoltageTriggerListener,
    VoltageTriggerListenerConfig,
)

LOG = logging.getLogger("dsart.services.voltage_retention")

__all__ = [
    "RetentionConfig",
    "staged_paths",
    "write_window_to_staging",
    "VoltageRetentionService",
    "main",
]

# Native specnum period (s): 1 specnum = 2 native samples = 65.536 µs.
_SPECNUM_S: float = 2 * 32.768e-6
#: Block cadence (s): 2048 specnums × 65.536 µs.
_BLOCK_S: float = 2048 * _SPECNUM_S


@dataclass(frozen=True)
class RetentionConfig:
    fada_key: int
    cn_id: int
    chgroup: int
    bind_host: str
    bind_port: int
    staging_dir: Path
    retention_blocks: int        # RAM ring depth (≈112 for 15 s)
    n_pre: int                   # blocks before the trigger block
    n_post: int                  # blocks after the trigger block
    dump_wait_s: float           # max wait for post-blocks to be captured
    queue_max: int = 32
    mon_key_fmt: str = "/mon/corr_rt/{cn}/voltage_retention"
    mon_interval_s: float = 5.0
    ready_sentinel_path: Optional[Path] = None
    max_blocks: int = 0          # 0 = run forever (tests cap it)

    @property
    def sb(self) -> str:
        return f"sb{self.chgroup:02d}"


# ---------------------------------------------------------------------------
# Staging I/O (pure-ish; unit-testable without PSRDADA)
# ---------------------------------------------------------------------------


def staged_paths(
    staging_dir: Path, event_name: str, chgroup: int,
) -> Tuple[Path, Path]:
    """Return ``(.out, .json)`` staging paths for an event on this node."""
    sb = f"sb{int(chgroup):02d}"
    base = f"{event_name}_{sb}"
    d = Path(staging_dir)
    return d / f"{base}_data.out", d / f"{base}.json"


def write_window_to_staging(
    *,
    ring: VoltageRing,
    event_name: str,
    event_specnum: int,
    cn_id: int,
    chgroup: int,
    staging_dir: Path,
    n_pre: int,
    n_post: int,
    mjd_target: float = 0.0,
) -> Dict[str, Any]:
    """Extract the dump window from ``ring`` and write it to staging.

    Writes the raw concatenated fada bytes (no header — legacy ``.out``
    convention) to ``<event>_<sb>_data.out`` and a sidecar manifest. The
    ``.out`` is written to a ``.tmp`` then atomically renamed so C3 never
    sees a partial file. Returns the manifest dict (also written as JSON).
    """
    target_block = specnum_to_block_n(int(event_specnum))
    extract = ring.extract_window(target_block, n_pre=n_pre, n_post=n_post)
    out_path, json_path = staged_paths(staging_dir, event_name, chgroup)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_written = 0
    with open(tmp_path, "wb") as fh:
        for _b, arr in extract.blocks:
            fh.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
            n_written += 1
    os.replace(tmp_path, out_path)

    manifest: Dict[str, Any] = {
        "event_name": event_name,
        "cn_id": int(cn_id),
        "chgroup": int(chgroup),
        "subband": f"sb{int(chgroup):02d}",
        "event_specnum": int(event_specnum),
        "target_block_n": int(target_block),
        "n_pre": int(n_pre),
        "n_post": int(n_post),
        "block_n_first": extract.first_block_n,
        "block_n_last": extract.last_block_n,
        "n_blocks_written": n_written,
        "n_blocks_dropped": extract.n_dropped,
        "dropped_block_ns": list(extract.dropped),
        "bytes_per_block": int(ring.bytes_per_block),
        "total_bytes": int(n_written * ring.bytes_per_block),
        "mjd_target": float(mjd_target),
        "ring_newest_block_n": int(ring.newest_block_n),
        "ring_oldest_block_n": int(ring.oldest_block_n),
        "written_at_unix": time.time(),
        "data_path": str(out_path),
    }
    with open(json_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest


def delete_staged(
    staging_dir: Path, event_name: str, chgroup: int,
) -> int:
    """Remove a staged ``.out``+``.json`` pair; return count removed."""
    out_path, json_path = staged_paths(staging_dir, event_name, chgroup)
    removed = 0
    for p in (out_path, json_path):
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except OSError as exc:
            LOG.warning("delete_staged: could not remove %s: %s", p, exc)
    return removed


# ---------------------------------------------------------------------------
# etcd mon wrapper (mockable, same idiom as the other services)
# ---------------------------------------------------------------------------


class _StoreWrapper:
    def __init__(self, mock: Optional[Any] = None) -> None:
        if mock is not None:
            self._store = mock
            self._available = True
            return
        try:
            from dsautils.dsa_store import DsaStore  # noqa: WPS433
            self._store = DsaStore()
            self._available = True
        except Exception as exc:  # noqa: BLE001
            LOG.warning("DsaStore unavailable (%s); mon export disabled", exc)
            self._store = None
            self._available = False

    def put_dict(self, key: str, value: Dict[str, Any]) -> None:
        if not self._available or self._store is None:
            return
        try:
            self._store.put_dict(key, dict(value))
        except Exception:  # noqa: BLE001
            LOG.exception("etcd put_dict(%s) failed", key)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class VoltageRetentionService:
    """Owns the ring + reader thread + dump worker + UDP listener."""

    def __init__(
        self,
        config: RetentionConfig,
        *,
        mon_store: Optional[Any] = None,
        ring: Optional[VoltageRing] = None,
    ) -> None:
        self._cfg = config
        self._ring = ring or VoltageRing(
            n_blocks=config.retention_blocks,
            bytes_per_block=FADA_BYTES_PER_BLOCK,
        )
        self._mon_store = _StoreWrapper(mock=mon_store)
        self._q: "queue.Queue[Tuple[str, str, int, float]]" = queue.Queue(
            maxsize=config.queue_max,
        )
        self._stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._listener: Optional[VoltageTriggerListener] = None
        self._counters: Dict[str, int] = {
            "blocks_read": 0,
            "dumps_done": 0,
            "dumps_failed": 0,
            "deletes_done": 0,
            "blocks_dropped_total": 0,
        }

    # ----- listener handlers (run on the asyncio loop) -----------------

    def _enqueue_dump(
        self, event_name: str, event_specnum: int, mjd_target: float = 0.0,
    ) -> bool:
        try:
            self._q.put_nowait(
                ("dump", event_name, int(event_specnum), float(mjd_target))
            )
            return True
        except queue.Full:
            return False

    def _enqueue_delete(self, event_name: str) -> bool:
        try:
            self._q.put_nowait(("delete", event_name, 0, 0.0))
            return True
        except queue.Full:
            return False

    # ----- reader thread (sole ring writer) ----------------------------

    def _reader_loop(self) -> None:
        try:
            from psrdada import Reader  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            LOG.error("psrdada import failed; reader disabled: %s", exc)
            return
        reader = Reader(self._cfg.fada_key)
        try:
            hdr = reader.getHeader()
            LOG.info(
                "voltage_retention: fada attached (%d hdr keys, UTC_START=%s)",
                len(hdr), hdr.get("UTC_START", "?"),
            )
            block_n = 0
            while not self._stop.is_set():
                try:
                    page = reader.getNextPage()
                except StopIteration:
                    LOG.info("voltage_retention: fada EOD")
                    break
                block_n += 1
                arr = np.asarray(page)
                if arr.nbytes != FADA_BYTES_PER_BLOCK:
                    LOG.error(
                        "voltage_retention: block #%d wrong size %d (want %d)",
                        block_n, arr.nbytes, FADA_BYTES_PER_BLOCK,
                    )
                    reader.markCleared()
                    continue
                self._ring.store(block_n, arr.reshape(-1))
                reader.markCleared()
                self._counters["blocks_read"] += 1
                if (
                    self._cfg.max_blocks
                    and self._counters["blocks_read"] >= self._cfg.max_blocks
                ):
                    break
        finally:
            try:
                reader.disconnect()
            except Exception:  # noqa: BLE001
                pass

    # ----- dump worker thread ------------------------------------------

    def _wait_for_post(self, target_block: int) -> None:
        """Block (≤ dump_wait_s) until the ring has captured the last
        post-window block, so a trigger that arrives before the burst's
        dispersion sweep finished still gets the full window."""
        need = target_block + self._cfg.n_post
        deadline = time.monotonic() + self._cfg.dump_wait_s
        while (
            not self._stop.is_set()
            and self._ring.newest_block_n < need
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                kind, name, specnum, mjd = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "delete":
                    n = delete_staged(
                        self._cfg.staging_dir, name, self._cfg.chgroup,
                    )
                    self._counters["deletes_done"] += 1
                    LOG.info(
                        "voltage_retention: deleted staged %s (%d files)",
                        name, n,
                    )
                    continue
                target = specnum_to_block_n(specnum)
                self._wait_for_post(target)
                manifest = write_window_to_staging(
                    ring=self._ring,
                    event_name=name,
                    event_specnum=specnum,
                    cn_id=self._cfg.cn_id,
                    chgroup=self._cfg.chgroup,
                    staging_dir=self._cfg.staging_dir,
                    n_pre=self._cfg.n_pre,
                    n_post=self._cfg.n_post,
                    mjd_target=mjd,
                )
                self._counters["blocks_dropped_total"] += int(
                    manifest["n_blocks_dropped"]
                )
                if int(manifest["n_blocks_written"]) == 0:
                    # Every requested block missed the ring: the staged
                    # file is empty and the dump is a FAILURE (e.g. a
                    # trigger specnum in the wrong units/epoch), not a
                    # success. Manifest + empty file are kept for
                    # forensics.
                    self._counters["dumps_failed"] += 1
                    LOG.error(
                        "voltage_retention: dump %s %s wrote 0 blocks "
                        "(all %d dropped; target=%d ring=%s..%s) — FAILED",
                        name, self._cfg.sb,
                        manifest["n_blocks_dropped"], target,
                        manifest["ring_oldest_block_n"],
                        manifest["ring_newest_block_n"],
                    )
                    continue
                self._counters["dumps_done"] += 1
                LOG.info(
                    "voltage_retention: staged %s %s — %d blocks (%d dropped) "
                    "%.2f GiB",
                    name, self._cfg.sb,
                    manifest["n_blocks_written"], manifest["n_blocks_dropped"],
                    manifest["total_bytes"] / (1 << 30),
                )
            except Exception:  # noqa: BLE001
                self._counters["dumps_failed"] += 1
                LOG.exception(
                    "voltage_retention: dump failed for %s specnum=%d",
                    name, specnum,
                )

    # ----- mon publish (asyncio) ---------------------------------------

    async def _mon_loop(self) -> None:
        key = self._cfg.mon_key_fmt.format(cn=self._cfg.cn_id)
        try:
            while not self._stop.is_set():
                payload: Dict[str, Any] = {
                    "ts_unix": time.time(),
                    "cn_id": self._cfg.cn_id,
                    "chgroup": self._cfg.chgroup,
                    "counters": dict(self._counters),
                    "ring": self._ring.mon(),
                    "queue_depth": self._q.qsize(),
                    "retention_s": self._cfg.retention_blocks * _BLOCK_S,
                    "window_s": (
                        (self._cfg.n_pre + self._cfg.n_post + 1) * _BLOCK_S
                    ),
                }
                if self._listener is not None:
                    payload["listener"] = self._listener.mon
                self._mon_store.put_dict(key, payload)
                await asyncio.sleep(self._cfg.mon_interval_s)
        except asyncio.CancelledError:
            return

    # ----- lifecycle ----------------------------------------------------

    async def run(self) -> int:
        loop = asyncio.get_running_loop()

        def _term() -> None:
            LOG.info("voltage_retention: SIGTERM/SIGINT")
            self._stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _term)
            except (NotImplementedError, RuntimeError):
                pass

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="vret-reader", daemon=True,
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="vret-worker", daemon=True,
        )
        self._reader_thread.start()
        self._worker_thread.start()

        self._listener = VoltageTriggerListener(
            config=VoltageTriggerListenerConfig(
                bind_host=self._cfg.bind_host,
                bind_port=self._cfg.bind_port,
                cn_id=self._cfg.cn_id,
                chgroup=self._cfg.chgroup,
            ),
            on_dump=self._enqueue_dump,
            on_delete=self._enqueue_delete,
        )
        await self._listener.start()

        if self._cfg.ready_sentinel_path is not None:
            try:
                p = Path(self._cfg.ready_sentinel_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            except OSError as exc:
                LOG.warning("ready sentinel touch failed: %s", exc)

        mon_task = asyncio.create_task(self._mon_loop(), name="vret-mon")
        LOG.info(
            "voltage_retention up: cn=%d %s ring=%d blocks (%.1f s) "
            "window=%d blocks (%.1f s) staging=%s bind=%s:%d",
            self._cfg.cn_id, self._cfg.sb, self._cfg.retention_blocks,
            self._cfg.retention_blocks * _BLOCK_S,
            self._cfg.n_pre + self._cfg.n_post + 1,
            (self._cfg.n_pre + self._cfg.n_post + 1) * _BLOCK_S,
            self._cfg.staging_dir, self._cfg.bind_host, self._cfg.bind_port,
        )
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.5)
        finally:
            mon_task.cancel()
            if self._listener is not None:
                await self._listener.stop()
            self._stop.set()
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _blocks_for_seconds(seconds: float) -> int:
    return int(math.ceil(float(seconds) / _BLOCK_S))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fada-key", default="fada",
                   help="fada PSRDADA key (hex like 0xfada or name 'fada').")
    p.add_argument("--cn-id", type=int, required=True)
    p.add_argument("--chgroup", type=int, required=True)
    p.add_argument("--bind-host", default="0.0.0.0",
                   help="corr-net IPv4 to bind the voltage trigger listener.")
    p.add_argument("--bind-port", type=int, default=11229)
    p.add_argument("--staging-dir",
                   default="/home/ubuntu/data/voltage_staging")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--retention-s", type=float, default=25.0)
    grp.add_argument("--retention-blocks", type=int, default=0,
                     help="Overrides --retention-s when > 0.")
    # The C1/C2 event label is the pulse arrival at the BOTTOM of the
    # band, so the dispersed burst lies BEFORE the target block —
    # n_pre must cover the full-band sweep at the deployed fine-DM max
    # (docs/voltage_dumps/VOLTAGE_DUMP_TIMING_FIX.md, 2026-07-15).
    p.add_argument("--n-pre", type=int, default=14,
                   help="Blocks before the trigger block (~1.88 s @ 14; "
                        "must cover the full-band dispersion sweep — the "
                        "event label is the arrival at the band BOTTOM).")
    p.add_argument("--n-post", type=int, default=8,
                   help="Blocks after the trigger block (~1.07 s @ 8; "
                        "only pulse width + intra-chgroup smear land "
                        "after the label).")
    p.add_argument("--dump-wait-s", type=float, default=5.0)
    p.add_argument("--queue-max", type=int, default=32)
    p.add_argument("--mon-interval-s", type=float, default=5.0)
    p.add_argument("--ready-sentinel-path", default=None)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p


def _parse_fada_key(s: str) -> int:
    """Mirror ``corr_fast_integration._key_to_int``: a 4-char buffer name
    like ``fada`` is the hex spelling ``0x0000fada``."""
    s = str(s).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    if len(s) == 4:
        return int(f"0x{s}", 16)
    if s.isdigit():
        return int(s)
    raise SystemExit(f"--fada-key {s!r} not a 4-char buffer name or hex key")


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    retention_blocks = (
        args.retention_blocks if args.retention_blocks > 0
        else _blocks_for_seconds(args.retention_s)
    )
    cfg = RetentionConfig(
        fada_key=_parse_fada_key(args.fada_key),
        cn_id=int(args.cn_id),
        chgroup=int(args.chgroup),
        bind_host=str(args.bind_host),
        bind_port=int(args.bind_port),
        staging_dir=Path(args.staging_dir),
        retention_blocks=int(retention_blocks),
        n_pre=int(args.n_pre),
        n_post=int(args.n_post),
        dump_wait_s=float(args.dump_wait_s),
        queue_max=int(args.queue_max),
        mon_interval_s=float(args.mon_interval_s),
        ready_sentinel_path=(
            Path(args.ready_sentinel_path)
            if args.ready_sentinel_path else None
        ),
        max_blocks=int(args.max_blocks),
    )
    svc = VoltageRetentionService(cfg)
    return asyncio.run(svc.run())


if __name__ == "__main__":
    sys.exit(main())
