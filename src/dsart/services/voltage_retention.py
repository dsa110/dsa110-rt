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
import errno
import json
import logging
import math
import os
import queue
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.constants import BLOCK_DURATION_S, FADA_BYTES_PER_BLOCK
from ..dump.voltage_ring import VoltageRing, specnum_to_block_n
from ..dump.voltage_trigger_listener import (
    VoltageTriggerListener,
    VoltageTriggerListenerConfig,
)

LOG = logging.getLogger("dsart.services.voltage_retention")

__all__ = [
    "RetentionConfig",
    "DiskFullError",
    "staged_paths",
    "write_window_to_staging",
    "check_disk_headroom",
    "VoltageRetentionService",
    "main",
]

# Native specnum period (s): 1 specnum = 2 native samples = 65.536 µs.
_SPECNUM_S: float = 2 * 32.768e-6
#: Block cadence (s): 2048 specnums × 65.536 µs.
_BLOCK_S: float = 2048 * _SPECNUM_S

# 2026-07-21 ENOSPC fleet stall: a C2 voltage-dump broadcast storm
# (dozens of triggers 19:20-21:42 UT) staged ~6.47 GiB/event/corr node
# into --staging-dir with NO free-space check anywhere in this file,
# filling the root filesystem on 9/16 corr nodes. The full disk stalled
# the meridian UVH5 writer sharing the filesystem -> bada back-pressure
# -> corr_slow blocked -> fada deadlock -> corr_fast TX froze on 3 nodes
# -> below --fan-in-min-corrs -> all 8 search consumers stalled 25 min.
# These are the code-side safe defaults for the disk-headroom guard
# below; see configs/dsart_pipeline_rt.yaml for the deployed knobs.
_DEFAULT_MIN_FREE_BYTES_FLOOR: int = 16 * (1 << 30)          # 16 GiB
_DEFAULT_STAGING_MAX_TOTAL_BYTES: int = 120 * (1 << 30)      # 120 GiB

# 2026-07-24 orphan-eviction guard. The 2026-07-21 storm-containment cap
# above has NO eviction: C3 only ever deletes events it adjudicates
# (KEEP-collect cleanup or REJECT), so a C2 dump-trigger whose event C2
# then DISCARDS (cube-incomplete under an RFI storm — the common case)
# strands its ~6.5 GiB/node fragment in staging forever. On 2026-07-23
# 31 such orphans (~1.81 TiB fleet-wide) accumulated over one evening and
# pinned the cap, so voltage_retention skipped ~half of every dump for
# ~18 h (dumps_skipped_staging_cap) — voltage capability was effectively
# dead. Fix: a periodic sweep that evicts staged fragments older than
# _DEFAULT_STAGING_ORPHAN_TTL_S. The TTL MUST exceed the worst-case time
# a legitimate KEEP fragment lives in staging before C3 collects+deletes
# it (c3.collect_timeout_s, currently 1800 s) with generous margin — a
# fragment older than the TTL was provably never collected, so it is an
# orphan and safe to drop.
_DEFAULT_STAGING_ORPHAN_TTL_S: float = 7200.0                # 2 h (4x c3 collect_timeout)
_DEFAULT_ORPHAN_SWEEP_INTERVAL_S: float = 300.0             # scan every 5 min


class DiskFullError(OSError):
    """Raised when a staging write hits ENOSPC mid-write.

    Distinguishes the 2026-07-21 incident failure mode (disk filled
    while writing a fragment) from generic dump failures so the worker
    loop can count/log it separately and clean up the partial fragment,
    while still surviving to process the next queued event.
    """


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
    # 2026-07-21 disk-headroom guard (ENOSPC fleet stall). A dump is
    # refused when free space < max(2 * expected_fragment_bytes,
    # min_free_bytes_floor); expected_fragment_bytes is derived from
    # n_pre/n_post below (no need to derive from window params — the
    # fragment size IS the window size). min_free_bytes_floor is a
    # fixed backstop for configs where 2x the fragment happens to be
    # small.
    min_free_bytes_floor: int = _DEFAULT_MIN_FREE_BYTES_FLOOR
    # Cumulative cap on bytes resident in staging_dir. Bounds the storm
    # case (many back-to-back real triggers) even when the disk has
    # plenty of headroom otherwise.
    staging_max_total_bytes: int = _DEFAULT_STAGING_MAX_TOTAL_BYTES
    # 2026-07-24 orphan-eviction guard. A staged fragment older than
    # staging_orphan_ttl_s is evicted by the periodic sweeper (a C2-
    # discarded event C3 never adjudicates, so it is never cleaned
    # otherwise — it would pin the cap above forever). TTL must exceed
    # c3.collect_timeout_s with margin. 0 disables the sweep.
    staging_orphan_ttl_s: float = _DEFAULT_STAGING_ORPHAN_TTL_S
    orphan_sweep_interval_s: float = _DEFAULT_ORPHAN_SWEEP_INTERVAL_S

    @property
    def sb(self) -> str:
        return f"sb{self.chgroup:02d}"

    @property
    def expected_fragment_bytes(self) -> int:
        """Bytes written per dump: (n_pre + n_post + 1) blocks."""
        return (self.n_pre + self.n_post + 1) * FADA_BYTES_PER_BLOCK


# ---------------------------------------------------------------------------
# Staging I/O (pure-ish; unit-testable without PSRDADA)
# ---------------------------------------------------------------------------


def _utc_start_from_capture_mon() -> Optional[int]:
    """Best-effort ``utc_start_specnum`` from the capture mon shm.

    2026-07-19: the fada DADA header does not carry UTC_START in
    production, so every dump manifest shipped ``utc_start_specnum:
    null``. The capture processes (ports 4011/4012) publish the armed
    specnum in their mon shm; either instance's value works (both
    captures arm on the same verb). Provenance-only — never raises.
    """
    try:
        from ..capture.mon_shm import MonShm
    except Exception:                                             # noqa: BLE001
        return None
    for port in (4011, 4012):
        try:
            snap = MonShm.open(port).snapshot()
            val = int(snap.utc_start_specnum)
            if val > 0:
                return val
        except Exception:                                         # noqa: BLE001
            continue
    return None


def staged_paths(
    staging_dir: Path, event_name: str, chgroup: int,
) -> Tuple[Path, Path]:
    """Return ``(.out, .json)`` staging paths for an event on this node."""
    sb = f"sb{int(chgroup):02d}"
    base = f"{event_name}_{sb}"
    d = Path(staging_dir)
    return d / f"{base}_data.out", d / f"{base}.json"


def _staging_dir_bytes(staging_dir: Path) -> int:
    """Sum of file sizes currently in ``staging_dir``.

    Startup baseline for the in-process cumulative-bytes counter (2026-
    07-21 storm-containment guard) — cheap (one ``os.scandir`` pass, no
    recursion) and tolerant of the directory not existing yet.
    """
    total = 0
    try:
        with os.scandir(staging_dir) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except FileNotFoundError:
        return 0
    return total


def _staged_fragment_bytes(
    staging_dir: Path, event_name: str, chgroup: int,
) -> int:
    """Size of a previously-staged ``.out`` fragment, or 0 if absent."""
    out_path, _json_path = staged_paths(staging_dir, event_name, chgroup)
    try:
        return out_path.stat().st_size
    except OSError:
        return 0


def scan_staged_events(
    staging_dir: Path, chgroup: int,
) -> List[Tuple[str, float]]:
    """List this node's staged events as ``(event_name, out_mtime_unix)``.

    Derives the event name from each ``{event}_{sb}_data.out`` fragment
    for this chgroup's ``sb`` (the ``.out`` mtime is the dump-completion
    time — writes land atomically via a ``.tmp`` rename). Used by the
    orphan sweeper; tolerant of a missing dir. Never raises.
    """
    sb = f"sb{int(chgroup):02d}"
    suffix = f"_{sb}_data.out"
    out: List[Tuple[str, float]] = []
    try:
        with os.scandir(staging_dir) as it:
            for entry in it:
                name = entry.name
                if not name.endswith(suffix):
                    continue
                try:
                    mtime = entry.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                out.append((name[: -len(suffix)], mtime))
    except FileNotFoundError:
        return []
    return out


def check_disk_headroom(
    staging_dir: Path, required_bytes: int,
) -> Tuple[bool, int]:
    """Return ``(free_bytes >= required_bytes, free_bytes)``.

    2026-07-21 ENOSPC fleet stall: the guard the dump worker consults
    before writing a fragment. ``staging_dir`` is created if it does not
    exist yet (a fresh node has no staging dir at all) so a never-
    written corr node gets a real ``statvfs`` reading rather than an
    ``OSError``.
    """
    staging_dir = Path(staging_dir)
    try:
        usage = shutil.disk_usage(staging_dir)
    except OSError:
        staging_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(staging_dir)
    return usage.free >= required_bytes, usage.free


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
    utc_start_specnum: Optional[int] = None,
    armed_mjd: Optional[float] = None,
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
    try:
        with open(tmp_path, "wb") as fh:
            for _b, arr in extract.blocks:
                fh.write(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
                n_written += 1
        os.replace(tmp_path, out_path)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            # 2026-07-21 fleet stall: this is the exact traceback site
            # (fh.write -> ENOSPC, logged 21:43:05 on n22). Clean up the
            # partial .tmp fragment so a storm of failed dumps doesn't
            # itself leave junk behind, then hand a typed error to the
            # caller so the worker loop can count/log it once and keep
            # going (the reader thread / RAM ring are unaffected either
            # way — this is the disk-worker path only).
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise DiskFullError(
                f"ENOSPC writing {tmp_path} "
                f"({n_written} of {len(extract.blocks)} blocks written)"
            ) from exc
        raise

    manifest: Dict[str, Any] = {
        # v2 (2026-07-19): block_n_* keys and ring block labels are now
        # ZERO-based absolute block numbers (block N = specnums
        # [2048N, 2048(N+1))), matching target_block_n's convention;
        # block_mjd_first is armed_mjd + block_n_first*BLOCK. v1
        # manifests had 1-based block_n_* (pages-since-attach) with
        # block_mjd_first = armed + (N-1)*BLOCK — the recorded TIME was
        # correct for the data in both versions; the window centering
        # and label conventions changed.
        "manifest_version": 2,
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
        # 2026-07-16 absolute-time provenance: the capture arm record.
        # armed_mjd (wall MJD at the utc_start verb) + block numbering
        # gives sample times on the SAME base as the slow-vis HDF5
        # archive, independent of the C2 label (which historically ran
        # LATE by the search pipeline's first-cube fill latency).
        # v2 (2026-07-19): block_n is ZERO-based absolute, so
        #   t(block N start) = armed_mjd + N * BLOCK_DURATION_S.
        # This is the field the localization chain (dsavim.metadata
        # .load_burst) anchors the voltage-MS time axis on — the C2
        # label (mjd_target) carries per-event centroid jitter of
        # ~0.1 s (~1.5 arcsec of RA) and must not be used for that.
        "utc_start_specnum": (
            int(utc_start_specnum) if utc_start_specnum is not None
            else None
        ),
        "armed_mjd": (
            float(armed_mjd) if armed_mjd is not None else None
        ),
        "block_mjd_first": (
            float(armed_mjd)
            + extract.first_block_n * BLOCK_DURATION_S / 86400.0
            if armed_mjd is not None and extract.first_block_n is not None
            else None
        ),
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

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._available or self._store is None:
            return None
        try:
            return self._store.get_dict(key)
        except Exception:  # noqa: BLE001
            LOG.warning("etcd get_dict(%s) failed", key)
            return None


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
        # absolute-time provenance for manifests (2026-07-16): the fada
        # header's UTC_START (SNAP-wall specnum the capture armed at)
        # captured by the reader thread, and armed_mjd from etcd read
        # lazily at the first dump.
        self._utc_start_specnum: Optional[int] = None
        self._armed_mjd: Optional[float] = None
        self._armed_mjd_read = False
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
        self._sweeper_thread: Optional[threading.Thread] = None
        self._listener: Optional[VoltageTriggerListener] = None
        self._counters: Dict[str, int] = {
            "blocks_read": 0,
            "dumps_done": 0,
            "dumps_failed": 0,
            "deletes_done": 0,
            "blocks_dropped_total": 0,
            # 2026-07-21 disk-headroom guard counters (ENOSPC fleet
            # stall). Exposed on the same mon payload as the rest of
            # _counters (see _mon_loop).
            "dumps_skipped_disk_full": 0,
            "dumps_skipped_staging_cap": 0,
            "dumps_failed_enospc": 0,
            # 2026-07-24 orphan-eviction guard: fragments the periodic
            # sweeper dropped because they outlived staging_orphan_ttl_s
            # (C2-discarded events C3 never adjudicates; counted apart
            # from C3-driven deletes_done for observability).
            "orphans_evicted": 0,
        }
        # In-process running total of bytes staged in config.staging_dir,
        # seeded from disk at startup, kept current on every successful
        # write/delete. Cheap stand-in for a full `du` on every dump —
        # bounds the C2-storm case per plan (many real triggers, disk
        # otherwise empty).
        self._staged_bytes: int = _staging_dir_bytes(config.staging_dir)

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

    def _enqueue_evict(self, event_name: str) -> bool:
        try:
            self._q.put_nowait(("evict", event_name, 0, 0.0))
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
            try:
                self._utc_start_specnum = int(hdr.get("UTC_START"))
            except (TypeError, ValueError):
                self._utc_start_specnum = None
            if self._utc_start_specnum is None:
                # 2026-07-19: production fada headers turned out not to
                # carry UTC_START (every manifest shipped null). Fall
                # back to the capture processes' mon shm, which always
                # has the armed specnum.
                self._utc_start_specnum = _utc_start_from_capture_mon()
            # 2026-07-19 block-numbering reconciliation: label pages
            # ZERO-based since attach so ring labels match the absolute
            # ``specnum_to_block_n`` convention (block N holds specnums
            # [2048N, 2048(N+1)); corr_fast/cube_pipeline anchor). The
            # previous 1-based count made every C2-triggered window
            # land one block EARLY in the data (the trigger specnum sat
            # in ring label floor(spec/2048)+1, but we extracted
            # floor(spec/2048)), and made the manifests' block_n_* keys
            # 1-based while target_block_n was 0-based. NB the labels
            # are absolute only when this service attaches at capture
            # start (systemd orders it with the pipeline); a
            # mid-capture attach offsets ALL labels and times, which is
            # why the manifest carries armed_mjd for downstream
            # cross-checks.
            block_n = -1
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
                if kind in ("delete", "evict"):
                    # Both remove a staged .out/.json pair and decrement
                    # the cap counter through this single (worker-thread)
                    # path so _staged_bytes stays race-free. "delete" is
                    # C3-driven (KEEP-collect cleanup / REJECT); "evict"
                    # is the TTL sweeper dropping an orphan. Counted
                    # separately for observability.
                    removed_bytes = _staged_fragment_bytes(
                        self._cfg.staging_dir, name, self._cfg.chgroup,
                    )
                    n = delete_staged(
                        self._cfg.staging_dir, name, self._cfg.chgroup,
                    )
                    if n:
                        self._staged_bytes = max(
                            0, self._staged_bytes - removed_bytes,
                        )
                    if kind == "evict":
                        self._counters["orphans_evicted"] += 1
                        LOG.warning(
                            "voltage_retention: evicted ORPHAN staged %s "
                            "(%d files, %.2f GiB) — older than TTL %.0f s; "
                            "C2-discarded event never collected by C3",
                            name, n, removed_bytes / (1 << 30),
                            self._cfg.staging_orphan_ttl_s,
                        )
                    else:
                        self._counters["deletes_done"] += 1
                        LOG.info(
                            "voltage_retention: deleted staged %s (%d files)",
                            name, n,
                        )
                    continue
                target = specnum_to_block_n(specnum)
                self._wait_for_post(target)

                # --- 2026-07-21 disk-headroom guard (ENOSPC fleet stall) ---
                # Checked here, in the disk-worker path only, BEFORE any
                # bytes are written. The reader thread / RAM ring are
                # untouched by either branch below — the trigger is
                # still absorbed off the queue exactly like a completed
                # dump, it just isn't staged to NVMe.
                expected = self._cfg.expected_fragment_bytes
                required_free = max(
                    2 * expected, self._cfg.min_free_bytes_floor,
                )
                free_ok, free_bytes = check_disk_headroom(
                    self._cfg.staging_dir, required_free,
                )
                if not free_ok:
                    self._counters["dumps_skipped_disk_full"] += 1
                    LOG.warning(
                        "voltage_retention: SKIPPING dump %s %s — disk "
                        "headroom guard: %.2f GiB free < %.2f GiB "
                        "required (2x expected fragment or floor); not "
                        "writing (2026-07-21 ENOSPC fleet stall guard)",
                        name, self._cfg.sb,
                        free_bytes / (1 << 30), required_free / (1 << 30),
                    )
                    continue
                if (
                    self._staged_bytes + expected
                    > self._cfg.staging_max_total_bytes
                ):
                    self._counters["dumps_skipped_staging_cap"] += 1
                    LOG.warning(
                        "voltage_retention: SKIPPING dump %s %s — "
                        "cumulative staging cap: %.2f GiB staged + "
                        "%.2f GiB fragment > %.2f GiB cap; not writing "
                        "(2026-07-21 C2 storm containment guard)",
                        name, self._cfg.sb,
                        self._staged_bytes / (1 << 30),
                        expected / (1 << 30),
                        self._cfg.staging_max_total_bytes / (1 << 30),
                    )
                    continue

                if not self._armed_mjd_read:
                    # once per process; the arm record is fixed for the
                    # lifetime of this capture epoch
                    self._armed_mjd_read = True
                    doc = self._mon_store.get_dict("/mon/snap/1/armed_mjd") or {}
                    try:
                        v = float(doc.get("armed_mjd") or 0.0)
                        self._armed_mjd = v if v > 40000.0 else None
                    except (TypeError, ValueError):
                        self._armed_mjd = None
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
                    utc_start_specnum=self._utc_start_specnum,
                    armed_mjd=self._armed_mjd,
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
                self._staged_bytes += int(manifest["total_bytes"])
                LOG.info(
                    "voltage_retention: staged %s %s — %d blocks (%d dropped) "
                    "%.2f GiB",
                    name, self._cfg.sb,
                    manifest["n_blocks_written"], manifest["n_blocks_dropped"],
                    manifest["total_bytes"] / (1 << 30),
                )
            except DiskFullError:
                # 2026-07-21 fleet stall: ENOSPC hit mid-write despite
                # the pre-write headroom check (a concurrent writer, or
                # the disk filling between the check and the write).
                # write_window_to_staging already cleaned up the partial
                # .tmp fragment. Count it separately from generic
                # failures and keep the worker loop running — this is
                # the behavior that survived the incident tonight; we're
                # only adding cleanup + a dedicated counter.
                self._counters["dumps_failed_enospc"] += 1
                LOG.error(
                    "voltage_retention: ENOSPC writing dump %s %s — "
                    "partial fragment cleaned up, continuing",
                    name, self._cfg.sb,
                )
            except Exception:  # noqa: BLE001
                self._counters["dumps_failed"] += 1
                LOG.exception(
                    "voltage_retention: dump failed for %s specnum=%d",
                    name, specnum,
                )

    # ----- orphan sweeper thread ---------------------------------------

    def _sweep_orphans(self) -> int:
        """Enqueue an ``evict`` for every staged fragment older than the
        TTL. Returns the number enqueued this pass.

        C3 only ever deletes events it adjudicates, so a C2 dump-trigger
        whose event C2 later discards (cube-incomplete, the RFI-storm
        common case) strands its fragment forever and eventually pins the
        cumulative-staging cap (2026-07-23: 31 orphans / ~1.81 TiB killed
        dumping for ~18 h). Any fragment older than ``staging_orphan_ttl_s``
        was provably never collected (>> c3.collect_timeout_s), so it is
        an orphan. Evicts run through the worker queue so the single
        worker thread owns every ``_staged_bytes`` mutation; a full queue
        just defers the rest to the next sweep.
        """
        ttl = float(self._cfg.staging_orphan_ttl_s)
        if ttl <= 0:
            return 0
        cutoff = time.time() - ttl
        n = 0
        for name, mtime in scan_staged_events(
            self._cfg.staging_dir, self._cfg.chgroup,
        ):
            if mtime < cutoff:
                if not self._enqueue_evict(name):
                    break                        # queue full; next pass
                n += 1
        if n:
            LOG.info(
                "voltage_retention: orphan sweep enqueued %d eviction(s) "
                "(> %.0f s old) on %s", n, ttl, self._cfg.sb,
            )
        return n

    def _sweeper_loop(self) -> None:
        interval = max(1.0, float(self._cfg.orphan_sweep_interval_s))
        if float(self._cfg.staging_orphan_ttl_s) <= 0:
            LOG.info("voltage_retention: orphan sweep disabled (ttl<=0)")
            return
        # small initial delay so a fresh start doesn't race the reader
        # attach / first mon publish
        if self._stop.wait(min(interval, 30.0)):
            return
        while not self._stop.is_set():
            try:
                self._sweep_orphans()
            except Exception:  # noqa: BLE001 — never let the sweep kill the thread
                LOG.exception("voltage_retention: orphan sweep failed")
            self._stop.wait(interval)

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
                    "staging_orphan_ttl_s": self._cfg.staging_orphan_ttl_s,
                    "staged_bytes": self._staged_bytes,
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
        self._sweeper_thread = threading.Thread(
            target=self._sweeper_loop, name="vret-sweeper", daemon=True,
        )
        self._reader_thread.start()
        self._worker_thread.start()
        self._sweeper_thread.start()

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
    # 2026-07-21 ENOSPC fleet stall (dozens of C2 dumps 19:20-21:42 UT
    # filled the root filesystem on 9/16 corr nodes; see module
    # docstring / RetentionConfig for the incident + cascade). These two
    # knobs are the disk-headroom + storm-containment guard.
    p.add_argument("--min-free-bytes-floor", type=int,
                   default=_DEFAULT_MIN_FREE_BYTES_FLOOR,
                   help="Refuse a dump if free space in --staging-dir "
                        "would drop below max(2x expected fragment "
                        "size, this floor). Bytes. Default 16 GiB.")
    p.add_argument("--staging-max-total-bytes", type=int,
                   default=_DEFAULT_STAGING_MAX_TOTAL_BYTES,
                   help="Refuse a dump if it would push the cumulative "
                        "bytes resident in --staging-dir over this cap "
                        "(bounds a C2 trigger storm even with an "
                        "otherwise-empty disk). Bytes. Default 120 GiB.")
    # 2026-07-24 orphan-eviction guard: without it, C2-discarded events
    # (never adjudicated by C3) strand their fragments and eventually
    # pin --staging-max-total-bytes forever (2026-07-23: killed dumping
    # ~18 h). TTL must exceed c3.collect_timeout_s with margin.
    p.add_argument("--staging-orphan-ttl-s", type=float,
                   default=_DEFAULT_STAGING_ORPHAN_TTL_S,
                   help="Evict a staged fragment older than this (a "
                        "C2-discarded orphan C3 never collected). Must "
                        "exceed c3.collect_timeout_s. 0 disables. "
                        "Seconds. Default 7200 (2 h).")
    p.add_argument("--orphan-sweep-interval-s", type=float,
                   default=_DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
                   help="How often the orphan sweeper scans --staging-dir. "
                        "Seconds. Default 300.")
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
        min_free_bytes_floor=int(args.min_free_bytes_floor),
        staging_max_total_bytes=int(args.staging_max_total_bytes),
        staging_orphan_ttl_s=float(args.staging_orphan_ttl_s),
        orphan_sweep_interval_s=float(args.orphan_sweep_interval_s),
    )
    svc = VoltageRetentionService(cfg)
    return asyncio.run(svc.run())


if __name__ == "__main__":
    sys.exit(main())
