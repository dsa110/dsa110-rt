"""h23-side in-memory ring buffer for 16-corr-node RFI window records.

A background thread (``RFIPoller``) hits each corr node's
``/api/recent`` endpoint at startup (warmup → backfill the ring),
then drops to polling ``/api/latest`` per node every ~2 s. New
records are appended to a per-cn deque, deduped on ``seq``, and
trimmed to the 30-min retention window.

Page renderers consume the store via :meth:`RFIWindowStore.snapshot`,
which returns a frozen view of all 16 corr nodes' rings ordered by
chgroup; that's the input to the per-antenna plot/table builder.

Concurrency: a single RLock protects the per-cn deques. Snapshots
copy out the deque contents so subsequent renders never observe
partial mutations.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import OrderedDict, deque
from typing import Dict, Optional

from corr_topology import CORR_NODES, CorrNode, CORR_NODES_BY_CN
from rfi_client import DecodedRFIMonRecord, RFIClient, build_clients

LOG = logging.getLogger("dsa_monitor.rfi_store")


# Retention horizon: per the M7.6 design, the dashboard keeps the
# last 30 min of windows. At 16 cubes/window × 134.218 ms = 2.147
# s/window, that's 30·60 / 2.147 ≈ 838 windows per cn. We size the
# deque to (RETENTION_S / WINDOW_PERIOD_S) + small slack.
RETENTION_S: float = 30 * 60                       # 30 minutes
WINDOW_PERIOD_S: float = 2.147                     # nominal 16-cube cadence
_PER_CN_DEQUE_CAP: int = int(RETENTION_S / WINDOW_PERIOD_S) + 32
"""~870 records/cn → ~108 MB total in-memory for 16 cn at production
NANTS=96 × NCHAN_DS=96 × NPOL=2."""

# Per-corr-node poller cadence. 2.0 s matches the producer cadence
# (one new record every ~2.147 s), so we usually pull exactly one
# new record per tick. Tunable via env / config.
POLL_PERIOD_S: float = 2.0

# Per-corr-node startup backfill: when the poller first attaches it
# pulls `BACKFILL_N` records via /api/recent so the dashboard isn't
# empty on the first render. 64 = the producer's default ring depth
# = ~137 s of history.
BACKFILL_N: int = 64


@dataclasses.dataclass(frozen=True)
class CorrNodeRingSnapshot:
    """Frozen view of one cn's ring for the render path."""

    cn: CorrNode
    records: tuple[DecodedRFIMonRecord, ...]      # oldest first
    last_seq: int                                 # 0 = never observed
    last_publish_unix: float                      # 0.0 if empty
    last_fetch_unix: float                        # 0.0 if never polled
    last_fetch_ok: bool


@dataclasses.dataclass(frozen=True)
class StoreSnapshot:
    """Snapshot of all 16 corr-nodes' rings, ordered by chgroup."""

    per_chgroup: tuple[CorrNodeRingSnapshot, ...]
    snapshot_unix: float

    @property
    def n_chgroup(self) -> int:
        return len(self.per_chgroup)


class RFIWindowStore:
    """In-memory per-corr-node ring buffer of decoded window records."""

    def __init__(
        self,
        *,
        retention_s: float = RETENTION_S,
        per_cn_cap: int = _PER_CN_DEQUE_CAP,
    ) -> None:
        self._retention_s = float(retention_s)
        self._lock = threading.RLock()
        # OrderedDict preserves chgroup order (CORR_NODES iteration).
        self._rings: OrderedDict[int, deque[DecodedRFIMonRecord]] = OrderedDict()
        self._last_seq: Dict[int, int] = {}
        self._last_fetch_unix: Dict[int, float] = {}
        self._last_fetch_ok: Dict[int, bool] = {}
        for cn in CORR_NODES:
            self._rings[cn.cn_id] = deque(maxlen=per_cn_cap)
            self._last_seq[cn.cn_id] = 0
            self._last_fetch_unix[cn.cn_id] = 0.0
            self._last_fetch_ok[cn.cn_id] = False

    # ------------------------------------------------------------------
    # Producer API (called by poller)
    # ------------------------------------------------------------------

    def append(self, recs: list[DecodedRFIMonRecord], *, cn_id: int) -> int:
        """Append decoded records for a specific cn. Dedupes against
        the last seen ``seq``. Trims records older than
        ``retention_s``. Returns the number of new records appended."""
        if not recs:
            with self._lock:
                self._last_fetch_unix[cn_id] = time.time()
                self._last_fetch_ok[cn_id] = True
            return 0
        n_new = 0
        with self._lock:
            ring = self._rings[cn_id]
            last_seq = self._last_seq.get(cn_id, 0)
            for r in recs:
                if r.seq <= last_seq:
                    continue
                ring.append(r)
                last_seq = r.seq
                n_new += 1
            self._last_seq[cn_id] = last_seq
            self._last_fetch_unix[cn_id] = time.time()
            self._last_fetch_ok[cn_id] = True
            self._trim_unlocked(cn_id)
        return n_new

    def mark_fetch_failed(self, cn_id: int) -> None:
        with self._lock:
            self._last_fetch_unix[cn_id] = time.time()
            self._last_fetch_ok[cn_id] = False

    # ------------------------------------------------------------------
    # Consumer API (called by render path)
    # ------------------------------------------------------------------

    def snapshot(self) -> StoreSnapshot:
        """Frozen view ordered by chgroup. Records inside each ring
        are oldest-first."""
        with self._lock:
            now = time.time()
            self._trim_all_unlocked()
            per_chgroup = []
            for cn in CORR_NODES:
                ring = self._rings[cn.cn_id]
                records = tuple(ring)
                last_publish = (records[-1].publish_unix
                                if records else 0.0)
                per_chgroup.append(CorrNodeRingSnapshot(
                    cn=cn,
                    records=records,
                    last_seq=self._last_seq[cn.cn_id],
                    last_publish_unix=last_publish,
                    last_fetch_unix=self._last_fetch_unix[cn.cn_id],
                    last_fetch_ok=self._last_fetch_ok[cn.cn_id],
                ))
        return StoreSnapshot(
            per_chgroup=tuple(per_chgroup),
            snapshot_unix=now,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trim_all_unlocked(self) -> None:
        for cn_id in self._rings:
            self._trim_unlocked(cn_id)

    def _trim_unlocked(self, cn_id: int) -> None:
        ring = self._rings[cn_id]
        if not ring:
            return
        horizon = time.time() - self._retention_s
        while ring and ring[0].publish_unix < horizon:
            ring.popleft()


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------


class RFIPoller:
    """Background thread that polls each corr-node exporter."""

    def __init__(
        self,
        store: RFIWindowStore,
        *,
        poll_period_s: float = POLL_PERIOD_S,
        backfill_n: int = BACKFILL_N,
    ) -> None:
        self._store = store
        self._poll_period_s = float(poll_period_s)
        self._backfill_n = int(backfill_n)
        self._clients = build_clients(CORR_NODES)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backfilled: set[int] = set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="rfi_poller", daemon=True,
        )
        self._thread.start()
        LOG.info("RFIPoller started: cadence=%.1fs cn_count=%d",
                 self._poll_period_s, len(self._clients))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            # Parallel-ish polling. We don't bother with a thread pool
            # because each fetch is a single tiny HTTP GET (~150 KB for
            # a single window) and 16 nodes serial is well within
            # the 2 s cadence even at ~50 ms / node.
            for cn_id, client in self._clients.items():
                if self._stop.is_set():
                    break
                self._poll_one(cn_id, client)
            elapsed = time.time() - t0
            sleep_s = max(0.1, self._poll_period_s - elapsed)
            self._stop.wait(sleep_s)

    def _poll_one(self, cn_id: int, client: RFIClient) -> None:
        try:
            if cn_id not in self._backfilled:
                # First successful contact: pull /api/recent so the
                # dashboard isn't empty on first render. Subsequent
                # ticks only pull /api/latest.
                recs = client.get_recent(self._backfill_n)
                if recs:
                    n = self._store.append(recs, cn_id=cn_id)
                    self._backfilled.add(cn_id)
                    LOG.info(
                        "cn=%d backfilled %d records (%d new)",
                        cn_id, len(recs), n,
                    )
                    return
                # No records yet — flag a failed fetch and try again next tick.
                self._store.mark_fetch_failed(cn_id)
                return
            rec = client.get_latest()
            if rec is None:
                self._store.mark_fetch_failed(cn_id)
                return
            self._store.append([rec], cn_id=cn_id)
        except Exception:
            LOG.exception("poll cn=%d failed", cn_id)
            self._store.mark_fetch_failed(cn_id)
