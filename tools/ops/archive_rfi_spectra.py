#!/usr/bin/env python3
"""Archive the RFI monitor spectra that feed the h23 dashboard plots.

Written 2026-08-10 for an on-site RFI source-tracking campaign: the
dashboard's plots are built from a 30-minute in-memory ring, so anything
older than that is gone and cannot be recovered afterwards.

This does NOT touch the dashboard. Each corr node runs its own
``rfi_monitor_export`` HTTP endpoint on port 5780, and the dashboard is
simply one consumer of it; this is a second, independent consumer of the
same endpoints. It imports ``rfi_client`` read-only so the base64/dtype
decode cannot drift from the dashboard's, but it shares no state with the
running service and cannot perturb it.

What is saved, per record (one per ~2.147 s window per node):

  s1_full_mean       (NANTS, NCHAN_DS, NPOL) float32 -- THE SPECTRA, the
                     array the per-antenna RFI plots are drawn from
  mask_count_final   (…) uint8  -- flag count, all detectors combined
  mask_count_sk      spectral kurtosis
  mask_count_bp      bandpass outlier
  mask_count_grp     group outlier (one decision broadcast across channels)
  mask_count_sumthr  sum-threshold
  mask_count_fa      flagants.dat (static)
  scalars            total_flag_fraction, ant_fraction_flagged, frac_* ...

The per-detector masks are kept deliberately: for tracking down a source,
knowing WHICH detector fired on a channel is usually more diagnostic than
the combined mask, and they cost little after compression.

Robustness, because a 12 h unattended run will hit hiccups:

* Records are deduped on the publisher's ``seq``, so overlapping fetches
  are harmless.
* On a seq gap the script backfills via ``/api/recent`` rather than
  accepting the hole. The node's own ring is ~64 records (~137 s), so a
  stall shorter than that loses nothing; longer gaps are logged
  explicitly in the manifest rather than passing silently.
* Every node is polled independently -- one unreachable node does not
  stop the others, and its downtime is recorded.
* Chunks are flushed to disk on a wall-clock boundary, so killing the
  script loses at most one partial chunk.

Usage::

    ./archive_rfi_spectra.py --hours 12 --out /dataz/.../rfi_capture_X
    ./archive_rfi_spectra.py --hours 12 --out DIR --no-masks   # spectra only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

#: The dashboard package is not installed, so add it to the path to reuse
#: its decode. Read-only import; nothing here mutates dashboard state.
_DASH = "/home/ubuntu/vikram/dev/dsa110-rt/tools/dashboard/dsa_monitor"
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

from corr_topology import CORR_NODES          # noqa: E402
from rfi_client import build_clients           # noqa: E402

LOG = logging.getLogger("archive_rfi_spectra")

MASK_KEYS = (
    "mask_count_final", "mask_count_sk", "mask_count_bp",
    "mask_count_grp", "mask_count_sumthr", "mask_count_fa",
)

#: Producer publishes one window per ~2.147 s; poll a little faster so we
#: never sit a whole window behind.
POLL_PERIOD_S = 2.0

#: Flush a chunk per node this often. 10 min => ~280 records/chunk,
#: ~50 MiB before compression: small enough that a kill costs little,
#: large enough that we are not writing thousands of tiny files.
CHUNK_S = 600.0

#: The node's own ring depth. A gap longer than this is unrecoverable and
#: gets recorded as a hole.
NODE_RING_N = 64

_stop = False


def _on_signal(signum, _frame):
    global _stop
    _stop = True
    LOG.warning("signal %s — finishing the current chunk and exiting", signum)


class NodeArchive:
    """Accumulates one corr node's records and flushes them to npz."""

    def __init__(self, cn_id: int, out_dir: str, keep_masks: bool) -> None:
        self.cn_id = cn_id
        self.out_dir = out_dir
        self.keep_masks = keep_masks
        self.last_seq: Optional[int] = None
        self.buf: List = []
        self.n_written = 0
        self.n_records = 0
        self.n_fetch_fail = 0
        self.gaps: List[dict] = []

    def note(self, rec) -> bool:
        """Add a record if unseen. Returns True if it was new."""
        if self.last_seq is not None and rec.seq <= self.last_seq:
            return False
        if self.last_seq is not None and rec.seq > self.last_seq + 1:
            missed = rec.seq - self.last_seq - 1
            self.gaps.append({"after_seq": self.last_seq, "to_seq": rec.seq,
                              "missed": missed, "at_unix": time.time(),
                              "recoverable": missed <= NODE_RING_N})
            LOG.warning("cn%02d seq gap %d -> %d (%d missed)",
                        self.cn_id, self.last_seq, rec.seq, missed)
        self.last_seq = rec.seq
        self.buf.append(rec)
        self.n_records += 1
        return True

    def flush(self) -> Optional[str]:
        if not self.buf:
            return None
        recs = self.buf
        self.buf = []
        first, last = recs[0], recs[-1]
        payload: Dict[str, np.ndarray] = {
            "seq": np.array([r.seq for r in recs], dtype=np.int64),
            "publish_unix": np.array([r.publish_unix for r in recs]),
            "block_n_start": np.array([r.block_n_start for r in recs], np.int64),
            "block_n_end": np.array([r.block_n_end for r in recs], np.int64),
            "n_cubes": np.array([r.n_cubes for r in recs], np.int32),
            "n_cubes_warmup": np.array([r.n_cubes_warmup for r in recs], np.int32),
            "s1_full_mean": np.stack([r.s1_full_mean for r in recs]),
        }
        if self.keep_masks:
            for k in MASK_KEYS:
                payload[k] = np.stack([getattr(r, k) for r in recs])
        # Scalars: (n_records, 3) per name, ordered (pol0, pol1, both).
        names = sorted(first.scalars)
        payload["scalar_names"] = np.array(names)
        payload["scalars"] = np.array(
            [[list(r.scalars.get(n, (np.nan,) * 3)) for n in names] for r in recs],
            dtype=np.float64,
        )
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(first.publish_unix))
        path = os.path.join(
            self.out_dir, "rfi_cn%02d_%s_seq%d-%d.npz"
            % (self.cn_id, stamp, first.seq, last.seq))
        np.savez_compressed(path, **payload)
        self.n_written += 1
        LOG.info("cn%02d wrote %d records -> %s (%.1f MiB)", self.cn_id,
                 len(recs), os.path.basename(path),
                 os.path.getsize(path) / 2 ** 20)
        return path

    def stats(self) -> dict:
        return {"cn_id": self.cn_id, "n_records": self.n_records,
                "n_files": self.n_written, "n_fetch_fail": self.n_fetch_fail,
                "last_seq": self.last_seq, "n_gaps": len(self.gaps),
                "gaps": self.gaps[:200]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--poll-s", type=float, default=POLL_PERIOD_S)
    ap.add_argument("--chunk-s", type=float, default=CHUNK_S)
    ap.add_argument("--no-masks", action="store_true",
                    help="save only the spectra + scalars (~40%% the size)")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    os.makedirs(a.out, exist_ok=True)
    clients = build_clients(CORR_NODES)
    arcs = {cn: NodeArchive(cn, a.out, not a.no_masks) for cn in clients}
    LOG.info("archiving %d corr nodes -> %s for %.1f h (masks=%s)",
             len(clients), a.out, a.hours, not a.no_masks)

    t_end = time.time() + a.hours * 3600.0
    t_next_flush = time.time() + a.chunk_s
    t0 = time.time()
    while not _stop and time.time() < t_end:
        tick = time.time()
        for cn, c in clients.items():
            arc = arcs[cn]
            try:
                rec = c.get_latest()
            except Exception as exc:                          # noqa: BLE001
                arc.n_fetch_fail += 1
                LOG.warning("cn%02d get_latest failed: %s", cn, exc)
                continue
            if rec is None:
                arc.n_fetch_fail += 1
                continue
            # A jump means we missed windows; pull the node's ring and
            # replay what is still there before taking the new record.
            if arc.last_seq is not None and rec.seq > arc.last_seq + 1:
                try:
                    for old in sorted(c.get_recent(NODE_RING_N),
                                      key=lambda r: r.seq):
                        if old.seq > arc.last_seq:
                            arc.note(old)
                except Exception as exc:                      # noqa: BLE001
                    LOG.warning("cn%02d backfill failed: %s", cn, exc)
            arc.note(rec)

        if time.time() >= t_next_flush:
            for arc in arcs.values():
                arc.flush()
            _write_manifest(a, arcs, t0, done=False)
            t_next_flush = time.time() + a.chunk_s

        slack = a.poll_s - (time.time() - tick)
        if slack > 0:
            time.sleep(slack)

    for arc in arcs.values():
        arc.flush()
    _write_manifest(a, arcs, t0, done=True)
    tot = sum(arc.n_records for arc in arcs.values())
    LOG.info("done: %d records over %.2f h from %d nodes",
             tot, (time.time() - t0) / 3600.0, len(arcs))
    return 0


def _write_manifest(a, arcs, t0, *, done: bool) -> None:
    size = 0
    for f in os.listdir(a.out):
        if f.endswith(".npz"):
            size += os.path.getsize(os.path.join(a.out, f))
    man = {
        "started_unix": t0, "updated_unix": time.time(), "complete": done,
        "requested_hours": a.hours, "elapsed_hours": (time.time() - t0) / 3600.0,
        "poll_period_s": a.poll_s, "chunk_s": a.chunk_s,
        "masks_saved": not a.no_masks,
        "bytes_on_disk": size,
        "source": "corr-node rfi_monitor_export :5780 /api/latest + /api/recent",
        "note": ("Independent consumer of the same endpoints the h23 dashboard "
                 "polls; the dashboard keeps only a 30-min in-memory ring. "
                 "s1_full_mean is the spectra array the per-antenna RFI plots "
                 "are built from."),
        "nodes": [arc.stats() for arc in sorted(arcs.values(), key=lambda x: x.cn_id)],
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
