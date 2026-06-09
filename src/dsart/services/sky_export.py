"""Static-sky snapshot exporter (corr → h23 sky monitor).

E2E correctness test 1 ("always seeing the sky"): every
``interval_s`` seconds, snapshot the slot-0 :class:`StaticSkyEMA`
running mean — the corr node's live estimate of the static-sky
gridded visibilities, in exactly the UV geometry the search-side
imager uses — and HTTP-POST it to the h23 dashboard's
``/sky/ingest`` endpoint. h23 combines the 16 chgroup snapshots
into one dirty image and serves a scrubbable greyscale movie on
the Sky tab.

Why slot 0: the per-coarse-DM EMA states differ only in the
stage-1 dedispersion shifts applied upstream; a static continuum
source is constant in time, so its time-averaged gridded
visibility is the same in every slot. Slot 0 (lowest coarse DM,
smallest shifts) is the cleanest choice.

EMA timescale: ``apply()`` runs once per fada block (134.218 ms)
per slot, with ``alpha = 0.001`` → half-life ``0.69/alpha ≈ 693``
blocks ≈ **93 s** (time constant 1/alpha = 1000 blocks ≈ 134 s).
A 30 s export cadence therefore samples the EMA ~3× per half-life;
adjacent movie frames are correlated, which is what you want for a
"is the sky still there" monitor.

Hot-path safety contract
========================

* :meth:`maybe_export` is called once per block from the main
  loop. In the common case (interval not yet elapsed) it is a
  single ``time.monotonic()`` compare — nanoseconds.
* On an export tick the only main-thread work is one D2H copy of
  the ``(N_filled,)`` complex64 EMA (~40 KB → sub-ms) plus an
  in-memory npz serialise (~100 µs). The HTTP POST happens on a
  daemon worker thread with a bounded queue; if the worker is
  behind (h23 slow/down) the snapshot is dropped, never queued
  unboundedly, and the RT loop is never blocked.
* Failures are logged at WARN with exponential suppression so a
  down dashboard doesn't spam the corr_fast log at 2 lines/min
  forever.

Snapshot-consistency note: in the 3-stream pipeliner the EMA
tensor is updated on the dedisp stream while we read it from the
main thread without a sync. A torn read mixes values that differ
by at most one ``alpha = 0.1%`` blend step — irrelevant for a
monitoring image, and not worth a cross-stream sync on the hot
path.
"""
from __future__ import annotations

import io
import json
import logging
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np

LOG = logging.getLogger("dsart.sky_export")

#: Wire-format version stamped into every payload so the h23 ingest
#: can reject snapshots from an incompatible corr build.
SKY_SNAPSHOT_VERSION: int = 1

#: Bounded handoff queue depth (snapshots, not bytes). 2 = the worker
#: may be one POST behind before we start dropping.
_QUEUE_DEPTH: int = 2


def build_snapshot_npz(
    vis: np.ndarray,
    *,
    ix_row: np.ndarray,
    ix_col: np.ndarray,
    meta: dict[str, Any],
) -> bytes:
    """Serialise one chgroup sky snapshot to in-memory ``.npz`` bytes.

    Args:
        vis: ``(N_filled,)`` complex64 — slot-0 EMA running mean.
        ix_row, ix_col: ``(N_filled,)`` uint16 — grid (row, col) of
            each filled cell (:class:`SparsityPattern` contract).
        meta: JSON-serialisable scalars (chgroup, n_grid,
            cell_lambda, amp_scale, block_n, unix_ts, ...).

    The pattern indices ride along in EVERY payload (~20 KB) so the
    h23 side is stateless w.r.t. corr restarts / pattern rebuilds.
    """
    vis = np.ascontiguousarray(vis, dtype=np.complex64)
    if vis.ndim != 1:
        raise ValueError(f"vis must be 1-D (N_filled,); got {vis.shape}")
    n_filled = vis.shape[0]
    ix_row = np.ascontiguousarray(ix_row, dtype=np.uint16)
    ix_col = np.ascontiguousarray(ix_col, dtype=np.uint16)
    if ix_row.shape != (n_filled,) or ix_col.shape != (n_filled,):
        raise ValueError(
            f"ix_row/ix_col shapes {ix_row.shape}/{ix_col.shape} != "
            f"({n_filled},)"
        )
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        version=np.int64(SKY_SNAPSHOT_VERSION),
        vis=vis,
        ix_row=ix_row,
        ix_col=ix_col,
        meta_json=np.bytes_(json.dumps(meta).encode("utf-8")),
    )
    return buf.getvalue()


def parse_snapshot_npz(body: bytes) -> dict[str, Any]:
    """Inverse of :func:`build_snapshot_npz`. Raises ``ValueError``
    on malformed / version-mismatched payloads.
    """
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as z:
            version = int(z["version"])
            if version != SKY_SNAPSHOT_VERSION:
                raise ValueError(
                    f"sky snapshot version {version} != "
                    f"{SKY_SNAPSHOT_VERSION}"
                )
            vis = np.asarray(z["vis"], dtype=np.complex64)
            ix_row = np.asarray(z["ix_row"], dtype=np.uint16)
            ix_col = np.asarray(z["ix_col"], dtype=np.uint16)
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed sky snapshot: {exc!r}") from exc
    n_filled = vis.shape[0]
    if vis.ndim != 1 or ix_row.shape != (n_filled,) or ix_col.shape != (n_filled,):
        raise ValueError(
            f"sky snapshot shape mismatch: vis={vis.shape} "
            f"ix_row={ix_row.shape} ix_col={ix_col.shape}"
        )
    return {"vis": vis, "ix_row": ix_row, "ix_col": ix_col, "meta": meta}


class SkySnapshotExporter:
    """Periodic, fail-soft EMA snapshot push (see module docstring).

    Args:
        url: h23 ingest endpoint, e.g.
            ``http://lxd110h23.pro.pvt:5778/sky/ingest``.
        interval_s: export cadence (wall-clock). Default 30 s.
        chgroup: this corr node's chgroup index 0..15.
        n_grid: gridder pattern side length (sanity echo for h23).
        cell_lambda: pattern per-cell λ-extent (h23 stamps pixel scale).
        pattern_id: pattern provenance hash (h23 logs mismatches).
        ix_row, ix_col: pattern filled-cell indices (sent verbatim).
        dec_deg: observing declination (display metadata).
        amp_scale: per-chgroup amplitude statistic — median |G| of the
            cal solutions for this subband (``cal_mag_p50``). With
            ``--cal-mode phase_only`` the gridded vis still carry the
            instrumental gain magnitudes; h23 divides this chgroup's
            vis by ``amp_scale**2`` (baseline gain = product of two
            antenna gains) so the 16-chgroup band sum is bandpass-
            flattened. 1.0 when no cal is loaded.
        timeout_s: per-POST HTTP timeout.
        min_cubes_seen: suppress export until the slot-0 EMA has seen
            at least this many cubes (a cold EMA is just the first
            cube — not a sky estimate). Default 64 (~8.6 s).
    """

    def __init__(
        self,
        url: str,
        *,
        interval_s: float = 30.0,
        chgroup: int,
        n_grid: int,
        cell_lambda: float,
        pattern_id: int,
        ix_row: np.ndarray,
        ix_col: np.ndarray,
        dec_deg: float,
        amp_scale: float = 1.0,
        timeout_s: float = 5.0,
        min_cubes_seen: int = 64,
    ) -> None:
        if not url:
            raise ValueError("SkySnapshotExporter: url must be non-empty")
        if interval_s <= 0:
            raise ValueError(f"interval_s={interval_s}, expected > 0")
        self.url = str(url)
        self.interval_s = float(interval_s)
        self.chgroup = int(chgroup)
        self.n_grid = int(n_grid)
        self.cell_lambda = float(cell_lambda)
        self.pattern_id = int(pattern_id)
        self.ix_row = np.ascontiguousarray(ix_row, dtype=np.uint16)
        self.ix_col = np.ascontiguousarray(ix_col, dtype=np.uint16)
        self.dec_deg = float(dec_deg)
        self.amp_scale = float(amp_scale)
        self.timeout_s = float(timeout_s)
        self.min_cubes_seen = int(min_cubes_seen)
        self.hostname = socket.gethostname()

        self.n_exported = 0
        self.n_dropped = 0
        self.n_failed = 0
        self._consecutive_failures = 0
        self._next_failure_log_at = 1                 # exponential suppression

        self._last_export_monotonic: float = 0.0      # 0 → export on first tick
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"sky-export-cg{self.chgroup}",
            daemon=True,
        )
        self._worker.start()
        LOG.info(
            "sky export: url=%s interval=%.0fs chgroup=%d n_filled=%d "
            "amp_scale=%.4g",
            self.url, self.interval_s, self.chgroup,
            self.ix_row.shape[0], self.amp_scale,
        )

    # ------------------------------------------------------------------
    # Hot-path entry point
    # ------------------------------------------------------------------

    def maybe_export(self, static_sky, *, block_n: int) -> bool:
        """Called once per block. Returns True iff a snapshot was taken.

        ``static_sky`` is the :class:`StaticSkyEMA` (or None — no-op).
        Never raises; never blocks on network.
        """
        now = time.monotonic()
        if now - self._last_export_monotonic < self.interval_s:
            return False
        if static_sky is None:
            return False
        try:
            cubes_seen = int(static_sky.cubes_seen_for(0))
            if cubes_seen < self.min_cubes_seen:
                return False
            mean = static_sky._running_mean_per_dm[0]
            if mean is None:
                return False
            self._last_export_monotonic = now
            vis = mean.detach().to("cpu", copy=True).numpy()
            meta = {
                "chgroup": self.chgroup,
                "hostname": self.hostname,
                "n_grid": self.n_grid,
                "cell_lambda": self.cell_lambda,
                "pattern_id": f"0x{self.pattern_id:016x}",
                "dec_deg": self.dec_deg,
                "amp_scale": self.amp_scale,
                "alpha": float(static_sky.alpha),
                "cubes_seen": cubes_seen,
                "block_n": int(block_n),
                "unix_ts": time.time(),
            }
            body = build_snapshot_npz(
                vis, ix_row=self.ix_row, ix_col=self.ix_col, meta=meta,
            )
        except Exception:                              # noqa: BLE001
            # Snapshot construction must never take down the RT loop.
            LOG.exception("sky export: snapshot build failed (continuing)")
            return False
        try:
            self._q.put_nowait(body)
        except queue.Full:
            self.n_dropped += 1
            LOG.warning(
                "sky export: worker behind, dropped snapshot "
                "(n_dropped=%d)", self.n_dropped,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                body = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._post(body)

    def _post(self, body: bytes) -> None:
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sky-Chgroup": str(self.chgroup),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp.read(512)
            self.n_exported += 1
            if self._consecutive_failures:
                LOG.info(
                    "sky export: recovered after %d failure(s)",
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            self._next_failure_log_at = 1
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            self.n_failed += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._next_failure_log_at:
                LOG.warning(
                    "sky export: POST %s failed (%s); %d consecutive "
                    "failure(s), next log at %d",
                    self.url, exc, self._consecutive_failures,
                    self._next_failure_log_at * 2,
                )
                self._next_failure_log_at *= 2

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2.0)

    def stats(self) -> dict[str, int]:
        return {
            "n_exported": self.n_exported,
            "n_dropped": self.n_dropped,
            "n_failed": self.n_failed,
        }


__all__ = [
    "SKY_SNAPSHOT_VERSION",
    "SkySnapshotExporter",
    "build_snapshot_npz",
    "parse_snapshot_npz",
]
