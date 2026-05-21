"""RFI monitor exporter sidecar (M7.6).

Lives next to ``corr_fast_integration`` as a sibling routine spawned
by ``dsart_rt``. Reads the per-window RFI shm published by
``corr_fast_integration`` (see :mod:`dsart.services.rfi_mon_shm`)
and does two things:

1. **etcd publisher**: pushes a compact scalar summary of the latest
   16-cube window to ``/mon/corr_rt/<cn>/rfi`` every
   ``--etcd-cadence-s`` (default 2 s). Contains the three metrics
   the M7.6 design called out:

     * total flag fraction of all cells,
     * fraction of channels flagged by bandpass-outlier (averaged
       across antennas),
     * fraction of antennas flagged (group-outlier),

   plus per-pol breakdowns and per-detector sub-fractions (Option C
   from the M7.6 plan).

2. **HTTP API**: lightweight stdlib ``http.server`` listening on
   ``--http-port`` (default 5780) that serves three endpoints
   consumed by the h23 dashboard:

     * ``GET /api/health``      → liveness + freshness JSON
     * ``GET /api/meta``        → segment dimensions / cadence JSON
     * ``GET /api/latest``      → JSON record with arrays as
                                   base64-encoded raw bytes
     * ``GET /api/recent?n=N``  → JSON list of up to N most-recent
                                   records (oldest first)

The HTTP transport binds ``0.0.0.0`` by default; sites that want a
private monitor port should set ``--http-bind 10.42.0.x`` (br1
mgmt) or use a firewall rule. No auth — same trust model as the
existing capture mon-shm sidecar.

Same conda env, same `dsart_rt` lifecycle as the capture sidecar
(see :mod:`dsart.services.capture_control` for the pattern). The
two are independent: capture mon-shm and RFI mon-shm are separate
segments with separate ABIs.

CLI::

    python -m dsart.services.rfi_monitor_export \\
        --cn-id 6 \\
        --etcd-cadence-s 2.0 \\
        --http-port 5780 \\
        --log-level INFO
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np

from dsart.services.rfi_mon_shm import (
    RFIMonRecord,
    RFIMonShmAbiMismatch,
    RFIMonShmNotPresent,
    RFIMonShmReader,
    shm_path,
)

LOG = logging.getLogger("dsart.rfi_monitor_export")


DEFAULT_ETCD_CADENCE_S: float = 2.0
DEFAULT_HTTP_PORT: int = 5780
# How stale the latest window can be before /api/health and the
# etcd payload flip ``degraded=true``. At the production cadence
# (one window every ~2.15 s) anything > 10 s implies corr_fast is
# either stuck, dead, or in cold-start.
DEFAULT_STALENESS_S: float = 10.0
SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# JSON encoding of an RFIMonRecord
# ---------------------------------------------------------------------------


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Return a JSON-friendly representation of a numpy array.

    Shape + dtype are explicit; bytes are base64-encoded raw little-
    endian little-endian bytes (matches the shm wire format).
    """
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def record_to_json_obj(
    rec: RFIMonRecord, *, include_arrays: bool = True,
) -> dict[str, Any]:
    """Convert an :class:`RFIMonRecord` to a JSON-encodable dict.

    Arrays are encoded as base64-of-bytes by :func:`_encode_array` so
    reconstructing them on h23 is a single ``np.frombuffer`` call.
    """
    obj: dict[str, Any] = {
        "seq": int(rec.seq),
        "publish_utc_ns": int(rec.publish_utc_ns),
        "block_n_start": int(rec.block_n_start),
        "block_n_end": int(rec.block_n_end),
        "n_cubes": int(rec.n_cubes),
        "n_cubes_warmup": int(rec.n_cubes_warmup),
        "scalars": {
            k: [float(v[0]), float(v[1]), float(v[2])]
            for k, v in rec.scalars.items()
        },
    }
    if include_arrays:
        obj["s1_full_mean"] = _encode_array(rec.s1_full_mean)
        obj["mask_count_final"] = _encode_array(rec.mask_count_final)
        obj["mask_count_sk"] = _encode_array(rec.mask_count_sk)
        obj["mask_count_bp"] = _encode_array(rec.mask_count_bp)
        obj["mask_count_grp"] = _encode_array(rec.mask_count_grp)
        obj["mask_count_sumthr"] = _encode_array(rec.mask_count_sumthr)
        obj["mask_count_fa"] = _encode_array(rec.mask_count_fa)
    return obj


def decode_array(d: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`_encode_array` — consumer-side helper for
    h23. Re-exported here so the dashboard can import it without
    duplicating the spec."""
    raw = base64.b64decode(d["data_b64"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])


# ---------------------------------------------------------------------------
# Etcd payload builder
# ---------------------------------------------------------------------------


def _mon_dict_unavailable(
    cn_id: int, *, reason: str,
) -> dict[str, Any]:
    """Etcd payload when no RFI shm or no records yet."""
    return {
        "schema_version": SCHEMA_VERSION,
        "cn_id": int(cn_id),
        "degraded": True,
        "shm_status": "missing_or_empty",
        "reason": reason,
        "time_unix": time.time(),
    }


def _mon_dict_from_record(
    rec: RFIMonRecord, *, cn_id: int, staleness_s: float,
) -> dict[str, Any]:
    """Etcd payload for the latest window.

    Schema (per Option C of the M7.6 design):

        schema_version, cn_id, time_unix, degraded, age_s,
        seq, block_n_{start,end}, n_cubes, n_cubes_warmup,

        # the three M7.6 headline metrics (per-pol + both):
        total_flag_fraction         {pol0, pol1, both}
        bandpass_channel_fraction   {pol0, pol1, both}
        ant_fraction_flagged        {pol0, pol1, both}

        # per-detector bonus breakdown:
        frac_sk                     {pol0, pol1, both}
        frac_bp                     {pol0, pol1, both}
        frac_grp                    {pol0, pol1, both}
        frac_sumthr                 {pol0, pol1, both}
        frac_fa                     {pol0, pol1, both}
    """
    publish_unix = rec.publish_utc_ns / 1e9
    age_s = max(0.0, time.time() - publish_unix)
    degraded = age_s > staleness_s

    def _triplet(name: str) -> dict[str, float]:
        v = rec.scalars[name]
        return {"pol0": float(v[0]), "pol1": float(v[1]),
                "both": float(v[2])}

    return {
        "schema_version": SCHEMA_VERSION,
        "cn_id": int(cn_id),
        "time_unix": time.time(),
        "publish_unix": publish_unix,
        "age_s": round(age_s, 3),
        "degraded": degraded,
        "seq": int(rec.seq),
        "block_n_start": int(rec.block_n_start),
        "block_n_end": int(rec.block_n_end),
        "n_cubes": int(rec.n_cubes),
        "n_cubes_warmup": int(rec.n_cubes_warmup),
        "total_flag_fraction": _triplet("total_flag_fraction"),
        "bandpass_channel_fraction": _triplet("bandpass_channel_fraction"),
        "ant_fraction_flagged": _triplet("ant_fraction_flagged"),
        "frac_sk": _triplet("frac_sk"),
        "frac_bp": _triplet("frac_bp"),
        "frac_grp": _triplet("frac_grp"),
        "frac_sumthr": _triplet("frac_sumthr"),
        "frac_fa": _triplet("frac_fa"),
    }


# ---------------------------------------------------------------------------
# Etcd store wrapper (mockable)
# ---------------------------------------------------------------------------


class _StoreWrapper:
    """Thin wrapper around DsaStore so the unit tests can mock it.

    Mirrors :class:`dsart.services.capture_control._StoreWrapper`.
    """

    def __init__(self, mock: Optional[Any] = None) -> None:
        if mock is not None:
            self._store = mock
        else:
            from dsautils.dsa_store import DsaStore  # noqa: WPS433
            self._store = DsaStore()

    def put_dict(self, key: str, value: dict[str, Any]) -> None:
        self._store.put_dict(key, value)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _RFIExportHandler(BaseHTTPRequestHandler):
    """Stdlib request handler bound to a shared
    :class:`RFIMonitorExportService` via ``self.server.service``."""

    # ThreadingHTTPServer adds the service field. Cast for clarity.
    server: "_RFIExportServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Re-route to our logger so journald shows it.
        LOG.debug("%s - - %s", self.client_address[0],
                  format % args)

    def _send_json(
        self, payload: dict[str, Any] | list[Any], *, status: int = 200,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        path = url.path
        qs = parse_qs(url.query)

        svc = self.server.service

        try:
            if path == "/api/health":
                self._send_json(svc.snapshot_health())
                return
            if path == "/api/meta":
                self._send_json(svc.snapshot_meta())
                return
            if path == "/api/latest":
                rec = svc.read_latest_safe()
                if rec is None:
                    self._send_json({"available": False}, status=503)
                    return
                self._send_json({
                    "available": True,
                    "schema_version": SCHEMA_VERSION,
                    "record": record_to_json_obj(rec, include_arrays=True),
                })
                return
            if path == "/api/recent":
                try:
                    n = int(qs.get("n", ["8"])[0])
                except ValueError:
                    self._send_error_json(400, "bad ?n=")
                    return
                n = max(1, min(n, 256))             # bound to avoid DoS
                recs = svc.read_recent_safe(n)
                self._send_json({
                    "available": True,
                    "schema_version": SCHEMA_VERSION,
                    "count": len(recs),
                    "records": [
                        record_to_json_obj(r, include_arrays=True)
                        for r in recs
                    ],
                })
                return
            self._send_error_json(404, f"unknown endpoint {path}")
        except Exception as e:                        # pragma: no cover
            LOG.exception("HTTP handler error on %s", path)
            try:
                self._send_error_json(500, f"server error: {e!r}")
            except Exception:
                pass


class _RFIExportServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a back-pointer to the exporter service."""

    daemon_threads = True

    def __init__(
        self, addr: tuple[str, int], handler_cls: type,
        *, service: "RFIMonitorExportService",
    ) -> None:
        super().__init__(addr, handler_cls)
        self.service = service


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RFIMonitorExportService:
    """Mon-publisher + HTTP exporter for one corr node.

    Lifecycle:
        ``run()`` blocks until SIGTERM/SIGINT or ``max_iters`` etcd
        ticks. HTTP server runs in a daemon thread.

    Concurrency:
        Single :class:`RFIMonShmReader` shared between the etcd
        publisher loop and HTTP handlers. ``read_latest`` /
        ``read_recent`` are seqlock-safe (multiple concurrent
        readers OK; producer is the corr_fast process).
    """

    def __init__(
        self,
        *,
        cn_id: int,
        store: Any,
        etcd_cadence_s: float = DEFAULT_ETCD_CADENCE_S,
        staleness_s: float = DEFAULT_STALENESS_S,
        http_bind: str = "0.0.0.0",
        http_port: int = DEFAULT_HTTP_PORT,
        reader_factory: Optional[Any] = None,
    ) -> None:
        self.cn_id = int(cn_id)
        self.store = store
        self.etcd_cadence_s = float(etcd_cadence_s)
        self.staleness_s = float(staleness_s)
        self.http_bind = http_bind
        self.http_port = int(http_port)
        self._reader_factory = reader_factory or (lambda: RFIMonShmReader(cn_id))

        self._reader: RFIMonShmReader | None = None
        self._stop = threading.Event()
        self._http_server: _RFIExportServer | None = None
        self._http_thread: threading.Thread | None = None
        self._started_utc_ns = time.time_ns()

    # ------------------------------------------------------------------
    # Public mon-key helper
    # ------------------------------------------------------------------

    @property
    def mon_key(self) -> str:
        return f"/mon/corr_rt/{self.cn_id}/rfi"

    # ------------------------------------------------------------------
    # Signals + lifecycle
    # ------------------------------------------------------------------

    def install_signals(self) -> None:
        # Only the main thread can install signal handlers; threading
        # tests / unit tests that spin up a service from a worker
        # thread will skip the handlers and stop via _stop.set().
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        LOG.info("received signal %d, shutting down", signum)
        self._stop.set()

    def run(self, *, max_iters: Optional[int] = None) -> int:
        self.install_signals()
        LOG.info(
            "rfi_monitor_export up: cn=%d mon_key=%s http=%s:%d "
            "etcd_cadence_s=%.1f host=%s",
            self.cn_id, self.mon_key, self.http_bind, self.http_port,
            self.etcd_cadence_s, socket.gethostname(),
        )
        # Initial attach (non-fatal: the producer may not have created
        # the shm yet; we retry inside the loop).
        self._try_attach()

        self._start_http()

        iters = 0
        try:
            while not self._stop.is_set():
                self._etcd_tick()
                iters += 1
                if max_iters is not None and iters >= max_iters:
                    LOG.info("max_iters %d reached, exiting", max_iters)
                    break
                if self._stop.wait(self.etcd_cadence_s):
                    break
        finally:
            self._shutdown_http()
            if self._reader is not None:
                try:
                    self._reader.close()
                except Exception:
                    pass

        LOG.info("rfi_monitor_export exiting cleanly (iters=%d)", iters)
        return 0

    # ------------------------------------------------------------------
    # Per-tick logic
    # ------------------------------------------------------------------

    def _try_attach(self) -> None:
        """Attach (or re-attach) to the producer shm. Idempotent;
        no-op if already attached and the ABI is still valid."""
        if self._reader is not None:
            return
        try:
            self._reader = self._reader_factory()
            LOG.info(
                "attached to %s: n_slots=%d n_ants=%d n_chan_ds=%d "
                "n_pol=%d window_size=%d freq_downsample=%d",
                shm_path(self.cn_id),
                self._reader.n_slots, self._reader.n_ants,
                self._reader.n_chan_ds, self._reader.n_pol,
                self._reader.window_size, self._reader.freq_downsample,
            )
        except RFIMonShmNotPresent:
            LOG.debug("rfi shm not present (yet); will retry")
        except RFIMonShmAbiMismatch as e:
            LOG.error("rfi shm ABI mismatch: %s", e)

    def _etcd_tick(self) -> None:
        self._try_attach()
        payload: dict[str, Any]
        if self._reader is None:
            payload = _mon_dict_unavailable(
                self.cn_id, reason="shm_not_present",
            )
        else:
            try:
                rec = self._reader.read_latest()
            except Exception:
                LOG.exception("read_latest failed")
                rec = None
            if rec is None:
                payload = _mon_dict_unavailable(
                    self.cn_id, reason="no_records_yet",
                )
            else:
                payload = _mon_dict_from_record(
                    rec, cn_id=self.cn_id,
                    staleness_s=self.staleness_s,
                )
        try:
            self.store.put_dict(self.mon_key, payload)
        except Exception:
            LOG.exception("etcd put_dict %s failed", self.mon_key)

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _start_http(self) -> None:
        self._http_server = _RFIExportServer(
            (self.http_bind, self.http_port),
            _RFIExportHandler,
            service=self,
        )
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="rfi_monitor_http",
            daemon=True,
        )
        self._http_thread.start()
        LOG.info("HTTP server listening on %s:%d",
                 self.http_bind, self.http_port)

    def _shutdown_http(self) -> None:
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                LOG.exception("HTTP shutdown failed (non-fatal)")
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # HTTP API helpers (called by the request handler)
    # ------------------------------------------------------------------

    def read_latest_safe(self) -> RFIMonRecord | None:
        self._try_attach()
        if self._reader is None:
            return None
        try:
            return self._reader.read_latest()
        except Exception:
            LOG.exception("HTTP read_latest failed")
            return None

    def read_recent_safe(self, n: int) -> list[RFIMonRecord]:
        self._try_attach()
        if self._reader is None:
            return []
        try:
            return self._reader.read_recent(n)
        except Exception:
            LOG.exception("HTTP read_recent failed")
            return []

    def snapshot_meta(self) -> dict[str, Any]:
        self._try_attach()
        if self._reader is None:
            return {
                "available": False,
                "cn_id": self.cn_id,
                "reason": "shm_not_present",
            }
        return {
            "available": True,
            "schema_version": SCHEMA_VERSION,
            "cn_id": self.cn_id,
            "n_slots": self._reader.n_slots,
            "n_ants": self._reader.n_ants,
            "n_chan_ds": self._reader.n_chan_ds,
            "n_pol": self._reader.n_pol,
            "window_size": self._reader.window_size,
            "freq_downsample": self._reader.freq_downsample,
            "startup_utc_ns": self._reader.startup_utc_ns,
            "host": socket.gethostname(),
        }

    def snapshot_health(self) -> dict[str, Any]:
        self._try_attach()
        if self._reader is None:
            return {
                "ok": False,
                "cn_id": self.cn_id,
                "reason": "shm_not_present",
                "uptime_s": (time.time_ns() - self._started_utc_ns) / 1e9,
            }
        try:
            seq = self._reader.read_publish_seq()
            rec = self._reader.read_latest()
        except Exception as e:
            return {
                "ok": False,
                "cn_id": self.cn_id,
                "reason": f"shm read error: {e!r}",
                "uptime_s": (time.time_ns() - self._started_utc_ns) / 1e9,
            }
        if rec is None:
            return {
                "ok": False,
                "cn_id": self.cn_id,
                "reason": "no_records_yet",
                "publish_seq": int(seq),
                "uptime_s": (time.time_ns() - self._started_utc_ns) / 1e9,
            }
        age = max(0.0, time.time() - rec.publish_utc_ns / 1e9)
        return {
            "ok": age <= self.staleness_s,
            "cn_id": self.cn_id,
            "publish_seq": int(seq),
            "latest_age_s": round(age, 3),
            "staleness_threshold_s": self.staleness_s,
            "uptime_s": (time.time_ns() - self._started_utc_ns) / 1e9,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cn-id", type=int, required=True,
                   help="corr node ID; also picks the shm "
                        "name /dev/shm/dsart-rfi-window-<cn-id>")
    p.add_argument("--etcd-cadence-s", type=float,
                   default=DEFAULT_ETCD_CADENCE_S,
                   help=f"etcd publish cadence (default: "
                        f"{DEFAULT_ETCD_CADENCE_S:g})")
    p.add_argument("--staleness-s", type=float,
                   default=DEFAULT_STALENESS_S,
                   help="max age before /api/health and etcd payload "
                        f"flip degraded=true (default: "
                        f"{DEFAULT_STALENESS_S:g})")
    p.add_argument("--http-bind", default="0.0.0.0",
                   help="HTTP bind address (default: 0.0.0.0)")
    p.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT,
                   help=f"HTTP port (default: {DEFAULT_HTTP_PORT})")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    store = _StoreWrapper()
    service = RFIMonitorExportService(
        cn_id=args.cn_id,
        store=store,
        etcd_cadence_s=args.etcd_cadence_s,
        staleness_s=args.staleness_s,
        http_bind=args.http_bind,
        http_port=args.http_port,
    )
    return service.run()


if __name__ == "__main__":
    sys.exit(main())
