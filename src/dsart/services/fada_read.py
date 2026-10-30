"""Robust page read of a psrdada ``Reader`` for the corr services (corr_fast 1x / 2x and corr_slow).

psrdada-python raises ``PSRDadaError`` ("NULL page read") whenever ``ipcbuf_get_next_read``
returns NULL -- including a blocking ``semop`` interrupted by any handled signal (EINTR; semop is
never restarted, and psrdada's ``ipc_semop`` does not print the errno: ``ipcutil.c:67`` tests
``flag | IPC_NOWAIT``, always false; ``&`` was meant). In the G1 full chain (2026-10-29) two
cold-started corr_fast died 0.2 s into their first blocking read with "ipcbuf_get_next_read:
error decrement FULL". No torch / triton import here: corr_slow keeps its light start-up.
"""
from __future__ import annotations

import logging
import time
from typing import Any

LOG = logging.getLogger("dsart.services.fada_read")

#: consecutive failed fada page reads tolerated (not at end-of-data) before giving up
PAGE_READ_RETRIES = 8


def next_fada_page(reader, state: "dict[str, Any] | None" = None, *,
                   retries: int = PAGE_READ_RETRIES, backoff_s: float = 0.05,
                   log: logging.Logger | None = None):
    """``reader.getNextPage()`` that survives a failed read. A failed read leaves the reader
    holding no page, so it is retried -- with a WARNING per retry -- unless a stop was
    requested (``state["stop"]``, set by the service's SIGTERM / SIGINT handler) or the ring is
    at end-of-data (refreshed by ``markCleared()``, which for a reader holding no page only reads
    ``ipcbuf_eod`` and resets the ring there); fatal (re-raised) after ``retries`` consecutive
    failures. A successful read is returned unchanged (the normal path is the bare call).
    Returns the page, or None (stop / end-of-data)."""
    from psrdada.exceptions import PSRDadaError  # noqa: PLC0415
    lg = log or LOG
    n_fail = 0
    while True:
        try:
            return reader.getNextPage()
        except StopIteration:
            return None
        except PSRDadaError as exc:
            n_fail += 1
            if state is not None and state.get("stop"):
                lg.info("fada read interrupted (%s) by a stop request", exc)
                return None
            try:
                reader.markCleared()
            except Exception:  # noqa: BLE001
                lg.exception("fada reader markCleared after a failed read")
            if getattr(reader, "isEndOfData", False):
                lg.info("fada read failed at end-of-data (%s): EOD", exc)
                return None
            if n_fail > int(retries):
                lg.error("fada read failed %d times in a row, not at end-of-data: %s", n_fail, exc)
                raise
            lg.warning("fada getNextPage failed (%s), not at end-of-data: retry %d/%d",
                       exc, n_fail, int(retries))
            time.sleep(backoff_s)
