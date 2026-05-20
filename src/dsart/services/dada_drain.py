"""dsart-rt drain-of-last-resort for a PSRDADA ring buffer.

Reads and discards every page from a PSRDADA buffer indefinitely (until
SIGTERM). Exists because the legacy ``dada_dbnull`` tool has three
blocking bugs for our usage pattern:

* Default `-X 64` MB makes it exit after a single 64 MiB transfer, so the
  dsart_rt routine supervisor has to respawn it in a tight loop. The
  respawn cadence (~2 s) is much slower than the producer's write rate
  (~99 ms/block, ~27 MiB/block for bada), so the ring fills 8-10 blocks
  per spawn cycle and the producer back-pressures. This was measured to
  cost ~16 ms/block of latency on corr_fast (158 ms -> 141.7 ms p50
  steady-state on the M7.1 5-min and 30-min soaks, n06, May 17 2026) by
  introducing bursty fada arrivals through the back-pressure chain
  (bada->corr_slow->fada->merge->junkdb).

* ``-S`` (loop forever on next XFER) only loops if the producer writes
  ``OBS_XFER`` in the header. ``dada_junkdb`` does NOT, so ``-S``
  immediately hits "header with no OBS_XFER, assuming END of XFERS" and
  exits after the same 64 MiB transfer.

* Setting ``-X`` to a large value (e.g. 4_000_000 = "4 TB") wraps because
  the C source computes ``transfer_bytes = transfer_size_mbytes *
  byte_base`` in ``int`` (32-bit signed) before assigning to ``uint64_t``.
  We measured 4000000 * 1000000 -> 1_385_447_424 = 1.29 GiB, which only
  lasts ~6 s of bada drain. Max safe value is ~2147 (≈ 2 GiB / 10 s).

This script avoids all three problems: it is a single Python process
that calls ``Reader.getNextPage(); reader.markCleared()`` in a loop
until SIGTERM, with no transfer-size cap and no spawn churn.

CLI surface:
    python -m dsart.services.dada_drain --key bada
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Any

LOG = logging.getLogger("dsart.dada_drain")


def _install_signals(state: dict[str, Any]) -> None:
    def _handler(signum: int, _frame: Any) -> None:  # noqa: ANN001
        LOG.info("received signal %d, requesting stop", signum)
        state["stop"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run(key: int, log_every: int = 1024) -> int:
    """Drain ``key`` until SIGTERM. Returns 0 on clean shutdown.

    ``log_every`` controls how often we print throughput-style stats so
    the operator can see the routine is alive without overwhelming the
    log. Default 1024 blocks ≈ 137 s at native bada cadence (134.218 ms),
    so a handful of log lines per minute.
    """
    from psrdada import Reader  # imported lazily — same pattern as
                                # services/corr_slow_compute.py so the
                                # module imports cleanly in test envs.

    state: dict[str, Any] = {"stop": False}
    _install_signals(state)

    LOG.info("connecting key=0x%x", key)
    reader = Reader(key)
    LOG.info("attached; draining…")

    n_drained = 0
    try:
        while not state["stop"]:
            try:
                page = reader.getNextPage()
            except StopIteration:
                LOG.info("reader StopIteration (EOD); exiting cleanly")
                break
            if reader.isEndOfData:
                LOG.info("EOD flag set; draining final page and exiting")
                reader.markCleared()
                break
            # We intentionally do nothing with ``page`` — this is a drain
            # of last resort. The producer's contract is that it must
            # treat the data as released once ``markCleared`` returns.
            _ = page  # silence linter; the cast forces PSRDADA cython to
                      # honour the page reservation.
            reader.markCleared()
            n_drained += 1
            if n_drained % log_every == 0:
                LOG.info("drained n=%d", n_drained)
    finally:
        # Reader doesn't expose a clean detach; rely on Python finalizer.
        LOG.info("shutdown: n_drained=%d", n_drained)

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Persistent PSRDADA drain.")
    ap.add_argument(
        "--key",
        required=True,
        help="PSRDADA hex key, e.g. 'bada' (case-insensitive, no 0x).",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=1024,
        help="Print throughput every N pages drained (default 1024).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        key = int(args.key, 16)
    except ValueError:
        LOG.error("invalid --key %r (must be hex)", args.key)
        return 2

    return run(key, log_every=args.log_every)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
