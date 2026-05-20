"""dsart-rt mon-publisher sidecar for the SNAP capture binaries.

Runs as a sibling routine of ``cap_a_real`` / ``cap_b_real`` under
``dsart_rt`` (see ``configs/dsart_pipeline_rt.yaml``). Polls each
capture binary's POSIX-shm segment at the mon-publisher cadence
(default 2 s) and pushes the counters into etcd under
``/mon/corr_rt/<n>/capture/<udp_port>``.

This is the production-readiness bridge that turns the legacy
syslog-only ``CAPSTATS`` line into structured, dashboard-consumable
mon-keys.

CLI:
    python -m dsart.services.capture_control \\
        --udp-ports 4011,4012 \\
        --cn-id 6 \\
        --mon-cadence-s 2.0 \\
        --log-level INFO

Etcd verb relay:
    *Not needed here.* The orchestrator's ``_verb_utc_start`` /
    ``_verb_utc_stop`` already deliver ``UTC_START-<seq>`` /
    ``UTC_STOP-<seq>`` UDP messages to ports 11223 and 11224 (which
    the capture binaries' ``control_thread`` listens on). The sidecar
    only observes the arming via the shm ``arm_state`` field; it
    does not relay verbs.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import struct
import sys
import time
from typing import Any, Optional

from dsart.capture.mon_shm import (
    ArmState,
    CaptureMonSnapshot,
    MonShm,
    MonShmAbiMismatch,
    MonShmNotPresent,
)

LOG = logging.getLogger("dsart.capture_control")

# Heartbeat staleness threshold. Above this, the sidecar publishes
# `degraded=true` even if all other counters look healthy -- means the
# C binary's stats_thread hasn't ticked in over a second, which
# either implies a hang or a SIGSTOP somewhere in the process group.
DEFAULT_STALENESS_MS = 1000.0


def _mon_key_for(cn_id: int, udp_port: int) -> str:
    """Etcd key for ``/mon/corr_rt/<cn_id>/capture/<udp_port>``.

    Keyed disjoint from the orchestrator's ``/mon/corr_rt/<cn_id>``
    so the mon-dict shape doesn't change for legacy consumers; new
    web-app dashboards just consume the ``capture/<port>`` sub-tree.
    """

    return f"/mon/corr_rt/{cn_id}/capture/{udp_port}"


def _snap_to_mon_dict(
    snap: CaptureMonSnapshot,
    *,
    staleness_threshold_ms: float,
) -> dict[str, Any]:
    """Convert a shm snapshot to the etcd payload schema.

    Schema is stable across calls; consumers (dashboard, grafana,
    operator CLIs) key on these names. Adding fields is fine; renaming
    or removing requires a schema-version bump.
    """

    degraded = snap.is_stale or snap.age_ms > staleness_threshold_ms
    return {
        "schema_version": 1,
        "udp_port": snap.udp_port,
        "control_port": snap.control_port,
        "pid": snap.pid,
        "arm_state": snap.arm_state.name,
        "arm_state_int": int(snap.arm_state),
        "utc_start_specnum": snap.utc_start_specnum,
        "utc_stop_specnum": snap.utc_stop_specnum,
        "last_seq_no": snap.last_seq_no,
        "socket_rcvbuf_bytes": snap.socket_rcvbuf_bytes,
        "rate_gbps": snap.rate_gbps,
        "rate_drop_mb_s": snap.rate_drop_mb_s,
        "rate_kernel_drop_pps": snap.rate_kernel_drop_pps,
        "n_recv_packets": snap.n_recv_packets,
        "n_recv_bytes": snap.n_recv_bytes,
        "n_dropped_payload": snap.n_dropped_payload,
        "n_dropped_kernel": snap.n_dropped_kernel,
        "n_seq_skipped": snap.n_seq_skipped,
        "n_too_late": snap.n_too_late,
        "n_wrong_size": snap.n_wrong_size,
        "n_recv_errors": snap.n_recv_errors,
        "n_block_writes": snap.n_block_writes,
        "startup_utc_ns": snap.startup_utc_ns,
        "last_update_utc_ns": snap.last_update_utc_ns,
        "age_ms": round(snap.age_ms, 2),
        "degraded": degraded,
    }


def _mon_dict_unavailable(udp_port: int, reason: str) -> dict[str, Any]:
    """Placeholder mon-dict when the shm is missing.

    Lets the dashboard see the lane is dark instead of just absent;
    operator can immediately tell whether the binary hasn't started
    yet vs. crashed mid-run.
    """

    return {
        "schema_version": 1,
        "udp_port": udp_port,
        "arm_state": "UNAVAILABLE",
        "arm_state_int": -1,
        "degraded": True,
        "shm_status": "missing",
        "reason": reason,
    }


class _StoreWrapper:
    """Thin wrapper around DsaStore so the unit tests can mock it.

    We import `DsaStore` lazily inside the constructor so this module
    imports cleanly in test envs that don't have ``dsautils`` on the
    path (e.g. the casa38 conda env on the developer's box).
    """

    def __init__(self, mock: Optional[Any] = None) -> None:
        if mock is not None:
            self._store = mock
        else:
            from dsautils.dsa_store import DsaStore  # noqa: WPS433
            self._store = DsaStore()

    def put_dict(self, key: str, value: dict[str, Any]) -> None:
        self._store.put_dict(key, value)


class CaptureControlService:
    """Mon-publisher sidecar -- one instance per ``dsart_rt`` node."""

    def __init__(
        self,
        *,
        udp_ports: tuple[int, ...],
        cn_id: int,
        store: Any,
        mon_cadence_s: float = 2.0,
        staleness_threshold_ms: float = DEFAULT_STALENESS_MS,
    ):
        self.udp_ports = udp_ports
        self.cn_id = cn_id
        self.store = store
        self.mon_cadence_s = mon_cadence_s
        self.staleness_threshold_ms = staleness_threshold_ms
        self._shms: dict[int, MonShm] = {}
        self._stop = False

    # ---- lifecycle ----------------------------------------------------

    def install_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        LOG.info("received signal %d, shutting down", signum)
        self._stop = True

    def run(self, *, max_iters: Optional[int] = None) -> int:
        self.install_signals()
        LOG.info(
            "capture_control up: cn=%d udp_ports=%s mon_cadence_s=%.1f host=%s",
            self.cn_id, list(self.udp_ports), self.mon_cadence_s,
            socket.gethostname(),
        )

        # Initial attach attempt (non-fatal: capture binaries may not
        # have opened their shms yet -- we'll retry inside the loop).
        for port in self.udp_ports:
            self._try_attach(port)

        iters = 0
        while not self._stop:
            self._tick()
            iters += 1
            if max_iters is not None and iters >= max_iters:
                LOG.info("max_iters %d reached, exiting", max_iters)
                break
            # The wait honours stop set during the wait window.
            for _ in range(max(1, int(self.mon_cadence_s * 10))):
                if self._stop:
                    break
                time.sleep(0.1)

        self._on_shutdown()
        return 0

    def _on_shutdown(self) -> None:
        for port, mon in list(self._shms.items()):
            try:
                mon.close()
            except Exception:  # noqa: BLE001
                LOG.exception("error closing shm for port %d", port)
        self._shms.clear()

    # ---- per-tick logic ----------------------------------------------

    def _try_attach(self, port: int) -> None:
        if port in self._shms:
            return
        try:
            self._shms[port] = MonShm.open(port)
            LOG.info("attached to shm for port %d", port)
        except MonShmNotPresent:
            LOG.debug("shm for port %d not present yet", port)
        except MonShmAbiMismatch as exc:
            LOG.error("ABI mismatch on port %d: %s", port, exc)

    def _detach(self, port: int) -> None:
        mon = self._shms.pop(port, None)
        if mon is not None:
            try:
                mon.close()
            except Exception:  # noqa: BLE001
                pass

    def _tick(self) -> None:
        """One mon-publisher tick: snapshot every attached shm; publish."""

        for port in self.udp_ports:
            # Always try to (re-)attach if we don't have it -- the
            # capture binary may have been restarted by the orchestrator.
            if port not in self._shms:
                self._try_attach(port)

            if port not in self._shms:
                payload = _mon_dict_unavailable(port, "shm not present")
                self._publish(port, payload)
                continue

            mon = self._shms[port]
            try:
                snap = mon.snapshot()
            except (OSError, struct.error) as exc:
                LOG.warning(
                    "snapshot failure on port %d: %s; reattaching",
                    port, exc,
                )
                self._detach(port)
                payload = _mon_dict_unavailable(port, f"snapshot error: {exc}")
                self._publish(port, payload)
                continue

            payload = _snap_to_mon_dict(
                snap, staleness_threshold_ms=self.staleness_threshold_ms
            )
            if payload["degraded"]:
                LOG.warning(
                    "port=%d DEGRADED arm=%s age_ms=%.0f n_recv=%d "
                    "kernel_drops=%d/s last_seq=%d",
                    port,
                    payload["arm_state"],
                    payload["age_ms"],
                    payload["n_recv_packets"],
                    payload["rate_kernel_drop_pps"],
                    payload["last_seq_no"],
                )
            else:
                LOG.info(
                    "port=%d arm=%s rate=%.3f Gb/s n_recv=%d "
                    "block_writes=%d kernel_drops=%d/s last_seq=%d",
                    port,
                    payload["arm_state"],
                    payload["rate_gbps"],
                    payload["n_recv_packets"],
                    payload["n_block_writes"],
                    payload["rate_kernel_drop_pps"],
                    payload["last_seq_no"],
                )
            self._publish(port, payload)

    def _publish(self, port: int, payload: dict[str, Any]) -> None:
        key = _mon_key_for(self.cn_id, port)
        try:
            self.store.put_dict(key, payload)
        except Exception:  # noqa: BLE001
            LOG.exception("put_dict %s failed", key)


# ---- CLI ------------------------------------------------------------------


def _parse_ports(s: str) -> tuple[int, ...]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0 or v >= 65536:
            raise argparse.ArgumentTypeError(f"invalid port: {tok}")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("must list at least one port")
    return tuple(out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--udp-ports",
        type=_parse_ports,
        default=(4011, 4012),
        help="Comma-separated list of UDP data ports to monitor "
             "(default: 4011,4012).",
    )
    p.add_argument(
        "--cn-id",
        type=int,
        required=True,
        help="corr-node ID (3..22) used for the etcd mon path.",
    )
    p.add_argument(
        "--mon-cadence-s",
        type=float,
        default=2.0,
        help="seconds between mon-publish ticks (default: 2.0).",
    )
    p.add_argument(
        "--staleness-threshold-ms",
        type=float,
        default=DEFAULT_STALENESS_MS,
        help="declare degraded if the C binary hasn't ticked in this many "
             "ms (default: 1000).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    store = _StoreWrapper()
    svc = CaptureControlService(
        udp_ports=args.udp_ports,
        cn_id=args.cn_id,
        store=store,
        mon_cadence_s=args.mon_cadence_s,
        staleness_threshold_ms=args.staleness_threshold_ms,
    )
    return svc.run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
