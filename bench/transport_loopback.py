#!/usr/bin/env python3
"""bench/transport_loopback.py — fast-vis cube transport TX/RX loopback bench.

M3 chunk 8 acceptance + characterisation bench. Stands up a
:class:`dsart.transport.tx.TransportTx` and a
:class:`dsart.transport.rx.TransportRx` on ``127.0.0.1`` (loopback
interface), fires 100 synthetic fast-vis cubes through the chunk-4
:class:`TransportTxStage` Protocol, and verifies:

* ``n_received == n_sent`` (no loss on loopback)
* ``n_crc_fail == 0`` (codec is bit-stable end-to-end)
* ``n_seq_gaps == 0`` (per-chgroup sequence numbers strictly monotonic)
* TX→RX latency p50, p99 (loopback baseline)
* throughput vs payload-size sweep at {16k, 32k, 64k, 128k} cells

Reports:

    bench/reports/<UTC>/<run_id>/M3-transport-loopback/
        report.html             — narrative + figures
        summary.json            — machine-readable metrics
        latency_histogram.png   — TX→RX latency distribution
        throughput_vs_payload_size.png   — throughput sweep

Loopback interface notes (per plan §8 line 2071): performance numbers
from ``127.0.0.1`` are NOT representative of the real 40 GbE fabric.
This bench validates the codec, sequence accounting, CRC, dtype
round-trip, and Protocol compliance — NOT the loss budget. Real-fabric
loss budget is M4b (`bench/net_pair.py` on h01 + the second machine).

CLI:

    python -m bench.transport_loopback \\
        [--host 127.0.0.1]
        [--port 0]                 # 0 → ephemeral free port
        [--n-cubes 100]
        [--n-filled 5800]          # cells per (dm, t) tile
        [--chgroup 0]
        [--dtype cfp16|cint8]
        [--out-dir bench/reports/<UTC>/<run_id>/M3-transport-loopback/]
        [--run-id transport-loopback]
        [--no-throughput-sweep]    # skip the {16k, 32k, 64k, 128k} sweep
        [--throughput-cubes-per-size 50]
"""

from __future__ import annotations

import argparse
import datetime
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.transport import (                                            # noqa: E402
    DEFAULT_MAX_PAYLOAD_BYTES,
    DTYPE_CFP16,
    DTYPE_CINT8,
    HEADER_BYTES,
    FastVisFrame,
    TransportRx,
    TransportTx,
)


# ---------------------------------------------------------------------------
# RX worker
# ---------------------------------------------------------------------------


class _RxWorker(threading.Thread):
    """Background RX loop: receives up to ``n_target`` frames, records
    arrival timestamps + decoded :class:`FastVisFrame` objects.

    Stops on (a) n_target reached, (b) ``stop_event`` set + drained,
    or (c) ``timeout_total_s`` wall-clock cap. Per-iteration uses a
    short socket timeout (100 ms) so the thread can exit cleanly when
    the caller signals.
    """

    def __init__(
        self,
        rx: TransportRx,
        n_target: int,
        *,
        timeout_total_s: float = 30.0,
    ) -> None:
        super().__init__(daemon=True)
        self._rx = rx
        self._rx._sock.settimeout(0.1)                                   # short poll
        self._n_target = n_target
        self._timeout_total_s = timeout_total_s
        self.frames: list[FastVisFrame] = []
        self.t_arrival_ns: list[int] = []
        self.error: BaseException | None = None
        self.stop_event = threading.Event()

    def run(self) -> None:
        deadline = time.monotonic() + self._timeout_total_s
        idle_polls = 0
        try:
            while len(self.frames) < self._n_target:
                if time.monotonic() > deadline:
                    return
                if self.stop_event.is_set() and idle_polls > 5:
                    return                                                # caller said stop + nothing pending
                try:
                    frame = self._rx.receive_one()
                except OSError as exc:
                    if getattr(exc, "errno", None) == 9:
                        return                                            # socket closed; bail out
                    print(f"  RX rejected (OSError): {exc}")
                    continue
                except Exception as exc:                                  # CRC / magic
                    print(f"  RX rejected: {exc}")
                    continue
                if frame is None:
                    idle_polls += 1
                    continue
                idle_polls = 0
                self.t_arrival_ns.append(time.monotonic_ns())
                self.frames.append(frame)
        except BaseException as exc:
            self.error = exc


# ---------------------------------------------------------------------------
# Synthetic cube generator
# ---------------------------------------------------------------------------


def _synth_cube(n_filled: int, *, seed: int) -> torch.Tensor:
    """One ``(N_DM=1, n_fast_vis=1, N_filled)`` complex64 sparse-COO cube.

    Random complex values uniform in [-1, 1] (re, im). The chunk-8
    transport doesn't care about the numerical content — just that it
    round-trips bit-stably.
    """
    rng = np.random.default_rng(seed)
    re = rng.uniform(-1.0, 1.0, size=n_filled).astype(np.float32)
    im = rng.uniform(-1.0, 1.0, size=n_filled).astype(np.float32)
    cube = torch.complex(
        torch.from_numpy(re).reshape(1, 1, n_filled),
        torch.from_numpy(im).reshape(1, 1, n_filled),
    )
    return cube


# ---------------------------------------------------------------------------
# Bench scenarios
# ---------------------------------------------------------------------------


def run_no_loss_round_trip(
    *,
    host: str,
    port: int,
    chgroup: int,
    n_cubes: int,
    n_filled: int,
    dtype_code: int,
) -> dict:
    """Send ``n_cubes`` cubes in single-threaded ping-pong (send 1,
    recv 1, repeat) so latency measurement is not GIL-distorted, then
    do a separate burst run for throughput. Returns a metrics dict.

    The ping-pong configuration measures *true* one-way TX→RX latency:
    each iteration measures `t_recv - t_send_returned` where the
    receiver was already polling on the socket (300 µs poll timeout).
    """
    rx = TransportRx(host, port, recv_timeout_s=2.0)
    try:
        # Short poll timeout so receive_one returns quickly when data
        # is in the socket buffer (it always is on loopback).
        rx._sock.settimeout(0.001)

        tx = TransportTx(
            host, rx.port, chgroup=chgroup,
            dtype_code=dtype_code,
            max_payload_bytes=max(DEFAULT_MAX_PAYLOAD_BYTES, n_filled * 8),
        )
        try:
            t_burst_start_ns = time.monotonic_ns()
            latencies_us: list[float] = []
            received_seqs: list[int] = []
            for i in range(n_cubes):
                cube = _synth_cube(n_filled, seed=i)
                t_pre_send_ns = time.monotonic_ns()
                sent = tx.transmit(
                    [cube], block_n=i, rfi_warming_up=False,
                )
                assert sent == 1, f"expected 1 frame, sent {sent}"
                # Spin-poll until the frame arrives. Loopback delivers
                # synchronously so this should be a single recvfrom.
                frame = None
                while frame is None:
                    try:
                        frame = rx.receive_one()
                    except Exception as exc:
                        raise RuntimeError(f"unexpected RX error: {exc}")
                t_recv_ns = time.monotonic_ns()
                latencies_us.append((t_recv_ns - t_pre_send_ns) / 1e3)
                received_seqs.append(frame.seq)
            t_burst_end_ns = time.monotonic_ns()
        finally:
            tx.close()

        if len(received_seqs) != n_cubes:
            raise RuntimeError(
                f"RX got {len(received_seqs)} frames, expected {n_cubes}"
            )

        latencies_us = np.asarray(latencies_us)

        seqs = np.asarray(received_seqs)
        ordered = bool((np.diff(seqs) == 1).all())

        bytes_sent = int(tx.bytes_sent)
        bytes_received = int(rx.stats.bytes_received)
        elapsed_s = (t_burst_end_ns - t_burst_start_ns) / 1e9

        dtype_name = "cfp16" if dtype_code == DTYPE_CFP16 else "cint8"
        bytes_per_cell = 4 if dtype_code == DTYPE_CFP16 else 2
        original_bytes = n_filled * 8                                     # complex64 source
        wire_payload_bytes = n_filled * bytes_per_cell
        compression_ratio = original_bytes / max(wire_payload_bytes, 1)

        return {
            "n_cubes": n_cubes,
            "n_filled": n_filled,
            "n_sent": int(tx.n_sent),
            "n_received": int(rx.stats.n_received),
            "n_crc_fail": int(rx.stats.n_crc_fail),
            "n_magic_fail": int(rx.stats.n_magic_fail),
            "n_seq_gaps": int(rx.stats.n_seq_gaps),
            "n_out_of_order": int(rx.stats.n_out_of_order),
            "monotonic_seq": bool(ordered),
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "elapsed_s": elapsed_s,
            "throughput_MB_per_s": (bytes_sent / 1e6) / max(elapsed_s, 1e-9),
            "p50_latency_us": float(np.percentile(latencies_us, 50)),
            "p99_latency_us": float(np.percentile(latencies_us, 99)),
            "max_latency_us": float(latencies_us.max()),
            "min_latency_us": float(latencies_us.min()),
            "latencies_us": latencies_us.tolist(),
            "dtype_name": dtype_name,
            "dtype_compression_ratio": compression_ratio,
            "wire_payload_bytes_per_frame": wire_payload_bytes + HEADER_BYTES,
        }
    finally:
        rx.close()


_MAX_UDP_PAYLOAD = 65507                                                  # IPv4 max


def run_throughput_sweep(
    *,
    host: str,
    chgroup: int,
    payload_sizes_bytes: list[int],
    cubes_per_size: int,
    dtype_code: int,
) -> list[dict]:
    """For each target payload size, send ``cubes_per_size`` cubes and
    measure aggregate throughput. Each scenario uses a fresh TX/RX pair
    on a free port. Sizes that would push wire-bytes above the
    65 507 B IPv4 UDP datagram cap are flagged as ``over_mtu=True``;
    real-fabric fragmentation is M4a's job (plan §4.3 ``n_frags``).
    """
    bytes_per_cell = 4 if dtype_code == DTYPE_CFP16 else 2
    out: list[dict] = []
    for target_bytes in payload_sizes_bytes:
        n_filled = max(1, target_bytes // bytes_per_cell)
        wire_bytes = n_filled * bytes_per_cell + HEADER_BYTES
        over_mtu = wire_bytes > _MAX_UDP_PAYLOAD
        if over_mtu:
            out.append({
                "target_payload_bytes": target_bytes,
                "n_filled": n_filled,
                "wire_bytes_per_frame": wire_bytes,
                "over_mtu": True,
                "note": (
                    f"wire={wire_bytes} > UDP max {_MAX_UDP_PAYLOAD}; "
                    "production §4.3 fragments via n_frags (M4a)"
                ),
                "n_sent": 0,
                "n_received": 0,
                "n_lost": cubes_per_size,
                "elapsed_s": 0.0,
                "throughput_MB_per_s": 0.0,
                "throughput_Gbps": 0.0,
            })
            continue

        rx = TransportRx(host, 0, recv_timeout_s=2.0)
        try:
            rx_worker = _RxWorker(
                rx, n_target=cubes_per_size, timeout_total_s=30.0,
            )
            rx_worker.start()
            time.sleep(0.05)

            tx = TransportTx(
                host, rx.port, chgroup=chgroup,
                dtype_code=dtype_code,
                max_payload_bytes=max(
                    DEFAULT_MAX_PAYLOAD_BYTES, target_bytes + HEADER_BYTES,
                ),
            )
            try:
                t0 = time.monotonic_ns()
                for i in range(cubes_per_size):
                    cube = _synth_cube(n_filled, seed=10_000 + i)
                    tx.transmit([cube], block_n=i, rfi_warming_up=False)
                t1 = time.monotonic_ns()
            finally:
                tx.close()
            rx_worker.stop_event.set()
            rx_worker.join(timeout=10.0)
            elapsed_s = (t1 - t0) / 1e9
            bytes_sent = int(tx.bytes_sent)
            n_lost = cubes_per_size - len(rx_worker.frames)
            out.append({
                "target_payload_bytes": target_bytes,
                "n_filled": n_filled,
                "wire_bytes_per_frame": wire_bytes,
                "over_mtu": False,
                "n_sent": int(tx.n_sent),
                "n_received": int(rx.stats.n_received),
                "n_lost": n_lost,
                "elapsed_s": elapsed_s,
                "throughput_MB_per_s": (bytes_sent / 1e6) / max(elapsed_s, 1e-9),
                "throughput_Gbps": (bytes_sent * 8 / 1e9)
                / max(elapsed_s, 1e-9),
            })
        finally:
            rx.close()
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _plot_latency(latencies_us: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(latencies_us)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=40, edgecolor="black")
    ax.axvline(np.percentile(arr, 50), color="green",
               linestyle="--", label=f"p50 = {np.percentile(arr, 50):.1f} µs")
    ax.axvline(np.percentile(arr, 99), color="red",
               linestyle="--", label=f"p99 = {np.percentile(arr, 99):.1f} µs")
    ax.set_xlabel("TX → RX latency (µs)")
    ax.set_ylabel("count")
    ax.set_title("Loopback fast-vis frame latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_throughput(sweep: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_mtu = [r for r in sweep if not r.get("over_mtu")]
    over_mtu = [r for r in sweep if r.get("over_mtu")]
    sizes_in = [r["target_payload_bytes"] / 1024.0 for r in in_mtu]
    rates_in = [r["throughput_Gbps"] for r in in_mtu]
    sizes_over = [r["target_payload_bytes"] / 1024.0 for r in over_mtu]

    fig, ax = plt.subplots(figsize=(7, 4))
    if sizes_in:
        ax.plot(sizes_in, rates_in, marker="o", linewidth=1.6,
                label="loopback (single-frame)")
    for s in sizes_over:
        ax.axvline(s, color="red", linestyle=":", alpha=0.5)
    if sizes_over:
        ax.text(
            0.97, 0.95,
            "red dashed = over UDP MTU\n(needs M4a fragmentation)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="red",
        )
    ax.set_xlabel("payload size per frame (KiB)")
    ax.set_ylabel("loopback throughput (Gbps)")
    ax.set_title("Throughput vs payload size (loopback, single-thread)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    if sizes_in:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _write_html(
    out_dir: Path,
    main_metrics: dict,
    sweep: list[dict],
) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in main_metrics.items()
        if k not in ("latencies_us",)
    )
    def _row(r: dict) -> str:
        flag = " (over MTU; needs fragmentation)" if r.get("over_mtu") else ""
        return (
            f"<tr><td>{r['target_payload_bytes']:>7}{flag}</td>"
            f"<td>{r['n_filled']:>6}</td>"
            f"<td>{r['wire_bytes_per_frame']:>8}</td>"
            f"<td>{r['n_sent']:>5}</td>"
            f"<td>{r['n_received']:>5}</td>"
            f"<td>{r['n_lost']:>5}</td>"
            f"<td>{r['throughput_Gbps']:.3f}</td></tr>"
        )
    sweep_rows = "".join(_row(r) for r in sweep)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>M3 chunk 8 — Transport loopback</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 980px; margin: 2em auto; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
.banner {{ padding: 12px; background: #e9f7ef; border: 1px solid #a8d5b9;
          border-radius: 4px; margin-bottom: 1em; }}
img {{ max-width: 720px; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style></head><body>
<h1>M3 chunk 8 — Transport loopback bench</h1>
<div class="banner">
  <strong>Scope:</strong> loopback fast-vis frame TX/RX validation. Tests
  the codec + sequence accounting + CRC + Protocol compliance end-to-end
  on <code>127.0.0.1</code>. Per plan §8 line 2071, loopback throughput
  numbers are <em>not</em> representative of the real 40 GbE fabric — the
  real-fabric loss budget is M4b's <code>bench/net_pair.py</code>.
</div>

<h2>Round-trip metrics ({main_metrics['n_cubes']} cubes, {main_metrics['n_filled']} filled cells, dtype={main_metrics['dtype_name']})</h2>
<table>{rows}</table>

<h2>Latency histogram</h2>
<img src="latency_histogram.png" />

<h2>Throughput sweep</h2>
<table>
<tr><th>target B</th><th>n_filled</th><th>wire B</th><th>n_sent</th><th>n_recv</th><th>n_lost</th><th>Gbps</th></tr>
{sweep_rows}
</table>
<img src="throughput_vs_payload_size.png" />

<h2>Pass criteria (chunk 8 DoD)</h2>
<ul>
<li>n_received == n_sent: {"PASS" if main_metrics['n_received'] == main_metrics['n_sent'] else "FAIL"}</li>
<li>n_crc_fail == 0: {"PASS" if main_metrics['n_crc_fail'] == 0 else "FAIL"}</li>
<li>n_seq_gaps == 0: {"PASS" if main_metrics['n_seq_gaps'] == 0 else "FAIL"}</li>
<li>monotonic_seq: {"PASS" if main_metrics['monotonic_seq'] else "FAIL"}</li>
</ul>

</body></html>
"""
    (out_dir / "report.html").write_text(html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0,
                   help="0 → kernel picks ephemeral free port")
    p.add_argument("--n-cubes", type=int, default=100)
    p.add_argument("--n-filled", type=int, default=5800,
                   help="cells per (dm, t) tile (~5800 ≈ N_grid=256 single-side fill)")
    p.add_argument("--chgroup", type=int, default=0)
    p.add_argument("--dtype", choices=("cfp16", "cint8"), default="cfp16")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="transport-loopback")
    p.add_argument("--no-throughput-sweep", action="store_true")
    p.add_argument("--throughput-cubes-per-size", type=int, default=50)
    args = p.parse_args(argv)

    if args.out_dir is None:
        utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = (
            REPO_ROOT / "bench" / "reports" / utc / args.run_id
            / "M3-transport-loopback"
        )
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype_code = DTYPE_CFP16 if args.dtype == "cfp16" else DTYPE_CINT8

    print(
        f"== M3 chunk 8 transport loopback bench =="
        f"\n  host={args.host} chgroup={args.chgroup} dtype={args.dtype}"
        f"\n  n_cubes={args.n_cubes} n_filled={args.n_filled}"
        f"\n  out_dir={out_dir}"
    )

    main_metrics = run_no_loss_round_trip(
        host=args.host, port=args.port, chgroup=args.chgroup,
        n_cubes=args.n_cubes, n_filled=args.n_filled,
        dtype_code=dtype_code,
    )

    print(
        f"\n  round-trip: sent={main_metrics['n_sent']} "
        f"received={main_metrics['n_received']} "
        f"crc_fail={main_metrics['n_crc_fail']} "
        f"seq_gaps={main_metrics['n_seq_gaps']} "
        f"p50={main_metrics['p50_latency_us']:.1f}µs "
        f"p99={main_metrics['p99_latency_us']:.1f}µs"
    )

    if args.no_throughput_sweep:
        sweep: list[dict] = []
    else:
        sweep = run_throughput_sweep(
            host=args.host, chgroup=args.chgroup,
            payload_sizes_bytes=[16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024],
            cubes_per_size=args.throughput_cubes_per_size,
            dtype_code=dtype_code,
        )
        for s in sweep:
            print(
                f"  sweep: target={s['target_payload_bytes']:>6}B "
                f"wire={s['wire_bytes_per_frame']:>6}B "
                f"throughput={s['throughput_Gbps']:.3f} Gbps "
                f"loss={s['n_lost']}"
            )

    _plot_latency(main_metrics["latencies_us"], out_dir / "latency_histogram.png")
    if sweep:
        _plot_throughput(sweep, out_dir / "throughput_vs_payload_size.png")

    summary = {
        "main": main_metrics,
        "throughput_sweep": sweep,
        "host": args.host,
        "chgroup": args.chgroup,
        "dtype": args.dtype,
        "n_cubes": args.n_cubes,
        "n_filled": args.n_filled,
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _write_html(out_dir, main_metrics, sweep)

    print(f"\n  wrote: {out_dir}/report.html")
    print(f"  wrote: {out_dir}/summary.json")

    pass_criteria = (
        main_metrics["n_received"] == main_metrics["n_sent"]
        and main_metrics["n_crc_fail"] == 0
        and main_metrics["n_seq_gaps"] == 0
        and main_metrics["monotonic_seq"]
    )
    return 0 if pass_criteria else 1


if __name__ == "__main__":
    sys.exit(main())
