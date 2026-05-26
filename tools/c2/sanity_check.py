#!/usr/bin/env python3
"""C2 smoke test — sends a fake C1 batch and checks the receiver picks it up.

Usage::

    python3 tools/c2/sanity_check.py --host lxd110h23 --port 11500

Optionally pass ``--mon-key /mon/c2/h23`` to also verify the
mon-points dict was updated within the timeout. Useful as the
``M7.4_BRINGUP.md`` smoke gate after starting ``dsart_c2.service``.

Exit code:
  * 0 — batch sent + accepted (and if --mon-key given, mon-points
        showed rows_in increment).
  * 1 — send / accept failed.
  * 2 — mon-points check failed.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

# Make ``dsart`` importable when this script is run from a checkout.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

from dsart.coinc import wire  # noqa: E402


def _build_batch() -> bytes:
    header = wire.build_header(
        cube_id=int(time.time()) & 0xFFFFFFFF,
        event_specnum_start=0,
        mjd_start=60781.123456789,
        sample_period_specnum=16,
        sample_period_us=1048.576,
        n_grid=256,
        n_fdm_in_cube=34,
        search_node_id=99,  # sentinel — never a real search node
        gpu_half=0,
        n_candidates=1,
    )
    row = wire.C1CandidateRow(
        snr=12.5,
        l_rad=1.5e-3,
        m_rad=-2.5e-3,
        l_pix=128,
        m_pix=128,
        dm_pc_cc=350.0,
        dm_idx_global=10,
        fine_dm_idx=0,
        event_specnum=int(time.time()),
        width_samples=4,
        kernel_id="unit:d1:b4",
        flags=0,
    )
    return wire.C1BatchEncoder.encode(header, [row])


def _read_mon_points(mon_key: str) -> dict | None:
    try:
        from dsautils.dsa_store import DsaStore  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"dsautils not importable ({exc}); skipping mon-key check")
        return None
    ds = DsaStore()
    try:
        return ds.get_dict(mon_key)
    except Exception as exc:  # noqa: BLE001
        print(f"etcd get_dict({mon_key}) failed: {exc}")
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="lxd110h23",
                   help="C2 service hostname (default: lxd110h23)")
    p.add_argument("--port", type=int, default=11500)
    p.add_argument("--mon-key", default=None,
                   help="If set, also verify rows_in incremented at this "
                        "etcd key (e.g. /mon/c2/h23).")
    p.add_argument("--mon-poll-s", type=float, default=8.0,
                   help="How long to wait for mon-points to refresh.")
    args = p.parse_args()

    blob = _build_batch()
    print(f"connecting to {args.host}:{args.port}…")
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5.0)
    except OSError as exc:
        print(f"FATAL: TCP connect failed: {exc}")
        return 1
    try:
        sock.sendall(blob)
    except OSError as exc:
        print(f"FATAL: send failed: {exc}")
        sock.close()
        return 1
    sock.close()
    print(f"sent {len(blob)} bytes (1 candidate, search_node_id=99 sentinel)")

    if args.mon_key:
        before = _read_mon_points(args.mon_key)
        if before is None:
            return 2
        before_rows = (before.get("counters", {}) or {}).get("rows_in", 0)
        deadline = time.monotonic() + args.mon_poll_s
        while time.monotonic() < deadline:
            time.sleep(1.0)
            cur = _read_mon_points(args.mon_key)
            if cur is None:
                return 2
            cur_rows = (cur.get("counters", {}) or {}).get("rows_in", 0)
            if cur_rows > before_rows:
                print(
                    f"OK: mon-points rows_in {before_rows} -> {cur_rows} "
                    f"(within {args.mon_poll_s:.0f}s)"
                )
                return 0
        print(
            f"FAIL: mon-points rows_in did not increment within "
            f"{args.mon_poll_s:.0f}s (stayed at {before_rows})"
        )
        return 2

    print("OK: batch sent. Verify on h23 with: journalctl --user -u dsart_c2 -n 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
