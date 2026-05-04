"""Minimal UDP emitter for §4.3 sparse-COO value channel (M0 loopback stub)."""

from __future__ import annotations

import argparse
import socket
import struct
import time

DSART_UDP_MAGIC = 0xD5A1107E
HEADER = struct.Struct("<IHHQQHHHHHHIQBBHffI8s")


def pack_datagram(
    *,
    seq: int,
    specnum: int,
    chgroup: int,
    dm_idx: int,
    frag_idx: int,
    n_frags: int,
    n_grid: int,
    n_filled: int,
    pattern_id: int,
    bits_per_cell: int,
    t_int_factor: int,
    scale: float,
    offset: float,
    payload: bytes,
) -> bytes:
    if len(payload) > 8964:
        raise ValueError("fragment payload exceeds conservative MTU headroom")
    hdr = HEADER.pack(
        DSART_UDP_MAGIC,
        1,
        0,
        seq,
        specnum,
        chgroup,
        dm_idx,
        frag_idx,
        n_frags,
        n_grid,
        0,
        n_filled,
        pattern_id,
        bits_per_cell,
        t_int_factor,
        0,
        scale,
        offset,
        len(payload),
        b"\0" * 8,
    )
    return hdr + payload


def verify_packet(pkt: bytes, expect_chgroup: int) -> None:
    if len(pkt) < HEADER.size:
        raise ValueError(f"short packet {len(pkt)}")
    fields = HEADER.unpack(pkt[: HEADER.size])
    magic = fields[0]
    version = fields[1]
    chgroup = fields[5]
    scale = fields[16]
    offset = fields[17]
    pay_len = fields[18]
    if magic != DSART_UDP_MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != 1:
        raise ValueError(f"bad version {version}")
    if chgroup != expect_chgroup:
        raise ValueError(f"chgroup mismatch got {chgroup} expected {expect_chgroup}")
    if len(pkt) != HEADER.size + pay_len:
        raise ValueError("payload length mismatch")
    _ = scale, offset


def send_one_block(
    *,
    host: str,
    port: int,
    chgroup: int,
    sleep_s: float,
) -> bytes:
    payload = (0).to_bytes(2, "little", signed=True) * 8  # tiny synthetic COO fragment (16 B)
    pkt = pack_datagram(
        seq=1,
        specnum=2048,
        chgroup=chgroup,
        dm_idx=0,
        frag_idx=0,
        n_frags=1,
        n_grid=256,
        n_filled=8,
        pattern_id=0xCAFEBABE_DEADBEEF,
        bits_per_cell=8,
        t_int_factor=8,
        scale=1.0,
        offset=0.0,
        payload=payload,
    )
    if sleep_s > 0:
        time.sleep(sleep_s)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
        tx.sendto(pkt, (host, port))
    return pkt


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port-base", type=int, default=9000)
    ap.add_argument("--chgroup", type=int, default=0)
    ap.add_argument("--rate", default="native", choices=("native", "fast"))
    ap.add_argument("--self-test", action="store_true")
    ns = ap.parse_args(argv)

    port = ns.port_base + ns.chgroup
    pace = 134.218e-3 if ns.rate == "native" else 0.0

    if ns.self_test:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind((ns.host, port))
        rx.settimeout(3.0)
        pkt_sent = send_one_block(host=ns.host, port=port, chgroup=ns.chgroup, sleep_s=pace)
        data, _addr = rx.recvfrom(65536)
        rx.close()
        verify_packet(data, ns.chgroup)
        print(
            f"fake_corr_to_search: self-test PASS chgroup={ns.chgroup} port={port} "
            f"bytes={len(data)} (sent={len(pkt_sent)})"
        )
        return 0

    send_one_block(host=ns.host, port=port, chgroup=ns.chgroup, sleep_s=pace)
    print(f"fake_corr_to_search: sent 1 datagram to {ns.host}:{port} chgroup={ns.chgroup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
