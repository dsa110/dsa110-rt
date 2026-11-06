"""tools/x2/snap_capture_check.py: header decode, pcap reader, sync offsets, run summary."""
import importlib.util
import struct
import sys
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "tools" / "x2" / "snap_capture_check.py"
_spec = importlib.util.spec_from_file_location("snap_capture_check", _P)
scc = importlib.util.module_from_spec(_spec)
sys.modules["snap_capture_check"] = scc
_spec.loader.exec_module(scc)


def _hdr(seq: int, ch: int, ant: int) -> bytes:
    # inverse of the capture binary's bit shuffle: 35-bit seq | 13-bit ch | 16-bit ant, big-endian
    return ((seq << 29) | (ch << 16) | ant).to_bytes(8, "big")


def test_decode_header_matches_capture_bit_layout():
    for seq, ch, ant in ((0, 0, 0), (1, 2, 3), (2**35 - 1, 2**13 - 1, 2**16 - 1), (123456789, 4321, 95)):
        assert scc.decode_header(_hdr(seq, ch, ant)) == (seq, ch, ant)


def _pcap(tmp_path, pkts):
    p = tmp_path / "x.pcap"
    with open(p, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, seq, ant in pkts:
            eth = b"\x00" * 12 + b"\x08\x00"
            ip = bytes([0x45]) + b"\x00" * 19
            udp = b"\x00" * 8
            body = eth + ip + udp + _hdr(seq, 0, ant)
            f.write(struct.pack("<IIII", int(t), int(round((t % 1) * 1e6)), len(body), len(body) + 4600))
            f.write(body)
    return str(p)


def test_pcap_reader_and_sync_offsets_flag_a_desynced_snap(tmp_path):
    rate = 30517.6                           # seq units per second (32.768 us)
    pkts = []
    for i in range(4000):
        t = 1000.0 + i * 2e-5
        for slot in range(16):
            lag = 30518 if slot == 7 else 0  # slot 7 armed one PPS second late
            pkts.append((t, int((t - 1000.0) * rate) + 5_000_000 - lag, 3 * slot))
    rows = scc.read_pcap_headers(_pcap(tmp_path, pkts))
    assert len(rows) == len(pkts) and rows[0][2] == 0 and rows[7][2] == 21
    s = scc.sync_offsets(rows)
    assert s["n_slots"] == 16
    assert abs(s["slots"]["7"]["offset_us"] + 1e6) < 2e3          # ~ -1 s
    assert all(abs(v["offset_us"]) < 100 for k, v in s["slots"].items() if k != "7")


def test_summarise_uses_armed_snapshots_only():
    def mon(rcv, drp, blk):
        return {"n_recv_packets": rcv, "n_dropped_payload": drp, "n_dropped_kernel": 0, "n_seq_skipped": 0,
                "n_too_late": 0, "n_block_writes": blk}
    snaps = [{"t": 0.0, "mon": {"4011": mon(10, 0, 0), "4012": mon(10, 0, 0)}},
             {"t": 10.0, "mon": {"4011": mon(1000, 5, 7), "4012": mon(1000, 0, 7)}},
             {"t": 20.0, "mon": {"4011": mon(2441406 + 1000, 105, 82), "4012": mon(2001000, 0, 82)}}]
    s = scc.summarise({"snaps": snaps, "sync": {}})
    assert s["4011"]["n_recv_packets"] == 2441406 and s["4011"]["n_dropped_payload"] == 100
    assert abs(s["4011"]["drop_frac"] - 100 / 2441506) < 1e-12
    assert abs(s["4011"]["recv_vs_expected"] - 1.0) < 1e-3
    assert s["4012"]["drop_frac"] == 0.0
