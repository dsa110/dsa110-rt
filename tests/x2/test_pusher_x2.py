"""x2 D11: Influx pusher builders for the x2 mon keys (additive)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "dashboard" / "dsart_rt_to_influx"))
import pusher  # noqa: E402

sys.path.remove(str(REPO / "tools" / "dashboard" / "dsart_rt_to_influx"))


class _Etcd:
    def get_prefix(self, prefix):
        return iter(())


class _W:
    def write(self, points):
        return len(list(points))


def _svc():
    return pusher.InfluxPusherService(etcd_client=_Etcd(), influx_writer=_W())


def test_dm_cutout_rx_corrfast_c3_builders() -> None:
    s = _svc()
    now = time.time()
    p = s._route("/mon/search_rt/9/dm/21", {
        "ts_wall_unix": now, "gpu_half": 1, "tier": "A", "opoint": "2x-x",
        "cubes_per_s": 4.97, "fanin_streams_present": 32, "ring_overrun_total": 0,
        "nested": {"x": 1}})
    assert len(p) == 1 and p[0].measurement == "search_rt_dm"
    assert p[0].tags == {"cn_id": "9", "host": "lxd110h09", "coarse_dm": "21",
                         "gpu_half": "1", "tier": "A", "opoint": "2x-x"}
    assert p[0].fields["cubes_per_s"] == 4.97 and "nested" not in p[0].fields
    p = s._route("/mon/search_rt/13/cutout/0", {
        "ts_wall_unix": now, "cutout_done_total": 3, "raw_bytes_total": 2_000_000_000})
    assert p[0].measurement == "search_rt_cutout" and p[0].tags["gpu_half"] == "0"
    p = s._route("/mon/search_rt/1/rx", {"ts_wall_unix": now, "rx_gbps": 12.1,
                                         "loss_frac": 1e-5})
    assert p[0].measurement == "search_rt_rx" and p[0].fields["rx_gbps"] == 12.1
    p = s._route("/mon/corr_rt/3/corr_fast", {
        "ts_wall_unix": now, "last_block_ms": 101.2, "block_n": 7})
    assert p[0].measurement == "corr_rt_corr_fast"
    assert p[0].tags == {"chgroup": "3", "cn_id": "6", "host": "lxd110h06"}
    p = s._route("/mon/corr_rt/3/corr_fast", {
        "ts_wall_unix": now, "last_block_ms": 101.2, "cn_id": 6, "opoint": "2x"})
    assert p[0].tags["cn_id"] == "6" and p[0].tags["opoint"] == "2x"
    p = s._route("/mon/c3/h23", {"ts_unix": now, "counters": {"kept": 4},
                                 "flag_only": True, "opoint": "1x"})
    assert p[0].measurement == "c3_service"
    assert p[0].fields["kept"] == 4 and p[0].fields["flag_only"] == 1


def test_compute_owner_tag_from_payload_only_when_present() -> None:
    now = time.time()
    base = {"ts_wall_unix": now, "c1_metering_active": 0, "n_blocks": 16}
    p1 = pusher.make_search_compute_points(
        dict(base), cn_id=2, gpu_half=1, coarse_dm_owner=pusher.COARSE_DM_OWNER)
    assert p1[0].tags["coarse_dm"] == "3" and "coarse_dms" not in p1[0].tags
    p2 = pusher.make_search_compute_points(
        dict(base, coarse_dms="3,11,19,27,35,43", opoint="2x-g"), cn_id=2,
        gpu_half=1, coarse_dm_owner=pusher.COARSE_DM_OWNER)
    assert "coarse_dm" not in p2[0].tags
    assert p2[0].tags["coarse_dms"] == "3,11,19,27,35,43"
    assert p2[0].tags["opoint"] == "2x-g"


def test_c2_extra_fields() -> None:
    p = pusher.make_c2_points({"ts_unix": time.time(), "cluster_rate": 12,
                               "cluster_rate_max": 400, "opoint": "2x-g",
                               "counters": {"events_complete": 2}}, host="h23")
    svc = [x for x in p if x.measurement == "c2_service"][0]
    assert svc.fields["cluster_rate"] == 12 and svc.fields["opoint"] == "2x-g"
    assert svc.fields["events_complete"] == 2
