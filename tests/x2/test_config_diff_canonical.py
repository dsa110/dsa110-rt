"""config_diff.canonical_sha256: a config hashes the same before and after an etcd (JSON) round trip."""
import json

from dsart.x2.config_diff import canonical_sha256


def test_int_keys_hash_like_their_json_round_trip():
    # c3.corr_nodes in configs/opoints/2x/dsart_search_rt.yaml: int keys 0..15; etcd stores strings,
    # and sort_keys orders 0,1,..,9,10 numerically but "0","1","10",.. lexicographically
    cfg = {"c3": {"corr_nodes": {i: {"udp_ip": f"10.41.0.{i}"} for i in range(16)}}, "a": [1, 2.5, "x"]}
    back = json.loads(json.dumps(cfg))
    assert list(back["c3"]["corr_nodes"])[:3] == ["0", "1", "2"]
    assert canonical_sha256(cfg) == canonical_sha256(back)


def test_content_changes_still_change_the_hash():
    a = {"c3": {"corr_nodes": {0: "x", 1: "y"}}}
    assert canonical_sha256(a) != canonical_sha256({"c3": {"corr_nodes": {0: "x", 1: "z"}}})
    assert canonical_sha256(a) != canonical_sha256({"c3": {"corr_nodes": {0: "x"}}})
