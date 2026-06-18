"""Tests for :mod:`dsart.coinc.voltage_collect` (C3 fragment collection)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from dsart.coinc.voltage_collect import (
    CorrNode,
    collect_fragments,
    collection_done,
    plan_fragments,
)


def _nodes(n: int) -> Dict[int, CorrNode]:
    return {
        g: CorrNode(chgroup=g, ssh_host=f"n{g:02d}.pro.pvt",
                    udp_ip=f"10.41.0.{g}")
        for g in range(n)
    }


def test_plan_fragments_naming(tmp_path: Path) -> None:
    plans = plan_fragments(
        event_name="260610hulw",
        corr_nodes=_nodes(3),
        staging_dir="/home/ubuntu/data/voltage_staging/",
        dest_dir=tmp_path / "voltages",
    )
    assert [p.chgroup for p in plans] == [0, 1, 2]
    p0 = plans[0]
    assert p0.remote == (
        "n00.pro.pvt:/home/ubuntu/data/voltage_staging/"
        "260610hulw_sb00_data.out"
    )
    assert p0.dest == tmp_path / "voltages" / "260610hulw_sb00_data.out"
    # zero-padded sb index
    assert plans[2].remote.endswith("260610hulw_sb02_data.out")


def test_collection_done_all_present() -> None:
    assert collection_done(n_present=16, n_total=16, min_fragments=8,
                           elapsed_s=0.0, timeout_s=100.0)


def test_collection_done_timeout_with_min() -> None:
    assert collection_done(n_present=8, n_total=16, min_fragments=8,
                           elapsed_s=101.0, timeout_s=100.0)
    # below min after timeout → not done
    assert not collection_done(n_present=7, n_total=16, min_fragments=8,
                               elapsed_s=101.0, timeout_s=100.0)
    # before timeout, incomplete → not done
    assert not collection_done(n_present=8, n_total=16, min_fragments=8,
                               elapsed_s=10.0, timeout_s=100.0)


def test_collect_fragments_uses_injected_puller(tmp_path: Path) -> None:
    plans = plan_fragments(
        event_name="ev", corr_nodes=_nodes(3),
        staging_dir="/stage", dest_dir=tmp_path,
    )
    pulled: List[Tuple[str, Path]] = []

    def fake_pull(remote: str, dest: Path) -> bool:
        pulled.append((remote, dest))
        # chgroup 1 not ready yet
        if "sb01" in remote:
            return False
        dest.write_bytes(b"data")
        return True

    present, n_present = collect_fragments(plans, pull_fn=fake_pull)
    assert n_present == 2
    assert present == {0: True, 1: False, 2: True}
    assert len(pulled) == 3


def test_collect_fragments_skips_already_present(tmp_path: Path) -> None:
    plans = plan_fragments(
        event_name="ev", corr_nodes=_nodes(2),
        staging_dir="/stage", dest_dir=tmp_path,
    )
    # Pre-stage chgroup 0's fragment.
    plans[0].dest.write_bytes(b"already-here")

    calls: List[str] = []

    def fake_pull(remote: str, dest: Path) -> bool:
        calls.append(remote)
        dest.write_bytes(b"x")
        return True

    present, n_present = collect_fragments(plans, pull_fn=fake_pull)
    assert n_present == 2
    # chgroup 0 was already present → puller only called for chgroup 1
    assert len(calls) == 1
    assert "sb01" in calls[0]
