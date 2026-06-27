"""Voltage-fragment collection helpers for C3 (h23).

C3 pulls the per-corr staged ``<event>_sb<NN>_data.out`` files into the
candidate directory's ``Level2/voltages/``. The pull policy mirrors the
legacy T3 ``manage_copied_files`` behaviour: wait for all ``n_chgroups``
(16) fragments, or accept ``>= min_fragments`` once ``timeout_s`` has
elapsed.

The pure decision/plan logic lives here (unit-testable); the actual
``rsync`` subprocess is a thin injected runner so tests never touch the
network.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger("dsart.coinc.voltage_collect")

__all__ = [
    "CorrNode",
    "FragmentPlan",
    "plan_fragments",
    "collection_done",
    "rsync_pull",
    "collect_fragments",
]


@dataclass(frozen=True)
class CorrNode:
    chgroup: int
    ssh_host: str           # e.g. "ubuntu@lxd110h03.pro.pvt"
    udp_ip: str             # corr-net IP for the delete-sentinel broadcast


@dataclass(frozen=True)
class FragmentPlan:
    chgroup: int
    remote: str             # rsync source "host:/path/<event>_sbNN_data.out"
    dest: Path              # local Level2/voltages/<event>_sbNN_data.out


def plan_fragments(
    *,
    event_name: str,
    corr_nodes: Mapping[int, CorrNode],
    staging_dir: str,
    dest_dir: Path,
) -> List[FragmentPlan]:
    """Build the per-chgroup rsync source/dest plan for an event."""
    plans: List[FragmentPlan] = []
    dest_dir = Path(dest_dir)
    for chgroup in sorted(corr_nodes):
        node = corr_nodes[chgroup]
        sb = f"sb{int(chgroup):02d}"
        fname = f"{event_name}_{sb}_data.out"
        remote = f"{node.ssh_host}:{staging_dir.rstrip('/')}/{fname}"
        plans.append(
            FragmentPlan(chgroup=chgroup, remote=remote, dest=dest_dir / fname)
        )
    return plans


def collection_done(
    *,
    n_present: int,
    n_total: int,
    min_fragments: int,
    elapsed_s: float,
    timeout_s: float,
    no_show_grace_s: float = float("inf"),
) -> bool:
    """True when collection should stop.

    Stops when:

    * all fragments are present (``n_present >= n_total``); or
    * the timeout elapsed with at least ``min_fragments`` in hand
      (legacy dsa110-T3 MIN_CT: proceed with a partial set); or
    * **no fragment showed up at all** after ``no_show_grace_s`` —
      the "this event was never dumped" early-out. Without it, an event
      that never produced any staged voltages (e.g. voltage dumps are
      disabled, or the trigger predates the corr retention window) would
      spin the collect loop forever, because ``n_present`` stays 0 and
      never reaches ``min_fragments``. ``no_show_grace_s`` defaults to
      ``inf`` (disabled) so existing callers/tests are unaffected; C3
      passes a finite value (``collect_no_show_grace_s``).
    """
    if n_present >= n_total:
        return True
    if elapsed_s >= timeout_s and n_present >= min_fragments:
        return True
    if n_present == 0 and elapsed_s >= no_show_grace_s:
        return True
    return False


def rsync_pull(remote: str, dest: Path, *, timeout_s: float = 600.0,
               bwlimit_kbps: int = 0) -> bool:
    """Pull one fragment via rsync; return True on success.

    A missing remote file (rsync exit 23/24) is a normal "not ready yet"
    and returns False without raising.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "--partial", "--inplace"]
    if bwlimit_kbps > 0:
        cmd.append(f"--bwlimit={int(bwlimit_kbps)}")
    cmd += [remote, str(dest)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        LOG.warning("rsync %s failed: %s", " ".join(shlex.quote(c) for c in cmd), exc)
        return False
    if proc.returncode == 0 and dest.is_file():
        return True
    if proc.returncode not in (0, 23, 24):
        LOG.warning(
            "rsync %s rc=%d stderr=%s",
            remote, proc.returncode, proc.stderr.decode("utf-8", "replace")[:200],
        )
    return False


def collect_fragments(
    plans: List[FragmentPlan],
    *,
    pull_fn: Callable[[str, Path], bool] = lambda r, d: rsync_pull(r, d),
) -> Tuple[Dict[int, bool], int]:
    """Attempt one pull pass over the not-yet-present fragments.

    Returns ``(present_by_chgroup, n_present)``. A fragment already on
    disk counts as present without re-pulling.
    """
    present: Dict[int, bool] = {}
    for plan in plans:
        if plan.dest.is_file() and plan.dest.stat().st_size > 0:
            present[plan.chgroup] = True
            continue
        ok = bool(pull_fn(plan.remote, plan.dest))
        present[plan.chgroup] = ok
    n_present = sum(1 for v in present.values() if v)
    return present, n_present
