"""Production inputs shared by corr_fast 2x, search_rx v4 and the owner engine.

One module, so the three processes cannot diverge (docs/x2/SPEC.md §3, §6, §9, §10):

* value map (SPEC §3): :func:`load_antpos` reads antenna E/N and the core-baseline
  mask from the production cal blob (``antennas.out``, through corr_fast's own
  ``load_antpos_from_cal_blob``) or from an antpos npz (``antpos_e`` / ``antpos_n`` /
  ``station``, e.g. ``tests/fixtures/x2/dsa110_antpos.npz``);
  :func:`canonical_dec_deg` is the DEC as corr_fast carries it (``math.radians`` when
  it parses ``--obs-dec-deg``, ``math.degrees`` at the value map), so a start-verb DEC
  on a 0.25 deg bin edge quantises to the same ``pattern_id`` in every process;
  :func:`value_map` builds with ``antpos_u = None`` and tee45 (``build_value_map_for_opoint``);
* ring v4 (SPEC §6): :func:`ring4_spec` is one search node's ``RingV4Spec`` with the
  format's row-rule scales; :func:`node_ring4` adds the per-stream pattern ids search_rx
  registers (the engine refuses a ring whose ids differ from its own value map);
* plan check (SPEC §10): :func:`check_plan_id`;
* time epoch (SPEC §9): frames carry absolute native indices (2 x SNAP specnum, corr_fast
  ``--x2-epoch capture``). :func:`capture_native0` is the native index corr_fast gives the
  first sample after an arm at specnum ``u``; :func:`read_arm_record` reads the capture arm
  record ``dsart_rt`` stamps at ``utc_start`` (``/mon/snap/1/armed_mjd``: ``armed_mjd`` =
  MJD of the arm specnum, ``arm_seq`` = that specnum); :func:`mjd_at_native_0` turns it into
  the MJD of native sample 0. 1x labels candidates from the same record (its search
  specnums start at the arm, so ``mjd_at_specnum_0 == armed_mjd``).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

__all__ = ["load_antpos", "canonical_dec_deg", "value_map", "value_map_for", "ring4_spec",
           "node_ring4", "node_index_of_cn", "check_plan_id", "load_plan", "capture_native0",
           "ArmRecord", "ARM_RECORD_KEY", "read_arm_record", "mjd_at_native_0", "NATIVE_S"]

LOG = logging.getLogger("dsart.x2.prod_inputs")

#: native sample (s)
NATIVE_S = 32.768e-6
#: native samples per SNAP specnum (SPEC §9: native index = 2 x specnum)
NATIVE_PER_SPECNUM = 2
#: corr_fast 2x frames need t0_native on this grid (``x2_native0_from_specnum``)
EPOCH_GRID_NATIVE = 32
#: the capture arm record ``dsart_rt._verb_utc_start`` writes
ARM_RECORD_KEY = "/mon/snap/1/armed_mjd"


# ---------------------------------------------------------------------------
# value map (SPEC §3)
# ---------------------------------------------------------------------------


def load_antpos(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(antpos_e, antpos_n, is_core_baseline_mask)`` from the cal blob corr_fast
    calibrates with (``--apply-cal``, production ``antennas.out``) or an antpos npz."""
    p = Path(path)
    if p.suffix == ".npz":
        from ..grid.sparsity_pattern import core_baseline_mask_from_station_numbers
        z = np.load(p)
        return (np.asarray(z["antpos_e"]), np.asarray(z["antpos_n"]),
                core_baseline_mask_from_station_numbers(z["station"]))
    from ..services.corr_fast_integration import load_antpos_from_cal_blob
    return load_antpos_from_cal_blob(p)


def canonical_dec_deg(obs_dec_deg) -> float:
    """The DEC of a start verb (``--obs-dec-deg CUSTOMDEC``) as corr_fast hands it to
    the value map: ``math.degrees(math.radians(x))``."""
    return math.degrees(math.radians(float(obs_dec_deg)))


def value_map(op, antpos_e, antpos_n, is_core_baseline_mask, *, dec_deg: float):
    """SPEC §3 production value map. ``dec_deg`` is the canonical DEC
    (corr_fast: ``math.degrees(cfg.obs_dec_rad)``; others: :func:`canonical_dec_deg`)."""
    from ..grid.baseline_grid import build_value_map_for_opoint
    return build_value_map_for_opoint(op, antpos_e, antpos_n, antpos_u=None,
                                      is_core_baseline_mask=is_core_baseline_mask,
                                      dec_deg=float(dec_deg))


def value_map_for(op, antpos_path, obs_dec_deg):
    """Value map from a cal blob / antpos npz path and the start verb's DEC."""
    e, n, mask = load_antpos(antpos_path)
    return value_map(op, e, n, mask, dec_deg=canonical_dec_deg(obs_dec_deg))


# ---------------------------------------------------------------------------
# plan (SPEC §2, §10)
# ---------------------------------------------------------------------------


def check_plan_id(plan, plan_id: Optional[str]) -> None:
    """SPEC §10: refuse a tiered plan that is not the set's ``${plan_id}`` (hex, with or
    without ``0x``). ``None`` / empty = no check (development runs)."""
    if not plan_id:
        return
    want = int(str(plan_id), 16)
    got = int(plan.plan_id)
    if got != want:
        raise ValueError(f"tiered DM plan plan_id 0x{got:016x} != --plan-id 0x{want:016x}: "
                         "the plan file is not the one this op-point set was rendered for")


def load_plan(path, op, plan_id: Optional[str] = None):
    """The v3 plan at ``path`` (``verify=True``), checked against ``plan_id`` and the
    op-point's ``n_sub``."""
    from .plan import TieredDmPlan
    plan = TieredDmPlan.from_npz(path, verify=True)
    check_plan_id(plan, plan_id)
    if int(plan.n_sub) != int(op.n_sub):
        raise ValueError(f"plan {path} has n_sub {plan.n_sub}, op-point {op.opoint_id} needs "
                         f"{op.n_sub}")
    return plan


# ---------------------------------------------------------------------------
# ring v4 (SPEC §6)
# ---------------------------------------------------------------------------


def node_index_of_cn(cn: int) -> int:
    """Search node index (``SEARCH_NODES`` order: n01 n02 n09 n13) of node id ``cn``."""
    from .plan import SEARCH_NODES
    name = f"n{int(cn):02d}"
    if name not in SEARCH_NODES:
        raise ValueError(f"cn {cn} is not a search node ({', '.join(SEARCH_NODES)})")
    return SEARCH_NODES.index(name)


def ring4_spec(plan, vm, op, *, node_index: int, retention_s: float = 20.0):
    """Ring v4 spec of search node ``node_index``: the DMs of halves 2i, 2i+1, the
    format's row-rule scales, retention >= ``retention_s``."""
    from ..transport.recv_ring import RingV4Spec
    from .quant import RowRule
    rule = RowRule.for_format(op.bits_main, op.n_sub)
    return RingV4Spec.for_node(plan, vm, op, node_index=int(node_index), rule=rule,
                               retention_s=float(retention_s))


def node_ring4(op, plan, *, node_index: int, antpos_path, obs_dec_deg,
               retention_s: float = 20.0):
    """``(spec, value_map, {stream: pattern_id})`` of search node ``node_index`` --
    what search_rx v4 creates and registers at start."""
    vm = value_map_for(op, antpos_path, obs_dec_deg)
    spec = ring4_spec(plan, vm, op, node_index=node_index, retention_s=retention_s)
    return spec, vm, {i: int(p) for i, p in enumerate(vm.stream_pattern_ids)}


# ---------------------------------------------------------------------------
# time epoch (SPEC §9)
# ---------------------------------------------------------------------------


def capture_native0(arm_specnum: int) -> int:
    """Native index corr_fast 2x gives the first sample of the first fada page after an
    arm at specnum ``arm_specnum``: 2 x specnum, rounded down to the 32-native grid
    (``corr_fast_integration.x2_native0_from_specnum(u) + 4096``)."""
    n = NATIVE_PER_SPECNUM * int(arm_specnum)
    return n - n % EPOCH_GRID_NATIVE


@dataclass(frozen=True)
class ArmRecord:
    """The capture arm record (``dsart_rt._verb_utc_start``)."""

    armed_mjd: float              # MJD of the arm specnum (PPS-derived or the verb's wall latch)
    arm_seq: int                  # the UTC_START specnum
    source: str = ""              # "pps_epoch" | "wall_latch"

    @property
    def mjd_at_native_0(self) -> float:
        return mjd_at_native_0(self.armed_mjd, self.arm_seq)


def mjd_at_native_0(armed_mjd: float, arm_seq: int) -> float:
    """MJD of native sample 0 when native sample :func:`capture_native0` (``arm_seq``)
    is at ``armed_mjd``."""
    return float(armed_mjd) - capture_native0(arm_seq) * NATIVE_S / 86400.0


def read_arm_record(store=None, *, tries: int = 3, retry_s: float = 1.0,
                    now=time.time, max_age_days: float = 30.0
                    ) -> Tuple[Optional[ArmRecord], str]:
    """``(record, "")`` or ``(None, reason)``. The record must carry a plausible
    ``armed_mjd`` (within ``max_age_days``, not in the future) and an integer
    ``arm_seq``. etcd read errors are retried ``tries`` times."""
    doc, why = None, ""
    for i in range(max(1, int(tries))):
        try:
            if store is None:
                from dsautils.dsa_store import DsaStore
                store = DsaStore()
            doc = store.get_dict(ARM_RECORD_KEY)
            why = ""
            break
        except Exception as exc:                                     # noqa: BLE001
            why = f"etcd read of {ARM_RECORD_KEY} failed: {exc}"
            LOG.warning("%s (try %d/%d)", why, i + 1, tries)
            if i + 1 < tries:
                time.sleep(retry_s)
    if why:
        return None, why
    if not isinstance(doc, dict) or doc.get("armed_mjd") is None:
        return None, f"{ARM_RECORD_KEY} is missing or has no armed_mjd (got {doc!r})"
    try:
        armed = float(doc["armed_mjd"])
    except (TypeError, ValueError):
        return None, f"{ARM_RECORD_KEY}.armed_mjd {doc.get('armed_mjd')!r} is not a number"
    seq = doc.get("arm_seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        return None, (f"{ARM_RECORD_KEY} has no integer arm_seq (got {seq!r}): the absolute "
                      "native epoch needs the arm specnum")
    now_mjd = 40587.0 + now() / 86400.0
    if not (now_mjd - max_age_days < armed <= now_mjd + 2.0 / 86400.0):
        return None, f"{ARM_RECORD_KEY}.armed_mjd {armed:.6f} implausible (now {now_mjd:.6f})"
    return ArmRecord(armed_mjd=armed, arm_seq=int(seq), source=str(doc.get("source") or "")), ""
