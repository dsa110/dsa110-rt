"""x2 real-time health + DM ownership view (docs/x2/downstream_inventory.md §4, D11).

Read-only: one pass over the x2 mon keys in etcd plus the active op-point's
tiered plan, judged against the plan's real-time gates:

* corr_fast block time p99 <= 134.218 ms (mean <= 85 %);
* search GPU duty <= 0.75 per half (aim 0.6);
* per-DM fan-in complete, ring lag bounded, no ring overruns;
* corner-turn frame loss <= 1e-4 per search node;
* cutout service healthy (no failures piling up).

Missing keys are reported as ``unknown`` (the publishers land with WS-S /
WS-T / WS-C), never as healthy. Nothing here writes to etcd.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Tuple

SEARCH_HALVES: Tuple[Tuple[int, int], ...] = tuple(
    (s, g) for s in (1, 2, 9, 13) for g in (0, 1))
CORR_CHGROUPS: Tuple[int, ...] = tuple(range(16))

BLOCK_BUDGET_MS = 134.217728
GATE_MEAN_FRAC = 0.85
DUTY_GATE = 0.75
DUTY_AIM = 0.6
RX_LOSS_GATE = 1e-4


@functools.lru_cache(maxsize=4)
def _plan(path: str) -> Any:
    from dsart.x2.plan import load_any_plan
    return load_any_plan(path)


def ownership(plan_path: Optional[str]) -> List[Dict[str, Any]]:
    """Per half: owned coarse DMs, their DM span and fine counts."""
    if not plan_path:
        return []
    try:
        plan = _plan(str(plan_path))
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"cannot load plan {plan_path}: {exc!r}"}]
    if getattr(plan, "owner_half", None) is None:
        return []
    out = []
    for h, (sid, g) in enumerate(SEARCH_HALVES):
        dms = [int(d) for d in plan.dms_of_half(h)]
        rows = []
        for d in dms:
            sl = plan.fines_of(d)
            rows.append({
                "coarse": d, "tier": str(plan.tier_name[int(plan.coarse_tier[d])]),
                "dm_lo": round(float(plan.fine_dm[sl.start]), 1),
                "dm_hi": round(float(plan.fine_dm[sl.stop - 1]), 1),
                "n_fdm": int(plan.n_fdm[d]),
            })
        out.append({"half": h, "sid": sid, "gpu_half": g, "coarse": rows,
                    "n_fine": int(sum(r["n_fdm"] for r in rows))})
    return out


def _state(ok: Optional[bool]) -> str:
    return "unknown" if ok is None else ("ok" if ok else "FAIL")


def gather(store: Any, record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """One snapshot of the x2 gates. ``store`` has ``get_dict``."""
    def get(key: str) -> Dict[str, Any]:
        try:
            d = store.get_dict(key)
        except Exception:  # noqa: BLE001
            return {}
        return dict(d) if isinstance(d, dict) else {}

    halves = []
    for sid, g in SEARCH_HALVES:
        c = get(f"/mon/search_rt/{sid}/compute/{g}")
        cut = get(f"/mon/search_rt/{sid}/cutout/{g}")
        duty = c.get("gpu_duty")
        halves.append({
            "half": f"s{sid}g{g}", "opoint": c.get("opoint"),
            "coarse_dms": c.get("coarse_dms"),
            "gpu_duty": duty,
            "duty_gate": _state(None if not isinstance(duty, (int, float))
                                else duty <= DUTY_GATE),
            "cubes_per_s": c.get("cubes_per_s"),
            "cubes_per_s_nominal": c.get("cubes_per_s_nominal"),
            "c1_metering_active": c.get("c1_metering_active"),
            "cutout_failed_total": cut.get("cutout_failed_total"),
            "cutout_done_total": cut.get("cutout_done_total"),
            "cutout_last_latency_ms": cut.get("cutout_last_latency_ms"),
        })
    nodes = []
    for sid in (1, 2, 9, 13):
        rx = get(f"/mon/search_rt/{sid}/rx")
        loss = rx.get("loss_frac")
        nodes.append({"node": f"n{sid:02d}", "rx_gbps": rx.get("rx_gbps"),
                      "loss_frac": loss,
                      "loss_gate": _state(None if not isinstance(loss, (int, float))
                                          else loss <= RX_LOSS_GATE)})
    corr = []
    for cg in CORR_CHGROUPS:
        cf = get(f"/mon/corr_rt/{cg}/corr_fast")
        p99 = cf.get("block_ms_p99")
        p50 = cf.get("block_ms_p50")
        corr.append({
            "chgroup": cg, "cn_id": cf.get("cn_id"), "block_ms_p50": p50,
            "block_ms_p99": p99, "last_block_ms": cf.get("last_block_ms"),
            "block_gate": _state(
                None if not isinstance(p99, (int, float)) else (
                    p99 <= BLOCK_BUDGET_MS and (
                        not isinstance(p50, (int, float))
                        or p50 <= GATE_MEAN_FRAC * BLOCK_BUDGET_MS))),
            "tx_gbps": cf.get("tx_gbps"),
        })
    gates = [h["duty_gate"] for h in halves] + [n["loss_gate"] for n in nodes] \
        + [c["block_gate"] for c in corr]
    overall = ("FAIL" if "FAIL" in gates else
               "unknown" if "unknown" in gates else "ok")
    plan_path = (record or {}).get("plan_path") if record else None
    return {
        "opoint": (record or {}).get("opoint") or "1x",
        "selection": (record or {}).get("selection"),
        "plan_id": (record or {}).get("plan_id"),
        "overall": overall,
        "halves": halves, "search_nodes": nodes, "corr": corr,
        "ownership": ownership(plan_path),
        "gates": {"block_budget_ms": BLOCK_BUDGET_MS, "mean_frac": GATE_MEAN_FRAC,
                  "duty_gate": DUTY_GATE, "duty_aim": DUTY_AIM,
                  "rx_loss_gate": RX_LOSS_GATE},
    }
