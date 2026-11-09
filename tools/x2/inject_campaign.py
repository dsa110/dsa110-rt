#!/usr/bin/env python
"""2x injection campaign (G3): measure K per (DM, width) bucket, then fire at chosen S/N and score it.

Runs on h23 against the dashboard (the same endpoints the inject bot uses; stop dsart_inject_bot first --
it fires hourly and its K repair rewrites K):

  calibrate  POST /control/inject_calibrate per bucket (K = S/N x sqrt(w) / F from the first match,
             fluence ladder x1/2/4 when undetected), then the SETTLED match doc is re-read
             (/mon/dsart/inject/matches/<inj_id>) and its best S/N recorded.
  grid       POST /control/inject with target_snr per bucket that has K; after the last shot the match
             docs are collected: observed / target S/N, matched DM / l / m / width / kernel / half.

Shots never overlap: >= --spacing-s between injections, >= --probe-gap-s after a probe, consecutive
shots differ in DM (interleaved order) and position (rotating l/m inside +-0.015 rad), so the C2 holdoff
(30 s per class), the matcher's wall-clock fallback (+28 s) and the 30 s noise time constant never couple
two shots. Every request and result is one JSON line in --out.

    python tools/x2/inject_campaign.py --phase both --out /dataz/dsa110/operations/inject/x2_campaign.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.parse
import urllib.request

DASH = "http://127.0.0.1:5778"
DMS = (150.0, 250.0, 500.0, 1000.0, 1250.0, 1500.0, 1700.0, 2300.0, 2900.0)
WIDTHS = (4, 8, 16, 32, 64, 128, 256, 512)
#: l/m rotation (rad), inside the bot's +-0.02 and away from the phase centre
POSITIONS = tuple((a * s, b * t) for a, b in ((0.012, 0.004), (0.004, 0.012), (0.009, 0.009))
                  for s, t in ((1, 1), (-1, 1), (-1, -1), (1, -1)))


def _post(path: str, fields: dict, timeout: float) -> dict:
    data = urllib.parse.urlencode({k: str(v) for k, v in fields.items() if v is not None}).encode()
    req = urllib.request.Request(DASH + path, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "{}")
        except ValueError:
            body = {}
        return dict(body, ok=False, http_status=e.code)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def _get(path: str, timeout: float = 30.0) -> dict:
    try:
        with urllib.request.urlopen(DASH + path, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def _match_doc(store, inj_id: str) -> dict | None:
    v, _ = store.get(f"/mon/dsart/inject/matches/{inj_id}")
    return json.loads(v) if v else None


def bucket_key(dm: float, width: int) -> str:
    base = f"dm{int(round(dm / 50.0) * 50):04d}"
    return base if int(width) == 4 else f"{base}@w{int(width):04d}"


def plan(dms, widths) -> list[tuple[float, int]]:
    """Interleaved order: widths outer, DMs inner and rotated per width, so consecutive shots never share
    a DM (the matcher's 5% DM window) and a bucket's neighbours in time differ in width too."""
    out = []
    for i, w in enumerate(widths):
        k = i % len(dms)
        for dm in dms[k:] + dms[:k]:
            out.append((dm, w))
    return out


def _log(fh, rec: dict) -> None:
    rec.setdefault("t", time.time())
    fh.write(json.dumps(rec, default=str) + "\n")
    fh.flush()
    print(json.dumps({k: rec[k] for k in rec if k in ("phase", "dm", "width", "ok", "K", "observed_snr",
                                                        "target_snr", "inj_id", "reason", "error")}),
          flush=True)


def run_calibrate(a, store, fh, buckets, pos) -> None:
    for dm, w in buckets:
        l, m = next(pos)
        t0 = time.time()
        r = _post("/control/inject_calibrate", dict(
            dm_pc_cm3=dm, width_samples=w, l_rad=l, m_rad=m, profile="gaussian",
            poll_timeout_s=a.poll_timeout_s, use_ladder=1, health_check=1,
            user="claude x2 campaign (calibrate)"), timeout=a.poll_timeout_s * 4 + 120)
        snr0 = r.get("observed_snr")
        if not r.get("ok") and "saturated" in str(r.get("reason")) and snr0:
            # the endpoint's ladder only steps UP (for non-detections): a probe on the clip rail is
            # repeated once, scaled to ~S/N 40, after the usual gap
            _log(fh, dict(phase="calibrate_saturated", dm=dm, width=w, observed_snr=snr0,
                          inj_id=r.get("inj_id"), response=r))
            _sleep_until(t0 + a.probe_gap_s)
            t0 = time.time()
            f = max(1e-6, float(r.get("fluence_used") or 5e-4) * 40.0 / float(snr0))
            r = _post("/control/inject_calibrate", dict(
                dm_pc_cm3=dm, width_samples=w, l_rad=l, m_rad=m, profile="gaussian",
                fluence_jy_ms=f"{f:.3e}", poll_timeout_s=a.poll_timeout_s, use_ladder=0,
                health_check=1, user="claude x2 campaign (calibrate, scaled)"),
                timeout=a.poll_timeout_s * 2 + 120)
        time.sleep(a.settle_s)
        doc = _match_doc(store, r.get("inj_id")) if r.get("inj_id") else None
        best = (doc or {}).get("best") or {}
        _log(fh, dict(phase="calibrate", dm=dm, width=w, l=l, m=m, bucket=bucket_key(dm, w), t_fire=t0,
                      ok=bool(r.get("ok")), K=r.get("K"), observed_snr=r.get("observed_snr"),
                      inj_id=r.get("inj_id"), reason=r.get("reason") or r.get("error"),
                      settled_best=best, response=r))
        _sleep_until(t0 + a.probe_gap_s)


def run_grid(a, store, fh, buckets, pos) -> None:
    ks = _get("/control/inject_calibrations")
    fired = []
    for rep in range(a.shots_per_bucket):
        for dm, w in buckets:
            snr = a.target_snr[(len(fired)) % len(a.target_snr)]
            l, m = next(pos)
            inj_id = f"x2grid_dm{int(dm)}_w{w}_s{int(snr)}_r{rep}_{int(time.time())}"
            t0 = time.time()
            r = _post("/control/inject", dict(
                inj_id=inj_id, l_rad=l, m_rad=m, dm_pc_cm3=dm, width_samples=w, profile="gaussian",
                target_snr=snr, user="claude x2 campaign (grid)"), timeout=120)
            rec = dict(phase="grid_fire", dm=dm, width=w, l=l, m=m, bucket=bucket_key(dm, w), inj_id=inj_id,
                       target_snr=snr, t_fire=t0, ok=bool(r.get("ok")), reason=r.get("error"), response=r)
            _log(fh, rec)
            if r.get("ok"):
                fired.append(rec)
            _sleep_until(t0 + a.spacing_s)
    time.sleep(a.collect_after_s)
    for rec in fired:
        doc = _match_doc(store, rec["inj_id"])
        best = (doc or {}).get("best") or {}
        obs = best.get("observed_snr")
        _log(fh, dict(phase="grid_result", dm=rec["dm"], width=rec["width"], bucket=rec["bucket"],
                      inj_id=rec["inj_id"], target_snr=rec["target_snr"], observed_snr=obs,
                      ratio=(obs / rec["target_snr"]) if obs else None, matched=bool(best), best=best,
                      calibrations_at_start=(ks.get("calibrations") or ks).get(rec["bucket"])
                      if isinstance(ks, dict) else None))


def _sleep_until(t: float) -> None:
    while time.time() < t:
        time.sleep(min(1.0, t - time.time()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("calibrate", "grid", "both"), default="both")
    ap.add_argument("--dms", default=",".join(str(d) for d in DMS))
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--target-snr", default="20", help="comma list, cycled over the grid shots")
    ap.add_argument("--shots-per-bucket", type=int, default=1)
    ap.add_argument("--spacing-s", type=float, default=60.0)
    ap.add_argument("--probe-gap-s", type=float, default=150.0, help="probe start to the next probe start")
    ap.add_argument("--poll-timeout-s", type=float, default=90.0)
    ap.add_argument("--settle-s", type=float, default=20.0, help="before re-reading a probe's match doc")
    ap.add_argument("--collect-after-s", type=float, default=150.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args(argv)
    a.target_snr = [float(x) for x in a.target_snr.split(",") if x]
    buckets = plan(tuple(float(x) for x in a.dms.split(",")), tuple(int(x) for x in a.widths.split(",")))
    if a.dry_run:
        for dm, w in buckets:
            print(dm, w, bucket_key(dm, w))
        print(len(buckets), "buckets; calibrate ~", round(len(buckets) * a.probe_gap_s / 3600, 1), "h; grid ~",
              round(len(buckets) * a.shots_per_bucket * a.spacing_s / 3600, 1), "h")
        return 0
    import etcd3
    store = etcd3.client(host="etcdv3service.pro.pvt", port=2379)
    pos = itertools.cycle(POSITIONS)
    with open(a.out, "a") as fh:
        _log(fh, dict(phase="start", args=vars(a), n_buckets=len(buckets)))
        if a.phase in ("calibrate", "both"):
            run_calibrate(a, store, fh, buckets, pos)
        if a.phase in ("grid", "both"):
            run_grid(a, store, fh, buckets, pos)
        _log(fh, dict(phase="done"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
