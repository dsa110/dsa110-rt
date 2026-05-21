# dsart-rt → InfluxDB pusher (h20, M7.6)

System service that mirrors the dsart-rt control plane's monitor-point
keys from etcd into the existing InfluxDB 1.7 instance on
`lxd110h20`, sitting alongside the legacy `etcd2db` bridge without
overlapping its key namespace.

Reference docs:

* `docs/m7/M7.6-MONITOR-POINTS.md`        — corr_rt key shapes
* `docs/m7/M7.6-MONITOR-POINTS-SEARCH.md` — search_rt key shapes

This implementation is the §6 / §5 reference-recipe pusher from those
docs, exactly.

## Architecture

```
  16 corr nodes  +  4 search nodes
       └── dsart_rt / capture_control / rfi_monitor_export
           └── etcdv3service.pro.pvt:2379
                            │
                            │  get_prefix(/mon/{corr_rt,search_rt,service/*}/)
                            ▼
   lxd110h20:  dsart_rt_to_influx.service (this code; system unit)
                            │
                            │  HTTP line protocol  (loopback)
                            ▼
                  lxd110h20:8086  influxd 1.7  (db `dsa110`)
                            │
                            ▼
                  lxd110h20:3000  Grafana 6.2.5
```

Co-located with `influxd` so the POST hop is loopback.  The legacy
`etcd2db` bridge keeps owning the legacy `/mon/{ant,beb,cal,wx,corr,
status,T1,T2,nsfrb*}/` prefixes; this pusher owns the disjoint
`/mon/{corr_rt,search_rt,service/{corr,search}_rt}/` prefixes.  No
overlap, no race.

## Measurements

| measurement | from | rows/poll | tag keys |
|---|---|---|---|
| `corr_rt_routine`   | `/mon/corr_rt/<cn>`             | 8 / cn (16 cn = 128) | `cn_id, host, instance, state, routine` |
| `corr_rt_buffer`    | `/mon/corr_rt/<cn>`             | 0–4 / cn (currently 0 — `metric: {}`) | `cn_id, host, buffer` |
| `corr_rt_capture`   | `/mon/corr_rt/<cn>/capture/<port>` | 1 / cn-port (32) | `cn_id, host, udp_port, control_port, arm_state` |
| `corr_rt_rfi`       | `/mon/corr_rt/<cn>/rfi`         | 3 / cn (48 — per-pol fan-out) | `cn_id, host, pol` (pol0\|pol1\|both) |
| `corr_rt_heartbeat` | `/mon/service/corr_rt/<cn>`     | 1 / cn (16) | `cn_id, host, state` |
| `search_rt_routine`   | `/mon/search_rt/<cn>`          | 3 / cn (4 cn = 12)  | `cn_id, host, instance, state, routine` + `coarse_dm` on the two compute halves |
| `search_rt_heartbeat` | `/mon/service/search_rt/<cn>`  | 1 / cn (4)  | `cn_id, host, state` |

Steady-state row count per **publish cycle** (every 2 s): ~240.  At
the default 1 s poll cadence with mod_revision dedup, roughly half
of all ticks emit zero rows.

## Cumulative-counter delta tracking

`corr_rt_capture` cumulative counters (`n_recv_packets`,
`n_recv_bytes`, `n_dropped_payload`, `n_dropped_kernel`,
`n_seq_skipped`, `n_too_late`, `n_wrong_size`, `n_recv_errors`,
`n_block_writes`) are emitted as **both** the raw counter and a
pre-diffed delta field `n_<field>_delta`.  The delta state is keyed
on `(cn_id, udp_port)` and **reset on `pid` flip** (capture binary
restart) or on any non-monotonic dip — so the time series never
shows a negative spike.

## UNAVAILABLE / degraded placeholders

The `capture_control` sidecar publishes a degraded placeholder
(`arm_state="UNAVAILABLE"`, `arm_state_int=-1`) whenever the C
capture binary's shm is missing or stale.  Per the M7.6 spec the
pusher emits **only** the tag set plus `degraded=1` for those — no
synthetic zero counters that would pollute the time series.

## Deploy on lxd110h20

The pusher runs in the `casa38` env (`/home/ubuntu/anaconda3/envs/casa38`)
because that env already has `dsautils`, `requests`, and
`influxdb` (the v1 Python client) installed.  No new packages
needed.

```bash
# 1. Clone (or rsync) the repo to /home/ubuntu/proj/dsa110-rt.
sudo -u ubuntu git clone git@github.com:dsa110/dsa110-rt.git \
    /home/ubuntu/proj/dsa110-rt

# 2. Install the wrapper + system unit.
sudo cp /home/ubuntu/proj/dsa110-rt/tools/dashboard/dsart_rt_to_influx/startDsartRtToInflux \
    /home/ubuntu/bin/
sudo chmod +x /home/ubuntu/bin/startDsartRtToInflux

sudo cp /home/ubuntu/proj/dsa110-rt/tools/dashboard/dsart_rt_to_influx/dsart_rt_to_influx.service \
    /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now dsart_rt_to_influx.service

# 3. Verify.
systemctl status dsart_rt_to_influx.service
journalctl -u dsart_rt_to_influx.service -f
curl -s -G "http://localhost:8086/query?db=dsa110" \
    --data-urlencode "q=SHOW MEASUREMENTS" | python3 -m json.tool
```

The expected steady-state `SHOW MEASUREMENTS` output is the
existing 7 plus the 7 new `*_rt_*` ones above.

## Configuration

All knobs are environment variables read by `startDsartRtToInflux`:

| var | default | meaning |
|-----|---------|---------|
| `DSART_RT_REPO`             | `/home/ubuntu/proj/dsa110-rt` | repo checkout root |
| `DSART_RT_TO_INFLUX_PY`     | `/home/ubuntu/anaconda3/envs/casa38/bin/python` | python interpreter |
| `DSART_RT_TO_INFLUX_URL`    | `http://localhost:8086` | influxd 1.x endpoint |
| `DSART_RT_TO_INFLUX_DB`     | `dsa110` | target database |
| `DSART_RT_TO_INFLUX_POLL_S` | `1.0`    | etcd poll cadence (s) |
| `DSART_RT_TO_INFLUX_LOG_LEVEL` | `INFO` | python logging level |

To override per-host, drop a `dsart_rt_to_influx.service.d/override.conf`
under `/etc/systemd/system/` with `[Service] Environment=...` lines.

## Local development + tests

The module is **self-contained** — no `import dsart`, no setup.py
dance.  Just run from the repo root:

```bash
# Tests (uses captured live payloads + a fake etcd / fake influx writer).
pytest tests/test_dsart_rt_to_influx_pusher.py -v

# Smoke run against the real etcd + Influx (e.g. on h20 in casa38):
python tools/dashboard/dsart_rt_to_influx/pusher.py \
    --max-iters 3 --log-level DEBUG
```

## Schema versioning

The pusher refuses to ingest `corr_rt_capture` / `corr_rt_rfi`
payloads whose `schema_version` doesn't match `SUPPORTED_SCHEMA_VERSION`
(currently `1`).  When a producer bumps its schema, bump that
constant in `pusher.py` and update the affected `make_*_points`
helper in the same commit.  See M7.6 corr doc §8 + search doc §8 for
the policy.

## Planned-but-not-yet-published keys

The pusher matches the three planned `search_rt` per-routine keys
documented in `M7.6-MONITOR-POINTS-SEARCH.md` §7 (`.../rx`,
`.../compute/<half>`, `.../cands`) and **logs a single error** + skips
them.  When the corresponding publisher ships, add a `make_*_points`
helper and route it from `InfluxPusherService._route`.

## Grafana dashboard

A companion Grafana dashboard is committed under
[`grafana/`](grafana/) and lives at uid `dsartRtMpV1` on the
`lxd110h20:3000` instance:

* `http://lxd110h20.sas.pvt:3000/d/dsartRtMpV1/dsart-rt-corr_rt-search_rt`

It renders all 6 new measurements grouped into seven rows (fleet
heartbeats, corr routine state, capture pipeline, capture health
flags, RFI, search routine state, service heartbeat cadences) using
the same look-and-feel as the existing `Correlator` dashboard.

Re-generate the JSON and POST it back to the live instance with:

```bash
# 1. Regenerate dashboard JSON (commit the diff if you want the change
#    tracked).
python tools/dashboard/dsart_rt_to_influx/grafana/build_dashboard.py

# 2. POST to Grafana on h20.  Default URL / creds match the
#    long-running install.  --post will overwrite the existing
#    dashboard at uid dsartRtMpV1.
python tools/dashboard/dsart_rt_to_influx/grafana/build_dashboard.py \
    --post --grafana-url http://localhost:3000 \
    --grafana-auth admin:adminLETmeIN
```

## File layout

```
tools/dashboard/dsart_rt_to_influx/
├── pusher.py                   The service (≈800 LOC including docstrings)
├── startDsartRtToInflux        Bash wrapper that systemd's ExecStart calls
├── dsart_rt_to_influx.service  systemd system unit
├── grafana/
│   ├── build_dashboard.py        Dashboard generator (idempotent, --post)
│   └── dsart_rt_dashboard.json   Committed snapshot
└── README.md                   this file

tests/
└── test_dsart_rt_to_influx_pusher.py  pytest suite (51 tests)
```
