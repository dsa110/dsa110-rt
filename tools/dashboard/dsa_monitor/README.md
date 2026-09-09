# DSA-110 monitoring dashboard (h23, M7.6)

Multi-tab Flask app that surfaces:

* **Antennas / RFI** — per-antenna pre-flag bandpass, 30-min waterfalls,
  flag spectra, and the existing `/mon/ant/<n>` etcd table for one
  selected antenna.
* **Array sums** — the core and each arm of the core treated as extra
  antennas: band-summed significance at 2.097 ms with the fired
  samples marked, E-W vs N-S arm symmetry against the 1:1 far-field
  line, and gain-normalised group spectra. This is where the
  array-common burst detector (M7.7) reports; see
  `docs/M7.7_array_burst_flagging.md`.
* **SEFDs** — passthrough/iframe of the existing SEFD dashboard
  (port 5777, run by `sefd_dashboard.service`).
* **Burst candidates** — placeholder for a future tab.

## Architecture

```
16 corr nodes (one per chgroup)
   ├── corr_fast_integration  (publisher; M7.6 window aggregator)
   │     └── /dev/shm/dsart-rfi-window-<cn>   POSIX-shm ring
   └── rfi_monitor_export     (sidecar; HTTP API on :5780)
                                          │
                                          ▼ HTTP poll every ~2 s
                                  h23  dsa_monitor.service (this app)
                                  ├── RFIPoller (background thread)
                                  ├── per-cn 30-min in-mem ring
                                  └── Flask renderers → matplotlib PNGs
```

There is **no shared filesystem**. The h23 app pulls window records
straight from each corr node over HTTP, decodes the base64-encoded
arrays, and stores the last 30 minutes locally (~110 MB resident).
Page refresh renders matplotlib PNGs from the in-memory ring — no
auto-refresh, no caching.

## Endpoints

| URL | What |
|-----|------|
| `/` or `/antennas` | Antennas/RFI tab (default) |
| `/antennas?ant=<idx>` | Same, with antenna pre-selected (`ant_idx` 0..95) |
| `/arraysum` | Array sums tab (array-burst detector) |
| `/arraysum?chgroup=<n>` | Same, with a sub-band pre-selected (0..15) |
| `/sefds` | SEFDs tab (iframe to `:5777`) |
| `/bursts` | Burst candidates placeholder |
| `/plot/bandpass.png?ant=<idx>` | Pre-flag bandpass spectrum |
| `/plot/bandpass_wf.png?ant=<idx>` | Pre-flag bandpass waterfall |
| `/plot/flag_spectrum.png?ant=<idx>` | Latest-window flag fraction spectrum |
| `/plot/flag_wf.png?ant=<idx>` | 30-min flag fraction waterfall |
| `/plot/array_burst_ts.png?chgroup=<n>` | Band-summed significance per group, 2.097 ms |
| `/plot/array_burst_arms.png?chgroup=<n>` | E-W vs N-S arm excess scatter |
| `/plot/array_burst_spec.png` | Gain-normalised group spectra, full band |
| `/control/slow_rfi` (GET/POST) | Slow-correlator RFI flagging toggle (etcd `/cnf/corr_slow_rfi`) |
| `/api/status` | JSON: per-cn ring sizes, last seq, last fetch time |

## Deployment on h23

```bash
# Once-only: install service (we assume the dsa110-rt repo is checked
# out at /home/ubuntu/proj/dsa110-rt).
mkdir -p ~/.config/systemd/user
cp /home/ubuntu/proj/dsa110-rt/tools/dashboard/dsa_monitor/dsa_monitor.service \
   ~/.config/systemd/user/dsa_monitor.service

systemctl --user daemon-reload
systemctl --user enable --now dsa_monitor.service
systemctl --user status  dsa_monitor.service

# Open in a browser:
#   http://lxd110h23:5778/
```

The service uses the existing `casa38` conda env (which already has
`Flask`, `matplotlib`, `numpy`, and `dsautils`). No extra packages
needed.

## Configuration

Environment variables (set in `dsa_monitor.service`):

| Var | Default | Meaning |
|-----|---------|---------|
| `DSA_MONITOR_PORT`     | `5778` | TCP port to bind |
| `DSA_MONITOR_BIND`     | `0.0.0.0` | Bind address |
| `SEFD_DASHBOARD_URL`   | `http://lxd110h23:5777/` | URL for the SEFDs iframe |
| `LOG_LEVEL`            | `INFO`  | Python logging level |

Corr-node URLs are derived from `corr_topology.CORR_NODES`; the
default is `http://n<NN>.pro.pvt:5780/` (the
`rfi_monitor_export` sidecar listening port). To change the port,
edit `corr_topology.RFI_HTTP_PORT_DEFAULT` to match what
`configs/dsart_pipeline_rt.yaml::routines.rfi_monitor_export.args`
uses.

## Layout

```
dsa_monitor/
├── app.py                    Flask app + module-init RFIPoller singleton
├── corr_topology.py          The 16-cn fleet + RFI exporter base URLs
├── rfi_client.py             HTTP client; decodes the exporter JSON
├── rfi_store.py              In-memory 30-min ring buffer (per cn)
├── plot_render.py            matplotlib PNG renderers
├── freq_mapping.py           Per-chgroup freq labelling (uses dsart.common)
├── ant_table.py              Per-antenna table from /mon/ant/<n> + RFI agg
├── array_sum_view.py         Array sums page data (core / arm summing groups)
├── templates/                Jinja2 (base / antennas / arraysum / sefds / bursts / control)
├── static/                   (empty placeholder for CSS / icons)
├── dsa_monitor.service       systemd user unit
└── README.md                 this file
```
