# DSA-110 node baseline

Captured from running nodes on **2026-08-01**: `n03` (corr) and `n01`
(search). This is the contract `verify_node.sh` and `versions.env`
assert against — if the fleet changes, update this file in the same
commit.

## Hardware / OS (identical on both roles)

| | |
|---|---|
| OS | Ubuntu 18.04.5 LTS (**EOL**, ESM only) |
| Kernel | 4.15.0-213-generic |
| CPU | Intel Xeon Silver 4210, 40 cores, 2 NUMA nodes |
| RAM | 172 GiB |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti, 11264 MiB each |
| Driver | 525.105.17 |
| CUDA | 11.1 (`/usr/local/cuda` → `cuda-11.1`) |
| gcc | 7 (CUDA 11.1's nvcc rejects gcc ≥ 10) |

## Network

| | corr (n03) | search (n01) |
|---|---|---|
| Data fabric | `enp129s0f0`, mlx5_core, **MTU 9000** | `br2` bridge over `enp129s0f0` |
| Data address | 10.41.0.x/24 | 10.41.0.x/24 |
| Management | `br1`, 10.42.0.x/24 | `br1`, 10.42.0.x/24 |
| Unused | `eno1`, `eno2` (ixgbe, MTU 1500) | — |

The driver is the **in-tree** `mlx5_core`, not MLNX_OFED. The MLNX_EN
4.7 bundle on the MaaS fileserver is deliberately not installed.

## Kernel tuning

Persisted in `/etc/sysctl.conf` on the fleet; provisioning writes
`/etc/sysctl.d/99-dsa110-node.conf` instead so re-runs cannot duplicate
lines.

```
net.core.rmem_max          = 536870912     # 512 MB
net.core.wmem_max          = 536870912
net.core.rmem_default      = 16777216
net.core.netdev_max_backlog= 250000
kernel.shmmax              = 68719476736   # 64 GiB
kernel.shmall              = 4294967296
```

`rmem_max` is the load-bearing one: `search_rx` requests a 256 MB
`SO_RCVBUF` and the kernel silently clamps to `rmem_max`. At the
default the capture drops packets with nothing in any log.

`shmmax` matters because PSRDADA holds ~**33.3 GiB across ~450 SysV
segments** on a running corr node.

No hugepages are configured.

## Shared memory

| | corr | search |
|---|---|---|
| `/dev/shm` | 87 G | 47 G |

`/dev/shm` backs the POSIX receive ring (`_recv_ring`); PSRDADA uses
SysV (`ipcs -m`). Both matter.

## Software

Two independent Python installs — conflating them is an easy mistake:

- `/home/ubuntu/miniforge3` → env **`dsa110-rt`** (py3.11). **This is
  what dsart runs from.** The orchestrator's interpreter on n03 is
  `/home/ubuntu/miniforge3/envs/dsa110-rt/bin/python3.11`. Search nodes
  also carry a `dsart-build` env.
- `/home/ubuntu/anaconda3` → env `casa38` (py3.8). Legacy calibration
  tooling; not needed for dsart.

`dsart` is an **editable** install: `__editable__.dsart-0.0.1.pth`
points at `/home/ubuntu/proj/dsa110-rt/src/dsart`, so the checked-out
tree is the running code.

Compiled extensions (built by `setup.py`, ABI-tagged per interpreter):

```
src/dsart/transport/_recv_ring.cpython-311-x86_64-linux-gnu.so
src/dsart/transport/_recv_epoll.cpython-311-x86_64-linux-gnu.so
src/dsart/capture/dsart_capture_manythread
```

PSRDADA installs to `/usr/local/bin/dada_*`.

## Repos

`/home/ubuntu/proj/`:

- `dsa110-rt` — the pipeline (`dsart`)
- `dsa110-shell/` — myrepos collection, ~26 sub-repos. Load-bearing:
  `dsa110-psrdada` (HEAD `72f07e9`), `dsa110-xGPU`, `dsa110-sigproc`,
  `dsa110-mbheimdall`, **`dsa110-xengine`**.

`dsa110-xengine` must exist even though the legacy search path is
retired: beamformer-weight distribution writes `antennas.out` into
`dsa110-xengine/utils/` on every corr node.

## Role-specific

Search nodes require `/home/ubuntu/data/dm_plans/*.npz` — `search_compute`
takes `--dm-plan-path` and will not start without it. Current:
`dm_plan_N8_dmmin100_tol1.364_csf8_cap10.2_v3.1.npz`.

Corr nodes stage voltage dumps in `/home/ubuntu/data/voltage_staging`
(~6.5 GiB per node per event).

## Not provisioned, by design

- **`dsart_rt` does not start at boot.** There is no systemd unit, cron
  entry or rc.local for it anywhere on the fleet; it is launched from
  h23 by `tools/ops/_m75_phaseB_16x4_launch.sh`. Provisioning makes a
  node *ready*, not *running*.
- SSH keys and accounts. MaaS injects the registered keys of the
  deploying user on every deploy. Nothing key-shaped is in this repo.
- PSRDADA buffers, created by `dsart_rt` from `/cnf/pipeline_rt`.
  Reader counts are physics-pinned; creating them out of band would
  fight the orchestrator.
