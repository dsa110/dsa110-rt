# DSA-110 disaster recovery — what it takes to come back

Written 2026-07-29, after a site power outage destroyed the NVMe in
`lxd110h20` (etcd + InfluxDB + Grafana host). Everything below was verified
against the live system during that outage, not inherited from older docs.

Companion document: **`dsa110-shell/deploy/README.md`** covers building a
*node* (MaaS/curtin, `install_tarballs`, conda, myrepos). This document
covers **h23** — the control host — plus the cross-cutting inventory of
things that are not in git.

---

## 1. Blast radius: what each failure costs you

| Failure | Consequence | Recoverable? |
| --- | --- | --- |
| One `dataz` disk (of 6) | none — raidz1 resilvers | yes |
| **Two `dataz` disks** | **48 TB science archive lost** | **no** |
| `/media/ubuntu/ssd` (sdh) | SEFD scanner source + 776 GB results | source now in git; results regenerable, slowly |
| `/media/ubuntu/data` (sdg) | that disk's contents | no parity |
| **h23 root NVMe** | **the whole control plane** — see §3 | only with this runbook |
| A corr/search node NVMe | that node's pipeline | yes — reprovision, see `dsa110-shell/deploy/` |
| `lxd110h20` (what just happened) | etcd, InfluxDB, Grafana, webserverUI; **all four service DNS aliases go NXDOMAIN** | yes — rebuilt 2026-07-30, see §7 |
| **`lxd110maas`** | **all node rebuilds break silently** | now mirrored — see §5 |
| **`dsa110maas`** | MaaS/DNS/DHCP for the whole cluster | now backed up — see §5 |

Disk redundancy as measured 2026-07-28:

| Device | Role | Redundancy | Notes |
| --- | --- | --- | --- |
| `nvme0n1` 477 G | h23 `/` | **NONE** | healthy (6% wear, 0 media errors) but a single point of failure |
| `sda`–`sdf` 6×14 TB | `dataz` raidz1, 48 TB | survives **1** disk | all PASSED, but **all 38,265 h (4.4 y), same batch** → correlated risk. raidz2 would be the right posture at this capacity |
| `sdg` 9.1 T (USB) | `/media/ubuntu/data` | **NONE** | 51,194 h (5.8 y) |
| `sdh` 3.7 T (USB) | `/media/ubuntu/ssd` | **NONE** | 95% full |

---

## 2. Immediate triage after a power event

1. **Check `/dataz` is read-write.** A hard power cycle can leave the pool
   imported read-only, which silently blocks candidate archiving, beamformer
   weights, calibration and the fleet's NFS mount:
   ```bash
   zpool get readonly dataz     # must be "off"
   mount | grep dataz           # must show rw
   ```
   If it says `on`, **`zfs set readonly=off` will NOT fix it** — it fails with
   a misleading `internal error: out of memory`. The pool was imported
   read-only and must be re-imported:
   ```bash
   lxc stop calibration23                       # holds /dataz bind mounts
   systemctl --user stop dsa_monitor sefd_dashboard dsart_slack_relay dsart_c3
   sudo fuser -vm /dataz                        # NOT `lsof +D` — it walks 48 TB
   sudo zpool export dataz                      # retry once if "busy"
   sudo zpool import dataz
   ```
2. **Check NFS.** `nfs-kernel-server` does not always survive; without it the
   whole fleet loses `/dataz/dsa110/operations`.
   `systemctl is-active nfs-kernel-server` → `sudo systemctl start nfs-kernel-server`.
3. **Check `/media/ubuntu/data` is mounted** — its `/etc/fstab` line is
   commented out, so it does **not** come back automatically.
4. **Expect etcd-dependent services to fail** if the etcd host is down:
   `dsart_c2`, `dsart_c3`, and `dsa110-operator-supervisor-h23` all exit or
   crash-loop on `etcd3.exceptions.ConnectionFailedError`. That is a symptom,
   not the disease.
5. **DNS**: when a host leaves `Deployed` state its name stops resolving
   cluster-wide. The mechanism, established 2026-07-30, is worse than "MaaS
   drops A records":

   - MaaS's **default** domain is `sas.pvt`, which is `authoritative=False` and
     is **not served by BIND at all**. Every machine MaaS commissions gets a
     name that can never resolve — which is also why `corrNN.sas.pvt` in the
     legacy `deploy` scripts is dead.
   - So every operational name is a hand-created `pro.pvt` **dnsresource**, and
     those were bound to `alloc_type: Observed` (DHCP-discovered) addresses on a
     specific machine interface. When the machine left `Deployed`, the address
     went null and **the alias itself disappeared** — `etcdv3service.pro.pvt`,
     `influxdbservice.pro.pvt`, `grafanaservice.pro.pvt` and
     `webserverUIservice.pro.pvt` all went NXDOMAIN together.

   They are now **CNAMEs to `lxd110h20.pro.pvt`**, a single literal A record —
   one place to change, and reboots/rescue/redeploy no longer delete the names.
   Verify with `dig +noall +answer etcdv3service.pro.pvt`, which should show
   `CNAME lxd110h20.pro.pvt` → `A`.

   Current addresses: `lxd110h20` (the new one, ex-h24) = 10.42.0.228 (pro) /
   10.41.0.254 (sas), `lxd110h23` = 10.42.0.232, `lxd110h17` = 10.42.0.148.
   10.42.0.126/.127 are *legacy LXD container* device reservations
   (`etcdv3service2`, `influxdbservice2`) and are **not** the bare-metal
   services. `dsa110maas` = 10.42.0.4 (also the resolver), `lxd110maas` =
   10.42.0.3.

---

## 3. Rebuilding h23 from bare metal

h23 is a MaaS-deployed machine (`lxd110h23`, Ubuntu 18.04, flat ext4 on a
512 GB NVMe). Provision it like any node (see `dsa110-shell/deploy/README.md`),
then restore the following. **Everything in this section is what makes h23
h23** — none of it comes back from MaaS.

### 3.1 Storage
- Import the ZFS pool: `zpool import dataz` (6 disks, raidz1 — the disks
  themselves survive an NVMe loss).
- Re-add `/media/ubuntu/ssd` (xfs, by-uuid, `nofail`) and
  `/media/ubuntu/data` to `/etc/fstab`.
- Recreate `/etc/exports`:
  `/dataz/dsa110/operations *(rw,sync,no_subtree_check,no_root_squash)`
  then `exportfs -ra`. The whole fleet mounts this.
- Netplan: `br2` = `enp129s0f0`, `10.41.0.5/24`, gw `10.41.0.4`, **MTU 9000**
  (the data fabric); `br1` = `eno1`, DHCP.

### 3.2 Credentials
h23 needs a number of credentials restored before its services will run: SSH
identities (GitHub push, and the key authorised on the corr nodes), plus API
tokens and webhooks for the alert/notification and monitoring integrations.

**This repository is public, so the inventory of what lives where is kept out
of it.** See the private companion:
`/dataz/dsa110/dr_archive/SECRETS_INVENTORY.md` (mode 600), which lists each
path, what it holds, and whether it can be restored or must be re-issued —
plus a rotation priority list.

Operationally: none of these are recoverable from git or from MaaS. If h23's
root disk is lost before they are copied to a secret store off-host, each one
must be re-issued from its provider and re-enrolled.

### 3.3 Conda environments
**Neither production env has a spec in git** (this is the largest rebuild
risk after secrets). Snapshots of both are exported to
`docs/env/` in this repo (`dsart_h23.yml`, `casa38.yml` + pip freezes) — treat
them as a starting point, not a guarantee, since `casa38` carries ~39
`setup.py develop` eggs accreted over years.

- **`dsart_h23`** (Python 3.11.15) — the production env for `dsa_monitor`,
  `dsart_c2`, `dsart_c3`, `dsart_slack_relay`, the operator supervisor and the
  `archive_bfweights` cron. Editable installs that must be recreated:
  `dsart` → `dsa110-rt`, `dsa110-operator`, `dsa110-event`, `dsa110-pyutils`.
  Also needs `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` + `protobuf 3.20.3`.
- **`casa38`** (Python 3.8.13) — SEFD scanner, `delete_level1_data` cron,
  dashboard subprocesses, and (after the migration) the calibration services.
  Editable: `dsa110-T3`, `dsa110-voltage-imaging`, plus `psrdada`,
  `dsa110_calib`, `dsa110_meridian_fs`, `lalsuite`, `ligo.skymap`, …

### 3.4 Services
User systemd units under `~/.config/systemd/user/`, kept alive by
**`loginctl enable-linger ubuntu`** — without linger, nothing starts at boot.

| Unit | Versioned at |
| --- | --- |
| `dsa_monitor.service` | `dsa110-rt/tools/dashboard/dsa_monitor/` |
| `dsart_c2.service`, `dsart_c3.service`, `dsart_slack_relay.service` | `dsa110-rt/systemd/` |
| `dsa110-operator-supervisor-h23.service` | `dsa110-operator/deploy/` (via `scripts/install_service.sh h23`) |
| `sefd_dashboard.service` | **now** `dsa110-rt/tools/sefd/` (was in no repo) |

⚠ `dsart_c2` is **not enabled** — after a reboot it stays down until started
by hand. The four drop-in overrides in `~/.config/systemd/user/*.d/` are not
in git either (dashboard `CANDS_MAX_EVENTS`, C2 inject tolerances, and two
log-redirect drop-ins).

Cron (`crontab -l`): `delete_level1_data.py` (04:00), `backup_annotations.sh`
(10:00), and `archive_bfweights.py` for `generated/` + `applied/` (monthly).

### 3.5 Non-repo state to restore
- `~/.dsa_monitor/annotations.db` — **human burst classifications**, the one
  irreplaceable operator-generated dataset on h23. Backed up daily to
  `/dataz/dsa110/database_human_annotations_backups` — i.e. **same host**;
  survives an NVMe loss but not an h23 loss.
- `~/.dsa_monitor/{slack_candidate_posts.db,slack_relay_state.json,backup_annotations.sh}`
- `/media/ubuntu/ssd/vikram/sefd/sefd_dashboard/` — scanner **state**
  (`state.json`) and 776 GB of `results/`. Source is now in git (§3.4).
- `/home/ubuntu/vikram/dev/` — the workspace is **not a git repo**; it holds
  `CLAUDE.md`, `REALTIME_FRB_SEARCH.md` (legacy physics ground truth) and
  `dsa110-rt_revamp_*.plan.md` (658 KB, cited throughout the code as
  "plan §N"). 113 MB, unversioned.

---

## 4. Rebuilding a corr/search node

See `dsa110-shell/deploy/README.md` for the full chain. Node-specific state
that is **not** in git and must be restored or regenerated:

- `~/.config/dsart-rt.env` — per-node `DSART_INSTANCE` (`pipeline_rt` vs
  `search_rt`) and `DSART_CN`. Regenerate with
  `dsa110-rt/tools/ops/dsart-rt services install`.
- `utils/antennas.out` — the live beamformer/cal solution, **git-ignored**,
  one copy per node. Regenerate by re-running `update_bfweights.py` against a
  fresh calibrator transit.
- `~/data/dm_plans/dm_plan_N8_dmmin100_tol1.364_csf8_cap10.2_v3.1.npz` — the
  DM plan the live config actually loads; identical across nodes, centrally
  generated, **not in git**. Regenerate with `tools/build_dm_plan.py`.
- Fringestopping tables (`~/data/fstables/`, 912 MB on a corr node) —
  auto-regenerated on cache miss, but slow.
- **sysctl tuning is runtime-only** — no `/etc/sysctl.d/` drop-in exists, so
  `tools/ops/sysctl.sh` must be re-run after every reboot or the M4b
  corner-turn ipfrag fix is silently lost.
- Compiled artifacts to rebuild: `_recv_epoll`/`_recv_ring` C extensions and
  `dsart_capture_manythread` (`setup.py build_ext`), PSRDADA
  (`tools/ops/install_psrdada.sh`), xGPU, sigproc.

---

## 5. Where the backups are

Created 2026-07-29 into `/dataz/dsa110/dr_archive/` (mode 700), ~5 GB:

| Path | What | Why it matters |
| --- | --- | --- |
| `lxd110maas-artifacts/` (3.1 GB) + `.md5` | the complete node-build artifact store mirrored off `http://10.42.0.3/maas/` | that host is absent from MaaS **and** DNS, hardcoded by IP; `dedisp.tar` and the LabJack installer exist nowhere else |
| `dsa110maas-config/maasdb-*.pgdump` (1.7 GB) | MaaS postgres dump — 25 machines, IPs, DHCP reservations, power credentials | **the first backup this DB has ever had** |
| `dsa110maas-config/dsa110maas-config-*.tar.gz` (257 MB) | `/etc/maas` (incl. preseeds), `/etc/bind`, `dhcpd.conf`, `~/proj/maas` (offline Mellanox driver bundle, pinned LXD snap), `~/bin` | the node build recipe + cluster DNS/DHCP |
| `run_on_cluster-scripts/` | the ~90 operational scripts that had only a README committed | now also pushed to the `run_on_cluster` repo |
| `calibration23/` | full local-mods patch + original notebooks (with outputs) | the container's uncommitted live edits |

⚠ **This archive is on h23's own `/dataz`.** It protects against losing
`lxd110maas`, `dsa110maas`, a node, or h23's root NVMe — but **not** against
losing h23 itself or a second `dataz` disk. Getting a copy off-site is the
single most valuable next step.

---

## 6. Open gaps (2026-07-29)

1. No off-site copy of anything, including the new DR archive.
2. `dataz` is raidz1 on six same-age 14 TB disks — one failure from the edge.
3. h23 root NVMe has no redundancy; both LXD containers live on it with
   **zero snapshots**.
4. Credentials (§3.2) exist in exactly one place each, and some need
   hardening/rotation — tracked in the private inventory, not here.
5. Conda envs are snapshots-of-a-snapshot; `casa38` may not solve from scratch.
6. `lxd110maas` is unmanaged and undiscoverable (absent from both MaaS and
   DNS, referenced only by hardcoded IP), and the artifact store it serves has
   no access control. See the private inventory for the follow-up actions.
7. MaaS postgres has no *scheduled* backup — only the manual dump in §5.
8. `dsart_c2` is not `enable`d; `/media/ubuntu/data` is not in fstab; sysctl
   tuning does not persist.
9. `etcdWx` (weather → etcd `/mon/wx` → influx `wxmon`) ran on the old h20 and
   **no binary, unit or source for it survives** anywhere we have looked — not
   in the staged install artifacts, not on the upstream artifact server. `wxmon`
   stays empty until it is rebuilt. (It was already crash-looping before the
   outage, so this is not a regression.)
10. Several `dsa110-cnf` YAMLs are **uncommitted** in git
    (`config_dsa96_corr.yaml`, `config_dsa96_search.yaml`, `corr_setup_96.yaml`,
    `config_dsa64_search_normal.yaml`), so the live legacy configs exist only on
    disk. There is also no recorded mapping from `/cnf/<key>` to source YAML —
    `dsautils/conf/cnfConfig.yml` lists the keys but not their files.

---

## 7. Rebuilding the monitoring host (lxd110h20)

Done 2026-07-30 by promoting the spare `lxd110h24` after the original h20's
NVMe died. **Full procedure, fixed start scripts, systemd units and configs are
in `dsa110-shell/deploy/h20-services/`** — read that README first; it documents
six real defects in the container-era scripts.

Outline:

1. **Retire the dead machine in MaaS**: exit rescue mode → release (with disk
   erasing **off**; `enable_disk_erasing_on_release` is already `false`) →
   rename aside (`lxd110h20-deadnvme`) rather than delete, so its hardware,
   power and MAC inventory survive. MaaS hostnames are unique, so the rename is
   what frees the name.
2. **Rename the replacement** to `lxd110h20` in MaaS, then `hostnamectl` +
   `/etc/hosts` on the host. `preserve_hostname: false`, so cloud-init would
   also set it from MaaS metadata on the next boot.
3. **Rebuild the DNS aliases as CNAMEs** onto one literal A record (see §2.5).
4. **Authorise h23's key.** Only h23's *passphrase-less* key works
   non-interactively — `id_ed25519` is passphrase-protected, so BatchMode SSH
   and any cron/tooling path silently fails with `Permission denied (publickey)`
   even though the server accepts the key. Also clear the old host key:
   `ssh-keygen -R lxd110h20.pro.pvt`, or every connection aborts with
   `POSSIBLE DNS SPOOFING DETECTED`.
   ⚠ Append with a leading newline — the shipped `authorized_keys` had **no
   trailing newline**, so a naive `echo >>` corrupts the last key *and* yours.
5. **Install the services** per the `deploy/h20-services/` README.
6. **Repopulate etcd**: `python tools/ops/push_dsart_to_etcd.py --instance all`
   for `/cnf/pipeline_rt` and `/cnf/search_rt`. Round-trip differs only by
   integer dict keys becoming strings through JSON, which consumers cast back
   (`c3.py`: `corr[int(chg)]`). `/cnf/inject/active/*`, `/cnf/spectral_line` and
   `/cnf/c2/*` are runtime/operator state, not version-controlled config, and
   default correctly when absent.
7. **Grafana**: the tarball ships a 2019 `grafana.db`, so the legacy dashboards
   (`DSA110-WX`, `Antenna Monitor`, `DSA119`, `Etcd`, `calibration`, …) survive
   a total host loss — but the instance does **not** start at `admin/admin`, and
   its datasources point at the long-dead `dsahead.ovro.pvt`. Repoint the
   `InfluxDB` datasource at `127.0.0.1:8086` and set a new admin password
   (`GRAFANA_AUTH` in `~/.dsart/secrets.env` on h23). The `dsartRtMpV1`
   dashboard re-imports from
   `tools/dashboard/dsart_rt_to_influx/grafana/dsart_rt_dashboard.json`.

   **Since 2026-08-03 you should not need the 2019 tarball.** That
   `grafana.db` was seven years stale — every dashboard edit since then
   lived only in the live sqlite DB on one host, and `dsartRtMpV1` was the
   sole board under version control. `tools/ops/backup_grafana.sh` now runs
   nightly from h23 (10:30 UTC, user crontab, `logger -t backup_grafana`)
   and writes to `/dataz/dsa110/dr_archive/grafana/<UTC-stamp>/` with a
   `latest` symlink and 14-day retention:

   | artifact | restores |
   | --- | --- |
   | `dashboards/<uid>__<slug>.json` | one file per dashboard, via `POST /api/dashboards/db` |
   | `datasources.json` | datasource definitions (passwords redacted by Grafana) |
   | `grafana_db.sql.gz` | everything — users, orgs, prefs, alerts, datasource secrets: `zcat … \| sqlite3 grafana.db` |

   Per-run restore commands are written into each backup's `MANIFEST.txt`.
   It runs on h23 rather than h20 on purpose: h23 is where `GRAFANA_AUTH`
   lives, `/dataz/…/dr_archive` is not writable from inside the h20
   container (LXD uid mapping), and a backup kept on the host being backed
   up is not a backup. The tree is `0700` — the sql dump carries datasource
   secrets and the admin password hash.

   Note h20 has Python 3.6 and no `sqlite3` CLI, so the dump uses
   `iterdump()` on a read-only handle inside a read transaction; neither
   `VACUUM INTO` (SQLite ≥ 3.27, h20 has 3.22) nor
   `sqlite3.Connection.backup()` (Python ≥ 3.7) is available there. If the
   ssh to h20 fails the dashboards are still captured and the run still
   succeeds — only the `db_dump: no` line in the manifest changes.
