# Migrating `calibration23` → bare-metal h23 systemd user services

**Status: DESIGN / DRAFT. Nothing in this directory is installed. No service
has been started, stopped, enabled or disabled. The container has not been
modified. No git operation was performed.**

Prepared 2026-07-29 on `lxd110h23`. Everything below was verified by running
read-only commands against both the container and the host; the "Verified"
notes say how.

> **etcd is DOWN right now.** `etcdv3service.pro.pvt:2379` does not resolve
> and 10.41.0.117:2379 is unreachable from h23. *Every one of these services
> is a pure etcd client* — `dsacalib.config.Configuration()` calls
> `cnf.Conf(use_etcd=True)` **and** `etcd.get_dict("/mon/snap/1/armed_mjd")`
> in its constructor, and `Conf.get()` does **not** fall back to the
> hardcoded defaults in `dsautils/cnf.py` (it does `etcd.get(key)[0]` then
> `json.loads`). So **all four services raise on startup while etcd is
> down**, and end-to-end verification (§6) is impossible until it returns.
> The offline preparation steps (§4 steps 1–5) can all be done now.

---

## 1. What is being migrated

Four workloads currently live in the `calibration23` LXD container (Ubuntu
16.04, EOL). They run under `~/.config/systemd/user/` **inside** the
container, but in practice the two long-running ones are started by hand in
`screen`, so they are lost on every container restart.

*Verified 2026-07-29: `screen -ls` in the container reports "No Sockets
found" and `ps aux` shows no python — **all of these are currently down**,
independently of the etcd outage.*

| Container unit | Script | New h23 unit |
| --- | --- | --- |
| `calibration_preprocessing.service` | `services/preprocess_service.py` | `dsa110-calib-preprocess.service` |
| `calibration.service` | `services/calibration_service.py` | `dsa110-calib-calibration.service` |
| `bfweights_copy.service` | `services/beamformerweights.py` | `dsa110-calib-bfweights.service` (aliased `bfweights_copy.service`) |
| crontab `0 */8 * * *` | `scripts/trigger_field_ms.py` | `dsa110-calib-fieldms.timer` + `.service` |

Not migrated, and out of scope: `realtime_calibration.service`,
`continuum_triggers.service`, `T2.service`, `injection.service`,
`sefd.service`, the dask units. **`realtime_calibration.py` matters
though** — see §1.4.

### 1.1 `preprocess_service.py` — hdf5 pull + transit trigger

* **Entry point:** `python preprocess_service.py`. **No CLI arguments.**
* **Watches:** etcd key `/cmd/cal`, callback filters for `cmd == "rsync"`.
  Payload `val` supplies `hostname` and `filename`.
* **Pipeline:** four `multiprocessing.Queue`s feeding three child processes —
  `rsync_handler` → `gather_files` → `assess_file`.
  1. `rsync` — builds `"{hostname}.pro.pvt:{filename} {CONFIG.hdf5dir}/"` and
     calls `dsacalib.preprocess.rsync_file`, which shells out to
     `. ~/.keychain/calibration-sh; rsync -avv --remove-source-files …`.
     Files matching `*spl*` are dropped after transfer.
  2. `gather` — buckets up to `MAX_ASSESS=4` timestamps concurrently, waits up
     to 15 min for all `ncorr` (=16) subband files of one timestamp.
  3. `assess` — reads `phase_center_dec` from the hdf5 header, calls
     `update_caltable(pt_dec)`, and if a calibrator transit falls inside the
     file's LST span emits `(calname, flist)`.
* **Writes:** hdf5 files into `CONFIG.hdf5dir`; etcd `/cmd/cal`
  `{"cmd": "calibrate", …}`; heartbeats `/mon/cal/{rsync,gather,assess}_process`
  and `/mon/service/calpreprocess` every 60 s.
* **External deps:** etcd; ssh/rsync **from** the corr nodes; the shared
  `hdf5_dir`; syslog via `dsautils.dsa_syslog`. No CASA.

### 1.2 `calibration_service.py` — measurement sets + calibration

* **Entry point:** `python calibration_service.py`. **No CLI arguments.**
* **Watches:** etcd `/cmd/cal`, filters `cmd == "calibrate"`; `val` carries
  `calname` and `flist`.
* **Does:** `convert_calibrator_pass_to_ms` → `{msdir}/{date}_{calname}.ms`
  (skipped if it already exists) → `calibrate_measurement_set` (CASA
  delay/bandpass/gain, `refants=['103']`) → `caltable_to_etcd` →
  `write_beamformer_solutions` into `beamformer_dir` → averages the last 24 h
  of solutions (`get_good_solution` / `filter_beamformer_solutions` /
  `average_beamformer_solutions`) → writes
  `{beamformer_dir}/beamformer_weights_{caltime.isot}.yaml` → publishes
  `/mon/cal/bfweights {"cmd": "update_weights"}`.
* **Also writes:** a summary PDF `{tempplots}/{date}_{calname}.pdf` and
  `{tempplots}/{caltime}_averagedweights.png` / `_phase.png`.
  All the `store_file(...)` calls that used to copy these to `webplots` are
  **commented out** in the current code, so `config.webplots` is computed but
  never used (see §2, row `/operations/webPLOTS`).
* **External deps:** etcd, CASA (`casatools`/`casatasks` 6.4.4.31), the shared
  `msdir` / `hdf5_dir` / `beamformer_dir`, matplotlib (Agg). No ssh.

### 1.3 `beamformerweights.py` — weight distribution (on demand)

* **Entry point:** `python beamformerweights.py`. **No CLI arguments.**
* **Watches:** etcd `/mon/cal/bfweights`, filters `cmd == "update_weights"`.
* **Does:** for each of the 16 `corr` `ch0` keys, rsyncs
  `{beamformer_dir}/{weight_files[i]}` → `{corr}.pro.pvt:{weightfile}` and
  `antenna_flags.txt` → `{corr}.pro.pvt:{flagfile}`, then archives a `.dat`
  copy plus a fleet YAML into `bfarchivedir`. Heartbeats
  `/mon/service/bfweightcopy`.
* **⚠ Writes `antenna_flags.txt` to a RELATIVE path** (i.e. `$CWD`) and then
  rsyncs that relative name. The working directory is load-bearing; all
  units here pin `WorkingDirectory=/home/ubuntu/servicewd` (exists on h23,
  verified).
* **Lifecycle:** started/stopped on demand by
  `~/dsa-notebooks/update_bfweights.py`, which literally runs
  `os.system('systemctl --user start bfweights_copy.service')`, sleeps 63 s,
  then stops it. That operator script is in turn driven by the dashboard's
  **"Update cals"** button — see §5.
* **`bfarchivedir` = `/operations/beamformer_weights/applied/` is
  operationally load-bearing:** `dsa110-rt`'s
  `tools/dashboard/dsa_monitor/cal_visibility.py` treats the newest fleet
  YAML there as ground truth for "last distributed solution".

### 1.4 `trigger_field_ms.py` — 8-hourly field MS trigger

* 23 lines. Puts `/cmd/cal {"cmd": "field", "val": {"trigname":
  "field<HH:MM:SS>", "mjds": <now>}}` and exits. No arguments.
* **🔴 The cron job is BROKEN and has been for some time.** The crontab runs
  `…/proj/dsa110-shell/dsa110-calib/scripts/trigger_field_ms.py`, which
  **does not exist** (verified: `ls` → "No such file or directory"; that
  scripts dir contains only `create_long_ms.py` and
  `remove_old_corrdata.py`). The only copy is
  `/home/ubuntu/dana/code/dsa110-calib/scripts/trigger_field_ms.py`.
  Cron mails nobody, so this has been failing silently every 8 hours.
* **🔴 Nothing being migrated consumes `cmd == "field"`.** `preprocess`
  returns unless `cmd == "rsync"`; `calibration` returns unless
  `cmd == "calibrate"`. The consumer is `realtime_calibration.py` (container
  `realtime_calibration.service`, from the separate `~/dana/code` checkout),
  which is **not** in scope here. Decide before enabling the timer: either
  migrate `realtime_calibration.py` too, or leave the timer disabled and
  accept that the trigger is a no-op put (which is the status quo anyway).

---

## 2. Path portability: container → h23

The container has three bind mounts (verified via `lxc config show`):

| Container path | h23 source | Type |
| --- | --- | --- |
| `/candidates` | `/dataz/dsa110/candidates` | disk |
| `/operations` | `/dataz/dsa110/operations` | disk |
| `/home/ubuntu/data` | `/home/ubuntu/data` | disk (**identical path**) |

**`/operations` and `/candidates` do not exist as paths on h23** (verified).
This is the single biggest source of silent breakage.

### 2.1 Live config values

These come from etcd `/cnf/cal` and `/cnf/corr`, pushed from
`/home/ubuntu/proj/dsa110-shell/dsa110-cnf/{config_cal.yaml,corr_setup_96.yaml}`
via `push_to_etcd.py`. **They are NOT the `dsautils/cnf.py` defaults** — those
name `/home/ubuntu/caldata/...`, a directory that does not exist on either
host, and `Conf.get()` never falls back to them.

| Config key | Container value | **h23 equivalent** | Evidence |
| --- | --- | --- | --- |
| `cal.msdir` | `/operations/calibration/` | `/dataz/dsa110/operations/calibration/` | 13 223 entries, newest `2026-07-25_1459+716.ms` |
| `cal.hdf5_dir` | `/operations/correlator/` | `/dataz/dsa110/operations/correlator/` | 16 282 files, newest `2026-07-25T10:35:44_sb02.hdf5` |
| `cal.beamformer_dir` | `/operations/beamformer_weights/generated/` | `/dataz/dsa110/operations/beamformer_weights/generated/` | 10 184 files, newest `…2026-07-24T20:26:52.yaml` |
| `cal.bfarchivedir` | `/operations/beamformer_weights/applied/` | `/dataz/dsa110/operations/beamformer_weights/applied/` | 328 files, newest `…2026-07-24T12:28:21.dat` |
| `cal.weightfile` | `/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out` | *remote path on corr nodes — unchanged* | verified present on n03/n04/n06/n21/n22 |
| `cal.flagfile` | `/home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat` | *remote path on corr nodes — unchanged* | verified present on n01 |
| `cal.caltable` | `/home/ubuntu/proj/dsa110-shell/dsa110-calib/dsacalib/data/calibrator_sources.csv` | same path on h23 — **does not exist on either host** | see §3.4 |
| `Configuration.tempplots` | `/home/ubuntu/data/webPLOTS/calibration/` | **identical** (`/home/ubuntu/data` is a 1:1 bind mount) | — |
| `Configuration.webplots` | `/operations/webPLOTS/calibration/` | `/dataz/dsa110/operations/webPLOTS/calibration/` | exists; **only referenced by commented-out code** |

### 2.2 🚩 Things that would silently write to the wrong place

1. **`/operations/...` is hardcoded in `dsacalib/config.py`** for `webplots`
   (`self.webplots = "/operations/webPLOTS/calibration/"`). It is currently
   dead code, but if anyone re-enables the `store_file(...)` calls in
   `calibration_service.py`, on h23 it would target a non-existent absolute
   path. **Mitigation:** create a compatibility symlink (§4 step 2) rather
   than editing `dsacalib`. That also future-proofs anything else in the
   `dsacalib`/`dsamfs` stack that assumes container-relative `/operations`.
2. **`tempplots = /home/ubuntu/data/webPLOTS/calibration/` does not exist** —
   on either host, because it is the same directory. The existing PNGs sit
   one level up in `/home/ubuntu/data/webPLOTS/`. `generate_summary_plot`'s
   `PdfPages(...)` therefore raises, and the failure is swallowed by
   `exception_logger(..., throw=False)`. This is a *pre-existing* bug, not a
   migration regression, but fix it while you are here (§4 step 2).
3. **`antenna_flags.txt` is relative** — see §1.3. If `WorkingDirectory` is
   ever dropped from the bfweights unit it will ship a stale or empty flag
   file to all 16 corr nodes without erroring.
4. **CASA writes `casa-<timestamp>.log` into `$CWD`.** In the container this
   accumulated ~200 GB of logs in the `services/` directory (largest single
   file 49 GB). `WorkingDirectory=/home/ubuntu/servicewd` contains it; add
   logrotate or a sweeper if you care.
5. **`rsync_file(..., remove_source_files=True)` is the default** in the
   preprocess path. Getting `hdf5dir` wrong does not merely misplace data —
   it **deletes the source hdf5 on the corr node** after a transfer to the
   wrong destination. Confirm `/cnf/cal.hdf5_dir` before first start.

---

## 3. Dependency check: can h23's `casa38` run these services?

**Headline: yes. Every import the four scripts need already resolves in
h23's `/home/ubuntu/anaconda3/envs/casa38`. There are no missing packages.**

Verified by running the same import probe under both interpreters. Both are
Python 3.8.13. All of `numpy scipy astropy h5py pandas yaml matplotlib
pyuvdata numba dask distributed etcd3 structlog casatools casatasks casacore
casadata antpos dsamfs`, all of `dsautils.{dsa_store,dsa_syslog,cnf,calstatus}`
and all of `dsacalib.{constants,config,preprocess,utils,routines,ms_io,
hdf5_io,weights,plotting,calib,fringestopping,fits_io}` import cleanly on h23.

The only import that fails is `dsautils.coordinates`, and it fails
**identically on both** with `ConnectionFailedError: etcd connection failed`
— that is the etcd outage, not a packaging gap. None of the four migrated
scripts import it.

### 3.1 Version drift (informational, no action strictly required)

| Package | container `casa38` | h23 `casa38` |
| --- | --- | --- |
| `dsa110-calib` | 3.0.0+98.g5f6ae4d | 3.0.0+98.g7c49bd4 |
| `dsa110-pyutils` | 3.8.2+15.ga68f464 | 3.8.2+15.gca59f1a |
| `pyuvdata` | **101.0.0** | **2.2.9.dev23+g40809cff** |
| `dsa110-meridian-fs` | 1.6.9 | 1.7.0 |
| `dsa110-antpos` | 1.4.4 | 1.4.0 |
| `scipy` | 1.10.1 | 1.7.3 |
| `numba` | 0.56.4 | 0.55.1 |
| `distributed` | 2023.3.2.1 | 2022.2.1 |
| `casatools`/`casatasks`/`casadata` | 6.4.4.31 / 2022.1.17 | **identical** |

The `pyuvdata` gap is the one to keep an eye on: `dsacalib.ms_io`
(`convert_calibrator_pass_to_ms`, `uvh5_to_ms`) is the heaviest pyuvdata
consumer. It **imports** fine on h23, but a major-version difference is not
exercised by an import. Treat the first `.ms` produced on h23 as the real
test (§6 step 3) and diff it against a container-produced one.

### 3.2 🔴 The installed `dsacalib` trees are NOT identical

Diffing the two installed packages, the Python is essentially the same — only
two files differ — but one of them matters:

* **`constants.py`**: container has `ovro_loc = me.observatory("OVRO")`, h23
  has `me.observatory("OVRO_MMA")` (h23 checkout commit 9c35b24, "OVRO ->
  OVRO_MMA"). **On h23's casadata, `me.observatory("OVRO")` returns an empty
  dict and logs `SEVERE Unknown observatory asked for`; only `OVRO_MMA`
  returns real coordinates** (verified). ⇒ **Keep h23's installed
  `dsacalib`. Do NOT overwrite it with the container's copy** or you will
  silently lose the observatory position.
* `uvh5_to_ms.py`: 4 lines differ.
* `T3imaging.py` exists only on h23.

### 3.3 🟡 Calibrator source tables: 204 in the container vs 5 on h23

`dsacalib/data/calibrator_sources_dec<+/-><DD>p<D>.csv` — the container has
204 pre-generated per-declination tables, h23 has 5.

This is **self-healing but slow and side-effecting**: `update_caltable()`
calls `generate_caltable()` when the table for the current pointing dec is
absent, which reads the bundled `data/vlacalibrators.txt` (present on h23,
761 KB — **no network access needed**) and writes the new CSV **into the
installed package directory**. h23's `…/site-packages/dsacalib/data` is
`ubuntu`-owned and writable, so this works.

Consequence: on first run at each new declination the assess step does a
noticeably slower O(N²) pass instead of a CSV read. Optional pre-warm: copy
the 204 CSVs from the container (§4 step 5). They are plain generated data —
copying them is safe and does not touch code.

### 3.4 🟡 `cal.caltable` points at a file that does not exist

`/home/ubuntu/proj/dsa110-shell/dsa110-calib/dsacalib/data/calibrator_sources.csv`
is absent on h23 (only a stale copy under `build/lib/`) and absent in the
container. Nothing in the four migrated scripts reads `cal.caltable` — they
all go through `update_caltable()`, which uses `resource_filename("dsacalib",
…)` against the *installed* package instead. So this is a latent config wart,
not a blocker. Do not "fix" it by pointing it at the installed package
without checking what else reads it.

### 3.5 🟡 `antennas_not_in_bf` may be `None`

`config_cal.yaml` has the key present with **no value**. It flows into
`write_beamformer_solutions(..., flagged_antennas=config.antennas_not_in_bf)`.
`dsautils/cnf.py` defines a 30-element default that is never used (no
fallback). Either etcd holds a hand-edited list that was never committed back
to `dsa110-cnf`, or this is a live bug. **Read the actual value out of etcd
the moment it comes back** (`/cnf/cal`) and reconcile before cutover — this
one cannot be checked offline.

### 3.6 Script versions differ between container and h23 checkout

The h23 `dsa110-calib` checkout is **not** a drop-in replacement for the
container's live working tree. The container has uncommitted modifications.

| Script | Status |
| --- | --- |
| `calibration_service.py` | **identical** |
| `preprocess_service.py` | **h23's copy is a different, older design** — uses `FSCRUNCH_Q`, and its `populate_queue(etcd_dict, queue=GATHER_Q, hdf5dir=…, subband_def=CONFIG.ch0)` skips the rsync stage entirely. The container's version is the live one. |
| `beamformerweights.py` | h23's targets `{corr}.sas.pvt`; the container's (uncommitted edit) targets `{corr}.pro.pvt`. **`.pro.pvt` is live** (see §4 step 1 / §5). |
| `trigger_field_ms.py` | **absent from h23** entirely. |

⇒ The cutover must copy the container's live scripts (§4 step 1). Doing so
puts uncommitted container edits into the h23 git checkout — **that is a git
working-tree change; get the operator's explicit sign-off** and decide
whether to commit them or stage them outside the repo.

---

## 4. Cutover procedure

Steps 1–5 are offline-safe and can be done now, while etcd is down. Steps
6–8 require etcd.

### Step 0 — prerequisites checklist

* [x] `casa38` env at `/home/ubuntu/anaconda3/envs/casa38` (Python 3.8.13) — present
* [x] `/home/ubuntu/servicewd` — present
* [x] `ubuntu` has `Linger=yes` (verified via `loginctl show-user ubuntu`) so
      user units survive logout
* [x] h23 can SSH key-only (BatchMode) to the corr nodes — verified against
      10.42.0.199 / 10.42.0.200 / 10.41.0.228, returning `lxd110h03`,
      `lxd110h22`, `lxd110h18`
* [ ] etcd reachable — **currently NO**
* [ ] Operator sign-off on touching the h23 `dsa110-calib` git working tree (§3.6)

### Step 1 — sync the live scripts onto h23

Copy from the container's working tree (source of truth), **not** from git:

```
lxc file pull calibration23/home/ubuntu/proj/dsa110-shell/dsa110-calib/services/preprocess_service.py   /home/ubuntu/proj/dsa110-shell/dsa110-calib/services/
lxc file pull calibration23/home/ubuntu/proj/dsa110-shell/dsa110-calib/services/beamformerweights.py    /home/ubuntu/proj/dsa110-shell/dsa110-calib/services/
lxc file pull calibration23/home/ubuntu/proj/dsa110-shell/dsa110-calib/services/calibration_service.py  /home/ubuntu/proj/dsa110-shell/dsa110-calib/services/
# fixes the broken cron path at the same time (§1.4):
lxc file pull calibration23/home/ubuntu/dana/code/dsa110-calib/scripts/trigger_field_ms.py              /home/ubuntu/proj/dsa110-shell/dsa110-calib/scripts/
```

Back up the h23 originals first. Do **not** touch
`site-packages/dsacalib` (§3.2).

### Step 2 — create the missing directories and the `/operations` shim

```
sudo ln -s /dataz/dsa110/operations  /operations     # §2.2 item 1
sudo ln -s /dataz/dsa110/candidates  /candidates     # parity with the container
mkdir -p /home/ubuntu/data/webPLOTS/calibration      # §2.2 item 2
```

The two symlinks make every hardcoded container path valid on h23 and mean
**zero edits to `dsacalib`**. Prefer them to patching the library.

### Step 3 — 🔴 add the 16 corr-node host entries to `/etc/hosts`

**This is the top hidden blocker.** `beamformerweights.py` rsyncs to
`{ch0_key}.pro.pvt`, and the live `/cnf/corr` `ch0` keys are **`lxd110hNN`
hostnames, not `corrNN`**. Those names resolve **only** because the container
carries 16 static `/etc/hosts` entries — they are **not in DNS**:
querying the same resolver (10.42.0.4) from h23 returns `NXDOMAIN` for
`lxd110h03.pro.pvt`, while the container resolves it to 10.42.0.199.

Without this step, weight distribution fails from h23 — and because
`rsync_file()` never raises (it only captures stdout), it fails **silently**
while still writing the `applied/` archive, so the dashboard would report a
distribution that never happened.

Append to h23's `/etc/hosts` (copied verbatim from the container; note
h18 is on the 10.41 subnet, not 10.42):

```
10.42.0.199 lxd110h03.pro.pvt
10.42.0.205 lxd110h04.pro.pvt
10.42.0.206 lxd110h05.pro.pvt
10.42.0.241 lxd110h06.pro.pvt
10.42.0.208 lxd110h07.pro.pvt
10.42.0.210 lxd110h08.pro.pvt
10.42.0.214 lxd110h10.pro.pvt
10.42.0.216 lxd110h11.pro.pvt
10.42.0.218 lxd110h12.pro.pvt
10.42.0.223 lxd110h14.pro.pvt
10.42.0.226 lxd110h15.pro.pvt
10.42.0.227 lxd110h16.pro.pvt
10.41.0.228 lxd110h18.pro.pvt
10.42.0.201 lxd110h19.pro.pvt
10.42.0.251 lxd110h21.pro.pvt
10.42.0.200 lxd110h22.pro.pvt
```

Verify all 16 afterwards:
`for h in 03 04 05 06 07 08 10 11 12 14 15 16 18 19 21 22; do ssh -o BatchMode=yes lxd110h$h.pro.pvt hostname; done`
— each must print `lxd110h<NN>`.

### Step 4 — SSH identity for rsync

`dsacalib.preprocess.rsync_file()` hardcodes
`. ~/.keychain/calibration-sh; rsync -avv …`.

* In the container, `~/.keychain/calibration-sh` is a symlink to
  `calibration23-sh`, a keychain-managed file exporting `SSH_AUTH_SOCK` /
  `SSH_AGENT_PID`. It was written 2022-06-16 and points at a long-dead agent,
  so in practice rsync is already falling back to the on-disk key.
* The container authenticates with its on-disk key in `~/.ssh/`. Key details
  (type, fingerprint, passphrase status) are recorded in the private notes
  rather than here, since this repo is public.
* h23 already has an equivalent keychain file at `~/.keychain/lxd110h23-sh`
  and a usable key. **h23's existing key is already
  authorized on the corr nodes** — verified by a BatchMode SSH to all three
  sampled corr IPs.

So h23 needs **no new key material**. Only the hardcoded *filename* has to
resolve, otherwise `sh` prints "No such file or directory" on every rsync
(harmless, since `;` continues to the rsync, but it pollutes the log):

```
ln -s lxd110h23-sh /home/ubuntu/.keychain/calibration-sh
```

Do **not** copy the container's private key to h23; it is unnecessary.
If you later want the corr nodes to accept *only* a dedicated identity, add
`RSYNC_RSH=ssh -i /home/ubuntu/.ssh/<key> -o BatchMode=yes` to the
EnvironmentFile instead.

### Step 5 — (optional) pre-warm the calibrator tables

```
mkdir -p /tmp/caltables && lxc file pull -r \
  calibration23/home/ubuntu/anaconda3/envs/casa38/lib/python3.8/site-packages/dsacalib/data /tmp/caltables/
cp -n /tmp/caltables/data/calibrator_sources_dec*.csv \
  /home/ubuntu/anaconda3/envs/casa38/lib/python3.8/site-packages/dsacalib/data/
```

`cp -n` so h23's existing tables win. Skip the two `#…#` Emacs autosave files.
Purely a warm-up (§3.3).

### Step 6 — install the units (requires etcd for a meaningful start)

```
mkdir -p /home/ubuntu/.config/dsa110-calib
cp /home/ubuntu/vikram/dev/_dr_staging/calib-migration/dsa110-calib.env \
   /home/ubuntu/.config/dsa110-calib/
cp /home/ubuntu/vikram/dev/_dr_staging/calib-migration/dsa110-calib-*.service \
   /home/ubuntu/vikram/dev/_dr_staging/calib-migration/dsa110-calib-*.timer \
   /home/ubuntu/.config/systemd/user/
systemctl --user daemon-reload
```

### Step 7 — stop the container side FIRST (avoid double-running)

Both stacks are etcd watchers on the same keys. Running both means two
rsyncs of the same hdf5 (with `--remove-source-files`!) and two competing
`.ms` builds. **Never overlap.**

```
lxc exec calibration23 -- sudo -u ubuntu XDG_RUNTIME_DIR=/run/user/1000 \
  systemctl --user disable --now calibration_preprocessing.service calibration.service
lxc exec calibration23 -- sudo -u ubuntu crontab -l > /tmp/calib23-crontab.bak   # keep a copy
# then remove the trigger_field_ms line from the container crontab
```

(They are already stopped today — see §1 — so in the current state this step
is just about the crontab and making sure they do not come back.)

### Step 8 — start the h23 units, one at a time

```
systemctl --user enable --now dsa110-calib-preprocess.service
# watch it settle, then:
systemctl --user enable --now dsa110-calib-calibration.service
# on-demand only; `enable` here just materialises the bfweights_copy alias:
systemctl --user enable dsa110-calib-bfweights.service
# only if a consumer for cmd=="field" exists (§1.4):
systemctl --user enable --now dsa110-calib-fieldms.timer
```

---

## 5. Dashboard integration — `dsa110-rt` change required

`/home/ubuntu/vikram/dev/dsa110-rt/tools/dashboard/dsa_monitor/bfweights_update.py`
powers the `/sefds` **"Update cals"** button, and it shells **into the
container**:

```python
return ["lxc", "exec", CONTAINER, "--", "sudo", "-u", "ubuntu",
        "XDG_RUNTIME_DIR=/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
        "bash", "-c", inner]
```

with `CONTAINER = "calibration23"`,
`CONTAINER_WORKDIR = "/home/ubuntu/dsa-notebooks"`,
`CONTAINER_PYTHON = "/home/ubuntu/anaconda3/envs/casa38/bin/python"`.

**Retiring the container breaks this button.** It must be changed to run
`update_bfweights.py` locally on h23. Also note `update_bfweights.py` itself
hardcodes `systemctl --user start bfweights_copy.service` — the
`Alias=bfweights_copy.service` in `dsa110-calib-bfweights.service` is exactly
what keeps that line working unmodified, provided you `systemctl --user
enable` that unit so the alias symlink is created.

You will also need `~/dsa-notebooks/update_bfweights.py` on h23 (verify
whether h23's copy matches the container's before relying on it).

---

## 6. Verification

**None of §6.2–§6.4 can be done until etcd returns.**

### 6.1 Offline (possible now)

```
# imports resolve
/home/ubuntu/anaconda3/envs/casa38/bin/python -c \
  "import dsacalib.routines, dsacalib.ms_io, dsacalib.weights, dsautils.dsa_store; print('ok')"
# observatory position is real (must NOT be an empty dict)
/home/ubuntu/anaconda3/envs/casa38/bin/python -c \
  "from casatools import measures; print(measures().observatory('OVRO_MMA'))"
# all 16 corr nodes reachable by the name beamformerweights.py will use
for h in 03 04 05 06 07 08 10 11 12 14 15 16 18 19 21 22; do \
  printf '%s ' $h; ssh -o BatchMode=yes -o ConnectTimeout=5 lxd110h$h.pro.pvt hostname; done
# units parse
systemd-analyze --user verify /home/ubuntu/.config/systemd/user/dsa110-calib-*.service
```

### 6.2 Config sanity, the moment etcd is back

```
/home/ubuntu/anaconda3/envs/casa38/bin/python -c \
  "from dsacalib.config import Configuration; print(Configuration())"
```

Confirm `msdir`, `hdf5dir`, `beamformer_dir` are the `/operations/...` values
in §2.1 (they will read as `/operations/...`; the §4 step 2 symlink makes
them valid), `refants == ['103']`, `ncorr == 16`, and **check
`antennas_not_in_bf` against §3.5**.

### 6.3 Per-service checks

| Service | Healthy signal |
| --- | --- |
| preprocess | `/mon/service/calpreprocess` heartbeat advancing every 60 s; `/mon/cal/{rsync,gather,assess}_process` show `ntasks_alive: 1`; new hdf5 landing in `hdf5_dir`; journal prints the `RSYNCING with command:` lines |
| calibration | `/mon/service/calibration` heartbeat; journal prints `N objects in calibration queue` every 5 min; on a transit, a new `{date}_{calname}.ms` under `msdir` and `/mon/cal/calibration` `status` flipping from `-1` to a real value |
| bfweights | `/mon/service/bfweightcopy` heartbeat while running; after a distribution, a new fleet YAML + 16 `.dat` in `applied/`; **and the corr nodes' `antennas.out` mtime actually advances** (see below) |
| fieldms | `systemctl --user list-timers dsa110-calib-fieldms.timer`; journal shows a clean exit 0 |

**The bfweights end-to-end check that actually matters** (because
`rsync_file` cannot fail loudly):

```
for h in 03 04 05 06 07 08 10 11 12 14 15 16 18 19 21 22; do \
  printf '%s ' $h; ssh -o BatchMode=yes lxd110h$h.pro.pvt \
  'stat -c %y /home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/antennas.out'; done
```

All 16 must show the new timestamp. Baseline captured 2026-07-29: all
sampled `ch0` members (h03, h04, h06, h21, h22) read
`2026-07-24 12:28:17`, matching the newest `applied/` `.dat` — i.e. the
container's last distribution was clean and complete. (Nodes n01, n02, n09,
n13 sit at `2026-06-03 17:20` — expected: they are **not** in the `ch0`
list of 16 and never receive weights.)

### 6.4 First-`.ms` cross-check

Because of the `pyuvdata` 101.0.0 → 2.2.9 gap (§3.1), compare the first
h23-produced measurement set against a container-produced one for the same
calibrator — antenna count, spw layout, UVW sign convention, and the
resulting bandpass amplitudes. Do this **before** letting an h23-generated
solution be distributed to the array.

---

## 7. Rollback

Fast, because nothing is destructive and the container is untouched.

```
# 1. stop + disable the h23 units
systemctl --user disable --now dsa110-calib-preprocess.service \
                                dsa110-calib-calibration.service \
                                dsa110-calib-fieldms.timer
systemctl --user disable        dsa110-calib-bfweights.service
systemctl --user daemon-reload

# 2. bring the container services back
lxc start calibration23                                  # if it was stopped
lxc exec calibration23 -- sudo -u ubuntu XDG_RUNTIME_DIR=/run/user/1000 \
  systemctl --user enable --now calibration_preprocessing.service calibration.service

# 3. restore the container crontab
lxc exec calibration23 -- sudo -u ubuntu crontab /tmp/calib23-crontab.bak

# 4. revert the dashboard change from §5, if it was made
```

Notes:

* The `/operations` and `/candidates` symlinks, the `/etc/hosts` entries, the
  `~/.keychain/calibration-sh` symlink and the extra calibrator CSVs are all
  **harmless to leave in place** after a rollback. Leave them; they make a
  second attempt cheaper.
* **Do restore the h23 `dsa110-calib` scripts** you overwrote in §4 step 1 if
  anything else on h23 imports from that checkout.
* The only genuinely irreversible risk in the whole plan is
  `rsync --remove-source-files` running against a wrong `hdf5dir` (§2.2
  item 5). Verify §6.2 before starting the preprocess service.

---

## 8. Open questions for the operator

1. **`antennas_not_in_bf`** — is the live etcd value a real list, or `None`
   as `config_cal.yaml` implies? (§3.5) Cannot be answered offline.
2. **`cmd == "field"`** — should `realtime_calibration.py` be migrated too, or
   should the 8-hourly timer stay disabled? (§1.4) Note the cron has been
   broken for a while, so "leave it off" restores no lost functionality.
3. **Container working-tree edits** — commit the `.pro.pvt` and
   `preprocess_service.py` changes to `dsa110-calib`, or keep the deployed
   scripts outside git? (§3.6)
4. **`pyuvdata` 101.0.0 vs 2.2.9** — is h23's older pyuvdata the intended
   long-term target, or should the container's newer one be reproduced in
   h23's `casa38` before cutover? (§3.1)
5. **`cal.caltable`** pointing at a non-existent file (§3.4) — dead config, or
   does something outside this migration read it?
