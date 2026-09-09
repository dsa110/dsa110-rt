# Node provisioning

Rebuild a DSA-110 corr or search node after a MaaS deploy.

Work is split by *kind*, not by convenience:

| | what | why there |
|---|---|---|
| `curtin_userdata_dsa110_node` | packages, user, NTP, sysctl, netplan/MTU | must exist before first boot |
| `provision_node.sh` | CUDA, conda, repos, native builds, dsart | idempotent, re-runnable, dry-runnable |

A cloud-init `per-once` hook runs exactly once, is awkward to re-run,
and fails into a log nobody reads. Everything that can live in a
re-runnable script does, so it can be iterated without a reprovision and
used to repair drift on an existing node.

## Quick start

```bash
# on the freshly deployed node, as root
git clone https://github.com/dsa110/dsa110-rt.git /home/ubuntu/proj/dsa110-rt
cd /home/ubuntu/proj/dsa110-rt/tools/provision

./provision_node.sh --role corr --dry-run                    # ALWAYS first
./provision_node.sh --role corr --apply --log /var/log/dsa-provision.log
./provision_node.sh --role corr --only verify
```

`--dry-run` is the default. There is no default-apply: this runs as root
against a node that may be carrying live buffers.

## Options

```
--role {corr|search}   required
--apply                actually make changes (needs root)
--dry-run              print what would happen (default)
--only  STAGE[,...]    run just these stages
--skip  STAGE[,...]    run everything except these
--list-stages          print the stage names
--log FILE             tee output
```

Stages, in dependency order:

```
preflight tuning mellanox cuda repos conda builds dsart role verify
```

`repos` precedes `conda` because the `dsa110-rt` env is built from the
repo's own `envs/dsa110-rt.yml`. `builds` precedes `dsart` because
dsart's C extensions link against the PSRDADA that `builds` installs.

Any stage can be run on its own — interface detection is lazy, so
`--only role` works without `preflight` having run.

## Verifying an existing node

`verify` mutates nothing, so it doubles as a drift check on a live node:

```bash
./provision_node.sh --role corr --only verify
```

## Files

```
versions.env                    every pinned version, one place
provision_node.sh               driver + dry-run harness
lib/00-preflight.sh             assert the machine looks like a node
lib/10-tuning.sh                sysctl, MTU, /dev/shm
lib/20-mellanox.sh              mlx5_core / firmware tools
lib/30-cuda.sh                  pinned CUDA + driver
lib/50-repos.sh                 dsa110-shell (mr) + dsa110-rt
lib/40-conda.sh                 miniforge + dsa110-rt env
lib/60-builds.sh                psrdada, xGPU, sigproc, xengine, mbheimdall
lib/70-dsart.sh                 editable install + C extensions
lib/80-role.sh                  corr vs search differences
lib/90-verify.sh                assert against MANIFEST.md
curtin_userdata_dsa110_node     MaaS preseed (base install only)
MANIFEST.md                     the fleet baseline this targets
```

## Installing on the MaaS server

The scripts are fetched over HTTP from `http://10.42.0.3/maas/`, served
out of `/var/www/html/maas/` on `dsa110maas`. Mirror them there:

```bash
rsync -av --exclude .git \
  tools/provision/ dsa@10.42.0.3:/var/www/html/maas/config/provision/
```

The curtin preseed goes into the MaaS region controller's preseed
directory as
`curtin_userdata_ubuntu_amd64_generic_bionic_<hostname>` (per machine)
or without the suffix (fleet-wide).

**Confirm that directory before installing.** MaaS on `dsa110maas` shows
neither a deb nor a snap package despite `regiond`/`rackd` running, so
the path has not been established. Installing to a guessed path means
the preseed silently does not apply, and the node comes up looking fine
but unconfigured.

A mirror of whatever is placed on `dsa110maas` is kept at
`/dataz/dsa110/maas-mirror/` on h23.

## Known gaps in the legacy provisioning this replaces

Recorded because each would produce a node that looks provisioned and
is not:

1. `dsa110-rt` absent entirely — the node could not run dsart.
2. `install_repos_c` (psrdada/xGPU/sigproc/xengine builds) was written
   for the LXC corr-container profile and is not referenced by the
   bare-metal preseed, so repos were cloned and nothing was compiled.
3. `apt-get -y install cuda` floats to the newest CUDA rather than the
   11.1 the fleet is built against.
4. anaconda3 was provisioned; miniforge3, which dsart actually uses,
   was not.
5. Several steps fetch from `http://lxd110maas.ovro.pvt/`, which does
   not resolve — those steps fetched nothing and carried on silently.
   Everything here uses the literal IP.
6. LXD cluster join and `lxc launch ubuntu:16.04 corrNN` are legacy;
   dsart runs on bare metal.

## Not handled here

- **Reboot persistence.** `dsart_rt` is not started at boot, matching
  current fleet behaviour.
- **SSH keys / accounts.** MaaS injects registered keys on deploy.
  Nothing key-shaped belongs in this repo.
- **OS upgrade.** 18.04 is EOL, but moving off it means re-validating
  CUDA, the driver and the whole native stack — a separate project, not
  a rebuild.
