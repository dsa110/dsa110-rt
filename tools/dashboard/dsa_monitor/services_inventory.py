"""Pinned fleet-services inventory.

Single source of truth for *which* services live on *which* hosts on
the DSA-110 real-time fleet, and the restart policy for each tier.
Imported by:

* :mod:`fleet_services` (status query + restart-all),
* :mod:`control_store` (audit-log shape),
* :mod:`tests.test_fleet_services` (shape pins).

Tiers
=====

``dsa_monitor_h23``      lxd110h23 systemd --user units: ``dsa_monitor.service``,
                         ``sefd_dashboard.service``. The dashboard itself.
``coincidencer_h23``     lxd110h23 systemd --user units: ``dsart_c2.service``
                         (coincidencer) and ``dsart_c3.service`` (voltage-dump
                         collector).
``support_h23``          lxd110h23 systemd --user units backing observing:
                         calibration preprocess + calibration, the Slack
                         relay, the C2 hiplots and declination.
                         These ran in the calibration23 LXC container until
                         it was retired (2026-07-31).
``grafana_h20``          lxd110h20 systemd --system units: ``etcdv3``,
                         ``influxdb``, ``grafana``. **Never restarted by
                         the dashboard** (see ``H20_HOSTNAMES``).
``dsart_orch_corr``      n03..n22 (16 hosts): ``dsart_rt`` Python process
                         (NOT a systemd unit; spawned by
                         ``tools/ops/_m75_phaseB_16x4_launch.sh`` STAGE 3a).
``dsart_orch_search``    n01,n02,n09,n13 (4 hosts): same ``dsart_rt``
                         process kind as the corr tier but instance
                         ``search_rt``.

Kinds
=====

* ``systemd_user``       — ``systemctl --user is-active <name>``
* ``systemd_system``     — ``systemctl is-active <name>`` (root or
                          system service)
* ``process``            — ``pgrep -af '<pattern>'``; returns 0 if any
                          match, 1 if none
* ``lxc_systemd_user``   — ``lxc exec <container> -- sudo -u ubuntu
                          systemctl --user is-active <name>``

The "Restart all" button on the Control tab walks
:func:`restartable_entries` (which drops anything whose ``host`` is
in :data:`H20_HOSTNAMES`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Tier name constants — used as the ``tier`` field and as keys in the
# restart-all per-tier loop. Pinned strings so the unit tests can lock
# the wire format.
# ---------------------------------------------------------------------------

TIER_DSA_MONITOR_H23: Final[str] = "dsa_monitor_h23"
TIER_COINCIDENCER_H23: Final[str] = "coincidencer_h23"
TIER_SUPPORT_H23: Final[str] = "support_h23"
TIER_GRAFANA_H20: Final[str] = "grafana_h20"
TIER_DSART_ORCH_CORR: Final[str] = "dsart_orch_corr"
TIER_DSART_ORCH_SEARCH: Final[str] = "dsart_orch_search"


# ---------------------------------------------------------------------------
# Kind constants
# ---------------------------------------------------------------------------

KIND_SYSTEMD_USER: Final[str] = "systemd_user"
KIND_SYSTEMD_SYSTEM: Final[str] = "systemd_system"
KIND_PROCESS: Final[str] = "process"
KIND_LXC_SYSTEMD_USER: Final[str] = "lxc_systemd_user"


# ---------------------------------------------------------------------------
# Host constants
# ---------------------------------------------------------------------------

#: Central host this dashboard runs on. Status queries here skip the
#: ssh layer; restart-all reaches the local systemd via
#: ``systemctl --user``.
HOST_H23: Final[str] = "lxd110h23"

#: LXC container on lxd110h23 that hosts the hiplot service. Reached
#: via ``lxc exec calibration23 -- ...`` from lxd110h23 (no ssh).
HOST_CALIBRATION23: Final[str] = "calibration23"

#: The grafana / influx / telegraf central host. Read-only from the
#: dashboard's perspective: status queries are allowed (over ssh)
#: but **never restarted**. See :data:`H20_HOSTNAMES`.
#: FQDN, not the bare name: h23 has no DNS or /etc/hosts entry for
#: "lxd110h20", so the ssh probe failed to resolve it and every row in
#: this tier reported "unknown" with no age. Only ".pro.pvt" is served
#: (see DISASTER_RECOVERY.md §7).
HOST_H20: Final[str] = "lxd110h20.pro.pvt"

#: Hostnames the dashboard MUST NEVER restart. Pinned as a frozenset
#: for O(1) lookup + immutability across the call sites.
H20_HOSTNAMES: Final[frozenset[str]] = frozenset({HOST_H20})


# Mirrors control_store.CORR_CN_IDS / SEARCH_CN_IDS without importing
# them here (control_store has heavier optional deps; the inventory
# stays import-cheap so it can be loaded by tests in isolation).
_CORR_CN_IDS: Final[tuple[int, ...]] = (
    3, 4, 5, 6, 7, 8,
    10, 11, 12,
    14, 15, 16,
    18, 19,
    21, 22,
)
_SEARCH_CN_IDS: Final[tuple[int, ...]] = (1, 2, 9, 13)


def _corr_host(cn: int) -> str:
    """``cn=3`` → ``n03.pro.pvt``. Matches the convention in
    ``tools/ops/_m75_phaseB_16x4_launch.sh``.
    """
    return f"n{int(cn):02d}.pro.pvt"


#: All 16 corr-node FQDNs, chgroup-ordered.
CORR_HOSTS: Final[tuple[str, ...]] = tuple(_corr_host(c) for c in _CORR_CN_IDS)

#: The 4 search-node FQDNs, in (1, 2, 9, 13) order.
SEARCH_HOSTS: Final[tuple[str, ...]] = tuple(
    _corr_host(c) for c in _SEARCH_CN_IDS
)

#: cn-id → FQDN lookup, for both tiers.
CN_TO_HOST: Final[dict[int, str]] = {
    cn: _corr_host(cn) for cn in (*_CORR_CN_IDS, *_SEARCH_CN_IDS)
}


# ---------------------------------------------------------------------------
# Entry shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceEntry:
    """One row of the inventory.

    ``cn_id`` is non-None only for the dsart-orchestrator tiers — it
    lets the re-spawn helper invoke the launch snippet without
    re-deriving the cn-id from the hostname.
    """

    tier: str
    host: str
    service: str           # systemd unit name OR process-name pattern
    kind: str              # one of the KIND_* constants
    cn_id: int | None = None      # set only for dsart_rt tiers
    instance: str | None = None   # "pipeline_rt" or "search_rt"

    def is_restartable(self) -> bool:
        """``False`` for anything on a host in :data:`H20_HOSTNAMES`."""
        return self.host not in H20_HOSTNAMES


# ---------------------------------------------------------------------------
# SERVICE_INVENTORY — the canonical list. Order is the same as it is
# displayed in the dashboard table.
# ---------------------------------------------------------------------------


def _build_inventory() -> tuple[ServiceEntry, ...]:
    rows: list[ServiceEntry] = []
    # Tier 1: dsa_monitor on h23 (this dashboard + the SEFD sidecar).
    rows.append(ServiceEntry(
        tier=TIER_DSA_MONITOR_H23, host=HOST_H23,
        service="dsa_monitor.service", kind=KIND_SYSTEMD_USER,
    ))
    rows.append(ServiceEntry(
        tier=TIER_DSA_MONITOR_H23, host=HOST_H23,
        service="sefd_dashboard.service", kind=KIND_SYSTEMD_USER,
    ))
    # Tier 2: C2 coincidencer + C3 voltage collector on h23.
    for unit in ("dsart_c2.service", "dsart_c3.service"):
        rows.append(ServiceEntry(
            tier=TIER_COINCIDENCER_H23, host=HOST_H23,
            service=unit, kind=KIND_SYSTEMD_USER,
        ))
    # Tier 3: the observing-support services on h23. These moved off the
    # calibration23 LXC container when it was retired (2026-07-31) and are
    # now plain systemd --user units alongside everything else.
    for unit in (
        "dsa110-calib-preprocess.service",
        "dsa110-calib-calibration.service",
        "dsart_slack_relay.service",
        # copydata.service removed 2026-08-19: legacy T1 path, unresolvable
        # source hosts, never functional. See SUPPORT_LOCAL_UNITS.
        "hiplot_c2.service",
        "declination.service",
    ):
        rows.append(ServiceEntry(
            tier=TIER_SUPPORT_H23, host=HOST_H23,
            service=unit, kind=KIND_SYSTEMD_USER,
        ))
    # Tier 4: etcd / influx / grafana on h20. READ-ONLY tier.
    #
    # etcdv3 is the control plane every other tier depends on, so its
    # absence here was the most consequential gap in the table. The unit
    # is "grafana.service" on this host, not "grafana-server.service" --
    # the old name matched nothing and the row could only ever report a
    # not-found state. telegraf is not installed and has been dropped.
    for unit in (
        "etcdv3.service",
        "influxdb.service",
        "grafana.service",
    ):
        rows.append(ServiceEntry(
            tier=TIER_GRAFANA_H20, host=HOST_H20,
            service=unit, kind=KIND_SYSTEMD_SYSTEM,
        ))
    # Tier 5: dsart_rt orchestrator process on each corr node.
    for cn in _CORR_CN_IDS:
        rows.append(ServiceEntry(
            tier=TIER_DSART_ORCH_CORR,
            host=_corr_host(cn),
            service="dsart_rt",
            kind=KIND_PROCESS,
            cn_id=int(cn),
            instance="pipeline_rt",
        ))
    # Tier 6: dsart_rt orchestrator process on each search node.
    for cn in _SEARCH_CN_IDS:
        rows.append(ServiceEntry(
            tier=TIER_DSART_ORCH_SEARCH,
            host=_corr_host(cn),
            service="dsart_rt",
            kind=KIND_PROCESS,
            cn_id=int(cn),
            instance="search_rt",
        ))
    return tuple(rows)


SERVICE_INVENTORY: Final[tuple[ServiceEntry, ...]] = _build_inventory()


# ---------------------------------------------------------------------------
# Convenience filters
# ---------------------------------------------------------------------------


def entries_by_tier(tier: str) -> tuple[ServiceEntry, ...]:
    return tuple(e for e in SERVICE_INVENTORY if e.tier == tier)


def restartable_entries() -> tuple[ServiceEntry, ...]:
    """Every inventory row whose host is NOT in :data:`H20_HOSTNAMES`."""
    return tuple(e for e in SERVICE_INVENTORY if e.is_restartable())


def corr_orch_hosts() -> tuple[str, ...]:
    """FQDNs of the 16 corr orchestrator hosts (re-spawn target)."""
    return CORR_HOSTS


def search_orch_hosts() -> tuple[str, ...]:
    """FQDNs of the 4 search orchestrator hosts (re-spawn target)."""
    return SEARCH_HOSTS


def all_orch_hosts() -> tuple[str, ...]:
    """All 20 dsart_rt host FQDNs (corr first, then search), in
    inventory order.
    """
    return tuple(CORR_HOSTS + SEARCH_HOSTS)
