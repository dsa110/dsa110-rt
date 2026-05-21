"""Corr-node fleet topology + HTTP exporter endpoints for the h23
dashboard (M7.6).

Pinned to the M7.5 phase-B fleet layout — the same 16-corr,
4-search assignment used by ``tools/ops/_m75_phaseB_16x4_launch.sh``
and ``tools/ops/_m72_snapshot.py``. cn 9, 13 sit on search hosts
(n09 / n13 run search_rt); the remaining 16 cn IDs are corr_rt.

For each corr cn we expose three identities:

  * ``cn_id`` : int 3..22 (skipping 9, 13, 17, 20). This is the
    routing key for etcd ``/mon/corr_rt/<cn>`` / ``/cmd/corr_rt/<cn>``.
  * ``host`` : short DNS name resolvable from h23 over br1
    (10.42.0.0/24). Append ``.pro.pvt`` to ssh / fetch.
  * ``chgroup`` : 0..15 sub-band index (canonical order, matches
    ``_cn_to_chgroup`` in ``dsart.services.dsart_rt``).

The HTTP exporter base URL is built from
``http://{host}.pro.pvt:{RFI_HTTP_PORT}/``; the dashboard's
``RFIClient`` does HTTP GETs against those for ``/api/latest``,
``/api/recent``, ``/api/meta`` and ``/api/health``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Mirrors tools/ops/_m75_phaseB_16x4_launch.sh::CORR_NODES_CN.
# Order matters: index in this tuple == chgroup index.
_CORR_NODES_CN_BY_CHGROUP: Final[tuple[int, ...]] = (
    3, 4, 5, 6, 7, 8,
    10, 11, 12,
    14, 15, 16,
    18, 19,
    21, 22,
)
assert len(_CORR_NODES_CN_BY_CHGROUP) == 16, "must have 16 chgroups"

RFI_HTTP_PORT_DEFAULT: Final[int] = 5780
"""Default HTTP port the rfi_monitor_export sidecar listens on."""


@dataclass(frozen=True)
class CorrNode:
    cn_id: int
    host: str                                      # short name; .pro.pvt appended for FQDN
    chgroup: int
    rfi_http_port: int = RFI_HTTP_PORT_DEFAULT

    @property
    def fqdn(self) -> str:
        return f"{self.host}.pro.pvt"

    @property
    def rfi_base_url(self) -> str:
        return f"http://{self.fqdn}:{self.rfi_http_port}"


def _cn_to_host(cn_id: int) -> str:
    """Hostname stem for a corr node — ``n<two-digit-cn>``.

    Matches the convention used by tools/ops/_m75_phaseB_16x4_launch.sh
    (e.g. cn=3 → n03, cn=22 → n22).
    """
    return f"n{cn_id:02d}"


CORR_NODES: Final[tuple[CorrNode, ...]] = tuple(
    CorrNode(cn_id=cn, host=_cn_to_host(cn), chgroup=g)
    for g, cn in enumerate(_CORR_NODES_CN_BY_CHGROUP)
)
"""All 16 corr nodes, chgroup-ordered (CORR_NODES[g].chgroup == g)."""


CORR_NODES_BY_CN: Final[dict[int, CorrNode]] = {
    n.cn_id: n for n in CORR_NODES
}
"""Lookup by cn_id."""


CORR_NODES_BY_CHGROUP: Final[dict[int, CorrNode]] = {
    n.chgroup: n for n in CORR_NODES
}
"""Lookup by chgroup."""


def all_rfi_base_urls() -> tuple[str, ...]:
    return tuple(n.rfi_base_url for n in CORR_NODES)
