"""HTTP client for the corr-node ``rfi_monitor_export`` exporters.

Decodes the JSON payload back into a typed
:class:`DecodedRFIMonRecord` (same fields as
``dsart.services.rfi_mon_shm.RFIMonRecord``, but stripped of the
shm-specific seq plumbing).

Uses Python stdlib only (``urllib.request``) so the dashboard can
run in any conda env with numpy + Flask installed.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Iterable, Optional

import numpy as np

LOG = logging.getLogger("dsa_monitor.rfi_client")


@dataclasses.dataclass(frozen=True)
class DecodedRFIMonRecord:
    """Consumer-side view of one window record fetched from a corr-node."""

    cn_id: int                                     # source corr node
    seq: int                                       # publisher-side seq
    publish_unix: float                            # POSIX time
    block_n_start: int
    block_n_end: int
    n_cubes: int
    n_cubes_warmup: int
    # scalars[name] -> (pol0, pol1, both)
    scalars: dict[str, tuple[float, float, float]]
    s1_full_mean: np.ndarray                       # (NANTS, NCHAN_DS, NPOL) fp32
    mask_count_final: np.ndarray                   # uint8
    mask_count_sk: np.ndarray
    mask_count_bp: np.ndarray
    mask_count_grp: np.ndarray
    mask_count_sumthr: np.ndarray
    mask_count_fa: np.ndarray


def _decode_array(d: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(d["data_b64"])
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])


def _decode_record(payload: dict[str, Any], *, cn_id: int) -> DecodedRFIMonRecord:
    return DecodedRFIMonRecord(
        cn_id=int(cn_id),
        seq=int(payload["seq"]),
        publish_unix=payload["publish_utc_ns"] / 1e9,
        block_n_start=int(payload["block_n_start"]),
        block_n_end=int(payload["block_n_end"]),
        n_cubes=int(payload["n_cubes"]),
        n_cubes_warmup=int(payload["n_cubes_warmup"]),
        scalars={
            k: (float(v[0]), float(v[1]), float(v[2]))
            for k, v in payload["scalars"].items()
        },
        s1_full_mean=_decode_array(payload["s1_full_mean"]),
        mask_count_final=_decode_array(payload["mask_count_final"]),
        mask_count_sk=_decode_array(payload["mask_count_sk"]),
        mask_count_bp=_decode_array(payload["mask_count_bp"]),
        mask_count_grp=_decode_array(payload["mask_count_grp"]),
        mask_count_sumthr=_decode_array(payload["mask_count_sumthr"]),
        mask_count_fa=_decode_array(payload["mask_count_fa"]),
    )


def _http_get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


class RFIClient:
    """Per-corr-node HTTP client. Stateless except for the cn_id +
    base URL; one instance per ``CorrNode``."""

    def __init__(
        self,
        cn_id: int,
        base_url: str,
        *,
        timeout_s: float = 3.0,
    ) -> None:
        self.cn_id = int(cn_id)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_health(self) -> Optional[dict[str, Any]]:
        try:
            return _http_get_json(
                self._url("/api/health"), timeout_s=self.timeout_s,
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            LOG.debug("get_health cn=%d: %s", self.cn_id, e)
            return None

    def get_meta(self) -> Optional[dict[str, Any]]:
        try:
            return _http_get_json(
                self._url("/api/meta"), timeout_s=self.timeout_s,
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            LOG.debug("get_meta cn=%d: %s", self.cn_id, e)
            return None

    def get_latest(self) -> Optional[DecodedRFIMonRecord]:
        try:
            body = _http_get_json(
                self._url("/api/latest"), timeout_s=self.timeout_s,
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            LOG.debug("get_latest cn=%d: %s", self.cn_id, e)
            return None
        if not body.get("available"):
            return None
        return _decode_record(body["record"], cn_id=self.cn_id)

    def get_recent(self, n: int) -> list[DecodedRFIMonRecord]:
        try:
            body = _http_get_json(
                self._url(f"/api/recent?n={int(n)}"),
                timeout_s=self.timeout_s,
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            LOG.debug("get_recent cn=%d: %s", self.cn_id, e)
            return []
        records = body.get("records", [])
        out: list[DecodedRFIMonRecord] = []
        for rec_obj in records:
            try:
                out.append(_decode_record(rec_obj, cn_id=self.cn_id))
            except Exception:
                LOG.exception("decode failed cn=%d", self.cn_id)
        return out


def build_clients(corr_nodes: Iterable[Any]) -> dict[int, RFIClient]:
    """Construct one :class:`RFIClient` per corr node, keyed by cn_id."""
    return {
        n.cn_id: RFIClient(cn_id=n.cn_id, base_url=n.rfi_base_url)
        for n in corr_nodes
    }
