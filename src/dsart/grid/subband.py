"""Sub-band gridding: split one chgroup into ``n_sub`` uv planes.

Why
===

The corr node dedisperses each summed channel to the COARSE DM and then
grids all 48 summed channels of its chgroup into one uv plane. Any
residual ``|dm_true - dm_coarse|`` therefore smears the burst across
the 11.72 MHz chgroup, and the search can only shift whole planes in
time, so no fine-DM trial can undo it:

    smear_us = 8.3 * B_MHz * nu_GHz^-3 * dm_err

At the deployed 100-1500 plan that is 3.18 samples of 1.048576 ms on
average. Gridding each of ``n_sub`` sub-bands to its own plane, with
stage 1 referencing each sub-band to its own top channel, lets the
search shift every sub-band separately; the uncorrectable width falls
``n_sub``-fold (0.8 samples at ``n_sub=4``).

Layout
======

The sub-band planes are concatenated along ONE cell axis so the fused
stage-1 dedisp+grid kernel runs unchanged, once, over all of them:

    cells [offsets[s], offsets[s+1])  <- sub-band s, pattern patterns[s]

Each ``(baseline, channel)`` source maps to a cell of ITS OWN
sub-band's pattern only (a channel never contributes to another
sub-band's plane). Downstream, the corr TX sends each sub-band as its
own stream with id ``chgroup * n_sub + s`` and its own ``pattern_id``;
the search treats the ``16 * n_sub`` streams as independent "corrs".

The gridding kernel support must be 1 (the production pillbox): the
combined map is built per ``(baseline, channel)`` with one tap each.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dsart.common.constants import NBASE, NCHAN_PER_CHGROUP
from dsart.grid.kernel import FastVisGridder
from dsart.grid.sparsity_pattern import SparsityPattern, build_pattern


def stream_id(chgroup: int, s: int, n_sub: int) -> int:
    """Wire/ring stream index for sub-band ``s`` of ``chgroup``."""
    return int(chgroup) * int(n_sub) + int(s)


def split_stream_id(sid: int, n_sub: int) -> tuple[int, int]:
    """Inverse of :func:`stream_id`: ``(chgroup, s)``."""
    return int(sid) // int(n_sub), int(sid) % int(n_sub)


def build_subband_patterns(
    antpos_e: np.ndarray,
    antpos_n: np.ndarray,
    *,
    chgroup: int,
    n_sub: int,
    dec_deg: float,
    n_grid: int,
    kernel_support: int,
    chan_sum_factor: int,
    cell_lambda: float,
    is_core_baseline_mask: np.ndarray | None,
) -> list[SparsityPattern]:
    """The ``n_sub`` sub-band patterns of one chgroup, in sub-band order.

    Corr and search both call this with identical inputs, so the
    per-stream ``pattern_id`` values agree by construction.
    """
    if n_sub < 1:
        raise ValueError(f"n_sub={n_sub} must be >= 1")
    if n_sub == 1:
        return [build_pattern(
            antpos_e, antpos_n, chgroup=chgroup, dec_deg=dec_deg,
            n_grid=n_grid, kernel_support=kernel_support,
            chan_sum_factor=chan_sum_factor, cell_lambda=cell_lambda,
            is_core_baseline_mask=is_core_baseline_mask,
        )]
    return [
        build_pattern(
            antpos_e, antpos_n, chgroup=chgroup, dec_deg=dec_deg,
            n_grid=n_grid, kernel_support=kernel_support,
            chan_sum_factor=chan_sum_factor, cell_lambda=cell_lambda,
            is_core_baseline_mask=is_core_baseline_mask,
            sub_band=(s, n_sub),
        )
        for s in range(n_sub)
    ]


@dataclass(frozen=True)
class SubbandLayout:
    """Concatenated-cell layout of one chgroup's ``n_sub`` sub-bands."""

    chgroup: int
    n_sub: int
    patterns: tuple[SparsityPattern, ...]
    offsets: np.ndarray            # (n_sub + 1,) int64 cell offsets
    cell_index_map: torch.Tensor   # (NBASE * nchan,) int64, sentinel = n_filled_total
    nchan: int

    @property
    def n_filled_total(self) -> int:
        return int(self.offsets[-1])

    @property
    def stream_ids(self) -> list[int]:
        return [stream_id(self.chgroup, s, self.n_sub) for s in range(self.n_sub)]

    @property
    def pattern_ids(self) -> list[int]:
        return [int(p.pattern_id) for p in self.patterns]

    def cell_slice(self, s: int) -> slice:
        return slice(int(self.offsets[s]), int(self.offsets[s + 1]))


def _pattern_keys(p: SparsityPattern) -> np.ndarray:
    """Sorted uint32 ``(row << 16) | col`` cell keys (build_pattern's order)."""
    return (p.ix_row.astype(np.uint32) << 16) | p.ix_col.astype(np.uint32)


def build_subband_layout(
    patterns: list[SparsityPattern],
    whole_gridder: FastVisGridder,
    *,
    device: torch.device | str = "cpu",
) -> SubbandLayout:
    """Combine per-sub-band patterns into one cell axis + one source map.

    ``whole_gridder`` is the chgroup's ordinary whole-chgroup gridder
    (production builds it anyway). Its ``(baseline, channel) -> cell``
    map is the geometric truth; every source of sub-band ``s`` is
    re-indexed from the whole pattern's cell list into sub-band ``s``'s
    cell list by key lookup. (``FastVisGridder.from_pattern`` cannot be
    used on a sub-band pattern: it rightly insists every channel's cell
    be present, and a sub-band pattern omits the other sub-bands'
    channels.)
    """
    n_sub = len(patterns)
    if n_sub < 1:
        raise ValueError("need at least one pattern")
    whole = whole_gridder.pattern
    chgroup = int(whole.chgroup)
    csf = int(whole.chan_sum_factor)
    nchan = NCHAN_PER_CHGROUP // csf
    if nchan % n_sub != 0:
        raise ValueError(f"n_sub={n_sub} must divide nchan={nchan}")
    if int(whole.kernel_support) != 1:
        raise ValueError(
            "sub-band gridding supports kernel_support=1 only "
            f"(got {whole.kernel_support})"
        )
    per = nchan // n_sub
    for s, p in enumerate(patterns):
        if int(p.chgroup) != chgroup or int(p.chan_sum_factor) != csf:
            raise ValueError("all sub-band patterns must share chgroup + csf")
        if int(p.kernel_support) != 1:
            raise ValueError("sub-band patterns must have kernel_support=1")
        if float(p.cell_lambda) != float(whole.cell_lambda):
            raise ValueError(
                "sub-band patterns must share the whole-chgroup cell_lambda "
                "(one pixel grid) or the search cannot sum them"
            )
        want = None if n_sub == 1 else (s, n_sub)
        if p.sub_band != want:
            raise ValueError(
                f"pattern {s} has sub_band={p.sub_band}, expected {want}"
            )

    offsets = np.zeros(n_sub + 1, dtype=np.int64)
    for s, p in enumerate(patterns):
        offsets[s + 1] = offsets[s] + int(p.n_filled)
    total = int(offsets[-1])

    n_whole = int(whole.n_filled)
    whole_keys = _pattern_keys(whole)
    cim_whole = whole_gridder.cell_index_map.detach().cpu().numpy()
    if cim_whole.shape != (NBASE * nchan,):
        raise ValueError(
            f"whole gridder map has shape {cim_whole.shape}, expected "
            f"({NBASE * nchan},)"
        )
    cim_whole = cim_whole.reshape(NBASE, nchan)

    combined = np.full((NBASE, nchan), total, dtype=np.int64)
    for s, p in enumerate(patterns):
        lo, hi = s * per, (s + 1) * per
        keys_s = _pattern_keys(p)
        cells = cim_whole[:, lo:hi]
        hit = cells < n_whole
        k = whole_keys[cells[hit]]
        idx = np.searchsorted(keys_s, k)
        if idx.size and (
            int(idx.max()) >= keys_s.size or not np.all(keys_s[idx] == k)
        ):
            raise RuntimeError(
                f"sub-band {s}: a source cell of its own channels is missing "
                f"from its pattern (pattern / gridder geometry drift)"
            )
        block = np.full(cells.shape, total, dtype=np.int64)
        block[hit] = idx.astype(np.int64) + offsets[s]
        combined[:, lo:hi] = block

    return SubbandLayout(
        chgroup=chgroup,
        n_sub=n_sub,
        patterns=tuple(patterns),
        offsets=offsets,
        cell_index_map=torch.from_numpy(combined.reshape(-1)).to(device),
        nchan=nchan,
    )


__all__ = [
    "SubbandLayout",
    "build_subband_layout",
    "build_subband_patterns",
    "split_stream_id",
    "stream_id",
]
