"""Isolated voltage retention ring (corr-node) — pure data structure.

The corr-node :mod:`dsart.services.voltage_retention` service attaches
``fada`` as a 3rd PSRDADA reader and, on every block, memcpys the raw
288 MiB block into this ring and immediately ``markCleared()``s the
PSRDADA page. The ring is the *isolation boundary* the operator asked
for: the only thing the hot fada reader does is fill the ring, so a slow
voltage dump (disk / network, driven by the separate dump worker) can
never back-pressure capture.

Concurrency model — a single-writer / single-reader-per-block **seqlock**,
no shared lock held across the big memcpy:

  * The reader thread is the sole writer. :meth:`store` marks the target
    slot in-flight (``index[slot] = -1``), does the 288 MiB copy, then
    publishes the block number (``index[slot] = block_n``). List-element
    assignment is atomic under the GIL.
  * The dump worker calls :meth:`copy_block`, which reads ``index[slot]``
    before and after copying the slot. If it isn't exactly ``block_n``
    both times the slot was being (or got) overwritten by the reader and
    the block is reported as *dropped* rather than returning torn bytes.

Because a given slot is only rewritten once every ``n_blocks`` (~15 s at
the production ring size), the seqlock retry/skip path is essentially
never taken except for the oldest block right as it rolls off — which is
exactly the block we are willing to lose.

This module is pure (no PSRDADA, no threads of its own, no I/O) so the
unit tests can drive it deterministically with a tiny ``bytes_per_block``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..common.constants import BLOCK_SAMPLES_SPECNUM, FADA_BYTES_PER_BLOCK

__all__ = [
    "specnum_to_block_n",
    "block_n_to_first_specnum",
    "window_block_range",
    "WindowExtract",
    "VoltageRing",
]


# Native specnum period: 1 specnum = 2 native samples = 65.536 µs.
# (Kept local so the ring math is self-contained; mirrors the value
# corr_fast / coincidencer use.)
def specnum_to_block_n(specnum: int) -> int:
    """Map an absolute native ``specnum`` to its fada ``block_n``.

    Uses the same anchor convention as ``corr_fast_integration`` /
    ``cube_pipeline`` (``block_specnum_start = block_n *
    BLOCK_SAMPLES_SPECNUM``), so a C2 trigger's ``event_specnum`` lands on
    the block the corr fleet labelled with the same ``block_n``.
    """
    if specnum < 0:
        raise ValueError(f"specnum={specnum} must be >= 0")
    return int(specnum) // BLOCK_SAMPLES_SPECNUM


def block_n_to_first_specnum(block_n: int) -> int:
    """First specnum contained in ``block_n``."""
    return int(block_n) * BLOCK_SAMPLES_SPECNUM


def window_block_range(
    target_block_n: int, n_pre: int, n_post: int,
) -> range:
    """Inclusive block window ``[target-n_pre, target+n_post]`` (clamped >=0)."""
    if n_pre < 0 or n_post < 0:
        raise ValueError(f"n_pre={n_pre}, n_post={n_post} must be >= 0")
    start = max(0, int(target_block_n) - int(n_pre))
    end = int(target_block_n) + int(n_post)
    return range(start, end + 1)


@dataclass(frozen=True)
class WindowExtract:
    """Result of :meth:`VoltageRing.extract_window`."""

    #: ``(block_n, bytes_view)`` for every block actually present, ascending.
    blocks: Tuple[Tuple[int, np.ndarray], ...]
    #: block_n values requested but not present (rolled off or never seen).
    dropped: Tuple[int, ...]
    #: first/last present block_n (or None when nothing was captured).
    first_block_n: Optional[int]
    last_block_n: Optional[int]

    @property
    def n_present(self) -> int:
        return len(self.blocks)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


class VoltageRing:
    """Fixed-capacity RAM ring of raw fada blocks, keyed by ``block_n``.

    Parameters
    ----------
    n_blocks:
        Slot count. Production: ``ceil(retention_s / 0.134218)`` (~112 for
        15 s).
    bytes_per_block:
        Block size; production :data:`FADA_BYTES_PER_BLOCK` (288 MiB).
        Tests pass a tiny value.
    buf:
        Optional preallocated ``(n_blocks, bytes_per_block)`` uint8 array
        (tests / NUMA-pinned production allocation). Allocated with
        ``np.empty`` (lazy pages) when omitted.
    """

    def __init__(
        self,
        *,
        n_blocks: int,
        bytes_per_block: int = FADA_BYTES_PER_BLOCK,
        buf: Optional[np.ndarray] = None,
    ) -> None:
        if n_blocks < 1:
            raise ValueError(f"n_blocks={n_blocks} must be >= 1")
        if bytes_per_block < 1:
            raise ValueError(f"bytes_per_block={bytes_per_block} must be >= 1")
        self._n = int(n_blocks)
        self._bpb = int(bytes_per_block)
        if buf is None:
            buf = np.empty((self._n, self._bpb), dtype=np.uint8)
        else:
            if buf.shape != (self._n, self._bpb) or buf.dtype != np.uint8:
                raise ValueError(
                    f"buf must be uint8 shape ({self._n}, {self._bpb}); "
                    f"got {buf.dtype} {buf.shape}"
                )
        self._buf = buf
        # Published block_n per slot; -1 = empty or in-flight. A Python
        # list so element assignment is GIL-atomic (the seqlock).
        self._index: List[int] = [-1] * self._n
        self._newest: int = -1
        self._n_stored: int = 0
        self._n_torn_skips: int = 0

    # ---- properties / introspection -----------------------------------

    @property
    def n_blocks(self) -> int:
        return self._n

    @property
    def bytes_per_block(self) -> int:
        return self._bpb

    @property
    def n_stored(self) -> int:
        """Total blocks stored over the ring's lifetime."""
        return self._n_stored

    @property
    def newest_block_n(self) -> int:
        """Highest block_n stored so far (-1 before the first store)."""
        return self._newest

    @property
    def oldest_block_n(self) -> int:
        """Lowest block_n still resident (-1 when empty)."""
        if self._newest < 0:
            return -1
        return max(0, self._newest - self._n + 1)

    def contains(self, block_n: int) -> bool:
        if block_n < 0:
            return False
        return self._index[int(block_n) % self._n] == int(block_n)

    # ---- writer (reader thread) ---------------------------------------

    def store(self, block_n: int, data: np.ndarray) -> None:
        """Copy ``data`` (uint8, length ``bytes_per_block``) into the slot
        for ``block_n``. Sole writer; seqlock-publishes the slot."""
        if block_n < 0:
            raise ValueError(f"block_n={block_n} must be >= 0")
        arr = np.asarray(data, dtype=np.uint8).reshape(-1)
        if arr.shape[0] != self._bpb:
            raise ValueError(
                f"data length {arr.shape[0]} != bytes_per_block {self._bpb}"
            )
        slot = int(block_n) % self._n
        self._index[slot] = -1          # mark in-flight (seqlock open)
        self._buf[slot, :] = arr        # the big memcpy
        self._index[slot] = int(block_n)  # publish (seqlock close)
        if int(block_n) > self._newest:
            self._newest = int(block_n)
        self._n_stored += 1

    # ---- reader (dump worker) -----------------------------------------

    def copy_block(self, block_n: int) -> Optional[np.ndarray]:
        """Return a private copy of ``block_n``'s bytes, or None if the
        block is absent / was overwritten mid-copy (seqlock check)."""
        if block_n < 0:
            return None
        b = int(block_n)
        slot = b % self._n
        if self._index[slot] != b:
            return None
        out = self._buf[slot].copy()
        if self._index[slot] != b:
            # Reader wrapped onto this slot during our copy → torn; drop.
            self._n_torn_skips += 1
            return None
        return out

    def extract_window(
        self, target_block_n: int, *, n_pre: int, n_post: int,
    ) -> WindowExtract:
        """Copy out every present block in ``[target-n_pre, target+n_post]``.

        Each present block is a fresh array (safe to write to disk while
        the reader keeps filling). Absent blocks are reported in
        ``dropped`` rather than raising.
        """
        blocks: List[Tuple[int, np.ndarray]] = []
        dropped: List[int] = []
        for b in window_block_range(target_block_n, n_pre, n_post):
            arr = self.copy_block(b)
            if arr is None:
                dropped.append(b)
            else:
                blocks.append((b, arr))
        first = blocks[0][0] if blocks else None
        last = blocks[-1][0] if blocks else None
        return WindowExtract(
            blocks=tuple(blocks),
            dropped=tuple(dropped),
            first_block_n=first,
            last_block_n=last,
        )

    def mon(self) -> dict:
        return {
            "n_blocks": self._n,
            "bytes_per_block": self._bpb,
            "n_stored": self._n_stored,
            "newest_block_n": self._newest,
            "oldest_block_n": self.oldest_block_n,
            "n_torn_skips": self._n_torn_skips,
            "ram_bytes": self._n * self._bpb,
        }
