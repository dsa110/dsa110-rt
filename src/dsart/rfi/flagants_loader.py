"""Legacy ``flagants.dat`` loader (M3 chunk 3c; plan §4.2 step 7).

Parses the legacy DSA-110 ``flagants.dat`` text file (one antenna
index per line, optionally with ``#``-prefix comments + blank lines)
into a fixed-size ``[NANTS]`` boolean mask.

Expected on-disk format (matches
``/home/ubuntu/proj/dsa110-shell/dsa110-xengine/utils/flagants.dat``):

.. code-block:: text

    # comments are OK on lines starting with '#' (after optional whitespace)
    # blank lines are OK
    47
    48
    52
    74

Each integer is treated as a 0-based antenna index in ``[0, NANTS)``;
out-of-range indices raise :class:`ValueError`. Duplicate indices are
silently de-duplicated (mask is idempotent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dsart.common.constants import NANTS

__all__ = ["load_flagants", "load_flagants_torch", "parse_flagants_text"]


def parse_flagants_text(text: str) -> list[int]:
    """Parse ``flagants.dat`` text into a deduplicated, sorted list of
    antenna indices in ``[0, NANTS)``.

    Args:
        text: raw file contents (UTF-8 str).

    Returns:
        Sorted list of unique ant indices (Python ints).

    Raises:
        ValueError: malformed line (non-int, out-of-range) — error
            message includes the offending line number (1-indexed).
    """
    seen: set[int] = set()
    for ln, raw in enumerate(text.splitlines(), start=1):
        # Strip whitespace, drop comment portion (everything after '#'
        # on a line, including leading-comment lines).
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            idx = int(line)
        except ValueError as exc:
            raise ValueError(
                f"flagants.dat line {ln}: cannot parse {raw!r} as int"
            ) from exc
        if not 0 <= idx < NANTS:
            raise ValueError(
                f"flagants.dat line {ln}: ant index {idx} out of range "
                f"[0, {NANTS})"
            )
        seen.add(idx)
    return sorted(seen)


def load_flagants(path: str | Path) -> np.ndarray:
    """Load ``flagants.dat`` from disk into a ``[NANTS]`` numpy bool mask.

    Args:
        path: path to a legacy ``flagants.dat`` text file.

    Returns:
        Numpy bool array of length :data:`NANTS`. ``mask[ant] = True``
        for every antenna listed in the file.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: malformed file (see :func:`parse_flagants_text`).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"flagants.dat not found at {p}")
    indices = parse_flagants_text(p.read_text(encoding="utf-8"))
    mask = np.zeros(NANTS, dtype=bool)
    if indices:
        mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def load_flagants_torch(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Like :func:`load_flagants` but returns a torch bool tensor on
    the requested device.

    Args:
        path: path to ``flagants.dat``.
        device: torch device for the returned tensor.

    Returns:
        Bool tensor of shape ``[NANTS]``.
    """
    np_mask = load_flagants(path)
    return torch.as_tensor(np_mask, device=device)
