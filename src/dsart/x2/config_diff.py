"""Structural diff of pipeline configs (etcd JSON vs YAML), for op-point pushes.

A unified text diff of two YAML/JSON configs re-aligns around inserted blocks
and invents phantom deletions (memory note "etcd config push drift"), so every
push of an op-point set is reviewed as a *structural* diff instead:

* dicts flatten to dotted paths (``coinc.dump_broadcast.hosts.3``);
* lists whose elements are dicts carrying a unique ``name`` (routines) or ``k``
  (buffers) are keyed by it (``routines[corr_fast].args``), so inserting a
  routine does not shift every later routine;
* other lists are indexed (``x2_todo[0]``);
* a changed whitespace-separated string (a routine's ``args``) is reported
  token by token (``-`` removed, ``+`` added), which is what a reviewer of a
  2x args change needs to see.

Stdlib only: imported by ``tools/ops/push_dsart_to_etcd.py`` and the dashboard.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

#: Keys that identify list elements (in priority order).
_LIST_ID_KEYS: tuple[str, ...] = ("name", "k")


@dataclass(frozen=True)
class DiffEntry:
    """One structural difference. ``kind`` is added / removed / changed."""

    path: str
    kind: str
    old: Any = None
    new: Any = None


def canonical_sha256(obj: Any) -> str:
    """sha256 of the canonical JSON form (sorted keys, no whitespace)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _list_id_key(items: list[Any]) -> str | None:
    if not items or not all(isinstance(x, dict) for x in items):
        return None
    for key in _LIST_ID_KEYS:
        ids = [x.get(key) for x in items]
        if all(isinstance(i, (str, int)) for i in ids) and len(set(ids)) == len(ids):
            return key
    return None


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten ``obj`` to ``{path: scalar}``. Empty containers are leaves."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix or "."] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        if not obj:
            out[prefix or "."] = []
        idk = _list_id_key(obj)
        for i, v in enumerate(obj):
            tag = str(v[idk]) if idk else str(i)
            out.update(flatten(v, f"{prefix}[{tag}]"))
    else:
        out[prefix or "."] = obj
    return out


def structural_diff(old: Any, new: Any) -> list[DiffEntry]:
    """Differences ``old`` -> ``new`` (``None`` old = everything added)."""
    fo = flatten(old) if old is not None else {}
    fn = flatten(new) if new is not None else {}
    entries: list[DiffEntry] = []
    for path in sorted(set(fo) | set(fn)):
        if path not in fo:
            entries.append(DiffEntry(path, "added", None, fn[path]))
        elif path not in fn:
            entries.append(DiffEntry(path, "removed", fo[path], None))
        elif fo[path] != fn[path] or type(fo[path]) is not type(fn[path]):
            entries.append(DiffEntry(path, "changed", fo[path], fn[path]))
    return entries


def _short(v: Any, n: int) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def token_diff(old: str, new: str) -> tuple[list[str], list[str]]:
    """(removed, added) whitespace tokens, order-preserving (difflib)."""
    a, b = old.split(), new.split()
    removed: list[str] = []
    added: list[str] = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(a[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(b[j1:j2])
    return removed, added


def format_diff(entries: Iterable[DiffEntry], *, max_value_len: int = 160) -> list[str]:
    """Human-readable lines, one block per entry."""
    lines: list[str] = []
    for e in entries:
        if e.kind == "added":
            lines.append(f"  + {e.path} = {_short(e.new, max_value_len)}")
        elif e.kind == "removed":
            lines.append(f"  - {e.path} (was {_short(e.old, max_value_len)})")
        elif (isinstance(e.old, str) and isinstance(e.new, str)
              and (len(e.old.split()) > 1 or len(e.new.split()) > 1)):
            removed, added = token_diff(e.old, e.new)
            lines.append(f"  ~ {e.path} (tokens)")
            if removed:
                lines.append("      - " + " ".join(removed))
            if added:
                lines.append("      + " + " ".join(added))
        else:
            lines.append(
                f"  ~ {e.path}: {_short(e.old, max_value_len)} -> "
                f"{_short(e.new, max_value_len)}"
            )
    return lines


def summarize(entries: Iterable[DiffEntry]) -> dict[str, int]:
    out = {"added": 0, "removed": 0, "changed": 0}
    for e in entries:
        out[e.kind] += 1
    return out
