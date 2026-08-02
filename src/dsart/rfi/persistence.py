"""Time-persistence latch for the fast-vis RFI flagger.

Every detector in :mod:`dsart.rfi` is *memoryless*: each cube is
flagged on its own statistics, so a transmitter that is on continuously
gets re-detected (and re-thresholded) 7.45 times a second. Cells that
sit right at a detector threshold therefore blink — flagged on some
cubes, clean on others — which leaves partially-excised RFI in the
fast-vis stream and modulates the effective bandpass at the cube
cadence.

This module adds the missing memory: an (ant, ch, pol) cell that the
detectors flag for a *whole* trailing window (``latch_window_s``, 30 s
by default) is **latched** and stays flagged for ``hold_s`` (15 min by
default) after the detectors stop firing on it. Persistent RFI is
excised persistently; a one-cube blip still only costs one cube.

Cost
====

The state is two ``[NANTS, NCHAN, NPOL]`` integer tensors (73 728 cells
each at the production shape → 0.59 MB total) and 4-5 elementwise
kernels per cube, with no host sync. Measured 2026-08-02 at the
production cube shape (96 × 384 × 2), 30 s window / 15 min hold:
**0.52 ms per cube on CPU** — 0.4 % of the 134 ms real-time budget,
and the CUDA path is strictly cheaper. See :class:`FlagPersistence` for
the exact recurrences.

Two equivalent formulations, picked by ``latch_frac``:

* ``latch_frac >= 1.0`` (the default, and the semantics "100 % flag
  fraction over the window"): the criterion "flagged in *every* one of
  the last ``W`` cubes" is exactly a consecutive-run length ``>= W``, so
  a single run counter suffices — **O(1) memory, no ring buffer**. This
  is the production path.
* ``latch_frac < 1.0``: a genuine rolling window is required, so we keep
  a ``[W, NANTS, NCHAN, NPOL]`` uint8 ring plus an int16 running count
  (16.5 MB at ``W=224``, still one add/sub per cube). Useful if the
  strict all-cubes rule turns out to be too brittle on sky — RFI that
  drops below threshold for one cube in 224 resets the strict counter.

Both paths hold the latch in the *same* ``_hold`` countdown, so the
latch/hold semantics are identical either way.

Feeding the latch
=================

:class:`dsart.rfi.combine.RFIFlagger` feeds the **detector** mask
(SK | bandpass | group | sum-threshold), deliberately excluding

* the persistence output itself — otherwise a latched cell keeps its own
  run alive and never expires, and
* the static ``flagants.dat`` overlay — those antennas are flagged
  unconditionally anyway, so latching them wastes nothing but says
  nothing either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dsart.common.constants import BLOCK_DURATION_S

#: Trailing window over which a cell must stay flagged before it latches.
LATCH_WINDOW_S_DEFAULT: float = 30.0

#: How long a latched cell stays flagged after the detectors let go.
HOLD_S_DEFAULT: float = 900.0

#: Flag fraction within the window required to latch. 1.0 == "flagged in
#: every cube of the window" and selects the O(1) run-counter path.
LATCH_FRAC_DEFAULT: float = 1.0


def seconds_to_cubes(seconds: float) -> int:
    """Cube count covering ``seconds`` at the fada block cadence.

    One ``flag_block`` call consumes one fada block =
    :data:`dsart.common.constants.BLOCK_DURATION_S` (134.218 ms), so
    30 s → 224 cubes and 900 s → 6705 cubes.
    """
    if seconds <= 0.0:
        return 0
    return max(1, int(math.ceil(seconds / BLOCK_DURATION_S)))


@dataclass(frozen=True, slots=True)
class PersistenceStats:
    """Cheap, sync-free snapshot of the latch state.

    ``n_latched`` and ``n_new_latched`` are 0-d int64 **tensors** on the
    flagger's device, not Python ints: reading them would force a
    host sync on the RT path. :class:`dsart.rfi.combine.RFIFlagger`
    folds them into the single sync it already pays for
    ``flag_fraction_total``.
    """

    n_latched: torch.Tensor      # 0-d int64: cells currently held
    n_new_latched: torch.Tensor  # 0-d int64: cells that latched this cube


class FlagPersistence:
    """Latch-and-hold memory over the per-cube detector mask.

    Args:
        latch_window_cubes: ``W`` — a cell latches once the detectors
            have flagged it for this many cubes (at ``latch_frac`` of
            them). ``0`` disables the latch entirely.
        hold_cubes: ``H`` — cubes a latched cell stays flagged after the
            detectors stop firing on it.
        latch_frac: fraction of the window that must be flagged. ``>=
            1.0`` uses the exact consecutive-run counter (no ring
            buffer); ``< 1.0`` allocates the rolling-window ring.

    The state is allocated lazily on the first :meth:`update` call so it
    picks up the cube shape / device of the live pipeline (tests run
    reduced shapes on CPU).

    Recurrences, per cube, with ``d`` the detector mask:

    * run-counter path::

          run <- (run + 1) * d           # zeroed wherever d is false
          at  <- run >= W                # criterion holds this cube

    * ring path::

          count <- count + d - ring[i];  ring[i] <- d;  i <- (i+1) % W
          at    <- count >= ceil(frac*W)

    and then, in both cases::

          hold <- H if at else hold      # refresh while the RFI is up
          emit    hold > 0
          hold <- hold if at else max(hold - 1, 0)

    so a cell stays flagged for exactly ``H`` cubes after the last cube
    on which the criterion held.
    """

    def __init__(
        self,
        *,
        latch_window_cubes: int,
        hold_cubes: int,
        latch_frac: float = LATCH_FRAC_DEFAULT,
    ) -> None:
        if latch_window_cubes < 0:
            raise ValueError(
                f"latch_window_cubes={latch_window_cubes}, expected >= 0"
            )
        if hold_cubes < 0:
            raise ValueError(f"hold_cubes={hold_cubes}, expected >= 0")
        if not 0.0 < latch_frac <= 1.0:
            raise ValueError(
                f"latch_frac={latch_frac}, expected in (0, 1]"
            )
        self._w = int(latch_window_cubes)
        self._h = int(hold_cubes)
        self._frac = float(latch_frac)
        self._use_ring = self._frac < 1.0
        # ceil so frac=1.0 would map to W exactly (the ring path is only
        # taken for frac < 1.0, but keep the arithmetic honest).
        self._count_thresh = max(1, int(math.ceil(self._frac * self._w)))

        self._hold: torch.Tensor | None = None      # int32 [A, C, P]
        self._run: torch.Tensor | None = None       # int32 [A, C, P]
        self._ring: torch.Tensor | None = None      # uint8 [W, A, C, P]
        self._count: torch.Tensor | None = None     # int16 [A, C, P]
        self._ring_pos: int = 0
        self._ring_filled: int = 0
        self._cubes_seen: int = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False when the latch can never fire (window or hold is 0)."""
        return self._w > 0 and self._h > 0

    @property
    def latch_window_cubes(self) -> int:
        return self._w

    @property
    def hold_cubes(self) -> int:
        return self._h

    @property
    def latch_frac(self) -> float:
        return self._frac

    @property
    def uses_ring(self) -> bool:
        """True when the rolling-window ring is allocated (frac < 1)."""
        return self._use_ring

    @property
    def state_bytes(self) -> int:
        """Bytes of resident latch state (0 before the first cube)."""
        tot = 0
        for t in (self._hold, self._run, self._ring, self._count):
            if t is not None:
                tot += t.numel() * t.element_size()
        return tot

    @property
    def cubes_seen(self) -> int:
        return self._cubes_seen

    def reset(self) -> None:
        """Drop every latch and re-arm from scratch."""
        self._hold = None
        self._run = None
        self._ring = None
        self._count = None
        self._ring_pos = 0
        self._ring_filled = 0
        self._cubes_seen = 0

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def _alloc(self, like: torch.Tensor) -> None:
        dev, shape = like.device, like.shape
        self._hold = torch.zeros(shape, dtype=torch.int32, device=dev)
        if self._use_ring:
            self._ring = torch.zeros(
                (self._w, *shape), dtype=torch.uint8, device=dev,
            )
            self._count = torch.zeros(shape, dtype=torch.int16, device=dev)
        else:
            self._run = torch.zeros(shape, dtype=torch.int32, device=dev)

    def update(
        self, detector_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, PersistenceStats]:
        """Advance the latch by one cube.

        Args:
            detector_mask: bool tensor ``[NANTS, NCHAN, NPOL]`` — the
                OR of the *detector* masks for this cube, WITHOUT the
                persistence output and WITHOUT the static flagants
                overlay (see the module docstring).

        Returns:
            ``(persist_mask, stats)``. ``persist_mask`` is a bool tensor
            of the same shape holding the currently-latched cells; the
            caller ORs it into the final mask. ``stats`` carries 0-d
            device tensors so nothing here forces a host sync.
        """
        if detector_mask.dtype != torch.bool:
            raise ValueError(
                f"detector_mask must be bool; got {detector_mask.dtype}"
            )
        if not self.enabled:
            zero = torch.zeros((), dtype=torch.int64,
                               device=detector_mask.device)
            return (
                torch.zeros_like(detector_mask),
                PersistenceStats(n_latched=zero, n_new_latched=zero),
            )

        if (
            self._hold is None
            or self._hold.shape != detector_mask.shape
            or self._hold.device != detector_mask.device
        ):
            self._alloc(detector_mask)
        assert self._hold is not None

        if self._use_ring:
            at_thresh, newly = self._advance_ring(detector_mask)
        else:
            at_thresh, newly = self._advance_run(detector_mask)

        # Refresh the countdown on every cube the criterion holds, so a
        # transmitter that stays up keeps its latch topped up and the
        # hold clock only starts once the detectors let go.
        self._hold = torch.where(
            at_thresh, torch.full_like(self._hold, self._h), self._hold,
        )
        # Emit from the PRE-decrement counter, then tick down only on
        # cubes where the criterion did not hold. That makes `hold_cubes`
        # literally "cubes flagged after the detectors let go": H, H-1,
        # ... 1 are all emitted, and the cell clears on the H+1-th.
        persist_mask = self._hold > 0
        self._hold = torch.where(
            at_thresh, self._hold, (self._hold - 1).clamp_(min=0),
        )
        self._cubes_seen += 1
        return persist_mask, PersistenceStats(
            n_latched=persist_mask.sum(dtype=torch.int64),
            n_new_latched=newly.sum(dtype=torch.int64),
        )

    # -- the two window formulations -----------------------------------

    def _advance_run(
        self, det: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consecutive-run counter — exact for ``latch_frac == 1.0``.

        ``(run + 1) * det`` increments where the detector fired and
        zeroes everywhere else in one fused elementwise pass; the run
        is clamped at ``W`` so a channel that is bad for a week can't
        overflow int32 (it would take ~9 years, but the clamp also keeps
        the tensor's dynamic range tiny for free).
        """
        assert self._run is not None
        self._run = ((self._run + 1) * det).clamp_(max=self._w)
        at_thresh = self._run >= self._w
        # "newly" == the cube on which the run first reaches W; on
        # subsequent cubes the run is clamped at W so `> hold==0` is the
        # discriminator instead.
        newly = at_thresh & (self._hold == 0)
        return at_thresh, newly

    def _advance_ring(
        self, det: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact rolling-window count — used when ``latch_frac < 1.0``."""
        assert self._ring is not None and self._count is not None
        slot = self._ring[self._ring_pos]
        det_u8 = det.to(torch.uint8)
        self._count += det_u8.to(torch.int16) - slot.to(torch.int16)
        self._ring[self._ring_pos] = det_u8
        self._ring_pos = (self._ring_pos + 1) % self._w
        self._ring_filled = min(self._ring_filled + 1, self._w)
        if self._ring_filled < self._w:
            # Don't latch on a partial window (cold start).
            empty = torch.zeros_like(det)
            return empty, empty
        at_thresh = self._count >= self._count_thresh
        newly = at_thresh & (self._hold == 0)
        return at_thresh, newly
