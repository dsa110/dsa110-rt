"""dsart.x2 -- the 2x time-resolution search (docs/x2/SPEC.md).

Everything the 2x op-point adds lives here or behind explicit 2x parameters,
so the 1x pipeline is untouched until the user switches op-points. Submodules
are imported explicitly (``from dsart.x2.plan import TieredDmPlan``); this
package init imports nothing, so a numpy-only consumer never pays for torch.
"""
