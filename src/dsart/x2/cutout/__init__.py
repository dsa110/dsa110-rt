"""x2 cutouts (docs/x2/SPEC.md §9, docs/x2/downstream_inventory.md §3).

Submodules (numpy-only; safe to import on h23 and in the dashboard):

* :mod:`.schema`   -- the ``dsart.cutout`` v1 NPZ schema: field table,
  builders shared with the producer (fine/coarse axes from the tiered
  plan, the numpy reference of the plane reductions), validator, atomic
  writer + sidecar manifest, reader.
* :mod:`.products` -- one reader for every downstream consumer: per-(half,
  coarse) DM-time planes from 2x cutouts OR 1x cubes (bit-identical to the
  pre-x2 readers), half counts for the C2 completion gate / dashboard.
"""
