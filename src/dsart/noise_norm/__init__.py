"""DSA-110 search-detector noise-normalisation subpackage (M5 chunk 3).

Two layers per plan §3.6.9 / §3.6.10 / §4.4:

  - **Layer 1** (``layer1.py``) — per-cube σ-clipped global scalar per
    fine_DM trial. Three iterations of 3σ clipping over the
    ``[T_det, N_grid, N_grid]`` slab; on cold start the first 5 cubes
    use the *median of the 5 most recent per-cube σs* (robust against
    single-cube RFI contamination), then from cube 6 onward the current
    cube's σ is used directly.

  - **Layer 2** (``layer2.py``) — per-conv-output empirical σ_k EMA after
    the conv bank, computed on the cube *interior* only
    (``[n_kernel_max_t//2, T_det − n_kernel_max_t//2]``) per the
    §3.6.12 ``T_det = 2 blocks`` re-pin. Welford running mean for the
    first ``noise_layer2_n_burnin`` cubes (default 30); EMA with γ from
    ``cube_cadence_s / τ_s`` afterwards. Setting
    ``flags.bit3 = noise_warmup`` on emitted Candidates is the
    Detector's responsibility (chunk 3 wires this via the
    ``Layer2State.is_warming_up`` predicate).

The ``DeterministicDetector`` (``forward.py``) now consumes a real
``Layer2State`` instance instead of the chunk-1/2 placeholder
``_sigma_k_placeholder`` buffer (D11). The Protocol surface
(``forward(cube, validity_mask, sigma_layer1)``) is unchanged.
"""

from .layer1 import (
    Layer1State,
    layer1_global_scalar,
    sigma_clipped_std,
)
from .layer2 import (
    Layer2State,
    layer2_interior_sigma,
)

__all__ = [
    "Layer1State",
    "Layer2State",
    "layer1_global_scalar",
    "layer2_interior_sigma",
    "sigma_clipped_std",
]
