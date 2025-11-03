"""P-field simulation using NSCORE back-transform.

Steps:
- Fit normal-score model on conditioning data (marginal CDF proxy).
- Simulate a standard normal field on grid nodes using a covariance model.
- Back-transform the simulated normal field to original scale via NSCORE model.
"""

from __future__ import annotations

import jax.numpy as jnp

from .gtsim import gtsim
from .nscore import fit_nscore, nscore_inverse


def pfsim(
    data_values: jnp.ndarray,
    grid_coords: jnp.ndarray,
    kernel_name: str,
    range_: float,
    nbands: int,
    seed: int,
) -> jnp.ndarray:
    """Continuous P-field simulation.

    Args:
        data_values: (n,) conditioning values to learn marginal CDF
        grid_coords: (m, d) nodes to simulate
        kernel_name: covariance model name for Gaussian field
        range_: model range for structure of P-field
        nbands: number of bands for turning-bands simulation
        seed: RNG seed

    Returns:
        Simulated field on original scale (m,)
    """
    # 1) Fit marginal CDF via NSCORE model from data
    model = fit_nscore(jnp.asarray(data_values))
    # 2) Simulate standard normal field with unit variance (sill=1)
    z = gtsim(grid_coords, kernel_name, sill=1.0, range_=range_, nugget=0.0, nbands=nbands, seed=seed)
    # 3) Back-transform to original scale
    x = nscore_inverse(z, model)
    return x


