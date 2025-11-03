"""LU (Cholesky) simulation of Gaussian random field on small grids.

Builds covariance matrix on grid nodes and performs Cholesky-based simulation.
Intended for moderate problem sizes due to O(n^2) memory.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ..core.random import make_prng_key
from ..core.kriging import _pairwise_distances  # reuse internal helper
from ..core.covariance import spherical, exponential, gaussian


def _get_kernel(name: str):
    n = name.lower()
    if n in ("spherical", "sph"):
        return spherical
    if n in ("exponential", "exp"):
        return exponential
    if n in ("gaussian", "gau"):
        return gaussian
    raise ValueError("Unknown kernel")


def lusim(
    grid_coords: jnp.ndarray,
    kernel_name: str,
    sill: float,
    range_: float,
    nugget: float,
    seed: int,
) -> jnp.ndarray:
    """One realization on nodes in Gaussian space.

    Args:
        grid_coords: (m, d)
        kernel_name: covariance model
        sill, range_, nugget: covariance parameters
        seed: RNG seed

    Returns:
        Simulated values (m,)
    """
    m = grid_coords.shape[0]
    kernel = _get_kernel(kernel_name)
    d = _pairwise_distances(grid_coords, grid_coords)
    c = kernel(d, sill, range_)
    if nugget != 0.0:
        c = c.at[jnp.diag_indices(m)].add(nugget)
    # Jitter for numerical stability
    c = c + 1e-8 * jnp.eye(m, dtype=c.dtype)
    L = jnp.linalg.cholesky(c)
    key = make_prng_key(seed)
    z = jax.random.normal(key, (m,))
    y = L @ z
    return y


