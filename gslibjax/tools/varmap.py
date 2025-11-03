"""Variogram map: semivariance as a function of lag vector (hx, hy).

Computes a 2D map by binning vector differences (dx, dy) into a grid of lag
cells and averaging semivariances.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


def variogram_map(
    coords: jnp.ndarray,
    values: jnp.ndarray,
    hx: float,
    hy: float,
    nx: int,
    ny: int,
    max_range: float | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a variogram map over (nx, ny) lag bins of size (hx, hy).

    Args:
        coords: (n, 2)
        values: (n,)
        hx, hy: lag cell sizes
        nx, ny: number of lag cells in +x and +y directions (map is symmetric)
        max_range: optional maximum distance to consider pairs

    Returns:
        (x_centers, y_centers, gamma_map) where gamma_map shape is (ny, nx)
    """
    n = coords.shape[0]
    ii, jj = jnp.triu_indices(n, k=1)
    dv = values[ii] - values[jj]
    gamma = 0.5 * (dv * dv)
    h = coords[jj] - coords[ii]
    if max_range is not None:
        keep = jnp.linalg.norm(h, axis=1) <= max_range
        h = h[keep]
        gamma = gamma[keep]

    # Map dx,dy to bins in [0, nx-1] and [0, ny-1] using absolute lags (symmetry)
    bx = jnp.clip(jnp.floor(jnp.abs(h[:, 0]) / jnp.maximum(hx, 1e-12)).astype(jnp.int32), 0, nx - 1)
    by = jnp.clip(jnp.floor(jnp.abs(h[:, 1]) / jnp.maximum(hy, 1e-12)).astype(jnp.int32), 0, ny - 1)
    flat = by * nx + bx
    length = nx * ny
    counts = jnp.bincount(flat, length=length)
    sums = jnp.bincount(flat, weights=gamma, length=length)
    gm = jnp.where(counts > 0, sums / counts, 0.0)
    gm = gm.reshape((ny, nx))
    xc = (jnp.arange(nx) + 0.5) * hx
    yc = (jnp.arange(ny) + 0.5) * hy
    return xc, yc, gm


