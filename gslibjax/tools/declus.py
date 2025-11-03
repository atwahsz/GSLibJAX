"""Cell declustering weights (simple version)."""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def declus_weights(
    coords: jnp.ndarray,
    cell_size: Tuple[float, float, float],
) -> jnp.ndarray:
    """Compute declustering weights by counting samples per cell.

    Args:
        coords: (n, d) d=2 or 3
        cell_size: (sx, sy, sz) use sz=1 for 2D

    Returns:
        Weights (n,) normalized to mean 1.
    """
    d = coords.shape[1]
    sx, sy, sz = cell_size
    if d == 2:
        grid_idx = jnp.stack([
            jnp.floor(coords[:, 0] / jnp.maximum(sx, 1e-12)),
            jnp.floor(coords[:, 1] / jnp.maximum(sy, 1e-12)),
        ], axis=1)
    else:
        grid_idx = jnp.stack([
            jnp.floor(coords[:, 0] / jnp.maximum(sx, 1e-12)),
            jnp.floor(coords[:, 1] / jnp.maximum(sy, 1e-12)),
            jnp.floor(coords[:, 2] / jnp.maximum(sz, 1e-12)),
        ], axis=1)
    # Hash grid indices to a single key
    base = jnp.array([73856093, 19349663, 83492791], dtype=jnp.int64)
    gi = jnp.array(grid_idx, dtype=jnp.int64)
    if d == 2:
        keys = (gi[:, 0] * base[0] + gi[:, 1] * base[1])
    else:
        keys = (gi[:, 0] * base[0] + gi[:, 1] * base[1] + gi[:, 2] * base[2])
    # Map keys to compact range via unique
    uk, inv = jnp.unique(keys, return_inverse=True)
    counts = jnp.bincount(inv, length=uk.shape[0])
    w = 1.0 / counts[inv]
    return w * (w.shape[0] / jnp.sum(w))


