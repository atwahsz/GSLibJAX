"""Neighbor search utilities in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def k_nearest_indices(points: jnp.ndarray, query: jnp.ndarray, k: int) -> jnp.ndarray:
    """Return indices of k nearest points to query.

    Args:
        points: (n, d)
        query: (d,)
        k: number of neighbors

    Returns:
        (k,) indices sorted by increasing distance.
    """
    dif = points - query[None, :]
    d2 = jnp.sum(dif * dif, axis=1)
    k = jnp.minimum(k, points.shape[0])
    vals, idx = jax.lax.top_k(-d2, k)
    # top_k returns largest; we used -d2 so sort ascending by distance
    order = jnp.argsort(-vals)
    return idx[order]


def octant_k_indices(points: jnp.ndarray, query: jnp.ndarray, k_per_octant: int, max_total: int | None = None) -> jnp.ndarray:
    """Select up to k_per_octant nearest points from each octant around query.

    Works for 2D (quadrants) and 3D (octants). Returns concatenated indices
    sorted within each octant by distance. If max_total is specified, truncate
    to that many indices overall.
    """
    dif = points - query[None, :]
    d2 = jnp.sum(dif * dif, axis=1)
    # Determine octant id based on sign bits
    if points.shape[1] == 2:
        # quadrants: (sx, sy) -> id in {0..3}
        sx = (dif[:, 0] >= 0).astype(jnp.int32)
        sy = (dif[:, 1] >= 0).astype(jnp.int32)
        oid = sx * 2 + sy
        num_parts = 4
    else:
        sx = (dif[:, 0] >= 0).astype(jnp.int32)
        sy = (dif[:, 1] >= 0).astype(jnp.int32)
        sz = (dif[:, 2] >= 0).astype(jnp.int32)
        oid = sx * 4 + sy * 2 + sz
        num_parts = 8

    # For each octant, select top-k by -d2
    def select_part(part_id):
        mask = (oid == part_id)
        # assign -inf to excluded to avoid selection
        score = jnp.where(mask, -d2, -jnp.inf)
        # choose up to k_per_octant; top_k pads with -inf if not enough
        vals, idx = jax.lax.top_k(score, jnp.minimum(k_per_octant, points.shape[0]))
        # filter out -inf entries (non-members)
        valid = jnp.isfinite(vals)
        idx = idx[valid]
        # sort within part by ascending distance
        vals_valid = vals[valid]
        order = jnp.argsort(-vals_valid)
        return idx[order]

    parts = [select_part(i) for i in range(num_parts)]
    cat = jnp.concatenate(parts, axis=0)
    if max_total is not None:
        max_total = jnp.minimum(max_total, cat.shape[0])
        cat = cat[: max_total]
    return cat


def within_radius_indices(points: jnp.ndarray, query: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Return indices of points within given euclidean radius of query."""
    dif = points - query[None, :]
    d2 = jnp.sum(dif * dif, axis=1)
    return jnp.nonzero(d2 <= (radius * radius), size=points.shape[0], fill_value=-1)[0]


