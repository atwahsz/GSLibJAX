"""Turning-bands simulation (GTSIM) using JAX.

Approach:
- Generate `nbands` random unit directions in 2D/3D.
- For each band, project node coordinates onto the direction to get scalars t.
- Build 1D covariance matrix C_ij = k(|t_i - t_j|) with unit sill.
- Simulate a 1D Gaussian field on nodes via Cholesky.
- Average band contributions and scale to match target sill.

This is a pragmatic implementation suitable for moderate node counts.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ..core.random import make_prng_key
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


def _random_unit_direction(key: jax.Array, dim: int) -> jnp.ndarray:
    v = jax.random.normal(key, (dim,))
    nrm = jnp.linalg.norm(v) + 1e-12
    return v / nrm


def gtsim(
    grid_coords: jnp.ndarray,
    kernel_name: str,
    sill: float,
    range_: float,
    nugget: float,
    nbands: int,
    seed: int,
) -> jnp.ndarray:
    """Turning-bands simulation on the provided nodes.

    Args:
        grid_coords: (m, d) node coordinates
        kernel_name: 'spherical'|'exponential'|'gaussian'
        sill: target variance of the field
        range_: model range
        nugget: small diagonal perturbation (added to all bands)
        nbands: number of bands to average
        seed: RNG seed

    Returns:
        Simulated values (m,)
    """
    m, d = grid_coords.shape
    kernel = _get_kernel(kernel_name)
    key = make_prng_key(seed)
    acc = jnp.zeros((m,), dtype=jnp.float32)

    def one_band(carry, i):
        key, acc = carry
        key, kd, kz = jax.random.split(key, 3)
        u = _random_unit_direction(kd, d)
        t = grid_coords @ u  # (m,)
        # 1D distances |ti - tj|
        dt = jnp.abs(t[:, None] - t[None, :])
        c = kernel(dt, 1.0, range_)
        if nugget != 0.0:
            c = c.at[jnp.diag_indices(m)].add(nugget)
        c = c + 1e-8 * jnp.eye(m, dtype=c.dtype)
        L = jnp.linalg.cholesky(c)
        z = jax.random.normal(kz, (m,))
        y = L @ z  # unit sill band
        acc = acc + y
        return (key, acc), None

    (key, acc), _ = jax.lax.scan(one_band, (key, acc), jnp.arange(nbands))
    # Average and scale to target sill
    avg = acc / nbands
    sim = avg * jnp.sqrt(jnp.maximum(sill, 1e-12))
    return sim


