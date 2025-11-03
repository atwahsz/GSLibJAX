"""Ordinary Kriging core routines using JAX.

Implements a basic ordinary kriging system with an arbitrary covariance kernel
compatible with the simple `CovModel` interface.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


CovKernel = Callable[[jnp.ndarray, float, float], jnp.ndarray]


def _pairwise_distances(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a2 = jnp.sum(a * a, axis=1, keepdims=True)
    b2 = jnp.sum(b * b, axis=1, keepdims=True).T
    d2 = jnp.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return jnp.sqrt(d2)


def covariance_matrix(coords: jnp.ndarray, kernel: CovKernel, sill: float, range_: float, nugget: float = 0.0) -> jnp.ndarray:
    """Build covariance matrix C for data locations.

    Args:
        coords: (n, d)
        kernel: covariance kernel(distance, sill, range_) -> covariance
        sill: kernel sill
        range_: kernel range
        nugget: nugget effect added to diagonal

    Returns:
        (n, n) covariance matrix.
    """
    d = _pairwise_distances(coords, coords)
    c = kernel(d, sill, range_)
    if nugget != 0.0:
        c = c.at[jnp.diag_indices(c.shape[0])].add(nugget)
    return c


def covariance_vector(x: jnp.ndarray, coords: jnp.ndarray, kernel: CovKernel, sill: float, range_: float) -> jnp.ndarray:
    """Covariance vector between prediction location ``x`` and data.

    Args:
        x: (d,)
        coords: (n, d)
        kernel: covariance kernel
        sill: sill
        range_: range

    Returns:
        (n,) covariance vector.
    """
    x2 = x[None, :]
    d = _pairwise_distances(coords, x2)[:, 0]
    return kernel(d, sill, range_)


def ordinary_kriging(
    coords: jnp.ndarray,
    values: jnp.ndarray,
    x: jnp.ndarray,
    kernel: CovKernel,
    sill: float,
    range_: float,
    nugget: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ordinary Kriging prediction and variance at location ``x``.

    Args:
        coords: (n, d) data coordinates
        values: (n,) data values
        x: (d,) prediction location
        kernel: covariance kernel(distance, sill, range_)
        sill: kernel sill
        range_: kernel range
        nugget: diagonal nugget on data covariance

    Returns:
        (estimate, variance)
    """
    n = coords.shape[0]
    c = covariance_matrix(coords, kernel, sill, range_, nugget)
    k = covariance_vector(x, coords, kernel, sill, range_)
    # Build OK system
    a = jnp.zeros((n + 1, n + 1), dtype=c.dtype)
    a = a.at[:n, :n].set(c)
    a = a.at[:n, n].set(1.0)
    a = a.at[n, :n].set(1.0)
    b = jnp.concatenate([k, jnp.array([1.0], dtype=c.dtype)])

    sol = jnp.linalg.solve(a, b)
    w = sol[:n]
    mu = sol[n]
    estimate = jnp.dot(w, values)
    c0 = kernel(jnp.array(0.0), sill, range_) + nugget
    variance = c0 - jnp.dot(w, k) - mu
    return estimate, variance


def ordinary_kriging_batch(
    coords: jnp.ndarray,
    values: jnp.ndarray,
    x_pred: jnp.ndarray,
    kernel: CovKernel,
    sill: float,
    range_: float,
    nugget: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized OK for multiple prediction points.

    Args are analogous to ``ordinary_kriging`` with ``x_pred`` shape (m, d).

    Returns:
        (estimates, variances) each shape (m,)
    """
    ok_fn = jax.vmap(lambda xp: ordinary_kriging(coords, values, xp, kernel, sill, range_, nugget), in_axes=(None))
    est, var = ok_fn(x_pred)
    return est, var


