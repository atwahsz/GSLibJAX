"""Variogram model evaluation (vmodel)."""

from __future__ import annotations

from typing import List, Tuple

import jax.numpy as jnp

from ..core.covariance import spherical, exponential, gaussian


def get_kernel(name: str):
    n = name.lower()
    if n in ("spherical", "sph"):
        return spherical
    if n in ("exponential", "exp"):
        return exponential
    if n in ("gaussian", "gau"):
        return gaussian
    raise ValueError("Unknown kernel")


def model_gamma(
    h: jnp.ndarray,
    nugget: float,
    structures: List[Tuple[str, float, float]],
) -> jnp.ndarray:
    """Compute model semivariogram gamma(h) = nugget + sum(sill - cov_k(h)).

    Each structure tuple is (name, sill, range).
    """
    gamma = jnp.zeros_like(h) + nugget
    for name, sill, range_ in structures:
        k = get_kernel(name)
        cov = k(h, sill, range_)
        gamma = gamma + (sill - cov)
    return gamma


