"""Basic covariance models comparable to GSLIB's variogram structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass
class CovModel:
    """Covariance model with sill and range parameterization.

    Attributes:
        sill: Contribution (variance) of the structure.
        range_: Practical range parameter.
        kernel: Callable mapping distance to covariance in [0, sill].
    """

    sill: float
    range_: float
    kernel: Callable[[jnp.ndarray, float, float], jnp.ndarray]

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.kernel(h, self.sill, self.range_)


def spherical(h: jnp.ndarray, sill: float, range_: float) -> jnp.ndarray:
    hr = jnp.clip(h / jnp.maximum(range_, 1e-12), a_min=0.0, a_max=1.0)
    cov = sill * (1.0 - (1.5 * hr - 0.5 * (hr ** 3)))
    return jnp.where(h <= range_, cov, 0.0)


def exponential(h: jnp.ndarray, sill: float, range_: float) -> jnp.ndarray:
    return sill * jnp.exp(-3.0 * h / jnp.maximum(range_, 1e-12))


def gaussian(h: jnp.ndarray, sill: float, range_: float) -> jnp.ndarray:
    return sill * jnp.exp(-3.0 * (h / jnp.maximum(range_, 1e-12)) ** 2)


def power(h: jnp.ndarray, sill: float, range_: float) -> jnp.ndarray:
    # Power variogram model is not covariance positive-definite for some params; placeholder
    alpha = jnp.clip(range_, 0.0, 2.0)
    return jnp.maximum(0.0, sill - (h ** alpha))


