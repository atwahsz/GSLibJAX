"""Normal score transform (nscore) in JAX-compatible style.

Provides forward and inverse transforms using empirical CDF with Gaussian mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class NScoreModel:
    """Stores the empirical mapping needed for inverse transform.

    Attributes:
        sorted_values: Sorted original data values.
        quantiles: Corresponding Gaussian quantiles (same length).
    """

    sorted_values: jnp.ndarray
    quantiles: jnp.ndarray


def fit_nscore(values: jnp.ndarray, eps: float = 1e-6) -> NScoreModel:
    """Fit normal-score mapping from data values.

    Args:
        values: Array of shape (n,).
        eps: Small number to avoid 0/1 CDF.

    Returns:
        NScoreModel.
    """
    v = jnp.asarray(values).reshape(-1)
    sorted_values = jnp.sort(v)
    n = sorted_values.shape[0]
    ranks = jnp.arange(1, n + 1, dtype=jnp.float32)
    # Empirical CDF with plotting position ~ (i - 0.5)/n
    u = jnp.clip((ranks - 0.5) / n, eps, 1 - eps)
    q = jax.scipy.stats.norm.ppf(u)
    return NScoreModel(sorted_values=sorted_values, quantiles=q)


def nscore_transform(values: jnp.ndarray, model: NScoreModel) -> jnp.ndarray:
    """Map data to normal scores via piecewise-linear interpolation on ECDF.

    Args:
        values: Array of shape (n,).
        model: Fitted model.

    Returns:
        Normal-score values shaped like input.
    """
    v = jnp.asarray(values).reshape(-1)
    sv, q = model.sorted_values, model.quantiles
    idx = jnp.clip(jnp.searchsorted(sv, v, side="left"), 1, sv.shape[0] - 1)
    x0 = sv[idx - 1]
    x1 = sv[idx]
    y0 = q[idx - 1]
    y1 = q[idx]
    t = jnp.where(x1 > x0, (v - x0) / (x1 - x0), 0.0)
    y = y0 + t * (y1 - y0)
    return y.reshape(values.shape)


def nscore_inverse(z: jnp.ndarray, model: NScoreModel) -> jnp.ndarray:
    """Inverse mapping from normal scores to original values via interpolation.

    Args:
        z: Normal-score values.
        model: Fitted model.

    Returns:
        Approximated original-scale values.
    """
    zf = jnp.asarray(z).reshape(-1)
    sv, q = model.sorted_values, model.quantiles
    idx = jnp.clip(jnp.searchsorted(q, zf, side="left"), 1, q.shape[0] - 1)
    x0 = q[idx - 1]
    x1 = q[idx]
    y0 = sv[idx - 1]
    y1 = sv[idx]
    t = jnp.where(x1 > x0, (zf - x0) / (x1 - x0), 0.0)
    y = y0 + t * (y1 - y0)
    return y.reshape(z.shape)


