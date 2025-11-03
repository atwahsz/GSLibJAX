"""Collocated Co-Kriging (CCoK) - simplified implementation.

This implements a pragmatic collocated cokriging estimator for a primary
variable Z1 using a collocated secondary Z2 at each prediction location.

Model assumptions:
- Ordinary kriging of Z1 provides base estimate and weights.
- Linear regression adjustment using collocated Z2* with coefficient
  beta = rho * sqrt(sill1 / sill2), where rho is the correlation between
  Z1 and Z2 at zero lag and sill1/sill2 are variances of primary/secondary.

This corresponds to a common approximation used when only collocated Z2 is
available at prediction nodes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ..core.anisotropy import apply_anisotropy
from ..core.kriging import ordinary_kriging_batch
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


def ccok_predict(
    data_coords: jnp.ndarray,
    data_values: jnp.ndarray,
    pred_coords: jnp.ndarray,
    secondary_collocated: jnp.ndarray,
    kernel_name: str,
    sill1: float,
    range1: float,
    nugget1: float,
    rho12: float,
    sill2: float,
    ranges: Optional[Tuple[float, float, float]] = None,
    angles_deg: Optional[Tuple[float, float, float]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Collocated co-kriging estimate and variance (approximate).

    Args:
        data_coords: (n, d) primary data locations
        data_values: (n,) primary values
        pred_coords: (m, d) prediction points
        secondary_collocated: (m,) collocated secondary value at each pred point
        kernel_name: covariance kernel for primary
        sill1, range1, nugget1: primary covariance params
        rho12: correlation at zero lag between primary and secondary
        sill2: variance of secondary
        ranges, angles_deg: optional anisotropy for coordinates

    Returns:
        (estimate, variance) arrays of shape (m,)
    """
    kernel = get_kernel(kernel_name)
    # Transform to isotropic space if provided
    if ranges is not None and angles_deg is not None:
        t_data = apply_anisotropy(data_coords, ranges, angles_deg)
        t_pred = apply_anisotropy(pred_coords, ranges, angles_deg)
        eff_range = 1.0
    else:
        t_data = data_coords
        t_pred = pred_coords
        eff_range = range1

    # Base OK estimate/variance for primary
    est_ok, var_ok = ordinary_kriging_batch(t_data, data_values, t_pred, kernel, sill1, eff_range, nugget1)

    # Collocated adjustment
    beta = float(rho12) * jnp.sqrt(jnp.maximum(sill1, 1e-12) / jnp.maximum(sill2, 1e-12))
    # Assume unknown mean -> centered around OK estimate baseline; we add beta * (z2* - mean2)
    # Without explicit mean2 we take z2* centered approximately by subtracting its sample mean.
    mean2 = jnp.mean(secondary_collocated)
    est = est_ok + beta * (secondary_collocated - mean2)

    # Variance reduced by collocated information approximately by beta^2 * var2_resid
    var2 = jnp.var(secondary_collocated)
    var = jnp.maximum(var_ok - (beta ** 2) * var2, 1e-8)
    return est, var


