"""Indicator Kriging (IK3D) simplified implementation.

For a set of thresholds, converts data to indicators and performs ordinary
kriging per indicator to estimate probabilities, then outputs probabilities and
optional most-probable category.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import jax
import jax.numpy as jnp

from ..core.anisotropy import apply_anisotropy
from ..core.neighbor import k_nearest_indices
from ..core.kriging import ordinary_kriging
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


def ik3d(
    data_coords: jnp.ndarray,
    data_values: jnp.ndarray,
    pred_coords: jnp.ndarray,
    thresholds: jnp.ndarray,
    kernel_name: str,
    range_: float,
    nugget: float,
    k_neighbors: int,
    ranges: Optional[Tuple[float, float, float]] = None,
    angles_deg: Optional[Tuple[float, float, float]] = None,
) -> jnp.ndarray:
    """Indicator kriging probabilities for thresholds.

    Returns an array of shape (m, nt) of probabilities P(Z <= t_k).
    """
    kernel = _get_kernel(kernel_name)
    # Transform coords if anisotropy specified
    if ranges is not None and angles_deg is not None:
        t_data = apply_anisotropy(data_coords, ranges, angles_deg)
        t_pred = apply_anisotropy(pred_coords, ranges, angles_deg)
        eff_range = 1.0
    else:
        t_data = data_coords
        t_pred = pred_coords
        eff_range = range_

    # Build indicator data per threshold: I = 1{Z <= t}
    nt = thresholds.shape[0]

    def predict_for_threshold(t: float) -> jnp.ndarray:
        ind = (data_values <= t).astype(jnp.float32)

        def predict_one(xp: jnp.ndarray):
            kk = jnp.minimum(k_neighbors, t_data.shape[0])
            idx = k_nearest_indices(t_data, xp, kk)
            sel_c = t_data[idx]
            sel_v = ind[idx]
            # Indicator variance sill ~ p(1-p); we use sill=1 in kernel; OK will handle mean via Lagrange
            est, var = ordinary_kriging(sel_c, sel_v, xp, kernel, 1.0, eff_range, nugget)
            # Clamp to probability range
            return jnp.clip(est, 0.0, 1.0)

        return jax.vmap(predict_one)(t_pred)

    probs = jax.vmap(predict_for_threshold)(thresholds)
    # shape (nt, m) -> (m, nt)
    return probs.T


