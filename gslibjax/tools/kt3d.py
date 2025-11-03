"""High-level kriging utilities.

Provides a simple interface over core ordinary kriging with common kernels.
"""

from __future__ import annotations

from typing import Tuple, Optional

import jax.numpy as jnp

import jax
from ..core.kriging import ordinary_kriging, ordinary_kriging_batch
from ..core.covariance import spherical, exponential, gaussian
from ..core.neighbor import k_nearest_indices, octant_k_indices, within_radius_indices
from ..core.anisotropy import apply_anisotropy


def get_kernel(name: str):
    name_l = name.lower()
    if name_l in ("spherical", "sph"):
        return spherical
    if name_l in ("exponential", "exp"):
        return exponential
    if name_l in ("gaussian", "gau"):
        return gaussian
    raise ValueError(f"Unknown kernel '{name}'. Use spherical|exponential|gaussian")


def ok_predict(
    data_coords: jnp.ndarray,
    data_values: jnp.ndarray,
    pred_coords: jnp.ndarray,
    kernel_name: str,
    sill: float,
    range_: float,
    nugget: float = 0.0,
    k_neighbors: Optional[int] = None,
    ranges: Optional[Tuple[float, float, float]] = None,
    angles_deg: Optional[Tuple[float, float, float]] = None,
    search_radius: Optional[float] = None,
    use_octants: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ordinary kriging predictions for multiple points.

    Args:
        data_coords: (n, d)
        data_values: (n,)
        pred_coords: (m, d)
        kernel_name: 'spherical'|'exponential'|'gaussian'
        sill: covariance sill
        range_: covariance range
        nugget: nugget on the diagonal

    Returns:
        (estimates, variances)
    """
    kernel = get_kernel(kernel_name)
    # Apply anisotropy transform if specified (coordinates scaled to isotropic)
    if ranges is not None and angles_deg is not None:
        t_data = apply_anisotropy(data_coords, ranges, angles_deg)
        t_pred = apply_anisotropy(pred_coords, ranges, angles_deg)
        # In transformed space, the effective model range along major axis is 1.0
        eff_range = 1.0
    else:
        t_data = data_coords
        t_pred = pred_coords
        eff_range = range_

    if k_neighbors is None or k_neighbors <= 0 or k_neighbors >= t_data.shape[0]:
        return ordinary_kriging_batch(t_data, data_values, t_pred, kernel, sill, eff_range, nugget)

    def predict_one(xp: jnp.ndarray):
        # Optional radius filter
        td = t_data
        tv = data_values
        if search_radius is not None and search_radius > 0:
            mask_idx = within_radius_indices(t_data, xp, float(search_radius))
            valid = mask_idx >= 0
            sel = mask_idx[valid]
            td = td[sel]
            tv = tv[sel]
        # Fall back to all if none within radius
        def choose_indices():
            if use_octants and td.shape[1] in (2, 3) and k_neighbors >= (4 if td.shape[1] == 2 else 8):
                kpo = int(jnp.ceil(k_neighbors / (4 if td.shape[1] == 2 else 8)))
                return octant_k_indices(td, xp, kpo, max_total=int(k_neighbors))
            return k_nearest_indices(td, xp, int(k_neighbors))

        idx = choose_indices()
        sel_c = t_data[idx]
        sel_v = data_values[idx]
        est, var = ordinary_kriging(sel_c, sel_v, xp, kernel, sill, eff_range, nugget)
        return est, var

    est, var = jax.vmap(predict_one)(t_pred)
    return est, var


