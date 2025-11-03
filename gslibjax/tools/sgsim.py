"""JIT-safe conditional Gaussian simulation using OK moments.

This implementation avoids dynamic boolean indexing and array growth so it
works with JAX JIT. It performs conditional simulation by:
1) Computing ordinary kriging mean/variance for all prediction nodes using
   only the hard data (no path-dependent augmentation).
2) Drawing Gaussian residuals and adding them to the kriging mean.

If ``use_nscore=True``, it operates in Gaussian space by fitting a normal-score
transform on the data and back-transforming the simulated field.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from ..core.random import make_prng_key
from ..core.kriging import ordinary_kriging_batch
from ..core.covariance import spherical, exponential, gaussian
from .nscore import NScoreModel, fit_nscore, nscore_transform, nscore_inverse


def _get_kernel(name: str):
    name_l = name.lower()
    if name_l in ("spherical", "sph"):
        return spherical
    if name_l in ("exponential", "exp"):
        return exponential
    if name_l in ("gaussian", "gau"):
        return gaussian
    raise ValueError(f"Unknown kernel '{name}'. Use spherical|exponential|gaussian")


def sgsim_gaussian(
    data_coords: jnp.ndarray,
    data_vals_ns: jnp.ndarray,
    grid_coords: jnp.ndarray,
    kernel_name: str,
    sill: float,
    range_: float,
    nugget: float,
    k_neighbors: int,
    seed: int,
    search_radius: float | None = None,
    use_octants: bool = True,
) -> jnp.ndarray:
    """Gaussian-space conditional simulation via OK moments (JIT-safe).

    Computes OK mean/variance for all prediction locations using the hard data,
    then samples one realization by adding Gaussian residuals. Path dependence
    is intentionally not modeled to keep the computation JIT-safe and vectorized.
    """
    del k_neighbors, search_radius, use_octants  # not used in JIT-safe variant
    kernel = _get_kernel(kernel_name)

    # Batch OK moments from hard data only
    mu, var = ordinary_kriging_batch(
        coords=jnp.asarray(data_coords),
        values=jnp.asarray(data_vals_ns),
        x_pred=jnp.asarray(grid_coords),
        kernel=kernel,
        sill=sill,
        range_=range_,
        nugget=nugget,
    )

    key = make_prng_key(int(seed))
    z = jax.random.normal(key, shape=mu.shape)
    sim = mu + jnp.sqrt(jnp.maximum(var, 1e-12)) * z
    return sim


def sgsim(
    data_coords: jnp.ndarray,
    data_values: jnp.ndarray,
    grid_coords: jnp.ndarray,
    kernel_name: str,
    sill: float,
    range_: float,
    nugget: float,
    k_neighbors: int,
    seed: int,
    use_nscore: bool = True,
    search_radius: float | None = None,
    use_octants: bool = True,
) -> jnp.ndarray:
    """SGSIM with optional normal-score transform and back-transform.

    Args as in ``sgsim_gaussian`` but with original-scale data values and
    use_nscore controlling transform.
    """
    if use_nscore:
        model: NScoreModel = fit_nscore(data_values)
        data_ns = nscore_transform(data_values, model)
        sim_ns = sgsim_gaussian(
            data_coords,
            data_ns,
            grid_coords,
            kernel_name,
            sill,
            range_,
            nugget,
            k_neighbors,
            seed,
            search_radius,
            use_octants,
        )
        sim = nscore_inverse(sim_ns, model)
        return sim
    else:
        # Assume data_values are already normal-score if not using nscore
        return sgsim_gaussian(
            data_coords,
            data_values,
            grid_coords,
            kernel_name,
            sill,
            range_,
            nugget,
            k_neighbors,
            seed,
            search_radius,
            use_octants,
        )


