"""Sequential Gaussian Simulation (SGSIM) in JAX-compatible style.

Provides a simplified SGS implementation:
- Works in Gaussian space; optional normal-score fit/back-transform.
- Uses k-nearest conditioning data (original + simulated) at each node.
"""

from __future__ import annotations

from typing import Tuple, Optional

import jax
import jax.numpy as jnp

from ..core.random import make_prng_key, split_key, normal
from ..core.neighbor import k_nearest_indices, octant_k_indices, within_radius_indices
from ..core.kriging import ordinary_kriging
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
    """Sequential Gaussian simulation given normal-score data.

    Args:
        data_coords: (n, d)
        data_vals_ns: (n,) normal-score values
        grid_coords: (m, d) nodes to simulate
        kernel_name: covariance model name
        sill, range_, nugget: covariance parameters
        k_neighbors: max neighbors from conditioning set
        seed: RNG seed

    Returns:
        Simulated normal-score values (m,)
    """
    kernel = _get_kernel(kernel_name)
    m = grid_coords.shape[0]
    key = make_prng_key(seed)
    # Random visiting order
    key, sub = jax.random.split(key)
    path = jax.random.permutation(sub, m)

    # Conditioning set starts with data
    cond_coords = data_coords
    cond_vals = data_vals_ns

    sim_vals = jnp.zeros((m,), dtype=jnp.float32)

    def body_fun(carry, idx):
        key, cond_coords, cond_vals, sim_vals = carry
        node_id = path[idx]
        x = grid_coords[node_id]
        # select neighbors
        all_coords = cond_coords
        all_vals = cond_vals
        # Optional radius filtering
        if search_radius is not None and search_radius > 0:
            idxr = within_radius_indices(all_coords, x, float(search_radius))
            valid = idxr >= 0
            selr = idxr[valid]
            all_coords = all_coords[selr]
            all_vals = all_vals[selr]
        kk = jnp.minimum(k_neighbors, all_coords.shape[0])
        # Octant-balanced selection when possible
        def choose_idx():
            if use_octants and all_coords.shape[1] in (2, 3) and k_neighbors >= (4 if all_coords.shape[1] == 2 else 8):
                kpo = int(jnp.ceil(k_neighbors / (4 if all_coords.shape[1] == 2 else 8)))
                return octant_k_indices(all_coords, x, kpo, max_total=int(k_neighbors))
            return k_nearest_indices(all_coords, x, kk)
        nidx = choose_idx()
        sel_c = all_coords[nidx]
        sel_v = all_vals[nidx]
        mean, var = ordinary_kriging(sel_c, sel_v, x, kernel, sill, range_, nugget)
        var = jnp.maximum(var, 1e-8)
        key, ns_key = jax.random.split(key)
        z = jax.random.normal(ns_key, ())
        sim_val = mean + jnp.sqrt(var) * z
        sim_vals = sim_vals.at[node_id].set(sim_val)
        # augment conditioning set
        cond_coords = jnp.concatenate([cond_coords, x[None, :]], axis=0)
        cond_vals = jnp.concatenate([cond_vals, jnp.array([sim_val])], axis=0)
        return (key, cond_coords, cond_vals, sim_vals), None

    (key, _, _, sim_vals), _ = jax.lax.scan(body_fun, (key, cond_coords, cond_vals, sim_vals), jnp.arange(m))
    return sim_vals


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


