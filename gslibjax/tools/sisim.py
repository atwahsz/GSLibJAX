"""Sequential Indicator Simulation (SISIM) with thresholds.

Simulates categorical outcomes defined by ordered thresholds t[0] < ... < t[nt-1].
Categories are indexed 0..nt where category k corresponds to:
  k = 0: (-inf, t0]
  k = i: (t_{i-1}, t_i] for 1<=i<=nt-1
  k = nt: (t_{nt-1}, +inf)

Uses indicator kriging of cumulative indicators I(Z<=t_i) to evaluate
probabilities per category at each node and samples a category.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from ..core.neighbor import k_nearest_indices, octant_k_indices, within_radius_indices
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


def _to_category_indices(values: jnp.ndarray, thresholds: jnp.ndarray) -> jnp.ndarray:
    # number of thresholds <= value gives category index
    # searchsorted returns index where to insert to keep order; side='right' counts <=
    return jnp.searchsorted(thresholds, values, side="right").astype(jnp.int32)


def sisim(
    data_coords: jnp.ndarray,
    data_values: jnp.ndarray,
    grid_coords: jnp.ndarray,
    thresholds: jnp.ndarray,
    kernel_name: str,
    range_: float,
    nugget: float,
    k_neighbors: int,
    seed: int,
    search_radius: float | None = None,
    use_octants: bool = True,
) -> jnp.ndarray:
    """Sequential Indicator Simulation producing category indices per node.

    Args:
        data_coords: (n, d)
        data_values: (n,)
        grid_coords: (m, d)
        thresholds: (nt,) sorted ascending
        kernel_name: covariance kernel for indicators (sill=1 used internally)
        range_: model range used for distances
        nugget: indicator nugget
        k_neighbors: neighbors per node
        seed: RNG seed
        search_radius: optional search radius
        use_octants: enable octant-balanced selection

    Returns:
        Simulated category indices (m,) in 0..nt
    """
    kernel = _get_kernel(kernel_name)
    nt = thresholds.shape[0]
    m = grid_coords.shape[0]

    # Conditioning set stores category indices
    data_cat = _to_category_indices(data_values, thresholds)
    cond_coords = data_coords
    cond_cat = data_cat

    key = jax.random.PRNGKey(seed)
    key, kperm = jax.random.split(key)
    path = jax.random.permutation(kperm, m)
    sim_cat = jnp.zeros((m,), dtype=jnp.int32)

    def body_fun(carry, i):
        key, cond_coords, cond_cat, sim_cat = carry
        node_id = path[i]
        x = grid_coords[node_id]

        # Filter by radius if provided
        cc = cond_coords
        ck = cond_cat
        if search_radius is not None and search_radius > 0:
            idxr = within_radius_indices(cc, x, float(search_radius))
            valid = idxr >= 0
            idxr = idxr[valid]
            cc = cc[idxr]
            ck = ck[idxr]

        # Neighbor indices selection helper
        def choose_indices():
            if use_octants and cc.shape[1] in (2, 3) and k_neighbors >= (4 if cc.shape[1] == 2 else 8):
                kpo = int(jnp.ceil(k_neighbors / (4 if cc.shape[1] == 2 else 8)))
                return octant_k_indices(cc, x, kpo, max_total=int(k_neighbors))
            return k_nearest_indices(cc, x, int(jnp.minimum(k_neighbors, cc.shape[0])))

        # Compute cumulative probabilities p_i = P(Z<=t_i)
        p_cum = []
        idx = choose_indices()
        sel_c = cc[idx]
        # For each threshold, build indicator values from category indices
        for ti in range(nt):
            ind = (ck[idx] <= ti).astype(jnp.float32)
            est, var = ordinary_kriging(sel_c, ind, x, kernel, 1.0, range_, nugget)
            p_cum.append(jnp.clip(est, 0.0, 1.0))
        p_cum = jnp.stack(p_cum, axis=0)  # (nt,)

        # Convert cumulative to class probabilities
        probs = jnp.zeros((nt + 1,), dtype=jnp.float32)
        probs = probs.at[0].set(p_cum[0])
        for k in range(1, nt):
            probs = probs.at[k].set(jnp.maximum(p_cum[k] - p_cum[k - 1], 0.0))
        probs = probs.at[nt].set(jnp.maximum(1.0 - p_cum[-1], 0.0))
        # Normalize
        s = jnp.sum(probs)
        probs = jnp.where(s > 0, probs / s, jnp.full_like(probs, 1.0 / (nt + 1)))

        key, ku = jax.random.split(key)
        u = jax.random.uniform(ku, ())
        cdf = jnp.cumsum(probs)
        cat = jnp.searchsorted(cdf, u, side="right")

        sim_cat = sim_cat.at[node_id].set(cat)
        # augment conditioning set with simulated category index
        cond_coords = jnp.concatenate([cond_coords, x[None, :]], axis=0)
        cond_cat = jnp.concatenate([cond_cat, jnp.array([cat], dtype=jnp.int32)], axis=0)
        return (key, cond_coords, cond_cat, sim_cat), None

    (key, _, _, sim_cat), _ = jax.lax.scan(body_fun, (key, cond_coords, cond_cat, sim_cat), jnp.arange(m))
    return sim_cat


