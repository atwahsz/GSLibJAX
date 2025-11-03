"""Simulated annealing (SASIM) to match a target variogram model.

This simplified SASIM adjusts continuous values at fixed coordinates to reduce
the squared error between the experimental omnidirectional variogram and a
target model variogram.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .gamv import experimental_variogram
from .vmodel import model_gamma


def _energy(values: jnp.ndarray, coords: jnp.ndarray, lag: float, nlag: int, nugget: float, structures) -> jnp.ndarray:
    centers, gamma_exp, npairs = experimental_variogram(values, coords, lag, nlag)
    gamma_model = model_gamma(centers, nugget, structures)
    w = jnp.where(npairs > 0, npairs.astype(jnp.float32), 0.0)
    return jnp.sum(w * (gamma_exp - gamma_model) ** 2)


def sasim(
    init_values: jnp.ndarray,
    coords: jnp.ndarray,
    lag: float,
    nlag: int,
    nugget: float,
    structures: Tuple[Tuple[str, float, float], ...],
    steps: int,
    temp0: float,
    temp_decay: float,
    noise_sigma: float,
    seed: int,
) -> jnp.ndarray:
    """Run simulated annealing to fit a target variogram model.

    Args:
        init_values: (m,) initial continuous values
        coords: (m, d) fixed coordinates
        lag, nlag: experimental variogram settings
        nugget, structures: target model params (see vmodel.model_gamma)
        steps: total annealing steps
        temp0: initial temperature
        temp_decay: multiplicative decay per step (e.g., 0.999)
        noise_sigma: proposal stddev for value perturbation
        seed: RNG seed

    Returns:
        Optimized values (m,)
    """
    key = jax.random.PRNGKey(seed)
    y = init_values.astype(jnp.float32)
    e = _energy(y, coords, lag, nlag, nugget, structures)

    def one_step(carry, t_idx):
        key, y, e = carry
        key, k1, k2 = jax.random.split(key, 3)
        i = jax.random.randint(k1, (), 0, y.shape[0])
        # propose local Gaussian perturbation
        delta = jax.random.normal(k2, ()) * noise_sigma
        y_new = y.at[i].add(delta)
        e_new = _energy(y_new, coords, lag, nlag, nugget, structures)
        temp = temp0 * (temp_decay ** (t_idx))
        accept = jnp.where(e_new < e, True, jax.random.uniform(key, ()) < jnp.exp(-(e_new - e) / jnp.maximum(temp, 1e-12)))
        y_next = jnp.where(accept, y_new, y)
        e_next = jnp.where(accept, e_new, e)
        return (key, y_next, e_next), None

    (key, y_opt, _), _ = jax.lax.scan(one_step, (key, y, e), jnp.arange(steps))
    return y_opt


