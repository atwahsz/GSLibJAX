from __future__ import annotations

import jax.numpy as jnp

from gslibjax.core.kriging import ordinary_kriging_batch
from gslibjax.core.covariance import spherical


def test_self_prediction_reproduces_data():
    # simple 2D coords and values
    coords = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    values = jnp.array([0.0, 1.0, 1.0, 2.0], dtype=jnp.float32)

    # Predict at same locations; with low nugget and reasonable range we should get values back
    est, var = ordinary_kriging_batch(coords, values, coords, spherical, sill=1.0, range_=10.0, nugget=1e-6)

    assert est.shape == (4,) and var.shape == (4,)
    # close to original values
    assert jnp.max(jnp.abs(est - values)) < 1e-3
    # small variance on data
    assert jnp.all(var < 1e-3)


