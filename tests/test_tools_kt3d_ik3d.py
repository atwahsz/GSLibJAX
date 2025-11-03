from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gslibjax.tools.kt3d import ok_predict
from gslibjax.tools.ik3d import ik3d


def test_ok_predict_shapes_and_anisotropy():
    xs = np.arange(6, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    data_coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    data_values = jnp.asarray((X + Y).ravel())

    pred_coords = data_coords[::5]
    est, var = ok_predict(
        data_coords,
        data_values,
        pred_coords,
        kernel_name="spherical",
        sill=1.0,
        range_=5.0,
        nugget=0.0,
        k_neighbors=8,
        ranges=(10.0, 5.0, 2.0),
        angles_deg=(0.0, 0.0, 0.0),
    )
    assert est.shape == (pred_coords.shape[0],)
    assert var.shape == (pred_coords.shape[0],)


def test_ik3d_probabilities():
    # simple grid and linear values
    xs = np.arange(6, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    data_coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    data_values = jnp.asarray((X + Y).ravel())
    pred_coords = data_coords[::10]
    thresholds = jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32)
    probs = ik3d(data_coords, data_values, pred_coords, thresholds, kernel_name="spherical", range_=5.0, nugget=0.0, k_neighbors=8)
    assert probs.shape == (pred_coords.shape[0], thresholds.shape[0])
    assert jnp.all(probs >= -1e-6) and jnp.all(probs <= 1.0 + 1e-6)


