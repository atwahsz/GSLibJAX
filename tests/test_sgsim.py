from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gslibjax.tools.sgsim import sgsim


def test_sgsim_shapes_and_determinism():
    # make a tiny synthetic 2D problem (keeps tests fast)
    xs = np.arange(6, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid_coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))

    # three conditioning points in a gradient field
    data_coords = jnp.array([[0.0, 0.0], [5.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)
    data_values = jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32)

    sim1 = sgsim(
        data_coords,
        data_values,
        grid_coords,
        kernel_name="spherical",
        sill=1.0,
        range_=5.0,
        nugget=0.0,
        k_neighbors=8,
        seed=42,
        use_nscore=True,
    )
    sim2 = sgsim(
        data_coords,
        data_values,
        grid_coords,
        kernel_name="spherical",
        sill=1.0,
        range_=5.0,
        nugget=0.0,
        k_neighbors=8,
        seed=42,
        use_nscore=True,
    )

    assert sim1.shape == (grid_coords.shape[0],)
    assert jnp.all(jnp.isfinite(sim1))
    # deterministic with same seed
    assert jnp.max(jnp.abs(sim1 - sim2)) == 0.0


