from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gslibjax.tools.gamv import experimental_variogram


def test_gamv_basic():
    # simple linear ramp field in 2D
    xs = np.arange(10, dtype=np.float32)
    ys = np.arange(8, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    values = jnp.asarray((X + Y).ravel())

    centers, gamma, npairs = experimental_variogram(values, coords, lag=1.0, nlag=5)
    assert centers.shape == (5,) and gamma.shape == (5,) and npairs.shape == (5,)
    assert jnp.all(npairs >= 0)
    assert jnp.all(jnp.isfinite(gamma))


