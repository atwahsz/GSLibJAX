from __future__ import annotations

import jax.numpy as jnp

from gslibjax.core.covariance import spherical, exponential, gaussian
from gslibjax.core.distance import squared_euclidean, rotation_matrix_3d
from gslibjax.core.anisotropy import apply_anisotropy


def test_covariance_monotonic():
    h = jnp.linspace(0, 10, 11)
    for kernel in (spherical, exponential, gaussian):
        c = kernel(h, sill=1.0, range_=5.0)
        assert c.shape == h.shape
        assert c[0] <= 1.0 + 1e-6
        # covariance nonnegative
        assert jnp.all(c >= -1e-8)


def test_squared_euclidean():
    a = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    b = jnp.array([[0.0, 0.0], [0.0, 2.0]])
    d2 = squared_euclidean(a, b)
    assert d2.shape == (2, 2)
    assert d2[0, 0] == 0.0
    assert jnp.isclose(d2[1, 1], 5.0)


def test_anisotropy_shapes():
    pts = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)
    tr = apply_anisotropy(pts, ranges=(10.0, 5.0, 2.0), angles_deg=(10.0, 0.0, 0.0))
    assert tr.shape == pts.shape


