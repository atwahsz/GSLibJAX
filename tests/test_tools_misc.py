from __future__ import annotations

import os, tempfile
import numpy as np
import jax.numpy as jnp
import pandas as pd

from gslibjax.tools.nscore import fit_nscore, nscore_transform, nscore_inverse
from gslibjax.tools.vmodel import model_gamma
from gslibjax.tools.declus import declus_weights
from gslibjax.tools.varmap import variogram_map
from gslibjax.tools.backtr import save_nscore_model, load_nscore_model
from gslibjax.tools.lusim import lusim
from gslibjax.tools.gtsim import gtsim


def test_nscore_roundtrip_and_model_save(tmp_path):
    x = jnp.array([0.1, 0.2, 0.3, 0.9], dtype=jnp.float32)
    m = fit_nscore(x)
    z = nscore_transform(x, m)
    xr = nscore_inverse(z, m)
    assert jnp.max(jnp.abs(x - xr)) < 1e-5
    p = tmp_path / "m.json"
    save_nscore_model(str(p), m)
    m2 = load_nscore_model(str(p))
    z2 = nscore_transform(x, m2)
    assert jnp.max(jnp.abs(z - z2)) < 1e-6


def test_vmodel_gamma_zero_lag():
    h = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    g = model_gamma(h, nugget=0.1, structures=[("spherical", 0.8, 10.0)])
    assert jnp.isclose(g[0], 0.1)


def test_declus_weights_mean_one():
    pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.1], [3.5, 3.5]], dtype=jnp.float32)
    w = declus_weights(pts, (1.0, 1.0, 1.0))
    assert jnp.isfinite(w).all() and w.shape == (4,)
    assert np.isclose(np.asarray(w).mean(), 1.0, atol=1e-6)


def test_varmap_shapes():
    xs = np.arange(6, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    vals = jnp.asarray((X + Y).ravel())
    xc, yc, gm = variogram_map(coords, vals, hx=1.0, hy=1.0, nx=3, ny=4)
    assert gm.shape == (4, 3)


def test_lusim_and_gtsim_shapes():
    xs = np.arange(6, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    lu = lusim(coords, kernel_name="spherical", sill=1.0, range_=5.0, nugget=0.0, seed=7)
    gt = gtsim(coords, kernel_name="spherical", sill=1.0, range_=5.0, nugget=0.0, nbands=8, seed=9)
    assert lu.shape == (coords.shape[0],)
    assert gt.shape == (coords.shape[0],)


