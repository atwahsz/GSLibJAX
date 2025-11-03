from __future__ import annotations

import os, tempfile
import jax.numpy as jnp
import pandas as pd

from gslibjax.core.neighbor import k_nearest_indices, octant_k_indices
from gslibjax.io.params import read_param_file
from gslibjax.io.gslib_table import read_gslib_table, write_gslib_table


def test_neighbors_basic():
    pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=jnp.float32)
    q = jnp.array([0.2, 0.2], dtype=jnp.float32)
    idx = k_nearest_indices(pts, q, 2)
    assert idx.shape[0] == 2
    idx2 = octant_k_indices(pts, q, 1, max_total=3)
    assert idx2.shape[0] <= 3


def test_params_and_table_io(tmp_path):
    # params: filter comments and blanks
    pfile = tmp_path / "p.par"
    pfile.write_text("""
    # comment
    data.tbl
    1
    2
    3
    """.strip())
    tokens = read_param_file(str(pfile))
    assert tokens[0] == "data.tbl"

    # table roundtrip
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    tpath = tmp_path / "t.tbl"
    write_gslib_table(df, str(tpath))
    df2 = read_gslib_table(str(tpath))
    assert list(df2.columns) == ["a", "b"]


