from __future__ import annotations

import os
import tempfile
import pandas as pd

from gslibjax.io.gslib_table import read_gslib_table, write_gslib_table


def test_gslib_io_roundtrip():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [2.0, 3.0], "v": [4.0, 5.0]})
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "t.tbl")
        write_gslib_table(df, pth, title="TEST")
        df2 = read_gslib_table(pth)
        assert list(df2.columns) == ["x", "y", "v"]
        assert df2.shape == df.shape


