from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..io.params import read_param_file
from ..tools.gamv import experimental_variogram

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(param_path: str = typer.Argument(..., help="GSLIB-style parameter file")):
    p = read_param_file(param_path)
    # Enhanced mapping for GAMV params:
    # 0: data file
    # 1: xcol index (1-based)
    # 2: ycol index (1-based)
    # 3: vcol index (1-based)
    # 4: lag
    # 5: nlag
    # 6: output file
    # Optional: 7: zcol index (1-based, if present)
    # Optional: 8-10: ang1, ang2, ang3 (degrees)
    # Optional: 11-13: ar1, ar2, ar3 (ranges)
    data_path = p[0]
    ix = int(p[1]) - 1
    iy = int(p[2]) - 1
    iv = int(p[3]) - 1
    lag = float(p[4])
    nlag = int(p[5])
    out_path = p[6]

    df = read_gslib_table(data_path)
    cols = list(df.columns)
    xcol, ycol, vcol = cols[ix], cols[iy], cols[iv]

    # Optional z column
    if len(p) > 7 and p[7].strip():
        iz = int(p[7]) - 1
        zcol = cols[iz]
        coords = jnp.stack([
            jnp.asarray(df[xcol].values),
            jnp.asarray(df[ycol].values),
            jnp.asarray(df[zcol].values),
        ], axis=1)
    else:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)

    vals = jnp.asarray(df[vcol].values)

    # Optional anisotropy
    angles = None
    ranges = None
    if len(p) >= 14:
        try:
            ang1, ang2, ang3 = float(p[8]), float(p[9]), float(p[10])
            ar1, ar2, ar3 = float(p[11]), float(p[12]), float(p[13])
            angles = (ang1, ang2, ang3)
            ranges = (ar1, ar2, ar3)
        except (ValueError, IndexError):
            pass

    centers, gamma, npairs = experimental_variogram(vals, coords, lag=lag, nlag=nlag, ranges=ranges, angles_deg=angles)
    out = pd.DataFrame({
        "h": pd.Series(centers),
        "gamma": pd.Series(gamma),
        "npairs": pd.Series(npairs),
    })
    write_gslib_table(out, out_path, title="GAMV Param Omnidirectional")


if __name__ == "__main__":
    app()


