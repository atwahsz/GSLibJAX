from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..io.params import read_param_file
from ..tools.kt3d import ok_predict

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(param_path: str = typer.Argument(..., help="GSLIB-style parameter file")):
    p = read_param_file(param_path)
    # Enhanced mapping for KT3D params:
    # 0: data file
    # 1: xcol idx, 2: ycol idx, 3: vcol idx
    # 4: grid file, 5: gxcol idx, 6: gycol idx
    # 7: kernel name, 8: sill, 9: range, 10: nugget
    # 11: k-neighbors, 12: output file
    # Optional: 13: zcol idx (data), 14: gzcol idx (grid)
    # Optional: 15-17: ang1, ang2, ang3
    # Optional: 18-20: ar1, ar2, ar3
    # Optional: 21: search_radius
    # Optional: 22: use_octants (0/1)
    dfile = p[0]
    ix, iy, iv = int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1
    gfile = p[4]
    igx, igy = int(p[5]) - 1, int(p[6]) - 1
    kernel = p[7]
    sill, range_, nugget = float(p[8]), float(p[9]), float(p[10])
    kneigh = int(p[11]) if int(p[11]) > 0 else None
    out = p[12]

    df = read_gslib_table(dfile)
    cols = list(df.columns)

    # Optional z columns
    if len(p) > 14 and p[13].strip() and p[14].strip():
        iz = int(p[13]) - 1
        igz = int(p[14]) - 1
        dcoords = jnp.stack([
            jnp.asarray(df[cols[ix]].values),
            jnp.asarray(df[cols[iy]].values),
            jnp.asarray(df[cols[iz]].values),
        ], axis=1)
        g = read_gslib_table(gfile)
        gcols = list(g.columns)
        pcoords = jnp.stack([
            jnp.asarray(g[gcols[igx]].values),
            jnp.asarray(g[gcols[igy]].values),
            jnp.asarray(g[gcols[igz]].values),
        ], axis=1)
    else:
        dcoords = jnp.stack([jnp.asarray(df[cols[ix]].values), jnp.asarray(df[cols[iy]].values)], axis=1)
        g = read_gslib_table(gfile)
        gcols = list(g.columns)
        pcoords = jnp.stack([jnp.asarray(g[gcols[igx]].values), jnp.asarray(g[gcols[igy]].values)], axis=1)

    dvals = jnp.asarray(df[cols[iv]].values)

    # Optional anisotropy
    angles = None
    ranges = None
    if len(p) >= 21:
        try:
            ang1, ang2, ang3 = float(p[15]), float(p[16]), float(p[17])
            ar1, ar2, ar3 = float(p[18]), float(p[19]), float(p[20])
            angles = (ang1, ang2, ang3)
            ranges = (ar1, ar2, ar3)
        except (ValueError, IndexError):
            pass

    # Optional search radius
    search_radius = None
    if len(p) > 21 and p[21].strip():
        try:
            search_radius = float(p[21])
        except (ValueError, IndexError):
            pass

    # Optional octants flag
    use_octants = True
    if len(p) > 22 and p[22].strip():
        try:
            use_octants = bool(int(p[22]))
        except (ValueError, IndexError):
            pass

    est, var = ok_predict(
        dcoords, dvals, pcoords, kernel, sill, range_, nugget,
        k_neighbors=kneigh, ranges=ranges, angles_deg=angles,
        search_radius=search_radius, use_octants=use_octants
    )
    gout = g.copy()
    vname = cols[iv]
    gout[vname + "_ok"] = pd.Series(est)
    gout[vname + "_var"] = pd.Series(var)
    write_gslib_table(gout, out, title="KT3D Param OK")


if __name__ == "__main__":
    app()


