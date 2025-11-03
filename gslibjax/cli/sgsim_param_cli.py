from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..io.params import read_param_file
from ..tools.sgsim import sgsim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(param_path: str = typer.Argument(..., help="GSLIB-style parameter file")):
    p = read_param_file(param_path)
    # SGSIM param mapping:
    # 0: data file
    # 1: xcol idx, 2: ycol idx, 3: vcol idx
    # 4: grid file, 5: gxcol idx, 6: gycol idx
    # 7: kernel name, 8: sill, 9: range, 10: nugget
    # 11: k-neighbors, 12: seed, 13: use_nscore (0/1), 14: output file
    # Optional: 15: zcol idx (data), 16: gzcol idx (grid)
    # Optional: 17-19: ang1, ang2, ang3
    # Optional: 20-22: ar1, ar2, ar3
    # Optional: 23: search_radius
    # Optional: 24: use_octants (0/1)
    dfile = p[0]
    ix, iy, iv = int(p[1]) - 1, int(p[2]) - 1, int(p[3]) - 1
    gfile = p[4]
    igx, igy = int(p[5]) - 1, int(p[6]) - 1
    kernel = p[7]
    sill, range_, nugget = float(p[8]), float(p[9]), float(p[10])
    kneigh = int(p[11])
    seed = int(p[12])
    use_ns = bool(int(p[13])) if len(p) > 13 else True
    out = p[14] if len(p) > 14 else p[13]

    df = read_gslib_table(dfile)
    cols = list(df.columns)

    # Optional z columns
    if len(p) > 16 and p[15].strip() and p[16].strip():
        iz = int(p[15]) - 1
        igz = int(p[16]) - 1
        dcoords = jnp.stack([
            jnp.asarray(df[cols[ix]].values),
            jnp.asarray(df[cols[iy]].values),
            jnp.asarray(df[cols[iz]].values),
        ], axis=1)
        g = read_gslib_table(gfile)
        gcols = list(g.columns)
        gcoords = jnp.stack([
            jnp.asarray(g[gcols[igx]].values),
            jnp.asarray(g[gcols[igy]].values),
            jnp.asarray(g[gcols[igz]].values),
        ], axis=1)
    else:
        dcoords = jnp.stack([jnp.asarray(df[cols[ix]].values), jnp.asarray(df[cols[iy]].values)], axis=1)
        g = read_gslib_table(gfile)
        gcols = list(g.columns)
        gcoords = jnp.stack([jnp.asarray(g[gcols[igx]].values), jnp.asarray(g[gcols[igy]].values)], axis=1)

    dvals = jnp.asarray(df[cols[iv]].values)

    # Optional search radius
    search_radius = None
    if len(p) > 23 and p[23].strip():
        try:
            search_radius = float(p[23])
        except (ValueError, IndexError):
            pass

    use_octants = True
    if len(p) > 24 and p[24].strip():
        try:
            use_octants = bool(int(p[24]))
        except (ValueError, IndexError):
            pass

    sim = sgsim(dcoords, dvals, gcoords, kernel, sill, range_, nugget, kneigh, seed, use_nscore=use_ns, search_radius=search_radius, use_octants=use_octants)
    gout = g.copy()
    vname = cols[iv]
    gout[vname + "_sgsim"] = pd.Series(sim)
    write_gslib_table(gout, out, title="SGSIM Param")


if __name__ == "__main__":
    app()

