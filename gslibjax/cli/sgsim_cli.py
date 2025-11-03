from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.sgsim import sgsim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    data_path: str = typer.Argument(..., help="Input conditioning data GSLIB table"),
    xcol: str = typer.Argument(..., help="Data X column"),
    ycol: str = typer.Argument(..., help="Data Y column"),
    zcol: str = typer.Option(None, help="Data Z column (optional)"),
    vcol: str = typer.Argument(..., help="Data value column"),
    grid_path: str = typer.Argument(..., help="Grid nodes GSLIB table"),
    gxcol: str = typer.Argument(..., help="Grid X column"),
    gycol: str = typer.Argument(..., help="Grid Y column"),
    gzcol: str = typer.Option(None, help="Grid Z column (optional)"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    sill: float = typer.Argument(..., help="Sill"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Nugget"),
    kneigh: int = typer.Option(12, help="k-nearest neighbors per node"),
    seed: int = typer.Option(12345, help="Random seed"),
    no_nscore: bool = typer.Option(False, help="Disable normal-score transform/back-transform"),
    search_radius: float = typer.Option(None, help="Search radius (units of coords)"),
    no_octants: bool = typer.Option(False, help="Disable octant-balanced selection"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with simulation"),
):
    d = read_gslib_table(data_path)
    if zcol is None:
        dc = jnp.stack([jnp.asarray(d[xcol].values), jnp.asarray(d[ycol].values)], axis=1)
    else:
        dc = jnp.stack([jnp.asarray(d[xcol].values), jnp.asarray(d[ycol].values), jnp.asarray(d[zcol].values)], axis=1)
    dv = jnp.asarray(d[vcol].values)

    g = read_gslib_table(grid_path)
    if gzcol is None:
        gc = jnp.stack([jnp.asarray(g[gxcol].values), jnp.asarray(g[gycol].values)], axis=1)
    else:
        gc = jnp.stack([jnp.asarray(g[gxcol].values), jnp.asarray(g[gycol].values), jnp.asarray(g[gzcol].values)], axis=1)

    sim = sgsim(dc, dv, gc, kernel, sill, range_, nugget, kneigh, seed, use_nscore=(not no_nscore), search_radius=search_radius, use_octants=(not no_octants))
    gout = g.copy()
    gout[vcol + "_sgsim"] = pd.Series(sim)
    write_gslib_table(gout, output_path, title="SGSIM Output")


if __name__ == "__main__":
    app()


