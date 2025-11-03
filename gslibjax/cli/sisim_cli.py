from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.sisim import sisim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    data_path: str = typer.Argument(..., help="Input conditioning data GSLIB table"),
    xcol: str = typer.Argument(..., help="Data X column"),
    ycol: str = typer.Argument(..., help="Data Y column"),
    vcol: str = typer.Argument(..., help="Data value column"),
    grid_path: str = typer.Argument(..., help="Grid nodes GSLIB table"),
    gxcol: str = typer.Argument(..., help="Grid X column"),
    gycol: str = typer.Argument(..., help="Grid Y column"),
    thresholds: str = typer.Argument(..., help="Comma-separated thresholds e.g. '0.1,0.3,0.5'"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Indicator nugget"),
    kneigh: int = typer.Option(12, help="k-nearest neighbors"),
    seed: int = typer.Option(12345, help="Random seed"),
    search_radius: float = typer.Option(None, help="Search radius (units of coords)"),
    no_octants: bool = typer.Option(False, help="Disable octant-balanced selection"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with category index"),
):
    d = read_gslib_table(data_path)
    dc = jnp.stack([jnp.asarray(d[xcol].values), jnp.asarray(d[ycol].values)], axis=1)
    dv = jnp.asarray(d[vcol].values)

    g = read_gslib_table(grid_path)
    gc = jnp.stack([jnp.asarray(g[gxcol].values), jnp.asarray(g[gycol].values)], axis=1)

    th = jnp.array([float(z) for z in thresholds.split(',')], dtype=jnp.float32)
    cat = sisim(dc, dv, gc, th, kernel, range_, nugget, kneigh, seed, search_radius=search_radius, use_octants=(not no_octants))
    gout = g.copy()
    gout["cat_idx"] = pd.Series(cat)
    write_gslib_table(gout, output_path, title="SISIM Categories")


if __name__ == "__main__":
    app()


