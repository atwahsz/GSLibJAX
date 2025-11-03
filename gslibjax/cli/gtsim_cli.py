from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.gtsim import gtsim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    grid_path: str = typer.Argument(..., help="Grid nodes GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    sill: float = typer.Argument(..., help="Sill (variance)"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Nugget"),
    nbands: int = typer.Option(64, help="Number of bands"),
    seed: int = typer.Option(12345, help="Random seed"),
    output_path: str = typer.Argument(..., help="Output GSLIB table"),
):
    g = read_gslib_table(grid_path)
    if zcol is None:
        coords = jnp.stack([jnp.asarray(g[xcol].values), jnp.asarray(g[ycol].values)], axis=1)
    else:
        coords = jnp.stack([jnp.asarray(g[xcol].values), jnp.asarray(g[ycol].values), jnp.asarray(g[zcol].values)], axis=1)
    sim = gtsim(coords, kernel, sill, range_, nugget, nbands, seed)
    gout = g.copy()
    gout["gtsim"] = pd.Series(sim)
    write_gslib_table(gout, output_path, title="GTSIM Output")


if __name__ == "__main__":
    app()


