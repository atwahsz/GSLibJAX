from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.lusim import lusim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    grid_path: str = typer.Argument(..., help="Grid nodes GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    sill: float = typer.Argument(..., help="Sill"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Nugget"),
    seed: int = typer.Option(12345, help="Random seed"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with simulation"),
):
    g = read_gslib_table(grid_path)
    coords = jnp.stack([jnp.asarray(g[xcol].values), jnp.asarray(g[ycol].values)], axis=1)
    sim = lusim(coords, kernel, sill, range_, nugget, seed)
    gout = g.copy()
    gout["lusim"] = pd.Series(sim)
    write_gslib_table(gout, output_path, title="LUSIM Output")


if __name__ == "__main__":
    app()


