from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.pfsim import pfsim

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    data_path: str = typer.Argument(..., help="Input data GSLIB table for marginal CDF"),
    vcol: str = typer.Argument(..., help="Data value column"),
    grid_path: str = typer.Argument(..., help="Grid nodes GSLIB table"),
    xcol: str = typer.Argument(..., help="Grid X column"),
    ycol: str = typer.Argument(..., help="Grid Y column"),
    zcol: str = typer.Option(None, help="Grid Z column (optional)"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    range_: float = typer.Argument(..., help="Range"),
    nbands: int = typer.Option(64, help="Number of bands"),
    seed: int = typer.Option(12345, help="Random seed"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with p-field simulation"),
):
    d = read_gslib_table(data_path)
    vals = jnp.asarray(d[vcol].values)
    g = read_gslib_table(grid_path)
    if zcol is None:
        coords = jnp.stack([jnp.asarray(g[xcol].values), jnp.asarray(g[ycol].values)], axis=1)
    else:
        coords = jnp.stack([jnp.asarray(g[xcol].values), jnp.asarray(g[ycol].values), jnp.asarray(g[zcol].values)], axis=1)
    sim = pfsim(vals, coords, kernel, range_, nbands, seed)
    gout = g.copy()
    gout["pfsim"] = pd.Series(sim)
    write_gslib_table(gout, output_path, title="PFSIM Output")


if __name__ == "__main__":
    app()


