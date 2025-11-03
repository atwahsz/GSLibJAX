from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.varmap import variogram_map

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    vcol: str = typer.Argument(..., help="Value column"),
    hx: float = typer.Argument(..., help="Lag cell size X"),
    hy: float = typer.Argument(..., help="Lag cell size Y"),
    nx: int = typer.Argument(..., help="Number of cells X"),
    ny: int = typer.Argument(..., help="Number of cells Y"),
    max_range: float = typer.Option(None, help="Max distance for pairs"),
    output_path: str = typer.Argument(..., help="Output GSLIB table (long format)"),
):
    df = read_gslib_table(input_path)
    coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    vals = jnp.asarray(df[vcol].values)
    xc, yc, gm = variogram_map(coords, vals, hx, hy, nx, ny, max_range)
    # Write long format: hx, hy, gamma
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            rows.append((float(xc[ix]), float(yc[iy]), float(gm[iy, ix])))
    out = pd.DataFrame(rows, columns=["hx", "hy", "gamma"])
    write_gslib_table(out, output_path, title="VARMAP")


if __name__ == "__main__":
    app()


