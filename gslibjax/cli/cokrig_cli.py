from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.cokrig import ccok_predict

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ccok(
    data_path: str = typer.Argument(..., help="Primary data GSLIB table"),
    xcol: str = typer.Argument(..., help="Data X"),
    ycol: str = typer.Argument(..., help="Data Y"),
    v1col: str = typer.Argument(..., help="Primary value column"),
    pred_path: str = typer.Argument(..., help="Prediction grid/table"),
    pxcol: str = typer.Argument(..., help="Pred X"),
    pycol: str = typer.Argument(..., help="Pred Y"),
    v2col: str = typer.Argument(..., help="Collocated secondary column"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    sill1: float = typer.Argument(..., help="Primary sill"),
    range1: float = typer.Argument(..., help="Primary range"),
    nugget1: float = typer.Option(0.0, help="Primary nugget"),
    rho12: float = typer.Argument(..., help="Corr(Z1,Z2) at zero lag"),
    sill2: float = typer.Argument(..., help="Secondary variance (sill)"),
    output_path: str = typer.Argument(..., help="Output table with ccok estimate/var"),
):
    d = read_gslib_table(data_path)
    dc = jnp.stack([jnp.asarray(d[xcol].values), jnp.asarray(d[ycol].values)], axis=1)
    dv1 = jnp.asarray(d[v1col].values)

    p = read_gslib_table(pred_path)
    pc = jnp.stack([jnp.asarray(p[pxcol].values), jnp.asarray(p[pycol].values)], axis=1)
    z2 = jnp.asarray(p[v2col].values)

    est, var = ccok_predict(dc, dv1, pc, z2, kernel, sill1, range1, nugget1, rho12, sill2)
    out = p.copy()
    out[v1col + "_ccok"] = pd.Series(est)
    out[v1col + "_var"] = pd.Series(var)
    write_gslib_table(out, output_path, title="CCoK Output")


if __name__ == "__main__":
    app()


