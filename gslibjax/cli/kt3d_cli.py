from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.kt3d import ok_predict

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ok(
    data_path: str = typer.Argument(..., help="Input data GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    vcol: str = typer.Argument(..., help="Value column"),
    pred_path: str = typer.Argument(..., help="Prediction points GSLIB table"),
    pxcol: str = typer.Argument(..., help="Pred X column"),
    pycol: str = typer.Argument(..., help="Pred Y column"),
    pzcol: str = typer.Option(None, help="Pred Z column (optional)"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    sill: float = typer.Argument(..., help="Sill"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Nugget"),
    k: int = typer.Option(0, help="k-nearest data for each prediction (0=all)"),
    search_radius: float = typer.Option(None, help="Search radius (units of coords)"),
    no_octants: bool = typer.Option(False, help="Disable octant-balanced selection"),
    ang1: float = typer.Option(None, help="Rotation ang1 (deg) Z"),
    ang2: float = typer.Option(None, help="Rotation ang2 (deg) Y"),
    ang3: float = typer.Option(None, help="Rotation ang3 (deg) X"),
    ar1: float = typer.Option(None, help="Range a1 (major)"),
    ar2: float = typer.Option(None, help="Range a2 (mid)"),
    ar3: float = typer.Option(None, help="Range a3 (minor)"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with estimates"),
):
    df = read_gslib_table(data_path)
    if zcol is None:
        dcoords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    else:
        dcoords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values), jnp.asarray(df[zcol].values)], axis=1)
    dvals = jnp.asarray(df[vcol].values)

    pred = read_gslib_table(pred_path)
    if pzcol is None:
        pcoords = jnp.stack([jnp.asarray(pred[pxcol].values), jnp.asarray(pred[pycol].values)], axis=1)
    else:
        pcoords = jnp.stack([jnp.asarray(pred[pxcol].values), jnp.asarray(pred[pycol].values), jnp.asarray(pred[pzcol].values)], axis=1)

    angles = None if ang1 is None or ang2 is None or ang3 is None else (float(ang1), float(ang2), float(ang3))
    ranges = None if ar1 is None or ar2 is None or ar3 is None else (float(ar1), float(ar2), float(ar3))
    est, var = ok_predict(
        dcoords, dvals, pcoords, kernel, sill, range_, nugget,
        k_neighbors=k, ranges=ranges, angles_deg=angles,
        search_radius=search_radius, use_octants=(not no_octants)
    )
    pred_out = pred.copy()
    pred_out[vcol + "_ok"] = pd.Series(est)
    pred_out[vcol + "_var"] = pd.Series(var)
    write_gslib_table(pred_out, output_path, title="KT3D OK Output")


if __name__ == "__main__":
    app()


