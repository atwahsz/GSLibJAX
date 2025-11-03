from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.gamv import experimental_variogram, directional_variogram, directional_variogram_3d

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def omni(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    vcol: str = typer.Argument(..., help="Value column"),
    lag: float = typer.Argument(..., help="Lag width"),
    nlag: int = typer.Argument(..., help="Number of lags"),
    ang1: float = typer.Option(None, help="Rotation ang1 (deg) Z"),
    ang2: float = typer.Option(None, help="Rotation ang2 (deg) Y"),
    ang3: float = typer.Option(None, help="Rotation ang3 (deg) X"),
    ar1: float = typer.Option(None, help="Range a1 (major)"),
    ar2: float = typer.Option(None, help="Range a2 (mid)"),
    ar3: float = typer.Option(None, help="Range a3 (minor)"),
    output_path: str = typer.Argument(..., help="Output GSLIB table for variogram"),
):
    df = read_gslib_table(input_path)
    if zcol is None:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    else:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values), jnp.asarray(df[zcol].values)], axis=1)
    vals = jnp.asarray(df[vcol].values)
    angles = None if ang1 is None or ang2 is None or ang3 is None else (float(ang1), float(ang2), float(ang3))
    ranges = None if ar1 is None or ar2 is None or ar3 is None else (float(ar1), float(ar2), float(ar3))
    centers, gamma, npairs = experimental_variogram(vals, coords, lag=lag, nlag=nlag, ranges=ranges, angles_deg=angles)
    out = pd.DataFrame({
        "h": pd.Series(centers),
        "gamma": pd.Series(gamma),
        "npairs": pd.Series(npairs),
    })
    write_gslib_table(out, output_path, title="GAMV Omnidirectional")


@app.command()
def directional(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    vcol: str = typer.Argument(..., help="Value column"),
    lag: float = typer.Argument(..., help="Lag width"),
    nlag: int = typer.Argument(..., help="Number of lags"),
    azimuth: float = typer.Argument(..., help="Azimuth in degrees"),
    atol: float = typer.Argument(..., help="Angular tolerance in degrees"),
    bandwidth: float = typer.Argument(..., help="Perpendicular bandwidth"),
    output_path: str = typer.Argument(..., help="Output GSLIB table for variogram"),
):
    df = read_gslib_table(input_path)
    coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    vals = jnp.asarray(df[vcol].values)
    centers, gamma, npairs = directional_variogram(
        vals, coords, lag=lag, nlag=nlag, azimuth_deg=azimuth, atol_deg=atol, bandwidth=bandwidth
    )
    out = pd.DataFrame({
        "h": pd.Series(centers),
        "gamma": pd.Series(gamma),
        "npairs": pd.Series(npairs),
    })
    write_gslib_table(out, output_path, title="GAMV Directional")


@app.command()
def directional3d(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Argument(..., help="Z column"),
    vcol: str = typer.Argument(..., help="Value column"),
    lag: float = typer.Argument(..., help="Lag width"),
    nlag: int = typer.Argument(..., help="Number of lags"),
    azimuth: float = typer.Argument(..., help="Azimuth in degrees"),
    dip: float = typer.Argument(..., help="Dip in degrees"),
    atol: float = typer.Argument(..., help="Angular tolerance in degrees"),
    band_h: float = typer.Argument(..., help="Horizontal bandwidth"),
    band_v: float = typer.Argument(..., help="Vertical bandwidth"),
    output_path: str = typer.Argument(..., help="Output GSLIB table for variogram"),
):
    df = read_gslib_table(input_path)
    coords = jnp.stack([
        jnp.asarray(df[xcol].values),
        jnp.asarray(df[ycol].values),
        jnp.asarray(df[zcol].values),
    ], axis=1)
    vals = jnp.asarray(df[vcol].values)
    centers, gamma, npairs = directional_variogram_3d(
        vals, coords, lag=lag, nlag=nlag, azimuth_deg=azimuth, dip_deg=dip, atol_deg=atol, bandwh_h=band_h, bandwh_v=band_v
    )
    out = pd.DataFrame({
        "h": pd.Series(centers),
        "gamma": pd.Series(gamma),
        "npairs": pd.Series(npairs),
    })
    write_gslib_table(out, output_path, title="GAMV Directional 3D")


if __name__ == "__main__":
    app()



