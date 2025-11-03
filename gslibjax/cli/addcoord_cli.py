from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    xref: float = typer.Option(0.0, help="Reference X"),
    yref: float = typer.Option(0.0, help="Reference Y"),
    zref: float = typer.Option(0.0, help="Reference Z"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with derived coords"),
):
    df = read_gslib_table(input_path)
    x = jnp.asarray(df[xcol].values)
    y = jnp.asarray(df[ycol].values)
    dx = x - xref
    dy = y - yref
    out = df.copy()
    if zcol is None:
        r = jnp.sqrt(dx * dx + dy * dy)
        theta = jnp.degrees(jnp.arctan2(dy, dx))
        out["r"] = pd.Series(r)
        out["theta_deg"] = pd.Series(theta)
    else:
        z = jnp.asarray(df[zcol].values)
        dz = z - zref
        r3 = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        out["r3"] = pd.Series(r3)
    write_gslib_table(out, output_path, title="ADDCOORD Output")


if __name__ == "__main__":
    app()


