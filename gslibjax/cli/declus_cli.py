from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.declus import declus_weights

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    sx: float = typer.Argument(..., help="Cell size X"),
    sy: float = typer.Argument(..., help="Cell size Y"),
    sz: float = typer.Option(1.0, help="Cell size Z"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with weights"),
):
    df = read_gslib_table(input_path)
    if zcol is None:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
        csz = (sx, sy, 1.0)
    else:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values), jnp.asarray(df[zcol].values)], axis=1)
        csz = (sx, sy, sz)
    w = declus_weights(coords, csz)
    out = df.copy()
    out["w_declus"] = pd.Series(w)
    write_gslib_table(out, output_path, title="DECLUS Weights")


if __name__ == "__main__":
    app()


