from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..core.anisotropy import apply_anisotropy

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    zcol: str = typer.Option(None, help="Z column (optional)"),
    ang1: float = typer.Argument(..., help="Rotation ang1 (deg) Z"),
    ang2: float = typer.Argument(..., help="Rotation ang2 (deg) Y"),
    ang3: float = typer.Argument(..., help="Rotation ang3 (deg) X"),
    ar1: float = typer.Argument(..., help="Range a1 (major)"),
    ar2: float = typer.Argument(..., help="Range a2 (mid)"),
    ar3: float = typer.Argument(..., help="Range a3 (minor)"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with rotated coords"),
):
    df = read_gslib_table(input_path)
    if zcol is None:
        coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    else:
        coords = jnp.stack([
            jnp.asarray(df[xcol].values),
            jnp.asarray(df[ycol].values),
            jnp.asarray(df[zcol].values),
        ], axis=1)
    t = apply_anisotropy(coords, (ar1, ar2, ar3), (ang1, ang2, ang3))
    out = df.copy()
    out["x_rot"] = pd.Series(t[:, 0])
    out["y_rot"] = pd.Series(t[:, 1])
    if t.shape[1] == 3:
        out["z_rot"] = pd.Series(t[:, 2])
    write_gslib_table(out, output_path, title="ROTCOORD Output")


if __name__ == "__main__":
    app()


