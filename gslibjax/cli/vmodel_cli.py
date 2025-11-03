from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..tools.vmodel import model_gamma
from ..io.gslib_table import write_gslib_table

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def eval(
    lag: float = typer.Argument(..., help="Lag width"),
    nlag: int = typer.Argument(..., help="Number of lags"),
    nugget: float = typer.Argument(..., help="Nugget"),
    structures: str = typer.Argument(..., help="Comma-separated structures name:sill:range;..."),
    output_path: str = typer.Argument(..., help="Output GSLIB table"),
):
    h = jnp.arange(1, nlag + 1, dtype=jnp.float32) * lag
    parts = []
    for s in structures.split(";"):
        name, sill, range_ = s.split(":")
        parts.append((name, float(sill), float(range_)))
    g = model_gamma(h, nugget, parts)
    df = pd.DataFrame({"h": pd.Series(h), "gamma": pd.Series(g)})
    write_gslib_table(df, output_path, title="VMODEL")


if __name__ == "__main__":
    app()


