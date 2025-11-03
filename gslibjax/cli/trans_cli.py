from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    column: str = typer.Argument(..., help="Column to transform"),
    op: str = typer.Argument(..., help="Operation: scale|shift|scale_shift|log|exp|standardize"),
    a: float = typer.Option(1.0, help="Scale factor (for scale/scale_shift)"),
    b: float = typer.Option(0.0, help="Shift (for shift/scale_shift)"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with transformed column"),
):
    df = read_gslib_table(input_path)
    x = jnp.asarray(df[column].values)
    if op == "scale":
        y = a * x
        suffix = "_scale"
    elif op == "shift":
        y = x + b
        suffix = "_shift"
    elif op == "scale_shift":
        y = a * x + b
        suffix = "_lin"
    elif op == "log":
        y = jnp.log(jnp.maximum(x, 1e-12))
        suffix = "_log"
    elif op == "exp":
        y = jnp.exp(x)
        suffix = "_exp"
    elif op == "standardize":
        mu = jnp.mean(x)
        sd = jnp.std(x) + 1e-12
        y = (x - mu) / sd
        suffix = "_std"
    else:
        raise typer.BadParameter("Unknown op. Use scale|shift|scale_shift|log|exp|standardize")

    out = df.copy()
    out[column + suffix] = pd.Series(y)
    write_gslib_table(out, output_path, title="TRANS Output")


if __name__ == "__main__":
    app()


