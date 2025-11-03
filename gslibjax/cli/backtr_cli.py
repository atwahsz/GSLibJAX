from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.backtr import load_nscore_model, back_transform

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input table with normal-score column"),
    ns_col: str = typer.Argument(..., help="Normal-score column name"),
    model_path: str = typer.Argument(..., help="Saved NSCORE model (JSON)"),
    output_path: str = typer.Argument(..., help="Output table"),
):
    df = read_gslib_table(input_path)
    model = load_nscore_model(model_path)
    back = back_transform(jnp.asarray(df[ns_col].values), model)
    df[ns_col + "_back"] = pd.Series(back)
    write_gslib_table(df, output_path, title="BACKTR Output")


if __name__ == "__main__":
    app()


