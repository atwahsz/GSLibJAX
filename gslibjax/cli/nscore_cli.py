from __future__ import annotations

import typer
import pandas as pd
import jax.numpy as jnp

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.nscore import fit_nscore, nscore_transform, nscore_inverse

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def forward(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    column: str = typer.Argument(..., help="Column to transform"),
    output_path: str = typer.Argument(..., help="Output GSLIB table"),
):
    df = read_gslib_table(input_path)
    model = fit_nscore(jnp.asarray(df[column].values))
    df[column + "_ns"] = pd.Series(
        nscore_transform(jnp.asarray(df[column].values), model)
    )
    write_gslib_table(df, output_path, title="NSCORE Output")


@app.command()
def inverse(
    input_path: str = typer.Argument(..., help="Input GSLIB table with _ns column"),
    ns_column: str = typer.Argument(..., help="Normal-score column name"),
    ref_column: str = typer.Argument(..., help="Reference original column name to fit model"),
    output_path: str = typer.Argument(..., help="Output GSLIB table"),
):
    df = read_gslib_table(input_path)
    model = fit_nscore(jnp.asarray(df[ref_column].values))
    df[ns_column + "_iv"] = pd.Series(
        nscore_inverse(jnp.asarray(df[ns_column].values), model)
    )
    write_gslib_table(df, output_path, title="NSCORE Inverse Output")


if __name__ == "__main__":
    app()


