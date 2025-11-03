from __future__ import annotations

import typer

from ..io.gslib_table import read_gslib_table
from ..tools.plotting import plot_variogram

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input variogram GSLIB table"),
    hcol: str = typer.Option("h", help="Lag column name"),
    gcol: str = typer.Option("gamma", help="Semivariance column name"),
    output_path: str = typer.Argument(..., help="Output image path (png)"),
):
    df = read_gslib_table(input_path)
    plot_variogram(df, hcol, gcol, output_path, title="Variogram")


if __name__ == "__main__":
    app()


