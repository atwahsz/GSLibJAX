from __future__ import annotations

import typer

from ..io.gslib_table import read_gslib_table
from ..tools.plotting import plot_scatter

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    output_path: str = typer.Argument(..., help="Output image path (png)"),
):
    df = read_gslib_table(input_path)
    plot_scatter(df, xcol, ycol, output_path, title=f"Scatter: {xcol} vs {ycol}")


if __name__ == "__main__":
    app()


