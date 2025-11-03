from __future__ import annotations

import typer

from ..io.gslib_table import read_gslib_table
from ..tools.plotting import plot_probplot

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table"),
    column: str = typer.Argument(..., help="Column to plot"),
    output_path: str = typer.Argument(..., help="Output image path (png)"),
):
    df = read_gslib_table(input_path)
    plot_probplot(df, column, output_path, title=f"Probability Plot: {column}")


if __name__ == "__main__":
    app()


