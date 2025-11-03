from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.sasim import sasim

app = typer.Typer(add_completion=False, no_args_is_help=True)


def parse_structures(spec: str):
    parts = []
    for s in spec.split(';'):
        name, sill, range_ = s.split(':')
        parts.append((name, float(sill), float(range_)))
    return tuple(parts)


@app.command()
def run(
    input_path: str = typer.Argument(..., help="Input GSLIB table with initial values"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    vcol: str = typer.Argument(..., help="Value column"),
    lag: float = typer.Argument(..., help="Lag width"),
    nlag: int = typer.Argument(..., help="Number of lags"),
    nugget: float = typer.Argument(..., help="Nugget"),
    structures: str = typer.Argument(..., help="name:sill:range;name:sill:range"),
    steps: int = typer.Option(10000, help="Annealing steps"),
    temp0: float = typer.Option(1.0, help="Initial temperature"),
    temp_decay: float = typer.Option(0.999, help="Temperature decay per step"),
    noise_sigma: float = typer.Option(0.1, help="Proposal stddev"),
    seed: int = typer.Option(12345, help="Random seed"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with optimized values"),
):
    df = read_gslib_table(input_path)
    coords = jnp.stack([jnp.asarray(df[xcol].values), jnp.asarray(df[ycol].values)], axis=1)
    vals0 = jnp.asarray(df[vcol].values)
    strucs = parse_structures(structures)
    vals = sasim(vals0, coords, lag, nlag, nugget, strucs, steps, temp0, temp_decay, noise_sigma, seed)
    out = df.copy()
    out[vcol + "_sasim"] = pd.Series(vals)
    write_gslib_table(out, output_path, title="SASIM Output")


if __name__ == "__main__":
    app()


