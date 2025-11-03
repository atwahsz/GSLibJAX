from __future__ import annotations

import typer
import jax.numpy as jnp
import pandas as pd

from ..io.gslib_table import read_gslib_table, write_gslib_table
from ..tools.ik3d import ik3d

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    data_path: str = typer.Argument(..., help="Input data GSLIB table"),
    xcol: str = typer.Argument(..., help="X column"),
    ycol: str = typer.Argument(..., help="Y column"),
    vcol: str = typer.Argument(..., help="Value column"),
    pred_path: str = typer.Argument(..., help="Prediction points GSLIB table"),
    pxcol: str = typer.Argument(..., help="Pred X column"),
    pycol: str = typer.Argument(..., help="Pred Y column"),
    thresholds: str = typer.Argument(..., help="Comma-separated thresholds (e.g., 0.1,0.3,0.5)"),
    kernel: str = typer.Argument(..., help="Kernel spherical|exponential|gaussian"),
    range_: float = typer.Argument(..., help="Range"),
    nugget: float = typer.Option(0.0, help="Nugget"),
    kneigh: int = typer.Option(12, help="k-nearest neighbors"),
    ang1: float = typer.Option(None, help="Rotation ang1 (deg) Z"),
    ang2: float = typer.Option(None, help="Rotation ang2 (deg) Y"),
    ang3: float = typer.Option(None, help="Rotation ang3 (deg) X"),
    ar1: float = typer.Option(None, help="Range a1 (major)"),
    ar2: float = typer.Option(None, help="Range a2 (mid)"),
    ar3: float = typer.Option(None, help="Range a3 (minor)"),
    output_path: str = typer.Argument(..., help="Output GSLIB table with probabilities"),
):
    d = read_gslib_table(data_path)
    dc = jnp.stack([jnp.asarray(d[xcol].values), jnp.asarray(d[ycol].values)], axis=1)
    dv = jnp.asarray(d[vcol].values)

    p = read_gslib_table(pred_path)
    pc = jnp.stack([jnp.asarray(p[pxcol].values), jnp.asarray(p[pycol].values)], axis=1)

    th = jnp.array([float(z) for z in thresholds.split(',')], dtype=jnp.float32)
    angles = None if ang1 is None or ang2 is None or ang3 is None else (float(ang1), float(ang2), float(ang3))
    ranges = None if ar1 is None or ar2 is None or ar3 is None else (float(ar1), float(ar2), float(ar3))
    probs = ik3d(dc, dv, pc, th, kernel, range_, nugget, kneigh, ranges, angles)
    out = p.copy()
    for i in range(probs.shape[1]):
        out[f"p_le_{float(th[i]):.6g}"] = pd.Series(probs[:, i])
    write_gslib_table(out, output_path, title="IK3D Probabilities")


if __name__ == "__main__":
    app()


