"""Minimal GSLIB table reader/writer compatible with classic format.

Notes:
- Classic GSLIB format: first line title, second line number of columns, then column names, then rows.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def read_gslib_table(path: str) -> pd.DataFrame:
    """Read a GSLIB table file into a DataFrame.

    Args:
        path: File path.

    Returns:
        DataFrame with columns named from header.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        raise ValueError("Empty file")

    title = lines[0]
    ncol = int(lines[1].split()[0])
    names = [lines[2 + i].strip() for i in range(ncol)]
    rows = [list(map(float, ln.split())) for ln in lines[2 + ncol :]]
    df = pd.DataFrame(rows, columns=names)
    return df


def write_gslib_table(df: pd.DataFrame, path: str, title: str = "GSLIB Table") -> None:
    """Write a DataFrame to GSLIB table format.

    Args:
        df: DataFrame.
        path: Output path.
        title: Title line.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{df.shape[1]}\n")
        for name in df.columns:
            f.write(f"{name}\n")
        for _, row in df.iterrows():
            f.write(" ".join(f"{float(v):.6g}" for v in row.values) + "\n")


