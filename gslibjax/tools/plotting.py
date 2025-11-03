from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(df: pd.DataFrame, column: str, bins: int, output_path: str, title: str = "Histogram") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[column].values, bins=bins, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, output_path: str, title: str = "Scatter") -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df[xcol].values, df[ycol].values, s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_variogram(df: pd.DataFrame, hcol: str, gcol: str, output_path: str, title: str = "Variogram") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[hcol].values, df[gcol].values, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel('h')
    ax.set_ylabel('gamma')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_probplot(df: pd.DataFrame, column: str, output_path: str, title: str = "Probability Plot") -> None:
    import numpy as np
    import scipy.stats as stats

    x = df[column].values
    x = x[~pd.isna(x)]
    osm, osr = stats.probplot(x, dist="norm", fit=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(osm, osr, s=10)
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


