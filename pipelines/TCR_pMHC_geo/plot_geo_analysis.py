#!/usr/bin/env python3
"""
Plot per-variable distributions and violin plots from a geometry CSV.

- Produces:
  1) <stem>_distributions.png  : histogram + KDE (if possible) for each numeric variable
  2) <stem>_violins.png        : violin plot for each numeric variable

Notes:
- Uses matplotlib only (no seaborn).
- Does NOT call plt.show(); saves to PNG.
"""

from __future__ import annotations
import os
import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde  # optional, for KDE overlay
except Exception:
    gaussian_kde = None


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Coerce all non-"structure" columns to numeric where possible
    for col in df.columns:
        if col == "structure":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _grid(n: int) -> tuple[int, int]:
    """Reasonable rows/cols for n subplots."""
    if n <= 0:
        return (1, 1)
    cols = 4 if n >= 8 else 3 if n >= 5 else 2 if n >= 2 else 1
    rows = math.ceil(n / cols)
    return rows, cols


def plot_distributions(
    df: pd.DataFrame,
    cols: List[str],
    out_path: Path,
    bins: int = 30,
) -> None:
    rows, ncols = _grid(len(cols))

    # Scale figure size with grid
    fig_w = max(10, 4 * ncols)
    fig_h = max(6, 3 * rows)
    fig, axes = plt.subplots(rows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        x = df[col].dropna().astype(float).values
        if x.size == 0:
            ax.set_title(col)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Histogram (density)
        ax.hist(x, bins=bins, density=True, alpha=0.6)

        # KDE overlay if possible and enough points
        if gaussian_kde is not None and x.size >= 5:
            try:
                kde = gaussian_kde(x)
                xs = np.linspace(np.min(x), np.max(x), 200)
                ax.plot(xs, kde(xs), linewidth=1.5)
            except Exception:
                # If KDE fails (e.g., singular covariance), skip silently
                pass

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")

    # Turn off any unused axes
    for j in range(len(cols), rows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violins(
    df: pd.DataFrame,
    cols: List[str],
    out_path: Path,
) -> None:
    rows, ncols = _grid(len(cols))

    fig_w = max(10, 4 * ncols)
    fig_h = max(6, 3 * rows)
    fig, axes = plt.subplots(rows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]

        x = df[col].dropna().astype(float).values
        if x.size == 0:
            ax.set_title(col)
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        parts = ax.violinplot(
            [x],
            positions=[1],
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )

        # Make it look clean without specifying colors
        ax.set_title(col)
        ax.set_xticks([1])
        ax.set_xticklabels([col], rotation=45, ha="right")
        ax.set_ylabel("Value")

        # Improve y-limits a bit
        ymin, ymax = np.nanmin(x), np.nanmax(x)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    for j in range(len(cols), rows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(csv_path, outdir, bins_num) -> None:

    df = load_csv(csv_path)

    # Exclude ID-like column(s)
    cols = numeric_columns(df, exclude=["structure"])
    if not cols:
        raise SystemExit("No numeric columns found to plot (after excluding 'structure').")

    outdir = outdir if outdir is not None else csv_path.parent
    stem = csv_path.stem

    dist_path = os.path.join(outdir ,f"{stem}_distributions.png")
    viol_path = os.path.join(outdir, f"{stem}_violins.png")

    plot_distributions(df, cols, Path(dist_path), bins=bins_num)
    plot_violins(df, cols, Path(viol_path))

    print(f"Wrote: {dist_path}")
    print(f"Wrote: {viol_path}")


if __name__ == "__main__":
    main(Path("/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_TCR3d_calc_geometry/geometry_results.csv"), outdir="/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_TCR3d_calc_geometry/plots", bins_num=30)
