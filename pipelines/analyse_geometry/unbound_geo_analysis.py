#!/usr/bin/env python
# run_geo_analysis_ab_chain.py
#
# Runs:
#  1) compute_TCR_angles() on a single directory
#  2) violin distribution grid (all ANGLE_COLUMNS in one figure)
#  3) per-angle scatter colored by germline_4tuple
#  4) PCA scatter colored by germline_4tuple
#
# Output goes to: <INPUT_DIR>/geo_analysis_outputs/

import sys
from pathlib import Path
import pandas as pd
import matplotlib
import re
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# If geo_analysis_tools.py is not on your PYTHONPATH, set this to its folder:
# Example:
# sys.path.append("/workspaces/Graphormer/TCR_Metrics/scripts")
GEO_TOOLS_DIR = None  # set to Path("...") if needed

if GEO_TOOLS_DIR is not None:
    sys.path.append(str(GEO_TOOLS_DIR))

from geo_analysis_tools import (
    compute_TCR_angles,
    plot_angle_distributions_grid,
    plot_angle_scatter_by_germline4tuple,
    pca_and_plot_by_germline4tuple,
)

# -------------------------
# CONFIG
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(s: str, max_len: int = 160) -> str:
    """
    Turn a long germline label into a filesystem-friendly string.
    """
    s = str(s)
    s = s.replace("*", "STAR").replace("|", "__").replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def pca_and_plot_one_per_germline4tuple(
    df: pd.DataFrame,
    ANGLE_COLUMNS: list,
    out_dir: Path,
    n_components: int = 2,
    min_group_size: int = 3,
    highlight_alpha: float = 0.95,
    background_alpha: float = 0.12,
    point_size: int = 18,
):
    """
    Fit PCA once on all rows (single-state df: all bound OR all unbound).
    Then write ONE PCA plot PER germline_4tuple, highlighting that group and
    showing all other points as a faint background.

    Requirements:
      - df contains ANGLE_COLUMNS
      - df contains 'germline_4tuple'
      - optional: 'tcr_name' (only for saving coords)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "germline_4tuple" not in df.columns:
        raise ValueError("df must contain 'germline_4tuple' column.")

    X = df.copy()

    # Ensure required columns exist
    for c in ANGLE_COLUMNS:
        if c not in X.columns:
            X[c] = np.nan

    # Keep only complete cases for PCA (no NaNs in features)
    feat = X[ANGLE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    mask = ~feat.isna().any(axis=1)
    X = X.loc[mask].copy()
    feat = feat.loc[mask].copy()

    if X.shape[0] < 5:
        raise ValueError(f"Not enough complete rows for PCA after dropping NaNs (n={X.shape[0]}).")

    X["germline_4tuple"] = X["germline_4tuple"].fillna("unknown").astype(str)

    # Standardize + PCA
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat.values.astype(float))

    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(feat_scaled)

    coords_df = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"], index=X.index)
    coords_df["germline_4tuple"] = X["germline_4tuple"].values
    if "tcr_name" in X.columns:
        coords_df["tcr_name"] = X["tcr_name"].astype(str).values

    # Save coordinates once
    coords_df.to_csv(out_dir / "pca_coords.csv", index=False)

    # Shared axes
    xmin, xmax = coords_df["PC1"].min(), coords_df["PC1"].max()
    ymin, ymax = coords_df["PC2"].min(), coords_df["PC2"].max()
    xpad = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
    ypad = 0.06 * (ymax - ymin) if ymax > ymin else 1.0
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    # Iterate groups
    group_sizes = coords_df["germline_4tuple"].value_counts()
    kept_groups = group_sizes[group_sizes >= min_group_size].index.tolist()

    # Optional: also save the list of groups plotted
    pd.DataFrame({"germline_4tuple": kept_groups, "n": group_sizes.loc[kept_groups].values}).to_csv(
        out_dir / "germline_groups_plotted.csv", index=False
    )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0

    for g in kept_groups:
        sub = coords_df[coords_df["germline_4tuple"] == g]
        if sub.empty:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))

        # Background (all points)
        ax.scatter(
            coords_df["PC1"].values,
            coords_df["PC2"].values,
            s=point_size,
            alpha=background_alpha,
            edgecolors="none",
            label="all other",
        )

        # Highlight current group
        ax.scatter(
            sub["PC1"].values,
            sub["PC2"].values,
            s=point_size + 6,
            alpha=highlight_alpha,
            edgecolors="none",
            label=f"{g} (n={len(sub)})",
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.set_title("PCA (one germline_4tuple highlighted)", fontsize=12)

        # Put the highlighted label under the plot (long labels)
        fig.text(0.5, 0.01, f"Highlighted: {g} (n={len(sub)})", ha="center", va="bottom", fontsize=8)

        ax.legend(loc="best", fontsize=8, frameon=True)
        fig.tight_layout(rect=[0, 0.03, 1, 1])

        out_path = out_dir / f"pca_highlight__{safe_slug(g)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_scatter_per_germline4tuple(df: pd.DataFrame, ANGLE_COLUMNS: list, out_dir: Path):
    """
    For each angle, make one plot PER germline_4tuple.

    Uses a single-state df (only bound OR only unbound).
    x-axis is sample index (rank) for that germline group.
    y-axis is the angle value.
    """
    ensure_dir(out_dir)

    if "germline_4tuple" not in df.columns:
        raise ValueError("df must contain 'germline_4tuple' (from normalize_germlines / compute_TCR_angles).")
    if "tcr_name" not in df.columns:
        df = df.copy()
        df["tcr_name"] = np.arange(len(df)).astype(str)

    df = df.copy()
    df["germline_4tuple"] = df["germline_4tuple"].fillna("unknown").astype(str)

    # One folder per angle for cleanliness
    for angle in ANGLE_COLUMNS:
        if angle not in df.columns:
            print(f"[plot_scatter_per_germline4tuple] Missing angle {angle}, skipping.")
            continue

        angle_dir = out_dir / f"{angle}"
        ensure_dir(angle_dir)

        # group by germline_4tuple
        for g, sub in df[["tcr_name", "germline_4tuple", angle]].dropna(subset=[angle]).groupby("germline_4tuple"):
            sub = sub.copy()
            if sub.empty:
                continue

            # Sort by angle for a stable visual (optional)
            sub = sub.sort_values(angle)

            x = np.arange(sub.shape[0])
            y = sub[angle].astype(float).values

            plt.figure(figsize=(8, 5))
            plt.scatter(x, y, s=22, alpha=0.85, edgecolors="none")
            plt.title(f"{angle} scatter for germline_4tuple\n{g}\n(n={sub.shape[0]})", fontsize=10)
            plt.xlabel("Index within germline group (sorted by value)")
            plt.ylabel(f"{angle} ({'deg' if angle != 'dc' else 'Å'})")
            plt.tight_layout()

            fname = f"{angle}__germline4tuple__{safe_slug(g)}.png"
            out_path = angle_dir / fname
            plt.savefig(out_path, dpi=300)
            plt.close()
        print(f"[OK] Wrote per-germline scatter plots for angle={angle} to {angle_dir}")


def main():
    ensure_dir(OUT_DIR)

    # 1) Compute geometry + germlines
    df, failed = compute_TCR_angles(INPUT_DIR)

    # Save raw table and failures
    df.to_csv(OUT_DIR / "geometry_with_germlines.csv", index=False)
    (OUT_DIR / "failed_tcrs.txt").write_text("\n".join(map(str, failed)) + ("\n" if failed else ""))
    print(f"[OK] Wrote geometry CSV: {OUT_DIR / 'geometry_with_germlines.csv'}")
    print(f"[OK] Wrote failed list:  {OUT_DIR / 'failed_tcrs.txt'} (n={len(failed)})")

    # 2) Violin distributions grid
    plot_angle_distributions_grid(
        df=df,
        out_path=OUT_DIR / "geometry_distributions_grid.png",
        ANGLE_COLUMNS=ANGLE_COLUMNS,
        ncols=3,
    )

    # 3) Scatter per germline_4tuple (one plot per germline per angle)
    plot_scatter_per_germline4tuple(
        df=df,
        ANGLE_COLUMNS=ANGLE_COLUMNS,
        out_dir=OUT_DIR / "scatter_per_germline4tuple",
    )

    # 4) PCA scatter colored by combined germlines
    pca_and_plot_by_germline4tuple(
        df=df,
        ANGLE_COLUMNS=ANGLE_COLUMNS,
        out_path=OUT_DIR / "pca_by_germline4tuple.png",
        max_legend_items=22,
        n_components=2,
    )
    pca_and_plot_one_per_germline4tuple(
    df=df,
    ANGLE_COLUMNS=ANGLE_COLUMNS,
    out_dir=OUT_DIR / "pca_one_plot_per_germline4tuple",
    min_group_size=3,   # adjust (e.g., 1 if you truly want all)
)

    print("\nDone. Outputs in:")
    print(f"  {OUT_DIR}")



if __name__ == "__main__":
    INPUT_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/ab_chain")
    OUT_DIR = Path("/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/unbound_geo_TCR3d_plots")

    ANGLE_COLUMNS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]

    main()

