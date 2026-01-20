#!/usr/bin/env python3
"""
Geometry determinism by (MHCα, germline_vj_pair)

Outputs:
  - Heatmap per feature: variability per (mhc_group, germline_vj_pair) cell, annotated with n
  - PCA scatter: each cell is a colored group (optionally collapsed to top-N groups)
  - Variance decomposition table per feature (eta^2 etc.)

Example:
  python geometry_group_determinism.py \
    --csv /path/to/human_TCR3d_unpaired_bound.csv \
    --outdir /path/to/out \
    --state bound \
    --var-metric mad \
    --min-cell-n 2 \
    --top-mhc 0 \
    --top-germline 40 \
    --top-groups 60
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Features
# -----------------------------
FEATURES: List[str] = [
    "BA","BC1","AC1","BC2","AC2","dc",
    "d_alpha_mhc","d_beta_mhc","d_alpha_beta",
    "theta_rA","phi_rA","theta_rB","phi_rB",
    "theta_pc1A","phi_pc1A","theta_pc1B","phi_pc1B",
    "alpha_cdr3_bend_deg","alpha_cdr3_apex_height_A","alpha_cdr3_apex_resi",
    "beta_cdr3_bend_deg","beta_cdr3_apex_height_A","beta_cdr3_apex_resi",
]

# If you later want circular handling, these are the candidates.
ANGLE_FEATURES = {
    "BA","BC1","AC1","BC2","AC2",
    "theta_rA","phi_rA","theta_rB","phi_rB",
    "theta_pc1A","phi_pc1A","theta_pc1B","phi_pc1B",
    "alpha_cdr3_bend_deg","beta_cdr3_bend_deg",
}


# -----------------------------
# Normalization utilities
# -----------------------------
def normalize_mhc_alpha_allele_to_2field(x: object) -> Optional[str]:
    """
    Collapse mhc_alpha_allele to GENE*XX:
      A*02:1150N -> A*02
      B*44:05    -> B*44
      DQA1*01:133Q -> DQA1*01

    Also strips common 'HLA' prefixes if present.
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # remove common prefixes
    s = s.replace("HLA-", "").replace("HLA:", "").replace("HLA", "").strip()
    s = re.sub(r"^[\s:;-]+", "", s).strip()

    # get token containing '*'
    tok = None
    m = re.search(r"\b([A-Za-z0-9]{1,10}\*\d{2}(?::\d{2,4}[A-Za-z0-9]*)?)\b", s)
    if m:
        tok = m.group(1)
    else:
        tok = next((t.strip(",;") for t in s.split() if "*" in t), None)
    if tok is None:
        return None

    # reduce to GENE*XX
    #tok = tok.split(":", 1)[0]
    return tok


# -----------------------------
# Variability metrics
# -----------------------------
def mad(x: np.ndarray) -> float:
    """Median absolute deviation (unscaled), robust variability."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def iqr(x: np.ndarray) -> float:
    """Interquartile range, robust variability."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)

def std(x: np.ndarray) -> float:
    """Sample standard deviation (ddof=1)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1))

VAR_METRICS = {"mad": mad, "iqr": iqr, "std": std}


# -----------------------------
# Plotting helpers
# -----------------------------
def _pick_text_color(val: float, vmin: float, vmax: float) -> str:
    """
    Choose white/black based on normalized intensity (simple heuristic).
    """
    if np.isnan(val) or vmax <= vmin:
        return "black"
    t = (val - vmin) / (vmax - vmin + 1e-12)
    return "white" if t > 0.55 else "black"


def plot_heatmap_with_n(
    mat: np.ndarray,
    nmat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    outpath: Path,
    fmt_val: str = "{:.2f}",
) -> None:
    """
    Heatmap where each cell shows:
      variability_value
      (n=...)
    """
    # Robust vmin/vmax for display
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig_w = min(max(10, 0.28 * len(col_labels) + 4), 28)
    fig_h = min(max(7,  0.28 * len(row_labels) + 3), 22)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=8)

    ax.set_title(title, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Variability", rotation=90)

    # annotate each cell
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            nij = int(nmat[i, j]) if np.isfinite(nmat[i, j]) else 0
            if nij <= 0 or np.isnan(mat[i, j]):
                continue
            txt = f"{fmt_val.format(mat[i,j])}\n(n={nij})"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=6,
                color=_pick_text_color(mat[i, j], vmin, vmax),
            )

    ax.set_xlabel("germline_vj_pair")
    ax.set_ylabel("MHCα group (GENE*XX)")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -----------------------------
# Variance decomposition (ANOVA-style)
# -----------------------------
def variance_decomposition_eta2(
    y: pd.Series,
    groups: pd.Series,
) -> Dict[str, float]:
    """
    ANOVA-style decomposition for y by groups:
      SS_total = sum (y - ybar)^2
      SS_between = sum n_g (ybar_g - ybar)^2
      SS_within = sum_g sum_i (y_i - ybar_g)^2
      eta2 = SS_between / SS_total

    Returns dict with counts and SS.
    """
    df = pd.DataFrame({"y": y, "g": groups}).dropna()
    if df.empty:
        return {
            "N": 0, "n_groups": 0,
            "SS_total": np.nan, "SS_between": np.nan, "SS_within": np.nan,
            "eta2": np.nan
        }

    yv = df["y"].astype(float).to_numpy()
    g = df["g"].astype(str)

    ybar = float(np.mean(yv))
    ss_total = float(np.sum((yv - ybar) ** 2))

    grouped = df.groupby("g")["y"]
    means = grouped.mean()
    counts = grouped.count()

    ss_between = float(np.sum(counts.to_numpy() * (means.to_numpy() - ybar) ** 2))

    # SS_within = SS_total - SS_between (numerically stable enough here)
    ss_within = float(ss_total - ss_between)

    eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan

    return {
        "N": int(df.shape[0]),
        "n_groups": int(counts.shape[0]),
        "SS_total": ss_total,
        "SS_between": ss_between,
        "SS_within": ss_within,
        "eta2": eta2,
    }


# -----------------------------
# PCA scatter (NumPy SVD)
# -----------------------------
def pca_2d_numpy(X: np.ndarray) -> np.ndarray:
    """
    PCA to 2D using SVD on standardized X.
    Returns (n_samples, 2).
    """
    # center
    Xc = X - np.mean(X, axis=0, keepdims=True)
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # PC scores = U * S
    Z = U[:, :2] * S[:2]
    return Z


def plot_pca_scatter_by_group(
    df: pd.DataFrame,
    features_for_pca: List[str],
    group_col: str,
    outpath: Path,
    top_groups: int = 60,
) -> None:
    """
    PCA scatter of geometry vectors, colored by group (mhc,germline cell).
    Optionally collapse to top-N groups by count.
    """
    needed = features_for_pca + [group_col]
    d = df[needed].copy()

    # drop rows with missing PCA features
    d = d.dropna(subset=features_for_pca + [group_col])
    if d.empty:
        return

    # collapse to top groups
    if top_groups > 0:
        counts = d[group_col].value_counts()
        keep = counts.index[:top_groups]
        d[group_col] = d[group_col].where(d[group_col].isin(keep), "Other")

    # build X and standardize
    X = d[features_for_pca].to_numpy(dtype=float)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    Z = pca_2d_numpy(Xz)
    d["PC1"] = Z[:, 0]
    d["PC2"] = Z[:, 1]

    groups = d[group_col].astype(str).unique().tolist()
    groups_sorted = sorted(groups, key=lambda g: (g == "Other", g))  # put Other last-ish
    n_groups = len(groups_sorted)

    cmap = plt.cm.get_cmap("tab20", max(n_groups, 1))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, g in enumerate(groups_sorted):
        sub = d[d[group_col] == g]
        ax.scatter(sub["PC1"], sub["PC2"], s=14, alpha=0.75, label=g, color=cmap(i))

    ax.set_title(f"PCA of geometry features (colored by {group_col})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Legend: only if not insane
    if n_groups <= 20:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(
            0.99, 0.01,
            f"{n_groups} groups (legend suppressed; see group_color_key.csv)",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9
        )

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    # Save group -> color index mapping
    key = pd.DataFrame({
        "group": groups_sorted,
        "color_index_tab20": list(range(len(groups_sorted)))
    })
    key.to_csv(outpath.parent / "group_color_key.csv", index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV (unpaired_bound.csv)")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--state", type=str, default="bound", help="Filter by state column (or set empty to skip)")
    ap.add_argument("--var-metric", type=str, default="mad", choices=sorted(VAR_METRICS.keys()))
    ap.add_argument("--min-cell-n", type=int, default=2, help="Minimum n per cell to display; others become NaN")
    ap.add_argument("--top-mhc", type=int, default=0, help="Keep only top N MHC groups by count (0 = keep all)")
    ap.add_argument("--top-germline", type=int, default=40, help="Keep only top N germline_vj_pair by count (0 = keep all)")
    ap.add_argument("--top-groups", type=int, default=60, help="Top N joint groups for PCA coloring (0 = keep all; may be unreadable)")
    ap.add_argument("--pca-features", type=str, default="",
                    help="Comma-separated subset of FEATURES for PCA (default: a practical core set)")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # optional state filter
    if args.state and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower() == args.state.lower()].copy()

    # required columns
    for col in ["mhc_alpha_allele", "germline_vj_pair"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # normalize MHC allele -> group
    df["mhc_alpha_group"] = df["mhc_alpha_allele"].map(normalize_mhc_alpha_allele_to_2field)
    df = df[df["mhc_alpha_group"].notna()].copy()
    df = df[df["germline_vj_pair"].notna()].copy()

    # ensure numeric
    for feat in FEATURES:
        if feat not in df.columns:
            raise ValueError(f"CSV missing feature column: {feat}")
        df[feat] = pd.to_numeric(df[feat], errors="coerce")

    # optional top filters for readability
    if args.top_mhc and args.top_mhc > 0:
        keep_m = df["mhc_alpha_group"].value_counts().index[:args.top_mhc]
        df = df[df["mhc_alpha_group"].isin(keep_m)].copy()

    if args.top_germline and args.top_germline > 0:
        keep_g = df["germline_vj_pair"].value_counts().index[:args.top_germline]
        df = df[df["germline_vj_pair"].isin(keep_g)].copy()

    # category ordering
    mhc_order = df["mhc_alpha_group"].value_counts().index.tolist()
    germ_order = df["germline_vj_pair"].value_counts().index.tolist()

    # joint group label for variance + PCA
    df["joint_group"] = df["mhc_alpha_group"].astype(str) + " | " + df["germline_vj_pair"].astype(str)

    # -------------------------
    # Heatmaps: variability per feature
    # -------------------------
    metric_fn = VAR_METRICS[args.var_metric]

    heat_dir = outdir / "heatmaps_variability"
    heat_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Rows (MHC groups): {len(mhc_order)} | Cols (germline_vj_pair): {len(germ_order)}")
    print(f"[INFO] Variability metric: {args.var_metric.upper()} | min_cell_n={args.min_cell_n}")

    for feat in FEATURES:
        g = df.groupby(["mhc_alpha_group", "germline_vj_pair"])[feat]

        # counts
        nmat = g.count().unstack().reindex(index=mhc_order, columns=germ_order)
        # variability
        vmat = g.apply(lambda s: metric_fn(s.to_numpy(dtype=float))).unstack().reindex(index=mhc_order, columns=germ_order)

        nmat = nmat.fillna(0).to_numpy(dtype=float)
        v = vmat.to_numpy(dtype=float)

        # mask low-n cells
        v[nmat < args.min_cell_n] = np.nan

        title = f"{feat} variability ({args.var_metric.upper()}) by (MHCα, germline_vj_pair)"
        outpath = heat_dir / f"heatmap_{feat}_{args.var_metric}.png"

        plot_heatmap_with_n(
            mat=v,
            nmat=nmat,
            row_labels=mhc_order,
            col_labels=germ_order,
            title=title,
            outpath=outpath,
            fmt_val="{:.2f}",
        )

    # -------------------------
    # Scatter plot: PCA of geometry vectors colored by joint group
    # -------------------------
    if args.pca_features.strip():
        pca_feats = [x.strip() for x in args.pca_features.split(",") if x.strip()]
    else:
        # pragmatic default: core geometry + distances (avoid apex_resi which is quasi-discrete)
        pca_feats = [
            "BA","BC1","AC1","BC2","AC2","dc",
            "d_alpha_mhc","d_beta_mhc","d_alpha_beta",
            "theta_rA","phi_rA","theta_rB","phi_rB",
            "theta_pc1A","phi_pc1A","theta_pc1B","phi_pc1B",
        ]

    for f in pca_feats:
        if f not in df.columns:
            raise ValueError(f"PCA feature not found: {f}")

    plot_pca_scatter_by_group(
        df=df,
        features_for_pca=pca_feats,
        group_col="joint_group",
        outpath=outdir / "pca_scatter_joint_groups.png",
        top_groups=args.top_groups,
    )

    # -------------------------
    # Statistical calculations: variance decomposition per feature
    # -------------------------
    rows = []
    for feat in FEATURES:
        stats = variance_decomposition_eta2(df[feat], df["joint_group"])
        stats["feature"] = feat
        rows.append(stats)

    stat_df = pd.DataFrame(rows)[
        ["feature","N","n_groups","SS_total","SS_between","SS_within","eta2"]
    ].sort_values("eta2", ascending=False)

    stat_csv = outdir / "variance_decomposition_by_feature.csv"
    stat_df.to_csv(stat_csv, index=False)

    print("\n[INFO] Variance explained by joint group (eta^2), top 10:")
    print(stat_df.head(10).to_string(index=False))
    print(f"\n[INFO] Wrote variance table: {stat_csv}")
    print(f"[INFO] Heatmaps: {heat_dir}")
    print(f"[INFO] PCA scatter: {outdir / 'pca_scatter_joint_groups.png'}")


if __name__ == "__main__":
    main()
