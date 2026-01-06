#!/usr/bin/env python3
"""
TCR geometry plotting + PCA projection utility.

Inputs (4 CSVs)
---------------
- TCR3d_unpaired_unbound.csv
- TCR3d_unpaired_bound.csv
- TCR3d_paired_unbound.csv
- TCR3d_paired_bound.csv

Expected columns (subset)
-------------------------
Core geometry:
  BA, BC1, AC1, BC2, AC2, dc
Optional CDR3:
  alpha_cdr3_bend_deg, alpha_cdr3_apex_height_A,
  beta_cdr3_bend_deg,  beta_cdr3_apex_height_A
Labels:
  tcr_name, state, germline_vj_pair (or alpha_vj/beta_vj), pdb_path

Outputs
-------
- Violin plots per feature across the 4 datasets (one plot per feature)
- Same violin plots with germline-coded scatter overlay
- Paired unbound-vs-bound scatter per feature (same tcr_name), and germline-coded version
- PCA fit on UNPAIRED UNBOUND using BA,BC1,AC1,BC2,AC2,dc
  - Plot unpaired unbound (fit) + unpaired bound (projected)
  - Plot unpaired unbound (fit) + paired (projected)
  - Germline-colored variants (color=germline, marker=dataset)

Usage
-----
python tcr_geometry_plots_and_pca.py \
  --data-dir /workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data \
  --out-dir  /workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/plots \
  --top-n-germlines 15 \
  --min-germline-count 6

Notes
-----
- Angles are wrapped to [-180, 180) for BA/BC1/AC1/BC2/AC2 and (if present) CDR3 bend.
- Germline coloring uses a collapsed label: top N with >= min_count; remainder -> "other".
- To keep plots responsive, scatter overlay can be downsampled via --max-scatter-points.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------
# Config / constants
# ---------------------------
CIRCULAR_COLS = ["BA", "BC1", "AC1", "BC2", "AC2"]
CDR_BEND_COLS = ["alpha_cdr3_bend_deg", "beta_cdr3_bend_deg"]

BASE_FEATURES = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]
OPTIONAL_FEATURES = [
    "alpha_cdr3_bend_deg",
    "alpha_cdr3_apex_height_A",
    "beta_cdr3_bend_deg",
    "beta_cdr3_apex_height_A",
]

DATASETS = [
    #("unpaired_unbound", "TCR3d_unpaired_unbound.csv"),
    #("unpaired_bound",   "TCR3d_unpaired_bound.csv"),
    #("paired_unbound",   "TCR3d_paired_unbound.csv"),
    #("paired_bound",     "TCR3d_paired_bound.csv"),
    ("human_unpaired_unbound", "human_TCR3d_unpaired_unbound.csv"),
    ("human_unpaired_bound",   "human_TCR3d_unpaired_bound.csv"),
    ("human_paired_unbound",   "human_TCR3d_paired_unbound.csv"),
    ("human_paired_bound",     "human_TCR3d_paired_bound.csv"),
    ##("mouse_unpaired_unbound", "mouse_TCR3d_unpaired_unbound.csv"),
    ##("mouse_unpaired_bound",   "mouse_TCR3d_unpaired_bound.csv"),
    ##("mouse_paired_unbound",   "mouse_TCR3d_paired_unbound.csv"),
    #("mouse_paired_bound",     "mouse_TCR3d_paired_bound.csv"),

]


# ---------------------------
# Numeric + angle utils
# ---------------------------
def wrap_deg_to_180(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return ((x + 180.0) % 360.0) - 180.0

def to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def sanitize_dataset_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def ensure_label_col(df: pd.DataFrame, preferred: str = "germline_vj_pair") -> Tuple[pd.DataFrame, str]:
    """
    Prefer germline_vj_pair, else alpha_vj|beta_vj, else alpha_germline|beta_germline, else 'unknown'.
    Returns (df_with_label, label_col_name)
    """
    df = df.copy()
    if preferred in df.columns:
        df[preferred] = df[preferred].fillna("unknown").astype(str)
        return df, preferred

    # Fallbacks
    if "alpha_vj" in df.columns and "beta_vj" in df.columns:
        df["germline_vj_pair"] = (
            df["alpha_vj"].fillna("unknown").astype(str) + "|" + df["beta_vj"].fillna("unknown").astype(str)
        )
        return df, "germline_vj_pair"

    if "alpha_germline" in df.columns and "beta_germline" in df.columns:
        df["germline_vj_pair"] = (
            df["alpha_germline"].fillna("unknown").astype(str) + "|" + df["beta_germline"].fillna("unknown").astype(str)
        )
        return df, "germline_vj_pair"

    df["germline_vj_pair"] = "unknown"
    return df, "germline_vj_pair"


# ---------------------------
# Germline collapsing for plotting
# ---------------------------
def collapse_categories(
    s: pd.Series,
    top_n: int,
    min_count: int
) -> Tuple[pd.Series, List[str], Dict[str, int]]:
    s = s.fillna("unknown").astype(str)
    vc = s.value_counts()
    kept = [c for c in vc.index.tolist() if vc[c] >= min_count][:top_n]
    s2 = s.where(s.isin(kept), other="other")
    counts = s2.value_counts().to_dict()
    cats = kept.copy()
    if (s2 == "other").any():
        cats.append("other")
    if "unknown" in s2.values and "unknown" not in cats:
        cats.append("unknown")
    return s2, cats, counts

def category_colors(categories: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    cm = {cat: cmap(i) for i, cat in enumerate(categories)}
    if "other" in cm:
        cm["other"] = (0.6, 0.6, 0.6, 0.6)
    if "unknown" in cm:
        cm["unknown"] = (0.3, 0.3, 0.3, 0.8)
    return cm


# ---------------------------
# Loading
# ---------------------------
def load_one_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset"] = dataset_name

    # Ensure state exists / consistent
    if "state" in df.columns:
        df["state"] = df["state"].fillna("unknown").astype(str)
    else:
        # infer from filename
        if "unbound" in dataset_name:
            df["state"] = "unbound"
        elif "bound" in dataset_name:
            df["state"] = "bound"
        else:
            df["state"] = "unknown"

    # Ensure germline label column exists
    df, label_col = ensure_label_col(df, preferred="germline_vj_pair")
    df["germline_label"] = df[label_col].fillna("unknown").astype(str)

    # Make numeric for possible features
    df = to_numeric_cols(df, BASE_FEATURES + OPTIONAL_FEATURES)

    # Wrap circular angles
    for c in CIRCULAR_COLS:
        if c in df.columns:
            df[c] = wrap_deg_to_180(df[c])
    for c in CDR_BEND_COLS:
        if c in df.columns:
            df[c] = wrap_deg_to_180(df[c])

    # Basic id columns
    if "tcr_name" not in df.columns:
        # fallback: if pdb_path exists, use stem
        if "pdb_path" in df.columns:
            df["tcr_name"] = df["pdb_path"].astype(str).map(lambda p: Path(p).stem)
        else:
            df["tcr_name"] = np.arange(len(df)).astype(str)

    return df


# ---------------------------
# Plotting: violin (matplotlib)
# ---------------------------
def violin_plot_4way(
    df_all: pd.DataFrame,
    feature: str,
    out_path: Path,
    title: str,
    dataset_order: List[str],
):
    # Collect per dataset vectors
    data = []
    labels = []
    ns = []
    for ds in dataset_order:
        v = df_all.loc[df_all["dataset"] == ds, feature].astype(float).values
        v = v[np.isfinite(v)]
        data.append(v)
        ns.append(len(v))
        labels.append(f"{ds}\n(n={len(v)})")

    if sum(ns) == 0:
        return

    fig = plt.figure(figsize=(11.0, 6.0))
    ax = fig.add_subplot(1, 1, 1)

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )

    # light styling
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(feature)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def violin_with_germline_scatter_overlay(
    df_all: pd.DataFrame,
    feature: str,
    out_path: Path,
    title: str,
    dataset_order: List[str],
    top_n_germlines: int,
    min_germline_count: int,
    max_scatter_points: int,
    seed: int,
):
    # Filter finite
    d = df_all[["dataset", "germline_collapsed", feature]].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d = d[np.isfinite(d[feature].values)]
    if d.empty:
        return

    # Optional downsample for scatter overlay
    rng = np.random.default_rng(seed)
    if len(d) > max_scatter_points:
        d = d.iloc[rng.choice(len(d), size=max_scatter_points, replace=False)].copy()

    # Build per dataset data for violin
    data = []
    labels = []
    for ds in dataset_order:
        v = df_all.loc[df_all["dataset"] == ds, feature].astype(float).values
        v = v[np.isfinite(v)]
        data.append(v)
        labels.append(ds)

    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_subplot(1, 1, 1)

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.85,
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.30)

    # Scatter overlay: jitter x positions
    cats = d["germline_collapsed"].fillna("unknown").astype(str).unique().tolist()
    # Keep deterministic ordering: most common first
    vc = d["germline_collapsed"].value_counts()
    cats = vc.index.tolist()
    cm = category_colors(cats)

    x_map = {ds: i + 1 for i, ds in enumerate(dataset_order)}
    x = d["dataset"].map(x_map).astype(float).values
    jitter = rng.uniform(-0.18, 0.18, size=len(d))
    xj = x + jitter

    for cat in cats:
        m = (d["germline_collapsed"].values == cat)
        if m.sum() == 0:
            continue
        ax.scatter(
            xj[m],
            d.loc[m, feature].values,
            s=14.0,
            alpha=0.70,
            c=[cm.get(cat, (0.6, 0.6, 0.6, 0.6))],
            edgecolors="none",
            label=f"{cat} (n={int(m.sum())})",
        )

    ax.set_xticks(np.arange(1, len(dataset_order) + 1))
    ax.set_xticklabels([f"{ds}\n(n={int(np.isfinite(df_all.loc[df_all['dataset']==ds, feature].astype(float).values).sum())})"
                        for ds in dataset_order])
    ax.set_title(title)
    ax.set_ylabel(feature)

    # Legend outside
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
        title=f"germline (top {top_n_germlines}, min n={min_germline_count})",
        title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Paired unbound vs bound scatter
# ---------------------------
def paired_unbound_vs_bound_table(
    paired_unbound: pd.DataFrame,
    paired_bound: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    Merge paired_unbound and paired_bound by tcr_name.
    If duplicates exist per tcr_name/state, take mean per feature.
    """
    agg_cols = ["tcr_name", "germline_label"]
    pu = paired_unbound.copy()
    pb = paired_bound.copy()

    # Reduce to one row per tcr_name by mean (numeric features)
    pu_agg = pu.groupby("tcr_name", as_index=False)[features].mean(numeric_only=True)
    pb_agg = pb.groupby("tcr_name", as_index=False)[features].mean(numeric_only=True)

    # Take germline from unbound preferentially, else from bound
    germ_pu = pu.groupby("tcr_name", as_index=False)["germline_label"].agg(lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna()) else "unknown")
    germ_pb = pb.groupby("tcr_name", as_index=False)["germline_label"].agg(lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna()) else "unknown")

    merged = pu_agg.merge(pb_agg, on="tcr_name", suffixes=("_unbound", "_bound"), how="inner")
    merged = merged.merge(germ_pu, on="tcr_name", how="left")
    merged = merged.merge(germ_pb, on="tcr_name", how="left", suffixes=("", "_boundlabel"))

    # Combine germline labels
    def pick_germ(row):
        g1 = str(row.get("germline_label", "unknown"))
        g2 = str(row.get("germline_label_boundlabel", "unknown"))
        if g1 and g1 != "unknown" and g1 != "nan":
            return g1
        if g2 and g2 != "unknown" and g2 != "nan":
            return g2
        return "unknown"

    merged["germline_label"] = merged.apply(pick_germ, axis=1)
    if "germline_label_boundlabel" in merged.columns:
        merged = merged.drop(columns=["germline_label_boundlabel"])

    return merged

def scatter_unbound_vs_bound(
    df_pair: pd.DataFrame,
    feature: str,
    out_path: Path,
    title: str,
    color_by_germline: bool,
    top_n_germlines: int,
    min_germline_count: int,
    seed: int,
):
    xcol = f"{feature}_unbound"
    ycol = f"{feature}_bound"
    if xcol not in df_pair.columns or ycol not in df_pair.columns:
        return

    d = df_pair[[xcol, ycol, "germline_collapsed"]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d[np.isfinite(d[xcol].values) & np.isfinite(d[ycol].values)]
    if d.empty:
        return

    fig = plt.figure(figsize=(7.8, 7.2))
    ax = fig.add_subplot(1, 1, 1)

    # Limits + diagonal
    xmin, xmax = float(np.min(d[xcol].values)), float(np.max(d[xcol].values))
    ymin, ymax = float(np.min(d[ycol].values)), float(np.max(d[ycol].values))
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    pad = 0.04 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], linewidth=1.2, alpha=0.6)

    if not color_by_germline:
        ax.scatter(d[xcol].values, d[ycol].values, s=22.0, alpha=0.80, edgecolors="none")
    else:
        vc = d["germline_collapsed"].value_counts()
        cats = vc.index.tolist()
        cm = category_colors(cats)
        for cat in cats:
            m = (d["germline_collapsed"].values == cat)
            if m.sum() == 0:
                continue
            ax.scatter(
                d.loc[m, xcol].values,
                d.loc[m, ycol].values,
                s=22.0,
                alpha=0.80,
                edgecolors="none",
                c=[cm.get(cat, (0.6, 0.6, 0.6, 0.6))],
                label=f"{cat} (n={int(m.sum())})",
            )
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=True,
            title=f"germline (top {top_n_germlines}, min n={min_germline_count})",
            title_fontsize=9,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{feature} (unbound)")
    ax.set_ylabel(f"{feature} (bound)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Directory containing the 4 CSV files")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for plots + tables")
    ap.add_argument("--top-n-germlines", type=int, default=55, help="Top N germlines to keep for coloring")
    ap.add_argument("--min-germline-count", type=int, default=2, help="Min count to keep germline; otherwise -> other")
    ap.add_argument("--max-scatter-points", type=int, default=2500, help="Downsample scatter overlay if too large")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 datasets
    dfs: Dict[str, pd.DataFrame] = {}
    for ds_name, fname in DATASETS:
        p = data_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV: {p}")
        dfs[ds_name] = load_one_csv(p, ds_name)

    # Concatenate
    df_all = pd.concat(list(dfs.values()), ignore_index=True)

    # Decide feature list: include all present from BASE_FEATURES + OPTIONAL_FEATURES
    feature_candidates = BASE_FEATURES + OPTIONAL_FEATURES
    features = [f for f in feature_candidates if f in df_all.columns]
    if not features:
        raise RuntimeError("No expected feature columns were found in the loaded CSVs.")

    # Collapse germlines globally (for consistent coloring)
    df_all["germline_label"] = df_all["germline_label"].fillna("unknown").astype(str)
    germ_collapsed, germ_cats, germ_counts = collapse_categories(
        df_all["germline_label"],
        top_n=args.top_n_germlines,
        min_count=args.min_germline_count,
    )
    df_all["germline_collapsed"] = germ_collapsed

    # Save a quick global summary
    summary = (
        df_all.groupby("dataset")
              .size()
              .reset_index(name="n_rows")
              .merge(df_all.groupby("dataset")["tcr_name"].nunique().reset_index(name="n_unique_tcrs"), on="dataset", how="left")
    )
    summary.to_csv(out_dir / "summary_counts_by_dataset.csv", index=False)

    pd.Series(germ_counts).sort_values(ascending=False).to_csv(out_dir / "germline_collapsed_counts.csv")

    dataset_order = [d[0] for d in DATASETS]

    # ---------------------------
    # 1) Violin plots: 4 datasets in one plot (per feature)
    # ---------------------------
    violin_dir = out_dir / "violin_by_dataset"
    violin_dir.mkdir(parents=True, exist_ok=True)

    for feat in features:
        title = f"Violin: {feat} across 4 datasets"
        violin_plot_4way(
            df_all=df_all,
            feature=feat,
            out_path=violin_dir / f"violin__{sanitize_dataset_name(feat)}.png",
            title=title,
            dataset_order=dataset_order,
        )

    # ---------------------------
    # 2) Violin plots + germline-colored scatter overlay
    # ---------------------------
    violin_germ_dir = out_dir / "violin_with_germline_overlay"
    violin_germ_dir.mkdir(parents=True, exist_ok=True)

    for feat in features:
        title = (
            f"Violin + germline overlay: {feat}\n"
            f"Color=germline (collapsed: top {args.top_n_germlines}, min n={args.min_germline_count})"
        )
        violin_with_germline_scatter_overlay(
            df_all=df_all,
            feature=feat,
            out_path=violin_germ_dir / f"violin_germline_overlay__{sanitize_dataset_name(feat)}.png",
            title=title,
            dataset_order=dataset_order,
            top_n_germlines=args.top_n_germlines,
            min_germline_count=args.min_germline_count,
            max_scatter_points=args.max_scatter_points,
            seed=args.seed,
        )

    # ---------------------------
    # 3) Paired: unbound feature vs bound feature (matched by tcr_name)
    # ---------------------------
    pair_dir = out_dir / "paired_unbound_vs_bound"
    pair_dir.mkdir(parents=True, exist_ok=True)

    paired_unbound = df_all[df_all["dataset"] == "paired_unbound"].copy()
    paired_bound   = df_all[df_all["dataset"] == "paired_bound"].copy()

    # Use features that exist in both; for scatter, only those columns
    pair_features = [f for f in features if f in paired_unbound.columns and f in paired_bound.columns]
    df_pair = paired_unbound_vs_bound_table(paired_unbound, paired_bound, pair_features)

    # add collapsed germline to paired table
    df_pair["germline_label"] = df_pair["germline_label"].fillna("unknown").astype(str)
    df_pair["germline_collapsed"], _, _ = collapse_categories(
        df_pair["germline_label"],
        top_n=args.top_n_germlines,
        min_count=args.min_germline_count,
    )

    df_pair.to_csv(pair_dir / "paired_unbound_vs_bound_merged.csv", index=False)

    for feat in pair_features:
        scatter_unbound_vs_bound(
            df_pair=df_pair,
            feature=feat,
            out_path=pair_dir / f"scatter__{sanitize_dataset_name(feat)}__plain.png",
            title=f"Paired TCRs: {feat} unbound vs bound (matched by tcr_name)",
            color_by_germline=False,
            top_n_germlines=args.top_n_germlines,
            min_germline_count=args.min_germline_count,
            seed=args.seed,
        )
        scatter_unbound_vs_bound(
            df_pair=df_pair,
            feature=feat,
            out_path=pair_dir / f"scatter__{sanitize_dataset_name(feat)}__by_germline.png",
            title=f"Paired TCRs: {feat} unbound vs bound (color=germline)",
            color_by_germline=True,
            top_n_germlines=args.top_n_germlines,
            min_germline_count=args.min_germline_count,
            seed=args.seed,
        )



if __name__ == "__main__":
    main()
