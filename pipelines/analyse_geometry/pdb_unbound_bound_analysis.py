#!/usr/bin/env python

import os
import sys
from pathlib import Path
import math
import ast

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None
# Optional (nice-to-have) for clustering and PCA
try:
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
except Exception:
    linkage = dendrogram = leaves_list = squareform = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

# --------------------------------------------------------------------
# Make sure we can import your TCR tools
# --------------------------------------------------------------------
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import TCR  # assuming this is correct import

# --------------------------------------------------------------------
# CONFIG – EDIT THESE PATHS IF NEEDED
# --------------------------------------------------------------------
UNBOUND_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_unbound_tcr_imgt")
BOUND_DIR   = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_bound_tcr_imgt")
OUTPUT_DIR  = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/geometry_bound_vs_unbound")

# Geometry variables for "TCR-only" geometry (angles in degrees + dc in Å)
ANGLE_COLUMNS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]

# Which of the above should be treated as circular angles (wrapped to [-180, 180))
CIRCULAR_ANGLE_COLS = {"BA", "BC1", "AC1", "BC2", "AC2"}  # do NOT include dc

# Germline columns to color by (per subplot)
GERMLINE_COLOR_COLS = ["alpha_v_gene", "alpha_j_gene", "beta_v_gene", "beta_j_gene"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# --------------------------------------------------------------------
# 1) GEOMETRY CALCULATION
# --------------------------------------------------------------------
def compute_angles_for_state(pdb_dir: Path, state: str) -> pd.DataFrame:
    results = []

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = sorted([f for f in pdb_dir.iterdir() if f.suffix.lower() == ".pdb"])
    print(f"[{state}] Found {len(pdb_files)} PDB files in {pdb_dir}")

    for pdb_path in pdb_files:
        tcr_name = pdb_path.stem  # e.g. 1ao7
        try:
            tcr = TCR(
                input_pdb=str(pdb_path),
                traj_path=None,
                contact_cutoff=5.0,
                min_contacts=50,
                legacy_anarci=True,
            )
        except Exception as e:
            print(f"  [WARN] Failed to initialise TCR for {pdb_path.name}: {e}")
            continue

        if not hasattr(tcr, "pairs") or len(tcr.pairs) == 0:
            print(f"  [WARN] No TCR alpha/beta pair found for {tcr_name}, skipping.")
            continue

        pair = tcr.pairs[0]
        alpha_germline = getattr(pair, "alpha_germline", None)
        beta_germline  = getattr(pair, "beta_germline", None)

        try:
            angle_dict = pair.calc_angles_tcr(out_path=None, vis=False)
            angle_dict["alpha_germline"] = alpha_germline
            angle_dict["beta_germline"] = beta_germline
            angle_df = pd.DataFrame([angle_dict])
        except Exception as e:
            print(f"  [WARN] calc_angles failed for {tcr_name}: {e}")
            continue

        if angle_df is None or angle_df.empty:
            print(f"  [WARN] Empty angle results for {tcr_name}, skipping.")
            continue

        if "pdb_name" in angle_df.columns:
            angle_df = angle_df.rename(columns={"pdb_name": "tcr_name"})
        else:
            angle_df["tcr_name"] = tcr_name

        angle_df["tcr_name"] = tcr_name
        angle_df["state"] = state
        results.append(angle_df)

    if not results:
        raise RuntimeError(f"No angle results collected for state={state} in dir={pdb_dir}")

    combined = pd.concat(results, ignore_index=True)
    print(f"[{state}] Collected geometry for {combined['tcr_name'].nunique()} TCRs.")
    return combined

# --------------------------------------------------------------------
# GERMLINE PROCESSING HELPERS
# --------------------------------------------------------------------
def _extract_gene_and_score(germline_obj, key: str):
    """
    germline_obj example:
      {'v_gene': [('human', 'TRAV12-2*01'), 0.9565], 'j_gene': [('human', 'TRAJ24*02'), 1.0]}
    Returns (gene_str, score_float) for requested key ('v_gene' or 'j_gene').
    """
    if not isinstance(germline_obj, dict):
        return ("unknown", np.nan)
    if key not in germline_obj:
        return ("unknown", np.nan)

    item = germline_obj.get(key)
    try:
        gene_tuple = item[0]  # ('human', 'TRAV12-2*01')
        score = float(item[1])
        gene = gene_tuple[1] if isinstance(gene_tuple, (list, tuple)) and len(gene_tuple) >= 2 else "unknown"
        if not gene:
            gene = "unknown"
        return (str(gene), score)
    except Exception:
        return ("unknown", np.nan)

def parse_germline_field(x):
    """
    Accepts dict or stringified dict; returns dict or {} on failure.
    """
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
    return {}

def normalize_germlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      alpha_v_gene, alpha_j_gene, alpha_v_score, alpha_j_score
      beta_v_gene,  beta_j_gene,  beta_v_score,  beta_j_score
      germline_v_pair, germline_vj_pair
    """
    out = df.copy()

    alpha_parsed = out["alpha_germline"].apply(parse_germline_field) if "alpha_germline" in out.columns else pd.Series([{}]*len(out))
    beta_parsed  = out["beta_germline"].apply(parse_germline_field)  if "beta_germline"  in out.columns else pd.Series([{}]*len(out))

    out["alpha_v_gene"], out["alpha_v_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["alpha_j_gene"], out["alpha_j_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["beta_v_gene"],  out["beta_v_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["beta_j_gene"],  out["beta_j_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["germline_v_pair"] = out["alpha_v_gene"].astype(str) + "|" + out["beta_v_gene"].astype(str)
    out["germline_vj_pair"] = (
        out["alpha_v_gene"].astype(str) + "-" + out["alpha_j_gene"].astype(str) +
        "|" +
        out["beta_v_gene"].astype(str) + "-" + out["beta_j_gene"].astype(str)
    )
    return out

def make_tcr_meta(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one row per tcr_name with germline calls.
    Prefer 'bound' if present.
    Adds a combined 4-gene label: alphaV|alphaJ|betaV|betaJ
    """
    needed = ["tcr_name", "state"] + GERMLINE_COLOR_COLS
    for c in needed:
        if c not in all_df.columns:
            all_df[c] = "unknown"

    meta = (
        all_df[needed]
        .dropna(subset=["tcr_name"])
        .sort_values(["tcr_name", "state"])                 # unbound then bound
        .drop_duplicates(subset=["tcr_name"], keep="last")  # keep bound if present
        .drop(columns=["state"])
    )

    for c in GERMLINE_COLOR_COLS:
        meta[c] = meta[c].fillna("unknown").astype(str)

    meta["germline_4tuple"] = (
        meta["alpha_v_gene"] + "|" + meta["alpha_j_gene"] + "|" + meta["beta_v_gene"] + "|" + meta["beta_j_gene"]
    )
    return meta

# --------------------------------------------------------------------
# 2) PLOT HELPERS (SAVE PLOTS, NO plt.show())
# --------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def plot_angle_distributions_by_state(df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)
    for angle in ANGLE_COLUMNS:
        if angle not in df.columns:
            print(f"[plot_angle_distributions_by_state] Missing angle column {angle}, skipping.")
            continue
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df,
            x="state",
            y=angle,
            inner="box",
            cut=0,
            palette="Set2",
        )
        plt.title(f"{angle} distribution: bound vs unbound")
        plt.xlabel("State")
        plt.ylabel(f"{angle} ({'deg' if angle != 'dc' else 'Å'})")
        out_path = out_dir / f"{angle}_violin_bound_vs_unbound.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path}")

def _scatter_by_category(
    ax,
    x,
    y,
    cat_series: pd.Series,
    title: str,
    max_legend_items: int = 12,
    show_legend: bool = True,
    show_multiplicity: bool = True,
    round_decimals: int = 3,
    size_base: float = 18.0,
    size_scale: float = 40.0,
):
    """
    Scatter plot colored by categories.

    Key improvement vs simple scatter:
    - If show_multiplicity=True, we collapse stacked points (same x,y within rounding)
      and encode multiplicity as marker size. NO jitter; coordinates are preserved.

    Parameters
    ----------
    round_decimals:
        Controls what counts as "same coordinate" (float stability). 3 is usually safe.
        If you want *exact* equality, set round_decimals=None and it will not round.
    size_base, size_scale:
        Marker size = size_base + size_scale * sqrt(count).
    """
    # Ensure arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    cats = cat_series.fillna("unknown").astype(str)

    # Collapse rare categories to keep legend readable
    vc = cats.value_counts(dropna=False)
    if len(vc) > max_legend_items:
        top = vc.index[: max_legend_items - 1]
        cats_plot = cats.where(cats.isin(top), other="other")
    else:
        cats_plot = cats.copy()

    unique = list(pd.unique(cats_plot))
    if "other" in unique:
        unique = [c for c in unique if c != "other"] + ["other"]

    # Stable color mapping
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    cat_to_color = {c: cmap(i) for i, c in enumerate(unique)}
    if "other" in cat_to_color:
        cat_to_color["other"] = (0.6, 0.6, 0.6, 0.6)

    # Build plotting frame
    dfp = pd.DataFrame({"x": x, "y": y, "cat": cats_plot.values})
    dfp = dfp[np.isfinite(dfp[["x", "y"]]).all(axis=1)].copy()

    if dfp.empty:
        ax.set_title(title, fontsize=10)
        return [], []

    if show_multiplicity:
        # Round coords to make "same point" robust for floats
        if round_decimals is None:
            dfp["xr"] = dfp["x"]
            dfp["yr"] = dfp["y"]
        else:
            dfp["xr"] = dfp["x"].round(round_decimals)
            dfp["yr"] = dfp["y"].round(round_decimals)

        # Count stacked points per (cat, xr, yr)
        grp = (
            dfp.groupby(["cat", "xr", "yr"], as_index=False)
               .size()
               .rename(columns={"size": "count"})
        )

        # Plot each category as one scatter (sizes encode multiplicity)
        for c in unique:
            sub = grp[grp["cat"] == c]
            if sub.empty:
                continue
            sizes = size_base + size_scale * np.sqrt(sub["count"].values.astype(float))
            ax.scatter(
                sub["xr"].values,
                sub["yr"].values,
                s=sizes,
                alpha=0.80,
                c=[cat_to_color[c]],
                edgecolors="none",
            )

        # Optional: annotate how much collapsing happened
        n_raw = len(dfp)
        n_collapsed = len(grp)
        n_stacked = n_raw - n_collapsed
        ax.text(
            0.02, 0.02,
            f"stacked: {n_stacked}/{n_raw}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )
    else:
        # Original behavior (will overplot)
        colors = dfp["cat"].map(cat_to_color).values
        ax.scatter(dfp["x"].values, dfp["y"].values, c=colors, s=18, alpha=0.8, edgecolors="none")

    ax.set_title(title, fontsize=10)

    # Legend handles
    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=cat_to_color[c],
            markersize=6,
            label=c,
        )
        for c in unique
    ]
    labels = unique

    if show_legend:
        ax.legend(
            handles=handles,
            labels=labels,
            title="germline",
            loc="best",
            fontsize=7,
            title_fontsize=8,
            frameon=True,
        )

    return handles, labels


def build_paired_geometry_table(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-TCR paired table with columns:
      BA_unbound, ..., dc_unbound,
      BA_bound,   ..., dc_bound,
      BA_shift,   ..., dc_shift

    This avoids the df_wide (tcr_name, angle, unbound, bound, shift) long format entirely.
    """

    # Aggregate one row per TCR per state
    unb = (
        all_df[all_df["state"] == "unbound"]
        .groupby("tcr_name")[ANGLE_COLUMNS]
        .mean(numeric_only=True)
    )
    bnd = (
        all_df[all_df["state"] == "bound"]
        .groupby("tcr_name")[ANGLE_COLUMNS]
        .mean(numeric_only=True)
    )

    # Keep only TCRs present in both
    common = unb.index.intersection(bnd.index)
    unb = unb.loc[common].copy()
    bnd = bnd.loc[common].copy()

    # Compute shifts
    shift = (bnd - unb).copy()

    # Wrap circular angle shifts to [-180, 180)
    for col in ANGLE_COLUMNS:
        if col in CIRCULAR_ANGLE_COLS:
            shift[col] = ((shift[col] + 180.0) % 360.0) - 180.0

    # Build paired output table
    paired = pd.concat(
        [
            unb.add_suffix("_unbound"),
            bnd.add_suffix("_bound"),
            shift.add_suffix("_shift"),
        ],
        axis=1,
    )
    paired.index = paired.index.astype(str)
    paired = paired.reset_index().rename(columns={"index": "tcr_name"})

    return paired

def plot_shift_distributions(paired_df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    for angle in ANGLE_COLUMNS:
        scol = f"{angle}_shift"
        if scol not in paired_df.columns:
            continue

        sub = paired_df[scol].dropna()
        if sub.empty:
            print(f"[plot_shift_distributions_from_paired] No shifts for angle {angle}, skipping.")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(sub, kde=True, bins=20)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.title(f"{angle} shift (bound - unbound)")
        plt.xlabel(f"Shift ({'deg' if angle != 'dc' else 'Å'})")
        plt.ylabel("Count")

        out_path = out_dir / f"{angle}_shift_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path}")

# --------------------------------------------------------------------
# 3) MOVEMENT VECTORS (Δ geometry), COSINE SIM, CLUSTERING, PCA
# --------------------------------------------------------------------
def wrap_deg_to_180(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return ((x + 180.0) % 360.0) - 180.0

def build_delta_matrix(df_wide: pd.DataFrame) -> pd.DataFrame:
    delta = df_wide.pivot(index="tcr_name", columns="angle", values="shift")
    for col in ANGLE_COLUMNS:
        if col not in delta.columns:
            delta[col] = np.nan
    delta = delta[ANGLE_COLUMNS]
    for col in ANGLE_COLUMNS:
        if col in CIRCULAR_ANGLE_COLS:
            delta[col] = wrap_deg_to_180(delta[col])
    return delta

def zscore_matrix(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, skipna=True, ddof=0)
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0
    return (X - mu) / sd_safe

def cosine_similarity(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def plot_cosine_heatmap(sim: np.ndarray, labels: list, out_path: Path):
    plt.figure(figsize=(12, 10))
    plt.imshow(sim, aspect="auto")
    plt.title("Cosine similarity of Δ geometry vectors")
    plt.colorbar(fraction=0.046, pad=0.04)
    n = len(labels)
    if n <= 60:
        plt.xticks(range(n), labels, rotation=90, fontsize=6)
        plt.yticks(range(n), labels, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

def plot_dendrogram(sim: np.ndarray, labels: list, out_path: Path):
    if linkage is None or dendrogram is None or squareform is None or leaves_list is None:
        print("  [INFO] scipy not available; skipping dendrogram.")
        return None
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    plt.figure(figsize=(14, 6))
    dendrogram(Z, labels=labels if len(labels) <= 200 else None, leaf_rotation=90)
    plt.title("Hierarchical clustering of Δ vectors (cosine distance)")
    plt.ylabel("Cosine distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")
    return leaves_list(Z)

def plot_pca(delta_scaled: pd.DataFrame, out_path: Path):
    if PCA is None:
        print("  [INFO] sklearn not available; skipping PCA plot.")
        return
    X = delta_scaled.dropna(axis=0, how="any")
    if X.shape[0] < 3:
        print("  [INFO] Not enough complete cases for PCA; skipping.")
        return
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X.values.astype(float))
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=12)
    plt.title("PCA of Δ geometry vectors")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

def movement_vector_analysis(paired_df: pd.DataFrame, out_dir: Path):
    ensure_dir(out_dir)

    # Build shift matrix: index=tcr_name, columns=ANGLE_COLUMNS
    shift_cols = {f"{c}_shift": c for c in ANGLE_COLUMNS if f"{c}_shift" in paired_df.columns}
    delta = (
        paired_df[["tcr_name"] + list(shift_cols.keys())]
        .dropna(subset=list(shift_cols.keys()), how="any")
        .set_index("tcr_name")
        .rename(columns=shift_cols)
    )

    if delta.shape[0] < 2:
        print("[WARN] Not enough complete Δ vectors to compute cosine similarity.")
        return

    delta_path = out_dir / "delta_vectors_raw.csv"
    delta.to_csv(delta_path)
    print(f"Saved Δ vectors (raw) to: {delta_path}")

    # (Optional scaling disabled by you previously)
    delta_scaled = delta.copy()
    scaled_path = out_dir / "delta_vectors_scaled.csv"
    delta_scaled.to_csv(scaled_path)
    print(f"Saved Δ vectors (scaled) to: {scaled_path}")

    labels = delta_scaled.index.astype(str).tolist()
    sim = cosine_similarity(delta_scaled.values.astype(float))

    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    sim_path = out_dir / "cosine_similarity.csv"
    sim_df.to_csv(sim_path)
    print(f"Saved cosine similarity matrix to: {sim_path}")

    order = plot_dendrogram(sim, labels, out_dir / "dendrogram.png")
    if order is not None:
        sim_ord = sim[np.ix_(order, order)]
        labels_ord = [labels[i] for i in order]
        plot_cosine_heatmap(sim_ord, labels_ord, out_dir / "cosine_heatmap.png")
    else:
        plot_cosine_heatmap(sim, labels, out_dir / "cosine_heatmap.png")

    plot_pca(delta_scaled, out_dir / "pca.png")

# --------------------------------------------------------------------
# PCA on geometry (fit on UNBOUND, project BOUND) + clustering + germline coloring
# Add this block to your script (below helpers), then call pca_gene_cluster_analysis(...)
# --------------------------------------------------------------------



def _pivot_geometry_wide(all_df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Returns wide geometry matrix for a given state:
      index=tcr_name, columns=ANGLE_COLUMNS, values=mean per TCR
    """
    sub = all_df[all_df["state"] == state].copy()
    wide = sub.pivot_table(index="tcr_name", values=ANGLE_COLUMNS, aggfunc="mean")
    # Ensure column order and existence
    for c in ANGLE_COLUMNS:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[ANGLE_COLUMNS]
    return wide


def _wrap_circular_columns(df: pd.DataFrame, cols=CIRCULAR_ANGLE_COLS) -> pd.DataFrame:
    """
    Wrap circular angle columns to [-180, 180). Leaves dc untouched.
    """
    out = df.copy()
    for c in out.columns:
        if c in cols:
            out[c] = ((out[c] + 180.0) % 360.0) - 180.0
    return out


def _build_gene_label(meta: pd.DataFrame, gene_mode: str) -> pd.Series:
    """
    gene_mode in {"alpha_v_gene","alpha_j_gene","beta_v_gene","beta_j_gene","germline_4tuple","germline_v_pair"}
    Returns Series indexed by tcr_name.
    """
    if gene_mode not in meta.columns:
        return pd.Series(index=meta.index, data="unknown", dtype=str)
    return meta[gene_mode].fillna("unknown").astype(str)

def plot_pca_fit_unbound_project_bound(
    coords_df: pd.DataFrame,
    color_by: str,
    out_path: Path,
    title: str,
    alpha: float = 0.35,
):
    """
    coords_df columns must include:
      PC1, PC2, state, and the color_by column.
    Two panels: unbound only (left), bound only (right), same axes.
    """
    # Build category mapping shared across both panels
    cats = coords_df[color_by].fillna("unknown").astype(str)
    vc = cats.value_counts()
    # Collapse very rare categories to keep legend readable
    max_legend_items = 14
    if len(vc) > max_legend_items:
        top = vc.index[: max_legend_items - 1]
        cats_plot = cats.where(cats.isin(top), other="other")
    else:
        cats_plot = cats.copy()

    # Determine consistent color mapping
    unique = list(pd.unique(cats_plot))
    if "other" in unique:
        unique = [c for c in unique if c != "other"] + ["other"]
    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    cat_to_color = {c: cmap(i) for i, c in enumerate(unique)}
    if "other" in cat_to_color:
        cat_to_color["other"] = (0.6, 0.6, 0.6, 0.6)

    coords_df = coords_df.copy()
    coords_df["_cat_plot"] = cats_plot

    # Shared limits
    xmin, xmax = coords_df["PC1"].min(), coords_df["PC1"].max()
    ymin, ymax = coords_df["PC2"].min(), coords_df["PC2"].max()
    xpad = 0.06 * (xmax - xmin) if xmax > xmin else 1.0
    ypad = 0.06 * (ymax - ymin) if ymax > ymin else 1.0
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, st in zip(axes, ["unbound", "bound"]):
        sub = coords_df[coords_df["state"] == st]
        for cat in unique:
            pts = sub[sub["_cat_plot"] == cat]
            if pts.empty:
                continue
            ax.scatter(
                pts["PC1"].values,
                pts["PC2"].values,
                s=18,
                alpha=alpha,
                c=[cat_to_color[cat]],
                edgecolors="none",
                label=cat,
            )
        ax.set_title(st)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Put legend under figure (can be large)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cat_to_color[c], markersize=6, label=c)
        for c in unique
    ]
    fig.legend(
        handles=handles,
        labels=unique,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        fontsize=8,
        title=color_by,
        title_fontsize=9,
        frameon=True,
    )
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

def cluster_and_score(
    X: np.ndarray,
    labels_true: np.ndarray,
    out_prefix: Path,
    method: str = "kmeans",
    k: int = None,
):
    """
    Cluster X and compare clusters to labels_true (gene categories) via ARI/NMI.
    Saves metrics to a small CSV.
    If k is None: uses number of unique labels_true (excluding 'unknown'/'other').
    """
    y = labels_true.astype(str)

    # Choose k based on label diversity (excluding unknown/other)
    uniq = sorted(set([v for v in y if v not in {"unknown", "other"}]))
    if k is None:
        k = max(2, len(uniq)) if len(uniq) >= 2 else 2

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        pred = model.fit_predict(X)
    elif method == "agglomerative":
        if AgglomerativeClustering is None:
            print("  [INFO] AgglomerativeClustering not available; skipping.")
            return None
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        pred = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Scores (against categorical labels)
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)

    # Silhouette (unsupervised quality; requires >=2 clusters and no singletons ideally)
    sil = np.nan
    try:
        if len(set(pred)) >= 2 and X.shape[0] >= k:
            sil = silhouette_score(X, pred)
    except Exception:
        sil = np.nan

    metrics = pd.DataFrame([{
        "method": method,
        "k": k,
        "ARI_vs_gene": ari,
        "NMI_vs_gene": nmi,
        "silhouette": sil,
        "n_samples": X.shape[0],
        "n_gene_labels": len(set(y)),
    }])
    metrics_path = Path(str(out_prefix) + f"_{method}_k{k}_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"  Saved {metrics_path}")

    # Save per-sample assignments
    assign = pd.DataFrame({
        "cluster": pred,
        "gene_label": y,
    })
    assign_path = Path(str(out_prefix) + f"_{method}_k{k}_assignments.csv")
    assign.to_csv(assign_path, index=False)
    print(f"  Saved {assign_path}")

    return pred, metrics

def pca_gene_cluster_analysis(all_df: pd.DataFrame, output_dir: Path, tcr_meta: pd.DataFrame):
    """
    1) Fit scaler+PCA on UNBOUND geometry
    2) Project BOUND into same PCA space
    3) Plot PC1/PC2 scatter colored by gene types (several options)
    4) Run clustering on PCA space and evaluate whether clusters match gene labels
    """
    ensure_dir(output_dir)

    # Build wide matrices
    X_unb = _pivot_geometry_wide(all_df, "unbound")
    X_bnd = _pivot_geometry_wide(all_df, "bound")

    # Keep common TCRs and drop missing
    common = sorted(set(X_unb.index) & set(X_bnd.index) & set(tcr_meta["tcr_name"].astype(str)))
    X_unb = X_unb.loc[common]
    X_bnd = X_bnd.loc[common]

    # Wrap circular angles (optional but recommended)
    X_unb = _wrap_circular_columns(X_unb)
    X_bnd = _wrap_circular_columns(X_bnd)

    # Drop any rows with NaNs in either state (so paired comparisons valid)
    mask_complete = (~X_unb.isna().any(axis=1)) & (~X_bnd.isna().any(axis=1))
    X_unb = X_unb.loc[mask_complete]
    X_bnd = X_bnd.loc[mask_complete]
    common = X_unb.index.tolist()

    if len(common) < 5:
        print("[WARN] Not enough complete paired TCRs for PCA analysis.")
        return

    # Standardize based on UNBOUND only
    scaler = StandardScaler()
    X_unb_scaled = scaler.fit_transform(X_unb.values.astype(float))
    X_bnd_scaled = scaler.transform(X_bnd.values.astype(float))

    # PCA fit on UNBOUND
    pca = PCA(n_components=2, random_state=0)
    U = pca.fit_transform(X_unb_scaled)
    B = pca.transform(X_bnd_scaled)

    # Save PCA model summary
    pca_summary = pd.DataFrame([{
        "PC1_var": float(pca.explained_variance_ratio_[0]),
        "PC2_var": float(pca.explained_variance_ratio_[1]),
        "features": ",".join(ANGLE_COLUMNS),
        "n_samples": len(common),
    }])
    pca_summary.to_csv(output_dir / "pca_fit_on_unbound_summary.csv", index=False)

    # Build coords dataframe (stack unbound/bound)
    meta_idx = tcr_meta.set_index("tcr_name").loc[common].copy()
    coords_unb = pd.DataFrame({"tcr_name": common, "PC1": U[:, 0], "PC2": U[:, 1], "state": "unbound"})
    coords_bnd = pd.DataFrame({"tcr_name": common, "PC1": B[:, 0], "PC2": B[:, 1], "state": "bound"})
    coords = pd.concat([coords_unb, coords_bnd], ignore_index=True)
    coords = coords.merge(meta_idx.reset_index(), on="tcr_name", how="left")

    coords.to_csv(output_dir / "pca_coords_unbound_fit_bound_project.csv", index=False)

    # Plot PCA colored by several gene labels
    color_options = [
        "alpha_v_gene",
        "alpha_j_gene",
        "beta_v_gene",
        "beta_j_gene",
        "germline_4tuple",
    ]
    for col in color_options:
        out_path = output_dir / f"pca_fit_unbound_project_bound_colored_by_{col}.png"
        title = f"PCA (fit on unbound, project bound) colored by {col}\nPC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
        plot_pca_fit_unbound_project_bound(coords, col, out_path, title)

    # --- Clustering in PCA space (use both states or unbound only?) ---
    # Usually: cluster UNBOUND coordinates and test if gene types align.
    # You can also cluster the combined set, but then state mixes in.
    X_cluster = U  # UNBOUND PCA coords
    cluster_out_prefix = output_dir / "cluster_on_unbound_PCA"

    # Evaluate clustering against each gene label
    for gene_col in ["alpha_v_gene", "alpha_j_gene", "beta_v_gene", "beta_j_gene", "germline_4tuple"]:
        y = meta_idx[gene_col].fillna("unknown").astype(str).values

        # Choose k from number of unique gene labels (capped for stability)
        uniq = [v for v in pd.unique(y) if v not in {"unknown", "other"}]
        k = max(2, min(len(uniq), 10)) if len(uniq) >= 2 else 2  # cap to 10

        # KMeans
        cluster_and_score(
            X=X_cluster,
            labels_true=y,
            out_prefix=Path(str(cluster_out_prefix) + f"_label_{gene_col}"),
            method="kmeans",
            k=k,
        )

        # Agglomerative (optional)
        if AgglomerativeClustering is not None:
            cluster_and_score(
                X=X_cluster,
                labels_true=y,
                out_prefix=Path(str(cluster_out_prefix) + f"_label_{gene_col}"),
                method="agglomerative",
                k=k,
            )

    # Optional: visualize clusters on PCA
    # Example: show KMeans clusters (k based on germline_4tuple) on unbound PCA
    y4 = meta_idx["germline_4tuple"].fillna("unknown").astype(str).values
    uniq4 = [v for v in pd.unique(y4) if v not in {"unknown", "other"}]
    k4 = max(2, min(len(uniq4), 10)) if len(uniq4) >= 2 else 2
    km = KMeans(n_clusters=k4, random_state=0, n_init="auto").fit(U)
    pred = km.labels_

    plt.figure(figsize=(7, 6))
    plt.scatter(U[:, 0], U[:, 1], c=pred, s=20, alpha=0.8, edgecolors="none")
    plt.title(f"KMeans clusters on UNBOUND PCA (k={k4})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_dir / f"unbound_pca_kmeans_clusters_k{k4}.png", dpi=300)
    plt.close()

    print(f"[OK] PCA+clustering analysis saved to: {output_dir}")

def plot_unbound_vs_bound_scatter_5panel(paired_df: pd.DataFrame, tcr_meta: pd.DataFrame, out_dir: Path):
    """
    For each angle, create a figure with:
      - 4 panels (2x2): alpha V, alpha J, beta V, beta J (legend inside each)
      - 1 panel (full width below): combined alphaV|alphaJ|betaV|betaJ (legend below figure)

    Uses paired_df columns like BA_unbound, BA_bound, BA_shift, etc.
    """
    ensure_dir(out_dir)
    meta = tcr_meta.set_index("tcr_name")

    # Merge metadata once
    df = paired_df.merge(meta, left_on="tcr_name", right_index=True, how="left")

    for c in GERMLINE_COLOR_COLS + ["germline_4tuple"]:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)

    for angle in ANGLE_COLUMNS:
        xcol = f"{angle}_unbound"
        ycol = f"{angle}_bound"
        if xcol not in df.columns or ycol not in df.columns:
            print(f"[plot_unbound_vs_bound_scatter_5panel_from_paired] Missing columns for {angle}, skipping.")
            continue

        sub = df.dropna(subset=[xcol, ycol]).copy()
        if sub.empty:
            print(f"[plot_unbound_vs_bound_scatter_5panel_from_paired] No data for angle {angle}, skipping.")
            continue

        x = sub[xcol].values
        y = sub[ycol].values

        min_val = np.nanmin([np.nanmin(x), np.nanmin(y)])
        max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
        pad = 0.05 * (max_val - min_val) if max_val > min_val else 1.0
        lo, hi = min_val - pad, max_val + pad

        fig = plt.figure(figsize=(12, 13))
        gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1, 1, 1.15], hspace=0.28, wspace=0.20)

        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        axC  = fig.add_subplot(gs[2, :])

        panel_specs = [
            (ax00, "alpha_v_gene", "Colored by alpha V gene"),
            (ax01, "alpha_j_gene", "Colored by alpha J gene"),
            (ax10, "beta_v_gene",  "Colored by beta V gene"),
            (ax11, "beta_j_gene",  "Colored by beta J gene"),
        ]

        for ax, col, ttl in panel_specs:
            _scatter_by_category(
                ax=ax,
                x=x,
                y=y,
                cat_series=sub[col],
                title=ttl,
                max_legend_items=12,
                show_legend=True,
            )
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"{angle} unbound ({'deg' if angle != 'dc' else 'Å'})")
            ax.set_ylabel(f"{angle} bound ({'deg' if angle != 'dc' else 'Å'})")

        # Combined legend below
        handles, labels_ = _scatter_by_category(
            ax=axC,
            x=x,
            y=y,
            cat_series=sub["germline_4tuple"],
            title="Colored by combined alphaV|alphaJ|betaV|betaJ",
            max_legend_items=30,
            show_legend=False,
        )
        axC.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axC.set_xlim(lo, hi)
        axC.set_ylim(lo, hi)
        axC.set_xlabel(f"{angle} unbound ({'deg' if angle != 'dc' else 'Å'})")
        axC.set_ylabel(f"{angle} bound ({'deg' if angle != 'dc' else 'Å'})")

        fig.legend(
            handles=handles,
            labels=labels_,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=6,
            fontsize=7,
            title="alphaV|alphaJ|betaV|betaJ",
            title_fontsize=8,
            frameon=True,
        )

        fig.suptitle(f"{angle}: unbound vs bound (colored by germlines)", fontsize=13, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])

        out_path = out_dir / f"{angle}_scatter_unbound_vs_bound_germlines_5panel.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

# --------------------------------------------------------------------
# 4) MAIN LOGIC
# --------------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    print("=== Computing angles for UNBOUND ===")
    df_unbound = compute_angles_for_state(UNBOUND_DIR, state="unbound")

    print("\n=== Computing angles for BOUND ===")
    df_bound = compute_angles_for_state(BOUND_DIR, state="bound")

    tcr_unbound = set(df_unbound["tcr_name"].unique())
    tcr_bound = set(df_bound["tcr_name"].unique())
    common_tcrs = sorted(tcr_unbound & tcr_bound)

    print(f"\nCommon TCRs with both bound & unbound: {len(common_tcrs)}")
    if not common_tcrs:
        print("No overlapping TCRs. Check naming or directories.")
        return

    df_unbound_common = df_unbound[df_unbound["tcr_name"].isin(common_tcrs)].copy()
    df_bound_common = df_bound[df_bound["tcr_name"].isin(common_tcrs)].copy()

    all_df = pd.concat([df_unbound_common, df_bound_common], ignore_index=True)
    all_df = normalize_germlines(all_df)

    # Build a per-TCR metadata table for coloring scatter plots
    tcr_meta = make_tcr_meta(all_df)
    tcr_meta.to_csv(OUTPUT_DIR / "tcr_germline_metadata.csv", index=False)

    angles_csv = OUTPUT_DIR / "tcr_geometry_bound_vs_unbound_all_angles.csv"
    all_df.to_csv(angles_csv, index=False)
    print(f"\nSaved combined angle table to: {angles_csv}")

    print("\n=== Plotting global angle distributions (bound vs unbound) ===")
    dist_dir = OUTPUT_DIR / "distributions"
    plot_angle_distributions_by_state(all_df, dist_dir)

    # PCA (fit on unbound) + project bound + clustering vs germlines
    pca_out = OUTPUT_DIR / "pca_fit_unbound_project_bound"
    pca_gene_cluster_analysis(all_df, pca_out, tcr_meta)
    paired_df=build_paired_geometry_table(all_df)
    paired_df.to_csv(OUTPUT_DIR / "tcr_geometry_paired_table.csv", index=False)

    # 6) Scatter plots: unbound vs bound (4-panel colored by germlines)
    print("\n=== Plotting unbound vs bound scatter per angle (4-panel germline coloring) ===")
    scatter_dir = OUTPUT_DIR / "scatter_unbound_vs_bound_germlines"
    plot_unbound_vs_bound_scatter_5panel(paired_df, tcr_meta, scatter_dir)


    print("\n=== Plotting shift distributions (bound - unbound) ===")
    shift_dir = OUTPUT_DIR / "shift_distributions"
    plot_shift_distributions(paired_df, shift_dir)

    # Shift summary
    print("\n=== Summary of shifts (bound - unbound) per angle ===")
    for angle in ANGLE_COLUMNS:
        scol = f"{angle}_shift"
        if scol not in paired_df.columns:
            continue
        sub = paired_df[scol].dropna()
        if sub.empty:
            continue
        print(f"  {angle}: mean={sub.mean(): .2f}, median={sub.median(): .2f}, std={sub.std(): .2f}, n={len(sub)}")

    # Movement-vector analysis from paired table
    print("\n=== Movement-vector analysis (Δ vectors across angles/dc) ===")
    mv_dir = OUTPUT_DIR / "movement_vectors"
    movement_vector_analysis(paired_df, mv_dir)

    print("\nDone. Check outputs under:")
    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
