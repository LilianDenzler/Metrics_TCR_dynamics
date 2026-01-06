# geo_analysis_tools.py
# ---------------------
# Geometry analysis utilities for TCR-only geometry (angles + dc) and germline coloring.
#
# Assumptions:
# - Input DataFrame represents a single state (all unbound OR all bound). No 'state' column required.
# - Germlines are available in columns 'alpha_germline' and 'beta_germline' (dicts or stringified dicts).
# - Angle columns exist in df (e.g., BA, BC1, AC1, BC2, AC2, dc).

import sys
import ast
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Make sure we can import your TCR tools
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
from TCR_TOOLS.classes.tcr import TCR  # assuming this is correct import


# ------------------------------------------------------------------------------
# Germline parsing / normalization
# ------------------------------------------------------------------------------
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
      germline_4tuple  (alphaV|alphaJ|betaV|betaJ)   <-- recommended combined identifier
    """
    out = df.copy()

    alpha_parsed = out["alpha_germline"].apply(parse_germline_field) if "alpha_germline" in out.columns else pd.Series([{}] * len(out))
    beta_parsed  = out["beta_germline"].apply(parse_germline_field)  if "beta_germline"  in out.columns else pd.Series([{}] * len(out))

    out["alpha_v_gene"], out["alpha_v_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["alpha_j_gene"], out["alpha_j_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["beta_v_gene"],  out["beta_v_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["beta_j_gene"],  out["beta_j_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["germline_v_pair"] = out["alpha_v_gene"].astype(str) + "|" + out["beta_v_gene"].astype(str)
    out["germline_vj_pair"] = (
        out["alpha_v_gene"].astype(str) + "-" + out["alpha_j_gene"].astype(str)
        + "|" +
        out["beta_v_gene"].astype(str) + "-" + out["beta_j_gene"].astype(str)
    )

    out["germline_4tuple"] = (
        out["alpha_v_gene"].astype(str) + "|" +
        out["alpha_j_gene"].astype(str) + "|" +
        out["beta_v_gene"].astype(str) + "|" +
        out["beta_j_gene"].astype(str)
    )

    return out


# ------------------------------------------------------------------------------
# Geometry computation
# ------------------------------------------------------------------------------
def compute_TCR_angles(pdb_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute TCR-only geometry for all PDBs in pdb_dir.
    Returns:
      combined_df: one row per successfully processed PDB (with germlines normalized)
      failed_tcrs: list of pdb stems that failed
    """
    results = []
    failed_tcrs = []

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = sorted([f for f in pdb_dir.iterdir() if f.suffix.lower() == ".pdb"])
    print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")

    for pdb_path in pdb_files:
        tcr_name = pdb_path.stem
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
            failed_tcrs.append(tcr_name)
            continue

        if not hasattr(tcr, "pairs") or len(tcr.pairs) == 0:
            print(f"  [WARN] No TCR alpha/beta pair found for {tcr_name}, skipping.")
            failed_tcrs.append(tcr_name)
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
            failed_tcrs.append(tcr_name)
            continue

        if angle_df is None or angle_df.empty:
            print(f"  [WARN] Empty angle results for {tcr_name}, skipping.")
            failed_tcrs.append(tcr_name)
            continue

        if "pdb_name" in angle_df.columns:
            angle_df = angle_df.rename(columns={"pdb_name": "tcr_name"})
        else:
            angle_df["tcr_name"] = tcr_name

        angle_df["tcr_name"] = tcr_name
        results.append(angle_df)

    if not results:
        raise RuntimeError(f"No angle results collected for dir={pdb_dir}")

    combined = pd.concat(results, ignore_index=True)
    print(f"Collected geometry for {combined['tcr_name'].nunique()} TCRs.")
    combined = normalize_germlines(combined)
    return combined, failed_tcrs


# ------------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _category_colors(
    cats: pd.Series,
    max_legend_items: int = 18,
    other_color=(0.6, 0.6, 0.6, 0.6),
):
    """
    Map categories to colors, collapsing rare categories to 'other'.
    Returns:
      cats_plot (Series), unique (List[str]), cat_to_color (dict)
    """
    cats = cats.fillna("unknown").astype(str)
    vc = cats.value_counts(dropna=False)

    if len(vc) > max_legend_items:
        top = vc.index[: max_legend_items - 1]
        cats_plot = cats.where(cats.isin(top), other="other")
    else:
        cats_plot = cats.copy()

    unique = list(pd.unique(cats_plot))
    if "other" in unique:
        unique = [c for c in unique if c != "other"] + ["other"]

    cmap = plt.get_cmap("tab20", max(len(unique), 1))
    cat_to_color = {c: cmap(i) for i, c in enumerate(unique)}
    if "other" in cat_to_color:
        cat_to_color["other"] = other_color

    return cats_plot, unique, cat_to_color


# ------------------------------------------------------------------------------
# PLOTTERS
# ------------------------------------------------------------------------------
def plot_angle_distributions_grid(
    df: pd.DataFrame,
    out_path: Path,
    ANGLE_COLUMNS: List[str],
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4.2, 3.6),
):
    """
    Violin distributions for each angle in a subplot grid (single-state df).
    Saves one PNG at out_path.
    """
    ensure_dir(out_path.parent)

    n = len(ANGLE_COLUMNS)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, angle in enumerate(ANGLE_COLUMNS):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        if angle not in df.columns:
            ax.axis("off")
            ax.set_title(f"{angle} (missing)", fontsize=10)
            continue

        y = pd.to_numeric(df[angle], errors="coerce")
        sub = pd.DataFrame({angle: y}).dropna()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{angle} (no data)", fontsize=10)
            continue

        sns.violinplot(
            data=sub,
            y=angle,
            inner="box",
            cut=0,
            ax=ax,
        )
        ax.set_title(angle, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(f"{'deg' if angle != 'dc' else 'Å'}")

    # Turn off unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Geometry distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_angle_scatter_by_germline4tuple(
    df: pd.DataFrame,
    ANGLE_COLUMNS: List[str],
    out_dir: Path,
    max_legend_items: int = 22,
):
    """
    For each angle in ANGLE_COLUMNS:
      x-axis = index (or rank) of samples
      y-axis = angle value
      colored by combined alphaV|alphaJ|betaV|betaJ (df['germline_4tuple'])

    This is a per-state diagnostic scatter (NOT unbound vs bound).
    """
    ensure_dir(out_dir)

    if "germline_4tuple" not in df.columns:
        raise ValueError("df must contain 'germline_4tuple'. Run normalize_germlines(df) first.")

    cats_plot, unique, cat_to_color = _category_colors(df["germline_4tuple"], max_legend_items=max_legend_items)

    # stable x coordinate: use dataframe order
    x = np.arange(len(df))

    for angle in ANGLE_COLUMNS:
        if angle not in df.columns:
            print(f"[plot_angle_scatter_by_germline4tuple] Missing {angle}, skipping.")
            continue

        y = pd.to_numeric(df[angle], errors="coerce")
        ok = np.isfinite(y.values)

        fig, ax = plt.subplots(figsize=(11, 5))
        for cat in unique:
            mask = (cats_plot == cat).values & ok
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask],
                y.values[mask],
                s=18,
                alpha=0.85,
                c=[cat_to_color[cat]],
                edgecolors="none",
                label=cat,
            )

        ax.set_title(f"{angle} values colored by germline_4tuple")
        ax.set_xlabel("sample index")
        ax.set_ylabel(f"{angle} ({'deg' if angle != 'dc' else 'Å'})")

        # legend under plot
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
            title="germline_4tuple",
            title_fontsize=9,
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.98])

        out_path = out_dir / f"{angle}_scatter_by_germline_4tuple.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def pca_and_plot_by_germline4tuple(
    df: pd.DataFrame,
    ANGLE_COLUMNS: List[str],
    out_path: Path,
    max_legend_items: int = 22,
    n_components: int = 2,
):
    """
    PCA on geometry features in ANGLE_COLUMNS for a single-state df.
    Produces PC1 vs PC2 scatter colored by germline_4tuple.
    Saves one PNG at out_path and also writes a CSV of PCA coordinates next to it.
    """
    ensure_dir(out_path.parent)

    if "germline_4tuple" not in df.columns:
        raise ValueError("df must contain 'germline_4tuple'. Run normalize_germlines(df) first.")

    # Build feature matrix
    X = df[ANGLE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Drop rows with any NaN in features
    keep = ~X.isna().any(axis=1)
    Xc = X.loc[keep].copy()
    meta = df.loc[keep].copy()

    if Xc.shape[0] < 3:
        raise ValueError(f"Not enough complete rows for PCA (n={Xc.shape[0]}).")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc.values.astype(float))

    pca = PCA(n_components=n_components, random_state=0)
    coords = pca.fit_transform(Xs)

    coords_df = pd.DataFrame({
        "tcr_name": meta["tcr_name"].astype(str).values if "tcr_name" in meta.columns else np.arange(len(meta)).astype(str),
        "PC1": coords[:, 0],
        "PC2": coords[:, 1] if coords.shape[1] > 1 else 0.0,
        "germline_4tuple": meta["germline_4tuple"].astype(str).values,
    })
    coords_csv = out_path.with_suffix(".coords.csv")
    coords_df.to_csv(coords_csv, index=False)

    cats_plot, unique, cat_to_color = _category_colors(coords_df["germline_4tuple"], max_legend_items=max_legend_items)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat in unique:
        m = (cats_plot == cat).values
        if not np.any(m):
            continue
        ax.scatter(
            coords_df.loc[m, "PC1"].values,
            coords_df.loc[m, "PC2"].values,
            s=20,
            alpha=0.85,
            c=[cat_to_color[cat]],
            edgecolors="none",
            label=cat,
        )

    ax.set_title(
        f"PCA of geometry features (colored by germline_4tuple)\n"
        f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%",
        fontsize=12,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

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
        title="germline_4tuple",
        title_fontsize=9,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")
    print(f"Saved {coords_csv}")

