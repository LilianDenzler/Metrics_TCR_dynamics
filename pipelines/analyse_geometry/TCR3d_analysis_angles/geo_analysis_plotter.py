#!/usr/bin/env python3
"""
Compute TCR geometry angles for UNBOUND and BOUND sets (no pairing assumed),
then plot per-state angle distributions colored by V/J alpha/beta gene combinations.

Outputs (per state, per metric):
  - overlay histogram (top N germline VJ-combos + 'other')
  - colored boxplot (top N germline VJ-combos + 'other')

Also writes:
  - CSV of all computed metrics with parsed germlines

Assumptions:
  - TCR_TOOLS.classes.tcr.TCR exists and pair.calc_angles_tcr(vis=False) returns a dict
  - pair.alpha_germline / pair.beta_germline exist (dict or stringified dict)
"""

import os
import sys
import ast
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Make sure we can import your TCR tools
# --------------------------------------------------------------------
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import TCR  # noqa: E402


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------


# Core geometry metrics you already use
ANGLE_COLUMNS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]

# Optional: include these if your calc_angles_tcr returns them
# (the script will only plot those that exist in the output table)
OPTIONAL_BEND_COLUMNS = [
    "alpha_cdr3_bend_deg",
    "alpha_cdr3_apex_height_A",
    "alpha_cdr3_apex_resi",
    "beta_cdr3_bend_deg",
    "beta_cdr3_apex_height_A",
    "beta_cdr3_apex_resi",
]

# Gene-combo plotting settings
TOP_N_COMBOS = 12  # keep plots readable; others collapsed to "other"
MIN_COUNT_TO_KEEP = 2  # combos with count < this will be collapsed to "other"

# --------------------------------------------------------------------
# GERMLINE PARSING
# --------------------------------------------------------------------
def parse_germline_field(x) -> Dict[str, Any]:
    """Accept dict or stringified dict; return dict (or {} on failure)."""
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

def _extract_gene_and_score(germline_obj: Dict[str, Any], key: str) -> Tuple[str, float]:
    """
    germline_obj example:
      {'v_gene': [('human', 'TRAV12-2*01'), 0.9565], 'j_gene': [('human', 'TRAJ24*02'), 1.0]}
    Returns (gene_str, score_float) for requested key ('v_gene' or 'j_gene').
    """
    if not isinstance(germline_obj, dict) or key not in germline_obj:
        return ("unknown", np.nan)

    item = germline_obj.get(key)
    try:
        gene_tuple = item[0]  # ('human', 'TRAV12-2*01')
        score = float(item[1])
        gene = gene_tuple[1] if isinstance(gene_tuple, (list, tuple)) and len(gene_tuple) >= 2 else "unknown"
        gene = str(gene) if gene else "unknown"
        return (gene, score)
    except Exception:
        return ("unknown", np.nan)

def normalize_germlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      alpha_v_gene, alpha_j_gene, alpha_v_score, alpha_j_score
      beta_v_gene,  beta_j_gene,  beta_v_score,  beta_j_score
      alpha_vj, beta_vj
      germline_vj_pair = alphaV-alphaJ|betaV-betaJ   (your requested combination)
    """
    out = df.copy()

    alpha_parsed = out["alpha_germline"].apply(parse_germline_field) if "alpha_germline" in out.columns else pd.Series([{}]*len(out))
    beta_parsed  = out["beta_germline"].apply(parse_germline_field)  if "beta_germline"  in out.columns else pd.Series([{}]*len(out))

    out["alpha_v_gene"], out["alpha_v_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["alpha_j_gene"], out["alpha_j_score"] = zip(*alpha_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["beta_v_gene"],  out["beta_v_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "v_gene")))
    out["beta_j_gene"],  out["beta_j_score"]  = zip(*beta_parsed.apply(lambda d: _extract_gene_and_score(d, "j_gene")))

    out["alpha_vj"] = out["alpha_v_gene"].astype(str) + "-" + out["alpha_j_gene"].astype(str)
    out["beta_vj"]  = out["beta_v_gene"].astype(str)  + "-" + out["beta_j_gene"].astype(str)

    # Requested: V/J alpha/beta gene combinations
    out["germline_vj_pair"] = out["alpha_vj"].astype(str) + "|" + out["beta_vj"].astype(str)
    return out


# --------------------------------------------------------------------
# 1) GEOMETRY CALCULATION (PER STATE)
# --------------------------------------------------------------------
def compute_angles_for_state(pdb_dir: Path, state: str) -> pd.DataFrame:
    results = []

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = sorted([f for f in pdb_dir.iterdir() if f.suffix.lower() == ".pdb"])
    print(f"[{state}] Found {len(pdb_files)} PDB files in {pdb_dir}")

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
            continue

        if not hasattr(tcr, "pairs") or len(tcr.pairs) == 0:
            print(f"  [WARN] No TCR alpha/beta pair found for {tcr_name}, skipping.")
            continue

        pair = tcr.pairs[0]
        alpha_germline = getattr(pair, "alpha_germline", None)
        beta_germline  = getattr(pair, "beta_germline", None)

        try:
            angle_dict = pair.calc_angles_tcr(out_path=None, vis=False)
            if not isinstance(angle_dict, dict):
                raise RuntimeError("calc_angles_tcr did not return a dict")

            angle_dict["alpha_germline"] = alpha_germline
            angle_dict["beta_germline"]  = beta_germline
            angle_dict["tcr_name"] = tcr_name
            angle_dict["state"] = state
            results.append(pd.DataFrame([angle_dict]))

        except Exception as e:
            print(f"  [WARN] calc_angles_tcr failed for {tcr_name}: {e}")
            continue

    if not results:
        raise RuntimeError(f"No angle results collected for state={state} in dir={pdb_dir}")

    combined = pd.concat(results, ignore_index=True)
    print(f"[{state}] Collected geometry for {combined['tcr_name'].nunique()} TCRs.")
    return combined


# --------------------------------------------------------------------
# 2) PLOTTING HELPERS
# --------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _collapse_categories(s: pd.Series, top_n: int, min_count: int) -> Tuple[pd.Series, List[str]]:
    """
    Keep the top_n categories (by count) that also have count >= min_count.
    Collapse the rest to 'other'. Returns (collapsed_series, kept_categories_in_order).
    """
    s = s.fillna("unknown").astype(str)
    vc = s.value_counts()
    kept = [c for c in vc.index.tolist() if vc[c] >= min_count][:top_n]
    s2 = s.where(s.isin(kept), other="other")
    # keep legend order: most frequent first, then 'other' if present
    cats = kept.copy()
    if (s2 == "other").any():
        cats.append("other")
    return s2, cats

def _get_color_map(categories: List[str]) -> Dict[str, Any]:
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
    if "other" in color_map:
        color_map["other"] = (0.6, 0.6, 0.6, 0.6)
    if "unknown" in color_map:
        color_map["unknown"] = (0.3, 0.3, 0.3, 0.8)
    return color_map

def plot_distributions_by_genecombo(
    df: pd.DataFrame,
    state: str,
    out_dir: Path,
    metrics: List[str],
    combo_col: str = "germline_vj_pair",
    top_n: int = TOP_N_COMBOS,
    min_count: int = MIN_COUNT_TO_KEEP,
):
    """
    For each metric:
      - histogram overlay (top combos + other)
      - colored boxplot (top combos + other)
    """
    ensure_dir(out_dir)

    n_tcrs = df["tcr_name"].nunique() if "tcr_name" in df.columns else len(df)

    # Collapse categories for this state
    combos_collapsed, cat_order = _collapse_categories(df[combo_col], top_n=top_n, min_count=min_count)
    color_map = _get_color_map(cat_order)

    # We'll work on a copy with collapsed categories
    d = df.copy()
    d["_combo_plot"] = combos_collapsed

    # For consistent binning per metric: use Freedman–Diaconis when possible, else fallback
    def _fd_bins(x: np.ndarray) -> int:
        x = x[np.isfinite(x)]
        if x.size < 2:
            return 10
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        if iqr <= 0:
            return 10
        bw = 2 * iqr * (x.size ** (-1/3))
        if bw <= 0:
            return 10
        bins = int(np.ceil((x.max() - x.min()) / bw)) if x.max() > x.min() else 10
        return int(np.clip(bins, 8, 40))

    for metric in metrics:
        if metric not in d.columns:
            continue

        x_all = pd.to_numeric(d[metric], errors="coerce").values
        x_all = x_all[np.isfinite(x_all)]
        if x_all.size == 0:
            continue

        bins = _fd_bins(x_all)

        # -----------------------
        # (A) Histogram overlay
        # -----------------------
        fig = plt.figure(figsize=(10.5, 6.0))
        ax = fig.add_subplot(1, 1, 1)

        # Plot each category separately
        for cat in cat_order:
            sub = pd.to_numeric(d.loc[d["_combo_plot"] == cat, metric], errors="coerce").dropna().values
            if sub.size == 0:
                continue
            ax.hist(
                sub,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=color_map.get(cat, None),
                label=f"{cat} (n={sub.size})",
            )

        ax.set_title(f"{state}: {metric} distribution by V/J combination (n={n_tcrs} TCRs)")
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        fig.tight_layout()

        out_hist = out_dir / f"{state}__{metric}__hist_by_germline_vj_pair.png"
        fig.savefig(out_hist, dpi=300)
        plt.close(fig)
        print(f"  Saved {out_hist}")

        # -----------------------
        # (B) Colored boxplot
        # -----------------------
        # Keep only categories with data for this metric
        data_by_cat = []
        cats_with_data = []
        for cat in cat_order:
            sub = pd.to_numeric(d.loc[d["_combo_plot"] == cat, metric], errors="coerce").dropna().values
            if sub.size == 0:
                continue
            data_by_cat.append(sub)
            cats_with_data.append(cat)

        if len(data_by_cat) >= 1:
            fig = plt.figure(figsize=(max(10, 0.65 * len(cats_with_data)), 6.2))
            ax = fig.add_subplot(1, 1, 1)

            bp = ax.boxplot(
                data_by_cat,
                labels=cats_with_data,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.3),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
            )
            for patch, cat in zip(bp["boxes"], cats_with_data):
                patch.set_facecolor(color_map.get(cat, (0.7, 0.7, 0.7, 0.7)))
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)

            ax.set_title(f"{state}: {metric} by V/J combination (top {top_n}, n={n_tcrs} TCRs)")
            ax.set_xlabel("alphaV-alphaJ | betaV-betaJ")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", labelrotation=60)
            fig.tight_layout()

            out_box = out_dir / f"{state}__{metric}__boxplot_by_germline_vj_pair.png"
            fig.savefig(out_box, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out_box}")

def plot_combo_counts(df: pd.DataFrame, state: str, out_dir: Path, combo_col: str = "germline_vj_pair"):
    """Bar chart of combo frequencies (top N plus 'other')."""
    ensure_dir(out_dir)
    n_tcrs = df["tcr_name"].nunique() if "tcr_name" in df.columns else len(df)

    combos_collapsed, cat_order = _collapse_categories(df[combo_col], top_n=TOP_N_COMBOS, min_count=MIN_COUNT_TO_KEEP)
    vc = combos_collapsed.value_counts()
    # keep order from cat_order
    counts = [int(vc.get(cat, 0)) for cat in cat_order]

    fig = plt.figure(figsize=(max(10, 0.65 * len(cat_order)), 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(len(cat_order)), counts)
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=60, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"{state}: V/J combo counts (n={n_tcrs} TCRs)")
    fig.tight_layout()

    out_path = out_dir / f"{state}__germline_vj_pair_counts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    print("=== Computing angles for UNBOUND ===")
    df_unbound = compute_angles_for_state(UNBOUND_DIR, state="unbound")

    print("\n=== Computing angles for BOUND ===")
    df_bound = compute_angles_for_state(BOUND_DIR, state="bound")

    # Combine, normalize germlines once, then split back
    all_df = pd.concat([df_unbound, df_bound], ignore_index=True)
    all_df = normalize_germlines(all_df)

    # Save the full table
    out_csv = OUTPUT_DIR / "tcr_geometry_unbound_and_bound_with_germlines.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"\nSaved full table: {out_csv}")

    # Per-state plotting
    metrics_to_plot = [c for c in (ANGLE_COLUMNS + OPTIONAL_BEND_COLUMNS) if c in all_df.columns]
    if not metrics_to_plot:
        raise RuntimeError("No metrics to plot found in output dataframe.")

    plot_root = OUTPUT_DIR / "plots_by_state"
    ensure_dir(plot_root)

    for state in ["unbound", "bound"]:
        df_state = all_df[all_df["state"] == state].copy()
        if df_state.empty:
            print(f"[WARN] No rows for state={state}, skipping plots.")
            continue

        state_dir = plot_root / state
        ensure_dir(state_dir)

        # 1) combo counts
        plot_combo_counts(df_state, state=state, out_dir=state_dir, combo_col="germline_vj_pair")

        # 2) per-metric distributions colored by combo
        plot_distributions_by_genecombo(
            df=df_state,
            state=state,
            out_dir=state_dir,
            metrics=metrics_to_plot,
            combo_col="germline_vj_pair",
            top_n=TOP_N_COMBOS,
            min_count=MIN_COUNT_TO_KEEP,
        )

    # Also write a short summary table (counts per state)
    summary = (
        all_df.groupby("state")["tcr_name"]
        .nunique()
        .reset_index()
        .rename(columns={"tcr_name": "n_unique_tcrs"})
    )
    summary_path = OUTPUT_DIR / "summary_counts_by_state.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    print("\nDone. Outputs under:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    UNBOUND_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/ab_chain")
    BOUND_DIR   = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes")
    OUTPUT_DIR  = Path("/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/TCR3d_analysis_angles")
    main()
