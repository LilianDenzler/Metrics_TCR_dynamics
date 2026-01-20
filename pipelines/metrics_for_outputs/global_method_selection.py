#!/usr/bin/env python3
"""
Global method selection + grouped violin plots for BOTH Mantel r and Trustworthiness,
extended with (1) pooled-region violin plots and (2) cross-representation comparison figures.

For each metrics JSON file type (representation):
  - ca_dist_metrics.json
  - coords_metrics.json
  - dihed_metrics.json

and for each score type:
  - mantel : score = data["mantel"]["gt"]["r"]     (higher better; ~[-1, 1])
  - trust  : score = data["trust"]["gt"]          (higher better; [0, 1])

the script generates:
A) Per-representation outputs (under OUT_ROOT/<rep>/<score_type>/)
   1) best_method_per_tcr_region__<score_type>.csv
   2) region-group count tables (paper-style region labels):
        best_method_counts__<group>__<score_type>.csv/.txt
   3) per-region subplot figures (as before):
        - loop-only fit: 6 subplots (A_CDR1..3, B_CDR1..3)
        - FR-fit single: 6 subplots (A_variable_A_CDR1..3, B_variable_B_CDR1..3)
        - FR-fit all:    2 subplots (A_variable_A_CDR1A_CDR2A_CDR3, B_variable_B_CDR1B_CDR2B_CDR3)
   4) pooled-region violin plots (ALL regions in group combined into one axis):
        pooled_<score_type>_violins__<group>.png

B) Cross-representation comparison figures (under OUT_COMPARE/<score_type>/)
   For each region-group, one figure with 3 subplots (ca vs coords vs dihed),
   each subplot is the pooled-region violin plot for that representation:
     - Loop-only Fit: Representation and Embedding Comparison
     - Framework Fit: Single CDR: Representation and Embedding Comparison
     - Framework Fit: All CDRs per chain: Representation and Embedding Comparison

Region title formatting (paper style):
  - A_CDR1 -> CDR1α ; B_CDR1 -> CDR1β
  - A_variable_A_CDR1 -> FR-fitted CDR1α ; etc.
  - A_variable_A_CDR1A_CDR2A_CDR3 -> FR-fitted all CDRα ; β analog

Median annotation:
  - violin median line shown (showmedians=True)
  - numeric median label placed BELOW the median line when possible (else above)
"""

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------
# Configuration
# -----------------------
BASE_DIR = "/workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_analysis_out"

FILTER_OUT_TICA = True

FIG_DPI = 250
METHOD_LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 12

ANNOTATE_MEDIANS = True
MEDIAN_DECIMALS = 3
MEDIAN_TEXT_OFFSET = 0.05  # in score units

# Region groups (desired keys)
REGIONS_GROUP1 = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
REGIONS_GROUP2 = [
    "A_variable_A_CDR1",
    "A_variable_A_CDR2",
    "A_variable_A_CDR3",
    "B_variable_B_CDR1",
    "B_variable_B_CDR2",
    "B_variable_B_CDR3",
]
REGIONS_GROUP3 = [
    "A_variable_A_CDR1A_CDR2A_CDR3",
    "B_variable_B_CDR1B_CDR2B_CDR3",
]

# Per-representation metrics files
REP_RUNS = [
    ("ca", "ca_dist_metrics.json", "Cα distance"),
    ("coords", "coords_metrics.json", "Cartesian"),
    ("dihed", "dihed_metrics.json", "Dihedrals"),
]

# Output directories
OUT_ROOT = "./global_method_selection_outputs"
OUT_COMPARE = "./global_method_selection_outputs_compare"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------
# JSON score parsing
# -----------------------
def get_mantel_r(data: Dict[str, Any]) -> Optional[float]:
    try:
        mantel = data.get("mantel", None)
        if not isinstance(mantel, dict):
            return None
        gt = mantel.get("gt", None)
        if not isinstance(gt, dict):
            return None
        r = gt.get("r", None)
        if isinstance(r, (int, float)):
            return float(r)
        return None
    except Exception:
        return None


def get_trust(data: Dict[str, Any]) -> Optional[float]:
    try:
        trust = data.get("trust", None)
        if not isinstance(trust, dict):
            return None
        gt = trust.get("gt", None)
        if isinstance(gt, (int, float)):
            return float(gt)
        return None
    except Exception:
        return None


def get_score(data: Dict[str, Any], score_type: str) -> Optional[float]:
    if score_type == "mantel":
        return get_mantel_r(data)
    if score_type == "trust":
        return get_trust(data)
    raise ValueError(f"Unknown score_type: {score_type}")


# -----------------------
# Path parsing
# -----------------------
def parse_path(json_path: Path, base_path: Path) -> Optional[Dict[str, str]]:
    """
    Extract tcr, region, method from path relative to base_path.

    rel parts: [tcr, region_part1, region_part2, ..., method, filename]
    """
    try:
        rel = json_path.relative_to(base_path).parts
        if len(rel) < 4:
            return None
        tcr = rel[0]
        method = rel[-2]
        region_parts = rel[1:-2]
        if not region_parts:
            return None
        region = "_".join(region_parts)
        return {"tcr": tcr, "region": region, "method": method}
    except Exception:
        return None


# -----------------------
# Region normalization & labeling
# -----------------------
def normalize_region(s: str) -> str:
    # canonicalize by removing underscores and lowercasing
    return re.sub(r"_+", "", str(s).strip().lower())


def resolve_region_name(df: pd.DataFrame, desired: str) -> Optional[str]:
    """
    Resolve 'desired' to an actual region in df.
    Primary: exact match.
    Secondary: normalized match (strip underscores, case-insensitive).
    Tertiary: substring fallback.
    """
    regions = df["region"].astype(str).unique().tolist()
    region_set = set(regions)

    if desired in region_set:
        return desired

    nd = normalize_region(desired)
    norm_map = {normalize_region(r): r for r in regions}
    if nd in norm_map:
        return norm_map[nd]

    for r in regions:
        if desired in r or r.endswith(desired):
            return r

    return None


def paper_region_label(desired_region_key: str) -> str:
    """
    Paper-style labels for tables and subplot titles.
    """
    # Loop-only
    if desired_region_key in ("A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"):
        chain = desired_region_key.split("_")[0]  # A/B
        loop = desired_region_key.split("_")[1]   # CDR1/2/3
        suffix = "α" if chain == "A" else "β"
        return f"{loop}{suffix}"

    # FR-fitted single loop
    if desired_region_key.startswith("A_variable_A_CDR") or desired_region_key.startswith("B_variable_B_CDR"):
        parts = desired_region_key.split("_")
        chain = parts[0]
        loop = parts[-1]
        suffix = "α" if chain == "A" else "β"
        return f"FR-fitted {loop}{suffix}"

    # FR-fitted all loops per chain
    if desired_region_key.startswith("A_variable_A_CDR1") and ("CDR2" in desired_region_key) and ("CDR3" in desired_region_key):
        return "FR-fitted all CDRα"
    if desired_region_key.startswith("B_variable_B_CDR1") and ("CDR2" in desired_region_key) and ("CDR3" in desired_region_key):
        return "FR-fitted all CDRβ"

    return desired_region_key


# -----------------------
# Data loading
# -----------------------
def load_scores(base_dir: str, data_filename: str, score_type: str) -> pd.DataFrame:
    base_path = Path(base_dir)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    json_files = glob.glob(f"{base_dir}/**/{data_filename}", recursive=True)
    if not json_files:
        raise FileNotFoundError(f"No '{data_filename}' found under {base_dir}")

    records: List[Dict[str, Any]] = []
    for f in json_files:
        p = Path(f)
        info = parse_path(p, base_path)
        if not info:
            continue
        try:
            with open(p, "r") as fh:
                data = json.load(fh)
            score = get_score(data, score_type=score_type)
            if score is None:
                continue
            records.append(
                {"tcr": info["tcr"], "region": info["region"], "method": info["method"], "score": float(score)}
            )
        except Exception as e:
            logging.warning(f"Skipping {p} due to error: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No valid records extracted for {data_filename} ({score_type}).")
    return df


# -----------------------
# Best method selection + count tables
# -----------------------
def pick_best_method_per_tcr_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (tcr, region), choose the method with the highest score.
    Tie-break: method name ascending (deterministic).
    """
    df2 = df.copy()
    df2["method"] = df2["method"].astype(str)
    df2 = df2.sort_values(by=["tcr", "region", "score", "method"], ascending=[True, True, False, True])
    best = df2.groupby(["tcr", "region"], as_index=False).first()
    best = best.rename(columns={"method": "best_method", "score": "best_score"})
    return best


def count_best_methods_table(df_best: pd.DataFrame, desired_to_actual: Dict[str, str]) -> pd.DataFrame:
    """
    Pivot: rows=paper-label region, cols=method, values=#unique TCRs where method is best in that region.
    """
    inv = {actual: desired for desired, actual in desired_to_actual.items()}

    tmp = df_best.copy()
    tmp = tmp[tmp["region"].isin(inv.keys())].copy()
    tmp["region_key"] = tmp["region"].map(inv)
    tmp["region_label"] = tmp["region_key"].map(paper_region_label)

    counts = (
        tmp.groupby(["region_label", "best_method"])["tcr"]
        .nunique()
        .reset_index(name="n_tcrs")
    )

    pivot = counts.pivot_table(
        index="region_label",
        columns="best_method",
        values="n_tcrs",
        fill_value=0,
        aggfunc="sum",
    )

    ordered_labels = [paper_region_label(k) for k in desired_to_actual.keys()]
    pivot = pivot.reindex(ordered_labels)

    col_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]
    return pivot


def sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:200] if len(s) > 200 else s


# -----------------------
# Plot helpers
# -----------------------
def make_method_color_map(methods: List[str]) -> Dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(methods))]
    return {m: colors[i] for i, m in enumerate(methods)}


def score_axis_config(score_type: str) -> Tuple[Tuple[float, float], str]:
    if score_type == "mantel":
        return (-1.0, 1.0), r"Mantel $r$"
    if score_type == "trust":
        return (0.0, 1.0), r"Trustworthiness"
    raise ValueError(score_type)


def annotate_median_value(ax, x: float, med: float, y_min: float, y_max: float):
    pad = 0.02 * (y_max - y_min)
    y_top = med - MEDIAN_TEXT_OFFSET  # place below median line (text top anchored here)
    if y_top <= y_min + pad:
        y_bot = min(y_max - pad, med + MEDIAN_TEXT_OFFSET)
        ax.text(
            x, y_bot, r"$\tilde{x}$=" + f"{med:.{MEDIAN_DECIMALS}f}",
            ha="center", va="bottom", fontsize=7, rotation=90
        )
    else:
        y_top = max(y_min + pad, y_top)
        ax.text(
            x, y_top, r"$\tilde{x}$=" + f"{med:.{MEDIAN_DECIMALS}f}",
            ha="center", va="top", fontsize=7, rotation=90
        )


def plot_violins_on_ax(
    ax: plt.Axes,
    df_sub: pd.DataFrame,
    methods_order: List[str],
    color_map: Dict[str, Any],
    score_type: str,
    show_ylabel: bool = True,
    title: Optional[str] = None,
):
    ylim, ylabel = score_axis_config(score_type)

    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)

    positions = list(range(1, len(methods_order) + 1))
    ax.set_xticks(positions)
    ax.set_xticklabels(methods_order, rotation=90, fontsize=METHOD_LABEL_FONTSIZE)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    medians_for_annotation: List[Tuple[int, float]] = []
    for pos, m in zip(positions, methods_order):
        vals = df_sub.loc[df_sub["method"] == m, "score"].dropna().values
        if vals.size == 0:
            continue

        med = float(np.median(vals))
        medians_for_annotation.append((pos, med))

        parts = ax.violinplot(
            dataset=[vals],
            positions=[pos],
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.85,
        )

        body = parts["bodies"][0]
        body.set_facecolor(color_map[m])
        body.set_edgecolor("black")
        body.set_alpha(0.9)
        body.set_linewidth(0.6)

        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.0)

    if ANNOTATE_MEDIANS and medians_for_annotation:
        y_min, y_max = ax.get_ylim()
        for pos, med in medians_for_annotation:
            annotate_median_value(ax, pos, med, y_min, y_max)


def pooled_region_df(df: pd.DataFrame, desired_regions: List[str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Pool rows for all regions in 'desired_regions' into one dataframe.
    Returns:
      df_pooled, desired_to_actual mapping (resolved)
    """
    desired_to_actual: Dict[str, str] = {}
    actual_regions: List[str] = []

    for desired in desired_regions:
        actual = resolve_region_name(df, desired)
        if actual is None:
            continue
        desired_to_actual[desired] = actual
        actual_regions.append(actual)

    actual_regions = sorted(set(actual_regions))
    df_pooled = df[df["region"].isin(actual_regions)].copy()
    return df_pooled, desired_to_actual


# -----------------------
# Per-representation plotting (per-region grids + pooled)
# -----------------------
def plot_per_region_grid(
    df: pd.DataFrame,
    desired_regions: List[str],
    out_path: str,
    nrows: int,
    ncols: int,
    score_type: str,
    figure_title: str,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    methods_order = sorted(df["method"].astype(str).unique().tolist())
    color_map = make_method_color_map(methods_order)

    fig_w = max(12, 4.2 * ncols)
    fig_h = max(6, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, desired in enumerate(desired_regions):
        rr = idx // ncols
        cc = idx % ncols
        ax = axes[rr][cc]

        actual = resolve_region_name(df, desired)
        title = paper_region_label(desired)

        if actual is None:
            ylim, ylabel = score_axis_config(score_type)
            ax.set_title(title, fontsize=TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data for region", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks(range(1, len(methods_order) + 1))
            ax.set_xticklabels(methods_order, rotation=90, fontsize=METHOD_LABEL_FONTSIZE)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            continue

        sub = df[df["region"] == actual].copy()
        plot_violins_on_ax(ax, sub, methods_order, color_map, score_type, show_ylabel=True, title=title)

    total_slots = nrows * ncols
    for j in range(len(desired_regions), total_slots):
        rr = j // ncols
        cc = j % ncols
        axes[rr][cc].axis("off")

    legend_handles = [mpatches.Patch(color=color_map[m], label=m) for m in methods_order]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(methods_order), 6), fontsize=8, frameon=False)

    fig.suptitle(figure_title, fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    logging.info(f"Wrote: {out_path}")


def plot_pooled_group_violin(
    df: pd.DataFrame,
    desired_regions: List[str],
    out_path: str,
    score_type: str,
    title: str,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    df_pooled, _ = pooled_region_df(df, desired_regions)
    if df_pooled.empty:
        logging.warning(f"Pooled plot skipped (no data): {out_path}")
        return

    methods_order = sorted(df_pooled["method"].astype(str).unique().tolist())
    color_map = make_method_color_map(methods_order)

    fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.55 * len(methods_order)), 6))
    plot_violins_on_ax(ax, df_pooled, methods_order, color_map, score_type, show_ylabel=True, title=title)

    # Legend
    legend_handles = [mpatches.Patch(color=color_map[m], label=m) for m in methods_order]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(methods_order), 6), fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    logging.info(f"Wrote: {out_path}")


# -----------------------
# Tables for groups (per representation)
# -----------------------
def write_tables_for_group(
    df_best: pd.DataFrame,
    desired_regions: List[str],
    desired_to_actual: Dict[str, str],
    out_dir: str,
    group_name: str,
    score_type: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # restrict mapping to desired set
    desired_to_actual = {k: v for k, v in desired_to_actual.items() if k in desired_regions and v is not None}
    if not desired_to_actual:
        logging.warning(f"Skipping table (no resolved regions) for group={group_name}, score={score_type}")
        return

    pivot = count_best_methods_table(df_best, desired_to_actual)

    csv_path = os.path.join(out_dir, f"best_method_counts__{sanitize_filename(group_name)}__{score_type}.csv")
    pivot.to_csv(csv_path)

    txt_path = os.path.join(out_dir, f"best_method_counts__{sanitize_filename(group_name)}__{score_type}.txt")
    with open(txt_path, "w") as f:
        f.write(pivot.to_string())
        f.write("\n")

    logging.info(f"Wrote: {csv_path}")
    logging.info(f"Wrote: {txt_path}")


# -----------------------
# Cross-representation comparison plots (pooled group as each subplot)
# -----------------------
def plot_cross_rep_comparison(
    dfs_by_rep: Dict[str, pd.DataFrame],
    rep_titles: Dict[str, str],
    desired_regions: List[str],
    score_type: str,
    out_path: str,
    fig_title: str,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
    axes = axes[0]

    # fixed rep order
    rep_order = ["ca", "coords", "dihed"]

    for i, rep_key in enumerate(rep_order):
        ax = axes[i]
        df = dfs_by_rep.get(rep_key, None)
        if df is None or df.empty:
            ylim, ylabel = score_axis_config(score_type)
            ax.set_title(rep_titles.get(rep_key, rep_key), fontsize=TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_xticks([])
            continue

        df_pooled, _ = pooled_region_df(df, desired_regions)
        if df_pooled.empty:
            ylim, ylabel = score_axis_config(score_type)
            ax.set_title(rep_titles.get(rep_key, rep_key), fontsize=TITLE_FONTSIZE)
            ax.text(0.5, 0.5, "No data for group", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_xticks([])
            continue

        methods_order = sorted(df_pooled["method"].astype(str).unique().tolist())
        color_map = make_method_color_map(methods_order)

        plot_violins_on_ax(
            ax=ax,
            df_sub=df_pooled,
            methods_order=methods_order,
            color_map=color_map,
            score_type=score_type,
            show_ylabel=True,
            title=rep_titles.get(rep_key, rep_key),
        )

    fig.suptitle(fig_title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    logging.info(f"Wrote: {out_path}")


# -----------------------
# Full run for one representation + score_type
# -----------------------
def run_one_rep(rep_key: str, data_filename: str, rep_out_dir: str, score_type: str) -> pd.DataFrame:
    """
    Executes all per-representation outputs for a given rep and score type.
    Returns the loaded (and filtered) df for downstream comparison plotting.
    """
    df = load_scores(BASE_DIR, data_filename, score_type=score_type)
    logging.info(f"[{rep_key} | {data_filename} | {score_type}] Loaded {len(df)} rows.")

    if FILTER_OUT_TICA:
        before = len(df)
        df = df[~df["method"].astype(str).str.contains("tica", na=False)]
        logging.info(f"[{rep_key} | {score_type}] Filtered out {before - len(df)} rows containing 'tica'. Remaining: {len(df)}")

    if df.empty:
        return df

    out_dir = os.path.join(rep_out_dir, score_type)
    os.makedirs(out_dir, exist_ok=True)

    # Best method per (TCR, region)
    df_best = pick_best_method_per_tcr_region(df)
    df_best.to_csv(os.path.join(out_dir, f"best_method_per_tcr_region__{score_type}.csv"), index=False)

    # Per-region grids
    plot_per_region_grid(
        df=df,
        desired_regions=REGIONS_GROUP1,
        out_path=os.path.join(out_dir, f"{score_type}_violins__loop_fit__CDRs_alpha_beta.png"),
        nrows=2,
        ncols=3,
        score_type=score_type,
        figure_title=f"Loop-only Fit: Single CDR Evaluation ({score_type})",
    )
    plot_per_region_grid(
        df=df,
        desired_regions=REGIONS_GROUP2,
        out_path=os.path.join(out_dir, f"{score_type}_violins__FR_fit__single_CDRs_alpha_beta.png"),
        nrows=2,
        ncols=3,
        score_type=score_type,
        figure_title=f"Framework Fit: Single CDR Evaluation ({score_type})",
    )
    plot_per_region_grid(
        df=df,
        desired_regions=REGIONS_GROUP3,
        out_path=os.path.join(out_dir, f"{score_type}_violins__FR_fit__all_CDRs_per_chain.png"),
        nrows=1,
        ncols=2,
        score_type=score_type,
        figure_title=f"Framework Fit: All CDRs Per Chain Evaluation ({score_type})",
    )

    # Tables per group (need desired->actual mappings)
    _, map1 = pooled_region_df(df, REGIONS_GROUP1)
    _, map2 = pooled_region_df(df, REGIONS_GROUP2)
    _, map3 = pooled_region_df(df, REGIONS_GROUP3)

    write_tables_for_group(df_best, REGIONS_GROUP1, map1, out_dir, "loop_only_fit_CDRs_alpha_beta", score_type)
    write_tables_for_group(df_best, REGIONS_GROUP2, map2, out_dir, "FR_fit_single_CDRs_alpha_beta", score_type)
    write_tables_for_group(df_best, REGIONS_GROUP3, map3, out_dir, "FR_fit_all_CDRs_per_chain", score_type)

    # NEW: pooled group violins (ALL regions in group combined)
    plot_pooled_group_violin(
        df=df,
        desired_regions=REGIONS_GROUP1,
        out_path=os.path.join(out_dir, f"pooled_{score_type}_violins__loop_only_fit_all_CDRs.png"),
        score_type=score_type,
        title=f"Loop-only Fit: All CDRs Combined ({score_type})",
    )
    plot_pooled_group_violin(
        df=df,
        desired_regions=REGIONS_GROUP2,
        out_path=os.path.join(out_dir, f"pooled_{score_type}_violins__FR_fit_single_CDRs.png"),
        score_type=score_type,
        title=f"Framework Fit: Single CDR (All) Combined ({score_type})",
    )
    plot_pooled_group_violin(
        df=df,
        desired_regions=REGIONS_GROUP3,
        out_path=os.path.join(out_dir, f"pooled_{score_type}_violins__FR_fit_all_CDRs_per_chain.png"),
        score_type=score_type,
        title=f"Framework Fit: All CDRs per chain (α+β) Combined ({score_type})",
    )

    logging.info(f"[{rep_key} | {score_type}] Per-representation outputs complete.")
    return df


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(OUT_COMPARE, exist_ok=True)

    rep_titles = {k: t for (k, _, t) in REP_RUNS}

    # For each score type, run per-representation analyses and then generate cross-rep comparison plots.
    for score_type in ("mantel", "trust"):
        dfs_by_rep: Dict[str, pd.DataFrame] = {}

        # Run each representation
        for rep_key, data_filename, _rep_title in REP_RUNS:
            rep_out_dir = os.path.join(OUT_ROOT, rep_key)
            os.makedirs(rep_out_dir, exist_ok=True)
            df = run_one_rep(rep_key, data_filename, rep_out_dir, score_type)
            dfs_by_rep[rep_key] = df

        # Cross-representation pooled comparison plots (NEW)
        out_dir_compare = os.path.join(OUT_COMPARE, score_type)
        os.makedirs(out_dir_compare, exist_ok=True)

        plot_cross_rep_comparison(
            dfs_by_rep=dfs_by_rep,
            rep_titles=rep_titles,
            desired_regions=REGIONS_GROUP1,
            score_type=score_type,
            out_path=os.path.join(out_dir_compare, f"{score_type}_compare__loop_only_fit__pooled.png"),
            fig_title="Loop-only Fit: Representation and Embedding Comparison",
        )

        plot_cross_rep_comparison(
            dfs_by_rep=dfs_by_rep,
            rep_titles=rep_titles,
            desired_regions=REGIONS_GROUP2,
            score_type=score_type,
            out_path=os.path.join(out_dir_compare, f"{score_type}_compare__FR_fit_single_CDR__pooled.png"),
            fig_title="Framework Fit: Single CDR: Representation and Embedding Comparison",
        )

        plot_cross_rep_comparison(
            dfs_by_rep=dfs_by_rep,
            rep_titles=rep_titles,
            desired_regions=REGIONS_GROUP3,
            score_type=score_type,
            out_path=os.path.join(out_dir_compare, f"{score_type}_compare__FR_fit_all_CDRs_per_chain__pooled.png"),
            fig_title="Framework Fit: All CDRs per chain: Representation and Embedding Comparison",
        )

        logging.info(f"[{score_type}] Cross-representation comparison figures complete.")

    logging.info("All done.")


if __name__ == "__main__":
    main()
