#!/usr/bin/env python3
"""
Generate violin plots for each geometry feature:
  1) grouped by MHC alpha allele (FULL token; no truncation)
  2) grouped by (MHC alpha allele, germline_vj_pair) combination

Fixes:
  - no MHC truncation
  - robustly skips groups with no finite data for a given feature (avoids zero-size arrays)

Example:
  python plot_by_mhc_germline_violin.py \
    --csv /path/to/human_TCR3d_unpaired_bound.csv \
    --outdir /path/to/outdir \
    --state bound \
    --mhc-top-k 0 --mhc-min-n 2 \
    --combo-top-k 60 --combo-min-n 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURES: List[str] = [
    "BA","BC1","AC1","BC2","AC2","dc",
    "d_alpha_mhc","d_beta_mhc","d_alpha_beta",
    "theta_rA","phi_rA","theta_rB","phi_rB",
    "theta_pc1A","phi_pc1A","theta_pc1B","phi_pc1B",
    "alpha_cdr3_bend_deg","alpha_cdr3_apex_height_A","alpha_cdr3_apex_resi",
    "beta_cdr3_bend_deg","beta_cdr3_apex_height_A","beta_cdr3_apex_resi",
]


def normalize_mhc_alpha_full_token(x: object) -> Optional[str]:
    """
    Return a stable "full allele token" without truncating to GENE*XX.

    Strategy:
      - strip whitespace
      - remove 'HLA-' / 'HLA:' / 'HLA' prefixes
      - pick the first token that contains '*' (typical allele token)
      - if none, attempt regex match; else None
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    s = s.replace("HLA-", "").replace("HLA:", "").replace("HLA", "").strip()
    s = re.sub(r"^[\s:;-]+", "", s).strip()

    # Prefer first whitespace-delimited token containing '*'
    for tok in s.split():
        if "*" in tok:
            return tok.strip(",;")

    # Fallback: regex anywhere
    m = re.search(r"\b([A-Za-z0-9]{1,10}\*\d{2}:\d{2,4}[A-Za-z0-9]*)\b", s)
    if m:
        return m.group(1)

    return None


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)[:180]


def prepare_dataframe(csv_path: Path, state: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if state is not None and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower() == state.lower()].copy()

    if "mhc_alpha_allele" not in df.columns:
        raise ValueError("CSV is missing required column: 'mhc_alpha_allele'")
    if "germline_vj_pair" not in df.columns:
        raise ValueError("CSV is missing required column: 'germline_vj_pair'")

    df["mhc_alpha_full"] = df["mhc_alpha_allele"].map(normalize_mhc_alpha_full_token)

    df["germline_vj_pair"] = df["germline_vj_pair"].astype(str)
    df.loc[df["germline_vj_pair"].str.lower().isin({"nan", "none"}), "germline_vj_pair"] = np.nan

    # Drop rows missing grouping keys
    df = df[df["mhc_alpha_full"].notna() & df["germline_vj_pair"].notna()].copy()

    # Ensure numeric
    for feat in FEATURES:
        if feat not in df.columns:
            raise ValueError(f"CSV is missing required feature column: {feat}")
        df[feat] = pd.to_numeric(df[feat], errors="coerce")

    # Combo label
    df["mhc_x_germline"] = df["mhc_alpha_full"] + "|" + df["germline_vj_pair"]

    return df


def choose_groups(
    df: pd.DataFrame,
    group_col: str,
    top_k: int,
    min_n: int,
    other_label: str = "Other",
) -> Tuple[pd.DataFrame, List[str], str]:
    counts = df[group_col].value_counts(dropna=True)
    keep = counts[counts >= min_n].index.tolist()
    if top_k > 0:
        keep = keep[:top_k]

    out_col = f"{group_col}_group"
    df[out_col] = df[group_col].where(df[group_col].isin(keep), other_label)

    # order by descending counts among kept
    group_counts = df[out_col].value_counts()
    ordered = [g for g in keep if g in group_counts.index]
    if other_label in group_counts.index:
        ordered.append(other_label)

    return df, ordered, out_col


def make_violin_plot(
    df: pd.DataFrame,
    group_col: str,
    ordered_groups: List[str],
    feature: str,
    outdir: Path,
    title: str,
    xlabel: str,
) -> None:
    """
    Robust violin plot:
      - builds arrays per group
      - DROPS groups where vals is empty after dropping NaNs / non-finite
      - avoids matplotlib zero-size reduction error
    """
    data = []
    labels = []
    ns = []

    for g in ordered_groups:
        vals = df.loc[df[group_col] == g, feature].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]  # drop NaN/inf
        if vals.size == 0:
            continue  # critical fix
        data.append(vals)
        ns.append(int(vals.size))
        labels.append(f"{g}\n(n={vals.size})")

    if len(data) == 0:
        # nothing to plot for this feature
        return

    n_groups = len(data)
    width = min(max(10.0, 0.55 * n_groups + 4.0), 30.0)
    height = 6.0

    fig, ax = plt.subplots(figsize=(width, height))

    parts = ax.violinplot(
        data,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for body in parts.get("bodies", []):
        body.set_alpha(0.65)

    # IQR + whiskers
    for i, vals in enumerate(data, start=1):
        if len(vals) < 2:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo = np.min(vals[vals >= (q1 - 1.5 * iqr)]) if np.any(vals >= (q1 - 1.5 * iqr)) else np.min(vals)
        hi = np.max(vals[vals <= (q3 + 1.5 * iqr)]) if np.any(vals <= (q3 + 1.5 * iqr)) else np.max(vals)

        ax.plot([i, i], [q1, q3], linewidth=6)
        ax.plot([i, i], [lo, q1], linewidth=1.5)
        ax.plot([i, i], [q3, hi], linewidth=1.5)

    ax.set_xticks(range(1, n_groups + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(feature)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"violin_{_sanitize_filename(feature)}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--state", type=str, default="bound")

    p.add_argument("--mhc-top-k", type=int, default=0, help="0 = keep all passing min-n")
    p.add_argument("--mhc-min-n", type=int, default=2)

    p.add_argument("--combo-top-k", type=int, default=60, help="0 = keep all passing min-n")
    p.add_argument("--combo-min-n", type=int, default=2)

    p.add_argument("--other-label", type=str, default="Other")
    args = p.parse_args()

    df = prepare_dataframe(args.csv, args.state)

    outdir_mhc = args.outdir / "by_mhc"
    outdir_combo = args.outdir / "by_mhc_x_germline"

    # 1) MHC-only
    df, ordered_mhc, mhc_group_col = choose_groups(
        df=df,
        group_col="mhc_alpha_full",
        top_k=args.mhc_top_k,
        min_n=args.mhc_min_n,
        other_label=args.other_label,
    )
    print("\nMHC groups (ordered) and counts:")
    vc_mhc = df[mhc_group_col].value_counts()
    print(vc_mhc.loc[[g for g in ordered_mhc if g in vc_mhc.index]])

    for feat in FEATURES:
        make_violin_plot(
            df=df,
            group_col=mhc_group_col,
            ordered_groups=ordered_mhc,
            feature=feat,
            outdir=outdir_mhc,
            title=f"{feat} by MHCα allele (full token)",
            xlabel="MHCα allele",
        )

    # 2) MHC x germline combo
    df, ordered_combo, combo_group_col = choose_groups(
        df=df,
        group_col="mhc_x_germline",
        top_k=args.combo_top_k,
        min_n=args.combo_min_n,
        other_label=args.other_label,
    )
    print("\n(MHC x germline) groups (top 30 shown) and counts:")
    vc_combo = df[combo_group_col].value_counts()
    show = [g for g in ordered_combo if g in vc_combo.index][:30]
    print(vc_combo.loc[show])
    if len(ordered_combo) > 30:
        print(f"... plus {len(ordered_combo) - 30} more groups")

    for feat in FEATURES:
        make_violin_plot(
            df=df,
            group_col=combo_group_col,
            ordered_groups=ordered_combo,
            feature=feat,
            outdir=outdir_combo,
            title=f"{feat} by (MHCα allele, germline_vj_pair)",
            xlabel="MHCα|germline_vj_pair",
        )

    print(f"\nSaved {len(FEATURES)} plots to:")
    print(f"  {outdir_mhc}")
    print(f"  {outdir_combo}")


if __name__ == "__main__":
    main()
