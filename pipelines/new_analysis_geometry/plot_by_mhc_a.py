#!/usr/bin/env python3
"""
Plot one violin plot per geometry feature, grouped by:
  (A) MHC alpha allele (collapsed to GENE*XX), and
  (B) TCR germline VJ combination (germline_vj_pair).

This produces two output subfolders:
  <outdir>/by_mhc_alpha/
  <outdir>/by_germline_vj_pair/

Usage:
  python plot_violins_by_mhc_and_germline.py \
    --csv /path/to/TCR3d_unpaired_bound.csv \
    --outdir /path/to/outdir \
    --state bound \
    --top-k 0 \
    --min-n 10

Notes
-----
- top_k=0 keeps all groups passing min_n
- Groups below min_n are collapsed to "Other"
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- Configure your geometry features here ----
FEATURES: List[str] = [
    "BA","BC1","AC1","BC2","AC2","dc",
    "d_alpha_mhc","d_beta_mhc","d_alpha_beta",
    "theta_rA","phi_rA","theta_rB","phi_rB",
    "theta_pc1A","phi_pc1A","theta_pc1B","phi_pc1B",
    "alpha_cdr3_bend_deg","alpha_cdr3_apex_height_A","alpha_cdr3_apex_resi",
    "beta_cdr3_bend_deg","beta_cdr3_apex_height_A","beta_cdr3_apex_resi",
]


def normalize_mhc_alpha_allele(x: object) -> Optional[str]:
    """
    Collapse mhc_alpha_allele to the 2-field group (GENE*XX), removing any ':...':
      A*02:1150N   -> A*02
      A*02:989N    -> A*02
      B*44:05      -> B*44
      DQA1*01:133Q -> DQA1*01
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None

    # remove common prefixes
    s = s.replace("HLA-", "").replace("HLA:", "").replace("HLA", "").strip()
    s = re.sub(r"^[\s:;-]+", "", s).strip()

    # Prefer an allele-like token that includes '*'
    # Capture GENE*XX optionally followed by :...
    m = re.search(r"\b([A-Za-z0-9]{1,10}\*\d{2})(?::\d{2,4}[A-Za-z0-9]*)?\b", s)
    if m:
        return m.group(1)

    # fallback: first token containing '*', else first token (still may be junk)
    tok = next((t.strip(",;") for t in s.split() if "*" in t), None)
    if tok is None:
        return None
    tok = tok.split(":", 1)[0]
    return tok


def _coerce_features_numeric(df: pd.DataFrame) -> None:
    for feat in FEATURES:
        if feat not in df.columns:
            raise ValueError(f"CSV is missing required feature column: {feat}")
        df[feat] = pd.to_numeric(df[feat], errors="coerce")


def prepare_grouped_dataframe(
    df: pd.DataFrame,
    group_col_raw: str,
    group_col_norm: str,
    normalizer,
    top_k: int,
    min_n: int,
    other_label: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates df[group_col_norm] using `normalizer` and df["group"] for plotting.
    Returns (df_with_group, ordered_group_labels).
    """
    if group_col_raw not in df.columns:
        raise ValueError(f"CSV is missing required column: {group_col_raw!r}")

    df[group_col_norm] = df[group_col_raw].map(normalizer)

    # Drop rows with no group label
    df = df[df[group_col_norm].notna()].copy()

    counts = df[group_col_norm].value_counts()
    keep = counts[counts >= min_n].index.tolist()
    keep = keep[:top_k] if top_k > 0 else keep

    df["group"] = df[group_col_norm].where(df[group_col_norm].isin(keep), other_label)

    group_counts = df["group"].value_counts()
    ordered = [g for g in keep if g in group_counts.index]
    if other_label in group_counts.index:
        ordered.append(other_label)

    return df, ordered


def make_violin_plot(
    df: pd.DataFrame,
    ordered_groups: List[str],
    feature: str,
    outdir: Path,
    title_prefix: str,
    xlabel: str,
    filename_prefix: str,
) -> None:
    data = []
    ns = []
    for g in ordered_groups:
        vals = df.loc[df["group"] == g, feature].dropna().to_numpy(dtype=float)
        data.append(vals)
        ns.append(len(vals))

    if sum(ns) == 0:
        return

    n_groups = len(ordered_groups)
    width = min(max(10.0, 0.55 * n_groups + 4.0), 22.0)
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

    # IQR box + whiskers
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

    labels = [f"{g}\n(n={n})" for g, n in zip(ordered_groups, ns)]
    ax.set_xticks(range(1, n_groups + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_title(f"{title_prefix}: {feature}", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(feature)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"{filename_prefix}_{feature}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--state", type=str, default="bound")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--other-label", type=str, default="Other")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.state is not None and "state" in df.columns:
        df = df[df["state"].astype(str).str.lower() == args.state.lower()].copy()

    _coerce_features_numeric(df)

    # ---------- (A) by MHC alpha allele ----------
    mhc_df, mhc_ordered = prepare_grouped_dataframe(
        df=df.copy(),
        group_col_raw="mhc_alpha_allele",
        group_col_norm="mhc_alpha_allele_norm",
        normalizer=normalize_mhc_alpha_allele,
        top_k=args.top_k,
        min_n=args.min_n,
        other_label=args.other_label,
    )

    out_mhc = args.outdir / "by_mhc_alpha"
    os.makedirs(out_mhc, exist_ok=True)
    print("\nMHCα allele groups and counts:")
    print(mhc_df["group"].value_counts().loc[mhc_ordered])

    for feat in FEATURES:
        make_violin_plot(
            mhc_df, mhc_ordered, feat, out_mhc,
            title_prefix="Grouped by MHCα allele",
            xlabel="MHCα allele (GENE*XX)",
            filename_prefix="violin_mhcA",
        )

    # ---------- (B) by TCR germline VJ pair ----------
    def normalize_germline_vj(x: object) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return s

    germ_df, germ_ordered = prepare_grouped_dataframe(
        df=df.copy(),
        group_col_raw="germline_vj_pair",
        group_col_norm="germline_vj_pair_norm",
        normalizer=normalize_germline_vj,
        top_k=args.top_k,
        min_n=args.min_n,
        other_label=args.other_label,
    )

    out_germ = args.outdir / "by_germline_vj_pair"
    os.makedirs(out_germ, exist_ok=True)
    print("\nGermline VJ pair groups and counts:")
    print(germ_df["group"].value_counts().loc[germ_ordered])

    for feat in FEATURES:
        make_violin_plot(
            germ_df, germ_ordered, feat, out_germ,
            title_prefix="Grouped by TCR germline VJ pair",
            xlabel="germline_vj_pair",
            filename_prefix="violin_germlineVJ",
        )

    print(f"\nSaved {len(FEATURES)} plots by MHCα to: {out_mhc}")
    print(f"Saved {len(FEATURES)} plots by germline VJ pair to: {out_germ}")


if __name__ == "__main__":
    main()
