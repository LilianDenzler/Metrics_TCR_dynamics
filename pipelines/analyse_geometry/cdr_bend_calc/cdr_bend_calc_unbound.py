#!/usr/bin/env python
"""
Analyse CDR3 bend metrics for:
  1) Unpaired sets:
      - unbound_pdbs_TCR3d
      - bound_pdbs_TCR3d
     -> distributions per set + overlay (unbound vs bound)

  2) Paired benchmark sets (same TCR names in both dirs):
      - expanded_benchmark_unbound_tcr_imgt/<tcr>.pdb
      - expanded_benchmark_bound_tcr_imgt/<tcr>.pdb
     -> distributions per set + overlay + shift (bound - unbound)

Outputs:
  - CSVs with computed metrics
  - PNG figures for bend angle, apex height, and apex resi distributions
  - Paired shift plots (hist + scatter)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths / imports (adjust only if your repo layout changes)
# ---------------------------------------------------------------------
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import TCR  # your project class





# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def list_pdbs(pdb_dir: Path) -> List[Path]:
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")
    return sorted([p for p in pdb_dir.iterdir() if p.suffix.lower() == ".pdb"])


def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def compute_geo_with_bends(pdb_path: Path, out_vis_root: Path, vis: bool = False) -> Optional[Dict]:
    """
    Compute angles + bend metrics for a single PDB via your TCR pipeline.
    Expects your modified geometry code to add:
      alpha_cdr3_bend_deg, alpha_cdr3_apex_height_A, alpha_cdr3_apex_resi
      beta_cdr3_bend_deg,  beta_cdr3_apex_height_A,  beta_cdr3_apex_resi
    """
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
        print(f"[WARN] Failed to initialize TCR for {tcr_name}: {e}")
        return None

    if not getattr(tcr, "pairs", None):
        print(f"[WARN] No pairs found for {tcr_name}")
        return None

    pair = tcr.pairs[0]

    alpha_germline = getattr(pair, "alpha_germline", None)
    beta_germline  = getattr(pair, "beta_germline", None)

    try:
        angle_dict = pair.calc_angles_tcr(out_path=None, vis=False)
    except Exception as e:
        print(f"[WARN] calc_angles_tcr failed for {tcr_name}: {e}")
        return None

    # attach identifiers
    angle_dict["tcr_name"] = tcr_name
    angle_dict["alpha_germline"] = alpha_germline
    angle_dict["beta_germline"] = beta_germline

    # normalize expected bend keys (keep originals too)
    # (If a key is missing, store NaN/unknown.)
    angle_dict["alpha_bend_deg"] = _safe_float(angle_dict.get("alpha_cdr3_bend_deg"))
    angle_dict["beta_bend_deg"]  = _safe_float(angle_dict.get("beta_cdr3_bend_deg"))

    angle_dict["alpha_apex_height_A"] = _safe_float(angle_dict.get("alpha_cdr3_apex_height_A"))
    angle_dict["beta_apex_height_A"]  = _safe_float(angle_dict.get("beta_cdr3_apex_height_A"))

    angle_dict["alpha_apex_resi"] = angle_dict.get("alpha_cdr3_apex_resi", "unknown")
    angle_dict["beta_apex_resi"]  = angle_dict.get("beta_cdr3_apex_resi", "unknown")

    return angle_dict


def compute_set(pdb_dir: Path, out_dir: Path, label: str, vis: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute metrics for all PDBs in a directory.
    Returns (df, failed_names).
    """
    out_vis_root = out_dir / f"vis_{label}"
    out_vis_root.mkdir(parents=True, exist_ok=True)

    pdbs = list_pdbs(pdb_dir)
    rows = []
    failed = []

    print(f"\n=== Computing set '{label}' from: {pdb_dir}  (n_files={len(pdbs)}) ===")

    for i, pdb_path in enumerate(pdbs, start=1):
        if i % 25 == 0 or i == 1 or i == len(pdbs):
            print(f"  [{label}] {i}/{len(pdbs)} ...")

        d = compute_geo_with_bends(pdb_path, out_vis_root=out_vis_root, vis=vis)
        if d is None:
            failed.append(pdb_path.stem)
            continue
        d["set_label"] = label
        rows.append(d)

    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows)

    # Save
    df.to_csv(out_dir / f"{label}_angles_bends.csv", index=False)
    (out_dir / f"{label}_failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))

    print(f"  -> {label}: computed n={len(df)}; failed n={len(failed)}")
    return df, failed


def compute_paired_set(
    unbound_dir: Path,
    bound_dir: Path,
    out_dir: Path,
    label: str,
    vis: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute unbound/bound metrics for matched names, then build a paired table with shifts.
    Returns: (df_unbound, df_bound, df_pairs)
    """
    out_vis_root = out_dir / f"vis_{label}"
    out_vis_root.mkdir(parents=True, exist_ok=True)

    unbound_map = {p.stem: p for p in list_pdbs(unbound_dir)}
    bound_map   = {p.stem: p for p in list_pdbs(bound_dir)}

    common = sorted(set(unbound_map.keys()) & set(bound_map.keys()))
    print(f"\n=== Computing paired set '{label}' ===")
    print(f"  Unbound files: {len(unbound_map)}")
    print(f"  Bound files:   {len(bound_map)}")
    print(f"  Common pairs:  {len(common)}")

    rows_u, rows_b = [], []
    failed_pairs = []

    for i, name in enumerate(common, start=1):
        if i % 25 == 0 or i == 1 or i == len(common):
            print(f"  [{label}] {i}/{len(common)} ...")

        d_u = compute_geo_with_bends(unbound_map[name], out_vis_root=out_vis_root / "unbound", vis=vis)
        d_b = compute_geo_with_bends(bound_map[name],   out_vis_root=out_vis_root / "bound",   vis=vis)

        if d_u is None or d_b is None:
            failed_pairs.append(name)
            continue

        d_u["set_label"] = f"{label}_unbound"
        d_b["set_label"] = f"{label}_bound"
        rows_u.append(d_u)
        rows_b.append(d_b)

    df_u = pd.DataFrame(rows_u)
    df_b = pd.DataFrame(rows_b)

    df_u.to_csv(out_dir / f"{label}_unbound_angles_bends.csv", index=False)
    df_b.to_csv(out_dir / f"{label}_bound_angles_bends.csv", index=False)
    (out_dir / f"{label}_failed_pairs.txt").write_text("\n".join(failed_pairs) + ("\n" if failed_pairs else ""))

    # Build paired shifts
    if df_u.empty or df_b.empty:
        df_pairs = pd.DataFrame()
    else:
        u = df_u.set_index("tcr_name")
        b = df_b.set_index("tcr_name")
        common2 = sorted(set(u.index) & set(b.index))

        def _col(name):  # convenience
            return name

        recs = []
        for t in common2:
            recs.append({
                "tcr_name": t,

                "alpha_bend_unbound": _safe_float(u.loc[t, _col("alpha_bend_deg")]),
                "alpha_bend_bound":   _safe_float(b.loc[t, _col("alpha_bend_deg")]),
                "alpha_bend_shift":   _safe_float(b.loc[t, _col("alpha_bend_deg")]) - _safe_float(u.loc[t, _col("alpha_bend_deg")]),

                "beta_bend_unbound": _safe_float(u.loc[t, _col("beta_bend_deg")]),
                "beta_bend_bound":   _safe_float(b.loc[t, _col("beta_bend_deg")]),
                "beta_bend_shift":   _safe_float(b.loc[t, _col("beta_bend_deg")]) - _safe_float(u.loc[t, _col("beta_bend_deg")]),

                "alpha_apexH_unbound": _safe_float(u.loc[t, _col("alpha_apex_height_A")]),
                "alpha_apexH_bound":   _safe_float(b.loc[t, _col("alpha_apex_height_A")]),
                "alpha_apexH_shift":   _safe_float(b.loc[t, _col("alpha_apex_height_A")]) - _safe_float(u.loc[t, _col("alpha_apex_height_A")]),

                "beta_apexH_unbound": _safe_float(u.loc[t, _col("beta_apex_height_A")]),
                "beta_apexH_bound":   _safe_float(b.loc[t, _col("beta_apex_height_A")]),
                "beta_apexH_shift":   _safe_float(b.loc[t, _col("beta_apex_height_A")]) - _safe_float(u.loc[t, _col("beta_apex_height_A")]),
            })

        df_pairs = pd.DataFrame(recs)
        df_pairs.to_csv(out_dir / f"{label}_paired_shifts.csv", index=False)

    print(f"  -> paired computed: unbound n={len(df_u)}, bound n={len(df_b)}, pairs n={len(df_pairs)}")
    return df_u, df_b, df_pairs


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _finite(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").astype(float).values
    x = x[np.isfinite(x)]
    return x


def plot_hist_single(df: pd.DataFrame, col: str, out_path: Path, title: str, xlabel: str, n: int, bins: int = 30):
    x = _finite(df[col]) if col in df.columns else np.array([])
    fig = plt.figure(figsize=(7, 5))
    if len(x) == 0:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
    else:
        plt.hist(x, bins=bins, alpha=0.8)
        plt.title(f"{title}\n(n={n})")
        plt.xlabel(xlabel)
        plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hist_overlay(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    label_a: str,
    label_b: str,
    n_a: int,
    n_b: int,
    bins: int = 30,
):
    xa = _finite(df_a[col]) if col in df_a.columns else np.array([])
    xb = _finite(df_b[col]) if col in df_b.columns else np.array([])

    fig = plt.figure(figsize=(7, 5))

    if len(xa) == 0 and len(xb) == 0:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
    else:
        # shared bins for comparability
        x_all = np.concatenate([xa, xb]) if (len(xa) and len(xb)) else (xa if len(xa) else xb)
        lo, hi = float(np.min(x_all)), float(np.max(x_all))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bins_edges = np.linspace(lo, hi, bins + 1)

        if len(xa):
            plt.hist(xa, bins=bins_edges, alpha=0.45, label=f"{label_a} (n={n_a})")
        if len(xb):
            plt.hist(xb, bins=bins_edges, alpha=0.45, label=f"{label_b} (n={n_b})")

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.legend()

        # caption-like note (as requested)
        plt.figtext(
            0.5, -0.02,
            f"{label_a} n={n_a}   |   {label_b} n={n_b}",
            ha="center", fontsize=9
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_apex_resi_bars(df: pd.DataFrame, col: str, out_path: Path, title: str, n: int, top_k: int = 25):
    if col not in df.columns:
        counts = pd.Series(dtype=int)
    else:
        counts = df[col].fillna("unknown").astype(str).value_counts()

    fig = plt.figure(figsize=(10, 5))
    if counts.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
    else:
        counts = counts.head(top_k)
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f"{title}\n(n={n}, top {top_k})")
        plt.xlabel("apex resi (CA)")
        plt.ylabel("count")
        plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_apex_resi_overlay_bars(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col: str,
    out_path: Path,
    title: str,
    label_a: str,
    label_b: str,
    n_a: int,
    n_b: int,
    top_k: int = 25,
):
    ca = df_a[col].fillna("unknown").astype(str).value_counts() if col in df_a.columns else pd.Series(dtype=int)
    cb = df_b[col].fillna("unknown").astype(str).value_counts() if col in df_b.columns else pd.Series(dtype=int)

    fig = plt.figure(figsize=(12, 5))
    if ca.empty and cb.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # union top categories by combined count
    comb = (ca.add(cb, fill_value=0)).sort_values(ascending=False).head(top_k)
    cats = list(comb.index)

    va = np.array([ca.get(k, 0) for k in cats], dtype=float)
    vb = np.array([cb.get(k, 0) for k in cats], dtype=float)

    x = np.arange(len(cats))
    w = 0.42
    plt.bar(x - w/2, va, width=w, label=f"{label_a} (n={n_a})")
    plt.bar(x + w/2, vb, width=w, label=f"{label_b} (n={n_b})")

    plt.title(title)
    plt.xlabel("apex resi (CA)")
    plt.ylabel("count")
    plt.xticks(x, cats, rotation=60, ha="right")
    plt.legend()

    plt.figtext(
        0.5, -0.02,
        f"{label_a} n={n_a}   |   {label_b} n={n_b}",
        ha="center", fontsize=9
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shift_hist(df_pairs: pd.DataFrame, col_shift: str, out_path: Path, title: str, xlabel: str):
    x = _finite(df_pairs[col_shift]) if (not df_pairs.empty and col_shift in df_pairs.columns) else np.array([])
    fig = plt.figure(figsize=(7, 5))
    if len(x) == 0:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
    else:
        plt.hist(x, bins=30, alpha=0.85)
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.title(f"{title}\n(n_pairs={len(df_pairs)})")
        plt.xlabel(xlabel)
        plt.ylabel("count")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scatter(df_pairs: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str, xlabel: str, ylabel: str):
    if df_pairs.empty or xcol not in df_pairs.columns or ycol not in df_pairs.columns:
        fig = plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    x = pd.to_numeric(df_pairs[xcol], errors="coerce").astype(float).values
    y = pd.to_numeric(df_pairs[ycol], errors="coerce").astype(float).values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    fig = plt.figure(figsize=(6, 6))
    if len(x) == 0:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
    else:
        plt.scatter(x, y, s=18, alpha=0.8, edgecolors="none")
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo -= pad
        hi += pad
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.title(f"{title}\n(n_pairs={len(df_pairs)})")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_plots_unpaired(df_unbound: pd.DataFrame, df_bound: pd.DataFrame, out_dir: Path, label_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_u = len(df_unbound)
    n_b = len(df_bound)

    # Numeric: bend + apex height (alpha and beta separately)
    numeric_specs = [
        ("alpha_bend_deg",        "α CDR3 bend angle (deg)",        "bend angle (deg)"),
        ("beta_bend_deg",         "β CDR3 bend angle (deg)",        "bend angle (deg)"),
        ("alpha_apex_height_A",   "α CDR3 apex height (Å)",         "apex height (Å)"),
        ("beta_apex_height_A",    "β CDR3 apex height (Å)",         "apex height (Å)"),
    ]

    for col, ttl, xlabel in numeric_specs:
        # per-set
        plot_hist_single(df_unbound, col, out_dir / f"{label_prefix}_unbound_{col}_hist.png", f"{ttl} (unbound)", xlabel, n=n_u)
        plot_hist_single(df_bound,   col, out_dir / f"{label_prefix}_bound_{col}_hist.png",   f"{ttl} (bound)",   xlabel, n=n_b)

        # overlay
        plot_hist_overlay(
            df_unbound, df_bound, col,
            out_dir / f"{label_prefix}_overlay_{col}_hist.png",
            title=f"{ttl} — unbound vs bound",
            xlabel=xlabel,
            label_a="unbound",
            label_b="bound",
            n_a=n_u,
            n_b=n_b,
        )

    # Categorical: apex resi (alpha and beta)
    cat_specs = [
        ("alpha_apex_resi", "α CDR3 apex residue (resi)"),
        ("beta_apex_resi",  "β CDR3 apex residue (resi)"),
    ]

    for col, ttl in cat_specs:
        plot_apex_resi_bars(df_unbound, col, out_dir / f"{label_prefix}_unbound_{col}_bar.png", f"{ttl} (unbound)", n=n_u)
        plot_apex_resi_bars(df_bound,   col, out_dir / f"{label_prefix}_bound_{col}_bar.png",   f"{ttl} (bound)",   n=n_b)

        plot_apex_resi_overlay_bars(
            df_unbound, df_bound, col,
            out_dir / f"{label_prefix}_overlay_{col}_bar.png",
            title=f"{ttl} — unbound vs bound (top categories)",
            label_a="unbound",
            label_b="bound",
            n_a=n_u,
            n_b=n_b,
        )


def make_all_plots_paired(df_u: pd.DataFrame, df_b: pd.DataFrame, df_pairs: pd.DataFrame, out_dir: Path, label_prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distributions per set + overlay (same as unpaired plotting)
    make_all_plots_unpaired(df_u, df_b, out_dir=out_dir / "distributions", label_prefix=label_prefix)

    # Shifts
    shift_dir = out_dir / "shifts"
    shift_dir.mkdir(parents=True, exist_ok=True)

    # Bend angle shifts
    plot_shift_hist(
        df_pairs, "alpha_bend_shift",
        shift_dir / f"{label_prefix}_alpha_bend_shift_hist.png",
        title="α CDR3 bend shift (bound - unbound)",
        xlabel="Δ bend angle (deg)"
    )
    plot_shift_hist(
        df_pairs, "beta_bend_shift",
        shift_dir / f"{label_prefix}_beta_bend_shift_hist.png",
        title="β CDR3 bend shift (bound - unbound)",
        xlabel="Δ bend angle (deg)"
    )

    # Apex height shifts
    plot_shift_hist(
        df_pairs, "alpha_apexH_shift",
        shift_dir / f"{label_prefix}_alpha_apexH_shift_hist.png",
        title="α CDR3 apex height shift (bound - unbound)",
        xlabel="Δ apex height (Å)"
    )
    plot_shift_hist(
        df_pairs, "beta_apexH_shift",
        shift_dir / f"{label_prefix}_beta_apexH_shift_hist.png",
        title="β CDR3 apex height shift (bound - unbound)",
        xlabel="Δ apex height (Å)"
    )

    # Paired scatter (unbound vs bound)
    scatter_dir = out_dir / "paired_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    plot_paired_scatter(
        df_pairs,
        "alpha_bend_unbound", "alpha_bend_bound",
        scatter_dir / f"{label_prefix}_alpha_bend_unbound_vs_bound.png",
        title="α CDR3 bend: unbound vs bound",
        xlabel="unbound bend (deg)",
        ylabel="bound bend (deg)",
    )
    plot_paired_scatter(
        df_pairs,
        "beta_bend_unbound", "beta_bend_bound",
        scatter_dir / f"{label_prefix}_beta_bend_unbound_vs_bound.png",
        title="β CDR3 bend: unbound vs bound",
        xlabel="unbound bend (deg)",
        ylabel="bound bend (deg)",
    )
    plot_paired_scatter(
        df_pairs,
        "alpha_apexH_unbound", "alpha_apexH_bound",
        scatter_dir / f"{label_prefix}_alpha_apexH_unbound_vs_bound.png",
        title="α CDR3 apex height: unbound vs bound",
        xlabel="unbound apex height (Å)",
        ylabel="bound apex height (Å)",
    )
    plot_paired_scatter(
        df_pairs,
        "beta_apexH_unbound", "beta_apexH_bound",
        scatter_dir / f"{label_prefix}_beta_apexH_unbound_vs_bound.png",
        title="β CDR3 apex height: unbound vs bound",
        xlabel="unbound apex height (Å)",
        ylabel="bound apex height (Å)",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Unpaired directories
    unpaired_out = OUT_DIR / "unpaired_TCR3d"
    unpaired_out.mkdir(parents=True, exist_ok=True)

    df_unbound, failed_unbound = compute_set(UNBOUND_DIR, out_dir=unpaired_out, label="unbound_TCR3d", vis=VIS_PER_TCR)
    df_bound,   failed_bound   = compute_set(BOUND_DIR,   out_dir=unpaired_out, label="bound_TCR3d",   vis=VIS_PER_TCR)

    plots_unpaired_dir = unpaired_out / "plots"
    make_all_plots_unpaired(df_unbound, df_bound, out_dir=plots_unpaired_dir, label_prefix="TCR3d_unpaired")

    # 2) Paired benchmark directories
    paired_out = OUT_DIR / "paired_benchmark"
    paired_out.mkdir(parents=True, exist_ok=True)

    df_u, df_b, df_pairs = compute_paired_set(
        unbound_dir=PAIRED_UNBOUND_DIR,
        bound_dir=PAIRED_BOUND_DIR,
        out_dir=paired_out,
        label="benchmark",
        vis=VIS_PER_TCR,
    )

    make_all_plots_paired(df_u, df_b, df_pairs, out_dir=paired_out / "plots", label_prefix="benchmark_paired")

    # Print quick summary
    print("\n=== DONE ===")
    print(f"Unpaired: unbound n={len(df_unbound)} (failed {len(failed_unbound)}), bound n={len(df_bound)} (failed {len(failed_bound)})")
    print(f"Paired benchmark: unbound n={len(df_u)}, bound n={len(df_b)}, pairs n={len(df_pairs)}")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # User inputs
    # ---------------------------------------------------------------------
    UNBOUND_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/ab_chain")
    BOUND_DIR   = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes")

    PAIRED_UNBOUND_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_unbound_tcr_imgt")
    PAIRED_BOUND_DIR   = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_bound_tcr_imgt")

    OUT_DIR = Path("/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/cdr_bend_calc/plots")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # If True, will run PyMOL vis (PSE/PNG) for each PDB (can be slow).
    VIS_PER_TCR = False
    main()
