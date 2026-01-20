#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# USER-CONTROLLED PERCENTILE SETTINGS
# =====================================================
P_LOW  = 2.0
P_HIGH = 98.0
P_TAG  = f"p{int(P_LOW)}_p{int(P_HIGH)}"

# =====================================================
# Metrics
# =====================================================
FIELDS = [
    "total_score",
    "interface_dG_tcr_pmhc",
    "dG_separated_tcr_pmhc",
    "delta_sasa_tcr_pmhc",
    "interface_dG_pep_mhc",
    "dG_separated_pep_mhc",
    "delta_sasa_pep_mhc",
    "dE_binding_tcr_pmhc",
]

FIELD_NAMES = {
    "total_score": "Total Rosetta Energy",
    "interface_dG_tcr_pmhc": "TCR–pMHC Interface ΔG",
    "dG_separated_tcr_pmhc": "TCR–pMHC Separated Interface Energy",
    "delta_sasa_tcr_pmhc": "TCR–pMHC Interface ΔSASA",
    "interface_dG_pep_mhc": "Peptide–MHC Interface ΔG",
    "dG_separated_pep_mhc": "Peptide–MHC Separated Interface Energy",
    "delta_sasa_pep_mhc": "Peptide–MHC Interface ΔSASA",
    "dE_binding_tcr_pmhc": "TCR–pMHC Binding Energy (ΔE_bind)",
}

# =====================================================
# Helpers
# =====================================================
def percentile_bounds(series, low=P_LOW, high=P_HIGH):
    return series.quantile(low / 100.0), series.quantile(high / 100.0)


def detect_percentile_excluded(df):
    excluded_mask = pd.Series(False, index=df.index)
    per_metric_counts = {}

    for c in FIELDS:
        lo, hi = percentile_bounds(df[c])
        mask = (df[c] < lo) | (df[c] > hi)
        per_metric_counts[c] = int(mask.sum())
        excluded_mask |= mask

    return excluded_mask, per_metric_counts


def _format_method_label(method):
    if method == "none":
        return ""
    if method == "rosetta_sidechain":
        return "after Rosetta sidechain packing"
    if method == "openmm":
        return "after OpenMM minimization"
    return method


# =====================================================
# Plotting
# =====================================================
def plot_distributions(df, out_png, method, antigen_name, trim=False):
    method_label = _format_method_label(method)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for ax, c in zip(axes, FIELDS):
        data = df[c]

        suffix = ""
        if trim:
            lo, hi = percentile_bounds(data)
            data = data[(data >= lo) & (data <= hi)]
            suffix = f" ({P_LOW:.0f}–{P_HIGH:.0f} percentile)"

        ax.hist(data.values, bins=50)
        ax.set_title(FIELD_NAMES[c])
        ax.set_ylabel("Number of structures")
        ax.tick_params(axis="x", labelrotation=45)

    title = f"Rosetta Energy Distributions {method_label}{suffix}"
    plt.suptitle(title, y=0.98, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.91, antigen_name, ha="center", fontsize=14, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================
# Public API
# =====================================================
def plot_rosetta_csv(csv_path, plots_dir, method, antigen_name):
    """
    Generate full + percentile-trimmed plots and exclusion reports
    for a single Rosetta CSV.
    """
    os.makedirs(plots_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"[WARN] CSV has no columns, skipping plot: {csv_path}")
        return

    if "TCR_name" not in df.columns:
        raise ValueError("Expected column 'TCR_name' not found")

    for c in FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FIELDS)

    base = os.path.join(
        plots_dir,
        os.path.splitext(os.path.basename(csv_path))[0]
    )

    # 1) Full distributions
    plot_distributions(df, base + ".png", method, antigen_name, trim=False)

    # 2) Trimmed distributions
    plot_distributions(df, base + f"_{P_TAG}.png", method, antigen_name, trim=True)

    # 3) Exclusion reporting
    excluded_mask, per_metric_counts = detect_percentile_excluded(df)
    df_excluded = df.loc[excluded_mask]

    with open(base + f"_excluded_pdbs_{P_TAG}.txt", "w") as f:
        for name in sorted(df_excluded["TCR_name"].astype(str).unique()):
            f.write(name + "\n")

    with open(base + f"_exclusion_summary_{P_TAG}.txt", "w") as f:
        f.write(f"Excluded {len(df_excluded)} / {len(df)} structures\n\n")
        for c in FIELDS:
            f.write(f"{c:35s}: {per_metric_counts[c]}\n")

    print(f"[PLOT] Generated plots for {os.path.basename(csv_path)}")
