#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# USER-CONTROLLED PERCENTILE SETTINGS (CHANGE ONLY HERE)
# =====================================================
P_LOW  = 2.0
P_HIGH = 98.0
P_TAG  = f"p{int(P_LOW)}_p{int(P_HIGH)}"  # used in filenames

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
# Percentile helpers
# =====================================================
def percentile_bounds(series, low=P_LOW, high=P_HIGH):
    lo = series.quantile(low / 100.0)
    hi = series.quantile(high / 100.0)
    return lo, hi


def detect_percentile_excluded(df, low=P_LOW, high=P_HIGH):
    """
    Returns:
        excluded_mask       : True if excluded in ANY metric
        per_metric_counts   : dict[field -> count]
    """
    excluded_mask = pd.Series(False, index=df.index)
    per_metric_counts = {}

    for c in FIELDS:
        lo, hi = percentile_bounds(df[c], low, high)
        mask = (df[c] < lo) | (df[c] > hi)
        per_metric_counts[c] = int(mask.sum())
        excluded_mask |= mask

    return excluded_mask, per_metric_counts


# =====================================================
# Plotting
# =====================================================
def plot_distributions(df, out_png, trim=False):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()

    for ax, c in zip(axes, FIELDS):
        data = df[c]

        if trim:
            lo, hi = percentile_bounds(data)
            data = data[(data >= lo) & (data <= hi)]
            suffix = f" ({P_LOW:.0f}–{P_HIGH:.0f} percentile)"
        else:
            suffix = ""

        ax.hist(data.values, bins=50)
        ax.set_title(f"{FIELD_NAMES[c]}{suffix}")
        ax.set_ylabel("Number of structures")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================
# Main per-CSV logic
# =====================================================
def main(csv_path: str, out_base: str):
    df = pd.read_csv(csv_path)

    if "TCR_name" not in df.columns:
        raise ValueError("Expected column 'TCR_name' not found in CSV")

    # numeric + drop NaNs
    for c in FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FIELDS)

    n_total = len(df)

    # --------------------------------------------------
    # 1) Original plots
    # --------------------------------------------------
    plot_distributions(df, out_base + ".png", trim=False)

    # --------------------------------------------------
    # 2) Trimmed plots
    # --------------------------------------------------
    plot_distributions(df, out_base + f"_{P_TAG}.png", trim=True)

    # --------------------------------------------------
    # 3) Detect excluded structures (reporting only)
    # --------------------------------------------------
    excluded_mask, per_metric_counts = detect_percentile_excluded(df)
    df_excluded = df.loc[excluded_mask]
    df_kept = df.loc[~excluded_mask]

    n_excluded = len(df_excluded)
    n_kept = len(df_kept)

    # --------------------------------------------------
    # 4) Write excluded PDB list
    # --------------------------------------------------
    excluded_list_path = out_base + f"_excluded_pdbs_{P_TAG}.txt"
    with open(excluded_list_path, "w") as f:
        f.write(f"Excluded structures ({P_LOW}–{P_HIGH} percentile trimming)\n")
        f.write(f"Source CSV: {os.path.basename(csv_path)}\n")
        f.write("=" * 60 + "\n\n")
        for name in sorted(df_excluded["TCR_name"].astype(str).unique()):
            f.write(name + "\n")

    # --------------------------------------------------
    # 5) Write exclusion summary
    # --------------------------------------------------
    summary_path = out_base + f"_exclusion_summary_{P_TAG}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Percentile trimming summary ({P_LOW}–{P_HIGH}%)\n")
        f.write(f"Source CSV: {os.path.basename(csv_path)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total structures      : {n_total}\n")
        f.write(f"Structures kept       : {n_kept}\n")
        f.write(f"Structures excluded   : {n_excluded}\n\n")

        f.write("Excluded per metric:\n")
        for c in FIELDS:
            f.write(f"  {c:35s}: {per_metric_counts[c]}\n")

    print(
        f"[INFO] {os.path.basename(csv_path)} | "
        f"excluded {n_excluded}/{n_total} structures "
        f"({P_LOW}–{P_HIGH}%)"
    )


# =====================================================
# Batch execution
# =====================================================
if __name__ == "__main__":
    csv_base_path = (
        "/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/"
        "process_TRAIT/omics_A0301_KLGGALQAK_tfold_Rosetta_Results/"
    )
    png_base_path = (
        "/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/"
        "process_TRAIT/plots/omics_tfold_pyrosetta/A0301_KLGGALQAK"
    )
    os.makedirs(png_base_path, exist_ok=True)

    for csv in os.listdir(csv_base_path):
        if not csv.endswith(".csv"):
            continue

        pos_neg = csv.split("_")[-1].replace(".csv", "")
        method = csv.split("_")[-2]

        base_name = f"random_{pos_neg}_{method}"
        csv_path = os.path.join(csv_base_path, csv)
        out_base = os.path.join(png_base_path, base_name)

        main(csv_path, out_base)
