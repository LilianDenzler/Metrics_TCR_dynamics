import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# --- Configuration ----------------------------------------------------

REGION_MAP = {
    ('A_CDR1',): 'A_CDR1',
    ('A_CDR2',): 'A_CDR2',
    ('A_CDR3',): 'A_CDR3',
    ('B_CDR1',): 'B_CDR1',
    ('B_CDR2',): 'B_CDR2',
    ('B_CDR3',): 'B_CDR3',
    ('A_variable', 'A_CDR1'): 'A_var_A_CDR1',
    ('A_variable', 'A_CDR2'): 'A_var_A_CDR2',
    ('A_variable', 'A_CDR3'): 'A_var_A_CDR3',
    ('B_variable', 'B_CDR1'): 'B_var_B_CDR1',
    ('B_variable', 'B_CDR2'): 'B_var_B_CDR2',
    ('B_variable', 'B_CDR3'): 'B_var_B_CDR3',
    ('A_variable', 'A_CDR1A_CDR2A_CDR3'): 'A_var_all_ACDRs',
    ('B_variable', 'B_CDR1B_CDR2B_CDR3'): 'B_var_all_BCDRs',
    ('A_variable', 'A_CDR1A_CDR2A_CDR3B_CDR1B_CDR2B_CDR3'): 'A_var_allCDRs',
    ('B_variable', 'A_CDR1A_CDR2A_CDR3B_CDR1B_CDR2B_CDR3'): 'B_var_allCDRs',
}

# Regex not strictly needed now, but kept for robustness if file format changes
JSD_REGEX = re.compile(r"Jensen[–-]Shannon divergence\s*:\s*([\d\.]+)")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------

def parse_jsd_file(filepath: str):
    """
    Reads a JSD_*.txt file and extracts the JSD value.

    Expected formats like:
        "Jensen–Shannon divergence : 0.1234"
        "Jensen-Shannon divergence : 0.1234"
    or anything with a single colon and value after it.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read().strip()

        # Try regex first (handles odd spacing / dash types)
        m = JSD_REGEX.search(content)
        if m:
            return float(m.group(1))

        # Fallback: split on ":" and take RHS
        if ":" in content:
            value = content.split(":")[-1].strip()
            return float(value)

        logging.error(f"Could not parse JSD value from {filepath}: content={content}")
        return None

    except Exception as e:
        logging.error(f"Could not read file {filepath}: {e}")
        return None


def get_path_info(filepath: str, base_dir: str, region_map: dict):
    """
    Parses a file path to get the TCR and Region name.

    Handles both:
      .../<MODEL>/<TCR>/<REGION>/.../<METHOD>/JSD_*.txt
      .../<MODEL>/<TCR>/<REGION>/.../<METHOD>/standardised_by_RMSD/JSD_*.txt

    Example non-standardised:
      /mnt/.../assessments2/all_ca_pca/Alphaflow_with_linker/1KGC/A_CDR1/ca_pca/JSD_histogram.txt

    Example standardised:
      /mnt/.../assessments2/all_ca_pca/Alphaflow_with_linker/1KGC/A_CDR1/ca_pca/standardised_by_RMSD/JSD_histogram.txt
    """
    try:
        p = Path(filepath)
        base_p = Path(base_dir)

        relative_parts = p.relative_to(base_p).parts
        # Example (raw):      ['1KGC', 'A_CDR1', 'ca_pca', 'JSD_histogram.txt']
        # Example (std):      ['1KGC', 'A_CDR1', 'ca_pca',
        #                      'standardised_by_RMSD', 'JSD_histogram.txt']

        if len(relative_parts) < 3:
            return None

        # TCR is always first piece; strip any extra suffix after underscore
        tcr = relative_parts[0].split("_")[0]

        # Determine index of method directory:
        # - right before 'standardised_by_RMSD' if present
        # - or second-to-last otherwise
        if "standardised_by_RMSD" in relative_parts:
            idx_std = relative_parts.index("standardised_by_RMSD")
            method_index = idx_std - 1
        else:
            method_index = len(relative_parts) - 2

        if method_index <= 1:
            return None

        # Region parts are everything between TCR and METHOD
        region_parts = tuple(relative_parts[1:method_index])

        column_name = region_map.get(region_parts)
        if column_name is None:
            # Not one of the regions we care about
            return None

        return {"tcr": tcr, "region": column_name}

    except Exception as e:
        logging.error(f"Error parsing path {filepath}: {e}")
        return None


def build_jsd_dataframe(
    jsd_hist_files,
    jsd_kde_files,
    base_dir: str,
    label: str,
):
    """
    Collects JSD data for a given variant (raw or RMSD-standardised).

    Returns a DataFrame with columns:
        ['tcr', 'region', 'jsd', 'jsd_kde'].
    """
    logging.info(f"Building JSD dataframe for variant: {label}")

    hist_entries = {}
    kde_entries = {}

    # Histogram-based JSD
    for fp in jsd_hist_files:
        info = get_path_info(fp, base_dir, REGION_MAP)
        if not info:
            continue
        jsd_val = parse_jsd_file(fp)
        if jsd_val is None:
            continue
        key = (info["tcr"], info["region"])
        hist_entries[key] = jsd_val

    # KDE-based JSD
    for fp in jsd_kde_files:
        info = get_path_info(fp, base_dir, REGION_MAP)
        if not info:
            continue
        jsd_val = parse_jsd_file(fp)
        if jsd_val is None:
            continue
        key = (info["tcr"], info["region"])
        kde_entries[key] = jsd_val

    all_keys = sorted(hist_entries.keys())
    all_data = []
    for key in all_keys:
        tcr, region = key
        all_data.append(
            {
                "tcr": tcr,
                "region": region,
                "jsd": hist_entries[key],
                "jsd_kde": kde_entries.get(key, np.nan),
            }
        )

    df = pd.DataFrame(all_data)
    if df.empty:
        logging.warning(f"No data collected for variant {label}.")
    return df


# ---------------------------------------------------------------------
# Plotting / table generation for one variant
# ---------------------------------------------------------------------

def process_variant(model: str, output_dir: str, df: pd.DataFrame, variant_label: str):
    """
    Runs the pivoting, stats, bar plots, and heatmaps for a given variant
    (raw or RMSD-standardised).
    """
    logging.info(f"Creating data tables for variant: {variant_label}")

    # Create tables for both jsd types
    for jsd_type in ["jsd", "jsd_kde"]:
        if jsd_type not in df.columns:
            logging.warning(f"{jsd_type} not in dataframe for variant {variant_label}, skipping.")
            continue

        logging.info(f"Processing JSD type: {jsd_type} [{variant_label}]")

        # --- 1. Pivot table TCR × Region ---
        try:
            pivot_df = df.pivot_table(index="tcr", columns="region", values=jsd_type)
        except Exception as e:
            logging.error(f"Error creating pivot table. Check for duplicate entries: {e}")
            duplicates = df[df.duplicated(subset=["tcr", "region"], keep=False)]
            logging.error(f"Duplicates:\n{duplicates}")
            return

        # Keep all regions in sorted order found in the data
        found_regions = sorted(df["region"].unique())
        pivot_df = pivot_df[found_regions]

        region_order = [
            "A_CDR1",
            "A_CDR2",
            "A_CDR3",
            "B_CDR1",
            "B_CDR2",
            "B_CDR3",
            "A_var_A_CDR1",
            "A_var_A_CDR2",
            "A_var_A_CDR3",
            "B_var_B_CDR1",
            "B_var_B_CDR2",
            "B_var_B_CDR3",
            "A_var_all_ACDRs",
            "B_var_all_BCDRs",
            "A_var_allCDRs",
            "B_var_allCDRs",
        ]
        # Restrict to what we actually have
        region_order = [r for r in region_order if r in pivot_df.columns]
        pivot_df = pivot_df.reindex(columns=region_order)

        print("\n" + "=" * 80)
        print(f"--- Table 1: JSD Values [{variant_label}, {jsd_type}] ---")
        print(pivot_df.to_markdown(floatfmt=".4f"))
        print("=" * 80)

        csv_filename = os.path.join(
            output_dir, f"{model}_{variant_label}_{jsd_type}_values_table.csv"
        )
        pivot_df.to_csv(csv_filename)

        # --- 2. Statistics table ---
        logging.info("Calculating statistics...")
        desc = pivot_df.describe()
        stats_df = desc.loc[["mean", "std", "min", "max"]]

        print("\n" + "=" * 80)
        print(
            f"--- Table 2: JSD Statistics (Mean / Std / Min / Max) "
            f"[{variant_label}, {jsd_type}] ---"
        )
        print(stats_df.to_markdown(floatfmt=".4f"))
        print("=" * 80)

        stats_csv_filename = os.path.join(
            output_dir, f"{model}_{variant_label}_{jsd_type}_statistics_table.csv"
        )
        stats_df.to_csv(stats_csv_filename)

        # --- 3. Barplot + heatmap ---
        try:
            mean_jsd_scores = stats_df.loc["mean"]
            std_jsd_scores = stats_df.loc["std"]
            min_jsd_scores = stats_df.loc["min"]
            max_jsd_scores = stats_df.loc["max"]

            # --- Barplot with errorbars and min/max ---
            fig_bar, ax = plt.subplots(figsize=(16, 8))
            x = np.arange(len(mean_jsd_scores))

            ax.bar(
                x,
                mean_jsd_scores.values,
                color="skyblue",
                edgecolor="black",
                zorder=3,
            )

            # ±1 std errorbars
            ax.errorbar(
                x,
                mean_jsd_scores.values,
                yerr=std_jsd_scores.values,
                fmt="none",
                ecolor="black",
                elinewidth=1.5,
                capsize=5,
                zorder=4,
                label="±1 std",
            )

            # Min–max vertical lines
            ax.vlines(
                x,
                min_jsd_scores.values,
                max_jsd_scores.values,
                colors="red",
                linestyles="dashed",
                linewidth=1.5,
                zorder=4,
                label="min–max",
            )

            ax.set_title(
                f'{model} Average Jensen-Shannon Divergence {jsd_type} '
                f'per Region [{variant_label}]',
                fontsize=18,
                fontweight="bold",
            )
            ax.set_ylabel(f"Mean {jsd_type}", fontsize=14)
            ax.set_xlabel("Region", fontsize=14)

            ax.set_xticks(x)
            ax.set_xticklabels(
                mean_jsd_scores.index, rotation=45, ha="right", fontsize=10
            )
            ax.tick_params(axis="y", labelsize=10)

            ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

            # Clamp to [0,1]
            ax.set_ylim(0, 1)

            # Annotate values on bars
            for i, (region, v) in enumerate(mean_jsd_scores.items()):
                y_text = min(v + 0.02, 0.98)
                ax.text(
                    i,
                    y_text,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.legend()

            fig_bar.tight_layout()
            plot_filename_mean = (
                f"{model}_{variant_label}_{jsd_type}_mean_per_region_plot.png"
            )
            fig_bar.savefig(os.path.join(output_dir, plot_filename_mean), dpi=150)
            plt.close(fig_bar)

            logging.info(
                f"Mean bar plot saved to: {os.path.join(output_dir, plot_filename_mean)}"
            )

            # --- Heatmap ---
            fig_hm, ax_hm = plt.subplots(
                figsize=(12, max(4, 0.4 * len(pivot_df)))
            )
            values = pivot_df.values

            vmin = np.nanmin(values)
            vmax = 1.0
            cmap = plt.colormaps.get("Reds_r")
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            im = ax_hm.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )

            ax_hm.set_xticks(np.arange(len(pivot_df.columns)))
            ax_hm.set_xticklabels(
                pivot_df.columns, rotation=45, ha="right"
            )
            ax_hm.set_yticks(np.arange(len(pivot_df.index)))
            ax_hm.set_yticklabels(pivot_df.index)

            ax_hm.set_xlabel("Region")
            ax_hm.set_ylabel("TCR")
            ax_hm.set_title(
                f'{model} {jsd_type} heatmap (TCR × Region) [{variant_label}]'
            )

            cbar = fig_hm.colorbar(im, ax=ax_hm)
            cbar.set_label(jsd_type)

            fig_hm.tight_layout()
            heatmap_filename = (
                f"{model}_{variant_label}_{jsd_type}_heatmap.png"
            )
            fig_hm.savefig(os.path.join(output_dir, heatmap_filename), dpi=150)
            plt.close(fig_hm)

            logging.info(
                f"Heatmap saved to: {os.path.join(output_dir, heatmap_filename)}"
            )

        except Exception as e:
            logging.error(
                f"Could not generate mean {jsd_type} bar/heatmap plots "
                f"for variant {variant_label}: {e}"
            )


# ---------------------------------------------------------------------
# Main per-model driver
# ---------------------------------------------------------------------

def main_for_model(base_dir: str, model: str, output_dir: str):
    """
    Main function for a single model:
      - Scan base_dir for JSD_* files
      - Build raw + RMSD-standardised DataFrames
      - Produce tables + plots
    """
    logging.info(f"Starting analysis for model={model} in: {base_dir}")

    glob_path_KDE = os.path.join(base_dir, "**", "JSD_KDE.txt")
    glob_path_hist = os.path.join(base_dir, "**", "JSD_histogram.txt")

    logging.info("Scanning for files...")
    jsd_kde_all = glob.glob(glob_path_KDE, recursive=True)
    jsd_hist_all = glob.glob(glob_path_hist, recursive=True)

    if not jsd_hist_all:
        logging.error(f"No 'JSD_histogram.txt' files found in {base_dir}. Please check the path.")
        return

    # Split into raw vs RMSD-standardised
    jsd_kde_raw = [f for f in jsd_kde_all if "standardised_by_RMSD" not in f]
    jsd_kde_std = [f for f in jsd_kde_all if "standardised_by_RMSD" in f]

    jsd_hist_raw = [f for f in jsd_hist_all if "standardised_by_RMSD" not in f]
    jsd_hist_std = [f for f in jsd_hist_all if "standardised_by_RMSD" in f]

    # --- RAW variant ---
    df_raw = build_jsd_dataframe(
        jsd_hist_files=jsd_hist_raw,
        jsd_kde_files=jsd_kde_raw,
        base_dir=base_dir,
        label="raw",
    )
    if not df_raw.empty:
        process_variant(model, output_dir, df_raw, variant_label="raw")
    else:
        logging.warning("No valid RAW data found; skipping raw variant.")

    # --- RMSD-standardised variant ---
    df_std = build_jsd_dataframe(
        jsd_hist_files=jsd_hist_std,
        jsd_kde_files=jsd_kde_std,
        base_dir=base_dir,
        label="standardised_by_RMSD",
    )
    if not df_std.empty:
        process_variant(model, output_dir, df_std, variant_label="standardised_by_RMSD")
    else:
        logging.warning("No valid RMSD-standardised data found; skipping that variant.")


# ---------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------
def run_make_tables(base_dir):
    model = os.path.basename(base_dir)
    output_dir = os.path.join(base_dir, "VISUALIZATIONS_AND_TABLES")
    os.makedirs(output_dir, exist_ok=True)

    main_for_model(base_dir, model, output_dir)
if __name__ == "__main__":
    # New base dir you mentioned:
    #   /mnt/larry/lilian/DATA/DIG_VARIATION_OUTPUTS/assessments2/all_ca_pca
    all_dir = "/mnt/larry/lilian/DATA/DIG_VARIATION_OUTPUTS/dig_variations_combinedMDGT/default"
    for model_folder in os.listdir(all_dir):
        if model_folder=="dig_with_init_cdr_mask_trA0.2_rotA0.2_trB0.1_rotB0.1":
            pass
        else:
            continue
        assess_dir = os.path.join(all_dir, model_folder)
        if not os.path.isdir(assess_dir):
            continue

        run_make_tables(assess_dir)

