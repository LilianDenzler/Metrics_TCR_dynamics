import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CUSTOM IMPORTS ---
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
from TCR_TOOLS.classes.tcr import TCR  # <- make sure this exists

# --- GLOBAL CONFIGURATION ---
TCR3d_TCR_Complexes = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes"
METADATA_FILE = "/mnt/larry/lilian/DATA/TCR3d_datasets/tcr_complexes_data.tsv"
OUTPUT_DIR = "/mnt/larry/lilian/DATA/TCR3d_datasets/angle_plots"  # <--- change if you want

ANGLE_COLUMNS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]  # Use all calculated angles

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)


def load_and_merge_data():
    """
    1. Runs the TCR angle calculation loop.
    2. Loads external metadata from TSV file.
    3. Merges the geometry data with the metadata.
    """
    all_angles_list = []

    print("--- Starting TCR Angle Calculation ---")
    if not os.path.isdir(TCR3d_TCR_Complexes):
        print(f"Error: TCR complexes directory not found at {TCR3d_TCR_Complexes}")
        return None

    # ---- 1. Angle Calculation Loop ----
    try:
        for pdb_file in sorted(os.listdir(TCR3d_TCR_Complexes)):
            try:
                if not pdb_file.endswith(".pdb"):
                    continue

                pdb_path = os.path.join(TCR3d_TCR_Complexes, pdb_file)
                tcr_name = pdb_file[:-4]  # strip ".pdb"

                # Instantiate TCR
                tcr = TCR(
                    input_pdb=pdb_path,
                    traj_path=None,
                    contact_cutoff=5.0,
                    min_contacts=50,
                    legacy_anarci=True,
                )
                if len(tcr.pairs) == 0:
                    print(f"Warning: No TCR pairs found in {tcr_name}. Skipping.")
                    continue
                if len(tcr.pairs) > 1:
                    print(f"Warning: {str(len(tcr.pairs))} TCR pairs found in {tcr_name}. Using the first pair only.")
                    input("Press Enter to continue...")
                for pair in tcr.pairs:
                    angle_results = pair.calc_angles()

                    if "pdb_name" in angle_results.columns:
                        # use TCR name instead of pdb_name
                        angle_results["tcr_name"] = tcr_name
                        angle_results = angle_results.drop(columns=["pdb_name"])
                        all_angles_list.append(angle_results)
                    else:
                        print(
                            f"Warning: Angle results for {tcr_name} did not contain 'pdb_name' column. Skipping."
                        )
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
                continue

        if not all_angles_list:
            print("FATAL: No angle results generated at all.")
            return None

        geometry_df = pd.concat(all_angles_list, ignore_index=True)

    except Exception as e:
        print(f"FATAL ERROR during TCR processing or concatenation: {e}")
        return None

    if geometry_df.empty:
        print("FATAL: geometry_df ended up empty. Check input files and TCR class.")
        return None

    # ---- 2. Load Metadata ----
    if not os.path.exists(METADATA_FILE):
        print(f"FATAL: Metadata file not found at {METADATA_FILE}")
        return None

    try:
        metadata_df = pd.read_csv(METADATA_FILE, sep="\t")
    except Exception as e:
        print(f"FATAL ERROR reading TSV file: {e}")
        return None

    # ---- 3. Merge Geometry and Metadata ----
    merged_df = pd.merge(
        geometry_df,
        metadata_df,
        left_on="tcr_name",
        right_on="PDB_ID",
        how="inner",
    )

    merged_df.rename(
        columns={
            "TCR_complex": "MHC_Class",
            "TCR_organism": "Organism",
        },
        inplace=True,
    )

    print("\n--- Data Merged ---")
    print(f"Total entries for analysis: {len(merged_df)}")

    return merged_df


def plot_angle_distributions(
    df,
    title: str,
    out_path: Path,
    hue_col: str | None = None,
    sub_filter_col: str | None = None,
    sub_filter_val: str | None = None,
):
    """
    Generate multi-panel KDE plots of angle distributions and save to file.
    """

    plot_df = df.copy()

    # Optional sub-filter, e.g. MHC_Class == 'CLASSI'
    if sub_filter_col and sub_filter_val:
        plot_df = plot_df[plot_df[sub_filter_col] == sub_filter_val]
        if plot_df.empty:
            print(
                f"Skipping plot '{out_path.name}': "
                f"No data for {sub_filter_col} = {sub_filter_val}."
            )
            return

    n_cols = 3
    n_rows = (len(ANGLE_COLUMNS) + n_cols - 1) // n_cols  # ceiling

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(18, 5 * n_rows), constrained_layout=True
    )
    axes = axes.flatten()

    for i, angle in enumerate(ANGLE_COLUMNS):
        if angle not in plot_df.columns:
            print(f"Warning: Angle column '{angle}' missing in the DataFrame.")
            fig.delaxes(axes[i])
            continue

        ax = axes[i]
        if hue_col and hue_col in plot_df.columns:
            sns.kdeplot(
                data=plot_df,
                x=angle,
                hue=hue_col,
                fill=True,
                ax=ax,
                linewidth=2.0,
                alpha=0.7,
            )
            # Try to put a legend; ignore issues when too many categories, etc.
            try:
                ax.legend(title=hue_col)
            except Exception:
                ax.get_legend().remove()
        else:
            sns.kdeplot(
                data=plot_df,
                x=angle,
                fill=True,
                ax=ax,
                color="indigo",
                linewidth=2.0,
                alpha=0.7,
            )

        ax.set_title(f"{angle} Distribution", fontsize=14)
        ax.set_xlabel("Angle Value (Degrees)", fontsize=12)

    # Remove any remaining unused axes
    for j in range(len(ANGLE_COLUMNS), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mhc_allele_comparison(df, out_dir: Path):
    """
    Box+swarm plot of BA angle vs MHC_allele.
    """

    if "MHC_allele" not in df.columns:
        print(
            f"Skipping MHC Allele Comparison Plot '{out_path.name}': 'MHC_allele' column missing."
        )
        return
    for angle in ANGLE_COLUMNS:
        out_path = out_dir/ f"{angle}_by_MHC_allele.png"

        if angle not in df.columns:
            print(f"Warning: Angle column '{angle}' missing in the DataFrame.")

        fig, ax = plt.subplots(figsize=(12, 7))

        sns.boxplot(
            data=df,
            x="MHC_allele",
            y=angle,
            showfliers=False,
            ax=ax,
        )

        # Only overlay swarm if the number of categories isn't insane
        n_alleles = df["MHC_allele"].nunique()
        if n_alleles <= 30:
            sns.swarmplot(
                data=df,
                x="MHC_allele",
                y=angle,
                color=".25",
                size=4,
                ax=ax,
            )

        ax.set_title(f"TCR {angle} Distribution by MHC Allele", fontsize=16)
        ax.set_xlabel("MHC Allele Group", fontsize=12)
        ax.set_ylabel(f"{angle} Angle Value (Degrees)", fontsize=12)
        ax.tick_params(axis="x", rotation=60)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    df_merged = load_and_merge_data()

    if df_merged is None or df_merged.empty:
        print("Analysis terminated: Cannot proceed without merged data.")
        sys.exit(1)

    print("\n--- Generating Plots (saved to disk) ---")

    # 1. All TCRs
    plot_angle_distributions(
        df_merged,
        title="All TCR Angle Distributions (Global View)",
        out_path=output_root / "angles_all_TCRs.png",
    )

    # 2. Class I only
    df_class_i = df_merged[df_merged["MHC_Class"] == "CLASSI"]
    plot_angle_distributions(
        df_class_i,
        title="Angle Distributions for MHC Class I Only",
        out_path=output_root / "angles_class_I.png",
    )

    # 3. Class II only
    df_class_ii = df_merged[df_merged["MHC_Class"] == "CLASSII"]
    plot_angle_distributions(
        df_class_ii,
        title="Angle Distributions for MHC Class II Only",
        out_path=output_root / "angles_class_II.png",
    )

    # 4. Human only
    df_human = df_merged[df_merged["Organism"] == "Human"]
    plot_angle_distributions(
        df_human,
        title="Angle Distributions for Human TCRs Only",
        out_path=output_root / "angles_human.png",
    )

    # 5. Mouse only
    df_mouse = df_merged[df_merged["Organism"] == "Mouse"]
    plot_angle_distributions(
        df_mouse,
        title="Angle Distributions for Mouse TCRs Only",
        out_path=output_root / "angles_mouse.png",
    )

    # 6. Human + Class I
    df_human_class_i = df_human[df_human["MHC_Class"] == "CLASSI"]
    plot_angle_distributions(
        df_human_class_i,
        title="Angle Distributions: Human & MHC Class I",
        out_path=output_root / "angles_human_class_I.png",
    )

    # 7. Human + Class II
    df_human_class_ii = df_human[df_human["MHC_Class"] == "CLASSII"]
    plot_angle_distributions(
        df_human_class_ii,
        title="Angle Distributions: Human & MHC Class II",
        out_path=output_root / "angles_human_class_II.png",
    )

    # 8. MHC allele vs BA
    plot_mhc_allele_comparison(
        df_merged,
        out_dir=output_root,
    )
    # 8. MHC allele vs BA
    os.makedirs(output_root/"mhc_allele_comparisons", exist_ok=True)
    os.makedirs(output_root/"mhc_allele_comparisons"/"human_class_I", exist_ok=True)
    os.makedirs(output_root/"mhc_allele_comparisons"/"human_class_II", exist_ok=True)
    plot_mhc_allele_comparison(
        df_human_class_i,
        out_dir=output_root/"mhc_allele_comparisons"/"human_class_I",
    )
    plot_mhc_allele_comparison(
        df_human_class_ii,
        out_dir=output_root/"mhc_allele_comparisons"/"human_class_II",
    )

    print("\nAnalysis complete. All plots saved to:")
    print(output_root)
