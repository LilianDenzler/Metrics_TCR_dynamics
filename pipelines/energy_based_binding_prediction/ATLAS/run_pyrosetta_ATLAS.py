import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from pipelines.energy_based_binding_prediction.run_pyrosetta import run_folder


def plot_results(merged_df: pd.DataFrame, vis_folder="plots"):
    os.makedirs(vis_folder, exist_ok=True)
    df = merged_df.copy()

    # ------------------------------------------------------------------
    # 0) Derive MHC_type from MHCname
    # ------------------------------------------------------------------
    # Make sure MHCname is a string, handle NaN safely
    if "MHCname" in df.columns:
        df["MHCname"] = df["MHCname"].fillna("").astype(str)

        human_mhc_1_prefixes = ('HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F', 'HLA-G')
        human_mhc_2_prefixes = ('HLA-DR', 'HLA-DQ', 'HLA-DP')
        mouse_mhc_1_prefixes = ('H-2K', 'H-2D', 'H-2L')
        mouse_mhc_2_prefixes = ('I-A', 'I-E')

        conditions = [
            df["MHCname"].str.startswith(human_mhc_1_prefixes),
            df["MHCname"].str.startswith(human_mhc_2_prefixes),
            df["MHCname"].str.startswith(mouse_mhc_1_prefixes),
            df["MHCname"].str.startswith(mouse_mhc_2_prefixes),
        ]
        choices = [
            "Human MHC I",
            "Human MHC II",
            "Mouse MHC I",
            "Mouse MHC II",
        ]

        df["MHC_type"] = np.select(conditions, choices, default="Other/Unknown")
    else:
        df["MHC_type"] = "Unknown"

    # ------------------------------------------------------------------
    # 1) Ensure numeric dtypes ONLY for known numeric columns
    # ------------------------------------------------------------------
    numeric_cols = [
        #"Kd_microM",
        #"Kon_per_M_per_s",
        #"Koff_per_S",
        "DeltaG_kcal_per_mol",
        "interface_dG_tcr_pmhc",
        "dG_separated_tcr_pmhc",
        "delta_sasa_tcr_pmhc",
        "interface_dG_pep_mhc",
        "dG_separated_pep_mhc",
        "delta_sasa_pep_mhc",
        "dE_binding_tcr_pmhc",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")



    # ------------------------------------------------------------------
    # Helper function: scatter + regression line + Pearson r
    # ------------------------------------------------------------------
    def scatter_with_corr(
        data,
        x_col,
        y_col,
        filename,
        xlabel,
        ylabel,
        title,
        hue_col=None,
        logx=False,
    ):
        # Drop rows with missing x or y
        use_cols = [x_col, y_col]
        if hue_col is not None:
            use_cols.append(hue_col)

        sub = data[use_cols].dropna()
        if sub.empty:
            print(f"[plot_results] No data for {x_col} vs {y_col}, skipping.")
            return

        # Pearson correlation
        r = sub[x_col].corr(sub[y_col], method="pearson")
        n = len(sub)

        fig, ax = plt.subplots(figsize=(6, 5))
        if hue_col is not None and hue_col in sub.columns:
            sns.scatterplot(
                data=sub,
                x=x_col,
                y=y_col,
                hue=hue_col,
                alpha=0.7,
                edgecolor="none",
                ax=ax,
            )
        else:
            sns.scatterplot(
                data=sub,
                x=x_col,
                y=y_col,
                alpha=0.7,
                edgecolor="none",
                ax=ax,
            )

        # Regression line (no CI band)
        sns.regplot(
            data=sub,
            x=x_col,
            y=y_col,
            scatter=False,
            color="black",
            ax=ax,
        )

        if logx:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nPearson r = {r:.2f} (n={n})")

        plt.tight_layout()
        out_path = os.path.join(vis_folder, filename)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[plot_results] Saved {out_path}")

    # ------------------------------------------------------------------
    # 3) Core comparisons: experimental vs TCR–pMHC energies/area
    # ------------------------------------------------------------------

    # ΔG_exp vs Rosetta interface_dG_tcr_pmhc
    scatter_with_corr(
        df,
        x_col="DeltaG_kcal_per_mol",
        y_col="interface_dG_tcr_pmhc",
        filename="exp_DeltaG_vs_interface_dG_tcr_pmhc.png",
        xlabel="Experimental ΔG (kcal/mol)",
        ylabel="Rosetta interface_dG TCR–pMHC (REU)",
        title="Experimental vs Rosetta interface ΔG (TCR–pMHC)",
        hue_col="MHC_type",
    )

    # ΔG_exp vs Rosetta dE_binding_tcr_pmhc
    scatter_with_corr(
        df,
        x_col="DeltaG_kcal_per_mol",
        y_col="dE_binding_tcr_pmhc",
        filename="exp_DeltaG_vs_dE_binding_tcr_pmhc.png",
        xlabel="Experimental ΔG (kcal/mol)",
        ylabel="Rosetta ΔE_binding TCR–pMHC (REU)",
        title="Experimental ΔG vs Rosetta ΔE_binding (TCR–pMHC)",
        hue_col="MHC_type",
    )

    # ------------------------------------------------------------------
    # 4) Distribution plots for peptide–MHC predictions
    # ------------------------------------------------------------------
    pep_metrics = [
        ("interface_dG_pep_mhc", "Peptide–MHC interface_dG (REU)"),
        ("dG_separated_pep_mhc", "Peptide–MHC dG_separated (REU)"),
        ("delta_sasa_pep_mhc", "Peptide–MHC ΔSASA (Å²)"),
    ]

    for col, label in pep_metrics:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(series, bins=30, kde=True, ax=ax)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        plt.tight_layout()
        out_path = os.path.join(vis_folder, f"distribution_{col}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[plot_results] Saved {out_path}")

    # ------------------------------------------------------------------
    # 5) Correlation heatmap across all metrics
    # ------------------------------------------------------------------
    cols_for_corr = [
        "DeltaG_kcal_per_mol",
        "interface_dG_tcr_pmhc",
        "dG_separated_tcr_pmhc",
        "delta_sasa_tcr_pmhc",
        "interface_dG_pep_mhc",
        "dG_separated_pep_mhc",
        "delta_sasa_pep_mhc",
        "dE_binding_tcr_pmhc",
    ]

    corr_df = df[[c for c in cols_for_corr if c in df.columns]].copy().dropna(how="all")
    if not corr_df.empty:
        corr = corr_df.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Spearman correlation: experimental vs Rosetta metrics")
        plt.tight_layout()
        out_path = os.path.join(vis_folder, "correlation_heatmap.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[plot_results] Saved {out_path}")
    else:
        print("[plot_results] No data for correlation heatmap.")

def select_MHC_classes(input_df: pd.DataFrame):

    # --- 0. Human MHC Class I and Class II ---
    human_prefixes = ('HLA-A', 'HLA-B','HLA-C', 'HLA-E', 'HLA-F', 'HLA-G','HLA-DR', 'HLA-DQ', 'HLA-DP')
    human_df = input_df[
        input_df['MHCname'].str.startswith(human_prefixes)
    ].copy()

    # --- 1. Human MHC Class I ---
    human_mhc_1_prefixes = ('HLA-A', 'HLA-B','HLA-C', 'HLA-E', 'HLA-F', 'HLA-G')
    human_mhc_1_df = input_df[
        input_df['MHCname'].str.startswith(human_mhc_1_prefixes)
    ].copy()

    # --- 2. Human MHC Class II ---
    human_mhc_2_prefixes = ('HLA-DR', 'HLA-DQ', 'HLA-DP')
    human_mhc_2_df = input_df[
        input_df['MHCname'].str.startswith(human_mhc_2_prefixes)
    ].copy()

    # --- 3. Mouse MHC Class I ---
    mouse_mhc_1_prefixes = ('H-2K', 'H-2D', 'H-2L')
    mouse_mhc_1_df = input_df[
        input_df['MHCname'].str.startswith(mouse_mhc_1_prefixes)
    ].copy()

    # --- 4. Mouse MHC Class II ---
    mouse_mhc_2_prefixes = ('I-A', 'I-E')
    mouse_mhc_2_df = input_df[
        input_df['MHCname'].str.startswith(mouse_mhc_2_prefixes)
    ].copy()

    # --- 5. all Mouse MHC ---
    mouse_prefixes = ('H-2K', 'H-2D', 'H-2L','I-A', 'I-E')
    mouse_df = input_df[
        input_df['MHCname'].str.startswith(mouse_prefixes)
    ].copy()
    return human_df, human_mhc_1_df, human_mhc_2_df, mouse_df, mouse_mhc_1_df, mouse_mhc_2_df


def merge_results(ATLAS_tsv_path="/mnt/larry/lilian/DATA/ATLAS/ATLAS.tsv", out_csv="atlas_results_summary.csv", vis_folder="plots"):
    atlas_df=pd.read_csv(ATLAS_tsv_path, sep="\t")
    atlas_true_pdb=atlas_df[atlas_df['true_PDB'].notna()]
    atlas_true_pdb=atlas_true_pdb[['MHCname','Kd_microM','Kon_per_M_per_s','Koff_per_S','DeltaG_kcal_per_mol','true_PDB']]


    results_df=pd.read_csv(out_csv)
    merged_df=pd.merge(atlas_true_pdb, results_df, left_on="true_PDB", right_on="TCR_name")
    human_df, human_mhc_1_df, human_mhc_2_df, mouse_df, mouse_mhc_1_df, mouse_mhc_2_df=select_MHC_classes(input_df=merged_df)
    os.makedirs(os.path.join(vis_folder,"human_all"), exist_ok=True)
    plot_results(human_df, vis_folder=os.path.join(vis_folder,"human_all"))
    os.makedirs(os.path.join(vis_folder,"human_mhc1"), exist_ok=True)
    plot_results(human_mhc_1_df, vis_folder=os.path.join(vis_folder,"human_mhc1"))
    os.makedirs(os.path.join(vis_folder,"human_mhc2"), exist_ok=True)
    plot_results(human_mhc_2_df, vis_folder=os.path.join(vis_folder,"human_mhc2"))
    os.makedirs(os.path.join(vis_folder,"mouse_all"), exist_ok=True)
    plot_results(mouse_df, vis_folder=os.path.join(vis_folder,"mouse_all"))

if __name__ == "__main__":
    base_dir="/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb"
    fixed_dir="/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_openmm_minimised"
    prefix="atlas_truepdb"
    vis_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/energy_based_binding_prediction/ATLAS/{prefix}_Rosetta_Results"
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)

    #ATLAS standardises, so that the chains are always named the same way
    tcr_alpha="D"
    tcr_beta="E"
    mhc_alpha="A"
    mhc_beta="B"
    peptide="C"
    modes = ["none", "rosetta_sidechain", "openmm"]#,"rosetta_full"]
    for mode in modes:
        if mode in ["rosetta_sidechain", "rosetta_full", "none"]:
            num_workers=4
        else:
            num_workers=2 #do 1 if using GPU
        out_csv = os.path.join(vis_folder,f"{prefix}_{mode}.csv")
        if Path(out_csv).exists():
            print(f"{out_csv} already exists, skipping...")
        else:
            run_folder(
                base_dir=base_dir,
                fixed_dir=fixed_dir,
                minimization_mode=mode,
                out_csv=out_csv,
                tcr_alpha_id=tcr_alpha,
                tcr_beta_id=tcr_beta,
                mhc_alpha_id=mhc_alpha,
                mhc_beta_id=mhc_beta,
                peptide_id=peptide,
                num_workers=num_workers,
            )
        merge_results(ATLAS_tsv_path="/mnt/larry/lilian/DATA/ATLAS/ATLAS.tsv", out_csv=out_csv, vis_folder=os.path.join(vis_folder,f"minimisation_{mode}_plots"))
