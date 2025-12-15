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
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # seaborn is optional; if not installed, we fall back to pure matplotlib
    try:
        import seaborn as sns
        _HAS_SNS = True
    except Exception:
        sns = None
        _HAS_SNS = False

    os.makedirs(vis_folder, exist_ok=True)
    df = merged_df.copy()

    # ------------------------------------------------------------------
    # 0) Derive MHC_type from MHCname
    # ------------------------------------------------------------------
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
        choices = ["Human MHC I", "Human MHC II", "Mouse MHC I", "Mouse MHC II"]
        df["MHC_type"] = np.select(conditions, choices, default="Other/Unknown")
    else:
        df["MHC_type"] = "Unknown"

    # ------------------------------------------------------------------
    # 1) Ensure numeric dtypes ONLY for known numeric columns (TCR3D case)
    # ------------------------------------------------------------------
    numeric_cols = [
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
    # Helpers
    # ------------------------------------------------------------------
    def _finite_series(series: pd.Series) -> np.ndarray:
        x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return x

    def _save(fig, filename):
        out_path = os.path.join(vis_folder, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[plot_results] Saved {out_path}")

    def dist_plot_overall_and_by_group(col: str, label: str, bins: int = 40):
        """
        Makes:
          1) Overall distribution (hist + optional KDE)
          2) Distribution by MHC_type (violin + box if seaborn, else grouped boxplot)
        """
        if col not in df.columns:
            return

        series = df[col].dropna()
        if series.empty:
            return

        # ---------- (1) Overall ----------
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = _finite_series(series)
        if x.size == 0:
            return

        if _HAS_SNS:
            sns.histplot(x, bins=bins, kde=True, ax=ax)
        else:
            ax.hist(x, bins=bins)

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution: {label}\n(n={len(x)})")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, f"dist_overall_{col}.png")

        # ---------- (2) By MHC_type ----------
        # Only plot if we have at least 2 groups with data
        if "MHC_type" not in df.columns:
            return

        sub = df[["MHC_type", col]].dropna()
        if sub.empty:
            return

        # Keep only groups with enough points to be meaningful
        counts = sub["MHC_type"].value_counts()
        keep_groups = counts[counts >= 5].index.tolist()  # adjust threshold if you want
        sub = sub[sub["MHC_type"].isin(keep_groups)]
        if sub["MHC_type"].nunique() < 2:
            return

        # Order groups by median (nice for comparisons)
        med_order = (
            sub.groupby("MHC_type")[col]
            .median()
            .sort_values()
            .index
            .tolist()
        )

        fig, ax = plt.subplots(figsize=(10.5, 4.8))

        if _HAS_SNS:
            sns.violinplot(data=sub, x="MHC_type", y=col, order=med_order, ax=ax, cut=0)
            sns.boxplot(
                data=sub, x="MHC_type", y=col, order=med_order,
                ax=ax, width=0.18, showfliers=False
            )
        else:
            # matplotlib fallback: grouped boxplot
            data_list = [ _finite_series(sub.loc[sub["MHC_type"] == g, col]) for g in med_order ]
            ax.boxplot(data_list, showfliers=False)
            ax.set_xticks(range(1, len(med_order) + 1))
            ax.set_xticklabels(med_order, rotation=25, ha="right")

        ax.set_xlabel("MHC type")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by MHC type")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, f"dist_by_MHCtype_{col}.png")

    def summary_table(cols: list):
        """
        Writes a small summary CSV (count/median/IQR/mean/std) overall and by MHC_type.
        """
        rows = []
        for col in cols:
            if col not in df.columns:
                continue
            x = _finite_series(df[col])
            if x.size == 0:
                continue
            rows.append({
                "metric": col,
                "group": "ALL",
                "n": int(x.size),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
                "median": float(np.median(x)),
                "q25": float(np.quantile(x, 0.25)),
                "q75": float(np.quantile(x, 0.75)),
            })

            if "MHC_type" in df.columns:
                for g, subg in df.groupby("MHC_type"):
                    y = _finite_series(subg[col])
                    if y.size == 0:
                        continue
                    rows.append({
                        "metric": col,
                        "group": str(g),
                        "n": int(y.size),
                        "mean": float(np.mean(y)),
                        "std": float(np.std(y, ddof=1)) if y.size > 1 else np.nan,
                        "median": float(np.median(y)),
                        "q25": float(np.quantile(y, 0.25)),
                        "q75": float(np.quantile(y, 0.75)),
                    })

        if rows:
            out = pd.DataFrame(rows)
            out_path = os.path.join(vis_folder, "metric_summary_stats.csv")
            out.to_csv(out_path, index=False)
            print(f"[plot_results] Saved {out_path}")

    # ------------------------------------------------------------------
    # 2) Distribution plots for TCR–pMHC and peptide–MHC metrics
    # ------------------------------------------------------------------
    metrics = [
        ("interface_dG_tcr_pmhc", "TCR–pMHC interface_dG (REU)"),
        ("dG_separated_tcr_pmhc", "TCR–pMHC dG_separated (REU)"),
        ("delta_sasa_tcr_pmhc", "TCR–pMHC ΔSASA (Å²)"),
        ("dE_binding_tcr_pmhc", "TCR–pMHC ΔE_binding (REU)"),
        ("interface_dG_pep_mhc", "Peptide–MHC interface_dG (REU)"),
        ("dG_separated_pep_mhc", "Peptide–MHC dG_separated (REU)"),
        ("delta_sasa_pep_mhc", "Peptide–MHC ΔSASA (Å²)"),
    ]

    for col, label in metrics:
        dist_plot_overall_and_by_group(col, label, bins=40)


    # ------------------------------------------------------------------
    # 4) Summary stats table
    # ------------------------------------------------------------------
    summary_table([m[0] for m in metrics])


def select_MHC_classes(input_df: pd.DataFrame):

    # --- 0. Human MHC Class I and Class II ---
    human_prefixes = ('HLA-A', 'HLA-B','HLA-C', 'HLA-E', 'HLA-F', 'HLA-G','HLA-DR', 'HLA-DQ', 'HLA-DP')
    human_df = input_df[
        input_df['MHC_allele'].str.startswith(human_prefixes)
    ].copy()

    # --- 1. Human MHC Class I ---
    human_mhc_1_prefixes = ('HLA-A', 'HLA-B','HLA-C', 'HLA-E', 'HLA-F', 'HLA-G')
    human_mhc_1_df = input_df[
        input_df['MHC_allele'].str.startswith(human_mhc_1_prefixes)
    ].copy()

    # --- 2. Human MHC Class II ---
    human_mhc_2_prefixes = ('HLA-DR', 'HLA-DQ', 'HLA-DP')
    human_mhc_2_df = input_df[
        input_df['MHC_allele'].str.startswith(human_mhc_2_prefixes)
    ].copy()

    # --- 3. Mouse MHC Class I ---
    mouse_mhc_1_prefixes = ('H-2K', 'H-2D', 'H-2L')
    mouse_mhc_1_df = input_df[
        input_df['MHC_allele'].str.startswith(mouse_mhc_1_prefixes)
    ].copy()

    # --- 4. Mouse MHC Class II ---
    mouse_mhc_2_prefixes = ('I-A', 'I-E')
    mouse_mhc_2_df = input_df[
        input_df['MHC_allele'].str.startswith(mouse_mhc_2_prefixes)
    ].copy()

    # --- 5. all Mouse MHC ---
    mouse_prefixes = ('H-2K', 'H-2D', 'H-2L','I-A', 'I-E')
    mouse_df = input_df[
        input_df['MHC_allele'].str.startswith(mouse_prefixes)
    ].copy()
    return human_df, human_mhc_1_df, human_mhc_2_df, mouse_df, mouse_mhc_1_df, mouse_mhc_2_df


def merge_results(TCR3d_tsv_path="/mnt/larry/lilian/DATA/TCR3d_datasets/tcr_complexes_data.tsv", out_csv="tcr3d_results_summary.csv", vis_folder="plots"):
    tcr3d_df=pd.read_csv(TCR3d_tsv_path, sep="\t")
    tcr3d_true_pdb=tcr3d_df[tcr3d_df['PDB_ID'].notna()]
    tcr3d_true_pdb=tcr3d_true_pdb[['MHC_allele','PDB_ID']]

    results_df=pd.read_csv(out_csv)
    merged_df=pd.merge(tcr3d_true_pdb, results_df, left_on="PDB_ID", right_on="TCR_name")
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
    base_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes"
    fixed_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_openmm_minimised"
    prefix="tcr3d_truepdb"
    vis_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/energy_based_binding_prediction/TCR3d/{prefix}_Rosetta_Results"
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)

    #tcr3d standardises, so that the chains are always named the same way
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
        merge_results(TCR3d_tsv_path="/mnt/larry/lilian/DATA/TCR3d_datasets/tcr_complexes_data.tsv", out_csv=out_csv, vis_folder=os.path.join(vis_folder,f"minimisation_{mode}_plots"))
