import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _clean_series(s: pd.Series) -> np.ndarray:
    """Convert to float array and drop NaNs/infs."""
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def violin_box(ax, data_list, labels, title, ylabel, ylim=None, rotate=0):
    """
    Draw violin distributions with an overlaid boxplot for the same data.
    data_list: list of 1D numpy arrays
    labels: list of strings
    """
    ax.violinplot(
        data_list,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    ax.boxplot(
        data_list,
        widths=0.15,
        showfliers=False,
        whis=1.5,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, axis="y", alpha=0.3)


def plot_outliers_scatter(df: pd.DataFrame, fig_path: str, outliers_path: str,
                          name_col: str = "tfold_name",
                          rmsd_col: str = "pMHC_aligned_rmsd",
                          tm_col: str = "pMHC_aligned_tm",
                          tm_thresh: float = 0.5,
                          rmsd_thresh: float = 8.0):
    """
    Outliers = TM < tm_thresh OR RMSD > rmsd_thresh (pMHC-aligned columns).
    Produces a figure with two subplots:
      - left: RMSD scatter with labels for outliers
      - right: TM scatter with labels for outliers
    """
    # Ensure numeric
    df = df.copy()
    df[rmsd_col] = pd.to_numeric(df.get(rmsd_col), errors="coerce")
    df[tm_col] = pd.to_numeric(df.get(tm_col), errors="coerce")

    # Build outlier mask (ignore rows where both are NaN)
    mask_valid = np.isfinite(df[rmsd_col].to_numpy(dtype=float)) | np.isfinite(df[tm_col].to_numpy(dtype=float))
    mask_outlier = (
        (df[tm_col] < tm_thresh) |
        (df[rmsd_col] > rmsd_thresh)
    ) & mask_valid

    outliers = df.loc[mask_outlier, [name_col, rmsd_col, tm_col]].copy()

    outliers.to_csv(outliers_path, index=False)

    # Nothing to plot
    if outliers.empty:
        print(f"[outliers] None found for TM<{tm_thresh} or RMSD>{rmsd_thresh}.")
        print(f"[outliers] Wrote empty file: {outliers_path}")
        return

    # Stable x positions for labeling
    outliers = outliers.sort_values(by=[rmsd_col, tm_col], ascending=[False, True], na_position="last")
    x = np.arange(len(outliers), dtype=int)

    fig, (ax_rmsd, ax_tm) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True)

    # RMSD subplot
    ax_rmsd.scatter(x, outliers[rmsd_col].to_numpy(dtype=float))
    ax_rmsd.axhline(rmsd_thresh, linestyle="--", linewidth=1)
    ax_rmsd.set_title(f"Outliers by RMSD (>{rmsd_thresh} Å) or TM (<{tm_thresh})")
    ax_rmsd.set_ylabel(f" RMSD (Å) ({rmsd_col})")
    ax_rmsd.set_xlabel("Outlier index")
    ax_rmsd.grid(True, axis="y", alpha=0.3)

    # TM subplot
    ax_tm.scatter(x, outliers[tm_col].to_numpy(dtype=float))
    ax_tm.axhline(tm_thresh, linestyle="--", linewidth=1)
    ax_tm.set_title(f"Outliers by TM (<{tm_thresh}) or RMSD (>{rmsd_thresh} Å)")
    ax_tm.set_ylabel(f" TM-score ({tm_col})")
    ax_tm.set_xlabel("Outlier index")
    ax_tm.set_ylim(0.0, 1.0)
    ax_tm.grid(True, axis="y", alpha=0.3)

    # Labels on both subplots
    # (Offset a bit so text is readable; still simple / robust)
    for i, (name, rmsd_val, tm_val) in enumerate(zip(outliers[name_col].astype(str),
                                                     outliers[rmsd_col].to_numpy(dtype=float),
                                                     outliers[tm_col].to_numpy(dtype=float))):
        if np.isfinite(rmsd_val):
            ax_rmsd.annotate(name, (x[i], rmsd_val), textcoords="offset points", xytext=(4, 4), ha="left", fontsize=8)
        if np.isfinite(tm_val):
            ax_tm.annotate(name, (x[i], tm_val), textcoords="offset points", xytext=(4, 4), ha="left", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print("[outliers] Wrote:", outliers_path)
    print("[outliers] Plot saved:", fig_path)
    print("[outliers] Count:", len(outliers))


def plot_distributions_from_csv(csv_path: str, out_dir: str, name_col: str = "tfold_name"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # -------------------------
    # 0) all aligned RMSD + TM
    # -------------------------
    all_rmsd_col = "all_aligned_rmsd"
    all_tm_col   = "all_aligned_tm"

    all_rmsd = _clean_series(df.get(all_rmsd_col, pd.Series(dtype=float)))
    all_tm   = _clean_series(df.get(all_tm_col, pd.Series(dtype=float)))

    fig0, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    violin_box(
        ax1,
        data_list=[all_rmsd],
        labels=[f"{all_rmsd_col}\n(n={len(all_rmsd)})"],
        title="all-aligned RMSD distribution",
        ylabel="RMSD (Å)",
    )

    violin_box(
        ax2,
        data_list=[all_tm],
        labels=[f"{all_tm_col}\n(n={len(all_tm)})"],
        title="all-aligned TM-score distribution",
        ylabel="TM-score",
        ylim=(0.0, 1.0),
    )

    fig0.tight_layout()
    fig0_path = os.path.join(out_dir, "dist_all_aligned_RMSD_TM.png")
    fig0.savefig(fig0_path, dpi=300)
    plt.close(fig0)

    # -------------------------
    # 1) pMHC aligned RMSD + TM
    # -------------------------
    pmhc_rmsd_col = "pMHC_aligned_rmsd"
    pmhc_tm_col   = "pMHC_aligned_tm"

    pmhc_rmsd = _clean_series(df.get(pmhc_rmsd_col, pd.Series(dtype=float)))
    pmhc_tm   = _clean_series(df.get(pmhc_tm_col, pd.Series(dtype=float)))

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    violin_box(
        ax1,
        data_list=[pmhc_rmsd],
        labels=[f"{pmhc_rmsd_col}\n(n={len(pmhc_rmsd)})"],
        title="pMHC-aligned RMSD distribution",
        ylabel="RMSD (Å)",
    )

    violin_box(
        ax2,
        data_list=[pmhc_tm],
        labels=[f"{pmhc_tm_col}\n(n={len(pmhc_tm)})"],
        title="pMHC-aligned TM-score distribution",
        ylabel="TM-score",
        ylim=(0.0, 1.0),
    )

    fig1.tight_layout()
    fig1_path = os.path.join(out_dir, "dist_pMHC_aligned_RMSD_TM.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # -------------------------
    # 2) All CDR RMSD (FR-aligned) in one distribution plot
    # -------------------------
    cdrs = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]

    cdr_rmsd_cols = [f"{cdr}_rmsd_fr_aligned" for cdr in cdrs]
    cdr_rmsd_data = [_clean_series(df.get(col, pd.Series(dtype=float))) for col in cdr_rmsd_cols]
    cdr_rmsd_labels = [f"{cdr}\n(n={len(arr)})" for cdr, arr in zip(cdrs, cdr_rmsd_data)]

    fig2, ax = plt.subplots(figsize=(10, 5))
    violin_box(
        ax,
        data_list=cdr_rmsd_data,
        labels=cdr_rmsd_labels,
        title="CDR RMSD distributions (aligned on same-chain framework; scored on CDR)",
        ylabel="RMSD (Å)",
        rotate=0,
    )
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "dist_CDR_RMSD_FR_aligned.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # -------------------------
    # 3) All CDR TM (FR-aligned) in one distribution plot
    # -------------------------
    cdr_tm_cols = [f"{cdr}_tm_fr_aligned" for cdr in cdrs]
    cdr_tm_data = [_clean_series(df.get(col, pd.Series(dtype=float))) for col in cdr_tm_cols]
    cdr_tm_labels = [f"{cdr}\n(n={len(arr)})" for cdr, arr in zip(cdrs, cdr_tm_data)]

    fig3, ax = plt.subplots(figsize=(10, 5))
    violin_box(
        ax,
        data_list=cdr_tm_data,
        labels=cdr_tm_labels,
        title="CDR TM-score distributions (aligned on same-chain framework; scored on CDR)",
        ylabel="TM-score",
        ylim=(0.0, 1.0),
        rotate=0,
    )
    fig3.tight_layout()
    fig3_path = os.path.join(out_dir, "dist_CDR_TM_FR_aligned.png")
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)

    # -------------------------
    # 4) All CDR RMSD (CDR-aligned) in one distribution plot
    # -------------------------
    cdr_rmsd_cols = [f"{cdr}_rmsd_cdr" for cdr in cdrs]
    cdr_rmsd_data = [_clean_series(df.get(col, pd.Series(dtype=float))) for col in cdr_rmsd_cols]
    cdr_rmsd_labels = [f"{cdr}\n(n={len(arr)})" for cdr, arr in zip(cdrs, cdr_rmsd_data)]

    fig4, ax = plt.subplots(figsize=(10, 5))
    violin_box(
        ax,
        data_list=cdr_rmsd_data,
        labels=cdr_rmsd_labels,
        title="CDR RMSD distributions (aligned on CDR; scored on CDR)",
        ylabel="RMSD (Å)",
        rotate=0,
    )
    fig4.tight_layout()
    fig4_path = os.path.join(out_dir, "dist_CDR_RMSD_CDR_aligned.png")
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)

    # -------------------------
    # 5) All CDR TM (CDR-aligned) in one distribution plot
    # -------------------------
    cdr_tm_cols = [f"{cdr}_tm_cdr" for cdr in cdrs]
    cdr_tm_data = [_clean_series(df.get(col, pd.Series(dtype=float))) for col in cdr_tm_cols]
    cdr_tm_labels = [f"{cdr}\n(n={len(arr)})" for cdr, arr in zip(cdrs, cdr_tm_data)]

    fig5, ax = plt.subplots(figsize=(10, 5))
    violin_box(
        ax,
        data_list=cdr_tm_data,
        labels=cdr_tm_labels,
        title="CDR TM-score distributions (aligned on CDR; scored on CDR)",
        ylabel="TM-score",
        ylim=(0.0, 1.0),
        rotate=0,
    )
    fig5.tight_layout()
    fig5_path = os.path.join(out_dir, "dist_CDR_TM_CDR_aligned.png")
    fig5.savefig(fig5_path, dpi=300)
    plt.close(fig5)

    # -------------------------
    # 6) Outlier scatter plots (requested)
    # -------------------------

    plot_outliers_scatter(
        df=df,
        fig_path=os.path.join(out_dir, "outliers_pMHC_aligned_RMSD_TM_scatter.png"),
        outliers_path=os.path.join(out_dir, "outliers_pMHC_aligned.csv"),
        name_col=name_col,
        rmsd_col="pMHC_aligned_rmsd",
        tm_col="pMHC_aligned_tm",
        tm_thresh=0.5,
        rmsd_thresh=8.0,
    )

    plot_outliers_scatter(
        df=df,
        fig_path=os.path.join(out_dir, "outliers_all_aligned_RMSD_TM_scatter.png"),
        outliers_path=os.path.join(out_dir, "outliers_all_aligned.csv"),
        name_col=name_col,
        rmsd_col="all_aligned_rmsd",
        tm_col="all_aligned_tm",
        tm_thresh=0.5,
        rmsd_thresh=8.0,
    )

    print("Saved distribution plots:")
    print(" -", fig1_path)
    print(" -", fig2_path)
    print(" -", fig3_path)
    print(" -", fig4_path)
    print(" -", fig5_path)


if __name__ == "__main__":
    out_dir = "/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_ATLAS_tfold_changedgeo/plots"
    csv_path = "/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_ATLAS_tfold_changedgeo/change_tfold_to_real_geometry_results.csv"
    plot_distributions_from_csv(csv_path, out_dir)
