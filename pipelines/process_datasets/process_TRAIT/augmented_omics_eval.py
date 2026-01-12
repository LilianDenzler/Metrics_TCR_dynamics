import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TCR_ID_COLS = ["TRAV", "TRAJ", "CDR3a", "TRBV", "TRBD", "TRBJ", "CDR3b"]
GERMLINE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]


def read_augmented_file(path: str) -> pd.DataFrame:
    """
    Reads one augmented TSV and returns only analysis-relevant columns.

    Assumptions:
      - Binding column is exactly "Binding" or "Non-binding"
      - pMHC exists as a column in file
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    required = ["Donor", "pMHC", "Binding", "Count"] + TCR_ID_COLS
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    df = df[required].copy()

    # Map binding to {pos, neg}
    df["binding_class"] = df["Binding"].map({"Binding": "pos", "Non-binding": "neg"})
    df = df.dropna(subset=["Donor", "pMHC", "binding_class"])

    # Count as numeric (abundance proxy)
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(1).clip(lower=1)

    # Ensure all TCR ID cols are strings
    for c in TCR_ID_COLS:
        df[c] = df[c].astype("string").fillna("")

    # Empty-safe keys
    if df.empty:
        df["tcr_key"] = pd.Series([], index=df.index, dtype="string")
        df["germline_key"] = pd.Series([], index=df.index, dtype="string")
        return df

    df["tcr_key"] = df[TCR_ID_COLS].agg("|".join, axis=1).astype("string")
    df["germline_key"] = df[GERMLINE_COLS].agg("+".join, axis=1).astype("string")
    return df


def load_all_augmented(augmented_pos, augmented_neg) -> pd.DataFrame:
    """
    Concatenate all augmented files into a single dataframe.
    """
    frames = []
    for f in augmented_pos:
        frames.append(read_augmented_file(f))
        print("loaded pos:", os.path.basename(f))
    for f in augmented_neg:
        frames.append(read_augmented_file(f))
        print("loaded neg:", os.path.basename(f))

    if not frames:
        raise RuntimeError("No augmented TSVs found.")
    return pd.concat(frames, ignore_index=True)


def plot_pos_fraction_per_pmhc(
    aug: pd.DataFrame,
    out_dir: str,
    top_n: int = 30,
    metric: str = "unique_tcr",
    ylim_pad_frac: float = 0.08,
):
    """
    Plot: For each pMHC, the fraction of binders (pos / (pos+neg)), with annotations as:
        pos/neg   (e.g., 69201/500000)

    Improvements vs previous:
      - y-axis auto-scales to slightly above max observed fraction among shown pMHCs
      - annotations rotated 45 degrees to avoid overlap
      - annotation shows pos/neg rather than n=total
    """
    os.makedirs(out_dir, exist_ok=True)

    if metric not in {"unique_tcr", "sum_count"}:
        raise ValueError("metric must be 'unique_tcr' or 'sum_count'")

    if metric == "unique_tcr":
        grp = (
            aug.groupby(["pMHC", "binding_class"])["tcr_key"]
            .nunique()
            .unstack(fill_value=0)
        )
    else:
        grp = (
            aug.groupby(["pMHC", "binding_class"])["Count"]
            .sum()
            .unstack(fill_value=0)
        )

    for cls in ["pos", "neg"]:
        if cls not in grp.columns:
            grp[cls] = 0

    grp["total"] = grp["pos"] + grp["neg"]
    grp = grp[grp["total"] > 0].copy()
    grp["pos_frac"] = grp["pos"] / grp["total"]

    # Show top_n by total evidence
    grp = grp.sort_values("total", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grp.index))
    ax.bar(x, grp["pos_frac"].values)

    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=60, ha="right")
    ax.set_ylabel("Binder fraction")
    ax.set_title(f"Binder fraction per pMHC ({metric}), top {top_n} by total")

    # Auto-scale y-axis to make small differences visible
    ymax = float(grp["pos_frac"].max()) if len(grp) else 1.0
    ymax = min(1.0, ymax * (1.0 + ylim_pad_frac) + 1e-6)  # small pad, never exceed 1
    ax.set_ylim(0, ymax)

    # Annotate as pos/neg, rotated 45 degrees
    for i, (pos, neg, frac) in enumerate(zip(grp["pos"].values, grp["neg"].values, grp["pos_frac"].values)):
        # place slightly above bar, within axis
        y_text = min(ymax * 0.98, frac + ymax * 0.02)
        ax.text(
            i,
            y_text,
            f"{int(pos)}/{int(neg)}",
            ha="left",
            va="bottom",
            fontsize=8,
            rotation=45,
            rotation_mode="anchor",
        )

    fig.tight_layout()
    fp = os.path.join(out_dir, f"pmhc_binder_fraction_{metric}_top{top_n}.png")
    fig.savefig(fp, dpi=250)
    plt.close(fig)
    print("[saved]", fp)


if __name__ == "__main__":
    omics_processed_dir = "/mnt/larry/lilian/DATA/TRAIT/omics_processed"
    out_dir = "/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/process_TRAIT/plots/distribution_analysis"
    os.makedirs(out_dir, exist_ok=True)

    augmented_neg = []
    augmented_pos = []
    stitch_fails_neg = []
    stitch_fails_pos = []

    for file in os.listdir(omics_processed_dir):
        if file.endswith("_binder_neg.augmented.tsv"):
            augmented_neg.append(os.path.join(omics_processed_dir, file))
        if file.endswith("_binder_pos.augmented.tsv"):
            augmented_pos.append(os.path.join(omics_processed_dir, file))
        if file.endswith("_neg.stitch_failures.tsv"):
            stitch_fails_neg.append(os.path.join(omics_processed_dir, file))
        if file.endswith("_pos.stitch_failures.tsv"):
            stitch_fails_pos.append(os.path.join(omics_processed_dir, file))

    aug = load_all_augmented(augmented_pos, augmented_neg)

    plot_pos_fraction_per_pmhc(aug, out_dir, top_n=60, metric="unique_tcr")
    plot_pos_fraction_per_pmhc(aug, out_dir, top_n=60, metric="sum_count")
