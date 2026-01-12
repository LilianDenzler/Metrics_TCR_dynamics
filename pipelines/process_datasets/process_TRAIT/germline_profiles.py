import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
omics_processed_dir = "/mnt/larry/lilian/DATA/TRAIT/omics_processed"
out_dir = "/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/process_TRAIT/plots/germline_profiles"
os.makedirs(out_dir, exist_ok=True)

# Choose how bar heights are computed:
#   - "sum_count": sums the Count column (approx. #cell observations)
#   - "n_unique_tcr": counts unique clonotypes (recommended)
WEIGHT_MODE = "n_unique_tcr"

# How many categories to show per plot (genes or germline combos)
TOP_N_CATS = 60  # 200 makes plots unreadable; increase if you really want
MIN_TOTAL_FOR_PLOT = 1

TCR_ID_COLS = ["TRAV", "TRAJ", "CDR3a", "TRBV", "TRBD", "TRBJ", "CDR3b"]
GERMLINE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
GENE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]


# -----------------------
# File discovery
# -----------------------
def list_files(base_dir):
    augmented_pos, augmented_neg, stitch_pos, stitch_neg = [], [], [], []
    for fn in os.listdir(base_dir):
        p = os.path.join(base_dir, fn)
        if fn.endswith("_binder_pos.augmented.tsv"):
            augmented_pos.append(p)
        elif fn.endswith("_binder_neg.augmented.tsv"):
            augmented_neg.append(p)
        elif fn.endswith("_pos.stitch_failures.tsv"):
            stitch_pos.append(p)
        elif fn.endswith("_neg.stitch_failures.tsv"):
            stitch_neg.append(p)
    return augmented_pos, augmented_neg, stitch_pos, stitch_neg


def infer_pmhc_from_filename(path, suffix):
    base = os.path.basename(path)
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return os.path.splitext(base)[0]


# -----------------------
# Reading
# -----------------------
def read_augmented(path):
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

    # numeric Count (fallback to 1 if missing)
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(1).clip(lower=1)

    # Keys
    for c in TCR_ID_COLS:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype("string").fillna("")

    if df.empty:
        df["tcr_key"] = pd.Series([], index=df.index, dtype="string")
        df["germline_key"] = pd.Series([], index=df.index, dtype="string")
    else:
        df["tcr_key"] = df[TCR_ID_COLS].agg("|".join, axis=1).astype("string")
        df["germline_key"] = df[GERMLINE_COLS].astype("string").fillna("").agg("+".join, axis=1).astype("string")

    return df


def read_stitch_failures(paths, label):
    recs = []
    for p in paths:
        pmhc = infer_pmhc_from_filename(
            p, "_pos.stitch_failures.tsv" if label == "pos" else "_neg.stitch_failures.tsv"
        )
        try:
            d = pd.read_csv(p, sep="\t")
            n = len(d)
        except Exception:
            n = np.nan
        recs.append({"pMHC": pmhc, "binding_class": label, "n_stitch_fail_rows": n})
    return pd.DataFrame(recs)


# -----------------------
# Plotting: separate plots per class
# -----------------------
def grouped_bar_single_class(
    df: pd.DataFrame,
    category_col: str,
    binding_class: str,
    title: str,
    out_path: str,
    top_n: int = TOP_N_CATS,
    min_total: int = MIN_TOTAL_FOR_PLOT,
):
    """
    Single-class bar plot:
      - X axis: categories (e.g., TRAV genes)
      - Y axis: either unique TCRs (nunique tcr_key) or sum(Count) depending on WEIGHT_MODE
      - Data filtered to binding_class in {"pos","neg"}.

    This avoids the visually uninformative "pos vs neg" grouped bars when neg dominates.
    """
    sub = df[df["binding_class"] == binding_class].dropna(subset=[category_col]).copy()
    if sub.empty:
        return

    if WEIGHT_MODE == "sum_count":
        grp = sub.groupby(category_col)["Count"].sum()
        ylabel = "Sum(Count)"
    elif WEIGHT_MODE == "n_unique_tcr":
        grp = sub.groupby(category_col)["tcr_key"].nunique()
        ylabel = "Unique TCRs"
    else:
        raise ValueError("WEIGHT_MODE must be 'sum_count' or 'n_unique_tcr'")

    grp = grp.sort_values(ascending=False)
    grp = grp[grp >= min_total]
    if grp.empty:
        return

    grp = grp.head(top_n)

    cats = grp.index.astype(str).tolist()
    vals = grp.values.astype(float)

    fig_w = max(10, 0.35 * len(cats))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(len(cats))
    ax.bar(x, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=60, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_csv(df, name):
    fp = os.path.join(out_dir, name)
    df.to_csv(fp, index=False)
    print(f"[saved] {fp}")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    aug_pos_files, aug_neg_files, stitch_pos_files, stitch_neg_files = list_files(omics_processed_dir)

    aug_dfs = []
    for f in (aug_pos_files + aug_neg_files):
        aug_dfs.append(read_augmented(f))

    if not aug_dfs:
        raise RuntimeError("No augmented TSVs found in directory. Check filenames/suffixes.")

    aug = pd.concat(aug_dfs, ignore_index=True)

    stitch = pd.concat(
        [read_stitch_failures(stitch_pos_files, "pos"),
         read_stitch_failures(stitch_neg_files, "neg")],
        ignore_index=True
    )

    # -----------------------
    # Summary tables
    # -----------------------
    pmhc_summary = (
        aug.groupby(["pMHC", "binding_class"])
           .agg(
               n_rows=("tcr_key", "size"),
               n_unique_tcr=("tcr_key", "nunique"),
               sum_count=("Count", "sum"),
               median_count=("Count", "median"),
               p95_count=("Count", lambda x: float(np.percentile(x, 95))),
           )
           .reset_index()
    )

    pmhc_fail = (
        stitch.groupby(["pMHC", "binding_class"])["n_stitch_fail_rows"]
              .sum()
              .reset_index()
    )
    pmhc_summary = pmhc_summary.merge(pmhc_fail, on=["pMHC", "binding_class"], how="left")
    pmhc_summary["n_stitch_fail_rows"] = pmhc_summary["n_stitch_fail_rows"].fillna(0).astype(int)
    save_csv(pmhc_summary, "pmhc_summary_by_binding.csv")

    donor_summary = (
        aug.groupby(["Donor", "binding_class"])
           .agg(
               n_rows=("tcr_key", "size"),
               n_unique_tcr=("tcr_key", "nunique"),
               sum_count=("Count", "sum"),
           )
           .reset_index()
    )
    save_csv(donor_summary, "donor_summary_by_binding.csv")

    # -----------------------
    # Gene / germline profile plots (SEPARATE for pos and neg)
    # -----------------------
    pmhcs = sorted(aug["pMHC"].dropna().unique().tolist())
    donors = sorted(aug["Donor"].dropna().unique().tolist())

    for pmhc in pmhcs:
        sub = aug[aug["pMHC"] == pmhc]
        for gene in GENE_COLS:
            # POS
            out_path = os.path.join(out_dir, f"pmhc_{pmhc}__{gene}_pos.png")
            grouped_bar_single_class(
                sub,
                category_col=gene,
                binding_class="pos",
                title=f"{pmhc}: {gene} usage (Binding only) [{WEIGHT_MODE}]",
                out_path=out_path,
            )
            # NEG
            out_path = os.path.join(out_dir, f"pmhc_{pmhc}__{gene}_neg.png")
            grouped_bar_single_class(
                sub,
                category_col=gene,
                binding_class="neg",
                title=f"{pmhc}: {gene} usage (Non-binding only) [{WEIGHT_MODE}]",
                out_path=out_path,
            )

        # Combined germline: per pMHC
        out_path = os.path.join(out_dir, f"pmhc_{pmhc}__GERMLINE_pos.png")
        grouped_bar_single_class(
            sub,
            category_col="germline_key",
            binding_class="pos",
            title=f"{pmhc}: germline (TRAV+TRAJ+TRBV+TRBJ) Binding only [{WEIGHT_MODE}]",
            out_path=out_path,
        )
        out_path = os.path.join(out_dir, f"pmhc_{pmhc}__GERMLINE_neg.png")
        grouped_bar_single_class(
            sub,
            category_col="germline_key",
            binding_class="neg",
            title=f"{pmhc}: germline (TRAV+TRAJ+TRBV+TRBJ) Non-binding only [{WEIGHT_MODE}]",
            out_path=out_path,
        )

    # Per donor (across all pMHCs)
    for donor in donors:
        sub = aug[aug["Donor"] == donor]
        for gene in GENE_COLS:
            out_path = os.path.join(out_dir, f"donor_{donor}__{gene}_pos.png")
            grouped_bar_single_class(
                sub,
                category_col=gene,
                binding_class="pos",
                title=f"{donor}: {gene} usage (Binding only) [{WEIGHT_MODE}]",
                out_path=out_path,
            )
            out_path = os.path.join(out_dir, f"donor_{donor}__{gene}_neg.png")
            grouped_bar_single_class(
                sub,
                category_col=gene,
                binding_class="neg",
                title=f"{donor}: {gene} usage (Non-binding only) [{WEIGHT_MODE}]",
                out_path=out_path,
            )

        out_path = os.path.join(out_dir, f"donor_{donor}__GERMLINE_pos.png")
        grouped_bar_single_class(
            sub,
            category_col="germline_key",
            binding_class="pos",
            title=f"{donor}: germline (TRAV+TRAJ+TRBV+TRBJ) Binding only [{WEIGHT_MODE}]",
            out_path=out_path,
        )
        out_path = os.path.join(out_dir, f"donor_{donor}__GERMLINE_neg.png")
        grouped_bar_single_class(
            sub,
            category_col="germline_key",
            binding_class="neg",
            title=f"{donor}: germline (TRAV+TRAJ+TRBV+TRBJ) Non-binding only [{WEIGHT_MODE}]",
            out_path=out_path,
        )

    print("Done. All outputs saved in:", out_dir)
