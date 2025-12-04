#!/usr/bin/env python

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # <<< add this
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# Make sure we can import your TCR tools
# --------------------------------------------------------------------
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import TCR  # assuming this is correct import

# --------------------------------------------------------------------
# CONFIG – EDIT THESE PATHS IF NEEDED
# --------------------------------------------------------------------
UNBOUND_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_unbound_tcr_imgt")
BOUND_DIR   = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_bound_tcr_imgt")
OUTPUT_DIR  = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/geometry_bound_vs_unbound")

ANGLE_COLUMNS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# --------------------------------------------------------------------
# 1) GEOMETRY CALCULATION
# --------------------------------------------------------------------
def compute_angles_for_state(pdb_dir: Path, state: str) -> pd.DataFrame:
    """
    Iterate over all .pdb files in pdb_dir, compute TCR geometry
    using TCR(...).pairs[i].calc_angles(), and return a DataFrame
    with columns: ['tcr_name', 'state', ...angles...].
    """
    results = []

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = sorted([f for f in pdb_dir.iterdir() if f.suffix.lower() == ".pdb"])
    print(f"[{state}] Found {len(pdb_files)} PDB files in {pdb_dir}")

    for pdb_path in pdb_files:
        tcr_name = pdb_path.stem  # e.g. 1ao7
        try:
            tcr = TCR(
                input_pdb=str(pdb_path),
                traj_path=None,
                contact_cutoff=5.0,
                min_contacts=50,
                legacy_anarci=True,
            )
        except Exception as e:
            print(f"  [WARN] Failed to initialise TCR for {pdb_path.name}: {e}")
            continue

        if not hasattr(tcr, "pairs") or len(tcr.pairs) == 0:
            print(f"  [WARN] No TCR alpha/beta pair found for {tcr_name}, skipping.")
            continue

        # Typically one alpha-beta pair; we just take the first.
        pair = tcr.pairs[0]
        try:
            angle_df = pair.calc_angles()  # usually returns a 1-row DataFrame
        except Exception as e:
            print(f"  [WARN] calc_angles failed for {tcr_name}: {e}")
            continue

        if angle_df is None or angle_df.empty:
            print(f"  [WARN] Empty angle results for {tcr_name}, skipping.")
            continue

        # Normalise column names: expect 'pdb_name' but we want 'tcr_name'
        if "pdb_name" in angle_df.columns:
            angle_df = angle_df.rename(columns={"pdb_name": "tcr_name"})
        else:
            # If not present, just add tcr_name
            angle_df["tcr_name"] = tcr_name

        angle_df["tcr_name"] = tcr_name
        angle_df["state"] = state

        results.append(angle_df)

    if not results:
        raise RuntimeError(f"No angle results collected for state={state} in dir={pdb_dir}")

    combined = pd.concat(results, ignore_index=True)
    print(f"[{state}] Collected geometry for {combined['tcr_name'].nunique()} TCRs.")
    return combined


# --------------------------------------------------------------------
# 2) PLOT HELPERS (SAVE PLOTS, NO plt.show())
# --------------------------------------------------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_angle_distributions_by_state(df: pd.DataFrame, out_dir: Path):
    """
    For each angle, make a violin/box distribution plot comparing
    unbound vs bound.
    """
    ensure_dir(out_dir)

    for angle in ANGLE_COLUMNS:
        if angle not in df.columns:
            print(f"[plot_angle_distributions_by_state] Missing angle column {angle}, skipping.")
            continue

        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df,
            x="state",
            y=angle,
            inner="box",
            cut=0,
            palette="Set2",
        )
        plt.title(f"{angle} distribution: bound vs unbound")
        plt.xlabel("State")
        plt.ylabel(f"{angle} (degrees)")
        out_path = out_dir / f"{angle}_violin_bound_vs_unbound.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path}")


def plot_unbound_vs_bound_scatter(df_wide: pd.DataFrame, out_dir: Path):
    """
    df_wide has columns:
        tcr_name, angle, unbound, bound, shift
    Make scatter plots: unbound vs bound with y=x reference.
    """
    ensure_dir(out_dir)

    for angle in ANGLE_COLUMNS:
        sub = df_wide[df_wide["angle"] == angle].dropna(subset=["unbound", "bound"])
        if sub.empty:
            print(f"[plot_unbound_vs_bound_scatter] No data for angle {angle}, skipping.")
            continue

        plt.figure(figsize=(6, 6))
        plt.scatter(sub["unbound"], sub["bound"], alpha=0.7)
        min_val = np.nanmin([sub["unbound"].min(), sub["bound"].min()])
        max_val = np.nanmax([sub["unbound"].max(), sub["bound"].max()])
        pad = 0.05 * (max_val - min_val)
        lo, hi = min_val - pad, max_val + pad

        # y=x line
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)

        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.xlabel(f"{angle} unbound (deg)")
        plt.ylabel(f"{angle} bound (deg)")
        plt.title(f"{angle}: unbound vs bound")

        out_path = out_dir / f"{angle}_scatter_unbound_vs_bound.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path}")


def plot_shift_distributions(df_shifts: pd.DataFrame, out_dir: Path):
    """
    df_shifts columns:
        tcr_name, angle, shift
    Plot distribution of bound - unbound shifts for each angle.
    """
    ensure_dir(out_dir)

    for angle in ANGLE_COLUMNS:
        sub = df_shifts[df_shifts["angle"] == angle].dropna(subset=["shift"])
        if sub.empty:
            print(f"[plot_shift_distributions] No shifts for angle {angle}, skipping.")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(sub["shift"], kde=True, bins=20)
        plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
        plt.title(f"{angle} shift (bound - unbound)")
        plt.xlabel("Shift (deg)")
        plt.ylabel("Count")
        out_path = out_dir / f"{angle}_shift_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved {out_path}")


# --------------------------------------------------------------------
# 3) MAIN LOGIC
# --------------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    # 1) Compute geometry for unbound and bound
    print("=== Computing angles for UNBOUND ===")
    df_unbound = compute_angles_for_state(UNBOUND_DIR, state="unbound")

    print("\n=== Computing angles for BOUND ===")
    df_bound = compute_angles_for_state(BOUND_DIR, state="bound")

    # 2) Keep only TCRs that exist in both states
    tcr_unbound = set(df_unbound["tcr_name"].unique())
    tcr_bound = set(df_bound["tcr_name"].unique())
    common_tcrs = sorted(tcr_unbound & tcr_bound)

    print(f"\nCommon TCRs with both bound & unbound: {len(common_tcrs)}")
    if not common_tcrs:
        print("No overlapping TCRs. Check naming or directories.")
        return

    df_unbound_common = df_unbound[df_unbound["tcr_name"].isin(common_tcrs)].copy()
    df_bound_common = df_bound[df_bound["tcr_name"].isin(common_tcrs)].copy()

    # 3) Combine into a single long DataFrame
    all_df = pd.concat([df_unbound_common, df_bound_common], ignore_index=True)

    # Save raw angle table
    angles_csv = OUTPUT_DIR / "tcr_geometry_bound_vs_unbound_all_angles.csv"
    all_df.to_csv(angles_csv, index=False)
    print(f"\nSaved combined angle table to: {angles_csv}")

    # 4) Plot distributions bound vs unbound
    print("\n=== Plotting global angle distributions (bound vs unbound) ===")
    dist_dir = OUTPUT_DIR / "distributions"
    plot_angle_distributions_by_state(all_df, dist_dir)

    # 5) Build wide & shifts per TCR & angle
    rows = []
    for angle in ANGLE_COLUMNS:
        if angle not in all_df.columns:
            print(f"[main] Angle {angle} missing in all_df, skipping.")
            continue

        # pivot: index=tcr_name, columns=state -> values=angle
        pivot = all_df.pivot_table(
            index="tcr_name",
            columns="state",
            values=angle,
            aggfunc="mean",  # just in case there are multiple entries
        )

        if "unbound" not in pivot.columns or "bound" not in pivot.columns:
            print(f"[main] For angle {angle}, missing unbound or bound column after pivot, skipping.")
            continue

        for tcr_name, row in pivot.iterrows():
            u_val = row.get("unbound", np.nan)
            b_val = row.get("bound", np.nan)
            if pd.isna(u_val) or pd.isna(b_val):
                continue
            shift = b_val - u_val
            rows.append(
                {
                    "tcr_name": tcr_name,
                    "angle": angle,
                    "unbound": u_val,
                    "bound": b_val,
                    "shift": shift,
                }
            )

    if not rows:
        print("No valid paired geometry values to compute shifts. Stopping.")
        return

    df_wide = pd.DataFrame(rows)
    wide_csv = OUTPUT_DIR / "tcr_geometry_bound_vs_unbound_paired.csv"
    df_wide.to_csv(wide_csv, index=False)
    print(f"Saved paired geometry (unbound, bound, shift) to: {wide_csv}")

    # 6) Scatter plots: unbound vs bound
    print("\n=== Plotting unbound vs bound scatter per angle ===")
    scatter_dir = OUTPUT_DIR / "scatter_unbound_vs_bound"
    plot_unbound_vs_bound_scatter(df_wide, scatter_dir)

    # 7) Shift distributions
    print("\n=== Plotting shift distributions (bound - unbound) ===")
    shift_dir = OUTPUT_DIR / "shift_distributions"
    plot_shift_distributions(df_wide[["tcr_name", "angle", "shift"]], shift_dir)

    # 8) Quick text summary of shifts (mean, std) to see conserved trends
    print("\n=== Summary of shifts (bound - unbound) per angle ===")
    for angle in ANGLE_COLUMNS:
        sub = df_wide[df_wide["angle"] == angle]["shift"].dropna()
        if sub.empty:
            continue
        mean_shift = sub.mean()
        std_shift = sub.std()
        median_shift = sub.median()
        print(
            f"  {angle}: "
            f"mean shift = {mean_shift: .2f} deg, "
            f"median = {median_shift: .2f} deg, "
            f"std = {std_shift: .2f} deg, "
            f"n = {len(sub)}"
        )

    print("\nDone. Check PNGs and CSVs under:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
