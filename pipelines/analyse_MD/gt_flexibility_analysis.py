import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.ops import *
from TCR_TOOLS.scoring.embeddings import run_coords, run_ca_dist, run_dihedrals
from TCR_TOOLS.core import io
from TCR_TOOLS.scoring.rmsd_tm import rmsd_tm

import pandas as pd
from TCR_TOOLS.scoring.PCA_methods import pca_project_two as pca_project
from TCR_Metrics.pipelines.metrics_for_outputs.analyse_metrics import get_best_metric
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import oriol_analysis

import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import mdtraj as md

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Regions / combined regions for overlays
# ------------------------------------------------------------------

# Simple CDR regions (per-chain)
OVERLAY_REGIONS = [
    "A_CDR1",
    "A_CDR2",
    "A_CDR3",
    "B_CDR1",
    "B_CDR2",
    "B_CDR3",
]

# Combined labels (as in your run_one_TCR)
COMBINED_REGION_LABELS = [
    "A_CDR1A_CDR2A_CDR3",
    "B_CDR1B_CDR2B_CDR3",
    "A_CDR1A_CDR2A_CDR3B_CDR1B_CDR2B_CDR3",
]

# For overlays of "regions aligned on whole variable domains"
DOMAIN_REGION_OVERLAYS = {
    "A_variable": [
        "A_CDR1",
        "A_CDR2",
        "A_CDR3",
        "A_CDR1A_CDR2A_CDR3",
        "A_CDR1A_CDR2A_CDR3B_CDR1B_CDR2B_CDR3",
    ],
    "B_variable": [
        "B_CDR1",
        "B_CDR2",
        "B_CDR3",
        "B_CDR1B_CDR2B_CDR3",
        "A_CDR1A_CDR2A_CDR3B_CDR1B_CDR2B_CDR3",
    ],
}


# ------------------------------------------------------------------
# Per-TCR flexibility – GT only
# ------------------------------------------------------------------

def compute_flexibility_vs_rep(
    gt_traj_aligned,
    region_names,
    atom_names,
    outdir,
    time_step_ps=None,
):
    """
    GT-only flexibility for a given region, assuming trajectory is ALREADY aligned upstream:

    1) Take the GT aligned trajectory and extract only that region & atoms.
    2) Compute an 'average structure' (mean coordinates over frames).
    3) Compute RMSD of each GT frame to this average (no additional fitting).
    4) Find the GT frame closest to this average (representative frame).
    5) Save RMSD time series, mean RMSD, reference PDB, and histogram.

    All RMSDs are plain positional deviations in the common aligned frame.
    """

    os.makedirs(outdir, exist_ok=True)

    # 1) Extract region-only trajectory (mdtraj.Trajectory object)
    gt_md = gt_traj_aligned.domain_subset(region_names, atom_names)

    # 2) Average GT structure (no further superpose here)
    # gt_md.xyz shape: (n_frames, n_atoms, 3)
    avg_xyz = gt_md.xyz.mean(axis=0)  # (n_atoms, 3)

    # 3) RMSD of each GT frame vs average (no best-fit; just deviations)
    diff_gt = gt_md.xyz - avg_xyz[None, :, :]      # (n_frames, n_atoms, 3)
    sq_dist_gt = (diff_gt ** 2).sum(axis=2)        # (n_frames, n_atoms)
    rmsd_gt = np.sqrt(sq_dist_gt.mean(axis=1))     # (n_frames,)

    # 4) Representative frame = GT frame closest to this average
    best_idx = int(np.argmin(rmsd_gt))
    ref_frame = gt_md[best_idx]
    ref_xyz = ref_frame.xyz[0]                     # (n_atoms, 3)

    # Save the reference PDB (region only)
    ref_pdb_path = os.path.join(outdir, "gt_reference_region.pdb")
    ref_frame.save_pdb(ref_pdb_path)

    # Optional: save region-only trajectory for inspection
    gt_md.save_xtc(os.path.join(outdir, "gt_region_traj.xtc"))
    gt_md[0].save_pdb(os.path.join(outdir, "gt_region_traj_firstframe.pdb"))

    # 5) Save raw RMSD values
    np.savetxt(os.path.join(outdir, "rmsd_gt_vs_ref.txt"), rmsd_gt)

    mean_rmsd_gt = float(rmsd_gt.mean())

    with open(os.path.join(outdir, "rmsd_flex_summary.txt"), "w") as f:
        f.write(f"Region: {region_names}\n")
        f.write(f"Mean RMSD GT vs ref: {mean_rmsd_gt:.3f} Å\n")

    # Histogram of RMSD (flexibility distribution)
    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    ax_h.hist(rmsd_gt, bins=40, alpha=0.8, label="GT")
    ax_h.set_xlabel("RMSD vs representative (Å)")
    ax_h.set_ylabel("Count")
    ax_h.set_title(f"RMSD distribution ({'+'.join(region_names)})")
    ax_h.legend()
    fig_h.tight_layout()
    fig_h.savefig(os.path.join(outdir, "rmsd_histogram.png"), dpi=150)
    plt.close(fig_h)

    return mean_rmsd_gt, best_idx, ref_pdb_path


def get_ref_gt_frame( gttcr_pair, region_names, atom_names):
    """
    Align model & GT trajectories using a region-representative GT frame
    chosen directly from the GT mdtraj trajectory.

    This version does NOT use make_region_representative_pairview and therefore
    avoids residue/atom count mismatches between Bio.PDB and mdtraj.
    """
    # --- 0) Get trajectory views ---
    tv_gt = gttcr_pair.traj      # TrajectoryView (GT)


    gt_traj = tv_gt._traj        # md.Trajectory for GT


    # --- 1) Region-only GT traj for picking representative frame ---
    region_traj = tv_gt.domain_subset(
        region_names=region_names,
        atom_names=atom_names,
        inplace=False,
    )

    # Align region frames to frame 0 so averaging is meaningful
    region_traj.superpose(region_traj, 0)

    # Average structure of the region across all frames
    avg_xyz = region_traj.xyz.mean(axis=0, keepdims=True)  # (1, n_atoms_region, 3)
    avg_traj = md.Trajectory(avg_xyz, region_traj.topology)

    # RMSD of each region frame vs the region-average
    rmsd_to_avg = md.rmsd(region_traj, avg_traj)
    best_idx = int(np.argmin(rmsd_to_avg))

    # Full GT reference frame (same index as in region_traj)
    ref_frame = gt_traj[best_idx]
    return ref_frame


def align(gttcr_pair, to_be_aligned_pair, region_names, atom_names, outdir="", save_aligned=True):
    tv_gt = gttcr_pair.traj      # TrajectoryView (GT)
    gt_traj = tv_gt._traj        # md.Trajectory for GT

    tv_pred = to_be_aligned_pair.traj   # TrajectoryView (model)
    pred_traj = tv_pred._traj    # md.Trajectory for model
    ref_frame=get_ref_gt_frame( gttcr_pair, region_names, atom_names)

    # --- 2) Atom indices used for alignment (GT and model) ---
    gt_idx = tv_gt.domain_idx(region_names=region_names, atom_names=atom_names)
    pred_idx = tv_pred.domain_idx(region_names=region_names, atom_names=atom_names)

    if len(gt_idx) != len(pred_idx):
        raise ValueError(
            f"Index size mismatch between GT and model for regions={region_names}: "
            f"gt={len(gt_idx)} vs model={len(pred_idx)}"
        )

    # --- 3) Align model trajectory to the GT reference frame ---
    pred_traj_aligned = pred_traj.superpose(
        ref_frame,
        atom_indices=pred_idx,
        ref_atom_indices=gt_idx,
    )

    # --- 5) Wrap back into TrajectoryView so downstream code keeps working ---
    from TCR_TOOLS.classes.tcr import TrajectoryView

    pred_tv_aligned = TrajectoryView(pred_traj_aligned, tv_pred._chain_map)

    # --- 6) Save reference PDB and aligned region snapshots if requested ---
    os.makedirs(outdir, exist_ok=True)

    # Reference frame PDB
    ref_pdb_path = os.path.join(outdir, "gt_align_ref_frame.pdb")
    ref_frame.save_pdb(ref_pdb_path)

    if save_aligned:
        pred_region = pred_tv_aligned.domain_subset(
            region_names=region_names,
            atom_names=atom_names,
            inplace=False,
        )

        pred_region[0].save_pdb(os.path.join(outdir, "aligned_region.pdb"))
        pred_region.save_xtc(os.path.join(outdir, "aligned_region.xtc"))

    # --- 7) RMSD/TM report (same as before) ---
    (n, k, rmsd_A, tm) = rmsd_tm(
        pred_tv_aligned,
        gttcr_pair,
        regions=region_names,
        atoms={"CA"},
    )

    print(f"{region_names} RMSD pred->gt: {rmsd_A}, TM: {tm}")

    with open(os.path.join(outdir, "rmsd_tm.txt"), "w") as f:
        f.write(f"Aligned on {region_names} and atoms {atom_names}\n")
        f.write(f"pred_to_gt: mean_RMSD: {np.mean(rmsd_A)}, mean_TM: {np.mean(tm)}\n")

    return pred_tv_aligned


def run_one_TCR(
    pdb_gt,
    xtc_gt,
    output_dir,
    closest_frames=False,
    save_aligned=True,
    specific_region=None,
):
    """
    GT-only pipeline for a single TCR:

    - Load GT structure and MD.
    - For each CDR: align on that CDR, compute GT flexibility vs representative frame.
    - For A_variable / B_variable: align on domain, then compute flexibility for
      each CDR + combined regions within that alignment.
    """
    # Load GT
    gttcr = TCR(
        input_pdb=pdb_gt,
        traj_path=xtc_gt,
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True,
    )
    gttcr_pair = gttcr.pairs[0]

    # 3) Get sequences (original vs IMGT, all chains)
    print(get_sequence_dict(gttcr_pair.full_structure))

    # ------------------------------------------------------------------
    # 1) Alignment of each CDR separately + flexibility (region-aligned)
    # ------------------------------------------------------------------
    for region in ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]:
        if specific_region is not None and region != specific_region:
            continue

        region_outdir = os.path.join(output_dir, region)
        os.makedirs(region_outdir, exist_ok=True)

        gt_traj_aligned = align(
            gttcr_pair,
            to_be_aligned_pair=gttcr_pair,
            region_names=[region],
            atom_names={"CA", "C", "N"},
            outdir=region_outdir,
            save_aligned=save_aligned,
        )

        flex_outdir = os.path.join(region_outdir, "flexibility_vs_rep")
        mean_gt, best_idx, ref_pdb = compute_flexibility_vs_rep(
            gt_traj_aligned=gt_traj_aligned,
            region_names=[region],
            atom_names={"CA", "C", "N"},
            outdir=flex_outdir,
            time_step_ps=None,  # set if you want ns on x-axis
        )
        print(
            f"[{region}] mean RMSD vs rep frame (GT): "
            f"{mean_gt:.3f} Å (rep frame index = {best_idx})"
        )

    # ------------------------------------------------------------------
    # 2) Alignment of whole variable domains + flexibility for subregions
    # ------------------------------------------------------------------
    for aligned_variable_domain in ["A_variable", "B_variable"]:
        name_domain = aligned_variable_domain
        domain_outdir = os.path.join(output_dir, name_domain)
        os.makedirs(domain_outdir, exist_ok=True)

        gt_traj_aligned = align(
            gttcr_pair,
            to_be_aligned_pair=gttcr_pair,
            region_names=[aligned_variable_domain],
            atom_names={"CA", "C", "N"},
            outdir=domain_outdir,
            save_aligned=save_aligned,
        )

        rmsd_tm_df = pd.DataFrame(columns=["Region", "RMSD", "TM"])
        rows = []

        # Per-region flexibility within that domain alignment
        for region in [
            "A_CDR1",
            "A_CDR2",
            "A_CDR3",
            "B_CDR1",
            "B_CDR2",
            "B_CDR3",
            aligned_variable_domain,
        ]:
            if specific_region is not None and region != specific_region:
                continue

            # GT vs static GT structure, domain-aligned
            n, k, rmsd_A, tm = rmsd_tm(
                gt_traj_aligned, gttcr_pair, regions=[region], atoms={"CA"}
            )
            rows.append(
                {
                    "Region": f"{region}gt_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            region_outdir = os.path.join(domain_outdir, region)
            os.makedirs(region_outdir, exist_ok=True)

            flex_outdir = os.path.join(region_outdir, "flexibility_vs_rep")
            mean_gt, best_idx, ref_pdb = compute_flexibility_vs_rep(
                gt_traj_aligned=gt_traj_aligned,
                region_names=[region],
                atom_names={"CA", "C", "N"},
                outdir=flex_outdir,
                time_step_ps=None,
            )
            print(
                f"[{region}] (aligned on {aligned_variable_domain}) "
                f"mean RMSD vs rep frame (GT): "
                f"{mean_gt:.3f} Å (rep frame index = {best_idx})"
            )

        # Multi-region (combined) flexibility
        combined_regions = [
            ["A_CDR1", "A_CDR2", "A_CDR3"],
            ["B_CDR1", "B_CDR2", "B_CDR3"],
            ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"],
        ]

        for region_list in combined_regions:
            # specific_region is a single-region filter; skip combined if set
            if specific_region is not None:
                continue

            label = "".join(region_list)

            n, k, rmsd_A, tm = rmsd_tm(
                gt_traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"}
            )
            rows.append(
                {
                    "Region": f"{label}gt_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            label_outdir = os.path.join(domain_outdir, label)
            os.makedirs(label_outdir, exist_ok=True)

            flex_outdir = os.path.join(label_outdir, "flexibility_vs_rep")
            mean_gt, best_idx, ref_pdb = compute_flexibility_vs_rep(
                gt_traj_aligned=gt_traj_aligned,
                region_names=region_list,
                atom_names={"CA", "C", "N"},
                outdir=flex_outdir,
                time_step_ps=None,
            )
            print(
                f"[{label}] (aligned on {aligned_variable_domain}) "
                f"mean RMSD vs rep frame (GT): "
                f"{mean_gt:.3f} Å (rep frame index = {best_idx})"
            )

        # Build once and save summary table (if not subsetted to a single region)
        if specific_region is None:
            rmsd_tm_df = pd.DataFrame(rows, columns=["Region", "RMSD", "TM"])
            rmsd_tm_df.to_csv(
                os.path.join(domain_outdir, "rmsd_tm_summary.csv"), index=False
            )


def run_all_TCRs(
    model_folder,
    model_name,
    output_dir_all,
    error_log,
    closest_frames=True,
    save_aligned=True,
    specific_TCR=None,
    specific_region=None,
):
    """
    GT-only version: we use model_folder only as a list of TCR folders.
    For each TCR, we load GT PDB/XTC from Cory_data and ignore any model output.
    """
    for folder in os.listdir(model_folder):

        try:
            full_path = os.path.join(model_folder, folder)
            if not os.path.isdir(full_path):
                continue

            TCR_NAME_full = folder
            TCR_NAME = TCR_NAME_full.split("_")[0]

            if specific_TCR is not None and TCR_NAME != specific_TCR:
                continue

            print(f"Processing TCR: {TCR_NAME} (GT MD only)")
            TCR_output_folder = os.path.join(output_dir_all, TCR_NAME_full)
            os.makedirs(TCR_output_folder, exist_ok=True)

            # GT paths (Cory data)
            pdb_gt = f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}.pdb"
            xtc_gt = f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}_Prod.xtc"

            if not (os.path.exists(pdb_gt) and os.path.exists(xtc_gt)):
                print(
                    f"Skipping {TCR_NAME} as ground truth files not found "
                    f"({pdb_gt}, {xtc_gt})."
                )
                continue

            run_one_TCR(
                pdb_gt=pdb_gt,
                xtc_gt=xtc_gt,
                output_dir=TCR_output_folder,
                closest_frames=closest_frames,
                save_aligned=save_aligned,
                specific_region=specific_region,
            )

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            with open(error_log, "a") as ef:
                ef.write(f"Error processing {folder}: {e}\n")
    #    continue


# ------------------------------------------------------------------
# Overlay GT RMSD distributions across TCRs
# ------------------------------------------------------------------

def _collect_rmsd_paths(model_dir, path_builder):
    """
    Generic helper: iterate TCR folders inside model_dir and try to load rmsd_gt_vs_ref
    using the provided path_builder(model_dir, tcr_folder) -> path.
    """
    rmsd_dict = {}

    if not os.path.isdir(model_dir):
        print(f"[WARN] Model dir {model_dir} does not exist.")
        return rmsd_dict

    for tcr_folder in os.listdir(model_dir):
        tcr_dir = os.path.join(model_dir, tcr_folder)
        if not os.path.isdir(tcr_dir):
            continue

        tcr_name = tcr_folder.split("_")[0]
        rmsd_path = path_builder(tcr_dir)

        if not os.path.exists(rmsd_path):
            continue

        try:
            vals = np.loadtxt(rmsd_path)
            if vals.ndim == 0:
                vals = np.array([float(vals)])
            rmsd_dict[tcr_name] = vals
        except Exception as e:
            print(f"[WARN] Could not load {rmsd_path}: {e}")
            continue

    return rmsd_dict


def plot_overlay_distributions(rmsd_dict, title, out_png_path):
    """
    Ridgeline plot of GT RMSD distributions for multiple TCRs.
    One KDE ridge per TCR, stacked vertically.
    """
    if not rmsd_dict:
        print(f"[INFO] No RMSD data for {title}. Skipping.")
        return

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    import scipy.stats as st

    # Sort TCRs for consistent ordering
    tcr_names = sorted(rmsd_dict.keys())
    n_tcr = len(tcr_names)

    # Figure height scales with number of TCRs
    fig_height = max(4.0, 0.35 * n_tcr)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Common x grid for all KDEs
    all_vals = np.concatenate([rmsd_dict[name] for name in tcr_names])
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    x_pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 256)

    ridge_height = 0.8  # height of each ridge above its baseline

    for i, name in enumerate(tcr_names):
        vals = rmsd_dict[name]

        kde = st.gaussian_kde(vals)
        y = kde(x_grid)
        y /= y.max() if y.max() > 0 else 1.0
        y = y * ridge_height

        baseline = i
        ax.fill_between(
            x_grid,
            baseline,
            baseline + y,
            alpha=0.7,
            linewidth=0.8,
            edgecolor="black",
        )
        ax.plot(x_grid, baseline + y, color="black", linewidth=0.8)

    ax.set_yticks(np.arange(n_tcr))
    ax.set_yticklabels(tcr_names)
    ax.set_ylabel("TCR")

    y_top = (n_tcr - 1) + ridge_height
    ax.set_ylim(-0.5, y_top + 0.7)

    ax.set_xlabel("RMSD vs representative frame (Å)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.subplots_adjust(left=0.2, right=0.85, top=0.92, bottom=0.15)

    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)
    print(f"Saved overlay RMSD distribution to {out_png_path}")


def create_overlay_plots_for_model_region_aligned(base_flex_dir, model_name, regions):
    """
    Overlays for simple region-aligned flexibility:
        {base_flex_dir}/{model_name}/{TCR}/<region>/flexibility_vs_rep/rmsd_gt_vs_ref.txt
    """
    overlay_base = os.path.join(base_flex_dir, "RMSD_DISTRIBUTION_OVERLAYS")
    model_dir = os.path.join(base_flex_dir, model_name)
    model_outdir = os.path.join(overlay_base, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    for region in regions:
        def builder(tcr_dir):
            return os.path.join(
                tcr_dir,
                region,
                "flexibility_vs_rep",
                "rmsd_gt_vs_ref.txt",
            )

        rmsd_dict = _collect_rmsd_paths(model_dir, builder)
        title = f"GT flexibility distributions – {model_name}, region {region} (region-aligned)"
        out_png = os.path.join(
            model_outdir,
            f"{model_name}_{region}_GT_RMSD_overlay_region_aligned.png",
        )
        plot_overlay_distributions(rmsd_dict, title, out_png)


def create_overlay_plots_for_model_domain_aligned(base_flex_dir, model_name, domain_region_map):
    """
    Overlays for regions / combined regions when trajectories are aligned
    on the whole variable domain (A_variable / B_variable).

    Paths:
        {base_flex_dir}/{model_name}/{TCR}/{domain}/{region_or_label}/flexibility_vs_rep/rmsd_gt_vs_ref.txt
    """
    overlay_base = os.path.join(base_flex_dir, "RMSD_DISTRIBUTION_OVERLAYS")
    model_dir = os.path.join(base_flex_dir, model_name)
    model_outdir = os.path.join(overlay_base, model_name)
    os.makedirs(model_outdir, exist_ok=True)

    for domain_name, region_list in domain_region_map.items():
        for region_label in region_list:
            def builder(tcr_dir, domain=domain_name, rlab=region_label):
                return os.path.join(
                    tcr_dir,
                    domain,
                    rlab,
                    "flexibility_vs_rep",
                    "rmsd_gt_vs_ref.txt",
                )

            rmsd_dict = _collect_rmsd_paths(model_dir, builder)
            title = (
                f"GT flexibility distributions – {model_name}, "
                f"{region_label} (aligned on {domain_name})"
            )
            fname = f"{model_name}_{domain_name}_{region_label}_GT_RMSD_overlay_domain_aligned.png"
            out_png = os.path.join(model_outdir, fname)
            plot_overlay_distributions(rmsd_dict, title, out_png)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == "__main__":
    base_out = "/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/flexibility_analysis/"
    os.makedirs(base_out, exist_ok=True)

    # "model_name" is now just a label for the GT MD dataset
    for model_name in ["GT_MD"]:
        output_dir_all = os.path.join(base_out, model_name)
        os.makedirs(output_dir_all, exist_ok=True)

        # still using this folder just to define the TCR list
        model_folder = (
            "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
        )
        error_log = os.path.join(output_dir_all, "error.txt")

        # 1) Run GT-only flexibility analysis for all TCRs
        run_all_TCRs(
            model_folder=model_folder,
            model_name=model_name,
            output_dir_all=output_dir_all,
            error_log=error_log,
            closest_frames=False,
            save_aligned=False,
            specific_TCR=None,
            specific_region=None,
        )

        # 2) After all TCRs are processed, create overlay plots

        # (a) region-aligned overlays
        create_overlay_plots_for_model_region_aligned(
            base_flex_dir=base_out,
            model_name=model_name,
            regions=OVERLAY_REGIONS,
        )

        # (b) domain-aligned overlays, including combined regions
        create_overlay_plots_for_model_domain_aligned(
            base_flex_dir=base_out,
            model_name=model_name,
            domain_region_map=DOMAIN_REGION_OVERLAYS,
        )
