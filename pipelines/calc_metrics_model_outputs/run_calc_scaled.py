import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.ops import *
from TCR_TOOLS.scoring.embeddings import run_coords, run_ca_dist, run_dihedrals
from TCR_TOOLS.core import io
from TCR_TOOLS.scoring.rmsd_tm import rmsd_tm
from TCR_TOOLS.scoring.rmsd_tm import run as run_rmsd_tm
from TCR_TOOLS.aligners.aligning import align_MD_same_TCR_profit  # still imported in case you need it

import pandas as pd
from TCR_TOOLS.scoring.PCA_methods import pca_project_two as pca_project
from dig_output_handler.process_output import process_output
from TCR_Metrics.pipelines.metrics_for_outputs.analyse_metrics import get_best_metric
from TCR_Metrics.pipelines.calc_metrics_model_outputs.visualise_closest_frames_pymol import render_zoomed_morph_image
from TCR_Metrics.pipelines.calc_metrics_model_outputs.Standardise_by_RMSD import rmsd_calibration_factor
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import oriol_analysis
from TCR_Metrics.pipelines.calc_metrics_model_outputs.benchmark_table_maker import run_make_tables
import os
from pathlib import Path
import yaml
import numpy as np
from matplotlib import pyplot as plt
import mdtraj as md   # needed in save_MD_frames...
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------
# Helper for global ranges (per region + mode + alignment context)
# ---------------------------------------------------------------------
def plot_standardised_embedding(
    Zg_scaled_centered: np.ndarray,
    Zp_scaled_centered: np.ndarray,
    corr: float,
    outpath: str,
    gt_label: str = "GT",
    model_label: str = "Model",
):
    """
    Scatter plot of the RMSD-standardised 2D embedding (already scaled+centered),
    with:
      - GT vs model clouds,
      - a scale bar showing 1 Å RMSD distance,
      - correlation (RMSD vs embedding distance) annotated in the legend.

    Assumes that the coordinate system of Z*_scaled_centered is already
    calibrated such that distances are in Å (via rmsd_calibration_factor).
    """
    Zg_sc = np.asarray(Zg_scaled_centered)
    Zp_sc = np.asarray(Zp_scaled_centered)

    fig, ax = plt.subplots(figsize=(6, 5))

    # GT vs model scatter
    ax.scatter(
        Zg_sc[:, 0],
        Zg_sc[:, 1],
        s=8,
        alpha=0.5,
        label=gt_label,
    )
    ax.scatter(
        Zp_sc[:, 0],
        Zp_sc[:, 1],
        s=8,
        alpha=0.5,
        label=model_label,
    )

    ax.set_xlabel("Dim 1 (Å-scaled)")
    ax.set_ylabel("Dim 2 (Å-scaled)")
    ax.set_title("Standardised embedding (RMSD-scaled)")

    # Make sure limits are set before drawing the scale bar
    ax.autoscale()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # --- 1 Å RMSD scale bar in embedding units ---
    # After RMSD scaling, 1 embedding unit ≈ 1 Å RMSD
    bar_len = 1.0  # 1 Å equivalent in this space
    x0 = x_min + 0.05 * x_range
    x1 = x0 + bar_len
    y0 = y_min + 0.08 * y_range  # a bit above bottom

    ax.plot([x0, x1], [y0, y0], color="black", linewidth=2)
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.03 * y_range,
        "1 Å RMSD",
        ha="center",
        va="bottom",
    )

    # --- Correlation in legend ---
    corr_handle = Line2D(
        [], [], color="none", label=f"RMSD–embedding corr = {corr:.2f}"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(corr_handle)
    labels.append(corr_handle.get_label())
    ax.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def make_region_key(assess_mode: str, alignment_context: str, region_names):
    """
    Build a unique key for a given (assess_mode, alignment_context, region_names).
    region_names can be a list like ["A_CDR1"] or ["A_CDR1","A_CDR2",...].
    """
    region_str = "+".join(region_names)
    return f"{assess_mode}|{alignment_context}|{region_str}"


def update_global_range(global_ranges: dict, region_key: str, Zg_scaled: np.ndarray, Zp_scaled: np.ndarray):
    """
    Update global min/max for this region_key given scaled+centered embeddings.
    """
    if Zp_scaled is None or Zp_scaled.size == 0:
        all_pts = Zg_scaled
    else:
        all_pts = np.concatenate([Zg_scaled, Zp_scaled], axis=0)

    x = all_pts[:, 0]
    y = all_pts[:, 1]

    local_xmin = float(x.min())
    local_xmax = float(x.max())
    local_ymin = float(y.min())
    local_ymax = float(y.max())

    if region_key not in global_ranges:
        global_ranges[region_key] = {
            "xmin": local_xmin,
            "xmax": local_xmax,
            "ymin": local_ymin,
            "ymax": local_ymax,
        }
    else:
        g = global_ranges[region_key]
        g["xmin"] = min(g["xmin"], local_xmin)
        g["xmax"] = max(g["xmax"], local_xmax)
        g["ymin"] = min(g["ymin"], local_ymin)
        g["ymax"] = max(g["ymax"], local_ymax)

# ---------------------------------------------------------------------
# Closest-frame analysis (unchanged)
# ---------------------------------------------------------------------

def save_MD_frames_closest_to_each_model_point_with_metrics(
    Zg,
    Zp,
    gt_traj_aligned,
    traj_aligned,
    outdir: str,
    region_names,
    atom_names,
    vis=False,
):
    os.makedirs(outdir, exist_ok=True)

    Zg = np.asarray(Zg)
    Zp = np.asarray(Zp)

    n_gt, d1 = Zg.shape
    n_model, d2 = Zp.shape
    assert d1 == d2, "Zg and Zp must have same embedding dimension"

    gt_md = getattr(gt_traj_aligned, "mdtraj", getattr(gt_traj_aligned, "_traj", gt_traj_aligned))
    pred_md = getattr(traj_aligned, "mdtraj", getattr(traj_aligned, "_traj", traj_aligned))

    rows = []

    for model_idx in range(n_model):
        z_model = Zp[model_idx]
        dists = np.linalg.norm(Zg - z_model[None, :], axis=1)
        gt_idx = int(np.argmin(dists))
        print(f"Model idx {model_idx} closest to GT idx {gt_idx} with dist {dists[gt_idx]:.4f}")

        if gt_idx >= gt_md.n_frames or model_idx >= pred_md.n_frames:
            print("[WARN] index beyond trajectory frames; skipping.")
            continue

        gt_frame = gt_md[gt_idx]
        model_frame = pred_md[model_idx]

        gt_pdb_path = os.path.join(outdir, f"gt_frame_{gt_idx}_model_{model_idx}.pdb")
        model_pdb_path = os.path.join(outdir, f"model_frame_{model_idx}.pdb")

        gt_frame.save_pdb(gt_pdb_path)
        model_frame.save_pdb(model_pdb_path)

        mov = md.load(model_pdb_path, top=model_pdb_path)
        ref = md.load(gt_pdb_path, top=gt_pdb_path)

        n, k, rmsd_A, tm = run_rmsd_tm(mov, ref, regions=region_names, atoms=atom_names)

        if hasattr(rmsd_A, "__len__"):
            rmsd_val = float(rmsd_A[0])
        else:
            rmsd_val = float(rmsd_A)

        if hasattr(tm, "__len__"):
            tm_val = float(tm[0])
        else:
            tm_val = float(tm)

        rows.append(
            {
                "model_index": model_idx,
                "closest_md_frame": gt_idx,
                "embedding_distance": float(dists[gt_idx]),
                "rmsd": rmsd_val,
                "tm_score": tm_val,
                "gt_pdb": gt_pdb_path,
                "model_pdb": model_pdb_path,
            }
        )

    if not rows:
        print("[WARN] No rows collected for closest-frame metrics.")
        return

    df = pd.DataFrame(rows)
    metrics_csv = os.path.join(outdir, "closest_frame_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"Saved closest-frame RMSD/TM table to: {metrics_csv}")

    stats_rows = []
    for col in ["rmsd", "tm_score"]:
        series = df[col]
        stats_rows.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
                "n": int(series.count()),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    summary_csv = os.path.join(outdir, "closest_frame_metrics_summary.csv")
    stats_df.to_csv(summary_csv, index=False)
    print(f"Saved summary stats to: {summary_csv}")

    mean_rmsd = stats_df.loc[stats_df["metric"] == "rmsd", "mean"].iloc[0]
    mean_tm = stats_df.loc[stats_df["metric"] == "tm_score", "mean"].iloc[0]
    print(f"Average RMSD over all model points: {mean_rmsd:.4f} Å")
    print(f"Average TM-score over all model points: {mean_tm:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["rmsd"], bins=30)
    axes[0].set_title("RMSD distribution")
    axes[0].set_xlabel("RMSD (Å)")
    axes[0].set_ylabel("Count")

    axes[1].hist(df["tm_score"], bins=30)
    axes[1].set_title("TM-score distribution")
    axes[1].set_xlabel("TM-score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(outdir, "closest_frame_metrics_hist.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved RMSD/TM distribution plot to: {plot_path}")

    if vis:
        render_zoomed_morph_image(
            closest_structures_folder=outdir,
            output_image=outdir + f"closest_frames_{''.join(region_names)}",
            regions=region_names,
            zoom_buffer=10,
        )
        print("Saved zoomed morph image to:", outdir + f"closest_frames_{region_names} ")

    # delete PDBs to save space
    for file in os.listdir(outdir):
        if file.endswith(".pdb"):
            os.remove(os.path.join(outdir, file))


# ---------------------------------------------------------------------
# Assess – COLLECT PHASE (compute embeddings, scale, center, save, update ranges)
# ---------------------------------------------------------------------

def get_assess_outdir(outdir_base: str, assess_type: str, mode: str) -> str:
    if assess_type == "dihed":
        return f"{outdir_base}/dihed_{mode}"
    if assess_type == "coords":
        return f"{outdir_base}/coords_{mode}"
    if assess_type == "ca":
        return f"{outdir_base}/ca_{mode}"
    raise ValueError(f"Unknown assess_type: {assess_type}")


def assess_collect(
    assess_mode: str,
    gt_traj_aligned,
    traj_aligned,
    regions,
    outdir_base="/workspaces/Graphormer/TCR_Metrics/test/test",
    closest_frames=True,
    global_ranges=None,
    region_key=None,
    fit_on="gt", #or "concat" or "pred"
):
    """
    Phase 1: for a single TCR and region:
      - compute embeddings (Zg, Zp),
      - calibrate & center using RMSD,
      - save scaled+centered embeddings to disk,
      - update global_ranges[region_key] with local min/max.
    """
    print("Running COLLECT for assess mode:", assess_mode, "on regions:", regions)

    assess_type = assess_mode.split("_")[0]   # dihed / coords / ca
    mode = "_".join(assess_mode.split("_")[1:])
    print("Assess type:", assess_type)
    print("Reducer mode:", mode)

    outdir = get_assess_outdir(outdir_base, assess_type, mode)
    os.makedirs(outdir, exist_ok=True)

    # --- 1) compute embeddings ---
    if assess_type == "dihed":
        Zg, Zp, info, tw, mt = run_dihedrals.run(
            gt_traj_aligned,
            traj_aligned,
            outdir=outdir,
            regions=regions,
            lag=5,
            reducer=mode,
            fit_on=fit_on,
            dihedrals=("phi", "psi"),
            encode="sincos",
            subsample=100,
            mantel_perms=999,
        )
    elif assess_type == "coords":
        Zg, Zp, info, tw, mt = run_coords.run(
            tv_gt=gt_traj_aligned,
            tv_pred=traj_aligned,
            outdir=outdir,
            regions=regions,
            lag=5,
            atoms=("CA", "C", "N"),
            reducer=mode,
            n_components=2,
            fit_on=fit_on,
            use_gt_scaler=False,
            subsample=100,
            mantel_method="spearman",
            mantel_perms=999,
        )
    elif assess_type == "ca":
        Zg, Zp, info, tw, mt = run_ca_dist.run(
            gt_traj_aligned,
            traj_aligned,
            outdir=outdir,
            regions=regions,
            fit_on=fit_on,
            reducer=mode,
            n_components=2,
            max_pairs=20000,
            subsample=100,
            lag=5,
            mantel_perms=999,
        )
    else:
        raise ValueError(f"Unknown assess_type: {assess_type}")

    if Zp is None:
        print("No model embeddings (Zp is None); skipping this region.")
        return

    # --- 2) RMSD calibration & centering ---
    k_scale, corr = rmsd_calibration_factor(
        Zg,
        gt_traj_aligned,
        region_names=regions,
        atom_names={"CA"},
    )
    print(f"RMSD calibration factor k: {k_scale}, corr: {corr}")
    if corr < 0.6:
        print(f"Low correlation ({corr}) between embedding distance and RMSD; cannot standardise by RMSD.")

    Zg_scaled = Zg * k_scale
    Zp_scaled = Zp * k_scale

    center = Zg_scaled.mean(axis=0)
    Zg_scaled_centered = Zg_scaled - center
    Zp_scaled_centered = Zp_scaled - center

    # --- 3) save embeddings for PMF phase ---
    emb_path = os.path.join(outdir, "embeddings_scaled_centered.npz")
    np.savez_compressed(emb_path, Zg=Zg_scaled_centered, Zp=Zp_scaled_centered)
    print(f"Saved scaled+centered embeddings to: {emb_path}")

    # --- 4) update global min/max for this region_key ---
    if global_ranges is not None and region_key is not None:
        update_global_range(global_ranges, region_key, Zg_scaled_centered, Zp_scaled_centered)

    # --- 5) optional closest-frames analysis (only needs Zg/Zp, not scaled) ---
    if closest_frames:
        closest_md_frames_dir = f"{outdir_base}/{assess_type}_{mode}/closest_frames"
        os.makedirs(closest_md_frames_dir, exist_ok=True)
        save_MD_frames_closest_to_each_model_point_with_metrics(
            Zg,
            Zp,
            gt_traj_aligned,
            traj_aligned,
            outdir=closest_md_frames_dir,
            region_names=regions,
            atom_names={"CA"},
        )
    xbins = 50
    ybins = 50
    temperature = 300.0


    # Here we pass global_range to oriol_analysis; tv_gt/tv_pred = None
    plot_pca(
        Zg,
        Zp,
        None,
        None,
        outfile=os.path.join(outdir, "pca_projection.png")
    )
    oriol_analysis(
        xbins,
        ybins,
        Zg,
        Zp,
        temperature,
        name="",
        outfolder=outdir,
        tv_gt=None,
        tv_pred=None,
        global_range=None,
    )
    # Also save a quick PCA projection for sanity (unscaled vs scaled_centered)
    outdir_standardised = os.path.join(outdir, "standardised_by_RMSD")
    os.makedirs(outdir_standardised, exist_ok=True)
    plot_standardised_embedding(
        Zg_scaled_centered,
        Zp_scaled_centered,
        corr=corr,  # <- from rmsd_calibration_factor
        outpath=os.path.join(outdir_standardised, "pca_projection_standardised.png"),
    )
    #input(os.path.join(outdir_standardised, "pca_projection_standardised.png"))


# ---------------------------------------------------------------------
# Assess – PMF PHASE (load scaled embeddings, use global bins, run oriol_analysis)
# ---------------------------------------------------------------------

def assess_pmf(
    assess_mode: str,
    regions,
    outdir_base: str,
    global_ranges: dict,
    region_key: str,
):
    """
    Phase 2: for a single TCR and region:
      - load scaled+centered embeddings,
      - run oriol_analysis with global bin ranges (shared across TCRs).
    """
    print("Running PMF for assess mode:", assess_mode, "on regions:", regions)

    if assess_mode == "all":
        # No global-binned PMF defined for the multi-mode screening case.
        print("assess_mode='all' – skipping PMF.")
        return

    assess_type = assess_mode.split("_")[0]
    mode = "_".join(assess_mode.split("_")[1:])

    outdir = get_assess_outdir(outdir_base, assess_type, mode)
    emb_path = os.path.join(outdir, "embeddings_scaled_centered.npz")

    if not os.path.exists(emb_path):
        print(f"[PMF] Missing embeddings file {emb_path}, skipping.")
        return

    data = np.load(emb_path)
    Zg_scaled_centered = data["Zg"]
    Zp_scaled_centered = data["Zp"]

    if region_key not in global_ranges:
        print(f"[PMF] No global range for key {region_key}, skipping.")
        return

    gr = global_ranges[region_key]
    global_range = (gr["xmin"], gr["xmax"], gr["ymin"], gr["ymax"])
    print(f"[PMF] Using global_range={global_range} for key {region_key}")

    xbins = 50
    ybins = 50
    temperature = 300.0

    outdir_standardised = os.path.join(outdir, "standardised_by_RMSD")
    os.makedirs(outdir_standardised, exist_ok=True)

    # Here we pass global_range to oriol_analysis; tv_gt/tv_pred = None
    oriol_analysis(
        xbins,
        ybins,
        Zg_scaled_centered,
        Zp_scaled_centered,
        temperature,
        name="",
        outfolder=outdir_standardised,
        tv_gt=None,
        tv_pred=None,
        global_range=global_range,
    )


# ---------------------------------------------------------------------
# Best-metric aggregation (unchanged)
# ---------------------------------------------------------------------

def calc_best_metric(outdir_base, rank_by="trust.gt"):
    best_methods = {}
    for folder in Path(outdir_base).iterdir():
        if folder.is_dir():
            if "A_variable" in str(folder):
                for subfolder in Path(outdir_base).iterdir():
                    if subfolder.is_dir():
                        region = subfolder.name
                        best_method, score = get_best_metric(subfolder, rank_by=rank_by)
                        full_region = "A_variable_" + region
                        best_methods[full_region] = {"method": best_method, "score": score}
            else:
                region = folder.name
                best_method, score = get_best_metric(folder, rank_by=rank_by)
                best_methods[region] = {"method": best_method, "score": score}
    print("\n=== Best methods per region ===")
    best_methods_df = pd.DataFrame.from_dict(best_methods, orient="index")
    print(best_methods_df)
    best_methods_df.to_csv(os.path.join(outdir_base, "best_methods_per_region_mantel.csv"))


# ---------------------------------------------------------------------
# Alignment (using representative frame)
# ---------------------------------------------------------------------
def align(digtcr_pair, gttcr_pair, region_names, atom_names, outdir="", save_aligned=True):
    """
    Align model & GT trajectories using a region-representative GT frame
    chosen directly from the GT mdtraj trajectory.

    This version does NOT use make_region_representative_pairview and therefore
    avoids residue/atom count mismatches between Bio.PDB and mdtraj.
    """
    # --- 0) Get trajectory views ---
    tv_gt = gttcr_pair.traj      # TrajectoryView (GT)
    tv_pred = digtcr_pair.traj   # TrajectoryView (model)

    gt_traj = tv_gt._traj        # md.Trajectory for GT
    pred_traj = tv_pred._traj    # md.Trajectory for model

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

    # --- 4) Align GT trajectory to the same reference frame (for consistency) ---
    gt_traj_aligned = gt_traj.superpose(
        ref_frame,
        atom_indices=gt_idx,
        ref_atom_indices=gt_idx,
    )

    # --- 5) Wrap back into TrajectoryView so downstream code keeps working ---
    from TCR_TOOLS.classes.tcr import TrajectoryView

    pred_tv_aligned = TrajectoryView(pred_traj_aligned, tv_pred._chain_map)
    gt_tv_aligned = TrajectoryView(gt_traj_aligned, tv_gt._chain_map)

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
        gt_region = gt_tv_aligned.domain_subset(
            region_names=region_names,
            atom_names=atom_names,
            inplace=False,
        )

        pred_region[0].save_pdb(os.path.join(outdir, "pred_aligned_region.pdb"))
        pred_region.save_xtc(os.path.join(outdir, "pred_aligned_region.xtc"))

        gt_region[0].save_pdb(os.path.join(outdir, "gt_aligned_region.pdb"))
        gt_region.save_xtc(os.path.join(outdir, "gt_aligned_region.xtc"))

    # --- 7) RMSD/TM report (same as before) ---
    (n, k, rmsd_A, tm) = rmsd_tm(
        pred_tv_aligned,
        gttcr_pair,
        regions=region_names,
        atoms={"CA"},
    )
    (n, k, rmsd_A_gt, tm_gt) = rmsd_tm(
        gt_tv_aligned,
        gttcr_pair,
        regions=region_names,
        atoms={"CA"},
    )
    print(f"{region_names} RMSD pred->gt: {rmsd_A}, TM: {tm}")
    print(f"{region_names} RMSD gt->gt: {rmsd_A_gt}, TM: {tm_gt}")

    with open(os.path.join(outdir, "rmsd_tm.txt"), "w") as f:
        f.write(f"Aligned on {region_names} and atoms {atom_names}\n")
        f.write(f"pred_to_gt: mean_RMSD: {np.mean(rmsd_A)}, mean_TM: {np.mean(tm)}\n")
        f.write(f"gt_to_gt:   mean_RMSD: {np.mean(rmsd_A_gt)}, mean_TM: {np.mean(tm_gt)}\n")

    return pred_tv_aligned, gt_tv_aligned


# ---------------------------------------------------------------------
# run_one_TCR – COLLECT PHASE
# ---------------------------------------------------------------------

def run_one_TCR_collect(
    pdb_gt,
    xtc_gt,
    pdb_pred,
    xtc_pred,
    output_dir,
    region_metric_config,
    global_ranges,
    closest_frames=True,
    save_aligned=True,
    specific_region=None,
    fit_on="gt",
):
    """
    Phase 1 for a single TCR:
      - align trajectories per region / domain,
      - run assess_collect to compute embeddings, scale, center, save,
      - update global_ranges.
    """
    gttcr = TCR(
        input_pdb=pdb_gt,
        traj_path=xtc_gt,
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True,
    )
    gttcr_pair = gttcr.pairs[0]
    print(get_sequence_dict(gttcr_pair.full_structure))

    try:
        digtcr = TCR(
            input_pdb=pdb_pred,
            traj_path=xtc_pred,
            contact_cutoff=5.0,
            min_contacts=50,
            legacy_anarci=True,
        )
        digtcr_pair = digtcr.pairs[0]
    except Exception:
        digtcr = TCR(
            input_pdb=pdb_pred,
            traj_path=None,
            contact_cutoff=5.0,
            min_contacts=50,
            legacy_anarci=True,
        )
        digtcr_pair = digtcr.pairs[0]
        digtcr_pair.attach_trajectory(
            xtc_pred,
            region_names=None,
            atom_names={"CA", "C", "N"},
        )



    # 1) each CDR separately (aligned on that CDR)
    for region in ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]:
        if specific_region is not None and region != specific_region:
            continue

        region_dir = os.path.join(output_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        traj_aligned, gt_traj_aligned = align(
            digtcr_pair,
            gttcr_pair,
            region_names=[region],
            atom_names={"CA", "C", "N"},
            outdir=region_dir,
            save_aligned=save_aligned,
        )

        cfg_region = region_metric_config.get(region, {})
        assess_mode = cfg_region.get(region, "all")
        region_key = make_region_key(assess_mode, "region", [region])

        assess_collect(
            assess_mode,
            gt_traj_aligned,
            traj_aligned,
            regions=[region],
            outdir_base=region_dir,
            closest_frames=closest_frames,
            global_ranges=global_ranges,
            region_key=region_key,
            fit_on=fit_on
        )

    # 2) whole variable domain + subregions + combined regions
    combined_regions_list = [
        ["A_CDR1", "A_CDR2", "A_CDR3"],
        ["B_CDR1", "B_CDR2", "B_CDR3"],
        ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"],
    ]

    for aligned_variable_domain in ["A_variable", "B_variable"]:
        name_domain = aligned_variable_domain
        domain_dir = os.path.join(output_dir, name_domain)
        os.makedirs(domain_dir, exist_ok=True)

        traj_aligned, gt_traj_aligned = align(
            digtcr_pair,
            gttcr_pair,
            region_names=[aligned_variable_domain],
            atom_names={"CA", "C", "N"},
            outdir=domain_dir,
            save_aligned=save_aligned,
        )

        rmsd_rows = []

        # subregions within that domain + the domain itself
        per_region_list = [
            "A_CDR1",
            "A_CDR2",
            "A_CDR3",
            "B_CDR1",
            "B_CDR2",
            "B_CDR3",
            aligned_variable_domain,
        ]

        for region in per_region_list:
            if specific_region is not None and region != specific_region:
                continue

            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rmsd_rows.append(
                {
                    "Region": f"{region}pred_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rmsd_rows.append(
                {
                    "Region": f"{region}gt_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            region_dir = os.path.join(domain_dir, region)
            os.makedirs(region_dir, exist_ok=True)

            cfg_region = region_metric_config.get(region, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            region_key = make_region_key(assess_mode, aligned_variable_domain, [region])

            assess_collect(
                assess_mode,
                gt_traj_aligned,
                traj_aligned,
                regions=[region],
                outdir_base=region_dir,
                closest_frames=closest_frames,
                global_ranges=global_ranges,
                region_key=region_key,
                fit_on=fit_on
            )

        # combined regions
        for region_list in combined_regions_list:
            if specific_region is not None and region_list != specific_region:
                continue

            label = "".join(region_list)
            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rmsd_rows.append(
                {
                    "Region": f"{label}pred_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rmsd_rows.append(
                {
                    "Region": f"{label}gt_to_gt",
                    "RMSD": float(np.mean(rmsd_A)),
                    "TM": float(np.mean(tm)),
                }
            )

            label_dir = os.path.join(domain_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            cfg_region = region_metric_config.get(label, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            region_key = make_region_key(assess_mode, aligned_variable_domain, region_list)

            assess_collect(
                assess_mode,
                gt_traj_aligned,
                traj_aligned,
                regions=region_list,
                outdir_base=label_dir,
                closest_frames=closest_frames,
                global_ranges=global_ranges,
                region_key=region_key,
                fit_on=fit_on
            )

        if specific_region is None:
            rmsd_tm_df = pd.DataFrame(rmsd_rows, columns=["Region", "RMSD", "TM"])
            rmsd_tm_df.to_csv(os.path.join(domain_dir, "rmsd_tm_summary.csv"), index=False)


# ---------------------------------------------------------------------
# PMF PHASE – run over existing outputs
# ---------------------------------------------------------------------

def run_pmf_for_TCR(
    TCR_output_folder: str,
    region_metric_config: dict,
    global_ranges: dict,
    closest_frames: bool = False,
    specific_region=None,
):
    """
    Phase 2 for a single TCR:
      - No alignment / no embeddings.
      - Just walk the same region/domain structure as in collect phase,
        and call assess_pmf with the right region_key and outdir_base.
    """

    # 1) simple CDRs aligned on themselves
    for region in ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]:
        if specific_region is not None and region != specific_region:
            continue

        region_dir = os.path.join(TCR_output_folder, region)
        if not os.path.isdir(region_dir):
            continue

        cfg_region = region_metric_config.get(region, {})
        assess_mode = cfg_region.get(region, "all")
        region_key = make_region_key(assess_mode, "region", [region])

        assess_pmf(
            assess_mode,
            regions=[region],
            outdir_base=region_dir,
            global_ranges=global_ranges,
            region_key=region_key,
        )

    # 2) domains + subregions + combined regions
    combined_regions_list = [
        ["A_CDR1", "A_CDR2", "A_CDR3"],
        ["B_CDR1", "B_CDR2", "B_CDR3"],
        ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"],
    ]

    for aligned_variable_domain in ["A_variable", "B_variable"]:
        domain_dir = os.path.join(TCR_output_folder, aligned_variable_domain)
        if not os.path.isdir(domain_dir):
            continue

        # per subregion + domain
        per_region_list = [
            "A_CDR1",
            "A_CDR2",
            "A_CDR3",
            "B_CDR1",
            "B_CDR2",
            "B_CDR3",
            aligned_variable_domain,
        ]

        for region in per_region_list:
            if specific_region is not None and region != specific_region:
                continue

            region_dir = os.path.join(domain_dir, region)
            if not os.path.isdir(region_dir):
                continue

            cfg_region = region_metric_config.get(region, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            region_key = make_region_key(assess_mode, aligned_variable_domain, [region])

            assess_pmf(
                assess_mode,
                regions=[region],
                outdir_base=region_dir,
                global_ranges=global_ranges,
                region_key=region_key,
            )

        # combined regions
        for region_list in combined_regions_list:
            if specific_region is not None and region_list != specific_region:
                continue

            label = "".join(region_list)
            label_dir = os.path.join(domain_dir, label)
            if not os.path.isdir(label_dir):
                continue

            cfg_region = region_metric_config.get(label, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            region_key = make_region_key(assess_mode, aligned_variable_domain, region_list)

            assess_pmf(
                assess_mode,
                regions=region_list,
                outdir_base=label_dir,
                global_ranges=global_ranges,
                region_key=region_key,
            )


# ---------------------------------------------------------------------
# run_all_TCRs – COLLECT + PMF
# ---------------------------------------------------------------------

def run_all_TCRs_collect(
    model_folder,
    model_name,
    output_dir_all,
    error_log,
    CONFIG_PATH,
    global_ranges,
    closest_frames=True,
    save_aligned=True,
    specific_TCR=None,
    specific_region=None,
    linker="GGGGS" * 3,
    fit_on="gt",
    GT_MD_FOLDER="/mnt/larry/lilian/DATA/Cory_data/"
):
    """
    Phase 1: go over all TCRs, align + compute embeddings + update global_ranges.
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

            print(f"[COLLECT] Processing TCR: {TCR_NAME}")
            TCR_output_folder = os.path.join(output_dir_all, TCR_NAME_full)
            os.makedirs(TCR_output_folder, exist_ok=True)

            output_xtc_path = os.path.join(TCR_output_folder, "pred.xtc")
            output_pdb_path= os.path.join(TCR_output_folder, "pred.pdb")

            dig_output_dir = os.path.join(model_folder, TCR_NAME_full, model_name)
            output_xtc_path, output_pdb_path = process_output(
                dig_output_dir,
                linker,
                output_xtc_path,
                output_pdb_path
            )

            with CONFIG_PATH.open() as f:
                region_metric_config = yaml.safe_load(f)

            pdb_gt = f"{GT_MD_FOLDER}/{TCR_NAME}/{TCR_NAME}.pdb"
            xtc_gt = f"{GT_MD_FOLDER}/{TCR_NAME}/{TCR_NAME}_Prod.xtc"


            if not (os.path.exists(pdb_gt) and os.path.exists(xtc_gt)):
                xtc_gt = f"{GT_MD_FOLDER}/{TCR_NAME}/{TCR_NAME}.xtc"
                if not (os.path.exists(pdb_gt) and os.path.exists(xtc_gt)):
                    print(f"Skipping {TCR_NAME} as ground truth files not found.")
                    continue

            run_one_TCR_collect(
                pdb_gt=pdb_gt,
                xtc_gt=xtc_gt,
                pdb_pred=output_pdb_path,
                xtc_pred=output_xtc_path,
                output_dir=TCR_output_folder,
                region_metric_config=region_metric_config,
                global_ranges=global_ranges,
                closest_frames=closest_frames,
                save_aligned=save_aligned,
                specific_region=specific_region,
                fit_on=fit_on
            )

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            with open(error_log, "a") as ef:
                ef.write(f"Error processing {folder}: {e}\n")
            continue

    # after collect phase, we can already compute best methods (Mantel etc.)
    #calc_best_metric(output_dir_all, rank_by="mantel.gt.r")


def run_all_TCRs_pmf(
    output_dir_all,
    CONFIG_PATH,
    global_ranges,
    closest_frames=False,
    specific_TCR=None,
    specific_region=None,
):
    """
    Phase 2: go over all TCRs and run PMF/JSD using global bin ranges.
    """
    with CONFIG_PATH.open() as f:
        region_metric_config = yaml.safe_load(f)

    for folder in os.listdir(output_dir_all):
        try:
            TCR_output_folder = os.path.join(output_dir_all, folder)
            if not os.path.isdir(TCR_output_folder):
                continue

            TCR_NAME_full = folder
            TCR_NAME = TCR_NAME_full.split("_")[0]

            if specific_TCR is not None and TCR_NAME != specific_TCR:
                continue

            print(f"[PMF] Processing TCR: {TCR_NAME}")
            run_pmf_for_TCR(
                TCR_output_folder=TCR_output_folder,
                region_metric_config=region_metric_config,
                global_ranges=global_ranges,
                closest_frames=closest_frames,
                specific_region=specific_region,
            )

        except Exception as e:
            print(f"[PMF] Error processing {folder}: {e}")
            # PMF errors don't go into the same error_log as collect; up to you.


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    fit_on="concat"
    #model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
    model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
    #for model_name in ["dig_with_init_cdr_mask_trA0.2_rotA0.2_trB0.1_rotB0.1","dig_with_init_trA0.2_rotA0.2_trB0.1_rotB0.1"]:
    #for model_name in ["Alphaflow_with_linker", "Dig_vanilla", "baseline_sampled_200_frames_1"]:
    for model_name in ["bioemu_filter", "bioemu_no_filter", "framedipt"]:
        config_modes = [
            #("config_assess_modes_all_ca_pca.yaml", "all_ca_pca"),
            #("config_assess_modes_all_ca_kpca_cosine.yaml", "all_ca_kpca_cosine"),
            ("config_assess_modes.yaml", "default"),
        ]

        for config_path, config_name in config_modes:
            CONFIG_PATH = HERE / config_path

            base_out = f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments_aaron/{config_name}"
            os.makedirs(base_out, exist_ok=True)

            output_dir_all = os.path.join(base_out, model_name)
            os.makedirs(output_dir_all, exist_ok=True)

            error_log = os.path.join(output_dir_all, "error.txt")

            # ---- PHASE 1: collect embeddings & global ranges ----
            global_ranges = {}
            if model_name == "AF3" or model_name=="framedipt":
                linker=""
            else:
                linker="GGGGS" * 3
            run_all_TCRs_collect(
                model_folder=model_folder,
                model_name=model_name,
                output_dir_all=output_dir_all,
                error_log=error_log,
                CONFIG_PATH=CONFIG_PATH,
                global_ranges=global_ranges,
                closest_frames=False,
                save_aligned=False,
                specific_TCR=None,
                specific_region=None,
                linker=linker,
                fit_on=fit_on
            )
            print("Global ranges after COLLECT phase:", global_ranges)

            # ---- PHASE 2: PMF / JSD with global bins per region ----
            run_all_TCRs_pmf(
                output_dir_all=output_dir_all,
                CONFIG_PATH=CONFIG_PATH,
                global_ranges=global_ranges,
                closest_frames=False,
                specific_TCR=None,
                specific_region=None,
            )
            run_make_tables(output_dir_all)

