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
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.ops import *
from TCR_TOOLS.aligners.aligning import align_MD_same_TCR_profit
import pandas as pd
from TCR_TOOLS.scoring.PCA_methods import pca_project_two as pca_project
from dig_output_handler.process_output import process_output
from TCR_Metrics.pipelines.metrics_for_outputs.analyse_metrics import get_best_metric
from TCR_Metrics.pipelines.calc_metrics_model_outputs.visualise_closest_frames_pymol import render_zoomed_morph_image
from TCR_Metrics.pipelines.calc_metrics_model_outputs.Standardise_by_RMSD import rmsd_calibration_factor
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import oriol_analysis
import os
from pathlib import Path
import yaml
import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent


def assess_screening(gt_traj_aligned, traj_aligned, regions,outdir_base="/workspaces/Graphormer/TCR_Metrics/test/test"):
    for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
            # Dihedrals
            Zg, Zp, info, tw, mt = run_dihedrals.run(
                gt_traj_aligned, traj_aligned,
                outdir=f"{outdir_base}/dihed_{mode}",
                regions=regions,lag=5,
                reducer=mode, fit_on="gt", dihedrals=("phi","psi"), encode="sincos",
                subsample=100,mantel_perms=999
            )
            if Zp is None:
                pass
            else:
                jsd = jensen_shannon_from_embeddings(Zg, Zp, base=2)
                print(f"Jensen–Shannon divergence (dihed, mode={mode}): {jsd:.4f}")
                #write to a txt file
                with open(f"{outdir_base}/dihed_{mode}/jsd_dihed_{mode}.txt", "w") as f:
                    f.write(f"Jensen–Shannon divergence : {jsd:.4f}\n")


            Zg, Zp, info, tw, mt = run_ca_dist.run(
            gt_traj_aligned, traj_aligned,
            outdir=f"{outdir_base}/ca_{mode}",
            regions=regions, fit_on="gt",
            reducer=mode, n_components=2, max_pairs=20000,
            subsample=100, lag=5,mantel_perms=999)
            if Zp is None:
                pass
            else:
                jsd = jensen_shannon_from_embeddings(Zg, Zp, base=2)
                print(f"Jensen–Shannon divergence (ca, mode={mode}): {jsd:.4f}")
                #write to a txt file
                with open(f"{outdir_base}/ca_{mode}/jsd_ca_{mode}.txt", "w") as f:
                    f.write(f"Jensen–Shannon divergence : {jsd:.4f}\n")

            Zg, Zp, info, tw, mt = run_coords.run(
            tv_gt=gt_traj_aligned, tv_pred=traj_aligned,
            outdir=f"{outdir_base}/coords_{mode}",
            regions=regions, lag=5,
            atoms=("CA","C","N"),
            reducer=mode, n_components=2, fit_on="gt", use_gt_scaler=False,
            subsample=100, mantel_method="spearman",mantel_perms=999)
            if Zp is None:
                pass
            else:
                jsd = jensen_shannon_from_embeddings(Zg, Zp, base=2)
                print(f"Jensen–Shannon divergence (coords, mode={mode}): {jsd:.4f}")
                #write to a txt file
                with open(f"{outdir_base}/coords_{mode}/jsd_coords_{mode}.txt", "w") as f:
                    f.write(f"Jensen–Shannon divergence : {jsd:.4f}\n")


def save_MD_frames_closest_to_each_model_point_with_metrics(
    Zg,
    Zp,
    gt_traj_aligned,
    traj_aligned,
    outdir: str,
    region_names,
    atom_names,
    vis=False
):
    """
    For each model datapoint (row in Zp), find closest GT point in Zg,
    save the corresponding MD (GT) frame and model frame to PDB,
    and compute RMSD + TM over the selected atoms.

    Assumes:
      - Zg.shape[0] == n_frames_gt_used
      - Zp.shape[0] == n_frames_model_used
      - gt_traj_aligned._traj.n_frames >= Zg.shape[0]
      - traj_aligned._traj.n_frames >= Zp.shape[0]

    Also saves:
      - closest_frame_metrics.csv              (per-pair table)
      - closest_frame_metrics_summary.csv      (mean/std/min/max/n for RMSD & TM)
      - closest_frame_metrics_hist.png         (histograms of RMSD and TM)
    """

    os.makedirs(outdir, exist_ok=True)

    Zg = np.asarray(Zg)
    Zp = np.asarray(Zp)

    n_gt, d1 = Zg.shape
    n_model, d2 = Zp.shape
    assert d1 == d2, "Zg and Zp must have same embedding dimension"

    # unwrap TrajectoryView -> mdtraj.Trajectory if needed
    gt_md = getattr(gt_traj_aligned, "mdtraj", getattr(gt_traj_aligned, "_traj", gt_traj_aligned))
    pred_md = getattr(traj_aligned, "mdtraj", getattr(traj_aligned, "_traj", traj_aligned))

    rows = []

    for model_idx in range(n_model):
        z_model = Zp[model_idx]

        # Euclidean distance in embedding space
        dists = np.linalg.norm(Zg - z_model[None, :], axis=1)
        gt_idx = int(np.argmin(dists))  # closest GT frame index
        print(f"Model idx {model_idx} closest to GT idx {gt_idx} with dist {dists[gt_idx]:.4f}")

        # Sanity cap in case embeddings come from subsampled frames
        if gt_idx >= gt_md.n_frames:
            print(f"[WARN] gt_idx {gt_idx} >= gt_md.n_frames; skipping.")
            continue
        if model_idx >= pred_md.n_frames:
            print(f"[WARN] model_idx {model_idx} >= pred_md.n_frames; skipping.")
            continue

        # Extract single-frame trajectories
        gt_frame = gt_md[gt_idx]
        model_frame = pred_md[model_idx]

        # Save frames to PDBs in outdir
        gt_pdb_path = os.path.join(outdir, f"gt_frame_{gt_idx}_model_{model_idx}.pdb")
        model_pdb_path = os.path.join(outdir, f"model_frame_{model_idx}.pdb")

        gt_frame.save_pdb(gt_pdb_path)
        model_frame.save_pdb(model_pdb_path)

        # Reload as mdtraj Trajectory (1 frame each, just to reuse run_rmsd_tm)
        mov = md.load(model_pdb_path, top=model_pdb_path)
        ref = md.load(gt_pdb_path, top=gt_pdb_path)

        # This should return scalars (or length-1 arrays) for RMSD (Å) and TM
        n, k, rmsd_A, tm = run_rmsd_tm(mov, ref, regions=region_names, atoms=atom_names)

        # If your run_rmsd_tm returns arrays, grab the first element
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

    # Save detailed table
    metrics_csv = os.path.join(outdir, "closest_frame_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"Saved closest-frame RMSD/TM table to: {metrics_csv}")

    # ---- Summary stats (mean, std, min, max, n) ----
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

    # Print averages to stdout
    mean_rmsd = stats_df.loc[stats_df["metric"] == "rmsd", "mean"].iloc[0]
    mean_tm = stats_df.loc[stats_df["metric"] == "tm_score", "mean"].iloc[0]
    print(f"Average RMSD over all model points: {mean_rmsd:.4f} Å")
    print(f"Average TM-score over all model points: {mean_tm:.4f}")

    # ---- Plots: spread of RMSD and TM ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # RMSD histogram
    axes[0].hist(df["rmsd"], bins=30)
    axes[0].set_title("RMSD distribution")
    axes[0].set_xlabel("RMSD (Å)")
    axes[0].set_ylabel("Count")

    # TM-score histogram
    axes[1].hist(df["tm_score"], bins=30)
    axes[1].set_title("TM-score distribution")
    axes[1].set_xlabel("TM-score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plot_path = os.path.join(outdir, "closest_frame_metrics_hist.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved RMSD/TM distribution plot to: {plot_path}")
    if vis==True:
        render_zoomed_morph_image(
            closest_structures_folder=outdir,
            output_image=outdir+f"closest_frames_{''.join(region_names)}",
            regions=region_names,
            zoom_buffer=10 # Keeping buffer for cmd.zoom call
            )
        print("Saved zoomed morph image to:", outdir+f"closest_frames_{region_names} ")

    #delete the pdb files to save space
    for file in os.listdir(outdir):
        if file.endswith(".pdb"):
            os.remove(os.path.join(outdir, file))



def assess(assess_mode,gt_traj_aligned, traj_aligned, regions,outdir_base="/workspaces/Graphormer/TCR_Metrics/test/test", closest_frames=True,global_range=None):
    # Dihedrals + dPCA (fit on GT)
    print("Running assess mode:", assess_mode,"!")
    if assess_mode=="all":
        assess_screening(gt_traj_aligned, traj_aligned, regions,outdir_base)
    assess_type=assess_mode.split("_")[0]
    mode=assess_mode.split("_")[1:]
    mode="_".join(mode)
    print("Assess mode:", assess_type)
    print("mode",mode)
    #for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
    if assess_type=="dihed":
        outdir=f"{outdir_base}/dihed_{mode}",
        Zg, Zp, info, tw, mt = run_dihedrals.run(
            gt_traj_aligned, traj_aligned,
            outdir=outdir,
            regions=regions,lag=5,
            reducer=mode, fit_on="concat", dihedrals=("phi","psi"), encode="sincos",
            subsample=100,mantel_perms=999
        )
    # Coordinates + PCA (concat), EVR + metrics
    #for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
    if assess_type=="coords":
        outdir=f"{outdir_base}/coords_{mode}",
        Zg, Zp, info, tw, mt = run_coords.run(
            tv_gt=gt_traj_aligned, tv_pred=traj_aligned,
            outdir=outdir,
            regions=regions, lag=5,
            atoms=("CA","C","N"),
            reducer=mode, n_components=2, fit_on="concat", use_gt_scaler=False,
            subsample=100, mantel_method="spearman",mantel_perms=999
        )
    # CA distances + kPCA (cosine)
    #for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
    if assess_type=="ca":
        outdir=f"{outdir_base}/ca_{mode}"
        Zg, Zp, info, tw, mt = run_ca_dist.run(
            gt_traj_aligned, traj_aligned,
            outdir=outdir,
            regions=regions, fit_on="concat",
            reducer=mode, n_components=2, max_pairs=20000,
            subsample=100, lag=5,mantel_perms=999
        )
    k_scale, corr=rmsd_calibration_factor(Zg,gt_traj_aligned,region_names=regions, atom_names={"CA"})
    if corr<0.6:
        input(f"RMSD calibration factor k: {k_scale}, corr: {corr}. Press Enter to continue...")
        raise ValueError(f"Low correlation ({corr}) between embedding distance and RMSD; cannot standardise by RMSD.")
    Zg_scaled = Zg * k_scale
    Zp_scaled = Zp * k_scale
    center = Zg_scaled.mean(axis=0)          # center taken from GT only
    Zg_scaled = Zg_scaled - center
    Zp_scaled = Zp_scaled - center         # subtract the same center

    xbins=50
    ybins=50
    temperature = 300
    outdir_standardised=f"{outdir}/standardised_by_RMSD"
    os.makedirs(outdir_standardised, exist_ok=True)
    plot_pca(Zg, Zp,  None,None,f"{outdir}/pca_projection.png")
    plot_pca(Zg_scaled, Zp_scaled,  None,None,f"{outdir_standardised}/pca_projection_standardised.png")
    oriol_analysis(xbins, ybins,Zg, Zp, temperature, name="",outfolder=outdir, tv_gt=gt_traj_aligned, tv_pred=traj_aligned)
    oriol_analysis(xbins, ybins,Zg_scaled, Zp_scaled, temperature, name="",outfolder=outdir_standardised, tv_gt=gt_traj_aligned, tv_pred=traj_aligned, global_range=global_range)

    if closest_frames==True:
        closest_md_frames_dir=f"{outdir_base}/{assess_type}_{mode}/closest_frames"
        os.makedirs(closest_md_frames_dir, exist_ok=True)
        save_MD_frames_closest_to_each_model_point_with_metrics(
        Zg,
        Zp,
        gt_traj_aligned,
        traj_aligned,
        outdir=closest_md_frames_dir,
        region_names=regions,
        atom_names={"CA"}
        )


def calc_best_metric(outdir_base, rank_by="trust.gt"):
    best_methods={}
    for folder in Path(outdir_base).iterdir():
        if folder.is_dir():
            if "A_variable" in str(folder):
                for subfolder in Path(outdir_base).iterdir():
                    if subfolder.is_dir():
                        region=subfolder.name
                        best_method,score=get_best_metric(subfolder, rank_by=rank_by)
                        full_region="A_variable_"+region
                        best_methods[full_region]={"method":best_method,"score":score}
            else:
                region=folder.name
                best_method,score=get_best_metric(folder, rank_by=rank_by)
                best_methods[region]={"method":best_method,"score":score}
    print("\n=== Best methods per region ===")
    best_methods_df=pd.DataFrame.from_dict(best_methods, orient="index")
    print(best_methods_df)
    best_methods_df.to_csv(os.path.join(outdir_base,"best_methods_per_region_mantel.csv"))


def align( digtcr_pair, gttcr_pair,region_names,atom_names, outdir="", save_aligned=True):

    """align_MD_same_TCR_profit(
        gttcr_pair,
        digtcr_pair,
        region_names,
        atom_names,
        inplace= False,
        profit_bin= "/workspaces/Graphormer/ProFitV3.3/bin/profit",
    )"""

    gttcr_pair_ref=gttcr_pair.make_region_representative_pairview(
        region_names=region_names,
        atom_names={"CA", "C", "N"},
        out_pdb_path=os.path.join(outdir, "gt_align_ref_frame.pdb"))


    # then refine alignment on the specific region
    traj_aligned, rmsd = digtcr_pair.traj.align_to_ref(
        gttcr_pair_ref,
        region_names=region_names,
        atom_names={"CA", "C", "N"},
        inplace=False,
    )

    gt_traj_aligned, rmsd = gttcr_pair.traj.align_to_ref(
        gttcr_pair_ref,
        region_names=region_names,
        atom_names={"CA", "C", "N"},
        inplace=False,
    )

    # write the aligned trajectories to the outdir
    if save_aligned:
        """traj_aligned._traj.save_xtc(os.path.join(outdir, "pred_aligned.xtc"))
        traj_aligned._traj[0].save_pdb(os.path.join(outdir, "pred_aligned.pdb"))
        gt_traj_aligned._traj.save_xtc(os.path.join(outdir, "gt_aligned.xtc"))
        gt_traj_aligned._traj[0].save_pdb(os.path.join(outdir, "gt_aligned.pdb"))
        digtcr_pair.traj._traj.save_xtc(os.path.join(outdir, "pred_unaligned.xtc"))
        digtcr_pair.traj._traj[0].save_pdb(os.path.join(outdir, "pred_unaligned.pdb"))
        gttcr_pair.traj._traj.save_xtc(os.path.join(outdir, "gt_unaligned.xtc"))
        gttcr_pair.traj._traj[0].save_pdb(os.path.join(outdir, "gt_unaligned.pdb"))"""

        # save only the aligned regions in a separate pdb and xtc
        aligned_region_traj = traj_aligned.domain_subset(region_names, atom_names)
        aligned_region_traj[0].save_pdb(os.path.join(outdir, "pred_aligned_region.pdb"))
        aligned_region_traj.save_xtc(os.path.join(outdir, "pred_aligned_region.xtc"))

        aligned_region_gt = gt_traj_aligned.domain_subset(region_names, atom_names)
        aligned_region_gt[0].save_pdb(os.path.join(outdir, "gt_aligned_region.pdb"))
        aligned_region_gt.save_xtc(os.path.join(outdir, "gt_aligned_region.xtc"))

    # quick region-level RMSD/TM summary
    (n, k, rmsd_A, tm) = rmsd_tm(
        traj_aligned, gttcr_pair, regions=region_names, atoms={"CA"}
    )
    (n, k, rmsd_A_gt, tm_gt) = rmsd_tm(
        gt_traj_aligned, gttcr_pair, regions=region_names, atoms={"CA"}
    )
    print(f"{region_names} RMSD pred->gt: {rmsd_A}, TM: {tm}")
    print(f"{region_names} RMSD gt->gt: {rmsd_A_gt}, TM: {tm_gt}")

    with open(os.path.join(outdir, "rmsd_tm.txt"), "w") as f:
        f.write(f"Aligned on {region_names} and atoms {atom_names}\n")
        f.write(
            f"pred_to_gt: mean_RMSD: {np.mean(rmsd_A)}, mean_TM: {np.mean(tm)}\n"
        )
        f.write(
            f"gt_to_gt:   mean_RMSD: {np.mean(rmsd_A_gt)}, mean_TM: {np.mean(tm_gt)}\n"
        )
    return traj_aligned, gt_traj_aligned

def run_one_TCR(pdb_gt, xtc_gt, pdb_pred, xtc_pred,output_dir,region_metric_config,closest_frames=True, save_aligned=True, specific_region=None):
    gttcr = TCR(
        input_pdb=pdb_gt,
        traj_path=xtc_gt,         # or an XTC/DCD if you have one
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True
    )
    gttcr_pair=gttcr.pairs[0]
    # 3) Get sequences (original vs IMGT, all chains)
    print(get_sequence_dict(gttcr_pair.full_structure))


    try:

        digtcr = TCR(
        input_pdb=pdb_pred,
        traj_path=xtc_pred,         # or an XTC/DCD if you have one
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True
        )
        digtcr_pair = digtcr.pairs[0]
    except:
        digtcr = TCR(
        input_pdb=pdb_pred,
        traj_path=None,         # or an XTC/DCD if you have one
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True
        )
        digtcr_pair = digtcr.pairs[0]
        digtcr_pair.attach_trajectory(
            xtc_pred,
            region_names=None,
            atom_names={"CA","C","N"}
            )

    #alignment of each CDR seperatly and assessment
    for region in ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]:
        if specific_region is not None:
            if region!=specific_region:
                continue
        os.makedirs(f"{output_dir}/{region}", exist_ok=True)
        traj_aligned, gt_traj_aligned=align(digtcr_pair, gttcr_pair,region_names=[region],atom_names={"CA","C","N"}, outdir=f"{output_dir}/{region}", save_aligned=save_aligned)
        cfg_region = region_metric_config.get(region, {})
        assess_mode = cfg_region.get(region, "all")
        assess(assess_mode,gt_traj_aligned, traj_aligned, regions=[region],outdir_base=f"{output_dir}/{region}", closest_frames=closest_frames)

    #alignment of the whole variable domain and assessment
    for aligned_variable_domain in ["A_variable","B_variable"]:
        name_domain=aligned_variable_domain
        os.makedirs(f"{output_dir}/{name_domain}", exist_ok=True)
        traj_aligned, gt_traj_aligned=align(digtcr_pair, gttcr_pair,region_names=[aligned_variable_domain],atom_names={"CA","C","N"}, outdir=f"{output_dir}/{name_domain}", save_aligned=save_aligned)
        rmsd_tm_df=pd.DataFrame(columns=["Region","RMSD","TM"])
        rows=[]
        for region in ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3", aligned_variable_domain]:
            if specific_region is not None:
                if region!=specific_region:
                    continue
            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rows.append({"Region": f"{region}pred_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rows.append({"Region": f"{region}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            os.makedirs(f"{output_dir}/{name_domain}/{region}", exist_ok=True)
            cfg_region = region_metric_config.get(region, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            print(assess_mode)
            assess(assess_mode, gt_traj_aligned, traj_aligned, regions=[region], outdir_base=f"{output_dir}/{name_domain}/{region}", closest_frames=closest_frames)

        # Multi-region (combined) loop
        combined_regions = [
            ["A_CDR1","A_CDR2","A_CDR3"],
            ["B_CDR1","B_CDR2","B_CDR3"],
            ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"],
        ]

        for region_list in combined_regions:
            if specific_region is not None:
                if region_list!=specific_region:
                    continue
            label = ''.join(region_list)
            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rows.append({"Region": f"{label}pred_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rows.append({"Region": f"{label}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            os.makedirs(f"{output_dir}/{name_domain}/{label}", exist_ok=True)
            cfg_region = region_metric_config.get(label, {})
            assess_mode = cfg_region.get(aligned_variable_domain, "all")
            assess(assess_mode, gt_traj_aligned, traj_aligned, regions=region_list, outdir_base=f"{output_dir}/{name_domain}/{label}",closest_frames=closest_frames)

        # Build once and save
        if specific_region is not None:
            continue
        rmsd_tm_df = pd.DataFrame(rows, columns=["Region", "RMSD", "TM"])
        rmsd_tm_df.to_csv(f"{output_dir}/{name_domain}/rmsd_tm_summary.csv", index=False)

def run_all_TCRs(model_folder, model_name, output_dir_all,error_log,CONFIG_PATH, closest_frames=True, save_aligned=True, specific_TCR=None, specific_region=None):
    for folder in os.listdir(model_folder):
        try:
            if not os.path.isdir(os.path.join(model_folder,folder)):
                continue
            TCR_NAME_full=folder
            TCR_NAME=TCR_NAME_full.split("_")[0]
            if specific_TCR is not None:
                if TCR_NAME!=specific_TCR:
                    continue

            print(f"Processing TCR: {TCR_NAME}")
            TCR_output_folder=os.path.join(output_dir_all,TCR_NAME_full)
            os.makedirs(TCR_output_folder,exist_ok=True)
            ##if Path(os.path.join(TCR_output_folder,"B_variable/B_variable/ca_kpca_cosine/plot_pmf.png")).exists():
            ##    print(f"Skipping {TCR_NAME} as already processed.")
            #    continue

            input_unlinked_pdb_path = os.path.join(TCR_output_folder,"unlinked_dig.pdb")
            output_xtc_path=os.path.join(TCR_output_folder,"unlinked_dig.xtc")
            dig_output_dir = os.path.join(model_folder,TCR_NAME_full,model_name)
            input_pdb=os.path.join(dig_output_dir,os.listdir(dig_output_dir)[1])
            i=1
            while input_pdb.endswith(".pdb")==False:
                input_pdb=os.path.join(dig_output_dir,os.listdir(dig_output_dir)[i])
                i+=1

            print(input_pdb)
            linker="GGGGS"*3
            output_xtc_path, input_unlinked_pdb_path=process_output(dig_output_dir, input_pdb,linker, output_xtc_path,input_unlinked_pdb_path)
            # Config path (same folder as this script/module)
            with CONFIG_PATH.open() as f:
                region_metric_config = yaml.safe_load(f)
            pdb_gt=f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}.pdb"
            xtc_gt=f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}_Prod.xtc"
            if not os.path.exists(pdb_gt) or not os.path.exists(xtc_gt):
                print(f"Skipping {TCR_NAME} as ground truth files not found.")
                continue
            run_one_TCR(
                pdb_gt=pdb_gt,
                xtc_gt=xtc_gt,
                pdb_pred=input_unlinked_pdb_path,
                xtc_pred=output_xtc_path,
                output_dir=TCR_output_folder,
                region_metric_config=region_metric_config,
                closest_frames=closest_frames,
                save_aligned=save_aligned,
                specific_region=specific_region
            )
            calc_best_metric(TCR_output_folder, rank_by="mantel.gt.r")

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            with open(error_log, "a") as ef:
                ef.write(f"Error processing {folder}: {e}\n")
            continue



if __name__ == "__main__":
    for model_name in ["Alphaflow_with_linker", 'Dig_vanilla', 'baseline_sampled_200_frames_1']:
        config_modes=[("config_assess_modes_all_ca_pca.yaml","all_ca_pca"),
                    ("config_assess_modes_all_ca_kpca_cosine.yaml","all_ca_kpca_cosine"),
                    ("config_assess_modes.yaml","default")]
        for config_path, config_name in config_modes:
            CONFIG_PATH = HERE / config_path
            output_dir_all=f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments/{config_name}"
            os.makedirs(output_dir_all,exist_ok=True)
            output_dir_all=f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments/{config_name}/{model_name}/"
            os.makedirs(output_dir_all,exist_ok=True)
            model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
            #model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
            error_log=os.path.join(output_dir_all,"error.txt")
            #chnage these according to needs closest_frames=False, save_aligned=False
            #if closest_frames=True pdb frames in MD closest to each model point will be saved and assessed
            #if save_aligned=True aligned trajectories will be saved to output folder
            run_all_TCRs(model_folder, model_name, output_dir_all,error_log,CONFIG_PATH, closest_frames=False, save_aligned=False)
    """
    model_name="dig_with_init_cdr_mask_trA0.2_rotA0.2_trB0.1_rotB0.1"
    config_path="config_assess_modes_all_ca_kpca_cosine.yaml"
    config_name="all_ca_kpca_cosine"
    CONFIG_PATH = HERE / config_path
    output_dir_all=f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments/{config_name}"
    os.makedirs(output_dir_all,exist_ok=True)
    output_dir_all=f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments/{config_name}/{model_name}/"
    os.makedirs(output_dir_all,exist_ok=True)
    #model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
    model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
    error_log=os.path.join(output_dir_all,"error.txt")
    run_all_TCRs(model_folder, model_name, output_dir_all,error_log,CONFIG_PATH, closest_frames=False, save_aligned=False, specific_TCR="7EA6", specific_region="B_CDR3")
    #run_all_TCRs(model_folder, model_name, output_dir_all,error_log,CONFIG_PATH, closest_frames=True, save_aligned=True, specific_TCR="6OVN", specific_region="B_CDR3")
    """