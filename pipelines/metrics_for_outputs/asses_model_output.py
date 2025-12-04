from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.ops import *
from TCR_TOOLS.scoring.embeddings import run_coords, run_ca_dist, run_dihedrals
from TCR_TOOLS.core import io
from TCR_TOOLS.scoring.rmsd_tm import rmsd_tm
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.ops import *
import pandas as pd
from TCR_TOOLS.scoring.PCA_methods import pca_project_two as pca_project

from dig_output_handler.process_output import process_output

import os
from pathlib import Path

def assess(gt_traj_aligned, traj_aligned, regions,outdir_base="/workspaces/Graphormer/TCR_Metrics/test/test", fit_on="concat"):
    # Dihedrals + dPCA (fit on GT)
    for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
        Zg, Zp, info, tw, mt = run_dihedrals.run(
            gt_traj_aligned, traj_aligned,
            outdir=f"{outdir_base}/dihed_{mode}",
            regions=regions,lag=5,
            reducer=mode, fit_on=fit_on, dihedrals=("phi","psi"), encode="sincos",
            subsample=100,mantel_perms=999
        )
    # Coordinates + PCA (concat), EVR + metrics
    for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
        Zg, Zp, info, tw, mt = run_coords.run(
            tv_gt=gt_traj_aligned, tv_pred=traj_aligned,
            outdir=f"{outdir_base}/coords_{mode}",
            regions=regions, lag=5,
            atoms=("CA","C","N"),
            reducer=mode, n_components=2, fit_on=fit_on, use_gt_scaler=True,
            subsample=100, mantel_method="spearman",mantel_perms=999
        )
    # CA distances + kPCA (cosine)
    for mode in ["pca", "kpca_rbf", "kpca_cosine", "kpca_poly","diffmap", "tica"]:
        Zg, Zp, info, tw, mt = run_ca_dist.run(
            gt_traj_aligned, traj_aligned,
            outdir=f"{outdir_base}/ca_{mode}",
            regions=regions, fit_on=fit_on,
            reducer=mode, n_components=2, max_pairs=20000,
            subsample=100, lag=5,mantel_perms=999
        )


def align( digtcr_pair, gttcr_pair,region_names,atom_names):
    traj_aligned, rmsd=digtcr_pair.traj.align_to_ref(gttcr_pair,
                        region_names=region_names,
                        atom_names={"CA","C","N"},
                        inplace=False)

    gt_traj_aligned, rmsd=gttcr_pair.traj.align_to_ref(gttcr_pair,
                        region_names=region_names,
                        atom_names={"CA","C","N"},
                        inplace=False)

    (n, k, rmsd_A, tm)=rmsd_tm(traj_aligned, gttcr_pair, regions=["A_CDR3"], atoms={"CA"})
    (n, k, rmsd_A, tm)=rmsd_tm(gt_traj_aligned, gttcr_pair, regions=["A_CDR3"], atoms={"CA"})
    print(f"A_CDR3 RMSD: {rmsd_A}, TM: {tm}")
    #write a txt file with the RMSD and TM
    with open("rmsd_tm.txt", "w") as f:
        f.write(f"Aligned on {region_names} and atoms {atom_names}\n")
        f.write(f"alignment results: mean_RMSD: {np.mean(rmsd_A)}, mean_TM: {np.mean(tm)}\n")
    return traj_aligned, gt_traj_aligned

def run_one_TCR(pdb_gt, xtc_gt, pdb_pred, xtc_pred,output_dir, fit_on="concat"):
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
        traj_aligned, gt_traj_aligned=align(digtcr_pair, gttcr_pair,region_names=[region],atom_names={"CA","C","N"})

        os.makedirs(f"{output_dir}/{region}", exist_ok=True)
        assess(gt_traj_aligned, traj_aligned, regions=[region],outdir_base=f"{output_dir}/{region}", fit_on=fit_on)

    #alignment of the whole variable domain and assessment
    for aligned_variable_domain in ["A_variable","B_variable"]:
        name_domain=aligned_variable_domain
        os.makedirs(f"{output_dir}/{name_domain}", exist_ok=True)
        traj_aligned, gt_traj_aligned=align(digtcr_pair, gttcr_pair,region_names=[aligned_variable_domain],atom_names={"CA","C","N"})
        rmsd_tm_df=pd.DataFrame(columns=["Region","RMSD","TM"])
        rows=[]
        for region in ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3", aligned_variable_domain]:
            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rows.append({"Region": f"{region}pred_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rows.append({"Region": f"{region}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            os.makedirs(f"{output_dir}/{name_domain}/{region}", exist_ok=True)
            assess(gt_traj_aligned, traj_aligned, regions=[region], outdir_base=f"{output_dir}/{name_domain}/{region}", fit_on=fit_on)

        # Multi-region (combined) loop
        combined_regions = [
            ["A_CDR1","A_CDR2","A_CDR3"],
            ["B_CDR1","B_CDR2","B_CDR3"],
            ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"],
        ]

        for region_list in combined_regions:
            label = ''.join(region_list)
            n, k, rmsd_A, tm = rmsd_tm(traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rows.append({"Region": f"{label}pred_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rows.append({"Region": f"{label}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})

            os.makedirs(f"{output_dir}/{name_domain}/{label}", exist_ok=True)
            assess(gt_traj_aligned, traj_aligned, regions=region_list, outdir_base=f"{output_dir}/{name_domain}/{label}", fit_on=fit_on)

        # Build once and save
        rmsd_tm_df = pd.DataFrame(rows, columns=["Region", "RMSD", "TM"])
        rmsd_tm_df.to_csv(f"{output_dir}/{name_domain}/rmsd_tm_summary.csv", index=False)
