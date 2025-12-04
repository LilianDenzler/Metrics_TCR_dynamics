################################################################################
# This script processes TCR MD simulations and computes different dimensionality reduction metrics.
# For each alignmnet and assessment region selection a different dimensionality reduction may be best.
# The script identifies the best dimensionality reduction method for each region.
# all of these methods are fit on the ground truth MD only and the metrics for how good an embedding method is
# are computed by comparing the predicted MD embedding to the ground truth MD.
# We use the trustworthiness metric to rank the different methods as well as the mantel test correlation.
# Using these metrics we select the best dimensionality reduction method for each region.
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
import os
import numpy as np
import pandas as pd
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.scoring.rmsd_tm import rmsd_tm
from TCR_Metrics.pipelines.metrics_for_outputs.run_test import  assess_screening, calc_best_metric

def align_to_self(gttcr_pair, region_names, atom_names, outdir):
    gt_traj_aligned, rmsd=gttcr_pair.traj.align_to_ref(gttcr_pair,
                        region_names=region_names,
                        atom_names=atom_names,
                        inplace=False)

    (n, k, rmsd_A, tm)=rmsd_tm(gt_traj_aligned, gttcr_pair, regions=region_names, atoms={"CA"})
    print(f"RMSD: {rmsd_A}, TM: {tm}")
    #write a txt file with the RMSD and TM
    with open(os.path.join(outdir,"rmsd_tm.txt"), "w") as f:
        f.write(f"Aligned on {region_names} and atoms {atom_names}\n")
        f.write(f"alignment results: mean_RMSD: {np.mean(rmsd_A)}, mean_TM: {np.mean(tm)}\n")
    return gt_traj_aligned


def run_one_TCR(pdb_gt, xtc_gt, output_dir):
    gttcr = TCR(
        input_pdb=pdb_gt,
        traj_path=xtc_gt,         # or an XTC/DCD if you have one
        contact_cutoff=5.0,
        min_contacts=50,
        legacy_anarci=True
    )
    gttcr_pair=gttcr.pairs[0]

    #alignment of each CDR seperatly and assessment
    for region in ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]:
        os.makedirs(f"{output_dir}/{region}", exist_ok=True)
        gt_traj_aligned=align_to_self(gttcr_pair, [region], atom_names={"CA","C","N"}, outdir=f"{output_dir}/{region}")
        assess_screening(gt_traj_aligned, None, regions=[region],outdir_base=f"{output_dir}/{region}")

    #alignment of the framework region:
    for aligned_variable_domain in ["A_variable","B_variable"]:
        name_domain=aligned_variable_domain
        os.makedirs(f"{output_dir}/{name_domain}", exist_ok=True)
        gt_traj_aligned=align_to_self(gttcr_pair, [region], atom_names={"CA","C","N"}, outdir=f"{output_dir}/{name_domain}")
        rmsd_tm_df=pd.DataFrame(columns=["Region","RMSD","TM"])
        rows=[]
        for region in ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3", aligned_variable_domain]:
            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=[region], atoms={"CA"})
            rows.append({"Region": f"{region}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})
            os.makedirs(f"{output_dir}/{name_domain}/{region}", exist_ok=True)
            assess_screening( gt_traj_aligned, None, regions=[region], outdir_base=f"{output_dir}/{name_domain}/{region}")

        # Multi-region (combined) loop
        combined_regions = [
            ["A_CDR1","A_CDR2","A_CDR3"],
            ["B_CDR1","B_CDR2","B_CDR3"],
            ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"],
        ]

        for region_list in combined_regions:
            label = ''.join(region_list)
            n, k, rmsd_A, tm = rmsd_tm(gt_traj_aligned, gttcr_pair, regions=region_list, atoms={"CA"})
            rows.append({"Region": f"{label}gt_to_gt", "RMSD": float(np.mean(rmsd_A)), "TM": float(np.mean(tm))})
            os.makedirs(f"{output_dir}/{name_domain}/{label}", exist_ok=True)
            assess_screening(gt_traj_aligned, None, regions=region_list, outdir_base=f"{output_dir}/{name_domain}/{label}")

        # Build once and save
        rmsd_tm_df = pd.DataFrame(rows, columns=["Region", "RMSD", "TM"])
        rmsd_tm_df.to_csv(f"{output_dir}/{name_domain}/rmsd_tm_summary.csv", index=False)
    calc_best_metric(outdir_base=output_dir, rank_by="mantel.gt.r")

def evaluate_all_outs(output_dir_all):
    for folder in os.listdir(output_dir_all):
        TCR_NAME=folder
        print(f"Evaluating TCR: {TCR_NAME}")
        TCR_output_folder=os.path.join(output_dir_all,TCR_NAME)


if __name__ == "__main__":
    output_dir_all="/workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_analysis_out"
    os.makedirs(output_dir_all,exist_ok=True)
    for folder in os.listdir("/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig3"):
        TCR_NAME=folder
        print(f"Processing TCR: {TCR_NAME}")
        TCR_output_folder=os.path.join(output_dir_all,TCR_NAME)
        os.makedirs(TCR_output_folder,exist_ok=True)
        run_one_TCR(
            pdb_gt=f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}.pdb",
            xtc_gt=f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}_Prod.xtc",
            output_dir=TCR_output_folder
        )