import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from pipelines.energy_based_binding_prediction.run_pyrosetta import run_folder

dataset_type="highest_count"
for neg_pos in ["neg","pos"]:
    base_dir=f"/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/A0301_KLGGALQAK/{dataset_type}/{neg_pos}"
    fixed_dir=f"/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/A0301_KLGGALQAK/{dataset_type}/{neg_pos}_openmm_minimised"

    os.makedirs(fixed_dir, exist_ok=True)
    prefix="omics_A0301_KLGGALQAK_tfold"
    vis_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/process_TRAIT/{prefix}_Rosetta_Results/{dataset_type}"
    os.makedirs(vis_folder, exist_ok=True)
    #tfold standardises, so that the chains are always named the same way
    tcr_alpha="A"
    tcr_beta="B"
    mhc_alpha="M"
    mhc_beta="N"
    peptide="P"
    modes = ["none", "rosetta_sidechain", "openmm"]#,"rosetta_full"]
    for mode in modes:
        if mode in ["rosetta_sidechain", "rosetta_full", "none"]:
            num_workers=4
        else:
            num_workers=2 #do 1 if using GPU
        out_csv = os.path.join(vis_folder,f"{prefix}_{mode}_{neg_pos}.csv")
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
