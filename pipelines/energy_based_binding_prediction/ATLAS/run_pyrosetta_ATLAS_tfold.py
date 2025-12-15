import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from pipelines.energy_based_binding_prediction.run_pyrosetta import run_folder
from pipelines.energy_based_binding_prediction.ATLAS.run_pyrosetta_ATLAS import *



base_dir="/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_tfold"
fixed_dir="/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_tfold_openmm_minimised"

os.makedirs(fixed_dir, exist_ok=True)
prefix="atlas_truepdb_tfold"
vis_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/energy_based_binding_prediction/ATLAS/{prefix}_Rosetta_Results"
os.makedirs(vis_folder, exist_ok=True)
#ATLAS standardises, so that the chains are always named the same way
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
    out_csv = os.path.join(vis_folder,f"{prefix}_{mode}.csv")
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
    merge_results(ATLAS_tsv_path="/mnt/larry/lilian/DATA/ATLAS/ATLAS.tsv", out_csv=out_csv, vis_folder=os.path.join(vis_folder,f"minimisation_{mode}_plots"))
