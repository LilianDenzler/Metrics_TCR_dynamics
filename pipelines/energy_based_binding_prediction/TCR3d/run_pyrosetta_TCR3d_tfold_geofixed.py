import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from pipelines.energy_based_binding_prediction.run_pyrosetta import run_folder
from pipelines.energy_based_binding_prediction.TCR3d.run_pyrosetta_TCR3d import *
from pipelines.TCR_pMHC_geo.change_tfold_to_real_geometry import main as change_geo
from pipelines.TCR_pMHC_geo.change_tfold_to_real_geometry import get_metrics


def make_geofixed_tcr3d(changed_dir, tfold_dir, true_pdb, out_base):
    change_geo(changed_dir, tfold_dir, true_pdb, out_base, aligned_dir=None)
    get_metrics(true_pdb, changed_dir, out_base,error_file=None,aligned_dir=None)


changed_dir = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold_geochanged"
changed_dir = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold_geochanged"
tfold_dir   = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold"
true_pdb    = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes"

fixed_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold_geochanged_openmm_minimised"
prefix="tcr3d_tfold_geochanged"
vis_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/energy_based_binding_prediction/TCR3d/{prefix}_Rosetta_Results"
os.makedirs(fixed_dir, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)
make_geofixed_tcr3d(changed_dir, tfold_dir, true_pdb, vis_folder)

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
            base_dir=changed_dir,
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
    merge_results(TCR3d_tsv_path="/mnt/larry/lilian/DATA/TCR3d_datasets/tcr_complexes_data.tsv",  out_csv=out_csv, vis_folder=os.path.join(vis_folder,f"minimisation_{mode}_plots"))
