import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
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
from pipelines.asses_model_ouput import run_one_TCR
import os
from pathlib import Path


def test():
    output_dir_all="/workspaces/Graphormer/TCR_Metrics/experiments/output_knapp_vs_cory"
    TCR_NAME="A6prmtop_first_frame"
    print(f"Processing TCR: {TCR_NAME}")
    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME)
    os.makedirs(TCR_output_folder,exist_ok=True)

    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME,"cory_vs_knapp_unbound")
    os.makedirs(TCR_output_folder,exist_ok=True)
    run_one_TCR(
        pdb_gt="/mnt/larry/lilian/DATA/Cory_data/A6/A6.pdb",
        xtc_gt="/mnt/larry/lilian/DATA/Cory_data/A6/Prod_Concat_A6_CMD.xtc",
        pdb_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.pdb",
        xtc_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.xtc",
        output_dir=TCR_output_folder,
        fit_on="concat"
    )

    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME,"cory_vs_knapp_unbound_fitcory")
    os.makedirs(TCR_output_folder,exist_ok=True)
    run_one_TCR(
        pdb_gt="/mnt/larry/lilian/DATA/Cory_data/A6/A6.pdb",
        xtc_gt="/mnt/larry/lilian/DATA/Cory_data/A6/Prod_Concat_A6_CMD.xtc",
        pdb_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.pdb",
        xtc_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.xtc",
        output_dir=TCR_output_folder,
        fit_on="gt"
    )
    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME,"cory_vs_knapp_unbound_fitknapp")
    os.makedirs(TCR_output_folder,exist_ok=True)
    run_one_TCR(
        pdb_gt="/mnt/larry/lilian/DATA/Cory_data/A6/A6.pdb",
        xtc_gt="/mnt/larry/lilian/DATA/Cory_data/A6/Prod_Concat_A6_CMD.xtc",
        pdb_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.pdb",
        xtc_pred=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.xtc",
        output_dir=TCR_output_folder,
        fit_on="pred"
    )

    quit()
    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME,"cory_vs_dig")
    os.makedirs(TCR_output_folder,exist_ok=True)
    input_unlinked_pdb_path = os.path.join(TCR_output_folder,"unlinked_dig.pdb")
    output_xtc_path=os.path.join(TCR_output_folder,"unlinked_dig.xtc")
    dig_output_dir = f"/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig3/{TCR_NAME}/dig_vanilla"
    input_pdb=f"/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig3/{TCR_NAME}/tcr_variable_linked.pdb"
    linker="GGGGS"*3
    output_xtc_path, input_unlinked_pdb_path=process_output(dig_output_dir, input_pdb,linker, output_xtc_path,input_unlinked_pdb_path)


    run_one_TCR(
        pdb_gt=f"/mnt/larry/lilian/DATA/Cory_data/A6/A6.pdb",
        xtc_gt=f"/mnt/larry/lilian/DATA/Cory_data/A6/Prod_Concat_A6_CMD.xtc",
        pdb_pred=input_unlinked_pdb_path,
        xtc_pred=output_xtc_path,
        output_dir=TCR_output_folder
    )

    TCR_output_folder=os.path.join(output_dir_all,TCR_NAME,"knapp_vs_dig")
    os.makedirs(TCR_output_folder,exist_ok=True)
    run_one_TCR(
        pdb_gt=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.pdb",
        xtc_gt=f"/mnt/larry/lilian/DATA/Unbound_Bound/TCR_only_nowater/TCR/combined_runs/A6_combined.xtc",
        pdb_pred=input_unlinked_pdb_path,
        xtc_pred=output_xtc_path,
        output_dir=TCR_output_folder
    )
if __name__ == "__main__":
    test()