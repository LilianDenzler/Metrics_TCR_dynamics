import sys
sys.path.append("/workspaces/Graphormer")
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from Bio.PDB import PDBIO
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from TCR_TOOLS.geometry.__init__ import DATA_PATH
from TCR_TOOLS.__init__ import CDR_FR_RANGES,VARIABLE_RANGE, A_FR_ALIGN, B_FR_ALIGN
#from ..testscript import run_folder
from TCR_Metrics.pipelines.TCR_pMHC_geo.plot_distributions import plot_distributions_from_csv


# -----------------------------
# Main pipeline
# -----------------------------
def main(input_dir, out_base):
    error_file = os.path.join(out_base, "calc_geometry_errors.txt")
    if os.path.exists(error_file):
        os.remove(error_file)


    all_rows = []

    for input_file in sorted(os.listdir(input_dir)):
        if not input_file.endswith(".pdb"):
            continue

        input_path = os.path.join(input_dir, input_file)
        input_name = input_file.replace(".pdb", "")

        try:
            true_complex = TCRpMHC(
            input_pdb=input_path,
            MHC_a_chain_id="A",
            MHC_b_chain_id="B",
            Peptide_chain_id="C"
            )
            results = true_complex.calc_geometry(out_path=None)
            all_rows.append({
                "structure": input_name,
                **results,
            })
        except Exception as e:
            with open(error_file, "a") as ef:
                ef.write(f"{input_path}\n")
            print(f"[ERROR] Failed on {input_file}: {e}")
            continue
    all_results_df= pd.DataFrame(all_rows)
    out_csv = os.path.join(out_base, "geometry_results.csv")
    all_results_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote geometry results to {out_csv}")
    return all_results_df



if __name__ == "__main__":
    input_dir = "/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes"
    out_base = "/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_TCR3d_calc_geometry"
    os.makedirs(out_base, exist_ok=True)
    all_results_df=main(input_dir, out_base)
    #onpy fails
    #read errors and rerun on those files
    error_file = os.path.join(out_base, "calc_geometry_errors.txt")
    second_round_error_file=os.path.join(out_base, "calc_geometry_errors_run2.txt")
    if os.path.exists(error_file):
        with open(error_file, "r") as ef:
            error_files = [line.strip() for line in ef.readlines()]
        if error_files:
            print(f"[INFO] Rerunning geometry calculations on {len(error_files)} error files.")
            for error_file_path in error_files:
                input_name = os.path.basename(error_file_path).replace(".pdb", "")
                try:
                    true_complex = TCRpMHC(
                        input_pdb=error_file_path,
                        MHC_a_chain_id="A",
                        MHC_b_chain_id="B",
                        Peptide_chain_id="C",
                        TCR_a_chain_id="D",
                        TCR_b_chain_id="E")
                    #write to pdb
                    results = true_complex.calc_geometry(out_path=None)
                    #append to all_results_df
                    all_results_df = pd.concat([all_results_df, pd.DataFrame([{"structure": input_name,**results}])])
                except Exception as e:
                        print(f"[ERROR] Rerun without TCR chains failed on {input_name}: {e}")
                        with open(second_round_error_file, "a") as ef2:
                            ef2.write(f"{error_file_path}\n")
                        continue
            out_csv = os.path.join(out_base, "geometry_results.csv")
            all_results_df.to_csv(out_csv, index=False)
            print(f"[OK] Wrote rerun geometry results to {out_csv}")