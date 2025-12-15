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

def calc_RMSD_TM(align_sel, calc_sel, input_complex_obj, changed_complex_obj, aligned_input_path, aligned_changed_path):
    rmsd, tm = input_complex_obj.rmsd_to(
        changed_complex_obj,
        align_sel=align_sel,
        score_sel=calc_sel,
        atom_mode="CA",
        align_first=True,
        return_tm=True,
        tm_mode="fixed",              # TM under your framework fit (recommended if that's what you mean)
        out_self_pdb=aligned_input_path,
        out_other_pdb=aligned_changed_path,
        apply_alignment_inplace = False
    )
    return rmsd, tm

def verify(input_pdb, changed_pdb, aligned_pdb_output_dir=None):
    a = TCRpMHC(input_pdb, MHC_a_chain_id="A", MHC_b_chain_id="B", Peptide_chain_id="C")
    b = TCRpMHC(changed_pdb, MHC_a_chain_id="M", MHC_b_chain_id="N", Peptide_chain_id="P")

    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt")) as f:
        A_consensus_res = [int(x) for x in f.read().strip().split(",") if x.strip()]

    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt")) as f:
        B_consensus_res = [int(x) for x in f.read().strip().split(",") if x.strip()]
    metrics={}
    align_sel = {
        "M": [],   # whole chain
        "N": [],   # whole chain
        "P": [],   # whole chain
        "A": [],   # whole chain
        "B": [],   # whole chain
    }
    score_sel = align_sel.copy()
    if aligned_pdb_output_dir == None:
        aligned_input_path= None
        aligned_changed_path= None
    else:
        aligned_input_path= os.path.join(aligned_pdb_output_dir, "input_aligned_all.pdb")
        aligned_changed_path= os.path.join(aligned_pdb_output_dir, "changed_aligned_all.pdb")
    all_aligned_RMSD, all_aligned_TM = calc_RMSD_TM(align_sel, score_sel, a, b, aligned_input_path, aligned_changed_path)
    #align to each CDR individually and calc score
    metrics["all_aligned"] = {
        "rmsd": all_aligned_RMSD,
        "tm": all_aligned_TM,
    }
    # Selection dict for your rmsd/fit code
    align_sel = {
        "M": [],   # whole chain
        "N": [],   # whole chain
        "P": [],   # whole chain
    }
    score_sel = align_sel.copy()
    if aligned_pdb_output_dir == None:
        aligned_input_path= None
        aligned_changed_path= None
    else:
        aligned_input_path= os.path.join(aligned_pdb_output_dir, "input_aligned.pdb")
        aligned_changed_path= os.path.join(aligned_pdb_output_dir, "changed_aligned.pdb")
    pMHC_aligned_RMSD, pMHC_aligned_TM = calc_RMSD_TM(align_sel, score_sel, a, b, aligned_input_path, aligned_changed_path)
    #align to each CDR individually and calc score
    metrics["pMHC_aligned"] = {
        "rmsd": pMHC_aligned_RMSD,
        "tm": pMHC_aligned_TM,
    }
    for cdr in ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]:
        start, end = CDR_FR_RANGES[cdr]
        if cdr.startswith("A_"):
            chain_id = "A"
            consensus_res = A_consensus_res
        else:
            chain_id = "B"
            consensus_res = B_consensus_res
        fr_align_sel = {
            chain_id: {"ids": consensus_res},
        }
        cdr_sel={chain_id: {"range": (start, end)}}
        if aligned_pdb_output_dir == None:
            aligned_input_path_fr= None
            aligned_changed_path_fr= None
            aligned_input_path_cdr= None
            aligned_changed_path_cdr= None
        else:
            aligned_input_path_fr= os.path.join(aligned_pdb_output_dir, f"input_aligned_{chain_id}FR_{cdr}.pdb")
            aligned_changed_path_fr= os.path.join(aligned_pdb_output_dir, f"changed_aligned_{chain_id}FR_{cdr}.pdb")
            aligned_input_path_cdr= os.path.join(aligned_pdb_output_dir, f"input_aligned_{cdr}.pdb")
            aligned_changed_path_cdr= os.path.join(aligned_pdb_output_dir, f"changed_aligned_{cdr}.pdb")
        rmsd_fr_aligned, tm_fr_aligned = calc_RMSD_TM(fr_align_sel, cdr_sel, a, b, aligned_input_path_fr, aligned_changed_path_fr)
        rmsd_cdr, tm_cdr = calc_RMSD_TM( cdr_sel, cdr_sel, a, b, aligned_input_path_cdr, aligned_changed_path_cdr)
        metrics[cdr] = {
            "rmsd_fr_aligned": rmsd_fr_aligned,
            "tm_fr_aligned": tm_fr_aligned,
            "rmsd_cdr": rmsd_cdr,
            "tm_cdr": tm_cdr,
        }
    #all ACDRs together aligend to AFR
    #a_acdrs = {"A": {"range": [(CDR_FR_RANGES["A_CDR1"][0], CDR_FR_RANGES["A_CDR1"][1]),(CDR_FR_RANGES["A_CDR2"][0], CDR_FR_RANGES["A_CDR2"][1]), (CDR_FR_RANGES["A_CDR3"][0], CDR_FR_RANGES["A_CDR3"][1])]},}
    return metrics

# -----------------------------
# Main pipeline
# -----------------------------
def main(changed_dir, tfold_dir, true_pdb, out_base, aligned_dir=None ):
    if aligned_dir is not None:
        os.makedirs(aligned_dir, exist_ok=True)

    os.makedirs(out_base, exist_ok=True)
    test_outdir = os.path.join(out_base, "test")
    os.makedirs(test_outdir, exist_ok=True)

    error_file = os.path.join(out_base, "change_tfold_to_real_geometry_errors.txt")
    if os.path.exists(error_file):
        os.remove(error_file)

    all_rows = []

    for tfold_file in sorted(os.listdir(tfold_dir)):
        if not tfold_file.endswith(".pdb"):
            continue

        tfold_path = os.path.join(tfold_dir, tfold_file)
        tfold_name = tfold_file.replace(".pdb", "")

        try:
            true_complex = TCRpMHC(
                input_pdb=os.path.join(true_pdb, tfold_file),
                MHC_a_chain_id="A",
                MHC_b_chain_id="B",
                Peptide_chain_id="C",
            )

            tmp_outpath = tempfile.mkdtemp()
            results = true_complex.calc_geometry(out_path=tmp_outpath)

            # Change geometry of the tfold complex to match "true" geometry params
            complex_obj = TCRpMHC(
                input_pdb=tfold_path,
                MHC_a_chain_id="M",
                MHC_b_chain_id="N",
                Peptide_chain_id="P",
            )

            tmp_outdir = tempfile.mkdtemp()
            rearranged_pdb = complex_obj.change_geometry(tmp_outdir, results)

            os.makedirs(changed_dir, exist_ok=True)
            changed_path = os.path.join(changed_dir, f"{tfold_name}.pdb")
            shutil.move(rearranged_pdb, changed_path)
            print("Changed geometry pdb saved to:", changed_path)
        except Exception as e:
            print("Error processing file:", tfold_file, "->", str(e))
            with open(error_file, "a") as ef:
                ef.write(f"{tfold_file}\t{str(e)}\n")
            continue


def get_metrics(true_dir, changed_dir, out_base,error_file=None,aligned_dir=None):
    all_rows = []
    for true_file in os.listdir(true_dir):
        if not true_file.endswith(".pdb"):
            continue

        try:
            true_path = os.path.join(true_dir, true_file)
            true_name = true_file.replace(".pdb", "")
            changed_path = os.path.join(changed_dir, f"{true_name}.pdb")

            if aligned_dir is not None:
                file_aligned_dir = os.path.join(aligned_dir, true_name)
                os.makedirs(file_aligned_dir, exist_ok=True)
            else:
                file_aligned_dir = None
            metrics = verify(
                input_pdb=os.path.join(true_path),
                changed_pdb=changed_path,
                aligned_pdb_output_dir=file_aligned_dir,
            )

            # Flatten to row
            row = {"tcr_name": true_name}
            for key in metrics:
                for metric_name, value in metrics[key].items():
                    row[f"{key}_{metric_name}"] = value
            all_rows.append(row)

        except Exception as e:
            print("Error processing file:", true_file, "->", str(e))
            if error_file is not None:
                pass
            else:
                error_file =os.path.join(out_base, "metrics_calc_errors.txt")
            with open(error_file, "a") as ef:
                ef.write(f"{true_file}\t{str(e)}\n")
            continue

    # Save results
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_base, "change_tfold_to_real_geometry_results.csv")
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    # Plot
    plot_dir = os.path.join(out_base, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_distributions_from_csv(csv_path, plot_dir, name_col="tcr_name")


if __name__ == "__main__":
    changed_dir = "/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_tfold_changedgeo"
    tfold_dir   = "/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_tfold"
    true_pdb    = "/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb"
    out_base = "/workspaces/Graphormer/TCR_Metrics/pipelines/TCR_pMHC_geo/outputs_ATLAS_tfold_changedgeo"
    #aligned_dir = "/mnt/larry/lilian/DATA/ATLAS/structures/true_pdb_tfold_changedgeo_alignments"

    main(changed_dir, tfold_dir, true_pdb, out_base, aligned_dir=None)
    get_metrics(true_pdb, changed_dir, out_base,error_file=None,aligned_dir=None)