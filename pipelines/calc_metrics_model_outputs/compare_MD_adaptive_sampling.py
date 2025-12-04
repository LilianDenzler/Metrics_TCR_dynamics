import sys
import shutil
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
from TCR_Metrics.pipelines.calc_metrics_model_outputs.run_calc_scaled import *

def run_all_TCRs_collect_MDs(
    md_folder, #"/mnt/larry/lilian/DATA/Oriol_adaptive_sampling/"
    output_dir_all,
    error_log,
    CONFIG_PATH,
    global_ranges,
    closest_frames=True,
    save_aligned=True,
    specific_TCR=None,
    specific_region=None,
    fit_on="gt",
):
    """
    Phase 1: go over all TCRs, align + compute embeddings + update global_ranges.
    """
    for folder in os.listdir(md_folder):
        try:
            full_path = os.path.join(md_folder, folder)
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
            og_output_xtc_path=os.path.join(md_folder,TCR_NAME,f"{TCR_NAME}.xtc")
            og_output_pdb_path=os.path.join(md_folder,TCR_NAME,f"{TCR_NAME}.pdb")
            shutil.copy(og_output_xtc_path, output_xtc_path)
            shutil.copy(og_output_pdb_path, output_pdb_path)

            with CONFIG_PATH.open() as f:
                region_metric_config = yaml.safe_load(f)

            pdb_gt = f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}.pdb"
            xtc_gt = f"/mnt/larry/lilian/DATA/Cory_data/{TCR_NAME}/{TCR_NAME}_Prod.xtc"

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


