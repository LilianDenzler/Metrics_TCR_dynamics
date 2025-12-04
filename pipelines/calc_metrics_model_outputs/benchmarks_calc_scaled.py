import sys
import shutil
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
from TCR_Metrics.pipelines.calc_metrics_model_outputs.run_calc_scaled import *
from TCR_Metrics.pipelines.calc_metrics_model_outputs.compare_MD_adaptive_sampling import run_all_TCRs_collect_MDs

HERE = Path(__file__).parent
fit_on="concat"
model_folder="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
config_modes = [
        ("config_assess_modes_all_ca_pca.yaml", "all_ca_pca"),
        ("config_assess_modes_all_ca_kpca_cosine.yaml", "all_ca_kpca_cosine"),
        ("config_assess_modes.yaml", "default"),
    ]
for config_path, config_name in config_modes:
    for model_name in ["Alphaflow_with_linker", "Dig_vanilla", "baseline_sampled_200_frames_1","adaptive_sampling_all", "baseline_adaptivesampling_200_frames_1"]:
        CONFIG_PATH = HERE / config_path
        base_out = f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/benchmarks/{config_name}"
        os.makedirs(base_out, exist_ok=True)
        output_dir_all = os.path.join(base_out, model_name)
        os.makedirs(output_dir_all, exist_ok=True)
        error_log = os.path.join(output_dir_all, "error.txt")
        # ---- PHASE 1: collect embeddings & global ranges ----
        global_ranges = {}
        if model_name =="adaptive_sampling_all":
            linker=""
            run_all_TCRs_collect_MDs(
                md_folder="/mnt/larry/lilian/DATA/Oriol_adaptive_sampling/",
                output_dir_all=output_dir_all,
                error_log=error_log,
                CONFIG_PATH=CONFIG_PATH,
                global_ranges=global_ranges,
                closest_frames=False,
                save_aligned=False,
                specific_TCR=None,
                specific_region=None,
                fit_on=fit_on
            )
        else:
            linker="GGGGS"*3
            run_all_TCRs_collect(
                model_folder=model_folder,
                model_name=model_name,
                output_dir_all=output_dir_all,
                error_log=error_log,
                CONFIG_PATH=CONFIG_PATH,
                global_ranges=global_ranges,
                closest_frames=False,
                save_aligned=False,
                specific_TCR=None,
                specific_region=None,
                linker=linker,
                fit_on=fit_on
            )
        print("Global ranges after COLLECT phase:", global_ranges)

        # ---- PHASE 2: PMF / JSD with global bins per region ----
        run_all_TCRs_pmf(
            output_dir_all=output_dir_all,
            CONFIG_PATH=CONFIG_PATH,
            global_ranges=global_ranges,
            closest_frames=False,
            specific_TCR=None,
            specific_region=None,
        )
        run_make_tables(output_dir_all)



