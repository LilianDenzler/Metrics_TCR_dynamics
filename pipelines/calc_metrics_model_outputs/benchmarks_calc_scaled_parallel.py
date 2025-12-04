import os
import sys
import shutil
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from TCR_Metrics.pipelines.calc_metrics_model_outputs.run_calc_scaled import (
    run_all_TCRs_collect,
    run_all_TCRs_pmf,
    run_make_tables,
)
from TCR_Metrics.pipelines.calc_metrics_model_outputs.compare_MD_adaptive_sampling import (
    run_all_TCRs_collect_MDs,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HERE = Path(__file__).parent

FIT_ON = "concat"
MODEL_FOLDER = "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
MD_FOLDER = "/mnt/larry/lilian/DATA/Oriol_adaptive_sampling/"
BASE_OUT_ROOT = "/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/benchmarks"

CONFIG_MODES = [
    ("config_assess_modes_all_ca_pca.yaml", "all_ca_pca"),
    ("config_assess_modes_all_ca_kpca_cosine.yaml", "all_ca_kpca_cosine"),
    ("config_assess_modes.yaml", "default"),
]

MODEL_NAMES = [
    "adaptive_sampling_all",
    "baseline_adaptivesampling_200_frames_1",
    "Alphaflow_with_linker",
    "Dig_vanilla",
    "baseline_sampled_200_frames_1",

]


# ---------------------------------------------------------------------
# WORKER FOR ONE (CONFIG, MODEL_NAME) COMBINATION
# ---------------------------------------------------------------------
def run_one(config_path: str, config_name: str, model_name: str):
    """
    Run:
      1) collect embeddings & global_ranges
      2) PMF/JSD
      3) make tables
    for a single (config, model_name) combo.
    """

    CONFIG_PATH = HERE / config_path

    base_out = os.path.join(BASE_OUT_ROOT, config_name)
    os.makedirs(base_out, exist_ok=True)

    output_dir_all = os.path.join(base_out, model_name)
    os.makedirs(output_dir_all, exist_ok=True)

    error_log = os.path.join(output_dir_all, "error.txt")

    # Per-job global_ranges
    global_ranges = {}
    if model_name in ["adaptive_sampling_all", "baseline_adaptivesampling_200_frames_1", "baseline_sampled_200_frames_1"]:
        linker = ""
    else:
        linker = "GGGGS" * 3


    # ---- PHASE 1: collect embeddings & global ranges ----
    if model_name == "adaptive_sampling_all":
        run_all_TCRs_collect_MDs(
            md_folder=MD_FOLDER,
            output_dir_all=output_dir_all,
            error_log=error_log,
            CONFIG_PATH=CONFIG_PATH,
            global_ranges=global_ranges,
            closest_frames=False,
            save_aligned=False,
            specific_TCR=None,
            specific_region=None,
            fit_on=FIT_ON,
        )
    else:
        run_all_TCRs_collect(
            model_folder=MODEL_FOLDER,
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
            fit_on=FIT_ON,
        )

    print(f"[{config_name} | {model_name}] Global ranges after COLLECT:", global_ranges)

    # ---- PHASE 2: PMF / JSD with global bins per region ----
    run_all_TCRs_pmf(
        output_dir_all=output_dir_all,
        CONFIG_PATH=CONFIG_PATH,
        global_ranges=global_ranges,
        closest_frames=False,
        specific_TCR=None,
        specific_region=None,
    )

    # ---- PHASE 3: make tables ----
    run_make_tables(output_dir_all)

    return config_name, model_name


# ---------------------------------------------------------------------
# MAIN: PARALLEL DISPATCH
# ---------------------------------------------------------------------
if __name__ == "__main__":
    HERE = Path(__file__).parent

    FIT_ON = "concat"
    MODEL_FOLDER = "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
    MD_FOLDER = "/mnt/larry/lilian/DATA/Oriol_adaptive_sampling/"
    BASE_OUT_ROOT = "/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/benchmarks"

    CONFIG_MODES = [
        ("config_assess_modes_all_ca_pca.yaml", "all_ca_pca"),
        ("config_assess_modes_all_ca_kpca_cosine.yaml", "all_ca_kpca_cosine"),
        ("config_assess_modes.yaml", "default"),
    ]

    MODEL_NAMES = [
        "adaptive_sampling_all",
        "baseline_adaptivesampling_200_frames_1",
        "Alphaflow_with_linker",
        "Dig_vanilla",
        "baseline_sampled_200_frames_1",

    ]

    # Tune this depending on CPU/GPU constraints
    MAX_WORKERS = 2  # e.g. 2–4; if GPU-limited, you can even set to 1

    jobs = []
    for config_path, config_name in CONFIG_MODES:
        for model_name in MODEL_NAMES:
            jobs.append((config_path, config_name, model_name))

    print(f"Submitting {len(jobs)} jobs with max_workers={MAX_WORKERS}...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(run_one, cfg_path, cfg_name, model_name): (cfg_name, model_name)
            for (cfg_path, cfg_name, model_name) in jobs
        }

        for fut in as_completed(futures):
            cfg_name, model_name = futures[fut]
            try:
                done_cfg, done_model = fut.result()
                print(f"[DONE] {done_cfg} | {done_model}")
            except Exception as e:
                print(f"[ERROR] {cfg_name} | {model_name}: {e}")
