import pandas as pd
import glob
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import logging
try:
    import yaml
except ImportError:
    logging.warning("PyYAML not found. Will not be able to generate YAML output.")
    logging.warning("Please run: pip install pyyaml")
    yaml = None

# --- Configuration ---
BASE_DIR = "/workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_analysis_out"
DATA_FILENAME = "ca_dist_metrics.json"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_mantel_score(data: dict) -> float | None:
    """
    Tries to find the mantel score from the loaded JSON data.

    Based on the example, the score is at data['mantel']['gt']['r']

    Args:
        data: The loaded JSON data (as a dict).

    Returns:
        The score as a float, or None if not found.
    """
    try:
        # Check if 'mantel' key exists and is a dictionary
        if isinstance(data, dict) and 'mantel' in data and isinstance(data['mantel'], dict):
            mantel_data = data['mantel']

            # Check if 'gt' key exists and is a dictionary
            if 'gt' in mantel_data and isinstance(mantel_data['gt'], dict):
                gt_data = mantel_data['gt']

                # Check if 'r' key exists and is a number
                if 'r' in gt_data and isinstance(gt_data['r'], (int, float)):
                    return float(gt_data['r'])

        # If any check fails, log a warning
        logging.warning(f"Could not find a valid score at ['mantel']['gt']['r'] in data.")
        return None

    except Exception as e:
        logging.error(f"Error while parsing score from data: {e}")
        return None

def parse_path(json_path: Path, base_path_obj: Path) -> dict | None:
    """
    Parses the full path to extract TCR, region, and method.

    The region is constructed from all directories between the TCR and the method.
    Example: /BASE/TCR_A/Region_Part1/Region_Part2/Method_X/file.json
    TCR: TCR_A
    Region: Region_Part1_Region_Part2
    Method: Method_X

    Args:
        json_path: The Path object for the JSON file.
        base_path_obj: The Path object for the base directory.

    Returns:
        A dictionary with 'tcr', 'region', 'method', or None on failure.
    """
    try:
        # Get all parts of the path relative to the base directory
        relative_parts = json_path.relative_to(base_path_obj).parts

        # We expect at least 4 parts: TCR, Region(s), Method, Filename
        if len(relative_parts) < 4:
            logging.warning(f"Skipping path, too short: {json_path}")
            return None

        tcr = relative_parts[0]
        method = relative_parts[-2] # Method is the second-to-last part

        # Region parts are everything between TCR and Method
        region_parts = relative_parts[1:-2]
        if not region_parts:
            logging.warning(f"Skipping path, no region parts found: {json_path}")
            return None

        # Join all region parts with underscores, as requested
        region_name = "_".join(region_parts)

        return {
            "tcr": tcr,
            "region": region_name,
            "method": method
        }

    except Exception as e:
        logging.error(f"Error parsing path {json_path}: {e}")
        return None

def analyze_all_tcrs(base_dir: str):
    """
    Main function to analyze all TCR metrics.
    """
    logging.info(f"Starting analysis in: {base_dir}")

    base_path_obj = Path(base_dir)
    if not base_path_obj.is_dir():
        logging.error(f"Base directory not found: {base_dir}")
        return

    # Find all data files recursively
    json_files = glob.glob(f"{base_dir}/**/{DATA_FILENAME}", recursive=True)

    if not json_files:
        logging.error(f"No '{DATA_FILENAME}' files found in {base_dir}")
        return

    logging.info(f"Found {len(json_files)} data files to analyze.")

    all_scores = []

    for file_str in json_files:
        file_path = Path(file_str)

        # 1. Parse the path
        path_info = parse_path(file_path, base_path_obj)
        if not path_info:
            continue

        # 2. Read the JSON and get the score
        try:
            with open(file_str, 'r') as f:
                data = json.load(f)

            score = get_mantel_score(data)

            if score is not None:
                all_scores.append({
                    "tcr": path_info["tcr"],
                    "region": path_info["region"],
                    "method": path_info["method"],
                    "score": score
                })
            else:
                logging.warning(f"No score found in file: {file_path}")

        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from: {file_path}")
        except IOError as e:
            logging.error(f"Could not read file {file_path}: {e}")

    if not all_scores:
        logging.error("No valid score data was extracted. Exiting.")
        return

    # --- Analysis ---

    # 1. Convert to DataFrame
    df = pd.DataFrame(all_scores)
    logging.info(f"Successfully processed {len(df)} valid data points.")
    # --- Filter out 'tica' methods ---
    original_count = len(df)
    df = df[~df['method'].str.contains("tica")]
    filtered_count = len(df)
    logging.info(f"Filtered out {original_count - filtered_count} 'tica' data points. {filtered_count} remaining.")

    if df.empty:
         logging.error("DataFrame is empty, cannot proceed with analysis.")
         return

    # --- ANALYSIS 1: Find the "Champion" (Single Highest Score) ---
    # This is the original analysis
    logging.info("Running Analysis 1: Finding 'Champion' method (single highest score)...")
    df_sorted_by_score = df.sort_values(by='score', ascending=False)
    best_per_region_champion = df_sorted_by_score.drop_duplicates(subset='region')
    best_per_region_champion = best_per_region_champion.sort_values(by='region').reset_index(drop=True)

    print("\n--- ANALYSIS 1: 'Champion' Method (Single Highest Score) ---")
    print(best_per_region_champion.to_string())

    # --- ANALYSIS 2: Find the "Most Robust" Method (Avg Performance) ---
    # This is the new analysis based on your request

    logging.info("Running Analysis 2: Finding 'Most Robust' method (highest mean, lowest std)...")
    # Group by region and method, then calculate stats for the scores
    method_stats = df.groupby(['region', 'method'])['score'].agg(
        mean_score='mean',
        std_dev='std',
        count='count'
    ).reset_index()

    # Handle cases with only one data point (where std is NaN)
    method_stats['std_dev'] = method_stats['std_dev'].fillna(0)

    # Sort to find the best: highest mean, then lowest std_dev
    method_stats_sorted = method_stats.sort_values(
        by=['region', 'mean_score', 'std_dev'],
        ascending=[True, False, True] # Sort by region (A-Z), then mean (High-Low), then std (Low-High)
    )

    print("\n--- ANALYSIS 2: Full Robustness Stats (Mean/Std per Method, All Regions) ---")
    print(method_stats_sorted.to_string())

    # Now, find the top robust method for each region
    best_robust_per_region = method_stats_sorted.drop_duplicates(subset='region', keep='first')
    best_robust_per_region = best_robust_per_region.reset_index(drop=True)

    print("\n--- ANALYSIS 2: 'Most Robust' Method (Best Avg Performance per Region) ---")
    print(best_robust_per_region.to_string())

    # --- Plotting ---
    # Plot the frequency of the "Most Robust" method

    print("\n--- Frequency of 'Most Robust' Methods ---")
    best_robust_method_counts = best_robust_per_region['method'].value_counts()
    print(best_robust_method_counts)

    try:
        plt.figure(figsize=(14, 8))
        best_robust_method_counts.plot(kind='bar', color='c', zorder=3)
        plt.title('Frequency: Which Method Was "Most Robust" Most Often?', fontsize=18)
        plt.ylabel('Number of Regions This Method Was Best For', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()

        plot_filename = "best_robust_method_frequency.png"
        plt.savefig(plot_filename)
        logging.info(f"Analysis plot saved to: {plot_filename}")

    except Exception as e:
        pass

    # --- ANALYSIS 3: Generate YAML Output File ---
    if yaml is None:
        logging.error("Skipping YAML file generation. Please install PyYAML.")
        return

    logging.info("Running Analysis 3: Generating YAML output file (best_methods_config.yml)...")
    yaml_data = {}

    # Use the 'best_robust_per_region' DataFrame from ANALYSIS 2
    for _, row in best_robust_per_region.iterrows():
        if "A_variable" in row['region']:
            base_region = row['region'].split('A_variable')[1].strip('_')
            align_region = "A_variable"
            if base_region == "":
                base_region = "A_variable"
        elif "B_variable" in row['region']:
            base_region = row['region'].split('B_variable')[1].strip('_')
            align_region = "B_variable"
            if base_region == "":
                base_region = "B_variable"
        else:
            base_region=row['region']
            align_region=row['region']
        method = row['method']

        if base_region not in yaml_data:
            yaml_data[base_region] = {}
        yaml_data[base_region][align_region] = method

    # Sort the final dictionary for a clean, human-readable output
    # Sort top-level keys (base_region)
    sorted_yaml_data = dict(sorted(yaml_data.items()))
    # Sort sub-keys (align_region) for each base_region
    for base, alignments in sorted_yaml_data.items():
        sorted_yaml_data[base] = dict(sorted(alignments.items()))

    output_filename = "best_methods_config.yml"
    try:
        with open(output_filename, 'w') as f:
            # Dump the YAML, using sort_keys=False because we already sorted it
            yaml.dump(sorted_yaml_data, f, default_flow_style=False, sort_keys=False, indent=2)
        logging.info(f"Successfully wrote YAML config to {output_filename}")

    except Exception as e:
        logging.error(f"Failed to write YAML file: {e}")
    # --- ANALYSIS 4: Find the "Global Best" Method (Best Avg Performance Across ALL Regions) ---
    logging.info("Running Analysis 4: Finding 'Global Best' method (best avg score across all regions)...")

    # Group by 'method' only, calculating stats across all regions and TCRs
    global_method_stats = df.groupby('method')['score'].agg(
        mean_score='mean',
        std_dev='std',
        count='count'
    ).reset_index()

    # Handle NaNs in std_dev (for methods that only appeared once, though unlikely)
    global_method_stats['std_dev'] = global_method_stats['std_dev'].fillna(0)

    # Sort to find the best: highest mean, then lowest std_dev
    global_method_stats_sorted = global_method_stats.sort_values(
        by=['mean_score', 'std_dev'],
        ascending=[False, True] # Sort by mean (High-Low), then std (Low-High)
    )

    print("\n--- ANALYSIS 4: Global Method Performance (Averaged Across All Regions & TCRs) ---")
    print(global_method_stats_sorted.to_string())

    if not global_method_stats_sorted.empty:
        global_best_method = global_method_stats_sorted.iloc[0]

        print("\n************************************************************")
        print(f"* The 'Global Best' method (highest average score) is: *")
        print(f"* Method:     {global_best_method['method']}")
        print(f"* Mean Score: {global_best_method['mean_score']:.4f}")
        print(f"* Std Dev:    {global_best_method['std_dev']:.4f}")
        print(f"* Data Points: {global_best_method['count']}")
        print("************************************************************")
    else:
        logging.warning("Could not determine a 'Global Best' method, no data processed.")



if __name__ == "__main__":
    analyze_all_tcrs(BASE_DIR)