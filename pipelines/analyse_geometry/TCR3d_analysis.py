from angle_utils import *
from angle_utils import main as compute_TCR_csv

def make_angle_csvs(data_dir="/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/"):
    #TCR3d unpaired, unbound:
    compute_TCR_csv(pdb_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/ab_chain", out_csv=f"{data_dir}/TCR3d_unpaired_unbound.csv", state="unbound")

    #TCR3d unpaired, bound:
    compute_TCR_csv(pdb_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes", out_csv=f"{data_dir}/TCR3d_unpaired_bound.csv", state="bound")

    #TCR3d paired, unbound:
    compute_TCR_csv(pdb_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_unbound_tcr_imgt", out_csv=f"{data_dir}/TCR3d_paired_unbound.csv", state="unbound")

    #TCR3d paired, bound:
    compute_TCR_csv(pdb_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/expanded_benchmark_bound_tcr_imgt", out_csv=f"{data_dir}/TCR3d_paired_bound.csv", state="bound")

def make_human_mouse_subsets(data_dir="/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/"):
    datasets = ["TCR3d_unpaired_unbound.csv", "TCR3d_unpaired_bound.csv",
                "TCR3d_paired_unbound.csv", "TCR3d_paired_bound.csv"]
    for dataset in datasets:
        df = pd.read_csv(f"{data_dir}/{dataset}")
        human_df = df[(df['alpha_germline'].str.contains("human")) & (df['beta_germline'].str.contains("human"))]
        mouse_df = df[(df['alpha_germline'].str.contains("mouse")) & (df['beta_germline'].str.contains("mouse"))]

        other_df= df[~df.index.isin(human_df.index) & ~df.index.isin(mouse_df.index)]
        mixed_mouse_human_df= df[((df['alpha_germline'].str.contains("mouse")) & (df['beta_germline'].str.contains("human"))) |
                                 ((df['alpha_germline'].str.contains("human")) & (df['beta_germline'].str.contains("mouse")))]
        #remove mixed from human and mouse dfs
        human_df= human_df[~human_df.index.isin(mixed_mouse_human_df.index)]
        mouse_df= mouse_df[~mouse_df.index.isin(mixed_mouse_human_df.index)]
        human_df.to_csv(f"{data_dir}/human_{dataset}", index=False)
        mouse_df.to_csv(f"{data_dir}/mouse_{dataset}", index=False)


if __name__ == "__main__":
    make_angle_csvs()
    make_human_mouse_subsets()