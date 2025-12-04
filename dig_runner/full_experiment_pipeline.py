import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
from dig_runner.pdb_to_inference_dig_all_variants import runall

all_cory_pdbs="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory"
output_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
pkl_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/evo2_embeddings"
dig_mode="init_cdr_mask"
regions_masked_true=["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]
noise_params={"tr_a":0.2, "rot_a":0.2, "tr_b":0.1, "rot_b":0.1}
runall(output_dir, all_cory_pdbs, pkl_dir=pkl_dir, n_samples=200,dig_mode=dig_mode, noise_params=noise_params, regions_masked_true=regions_masked_true)


all_cory_pdbs="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory"
output_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
pkl_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/evo2_embeddings"
dig_mode="init"
noise_params={"tr_a":0.2, "rot_a":0.2, "tr_b":0.1, "rot_b":0.1}

runall(output_dir, all_cory_pdbs, pkl_dir=pkl_dir, n_samples=200,dig_mode=dig_mode, noise_params=noise_params)
