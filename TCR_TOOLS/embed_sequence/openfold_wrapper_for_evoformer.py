##cd /workspaces/Graphormer/openfold
##mamba env create -n openfold_env -f environment.yml
#mamba activate openfold_env
from pdb2fasta import pdb_to_fasta
import os
import sys
sys.path.append("/workspaces/Graphormer/openfold")
import openfold
import subprocess
import shlex
import torch
from MMseq2_GPU_MSA_MAKER import run_pipeline_with_precomputed_alignments

print("cuda_available:", torch.cuda.is_available(),
      "device_count:", torch.cuda.device_count(),
      "current_device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
def run(pdb_path, fasta_dir, output_dir,precom_alignments_dir=None, usemmseq2_gpu=False):
    #make fasta file
    fasta_path = os.path.join(fasta_dir, "seq.fasta")
    pdb_to_fasta(pdb_path, fasta_path)
    print(f"FASTA file created at {fasta_path}")
    #run python command
    if usemmseq2_gpu:
        precom_alignments_dir=run_pipeline_with_precomputed_alignments(
            fasta_path=fasta_path,
            align_dir=os.path.join(output_dir,"alignments"),
            output_dir=output_dir,
            uniref90_db="/workspaces/Graphormer/mmseqs2_data/uniref90_db/uniref90",
            model_device="cuda:0",
            config_preset="model_1_ptm",
            template_mmcif_dir="/mnt/bob/shared/alphafold/pdb_mmcif/mmcif_files",
            threads="32",
            gpu_ids="0,1",
            conda_env="nvcc")

    command=f"""python3 run_pretrained_openfold_shortened.py \
        {fasta_dir} \
        /mnt/bob/shared/alphafold/pdb_mmcif/mmcif_files \
        --uniref90_database_path /mnt/bob/shared/alphafold/uniref90/uniref90.fasta \
        --mgnify_database_path /mnt/bob/shared/alphafold/mgnify/mgy_clusters_2022_05.fa \
        --pdb_seqres_database_path /mnt/bob/shared/alphafold/pdb_seqres/pdb_seqres.txt \
        --uniref30_database_path /mnt/bob/shared/alphafold/uniref30/UniRef30_2021_03 \
        --uniprot_database_path /mnt/bob/shared/alphafold/uniprot/uniprot.fasta \
        --bfd_database_path /mnt/bob/shared/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --jackhmmer_binary_path /home/vscode/.conda/envs/openfold/bin/jackhmmer \
        --hhblits_binary_path /home/vscode/.conda/envs/openfold/bin/hhblits \
        --hmmsearch_binary_path /home/vscode/.conda/envs/openfold/bin/hmmsearch \
        --hmmbuild_binary_path /home/vscode/.conda/envs/openfold/bin/hmmbuild \
        --kalign_binary_path /home/vscode/.conda/envs/openfold/bin/kalign \
        --config_preset "model_1_multimer_v3" \
        --model_device "cuda:0" \
        --output_dir {output_dir} \
        --save_outputs"""
    commandv1=f"""conda run -n openfold python3 -u run_pretrained_openfold_shortened.py \
        {fasta_dir} \
        /mnt/bob/shared/alphafold/pdb_mmcif/mmcif_files \
        --output_dir {output_dir} \
        --config_preset "model_1_ptm" \
        --uniref90_database_path /mnt/bob/shared/alphafold/uniref90/uniref90.fasta \
        --mgnify_database_path /mnt/bob/shared/alphafold/mgnify/mgy_clusters_2022_05.fa \
        --pdb70_database_path /mnt/bob/shared/alphafold/pdb70/pdb70 \
        --uniclust30_database_path /mnt/bob/shared/alphafold/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
        --bfd_database_path /mnt/bob/shared/alphafold/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
        --model_device "cuda:0" \
        --save_outputs {'--use_precomputed_alignments '+str(precom_alignments_dir) if precom_alignments_dir else ''} """
    command_list = shlex.split(commandv1)
    print("--- Starting OpenFold subprocess ---")
    # Using subprocess.run will wait for the command to complete.
    # It will print the stdout and stderr from the subprocess directly to your current console in real-time.
    result = subprocess.run(command_list, check=True) # check=True will raise an error if the script fails
    print("--- OpenFold subprocess finished ---")
    #os.system(commandv1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the input PDB file")
    parser.add_argument("--fasta_dir", type=str, required=True, help="Directory to save the FASTA file")
    parser.add_argument("--output_dir", type=str,required=True, help="Directory to save the OpenFold output")
    parser.add_argument("--precom_alignments_dir", type=str, default=None, help="Directory of precomputed alignments (optional)")
    parser.add_argument("--usemmseq2_gpu", action='store_true', help="Whether to use MMseq2 with GPU acceleration for MSA generation")
    args = parser.parse_args()

    pdb_path = args.pdb_path
    fasta_dir = args.fasta_dir
    output_dir = args.output_dir
    precom_alignments_dir = args.precom_alignments_dir
    usemmseq2_gpu = args.usemmseq2_gpu
    #change dir
    os.chdir("/workspaces/Graphormer/openfold")
    #mkdir fasta_dir if not exists
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)
    run(pdb_path, fasta_dir, output_dir, precom_alignments_dir=precom_alignments_dir, usemmseq2_gpu=usemmseq2_gpu)
    # run as
    # python openfold_wrapper_for_evoformer.py --pdb_path /path/to/pdb.pdb --fasta_dir /path/to/fasta_dir --fasta_path /path/to/fasta.fasta --output_dir /path/to/output_dir