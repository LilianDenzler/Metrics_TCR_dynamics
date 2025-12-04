import os
import subprocess
from pathlib import Path
import shutil
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.core.io import write_pdb
import sys
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import json
import os
import numpy as np
import time

def save_noise_params(noise_params, final_out_subdir):
    json_path = os.path.join(final_out_subdir, "noise_params.json")
    with open(json_path, "w") as f:
        json.dump(noise_params, f, indent=4)
    print(f"✅ Saved noise params to {json_path}")
    return json_path


def pdb_to_fasta(pdb_path, fasta_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_path)

    seq_records = []

    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            sequence = "".join(seq1(res.get_resname()) for res in residues)
            record = SeqRecord(
                Seq(sequence),
                id=f"{structure.id}_{chain.id}",
                description=f"Chain {chain.id} from {pdb_path}"
            )
            seq_records.append(record)

    SeqIO.write(seq_records, fasta_path, "fasta")
    print(f"✅ Wrote {len(seq_records)} chains to {fasta_path}")


def run_prep(final_out, input_pdb,pdb_name, linker_sequence="GGGGS" * 3, regions_masked_true=None,pkl_dir=None):
    tcr=TCR(input_pdb)
    pv = tcr.pairs[0]
    linked_out_file=os.path.join(final_out, f"{pdb_name}_linked.pdb")
    print(linked_out_file)
    #if Path(linked_out_file).exists():
    #    print(f"Linked PDB already exists. Skipping linking step.")
    #else:
    linked_structure=pv.linked_structure(linker_sequence)
    if regions_masked_true:
        binary_res_mask=pv.linked_resmask( regions_masked_true)
    else:
        binary_res_mask=None
    print(f"Writing linked structure to {linked_structure}")
    write_pdb(linked_out_file, linked_structure)

    if Path(os.path.join(final_out, f"{pdb_name}.pkl")).exists():
        print(f"OpenFold output already exists. Skipping Evoformer embedding extraction.")
    if Path(os.path.join(pkl_dir, f"{pdb_name}.pkl")).exists():
        print(f"Copying precomputed Evo2 embeddings from {pkl_dir}")
        shutil.copyfile(os.path.join(pkl_dir, f"{pdb_name}.pkl"), os.path.join(final_out, f"{pdb_name}.pkl"))
        pdb_to_fasta(linked_out_file, os.path.join(final_out, f"{pdb_name}.fasta"))
    else:
        if not Path(os.path.join(final_out, "fasta")).exists():
            os.makedirs(os.path.join(final_out, "fasta"))
        if not Path(os.path.join(final_out, "openfold_output_Evo2")).exists():
            os.makedirs(os.path.join(final_out, "openfold_output_Evo2"))

        #precompute alignments with GPU acceleration
        subprocess.run(["conda", "run", "-n", "openfold", "python",
                "/workspaces/Graphormer/distributional_graphormer/protein/full_pipeline/evoformer_representation/openfold_wrapper_for_evoformer.py",
                "--pdb_path", linked_out_file,
                "--fasta_dir", os.path.join(final_out, "fasta"),
                #"--fasta_path", os.path.join(final_out, "fasta", f"{pdb_name}.fasta"),
                "--output_dir", os.path.join(final_out, "openfold_output_Evo2")])
        pkl_out=list(Path(os.path.join(final_out, "openfold_output_Evo2",'predictions')).glob("*.pkl"))[0]
        fasta_out=list(Path(os.path.join(final_out, "fasta")).glob("*.fasta"))[0]
        subprocess.run(["mv", fasta_out, os.path.join(final_out, f"{pdb_name}.fasta")])
        subprocess.run(["mv",pkl_out, os.path.join(final_out, f"{pdb_name}.pkl")])
        print("Evoformer embedding extraction finished.")
        shutil.rmtree(os.path.join(final_out, "fasta"))
        shutil.rmtree(os.path.join(final_out, "openfold_output_Evo2"))
    #run inference
    if Path(os.path.join(final_out, f"{pdb_name}.init_state.npz")).exists():
        print(f"Init state file already exists. Skipping init state extraction step.")
    else:
        subprocess.run(["python","/workspaces/Graphormer/distributional_graphormer/protein/full_pipeline/get_init_state.py",
                    linked_out_file,
                    "--out_path", os.path.join(final_out, f"{pdb_name}.init_state.npz")])
        print("Init state extraction finished.")
    return Path(final_out),binary_res_mask


def run_vanilla_no_init_state(final_out, pdb_name, n_samples=100):
    final_out_subdir=os.path.join(final_out, "dig_vanilla")
    os.makedirs(final_out_subdir, exist_ok=True)
    time_start=time.time()
    subprocess.run([sys.executable, "/workspaces/Graphormer/distributional_graphormer/protein/run_inference.py",
                    "-c", "/workspaces/Graphormer/distributional_graphormer/protein/checkpoints/checkpoint-520k.pth",
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    #NO INIT STATE ARGUMENT HERE
                    "-n", str(n_samples),
                    #NO NOISE MASK ARGUMENT HERE
                    #NO NOISE PARAMS ARGUMENT HERE
                    "--use-gpu",
                    "--use-tqdm"])
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(os.path.join(final_out, "dig_vanilla"))

def run_with_init_state(final_out, pdb_name, n_samples=100, noise_params=None):
    final_out_subdir=os.path.join(final_out, "dig_with_init"+f"_trA{str(noise_params['tr_a'])}_rotA{str(noise_params['rot_a'])}_trB{str(noise_params['tr_b'])}_rotB{str(noise_params['rot_b'])}/")
    if not Path(final_out_subdir).exists():
        os.makedirs(final_out_subdir)
    if any(Path(final_out_subdir).glob("*_199.pdb")):
        print(f"✅ Inference outputs already exist in {final_out_subdir}. Skipping inference step.")
        return Path(final_out_subdir)
    #write noise params to txt file
    noise_param_json=save_noise_params(noise_params, final_out_subdir)
    time_start=time.time()
    subprocess.run([sys.executable, "/workspaces/Graphormer/distributional_graphormer/protein/run_inference_addnoise.py",
                    "-c", "/workspaces/Graphormer/distributional_graphormer/protein/checkpoints/checkpoint-520k.pth",
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm",
                    #NO NOISE MASK ARGUMENT HERE
                    "--noise-params-json", noise_param_json],cwd="/tmp")
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(final_out_subdir)


def run_with_init_state_cdr_mask(final_out, pdb_name, n_samples=100, noise_params=None, regions_masked_true=None, binary_res_mask=None):
    final_out_subdir=os.path.join(final_out, "dig_with_init_cdr_mask"+f"_trA{str(noise_params['tr_a'])}_rotA{str(noise_params['rot_a'])}_trB{str(noise_params['tr_b'])}_rotB{str(noise_params['rot_b'])}/")
    if not Path(final_out_subdir).exists():
        os.makedirs(final_out_subdir)
    #write noise params to txt file
    noise_param_json=save_noise_params(noise_params, final_out_subdir)
    binary_res_mask_path=os.path.join(final_out_subdir, f"binary_res_mask.npy")
    np.save(binary_res_mask_path, binary_res_mask)
    print(f"✅ Saved binary residue mask to {binary_res_mask_path}")
    #save regions masked true to json
    regions_masked_true_json_path=os.path.join(final_out_subdir, "regions_masked_true.json")
    with open(regions_masked_true_json_path, "w") as f:
        json.dump(regions_masked_true, f, indent=4)
    print(f"✅ Saved regions masked true to {regions_masked_true_json_path}")
    time_start=time.time()
    subprocess.run([sys.executable, "/workspaces/Graphormer/distributional_graphormer/protein/run_inference_addnoise.py",
                    "-c", "/workspaces/Graphormer/distributional_graphormer/protein/checkpoints/checkpoint-520k.pth",
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", final_out_subdir,
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm",
                    "--binary_res_mask_path", binary_res_mask_path,
                    "--noise-params-json", noise_param_json],cwd="/tmp")
    time_end=time.time()
    print(f"Total inference time: {time_end - time_start} seconds")
    total_time_per_sample=(time_end - time_start)/n_samples
    #write to file
    with open(os.path.join(final_out_subdir, "time_per_sample.txt"), "w") as f:
        f.write(f"Total inference time: {time_end - time_start} seconds\n")
        f.write(f"Time per sample: {total_time_per_sample} seconds\n")
    return Path(final_out_subdir)


def runall(output_dir_all, all_pdb_folder, pkl_dir, n_samples=200,dig_mode="vanilla_no_init", noise_params={"tr_a":0.2, "rot_a":0.2, "tr_b":2.5, "rot_b":1.5},regions_masked_true=["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]):
    pdbs = list(Path(all_pdb_folder).glob("*.pdb"))
    if not os.path.exists(output_dir_all):
        os.makedirs(output_dir_all)
    for pdb_path in pdbs:
        pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
        output_dir=os.path.join(output_dir_all,pdb_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        preprocess_out,binary_res_mask=run_prep(output_dir, pdb_path,  pdb_name, linker_sequence="GGGGS" * 3, regions_masked_true=regions_masked_true, pkl_dir=pkl_dir)
        print(binary_res_mask)
        if dig_mode=="vanilla_no_init":
            vanilla_no_init_frame_dir=run_vanilla_no_init_state(preprocess_out, pdb_name, n_samples=n_samples)
        elif dig_mode=="init":
            init_frame_dir=run_with_init_state(preprocess_out, pdb_name, n_samples=n_samples, noise_params=noise_params)
        elif dig_mode=="init_cdr_mask":
            init_cdr_mask_frame_dir=run_with_init_state_cdr_mask(preprocess_out, pdb_name, n_samples=n_samples, noise_params=noise_params,regions_masked_true=regions_masked_true, binary_res_mask=binary_res_mask)
        elif dig_mode=="init_geometry_sample":
            print("Not implemented yet.")
        elif dig_mode=="init_cdr_mask_geometry_sample":
            print("Not implemented yet.")

if __name__ == "__main__":
    all_cory_pdbs="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory"
    output_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_dig_variations"
    pkl_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/evo2_embeddings"
    runall(output_dir, all_cory_pdbs, pkl_dir=pkl_dir,dig_mode="vanilla_with_init")
