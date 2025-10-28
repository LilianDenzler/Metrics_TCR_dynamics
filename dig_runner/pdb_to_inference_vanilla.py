import os
import subprocess
from pathlib import Path
import shutil
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.core.io import write_pdb
import sys
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

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


def run_prep(final_out, input_pdb,pdb_name, linker_sequence="GGGGS" * 3, pkl_dir=None):
    tcr=TCR(input_pdb)
    pv = tcr.pairs[0]
    linked_out_file=os.path.join(final_out, f"{pdb_name}_linked.pdb")
    print(linked_out_file)
    if Path(linked_out_file).exists():
        print(f"Linked PDB already exists. Skipping linking step.")
    else:
        linked_structure=pv.linked_structure("GGGGS" * 3)
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
    return Path(os.path.join(final_out, pdb_name))


def run_vanilla_no_init_state(final_out, pdb_name, n_samples=100):
    if not Path(os.path.join(final_out, "dig_vanilla")).exists():
        os.makedirs(os.path.join(final_out, "dig_vanilla"))
    subprocess.run(["python", "/workspaces/Graphormer/distributional_graphormer/protein/run_inference.py",
                    "-c", "/workspaces/Graphormer/distributional_graphormer/protein/checkpoints/checkpoint-520k.pth",
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", os.path.join(final_out, "dig_vanilla/"),
                    #"--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm"])
    return Path(os.path.join(final_out, "dig_vanilla"))

def run_vanilla_with_init_state(final_out, pdb_name, n_samples=100):
    if not Path(os.path.join(final_out, "dig_vanilla_with_init")).exists():
        os.makedirs(os.path.join(final_out, "dig_vanilla_with_init"))
    subprocess.run(["python", "/workspaces/Graphormer/distributional_graphormer/protein/run_inference.py",
                    "-c", "/workspaces/Graphormer/distributional_graphormer/protein/checkpoints/checkpoint-520k.pth",
                    "-i", os.path.join(final_out,  f"{pdb_name}.pkl"),
                    "-s", os.path.join(final_out,  f"{pdb_name}.fasta"),
                    "-o", pdb_name,
                    "--output-prefix", os.path.join(final_out, "dig_vanilla_with_init/"),
                    "--init-state", os.path.join(final_out, f"{pdb_name}.init_state.npz"),
                    "-n", str(n_samples),
                    "--use-gpu",
                    "--use-tqdm"])
    return Path(os.path.join(final_out, "dig_vanilla_with_init"))

def runall(output_dir_all, all_pdb_folder, n_samples=500):
    pdbs = list(Path(all_pdb_folder).glob("*.pdb"))
    if not os.path.exists(output_dir_all):
        os.makedirs(output_dir_all)
    for pdb_path in pdbs:
        pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
        output_dir=os.path.join(output_dir_all,pdb_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tmp_out_dir = os.path.join(output_dir, pdb_name)
        if os.path.exists(os.path.join(output_dir, pdb_name+".pkl")):
            print(f"{pdb_name} pkl already exists.")
            #copy the pkl to the preprocess_out
            shutil.copyfile(os.path.join(output_dir, pdb_name+".pkl"), os.path.join(output_dir, pdb_name,pdb_name+".pkl"))
        if not os.path.exists(tmp_out_dir):
            os.makedirs(tmp_out_dir)
        preprocess_out=run(output_dir, pdb_path,  pdb_name, linker_sequence="GGGGS" * 3)
        vanilla_no_init_frame_dir=run_vanilla_no_init_state(preprocess_out, pdb_name, n_samples=n_samples)


if __name__ == "__main__":
    all_cory_pdbs="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory"
    output_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig"
    runall(output_dir, all_cory_pdbs)
    #quit()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the input PDB file")
    parser.add_argument("--output_dir", type=str,required=True, help="Directory to save the OpenFold output")
    args = parser.parse_args()

    pdb_path = args.pdb_path
    output_dir = args.output_dir
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    output_dir = os.path.join(output_dir, pdb_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tmp_out_dir = os.path.join(output_dir, "tmp_"+pdb_name)
    if not os.path.exists(tmp_out_dir):
        os.makedirs(tmp_out_dir)
    run("/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory/1KGC.pdb",
        "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/input_pdbs_cory/1KGC.pdb",
        "1KGC",
        "GGGGS" * 3)
