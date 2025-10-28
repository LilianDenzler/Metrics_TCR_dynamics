import os, shutil
import mdtraj as md
import numpy as np
from Bio.SeqUtils import seq1
import glob


def split(traj, linker):
    seq = ''.join(seq1(res.name) for res in traj.topology.residues)
    if linker not in seq:
        raise ValueError("Linker sequence not found in input PDB.")
    # Slice trajectory
    linker_idx = seq.find(linker)
    len_A = linker_idx
    len_linker = len(linker)
    residues_A = list(traj.topology.residues)[:len_A]
    residues_B = list(traj.topology.residues)[len_A + len_linker:]
    atoms_A = [atom.index for res in residues_A for atom in res.atoms]
    atoms_B = [atom.index for res in residues_B for atom in res.atoms]

    traj_A = traj.atom_slice(atoms_A)
    traj_B = traj.atom_slice(atoms_B)
    combined_traj = traj_A.stack(traj_B)
    return combined_traj

def unlink_pdbs(output_folder, linker="GGGGS"*3, output_xtc_path=None):
    """removing the linker from each output pdb"""
    #one traj for all pdbs
    aligned_frames = []
    for file in os.listdir(output_folder):
        if file.endswith(".pdb"):
            input_pdb = os.path.join(output_folder, file)
            traj = md.load(input_pdb)
            split_traj = split(traj, linker)
            aligned_frames.append(split_traj)
    combined_traj = md.join(aligned_frames)
    combined_traj.save_xtc(output_xtc_path)
    return combined_traj, output_xtc_path


def process_output(dig_output_dir, input_pdb,linker="GGGGS"*3, output_xtc_path=None,input_unlinked_pdb_path=None):
    traj_input_linked = md.load(input_pdb)
    traj_input_unlinked = split(traj_input_linked, linker)
    # Save unlinked input PDB for reference
    traj_input_unlinked.save_pdb(input_unlinked_pdb_path)
    # Unlink chains from the output PDB files
    combined_traj, output_xtc_path=unlink_pdbs(dig_output_dir, linker=linker, output_xtc_path=output_xtc_path)
    return output_xtc_path, input_unlinked_pdb_path

if __name__ == "__main__":
    input_unlinked_pdb_path = "/workspaces/Graphormer/TCR_Metrics/test/unlinked_dig.pdb"
    output_xtc_path="/workspaces/Graphormer/TCR_Metrics/test/unlinked_dig.xtc"
    dig_output_dir = "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig2/1KGC/dig_vanilla"
    input_pdb="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig2/1KGC/tcr_variable_linked.pdb"
    linker="GGGGS"*3
    output_xtc_path, input_unlinked_pdb_path=process_output(dig_output_dir, input_pdb,linker, output_xtc_path,input_unlinked_pdb_path)
