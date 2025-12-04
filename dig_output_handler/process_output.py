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

def unlink_pdbs(output_folder, linker="GGGGS"*3, output_xtc_path=None, output_pdb_path=None):
    """removing the linker from each output pdb"""
    #one traj for all pdbs
    aligned_frames = []
    for file in os.listdir(output_folder):
        if file.endswith(".pdb"):
            input_pdb = os.path.join(output_folder, file)
            traj = md.load(input_pdb)
            if linker=="":
                split_traj = traj
            else:
                split_traj = split(traj, linker)

            aligned_frames.append(split_traj)
    combined_traj = md.join(aligned_frames)
    combined_traj.save_xtc(output_xtc_path)
    combined_traj[0].save_pdb(output_pdb_path)
    return output_xtc_path,output_pdb_path


def process_output(dig_output_dir, linker="GGGGS"*3, output_xtc_path=None, output_pdb_path=None):
    # Unlink chains from the output PDB files
    output_xtc_path, output_pdb_path=unlink_pdbs(dig_output_dir, linker=linker, output_xtc_path=output_xtc_path, output_pdb_path=output_pdb_path)
    return output_xtc_path, output_pdb_path

