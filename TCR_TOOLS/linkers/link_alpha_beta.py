from Bio.PDB import PDBParser, PDBIO, Chain, Model, Structure, Residue, Atom
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
import argparse
from Bio.SeqUtils import seq1, seq3
from Bio.PDB import Chain, Model, Structure
from Bio.Seq import Seq
from pathlib import Path
import json
import sys
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.linkers.build_linker_modeller import build_model

import os
import glob
SANITIZE_MAP = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
    "CYX": "CYS",
    "MSE": "MET",
    "SEC": "CYS",   # or "SEC" if you have parameters; MODELLER doesn't by default
    "SEP": "SER", "TPO": "THR", "PTR": "TYR",
}


def sanitize_pdb_for_modeller(structure, output_pdb):
    """
    Write a sanitized copy for MODELLER and return a mapping:
    {(chain_id, resseq, icode): (orig_resname, new_resname)}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", structure)

    mapping = {}  # key=(chain, resseq, icode) -> (orig, new)
    for model in structure:
        for chain in model:
            for res in chain:
                orig = res.get_resname().strip().upper()
                new = SANITIZE_MAP.get(orig, orig)
                if new != orig:
                    mapping[(chain.id, res.id[1], res.id[2])] = (orig, new)
                    res.resname = new

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    return mapping



def renumber_pdb_consecutively(pdb_path, output_path="renumbered.pdb", chain_id="A"):
    """
    Renumber all residues consecutively across a single chain and save to a new PDB file.
    If the input has multiple chains, only the specified chain is used.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    new_residues = []
    for new_idx, res in enumerate(chain.get_residues(), start=1):
        # Update residue ID (keep same insertion code and hetero flag)
        res.id = (' ', new_idx, ' ')
        new_residues.append(res)

    # Create a new structure with the renumbered residues
    new_chain = Chain.Chain(chain_id)
    for res in new_residues:
        new_chain.add(res)

    new_model = Model.Model(0)
    new_model.add(new_chain)

    new_structure = Structure.Structure("renumbered")
    new_structure.add(new_model)

    # Save
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)
    print(f"✅ Saved renumbered PDB to: {output_path}")
    return output_path

def create_linker_residues(start_id,LINKER_SEQUENCE):
    linker_residues = []
    for i, aa in enumerate(LINKER_SEQUENCE):
        #one to three letter code

        resname = seq3(aa)
        resname = 'GLY' if aa == 'G' else 'SER'
        res_id = (' ', start_id + i, ' ')
        residue = Residue.Residue(res_id, resname, '')

        # Add full backbone
        for j, atom_name in enumerate(['N', 'CA', 'C', 'O']):
            atom = Atom.Atom(
                name=atom_name,
                coord=np.array([0.0 + j, 0.0, 0.0]),  # Spread atoms slightly to avoid colocation
                bfactor=0.0,
                occupancy=1.0,
                altloc=' ',
                fullname=f"{atom_name:>4}",
                serial_number=10000 + i * 4 + j,
                element=atom_name[0]
            )
            residue.add(atom)

        linker_residues.append(residue)

    return linker_residues

def build_linked_chain(alpha_residues, linker_residues, beta_residues, beta_offset=200):
    new_chain = Chain.Chain('A')

    # Add alpha residues as-is
    for res in alpha_residues:
        new_chain.add(res.copy())

    # Add linker
    for res in linker_residues:
        new_chain.add(res)

    # Add beta residues with offset
    for res in beta_residues:
        res_copy = res.copy()
        old_id = res_copy.id
        res_copy.id = (' ', old_id[1] + beta_offset, old_id[2])
        new_chain.add(res_copy)

    return new_chain

def save_structure(chain, out_path):
    structure = Structure.Structure('linked')
    model = Model.Model(0)
    model.add(chain)
    structure.add(model)

    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    print(f"✅ Saved linked structure to {out_path}")

def make_dummy_pdb(pdb_path, output_pdb, alpha_chain_id, beta_chain_id,LINKER_SEQUENCE):
    model= PDBParser(QUIET=True).get_structure("input", pdb_path)[0]
    alpha_residues = list(model[alpha_chain_id].get_residues())
    beta_residues = list(model[beta_chain_id].get_residues())
    alpha_sequence = "".join([seq1(SANITIZE_MAP.get(res.get_resname().strip().upper(),res.get_resname())) for res in alpha_residues])
    beta_sequence = "".join([seq1(SANITIZE_MAP.get(res.get_resname().strip().upper(),res.get_resname())) for res in beta_residues])

    # Linker starts right after last alpha residue
    last_alpha_res_id = alpha_residues[-1].id[1]
    linker_residues = create_linker_residues(last_alpha_res_id + 1, LINKER_SEQUENCE)

    # Combine all
    linked_chain = build_linked_chain(alpha_residues, linker_residues, beta_residues)

    # Save output
    save_structure(linked_chain, output_pdb)
    return output_pdb, alpha_sequence, beta_sequence

def remove_linker_by_sequence(pdb_lines, linker_sequence, chain_id='A'):
    """
    Removes linker residues from PDB lines using a known linker sequence.
    This approach assumes residues are ordered and uses a sliding window.
    """

    # Collect residues in order
    residues = []
    res_indices = []
    current_res = None
    current_res_seq = ""

    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')) and line[21] == chain_id:
            res_id = line[22:26].strip()
            resname = line[17:20].strip()
            key = (res_id, resname)

            if key != current_res:
                try:
                    aa = seq1(resname)
                    current_res_seq += aa
                    res_indices.append(res_id)
                    current_res = key
                except KeyError:
                    # Skip unknown or non-standard residue
                    continue

    # Find linker position in the sequence
    linker_seq_str = "".join(linker_sequence)
    idx = current_res_seq.find(linker_seq_str)
    if idx == -1:
        raise ValueError("Linker sequence not found in chain.")

    linker_ids = set(res_indices[idx:idx+len(linker_sequence)])

    # Remove linker lines
    cleaned_lines = []
    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')) and line[21] == chain_id:
            res_id = line[22:26].strip()
            if res_id in linker_ids:
                continue  # skip linker
        cleaned_lines.append(line)

    return cleaned_lines, linker_ids

def fully_remove_linker(consec_linked_dummy, linker_sequence, linked_pdb_no_linker):
    with open(consec_linked_dummy) as f:
        pdb_lines = f.readlines()

    cleaned,linker_ids = remove_linker_by_sequence(pdb_lines, linker_sequence, chain_id="A")

    with open(linked_pdb_no_linker, "w") as f:
        f.writelines(cleaned)

    return linked_pdb_no_linker,linker_ids

#Ouptut PDB has dummy linker which will hve very wrong structure, need to use modeller
def write_modeller_alignment(alpha_sequence,beta_sequence, linker_sequence, alignment_path="alignment.ali", model_name="scfv_model"):
    """Generates a Modeller .ali file from a single-chain PDB structure."""
    full_sequence = alpha_sequence + linker_sequence + beta_sequence
    full_sequence_nolinker = alpha_sequence + "-"*len(linker_sequence) + beta_sequence

    chain_id = "A"
    with open(alignment_path, "w") as f:
        f.write(f">P1;{model_name}\n")
        f.write(f"structureX:{model_name}:1:{chain_id}:{len(full_sequence)}:{chain_id}::::\n")
        f.write(full_sequence_nolinker + "*\n\n")
        f.write(f">P1;{model_name}_output\n")
        f.write(f"sequence:{model_name}_output:::::::0.00: 0.00\n")
        f.write(full_sequence + "*\n")

    print(f"✅ Written alignment file to: {alignment_path}")
    print(f"✅ Chain ID detected: {chain_id}")
    print(f"✅ Sequence length: {len(full_sequence)} residues")

def modeller_run(linker_dir,alignment_file,model_name,linker_pos_start,linker_pos_end):
    #original_cwd = os.getcwd()
    os.chdir(linker_dir)
    build_model( linker_dir, alignment_file, model_name,
                linker_pos_start, linker_pos_end)
    #os.chdir(original_cwd)


def run_build(pdb_path,structure, alignment_path,alpha_chain_id="A", beta_chain_id="B", LINKER_SEQUENCE="GGGGS"*3):
    sanitized_pdb_path = pdb_path.replace(".pdb", "_sanitized.pdb")
    dummy_linked_pdb_path = sanitized_pdb_path.replace(".pdb", "_output.pdb")
    linked_pdb_consec = dummy_linked_pdb_path.replace(".pdb", "_consec.pdb")
    linked_pdb_no_linker = linked_pdb_consec.replace(".pdb", "_no_linker.pdb")


    dummy_linked_pdb_path, alpha_sequence, beta_sequence=make_dummy_pdb(pdb_path, dummy_linked_pdb_path, alpha_chain_id, beta_chain_id,LINKER_SEQUENCE)
    dummy_linked_pdb_path=renumber_pdb_consecutively(dummy_linked_pdb_path, dummy_linked_pdb_path, chain_id="A")
    mapping=sanitize_pdb_for_modeller(dummy_linked_pdb_path, dummy_linked_pdb_path)
    linked_pdb_no_linker,linker_ids=fully_remove_linker(dummy_linked_pdb_path, LINKER_SEQUENCE, linked_pdb_no_linker)

    write_modeller_alignment(alpha_sequence,beta_sequence, LINKER_SEQUENCE, alignment_path, model_name=linked_pdb_no_linker.split("/")[-1].replace(".pdb",""))

    linker_start = len(alpha_sequence) + 1  # 1-based index
    linker_end = linker_start + len(LINKER_SEQUENCE) +1

    # Determine linker positions in the renumbered PDB

    modeller_run(os.path.dirname(linked_pdb_consec), alignment_path, model_name=linked_pdb_no_linker.split("/")[-1].replace(".pdb",""), linker_pos_start=linker_start, linker_pos_end=linker_end)
    final=cleanup_modeller_outputs(os.path.dirname(linked_pdb_consec), keep=pdb_path.split("/")[-1])
    final_out=pdb_path.replace(".pdb", "_linked.pdb")
    reverse_sanitize_pdb(final, mapping, final_out)
    check_output(final_out, pdb_path, alpha_chain_id=alpha_chain_id, beta_chain_id=beta_chain_id, LINKER_SEQUENCE=LINKER_SEQUENCE)
    os.remove(final)
    return final_out

def cleanup_modeller_outputs(directory, keep_pattern="*.BL*.pdb", keep=None):
    os.chdir(directory)
    keep_files = list(glob.glob(keep_pattern))
    final_modeler= list(keep_files)[0]
    all_files = set(glob.glob("*"))
    if keep:
        print(keep)
        keep_files.append(keep)
    to_delete = all_files - set(keep_files)

    for file in to_delete:
        os.remove(file)
        print(f"Deleted: {file}")
    for file in keep_files:
        print(f"Kept: {file}")
    return final_modeler


def check_output(output_pdb, input_pdb, alpha_chain_id="A", beta_chain_id="B",LINKER_SEQUENCE="GGGGS"*3):
    model= PDBParser(QUIET=True).get_structure("input", input_pdb)[0]
    alpha_residues = list(model[alpha_chain_id].get_residues())
    beta_residues = list(model[beta_chain_id].get_residues())
    alpha_sequence = "".join([seq1(res.get_resname()) for res in alpha_residues])
    beta_sequence = "".join([seq1(res.get_resname()) for res in beta_residues])
    model_out= PDBParser(QUIET=True).get_structure("output", output_pdb)[0]
    out_seq = list(model_out[alpha_chain_id].get_residues())
    out_sequence = "".join([seq1(res.get_resname()) for res in out_seq])

    full_sequence = alpha_sequence + LINKER_SEQUENCE + beta_sequence
    assert full_sequence == out_sequence, f"Output sequence does not match expected. Expected: {full_sequence}, Got: {out_sequence}"

def reverse_sanitize_pdb(sanitized_pdb, mapping, output_pdb):
    """
    Reverse the sanitization process by restoring original residue names.
    `mapping` is the dictionary returned by `sanitize_pdb_for_modeller`.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sanitized", sanitized_pdb)

    for model in structure:
        for chain in model:
            for res in chain:
                key = (chain.id, res.id[1], res.id[2])
                if key in mapping:
                    orig_resname, _ = mapping[key]
                    res.resname = orig_resname

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"✅ Reversed sanitization and saved to: {output_pdb}")

def run_standalone(tcr1, output_folder,LINKER_SEQUENCE):
    tcr = TCR(input_pdb=tcr1)
    tcr1=tcr.pairs[0]
    variable_tcr1=tcr1.variable_structure()
    variable_out=os.path.join(output_folder, "tcr_variable.pdb")
    write_pdb(variable_out, variable_tcr1)
    alpha_chain_id=tcr1.alpha_chain_id
    beta_chain_id=tcr1.beta_chain_id
    alignment_path=f"{output_folder}/linked_variable_imgt.ali"
    final_out=run_build(variable_out, variable_tcr1,alignment_path,alpha_chain_id, beta_chain_id, LINKER_SEQUENCE)
    return final_out

def run(tcr1, output_folder,LINKER_SEQUENCE):
    variable_tcr1 = tcr1.variable_structure
    variable_out=os.path.join(output_folder, "tcr_variable.pdb")
    write_pdb(variable_out, variable_tcr1)
    alpha_chain_id=tcr1.alpha_chain_id
    beta_chain_id=tcr1.beta_chain_id
    alignment_path=f"{output_folder}/linked_variable_imgt.ali"
    final_out=run_build(variable_out, variable_tcr1,alignment_path,alpha_chain_id, beta_chain_id, LINKER_SEQUENCE)
    return final_out

if __name__=="__main__":
    LINKER_SEQUENCE = "GGGGS" * 3  # 15 residues
    output_folder="/workspaces/Graphormer/TCR_Metrics/test"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    final_out=run("/mnt/larry/lilian/DATA/Cory_data/6OVN/6OVN.pdb", output_folder,LINKER_SEQUENCE)

