from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import Structure as BPStructure, Model as BPModel, Chain as BPChain, Residue as BPResidue, Atom as BPAtom
from Bio.SeqUtils import seq1

def copy_subset(
    struct: BPStructure.Structure,
    pred: Callable[[str, BPResidue.Residue, Optional[str]], bool],
) -> BPStructure.Structure:
    """
    Keep atoms that satisfy pred(chain_id, residue, atom_name).
    Drops empty residues/chains. Preserves order & residue ids (incl. insertion codes).
    """
    new_struct = BPStructure.Structure(str(struct.id) + "_subset")
    new_model = BPModel.Model(0)
    new_struct.add(new_model)

    model0 = next(struct.get_models())
    for chain in model0:
        new_chain = BPChain.Chain(chain.id)
        kept_any = False
        for residue in chain:
            chain_id = chain.id
            hetflag, resseq, icode = residue.id
            res_copy = BPResidue.Residue((hetflag, int(resseq), str(icode)), residue.get_resname(), residue.segid)
            atom_kept = False
            for atom in residue.get_atoms():
                if pred(chain_id, residue, atom.get_name()):
                    a = BPAtom.Atom(
                        name=atom.get_name(),
                        coord=np.asarray(atom.coord, dtype=float),
                        bfactor=float(atom.get_bfactor()),
                        occupancy=float(atom.get_occupancy() or 1.0),
                        altloc=atom.get_altloc(),
                        fullname=atom.get_fullname(),
                        serial_number=atom.get_serial_number(),
                        element=atom.element
                    )
                    res_copy.add(a)
                    atom_kept = True
            if atom_kept:
                new_chain.add(res_copy)
                kept_any = True
        if kept_any:
            new_model.add(new_chain)
    return new_struct

def sanitize_structure(structure: BPStructure.Structure) -> BPStructure.Structure:
    """
    Return a NEW structure with common non-standard residues mapped to standards.
    Does not mutate input.
    """
    SANITIZE_MAP = {
        "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
        "CYX": "CYS", "MSE": "MET", "SEC": "CYS",
        "SEP": "SER", "TPO": "THR", "PTR": "TYR",
    }
    new_struct = BPStructure.Structure(structure.id + "_sanitized")
    new_model = BPModel.Model(0); new_struct.add(new_model)

    model0 = next(structure.get_models())
    for chain in model0:
        new_chain = BPChain.Chain(chain.id)
        for res in chain:
            hetflag, resseq, icode = res.id
            resname = SANITIZE_MAP.get(res.get_resname().strip().upper(), res.get_resname())
            res_copy = BPResidue.Residue((hetflag, int(resseq), str(icode)), resname, res.segid)
            for atom in res.get_atoms():
                res_copy.add(atom.copy())
            new_chain.add(res_copy)
        new_model.add(new_chain)
    return new_struct

def get_sequence_dict(structure: BPStructure.Structure) -> Dict[str, str]:
    """Return {chain_id: sequence} without modifying the structure."""
    model0 = next(structure.get_models())
    out = {}
    for chain in model0:
        seq = []
        for res in chain:
            try: seq.append(seq1(res.get_resname()))
            except Exception: seq.append("X")
        out[chain.id] = "".join(seq)
    return out
