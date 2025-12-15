from Bio.PDB import PDBIO, PDBParser, Structure as BPStructure
import biotite.structure as bts
import numpy as np
import os, tempfile
import mdtraj as md

def load_pdb(path: str, struct_id: str = "structure") -> BPStructure.Structure:
    return PDBParser(QUIET=True).get_structure(struct_id, path)

def write_pdb(path: str, structure: BPStructure.Structure) -> None:
    """
    Write a PDB file with guaranteed numeric occupancy and B-factor
    for all ATOM/HETATM records, so that Biotite can parse it.

    - If occupancy is None, set it to 1.0
    - If B-factor is None, set it to 0.0
    """
    # Fix missing occupancies / B-factors in-place
    for atom in structure.get_atoms():
        if atom.get_occupancy() is None:
            atom.set_occupancy(1.0)
            #warning: Biotite cannot handle None occupancy
        if atom.get_bfactor() is None:
            atom.set_bfactor(0.0)
            #warning: Biotite cannot handle None B-factor

    io = PDBIO()
    io.set_structure(structure)
    io.save(path)

def structure_to_atomarray(structure: BPStructure.Structure) -> bts.AtomArray:
    model0 = next(structure.get_models())
    rows = []
    for chain in model0:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            icode = icode or " "
            for atom in residue:
                rows.append((
                    chain.id, residue.get_resname(), int(resseq),
                    bool(hetflag != " "), str(hetflag),
                    atom.get_name().strip(), atom.coord.astype(float),
                    (atom.element or ""), float(atom.get_bfactor()),
                    float(atom.get_occupancy() or 1.0),
                    (atom.get_altloc() or ""), str(icode),
                ))
    if not rows:
        return bts.AtomArray(0)

    chain_id  = np.array([r[0] for r in rows], dtype=object)
    res_name  = np.array([r[1] for r in rows], dtype=object)
    res_id    = np.array([r[2] for r in rows], dtype=int)
    hetero    = np.array([r[3] for r in rows], dtype=bool)
    het_flag  = np.array([r[4] for r in rows], dtype=object)
    atom_name = np.array([r[5] for r in rows], dtype=object)
    coord     = np.stack([r[6] for r in rows]).astype(float)
    element   = np.array([r[7] for r in rows], dtype=object)
    b_factor  = np.array([r[8] for r in rows], dtype=float)
    occupancy = np.array([r[9] for r in rows], dtype=float)
    altloc    = np.array([r[10] for r in rows], dtype=object)
    ins_code  = np.array([r[11] for r in rows], dtype=object)

    arr = bts.AtomArray(coord.shape[0])
    arr.coord = coord
    arr.chain_id = chain_id
    arr.res_name = res_name
    arr.res_id = res_id
    arr.hetero = hetero
    arr.atom_name = atom_name
    arr.element = element
    arr.b_factor = b_factor
    arr.occupancy = occupancy
    arr.altloc_id = altloc
    # store extras
    setattr(arr, "ins_code", ins_code)
    setattr(arr, "het_flag", het_flag)
    return arr

def atomarray_to_structure(atom_array: bts.AtomArray) -> BPStructure.Structure:
    from Bio.PDB import Structure as BPStructure, Model as BPModel, Chain as BPChain, Residue as BPResidue, Atom as BPAtom
    from collections import defaultdict

    new_struct = BPStructure.Structure("from_biotite")
    model = BPModel.Model(0); new_struct.add(model)

    chain_ids = atom_array.chain_id
    res_ids   = atom_array.res_id
    res_names = atom_array.res_name
    atom_names= atom_array.atom_name
    coords    = atom_array.coord
    elements  = getattr(atom_array, "element", None)
    b_factors = getattr(atom_array, "b_factor", None)
    occs      = getattr(atom_array, "occupancy", None)
    altlocs   = getattr(atom_array, "altloc_id", None)
    ins_codes = getattr(atom_array, "ins_code", None)
    het_flags = getattr(atom_array, "het_flag", None)
    hetero    = getattr(atom_array, "hetero", None)

    by_chain = defaultdict(lambda: defaultdict(list))
    for i in range(atom_array.array_length()):
        icode = str(ins_codes[i]) if ins_codes is not None else " "
        hetf  = str(het_flags[i]) if het_flags is not None else ("H_" if (hetero is not None and bool(hetero[i])) else " ")
        key = (int(res_ids[i]), icode, str(res_names[i]), hetf)
        by_chain[str(chain_ids[i])][key].append(i)

    for cid, res_map in by_chain.items():
        chain = BPChain.Chain(cid)
        for (resseq, icode, resname, hetf), idxs in sorted(res_map.items()):
            res = BPResidue.Residue((hetf, int(resseq), str(icode)), str(resname), segid="")
            for i in idxs:
                elem = str(elements[i]) if elements is not None else None
                bfac = float(b_factors[i]) if b_factors is not None else 0.0
                occ  = float(occs[i]) if occs is not None else 1.0
                alt  = str(altlocs[i]) if altlocs is not None else ""
                atom = BPAtom.Atom(
                    name=str(atom_names[i]),
                    coord=np.asarray(coords[i], dtype=float),
                    bfactor=bfac, occupancy=occ, altloc=alt,
                    fullname=str(atom_names[i]).rjust(4),
                    serial_number=i+1, element=elem
                )
                res.add(atom)
            chain.add(res)
        model.add(chain)
    return new_struct



def mdtraj_from_biopython_path(structure):
    """Convert a Bio.PDB Structure -> mdtraj.Trajectory via a temp PDB file."""
    io = PDBIO(); io.set_structure(structure)
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        io.save(tmp.name)
        path = tmp.name
    try:
        ref_md = md.load(path)  # single-frame Trajectory
    finally:
        try: os.remove(path)
        except OSError: pass
    return ref_md