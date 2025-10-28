from typing import Callable, Dict, List, Tuple, Optional, Set
# import your constants
from ..__init__ import CDR_FR_RANGES, VARIABLE_RANGE

from typing import Callable, Dict, List, Tuple, Optional, Set
from Bio.PDB.Residue import Residue as BPResidue
from Bio.PDB.Atom import Atom as BPAtom


def make_region_atom_predicate(
    regions: Optional[List[Tuple[str, Tuple[int, int]]]],
    chain_map: Dict[str, str],
    atom_names: Optional[Set[str]] = None,
    include_het: bool = True):
    """
    Returns pred(chain_id, residue, atom_name) -> bool

    - Regions:
        * If regions is None/empty: accept all residues (subject to include_het).
        * Else: accept residues whose numeric resseq is within any interval for that chain.
    - Atoms:
        * If atom_name is None: treat as residue-level query → ignore atom filter.
        * If atom_names is None: accept all atom names.
        * Else: require atom_name ∈ atom_names (case-insensitive).
    """
    atom_set = {a.strip().upper() for a in atom_names} if atom_names is not None else None

    intervals_by_chain: Dict[str, List[Tuple[int, int]]] = {}
    if regions:
        for rname, (s, e) in regions:
            cid = chain_map["alpha"] if rname.startswith("A_") else chain_map["beta"]
            intervals_by_chain.setdefault(cid, []).append((int(s), int(e)))

    def _res_ok(chain_id: str, residue: BPResidue) -> bool:
        hetflag, resseq, _icode = residue.id  # (' ', 100, 'A') or ('H_MSE', 25, ' ')
        if not include_het and hetflag.strip() != "":
            return False
        if not intervals_by_chain:
            return True
        if chain_id not in intervals_by_chain:
            return False
        rnum = int(resseq)
        return any(s <= rnum <= e for (s, e) in intervals_by_chain[chain_id])

    def _atom_ok(atom_name: Optional[str]) -> bool:
        if atom_name is None:
            return True                       # residue-level query → don't atom-filter
        if atom_set is None:
            return True                       # no atom filter requested
        return atom_name.strip().upper() in atom_set

    def pred(chain_id: str, residue: BPResidue, atom_name: Optional[str] = None) -> bool:
        return _res_ok(chain_id, residue) and _atom_ok(atom_name)

    return pred



def region_list_for_names(region_names, variable_range=VARIABLE_RANGE):
    regions = []
    for name in region_names:
        if name in ("A_variable","B_variable"):
            regions.append((name, variable_range))
        else:
            regions.append((name, CDR_FR_RANGES[name]))
    return regions
