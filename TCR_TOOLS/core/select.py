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


def make_idx_atom_predicate(
    A_residue_list: List[int],
    B_residue_list: List[int],
    chain_map: Dict[str, str],
    atom_names: Optional[Set[str]] = None,
    include_het: bool = True,
):
    """
    Returns a predicate:

        pred(chain_id, residue, atom_name) -> bool

    that is True iff:
      - the residue belongs to the requested residue list for that chain
        (A_residue_list for alpha, B_residue_list for beta), and
      - the atom_name passes the optional atom filter, and
      - (optionally) hetero residues are included.

    Parameters
    ----------
    A_residue_list : list of int
        Residue numbers (resseq) to include on the alpha chain.
    B_residue_list : list of int
        Residue numbers (resseq) to include on the beta chain.
    chain_map : dict
        Maps logical names {"alpha", "beta"} to actual chain IDs in the PDB.
    atom_names : set of str, optional
        Allowed atom names (e.g. {"CA", "N"}). If None, all atoms are allowed.
    include_het : bool
        If False, hetero residues (hetflag != " ") are excluded.
    """

    atom_set = {a.strip().upper() for a in atom_names} if atom_names is not None else None

    # Map residue lists to actual chain IDs and convert to sets for fast lookup
    residues_by_chain: Dict[str, Set[int]] = {}

    A_cid = chain_map.get("alpha")
    B_cid = chain_map.get("beta")

    if A_cid is not None and A_residue_list is not None:
        residues_by_chain[A_cid] = set(int(r) for r in A_residue_list)

    if B_cid is not None and B_residue_list is not None:
        residues_by_chain[B_cid] = set(int(r) for r in B_residue_list)

    def _res_ok(chain_id: str, residue: BPResidue) -> bool:
        hetflag, resseq, _icode = residue.id  # e.g. (' ', 100, ' ') or ('H_MSE', 25, ' ')
        if not include_het and hetflag.strip() != "":
            return False

        # If no residue filters are defined at all, accept everything
        if not residues_by_chain:
            return True

        if chain_id not in residues_by_chain:
            return False

        rnum = int(resseq)
        return rnum in residues_by_chain[chain_id]

    def _atom_ok(atom_name: Optional[str]) -> bool:
        if atom_name is None:
            # residue-level query → don't filter on atom
            return True
        if atom_set is None:
            # no atom filter requested
            return True
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
