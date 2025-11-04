# tcr.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set, Callable

import os
import tempfile
import numpy as np
import mdtraj as md
from Bio.PDB import (
    PDBParser,
    Structure as BPStructure,
    Model as BPModel,
    Chain as BPChain,
    Residue as BPResidue,
    Atom as BPAtom,
)

# --- project imports (pure utilities & constants) ---
from TCR_TOOLS.core import ops, io
import tempfile, os

from TCR_TOOLS.core.select import make_region_atom_predicate, region_list_for_names
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE

# --- numbering/pairing/alignment (your existing modules) ---
from TCR_TOOLS.numbering.main import fix_duplicate_resseqs_by_insertion_code,apply_imgt_renumbering
from TCR_TOOLS.numbering.tcr_pairing import pair_tcrs_by_interface

from TCR_TOOLS.geometry.calc_geometry_MD import run as calc_traj_angles
# -------------------------------------------------------------------
# Trajectory view (thin helper over mdtraj)
# -------------------------------------------------------------------
class TrajectoryView:
    """
    Thin wrapper around an mdtraj.Trajectory that knows how to slice by IMGT regions.
    It expects: chain ids in the topology match those in the provided chain_map.
    """

    def __init__(self, traj: md.Trajectory, chain_map: Dict[str, str]):
        """
        Params
        ------
        traj : md.Trajectory
            The attached trajectory with a topology consistent with IMGT numbering
            (or consistently chosen numbering).
        chain_map : {"alpha": "<chain_id>", "beta": "<chain_id>"}
            Chain ids as they appear in the trajectory topology.
        """
        self._traj = traj
        self._top = traj.topology
        self._chain_map = chain_map  # e.g., {"alpha": "A", "beta": "B"}

    @property
    def mdtraj(self) -> md.Trajectory:
        return self._traj
    @property
    def topology(self):
        return self._traj.topology
    @property
    def n_frames(self):
        return self._traj.n_frames

    def _region_atom_indices(
        self,
        region_names: List[str],
        cdr_fr_ranges: Dict[str, Tuple[int, int]],
        variable_range: Tuple[int, int],
        atom_names: Optional[Set[str]] = None,
        pass_names=False
    ) -> List[int]:
        # Build interval map per chain in traj
        intervals_by_chain: Dict[str, List[Tuple[int, int]]] = {}
        for name in region_names:
            if name in ("A_variable", "B_variable"):
                start, end = variable_range
                cid = self._chain_map["alpha"] if name.startswith("A_") else self._chain_map["beta"]
            else:
                if name not in cdr_fr_ranges:
                    raise KeyError(f"Unknown region: {name}")
                start, end = cdr_fr_ranges[name]
                cid = self._chain_map["alpha"] if name.startswith("A_") else self._chain_map["beta"]
            intervals_by_chain.setdefault(cid, []).append((start, end))

        atom_idxs: List[int] = []
        atom_list=[]
        for atom in self._top.atoms:
            res = atom.residue
            if res is None or res.chain is None:
                continue
            chain_obj = res.chain
            # mdtraj Topology has .chain_id for PDB; fall back to index if absent
            chain_id = getattr(chain_obj, "chain_id", None) or str(chain_obj.index)

            if chain_id not in intervals_by_chain:
                continue
            resnum = int(getattr(res, "resSeq", res.index))  # IMGT numbering expected if serialized from IMGT PDB
            for (s, e) in intervals_by_chain[chain_id]:
                if s <= resnum <= e:
                    if atom_names is None or atom.name in atom_names:
                        print(f"Selecting atom: chain={chain_id} resnum={resnum} name={atom.name}")
                        atom_idxs.append(atom.index)
                        atom_list.append((chain_id, resnum, atom.name))
                    break
        if atom_idxs == []:
            raise ValueError(f"No atoms matched regions={region_names} (atom_names={atom_names}).")
        if pass_names:
            return atom_idxs, atom_list
        else:
            return atom_idxs

    def domain_idx(
        self,
        region_names: List[str],
        atom_names: Optional[Set[str]] = None,
        cdr_fr_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        variable_range: Optional[Tuple[int, int]] = None,
        pass_names: bool = False
    ) -> List[int]:
        if cdr_fr_ranges is None:
            cdr_fr_ranges = CDR_FR_RANGES
        if variable_range is None:
            variable_range = VARIABLE_RANGE
        return self._region_atom_indices(region_names, cdr_fr_ranges, variable_range, atom_names, pass_names=pass_names)

    def domain_subset(
        self,
        region_names: List[str],
        atom_names: Optional[Set[str]] = None,
        cdr_fr_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        variable_range: Optional[Tuple[int, int]] = None,
        inplace: bool = False,
    ) -> md.Trajectory:
        idxs = self.domain_idx(region_names, atom_names, cdr_fr_ranges, variable_range)
        if inplace:
            self._traj.atom_slice(idxs, inplace=True)
            return self._traj
        return self._traj.atom_slice(idxs, inplace=False)

    def align_to_ref(self, pv_ref: TCRPairView,
                     region_names: List[str],
                     atom_names: Optional[Set[str]] = None,
                     inplace: bool = False):
        ref=io.mdtraj_from_biopython_path(pv_ref.full_structure)
        ref_idx, refnames,  = pv_ref.domain_idx(region_names=region_names, atom_names=atom_names, pass_names=True)   # Bio.PDB order
        traj_idx, trajnames = self.domain_idx(region_names=region_names, atom_names=atom_names,pass_names=True)     # MDTraj order

        if len(ref_idx) != len(traj_idx):
            shared = [x for x in refnames if x in trajnames]

            # Filter TCR A to keep only shared elements
            mask_a = [x in shared for x in refnames]
            ref_idx = [x for x, keep in zip(ref_idx, mask_a) if keep]

            # Filter TCR B to keep only shared elements
            mask_b = [x in shared for x in trajnames]
            traj_idx = [x for x, keep in zip(traj_idx, mask_b) if keep]
            if len(ref_idx) != len(traj_idx):
                raise ValueError(f"Index size mismatch: ref={len(ref_idx)} vs traj={len(traj_idx)}")
        work_traj = self._traj if inplace else self._traj[:]
        # Align in-place; get per-frame RMSD (nm)
        work_traj.superpose(
            ref,
            atom_indices=np.asarray(traj_idx, dtype=int),
            ref_atom_indices=np.asarray(ref_idx,  dtype=int),
            parallel=True
        )
        rmsds_nm=md.rmsd(work_traj,
                      ref,
                      atom_indices=np.asarray(traj_idx, dtype=int),
                      ref_atom_indices=np.asarray(ref_idx,  dtype=int),
                      parallel=True)

        rmsds_A = rmsds_nm * 10.0  # convert nm -> Å
        # Work on a copy if requested
        if inplace:
            # update self in-place and return self (plus optional rmsd)
            self._traj = work_traj
            self._top = work_traj.topology
            return (self, rmsds_A)
        else:
            # build a NEW TrajectoryView with the aligned traj copy
            work_traj.topology=self._top
            new_view = self.__class__(work_traj, self._chain_map)
            new_view._traj.topology = work_traj.topology
            new_view._top = work_traj.topology
            return (new_view, rmsds_A)


# -------------------------------------------------------------------
# Small helper: rename two chains (pure OR inplace variant)
# -------------------------------------------------------------------
def rename_two_chains_exact_inplace(
    structure: BPStructure.Structure,
    old_a: str,
    old_b: str,
    new_a: str,
    new_b: str,
) -> None:
    """In-place: change two chain IDs exactly."""
    model0 = next(structure.get_models())
    for chain in model0:
        if chain.id == old_a:
            chain.id = new_a
        elif chain.id == old_b:
            chain.id = new_b


# -------------------------------------------------------------------
# Per-pair view (renamed chains, cached subsets, linking)
# -------------------------------------------------------------------
@dataclass
class TCRPairView:
    """An in-memory view for a single paired receptor (post-IMGT, renamed for this view)."""

    index: int
    alpha_chain_id: str            # original chain id in source PDB (needed for traj mapping)
    beta_chain_id: str             # original chain id in source PDB
    alpha_type: str                # "A" or "G" in THIS view
    beta_type: str                 # "B" or "D" in THIS view
    full_structure: BPStructure.Structure   # IMGT-numbered; chains renamed to A/B or G/D for this view
    chain_map: Dict[str, str]               # {"alpha": "<A|G>", "beta": "<B|D>"}
    alpha_germline: Optional[str] = None
    beta_germline: Optional[str] = None

    # Backref to owning TCR + cached per-pair traj view
    tcr_owner: "TCR" = field(repr=False, default=None)
    _cached_traj_view: Optional[TrajectoryView] = None

    # Cached structures
    _variable_structure: Optional[BPStructure.Structure] = field(init=False, default=None, repr=False)
    _linked_structure: Optional[BPStructure.Structure] = field(init=False, default=None, repr=False)

    @property
    def variable_structure(self) -> BPStructure.Structure:
        """
        Lazily compute and cache the variable-domain structure (A/B or G/D variable ranges).
        """
        if self._variable_structure is None:
            regions = [("A_variable", VARIABLE_RANGE), ("B_variable", VARIABLE_RANGE)]
            pred = make_region_atom_predicate(
                regions=regions,
                chain_map=self.chain_map,
            )
            self._variable_structure = ops.copy_subset(self.full_structure, pred)
        return self._variable_structure

    def _pair_topology_from_biopdb(self,pair_full_structure) -> md.Topology:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            io.write_pdb(tmp.name, pair_full_structure)      # writes renamed A/B (or G/D), IMGT-numbered
            top = md.load_frame(tmp.name, index=0).topology  # read topology only
            return top
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass
    def domain_subset(self, region_names: Optional[List[str]]= None, atom_names: Optional[Set[str]]= None, include_het=True) -> BPStructure.Structure:
        """
        Get a NEW structure subset for provided region names (e.g., ["A_CDR1","B_CDR3"]).
        """
        regions = region_list_for_names(region_names, VARIABLE_RANGE)
        pred = make_region_atom_predicate(
            regions=regions,
            chain_map=self.chain_map,
            atom_names=atom_names,
            include_het=include_het
        )
        return ops.copy_subset(self.full_structure, pred)


    def domain_idx(
        self,
        region_names: Optional[List[str]] = None,
        atom_names: Optional[Set[str]] = None,
        include_het: bool = True,
        pass_names: bool = False
    ) -> List[int]:
        regions = region_list_for_names(region_names, VARIABLE_RANGE) if region_names else None
        pred = make_region_atom_predicate(regions, self.chain_map, atom_names, include_het)

        idxs: List[int] = []
        atom_list=[]
        for i, atom in enumerate(self.full_structure.get_atoms()):
            res = atom.get_parent()
            if res is None: continue
            chain = res.get_parent()
            if chain is None: continue
            if pred(chain.id, res, atom.get_name()):
                idxs.append(i)
                atom_list.append((chain.id,res.get_id()[1],atom.get_name()))

        if not idxs:
            raise ValueError(f"No atoms matched regions={region_names} (atom_names={atom_names}).")
        if pass_names:
            return idxs, atom_list
        else:
            return idxs


    def linked_structure(self, linker_sequence: str) -> BPStructure.Structure:
        """
        Build a linked variable-domain TCR using a temporary workspace.
        Returns a Bio.PDB Structure; all temp files are deleted afterwards.
        Caches result for subsequent calls.
        """
        if self._linked_structure is not None:
            return self._linked_structure
        from TCR_TOOLS.linkers.link_alpha_beta import run
        with tempfile.TemporaryDirectory(prefix="tcr_link_") as tmpdir:
            linked_struct_path = run(self, tmpdir, linker_sequence)
            if not linked_struct_path or not os.path.exists(linked_struct_path):
                raise FileNotFoundError("Linking failed (no output PDB).")
            parser = PDBParser(QUIET=True)
            linked_structure = parser.get_structure("linked", linked_struct_path)
            linked_structure=linked_structure[0]  # get first model
        self._linked_structure =linked_structure
        return self._linked_structure

    def cdr_fr_sequences(self) -> Dict[str, str]:
        """
        Return sequences for all CDR/FR regions from this pair view.
        """
        seqs: Dict[str, str] = {}
        for rname, (s, e) in CDR_FR_RANGES.items():
            # residue-level predicate for this single region
            regions = [(rname, (s, e))]
            pred=make_region_atom_predicate(
                regions, self.chain_map, atom_names=None, include_het=True
            )
            subset = ops.copy_subset(self.full_structure, pred)

            d = ops.get_sequence_dict(subset)  # {chain_id: seq}
            cid = self.chain_map["alpha"] if rname.startswith("A_") else self.chain_map["beta"]
            seqs[rname] = d.get(cid, "")
        return seqs


    @property
    def traj(self) -> TrajectoryView:
        """
        Per-pair trajectory view sliced from the parent TCR's attached trajectory.
        Usage:
            sub = pv1.traj.domain_subset(["A_CDR1","A_CDR2"])
        """
        if self._cached_traj_view is not None:

            return self._cached_traj_view

    def calc_angles_traj(self):
        #write as tmp pdb and tmp xtc
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp_xtc = tempfile.NamedTemporaryFile(suffix=".xtc", delete=False)
        tmptop=self.traj._traj[0]
        tmptop.save_pdb(tmp_pdb.name)
        self.traj._traj.save_xtc(tmp_xtc.name)
        angle_results=calc_traj_angles(tmp_xtc.name,tmp_pdb.name)
        return angle_results

    # Cache control
    def invalidate_caches(self) -> None:
        self._variable_structure = None
        self._linked_structure = None
        self._cached_traj_view = None

    def attach_trajectory(
        self,
        traj_path: str,
        region_names: Optional[List[str]] = None,   # None = all residues
        atom_names: Optional[Set[str]] = None,      # e.g. {"CA","C","N"} or add "O"
        include_het: bool = True,
    ) -> None:
        regions = region_list_for_names(region_names, VARIABLE_RANGE) if region_names else None
        pred = make_region_atom_predicate(regions, self.chain_map, atom_names, include_het)

        # single-pass subset that matches your backbone-only XTC selection
        sub_struct = ops.copy_subset(self.full_structure, pred)

        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            model0 = next(sub_struct.get_models())
            io.write_pdb(tmp.name, sub_struct)
            tmp_path = tmp.name; tmp.close()
            pair_top = self._pair_topology_from_biopdb(sub_struct)
            traj = md.load(traj_path, top=tmp_path)
            traj.topology = pair_top
            self._cached_traj_view = TrajectoryView(traj, self.chain_map)

        except Exception:
            try: os.unlink(tmp.name)
            except Exception: pass
            raise

# -------------------------------------------------------------------
# TCR root object (owns original/imgt structures and per-pair views)
# -------------------------------------------------------------------
@dataclass
class TCR:
    input_pdb: str
    traj_path: Optional[str] = None
    contact_cutoff: float = 5.0
    min_contacts: int = 50
    legacy_anarci: bool = True

    original_structure: BPStructure.Structure = field(init=False)
    imgt_all_structure: BPStructure.Structure = field(init=False)
    germline_info: Dict[str, str] = field(init=False, default_factory=dict)
    pairs: List[TCRPairView] = field(init=False, default_factory=list)

    # mdtraj state
    _traj: Optional[md.Trajectory] = field(init=False, default=None, repr=False)

    def _mdtraj_chain_id(self,chain) -> str:
                cid = getattr(chain, "chain_id", None) or getattr(chain, "id", None)
                return (str(cid).strip() if cid else str(chain.index))
    def _pair_topology_from_biopdb(self,pair_full_structure) -> md.Topology:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            io.write_pdb(tmp.name, pair_full_structure)      # writes renamed A/B (or G/D), IMGT-numbered
            top = md.load_frame(tmp.name, index=0).topology  # read topology only
            return top
        finally:
            try: os.unlink(tmp.name)
            except Exception: pass

    def __post_init__(self):
        parser = PDBParser(QUIET=True)

        # 1) load original
        self.original_structure = parser.get_structure("orig", self.input_pdb)[0]
        tmp_fix = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        fix_duplicate_resseqs_by_insertion_code(self.input_pdb, tmp_fix.name)
        #write to temp pdb
        self.input_pdb=tmp_fix.name
        # 2) pair & get numbering maps
        pairs, per_chain_map, germline_info = pair_tcrs_by_interface(
            self.input_pdb,
            contact_cutoff=self.contact_cutoff,
            min_contacts=self.min_contacts,
            legacy_anarci=self.legacy_anarci,
        )

        # 3) IMGT-renumber full structure (no chain renaming here)
        self.imgt_all_structure = parser.get_structure("imgt_all", self.input_pdb)
        apply_imgt_renumbering(self.imgt_all_structure, per_chain_map)
        print("IMGT renumbering applied.")

        # 4) build per-pair views (renamed A/B or G/D in each view only)
        self.pairs = []
        for idx, pair in enumerate(pairs, start=1):
            s = parser.get_structure(f"pair{idx}", self.input_pdb)
            apply_imgt_renumbering(s, per_chain_map)

            aid, bid = pair["alpha_chain"], pair["beta_chain"]
            cta, ctb = pair["alpha_chain_type"], pair["beta_chain_type"]  # "A"/"G" and "B"/"D"

            # Choose standardized view labels
            if {cta, ctb} == {"A", "B"}:
                new_a, new_b = "A", "B"
            elif {cta, ctb} == {"G", "D"}:
                new_a, new_b = "G", "D"
            else:
                # Fallback to A/B
                new_a, new_b = "A", "B"

            # rename chains for this view only

            chain_map = {"alpha": new_a, "beta": new_b}

            # keep just those two chains in the view
            def _only_pair(cid: str, _res: BPResidue, _atom_name: Optional[str] = None) -> bool:
                return cid in (aid, bid)

            s_view = ops.copy_subset(s, _only_pair)
            rename_two_chains_exact_inplace(s_view, aid, bid, new_a, new_b)
            #write to temp pdb
            tmp_view_pair = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
            io.write_pdb(tmp_view_pair.name, s_view)

            # germline text (if available)
            if germline_info:
                alpha_germline = germline_info.get(aid)
                beta_germline = germline_info.get(bid)
            else:
                alpha_germline = None
                beta_germline = None


            if self.traj_path:
                self._traj = md.load(self.traj_path, top=self.input_pdb)
                top = self._traj.topology
                keep = {aid, bid}  # original PDB chain IDs present in the whole-traj topology
                atom_idx = [a.index for a in top.atoms if self._mdtraj_chain_id(a.residue.chain) in keep]
                self._traj.atom_slice(atom_idx, inplace=False)
                subtraj = self._traj.atom_slice(atom_idx, inplace=False)
                pair_top = self._pair_topology_from_biopdb(s_view)
                print(f"Atom count after restriction: coords={subtraj.n_atoms} vs topo={pair_top.n_atoms}.")
                # 3) sanity: atom counts must match; if not, restrict the Bio.PDB view to the same atom set
                if subtraj.n_atoms != pair_top.n_atoms:
                    # Build a restricted Bio.PDB structure that includes only the atom names present in the subtraj
                    # (e.g., backbone-only XTC). This preserves ordering.
                    atom_names_in_sub = {a.name.strip() for a in subtraj.topology.atoms}
                    def _only_pair_and_atoms(cid: str, res: BPResidue, atom_name: Optional[str] = None) -> bool:
                        if cid not in (new_a, new_b):  # A/B or G/D in pair.full_structure
                            return False
                        return True if atom_name is None else (atom_name.strip() in atom_names_in_sub)
                    restricted = ops.copy_subset(s_view, _only_pair_and_atoms)
                    pair_top = self._pair_topology_from_biopdb(restricted)
                    # recheck

                    if subtraj.n_atoms != pair_top.n_atoms:
                        raise RuntimeError(
                            f"Atom count mismatch after restriction: coords={subtraj.n_atoms} vs topo={pair_top.n_atoms}. "
                            "Ensure your whole-traj and pair.full_structure contain the same atom subset and order."
                        )

                # 4) rebind coordinates to the pair topology (keeps times/cells)
                new_traj = md.Trajectory(
                    xyz=subtraj.xyz.copy(),
                    topology=pair_top,
                    time=getattr(subtraj, "time", None)
                )

            self.pairs.append(
                TCRPairView(
                    index=idx,
                    alpha_chain_id=aid,  # ORIGINAL ids (for trajectory mapping)
                    beta_chain_id=bid,
                    alpha_type=new_a,
                    beta_type=new_b,
                    full_structure=s_view,
                    chain_map=chain_map,
                    alpha_germline=alpha_germline,
                    beta_germline=beta_germline,
                    tcr_owner=self,
                    _cached_traj_view=TrajectoryView(new_traj, {"alpha": new_a, "beta":  new_b }) if self._traj else None,
                )
            )