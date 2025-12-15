from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set

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
from Bio.SeqUtils import seq1

# --- project imports (pure utilities & constants) ---
from TCR_TOOLS.core import ops, io
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.core.select import (
    make_region_atom_predicate,
    region_list_for_names,
    make_idx_atom_predicate,
)
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE

# --- numbering/pairing/alignment ---
from TCR_TOOLS.numbering.main import (
    fix_duplicate_resseqs_by_insertion_code,
    apply_imgt_renumbering,
)
from TCR_TOOLS.numbering.tcr_pairing import pair_tcrs_by_interface

# --- geometry (old + new) ---
from TCR_TOOLS.geometry.calc_geometry_MD import run as calc_traj_angles
from TCR_TOOLS.geometry.calc_geometry import run as calc_angles_tcr_only_old
from TCR_TOOLS.geometry.change_geometry2 import run as change_angles_tcr_only_old

#from TCR_TOOLS.geometry.calc_geometry_new import process as calc_angles_tcr_only
#from TCR_TOOLS.geometry.change_geometry_new import apply_geometry_to_tcr as change_angles_tcr_only


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
        Parameters
        ----------
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
        pass_names: bool = False,
    ) -> List[int]:
        # Build interval map per chain in traj
        intervals_by_chain: Dict[str, List[Tuple[int, int]]] = {}
        for name in region_names:
            if name in ("A_variable", "B_variable"):
                start, end = variable_range
                cid = (
                    self._chain_map["alpha"]
                    if name.startswith("A_")
                    else self._chain_map["beta"]
                )
            else:
                if name not in cdr_fr_ranges:
                    raise KeyError(f"Unknown region: {name}")
                start, end = cdr_fr_ranges[name]
                cid = (
                    self._chain_map["alpha"]
                    if name.startswith("A_")
                    else self._chain_map["beta"]
                )
            intervals_by_chain.setdefault(cid, []).append((start, end))

        atom_idxs: List[int] = []
        atom_list = []

        for atom in self._top.atoms:
            res = atom.residue
            if res is None or res.chain is None:
                continue
            chain_obj = res.chain
            # mdtraj Topology has .chain_id for PDB; fall back to index if absent
            chain_id = getattr(chain_obj, "chain_id", None) or str(chain_obj.index)

            if chain_id not in intervals_by_chain:
                continue

            # IMGT numbering expected if serialized from IMGT PDB
            resnum = int(getattr(res, "resSeq", res.index))

            for (s, e) in intervals_by_chain[chain_id]:
                if s <= resnum <= e:
                    if atom_names is None or atom.name in atom_names:
                        atom_idxs.append(atom.index)
                        atom_list.append((chain_id, resnum, atom.name))
                    break

        if not atom_idxs:
            raise ValueError(
                f"No atoms matched regions={region_names} (atom_names={atom_names})."
            )

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
        pass_names: bool = False,
    ) -> List[int]:
        if cdr_fr_ranges is None:
            cdr_fr_ranges = CDR_FR_RANGES
        if variable_range is None:
            variable_range = VARIABLE_RANGE
        return self._region_atom_indices(
            region_names,
            cdr_fr_ranges,
            variable_range,
            atom_names,
            pass_names=pass_names,
        )

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

    def align_to_ref(
        self,
        pv_ref: "TCRPairView",
        region_names: List[str],
        atom_names: Optional[Set[str]] = None,
        inplace: bool = False,
    ):
        ref = io.mdtraj_from_biopython_path(pv_ref.full_structure)
        ref_idx, refnames = pv_ref.domain_idx(
            region_names=region_names, atom_names=atom_names, pass_names=True
        )  # Bio.PDB order
        traj_idx, trajnames = self.domain_idx(
            region_names=region_names, atom_names=atom_names, pass_names=True
        )  # MDTraj order

        if len(ref_idx) != len(traj_idx):
            shared = [x for x in refnames if x in trajnames]

            # Filter ref to keep only shared elements
            mask_a = [x in shared for x in refnames]
            ref_idx = [x for x, keep in zip(ref_idx, mask_a) if keep]

            # Filter traj to keep only shared elements
            mask_b = [x in shared for x in trajnames]
            traj_idx = [x for x, keep in zip(traj_idx, mask_b) if keep]

            if len(ref_idx) != len(traj_idx):
                raise ValueError(
                    f"Index size mismatch: ref={len(ref_idx)} vs traj={len(traj_idx)}"
                )

        work_traj = self._traj if inplace else self._traj[:]

        # Align in-place; get per-frame RMSD (nm)
        work_traj.superpose(
            ref,
            atom_indices=np.asarray(traj_idx, dtype=int),
            ref_atom_indices=np.asarray(ref_idx, dtype=int),
            parallel=True,
        )
        rmsds_nm = md.rmsd(
            work_traj,
            ref,
            atom_indices=np.asarray(traj_idx, dtype=int),
            ref_atom_indices=np.asarray(ref_idx, dtype=int),
            parallel=True,
        )
        rmsds_A = rmsds_nm * 10.0  # convert nm -> Å

        if inplace:
            self._traj = work_traj
            self._top = work_traj.topology
            return self, rmsds_A
        else:
            work_traj.topology = self._top
            new_view = self.__class__(work_traj, self._chain_map)
            new_view._traj.topology = work_traj.topology
            new_view._top = work_traj.topology
            return new_view, rmsds_A


# -------------------------------------------------------------------
# Small helper: rename two chains (pure OR inplace variant)
# -------------------------------------------------------------------
def rename_two_chains_exact_inplace(
    structure: BPStructure,
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
    name:str
    alpha_chain_id: str            # original chain id in source PDB (needed for traj mapping)
    beta_chain_id: str             # original chain id in source PDB
    alpha_type: str                # "A" or "G" in THIS view
    beta_type: str                 # "B" or "D" in THIS view
    full_structure: BPStructure    # IMGT-numbered; chains renamed to A/B or G/D for this view
    chain_map: Dict[str, str]      # {"alpha": "<A|G>", "beta": "<B|D>"}
    alpha_germline: Optional[str] = None
    beta_germline: Optional[str] = None

    # Backref to owning TCR + cached per-pair traj view
    tcr_owner: "TCR" = field(repr=False, default=None)
    _cached_traj_view: Optional[TrajectoryView] = None

    # Cached structures
    _variable_structure: Optional[BPStructure] = field(init=False, default=None, repr=False)
    _linked_structure: Optional[BPStructure] = field(init=False, default=None, repr=False)

    @property
    def variable_structure(self) -> BPStructure:
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

    def _pair_topology_from_biopdb(self, pair_full_structure) -> md.Topology:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            io.write_pdb(tmp.name, pair_full_structure)      # writes renamed A/B (or G/D), IMGT-numbered
            top = md.load_frame(tmp.name, index=0).topology  # read topology only
            return top
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def domain_subset(
        self,
        region_names: Optional[List[str]] = None,
        atom_names: Optional[Set[str]] = None,
        include_het: bool = True,
    ) -> BPStructure:
        """
        Get a NEW structure subset for provided region names (e.g., ["A_CDR1","B_CDR3"]).
        """
        regions = region_list_for_names(region_names, VARIABLE_RANGE)
        pred = make_region_atom_predicate(
            regions=regions,
            chain_map=self.chain_map,
            atom_names=atom_names,
            include_het=include_het,
        )
        return ops.copy_subset(self.full_structure, pred)

    def extract_sequence_from_structure(self, structure: BPStructure) -> str:
        """Return concatenated one-letter AA sequence of the structure (all chains, in order)."""
        seq = []
        for res in structure.get_residues():
            seq.append(seq1(res.resname))
        return "".join(seq)

    def subset_by_idx(
        self,
        A_residue_list,
        B_residue_list,
        atom_names,
        include_het: bool = True,
    ):
        pred = make_idx_atom_predicate(
            A_residue_list=A_residue_list,
            B_residue_list=B_residue_list,
            chain_map=self.chain_map,
            atom_names=atom_names,
            include_het=include_het,
        )
        return ops.copy_subset(self.full_structure, pred)

    def domain_idx(
        self,
        region_names: Optional[List[str]] = None,
        atom_names: Optional[Set[str]] = None,
        include_het: bool = True,
        pass_names: bool = False,
    ) -> List[int]:
        regions = region_list_for_names(region_names, VARIABLE_RANGE) if region_names else None
        pred = make_region_atom_predicate(regions, self.chain_map, atom_names, include_het)

        idxs: List[int] = []
        atom_list = []

        for i, atom in enumerate(self.full_structure.get_atoms()):
            res = atom.get_parent()
            if res is None:
                continue
            chain = res.get_parent()
            if chain is None:
                continue
            if pred(chain.id, res, atom.get_name()):
                idxs.append(i)
                atom_list.append((chain.id, res.get_id()[1], atom.get_name()))

        if not idxs:
            raise ValueError(
                f"No atoms matched regions={region_names} (atom_names={atom_names})."
            )

        if pass_names:
            return idxs, atom_list
        else:
            return idxs

    def linked_structure(self, linker_sequence: str) -> BPStructure:
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
            linked_structure = linked_structure[0]  # get first model

        self._linked_structure = linked_structure
        self._linked_structure.linker_sequence = linker_sequence

        full_linked_sequence = self.extract_sequence_from_structure(self._linked_structure)
        self._linked_structure.full_linked_sequence = full_linked_sequence
        return self._linked_structure

    def linked_resmask(self, regions):
        if self._linked_structure is None:
            raise ValueError("Linked structure not yet built; call linked_structure() first.")

        non_linked_imgt_mask_dict = self.cdr_fr_resmask(
            region_names=regions,
            structure_used=self.variable_structure,
            atom_names={"CA"},
            include_het=True,
        )
        linker_Nones = [False] * len(self._linked_structure.linker_sequence)
        try:
            full_linked_mask = (
                non_linked_imgt_mask_dict["A"] + linker_Nones + non_linked_imgt_mask_dict["B"]
            )
        except KeyError:
            full_linked_mask = (
                non_linked_imgt_mask_dict["G"] + linker_Nones + non_linked_imgt_mask_dict["D"]
            )

        assert len(full_linked_mask) == len(self._linked_structure.full_linked_sequence)
        return full_linked_mask

    def cdr_fr_sequences(self) -> Dict[str, str]:
        """
        Return sequences for all CDR/FR regions from this pair view.
        """
        seqs: Dict[str, str] = {}
        for rname, (s, e) in CDR_FR_RANGES.items():
            regions = [(rname, (s, e))]
            pred = make_region_atom_predicate(
                regions, self.chain_map, atom_names=None, include_het=True
            )
            subset = ops.copy_subset(self.full_structure, pred)
            d = ops.get_sequence_dict(subset)  # {chain_id: seq}
            cid = self.chain_map["alpha"] if rname.startswith("A_") else self.chain_map["beta"]
            seqs[rname] = d.get(cid, "")
        # Note: d from last subset is unused but kept to match your original code
        return seqs

    def cdr_fr_resmask(
        self,
        structure_used,
        region_names: Optional[List[str]] = None,
        atom_names: Optional[Set[str]] = "CA",
        include_het: bool = True,
    ):
        """
        Return boolean masks for all CDR/FR regions from this pair view.
        """
        regions = region_list_for_names(region_names)
        pred = make_region_atom_predicate(regions, self.chain_map, atom_names, include_het)

        mask_dict = {}
        # get chain ids
        for chain in structure_used.get_chains():
            mask_dict[chain.id] = []

        for atom in structure_used.get_atoms():
            res = atom.get_parent()
            if res is None:
                continue
            chain = res.get_parent()
            if chain is None:
                continue
            if atom.get_name() not in atom_names:
                continue
            if pred(chain.id, res, atom.get_name()):
                mask_dict[chain.id].append(True)
            else:
                mask_dict[chain.id].append(False)
        return mask_dict

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
        # write as tmp pdb and tmp xtc
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp_xtc = tempfile.NamedTemporaryFile(suffix=".xtc", delete=False)

        tmptop = self.traj._traj[0]
        tmptop.save_pdb(tmp_pdb.name)
        self.traj._traj.save_xtc(tmp_xtc.name)

        angle_results = calc_traj_angles(tmp_xtc.name, tmp_pdb.name)
        return angle_results

    def calc_angles_tcr(self, out_path, vis: bool = True):
        write_pdb(os.path.join(out_path, "input_tcr.pdb"), self.full_structure)
        angle_results = calc_angles_tcr_only_old(
            os.path.join(out_path, "input_tcr.pdb"),
            out_path=out_path,
            vis=vis,
            cleanup_tmp=True,
            alpha_chain_id=self.chain_map["alpha"],
            beta_chain_id=self.chain_map["beta"],
        )
        return angle_results

    def change_angles_tcr(
        self,
        out_path,
        angles,
        vis: bool = True):

        BA = angles["BA"]
        BC1 = angles["BC1"]
        BC2 = angles["BC2"]
        AC1 = angles["AC1"]
        AC2 = angles["AC2"]
        dc = angles["dc"]

        write_pdb(os.path.join(out_path, "input_tcr.pdb"), self.full_structure)
        alpha_chain_id = self.chain_map["alpha"]
        beta_chain_id = self.chain_map["beta"]

        final_aligned_pdb, final_aligned_structure = change_angles_tcr_only_old(
            os.path.join(out_path, "input_tcr.pdb"),
            BA,
            BC1,
            BC2,
            AC1,
            AC2,
            dc,
            out_path=out_path,
            vis=vis,
            alpha_chain_id=alpha_chain_id,
            beta_chain_id=beta_chain_id,
        )
        return final_aligned_pdb, final_aligned_structure

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
            tmp_path = tmp.name
            tmp.close()

            pair_top = self._pair_topology_from_biopdb(sub_struct)
            traj = md.load(traj_path, top=tmp_path)
            traj.topology = pair_top

            self._cached_traj_view = TrajectoryView(traj, self.chain_map)
        except Exception:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise


# -------------------------------------------------------------------
# TCR root object (owns original/imgt structures and per-pair views)
# -------------------------------------------------------------------
@dataclass
class TCR:
    input_pdb: str
    name: Optional[str] = None
    traj_path: Optional[str] = None
    contact_cutoff: float = 5.0
    min_contacts: int = 50
    legacy_anarci: bool = True
    scheme: str = "imgt"

    original_structure: BPStructure = field(init=False)
    imgt_all_structure: BPStructure = field(init=False)
    germline_info: Dict[str, str] = field(init=False, default_factory=dict)
    pairs: List[TCRPairView] = field(init=False, default_factory=list)
    manual_chain_types: Dict[str, str] = field(default_factory=dict)  # optional manual chain type overrides

    # mdtraj state
    _traj: Optional[md.Trajectory] = field(init=False, default=None, repr=False)

    def _mdtraj_chain_id(self, chain) -> str:
        cid = getattr(chain, "chain_id", None) or getattr(chain, "id", None)
        return str(cid).strip() if cid else str(chain.index)

    def _pair_topology_from_biopdb(self, pair_full_structure) -> md.Topology:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        try:
            io.write_pdb(tmp.name, pair_full_structure)      # writes renamed A/B (or G/D), IMGT-numbered
            top = md.load_frame(tmp.name, index=0).topology  # read topology only
            return top
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def __post_init__(self):
        parser = PDBParser(QUIET=True)
        if not self.name:
            self.name = os.path.basename(self.input_pdb).split(".")[0]

        # 1) load original
        self.original_structure = parser.get_structure("orig", self.input_pdb)[0]

        tmp_fix = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        fix_duplicate_resseqs_by_insertion_code(self.input_pdb, tmp_fix.name)
        self.input_pdb = tmp_fix.name

        # 2) pair & get numbering maps
        pairs, per_chain_map, germline_info, chain_types_dict = pair_tcrs_by_interface(
            self.input_pdb,
            contact_cutoff=self.contact_cutoff,
            min_contacts=self.min_contacts,
            legacy_anarci=self.legacy_anarci,
            manual_chain_types=self.manual_chain_types,
            scheme=self.scheme,
        )
        self.per_chain_map = per_chain_map

        if self.manual_chain_types:
            self.chain_types_dict = self.manual_chain_types
        else:
            self.chain_types_dict = chain_types_dict

        # attempt to recover original CDR/FR ranges (may fail for some inputs)
        try:
            chain_map_A = self.per_chain_map[self.chain_types_dict.get("A", "G")]
            chain_map_B = self.per_chain_map[self.chain_types_dict.get("B", "D")]

            inv_map_A = {newval: (k, res) for k, (newval, res) in chain_map_A.items()}
            inv_map_B = {newval: (k, res) for k, (newval, res) in chain_map_B.items()}

            old_CDR_FR_RANGES = {}
            for name, rng in CDR_FR_RANGES.items():
                print(f"{name}: {rng}")
                if "A" in name:
                    use_map = inv_map_A
                if "B" in name:
                    use_map = inv_map_B

                start, end = rng
                old_start = use_map[(start, " ")]
                old_end = use_map[(end, " ")]
                old_CDR_FR_RANGES[name] = (old_start, old_end)
            self.original_CDR_FR_RANGES = old_CDR_FR_RANGES
        except Exception:
            self.original_CDR_FR_RANGES = None

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
            def _only_pair(
                cid: str,
                _res: BPResidue,
                _atom_name: Optional[str] = None,
            ) -> bool:
                return cid in (aid, bid)

            s_view = ops.copy_subset(s, _only_pair)
            rename_two_chains_exact_inplace(s_view, aid, bid, new_a, new_b)

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
                atom_idx = [
                    a.index
                    for a in top.atoms
                    if self._mdtraj_chain_id(a.residue.chain) in keep
                ]
                self._traj.atom_slice(atom_idx, inplace=False)
                subtraj = self._traj.atom_slice(atom_idx, inplace=False)

                pair_top = self._pair_topology_from_biopdb(s_view)
                print(
                    f"Atom count after restriction: coords={subtraj.n_atoms} "
                    f"vs topo={pair_top.n_atoms}."
                )

                # sanity: atom counts must match; if not, restrict the Bio.PDB view
                if subtraj.n_atoms != pair_top.n_atoms:
                    atom_names_in_sub = {a.name.strip() for a in subtraj.topology.atoms}

                    def _only_pair_and_atoms(
                        cid: str,
                        res: BPResidue,
                        atom_name: Optional[str] = None,
                    ) -> bool:
                        if cid not in (new_a, new_b):  # A/B or G/D in pair.full_structure
                            return False
                        return True if atom_name is None else (
                            atom_name.strip() in atom_names_in_sub
                        )

                    restricted = ops.copy_subset(s_view, _only_pair_and_atoms)
                    pair_top = self._pair_topology_from_biopdb(restricted)

                    if subtraj.n_atoms != pair_top.n_atoms:
                        raise RuntimeError(
                            "Atom count mismatch after restriction: "
                            f"coords={subtraj.n_atoms} vs topo={pair_top.n_atoms}. "
                            "Ensure your whole-traj and pair.full_structure contain "
                            "the same atom subset and order."
                        )

                new_traj = md.Trajectory(
                    xyz=subtraj.xyz.copy(),
                    topology=pair_top,
                    time=getattr(subtraj, "time", None),
                )
            else:
                new_traj = None

            self.pairs.append(
                TCRPairView(
                    index=idx,
                    name=self.name,
                    alpha_chain_id=aid,  # ORIGINAL ids (for trajectory mapping)
                    beta_chain_id=bid,
                    alpha_type=new_a,
                    beta_type=new_b,
                    full_structure=s_view,
                    chain_map=chain_map,
                    alpha_germline=alpha_germline,
                    beta_germline=beta_germline,
                    tcr_owner=self,
                    _cached_traj_view=TrajectoryView(new_traj, {"alpha": new_a, "beta": new_b})
                    if self._traj
                    else None,
                )
            )
