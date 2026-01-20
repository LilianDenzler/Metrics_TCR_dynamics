# tcr.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set, Callable
from io import StringIO
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
from typing import Optional, Tuple, Union, Dict
from Bio.SeqUtils import seq1

from Bio.SeqUtils import seq1
# --- project imports (pure utilities & constants) ---
from TCR_TOOLS.core import ops, io
from TCR_TOOLS.core.io import write_pdb
import tempfile, os
from TCR_TOOLS.classes.tcr import TCRPairView, TrajectoryView
from TCR_TOOLS.core.select import make_region_atom_predicate, region_list_for_names
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE

# --- numbering/pairing/alignment (your existing modules) ---
from TCR_TOOLS.numbering.main import fix_duplicate_resseqs_by_insertion_code,apply_imgt_renumbering
from TCR_TOOLS.numbering.tcr_pairing import pair_tcrs_by_interface

from TCR_TOOLS.geometry.TCR_PMHC_geo_new import run as calc_complex_angles_single_pdb
from TCR_TOOLS.geometry.TCR_PMHC_change_geo import  run as change_complex_angles_single_pdb
from Bio.PDB import PDBIO
from Bio.PDB import Superimposer
import re
from copy import deepcopy
from TCR_TOOLS.geometry.__init__ import DATA_PATH
from TCR_TOOLS.__init__ import CDR_FR_RANGES,VARIABLE_RANGE, A_FR_ALIGN, B_FR_ALIGN,IMGT_DB_PATH
from TCR_TOOLS.numbering.get_HLA_from_seq import blastp_imgt_hla

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
# TCR root object (owns original/imgt structures and per-pair views)
# -------------------------------------------------------------------
@dataclass
class TCRpMHC:
    input_pdb: str
    name: Optional[str] = None
    #if i already know exactely what the chanid of the mhc chain,tcr chains etc,  is in my input pdb can put it here, to override any automatic assignment
    MHC_a_chain_id: Optional[str] = None
    MHC_b_chain_id: Optional[str] = None
    Peptide_chain_id: Optional[str] = None
    TCR_a_chain_id: Optional[str] = None
    TCR_b_chain_id: Optional[str] = None

    contact_cutoff: float = 5.0
    min_contacts: int = 50
    legacy_anarci: bool = True
    traj_path: Optional[str] = None
    scheme: str = "imgt"

    original_structure: BPStructure.Structure = field(init=False)
    imgt_all_structure: BPStructure.Structure = field(init=False)
    variable_structure: BPStructure.Structure = field(init=False, default=None)
    _traj: Optional[md.Trajectory] = field(init=False, default=None, repr=False)
    mhc_alpha_allele: Optional[dict] = field(init=False, default=None)
    mhc_beta_allele: Optional[dict] = field(init=False, default=None)


    @staticmethod
    def _extract_chain_sequence_from_structure(structure, chain_id: str) -> str:
        model0 = next(structure.get_models())
        if chain_id not in model0:
            raise KeyError(f"Chain {chain_id!r} not found in structure.")
        chain = model0[chain_id]

        aa = []
        for res in chain:
            hetflag, resseq, icode = res.id
            if hetflag != " ":
                continue
            try:
                aa.append(seq1(res.get_resname(), custom_map={"UNK": "X"}))
            except Exception:
                continue
        return "".join(aa)

    @staticmethod
    def _count_polymer_residues(structure, chain_id: str) -> int:
        model0 = next(structure.get_models())
        chain = model0[chain_id]
        n = 0
        for res in chain:
            hetflag, _, _ = res.id
            if hetflag == " ":
                n += 1
        return n

    def _chain_exists(self, structure, chain_id: Optional[str]) -> bool:
        if not chain_id:
            return False

        model0 = next(structure.get_models())
        if chain_id not in model0:
            return False

        chain = model0[chain_id]

        # Must have at least one polymer residue that looks like an amino acid.
        for res in chain:
            hetflag, _, _ = res.id
            if hetflag != " ":
                continue
            try:
                aa = seq1(res.get_resname(), custom_map={"UNK": "X"})
            except Exception:
                continue
            if aa and aa != "?":
                return True

        return False


    def _mhc_role_from_hit(self, hit: dict) -> Optional[str]:
        allele_title = (hit or {}).get("stitle")
        gene=allele_title.split("*")[0]
        if not gene:
            return None

        # Class I heavy chain genes (HLA-A/B/C + non-classical)
        mhc1_genes = {"A", "B", "C", "E", "F", "G"}

        # Class II (alpha vs beta)
        mhc2_alpha_genes = {"DRA", "DPA1", "DQA1", "DQA2", "DMA", "DOA"}
        mhc2_beta_genes  = {"DRB1", "DRB3", "DRB4", "DRB5", "DPB1", "DQB1", "DQB2", "DMB", "DOB"}

        if gene in mhc1_genes:
            return "mhc1_heavy"
        if gene in mhc2_alpha_genes:
            return "mhc2_alpha"
        if gene in mhc2_beta_genes:
            return "mhc2_beta"
        return None

    def _infer_pmhc_chain_ids_if_needed(self) -> None:
        """
        Infer MHC alpha/beta and peptide chain IDs when not provided.

        Uses the *gene* parsed from the top IPD-IMGT/HLA BLAST hit to decide:
        - MHC-I heavy: HLA-A/B/C/E/F/G  -> MHC_a_chain_id
        - MHC-II alpha: DRA/DPA1/DQA1   -> MHC_a_chain_id
        - MHC-II beta:  DRB1/DPB1/DQB1  -> MHC_b_chain_id
        Peptide = remaining short chain, validated <= 12 polymer residues.
        """

        model0 = next(self.imgt_all_structure.get_models())

        # TCR chains must already be known (original PDB IDs)
        tcr_alpha = self.chain_types_dict.get("A")
        tcr_beta  = self.chain_types_dict.get("B")
        if tcr_alpha is None or tcr_beta is None:
            raise ValueError("TCR chains must be identified before pMHC inference.")

        non_tcr_chain_ids = [c.id for c in model0 if c.id not in {tcr_alpha, tcr_beta}]
        if not non_tcr_chain_ids:
            raise ValueError("No non-TCR chains left to infer pMHC from.")

        seq_by_chain = {cid: self._extract_chain_sequence_from_structure(self.imgt_all_structure, cid)
                        for cid in non_tcr_chain_ids}
        len_by_chain = {cid: len(seq_by_chain[cid]) for cid in non_tcr_chain_ids}

        # BLAST only chains plausibly MHC-like
        mhc_candidates = [cid for cid in non_tcr_chain_ids if len_by_chain[cid] >= 50]

        hits = []
        for cid in mhc_candidates:
            hit = blastp_imgt_hla(seq_by_chain[cid], imgt_db_path=IMGT_DB_PATH)
            if hit:
                role = self._mhc_role_from_hit(hit)
                if role:
                    hits.append((cid, role, hit))

        # Partition hits by role
        mhc1_heavy = [(cid, hit) for (cid, role, hit) in hits if role == "mhc1_heavy"]
        mhc2_alpha = [(cid, hit) for (cid, role, hit) in hits if role == "mhc2_alpha"]
        mhc2_beta  = [(cid, hit) for (cid, role, hit) in hits if role == "mhc2_beta"]

        def _best(hit_list):
            # choose best by bitscore then identity then aln_len
            return sorted(
                hit_list,
                key=lambda x: (x[1].get("bitscore", -1.0), x[1].get("identity", -1.0), x[1].get("aln_len", -1)),
                reverse=True,
            )[0] if hit_list else None

        # Decide whether this looks like class II (both alpha+beta) or class I (heavy only)
        class2_alpha_best = _best(mhc2_alpha)
        class2_beta_best  = _best(mhc2_beta)
        class1_best       = _best(mhc1_heavy)

        used = set()

        # --- Assign MHC chains by allele gene ---
        if (class2_alpha_best is not None) or (class2_beta_best is not None):
            # Treat as class II if either chain indicates class II; require both if possible.
            if self.MHC_a_chain_id is None:
                if class2_alpha_best is None:
                    raise ValueError("Found MHC-II-like hits but none mapped to an alpha-chain gene (e.g., DRA/DPA1/DQA1).")
                self.MHC_a_chain_id = class2_alpha_best[0]
            if self.MHC_b_chain_id is None:
                if class2_beta_best is None:
                    raise ValueError("Found MHC-II-like hits but none mapped to a beta-chain gene (e.g., DRB1/DPB1/DQB1).")
                self.MHC_b_chain_id = class2_beta_best[0]

            used.update([self.MHC_a_chain_id, self.MHC_b_chain_id])

        else:
            # Treat as class I: heavy chain allele indicates HLA-A/B/C/...
            if self.MHC_a_chain_id is None:
                if class1_best is None:
                    debug = ", ".join([f"{cid}(len={len_by_chain[cid]})" for cid in non_tcr_chain_ids])
                    raise ValueError(
                        "Could not infer MHC-I heavy chain: no class-I gene hits (A/B/C/E/F/G).\n"
                        f"Non-TCR chains: {debug}"
                    )
                self.MHC_a_chain_id = class1_best[0]
            used.add(self.MHC_a_chain_id)

            # For class I, MHC_b is typically beta2m (not in IMGT/HLA). Use a structural heuristic:
            # pick the longest remaining non-TCR chain that is not the heavy chain, but do NOT steal the peptide.
            if self.MHC_b_chain_id is None:
                remaining = [cid for cid in non_tcr_chain_ids if cid not in used]
                # peptide candidates are very short; exclude them first
                non_pep_like = [cid for cid in remaining if self._count_polymer_residues(self.imgt_all_structure, cid) > 12]
                if non_pep_like:
                    # beta2m typically ~90–120 aa; choose best by closeness to 100 then length
                    self.MHC_b_chain_id = sorted(
                        non_pep_like,
                        key=lambda cid: (abs(len_by_chain[cid] - 100), -len_by_chain[cid]),
                    )[0]
                    used.add(self.MHC_b_chain_id)
                else:
                    self.MHC_b_chain_id = None  # allowed

        # --- Infer peptide from leftover chains (must be <=12 polymer residues) ---
        if self.Peptide_chain_id is None:
            leftovers = [cid for cid in non_tcr_chain_ids if cid not in used]
            if not leftovers:
                raise ValueError("No leftover chain available to assign as peptide after MHC assignment.")

            # choose shortest by polymer residues
            leftovers = sorted(leftovers, key=lambda cid: self._count_polymer_residues(self.imgt_all_structure, cid))
            pep_cid = leftovers[0]
            pep_len = self._count_polymer_residues(self.imgt_all_structure, pep_cid)

            if pep_len > 12:
                detail = ", ".join([f"{cid}(poly={self._count_polymer_residues(self.imgt_all_structure, cid)},len={len_by_chain[cid]})"
                                    for cid in leftovers])
                raise ValueError(
                    f"Peptide inference failed: best leftover chain {pep_cid!r} has {pep_len} polymer residues (>12).\n"
                    f"Leftovers: {detail}"
                )

            self.Peptide_chain_id = pep_cid

    def _get_MHC_alleles(self):
        # alpha (heavy or MHC-II alpha)
        if self._chain_exists(self.imgt_all_structure, self.newMHC_a_chain_id):
            alpha_seq = self._extract_chain_sequence_from_structure(self.imgt_all_structure, self.newMHC_a_chain_id)
            self.mhc_alpha_allele = blastp_imgt_hla(alpha_seq, imgt_db_path=IMGT_DB_PATH)
        else:
            self.mhc_alpha_allele = None

        # beta (MHC-II beta or beta2m heuristic) — may be missing / may not BLAST-hit
        if self._chain_exists(self.imgt_all_structure, self.newMHC_b_chain_id):
            beta_seq = self._extract_chain_sequence_from_structure(self.imgt_all_structure, self.newMHC_b_chain_id)
            self.mhc_beta_allele = blastp_imgt_hla(beta_seq, imgt_db_path=IMGT_DB_PATH)
        else:
            self.mhc_beta_allele = None


    def rename_pMHC_chains(self):

        old_alpha_chainid=self.chain_types_dict.get("A","G")
        old_beta_chainid=self.chain_types_dict.get("B","D")
        """Rename MHC and peptide chains to standard IDs."""
        chain_id_map = {
            old_alpha_chainid: "A",
            old_beta_chainid:  "B",
        }

        # Only include pMHC chains if they were provided/inferred
        if self.MHC_a_chain_id is not None:
            chain_id_map[self.MHC_a_chain_id] = "M"
        if self.MHC_b_chain_id is not None:
            chain_id_map[self.MHC_b_chain_id] = "N"
        if self.Peptide_chain_id is not None:
            chain_id_map[self.Peptide_chain_id] = "P"
        first_pass= {}
        second_pass={}
        new_chain_id_map = {}
        for key, value in chain_id_map.items():
            if key in list(chain_id_map.values()):
                #remove from chain_id_map and add to second pass
                first_pass[key]=value
                #second_pass[chain_id_map.get(value)]=key
                #remove from original map
            elif value in list(chain_id_map.keys()):
                second_pass[key]=value
            else:
                new_chain_id_map[key]=value

        structure = self.imgt_all_structure
        for model in structure:
            for chain in model:
                if chain.id in new_chain_id_map.keys():
                    print(f"Renaming chain {chain.id} to {new_chain_id_map[chain.id]}")
                    chain.id = new_chain_id_map[chain.id]

        for model in structure:
            for chain in model:
                if chain.id in first_pass.keys():
                    print(f"Renaming chain {chain.id} to {first_pass[chain.id]}")
                    chain.id = first_pass[chain.id]
        for model in structure:
            for chain in model:
                if chain.id in second_pass.keys():
                    print(f"Renaming chain {chain.id} to {second_pass[chain.id]}")
                    chain.id = second_pass[chain.id]

        self.imgt_all_structure=structure
        self.newMHC_a_chain_id="M"
        self.newMHC_b_chain_id = "N" if self.MHC_b_chain_id is not None else None
        self.newPeptide_chain_id="P"
        self.newTCR_a_chain_id="A"
        self.newTCR_b_chain_id="B"

    def __post_init__(self):
        if self.name is None:
            self.name = os.path.basename(self.input_pdb).replace(".pdb", "")
        parser = PDBParser(QUIET=True)

        # 1) load original
        self.original_structure = parser.get_structure("orig", self.input_pdb)[0]
        tmp_fix = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        fix_duplicate_resseqs_by_insertion_code(self.input_pdb, tmp_fix.name)
        #write to temp pdb
        self.input_pdb=tmp_fix.name
        # 2) pair & get numbering maps
        pairs, per_chain_map, germline_info,chain_types_dict = pair_tcrs_by_interface(
            self.input_pdb,
            contact_cutoff=self.contact_cutoff,
            min_contacts=self.min_contacts,
            legacy_anarci=self.legacy_anarci,
            scheme=self.scheme
        )

        self.per_chain_map = per_chain_map
        self.chain_types_dict=chain_types_dict

        print("IDENTIFIED TCR CHAIN TYPES: ",self.chain_types_dict)
        if self.TCR_a_chain_id is not None and self.TCR_b_chain_id is not None:
            print("TCR CHAIN ID OVERRIDE")
            self.chain_types_dict={}
            self.chain_types_dict["A"]=self.TCR_a_chain_id
            self.chain_types_dict["B"]=self.TCR_b_chain_id

        if "A" not in self.chain_types_dict.keys() and "G" not in self.chain_types_dict.keys() and "L" not in self.chain_types_dict.keys() and "K" not in self.chain_types_dict.keys():
            raise ValueError("Could not identify TCR alpha chain in structure.")
        else:
            alpha_aliases = ("A", "G", "L","K")
            present = [k for k in alpha_aliases if k in self.chain_types_dict]
            old_key = present[0]
            self.chain_types_dict["A"] = self.chain_types_dict.pop(old_key)
        if "B" not in self.chain_types_dict.keys() and "D" not in self.chain_types_dict.keys() and "H" not in self.chain_types_dict.keys():
            raise ValueError("Could not identify TCR beta chain in structure.")
        else:
            beta_aliases = ("B", "D", "H")
            present = [k for k in beta_aliases if k in self.chain_types_dict]
            old_key = present[0]
            self.chain_types_dict["B"] = self.chain_types_dict.pop(old_key)

        try:
            chain_map_A=self.per_chain_map[self.chain_types_dict.get("A","G")]
            chain_map_B=self.per_chain_map[self.chain_types_dict.get("B","D")]

            inv_map_A = {newval: (k,res) for k, (newval, res) in chain_map_A.items()}
            inv_map_B = {newval: (k,res) for k, (newval, res) in chain_map_B.items()}
            old_CDR_FR_RANGES={}
            for name, range in CDR_FR_RANGES.items():
                print(f"{name}: {range}")
                if "A" in name:
                    use_map=inv_map_A
                if "B" in name:
                    use_map=inv_map_B

                start, end = range
                #get key with start somewhere in value
                old_start=use_map[(start, ' ')]
                old_end=use_map[(end, ' ')]
                old_CDR_FR_RANGES[name]=(old_start, old_end)
            self.original_CDR_FR_RANGES=old_CDR_FR_RANGES
        except:
            self.original_CDR_FR_RANGES=None


        # 3) IMGT-renumber full structure (no chain renaming here)
        self.imgt_all_structure = parser.get_structure("imgt_all", self.input_pdb)
        apply_imgt_renumbering(self.imgt_all_structure, per_chain_map)
        if self.MHC_a_chain_id is None or self.Peptide_chain_id is None:
            self._infer_pmhc_chain_ids_if_needed()
        self.rename_pMHC_chains()

        # then allele lookup on standardized chain IDs M/N:
        self._get_MHC_alleles()

        #check TCRpMHC is complete
        for check_chain in [self.newMHC_a_chain_id, self.newPeptide_chain_id, self.newTCR_a_chain_id, self.newTCR_b_chain_id]:
            if not self._chain_exists(self.imgt_all_structure, check_chain):
                print("error", check_chain)
                raise RuntimeError("Not all chains are in the input file!")
            else:
                print("correct", check_chain)


        print("IMGT renumbering applied.")
        self.pairs = []
        if len(pairs)==0 and self.TCR_a_chain_id is not None and self.TCR_b_chain_id is not None:
            print("NO PAIRS FOUND, BUT TCR CHAIN ID OVERRIDE - CREATING SINGLE VIEW")
            pairs=[{"alpha_chain":self.TCR_a_chain_id,
                    "beta_chain":self.TCR_b_chain_id,
                    "alpha_chain_type":"A",
                    "beta_chain_type":"B"}]
        for idx, pair in enumerate(pairs, start=1):
            print(pair)
            print(per_chain_map)
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
                    _cached_traj_view=TrajectoryView(new_traj, {"alpha": new_a, "beta":  new_b }) if self._traj else None,
                )
            )

    def calc_geometry(self, out_path=None):
        print(self.newTCR_a_chain_id)
        print(self.newTCR_b_chain_id)
        df=calc_complex_angles_single_pdb(self,
                                          out_path=out_path,
                                          vis=False,
                                          alpha_chain_id=self.newTCR_a_chain_id,
                                          beta_chain_id=self.newTCR_b_chain_id,
                                          mhc_chain_ids=(self.newMHC_a_chain_id, self.newMHC_b_chain_id),
                                          pep_chain_id=self.newPeptide_chain_id)
        return df

    def variable_tcr_structure(self) -> BPStructure.Structure:
        """
        Lazily compute and cache the variable-domain structure (A/B or G/D variable ranges).
        """
        if self.variable_structure is None:
            regions = [("A_variable", VARIABLE_RANGE), ("B_variable", VARIABLE_RANGE)]
            pred = make_region_atom_predicate(
                regions=regions,
                chain_map={"alpha": "A", "beta": "B"},
            )
            self.variable_structure = ops.copy_subset(self.imgt_all_structure, pred)
        return self.variable_structure

    def change_geometry(self,output_dir,geo_dict, inplace: bool = False) -> str:

        io = PDBIO()
        io.set_structure(self.imgt_all_structure)
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        io.save(tmp_pdb.name)
        input_pdb = tmp_pdb.name
        full_final_pdb=change_complex_angles_single_pdb(
                tmp_pdb.name,
                BA=geo_dict["BA"],
                BC1=geo_dict["BC1"],
                BC2=geo_dict["BC2"],
                AC1=geo_dict["AC1"],
                AC2=geo_dict["AC2"],
                dc=geo_dict["dc"],
                d_alpha_mhc=geo_dict["d_alpha_mhc"],
                d_beta_mhc=geo_dict["d_beta_mhc"],
                d_alpha_beta=geo_dict["d_alpha_beta"],
                theta_rA=geo_dict["theta_rA"],
                phi_rA=geo_dict["phi_rA"],
                theta_rB=geo_dict["theta_rB"],
                phi_rB=geo_dict["phi_rB"],
                theta_pc1A=geo_dict["theta_pc1A"],
                phi_pc1A=geo_dict["phi_pc1A"],
                theta_pc1B=geo_dict["theta_pc1B"],
                phi_pc1B=geo_dict["phi_pc1B"],
                out_dir=output_dir,
                alpha_chain_id=self.newTCR_a_chain_id,
                beta_chain_id=self.newTCR_b_chain_id,
                mhc_chain_ids=(self.newMHC_a_chain_id, self.newMHC_b_chain_id),
                pep_chain_id=self.newPeptide_chain_id,)
        return full_final_pdb




    def subset_structure(
        self,
        chain_to_resids: Dict[str, object],
        structure: Optional[BPStructure.Structure] = None,
        include_hetero: bool = False,
        ) -> BPStructure.Structure:
        '''
        Return a Bio.PDB Structure subset using a dict like:
          {
            "A": [ (23,' '), (24,' '), "25A", 26 ],
            "B": {"range": (1, 110)},
            "M": {"ids": [10,11,12]},
          }

        Uses self.imgt_all_structure by default (already IMGT-renumbered and chain-standardized).
        Residues are matched by (resseq, insertion_code).
        '''
        if structure is None:
            structure = self.imgt_all_structure

        # Normalize to: chain -> set((resseq, icode))
        chain_to_set = {}
        for cid, spec in chain_to_resids.items():
            chain_to_set[cid] = _normalize_residue_spec(spec)

        def _pred(cid: str, res: BPResidue, atom_name: Optional[str] = None) -> bool:
            # keep only specified chains
            if cid not in chain_to_set:
                return False

            # optionally drop hetero residues
            hetflag, resseq, icode = res.id
            if (not include_hetero) and (hetflag != " "):
                return False

            wanted = chain_to_set[cid]
            # If user passed empty set for a chain, interpret as "keep entire chain"
            if len(wanted) == 0:
                return True

            return (int(resseq), (icode if icode else " ")) in wanted

        return ops.copy_subset(structure, _pred)


    def _collect_matched_atoms_for_alignment(
        self,
        other: "TCRpMHC",
        chain_to_resids: Dict[str, object],
        atom_mode: str = "CA",
        structure_self: Optional[BPStructure.Structure] = None,
        structure_other: Optional[BPStructure.Structure] = None):
        '''
        Build matched (fixed_atoms, moving_atoms) lists for superposition.
        Matching key: (chain_id, resseq, icode, atom_name).
        atom_mode: "CA" or "heavy"
        '''
        if structure_self is None:
            structure_self = self.imgt_all_structure
        if structure_other is None:
            structure_other = other.imgt_all_structure

        # Extract the region subsets from both, so both are constrained to the same logical selection
        sub_self = self.subset_structure(chain_to_resids, structure=structure_self)
        sub_other = other.subset_structure(chain_to_resids, structure=structure_other)

        model_s = next(sub_self.get_models())
        model_o = next(sub_other.get_models())

        def iter_atoms(model, atom_mode):
            for chain in model:
                cid = chain.id
                for res in chain:
                    hetflag, resseq, icode = res.id
                    if hetflag != " ":
                        continue
                    rk = (cid, int(resseq), icode if icode else " ")
                    if atom_mode == "CA":
                        if "CA" in res:
                            yield (rk, "CA"), res["CA"]
                    elif atom_mode == "heavy":
                        for a in res.get_atoms():
                            if a.element != "H":
                                yield (rk, a.get_name()), a
                    else:
                        raise ValueError("atom_mode must be 'CA' or 'heavy'")

        atoms_s = dict(iter_atoms(model_s, atom_mode))
        atoms_o = dict(iter_atoms(model_o, atom_mode))

        keys = sorted(set(atoms_s.keys()) & set(atoms_o.keys()))
        if len(keys) < 3:
            raise RuntimeError(
                f"Not enough matched atoms for alignment (matched={len(keys)}). "
                "Check chain IDs and residue IDs."
            )

        fixed_atoms = [atoms_o[k] for k in keys]   # other is reference/fixed
        moving_atoms = [atoms_s[k] for k in keys]  # self will be moved onto other
        return fixed_atoms, moving_atoms


    def fit_to(
        self,
        other: "TCRpMHC",
        chain_to_resids: Dict[str, object],
        atom_mode: str = "CA",
        apply_to_self: bool = False,
    ) -> float:
        '''
        Superimpose *this* object onto `other` using a user-defined selection.

        Parameters
        ----------
        chain_to_resids:
            dict mapping chain -> residue spec (see subset_structure docstring).
            If you pass an empty set/dict/list for a chain, it means "use whole chain".
            Example:
              {"M": {"range": (1, 180)}, "N": {"range": (1, 180)}, "C": [], "A": {"range": (1, 110)}, "B": {"range": (1, 110)}}

        atom_mode:
            "CA" or "heavy"
        apply_to_self:
            If False: compute RMSD but do not modify coordinates.
            If True: apply the fitted transform to *self.imgt_all_structure* in-place.

        Returns
        -------
        RMSD of selected atoms after superposition.
        '''
        fixed_atoms, moving_atoms = self._collect_matched_atoms_for_alignment(
            other=other,
            chain_to_resids=chain_to_resids,
            atom_mode=atom_mode,
        )

        sup = Superimposer()
        sup.set_atoms(fixed_atoms, moving_atoms)

        if apply_to_self:
            # Apply transform to all atoms in the full structure (not just the subset)
            model0 = next(self.imgt_all_structure.get_models())
            all_atoms = [a for a in model0.get_atoms()]
            sup.apply(all_atoms)

        return float(sup.rms)


    def rmsd_to(
        self,
        other: "TCRpMHC",
        atom_mode: str = "CA",
        align_first: bool = True,
        out_self_pdb: Optional[str] = None,
        out_other_pdb: Optional[str] = None,
        apply_alignment_inplace: bool = False,
        # --- TM options ---
        return_tm: bool = False,
        tm_reduce: str = "max",      # for tm_align: "max" | "mean" | "chain1" | "chain2"
        tm_mode: str = "tmalign",    # "tmalign" (tmtools) | "fixed" (your fit)
        # --- NEW: separate selections ---
        align_sel: Optional[Dict[str, object]] = None,
        score_sel: Optional[Dict[str, object]] = None,
    ) -> Union[float, Tuple[float, float]]:
        '''
        Compute RMSD (and optionally TM-score) between self and other.

        You may specify:
          - align_sel: selection used to align mover onto reference
          - score_sel: selection used to compute RMSD/TM after alignment


        '''


        # Helper: build coords + sequence from atoms (for tm_align only)
        def _atoms_to_coords_and_seq(atoms):
            coords = np.asarray([a.get_coord() for a in atoms], dtype=float)
            seq = []
            for a in atoms:
                res = a.get_parent()
                try:
                    seq.append(seq1(res.get_resname(), custom_map={"UNK": "X"}))
                except Exception:
                    seq.append("X")
            return coords, "".join(seq)

        # Choose mover/reference structures
        if align_first:
            if apply_alignment_inplace:
                mover = self
                mover_struct = self.imgt_all_structure
            else:
                mover = deepcopy(self)
                mover.imgt_all_structure = deepcopy(self.imgt_all_structure)
                mover_struct = mover.imgt_all_structure

            ref = other
            ref_struct = other.imgt_all_structure

            # 1) Align using align_sel (apply to mover)
            mover.fit_to(ref, align_sel, atom_mode=atom_mode, apply_to_self=True)

            # 2) Collect atoms for scoring using score_sel AFTER alignment
            fixed_atoms, moving_atoms = mover._collect_matched_atoms_for_alignment(
                other=ref,
                chain_to_resids=score_sel,
                atom_mode=atom_mode,
            )

            # Save aligned structures if requested
            if out_self_pdb is not None:
                write_pdb(out_self_pdb, mover_struct)
            if out_other_pdb is not None:
                write_pdb(out_other_pdb, ref_struct)

        else:
            # No alignment; score on current coords using score_sel
            fixed_atoms, moving_atoms = self._collect_matched_atoms_for_alignment(
                other=other,
                chain_to_resids=score_sel,
                atom_mode=atom_mode,
            )
            if out_self_pdb is not None:
                write_pdb(out_self_pdb, self.imgt_all_structure)
            if out_other_pdb is not None:
                write_pdb(out_other_pdb, other.imgt_all_structure)

        # RMSD on scoring set
        sup = Superimposer()
        sup.set_atoms(fixed_atoms, moving_atoms)
        rmsd = float(sup.rms)

        if not return_tm:
            return rmsd

        # TM-score
        if tm_mode == "fixed":
            # TM-score under *your* chosen superposition (framework fit if align_first=True)
            coords_fixed = np.asarray([a.get_coord() for a in fixed_atoms], dtype=float)
            coords_moving = np.asarray([a.get_coord() for a in moving_atoms], dtype=float)
            tm = _tm_score_fixed(coords_moving, coords_fixed)
            return rmsd, tm

        elif tm_mode == "tmalign":
            # TM-align will re-align internally (best possible TM-score for the scoring selection)
            try:
                from tmtools import tm_align
            except ImportError as e:
                raise ImportError("tmtools is not installed. Install with: pip install tmtools") from e

            coords_fixed, seq_fixed = _atoms_to_coords_and_seq(fixed_atoms)
            coords_moving, seq_moving = _atoms_to_coords_and_seq(moving_atoms)

            res = tm_align(coords_moving, coords_fixed, seq_moving, seq_fixed)

            tm1 = float(res.tm_norm_chain1)
            tm2 = float(res.tm_norm_chain2)

            if tm_reduce == "chain1":
                tm = tm1
            elif tm_reduce == "chain2":
                tm = tm2
            elif tm_reduce == "mean":
                tm = 0.5 * (tm1 + tm2)
            else:
                tm = max(tm1, tm2)

            return rmsd, tm

        else:
            raise ValueError("tm_mode must be one of: 'fixed' | 'tmalign'")


def _normalize_residue_spec(spec):
    '''
    Normalize many possible residue specs into a set of (resseq, icode).

    spec can be:
      - list/tuple/set of residue ids: [105, "106A", (107,' ')]
      - dict with keys:
          {"ids": [...]} OR {"range": (start, end)} OR {"ranges": [(s,e),...]} OR combos
        where start/end can be int, "105A", or (105,'A')
    '''
    out = set()

    if spec is None:
        return out

    if isinstance(spec, (list, tuple, set)):
        for x in spec:
            out.add(_parse_res_id(x))
        return out

    if isinstance(spec, dict):
        if "ids" in spec and spec["ids"] is not None:
            for x in spec["ids"]:
                out.add(_parse_res_id(x))

        if "range" in spec and spec["range"] is not None:
            s, e = spec["range"]
            s_res, s_ic = _parse_res_id(s)
            e_res, e_ic = _parse_res_id(e)
            if s_ic != " " or e_ic != " ":
                raise ValueError("For 'range', use plain residue numbers without insertion codes.")
            for r in range(min(s_res, e_res), max(s_res, e_res) + 1):
                out.add((r, " "))

        if "ranges" in spec and spec["ranges"] is not None:
            for (s, e) in spec["ranges"]:
                s_res, s_ic = _parse_res_id(s)
                e_res, e_ic = _parse_res_id(e)
                if s_ic != " " or e_ic != " ":
                    raise ValueError("For 'ranges', use plain residue numbers without insertion codes.")
                for r in range(min(s_res, e_res), max(s_res, e_res) + 1):
                    out.add((r, " "))

        return out

    raise ValueError(f"Unsupported residue spec type: {type(spec)}")

def _tm_score_fixed(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
        '''
        TM-score computed on already-superposed coordinates with 1-to-1 correspondence.
        This follows the standard TM-score weighting:
        TM = (1/L) * Σ 1 / (1 + (d_i/d0)^2)
        using the usual d0(L) definition (for L>=15).
        '''
        coords_a = np.asarray(coords_a, float)
        coords_b = np.asarray(coords_b, float)
        assert coords_a.shape == coords_b.shape
        L = coords_a.shape[0]
        if L == 0:
            return float("nan")

        d = np.linalg.norm(coords_a - coords_b, axis=1)

        # Standard d0(L) used by TM-score / TM-align
        # For small L, clamp to avoid nonsensical values.
        if L < 15:
            d0 = 0.5  # conventional fallback; TM-align uses safeguards for small L
        else:
            d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
            d0 = max(d0, 0.5)

        return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))

def _parse_res_id(x):
    '''
    Accept:
      - int (e.g., 105) => (105, ' ')
      - tuple (resseq, icode) => (resseq, icode)
      - string:
          "105"   => (105, ' ')
          "105A"  => (105, 'A')
          "105 "  => (105, ' ')
    '''
    if isinstance(x, tuple) and len(x) == 2:
        resseq = int(x[0])
        icode = str(x[1]) if x[1] is not None else " "
        if icode == "":
            icode = " "
        return (resseq, icode)

    if isinstance(x, int):
        return (x, " ")

    if isinstance(x, str):
        s = x.strip()
        m = re.match(r"^(\d+)\s*([A-Za-z]?)$", s)
        if not m:
            raise ValueError(f"Unrecognized residue id: {x!r} (use 105, '105A', or (105,'A'))")
        resseq = int(m.group(1))
        icode = m.group(2) if m.group(2) else " "
        return (resseq, icode)

    raise ValueError(f"Unsupported residue id type: {type(x)}")

