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
from Bio.SeqUtils import seq1
# --- project imports (pure utilities & constants) ---
from TCR_TOOLS.core import ops, io
import tempfile, os

from TCR_TOOLS.core.select import make_region_atom_predicate, region_list_for_names
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE

# --- numbering/pairing/alignment (your existing modules) ---
from TCR_TOOLS.numbering.main import fix_duplicate_resseqs_by_insertion_code,apply_imgt_renumbering
from TCR_TOOLS.numbering.tcr_pairing import pair_tcrs_by_interface

from TCR_TOOLS.geometry.TCR_PMHC_geo import run as calc_complex_angles_single_pdb
from TCR_TOOLS.geometry.TCR_PMHC_change_geo import  run_change_geometry as change_complex_angles_single_pdb
from Bio.PDB import PDBIO


# -------------------------------------------------------------------
# TCR root object (owns original/imgt structures and per-pair views)
# -------------------------------------------------------------------
@dataclass
class TCRpMHC:
    input_pdb: str
    MHC_a_chain_id: str = "M"
    MHC_b_chain_id: str = "N"
    Peptide_chain_id: str = "P"
    contact_cutoff: float = 5.0
    min_contacts: int = 50
    legacy_anarci: bool = True

    scheme: str = "imgt"

    original_structure: BPStructure.Structure = field(init=False)
    imgt_all_structure: BPStructure.Structure = field(init=False)
    variable_structure: BPStructure.Structure = field(init=False, default=None)

    def rename_pMHC_chains(self):

        old_alpha_chainid=self.chain_types_dict.get("A","G")
        old_beta_chainid=self.chain_types_dict.get("B","D")
        """Rename MHC and peptide chains to standard IDs."""
        chain_id_map = {
            self.MHC_a_chain_id: "M",
            self.MHC_b_chain_id: "N",
            self.Peptide_chain_id: "P",
            old_alpha_chainid: "A",
            old_beta_chainid: "B"
        }
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

    def __post_init__(self):
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
        self.rename_pMHC_chains()
        print("IMGT renumbering applied.")

    def calc_geometry(self):
        #write to pdb
        io = PDBIO()
        io.set_structure(self.imgt_all_structure)
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        io.save(tmp_pdb.name)
        input_pdb = tmp_pdb.name
        df=calc_complex_angles_single_pdb(input_pdb, out_path="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test", vis=True, cleanup_tmp=False)
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

    def change_geometry(self,output_pdb,geo_dict):

        io = PDBIO()
        io.set_structure(self.imgt_all_structure)
        tmp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        io.save(tmp_pdb.name)
        input_pdb = tmp_pdb.name

        full_final_pdb=change_complex_angles_single_pdb(
                input_pdb_full=input_pdb,
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
                out_pdb=output_pdb)

