import os
import sys
import math
import tempfile
import argparse
import warnings
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pdbfixer import PDBFixer
import numpy as np
import pandas as pd

import pyrosetta
from pyrosetta import rosetta

# OpenMM (for the OpenMM minimisation mode)
from openmm import app
from openmm import unit, Platform, LangevinIntegrator
from openmm.app import PDBFile, Modeller, Simulation, ForceField
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import shutil
warnings.filterwarnings("ignore", ".*is discontinuous.*")

"""
This script will be used for running the energy minimization and Rosetta energy calcualtions for TCR complex structures
Important things to note are:
- Before running, please check all TCR complex pdb files are standardised (i.e. chains are uniformely named, TCR-a always D, TCR-b always E etc. the actual names can be adapten in the main run block)
- Also, the complexes should be standardised so that MHC is truncated to same length (recommended is only the alpha1 and alpha2 domains / alpha1 and beta1 domains if MHC2)
TCRs should also be uniformely truncated (i.e. always just variable domain, or always full. recommended is only variable domain)
- Openmm energy minimisation yields the best results in my preliminary experiments
- rosetta sidechain packing is very fast compared to , and can be run in addition to openmm energy minimisation
- running rosetta with no energy minimisation is recommended for a baseline comparison
"""

# Per-process PyRosetta init flag
_PYROSETTA_INITIALIZED = False


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------

def ensure_pyrosetta_init():
    global _PYROSETTA_INITIALIZED
    if not _PYROSETTA_INITIALIZED:
        # You can add extra flags here if needed
        pyrosetta.init("-mute all")
        _PYROSETTA_INITIALIZED = True


def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0.0 else v


def angle_between(v1, v2):
    v1 = as_unit(v1)
    v2 = as_unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def to_spherical_angles(vec):
    v = as_unit(vec)
    x, y, z = v
    theta = math.degrees(math.acos(np.clip(z, -1.0, 1.0)))  # from +z
    phi = math.degrees(math.atan2(y, x))                    # azimuth
    return theta, phi


# --------------------------------------------------------------------------
# Rosetta helpers
# --------------------------------------------------------------------------

def split_pose_by_chains(pose, chains_1, chains_2):
    """
    Split a pose into two subposes by chain IDs.

    chains_1: string of chain IDs for partner 1 (e.g. "AC")
    chains_2: string of chain IDs for partner 2 (e.g. "DE")
    """
    pdb_info = pose.pdb_info()

    def make_subpose(chain_str):
        res_indices = rosetta.utility.vector1_unsigned_long()
        for i in range(1, pose.size() + 1):
            if pdb_info.chain(i) in chain_str:
                res_indices.append(i)

        subpose = rosetta.core.pose.Pose()
        rosetta.core.pose.pdbslice(subpose, pose, res_indices)
        subpose.conformation().detect_disulfides()
        return subpose

    p1 = make_subpose(chains_1)
    p2 = make_subpose(chains_2)
    return p1, p2


def build_chain_sets_explicit(
    pose,
    tcr_alpha_id,
    tcr_beta_id,
    mhc_alpha_id,
    mhc_beta_id,
    peptide_id,
):
    """
    Use explicitly provided chain IDs, but only keep those actually present
    in the pose. Build:
      - tcr_chains         = TCRα + TCRβ (if present)
      - mhc_only_chains    = MHCα + MHCβ (if present)
      - peptide_chain      = peptide_id (if present)
      - pmhc_chains        = mhc_only_chains + peptide_chain
      - partners_tcr_pmhc  = pmhc_chains _ tcr_chains
      - partners_pep_mhc   = mhc_only_chains _ peptide_chain
    """
    pdb_info = pose.pdb_info()
    chains_present = {pdb_info.chain(i) for i in range(1, pose.size() + 1)}

    def keep_if_present(chain_id):
        if chain_id and chain_id in chains_present:
            return chain_id
        if chain_id:
            print(f"[WARN] Requested chain '{chain_id}' not present in PDB.")
        return ""

    tcr_alpha = keep_if_present(tcr_alpha_id)
    tcr_beta = keep_if_present(tcr_beta_id)
    mhc_alpha = keep_if_present(mhc_alpha_id)
    mhc_beta = keep_if_present(mhc_beta_id)
    peptide_chain = keep_if_present(peptide_id)

    tcr_chains = "".join(c for c in (tcr_alpha, tcr_beta) if c)
    mhc_only_chains = "".join(c for c in (mhc_alpha, mhc_beta) if c)
    pmhc_chains = mhc_only_chains + (peptide_chain or "")

    partners_tcr_pmhc = (
        f"{pmhc_chains}_{tcr_chains}"
        if pmhc_chains and tcr_chains
        else ""
    )
    partners_pep_mhc = (
        f"{mhc_only_chains}_{peptide_chain}"
        if mhc_only_chains and peptide_chain
        else ""
    )

    print(f"Chains present        : {sorted(chains_present)}")
    print(f"  TCR chains          : {tcr_chains or 'None'}")
    print(f"  MHC-only chains     : {mhc_only_chains or 'None'}")
    print(f"  Peptide chain       : {peptide_chain or 'None'}")
    print(f"  pMHC chains (MHC+pep): {pmhc_chains or 'None'}")
    print(f"  partners TCR–pMHC   : {partners_tcr_pmhc or 'N/A'}")
    print(f"  partners pep–MHC    : {partners_pep_mhc or 'N/A'}")

    return {
        "tcr_chains": tcr_chains,
        "pmhc_chains": pmhc_chains,
        "peptide_chain": peptide_chain,
        "mhc_only_chains": mhc_only_chains,
        "partners_tcr_pmhc": partners_tcr_pmhc,
        "partners_pep_mhc": partners_pep_mhc,
    }


def binding_energy(pose, chains_tcr, chains_pmhc):
    """
    Compute ΔE_binding = E_complex - (E_TCR + E_pMHC) in Rosetta Energy Units.
    """
    scorefxn = rosetta.core.scoring.get_score_function()

    complex_pose = pose.clone()
    tcr, pmhc = split_pose_by_chains(complex_pose, chains_tcr, chains_pmhc)

    E_complex = scorefxn(complex_pose)
    E_tcr = scorefxn(tcr)
    E_pMHC = scorefxn(pmhc)

    dE = E_complex - (E_tcr + E_pMHC)

    print("E_complex:", E_complex)
    print("E_TCR    :", E_tcr)
    print("E_pMHC   :", E_pMHC)
    print("ΔE_binding (Rosetta units):", dE)

    return dE, E_complex


def analyze_interface(pose, partners_string):
    """
    Run InterfaceAnalyzerMover for a given partners string (e.g. 'ABC_DE').

    Returns:
        dict with interface_dG, dG_separated, delta_sasa
        or dict with NaNs if partners_string is empty.
    """
    if not partners_string:
        return {
            "interface_dG": float("nan"),
            "dG_separated": float("nan"),
            "delta_sasa": float("nan"),
        }

    scorefxn = rosetta.core.scoring.get_score_function()

    iam = rosetta.protocols.analysis.InterfaceAnalyzerMover(
        partners_string,
        False,          # tracer off
        scorefxn
    )

    pose_copy = pose.clone()
    iam.apply(pose_copy)

    interface_dG = iam.get_interface_dG()
    dG_separated = iam.get_separated_interface_energy()
    delta_sasa = iam.get_interface_delta_sasa()

    print(f"[InterfaceAnalyzer] partners={partners_string}")
    print("  interface_dG   :", interface_dG)
    print("  dG_separated   :", dG_separated)
    print("  ΔSASA          :", delta_sasa)

    return {
        "interface_dG": interface_dG,
        "dG_separated": dG_separated,
        "delta_sasa": delta_sasa,
    }


def relax_pose(pose, scorefxn, n_cycles=5):
    """
    Full Rosetta FastRelax (backbone + sidechains + repacking).
    """
    relax = rosetta.protocols.relax.FastRelax(scorefxn, n_cycles)
    relax.apply(pose)
    return pose


def quick_minimize_pose(pose, scorefxn):
    """
    Very fast local minimisation:
      - sidechain chi minimisation
      - rigid-body jumps
      - no backbone motion
      - no repacking
    """
    mm = rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(True)
    mm.set_jump(True)

    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.movemap(mm)
    min_mover.score_function(scorefxn)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.apply(pose)
    return pose


# --------------------------------------------------------------------------
# OpenMM minimisation (Amber19, GPU-enabled)
# --------------------------------------------------------------------------

def get_openmm_platform():
    """
    Try CUDA, then OpenCL, then CPU.
    """
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return Platform.getPlatformByName(name)
        except Exception:
            continue
    return Platform.getPlatform(0)

def run_openmm_minimization(pdb_in, pdb_out, max_iterations=1000):
    """
    Energy-minimise a PDB with OpenMM using Amber19 forcefield.

    Uses amber19-all.xml and amber19/tip3pfb.xml (no explicit solvent).
    Only hydrogens are added via Modeller; chain IDs/topology are preserved.
    """
    print(f"[OpenMM] Minimising {pdb_in} → {pdb_out}")

    # 1) Read PDB
    pdb = PDBFile(pdb_in)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # 2) Build modeller and add hydrogens
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens()   # adds missing Hs only

    # 3) Build system from *modeller.topology*
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )

    # 4) Integrator and platform
    integrator = LangevinIntegrator(
        300 * kelvin,
        1.0 / picosecond,
        0.004 * picosecond,
    )

    for plat_name in ("CUDA", "CPU"):
        try:
            platform = Platform.getPlatformByName(plat_name)
            props = {}
            if plat_name == "CUDA":
                props = {"CudaDeviceIndex": "0", "CudaPrecision": "mixed"}
            print(plat_name)
            simulation = Simulation(modeller.topology, system, integrator, platform, props)
            simulation.context.setPositions(modeller.positions)
            print("[OpenMM] Using platform:", plat_name)
            break
        except Exception as e:
            print(f"[OpenMM] {plat_name} init failed: {e}")
            simulation = None

    # 6) Minimize
    simulation.minimizeEnergy(maxIterations=max_iterations)

    # 7) Get final positions and energy
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    energy = state.getPotentialEnergy()
    print(f"[OpenMM] Minimized energy: {energy}")

    # 8) Write minimized structure
    with open(pdb_out, "w") as output:
        PDBFile.writeFile(modeller.topology, state.getPositions(), output, keepIds=True)

    return pdb_out, energy



def minimize_with_openmm(pdb_in, pdb_out, max_iterations=1000, pdb_path_fixed=None):
    """
    Energy-minimise a PDB with OpenMM using Amber19 forcefield.

    Uses amber19-all.xml and amber19/tip3pfb.xml (no explicit solvent added;
    the water FF file is present to keep parameters consistent if you later
    add solvent).
    """
    print(f"[OpenMM] Minimising {pdb_in} → {pdb_out}")
    try:
        run_openmm_minimization(pdb_in, pdb_out, max_iterations=max_iterations)

    except Exception as e:
        print(f"[OpenMM] Error during minimisation: {e}")
        fixer = PDBFixer(filename=pdb_in)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # pH 7
        if pdb_path_fixed:
            pdb_fixed=pdb_path_fixed
        else:
            pdb_fixed=pdb_out
        with open(pdb_fixed, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        try:
            pdb_out, energy=run_openmm_minimization(pdb_fixed, pdb_out, max_iterations=max_iterations)
        except Exception as e2:
            print(f"[OpenMM] Second minimisation attempt failed: {e2}")
            return "ERROR","ERROR"
    return pdb_out, energy


# --------------------------------------------------------------------------
# Per-PDB logic
# --------------------------------------------------------------------------

def process_one_complex(
    pdb_path,
    minimization_mode,
    tcr_alpha_id,
    tcr_beta_id,
    mhc_alpha_id,
    mhc_beta_id,
    peptide_id,
    pdb_path_fixed=None,
):
    """
    Load one TCR–pMHC complex and return all metrics as a dict.

    minimization_mode: one of
        - "none"
        - "rosetta_sidechain"
        - "rosetta_full"
        - "openmm"
    """
    ensure_pyrosetta_init()

    print(f"\n=== Processing {pdb_path} (mode={minimization_mode}) ===")

    # For OpenMM we minimise on the raw PDB then reload into PyRosetta.
    pdb_for_rosetta = pdb_path
    tmp_dir = None
    if minimization_mode == "openmm":
        tmp_dir = tempfile.mkdtemp(prefix="openmm_min_")
        pdb_min = os.path.join(tmp_dir, "minimized_openmm.pdb")
        pdb_for_rosetta, openmm_energy = minimize_with_openmm(pdb_path, pdb_min, pdb_path_fixed=pdb_path_fixed)
        shutil.copy(pdb_for_rosetta, pdb_path_fixed)

        if pdb_for_rosetta == "ERROR":
            raise RuntimeError("OpenMM minimisation failed.")

    pose = rosetta.core.import_pose.pose_from_file(pdb_for_rosetta)

    scorefxn = rosetta.core.scoring.get_score_function()
    if minimization_mode == "rosetta_sidechain":
        pose = quick_minimize_pose(pose, scorefxn)
    elif minimization_mode == "rosetta_full":
        pose = relax_pose(pose, scorefxn)

    total_complex_score = scorefxn(pose)
    print("Total Rosetta score (after chosen minimisation):", total_complex_score)

    # Use explicit chain IDs
    chain_sets = build_chain_sets_explicit(
        pose,
        tcr_alpha_id,
        tcr_beta_id,
        mhc_alpha_id,
        mhc_beta_id,
        peptide_id,
    )
    tcr_chains = chain_sets["tcr_chains"]
    pmhc_chains = chain_sets["pmhc_chains"]
    peptide_chain = chain_sets["peptide_chain"]
    mhc_only_chains = chain_sets["mhc_only_chains"]
    partners_tcr_pmhc = chain_sets["partners_tcr_pmhc"]
    partners_pep_mhc = chain_sets["partners_pep_mhc"]

    # Interface metrics: TCR–pMHC
    iface_tcr_pmhc = analyze_interface(pose, partners_tcr_pmhc)

    # Interface metrics: peptide–MHC
    iface_pep_mhc = analyze_interface(pose, partners_pep_mhc)

    # Binding energy TCR–pMHC (complex vs. TCR + pMHC)
    dE_binding, E_complex = (float("nan"), float("nan"))
    if tcr_chains and pmhc_chains:
        dE_binding, E_complex = binding_energy(
            pose, chains_tcr=tcr_chains, chains_pmhc=pmhc_chains
        )
    else:
        # If we could not define partners, fall back to total score as E_complex
        E_complex = total_complex_score

    metrics = {
        "minimization_mode": minimization_mode,
        "total_score": E_complex,
        "tcr_chains": tcr_chains,
        "pmhc_chains": pmhc_chains,
        "peptide_chain": peptide_chain,
        "mhc_only_chains": mhc_only_chains,
        "partners_tcr_pmhc": partners_tcr_pmhc,
        "partners_pep_mhc": partners_pep_mhc,
        # TCR–pMHC interface
        "interface_dG_tcr_pmhc": iface_tcr_pmhc["interface_dG"],
        "dG_separated_tcr_pmhc": iface_tcr_pmhc["dG_separated"],
        "delta_sasa_tcr_pmhc": iface_tcr_pmhc["delta_sasa"],
        # peptide–MHC interface
        "interface_dG_pep_mhc": iface_pep_mhc["interface_dG"],
        "dG_separated_pep_mhc": iface_pep_mhc["dG_separated"],
        "delta_sasa_pep_mhc": iface_pep_mhc["delta_sasa"],
        # Binding energy (Rosetta units)
        "dE_binding_tcr_pmhc": dE_binding,
    }

    if minimization_mode == "openmm":
        metrics["openmm_energy"] = openmm_energy

    print("Metrics:", metrics)
    return metrics


# Wrapper so we can use ProcessPoolExecutor
def worker_wrapper(args):
    (
        pdb_path,
        TCR_name,
        minimization_mode,
        tcr_alpha_id,
        tcr_beta_id,
        mhc_alpha_id,
        mhc_beta_id,
        peptide_id,
        pdb_path_fixed,
    ) = args

    try:
        metrics = process_one_complex(
            pdb_path=pdb_path,
            minimization_mode=minimization_mode,
            tcr_alpha_id=tcr_alpha_id,
            tcr_beta_id=tcr_beta_id,
            mhc_alpha_id=mhc_alpha_id,
            mhc_beta_id=mhc_beta_id,
            peptide_id=peptide_id,
            pdb_path_fixed=pdb_path_fixed,
        )
        row = {"TCR_name": TCR_name}
        row.update(metrics)
        return row
    except Exception as e:
        print(f"[ERROR] {pdb_path}: {e}")
        # Return a row with error info (optional)
        return {
            "TCR_name": TCR_name,
            "error": str(e),
            "minimization_mode": minimization_mode,
        }


# --------------------------------------------------------------------------
# Batch over directory (parallel)
# --------------------------------------------------------------------------

def run_folder(
    base_dir,
    fixed_dir,
    minimization_mode,
    out_csv,
    tcr_alpha_id,
    tcr_beta_id,
    mhc_alpha_id,
    mhc_beta_id,
    peptide_id,
    num_workers=None,
):
    """
    Process all PDBs in a folder with a given minimisation mode
    and write a CSV with all metrics. Parallel across PDBs.

    num_workers: number of worker processes (defaults to CPU count).
                 Be careful with 'openmm' mode if you have a single GPU:
                 too many workers can oversubscribe GPU memory.
    """
    pdb_files = [
        f for f in sorted(os.listdir(base_dir))
        if f.lower().endswith(".pdb")
    ]
    tasks = []
    for pdb_file in pdb_files:
        TCR_name = pdb_file.rsplit(".pdb", 1)[0]
        file_path = os.path.join(base_dir, pdb_file)
        tasks.append(
            (
                file_path,
                TCR_name,
                minimization_mode,
                tcr_alpha_id,
                tcr_beta_id,
                mhc_alpha_id,
                mhc_beta_id,
                peptide_id,
                os.path.join(fixed_dir, pdb_file)
            )
        )

    if num_workers is None or num_workers < 1:
        num_workers = multiprocessing.cpu_count()

    print(
        f"\n[RUN FOLDER] mode={minimization_mode}, "
        f"PDBs={len(tasks)}, workers={num_workers}"
    )

    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_wrapper, t): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                records.append(row)

    df = pd.DataFrame(records)
    print("\nFinal DataFrame:")
    print(df)

    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir="PATH/TO/FOLDER/WITH/PDB/FILES"
    fixed_dir="PATH/TO/FOLDER/WITH/PDB/FILES/opnemm_minimised" #here we store the openmm minimized structures for future reference
    os.makedirs(fixed_dir, exist_ok=True)
    #standardised, so that the chains are always named the same way, THIS IS IMPORTANT, MUST CHECK BEFORE RUNNING, otherwise wrong interfaces are used for calculation!
    tcr_alpha="D"
    tcr_beta="E"
    mhc_alpha="A"
    mhc_beta="B"
    peptide="C"
    modes = ["none", "rosetta_sidechain", "rosetta_full", "openmm"]
    for mode in modes:
        num_workers=4 #do 1 if using only 1 GPU
        out_csv = f"PATH_TO_OUTPUT_FILE.csv"
        run_folder(
            base_dir=base_dir,
            fixed_dir=fixed_dir,
            minimization_mode=mode,
            out_csv=out_csv,
            tcr_alpha_id=tcr_alpha,
            tcr_beta_id=tcr_beta,
            mhc_alpha_id=mhc_alpha,
            mhc_beta_id=mhc_beta,
            peptide_id=peptide,
            num_workers=num_workers
        )
