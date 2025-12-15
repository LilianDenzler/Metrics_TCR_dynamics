import os
import math
import warnings
import tempfile
from pathlib import Path
from collections import namedtuple

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")

# --- TCR-only change geometry (internal BA/BC1/AC1/BC2/AC2/dc) ---

from TCR_TOOLS.geometry.change_geometry2 import build_geometry_from_angles, rigid_transform_from_frames
from TCR_TOOLS.geometry.calc_geometry import get_tcr_points_world
from TCR_TOOLS.geometry.TCR_PMHC_geo_new import mhc_canonical_frame, get_chain_ca_coords, as_unit,pep_end_to_end_dir_from_structure
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.core.io import write_pdb
warnings.filterwarnings("ignore", ".*is discontinuous.*")

# -------------------------
# Basic geometry helpers
# -------------------------
Points = namedtuple("Points", ["C", "V1", "V2"])  # centroid + PC1/PC2 endpoints




def sph_to_cart(theta_deg, phi_deg):
    """Inverse of to_spherical_angles()."""
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    s = math.sin(th)
    x = s * math.cos(ph)
    y = s * math.sin(ph)
    z = math.cos(th)
    return np.array([x, y, z], dtype=float)


def to_spherical_angles(vec):
    """
    vec in some frame (3-vector) -> (theta, phi) in degrees.

    theta: angle from +z axis (0..180)
    phi  : azimuth in xy-plane from +x, toward +y (-180..180]
    """
    v = as_unit(vec)
    x, y, z = v
    theta = math.degrees(math.acos(np.clip(z, -1.0, 1.0)))
    phi = math.degrees(math.atan2(y, x))
    return theta, phi


def kabsch_rotation(P_src, P_tgt):
    """
    Kabsch algorithm: best rotation R such that R @ P_src[i] ~= P_tgt[i].
    P_src, P_tgt : (N,3) arrays of vectors.
    """
    P_src = np.asarray(P_src, dtype=float)
    P_tgt = np.asarray(P_tgt, dtype=float)

    H = P_src.T @ P_tgt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def flip_tcr_about_mhc_x_inplace(structure, R_mhc, com_mhc, alpha_chain_id, beta_chain_id):
    """
    Flip the TCR (chains alpha/beta) 180° about the MHC canonical x-axis.
    This changes which side of the MHC plane (+/- z) the TCR sits on,
    while preserving the MHC frame definition.
    """
    Rx = np.array([[1, 0, 0],
                   [0,-1, 0],
                   [0, 0,-1]], dtype=float)  # 180° about x

    model = next(structure.get_models())
    for chain in model:
        if chain.id not in {alpha_chain_id, beta_chain_id}:
            continue
        for residue in chain:
            for atom in residue:
                p = atom.coord.astype(float)
                p_mhc = R_mhc @ (p - com_mhc)
                p_mhc = Rx @ p_mhc
                atom.coord = (R_mhc.T @ p_mhc + com_mhc).astype(np.float32)


# -------------------------
# TCR extraction & embedding
# -------------------------
class TCRSelect(Select):
    def __init__(self, alpha_chain_id, beta_chain_id):
        self.allowed = {alpha_chain_id, beta_chain_id}

    def accept_chain(self, chain):
        return chain.id in self.allowed



def apply_tcr_internal_change_to_complex(complex_structure,
                                         changed_tcr_structure,
                                         alpha_chain_id="A",
                                         beta_chain_id="B"):
    """
    Copy coordinates from changed TCR (chains A/B) into the complex structure.
    Matching is by (chain_id, residue_id_tuple, atom_name).
    """
    changed_coords = {}
    model_tcr = next(changed_tcr_structure.get_models())
    for chain in model_tcr:
        if chain.id not in {alpha_chain_id, beta_chain_id}:
            continue
        for residue in chain:
            res_id = residue.get_id()      # (' ', resseq, icode)
            for atom in residue:
                key = (chain.id, res_id, atom.get_name())
                changed_coords[key] = atom.coord.copy()

    model_cx = next(complex_structure.get_models())
    n_updated = 0
    for chain in model_cx:
        if chain.id not in {alpha_chain_id, beta_chain_id}:
            continue
        for residue in chain:
            res_id = residue.get_id()
            for atom in residue:
                key = (chain.id, res_id, atom.get_name())
                if key in changed_coords:
                    atom.coord = changed_coords[key].astype(np.float32)
                    n_updated += 1

    print(f"[change-geom] Updated {n_updated} TCR atom coordinates in complex.")

def change_TCR_geometry(input_pdb_path, alpha_chain_id, beta_chain_id, out_dir,
                        BA, BC1, BC2, AC1, AC2, dc):

    # old points (world) from original structure (only needed for R_A/R_B)
    A_old, B_old = get_tcr_points_world(input_pdb_path, out_dir,
                                        alpha_chain_id=alpha_chain_id,
                                        beta_chain_id=beta_chain_id)

    A_C, A_V1, A_V2, B_C, B_V1, B_V2 = build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc)
    A_target = Points(C=A_C, V1=A_V1, V2=A_V2)
    B_target = Points(C=B_C, V1=B_V1, V2=B_V2)

    R_A, t_A = rigid_transform_from_frames(A_old, A_target)
    R_B, t_B = rigid_transform_from_frames(B_old, B_target)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", input_pdb_path)

    for model in structure:
        for chain in model:
            if chain.id not in (alpha_chain_id, beta_chain_id):
                continue
            R, t = (R_A, t_A) if chain.id == alpha_chain_id else (R_B, t_B)
            for residue in chain:
                for atom in residue:
                    coord = np.asarray(atom.get_coord(), float)
                    atom.set_coord(R @ coord + t)

    changed_path = os.path.join(out_dir, "changed_tcr.pdb")
    write_pdb(changed_path, structure)

    # IMPORTANT: recompute points in WORLD frame from the changed structure
    A_world, B_world = get_tcr_points_world(changed_path, out_dir,
                                           alpha_chain_id=alpha_chain_id,
                                           beta_chain_id=beta_chain_id)

    return changed_path, structure, A_world, B_world

# -------------------------
# Core function
# -------------------------
def run(
    input_complex_pdb,
    # internal TCR invariants (for TCR-only script)
    BA,
    BC1,
    BC2,
    AC1,
    AC2,
    dc,
    # external distances:
    d_alpha_mhc,
    d_beta_mhc,
    d_alpha_beta,
    # external angles in MHC canonical frame:
    theta_rA,
    phi_rA,
    theta_rB,
    phi_rB,
    theta_pc1A,
    phi_pc1A,
    theta_pc1B,
    phi_pc1B,
    out_dir=None,
    alpha_chain_id="A",
    beta_chain_id="B",
    mhc_chain_ids=("M", "N"),
    pep_chain_id=("C")
):
    """
    Change geometry of a TCR–pMHC complex in TWO stages:

      1) INTERNAL TCR GEOMETRY:
         - Call the TCR-only change_geometry2.run() to impose
           (BA, BC1, BC2, AC1, AC2, dc) on the TCR alone.
         - Embed the changed TCR back into the complex (MHC unchanged).

      2) EXTERNAL TCR–MHC GEOMETRY:
         - In the MHC canonical frame, reposition the ENTIRE (already-changed)
           TCR as a rigid body so that:
             * |COM(α) - COM(MHC)| = d_alpha_mhc
             * |COM(β) - COM(MHC)| = d_beta_mhc
             * |COM(β) - COM(α)|  ≈ d_alpha_beta
             * directions of MHC→α, MHC→β, and PC1_A are given by the
               spherical angles (theta_rA, phi_rA, ... etc.).

    Parameters
    ----------
    input_complex_pdb : str
        Path to PDB containing TCR (A/B), MHC (M/N), peptide (e.g. C), etc.
    BA, BC1, BC2, AC1, AC2, dc : float
        Internal TCR geometry invariants (same definitions as TCR-only calc).
    d_alpha_mhc, d_beta_mhc, d_alpha_beta : float
        Distances (Å) for TCR–MHC geometry.
    theta_rA, phi_rA, theta_rB, phi_rB : float
        Spherical angles (deg) for the COM vectors in MHC canonical frame.
    theta_pc1A, phi_pc1A, theta_pc1B, phi_pc1B : float
        Spherical angles (deg) for PC1 directions in MHC canonical frame.
    out_path : Optional[str]
        Output PDB path; if None, a temp file is created.
    alpha_chain_id, beta_chain_id : str
        Chain IDs of TCR alpha/beta in the complex.
    mhc_chain_ids : tuple[str, ...]
        Chain IDs for the MHC heavy chains (e.g. ("M","N")).

    Returns
    -------
    final_pdb : str
        Path to the geometry-changed complex.
    """
    #parser = PDBParser(QUIET=True)
    #complex_structure = parser.get_structure("complex_orig", input_complex_pdb)
    if out_dir is None:
        out_dir=tempfile.mkdtemp()
    complex_path, complex_structure,Apts_world, Bpts_world = change_TCR_geometry(input_complex_pdb, alpha_chain_id, beta_chain_id, out_dir, BA, BC1, BC2, AC1, AC2, dc)


    # ------------------- Stage 2: external TCR–MHC rigid-body repositioning -------------------

    # (2a) Build MHC canonical frame from complex (MHC chains only)
    coords_mhc = get_chain_ca_coords(complex_structure, mhc_chain_ids)
    coords_pep = get_chain_ca_coords(complex_structure, [pep_chain_id])
    pep_dir_world = pep_end_to_end_dir_from_structure(complex_structure, pep_chain_id)
    R_mhc, com_mhc, pc1, pc2, n_mhc_plane = mhc_canonical_frame(coords_mhc, coords_pep, pep_dir_world=pep_dir_world)

    def world_to_mhc(p):
        return R_mhc @ (np.asarray(p, float) - com_mhc)

    def mhc_to_world(p_mhc):
        return R_mhc.T @ np.asarray(p_mhc, float) + com_mhc

    # COMs in MHC frame
    rA_src = world_to_mhc(Apts_world.C)
    rB_src = world_to_mhc(Bpts_world.C)

    # PC1 directions in MHC frame
    pc1A_src_dir = world_to_mhc(Apts_world.V1) - rA_src
    pc1B_src_dir = world_to_mhc(Bpts_world.V1) - rB_src
    pc1A_src_dir = as_unit(pc1A_src_dir)
    pc1B_src_dir = as_unit(pc1B_src_dir)

    # Axis from beta to alpha in MHC frame (used as second vector for Kabsch)
    axis_src = as_unit(rA_src - rB_src)

    # (2c) Target COMs & directions from invariants (MHC canonical frame)
    rA_tgt = d_alpha_mhc * sph_to_cart(theta_rA, phi_rA)
    rB_tgt = d_beta_mhc * sph_to_cart(theta_rB, phi_rB)

    axis_tgt = as_unit(rA_tgt - rB_tgt)
    d_ab_from_targets = np.linalg.norm(rB_tgt - rA_tgt)
    print(
        f"[change-geom] d_alpha_beta target={d_alpha_beta:.2f} Å, "
        f"implied by COM vectors={d_ab_from_targets:.2f} Å"
    )

    pc1A_tgt_dir = sph_to_cart(theta_pc1A, phi_pc1A)
    pc1B_tgt_dir = sph_to_cart(theta_pc1B, phi_pc1B)

    # (2d) Rotation in MHC frame:
    # map (pc1A_src, axis_src) -> (pc1A_tgt, axis_tgt) with a pure rotation
    R_tcr_mhc = kabsch_rotation(
        np.stack([pc1A_src_dir, axis_src], axis=0),
        np.stack([pc1A_tgt_dir, axis_tgt], axis=0),
    )

    # Translation so that rA_src -> rA_tgt in MHC frame
    t_tcr_mhc = rA_tgt - R_tcr_mhc @ rA_src

    # (2e) Apply transform to all TCR atoms (chains A,B) in the complex
    model_cx = next(complex_structure.get_models())
    for chain in model_cx:
        if chain.id not in {alpha_chain_id, beta_chain_id}:
            continue
        for residue in chain:
            for atom in residue:
                p_world = atom.coord.astype(float)
                p_mhc = world_to_mhc(p_world)
                p_mhc_new = R_tcr_mhc @ p_mhc + t_tcr_mhc
                p_world_new = mhc_to_world(p_mhc_new)
                atom.coord = p_world_new.astype(np.float32)
    # --- Enforce: TCR must be on peptide side of MHC plane ---
    # peptide side sign (robust: median signed distance of peptide CA to MHC plane)
    signed_pep = (coords_pep - com_mhc) @ n_mhc_plane
    pep_side = np.sign(np.median(signed_pep))

    # tcr side sign (median signed distance of TCR CA to MHC plane)
    tcr_ca = []
    model = next(complex_structure.get_models())
    for ch in model:
        if ch.id in {alpha_chain_id, beta_chain_id}:
            for res in ch:
                if "CA" in res:
                    tcr_ca.append(res["CA"].coord)

    tcr_ca = np.asarray(tcr_ca, float)
    signed_tcr = (tcr_ca - com_mhc) @ n_mhc_plane
    tcr_side = np.sign(np.median(signed_tcr))

    if pep_side != 0 and tcr_side != 0 and pep_side != tcr_side:
        print("[change-geom] TCR ended up on wrong side of MHC plane. Flipping 180° about MHC x-axis.")
        flip_tcr_about_mhc_x_inplace(
            complex_structure, R_mhc, com_mhc, alpha_chain_id, beta_chain_id
        )

    # QC logs
    d_alpha_final = float(np.linalg.norm(rA_tgt))
    d_beta_final = float(np.linalg.norm(rB_tgt))
    d_ab_final = float(np.linalg.norm(rB_tgt - rA_tgt))

    print(
        f"[change-geom] Final d_alpha_mhc={d_alpha_final:.2f} Å "
        f"(target {d_alpha_mhc:.2f})"
    )
    print(
        f"[change-geom] Final d_beta_mhc ={d_beta_final:.2f} Å "
        f"(target {d_beta_mhc:.2f})"
    )
    print(
        f"[change-geom] Final d_alpha_beta={d_ab_final:.2f} Å "
        f"(target {d_alpha_beta:.2f})"
    )

    pc1A_final_dir = R_tcr_mhc @ pc1A_src_dir
    pc1B_final_dir = R_tcr_mhc @ pc1B_src_dir
    thA_final, phA_final = to_spherical_angles(pc1A_final_dir)
    thB_final, phB_final = to_spherical_angles(pc1B_final_dir)

    print(
        f"[change-geom] PC1_A angles final (θ,φ)=({thA_final:.1f},{phA_final:.1f}) "
        f"target=({theta_pc1A:.1f},{phi_pc1A:.1f})"
    )
    print(
        f"[change-geom] PC1_B angles final (θ,φ)=({thB_final:.1f},{phB_final:.1f}) "
        f"target=({theta_pc1B:.1f},{phi_pc1B:.1f})"
    )

    # ------------------- Save final complex -------------------
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    out_path = str(Path(out_dir) / "complex_geometry_changed.pdb")

    io_out = PDBIO()
    io_out.set_structure(complex_structure)
    io_out.save(out_path)
    print(f"💾 Geometry-changed complex written to: {out_path}")

    return out_path
