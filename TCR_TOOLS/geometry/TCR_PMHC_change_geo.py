import os
import math
import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
from Bio.PDB import PDBParser, PDBIO

import biotite.structure as bts
import biotite.structure.io as btsio

from . import DATA_PATH
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.geometry.change_geometry import run as change_angles_single_pdb
warnings.filterwarnings("ignore", ".*is discontinuous.*")

Points = namedtuple("Points", ["C", "V1", "V2"])  # centroid + PC1/PC2 endpoints


# ========= helpers (must match calc script) =========

def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def center_of_mass(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        raise ValueError("No coordinates provided for center_of_mass()")
    return coords.mean(axis=0)


def pca_axes(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] < 3:
        raise ValueError("Need at least 3 points for PCA.")
    mean = coords.mean(axis=0)
    X = coords - mean
    cov = X.T @ X / (X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    for i in range(vecs.shape[1]):
        vecs[:, i] = as_unit(vecs[:, i])
    return vals, vecs, mean


def mhc_canonical_frame(coords_mhc):
    """
    Same as in calc script:
      origin = COM(MHC)
      z-axis = plane normal (PC1 x PC2)
      x-axis = PC1 projected into plane
      y-axis = z × x

    Returns R_mhc that maps WORLD -> MHC frame.
    """
    _, vecs_mhc, com_mhc = pca_axes(coords_mhc)
    pc1 = vecs_mhc[:, 0]
    pc2 = vecs_mhc[:, 1]

    n_mhc = as_unit(np.cross(pc1, pc2))  # plane normal
    x_axis = as_unit(pc1 - np.dot(pc1, n_mhc) * n_mhc)
    z_axis = n_mhc
    y_axis = as_unit(np.cross(z_axis, x_axis))

    R_mhc = np.vstack([x_axis, y_axis, z_axis])  # rows = basis (world->frame)
    return R_mhc, com_mhc, pc1, pc2, n_mhc


def get_chain_ca_coords(structure, chain_ids):
    coords = []
    model = next(structure.get_models())
    chain_ids = set(chain_ids)
    for chain in model:
        if chain.id in chain_ids:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].coord)
    if not coords:
        raise ValueError(f"No CA atoms found for chains {chain_ids}")
    return np.array(coords, dtype=float)


def to_spherical_angles(vec):
    """
    Same convention as calc script:
      theta: angle from +z (0..180)
      phi  : azimuth from +x in x–y plane, in [-180, 180)
    """
    v = as_unit(vec)
    x, y, z = v
    theta = math.degrees(math.acos(np.clip(z, -1.0, 1.0)))
    phi = math.degrees(math.atan2(y, x))
    return theta, phi


def sph_to_cart(theta_deg, phi_deg):
    """
    Inverse of to_spherical_angles().
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    s = math.sin(th)
    x = s * math.cos(ph)
    y = s * math.sin(ph)
    z = math.cos(th)
    return np.array([x, y, z], dtype=float)


def read_pseudo_points(pdb_path, chain_id_main):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("consensus", pdb_path)

    if chain_id_main not in [ch.id for ch in s[0]]:
        raise ValueError(f"Chain '{chain_id_main}' not found in {pdb_path}")

    try:
        z = s[0]["Z"]
    except KeyError:
        raise ValueError(
            f"Chain 'Z' with pseudoatoms (CEN/PC1/PC2) not found in {pdb_path}"
        )

    def _get_first_atom(res_name):
        for res in z:
            if res.get_resname() == res_name:
                for atom in res:
                    return atom.get_coord()
        raise ValueError(f"Pseudoatom '{res_name}' not found in chain Z of {pdb_path}")

    C = _get_first_atom("CEN")
    V1 = _get_first_atom("PC1")
    V2 = _get_first_atom("PC2")
    return Points(C=np.array(C, float), V1=np.array(V1, float), V2=np.array(V2, float))


def apply_affine_to_atomarray(atomarray, transform):
    arr = atomarray.copy()
    M = np.asarray(transform.as_matrix(), dtype=np.float64)
    if M.shape == (1, 4, 4):
        M = M[0]
    R = M[:3, :3]
    t = M[:3, 3]
    coords = np.asarray(arr.coord, dtype=np.float64)
    new_coords = (coords @ R.T) + t
    arr.coord = new_coords.astype(np.float32)
    return arr


def align_with_biotite(
    static_pdb_file: str,
    mobile_pdb_file: str,
    output_pdb_file: str,
    chain_name: str,
    static_consenus_res: list = None,
    mobile_consenus_res: list = None,
):
    static_structure = btsio.load_structure(static_pdb_file, model=1)
    mobile_structure = btsio.load_structure(mobile_pdb_file, model=1)

    static_mask = (static_structure.atom_name == "CA") & (
        static_structure.chain_id == chain_name
    )
    mobile_mask = (mobile_structure.atom_name == "CA") & (
        mobile_structure.chain_id == chain_name
    )

    static_ca = static_structure[static_mask]
    mobile_ca = mobile_structure[mobile_mask]
    if static_consenus_res:
        static_ca = static_ca[np.isin(static_ca.res_id, static_consenus_res)]
    if mobile_consenus_res:
        mobile_ca = mobile_ca[np.isin(mobile_ca.res_id, mobile_consenus_res)]
    if static_ca.array_length() < 4 or mobile_ca.array_length() < 4:
        raise ValueError(f"Not enough CA atoms to align on chain {chain_name}")

    _, transform, _, _ = bts.superimpose_structural_homologs(
        fixed=static_ca, mobile=mobile_ca, max_iterations=1
    )
    mobile_full_aligned = apply_affine_to_atomarray(mobile_structure, transform)
    btsio.save_structure(output_pdb_file, mobile_full_aligned)
    print(f"💾 Saved aligned structure to '{output_pdb_file}'")
    return transform


def kabsch_rotation(src_vecs, tgt_vecs):
    """
    Pure rotation (no translation) mapping src_vecs -> tgt_vecs.
    src_vecs, tgt_vecs: (N,3) arrays of vectors (assumed from origin).
    """
    P = np.asarray(src_vecs, dtype=float)
    Q = np.asarray(tgt_vecs, dtype=float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("src_vecs and tgt_vecs must be Nx3 arrays")

    C = P.T @ Q  # since all from origin
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R



def get_TCR_points(input_pdb_full,out_pdb):
    # 2) Map consensus pseudoatoms -> WORLD -> MHC frame (source geometry)
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")

    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    tmp_dir = Path(out_pdb).parent / "tmp_change_geom"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    aligned_input_path = str(tmp_dir / "aligned_input.pdb")
    aligned_consB_path = str(tmp_dir / "aligned_consB.pdb")

    # Align input to consensus A (same as calc)
    transform_input = align_with_biotite(
        static_pdb_file=consA_pca_path,
        mobile_pdb_file=input_pdb_full,
        output_pdb_file=aligned_input_path,
        chain_name="A",
        static_consenus_res=A_consenus_res,
        mobile_consenus_res=A_consenus_res,
    )

    # Align consensus B to aligned input (same as calc)
    _ = align_with_biotite(
        static_pdb_file=aligned_input_path,
        mobile_pdb_file=consB_pca_path,
        output_pdb_file=aligned_consB_path,
        chain_name="B",
        static_consenus_res=B_consenus_res,
        mobile_consenus_res=B_consenus_res,
    )

    # Pseudoatoms in "consensus-aligned" frame
    Apts_cons = read_pseudo_points(consA_pca_path, chain_id_main="A")
    Bpts_cons = read_pseudo_points(aligned_consB_path, chain_id_main="B")

    # Map pseudoatoms back to WORLD frame
    M_bi = np.asarray(transform_input.as_matrix(), dtype=np.float64)
    if M_bi.shape == (1, 4, 4):
        M_bi = M_bi[0]
    R_bi = M_bi[:3, :3]
    t_bi = M_bi[:3, 3]
    R_inv = R_bi.T

    def cons_to_world(p_cons):
        p_cons = np.asarray(p_cons, dtype=float)
        return R_inv @ (p_cons - t_bi)

    Apts_world = Points(
        C=cons_to_world(Apts_cons.C),
        V1=cons_to_world(Apts_cons.V1),
        V2=cons_to_world(Apts_cons.V2),
    )
    Bpts_world = Points(
        C=cons_to_world(Bpts_cons.C),
        V1=cons_to_world(Bpts_cons.V1),
        V2=cons_to_world(Bpts_cons.V2),
    )
    return Apts_world, Bpts_world
# ========= main: change geometry =========


def change_TCR_geometry(input_pdb, out_path, BA, BC1, BC2, AC1, AC2, dc):
    final_aligned_pdb=change_angles_single_pdb(input_pdb,out_path, BA, BC1, BC2, AC1, AC2, dc)
    input(f"Final aligned pdb saved to: {final_aligned_pdb}. Press Enter to continue...")
    return final_aligned_pdb




def run_change_geometry(
    input_pdb_full,
    # internal invariants (kept for completeness, not used right now)
    BA,
    BC1,
    AC1,
    BC2,
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
    out_pdb,
):
    """
    Reposition the TCR relative to the MHC using ONLY the invariants
    produced by your calc script.

    MHC atoms stay fixed; only chains A and B are rigidly moved.
    """
    #os.makedirs(out_pdb, exist_ok=True)
    #out_pdb=change_TCR_geometry(input_pdb_full, out_pdb, BA, BC1, BC2, AC1, AC2, dc)
    #input_pdb_full=out_pdb

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb_full)
    model = next(structure.get_models())

    # 1) Canonical MHC frame of *this* complex
    coords_mhc = get_chain_ca_coords(structure, ["M", "N"])
    R_mhc, com_mhc, _, _, _ = mhc_canonical_frame(coords_mhc)

    def world_to_mhc(p):
        return R_mhc @ (np.asarray(p, float) - com_mhc)

    def mhc_to_world(p_mhc):
        return R_mhc.T @ np.asarray(p_mhc, float) + com_mhc

    Apts_world, Bpts_world=get_TCR_points(input_pdb_full,out_pdb)

    # 3) Source COMs & PC1 dirs in MHC frame
    rA_src = world_to_mhc(Apts_world.C)
    rB_src = world_to_mhc(Bpts_world.C)

    pc1A_src_dir = world_to_mhc(Apts_world.V1) - rA_src  # still in MHC frame but relative to COM
    pc1B_src_dir = world_to_mhc(Bpts_world.V1) - rB_src
    pc1A_src_dir = as_unit(pc1A_src_dir)
    pc1B_src_dir = as_unit(pc1B_src_dir)

    axis_src = as_unit(rA_src - rB_src)

    # 4) Target COMs & PC1 dirs from invariants (in MHC frame)
    rA_tgt = d_alpha_mhc * sph_to_cart(theta_rA, phi_rA)
    rB_tgt = d_beta_mhc * sph_to_cart(theta_rB, phi_rB)

    axis_tgt = as_unit(rA_tgt - rB_tgt)
    d_ab_from_targets = np.linalg.norm(rB_tgt - rA_tgt)
    print(
        f"[change-geom] d_alpha_beta target={d_alpha_beta:.2f} Å, "
        f"from (d_alpha,d_beta,angles)={d_ab_from_targets:.2f} Å"
    )

    pc1A_tgt_dir = sph_to_cart(theta_pc1A, phi_pc1A)
    pc1B_tgt_dir = sph_to_cart(theta_pc1B, phi_pc1B)

    # 5) Rotation in MHC frame: map (pc1A_src, axis_src) -> (pc1A_tgt, axis_tgt)
    R_tcr_mhc = kabsch_rotation(
        np.stack([pc1A_src_dir, axis_src], axis=0),
        np.stack([pc1A_tgt_dir, axis_tgt], axis=0),
    )

    # Translation so that rA_src -> rA_tgt
    t_tcr_mhc = rA_tgt - R_tcr_mhc @ rA_src

    # 6) Apply transform to all TCR atoms (chains A,B)
    for chain in model:
        if chain.id not in {"A", "B"}:
            continue
        for residue in chain:
            for atom in residue:
                p_world = atom.coord.astype(float)
                p_mhc = world_to_mhc(p_world)
                p_mhc_new = R_tcr_mhc @ p_mhc + t_tcr_mhc
                p_world_new = mhc_to_world(p_mhc_new)
                atom.coord = p_world_new.astype(np.float32)

    # 7) QC: recompute values from target configuration (should match invariants)
    d_alpha_final = np.linalg.norm(rA_tgt)
    d_beta_final = np.linalg.norm(rB_tgt)
    d_ab_final = np.linalg.norm(rB_tgt - rA_tgt)

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
        f"[change-geom] PC1_A angles final (theta,phi)=({thA_final:.1f},{phA_final:.1f}) "
        f"target=({theta_pc1A:.1f},{phi_pc1A:.1f})"
    )
    print(
        f"[change-geom] PC1_B angles final (theta,phi)=({thB_final:.1f},{phB_final:.1f}) "
        f"target=({theta_pc1B:.1f},{phi_pc1B:.1f})"
    )

    # 8) Save structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)
    print(f"💾 Geometry-changed complex written to: {out_pdb}")


    return out_pdb
