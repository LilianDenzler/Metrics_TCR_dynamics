import os
import math
import shutil
import tempfile
import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

# Biopython for quick writing/reading
from Bio.PDB import PDBParser

# Biotite for structural alignment
import biotite.structure as bts
import biotite.structure.io as btsio

from . import DATA_PATH

warnings.filterwarnings("ignore", ".*is discontinuous.*")


# -------------------------
# Geometry helpers
# -------------------------
Points = namedtuple("Points", ["C", "V1", "V2"])  # endpoints (absolute coords)


def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_between(v1, v2):
    v1 = as_unit(v1)
    v2 = as_unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def angle_vec_plane(v, plane_normal):
    """
    Angle between a vector and a plane in [0, 90] degrees.
    0° -> vector lies in the plane
    90° -> vector is perpendicular to the plane
    """
    v = as_unit(v)
    n = as_unit(plane_normal)
    cos_phi = np.clip(abs(float(np.dot(v, n))), -1.0, 1.0)
    phi = math.degrees(math.acos(cos_phi))
    return 90.0 - phi


def center_of_mass(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.size == 0:
        raise ValueError("No coordinates provided for center_of_mass()")
    return coords.mean(axis=0)


def pca_axes(coords):
    """
    PCA on coords -> eigenvalues, eigenvectors, mean.
    Eigenvectors are columns, sorted by descending eigenvalue.
    """
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


def angle_between_planes(n1, n2):
    """
    Angle between planes with normals n1, n2 in [0, 90] degrees.
    (QC metric; not used as invariant.)
    """
    n1 = as_unit(n1)
    n2 = as_unit(n2)
    cosang = abs(float(np.clip(np.dot(n1, n2), -1.0, 1.0)))
    return math.degrees(math.acos(cosang))


def get_chain_ca_coords(structure, chain_ids):
    """
    Get all CA coordinates for a list of chain IDs from a Biopython Structure.
    """
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
    vec in some frame (3-vector) -> (theta, phi) in degrees.

    theta: angle from +z axis (0..180)
    phi  : azimuth in xy-plane from +x, toward +y (-180..180]
    """
    v = as_unit(vec)
    x, y, z = v
    theta = math.degrees(math.acos(np.clip(z, -1.0, 1.0)))
    phi = math.degrees(math.atan2(y, x))
    return theta, phi


# -------------------------
# MHC canonical frame
# -------------------------

def mhc_canonical_frame(coords_mhc):
    """
    Given all MHC CA coords, build a canonical MHC frame:

      - origin at COM(MHC)
      - z-axis along MHC plane normal (PC1 x PC2)
      - x-axis along in-plane PC1
      - y-axis = z × x

    Returns:
      R_mhc    : 3x3 rotation mapping WORLD -> MHC frame
      com_mhc  : 3-vector COM in world coords
      pc1      : PC1 of MHC in world coords
      pc2      : PC2 of MHC in world coords
      n_mhc    : plane normal in world coords
    """
    _, vecs_mhc, com_mhc = pca_axes(coords_mhc)
    pc1 = vecs_mhc[:, 0]
    pc2 = vecs_mhc[:, 1]

    n_mhc = as_unit(np.cross(pc1, pc2))  # plane normal
    # in-plane x-axis (ensure orthogonal to normal)
    x_axis = as_unit(pc1 - np.dot(pc1, n_mhc) * n_mhc)
    z_axis = n_mhc
    y_axis = as_unit(np.cross(z_axis, x_axis))

    # rows of R_mhc are the basis vectors -> projection (world -> frame)
    R_mhc = np.vstack([x_axis, y_axis, z_axis])
    return R_mhc, com_mhc, pc1, pc2, n_mhc


# -------------------------
# Read pseudoatoms (CEN/PC1/PC2) from PDB
# -------------------------
def read_pseudo_points(pdb_path, chain_id_main):
    """
    Returns Points for the given chain from a consensus PDB that contains
    pseudoatoms on chain 'Z' with residue names: CEN, PC1, PC2.
    The main protein chain_id_main is used only to sanity-check presence;
    pseudoatoms are read from chain Z.
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("consensus", pdb_path)

    # Sanity: ensure the main chain exists
    if chain_id_main not in [ch.id for ch in s[0]]:
        raise ValueError(f"Chain '{chain_id_main}' not found in {pdb_path}")

    try:
        z = s[0]["Z"]
    except KeyError:
        raise ValueError(f"Chain 'Z' with pseudoatoms (CEN/PC1/PC2) not found in {pdb_path}")

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


# -------------------------
# Biotite alignment
# -------------------------
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
    """
    Align 'mobile' to 'static' using C-alpha atoms of a single protein chain.
    Saves the aligned full-atom mobile structure to output_pdb_file.
    Returns the Biotite transform used (AffineTransformation).
    """
    static_structure = btsio.load_structure(static_pdb_file, model=1)
    mobile_structure = btsio.load_structure(mobile_pdb_file, model=1)

    static_mask = (static_structure.atom_name == "CA") & (static_structure.chain_id == chain_name)
    mobile_mask = (mobile_structure.atom_name == "CA") & (mobile_structure.chain_id == chain_name)

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


# -------------------------
# PyMOL label-only helper (optional)
# -------------------------
def add_cgo_arrow(start, end, color, radius=0.3):
    return f"""[
        cgo.CYLINDER,{start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                     {end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                     {radius},
                     {color[0]},{color[1]},{color[2]},
                     {color[0]},{color[1]},{color[2]},
        cgo.CONE,{end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                 {start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                 {radius*1.5},0.0,
                 {color[0]},{color[1]},{color[2]},
                 {color[0]},{color[1]},{color[2]},1.0
    ]"""


# -------------------------
# Core processing (NO geometry modification of the input)
# -------------------------
def process(
    input_pdb,
    consA_with_pca,
    consB_with_pca,
    out_dir,
    vis_folder=None,
    A_consenus_res=None,
    B_consenus_res=None,
):
    """
    Measure intra-TCR and TCR–pMHC geometry on the ORIGINAL input.

      Intra-TCR:
        - BA, BC1, AC1, BC2, AC2, dc

      External TCR–MHC geometry in canonical MHC frame
      (COM(MHC)=0, n_MHC=+z, PC1_MHC=+x):

        Distances:
          - d_alpha_mhc : |COM(α) - COM(MHC)|
          - d_beta_mhc  : |COM(β) - COM(MHC)|
          - d_alpha_beta: |COM(β) - COM(α)|

        Directions encoded as spherical angles:
          - theta_rA,  phi_rA     : direction of MHC→α COM
          - theta_rB,  phi_rB     : direction of MHC→β COM
          - theta_pc1A, phi_pc1A  : direction of PC1_A
          - theta_pc1B, phi_pc1B  : direction of PC1_B

      QC / legacy (still computed, not used for reconstruction):
        - incident_angle, tilt_alpha_mhc, tilt_beta_mhc, dc_mhc_plane_angle
        - alpha_side_sign, beta_side_sign, dc_side_sign
    """
    parser = PDBParser(QUIET=True)

    os.makedirs(out_dir, exist_ok=True)
    aligned_input_path = os.path.join(out_dir, "aligned_input.pdb")
    aligned_consB_path = os.path.join(out_dir, "aligned_consB.pdb")

    # Align input (mobile) to consensus A (static) on chain A
    transform_input = align_with_biotite(
        static_pdb_file=consA_with_pca,
        mobile_pdb_file=input_pdb,
        output_pdb_file=aligned_input_path,
        chain_name="A",
        static_consenus_res=A_consenus_res,
        mobile_consenus_res=A_consenus_res,
    )

    # Align consensus B (mobile) to aligned input (static) on chain B
    _ = align_with_biotite(
        static_pdb_file=aligned_input_path,
        mobile_pdb_file=consB_with_pca,
        output_pdb_file=aligned_consB_path,
        chain_name="B",
        static_consenus_res=B_consenus_res,
        mobile_consenus_res=B_consenus_res,
    )

    # Read Points from pseudoatoms (in consensus-A / aligned frame)
    Apts_cons = read_pseudo_points(consA_with_pca, chain_id_main="A")
    Bpts_cons = read_pseudo_points(aligned_consB_path, chain_id_main="B")

    # ----------------- TCR internal geometry -----------------
    Cvec_cons = as_unit(Bpts_cons.C - Apts_cons.C)
    A1_cons = as_unit(Apts_cons.V1 - Apts_cons.C)
    A2_cons = as_unit(Apts_cons.V2 - Apts_cons.C)
    B1_cons = as_unit(Bpts_cons.V1 - Bpts_cons.C)
    B2_cons = as_unit(Bpts_cons.V2 - Bpts_cons.C)

    # BA (signed dihedral-like torsion) using plane-projected method
    nx = np.cross(A1_cons, Cvec_cons)
    ny = np.cross(Cvec_cons, nx)
    Lp = as_unit([0.0, np.dot(A1_cons, nx), np.dot(A1_cons, ny)])
    Hp = as_unit([0.0, np.dot(B1_cons, nx), np.dot(B1_cons, ny)])
    BA = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA = -BA

    BC1 = angle_between(B1_cons, -Cvec_cons)
    AC1 = angle_between(A1_cons, Cvec_cons)
    BC2 = angle_between(B2_cons, -Cvec_cons)
    AC2 = angle_between(A2_cons, Cvec_cons)
    dc = float(np.linalg.norm(Bpts_cons.C - Apts_cons.C))

    # ----------------- Map pseudoatoms back to ORIGINAL frame -----------------
    M = np.asarray(transform_input.as_matrix(), dtype=np.float64)
    if M.shape == (1, 4, 4):
        M = M[0]
    R = M[:3, :3]
    t = M[:3, 3]
    R_inv = R.T

    def to_orig(p_cons):
        p_cons = np.asarray(p_cons, dtype=float)
        return R_inv @ (p_cons - t)

    A_C_orig = to_orig(Apts_cons.C)
    A_V1_orig = to_orig(Apts_cons.V1)
    A_V2_orig = to_orig(Apts_cons.V2)
    B_C_orig = to_orig(Bpts_cons.C)
    B_V1_orig = to_orig(Bpts_cons.V1)
    B_V2_orig = to_orig(Bpts_cons.V2)

    # Points in ORIGINAL frame
    Apts_orig = Points(C=A_C_orig, V1=A_V1_orig, V2=A_V2_orig)
    Bpts_orig = Points(C=B_C_orig, V1=B_V1_orig, V2=B_V2_orig)

    # ----------------- TCR–pMHC geometry on ORIGINAL input -----------------
    orig_struct = parser.get_structure("input_orig", input_pdb)

    # MHC CA coords in original frame (chains M and optional N)
    coords_mhc = get_chain_ca_coords(orig_struct, ["M", "N"])

    # Canonical MHC frame (COM at origin, normal along +z, PC1 along +x)
    R_mhc, com_mhc, mhc_pc1, mhc_pc2, n_mhc_plane = mhc_canonical_frame(coords_mhc)

    # Use centroids from pseudoatoms as com_alpha and com_beta (your CENs)
    com_alpha = Apts_orig.C
    com_beta = Bpts_orig.C

    # TCR PCA axis directions in original frame
    A1_orig = as_unit(Apts_orig.V1 - Apts_orig.C)  # PC1_A
    B1_orig = as_unit(Bpts_orig.V1 - Bpts_orig.C)  # PC1_B

    # TCR plane from PC1 alpha and PC1 beta
    n_tcr_plane = as_unit(np.cross(A1_orig, B1_orig))

    # QC: TCR plane vs MHC plane
    incident_angle = angle_between_planes(n_tcr_plane, n_mhc_plane)

    # Distances COM(alpha/beta) → COM(MHC) (world frame)
    v_alpha_to_mhc = com_mhc - com_alpha   # MHC <- alpha
    v_beta_to_mhc = com_mhc - com_beta     # MHC <- beta
    d_alpha_mhc = float(np.linalg.norm(v_alpha_to_mhc))
    d_beta_mhc = float(np.linalg.norm(v_beta_to_mhc))
    d_alpha_beta = float(np.linalg.norm(com_beta - com_alpha))

    # Legacy plane-based angles (QC)
    dc_vec_orig = B_C_orig - A_C_orig  # A_C -> B_C
    tilt_alpha_mhc = angle_vec_plane(A1_orig, n_mhc_plane)
    tilt_beta_mhc = angle_vec_plane(B1_orig, n_mhc_plane)
    dc_mhc_plane_angle = angle_vec_plane(dc_vec_orig, n_mhc_plane)

    # Which side of the MHC plane the TCR sits on (QC)
    signed_alpha_side = float(np.sign(np.dot(com_alpha - com_mhc, n_mhc_plane)))
    signed_beta_side = float(np.sign(np.dot(com_beta - com_mhc, n_mhc_plane)))
    signed_dc_side = float(np.sign(np.dot(dc_vec_orig, n_mhc_plane)))

    # ----------------- Canonical MHC-frame representation -----------------
    # COM positions in MHC frame (MHC COM at origin)
    rA = R_mhc @ (com_alpha - com_mhc)
    rB = R_mhc @ (com_beta - com_mhc)

    # Directions of the MHC→α, MHC→β COM vectors
    # (these are just rA and rB, since COM(MHC) is at origin)
    theta_rA, phi_rA = to_spherical_angles(rA)
    theta_rB, phi_rB = to_spherical_angles(rB)

    # PC1_A and PC1_B direction in MHC frame
    pc1A_mhc = R_mhc @ A1_orig
    pc1B_mhc = R_mhc @ B1_orig
    theta_pc1A, phi_pc1A = to_spherical_angles(pc1A_mhc)
    theta_pc1B, phi_pc1B = to_spherical_angles(pc1B_mhc)

    # ----------------- Visualization outputs (optional) -----------------
    if vis_folder:
        vis_folder = Path(vis_folder)
        vis_folder.mkdir(exist_ok=True, parents=True)

        generate_pymol_script(
            input_pdb_path=os.path.abspath(input_pdb),
            Apts=Apts_orig,
            Bpts=Bpts_orig,
            com_mhc=com_mhc,
            n_tcr_plane=n_tcr_plane,
            n_mhc_plane=n_mhc_plane,
            d_alpha_mhc=d_alpha_mhc,
            d_beta_mhc=d_beta_mhc,
            d_alpha_beta=d_alpha_beta,
            incident_angle_deg=incident_angle,
            tilt_alpha_mhc=tilt_alpha_mhc,
            tilt_beta_mhc=tilt_beta_mhc,
            dc_mhc_plane_angle=dc_mhc_plane_angle,
            vis_folder=str(vis_folder),
        )

    return {
        "pdb_name": Path(input_pdb).stem,
        # internal TCR
        "BA": BA,
        "BC1": BC1,
        "AC1": AC1,
        "BC2": BC2,
        "AC2": AC2,
        "dc": dc,
        # external distances (MHC frame)
        "d_alpha_mhc": d_alpha_mhc,
        "d_beta_mhc": d_beta_mhc,
        "d_alpha_beta": d_alpha_beta,
        # spherical directions in canonical MHC frame
        "theta_rA": theta_rA,
        "phi_rA": phi_rA,
        "theta_rB": theta_rB,
        "phi_rB": phi_rB,
        "theta_pc1A": theta_pc1A,
        "phi_pc1A": phi_pc1A,
        "theta_pc1B": theta_pc1B,
        "phi_pc1B": phi_pc1B,
        # QC / legacy
        "tilt_alpha_mhc": tilt_alpha_mhc,
        "tilt_beta_mhc": tilt_beta_mhc,
        "dc_mhc_plane_angle": dc_mhc_plane_angle,
        "incident_angle": incident_angle,
        "alpha_side_sign": signed_alpha_side,
        "beta_side_sign": signed_beta_side,
        "dc_side_sign": signed_dc_side,
        "input_aligned": os.path.abspath(aligned_input_path),
    }


# -------------------------
# PyMOL visualization on ORIGINAL input frame (optional)
# -------------------------
def generate_pymol_script(
    input_pdb_path,
    Apts,
    Bpts,
    com_mhc,
    n_tcr_plane,
    n_mhc_plane,
    d_alpha_mhc,
    d_beta_mhc,
    d_alpha_beta,
    incident_angle_deg,
    tilt_alpha_mhc,
    tilt_beta_mhc,
    dc_mhc_plane_angle,
    vis_folder,
):
    pdb_name = Path(input_pdb_path).stem
    scale = 1.0  # scale for PCA pseudoatom endpoints

    # TCR pseudoatom geometry in ORIGINAL frame
    A_C = Apts.C.tolist()
    B_C = Bpts.C.tolist()
    a1_end = (Apts.C + scale * (Apts.V1 - Apts.C)).tolist()
    a2_end = (Apts.C + scale * (Apts.V2 - Apts.C)).tolist()
    b1_end = (Bpts.C + scale * (Bpts.V1 - Bpts.C)).tolist()
    b2_end = (Bpts.C + scale * (Bpts.V2 - Bpts.C)).tolist()

    # COMs
    com_mhc_list = com_mhc.tolist()
    com_alpha_list = A_C
    com_beta_list = B_C

    mid_tcr = 0.5 * (Apts.C + Bpts.C)
    mid_tcr_list = mid_tcr.tolist()
    norm_len = 25.0

    tcrnorm_end = (mid_tcr + norm_len * as_unit(n_tcr_plane)).tolist()
    mhcnorm_end = (com_mhc + norm_len * as_unit(n_mhc_plane)).tolist()

    # Simple label text
    label_text = (
        f"incident={incident_angle_deg:.1f}°\\n"
        f"dA_MHC={d_alpha_mhc:.1f}Å, dB_MHC={d_beta_mhc:.1f}Å, dAB={d_alpha_beta:.1f}Å\\n"
        f"(tiltA={tilt_alpha_mhc:.1f}°, tiltB={tilt_beta_mhc:.1f}°, dc_plane={dc_mhc_plane_angle:.1f}°)"
    )
    label_literal = repr(label_text)

    incident_label_pos = (mid_tcr + 1.3 * norm_len * as_unit(n_tcr_plane)).tolist()

    png_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.png")
    pse_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.pse")
    vis_script = os.path.join(vis_folder, "vis.py")

    script = f"""
import numpy as np
from pymol import cmd, cgo

cmd.load("{input_pdb_path}","complex_{pdb_name}")

cmd.bg_color("white")
cmd.hide("everything","all")

# Show full complex
cmd.show("cartoon","complex_{pdb_name}")

# Color TCR chains
cmd.color("marine","complex_{pdb_name} and chain A")
cmd.color("teal","complex_{pdb_name} and chain B")

# Color MHC + peptide
cmd.color("wheat","complex_{pdb_name} and chain M")
cmd.color("wheat","complex_{pdb_name} and chain N")
cmd.color("salmon","complex_{pdb_name} and chain C")

# Pseudoatoms for TCR centroids & PCA endpoints (original frame)
cmd.pseudoatom("centroid_A_{pdb_name}", pos={A_C}, color="red")
cmd.pseudoatom("centroid_B_{pdb_name}", pos={B_C}, color="orange")
cmd.pseudoatom("PCA_A1_{pdb_name}", pos={a1_end}, color="white")
cmd.pseudoatom("PCA_A2_{pdb_name}", pos={a2_end}, color="white")
cmd.pseudoatom("PCA_B1_{pdb_name}", pos={b1_end}, color="white")
cmd.pseudoatom("PCA_B2_{pdb_name}", pos={b2_end}, color="white")
cmd.show("spheres","centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")
cmd.set("sphere_scale", 0.6, "centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")

# CGO arrows: TCR PCA axes + dc vector (all in original frame)
cmd.load_cgo({add_cgo_arrow(A_C, a1_end, (0.2, 0.5, 1.0))}, "PC1_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, a2_end, (0.1, 0.8, 0.1))}, "PC2_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b1_end, (1.0, 0.5, 0.2))}, "PC1_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b2_end, (0.8, 0.8, 0.1))}, "PC2_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, B_C, (0.5, 0.0, 0.5))}, "dc_vec_{pdb_name}")

# Measurements for TCR internal geometry (in original frame)
cmd.distance("dc_len_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC1_ang_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC2_ang_{pdb_name}", "PCA_B2_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("AC1_ang_{pdb_name}", "PCA_A1_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.angle("AC2_ang_{pdb_name}", "PCA_A2_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.dihedral("BA_dih_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}", "PCA_A1_{pdb_name}")

cmd.enable("dc_len_{pdb_name}")
cmd.enable("BC1_ang_{pdb_name}")
cmd.enable("BC2_ang_{pdb_name}")
cmd.enable("AC1_ang_{pdb_name}")
cmd.enable("AC2_ang_{pdb_name}")
cmd.enable("BA_dih_{pdb_name}")

# Arrows: TCR plane normal and MHC plane normal (QC)
cmd.load_cgo({add_cgo_arrow(mid_tcr_list, tcrnorm_end, (1.0, 1.0, 0.0))}, "TCR_plane_normal_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(com_mhc_list, mhcnorm_end, (1.0, 0.0, 0.0))}, "MHC_plane_normal_{pdb_name}")

# COM pseudoatoms for QC
cmd.pseudoatom("COM_alpha_chain_{pdb_name}", pos={com_alpha_list}, color="cyan")
cmd.pseudoatom("COM_beta_chain_{pdb_name}",  pos={com_beta_list},  color="magenta")
cmd.pseudoatom("COM_MHC_{pdb_name}",         pos={com_mhc_list},   color="yellow")
cmd.show("spheres", "COM_alpha_chain_{pdb_name} or COM_beta_chain_{pdb_name} or COM_MHC_{pdb_name}")
cmd.set("sphere_scale", 0.8, "COM_alpha_chain_{pdb_name} or COM_beta_chain_{pdb_name} or COM_MHC_{pdb_name}")

# Distances alpha/beta COM -> MHC COM
cmd.distance("alpha_MHC_dist_{pdb_name}", "COM_alpha_chain_{pdb_name}", "COM_MHC_{pdb_name}")
cmd.distance("beta_MHC_dist_{pdb_name}",  "COM_beta_chain_{pdb_name}",  "COM_MHC_{pdb_name}")
cmd.distance("AB_dist_{pdb_name}",        "COM_alpha_chain_{pdb_name}", "COM_beta_chain_{pdb_name}")
cmd.enable("alpha_MHC_dist_{pdb_name}")
cmd.enable("beta_MHC_dist_{pdb_name}")
cmd.enable("AB_dist_{pdb_name}")

# Label for angles/distances
cmd.pseudoatom("incident_label_{pdb_name}", pos={incident_label_pos})
cmd.label("incident_label_{pdb_name}", {label_literal})
cmd.show("labels", "incident_label_{pdb_name}")

# Global style
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.2)
cmd.set("dash_round_ends", 0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)
cmd.set("label_angle_digits", 1)

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png(r"{png_path}", dpi=300, ray=1)
cmd.save(r"{pse_path}")
cmd.quit()
"""
    with open(vis_script, "w") as f:
        f.write(script)
    os.system(f"pymol -cq {vis_script}")
    print(f"✅ PyMOL script written & run: {vis_script}")


# -------------------------
# Public API
# -------------------------
def run(input_pdb_fv, out_path=None, vis=False, cleanup_tmp=True):
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")

    # read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    # output directory logic
    if out_path is None:
        out_dir = Path(tempfile.mkdtemp(prefix="tcr_geometry_"))
        is_tmp = True
    else:
        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        is_tmp = False

    vis_folder = None
    if vis:
        vis_folder = out_dir / "vis"
        vis_folder.mkdir(exist_ok=True)

    # main processing
    result = process(
        input_pdb=input_pdb_fv,
        consA_with_pca=consA_pca_path,
        consB_with_pca=consB_pca_path,
        out_dir=str(out_dir),
        vis_folder=str(vis_folder) if vis else None,
        A_consenus_res=A_consenus_res,
        B_consenus_res=B_consenus_res,
    )

    cols = [
        "pdb_name",
        # internal TCR geometry
        "BA", "BC1", "AC1", "BC2", "AC2", "dc",
        # external distances in MHC canonical frame
        "d_alpha_mhc", "d_beta_mhc", "d_alpha_beta",
        # spherical directions in MHC canonical frame
        "theta_rA", "phi_rA",
        "theta_rB", "phi_rB",
        "theta_pc1A", "phi_pc1A",
        "theta_pc1B", "phi_pc1B",
        # QC
        "tilt_alpha_mhc", "tilt_beta_mhc", "dc_mhc_plane_angle",
        "incident_angle", "alpha_side_sign", "beta_side_sign", "dc_side_sign",
    ]
    df = pd.DataFrame([result])[cols]

    if vis:
        print(f"🖼️  Figures/PSE written under: {vis_folder}")

    if is_tmp and cleanup_tmp:
        shutil.rmtree(out_dir, ignore_errors=True)

    return df
