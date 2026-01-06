import os
import math
import shutil
import tempfile
import warnings
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from . import DATA_PATH
warnings.filterwarnings("ignore", ".*is discontinuous.*")
from TCR_TOOLS.core.io import write_pdb
from TCR_TOOLS.geometry.calc_geometry import get_tcr_points_world, calc_tcr_geo_main
# -------------------------
# Geometry helpers
# -------------------------
Points = namedtuple("Points", ["C", "V1", "V2"])  # centroid + PC1/PC2 endpoints


def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angle_between(v1, v2):
    v1 = as_unit(v1)
    v2 = as_unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def signed_angle(v, ref, normal):
    """
    Signed angle between v and ref, using `normal` to define orientation.
    Result in degrees, in (-180, 180].
    """
    v = as_unit(v)
    ref = as_unit(ref)
    normal = as_unit(normal)

    cosang = np.clip(np.dot(v, ref), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    # sign from orientation relative to normal
    sign = np.sign(np.dot(np.cross(ref, v), normal))
    if sign == 0:
        sign = 1.0
    return ang * sign


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

def pep_end_to_end_dir_from_structure(structure, pep_chain_id: str) -> np.ndarray:
    """
    Peptide N->C direction using CA atoms: last_CA - first_CA.
    Returns a unit vector in WORLD coordinates.
    """
    model = next(structure.get_models())
    pep_chain = None
    for ch in model:
        if ch.id == pep_chain_id:
            pep_chain = ch
            break
    if pep_chain is None:
        raise ValueError(f"Peptide chain '{pep_chain_id}' not found.")

    residues = [r for r in pep_chain if ("CA" in r)]
    residues.sort(key=lambda r: (r.id[1], r.id[2]))  # (resseq, icode)
    if len(residues) < 2:
        raise ValueError("Peptide has <2 CA atoms; cannot define N->C direction.")

    v = residues[-1]["CA"].coord - residues[0]["CA"].coord
    return as_unit(v)

# -------------------------
# MHC canonical frame
# -------------------------
def mhc_canonical_frame(coords_mhc, coords_pep=None, pep_dir_world=None):
    """
    Canonical MHC frame:
      - origin at COM(MHC)
      - +z points towards peptide side (using peptide COM)
      - +x aligned (in-plane) with peptide N->C direction projected into MHC plane
        (removes the remaining 180° rotation ambiguity around z)
    """
    _, vecs_mhc, com_mhc = pca_axes(coords_mhc)
    pc1 = vecs_mhc[:, 0]
    pc2 = vecs_mhc[:, 1]

    # plane normal (sign ambiguous until we fix it)
    n_mhc = as_unit(np.cross(pc1, pc2))

    # 1) Fix +z using peptide COM side (your existing logic)
    if coords_pep is not None:
        com_pep = center_of_mass(coords_pep)
        if np.dot(n_mhc, com_pep - com_mhc) < 0.0:
            n_mhc = -n_mhc
            pc1 = -pc1
            pc2 = -pc2

    # 2) Fix +x sign in-plane using peptide N->C direction projected onto plane
    #    This removes the 180° rotation around z that otherwise remains.
    if pep_dir_world is not None:
        # Project peptide direction into MHC plane
        pep_proj = pep_dir_world - np.dot(pep_dir_world, n_mhc) * n_mhc
        if np.linalg.norm(pep_proj) > 1e-6:
            pep_proj = as_unit(pep_proj)

            # Ensure pc1 points along pep_proj (not opposite)
            # If opposite, flip pc1 and pc2 to preserve right-handedness.
            if np.dot(pc1, pep_proj) < 0.0:
                pc1 = -pc1
                pc2 = -pc2

    # Recompute normal to be consistent with final pc1/pc2
    n_mhc = as_unit(np.cross(pc1, pc2))

    # Build orthonormal basis (x in plane, z normal)
    x_axis = as_unit(pc1 - np.dot(pc1, n_mhc) * n_mhc)
    z_axis = n_mhc
    y_axis = as_unit(np.cross(z_axis, x_axis))

    R_mhc = np.vstack([x_axis, y_axis, z_axis])
    return R_mhc, com_mhc, pc1, pc2, n_mhc

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
def process(tcrpmhc_complex, out_dir, vis_folder=None,
            alpha_chain_id="A", beta_chain_id="B",
            mhc_chain_ids=("M", "N"), pep_chain_id="C"):
    """
    Measure intra-TCR and TCR–pMHC geometry on the ORIGINAL input.

      Intra-TCR (internal, from centroid+PCA framework geometry):
        - BA, BC1, AC1, BC2, AC2, dc   (signed AC/BC, like TCR calc_angles)

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
    pdb_full_renumbered=os.path.join(out_dir, "full_renumbered.pdb")
    write_pdb(pdb_full_renumbered,tcrpmhc_complex.imgt_all_structure)

    angles_dict_tcr=calc_tcr_geo_main(pdb_full_renumbered, out_dir, alpha_chain_id=alpha_chain_id,beta_chain_id=beta_chain_id, vis_folder=None)
    AC1=angles_dict_tcr["AC1"]
    AC2=angles_dict_tcr["AC2"]
    BC1=angles_dict_tcr["BC1"]
    BC2=angles_dict_tcr["BC2"]
    BA=angles_dict_tcr["BA"]
    dc=angles_dict_tcr["dc"]
    print(AC1, AC2, BC1, BC2, BA, dc)
    Apts_orig, Bpts_orig,alpha_cdr3, beta_cdr3 = get_tcr_points_world(pdb_full_renumbered, out_dir, alpha_chain_id=alpha_chain_id,beta_chain_id=beta_chain_id)

    # ----------------- TCR–pMHC geometry on ORIGINAL input -----------------
    # MHC CA coords in original frame (chains M / N)
    coords_mhc = get_chain_ca_coords(tcrpmhc_complex.imgt_all_structure, list(mhc_chain_ids))
    coords_pep = get_chain_ca_coords(tcrpmhc_complex.imgt_all_structure, [pep_chain_id])
    # Canonical MHC frame (COM at origin, normal along +z, PC1 along +x)
    pep_dir_world = pep_end_to_end_dir_from_structure(tcrpmhc_complex.imgt_all_structure, pep_chain_id)
    R_mhc, com_mhc, pc1, pc2, n_mhc_plane = mhc_canonical_frame(coords_mhc, coords_pep, pep_dir_world=pep_dir_world)




    # Use PCA centroids as COMs
    com_alpha = Apts_orig.C
    com_beta = Bpts_orig.C

    # TCR PCA axis directions in original frame
    A1_orig = Apts_orig.V1 - Apts_orig.C  # PC1_A
    B1_orig = Bpts_orig.V1 - Bpts_orig.C   # PC1_B

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
    dc_vec_orig = Bpts_orig.C - Apts_orig.C  # A_C -> B_C
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
            input_pdb_path=pdb_full_renumbered,
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
        print(f"Visualization saved to {vis_folder}")

    return {
        # internal TCR (signed angles)
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
        "alpha_cdr3_bend_deg": alpha_cdr3["bend_angle_deg"],
        "alpha_cdr3_apex_height_A": alpha_cdr3["apex_height_A"],
        #"alpha_cdr3_anchor_pre": alpha_cdr3["anchor_pre_resi"],
        #"alpha_cdr3_anchor_post": alpha_cdr3["anchor_post_resi"],
        "alpha_cdr3_apex_resi": alpha_cdr3["apex_resi"],
        "beta_cdr3_bend_deg": beta_cdr3["bend_angle_deg"],
        "beta_cdr3_apex_height_A": beta_cdr3["apex_height_A"],
        #"beta_cdr3_anchor_pre": beta_cdr3["anchor_pre_resi"],
        #"beta_cdr3_anchor_post": beta_cdr3["anchor_post_resi"],
        "beta_cdr3_apex_resi": beta_cdr3["apex_resi"],
        # QC / legacy
        #"tilt_alpha_mhc": tilt_alpha_mhc,
        #"tilt_beta_mhc": tilt_beta_mhc,
        #"dc_mhc_plane_angle": dc_mhc_plane_angle,
        #"incident_angle": incident_angle,
        #"alpha_side_sign": signed_alpha_side,
        #"beta_side_sign": signed_beta_side,
        #"dc_side_sign": signed_dc_side
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
    from pathlib import Path as _Path

    pdb_name = _Path(input_pdb_path).stem
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
    label_text = (""
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

# Color MHC + peptide (assuming M/N = MHC, C = peptide)
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
def run(tcrpmhc_complex, out_path=None, vis=False, cleanup_tmp=True,
        alpha_chain_id="A", beta_chain_id="B", mhc_chain_ids=("M", "N"), pep_chain_id="C"):
    # output directory logic
    if out_path is None:
        out_dir = Path(tempfile.mkdtemp(prefix="tcr_pmhc_geometry_"))
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
        tcrpmhc_complex=tcrpmhc_complex,
        out_dir=str(out_dir),
        vis_folder=str(vis_folder) if vis else None,
        alpha_chain_id=alpha_chain_id,
        beta_chain_id=beta_chain_id,
        mhc_chain_ids=mhc_chain_ids,
        pep_chain_id=pep_chain_id,
    )

    if vis:
        print(f"🖼️  Figures/PSE written under: {vis_folder}")

    if is_tmp and cleanup_tmp:
        shutil.rmtree(out_dir, ignore_errors=True)

    return result
