import os
import numpy as np
import warnings
from pathlib import Path
import math
import json
import tempfile
import biotite.structure as bts
import biotite.structure.io as btsio
from Bio.PDB import PDBParser, PDBIO, Select
from . import DATA_PATH
# Suppress PDB parsing warnings for cleaner output
warnings.filterwarnings("ignore", ".*is discontinuous.*")
from collections import namedtuple
Points = namedtuple("Points", ["C", "V1", "V2"])  # centroid + PC1 + PC2
from TCR_TOOLS.geometry.calc_geometry import get_tcr_points_world, as_unit, angle_between,signed_angle
from TCR_TOOLS.core.io import write_pdb
from pathlib import Path


# ========================
# Geometry and Math Helpers
# ========================
def normalize(v, length=1.0):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return (v / n) * length

def rotate_around_axis(v, axis, angle):
    """
    Rotate vector v around 'axis' (unit or non-unit) by 'angle' (radians)
    using Rodrigues' rotation formula.
    """
    v = np.asarray(v, float)
    axis = as_unit(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    return (
        v * c
        + np.cross(axis, v) * s
        + axis * np.dot(axis, v) * (1.0 - c)
    )

def build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc):

    # ---- 0) degrees → radians ----
    ba   = np.deg2rad(BA)
    bc1  = np.deg2rad(BC1)
    bc2  = np.deg2rad(BC2)
    ac1  = np.deg2rad(AC1)
    ac2  = np.deg2rad(AC2)

    # ---- 1) centroids and Cvec ----
    A_C = np.array([0.0, 0.0, 0.0], dtype=float)
    B_C = np.array([dc, 0.0, 0.0], dtype=float)
    Cvec = as_unit(B_C - A_C)   # this will be +x

    # ============================================================
    # A DOMAIN: A1 from AC1, then A2 from AC2 with A1 ⟂ A2
    # ============================================================

    # AC1 in the calc script is the signed angle between A1 and Cvec
    # with normal nA = A1 x Cvec. One can show that
    #   AC1 = - angle(A1, Cvec)
    # So the unsigned tilt is |AC1|, and the sign just picks a
    # direction in the y–z plane. A1 we put in the x–y plane.
    thetaA = -ac1  # unsigned tilt = |ac1|, sign handled by minus
    A1 = np.array([np.cos(thetaA), np.sin(thetaA), 0.0], dtype=float)
    A1 = as_unit(A1)

    # local basis for A: u along Cvec, v in plane of (A1,u), w = u × v
    u = Cvec
    v_raw = A1 - np.dot(A1, u) * u
    if np.linalg.norm(v_raw) < 1e-8:
        v_raw = np.array([0.0, 1.0, 0.0])
    v = as_unit(v_raw)
    w = as_unit(np.cross(u, v))

    # write A1 in (u,v): A1 = cos(phi1) u + sin(phi1) v
    cos_phi1 = np.dot(A1, u)
    sin_phi1 = np.sqrt(max(0.0, 1.0 - cos_phi1**2))

    # A2 = α u + β v + γ w
    # constraints:
    #   1) A2·u       = cos(|AC2|)
    #   2) A1·A2      = 0
    #   3) ||A2|| = 1
    alpha = np.cos(abs(ac2))          # angle to Cvec is |AC2|
    if sin_phi1 < 1e-6:
        # degenerate case: A1 ∥ Cvec
        beta = 0.0
    else:
        # from A1·A2 = 0:
        #   (cos_phi1) α + (sin_phi1) β = 0  →  β = -cos_phi1 α / sin_phi1
        beta = -cos_phi1 * alpha / sin_phi1

    tmp = 1.0 - alpha*alpha - beta*beta
    if tmp < 0.0:
        tmp = 0.0
    # sign of AC2 chooses which side of the A-domain plane (via w)
    gamma = np.sign(AC2) * np.sqrt(tmp)

    A2 = alpha*u + beta*v + gamma*w
    A2 = as_unit(A2)   # numerical clean-up only (should already be unit)

    # ============================================================
    # B DOMAIN: B1 from BC1 + BA, then B2 from BC2 with B1 ⟂ B2
    # ============================================================
    ba   = np.deg2rad(BA)
    bc1  = np.deg2rad(BC1)
    bc2  = np.deg2rad(BC2)

    # 1) B1_init with correct *unsigned* tilt BC1 to -Cvec
    thetaB = abs(bc1)
    B1_init = np.array([-np.cos(thetaB), np.sin(thetaB), 0.0], dtype=float)
    B1_init = as_unit(B1_init)

    # 2) adjust around Cvec so BA matches (same as calc)
    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp_init = as_unit([0.0, np.dot(B1_init, nx), np.dot(B1_init, ny)])
    BA_init = angle_between(Lp, Hp_init)
    if np.cross(Lp, Hp_init)[0] < 0:
        BA_init = -BA_init
    BA_init_rad = np.deg2rad(BA_init)

    delta = ba - BA_init_rad

    def rotate_around_axis(v, axis, angle):
        v = np.asarray(v, float)
        axis = as_unit(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        return (
            v * c
            + np.cross(axis, v) * s
            + axis * np.dot(axis, v) * (1.0 - c)
        )

    B1 = rotate_around_axis(B1_init, Cvec, delta)
    B1 = as_unit(B1)

    # 3) Build a candidate B2 from BC2 magnitude, perpendicular to B1
    u = Cvec
    v2_raw = B1 - np.dot(B1, u) * u
    if np.linalg.norm(v2_raw) < 1e-8:
        v2_raw = np.array([0.0, 1.0, 0.0])
    v2 = as_unit(v2_raw)
    w2 = as_unit(np.cross(u, v2))

    b_u = np.dot(B1, u)
    b_v = np.dot(B1, v2)

    # angle(B2, -u) = |BC2|
    alpha_b = -np.cos(abs(bc2))
    beta_b  = 0.0 if abs(b_v) < 1e-6 else -b_u * alpha_b / b_v

    tmp2 = 1.0 - alpha_b*alpha_b - beta_b*beta_b
    if tmp2 < 0.0:
        tmp2 = 0.0

    # start with one sign (your original choice)
    gamma_b = np.sign(BC2) * np.sqrt(tmp2)
    B2 = alpha_b*u + beta_b*v2 + gamma_b*w2
    B2 = as_unit(B2)

    # --------------------------------------------------------
    # NEW: enforce BC2 consistency with calc_tcr_geo_main
    # --------------------------------------------------------
    def canon_BC2_for(B2_vec):
        # exactly replicate calc code:
        nA_loc = as_unit(np.cross(A1, Cvec))
        nB_loc = as_unit(np.cross(B1, Cvec))

        BC2_raw = signed_angle(B2_vec, -Cvec, nB_loc)
        BC2_can = BC2_raw
        if np.dot(B2_vec, nA_loc) > 0:
            BC2_can = -BC2_can
        return BC2_can, nA_loc

    BC2_can, nA_loc = canon_BC2_for(B2)

    if abs(BC2_can - BC2) > 1e-3:
        # reflect B2 across the plane with normal nA (A-domain plane)
        B2_reflect = B2 - 2.0 * np.dot(B2, nA_loc) * nA_loc
        B2_reflect = as_unit(B2_reflect)
        BC2_can_reflect, _ = canon_BC2_for(B2_reflect)

        # pick whichever matches BC2 better
        if abs(BC2_can_reflect - BC2) < abs(BC2_can - BC2):
            B2 = B2_reflect
    # --------------------------------------------------------

    # Endpoints
    A_V1 = A_C + A1
    A_V2 = A_C + A2
    B_V1 = B_C + B1
    B_V2 = B_C + B2

    return A_C, A_V1, A_V2, B_C, B_V1, B_V2


def build_geometry_from_angles_old(BA, BC1, BC2, AC1, AC2, dc):
    """
    Reconstruct canonical geometry from angles produced by the calc script.

    Conventions matched to calc_geometry.process():
      - Cvec = unit(B_C - A_C)
      - A1 = unit(A_V1 - A_C), A2 = unit(A_V2 - A_C)
      - B1 = unit(B_V1 - B_C), B2 = unit(B_V2 - B_C)
      - BA from projected Lp/Hp as in calc script
      - AC1, AC2, BC1, BC2 are *signed* angles as in the newer calc script
      - A1 ⟂ A2, B1 ⟂ B2
    """

    # ---- 0) degrees → radians ----
    ba   = np.deg2rad(BA)
    bc1  = np.deg2rad(BC1)
    bc2  = np.deg2rad(BC2)
    ac1  = np.deg2rad(AC1)
    ac2  = np.deg2rad(AC2)

    # ---- 1) centroids and Cvec ----
    A_C = np.array([0.0, 0.0, 0.0], dtype=float)
    B_C = np.array([dc, 0.0, 0.0], dtype=float)
    Cvec = as_unit(B_C - A_C)   # this will be +x

    # ============================================================
    # A DOMAIN: A1 from AC1, then A2 from AC2 with A1 ⟂ A2
    # ============================================================

    # AC1 in the calc script is the signed angle between A1 and Cvec
    # with normal nA = A1 x Cvec. One can show that
    #   AC1 = - angle(A1, Cvec)
    # So the unsigned tilt is |AC1|, and the sign just picks a
    # direction in the y–z plane. A1 we put in the x–y plane.
    thetaA = -ac1  # unsigned tilt = |ac1|, sign handled by minus
    A1 = np.array([np.cos(thetaA), np.sin(thetaA), 0.0], dtype=float)
    A1 = as_unit(A1)

    # local basis for A: u along Cvec, v in plane of (A1,u), w = u × v
    u = Cvec
    v_raw = A1 - np.dot(A1, u) * u
    if np.linalg.norm(v_raw) < 1e-8:
        v_raw = np.array([0.0, 1.0, 0.0])
    v = as_unit(v_raw)
    w = as_unit(np.cross(u, v))

    # write A1 in (u,v): A1 = cos(phi1) u + sin(phi1) v
    cos_phi1 = np.dot(A1, u)
    sin_phi1 = np.sqrt(max(0.0, 1.0 - cos_phi1**2))

    # A2 = α u + β v + γ w
    # constraints:
    #   1) A2·u       = cos(|AC2|)
    #   2) A1·A2      = 0
    #   3) ||A2|| = 1
    alpha = np.cos(abs(ac2))          # angle to Cvec is |AC2|
    if sin_phi1 < 1e-6:
        # degenerate case: A1 ∥ Cvec
        beta = 0.0
    else:
        # from A1·A2 = 0:
        #   (cos_phi1) α + (sin_phi1) β = 0  →  β = -cos_phi1 α / sin_phi1
        beta = -cos_phi1 * alpha / sin_phi1

    tmp = 1.0 - alpha*alpha - beta*beta
    if tmp < 0.0:
        tmp = 0.0
    # sign of AC2 chooses which side of the A-domain plane (via w)
    gamma = np.sign(AC2) * np.sqrt(tmp)

    A2 = alpha*u + beta*v + gamma*w
    A2 = as_unit(A2)   # numerical clean-up only (should already be unit)

    # ============================================================
    # B DOMAIN: B1 from BC1 + BA, then B2 from BC2 with B1 ⟂ B2
    # ============================================================

    # First choose B1_init with correct *unsigned* tilt BC1 to -Cvec
    thetaB = abs(bc1)
    B1_init = np.array([-np.cos(thetaB), np.sin(thetaB), 0.0], dtype=float)
    B1_init = as_unit(B1_init)

    # Compute BA_init for this choice, using the SAME formula as calc script
    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp_init = as_unit([0.0, np.dot(B1_init, nx), np.dot(B1_init, ny)])
    BA_init = angle_between(Lp, Hp_init)          # degrees
    BA_init = np.deg2rad(BA_init)                 # radians
    if np.cross(Lp, Hp_init)[0] < 0:
        BA_init = -BA_init

    # rotate around Cvec to make BA match the target BA
    delta = ba - BA_init
    B1 = rotate_around_axis(B1_init, Cvec, delta)
    B1 = as_unit(B1)

    # local basis for B: u along Cvec, v2 in plane of (B1,u), w2 = u × v2
    u = Cvec
    v2_raw = B1 - np.dot(B1, u) * u
    if np.linalg.norm(v2_raw) < 1e-8:
        v2_raw = np.array([0.0, 1.0, 0.0])
    v2 = as_unit(v2_raw)
    w2 = as_unit(np.cross(u, v2))

    # express B1 in (u,v2)
    b_u = np.dot(B1, u)
    b_v = np.dot(B1, v2)

    # B2 = α_b u + β_b v2 + γ_b w2
    # constraints:
    #   1) angle(B2, -u) = |BC2| → B2·(-u) = cos(|BC2|)
    #      → α_b = -cos(|BC2|)
    #   2) B1·B2 = 0 → b_u α_b + b_v β_b = 0
    #   3) ||B2|| = 1
    alpha_b = -np.cos(abs(bc2))

    if abs(b_v) < 1e-6:
        beta_b = 0.0
    else:
        beta_b = -b_u * alpha_b / b_v

    tmp2 = 1.0 - alpha_b*alpha_b - beta_b*beta_b
    if tmp2 < 0.0:
        tmp2 = 0.0
    # sign of BC2 picks which side of B-domain plane
    gamma_b = np.sign(BC2) * np.sqrt(tmp2) #negative as we did convention in calc

    B2 = alpha_b*u + beta_b*v2 + gamma_b*w2
    B2 = as_unit(B2)

    # ============================================================
    # Endpoints
    # ============================================================
    A_V1 = A_C + A1
    A_V2 = A_C + A2
    B_V1 = B_C + B1
    B_V2 = B_C + B2

    return A_C, A_V1, A_V2, B_C, B_V1, B_V2


def apply_transformation(coords, R, t):
    """Applies rotation (R) and translation (t) to a set of coordinates."""
    return (coords @ R.T) + t



def frame_from_points(pts):
    origin = np.asarray(pts.C, float)
    e1 = as_unit(pts.V1 - pts.C)
    e2 = as_unit(pts.V2 - pts.C)
    e3 = np.cross(e1, e2)
    n3 = np.linalg.norm(e3)
    if n3 < 1e-8:
        e2 = as_unit(e2 + np.array([0.0, 0.0, 1e-3]))
        e3 = np.cross(e1, e2)
    e3 = as_unit(e3)
    e2 = as_unit(np.cross(e3, e1))
    R = np.column_stack([e1, e2, e3])
    return origin, R

def rigid_transform_from_frames(old_pts, new_pts):
    old_origin, R_old = frame_from_points(old_pts)
    new_origin, R_new = frame_from_points(new_pts)
    R = R_new @ R_old.T
    t = new_origin - R @ old_origin
    return R, t
def add_cgo_arrow(start, end, color, radius=0.3):
    """Generates a CGO string for a PyMOL arrow object."""
    # This function is unchanged but included for completeness
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

def generate_pymol_script(aligned_pdb, A_pts, B_pts, out_prefix, vis_folder):
    """
    Generates a PyMOL script to visualize the final alignment with longer PCA
    axes and visible centroids.
    """
    pdb_name = Path(aligned_pdb).stem

    # --- FIX: Increased scale factor for longer PCA axes ---
    scale = 10.0
    A_V1=A_pts.V1
    A_V2=A_pts.V2
    B_V1=B_pts.V1
    B_V2=B_pts.V2
    A_C=A_pts.C
    B_C=B_pts.C

    # Calculate endpoints for the CGO arrows
    a1_end = A_C + scale * (A_V1 - A_C)
    a2_end = A_C + scale * (A_V2 - A_C)
    b1_end = B_C + scale * (B_V1 - B_C)
    b2_end = B_C + scale * (B_V2 - B_C)

    script = f"""
import numpy as np
from pymol import cmd, cgo
cmd.load("{aligned_pdb}","aligned_{pdb_name}")
cmd.bg_color("white")
cmd.hide("everything","all")

cmd.show("cartoon","aligned_{pdb_name}")
# --- FIX: Using standard PyMOL color names ---
cmd.color("marine","aligned_{pdb_name} and chain A")
cmd.color("teal","aligned_{pdb_name} and chain B")



# --- FIX: Explicitly create and show spheres for the centroids ---
cmd.pseudoatom("centroid_A", pos={list(A_C)}, color="red")
cmd.pseudoatom("centroid_B", pos={list(B_C)}, color="orange")
cmd.pseudoatom("PCA_A1", pos={list(a1_end)}, color="white")
cmd.pseudoatom("PCA_A2", pos={list(a2_end)}, color="white")
cmd.pseudoatom("PCA_B1", pos={list(b1_end)}, color="white")
cmd.pseudoatom("PCA_B2", pos={list(b2_end)}, color="white")
cmd.show("spheres", "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")
cmd.set("sphere_scale", 0.5, "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")

# Load CGO arrows representing the scaled PCA axes
cmd.load_cgo({add_cgo_arrow(A_C, a1_end, (0.2, 0.5, 1.0))}, "PC1_A")
cmd.load_cgo({add_cgo_arrow(A_C, a2_end, (0.1, 0.8, 0.1))}, "PC2_A")
cmd.load_cgo({add_cgo_arrow(B_C, b1_end, (1.0, 0.5, 0.2))}, "PC1_B")
cmd.load_cgo({add_cgo_arrow(B_C, b2_end, (0.8, 0.8, 0.1))}, "PC2_B")
cmd.load_cgo({add_cgo_arrow(A_C, B_C, (0.5,0.0,0.5))},"dc")


# --- measurements (wizard-equivalent) ---
cmd.distance("dc_len", "centroid_B", "centroid_A")                 # distance dc
cmd.angle("BC1_ang", "PCA_B1", "centroid_B", "centroid_A")         # angle BC1
cmd.angle("BC2_ang", "PCA_B2", "centroid_B", "centroid_A")         # angle BC2
cmd.angle("AC1_ang", "PCA_A1", "centroid_A", "centroid_B")         # angle AC1
cmd.angle("AC2_ang", "PCA_A2", "centroid_A", "centroid_B")         # angle AC2
cmd.dihedral("BA_dih", "PCA_B1", "centroid_B", "centroid_A", "PCA_A1")  # dihedral BA

# Ensure measurement objects are visible
cmd.enable("dc_len")
cmd.enable("BC1_ang")
cmd.enable("BC2_ang")
cmd.enable("AC1_ang")
cmd.enable("AC2_ang")
cmd.enable("BA_dih")

# Global styling for measurement dashes & labels (applies to all three)
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)  # for distances
cmd.set("label_angle_digits", 1)     # for angles/dihedrals

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png("{os.path.join(vis_folder, out_prefix + "_final_vis.png")}", dpi=300, ray=1)
cmd.save("{os.path.join(vis_folder, out_prefix + "_final_vis.pse")}")
cmd.quit()
"""
    vis_script_path = os.path.join(vis_folder, out_prefix + "_final_vis.py")
    with open(vis_script_path, "w") as f:
        f.write(script)
    os.system(f"pymol -cq {vis_script_path}")
    print(f"✅ PyMOL script written: {vis_script_path}")
    return vis_script_path




def run(input_pdb, BA, BC1, BC2, AC1, AC2, dc, out_path=None, vis=True, cleanup_tmp=True, alpha_chain_id="A", beta_chain_id="B"):


    # ----------------- output directory logic -----------------
    if out_path is None:
        # make a tmp folder for output
        out_dir = Path(tempfile.mkdtemp(prefix="tcr_geometry_"))
        is_tmp = True
    else:
        out_dir = Path(out_path)
        out_dir.mkdir(exist_ok=True, parents=True)
        is_tmp = False

    # vis subfolder (only if requested)
    vis_folder = None
    if vis:
        vis_folder = out_dir / "vis"
        vis_folder.mkdir(exist_ok=True)

    # ----------------- main processing -----------------
    A_old, B_old = get_tcr_points_world(input_pdb, out_dir, alpha_chain_id=alpha_chain_id,beta_chain_id=beta_chain_id)

    # Build target geometry
    A_C, A_V1, A_V2, B_C, B_V1, B_V2 = build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc)
    A_target = Points(C=A_C, V1=A_V1, V2=A_V2)
    B_target = Points(C=B_C, V1=B_V1, V2=B_V2)

    # Domain-wise transforms to go from old (consensus frame) → target geometry
    R_A, t_A = rigid_transform_from_frames(A_old, A_target)
    R_B, t_B = rigid_transform_from_frames(B_old, B_target)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input_structure", input_pdb)


    pdb_name = Path(input_pdb).stem
    out_pdb = os.path.join(out_dir, f"{pdb_name}_geometry_changed_aligned.pdb")

    for model in structure:
        for chain in model:
            cid = chain.id
            if cid not in (alpha_chain_id, beta_chain_id):
                continue
            R, t = (R_A, t_A) if cid == alpha_chain_id else (R_B, t_B)
            for residue in chain:
                for atom in residue:
                    coord = np.asarray(atom.get_coord(), float)
                    new_coord = R @ coord + t
                    atom.set_coord(new_coord)
    write_pdb(out_pdb, structure)

    if vis:
        vis_folder = str(vis_folder)
        generate_pymol_script(out_pdb, A_target, B_target, out_prefix="vis", vis_folder=vis_folder)

    original_structure = btsio.load_structure(input_pdb, model=1)
    final_aligned_structure = btsio.load_structure(out_pdb, model=1)

    original_alpha = original_structure[original_structure.chain_id == alpha_chain_id]
    final_alpha    = final_aligned_structure[final_aligned_structure.chain_id == alpha_chain_id]

    _, transform, _, _ = bts.superimpose_structural_homologs(
        fixed=original_alpha,
        mobile=final_alpha
    )

    M = np.asarray(transform.as_matrix(), dtype=np.float64)[0]
    R, t = M[:3, :3], M[:3, 3]

    final_aligned_structure.coord = (final_aligned_structure.coord @ R.T) + t

    final_out = os.path.join(out_dir, f"{pdb_name}_geometry_changed_original_frame.pdb")
    btsio.save_structure(final_out, final_aligned_structure)

    return final_out, final_aligned_structure
