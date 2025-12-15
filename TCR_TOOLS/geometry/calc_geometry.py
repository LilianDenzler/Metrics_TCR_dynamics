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
from Bio.PDB import PDBParser, PDBIO

# Biotite for structural alignment
import biotite.structure as bts
import biotite.structure.io as btsio

from . import DATA_PATH
from importlib.resources import files
from pathlib import Path

warnings.filterwarnings("ignore", ".*is discontinuous.*")
from pathlib import Path


# -------------------------
# Geometry helpers
# -------------------------
Points = namedtuple("Points", ["C", "V1", "V2"])  # endpoints (absolute coords)

def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def signed_angle(v, ref, normal):
    """
    Returns signed angle between v and ref, using `normal` to define orientation.
    """
    v = as_unit(v)
    ref = as_unit(ref)
    ang = math.degrees(math.acos(np.clip(np.dot(v, ref), -1.0, 1.0)))
    # sign comes from orientation relative to the normal
    sign = np.sign(np.dot(np.cross(ref, v), normal))
    return ang * sign


def angle_between(v1, v2):
    v1 = as_unit(v1); v2 = as_unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

# -------------------------
# Read pseudoatoms (CEN/PC1/PC2) from PDB
# Expectation: they live as residues named 'CEN','PC1','PC2' on chain 'Z'
# (This matches your change-geometry script behavior.)
# -------------------------
def read_pseudo_points(s, chain_id_main):
    """
    Returns Points for the given chain from a structure that contains
    pseudoatoms on chain 'Z' with residue names: CEN, PC1, PC2.

    Supports:
      - Biopython Structure
      - Biotite AtomArray

    `chain_id_main` is used only for a sanity check
    (does the main TCR chain exist?).
    """

    # ---------- Case 1: Biopython Structure ----------
    if hasattr(s, "get_models"):  # Biopython Structure-like
        model0 = next(s.get_models())
        chain_ids = [ch.id for ch in model0]

        if chain_id_main not in chain_ids:
            raise ValueError(f"Chain '{chain_id_main}' not found i")

        try:
            z = model0["Z"]
        except KeyError:
            raise ValueError(
                f"Chain 'Z' with pseudoatoms (CEN/PC1/PC2) not found"
            )

        def _get_first_atom(res_name):
            for res in z:
                if res.get_resname() == res_name:
                    for atom in res:
                        return atom.get_coord()
            raise ValueError(
                f"Pseudoatom '{res_name}' not found in chain Z "
            )

        C = _get_first_atom("CEN")
        V1 = _get_first_atom("PC1")
        V2 = _get_first_atom("PC2")

        return Points(
            C=np.array(C, float),
            V1=np.array(V1, float),
            V2=np.array(V2, float),
        )

    # ---------- Case 2: Biotite AtomArray ----------
    if isinstance(s, bts.AtomArray) or hasattr(s, "array_length"):
        arr = s

        # sanity check main chain
        unique_chains = np.unique(arr.chain_id)
        if chain_id_main not in unique_chains:
            raise ValueError(
                f"Chain '{chain_id_main}' not found in AtomArray "
            )

        # select pseudoatoms on chain 'Z'
        mask_Z = (arr.chain_id == "Z")
        if not np.any(mask_Z):
            raise ValueError(
                f"Chain 'Z' with pseudoatoms (CEN/PC1/PC2) not found in AtomArray "
            )

        arr_Z = arr[mask_Z]

        def _get_first_coord(res_name):
            mask = (arr_Z.res_name == res_name)
            if not np.any(mask):
                raise ValueError(
                    f"Pseudoatom '{res_name}' not found in chain Z of AtomArray"
                )
            return np.asarray(arr_Z[mask].coord[0], float)

        C  = _get_first_coord("CEN")
        V1 = _get_first_coord("PC1")
        V2 = _get_first_coord("PC2")

        return Points(C=C, V1=V1, V2=V2)

    # ---------- Fallback ----------
    raise TypeError(
        f"Unsupported structure type for read_pseudo_points(): {type(s)}"
    )

# -------------------------
# Biotite alignment
# -------------------------
def apply_affine_to_atomarray(atomarray, transform):
    arr = atomarray.copy()
    M = np.asarray(transform.as_matrix(), dtype=np.float64)
    if M.shape == (1, 4, 4):
        M = M[0]
    R = M[:3, :3]
    t = M[:3,  3]
    coords = np.asarray(arr.coord, dtype=np.float64)
    new_coords = (coords @ R.T) + t
    arr.coord = new_coords.astype(np.float32)
    return arr

def align_with_biotite(static, mobile, chain_name: str,static_consenus_res: list =None, mobile_consenus_res: list =None):
    """
    Align 'mobile' to 'static' using C-alpha atoms of a single protein chain.
    Saves the aligned full-atom mobile structure to output_pdb_file.
    Pseudoatoms (chain Z) ride along with the same transform if present.
    """
    static_mask = (static.atom_name == "CA") & (static.chain_id == chain_name)
    mobile_mask = (mobile.atom_name == "CA") & (mobile.chain_id == chain_name)
    static_ca = static[static_mask]
    mobile_ca = mobile[mobile_mask]
    if static_consenus_res:
        static_ca = static_ca[np.isin(static_ca.res_id, static_consenus_res)]
    if mobile_consenus_res:
        mobile_ca = mobile_ca[np.isin(mobile_ca.res_id, mobile_consenus_res)]
    if static_ca.array_length() < 4 or mobile_ca.array_length() < 4:
        raise ValueError(f"Not enough CA atoms to align on chain {chain_name}")

    _, transform, _, _ = bts.superimpose_structural_homologs(
        fixed=static_ca, mobile=mobile_ca, max_iterations=1
    )
    mobile_full_aligned = apply_affine_to_atomarray(mobile, transform)
    return mobile_full_aligned

# -------------------------
# CGO arrow helper (for PyMOL script text)
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
def get_tcr_points_world(
    input_pdb: str,
    out_dir: str,
    alpha_chain_id: str = "A",
    beta_chain_id: str = "B",
    vis_folder: str = None,
) -> tuple[Points, Points]:
    """
    Return Apts, Bpts as Points in the ORIGINAL complex frame.

    Steps:
      1) Align input to consensus A → aligned_input (same as TCR-only code).
      2) Align consensus B to aligned_input → aligned_consB.
      3) Read pseudoatoms from consensus A and aligned_consB (consensus frame).
      4) Use Biotite to compute the transform that maps aligned_input back
         to the original complex, and apply it to the pseudoatoms.
    """
    # --- consensus PDBs with PCA pseudoatoms ---
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")

    # --- consensus residue indices ---
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt")) as f:
        A_consenus_res = [int(x) for x in f.read().strip().split(",") if x.strip()]
    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt")) as f:
        B_consenus_res = [int(x) for x in f.read().strip().split(",") if x.strip()]


    # ---- 1) align input → consensus A on chain A (writes aligned_input_path) ----
    consA_pca_structure= btsio.load_structure(consA_pca_path, model=1)
    consB_pca_structure= btsio.load_structure(consB_pca_path, model=1)

    input_structure = btsio.load_structure(input_pdb, model=1)

    aligned_input_to_alpha=align_with_biotite(
        static=consA_pca_structure,
        mobile=input_structure,
        chain_name="A",
        static_consenus_res=A_consenus_res,
        mobile_consenus_res=A_consenus_res,
    )

    # ---- 2) align consensus B → aligned_input on chain B (writes aligned_consB_path) ----
    aligned_consB=align_with_biotite(
        static=aligned_input_to_alpha,
        mobile=consB_pca_structure,
        chain_name="B",
        static_consenus_res=B_consenus_res,
        mobile_consenus_res=B_consenus_res,
    )

    # ---- 3) read pseudoatom points in "consensus / aligned" frame ----
    A_cons = read_pseudo_points(consA_pca_structure,       chain_id_main="A")
    B_cons = read_pseudo_points(aligned_consB,   chain_id_main="B")

    # ---- 4) compute transform: aligned_input → original input (WORLD) ----
    orig_alpha    = input_structure[input_structure.chain_id    == alpha_chain_id]
    aligned_alpha = aligned_input_to_alpha[aligned_input_to_alpha.chain_id == alpha_chain_id]

    # This gives a transform that moves `mobile` (aligned_alpha) onto `fixed` (orig_alpha)
    _, transform, _, _ = bts.superimpose_structural_homologs(
        fixed=orig_alpha,
        mobile=aligned_alpha,
    )

    M = np.asarray(transform.as_matrix(), dtype=np.float64)
    if M.shape == (1, 4, 4):
        M = M[0]
    R = M[:3, :3]
    t = M[:3, 3]

    # coords_world = coords_aligned @ R.T + t
    def cons_to_world(p):
        p = np.asarray(p, float)
        return p @ R.T + t

    A_world = Points(
        C  = cons_to_world(A_cons.C),
        V1 = cons_to_world(A_cons.V1),
        V2 = cons_to_world(A_cons.V2),
    )
    B_world = Points(
        C  = cons_to_world(B_cons.C),
        V1 = cons_to_world(B_cons.V1),
        V2 = cons_to_world(B_cons.V2),
    )
    if vis_folder is not None:
        #transform the consB_pca_path protein structure
        consB_aligned_tcr = apply_affine_to_atomarray(consB_pca_structure, transform)
        consB_in_tcr_frame = os.path.join(vis_folder, "consB_aligned_for_points.pdb")
        btsio.save_structure(consB_in_tcr_frame, consB_aligned_tcr)
        #transform the consA_pca_path protein structure
        consA_aligned_tcr = apply_affine_to_atomarray(consA_pca_structure, transform)
        consA_in_tcr_frame = os.path.join(vis_folder, "consA_aligned_for_points.pdb")
        btsio.save_structure(consA_in_tcr_frame, consA_aligned_tcr)
        generate_pymol_script(
            input_aligned_viz=input_pdb,
            consA_pdb=consA_in_tcr_frame,
            consB_pdb=consB_in_tcr_frame,
            Apts=A_world,
            Bpts=B_world,
            vis_folder=vis_folder,
        )

    return A_world, B_world

def calc_tcr_geo_main(input_pdb, out_dir, alpha_chain_id="A", beta_chain_id="B", vis_folder=None):
    """
    1) Renumbered input is aligned to consensus A (chain A).
    2) Consensus B is aligned to the aligned input (chain B).
    3) Read pseudoatoms (CEN/PC1/PC2) from consensus A and aligned consensus B.
    4) Compute BA, BC1/2, AC1/2, dc from those points.
    5) Optionally write and run a PyMOL visualization script.
    """
    consA_with_pca = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_with_pca = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")

    # read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    Apts, Bpts = get_tcr_points_world(input_pdb=input_pdb,out_dir=out_dir,alpha_chain_id=alpha_chain_id,beta_chain_id=beta_chain_id, vis_folder=vis_folder)

    # Compute geometry
    Cvec = as_unit(Bpts.C - Apts.C)
    A1 = as_unit(Apts.V1 - Apts.C)
    A2 = as_unit(Apts.V2 - Apts.C)
    B1 = as_unit(Bpts.V1 - Bpts.C)
    B2 = as_unit(Bpts.V2 - Bpts.C)

    # BA (signed dihedral-like torsion) using plane-projected method
    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp = as_unit([0.0, np.dot(B1, nx), np.dot(B1, ny)])
    BA = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA = -BA

    # Define orientation normals for each domain
    nA = as_unit(np.cross(A1, Cvec))
    nB = as_unit(np.cross(B1, Cvec))

    AC1 = signed_angle(A1,  Cvec, nA)
    AC2 = signed_angle(A2,  Cvec, nA)
    BC1 = signed_angle(B1, -Cvec, nB)
    BC2 = signed_angle(B2, -Cvec, nB)

    # Canonicalize
    if AC1 > 0:
        AC1 = -AC1          # AC1 always negative

    # Optional canonicalization for BC2:
    if np.dot(B2, nA) > 0:  # or any other global convention you like
        BC2 = -BC2


    dc  = float(np.linalg.norm(Bpts.C - Apts.C))



    return {
        "BA": BA, "BC1": BC1, "AC1": AC1, "BC2": BC2, "AC2": AC2, "dc": dc
    }

# -------------------------
# PyMOL visualization (matches your previous style)
# -------------------------
def generate_pymol_script(input_aligned_viz, consA_pdb, consB_pdb, Apts, Bpts, vis_folder):
    pdb_name = Path(input_aligned_viz).stem
    scale = 1.0

    # Precompute lists for f-string insertion (avoid inline math in { ... } exprs)
    A_C = Apts.C.tolist()
    B_C = Bpts.C.tolist()
    a1_end = (Apts.C + scale * (Apts.V1 - Apts.C)).tolist()
    a2_end = (Apts.C + scale * (Apts.V2 - Apts.C)).tolist()
    b1_end = (Bpts.C + scale * (Bpts.V1 - Bpts.C)).tolist()
    b2_end = (Bpts.C + scale * (Bpts.V2 - Bpts.C)).tolist()

    png_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.png")
    pse_path = os.path.join(vis_folder, f"{pdb_name}_final_vis.pse")
    vis_script = os.path.join(vis_folder, "vis.py")

    script = f"""
import numpy as np
from pymol import cmd, cgo

cmd.load("{input_aligned_viz}","input_{pdb_name}")
cmd.load("{consA_pdb}","consA_{pdb_name}")
cmd.load("{consB_pdb}","consB_{pdb_name}")

cmd.bg_color("white")
cmd.hide("everything","all")

# Input TCR (aligned) colors
cmd.show("cartoon","input_{pdb_name}")
cmd.color("marine","input_{pdb_name} and chain A")
cmd.color("teal","input_{pdb_name} and chain B")

# Consensus overlays
cmd.show("cartoon","consA_{pdb_name} or consB_{pdb_name}")
cmd.color("gray70","consA_{pdb_name}")
cmd.color("gray70","consB_{pdb_name}")
cmd.set("cartoon_transparency", 0.5, "consA_{pdb_name} or consB_{pdb_name}")

# Pseudoatoms for centroids & scaled PC endpoints
cmd.pseudoatom("centroid_A_{pdb_name}", pos={A_C}, color="red")
cmd.pseudoatom("centroid_B_{pdb_name}", pos={B_C}, color="orange")
cmd.pseudoatom("PCA_A1_{pdb_name}", pos={a1_end}, color="white")
cmd.pseudoatom("PCA_A2_{pdb_name}", pos={a2_end}, color="white")
cmd.pseudoatom("PCA_B1_{pdb_name}", pos={b1_end}, color="white")
cmd.pseudoatom("PCA_B2_{pdb_name}", pos={b2_end}, color="white")
cmd.show("spheres","centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")
cmd.set("sphere_scale", 0.5, "centroid_A_{pdb_name} or centroid_B_{pdb_name} or PCA_A1_{pdb_name} or PCA_A2_{pdb_name} or PCA_B1_{pdb_name} or PCA_B2_{pdb_name}")

# CGO arrows: use precomputed endpoints
cmd.load_cgo({add_cgo_arrow(A_C, a1_end, (0.2, 0.5, 1.0))}, "PC1_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, a2_end, (0.1, 0.8, 0.1))}, "PC2_A_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b1_end, (1.0, 0.5, 0.2))}, "PC1_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(B_C, b2_end, (0.8, 0.8, 0.1))}, "PC2_B_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(A_C, B_C, (0.5, 0.0, 0.5))}, "dc_vec_{pdb_name}")

# Measurements (wizard-equivalent)
cmd.distance("dc_len_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC1_ang_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("BC2_ang_{pdb_name}", "PCA_B2_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}")
cmd.angle("AC1_ang_{pdb_name}", "PCA_A1_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.angle("AC2_ang_{pdb_name}", "PCA_A2_{pdb_name}", "centroid_A_{pdb_name}", "centroid_B_{pdb_name}")
cmd.dihedral("BA_dih_{pdb_name}", "PCA_B1_{pdb_name}", "centroid_B_{pdb_name}", "centroid_A_{pdb_name}", "PCA_A1_{pdb_name}")

# Ensure measurement objects are visible
cmd.enable("dc_len_{pdb_name}")
cmd.enable("BC1_ang_{pdb_name}")
cmd.enable("BC2_ang_{pdb_name}")
cmd.enable("AC1_ang_{pdb_name}")
cmd.enable("AC2_ang_{pdb_name}")
cmd.enable("BA_dih_{pdb_name}")

# Global style
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.2)
cmd.set("dash_round_ends", 0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)
cmd.set("label_angle_digits", 1)

# Chain labels on INPUT TCR only
def _centroid(selection):
    arr = cmd.get_coords(selection, state=1)
    if arr is None or len(arr) == 0:
        return None
    import numpy as _np
    return _np.mean(arr, axis=0)

alpha_sel = "input_{pdb_name} and chain A and polymer"
beta_sel  = "input_{pdb_name} and chain B and polymer"
alpha_pos = _centroid(alpha_sel)
beta_pos  = _centroid(beta_sel)
if alpha_pos is not None:
    cmd.pseudoatom("label_alpha_chain_{pdb_name}", pos=alpha_pos.tolist(), label="TCR α")
if beta_pos is not None:
    cmd.pseudoatom("label_beta_chain_{pdb_name}", pos=beta_pos.tolist(), label="TCR β")
cmd.hide("everything", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.show("labels", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_size", 20, "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_color", "black", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")
cmd.set("label_outline_color", "white", "label_alpha_chain_{pdb_name} or label_beta_chain_{pdb_name}")

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png(r"{png_path}", dpi=300, ray=1)
cmd.save(r"{pse_path}")
cmd.quit()
"""
    with open(vis_script, "w") as f:
        f.write(script)
    os.system(f"pymol -cq {vis_script}")
    print(f"✅ PyMOL script written: {vis_script}")

# -------------------------
# Public API (similar to your previous run() signature)
# -------------------------
def run(input_pdb_fv, out_path=None, vis=True, cleanup_tmp=True, alpha_chain_id="A", beta_chain_id="B"):

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
    result = calc_tcr_geo_main(
        input_pdb=input_pdb_fv,
        alpha_chain_id="A",
        beta_chain_id="B",
        out_dir=str(out_dir),
        vis_folder=str(vis_folder) if vis else None,
    )

    # Save CSV row
    df = pd.DataFrame([result])[["BA", "BC1", "AC1", "BC2", "AC2", "dc"]]

    if vis:
        print(f"🖼️  Figures/PSE written under: {vis_folder}")

    # ----------------- optional cleanup of tmp dir -----------------
    # Only delete if:
    #  - we created a temporary dir (out_path is None)
    #  - and caller allows cleanup
    if is_tmp and cleanup_tmp:
        shutil.rmtree(out_dir, ignore_errors=True)

    return result
