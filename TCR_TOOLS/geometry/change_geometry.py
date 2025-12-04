import os
import numpy as np
import warnings
from pathlib import Path
import math
import json
import tempfile
import biotite.structure as bts
import biotite.structure.io as btsio
from . import DATA_PATH
# Suppress PDB parsing warnings for cleaner output
warnings.filterwarnings("ignore", ".*is discontinuous.*")


from pathlib import Path

def verify_built_geometry(A_C, A_V1, A_V2, B_C, B_V1, B_V2, BA, BC1, BC2, AC1, AC2, dc):
    # 3) Plug into your calc-style geometry to see if you recover the same values
    from math import degrees

    def as_unit(v):
        v = np.asarray(v, float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def angle_between(v1, v2):
        v1 = as_unit(v1); v2 = as_unit(v2)
        return degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    A1 = as_unit(A_V1 - A_C)
    A2 = as_unit(A_V2 - A_C)
    B1 = as_unit(B_V1 - B_C)
    B2 = as_unit(B_V2 - B_C)
    Cvec = as_unit(B_C - A_C)

    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
    Hp = as_unit([0.0, np.dot(B1, nx), np.dot(B1, ny)])
    BA_calc = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA_calc = -BA_calc

    BC1_calc = angle_between(B1, -Cvec)
    AC1_calc = angle_between(A1,  Cvec)
    BC2_calc = angle_between(B2, -Cvec)
    AC2_calc = angle_between(A2,  Cvec)
    dc_calc  = np.linalg.norm(B_C - A_C)

    print("BA   target / calc:", BA,  BA_calc)
    print("BC1  target / calc:", BC1, BC1_calc)
    print("AC1  target / calc:", AC1, AC1_calc)
    print("BC2  target / calc:", BC2, BC2_calc)
    print("AC2  target / calc:", AC2, AC2_calc)
    print("dc   target / calc:", dc,  dc_calc)

def write_renumbered_fv(out_folder, in_path):
    """
    Uses your ANARCII renumbering to produce an IMGT-numbered FV PDB.
    Mirrors your existing helper signature/behavior.
    """
    outputs=process_pdb(
        input_pdb=in_path,
        out_prefix=out_folder,
        write_fv= True
        )
    full_imgt=outputs["pairs"][0]["files"]["full"]
    variable_pdb_imgt=outputs["pairs"][0]["files"]["variable"]
    return full_imgt,variable_pdb_imgt

# ========================
# Geometry and Math Helpers
# ========================
def normalize(v, length=1.0):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return (v / n) * length


def build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc):
    """
    Construct a canonical TCR geometry that is EXACTLY consistent with the
    calc script's definitions of:

       BA, BC1, BC2, AC1, AC2, dc

    The construction matches the calc code:

      Cvec = unit(B_C - A_C)
      A1   = unit(A_V1 - A_C)
      B1   = unit(B_V1 - B_C)
      ...
      BA   = signed angle between projected A1/B1 in the A-defined frame.

    We fix a convenient global frame:

      - A centroid at (0, 0, 0)
      - B centroid at (dc, 0, 0)   → Cvec = +x
      - A1 lies in the x–y plane
      - B1 is rotated around +x so that the resulting BA matches the target.

    A2/B2 are chosen to satisfy AC2/BC2 and avoid degeneracies, but do not
    affect BA by construction (only their angles to Cvec matter in the calc).
    """

    # Convert to radians
    ba   = np.deg2rad(BA)
    bc1  = np.deg2rad(BC1)
    bc2  = np.deg2rad(BC2)
    ac1  = np.deg2rad(AC1)
    ac2  = np.deg2rad(AC2)

    # --- Centroids ---
    A_C = np.array([0.0, 0.0, 0.0])
    B_C = np.array([dc, 0.0, 0.0])
    # Cvec = B_C - A_C = (dc,0,0) → unit (1,0,0)

    # =========================
    # 1) PC1_A direction (A1)
    # =========================
    # Constrain angle(A1, +x) = AC1 → put A1 in x–y plane
    # so that the calc code's AC1 = angle(A1, Cvec) is satisfied.
    A1_dir = np.array([
        np.cos(ac1),
        np.sin(ac1),
        0.0,
    ])
    A1_dir = normalize(A1_dir)

    # =========================
    # 2) PC1_B direction (B1)
    # =========================
    # We need:
    #   BC1 = angle(B1, -Cvec)  and
    #   BA  = torsion defined in the calc script.
    #
    # For Cvec = +x, angle(B1, -x) = BC1  ⇒  polar angle θ_B1 = π - BC1
    # Now parameterise rotation around +x by an azimuth α in the (y,z) plane.
    # One can show algebraically that in the calc script's BA definition,
    # the resulting BA equals exactly α. So we set α = BA (radians).
    theta_B1 = np.pi - bc1
    sin_t1   = np.sin(theta_B1)

    # Azimuth around +x chosen to match BA
    alpha = ba

    B1_dir = np.array([
        np.cos(theta_B1),
        sin_t1 * np.cos(alpha),
        sin_t1 * np.sin(alpha),
    ])
    B1_dir = normalize(B1_dir)

    # =========================
    # 3) PC2_A direction (A2)
    # =========================
    # AC2 = angle(A2, +x). There is free azimuth; we choose a plane that
    # is generically not collinear with A1. Putting A2 in the x–z plane
    # works well and keeps the construction simple.
    theta_A2 = ac2
    A2_dir = np.array([
        np.cos(theta_A2),
        0.0,
        np.sin(theta_A2),
    ])
    A2_dir = normalize(A2_dir)

    # =========================
    # 4) PC2_B direction (B2)
    # =========================
    # Similarly, BC2 = angle(B2, -x) ⇒ polar θ_B2 = π - BC2. We can put B2
    # in the x–y plane for simplicity (any azimuth works for the calc AC/BC2).
    theta_B2 = np.pi - bc2
    B2_dir = np.array([
        np.cos(theta_B2),
        np.sin(theta_B2),
        0.0,
    ])
    B2_dir = normalize(B2_dir)

    # =========================
    # 5) Convert to absolute endpoints
    # =========================
    A_V1 = A_C + A1_dir
    A_V2 = A_C + A2_dir
    B_V1 = B_C + B1_dir
    B_V2 = B_C + B2_dir

    return A_C, A_V1, A_V2, B_C, B_V1, B_V2

def apply_transformation(coords, R, t):
    """Applies rotation (R) and translation (t) to a set of coordinates."""
    return (coords @ R.T) + t

def change_geometry(cons_pdb_with_pca, chain_id, target_centroid, target_v1, target_v2):
    """
    Moves a consensus chain structure to a new target geometry by reading its
    source geometry from pseudoatoms.
    """
    structure = btsio.load_structure(cons_pdb_with_pca, model=1)
    chain = structure[structure.chain_id == chain_id]

    # 1. Determine the source geometry from pseudoatoms in the same file
    try:
        source_centroid = structure[(structure.res_name == 'CEN') & (structure.chain_id == 'Z')].coord[0]
        source_v1_end = structure[(structure.res_name == 'PC1') & (structure.chain_id == 'Z')].coord[0]
        source_v2_end = structure[(structure.res_name == 'PC2') & (structure.chain_id == 'Z')].coord[0]
        source_v1_dir = normalize(source_v1_end - source_centroid)
        source_v2_dir = normalize(source_v2_end - source_centroid)
    except IndexError:
        raise ValueError(f"Could not find CEN, PC1, PC2 pseudoatoms in {cons_pdb_with_pca}")

    # 2. Define the target geometry vectors relative to the centroid
    target_v1_dir = normalize(target_v1 - target_centroid)
    target_v2_dir = normalize(target_v2 - target_centroid)

    # 3. Calculate transformation
    R_source = np.stack([source_v1_dir, source_v2_dir, np.cross(source_v1_dir, source_v2_dir)], axis=1)
    R_target = np.stack([target_v1_dir, target_v2_dir, np.cross(target_v1_dir, target_v2_dir)], axis=1)
    rotation = R_target @ np.linalg.inv(R_source)

    # 4. Apply transformation to the entire chain
    chain.coord = apply_transformation(chain.coord, np.identity(3), -source_centroid) # Move to origin
    chain.coord = apply_transformation(chain.coord, rotation, np.zeros(3))           # Rotate
    chain.coord = apply_transformation(chain.coord, np.identity(3), target_centroid)  # Move to target

    return chain

def move_chains_to_geometry(new_consensus_pdb, input_pdb, output_pdb,A_consenus_res, B_consenus_res):
    """
    Aligns the chains of an input PDB to the newly generated consensus geometry.
    """
    aligned_chain_A = align_chain_to_consensus(input_pdb, new_consensus_pdb, "A", static_consenus_res=A_consenus_res, mobile_consenus_res=A_consenus_res)
    aligned_chain_B = align_chain_to_consensus(input_pdb, new_consensus_pdb, "B", static_consenus_res=B_consenus_res, mobile_consenus_res=B_consenus_res)
    #other chains in input_pdb remain unchanged
    structure = btsio.load_structure(input_pdb, model=1)
    other_chains = structure[(structure.chain_id != "A") & (structure.chain_id != "B")]
    final_aligned_structure = aligned_chain_A + aligned_chain_B +other_chains
    btsio.save_structure(output_pdb, final_aligned_structure)
    print(f"Saved final aligned structure to: {output_pdb}")
    return final_aligned_structure

def align_chain_to_consensus(mobile_pdb_path, static_pdb_path, chain_id, mobile_consenus_res=None, static_consenus_res=None):
    """Helper to align a single chain and return the transformed AtomArray."""
    static_struct = btsio.load_structure(static_pdb_path, model=1)
    mobile_struct = btsio.load_structure(mobile_pdb_path, model=1)
    static_ca = static_struct[(static_struct.atom_name == "CA") & (static_struct.chain_id == chain_id)]
    mobile_ca = mobile_struct[(mobile_struct.atom_name == "CA") & (mobile_struct.chain_id == chain_id)]
    if static_consenus_res:
        static_ca = static_ca[np.isin(static_ca.res_id, static_consenus_res)]
    if mobile_consenus_res:
        mobile_ca = mobile_ca[np.isin(mobile_ca.res_id, mobile_consenus_res)]
    if static_ca.array_length() < 4 or mobile_ca.array_length() < 4:
        raise ValueError(f"Not enough C-alpha atoms for alignment on chain {chain_id}")

    _, transform, _, _ = bts.superimpose_structural_homologs(fixed=static_ca, mobile=mobile_ca)

    mobile_chain_full = mobile_struct[mobile_struct.chain_id == chain_id]

    # Apply transformation using a robust matrix operation
    M = np.asarray(transform.as_matrix(), dtype=np.float64)[0]
    R, t = M[:3, :3], M[:3, 3]
    mobile_chain_full.coord = (mobile_chain_full.coord @ R.T) + t

    return mobile_chain_full

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

def generate_pymol_script(aligned_pdb, new_consensus_pdb, A_C, A_V1, A_V2, B_C, B_V1, B_V2, out_prefix, vis_folder):
    """
    Generates a PyMOL script to visualize the final alignment with longer PCA
    axes and visible centroids.
    """
    pdb_name = Path(aligned_pdb).stem

    # --- FIX: Increased scale factor for longer PCA axes ---
    scale = 10.0

    # Calculate endpoints for the CGO arrows
    a1_end = A_C + scale * (A_V1 - A_C)
    a2_end = A_C + scale * (A_V2 - A_C)
    b1_end = B_C + scale * (B_V1 - B_C)
    b2_end = B_C + scale * (B_V2 - B_C)

    script = f"""
import numpy as np
from pymol import cmd, cgo
cmd.load("{aligned_pdb}","aligned_{pdb_name}")
cmd.load("{new_consensus_pdb}","consensus_geom")
cmd.bg_color("white")
cmd.hide("everything","all")

cmd.show("cartoon","aligned_{pdb_name}")
# --- FIX: Using standard PyMOL color names ---
cmd.color("marine","aligned_{pdb_name} and chain A")
cmd.color("teal","aligned_{pdb_name} and chain B")

cmd.show("cartoon","consensus_geom and polymer")
# --- FIX: Using a more standard gray color ---
cmd.color("gray","consensus_geom")
cmd.set("cartoon_transparency", 0.5, "consensus_geom and polymer")

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
    return vis_script_path

def run(input_pdb, out_path, BA, BC1, BC2, AC1, AC2, dc, vis=True):
    # --- Define paths to input data ---
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")
    #read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    # --- Setup output directories ---
    pdb_name = Path(input_pdb).stem
    out_dir = Path(out_path); out_dir.mkdir(exist_ok=True)
    tmp_out = out_dir / pdb_name; tmp_out.mkdir(exist_ok=True)
    if vis:
        vis_folder = tmp_out / "vis"; vis_folder.mkdir(exist_ok=True)


    #renumbered_pdb,renumbered_pdb_fv=write_renumbered_fv(tmp_out, input_pdb)

    # 1. Build the target geometry from the input angles
    A_C, A_V1, A_V2, B_C, B_V1, B_V2 = build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc)
    verify_built_geometry(A_C, A_V1, A_V2, B_C, B_V1, B_V2, BA, BC1, BC2, AC1, AC2, dc)
    input()

    # 2. Move the consensus chains to this new target geometry
    new_chain_A = change_geometry(consA_pca_path, "A", A_C, A_V1, A_V2)
    new_chain_B = change_geometry(consB_pca_path, "B", B_C, B_V1, B_V2)

    # 3. Combine the moved chains and add pseudoatoms for the new geometry
    new_consensus_structure = new_chain_A + new_chain_B

    # Create pseudoatoms for the new target geometry for validation
    target_pseudoatoms_A = bts.array([
        bts.Atom(coord=A_C, atom_name="CA", res_id=900, res_name="GEA", chain_id="X", element="X"),
        bts.Atom(coord=A_V1, atom_name="V1", res_id=900, res_name="GEA", chain_id="X", element="X"),
        bts.Atom(coord=A_V2, atom_name="V2", res_id=900, res_name="GEA", chain_id="X", element="X")
    ])
    target_pseudoatoms_B = bts.array([
        bts.Atom(coord=B_C, atom_name="CB", res_id=901, res_name="GEB", chain_id="Y", element="X"),
        bts.Atom(coord=B_V1, atom_name="V1", res_id=901, res_name="GEB", chain_id="Y", element="X"),
        bts.Atom(coord=B_V2, atom_name="V2", res_id=901, res_name="GEB", chain_id="Y", element="X")
    ])

    structure_with_geom = new_consensus_structure + target_pseudoatoms_A + target_pseudoatoms_B
    new_consensus_pdb = str(tmp_out / "consensus_oriented.pdb")
    btsio.save_structure(new_consensus_pdb, structure_with_geom)
    print(f"Saved new target geometry with pseudoatoms to: {new_consensus_pdb}")

    # 4. Align the input TCR chains to the new consensus geometry
    final_aligned_pdb = str(tmp_out / f"{pdb_name}_oriented.pdb")
    final_aligned_structure=move_chains_to_geometry(new_consensus_pdb, input_pdb, final_aligned_pdb, A_consenus_res, B_consenus_res)

    # 5. Generate visualization script
    if vis:
        vis_script = generate_pymol_script(
            final_aligned_pdb, new_consensus_pdb,
            A_C, A_V1, A_V2, B_C, B_V1, B_V2, pdb_name, str(vis_folder)
        )

        print(f"\n✅ PyMOL script saved. Run with:\n   pymol -cq {vis_script}")
        os.system(f"pymol -cq {vis_script}")
        print(f"Output files saved in: {tmp_out}")
    #align final_aligned_structure alocha chain to original alpha chain
    final_aligned_pdb_ogorientation=str(tmp_out / f"{pdb_name}_oriented_ogorientation.pdb")
    original_structure = btsio.load_structure(input_pdb, model=1)
    original_alpha = original_structure[original_structure.chain_id == "A"]
    final_alpha = final_aligned_structure[final_aligned_structure.chain_id == "A"]
    _, transform, _, _ = bts.superimpose_structural_homologs(fixed=original_alpha, mobile=final_alpha)
    M = np.asarray(transform.as_matrix(), dtype=np.float64)[0]
    R, t = M[:3, :3], M[:3, 3]
    final_aligned_structure.coord = (final_aligned_structure.coord @ R.T) + t
    btsio.save_structure(final_aligned_pdb_ogorientation, final_aligned_structure)

    return final_aligned_pdb
