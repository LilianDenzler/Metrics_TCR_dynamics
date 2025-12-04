from pymol import cmd
import os
import time
import glob
import numpy as np
CDR_FR_RANGES = {
    # Alpha-like (A)
    "A_FR1":   (3, 26),
    "A_CDR1":  (27, 38),
    "A_FR2":   (39, 55),
    "A_CDR2":  (56, 65),
    "A_FR3":   (66, 104),
    "A_CDR3":  (105, 117),
    "A_FR4":   (118, 125),
    # Beta-like (B)
    "B_FR1":   (3, 26),
    "B_CDR1":  (27, 38),
    "B_FR2":   (39, 55),
    "B_CDR2":  (56, 65),
    "B_FR3":   (66, 104),
    "B_CDR3":  (105, 117),
    "B_FR4":   (118, 125),
}

VIEW_ANGLES = [
    # (Angle, Axis, Suffix)
    (0, 'x', 'front'),       # No rotation from initial zoom/orient
    (90, 'y', 'side'),       # Rotate 90 degrees around Y-axis (vertical)
    (-90, 'y', 'other_side'),# Rotate -90 degrees around Y-axis
    (90, 'x', 'top'),        # Rotate 90 degrees around X-axis (horizontal)
]

def get_red_region(regions):
    all_reds=[]
    for region in regions:
        if region=="A_variable":
            red_region="chain A and resi 3-125"
            all_reds.append(red_region)
            continue
        if region=="B_variable":
            red_region="chain B and resi 3-125"
            all_reds.append(red_region)
            continue
        if region.startswith("A_"):
            chain="A"
        if region.startswith("B_"):
            chain="B"
        if "CDR1" in region:
            resi="(resi 27-38)"
        if "CDR2" in region:
            resi="(resi 56-65)"
        if "CDR3" in region:
            resi="(resi 105-117)"
        red_region=f"chain {chain} and {resi}"
        all_reds.append(red_region)
    red_residues=" or ".join(all_reds)
    return red_residues


def render_zoomed_morph_image(
    closest_structures_folder,
    output_image="zoomed_view",  # base name; suffixes & extensions added
    regions=None,
    zoom_buffer=10
):
    """
    Run from normal Python:

        from this_module import render_zoomed_morph_image
        render_zoomed_morph_image("/path/to/closest_frames", "out/zoomed", "A_CDR3")

    Produces multiple PSE/PNG views: out/zoomed_front.pse, out/zoomed_left.png, etc.
    """
    red_regions_cmd = get_red_region(regions)

    cmd.reinitialize()

    # --- Load structures ---
    for fname in os.listdir(closest_structures_folder):
        if not fname.endswith(".pdb"):
            continue
        obj_name = os.path.splitext(fname)[0]
        path = os.path.join(closest_structures_folder, fname)
        cmd.load(path, obj_name)

    cmd.hide("everything")
    cmd.show("cartoon")

    # Color by name pattern: assume you used "gt_..." and "model_..." when saving
    cmd.color("blue",  "gt*")
    cmd.color("green", "model*")

    # Overall transparency
    cmd.set("cartoon_transparency", 0.9, "model*")
    cmd.set("cartoon_transparency", 0.9, "gt*")

    # Target region opaque
    cmd.set("cartoon_transparency", 0.0, f"model* and {red_regions_cmd}")
    cmd.set("cartoon_transparency", 0.0, f"gt* and {red_regions_cmd}")

    # Orient and zoom on region
    print("Zooming in on loop...")
    cmd.orient()
    cmd.zoom(f"gt* and {red_regions_cmd}", buffer=zoom_buffer)

    # Save base view
    base_view = cmd.get_view()

    # Render each angle
    for angle, axis, suffix in VIEW_ANGLES:
        # Reset to base view
        cmd.set_view(base_view)

        # Rotate around axis if needed
        if angle != 0:
            cmd.turn(axis, angle)

        # Save PSE
        pse_path = f"{output_image}_{suffix}.pse"
        cmd.save(pse_path)
        break #ONLY SAVE ONE VIEW, Rendering in pymol gui is faster

        # Also save a PNG image
        #png_path = f"{output_image}_{suffix}.png"
        #cmd.png(png_path, width=800, height=600, dpi=150, ray=1)

        #print(f"✅ Saved view '{suffix}' to {pse_path} and {png_path}")

    cmd.delete("all")