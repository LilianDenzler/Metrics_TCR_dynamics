# visualise_spread_vmd.py
# VMD Python 2.x compatible, no os / stdlib imports

from VMD import evaltcl
import molecule
import molrep
import render
import display
import atomsel

# ----------------------------------------------------------------------
# Selection helper
# ----------------------------------------------------------------------
def get_red_region(red_region):
    """Map region name to a VMD atom selection string."""
    if red_region == "A_FR":
        red_sel = "(chain A) and (resid 3 to 26 or resid 39 to 55 or resid 66 to 104 or resid 118 to 125)"
    elif red_region == "B_FR":
        red_sel = "(chain B) and (resid 3 to 26 or resid 39 to 55 or resid 66 to 104 or resid 118 to 125)"
    elif red_region == "A_CDR1":
        red_sel = "chain A and resid 27 to 38"
    elif red_region == "A_CDR2":
        red_sel = "(chain A) and (resid 56 to 65)"
    elif red_region == "A_CDR3":
        red_sel = "(chain A) and (resid 105 to 117)"
    elif red_region == "B_CDR1":
        red_sel = "(chain B) and (resid 27 to 38)"
    elif red_region == "B_CDR2":
        red_sel = "(chain B) and (resid 56 to 65)"
    elif red_region == "B_CDR3":
        red_sel = "(chain B) and (resid 105 to 117)"
    elif red_region == "all_A_CDRs":
        red_sel = "(chain A) and (resid 27 to 38 or resid 56 to 65 or resid 105 to 117)"
    elif red_region == "all_B_CDRs":
        red_sel = "(chain B) and (resid 27 to 38 or resid 56 to 65 or resid 105 to 117)"
    elif red_region == "all_CDRs":
        red_sel = "(chain A or chain B) and (resid 27 to 38 or resid 56 to 65 or resid 105 to 117)"
    else:
        red_sel = "protein"
    return red_sel


# ----------------------------------------------------------------------
# VMD helpers
# ----------------------------------------------------------------------


def reset_vmd():
    """Clear all molecules."""
    mids = molecule.listall()
    for mid in mids:
        molecule.delete(mid)


def center_on_selection(molid, selection_str):
    """Recenter and reset view."""
    sel = atomsel.atomsel(selection_str, molid=molid)
    if sel.num() == 0:
        sel = atomsel.atomsel("protein", molid=molid)

    coords = sel.center()  # [x, y, z]
    cmd = "molinfo %d set {center} {{%0.3f %0.3f %0.3f}}" % (
        molid,
        coords[0],
        coords[1],
        coords[2],
    )
    evaltcl(cmd)
    evaltcl("display resetview")


def add_cartoon_rep(molid, selection, color_id, material_name):
    """Create a cartoon representation and return its index."""
    molrep.addrep(molid)
    rep_index = molrep.num(molid) - 1
    molrep.modselect(rep_index, molid, selection)
    molrep.modstyle(rep_index, molid, "NewCartoon", "", "")
    molrep.modcolor(rep_index, molid, "ColorID %d" % color_id)
    molrep.modmaterial(rep_index, molid, material_name)
    return rep_index


def file_exists(path):
    """Check file existence via Tcl."""
    res = evaltcl('file exists "%s"' % path)
    return res.strip() == "1"


def mkdir_p(path):
    """mkdir -p via Tcl."""
    evaltcl('file mkdir "%s"' % path)


def list_subdirs(path):
    """Return list of immediate subdirectory names of 'path' using Tcl glob."""
    cmd = 'lsort [glob -nocomplain -type d "%s/*"]' % path
    out = evaltcl(cmd).strip()
    if out == "":
        return []
    full_paths = out.split()
    names = []
    for p in full_paths:
        parts = p.split("/")
        if len(parts) > 0:
            names.append(parts[-1])
    return names


# ----------------------------------------------------------------------
# Core rendering
# ----------------------------------------------------------------------
# --- IMPORTANT: This helper function must also be correct ---
def center_on_selection(molid, selection_str):
    """Approximate PyMOL zoom: recenter and reset view."""
    sel = atomsel.atomsel(selection_str, molid=molid)
    coords = sel.center()  # [x, y, z]

    cmd = "molinfo %d set {center} {{%0.3f %0.3f %0.3f}}" % (
        molid,
        coords[0],
        coords[1],
        coords[2],
    )
    evaltcl(cmd)

    # --- This is the correct call to zoom the camera ---
    evaltcl('display resetview')


# ----------------------------------------------------------------------
# Core rendering (Complete Corrected Function)
# ----------------------------------------------------------------------

def render_zoomed_morph_image(
    pred_xtc,
    pred_pdb,
    gt_xtc,
    gt_pdb,
    output_image,
    red_regions_cmd,
    stride,
    max_frame,
    zoom_buffer,
    cartoon_transparency,
):
    """Load GT & pred, highlight CDR, draw ALL frames, zoom, ray trace image."""
    reset_vmd()

    print("Rendering", output_image)

    # --- 1) Check files ---
    if not (file_exists(gt_pdb) and file_exists(gt_xtc)):
        print("  Missing GT files:", gt_pdb, gt_xtc)
        return
    if not (file_exists(pred_pdb) and file_exists(pred_xtc)):
        print("  Missing pred files:", pred_pdb, pred_xtc)
        return

    # --- 2) Load GT and pred (+ trajectories) ---
    molid_gt = molecule.load("pdb", gt_pdb)
    molecule.read(molid_gt, "xtc", gt_xtc)

    molid_pred = molecule.load("pdb", pred_pdb)
    molecule.read(molid_pred, "xtc", pred_xtc)

    nframes_gt = int(evaltcl("molinfo %d get numframes" % molid_gt))
    nframes_pred = int(evaltcl("molinfo %d get numframes" % molid_pred))

    print("  numframes_gt =", nframes_gt, "numframes_pred =", nframes_pred)

    if nframes_gt < 1 or nframes_pred < 1:
        print("  No frames in one of the trajectories, skipping.")
        return

    # Use shared number of frames, capped by max_frame
    nframes   = nframes_gt if nframes_gt < nframes_pred else nframes_pred
    last_frame = nframes - 1
    if last_frame > max_frame - 1:
        last_frame = max_frame - 1

    print("  Using frames 0..%d step %d" % (last_frame, stride))

    molecule.set_frame(molid_gt, 0)
    molecule.set_frame(molid_pred, 0)
    print("######Frames set")

    # --- 3) Keep default display (safe) ---
    print("######Display (defaults)")

    # --- 4) Clear default reps ---
    while molrep.num(molid_gt) > 0:
        molrep.delrep(molid_gt, 0)
    while molrep.num(molid_pred) > 0:
        molrep.delrep(molid_pred, 0)
    print("######Reps cleared")

    # --- 5) Set up transparency ---
    vmd_opacity = 1.0 - cartoon_transparency

    # --- THIS IS THE FIX ---
    # The order is 'Transparent' (material) then '%f' (value)
    evaltcl("material change opacity Transparent %f" % vmd_opacity)
    # -----------------------

    print("######Material 'Transparent' opacity set to", vmd_opacity)

    # --- 6) Backbone reps: Use 'Transparent' material ---
    add_cartoon_rep(
        molid_gt,
        rep_index=0,
        selection="protein",
        color_id=8,
        material_name="Transparent",
    )
    add_cartoon_rep(
        molid_pred,
        rep_index=0,
        selection="protein",
        color_id=0,
        material_name="Transparent",
    )
    print("######Backbone reps added")

    # --- 7) Highlight CDRs: Use 'Transparent' material ---
    if red_regions_cmd is None:
        red_regions_cmd = "protein"

    add_cartoon_rep(
        molid_pred,
        rep_index=1,
        selection=red_regions_cmd,
        color_id=1,          # red
        material_name="Transparent",
    )
    add_cartoon_rep(
        molid_gt,
        rep_index=1,
        selection=red_regions_cmd,
        color_id=9,          # magenta-ish
        material_name="Transparent",
    )
    print("######CDR reps added")

    # --- 8) Draw ALL frames for each rep (GT + pred) ---
    framespec = "0:%d:%d" % (last_frame, stride)
    evaltcl("mol drawframes %d 0 {%s}" % (molid_gt,   framespec))
    evaltcl("mol drawframes %d 1 {%s}" % (molid_gt,   framespec))
    evaltcl("mol drawframes %d 0 {%s}" % (molid_pred, framespec))
    evaltcl("mol drawframes %d 1 {%s}" % (molid_pred, framespec))
    print("######Drawframes set:", framespec)

    # --- 9) Center on the whole protein ---
    center_on_selection(molid_gt, "protein")
    print("######View centered and reset")

    # --- 10) Save state and GPU render ---
    base = output_image
    state_file = base + ".vmd"
    img_file   = base + ".tga"

    print("######Updating display...")
    display.update()

    evaltcl('save_state "%s"' % state_file)
    render.render("TachyonLOptiXInternal", img_file)

    print("  Saved VMD state to", state_file)
    print("  Rendered image to", img_file)
# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------
settings = [
    ("A_CDR1", "A_CDR1"),
    ("A_CDR2", "A_CDR2"),
    ("A_CDR3", "A_CDR3"),
    ("B_CDR1", "B_CDR1"),
    ("B_CDR2", "B_CDR2"),
    ("B_CDR3", "B_CDR3"),
    ("all_A_CDRs", "A_variable"),
    ("all_B_CDRs", "B_variable"),
    ("all_CDRs", "A_variable"),
]

model_name   = "AFL_linker"
model_out_dir = "/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow"
output_root   = "/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/visualisations"


def main():
    print(">>> running visualise_spread_vmd.py <<<")

    mkdir_p(output_root)
    output_dir_model = output_root + "/" + model_name
    mkdir_p(output_dir_model)

    tcr_names = list_subdirs(model_out_dir)
    if len(tcr_names) == 0:
        print("No TCR dirs found under", model_out_dir)

    for tcr in tcr_names:
        output_dir = output_dir_model + "/" + tcr
        mkdir_p(output_dir)

        for red_region, aligned_on in settings:
            base_path = model_out_dir + "/" + tcr + "/" + aligned_on

            pred_xtc = base_path + "/pred_aligned.xtc"
            pred_pdb = base_path + "/pred_aligned.pdb"
            gt_xtc   = base_path + "/gt_aligned.xtc"
            gt_pdb   = base_path + "/gt_aligned.pdb"

            output_image = output_dir + "/" + red_region + "_alignedon_" + aligned_on
            red_sel = get_red_region(red_region)

            print("Processing", tcr, red_region, "aligned_on", aligned_on)
            render_zoomed_morph_image(
                pred_xtc=pred_xtc,
                pred_pdb=pred_pdb,
                gt_xtc=gt_xtc,
                gt_pdb=gt_pdb,
                output_image=output_image,
                red_regions_cmd=red_sel,
                stride=1,          # <<< ALL FRAMES
                max_frame=1000000, # huge cap, effectively ignore
                zoom_buffer=10,
                cartoon_transparency=0.7,
            )

    evaltcl("quit")


if __name__ == "__main__":
    main()
