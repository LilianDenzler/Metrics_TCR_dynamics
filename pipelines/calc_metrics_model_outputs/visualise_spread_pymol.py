from pymol import cmd
import os
import time
def get_red_region(red_region):
    if red_region=="A_FR":
        red_residues="chain A and (resi 3-26+39-55+66-104+118-125)"
    if red_region=="B_FR":
        red_residues="chain B and (resi 3-26+39-55+66-104+118-125)"
    if red_region=="A_CDR1":
        red_residues="chain A and (resi 27-38)"
    if red_region=="A_CDR2":
        red_residues="chain A and (resi 56-65)"
    if red_region=="A_CDR3":
        red_residues="chain A and (resi 105-117)"
    if red_region=="B_CDR1":
        red_residues="chain B and (resi 27-38)"
    if red_region=="B_CDR2":
        red_residues="chain B and (resi 56-65)"
    if red_region=="B_CDR3":
        red_residues="chain B and (resi 105-117)"
    if red_region=="all_A_CDRs":
        red_residues="chain A and (resi 27-38+56-65+105-117)"
    if red_region=="all_B_CDRs":
        red_residues="chain B and (resi 27-38+56-65+105-117)"
    if red_region=="all_CDRs":
        red_residues="(chain A + chain B) and (resi 27-38+56-65+105-117)"
    return red_residues

def render_zoomed_morph_image(
    pred_xtc="pred_aligned.xtc",
    pred_pdb="pred_aligned.pdb",
    gt_xtc="gt_aligned.xtc",
    gt_pdb="gt_aligned.pdb",
    output_image="zoomed_view.png",
    red_regions_cmd=None,
    stride=1000,
    max_frame=10000,
    zoom_buffer=10,
    cartoon_transparency=0.7
):
    cmd.reinitialize()

    # Load trajectories
    cmd.load(gt_pdb, "gt_aligned")
    cmd.load_traj(gt_xtc, "gt_aligned")
    cmd.load(pred_pdb, "pred_aligned")
    cmd.load_traj(pred_xtc, "pred_aligned")
    print("Loaded trajectories.")
    cmd.hide("everything")
    cmd.show("cartoon")

    # Trim GT to 1–128
    cmd.select("gt_aligned_trim", "gt_aligned and resi 1-128")
    cmd.create("gt_ali", "gt_aligned_trim")


    # Color pred loop red
    cmd.select("red_loop_pred", f"pred_aligned and {red_regions_cmd}")
    cmd.color("red", "red_loop_pred")

    # Split pred into states
    print("Splitting frames from prediction...")
    cmd.split_states("pred_aligned", prefix="pred")
    cmd.delete("pred_aligned")  # Free up memory between renderings

    # Sample frames from GT
    print("Sampling frames from ground truth...")
    sample_idx = 0
    for frame in range(1, max_frame + 1, stride):
        sample_idx += 1
        obj = f"gt_{sample_idx}"
        cmd.create(obj, "gt_ali", frame, 1)   # <- each sample is its own object
        cmd.set("cartoon_transparency", cartoon_transparency, obj)
    cmd.delete("gt_ali")  # Free up memory between renderings
    cmd.delete("gt_aligned")  # Free up memory between renderings
    cmd.delete("gt_aligned_trim")  # Free up memory between renderings
    # Color GT loop magenta
    cmd.select("red_loop_gt", f"gt_* and {red_regions_cmd}")
    cmd.color("magenta", "red_loop_gt")

    # Set transparency on GT
    print("Setting cartoon transparency for GT...")
    cmd.set("cartoon_transparency", cartoon_transparency, "pred*")
    cmd.set("cartoon_transparency", cartoon_transparency, "gt*")
    cmd.set("cartoon_transparency", 0.0, "red_loop_pred")
    cmd.set("cartoon_transparency", 0.5, "red_loop_gt")

    # Zoom in on red loop
    print("Zooming in on loop...")
    cmd.zoom("red_loop_gt", buffer=zoom_buffer)

    # Render and save image
    # === Speed & GPU-acceleration settings ===
    #cmd.set("use_shaders", 1)
    #cmd.set("ray_trace_mode", 1)
    #cmd.set("ray_trace_gain", 0.1)
    #cmd.set("ray_trace_disco_factor", 1)
    #cmd.set("ray_trace_fog", 0)


    cmd.save(os.path.join(output_image+".pse"))
    cmd.png(os.path.join(output_image+".png"),width=600, height=500, dpi=50, ray=1)
    print(f"✅ Saved image to {output_image}")
    cmd.delete("all")  # Free up memory between renderings

settings=[
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

model_name="AFL_linker"
model_out_dir="/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/"

for TCR_NAME in os.listdir(model_out_dir):
    output_dir="/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/visualisations/"
    os.makedirs(output_dir, exist_ok=True)
    output_dir_model=os.path.join(output_dir, model_name)
    os.makedirs(output_dir_model, exist_ok=True)
    output_dir=os.path.join(output_dir_model, TCR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    TCR_dir=os.path.join(model_out_dir, TCR_NAME)
    for red_region, aligned_on in settings:
            output_image=os.path.join(output_dir,f"{red_region}_alignedon_{aligned_on}")
            red_regions_cmd=get_red_region(red_region)
            start_time=time.time()
            render_zoomed_morph_image(
                pred_xtc=f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/{TCR_NAME}/{aligned_on}/pred_aligned.xtc",
                pred_pdb=f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/{TCR_NAME}/{aligned_on}/pred_aligned.pdb",
                gt_xtc=f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/{TCR_NAME}/{aligned_on}/gt_aligned.xtc",
                gt_pdb=f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/{TCR_NAME}/{aligned_on}/gt_aligned.pdb",
                output_image=output_image,
                red_regions_cmd=red_regions_cmd
                )
            end_time=time.time()
            print(f"Time taken to render {output_image}: {end_time-start_time} seconds")