from pymol import cmd
import os


def make_alignment_animation(
    pdbA,
    xtcA,
    pdbB,
    xtcB,
    red_residues="chain A and (resi 27-38)",
    method="linear",
    refinement=0,
    stride=1,
    max_pairs=None,
    zoom_frames=30,
    rotate_frames=60,
    export_movie=True,
    movie_png_dir="movie_frames",
    movie_out="morph_movie.mp4",
    fps=30):
    """Create an alignment morphing animation between two trajectories.

    Args:
        pdbA (str): Path to the PDB file for trajectory A (unaligned).
        xtcA (str): Path to the XTC file for trajectory A.
        pdbB (str): Path to the PDB file for trajectory B (aligned).
        xtcB (str): Path to the XTC file for trajectory B.
        method (str): Morphing method ('linear', 'cubic', etc.).
        refinement (int): Number of refinement steps during morphing.
        stride (int): Frame stride when selecting frame pairs.
        max_pairs (int or None): Maximum number of frame pairs to process.
        zoom_frames (int): Number of frames for zoom-in effect.
        rotate_frames (int): Number of frames for rotation effect.
        export_movie (bool): Whether to export the movie as an MP4 file.
        movie_png_dir (str): Directory to save PNG frames for movie export.
        movie_out (str): Output path for the final movie file.
        fps (int): Frames per second for the output movie.
        """
    cmd.reinitialize()

    # Load trajectories
    cmd.load(pdbA, "unaligned")
    cmd.load_traj(xtcA, "unaligned")
    cmd.load(pdbB, "aligned")
    cmd.load_traj(xtcB, "aligned")

    # Determine how many frame-pairs
    nA = cmd.count_states("unaligned")
    nB = cmd.count_states("aligned")
    n = min(nA, nB)
    idxs = list(range(1, n + 1, stride))
    if max_pairs is not None:
        idxs = idxs[:max_pairs]

    morph_names = []
    for k, i in enumerate(idxs, start=1):
        Ai = f"unaligned_{i}"
        Bi = f"aligned_{i}"
        Mi = f"morph_{i}"
        cmd.create(Ai, "unaligned", i, 1)
        cmd.create(Bi, "aligned", i, 1)
        cmd.morph(Mi, Ai, Bi, method=method, refinement=refinement)
        morph_names.append(Mi)

    # Cleanup
    cmd.delete("unaligned")
    cmd.delete("aligned")
    cmd.delete("unaligned_*")
    cmd.delete("aligned_*")
    cmd.hide("everything", "all")
    cmd.show("cartoon", "morph_*")

    # Color
    cmd.color("tv_blue",  "morph_* and chain A")
    cmd.color("tv_green", "morph_* and chain B")
    cmd.color("red",      red_residues)
    cmd.select("loop_sel", red_residues)

    # Count states
    first_morph = morph_names[0]
    n_states = cmd.count_states(first_morph)

    # Frame plan
    zoom_start = n_states + 1
    zoom_end = zoom_start + zoom_frames - 1
    rot_start = zoom_end + 1
    rot_end = rot_start + rotate_frames - 1
    total_frames = rot_end

    cmd.set("movie_auto_interpolate", 0)
    cmd.mset(f"1 -{n_states} {n_states} x{zoom_frames} {n_states} x{rotate_frames}")
    cmd.mview("clear")

    # Frame 1: start of morph
    cmd.frame(1)
    cmd.orient("morph_*")
    cmd.zoom("morph_*", buffer=15)
    cmd.mview("store", 1)

    # Frame n_states: end of morph (static)
    cmd.frame(n_states)
    cmd.mview("store", n_states)

    # Capture full view
    cmd.frame(n_states)
    view_full = cmd.get_view()

    # Capture zoomed view on red loop
    cmd.zoom("loop_sel", buffer=15)
    view_zoom = cmd.get_view()

    # Interpolate zoom from view_full to view_zoom
    for i, f in enumerate(range(zoom_start, zoom_end + 1)):
        interp = i / (zoom_frames - 1)
        view_interp = [
            view_full[j] * (1 - interp) + view_zoom[j] * interp
            for j in range(18)
        ]
        cmd.frame(f)
        cmd.set_view(view_interp)
        cmd.mview("store", f)

    # Rotate after zoom-in
    cmd.set_view(view_zoom)
    for i, f in enumerate(range(rot_start, rot_end + 1)):
        cmd.frame(f)
        cmd.turn("y", 360 / rotate_frames)
        cmd.mview("store", f)

    # Enable camera smoothing
    cmd.set("movie_auto_interpolate", 1)

    # Play interactively
    cmd.frame(1)
    cmd.mplay()

    # Export movie
    if export_movie:
        os.makedirs(movie_png_dir, exist_ok=True)
        cmd.mstop()
        cmd.frame(1)
        cmd.mpng(f"{movie_png_dir}/frame")
        os.system(
            f'ffmpeg -y -framerate {fps} -i {movie_png_dir}/frame%04d.png '
            f'-pix_fmt yuv420p "{movie_out}"'
        )
    #delete frame directory
        if os.path.exists(movie_png_dir):
            import shutil
            shutil.rmtree(movie_png_dir)

def run_animation(aligned_on="B_CDR3"):
    if aligned_on=="A_variable":
        red_residues="chain A and (resi 3-26+39-55+66-104+118-125)"
    if aligned_on=="B_variable":
        red_residues="chain B and (resi 3-26+39-55+66-104+118-125)"
    if aligned_on=="B_CDR3":
        red_residues="chain B and (resi 105-117)"
    if aligned_on=="A_CDR1":
        red_residues="chain A and (resi 27-38)"

    pdbA = f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/3QEU_B/{aligned_on}/gt_unaligned.pdb"
    xtcA = f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/3QEU_B/{aligned_on}/gt_unaligned.xtc"
    pdbB = f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/3QEU_B/{aligned_on}/gt_aligned.pdb"
    xtcB = f"/workspaces/Graphormer/TCR_Metrics/outputs/outputs_dig_aaron_benchmark/Alphaflow/3QEU_B/{aligned_on}/gt_aligned.xtc"


    frames_per_morph = 60
    max_pairs = None
    stride = 1000
    method = "linear"
    refinement = 3
    export_movie = True
    fps = 24

    zoom_frames = 24           # smooth zoom in (~1 sec)
    rotate_frames = 54         # rotate around loop (~1 sec)
    movie_png_dir = f"morph_frames_{aligned_on}"
    movie_out = f"morph_{aligned_on}.mp4"
    make_alignment_animation(
        pdbA,
        xtcA,
        pdbB,
        xtcB,
        red_residues=red_residues,
        method=method,
        refinement=refinement,
        stride=stride,
        max_pairs=max_pairs,
        zoom_frames=zoom_frames,
        rotate_frames=rotate_frames,
        export_movie=export_movie,
        movie_png_dir=movie_png_dir,
        movie_out=movie_out,
        fps=fps)

cmd.do('run_animation("B_CDR3")')
cmd.do('run_animation("A_variable")')
cmd.do('run_animation("B_variable")')
cmd.do('run_animation("A_CDR1")')
