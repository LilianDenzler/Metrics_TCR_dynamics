from TCR_TOOLS.geometry.calc_geometry import *
import MDAnalysis as mda
import tempfile
from tqdm import tqdm
import sys, os
from contextlib import redirect_stdout
import warnings
from . import DATA_PATH
from pathlib import Path

warnings.filterwarnings("ignore", message=".*formalcharges.*")

def run(input_traj, input_top):
    consA_pca_path = os.path.join(DATA_PATH, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(DATA_PATH, "chain_B/average_structure_with_pca.pdb")
     #read file with consensus alignment residues as list of integers
    with open(os.path.join(DATA_PATH, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(DATA_PATH, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    u = mda.Universe(input_top, input_traj)
    # Prepare arrays for results; list-of-dicts is fine but arrays are faster
    frames = []
    times  = []
    BA_arr  = []
    BC1_arr = []
    AC1_arr = []
    BC2_arr = []
    AC2_arr = []
    dc_arr  = []

    # Stream frames
    for ts in tqdm(u.trajectory, total=len(u.trajectory), desc="Processing frames"):
        # Write out current frame to PDB
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            with tempfile.TemporaryDirectory() as td:
                tmp_pdb = Path(td) / "frame.pdb"
                u.atoms.write(tmp_pdb.as_posix())  # writes PDB

                # Process (align + compute angles + visualize)
                result_frame = process(
                    input_pdb=tmp_pdb,
                    consA_with_pca=consA_pca_path,
                    consB_with_pca=consB_pca_path,
                    out_dir=str(td),
                    vis_folder=None,
                    A_consenus_res=A_consenus_res,
                    B_consenus_res=B_consenus_res
                )
            BA, BC1, AC1, BC2, AC2, dc = (
                result_frame["BA"],
                result_frame["BC1"],
                result_frame["AC1"],
                result_frame["BC2"],
                result_frame["AC2"],
                result_frame["dc"],
            )
            frames.append(ts.frame)
            times.append(getattr(ts, "time", np.nan))  # ps if present
            BA_arr.append(BA); BC1_arr.append(BC1); AC1_arr.append(AC1)
            BC2_arr.append(BC2); AC2_arr.append(AC2); dc_arr.append(dc)
    df = pd.DataFrame({
        "frame": frames,
        "time_ps": times,
        "BA": BA_arr,
        "BC1": BC1_arr,
        "AC1": AC1_arr,
        "BC2": BC2_arr,
        "AC2": AC2_arr,
        "dc": dc_arr,
    })
    return df


