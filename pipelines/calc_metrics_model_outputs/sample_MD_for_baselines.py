#Randomly sample 200 structures from MD trajectories as baseline
#This is done to get the JSD of "perfect" model outputs (i.e. the MD simulations themselves) as a baseline for comparison to model outputs
import sys
import glob
sys.path.append("/workspaces/Graphormer/")
import MDAnalysis as mda
import numpy as np
import os
from TCR_Metrics.pipelines.calc_metrics_model_outputs.run_calc import *
def folder_to_xtc(input_dir, output_xtc, output_pdb):
    """
    Loads all PDB files from a directory into a single MDAnalysis trajectory,
    saves the coordinates to an XTC file, and saves the first frame as the
    topology PDB.
    """

    # 1. Gather and Sort PDB Files
    # Get all .pdb files and sort them numerically to ensure correct frame order.
    pdb_files = sorted(glob.glob(os.path.join(input_dir, '*.pdb')))

    if not pdb_files:
        print(f"Error: No PDB files found in '{input_dir}'.")
        return

    # 2. Load the Trajectory
    # MDAnalysis allows loading the first PDB as the topology and the rest
    # as coordinate frames.
    try:
        # Load the first PDB file as the topology, and all subsequent files as trajectory coordinates.
        universe = mda.Universe(pdb_files[0], pdb_files[1:])
    except Exception as e:
        print(f"Error loading files into MDAnalysis: {e}")
        return

    n_frames = universe.trajectory.n_frames
    print(f"Loaded {len(pdb_files)} files, resulting in a trajectory of {n_frames} frames.")

    # 3. Save the Topology PDB
    # The topology is the structure information (atoms, residues, etc.), usually from the first frame.
    # Select all atoms for writing.
    all_atoms = universe.select_atoms("all")

    with mda.Writer(output_pdb, all_atoms.n_atoms) as pdb_writer:
        # Write the first frame (state=0) to the topology PDB
        universe.trajectory[0]
        pdb_writer.write(all_atoms)

    print(f"✅ Saved topology (first frame) to: {output_pdb}")

    # 4. Save the XTC Trajectory
    # XTC is a compressed coordinate format.
    # We use 'all_atoms' selection to write the coordinates of the entire system.
    with mda.Writer(output_xtc, all_atoms.n_atoms) as xtc_writer:
        for ts in universe.trajectory:
            xtc_writer.write(all_atoms)

    print(f"✅ Saved {n_frames} frames to XTC file: {output_xtc}")


def sample_trajectory_frames(pdb_path, traj_path, output_dir, num_samples):
    """
    Loads a trajectory, randomly samples a specified number of frames,
    and saves them as separate PDB files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        # Load the trajectory using the PDB for topology and XTC for coordinates
        universe = mda.Universe(pdb_path, traj_path)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return

    n_frames = universe.trajectory.n_frames

    if n_frames == 0:
        print("Error: Trajectory contains no frames.")
        return

    # Check if trajectory is shorter than the desired sample size
    if n_frames < num_samples:
        print(f"Warning: Trajectory has only {n_frames} frames. Sampling all of them.")
        num_samples = n_frames

    # 1. Generate unique, random indices (0-indexed)
    # replace=False ensures indices are unique
    random_indices = np.random.choice(
        n_frames,
        size=num_samples,
        replace=False
    )
    # Sorting is done for slightly better I/O efficiency, though not strictly required
    sorted_indices = np.sort(random_indices)

    print(f"Total frames: {n_frames}. Sampling {num_samples} frames.")

    # Select all atoms for writing
    all_atoms = universe.select_atoms("all")

    # 2. Loop through the sampled indices and save
    counter = 0
    for frame_index in sorted_indices:
        # Jump directly to the desired frame index using MDAnalysis indexing
        universe.trajectory[frame_index]

        # Define output filename, using 0-padding for frame index and sample count
        output_filename = os.path.join(
            output_dir,
            f"frame_{frame_index:05d}_sample_{counter:03d}.pdb"
        )

        # Write the current frame
        with mda.Writer(output_filename, all_atoms.n_atoms) as writer:
            writer.write(all_atoms)

        counter += 1
        print(f"Saved frame {frame_index} to {output_filename}")

def make_cory_subsamples(model_name="baseline_sampled_200_frames_1", NUM_SAMPLES=200,cory_folder="/mnt/larry/lilian/DATA/Cory_data/", benchmarks_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"):
    for folder in os.listdir(cory_folder):
        if not os.path.isdir(os.path.join(cory_folder,folder)):
            continue
        TCR_NAME_full=folder
        TCR_NAME=TCR_NAME_full.split("_")[0]

        print(f"Processing TCR: {TCR_NAME}")
        pdb_gt=f"{cory_folder}/{TCR_NAME}/{TCR_NAME}.pdb"
        xtc_gt=f"{cory_folder}/{TCR_NAME}/{TCR_NAME}_Prod.xtc"
        samples_output_dir=os.path.join(benchmarks_dir,TCR_NAME_full,model_name)
        sample_trajectory_frames(pdb_gt, xtc_gt, samples_output_dir, NUM_SAMPLES)


def make_oriol_adaptive_sampling_subsamples(model_name="baseline_adaptivesampling_200_frames_1", NUM_SAMPLES=200,adaptive_sampling_folder="/mnt/larry/lilian/DATA/Oriol_adaptive_sampling", benchmarks_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"):
    for folder in os.listdir(adaptive_sampling_folder):
        if not os.path.isdir(os.path.join(adaptive_sampling_folder,folder)):
            continue
        TCR_NAME_full=folder
        TCR_NAME=TCR_NAME_full.split("_")[0]

        print(f"Processing TCR: {TCR_NAME}")
        pdb_gt=f"{adaptive_sampling_folder}/{TCR_NAME}/{TCR_NAME}.pdb"
        xtc_gt=f"{adaptive_sampling_folder}/{TCR_NAME}/{TCR_NAME}.xtc"
        samples_output_dir=os.path.join(benchmarks_dir,TCR_NAME_full,model_name)
        sample_trajectory_frames(pdb_gt, xtc_gt, samples_output_dir, NUM_SAMPLES)

if __name__ == "__main__":
    make_oriol_adaptive_sampling_subsamples(model_name="baseline_adaptivesampling_200_frames_1", NUM_SAMPLES=200,adaptive_sampling_folder="/mnt/larry/lilian/DATA/Oriol_adaptive_sampling", benchmarks_dir="/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/")
