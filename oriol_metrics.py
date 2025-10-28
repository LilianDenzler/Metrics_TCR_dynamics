import argparse
import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd

# Constants
R_KJ_PER_MOL_K = 0.00831446261815324  # kJ/(mol*K)

# -----------------------------
# Loading functions
# -----------------------------


def load_md_xtc(md_xtc_paths, topology_path):
    """
    Loads a list of xtc trajectories into memory
    """
    traj = md.load(md_xtc_paths, top=topology_path)
    return traj

def strip_backbone_atoms(md_traj):
    """
    Extracts the atoms of the backbone to match backbone frame representation.
    """
    # Select atom indices
    atom_indices = md_traj.topology.select('name CA or name C or name N')

    # Slice trajectory
    backbone_traj = md_traj.atom_slice(atom_indices)
    return backbone_traj

def load_model_numpy(npy_paths, topology):
    """
    Loads a trajectory stored into npz files.
    """
    if isinstance(npy_paths, str):
        npy_paths = [npy_paths]
    all_coords = []
    for npy_path in npy_paths:
        arr = np.load(npy_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = list(arr.keys())
            if len(keys) == 0:
                raise ValueError(".npz file contains no arrays")
            arr = arr[keys[0]]
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 3:
            coords = arr.copy()
        elif arr.ndim == 2 and arr.shape[1] % 3 == 0:
            n_atoms = arr.shape[1] // 3
            coords = arr.reshape((arr.shape[0], n_atoms, 3))
        else:
            raise ValueError("Invalid numpy shape for coordinates")
        all_coords.append(coords)
    all_coords = np.concatenate(all_coords, axis=0)
    traj = md.Trajectory(all_coords, topology)
    return traj

def load_model_pdb(pdb_files, topology):
    """
    Loads pdbs and creates an mdtraj trajectory matching a given topology.
    """
    combined_xyz = []

    for pdb_path in pdb_files:
        traj = md.load_pdb(pdb_path)

        # Ensure atom count matches reference
        if traj.n_atoms != topology.n_atoms:
            raise ValueError(f"Atom count mismatch in {pdb_path}: "
                             f"{traj.n_atoms} vs {topology.n_atoms}")

        combined_xyz.append(traj.xyz)

    # Concatenate all frames
    all_frames = np.concatenate(combined_xyz, axis=0)

    # Create combined trajectory
    combined_traj = md.Trajectory(all_frames, topology=topology)
    return combined_traj


# -----------------------------
# Saving functions
# -----------------------------


def save_pmf(fe, xedges, yedges, filename):
    """
    Save PMF (free energy) and bin edges to a .npz file.
    """
    np.savez_compressed(
        filename,
        fe=fe,
        xedges=xedges,
        yedges=yedges
    )
    print(f"PMF saved to {filename}")


# -----------------------------
# Processing functions
# -----------------------------


def select_atom_indices_for_residues(topology, selected_res_array):
    """
    Creates an np.array with all the atom indexes belonging to a list of residues
    """
    atom_indices = []
    for res in topology.residues:
        if res.resSeq in selected_res_array:
            atom_indices.extend([a.index for a in res.atoms])
    return np.array(atom_indices, dtype=int)

def compute_pca(traj_all, atom_indices, n_components=2):
    """
    Computes a PCA for the selected atom indices and returns the PCA object and projection.
    """
    coords = traj_all.xyz[:, atom_indices, :]
    X = coords.reshape(traj_all.n_frames, -1)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X)
    return pca, proj

def limit_energies(fe_profile, max_ene):
    """
    Sets all infinite and high values to a predefined max value
    """
    for jy in range(len(fe_profile[0,:])):
        for jx in range(len(fe_profile[:,0])):
            if np.isinf(fe_profile[jx,jy]) or fe_profile[jx,jy] > max_ene:
                fe_profile[jx,jy]=max_ene
    return fe_profile

def compute_pmf_hist(proj, nbins=(100,100), temperature=300.0, pseudo_counts=1e-9, max_ene=None):
    """
    Computes a PMF from a projection trough a histogram approach.
    """
    x, y = proj[:,0], proj[:,1]
    xedges = np.linspace(x.min(), x.max(), nbins[0]+1)
    yedges = np.linspace(y.min(), y.max(), nbins[1]+1)
    occupancy, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    prob = occupancy.T + pseudo_counts  # pseudo-count to avoid 0 probabilities
    prob /= prob.sum()
    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob)
    fe -= np.nanmin(fe)
    if max_ene:
        fe = limit_energies(fe, max_ene)

    return fe, xedges, yedges

def compute_pmf_kde(proj, nbins=(100,100), temperature=300.0, pseudo_counts=1e-9, max_ene=None, bandwidth=None):
    """
    Computes a PMF from a projection trough a KDE approach.
    """
    x, y = proj[:,0], proj[:,1]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    xgrid = np.linspace(x.min(), x.max(), nbins[0])
    ygrid = np.linspace(y.min(), y.max(), nbins[1])
    X, Y = np.meshgrid(xgrid, ygrid)
    density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    prob = density * dx * dy
    prob /= prob.sum()
    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob + pseudo_counts)
    fe -= np.nanmin(fe)
    xedges = np.linspace(x.min(), x.max(), nbins[0]+1)
    yedges = np.linspace(y.min(), y.max(), nbins[1]+1)
    if max_ene:
        fe = limit_energies(fe, max_ene)

    return fe, xedges, yedges


# -----------------------------
# Comparison functions
# -----------------------------


def align_pmfs_and_masks(fe_md, fe_model):
    """
    Checks for infinite values and extract masks for both conditions
    """
    mask_md = np.isfinite(fe_md)
    mask_model = np.isfinite(fe_model)
    common_mask = mask_md & mask_model
    md_only_mask = mask_md & ~mask_model
    model_only_mask = ~mask_md & mask_model
    return common_mask, md_only_mask, model_only_mask

def compute_mse_masked(fe_md, fe_model, mask):
    """
    Compute the Mean Squared Error between two profiles.
    Masking out non matching regions defined by mask.
    """
    diff = fe_md[mask] - fe_model[mask]
    return np.mean(diff**2) if diff.size > 0 else np.nan

def compute_mse(fe_md, fe_model):
    """
    Compute the Mean Squared Error between two profiles.
    """
    diff = fe_md - fe_model
    return np.mean(diff**2)


# -----------------------------
# Plotting functions
# -----------------------------


def plot_pca(proj, name):
    """
    Bare-bones plotting funnction for quick visualization of PCA.
    """
    plt.figure(figsize=(6,5))
    plt.scatter(proj[:,0], proj[:,1], s=10, alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of selected atoms')
    plt.savefig(f"pca_{name}_all.png")


def plot_kde(fe_md, fe_model, fe_diff, xedges, yedges, name):
    """
    Bare-bones plotting funnction for quick visualization of KDE PMF and their differences.
    """
    plt.figure(figsize=(18,5))  # wider figure to fit 3 subplots

    # MD PMF
    plt.subplot(1,3,1)
    plt.imshow(
        fe_md,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation='bilinear'
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('MD PMF')

    # Model PMF
    plt.subplot(1,3,2)
    plt.imshow(
        fe_model,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation='bilinear'
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('Model PMF')

    # Difference PMF
    plt.subplot(1,3,3)
    plt.imshow(
        fe_diff,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='bwr',  # blue-red diverging colormap for differences
        interpolation='bilinear'
    )
    plt.colorbar(label='FE difference (kJ/mol)')
    plt.title('FE Difference (MD - Model)')

    plt.tight_layout()
    plt.savefig(f"{name}_pmfs_diff.png")
    plt.close()


def plot_pmf(fe_md, fe_model, fe_diff, xedges, yedges, name):
    """
    Bare-bones plotting funnction for quick visualization of PMF and their differences.
    """

    plt.figure(figsize=(18,5))

    # MD PMF
    plt.subplot(1,3,1)
    plt.imshow(
        fe_md,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('MD PMF')

    # Model PMF
    plt.subplot(1,3,2)
    plt.imshow(
        fe_model,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('Model PMF')

    # Difference PMF
    plt.subplot(1,3,3)
    plt.imshow(
        fe_diff,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='bwr'  # blue-red diverging colormap for differences
    )
    plt.colorbar(label='FE difference (kJ/mol)')
    plt.title('FE Difference (MD - Model)')

    plt.tight_layout()
    plt.savefig(f"{name}_pmfs_nokde_diff.png")
    plt.close()

# -----------------------------
# Example of main pipeline
# -----------------------------


def main(md_files, model_files, topology, xbins, ybins, selected_residues, name, temperature):

    print("Loading MD...")
    md_traj = load_md_xtc(md_files, topology)
    md_traj = strip_backbone_atoms(md_traj)
    print(f"MD frames: {md_traj.n_frames}")

    print("Loading model...")
    #model_traj = load_model_numpy(model_files, md_traj.topology)
    model_traj = load_model_pdb(model_files, md_traj.topology)
    print(f"Model frames: {model_traj.n_frames}")

    # Ensure model_traj has the same unit cell as md_traj
    model_traj.unitcell_lengths = np.tile(md_traj.unitcell_lengths[0], (model_traj.n_frames, 1))
    model_traj.unitcell_angles  = np.tile(md_traj.unitcell_angles[0],  (model_traj.n_frames, 1))

    traj_all = md_traj.join(model_traj)
    N_md, N_model = md_traj.n_frames, model_traj.n_frames

    # superpose all frames. Different options may be used here depending on the needs.
    traj_all.superpose(traj_all, 0)

    # Get the atom indices for the selected residues
    atom_indices = select_atom_indices_for_residues(traj_all.topology, selected_residues)
    print(f"Selected {len(atom_indices)} atoms")

    # Compute a PCA over the cartesian coordinates. Other options are also available
    # One can also compute a PCA over the MD frames and project the infered frame on those instead of computing a PCA of everything.
    pca, proj_all = compute_pca(traj_all, atom_indices)

    # Compute a TiCa over the MD trajectories. (Important, time decorrelation makes no sense on time independent data).

    # Save pca
    plot_pca(proj_all, name)

    # Extract frames from MD and model from the projection.
    proj_md, proj_model = proj_all[:N_md], proj_all[N_md:]

    # Histogram calculation
    nbins = (xbins, ybins)

    # Histograms without pseudocounts to find non overlapping regions
    fe_md_nopseudo, xedges, yedges = compute_pmf_hist(proj_md, nbins, temperature, pseudo_counts=0)
    fe_model_nopseudo, _, _ = compute_pmf_hist(proj_model, nbins, temperature, pseudo_counts=0)

    # Histogram with pseudocounts and max energy
    max_ene = np.max(fe_md_nopseudo[np.isfinite(fe_md_nopseudo)]) + (R_KJ_PER_MOL_K * temperature) # Maximum seen energy + kbt
    fe_md, xedges, yedges = compute_pmf_hist(proj_md, nbins, temperature, pseudo_counts=1e-9, max_ene=max_ene)
    fe_model, _, _ = compute_pmf_hist(proj_model, nbins, temperature, pseudo_counts=1e-9, max_ene=max_ene)

    # Save
    save_pmf(fe_md, xedges, yedges, f"{name}_histogram_MD.npz")
    save_pmf(fe_model, xedges, yedges, f"{name}_histogram_diffusion.npz")

    # KDE PMF calculation for smooth visualization
    fe_md_kde, xedges_kde, yedges_kde = compute_pmf_kde(proj_md, nbins, temperature, pseudo_counts=1e-9)
    fe_model_kde, _, _ = compute_pmf_kde(proj_model, nbins, temperature, pseudo_counts=1e-9)

    # Save
    save_pmf(fe_md_kde, xedges_kde, yedges_kde, f"{name}_KDE_PMF_MD.npz")
    save_pmf(fe_model_kde, xedges_kde, yedges_kde, f"{name}_KDE_PMF_diffusion.npz")

    # Align PMFS and mask for Histogram2D
    common_mask, md_only_mask, model_only_mask = align_pmfs_and_masks(fe_md_nopseudo, fe_model_nopseudo)

    mse_overlap = compute_mse_masked(fe_md_nopseudo, fe_model_nopseudo, common_mask)
    mse_complete = compute_mse(fe_md, fe_model)

    print(f"MSE (overlap only): {mse_overlap:.3f}")
    print(f"MSE (all space): {mse_complete:.3f}")

    # Save regions that do not overlap
    missing_info = pd.DataFrame({
        'md_only_bins': md_only_mask.flatten(),
        'model_only_bins': model_only_mask.flatten()
    })
    missing_info.to_csv(f"{name}_missing_bins.csv", index=False)

    # KDE difference for plotting
    fe_diff_kde = fe_md_kde - fe_model_kde
    fe_diff = fe_md - fe_model

    plot_kde(fe_md_kde, fe_model_kde, fe_diff_kde, xedges_kde, yedges_kde, name)
    plot_pmf(fe_md, fe_model, fe_diff, xedges, yedges, name)


# -----------------------------

if __name__ == '__main__':
    import glob
    md_files = glob.glob("/mnt/larry/lilian/DATA/Cory_data/1KGC/*xtc")
    #md_files = glob.glob("../bias/A6_bias_combined_traj.xtc")
    #model_files = glob.glob("../difusion/*.npyz")
    #model_files = glob.glob("../difusion/*_fixed.pdb")
    model_files = glob.glob("/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/output_vanilla_dig2/1KGC/dig_vanilla/*.pdb")
    topology = "/mnt/larry/lilian/DATA/Cory_data/1KGC/1KGC.pdb"
    xbins, ybins = 50, 50
    selected_residues_ranges = [[25,31], [48,54], [88,99], [137, 141], [159,164], [202,215]] # Residues for A6
    #selected_residues_ranges = [[26,32], [49,55], [89,99], [136, 141], [158,164], [201,212]] # Residue for DMF5
    #selected_residues_ranges = [[1,225]]
    selected_residues = [i for start, end in selected_residues_ranges for i in range(start, end)]
    name = "A6_adaptive"
    temperature = 300
    main(md_files, model_files, topology, xbins, ybins, selected_residues, name, temperature)

    # ToDo: Add reweighting using statistical weights for bias simulations.