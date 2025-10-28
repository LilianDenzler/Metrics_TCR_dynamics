import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import os
from TCR_TOOLS.scoring.plotters import *
# Constants
R_KJ_PER_MOL_K = 0.00831446261815324  # kJ/(mol*K)



def limit_energies(fe_profile, max_ene):
    """
    Sets all infinite and high values to a predefined max value
    """
    for jy in range(len(fe_profile[0,:])):
        for jx in range(len(fe_profile[:,0])):
            if np.isinf(fe_profile[jx,jy]) or fe_profile[jx,jy] > max_ene:
                fe_profile[jx,jy]=max_ene
    return fe_profile

def compute_pmf_hist(proj, nbins=(100,100), temperature=300.0,
                     pseudo_counts=1e-9, max_ene=None, xedges=None, yedges=None):
    x, y = proj[:,0], proj[:,1]
    if xedges is None: xedges = np.linspace(x.min(), x.max(), nbins[0]+1)
    if yedges is None: yedges = np.linspace(y.min(), y.max(), nbins[1]+1)
    occupancy, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    prob = occupancy.T + pseudo_counts
    prob /= prob.sum()
    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob)
    fe -= np.nanmin(fe)
    if max_ene:
        fe = limit_energies(fe, max_ene)
    return fe, xedges, yedges

def compute_pmf_kde(proj, nbins=(100,100), temperature=300.0,
                    pseudo_counts=1e-9, max_ene=None, bandwidth=None,
                    xedges=None, yedges=None, bw_adjust=1.0):
    x, y = proj[:,0], proj[:,1]

    # shared grid centers from shared edges if provided
    if xedges is None:
        xedges = np.linspace(x.min(), x.max(), nbins[0]+1)
    if yedges is None:
        yedges = np.linspace(y.min(), y.max(), nbins[1]+1)
    xgrid = 0.5*(xedges[:-1] + xedges[1:])
    ygrid = 0.5*(yedges[:-1] + yedges[1:])

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    if bw_adjust != 1.0:
        kde.set_bandwidth(bw_method=kde.factor * bw_adjust)  # narrower if <1

    X, Y = np.meshgrid(xgrid, ygrid, indexing='xy')
    density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # normalize to probability on the grid
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    prob = density * dx * dy
    prob /= prob.sum()

    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob + pseudo_counts)
    fe -= np.nanmin(fe)
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


def oriol_analysis(xbins, ybins, proj_md, proj_model, temperature: float, name: str, outfolder: str):
    # Histogram calculation
    nbins = (xbins, ybins)
    x_all = np.concatenate([proj_md[:,0], proj_model[:,0]])
    y_all = np.concatenate([proj_md[:,1], proj_model[:,1]])
    xedges = np.linspace(x_all.min(), x_all.max(), nbins[0]+1)
    yedges = np.linspace(y_all.min(), y_all.max(), nbins[1]+1)

    # Histograms without pseudocounts to find non overlapping regions
    fe_md_nopseudo, _, _ = compute_pmf_hist(proj_md, nbins, temperature, pseudo_counts=0,xedges=xedges, yedges=yedges)
    fe_model_nopseudo, _, _ = compute_pmf_hist(proj_model, nbins, temperature, pseudo_counts=0,xedges=xedges, yedges=yedges)

    # Histogram with pseudocounts and max energy
    max_ene = np.max(fe_md_nopseudo[np.isfinite(fe_md_nopseudo)]) + (R_KJ_PER_MOL_K * temperature) # Maximum seen energy + kbt
    fe_md, xedges, yedges = compute_pmf_hist(proj_md, nbins, temperature, pseudo_counts=1e-9, max_ene=max_ene,xedges=xedges, yedges=yedges)
    fe_model, _, _ = compute_pmf_hist(proj_model, nbins, temperature, pseudo_counts=1e-9, max_ene=max_ene,xedges=xedges, yedges=yedges)

    # Save
    save_pmf(fe_md, xedges, yedges, os.path.join(outfolder,f"{name}_histogram_MD.npz"))
    save_pmf(fe_model, xedges, yedges, os.path.join(outfolder,f"{name}_histogram_diffusion.npz"))

    # KDE PMF calculation for smooth visualization
    fe_md_kde, xedges_kde, yedges_kde = compute_pmf_kde(proj_md, nbins, temperature, pseudo_counts=1e-9,xedges=xedges, yedges=yedges)
    fe_model_kde, _, _ = compute_pmf_kde(proj_model, nbins, temperature, pseudo_counts=1e-9,xedges=xedges, yedges=yedges)

    # Save
    save_pmf(fe_md_kde, xedges_kde, yedges_kde, os.path.join(outfolder,f"{name}_KDE_PMF_MD.npz"))
    save_pmf(fe_model_kde, xedges_kde, yedges_kde, os.path.join(outfolder,f"{name}_KDE_PMF_diffusion.npz"))

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
    missing_info.to_csv(os.path.join(outfolder,f"{name}_missing_bins.csv"), index=False)

    # KDE difference for plotting
    fe_diff_kde = fe_md_kde - fe_model_kde
    fe_diff = fe_md - fe_model

    plot_kde(fe_md_kde, fe_model_kde, fe_diff_kde, xedges_kde, yedges_kde, os.path.join(outfolder,"plot_kde.png"))
    plot_pmf(fe_md, fe_model, fe_diff, xedges, yedges, os.path.join(outfolder,"plot_pmf.png"))
