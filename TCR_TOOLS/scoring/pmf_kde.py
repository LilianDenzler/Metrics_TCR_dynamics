import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import os
from TCR_TOOLS.scoring.plotters import *
from scipy.spatial.distance import jensenshannon

# Constants
R_KJ_PER_MOL_K = 0.00831446261815324  # kJ/(mol*K)


def js_from_fe(fe_md, fe_model, temperature=300.0, base=2):
    """
    Compute Jensen–Shannon divergence between two PMFs defined on the same grid.

    fe_md, fe_model: 2D arrays of free energies (kJ/mol) on the same x/y grid
                     (e.g. from compute_pmf_hist with shared xedges, yedges).
    Returns: JS divergence (not distance), in `base` units (base=2 -> bits).
    """
    # Convert FE back to unnormalized probabilities:
    # fe = -kT log p  =>  p ∝ exp(-fe / kT)
    kT = R_KJ_PER_MOL_K * temperature

    p_md = np.exp(-fe_md / kT)
    p_model = np.exp(-fe_model / kT)

    # Mask any weird entries (inf/nan)
    mask = np.isfinite(p_md) & np.isfinite(p_model)
    p_md = p_md[mask].ravel()
    p_model = p_model[mask].ravel()

    # Avoid exact zeros, renormalize independently
    eps = 1e-12
    p_md = np.clip(p_md, eps, None)
    p_model = np.clip(p_model, eps, None)
    p_md /= p_md.sum()
    p_model /= p_model.sum()

    # SciPy's jensenshannon returns a distance, so square it for divergence
    js_dist = jensenshannon(p_md, p_model, base=base)
    js_div = js_dist ** 2
    return js_div


def limit_energies(fe_profile, max_ene):
    """
    Sets all infinite and high values to a predefined max value
    """
    for jy in range(len(fe_profile[0,:])):
        for jx in range(len(fe_profile[:,0])):
            if np.isinf(fe_profile[jx,jy]) or fe_profile[jx,jy] > max_ene:
                fe_profile[jx,jy]=max_ene
    return fe_profile

def compute_pmf_hist(proj, xedges=None, yedges=None, temperature=300.0, pseudo_counts=1e-9, max_ene=None):
    """
    Computes a PMF from a projection trough a histogram approach.
    """
    x, y = proj[:,0], proj[:,1]

    occupancy, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    prob = occupancy.T + pseudo_counts  # pseudo-count to avoid 0 probabilities
    prob /= prob.sum()
    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob)
    #fe -= np.nanmin(fe)
    if max_ene:
        fe = limit_energies(fe, max_ene)

    return fe, xedges, yedges

def compute_pmf_kde(
    proj,
    xgrid=None, ygrid=None,
    xedges=None, yedges=None,
    temperature=300.0, pseudo_counts=1e-9, max_ene=None, bandwidth='scott'
):
    x, y = proj[:,0], proj[:,1]
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
    #kde.set_bandwidth(bw_method=kde.factor * 0.9)

    X, Y = np.meshgrid(xgrid, ygrid)
    density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    prob = density * dx * dy
    prob /= prob.sum()

    kT = R_KJ_PER_MOL_K * temperature
    fe = -kT * np.log(prob + pseudo_counts)
    #fe -= np.nanmin(fe)
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
# --------------------------------------
# Highest-density points (no clustering)
# --------------------------------------
def identify_high_density_points(
    proj_md,
    proj_model,
    tv_gt,
    tv_pred,
    outfolder,
    n_cluster=200,  # kept for API compatibility, but not used anymore
    n_dirs=10,
    use_extremes=True,
):
    """
    Identify:
      - global highest-density point (KDE over sample points) for MD and model,
      - a set of extreme points chosen over *all* frames using farthest-point
        sampling (spread-out extremes, not restricted to any cluster),
      - the model point closest to the MD highest-density point.

    Returns:
        gt_points        : list of (x, y) for MD selected points
                           [0] = peak; [1:] = extremes
        model_points     : list of (x, y) for model selected points
                           [0] = peak; [1: -1] = extremes; last = nearest-to-GT-peak
        gt_initial_point : (x0, y0) for proj_md[0]
        model_initial_point : (x0, y0) for proj_model[0] or None
        gt_point_ids     : list of frame indices aligned with gt_points
        model_point_ids  : list of frame indices aligned with model_points

    Side effects (if tv_gt / tv_pred not None):
        Saves PDBs:
          - gt_initial_frame0.pdb
          - gt_peak_highdensity_frame{idx}.pdb
          - gt_extreme_frame{idx}.pdb  (for each extreme)
          - model_initial_frame0.pdb
          - model_peak_highdensity_frame{idx}.pdb
          - model_extreme_frame{idx}.pdb
          - model_nearest_to_gt_peak_frame{idx}.pdb
    """
    os.makedirs(outfolder, exist_ok=True)

    # ---- helper: peak + spread-out extremes via farthest-point sampling ----
    def compute_peak_and_extremes(coords, n_dirs):
        """
        coords: array shape (2, N)

        Returns:
          peak_idx: int
          peak_point: (2,)
          extreme_indices: list[int]
              - first extreme = point farthest from peak
              - additional extremes chosen by farthest-point sampling
                to be spread out in space.
        """
        N = coords.shape[1]
        if N == 0:
            return None, None, []

        # KDE over sample points to get density at each frame
        kde = gaussian_kde(coords)
        densities = kde(coords)  # length N

        # global max-density point (peak)
        peak_idx = int(np.argmax(densities))
        peak_point = coords[:, peak_idx]

        # if we don't want extremes or there aren't enough points
        if use_extremes is False:
            return peak_idx, peak_point, []
        if N == 1 or n_dirs <= 0:
            return peak_idx, peak_point, []

        indices = np.arange(N)

        # ---- first extreme: farthest from peak over ALL frames (except the peak) ----
        mask_not_peak = indices != peak_idx
        candidates = indices[mask_not_peak]
        if candidates.size == 0:
            return peak_idx, peak_point, []

        # distances from peak
        diffs = coords[:, candidates] - peak_point[:, None]  # (2, M)
        dists = np.linalg.norm(diffs, axis=0)                # (M,)
        first_ext_idx = int(candidates[np.argmax(dists)])

        extremes = [first_ext_idx]

        # if we only want one extreme or only 2 frames in total, stop here
        if N <= 2 or n_dirs <= 1:
            return peak_idx, peak_point, extremes

        # ---- additional extremes via farthest-point sampling ----
        selected = list(extremes)
        candidate_list = [i for i in indices if i not in selected and i != peak_idx]

        # we want up to n_dirs extremes total
        target_num = min(n_dirs, 1 + len(candidate_list))  # 1 already in extremes

        while len(selected) < target_num and candidate_list:
            best_cand = None
            best_score = -1.0

            for idx in candidate_list:
                # distance from this candidate to each already selected extreme
                v = coords[:, idx][:, None] - coords[:, selected]  # (2, len(selected))
                d = np.linalg.norm(v, axis=0)
                score = float(d.min())  # candidate's distance to nearest selected extreme

                if score > best_score:
                    best_score = score
                    best_cand = idx

            selected.append(best_cand)
            candidate_list.remove(best_cand)

        extreme_indices = selected
        return peak_idx, peak_point, extreme_indices

    # ---- MD side ----
    gt_initial_point = proj_md[0, :]
    gt_points = []
    gt_point_ids = []
    model_points = None
    model_point_ids = None
    model_initial_point = None

    x_md, y_md = proj_md[:, 0], proj_md[:, 1]
    coords_md = np.vstack([x_md, y_md])  # (2, N_md)
    peak_idx_md, gt_peak_point, extreme_idx_md = compute_peak_and_extremes(
        coords_md, n_dirs=n_dirs
    )

    # collect MD-selected points for plotting:
    # first entry in gt_points will be the peak; rest are extremes
    if gt_peak_point is not None:
        gt_points = [gt_peak_point] + [coords_md[:, i] for i in extreme_idx_md]
        gt_point_ids = [peak_idx_md] + list(extreme_idx_md)

    # save PDBs for MD
    if tv_gt is not None:
        # initial frame
        tv_gt._traj[0].save_pdb(
            os.path.join(outfolder, "gt_initial_frame0.pdb")
        )
        # highest-density frame
        if peak_idx_md is not None:
            tv_gt._traj[peak_idx_md].save_pdb(
                os.path.join(
                    outfolder, f"gt_peak_highdensity_frame{peak_idx_md}.pdb"
                )
            )
        # extremes
        for i in extreme_idx_md:
            tv_gt._traj[i].save_pdb(
                os.path.join(
                    outfolder,
                    f"gt_extreme_frame{i}.pdb",
                )
            )

    # ---- Model side ----
    if proj_model is not None and proj_model.shape[0] > 0:
        model_initial_point = proj_model[0, :]

        x_model, y_model = proj_model[:, 0], proj_model[:, 1]
        coords_model = np.vstack([x_model, y_model])  # (2, N_model)
        peak_idx_model, model_peak_point, extreme_idx_model = compute_peak_and_extremes(
            coords_model, n_dirs=n_dirs
        )

        model_points = []
        model_point_ids = []
        if model_peak_point is not None:
            model_points = [model_peak_point] + [coords_model[:, i] for i in extreme_idx_model]
            model_point_ids = [peak_idx_model] + list(extreme_idx_model)

        # nearest model point to highest-density MD point
        nearest_idx_model = None
        if gt_peak_point is not None:
            coords_model_T = coords_model.T  # (N, 2)
            diff = coords_model_T - gt_peak_point  # broadcast; shape (N, 2)
            dists = np.linalg.norm(diff, axis=1)
            nearest_idx_model = int(np.argmin(dists))
            # add this to model_points as well (last entry)

        # save PDBs for model
        if tv_pred is not None:
            # initial frame
            tv_pred._traj[0].save_pdb(
                os.path.join(outfolder, "model_initial_frame0.pdb")
            )
            # highest-density frame
            if peak_idx_model is not None:
                tv_pred._traj[peak_idx_model].save_pdb(
                    os.path.join(
                        outfolder,
                        f"model_peak_highdensity_frame{peak_idx_model}.pdb",
                    )
                )
            # extremes
            for i in extreme_idx_model:
                tv_pred._traj[i].save_pdb(
                    os.path.join(
                        outfolder,
                        f"model_extreme_frame{i}.pdb",
                    )
                )
            # nearest to MD peak
            if nearest_idx_model is not None:
                tv_pred._traj[nearest_idx_model].save_pdb(
                    os.path.join(
                        outfolder,
                        f"model_nearest_to_gt_peak_frame{nearest_idx_model}.pdb",
                    )
                )

    return (
        gt_points,
        model_points,
        gt_initial_point,
        model_initial_point,
        gt_point_ids,
        model_point_ids,
    )

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
        yedges=yedges,
    )
    print(f"PMF saved to {filename}")



def oriol_analysis(
    xbins,
    ybins,
    proj_md,
    proj_model,
    temperature: float,
    name: str,
    outfolder: str,
    tv_gt,
    tv_pred,
    use_extremes=False,
    global_range=None,
):
    # Histogram calculation
    nbins = (xbins, ybins)
    x_all = np.concatenate([proj_md[:, 0], proj_model[:, 0]])
    y_all = np.concatenate([proj_md[:, 1], proj_model[:, 1]])
    if global_range is not None:
        xmin, xmax, ymin, ymax = global_range
    else:
        xmin=x_all.min()
        xmax=x_all.max()
        ymin=y_all.min()
        ymax=y_all.max()
    xedges = np.linspace(xmin, xmax, nbins[0]+1)
    yedges = np.linspace(ymin, ymax, nbins[1]+1)
    xgrid = np.linspace(xmin, xmax, nbins[0])
    ygrid = np.linspace(ymin, ymax, nbins[1])

    # Histograms without pseudocounts to find non overlapping regions
    fe_md_nopseudo, xedges_nopseudo, yedges_nopseudo = compute_pmf_hist(proj_md,xedges=xedges, yedges=yedges,temperature=temperature, pseudo_counts=0)
    fe_model_nopseudo, _, _ = compute_pmf_hist(proj_model, xedges=xedges, yedges=yedges, temperature=temperature, pseudo_counts=0)

    # Histogram with pseudocounts and max energy
    max_ene = np.max(fe_md_nopseudo[np.isfinite(fe_md_nopseudo)]) + (R_KJ_PER_MOL_K * temperature) # Maximum seen energy + kbt
    fe_md, xedges, yedges = compute_pmf_hist(proj_md, xedges=xedges, yedges=yedges,temperature=temperature, pseudo_counts=1e-9, max_ene=max_ene)
    fe_model, _, _ = compute_pmf_hist(proj_model, xedges=xedges, yedges=yedges,temperature=temperature, pseudo_counts=1e-9, max_ene=max_ene)

    # KDE PMF calculation for smooth visualization
    fe_md_kde, xedges_kde, yedges_kde = compute_pmf_kde(proj_md,  xgrid=xgrid, ygrid=ygrid, xedges=xedges, yedges=yedges,temperature=temperature, pseudo_counts=1e-9)
    fe_model_kde, _, _ = compute_pmf_kde(proj_model,  xgrid=xgrid, ygrid=ygrid, xedges=xedges, yedges=yedges, temperature=temperature, pseudo_counts=1e-9)
    fe_md_kde_nopseudo, xedges_kde_nopseudo, yedges_kde_nopseudo = compute_pmf_kde(proj_md,  xgrid=xgrid, ygrid=ygrid, xedges=xedges, yedges=yedges,temperature=temperature, pseudo_counts=0)
    fe_model_kde_nopseudo, _, _ = compute_pmf_kde(proj_model,  xgrid=xgrid, ygrid=ygrid, xedges=xedges, yedges=yedges, temperature=temperature, pseudo_counts=0)



    # Save
    save_pmf(fe_md, xedges, yedges, os.path.join(outfolder, f"{name}_histogram_MD.npz"))
    save_pmf(fe_model,xedges,yedges,os.path.join(outfolder, f"{name}_histogram_diffusion.npz"))
    save_pmf(fe_md_kde,xedges_kde,yedges_kde,os.path.join(outfolder, f"{name}_KDE_PMF_MD.npz"))
    save_pmf(fe_model_kde,xedges_kde,yedges_kde,os.path.join(outfolder, f"{name}_KDE_PMF_diffusion.npz"))

    # --- JSD from histogram PMFs ---
    js_hist = js_from_fe(fe_md, fe_model, temperature=temperature, base=2)
    print(f"Jensen–Shannon divergence (hist PMF, base=2): {js_hist:.4f} bits")

    # Optionally save to a txt file
    js_path = os.path.join(outfolder, f"JSD_histogram.txt")
    with open(js_path, "w") as f:
        f.write(f"Jensen–Shannon divergence (hist PMF, base=2): {js_hist:.6f}\n")
        js_kde = js_from_fe(fe_md_kde, fe_model_kde, temperature=temperature, base=2)
    print(f"Jensen–Shannon divergence (KDE PMF, base=2): {js_kde:.4f} bits")

    js_kde_path = os.path.join(outfolder, f"JSD_KDE.txt")
    with open(js_kde_path, "w") as f:
        f.write(f"Jensen–Shannon divergence (KDE PMF, base=2): {js_kde:.6f}\n")

    # Align PMFS and mask for Histogram2D
    common_mask, md_only_mask, model_only_mask = align_pmfs_and_masks(fe_md_kde_nopseudo, fe_model_kde_nopseudo)



    #mse_overlap = compute_mse_masked(fe_md_nopseudo, fe_model_nopseudo, common_mask)
    #mse_complete = compute_mse(fe_md, fe_model)

    #print(f"MSE (overlap only): {mse_overlap:.3f}")
    #print(f"MSE (all space): {mse_complete:.3f}")

    # Save regions that do not overlap
    #missing_info = pd.DataFrame({"md_only_bins": md_only_mask.flatten(),"model_only_bins": model_only_mask.flatten()})
    #missing_info.to_csv(os.path.join(outfolder, f"{name}_missing_bins.csv"),index=False)

    # KDE difference for plotting



    #fe_diff_kde = (fe_md_kde - np.nanmax(fe_md_kde)) - (fe_model_kde - np.nanmax(fe_model_kde))
    #fe_diff= (fe_md - np.nanmax(fe_md)) - (fe_model - np.nanmax(fe_model))
    fe_diff_kde=fe_md_kde- fe_model_kde
    fe_diff=fe_md - fe_model
    fe_md_kde -= np.nanmin(fe_md_kde)
    fe_model_kde -= np.nanmin(fe_model_kde)
    fe_md -= np.nanmin(fe_md)
    fe_model -= np.nanmin(fe_model)


    (gt_points,
        model_points,
        gt_initial_point,
        model_initial_point,
        gt_point_ids,
        model_point_ids,
    ) = identify_high_density_points(proj_md, proj_model, tv_gt, tv_pred, outfolder, use_extremes=use_extremes)

    plot_kde(
        fe_md_kde,
        fe_model_kde,
        fe_diff_kde,
        xedges_kde,
        yedges_kde,
        os.path.join(outfolder, "plot_kde_withhighest.png"),
        gt_points=gt_points,
        model_points=model_points,
        gt_initial=gt_initial_point,
        model_initial=model_initial_point,
        gt_point_ids=gt_point_ids,
        model_point_ids=model_point_ids,
    )

    plot_pmf(
        fe_md,
        fe_model,
        fe_diff,
        xedges,
        yedges,
        os.path.join(outfolder, "plot_pmf_withhighest.png"),
        gt_points=gt_points,
        model_points=model_points,
        gt_initial=gt_initial_point,
        model_initial=model_initial_point,
        gt_point_ids=gt_point_ids,
        model_point_ids=model_point_ids,
    )

    # plain versions without the peak markers
    plot_kde(
        fe_md_kde,
        fe_model_kde,
        fe_diff_kde,
        xedges_kde,
        yedges_kde,
        os.path.join(outfolder, "plot_kde.png"),
        gt_initial=gt_initial_point,
        model_initial=model_initial_point,
    )

    plot_pmf(
        fe_md,
        fe_model,
        fe_diff,
        xedges,
        yedges,
        os.path.join(outfolder, "plot_pmf.png"),
        gt_initial=gt_initial_point,
        model_initial=model_initial_point,
    )
