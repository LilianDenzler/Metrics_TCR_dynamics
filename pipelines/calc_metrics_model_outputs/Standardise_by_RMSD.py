import numpy as np
import mdtraj as md
import os
from typing import List, Set, Tuple, Optional
import csv


def rmsd_calibration_factor(
    Z,
    traj_aligned,
    frame_indices=None,
    region_names=None,
    atom_names={"CA"},
    n_pairs=1000,
    random_state=0,
):
    """
    Fit a linear factor k such that

        RMSD_ij ≈ k * ||Z_i - Z_j||

    using random frame pairs from the GT trajectory.

    Parameters
    ----------
    Z : (n_frames_used, d) array
        Embedding coordinates (e.g. Zg from run_ca_dist).
    traj_aligned : TrajectoryView
        The aligned trajectory that corresponds to the embeddings
        (e.g. gt_traj_aligned).
    frame_indices : array-like or None
        Mapping from embedding rows to underlying mdtraj frame indices.
        If None, assumes 0..n_frames_used-1.
    region_names : list[str] or None
        Region(s) to use for RMSD (e.g. ["A_CDR1"]).
    atom_names : set[str]
        Atoms to use in RMSD (default {"CA"}).
    n_pairs : int
        Number of random frame pairs to sample.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    k : float
        Scale factor in Å / embedding_unit.
    corr : float
        Pearson correlation between embedding distance and RMSD
        for the sampled pairs (sanity check).
    """
    # unwrap TrajectoryView -> mdtraj.Trajectory for the region
    aligned_region_traj = traj_aligned.domain_subset(region_names, atom_names)

    Z = np.asarray(Z, dtype=float)
    n_embed = Z.shape[0]
    if n_embed < 2:
        raise ValueError("Need at least 2 embedded frames for calibration.")

    if frame_indices is None:
        frame_indices = np.arange(n_embed)
    frame_indices = np.asarray(frame_indices, dtype=int)

    if frame_indices.shape[0] != n_embed:
        raise ValueError("frame_indices length must match number of rows in Z.")

    rng = np.random.default_rng(random_state)

    # how many unique pairs are even possible?
    max_pairs = n_embed * (n_embed - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    # sample unique (i,j) with i<j
    pairs = set()
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n_embed, size=2)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        pairs.add((i, j))
    pairs = np.array(list(pairs), dtype=int)

    d_embed = np.empty(len(pairs), dtype=float)
    d_rmsd = np.empty(len(pairs), dtype=float)

    for k_idx, (i, j) in enumerate(pairs):
        fi = int(frame_indices[i])
        fj = int(frame_indices[j])

        # distance in embedding
        d_embed[k_idx] = np.linalg.norm(Z[i] - Z[j])

        # RMSD between frames fi and fj (nm -> Å)
        d_nm = md.rmsd(aligned_region_traj[fj], aligned_region_traj[fi])[0]
        d_rmsd[k_idx] = float(d_nm * 10.0)  # Å

    # avoid division by ~0
    mask = d_embed > 1e-8
    d_embed = d_embed[mask]
    d_rmsd = d_rmsd[mask]
    if d_embed.size == 0:
        raise ValueError("All embedding distances are ~0; cannot fit calibration factor.")

    # least-squares slope through origin: RMSD ≈ k * d_embed
    k = float(np.dot(d_embed, d_rmsd) / np.dot(d_embed, d_embed))

    # correlation (sanity check)
    if d_embed.size > 1:
        corr = float(np.corrcoef(d_embed, d_rmsd)[0, 1])
    else:
        corr = np.nan

    return k, corr


def _rmsd_fixed_frame_nm(xyz_i_nm: np.ndarray, xyz_j_nm: np.ndarray) -> float:
    """
    Fixed-frame RMSD between two frames (NO superposition).
    xyz_*_nm: (k,3) in nm
    returns RMSD in nm
    """
    diff = xyz_i_nm - xyz_j_nm  # (k,3)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def rmsd_calibration_factor_all_space(
    Zg_list: List[np.ndarray],
    gt_traj_aligned_list,
    region_names: List[str],
    atom_names: Set[str] = {"CA"},
    n_pairs: int = 5000,
    random_state: int = 0,
    out_csv: Optional[str] = None,
    return_center: bool = True,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    GLOBAL calibration for a shared embedding space.

    Instead of needing a global frame index mapping, we sample random pairs
    *within randomly chosen TCRs* and pool (embed_distance, RMSD) pairs.

    Inputs
    ------
    Zg_list : list of (n_i, d) arrays
        GT embeddings per TCR (in the shared reducer space).
    gt_traj_aligned_list : list of TrajectoryView
        Aligned GT trajectories (same alignment context as embeddings).
    region_names : list[str]
        Assessment region used for RMSD computation.
    atom_names : set[str]
        Atoms used for RMSD (default CA).
    n_pairs : int
        Total number of random pairs to sample across all TCRs.
    out_csv : str or None
        If provided, writes a single-row CSV log with k and corr.

    Returns
    -------
    k : float
        Scale factor so that k*||ΔZ|| ≈ RMSD(Å)
    corr : float
        Pearson correlation between ||ΔZ|| and RMSD over sampled pairs
    center_scaled : (d,) array or None
        Global center of scaled GT embeddings (for shared centering)
    """
    rng = np.random.default_rng(random_state)

    if len(Zg_list) == 0:
        raise ValueError("Zg_list is empty.")
    if len(Zg_list) != len(gt_traj_aligned_list):
        raise ValueError("Zg_list and gt_traj_aligned_list must have same length.")

    dims = [z.shape[1] for z in Zg_list if z is not None and z.size > 0]
    if len(dims) == 0:
        raise ValueError("No non-empty embeddings in Zg_list.")
    d = dims[0]
    if any(dd != d for dd in dims):
        raise ValueError("All Zg embeddings must have the same dimension.")

    # Precompute region-only trajectories for RMSD
    region_trajs = []
    valid_tcr = []
    for i, tv in enumerate(gt_traj_aligned_list):
        z = Zg_list[i]
        if z is None or z.shape[0] < 2:
            continue
        reg = tv.domain_subset(region_names, atom_names)
        if reg.n_frames < 2:
            continue
        # embeddings and reg traj must correspond in frame count (or at least support indexing)
        # If you subsampled embeddings upstream, you must pass consistent frame selection.
        if z.shape[0] > reg.n_frames:
            raise ValueError(
                f"TCR {i}: embeddings have {z.shape[0]} rows but region traj has {reg.n_frames} frames. "
                "If you subsampled embeddings, you must apply the same subsample to the trajectory for RMSD calibration."
            )
        region_trajs.append(reg)
        valid_tcr.append(i)

    if len(valid_tcr) == 0:
        raise ValueError("No valid TCRs (need at least 2 frames) for global RMSD calibration.")

    d_embed = []
    d_rmsd = []

    # sample pairs across random TCRs (within each chosen TCR)
    for _ in range(n_pairs):
        tcr_local = int(rng.integers(0, len(valid_tcr)))
        i_global = valid_tcr[tcr_local]
        Z = Zg_list[i_global]
        reg = region_trajs[tcr_local]

        n = Z.shape[0]
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        if a > b:
            a, b = b, a

        de = float(np.linalg.norm(Z[a] - Z[b]))
        if de <= 1e-8:
            continue

        # md.rmsd expects trajectories; compare single frames (nm -> Å)
        dr_nm = md.rmsd(reg[b], reg[a])[0]
        dr = float(dr_nm * 10.0)

        d_embed.append(de)
        d_rmsd.append(dr)

    if len(d_embed) < 10:
        raise ValueError("Too few valid (non-zero) pairs for calibration; increase n_pairs or check embeddings.")

    d_embed = np.asarray(d_embed, float)
    d_rmsd = np.asarray(d_rmsd, float)

    k = float(np.dot(d_embed, d_rmsd) / np.dot(d_embed, d_embed))
    corr = float(np.corrcoef(d_embed, d_rmsd)[0, 1]) if d_embed.size > 1 else np.nan

    center_scaled = None
    if return_center:
        Z_all = np.concatenate([z for z in Zg_list if z is not None and z.size > 0], axis=0)
        center_scaled = (Z_all * k).mean(axis=0)

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with out_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["region", "atoms", "n_pairs_used", "k_A_per_unit", "corr"])
            w.writerow(["+".join(region_names), "+".join(sorted(atom_names)), len(d_embed), k, corr])

    return k, corr, center_scaled