import numpy as np
import mdtraj as md


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
