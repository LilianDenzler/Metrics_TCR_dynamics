from sklearn.manifold import trustworthiness
from typing import Optional, Tuple, Dict
import numpy as np
import numpy as np
import mdtraj as md
from typing import Optional, Dict, Tuple
from scipy.spatial.distance import pdist, squareform, cdist


#------------ trustworthiness ------------
def trustworthiness_from_embedding(
    X_high: np.ndarray,
    Z_low:  np.ndarray,
    n_neighbors: int = 10,
    subsample: Optional[int] = None,
    random_state: Optional[int] = 0,
    metric: str = "euclidean",
) -> float:
    """
    Compute sklearn.manifold.trustworthiness between the original space (X_high)
    and a low-dimensional embedding (Z_low).

    Parameters
    ----------
    X_high : (n_samples, n_features) original data (e.g., flattened coords)
    Z_low  : (n_samples, n_components) embedding (e.g., PCA/kPCA/t-SNE/etc)
    n_neighbors : neighborhood size (typical: 5–30)
    subsample : if set, evaluate on a random subset of this many samples
    random_state : RNG seed for reproducibility (when subsampling)
    metric : distance metric for neighborhoods in X_high and Z_low

    Returns
    -------
    tw : float in [0, 1]; higher = better neighborhood preservation
    """
    if X_high.shape[0] != Z_low.shape[0]:
        raise ValueError("X_high and Z_low must have the same number of rows (samples).")

    if subsample is not None and subsample < X_high.shape[0]:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_high.shape[0], size=subsample, replace=False)
        Xh = X_high[idx]
        Zl = Z_low[idx]
    else:
        Xh, Zl = X_high, Z_low

    return trustworthiness(Xh, Zl, n_neighbors=n_neighbors, metric=metric)

def _tv_to_X(tv, region_names, atom_names) -> np.ndarray:
    idx = tv.domain_idx(region_names=region_names, atom_names=atom_names)
    # flatten (frames, atoms*3); coords are in nm (unit doesn’t matter for trustworthiness)
    return tv.mdtraj.xyz[:, idx, :].reshape(tv.mdtraj.n_frames, -1)

def trustworthiness_gt_pred(
    tv_gt,
    tv_pred,
    Z_gt: np.ndarray,
    Z_pred: np.ndarray,
    region_names,
    atom_names,
    n_neighbors: int = 10,
    subsample: Optional[int] = None,
    random_state: Optional[int] = 0,
    metric: str = "euclidean",
) -> Dict[str, float]:
    """
    Compute trustworthiness separately for GT and Pred under the same selection.

    Returns
    -------
    dict with keys 'gt' and 'pred'
    """
    X_gt   = _tv_to_X(tv_gt,   region_names, atom_names)
    X_pred = _tv_to_X(tv_pred, region_names, atom_names)

    tw_gt   = trustworthiness_from_embedding(X_gt,   Z_gt,   n_neighbors, subsample, random_state, metric)
    tw_pred = trustworthiness_from_embedding(X_pred, Z_pred, n_neighbors, subsample, random_state, metric)
    return {"gt": tw_gt, "pred": tw_pred}


# ---------- Mantel test between RMSD and embedding distances ----------
def _mantel_test(D1: np.ndarray,
                 D2: np.ndarray,
                 method: str = "pearson",
                 permutations: int = 9999,
                 random_state: int = 0) -> Tuple[float, float]:
    """
    Mantel correlation between two square, symmetric distance matrices.
    Returns (r, p_value).
    """
    if D1.shape != D2.shape or D1.shape[0] != D1.shape[1]:
        raise ValueError("D1 and D2 must be square distance matrices of the same shape.")
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    x = D1[iu].astype(float)
    y = D2[iu].astype(float)

    # correlation
    if method == "pearson":
        def corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            denom = np.sqrt((a*a).sum() * (b*b).sum())
            return 0.0 if denom == 0 else float((a*b).sum() / denom)
    elif method == "spearman":
        from scipy.stats import rankdata
        def corr(a, b):
            ra, rb = rankdata(a), rankdata(b)
            ra = ra - ra.mean(); rb = rb - rb.mean()
            denom = np.sqrt((ra*ra).sum() * (rb*rb).sum())
            return 0.0 if denom == 0 else float((ra*rb).sum() / denom)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    r_obs = corr(x, y)

    # permutation null by permuting labels (rows/cols) of D2
    rng = np.random.default_rng(random_state)
    greater = 0
    for _ in range(int(permutations)):
        p = rng.permutation(n)
        y_perm = D2[p][:, p][iu]
        if corr(x, y_perm) >= r_obs:
            greater += 1
    p_value = (greater + 1) / (permutations + 1)  # add-one smoothing
    return r_obs, p_value


# ---------- RMSD helpers ----------
def _pairwise_rmsd(traj: md.Trajectory,
                   atom_indices: np.ndarray) -> np.ndarray:
    """Full pairwise RMSD matrix (Å) for one trajectory on a given atom selection."""
    n = traj.n_frames
    M = np.empty((n, n), dtype=float)
    for i in range(n):
        # md.rmsd returns nm; compare all frames to reference frame=i
        M[:, i] = md.rmsd(traj, traj, frame=i,
                          atom_indices=atom_indices) * 10.0
    # symmetrize numerically
    return 0.5 * (M + M.T)

def _cross_rmsd(trajA: md.Trajectory, trajB: md.Trajectory,
                idxA: np.ndarray, idxB: np.ndarray) -> np.ndarray:
    """Cross RMSD matrix (Å): rows=B frames, cols=A frames."""
    nA, nB = trajA.n_frames, trajB.n_frames
    C = np.empty((nB, nA), dtype=float)
    for i in range(nA):
        # RMSD of all B frames to A's frame i (supports differing atom lists via ref_atom_indices)
        C[:, i] = md.rmsd(trajB, trajA, frame=i,
                          atom_indices=idxB,
                          ref_atom_indices=idxA) * 10.0
    return C  # shape (nB, nA)


# ---------- Main API ----------
def mantel_rmsd_vs_embedding(
    tv_gt, tv_pred,
    Z_gt: np.ndarray, Z_pred: np.ndarray,
    region_names,
    atom_names={"CA"},
    method: str = "pearson",
    permutations: int = 9999,
    random_state: int = 0,
    subsample_gt: Optional[int] = None,
    subsample_pred: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare structural RMSD distances to embedding distances with Mantel tests.

    Returns a dict with three entries 'gt', 'pred', 'combined', each containing:
        {'r': mantel_correlation, 'p': p_value}

    Notes
    -----
    - Uses same region/atom selection on GT and Pred; the atom counts must match.
    - For speed, set subsample_* to limit frames (random subset).
    """
    # 0) Select atoms (must match in count)
    idx_gt   = np.asarray(tv_gt.domain_idx(region_names=region_names, atom_names=atom_names), dtype=int)
    idx_pred = np.asarray(tv_pred.domain_idx(region_names=region_names, atom_names=atom_names), dtype=int)
    if len(idx_gt) != len(idx_pred):
        raise ValueError(f"Atom selection mismatch: GT={len(idx_gt)} vs Pred={len(idx_pred)}.")

    # 1) Optional frame subsampling (keep same order for embeddings)
    rng = np.random.default_rng(random_state)
    n_gt, n_pr = tv_gt.mdtraj.n_frames, tv_pred.mdtraj.n_frames

    if subsample_gt is not None and subsample_gt < n_gt:
        sel_gt = np.sort(rng.choice(n_gt, size=subsample_gt, replace=False))
    else:
        sel_gt = np.arange(n_gt)

    if subsample_pred is not None and subsample_pred < n_pr:
        sel_pr = np.sort(rng.choice(n_pr, size=subsample_pred, replace=False))
    else:
        sel_pr = np.arange(n_pr)

    traj_gt   = tv_gt.mdtraj[sel_gt]
    traj_pred = tv_pred.mdtraj[sel_pr]
    Zg = Z_gt[sel_gt]
    Zp = Z_pred[sel_pr]

    # 2) Structural distance matrices (Å)
    Dg = _pairwise_rmsd(traj_gt,   idx_gt)
    Dp = _pairwise_rmsd(traj_pred, idx_pred)

    # combined block
    C  = _cross_rmsd(traj_gt, traj_pred, idx_gt, idx_pred)  # (n_pr, n_gt)
    # assemble:
    #   [ Dg        C.T ]
    #   [ C         Dp  ]
    Dall = np.block([[Dg, C.T],
                     [C,  Dp]])

    # 3) Embedding distance matrices (Euclidean in embedding space)
    Eg   = squareform(pdist(Zg, metric="euclidean"))
    Ep   = squareform(pdist(Zp, metric="euclidean"))
    Ec12 = cdist(Zp, Zg, metric="euclidean")  # cross
    Eall = np.block([[Eg,  Ec12.T],
                     [Ec12, Ep]])

    # 4) Mantel tests
    r_g,  p_g  = _mantel_test(Dg,   Eg,   method=method, permutations=permutations, random_state=random_state)
    r_p,  p_p  = _mantel_test(Dp,   Ep,   method=method, permutations=permutations, random_state=random_state)
    r_all,p_all= _mantel_test(Dall, Eall, method=method, permutations=permutations, random_state=random_state)

    return {
        "gt":       {"r": r_g,   "p": p_g},
        "pred":     {"r": r_p,   "p": p_p},
        "combined": {"r": r_all, "p": p_all},
    }
