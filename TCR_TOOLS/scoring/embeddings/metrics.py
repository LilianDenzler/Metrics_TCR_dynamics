import numpy as np
import mdtraj as md
from typing import Optional, Tuple, Dict
from sklearn.manifold import trustworthiness as sk_trust
from scipy.spatial.distance import pdist, squareform, cdist

def trustworthiness(X, Z, n_neighbors=10, subsample: Optional[int]=None, seed=0) -> float:
    if subsample and subsample < len(X):
        rs = np.random.default_rng(seed)
        idx = rs.choice(len(X), size=subsample, replace=False)
        X, Z = X[idx], Z[idx]
    return float(sk_trust(X, Z, n_neighbors=n_neighbors, metric="euclidean"))

def mantel(D1: np.ndarray, D2: np.ndarray, method="spearman", permutations=9999, seed=0) -> Tuple[float,float]:
    if D1.shape != D2.shape or D1.shape[0] != D1.shape[1]:
        raise ValueError("Mantel expects same-shaped square distance matrices.")
    n = D1.shape[0]; iu = np.triu_indices(n, 1)
    x = D1[iu].astype(float); y = D2[iu].astype(float)

    if method == "pearson":
        def corr(a,b):
            a=a-a.mean(); b=b-b.mean()
            den = np.sqrt((a*a).sum()*(b*b).sum()); return 0.0 if den==0 else float((a*b).sum()/den)
    else:
        from scipy.stats import rankdata
        def corr(a,b):
            ra, rb = rankdata(a), rankdata(b)
            ra=ra-ra.mean(); rb=rb-rb.mean()
            den = np.sqrt((ra*ra).sum()*(rb*rb).sum()); return 0.0 if den==0 else float((ra*rb).sum()/den)

    r_obs = corr(x,y)
    rs = np.random.default_rng(seed); ge=0
    for _ in range(int(permutations)):
        p = rs.permutation(n)
        y_perm = D2[p][:,p][iu]
        if corr(x,y_perm) >= r_obs: ge += 1
    pval = (ge+1)/(permutations+1)
    return r_obs, pval

def rmsd_matrix(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    n = traj.n_frames
    M = np.empty((n,n), float)
    for i in range(n):
        M[:, i] = md.rmsd(traj, traj, frame=i, atom_indices=atom_indices) * 10.0
    return 0.5*(M+M.T)

def mantel_rmsd_vs_embedding(
    tv_gt,
    tv_pred,
    Z_gt: np.ndarray,
    Z_pred: Optional[np.ndarray],
    region_names,
    atom_names={"CA"},
    method: str = "spearman",
    permutations: int = 9999,
    seed: int = 0,
    subsample_gt: Optional[int] = None,
    subsample_pred: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Mantel correlations between RMSD matrices and embedding distances.

    Supports:
      - GT-only: pass tv_pred=None and/or Z_pred=None → returns {"gt": {...}}
      - GT+Pred: as before → returns {"gt": {...}, "pred": {...}, "combined": {...}}
    """
    rs = np.random.default_rng(seed)

    # --- GT selection and subsampling ---
    idx_gt = np.asarray(tv_gt.domain_idx(region_names=region_names,
                                         atom_names=atom_names), int)
    Tgt = tv_gt.mdtraj

    if subsample_gt is None:
        Igt = np.arange(Tgt.n_frames)
    else:
        Igt = np.sort(rs.choice(Tgt.n_frames, subsample_gt, replace=False))

    Tgt = Tgt[Igt]
    Zg = Z_gt[Igt]

    # RMSD distances in structure space (GT)
    Dg = rmsd_matrix(Tgt, idx_gt)
    # Distances in embedding space (GT)
    Eg = squareform(pdist(Zg))

    # Mantel GT
    rg, pg = mantel(Dg, Eg, method, permutations, seed)
    out: Dict[str, Dict[str, float]] = {"gt": {"r": rg, "p": pg}}

    # --- If no pred provided, stop here ---
    has_pred = (tv_pred is not None) and (Z_pred is not None)
    if not has_pred:
        return out

    # --- Pred selection and subsampling ---
    idx_pr = np.asarray(tv_pred.domain_idx(region_names=region_names,
                                           atom_names=atom_names), int)
    if len(idx_gt) != len(idx_pr):
        raise ValueError("Atom selection mismatch for RMSD Mantel (GT vs Pred).")

    Tpr = tv_pred.mdtraj
    if subsample_pred is None:
        Ipr = np.arange(Tpr.n_frames)
    else:
        Ipr = np.sort(rs.choice(Tpr.n_frames, subsample_pred, replace=False))

    Tpr = Tpr[Ipr]
    Zp = Z_pred[Ipr]

    # RMSD distances in structure space (Pred)
    Dp = rmsd_matrix(Tpr, idx_pr)

    # Cross RMSD between GT and Pred
    C = np.empty((Tpr.n_frames, Tgt.n_frames), float)
    for i in range(Tgt.n_frames):
        # md.rmsd uses angstrom; multiply by 10.0 if you want nm→Å or vice versa
        C[:, i] = md.rmsd(Tpr, Tgt, frame=i,
                          atom_indices=idx_pr,
                          ref_atom_indices=idx_gt) * 10.0

    # Combined structural-distance matrix
    Dall = np.block([[Dg, C.T],
                     [C,  Dp]])

    # Embedding distances (Pred + combined)
    Ep   = squareform(pdist(Zp))
    Ec12 = cdist(Zp, Zg)
    Eall = np.block([[Eg,   Ec12.T],
                     [Ec12, Ep]])

    # Mantel Pred + combined
    rp, pp = mantel(Dp, Ep, method, permutations, seed)
    ra, pa = mantel(Dall, Eall, method, permutations, seed)

    return {"gt":{"r":rg,"p":pg},"pred":{"r":rp,"p":pp},"combined":{"r":ra,"p":pa}}
