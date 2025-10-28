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
    tv_gt, tv_pred, Z_gt, Z_pred, region_names, atom_names={"CA"},
    method="spearman", permutations=9999, seed=0,
    subsample_gt: Optional[int]=None, subsample_pred: Optional[int]=None
) -> Dict[str, Dict[str,float]]:
    idx_gt = np.asarray(tv_gt.domain_idx(region_names=region_names, atom_names=atom_names), int)
    idx_pr = np.asarray(tv_pred.domain_idx(region_names=region_names, atom_names=atom_names), int)
    if len(idx_gt) != len(idx_pr):
        raise ValueError("Atom selection mismatch for RMSD Mantel.")
    rs = np.random.default_rng(seed)
    Tgt = tv_gt.mdtraj
    Tpr = tv_pred.mdtraj
    Igt = np.arange(Tgt.n_frames) if not subsample_gt else np.sort(rs.choice(Tgt.n_frames, subsample_gt, replace=False))
    Ipr = np.arange(Tpr.n_frames) if not subsample_pred else np.sort(rs.choice(Tpr.n_frames, subsample_pred, replace=False))
    Tgt = Tgt[Igt]; Tpr = Tpr[Ipr]; Zg = Z_gt[Igt]; Zp = Z_pred[Ipr]

    Dg = rmsd_matrix(Tgt, idx_gt)
    Dp = rmsd_matrix(Tpr, idx_pr)
    C  = np.empty((Tpr.n_frames, Tgt.n_frames), float)
    for i in range(Tgt.n_frames):
        C[:, i] = md.rmsd(Tpr, Tgt, frame=i, atom_indices=idx_pr, ref_atom_indices=idx_gt) * 10.0
    Dall = np.block([[Dg, C.T],[C, Dp]])

    Eg   = squareform(pdist(Zg))
    Ep   = squareform(pdist(Zp))
    Ec12 = cdist(Zp, Zg)
    Eall = np.block([[Eg, Ec12.T],[Ec12, Ep]])

    rg, pg = mantel(Dg, Eg, method, permutations, seed)
    rp, pp = mantel(Dp, Ep, method, permutations, seed)
    ra, pa = mantel(Dall, Eall, method, permutations, seed)
    return {"gt":{"r":rg,"p":pg},"pred":{"r":rp,"p":pp},"combined":{"r":ra,"p":pa}}
