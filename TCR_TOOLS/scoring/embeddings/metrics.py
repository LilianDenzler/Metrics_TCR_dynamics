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

def cross_rmsd_matrix(
    traj_a: md.Trajectory,
    traj_b: md.Trajectory,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    superpose: bool = True,
) -> np.ndarray:
    """
    Cross RMSD matrix (Å): shape (n_a, n_b), distances between frames in traj_a and traj_b.

    superpose=True  -> best-fit RMSD per pair (mdtraj.rmsd with ref frame)
    superpose=False -> fixed-frame RMSD in current coordinates (NO re-fit)
                       Requires traj_a and traj_b to already be in the SAME coordinate frame.
    """
    Ta = traj_a.atom_slice(idx_a)
    Tb = traj_b.atom_slice(idx_b)

    Xa = Ta.xyz  # (na,k,3) nm
    Xb = Tb.xyz  # (nb,k,3) nm
    na, k, _ = Xa.shape
    nb = Xb.shape[0]

    if superpose:
        C = np.empty((na, nb), float)
        for j in range(nb):
            # RMSD of all frames in Ta to frame j of Tb, best-fit on selected atoms
            C[:, j] = md.rmsd(Ta, Tb, frame=j) * 10.0
        return C

    # fixed-frame cross RMSD
    Ya = Xa.reshape(na, 3 * k)
    Yb = Xb.reshape(nb, 3 * k)
    G = Ya @ Yb.T
    sa = np.sum(Ya * Ya, axis=1)
    sb = np.sum(Yb * Yb, axis=1)
    D2 = sa[:, None] + sb[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    C_nm = np.sqrt(D2 / k)
    return C_nm * 10.0  # Å


def rmsd_matrix(traj: md.Trajectory, atom_indices: np.ndarray, rmsd_superpose: bool = True) -> np.ndarray:
    """
    Pairwise RMSD matrix (Å) between frames, using either:
      - rmsd_superpose=True: best-fit RMSD (mdtraj.rmsd; re-fits per pair)
      - rmsd_superpose=False: fixed-frame RMSD (NO re-fit; uses current coordinates)
    Assumes traj.xyz is in nm (mdtraj default). Output is in Å.
    """
    T = traj.atom_slice(atom_indices)
    X = T.xyz  # (n, k, 3) in nm
    n, k, _ = X.shape

    if rmsd_superpose:
        M = np.empty((n, n), float)
        for i in range(n):
            # md.rmsd returns nm; convert to Å
            M[:, i] = md.rmsd(T, T, frame=i) * 10.0
        return 0.5 * (M + M.T)

    # fixed-frame RMSD: RMSD(i,j) = sqrt( mean_a ||X_i(a)-X_j(a)||^2 )
    # Vectorize via flattened coordinates and quadratic form
    Y = X.reshape(n, 3 * k)  # nm
    G = Y @ Y.T              # (n,n) nm^2
    sq = np.diag(G)          # (n,) nm^2
    # squared euclidean distance in R^(3k)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)

    # Convert squared euclidean to RMSD: divide by k, sqrt
    M_nm = np.sqrt(D2 / k)
    return M_nm * 10.0  # Å

def mantel_rmsd_vs_embedding(
    tv_gt,
    tv_pred,
    Z_gt: np.ndarray,
    Z_pred: Optional[np.ndarray],
    region_names,
    rmsd_superpose: bool = False,
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
    Dg = rmsd_matrix(Tgt, idx_gt, rmsd_superpose=rmsd_superpose)
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
    Dp = rmsd_matrix(Tpr, idx_pr, rmsd_superpose=rmsd_superpose)

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
