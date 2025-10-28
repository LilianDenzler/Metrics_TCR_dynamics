from typing import Optional, Set, List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, List, Tuple, Dict, Iterable
import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import mdtraj as md
from itertools import combinations
from typing import List, Optional, Set, Tuple, Dict, Callable, Any

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

# ---------- utilities ----------

def evr_for_dataset(pca_like, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Explained-variance ratio of each component for dataset X
    under an already-fitted (W)PCA basis.
    Returns (per_pc_ratio, per_pc_var, total_var) for X only.
    """
    Xc = X - pca_like.mean_
    Z  = Xc @ pca_like.components_.T
    per_pc_var = Z.var(axis=0, ddof=1)
    total_var  = Xc.var(axis=0, ddof=1).sum()
    per_pc_ratio = per_pc_var / (total_var if total_var > 0 else 1.0)
    return per_pc_ratio, per_pc_var, total_var

def _evr_for_dataset_linear(pca_like, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """EVR per dataset for linear (W)PCA basis."""
    Xc = X - pca_like.mean_
    Z = Xc @ pca_like.components_.T
    per_pc_var = Z.var(axis=0, ddof=1)
    total_var = Xc.var(axis=0, ddof=1).sum()
    per_pc_ratio = per_pc_var / (total_var if total_var > 0 else 1.0)
    return per_pc_ratio, per_pc_var, total_var

def median_gamma(X: np.ndarray, subsample: int = 4000, seed: int = 0) -> float:
    """Median distance heuristic for RBF gamma."""
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    D = pairwise_distances(X[idx], metric="euclidean")
    med = np.median(D[np.triu_indices_from(D, k=1)])
    if not np.isfinite(med) or med <= 0:
        med = np.std(D)
    return 1.0 / (2.0 * (med**2) + 1e-12)



class PCAWeighted:
    """
    Minimal weighted PCA via row-scaling + SVD.
    Equivalent to PCA on a weighted covariance with sample_weight.

    Attributes:
        mean_ : (d,) weighted mean
        components_ : (k, d) principal directions
        explained_variance_ratio_ : (k,) EVR on the FIT set (weighted)
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_: np.ndarray = None
        self.components_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None

    def fit(self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        n, d = X.shape
        if sample_weight is None:
            w = np.ones(n, dtype=float) / n
        else:
            w = np.asarray(sample_weight, dtype=float)
            assert w.shape == (n,)
            s = w.sum()
            if s <= 0:
                raise ValueError("sample_weight must sum to a positive value.")
            w = w / s

        # weighted mean + centering
        self.mean_ = np.average(X, axis=0, weights=w)
        Xc = X - self.mean_

        # row-scale by sqrt(weights)
        Xw = Xc * np.sqrt(w)[:, None]

        # SVD
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        self.components_ = Vt[: self.n_components]

        # EVR on the fitted (weighted) dataset
        eigvals = (S ** 2)
        tot = eigvals.sum()
        if tot > 0:
            self.explained_variance_ratio_ = eigvals[: self.n_components] / tot
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components, dtype=float)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T


# ---------- main API ----------

def pca_project_two(
    tv_gt,                     # TrajectoryView for ground truth
    tv_pred,                   # TrajectoryView for model trajectory
    region_names: List[str],
    atom_names: Optional[Set[str]] = {"CA"},
    n_components: int = 2,
    fit_on: str = "gt",        # "gt" | "concat" | "pred" | "concat_weighted"
    weight_mode: str = "equal_sets",  # used when fit_on="concat_weighted"
    sample_weight_custom: Optional[np.ndarray] = None,
) -> Tuple[object, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Fit PCA or Weighted PCA and project both datasets.
    Returns:
        pca_like : PCA or PCAWeighted instance
        Z_gt     : (n_gt, n_components)
        Z_pred   : (n_pred, n_components)
        info     : dict with:
            - evr_fit                : EVR on the data used for fitting
            - evr_gt / evr_pred      : EVR evaluated per dataset under the fitted basis
            - evr_cum_gt / _pred     : cumulative EVR
            - total_var_gt / _pred   : total variance per dataset (original coord space)
    """
    # 1) select matching atoms (indices are in their own mdtraj topologies)
    idx_gt   = tv_gt.domain_idx(region_names=region_names, atom_names=atom_names)
    idx_pred = tv_pred.domain_idx(region_names=region_names, atom_names=atom_names)
    if len(idx_gt) != len(idx_pred):
        raise ValueError(f"Atom count mismatch: gt={len(idx_gt)} vs pred={len(idx_pred)}")

    # 2) feature matrices
    X_gt   = tv_gt.mdtraj.xyz[:,   idx_gt,  :].reshape(tv_gt.mdtraj.n_frames,  -1)
    X_pred = tv_pred.mdtraj.xyz[:, idx_pred, :].reshape(tv_pred.mdtraj.n_frames, -1)

    # 3) fit variants
    if fit_on == "gt":
        X_fit = X_gt
        pca_like = PCA(n_components=n_components).fit(X_fit)

    elif fit_on == "pred":
        X_fit = X_pred
        pca_like = PCA(n_components=n_components).fit(X_fit)

    elif fit_on == "concat":
        X_fit = np.vstack([X_gt, X_pred])
        pca_like = PCA(n_components=n_components).fit(X_fit)

    elif fit_on == "concat_weighted":
        X_fit = np.vstack([X_gt, X_pred])
        n_gt, n_pred = len(X_gt), len(X_pred)

        if sample_weight_custom is not None:
            if sample_weight_custom.shape != (len(X_fit),):
                raise ValueError("sample_weight_custom must match len(X_gt)+len(X_pred).")
            w = sample_weight_custom
        else:
            if weight_mode == "equal_sets":
                # total weight 0.5 for GT and 0.5 for Pred (independent of counts)
                w = np.concatenate([
                    np.full(n_gt,  0.5 / max(1, n_gt)),
                    np.full(n_pred,0.5 / max(1, n_pred)),
                ])
            elif weight_mode == "by_counts":
                # same as unweighted concat (each sample equal)
                w = np.ones(len(X_fit), dtype=float)
            else:
                raise ValueError("weight_mode must be 'equal_sets' or 'by_counts'.")

        pca_like = PCAWeighted(n_components=n_components).fit(X_fit, sample_weight=w)

    else:
        raise ValueError("fit_on must be one of: 'gt', 'pred', 'concat', 'concat_weighted'.")

    # 4) transform
    Z_gt   = pca_like.transform(X_gt)
    Z_pred = pca_like.transform(X_pred)

    # 5) EVR per dataset under this basis
    evr_gt,  var_pc_gt,  tot_gt   = evr_for_dataset(pca_like, X_gt)
    evr_pred,var_pc_pred,tot_pred = evr_for_dataset(pca_like, X_pred)

    # EVR on the fit set (linear PCA: attribute exists; weighted: we computed it)
    evr_fit = getattr(pca_like, "explained_variance_ratio_", None)

    info = {
        "evr_fit": evr_fit,
        "evr_gt": evr_gt,
        "evr_pred": evr_pred,
        "evr_cum_gt": np.cumsum(evr_gt),
        "evr_cum_pred": np.cumsum(evr_pred),
        "total_var_gt": tot_gt,
        "total_var_pred": tot_pred,
        "per_pc_var_gt": var_pc_gt,
        "per_pc_var_pred": var_pc_pred,
    }
    evr_gt=[100.0 * evr_gt[i] for i in range(len(evr_gt))]
    evr_pred=[100.0 * evr_pred[i] for i in range(len(evr_pred))]
    return pca_like, Z_gt, Z_pred, evr_gt, evr_pred

#---------- 1) Build dihedral feature matrices (sin/cos encoded) ----------

def _allowed_residue_indices_from_regions(tv, region_names: Optional[List[str]]) -> Optional[set]:
    """Return None (use all) or set of mdtraj residue indices allowed by regions."""
    if not region_names:
        return None
    # use CA selection to map atoms->res
    atom_idx = tv.domain_idx(region_names=region_names, atom_names={"CA"})
    res_idx = {tv.mdtraj.topology.atom(i).residue.index for i in atom_idx}
    return res_idx

def _filter_cols_by_center_res(dihed_atom_idx: np.ndarray, traj: md.Trajectory,
                               allowed_res: Optional[set]) -> np.ndarray:
    """dihed_atom_idx: (m,4) atom indices per dihedral; keep columns whose 'center' residue is allowed.
    We take atom #1 (2nd atom) as the central residue (works for φ/ψ/ω/χ1)."""
    if allowed_res is None:
        return np.arange(dihed_atom_idx.shape[0])
    centers = [traj.topology.atom(int(row[1])).residue.index for row in dihed_atom_idx]
    keep = np.array([i for i, r in enumerate(centers) if r in allowed_res], dtype=int)
    return keep

def _compute_one(traj: md.Trajectory, kind: str):
    kind = kind.lower()
    if kind == "phi":
        idx, ang = md.compute_phi(traj)
    elif kind == "psi":
        idx, ang = md.compute_psi(traj)
    elif kind == "omega":
        idx, ang = md.compute_omega(traj)
    elif kind in ("chi1","chi2","chi3","chi4","chi5"):
        order = int(kind[-1])
        idx, ang = md.compute_chi(traj, order=order)
    else:
        raise ValueError(f"Unknown dihedral type: {kind}")
    return idx, ang  # idx: (m,4), ang: (n_frames, m) in radians

def dihedral_features(
    tv,
    dihedrals: Iterable[str] = ("phi","psi"),
    region_names: Optional[List[str]] = None,
    encode: str = "sincos",  # 'sincos' or 'radians'
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a per-frame feature matrix from selected dihedrals of a TrajectoryView.
    Returns:
        X : (n_frames, n_feat)  where n_feat = sum over selected dihedrals of (2*m) if 'sincos' else m
        meta : dict with angles per kind for debugging {'phi': angles, ...}
    Notes:
        - Angles are in radians; 'sincos' stacks [cos, sin] for each dihedral to remove periodicity.
        - region_names filters which residues contribute dihedrals.
    """
    traj = tv.mdtraj
    allowed_res = _allowed_residue_indices_from_regions(tv, region_names)

    X_parts = []
    meta = {}
    for kind in dihedrals:
        idx, ang = _compute_one(traj, kind)  # ang: (n_frames, m_all)
        keep_cols = _filter_cols_by_center_res(idx, traj, allowed_res)
        if keep_cols.size == 0:
            continue
        A = ang[:, keep_cols]  # (n_frames, m)
        meta[kind] = A
        if encode == "sincos":
            X_parts.append(np.cos(A))
            X_parts.append(np.sin(A))
        elif encode == "radians":
            X_parts.append(A)
        else:
            raise ValueError("encode must be 'sincos' or 'radians'")
    if not X_parts:
        raise ValueError("No dihedral features selected (check region_names / dihedral types).")
    X = np.hstack(X_parts)
    return X, meta

def pca_on_dihedrals(
    tv_gt,
    tv_pred,
    region_names: Optional[List[str]] = None,
    dihedrals: Iterable[str] = ("phi","psi"),
    encode: str = "sincos",
    n_components: int = 2,
    method: str = "pca_weighted",     # 'pca_weighted' | 'kpca_rbf' | 'kpca_cosine'
    fit_on="concat" ,
    pred_share: float = 0.2,          # only for pca_weighted
    kpca_gamma: Optional[float] = None,  # if None (rbf): auto via median heuristic
) -> Tuple[object, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Build dihedral features for GT and Pred, then embed with (weighted) PCA or Kernel PCA.

    Returns:
        model    : PCAWeighted or KernelPCA
        Z_gt     : (n_gt, n_components)
        Z_pred   : (n_pred, n_components)
        info     : dict with EVRs (for linear) or lambdas (for kPCA) and bookkeeping
    """
    # 1) features
    X_gt,   _ = dihedral_features(tv_gt,   dihedrals=dihedrals, region_names=region_names, encode=encode)
    X_pred, _ = dihedral_features(tv_pred, dihedrals=dihedrals, region_names=region_names, encode=encode)

    info: Dict[str, np.ndarray] = {}

    if method == "pca_weighted":
        # equal-set weighting with adjustable Pred share
        X_fit = np.vstack([X_gt, X_pred])
        n_gt, n_pred = len(X_gt), len(X_pred)
        lam = float(np.clip(pred_share, 1e-6, 0.5))
        w = np.concatenate([
            np.full(n_gt,  (1.0 - lam) / max(1, n_gt)),
            np.full(n_pred,      lam    / max(1, n_pred)),
        ])
        model = PCAWeighted(n_components=n_components).fit(X_fit, sample_weight=w)
        Z_gt   = model.transform(X_gt)
        Z_pred = model.transform(X_pred)

        # EVR per dataset (linear only)
        evr_gt, _, _   = _evr_for_dataset_linear(model, X_gt)
        evr_pred, _, _ = _evr_for_dataset_linear(model, X_pred)
        info["evr_fit"]  = model.explained_variance_ratio_
        info["evr_gt"]   = evr_gt
        info["evr_pred"] = evr_pred
    if method == "pca":
            # 3) fit variants
        if fit_on == "gt":
            X_fit = X_gt
            model = PCA(n_components=n_components).fit(X_fit)

        elif fit_on == "pred":
            X_fit = X_pred
            model = PCA(n_components=n_components).fit(X_fit)

        elif fit_on == "concat":
            X_fit = np.vstack([X_gt, X_pred])
        model = PCA(n_components=n_components).fit(X_fit)
        Z_gt   = model.transform(X_gt)
        Z_pred = model.transform(X_pred)

        # EVR per dataset (linear only)
        evr_gt, _, _   = _evr_for_dataset_linear(model, X_gt)
        evr_pred, _, _ = _evr_for_dataset_linear(model, X_pred)
        info["evr_fit"]  = model.explained_variance_ratio_
        info["evr_gt"]   = evr_gt
        info["evr_pred"] = evr_pred

    elif method in ("kpca_rbf", "kpca_cosine"):
        # Fit on balanced union for stability (weights not supported by KernelPCA)
        X_fit = np.vstack([X_gt, X_pred])
        if method == "kpca_rbf":
            gamma = kpca_gamma if kpca_gamma is not None else median_gamma(X_fit)
            model = KernelPCA(n_components=n_components, kernel="rbf", gamma=gamma, fit_inverse_transform=False)
        else:
            model = KernelPCA(n_components=n_components, kernel="cosine", fit_inverse_transform=False)
        Z_fit  = model.fit_transform(X_fit)  # not used except to establish basis
        Z_gt   = model.transform(X_gt)
        Z_pred = model.transform(X_pred)

        # EVR is not defined like linear PCA; report kernel eigenvalue fractions (fit-set proxy)
        if hasattr(model, "lambdas_"):
            lamb = np.asarray(model.lambdas_)
            info["lambda_ratio_fit"] = lamb[:n_components] / np.sum(lamb)

    else:
        raise ValueError("method must be 'pca_weighted', 'kpca_rbf', or 'kpca_cosine'.")

    return model, Z_gt, Z_pred, info


