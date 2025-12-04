from typing import Optional, Tuple, Dict, Any,Iterable
from sklearn.decomposition import PCA, KernelPCA
from numpy.linalg import svd
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
import numpy as np
np.bool = np.bool_
import pyemma
from deeptime.decomposition import TICA as DT_TICA

class PCAWeighted:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        n = X.shape[0]
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        w = w / w.sum()
        self.mean_ = np.average(X, axis=0, weights=w)
        Xc = X - self.mean_
        Xw = Xc * np.sqrt(w)[:, None]
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        eig = S**2
        self.explained_variance_ratio_ = eig[: self.n_components] / (eig.sum() if eig.sum() > 0 else 1.0)
        return self

    def transform(self, X: np.ndarray):
        return (X - self.mean_) @ self.components_.T

def evr_on_dataset_linear(model, X: np.ndarray) -> np.ndarray:
    Xc = X - model.mean_
    Z  = Xc @ model.components_.T
    var = Z.var(axis=0, ddof=1)
    tot = Xc.var(axis=0, ddof=1).sum()
    return var / (tot if tot > 0 else 1.0)

def fit_pca_linear(
    X_gt, X_pred, n_components=2, fit_on="gt", pred_share=0.2, use_gt_scaler=False
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    if use_gt_scaler:
        mu = X_gt.mean(axis=0)
        sd = X_gt.std(axis=0, ddof=1); sd[sd==0]=1.0
        Xg = (X_gt - mu)/sd
        if X_pred is not None:
            Xp = (X_pred - mu)/sd
        else:
            Xp=None
    else:
        Xg, Xp = X_gt, X_pred

    info: Dict[str, Any] = {}
    if fit_on == "concat_weighted":
        Xfit = np.vstack([Xg, Xp])
        n_g, n_p = len(Xg), len(Xp)
        lam = float(np.clip(pred_share, 1e-6, 0.5))
        w = np.concatenate([
            np.full(n_g, (1.0 - lam)/max(1,n_g)),
            np.full(n_p, lam/max(1,n_p))
        ])
        model = PCAWeighted(n_components=n_components).fit(Xfit, sample_weight=w)
        Zg, Zp = model.transform(Xg), model.transform(Xp)
        info["evr_fit"]  = model.explained_variance_ratio_
        info["evr_gt"]   = evr_on_dataset_linear(model, Xg)
        info["evr_pred"] = evr_on_dataset_linear(model, Xp)
        return model, Zg, Zp, info

    if fit_on == "gt":   model = PCA(n_components=n_components).fit(Xg)
    elif fit_on == "pred": model = PCA(n_components=n_components).fit(Xp)
    elif fit_on == "concat": model = PCA(n_components=n_components).fit(np.vstack([Xg, Xp]))
    else: raise ValueError("fit_on must be gt|pred|concat|concat_weighted")
    if Xp is None:
        Zg= model.transform(Xg)
        Zp=None
        info["evr_fit"]  = getattr(model, "explained_variance_ratio_", None)
        info["evr_gt"]   = evr_on_dataset_linear(model, Xg)
        info["evr_pred"] = None
        return model, Zg, Zp, info
    else:

        Zg, Zp = model.transform(Xg), model.transform(Xp)
        info["evr_fit"]  = getattr(model, "explained_variance_ratio_", None)
        info["evr_gt"]   = evr_on_dataset_linear(model, Xg)
        info["evr_pred"] = evr_on_dataset_linear(model, Xp)
        return model, Zg, Zp, info

def median_gamma(X: np.ndarray, subsample=4000, seed=0) -> float:
    from sklearn.metrics import pairwise_distances
    import numpy as np
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(X), size=min(subsample, len(X)), replace=False)
    D = pairwise_distances(X[idx])
    med = np.median(D[np.triu_indices_from(D,1)])
    if not np.isfinite(med) or med <= 0: med = np.std(D)
    return 1.0 / (2.0 * (med**2) + 1e-12)

def fit_kpca(
    X_gt, X_pred, n_components=2, kernel="rbf", gamma=None, degree=3, fit_on="gt"
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    if kernel == "rbf" and gamma is None:
        if X_pred is None:
            gamma = median_gamma(X_gt)
        else:
            gamma = median_gamma(np.vstack([X_gt, X_pred]))
    if kernel == "poly" and degree is None:
        degree = 3.0  # default poly degree

    # initialize model
    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma if kernel == "rbf" else None,
        degree=float(degree) if kernel == "poly" else 3.0,
        fit_inverse_transform=False
    )
    if fit_on == "gt":
        model.fit(X_gt)
    elif fit_on == "pred":
        model.fit(X_pred)
    elif fit_on == "concat":
        model.fit(np.vstack([X_gt, X_pred]))

    Zg = model.transform(X_gt)
    if X_pred is None:
        Zp = None
    else:
        Zp = model.transform(X_pred)
    info = {}
    if hasattr(model, "lambdas_"):
        lam = np.asarray(model.lambdas_)
        info["lambda_ratio_fit"] = lam[:n_components] / lam.sum()
    return model, Zg, Zp, info




def fit_tica(
    X_gt: np.ndarray,
    X_pred: np.ndarray,
    n_components: int = 2,
    lag: int = 10,
    reversible: bool = True,          # common setting
    var_cutoff: float | None = None,  # keep as None when specifying n_components
    epsilon: float = 1e-12,
    ) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit TICA on the time-ordered GT feature matrix (frames x features),
    then project GT and Pred into the same TICA space.
    """
    # Build and fit
    from deeptime.decomposition import VAMP

    def choose_tica_lag_by_vamp2(
        X: np.ndarray,
        lags: Iterable[int] = (1, 2, 5, 10, 20, 50),
        n_components: int = 10,
        score_k: int = 10,  # VAMP-2 uses top-k singular values; pick something >= n_components
    ) -> Tuple[int, Dict[int, float]]:
        """
        Returns (best_lag, scores), where scores maps lag -> VAMP2 score.
        X must be time-ordered (frames x features).
        """
        scores = {}
        for lag in lags:
            vamp = VAMP(lagtime=lag, dim=n_components).fit(X).fetch_model()
            scores[lag] = float(vamp.score(score_k))  # higher is better
        # pick the lag with max score (or implement 'elbow' if you prefer stability)
        best_lag = max(scores, key=scores.get)
        return best_lag, scores
    best_lag, vamp_scores=choose_tica_lag_by_vamp2(X_gt)
    tica = DT_TICA(
        lagtime=best_lag,
        var_cutoff=None,
        dim=10,
        scaling="kinetic_map",
    ).fit(X_gt).fetch_model()

    # Transform
    Zg = tica.transform(X_gt)
    if X_pred is None:
        Zp = None
    else:
        Zp = tica.transform(X_pred)

    if Zg.ndim == 2 and Zg.shape[1] > n_components:
        Zg = Zg[:, :n_components]
    if Zp is not None:
        if Zp.ndim == 2 and Zp.shape[1] > n_components:
            Zp = Zp[:, :n_components]

    # --- 4) diagnostics: values and timescales ---
    # Prefer provided attributes; fall back to safe computations
    vals = None
    for attr in ("eigenvalues_", "singular_values_"):
        if hasattr(tica, attr):
            vals = np.asarray(getattr(tica, attr))
            break
    if vals is None:
        # very defensive fallback: estimate via correlation operator spectrum (not typical)
        vals = np.ones(n_components, dtype=float)

    if hasattr(tica, "timescales_"):
        timescales = np.asarray(tica.timescales_)
    else:
        # implied timescales (frames): -tau / ln(|λ|)
        lam = np.clip(np.abs(vals), epsilon, 1 - epsilon)
        timescales = -best_lag / np.log(lam)

    info = {
        "best_lag": best_lag,
        "vamp2_scores": vamp_scores,          # {lag: score}
        "values": vals[:n_components].tolist(),
        "timescales": timescales[:n_components].tolist(),
    }
    return tica, Zg, Zp, info



class DiffusionMaps:
    """
    Diffusion Maps with Gaussian kernel (alpha=0 random-walk norm).
    Fit on X_fit (usually concat GT+Pred). Nyström for transform().
    """
    def __init__(self, n_components=2, epsilon=None):
        self.n_components = n_components
        self.epsilon = epsilon
        # fitted attributes
        self.X_fit_ = None
        self.kernel_rowsum_ = None  # row sums for normalization
        self.evals_ = None
        self.evecs_ = None

    @staticmethod
    def _median_epsilon(X, subsample=4000, seed=0):
        rs = np.random.default_rng(seed)
        idx = rs.choice(len(X), size=min(subsample, len(X)), replace=False)
        D = pairwise_distances(X[idx], metric="euclidean")
        med = np.median(D[np.triu_indices_from(D, 1)])
        if not np.isfinite(med) or med <= 0: med = np.std(D)
        return med**2  # epsilon ~ median^2

    def fit(self, X: np.ndarray, seed=0):
        self.X_fit_ = np.asarray(X, float)
        n = X.shape[0]
        eps = self.epsilon or self._median_epsilon(X, seed=seed)
        # Gaussian kernel
        D2 = pairwise_distances(X, squared=True)
        K = np.exp(-D2 / (4.0 * eps))

        # random-walk normalization (alpha=0): row-stochastic P
        d = K.sum(axis=1, keepdims=True)
        d[d == 0] = 1.0
        P = K / d

        # eigen-decomposition of P (largest eigenvalues)
        # dense eigh is ok for a few thousand frames; for huge N use sparse
        w, U = eigh(0.5 * (P + P.T))  # symmetrize for numerical stability
        idx = np.argsort(w)[::-1]
        w, U = w[idx], U[:, idx]
        # drop the trivial eigenvector (≈1.0), keep next n_components
        self.evals_ = w[1:self.n_components+1]
        self.evecs_ = U[:, 1:self.n_components+1]
        self.kernel_rowsum_ = d.ravel()
        self._eps_ = eps
        return self

    def transform(self, Xnew: np.ndarray) -> np.ndarray:
        if self.X_fit_ is None: raise RuntimeError("Call fit() first.")
        Xnew = np.asarray(Xnew, float)
        # Nyström extension
        D2 = pairwise_distances(Xnew, self.X_fit_, squared=True)
        Kx = np.exp(-D2 / (4.0 * self._eps_))
        dx = Kx.sum(axis=1, keepdims=True)
        dx[dx == 0] = 1.0
        Px = Kx / dx  # row-stochastic vs training set

        # Project using training eigenvectors/eigenvalues
        # (right-eigenvectors of P form diffusion coordinates)
        # scale by 1/lambda to be consistent with standard Nyström
        Z = Px @ self.evecs_
        Z = Z / np.maximum(self.evals_, 1e-12)  # broadcast
        return Z

def fit_diffmap(
    X_gt: np.ndarray, X_pred: np.ndarray,
    n_components=2, epsilon=None, seed=0, fit_on="gt"
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    if fit_on == "gt":
        Xfit = X_gt
    elif fit_on == "pred":
        Xfit = X_pred
    elif fit_on == "concat":
        Xfit = np.vstack([X_gt, X_pred])
    dm = DiffusionMaps(n_components=n_components, epsilon=epsilon).fit(Xfit, seed=seed)
    Zg= dm.transform(X_gt)
    if X_pred is None:
        Zp=None
    else:
        Zp = dm.transform(X_pred)
    info = {"diffmap_evals": dm.evals_.tolist(), "epsilon": float(dm._eps_)}
    return dm, Zg, Zp, info
