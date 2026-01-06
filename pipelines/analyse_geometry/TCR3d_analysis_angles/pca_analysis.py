#!/usr/bin/env python3
"""
Germline/state effect analysis for TCR geometry encodings.

Key outputs
-----------
A) Per-state analyses (UNBOUND and BOUND handled separately):
   - Multiple encodings (raw/zscore/sincos/PCA/optional KPCA/RP/UMAP)
   - Permutation MANOVA-style F-test (Euclidean pseudo-F) + p-value
   - Cross-validated classification using "new clf":
       macro-F1, balanced accuracy, multiclass MCC
     + optional permutation p-value for macro-F1
   - 2D scatter plots per encoding (colored by germline; legend shows counts)

B) Joint PCA analyses (fit PCA on UNBOUND ∪ BOUND together, then plot both states):
   - Joint PCA for:
       (i) raw_deg (wrapped angles + dc)
      (ii) sincos (sin/cos angles + dc)
   - Plots:
       1) All samples: color by germline, shape by state
       2) All samples: state only (germlines merged)
       3) One plot per germline: unbound vs bound colored differently (plus shapes)

   - Joint separability (in JOINT PCA space; using first N PCs, default N=10):
       1) Separate germlines (ignore state): y = germline
       2) Separate unbound vs bound (ignore germline): y = state
       3) Separate germline+state together: y = f"{germline}__{state}"

Usage
-----
python germline_geometry_analysis.py \
  --csv /path/to/tcr_geometry_unbound_and_bound_with_germlines.csv \
  --out /path/to/output_dir \
  --label-col germline_vj_pair \
  --states unbound,bound \
  --min-per-class 6 \
  --top-n-plot 50 \
  --n-perm-F 499 \
  --n-perm-f1 120

Optionally include CDR3 features (if present):
  --include-cdr3
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# Optional: UMAP (only if installed)
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ---------------------------
# Utility: angle wrapping
# ---------------------------
CIRCULAR_COLS = ["BA", "BC1", "AC1", "BC2", "AC2"]

def wrap_deg_to_180(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return ((x + 180.0) % 360.0) - 180.0

def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def radians(x_deg: np.ndarray) -> np.ndarray:
    return (np.pi / 180.0) * x_deg


# ---------------------------
# Coloring / plotting helpers
# ---------------------------
def collapse_categories(
    s: pd.Series,
    top_n: int,
    min_count: int
) -> Tuple[pd.Series, List[str], Dict[str, int]]:
    """
    Keep up to top_n categories with >= min_count. Others -> 'other'.
    Returns (collapsed_series, ordered_categories, counts_dict).
    """
    s = s.fillna("unknown").astype(str)
    vc = s.value_counts()
    kept = [c for c in vc.index.tolist() if vc[c] >= min_count][:top_n]
    s2 = s.where(s.isin(kept), other="other")
    counts = s2.value_counts().to_dict()
    cats = kept.copy()
    if (s2 == "other").any():
        cats.append("other")
    return s2, cats, counts

def category_colors(categories: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    cm = {cat: cmap(i) for i, cat in enumerate(categories)}
    if "other" in cm:
        cm["other"] = (0.6, 0.6, 0.6, 0.6)
    if "unknown" in cm:
        cm["unknown"] = (0.3, 0.3, 0.3, 0.8)
    return cm

def scatter_2d(
    X2: np.ndarray,
    y: pd.Series,
    out_path: Path,
    title: str,
    top_n: int,
    min_count: int,
    alpha: float = 0.8,
    size: float = 18.0,
):
    y2, cats, _counts = collapse_categories(y, top_n=top_n, min_count=min_count)
    cm = category_colors(cats)

    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(1, 1, 1)

    for cat in cats:
        mask = (y2.values == cat)
        if mask.sum() == 0:
            continue
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            s=size,
            alpha=alpha,
            c=[cm.get(cat, (0.6, 0.6, 0.6, 0.6))],
            edgecolors="none",
            label=f"{cat} (n={int(mask.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def joint_pca_plot_color_germline_shape_state(
    coords: pd.DataFrame,
    label_col: str,
    out_path: Path,
    title: str,
    top_n: int,
    min_count: int,
    alpha: float = 0.75,
    size: float = 18.0,
):
    """
    coords must contain columns: PC1, PC2, state, label_col
    - color by germline (collapsed)
    - marker by state
    - legend entries show counts per germline; separate legend shows counts per state
    """
    required = {"PC1", "PC2", "state", label_col}
    missing = required - set(coords.columns)
    if missing:
        raise KeyError(f"joint_pca_plot_color_germline_shape_state missing cols: {missing}")

    y_raw = coords[label_col].fillna("unknown").astype(str)
    y2, cats, _counts = collapse_categories(y_raw, top_n=top_n, min_count=min_count)
    cm = category_colors(cats)

    # Markers per state
    state_series = coords["state"].fillna("unknown").astype(str)
    state_markers = {"unbound": "o", "bound": "^", "unknown": "s"}

    fig = plt.figure(figsize=(10.5, 7.8))
    ax = fig.add_subplot(1, 1, 1)

    for st, mk in state_markers.items():
        st_mask = (state_series.values == st)
        if st_mask.sum() == 0:
            continue
        for cat in cats:
            m = st_mask & (y2.values == cat)
            if m.sum() == 0:
                continue
            ax.scatter(
                coords.loc[m, "PC1"].values,
                coords.loc[m, "PC2"].values,
                s=size,
                alpha=alpha,
                marker=mk,
                c=[cm.get(cat, (0.6, 0.6, 0.6, 0.6))],
                edgecolors="none",
            )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Germline legend with counts
    germ_handles = []
    for cat in cats:
        ncat = int((y2.values == cat).sum())
        germ_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                       markerfacecolor=cm.get(cat, (0.6, 0.6, 0.6, 0.6)),
                       markeredgecolor="none",
                       label=f"{cat} (n={ncat})")
        )

    # State legend with counts
    state_handles = []
    for st, mk in state_markers.items():
        nst = int((state_series.values == st).sum())
        if nst == 0:
            continue
        state_handles.append(
            plt.Line2D([0], [0], marker=mk, linestyle="", markersize=7,
                       markerfacecolor="black", markeredgecolor="none",
                       label=f"{st} (n={nst})")
        )

    leg1 = ax.legend(handles=germ_handles, title=f"{label_col} (min n={min_count})",
                     loc="center left", bbox_to_anchor=(1.02, 0.65),
                     fontsize=8, title_fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=state_handles, title="state",
              loc="center left", bbox_to_anchor=(1.02, 0.20),
              fontsize=9, title_fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def joint_pca_plot_state_only(
    coords: pd.DataFrame,
    out_path: Path,
    title: str,
    alpha: float = 0.75,
    size: float = 18.0,
):
    """
    coords must contain columns: PC1, PC2, state
    Germlines merged; show only state separation.
    """
    required = {"PC1", "PC2", "state"}
    missing = required - set(coords.columns)
    if missing:
        raise KeyError(f"joint_pca_plot_state_only missing cols: {missing}")

    state_series = coords["state"].fillna("unknown").astype(str)
    state_markers = {"unbound": "o", "bound": "^", "unknown": "s"}
    state_colors = {"unbound": (0.2, 0.4, 0.8, 0.85), "bound": (0.85, 0.25, 0.25, 0.85), "unknown": (0.5, 0.5, 0.5, 0.6)}

    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(1, 1, 1)

    for st, mk in state_markers.items():
        m = (state_series.values == st)
        if m.sum() == 0:
            continue
        ax.scatter(
            coords.loc[m, "PC1"].values,
            coords.loc[m, "PC2"].values,
            s=size,
            alpha=alpha,
            marker=mk,
            c=[state_colors.get(st, (0.5, 0.5, 0.5, 0.6))],
            edgecolors="none",
            label=f"{st} (n={int(m.sum())})",
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def joint_pca_plot_per_germline(
    coords: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    top_n: int,
    min_count: int,
    alpha: float = 0.75,
    size: float = 18.0,
):
    """
    One plot per (collapsed) germline category.
    Unbound vs bound colored differently; shapes also different.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    y_raw = coords[label_col].fillna("unknown").astype(str)
    y2, cats, _counts = collapse_categories(y_raw, top_n=top_n, min_count=min_count)
    coords2 = coords.copy()
    coords2["_germ"] = y2

    state_markers = {"unbound": "o", "bound": "^", "unknown": "s"}
    state_colors = {"unbound": (0.2, 0.4, 0.8, 0.85), "bound": (0.85, 0.25, 0.25, 0.85), "unknown": (0.5, 0.5, 0.5, 0.6)}

    for g in cats:
        subg = coords2[coords2["_germ"] == g].copy()
        if subg.empty:
            continue

        n_unb = int((subg["state"].astype(str) == "unbound").sum())
        n_bnd = int((subg["state"].astype(str) == "bound").sum())

        fig = plt.figure(figsize=(8.8, 6.8))
        ax = fig.add_subplot(1, 1, 1)

        for st, mk in state_markers.items():
            m = (subg["state"].astype(str).values == st)
            if m.sum() == 0:
                continue
            ax.scatter(
                subg.loc[m, "PC1"].values,
                subg.loc[m, "PC2"].values,
                s=size,
                alpha=alpha,
                marker=mk,
                c=[state_colors.get(st, (0.5, 0.5, 0.5, 0.6))],
                edgecolors="none",
                label=f"{st} (n={int(m.sum())})",
            )

        ax.set_title(f"{label_col} = {g}\nunbound n={n_unb}, bound n={n_bnd}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", frameon=True)

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(g))[:180]
        out_path = out_dir / f"joint_pca__{label_col}__{safe}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------
# Permutation MANOVA-style test (Euclidean)
# ---------------------------
def euclidean_group_F(X: np.ndarray, labels: np.ndarray) -> float:
    """
    MANOVA-style pseudo-F in Euclidean space:
      F = (SS_between/(k-1)) / (SS_within/(N-k))
    """
    labels = labels.astype(str)
    N = X.shape[0]
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2 or N <= k:
        return np.nan

    grand = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - grand) ** 2).sum())

    ss_within = 0.0
    for g in uniq:
        Xg = X[labels == g]
        if Xg.shape[0] < 2:
            continue
        cg = Xg.mean(axis=0, keepdims=True)
        ss_within += float(((Xg - cg) ** 2).sum())

    ss_between = ss_total - ss_within
    df_between = k - 1
    df_within = N - k
    if df_within <= 0:
        return np.nan

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if ss_within > 0 else np.nan
    if not np.isfinite(ms_within) or ms_within == 0:
        return np.nan
    return ms_between / ms_within

def permutation_pvalue_F(
    X: np.ndarray,
    labels: np.ndarray,
    n_perm: int = 999,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = labels.astype(str)

    F_obs = euclidean_group_F(X, labels)
    if not np.isfinite(F_obs):
        return (F_obs, np.nan)

    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        Fp = euclidean_group_F(X, perm)
        if np.isfinite(Fp) and Fp >= F_obs:
            ge += 1
    p = (1.0 + ge) / (1.0 + n_perm)
    return (F_obs, p)


# ---------------------------
# Classification + permutation test
# ---------------------------
def make_new_clf(seed: int = 0) -> Pipeline:
    """
    "New clf": scaler + L2 logistic regression (no deprecated multi_class arg).
    """
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=0.5,
            max_iter=10000,
            tol=1e-4,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        ))
    ])
    return clf

def cv_predict_metrics(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> Tuple[float, float, float]:
    """
    Stratified CV with explicit fold predictions, returning:
      (macro_f1, balanced_accuracy, multiclass_mcc)
    Uses the "new clf".
    """
    y = y.astype(str)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base_clf = make_new_clf(seed=seed)

    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for tr_idx, te_idx in cv.split(X, y):
        clf = clone(base_clf)
        clf.fit(X[tr_idx], y[tr_idx])
        yp = clf.predict(X[te_idx])
        y_true_all.extend(y[te_idx].tolist())
        y_pred_all.extend(yp.tolist())

    y_true = np.asarray(y_true_all, dtype=str)
    y_pred = np.asarray(y_pred_all, dtype=str)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))
    mcc      = float(matthews_corrcoef(y_true, y_pred))  # multiclass MCC supported

    return macro_f1, bal_acc, mcc

def permutation_pvalue_macro_f1(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Permutation p-value for macro-F1 (CV mean via fold predictions).
    Returns (f1_obs, p_value).
    """
    rng = np.random.default_rng(seed)
    y = y.astype(str)

    f1_obs, _, _ = cv_predict_metrics(X, y, seed=seed)
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        f1p, _, _ = cv_predict_metrics(X, yp, seed=seed)
        if f1p >= f1_obs:
            ge += 1
    p = (1.0 + ge) / (1.0 + n_perm)
    return (f1_obs, p)


# ---------------------------
# Encodings
# ---------------------------
def encode_raw_deg(df: pd.DataFrame, include_cdr3: bool) -> Tuple[np.ndarray, List[str]]:
    cols = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]
    if include_cdr3:
        extra = [
            "alpha_cdr3_bend_deg",
            "alpha_cdr3_apex_height_A",
            "beta_cdr3_bend_deg",
            "beta_cdr3_apex_height_A",
        ]
        cols += [c for c in extra if c in df.columns]

    X = df[cols].copy()
    for c in CIRCULAR_COLS:
        if c in X.columns:
            X[c] = wrap_deg_to_180(X[c])
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.values.astype(float), cols

def encode_sincos(df: pd.DataFrame, include_cdr3: bool) -> Tuple[np.ndarray, List[str]]:
    base_cols = ["BA", "BC1", "AC1", "BC2", "AC2"]
    out: Dict[str, np.ndarray] = {}

    for c in base_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required angle col: {c}")
        a = wrap_deg_to_180(df[c]).astype(float).values
        r = radians(a)
        out[f"{c}_sin"] = np.sin(r)
        out[f"{c}_cos"] = np.cos(r)

    if "dc" not in df.columns:
        raise KeyError("Missing required col: dc")
    out["dc"] = pd.to_numeric(df["dc"], errors="coerce").values.astype(float)

    if include_cdr3:
        for bend_col in ["alpha_cdr3_bend_deg", "beta_cdr3_bend_deg"]:
            if bend_col in df.columns:
                a = wrap_deg_to_180(df[bend_col]).astype(float).values
                r = radians(a)
                out[f"{bend_col}_sin"] = np.sin(r)
                out[f"{bend_col}_cos"] = np.cos(r)
        for hcol in ["alpha_cdr3_apex_height_A", "beta_cdr3_apex_height_A"]:
            if hcol in df.columns:
                out[hcol] = pd.to_numeric(df[hcol], errors="coerce").values.astype(float)

    cols = list(out.keys())
    X = np.column_stack([out[c] for c in cols]).astype(float)
    return X, cols

def apply_scaler(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def encode_pca(Xs: np.ndarray, n_components: int = 5, seed: int = 0) -> Tuple[np.ndarray, PCA]:
    n_components = min(n_components, Xs.shape[1], Xs.shape[0] - 1) if Xs.shape[0] > 2 else min(n_components, Xs.shape[1])
    pca = PCA(n_components=max(2, n_components), random_state=seed)
    Z = pca.fit_transform(Xs)
    return Z, pca

def encode_kernel_pca_rbf(Xs: np.ndarray, n_components: int = 5, gamma: Optional[float] = None, seed: int = 0) -> Tuple[np.ndarray, KernelPCA]:
    n_components = min(n_components, Xs.shape[1], Xs.shape[0] - 1) if Xs.shape[0] > 2 else min(n_components, Xs.shape[1])
    kpca = KernelPCA(
        n_components=max(2, n_components),
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=False,
        random_state=seed,
    )
    Z = kpca.fit_transform(Xs)
    return Z, kpca

def encode_randproj(Xs: np.ndarray, n_components: int = 5, seed: int = 0) -> Tuple[np.ndarray, GaussianRandomProjection]:
    n_components = min(n_components, Xs.shape[1], max(2, Xs.shape[0] - 1))
    rp = GaussianRandomProjection(n_components=max(2, n_components), random_state=seed)
    Z = rp.fit_transform(Xs)
    return Z, rp

def encode_umap_2d(Xs: np.ndarray, seed: int = 0) -> Optional[np.ndarray]:
    if not HAS_UMAP:
        return None
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=seed)
    return reducer.fit_transform(Xs)


# ---------------------------
# Per-state analysis
# ---------------------------
def run_state_analysis(
    df: pd.DataFrame,
    state: str,
    label_col: str,
    out_dir: Path,
    include_cdr3: bool,
    min_per_class: int,
    top_n_plot: int,
    n_perm_F: int,
    n_perm_f1: int,
    seed: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    d = df[df["state"] == state].copy()
    if d.empty:
        print(f"[WARN] No rows for state={state}.")
        return

    if label_col not in d.columns:
        raise KeyError(f"Label column '{label_col}' not in dataframe.")

    # Filter label classes by count
    y_all = d[label_col].fillna("unknown").astype(str)
    vc = y_all.value_counts()
    keep = vc[vc >= min_per_class].index.tolist()
    d = d[y_all.isin(keep)].copy()
    y = d[label_col].fillna("unknown").astype(str).values

    n = d["tcr_name"].nunique() if "tcr_name" in d.columns else len(d)
    k = len(np.unique(y))
    print(f"[{state}] n={n} (after filtering), k={k} classes kept (min_per_class={min_per_class}).")

    # raw features (drop NaNs)
    X_raw, _raw_cols = encode_raw_deg(d, include_cdr3=include_cdr3)
    mask = np.isfinite(X_raw).all(axis=1)
    d = d.loc[mask].copy()
    y = d[label_col].fillna("unknown").astype(str).values
    X_raw, _raw_cols = encode_raw_deg(d, include_cdr3=include_cdr3)

    if X_raw.shape[0] < 10 or len(np.unique(y)) < 2:
        print(f"[WARN] Not enough data after NaN filtering for state={state}.")
        return

    # sin/cos features (align rows; drop NaNs)
    X_sc, _sc_cols = encode_sincos(d, include_cdr3=include_cdr3)
    mask_sc = np.isfinite(X_sc).all(axis=1)
    if not mask_sc.all():
        d = d.loc[mask_sc].copy()
        y = d[label_col].fillna("unknown").astype(str).values
        X_raw, _ = encode_raw_deg(d, include_cdr3=include_cdr3)
        X_sc, _  = encode_sincos(d, include_cdr3=include_cdr3)

    # Standardized variants
    X_raw_z, _ = apply_scaler(X_raw)
    X_sc_z, _  = apply_scaler(X_sc)

    # PCA variants (per-state PCA)
    Z_pca_raw, pca_raw = encode_pca(X_raw_z, n_components=5, seed=seed)
    Z_pca_sc,  pca_sc  = encode_pca(X_sc_z,  n_components=5, seed=seed)

    # Optional encodings
    Z_kpca = None
    try:
        Z_kpca, _ = encode_kernel_pca_rbf(X_sc_z, n_components=5, gamma=None, seed=seed)
    except Exception:
        Z_kpca = None

    Z_rp, _ = encode_randproj(X_sc_z, n_components=5, seed=seed)
    Z_umap2 = encode_umap_2d(X_sc_z, seed=seed) if HAS_UMAP else None

    encodings: Dict[str, np.ndarray] = {
        "raw_deg": X_raw,
        "zscore_raw_deg": X_raw_z,
        "sincos": X_sc,
        "zscore_sincos": X_sc_z,
        "pca_raw__fit_within_state": Z_pca_raw,
        "pca_sincos__fit_within_state": Z_pca_sc,
        "randproj_sincos": Z_rp,
    }
    if Z_kpca is not None:
        encodings["kernel_pca_rbf_sincos"] = Z_kpca
    if Z_umap2 is not None:
        encodings["umap2_sincos"] = Z_umap2

    # Evaluate each encoding
    rows = []
    for name, X in encodings.items():
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[0] != len(y):
            continue

        F_obs, p_F = permutation_pvalue_F(X, y, n_perm=n_perm_F, seed=seed)

        try:
            f1, bac, mcc = cv_predict_metrics(X, y, seed=seed)
        except Exception:
            f1, bac, mcc = (np.nan, np.nan, np.nan)

        try:
            f1_obs, p_f1 = permutation_pvalue_macro_f1(X, y, n_perm=n_perm_f1, seed=seed)
        except Exception:
            f1_obs, p_f1 = (np.nan, np.nan)

        extra = {}
        if "pca_raw__fit_within_state" == name and hasattr(pca_raw, "explained_variance_ratio_"):
            extra["pca_var_PC1"] = float(pca_raw.explained_variance_ratio_[0])
            extra["pca_var_PC2"] = float(pca_raw.explained_variance_ratio_[1])
        if "pca_sincos__fit_within_state" == name and hasattr(pca_sc, "explained_variance_ratio_"):
            extra["pca_var_PC1"] = float(pca_sc.explained_variance_ratio_[0])
            extra["pca_var_PC2"] = float(pca_sc.explained_variance_ratio_[1])

        rows.append({
            "state": state,
            "label_col": label_col,
            "encoding": name,
            "n_samples": int(X.shape[0]),
            "n_classes": int(len(np.unique(y))),
            "F_obs": float(F_obs) if np.isfinite(F_obs) else np.nan,
            "p_perm_F": float(p_F) if np.isfinite(p_F) else np.nan,
            "cv_macro_f1": float(f1) if np.isfinite(f1) else np.nan,
            "cv_bal_acc": float(bac) if np.isfinite(bac) else np.nan,
            "cv_mcc": float(mcc) if np.isfinite(mcc) else np.nan,
            "perm_p_macro_f1": float(p_f1) if np.isfinite(p_f1) else np.nan,
            **extra,
        })

        # 2D visualization: if >2 dims, PCA to 2D for plotting only
        try:
            if X.shape[1] == 2:
                X2 = X
            else:
                X2 = PCA(n_components=2, random_state=seed).fit_transform(X)

            plot_path = out_dir / f"{state}__{label_col}__{name}__scatter2d.png"
            title = f"{state} | {label_col} | {name}\n(n={X.shape[0]}, k={len(np.unique(y))})"
            scatter_2d(
                X2=X2,
                y=pd.Series(y),
                out_path=plot_path,
                title=title,
                top_n=top_n_plot,
                min_count=min_per_class,
                alpha=0.80,
                size=18.0,
            )
        except Exception as e:
            print(f"[WARN] plotting failed for {state}/{name}: {e}")

    res = pd.DataFrame(rows).sort_values(
        ["state", "p_perm_F", "perm_p_macro_f1"],
        ascending=[True, True, True]
    )
    res_path = out_dir / f"{state}__encoding_comparison_results.csv"
    res.to_csv(res_path, index=False)
    print(f"[{state}] Saved results: {res_path}")

    used_path = out_dir / f"{state}__filtered_dataset_used.csv"
    d.to_csv(used_path, index=False)
    print(f"[{state}] Saved filtered data: {used_path}")


# ---------------------------
# Joint PCA on (unbound ∪ bound) + joint separability tasks
# ---------------------------
def _filter_by_min_count(series: pd.Series, min_per_class: int) -> Tuple[pd.Series, List[str]]:
    s = series.fillna("unknown").astype(str)
    vc = s.value_counts()
    keep = vc[vc >= min_per_class].index.tolist()
    return s[s.isin(keep)], keep

def run_joint_pca(
    df: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    include_cdr3: bool,
    min_per_class: int,
    top_n_plot: int,
    seed: int,
    joint_pca_n_components: int,
    n_perm_joint_f1: int,
):
    """
    Fit PCA on ALL samples (unbound ∪ bound) and output joint plots.
    For each of raw_deg and sincos:
      - Fit scaler + PCA on pooled data
      - Save coords (PC1/PC2), loadings, summary
      - Evaluate separability in JOINT PCA space (first N PCs):
          (1) germline-only
          (2) state-only (binary; only unbound/bound)
          (3) germline+state combined
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in dataframe.")
    if "state" not in df.columns:
        raise KeyError("Missing 'state' column in dataframe.")

    # Keep only known states for joint work
    d0 = df.copy()
    d0["state"] = d0["state"].fillna("unknown").astype(str)
    d0 = d0[d0["state"].isin(["unbound", "bound"])].copy()
    if d0.empty:
        print("[WARN] Joint PCA: no rows with state in {unbound, bound}.")
        return

    enc_specs = [
        ("raw_deg", encode_raw_deg),
        ("sincos", encode_sincos),
    ]

    for enc_name, enc_fn in enc_specs:
        enc_dir = out_dir / f"joint_{enc_name}"
        enc_dir.mkdir(parents=True, exist_ok=True)

        # Build features
        X, cols = enc_fn(d0, include_cdr3=include_cdr3)
        X = np.asarray(X, float)

        # Drop NaNs
        mask = np.isfinite(X).all(axis=1)
        d = d0.loc[mask].copy()
        X = X[mask]
        if X.shape[0] < 20:
            print(f"[WARN] Joint PCA ({enc_name}): too few samples after NaN filtering.")
            continue

        # Standardize pooled
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # PCA pooled: keep N PCs for evaluation, but always save PC1/PC2 for plotting
        n_pc = int(min(joint_pca_n_components, Xs.shape[1], max(2, Xs.shape[0] - 1)))
        pca_full = PCA(n_components=n_pc, random_state=seed)
        Z = pca_full.fit_transform(Xs)

        # For plotting
        PC1 = Z[:, 0]
        PC2 = Z[:, 1]

        coords = d.copy()
        coords["PC1"] = PC1
        coords["PC2"] = PC2

        # Save coords + PCA summary + loadings
        coords_path = enc_dir / f"coords_PC1_PC2.csv"
        coords.to_csv(coords_path, index=False)

        summary = pd.DataFrame([{
            "encoding": enc_name,
            "n_samples": int(coords.shape[0]),
            "include_cdr3": bool(include_cdr3),
            "n_pcs_kept_for_eval": int(n_pc),
            "pc1_var": float(pca_full.explained_variance_ratio_[0]),
            "pc2_var": float(pca_full.explained_variance_ratio_[1]) if len(pca_full.explained_variance_ratio_) > 1 else np.nan,
            "cum_var_top2": float(np.sum(pca_full.explained_variance_ratio_[:2])),
            "cum_var_all_kept": float(np.sum(pca_full.explained_variance_ratio_)),
            "features": ",".join(cols),
        }])
        summary.to_csv(enc_dir / f"pca_summary.csv", index=False)

        # Loadings for PC1/PC2 (only)
        loadings = pd.DataFrame(
            pca_full.components_[:2].T,
            index=cols,
            columns=["PC1_loading", "PC2_loading"],
        )
        loadings.to_csv(enc_dir / f"pca_loadings_PC1_PC2.csv")

        # Plots
        title1 = (
            f"Joint PCA fit on all samples (unbound ∪ bound) | encoding={enc_name}\n"
            f"Color={label_col} (min n={min_per_class}), shape=state\n"
            f"PC1={pca_full.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2={pca_full.explained_variance_ratio_[1]*100:.1f}%"
        )
        joint_pca_plot_color_germline_shape_state(
            coords=coords,
            label_col=label_col,
            out_path=enc_dir / f"plot_color_germline_shape_state.png",
            title=title1,
            top_n=top_n_plot,
            min_count=min_per_class,
        )

        title2 = (
            f"Joint PCA fit on all samples (unbound ∪ bound) | encoding={enc_name}\n"
            f"Germlines merged; state only\n"
            f"PC1={pca_full.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2={pca_full.explained_variance_ratio_[1]*100:.1f}%"
        )
        joint_pca_plot_state_only(
            coords=coords,
            out_path=enc_dir / f"plot_state_only.png",
            title=title2,
        )

        per_dir = enc_dir / f"per_germline"
        joint_pca_plot_per_germline(
            coords=coords,
            label_col=label_col,
            out_dir=per_dir,
            top_n=top_n_plot,
            min_count=min_per_class,
        )

        # ---------------------------
        # Joint separability tasks in joint PCA space (Z, first n_pc PCs)
        # ---------------------------
        results_rows = []

        # Task 1: germline separation (ignore state)
        y_germ_all = d[label_col].fillna("unknown").astype(str)
        vc = y_germ_all.value_counts()
        keep_g = vc[vc >= min_per_class].index.tolist()
        m_g = y_germ_all.isin(keep_g).values
        if m_g.sum() >= 20 and len(np.unique(y_germ_all[m_g].values)) >= 2:
            y = y_germ_all[m_g].values.astype(str)
            Xtask = Z[m_g, :]
            f1, bac, mcc = cv_predict_metrics(Xtask, y, seed=seed)
            p_f1 = np.nan
            if n_perm_joint_f1 and n_perm_joint_f1 > 0:
                _, p_f1 = permutation_pvalue_macro_f1(Xtask, y, n_perm=n_perm_joint_f1, seed=seed)
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_germline_ignore_state",
                "n_samples": int(Xtask.shape[0]),
                "n_classes": int(len(np.unique(y))),
                "cv_macro_f1": f1,
                "cv_bal_acc": bac,
                "cv_mcc": mcc,
                "perm_p_macro_f1": p_f1,
            })
        else:
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_germline_ignore_state",
                "n_samples": int(m_g.sum()),
                "n_classes": int(len(np.unique(y_germ_all[m_g].values.astype(str)))) if m_g.sum() > 0 else 0,
                "cv_macro_f1": np.nan,
                "cv_bal_acc": np.nan,
                "cv_mcc": np.nan,
                "perm_p_macro_f1": np.nan,
            })

        # Task 2: state separation (ignore germline) (binary)
        y_state = d["state"].astype(str).values
        if len(np.unique(y_state)) >= 2 and len(y_state) >= 20:
            f1, bac, mcc = cv_predict_metrics(Z, y_state, seed=seed)
            p_f1 = np.nan
            if n_perm_joint_f1 and n_perm_joint_f1 > 0:
                _, p_f1 = permutation_pvalue_macro_f1(Z, y_state, n_perm=n_perm_joint_f1, seed=seed)
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_state_ignore_germline",
                "n_samples": int(Z.shape[0]),
                "n_classes": int(len(np.unique(y_state))),
                "cv_macro_f1": f1,
                "cv_bal_acc": bac,
                "cv_mcc": mcc,
                "perm_p_macro_f1": p_f1,
            })
        else:
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_state_ignore_germline",
                "n_samples": int(Z.shape[0]),
                "n_classes": int(len(np.unique(y_state))),
                "cv_macro_f1": np.nan,
                "cv_bal_acc": np.nan,
                "cv_mcc": np.nan,
                "perm_p_macro_f1": np.nan,
            })

        # Task 3: germline+state combined
        y_comb = (d[label_col].fillna("unknown").astype(str) + "__" + d["state"].astype(str))
        vc2 = y_comb.value_counts()
        keep_c = vc2[vc2 >= min_per_class].index.tolist()
        m_c = y_comb.isin(keep_c).values
        if m_c.sum() >= 20 and len(np.unique(y_comb[m_c].values.astype(str))) >= 2:
            y = y_comb[m_c].values.astype(str)
            Xtask = Z[m_c, :]
            f1, bac, mcc = cv_predict_metrics(Xtask, y, seed=seed)
            p_f1 = np.nan
            if n_perm_joint_f1 and n_perm_joint_f1 > 0:
                _, p_f1 = permutation_pvalue_macro_f1(Xtask, y, n_perm=n_perm_joint_f1, seed=seed)
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_germline_and_state",
                "n_samples": int(Xtask.shape[0]),
                "n_classes": int(len(np.unique(y))),
                "cv_macro_f1": f1,
                "cv_bal_acc": bac,
                "cv_mcc": mcc,
                "perm_p_macro_f1": p_f1,
            })
        else:
            results_rows.append({
                "encoding": enc_name,
                "task": "joint_separate_germline_and_state",
                "n_samples": int(m_c.sum()),
                "n_classes": int(len(np.unique(y_comb[m_c].values.astype(str)))) if m_c.sum() > 0 else 0,
                "cv_macro_f1": np.nan,
                "cv_bal_acc": np.nan,
                "cv_mcc": np.nan,
                "perm_p_macro_f1": np.nan,
            })

        joint_res = pd.DataFrame(results_rows)
        joint_res_path = enc_dir / "joint_classification_results_in_joint_PCA_space.csv"
        joint_res.to_csv(joint_res_path, index=False)

        print(f"[JOINT PCA {enc_name}] Saved coords: {coords_path}")
        print(f"[JOINT PCA {enc_name}] Saved joint separability: {joint_res_path}")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to tcr_geometry_unbound_and_bound_with_germlines.csv")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--label-col", type=str, default="germline_vj_pair",
                    help="Which germline label to test (e.g. germline_vj_pair, alpha_v_gene, beta_v_gene, ...)")
    ap.add_argument("--states", type=str, default="unbound,bound", help="Comma-separated states to analyze")
    ap.add_argument("--min-per-class", type=int, default=6, help="Minimum examples per class to keep")
    ap.add_argument("--top-n-plot", type=int, default=50, help="Top N classes to show explicitly in plots; rest -> other")
    ap.add_argument("--include-cdr3", action="store_true",
                    help="Include CDR3 features if present (bend + apex height) in addition to BA..dc")
    ap.add_argument("--n-perm-F", type=int, default=499, help="Permutations for MANOVA-style F test")
    ap.add_argument("--n-perm-f1", type=int, default=120, help="Permutations for macro-F1 p-value (per-state)")
    ap.add_argument("--joint-pca-n-components", type=int, default=10, help="Number of PCs to keep for joint PCA classification")
    ap.add_argument("--n-perm-joint-f1", type=int, default=0,
                    help="If >0, permutation p-value for macro-F1 in joint PCA tasks (can be expensive)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed_base = ["BA", "BC1", "AC1", "BC2", "AC2", "dc", "state", args.label_col]
    missing = [c for c in needed_base if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # Make numeric
    df = safe_numeric(
        df,
        [
            "BA", "BC1", "AC1", "BC2", "AC2", "dc",
            "alpha_cdr3_bend_deg", "alpha_cdr3_apex_height_A",
            "beta_cdr3_bend_deg", "beta_cdr3_apex_height_A",
        ],
    )

    # Wrap core circular geometry angles
    for c in CIRCULAR_COLS:
        if c in df.columns:
            df[c] = wrap_deg_to_180(df[c])

    # Summary counts by state
    if "tcr_name" in df.columns:
        summary = (
            df.groupby("state")["tcr_name"]
              .nunique()
              .reset_index()
              .rename(columns={"tcr_name": "n_unique_tcrs"})
        )
    else:
        summary = df.groupby("state").size().reset_index(name="n_rows")
    summary.to_csv(out_dir / "summary_counts_by_state.csv", index=False)

    # (1) Per-state analyses
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    for st in states:
        run_state_analysis(
            df=df,
            state=st,
            label_col=args.label_col,
            out_dir=out_dir / f"state_{st}",
            include_cdr3=args.include_cdr3,
            min_per_class=args.min_per_class,
            top_n_plot=args.top_n_plot,
            n_perm_F=args.n_perm_F,
            n_perm_f1=args.n_perm_f1,
            seed=args.seed,
        )

    # (2) Joint PCA analyses + joint separability tasks
    run_joint_pca(
        df=df,
        label_col=args.label_col,
        out_dir=out_dir / "joint_pca_fit_on_all_samples",
        include_cdr3=args.include_cdr3,
        min_per_class=args.min_per_class,
        top_n_plot=args.top_n_plot,
        seed=args.seed,
        joint_pca_n_components=args.joint_pca_n_components,
        n_perm_joint_f1=args.n_perm_joint_f1,
    )

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
