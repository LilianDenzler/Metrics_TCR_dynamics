import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr import TCR

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from dataset_handler import GeometryDatasetHandler

from visualise_model_results import visualize_evaluation_results


# -----------------------
# Config
# -----------------------
CSV_PATH = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/TCR3d_unpaired_unbound.csv"
SEQ_CACHE_CSV = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/TCR3d_unpaired_unbound_with_variable_seqs.csv"
RESULTS_DIR = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/geometry_prediction/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COL = "tcr_name"
PDB_COL = "pdb_path"
ALPHA_SEQ_COL = "alpha_variable_seq"
BETA_SEQ_COL = "beta_variable_seq"
GERMLINE_COL = "germline_vj_pair"
GEOM_COLS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESMC_MODEL_NAME = "esmc_600m"  # or "esmc_300m"

RANDOM_STATE = 0
RIDGE_ALPHA = 1.0

# Evaluation protocol:
TCR_CV_FOLDS = 10                 # TCR-held-out (GroupKFold by tcr_name)
GERMLINE_HOLDOUT_FOLDS = 5        # germline-held-out folds
GERMLINE_HOLDOUT_REPEATS = 10     # repeat R times with different random seeds
MIN_TCRS_PER_GERMLINE = 1         # set to 1 if you want; 3 is often more stable

# Angles:
ANGLE_ENCODING = "raw"            # "raw" or "sincos"; raw supported with circular error here


# -----------------------
# Helpers: sequence extraction
# -----------------------
def ensure_variable_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if os.path.isfile(SEQ_CACHE_CSV):
        cached = pd.read_csv(SEQ_CACHE_CSV)
        if ALPHA_SEQ_COL in cached.columns and BETA_SEQ_COL in cached.columns:
            return cached

    out = df.copy()
    if ALPHA_SEQ_COL not in out.columns:
        out[ALPHA_SEQ_COL] = pd.NA
    if BETA_SEQ_COL not in out.columns:
        out[BETA_SEQ_COL] = pd.NA

    for idx, row in out.iterrows():
        if pd.notna(out.at[idx, ALPHA_SEQ_COL]) and pd.notna(out.at[idx, BETA_SEQ_COL]):
            continue

        pdb_path = row[PDB_COL]
        try:
            tcr = TCR(
                input_pdb=str(pdb_path),
                traj_path=None,
                contact_cutoff=5.0,
                min_contacts=50,
                legacy_anarci=True,
            )
            pair = tcr.pairs[0]
            seqs = pair.cdr_fr_sequences()
            out.at[idx, ALPHA_SEQ_COL] = seqs.get("A_variable", None)
            out.at[idx, BETA_SEQ_COL] = seqs.get("B_variable", None)

        except Exception as e:
            print(f"[WARN] Failed to init TCR for {pdb_path}: {e}")
            continue

    out.to_csv(SEQ_CACHE_CSV, index=False)
    return out


# -----------------------
# ESMC embedding: pooled per-residue embeddings
# -----------------------
def pool_per_residue_embeddings(emb) -> np.ndarray:
    if isinstance(emb, torch.Tensor):
        x = emb.detach().cpu().float().numpy()
    else:
        x = np.asarray(emb, dtype=np.float32)

    if x.ndim == 1:
        return x
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Unexpected embeddings shape: {x.shape}")
    return x.mean(axis=0)


class ESMCEmbedderEmbeddingsOnly:
    def __init__(self, model_name: str = ESMC_MODEL_NAME, device: str = DEVICE):
        self.device = device
        self.model = ESMC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.cfg = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=False)

    @torch.no_grad()
    def embed_sequence_pooled(self, sequence: str) -> np.ndarray:
        protein = ESMProtein(sequence=str(sequence))
        protein_tensor = self.model.encode(protein)
        out = self.model.logits(protein_tensor, self.cfg)
        emb = getattr(out, "embeddings", None)
        if emb is None:
            raise RuntimeError("No embeddings returned. Check LogitsConfig(return_embeddings=True).")
        return pool_per_residue_embeddings(emb)

    def embed_many(self, sequences: list[str]) -> list[object]:
        out: list[object] = []
        for i, seq in enumerate(sequences):
            if seq is None or (isinstance(seq, float) and np.isnan(seq)):
                out.append(None)
                continue
            try:
                out.append(self.embed_sequence_pooled(seq))
            except Exception as e:
                print(f"[WARN] embedding failed at i={i}: {e}")
                out.append(None)
        return out


def stack_keep(vec_list, Y: np.ndarray, groups: np.ndarray, germlines: np.ndarray):
    keep = [i for i, v in enumerate(vec_list) if v is not None]
    if len(keep) < 10:
        raise ValueError("Too few valid embeddings after filtering failures.")
    X = np.vstack([vec_list[i] for i in keep])
    Yk = Y[keep]
    gk = groups[keep].astype(str)
    glk = germlines[keep].astype(str)
    return X, Yk, gk, glk, np.array(keep, dtype=int)


# -----------------------
# Metrics (raw + sincos)
# -----------------------
ANGLE_BASE_NAMES = ["BA", "BC1", "AC1", "BC2", "AC2"]

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))

def angle_errors_from_sincos(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    name_to_idx = {n: i for i, n in enumerate(target_names)}
    out = {}
    for a in ANGLE_BASE_NAMES:
        si = name_to_idx.get(f"{a}_sin", None)
        ci = name_to_idx.get(f"{a}_cos", None)
        if si is None or ci is None:
            continue
        ang_true = np.degrees(np.arctan2(y_true[:, si], y_true[:, ci]))
        ang_pred = np.degrees(np.arctan2(y_pred[:, si], y_pred[:, ci]))
        diff = (ang_pred - ang_true + 180.0) % 360.0 - 180.0
        out[f"{a}_mae_deg"] = float(np.mean(np.abs(diff)))
        out[f"{a}_rmse_deg"] = float(np.sqrt(np.mean(diff ** 2)))
    return out

def angle_errors_from_raw(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    name_to_idx = {n: i for i, n in enumerate(target_names)}
    out = {}
    for a in ANGLE_BASE_NAMES:
        if a not in name_to_idx:
            continue
        j = name_to_idx[a]
        diff = (y_pred[:, j] - y_true[:, j] + 180.0) % 360.0 - 180.0
        out[f"{a}_mae_deg"] = float(np.mean(np.abs(diff)))
        out[f"{a}_rmse_deg"] = float(np.sqrt(np.mean(diff ** 2)))
    return out

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str], angle_encoding: str) -> dict:
    metrics = {"rmse_all": _rmse(y_true, y_pred), "mae_all": _mae(y_true, y_pred)}

    if "dc" in target_names:
        j = target_names.index("dc")
        metrics["rmse_dc"] = _rmse(y_true[:, j], y_pred[:, j])
        metrics["mae_dc"] = _mae(y_true[:, j], y_pred[:, j])

    if angle_encoding == "sincos":
        metrics.update(angle_errors_from_sincos(y_true, y_pred, target_names))
    elif angle_encoding == "raw":
        metrics.update(angle_errors_from_raw(y_true, y_pred, target_names))

    return metrics

def macro_rmse_by_germline(y_true: np.ndarray, y_pred: np.ndarray, germlines: np.ndarray) -> float:
    rmses = []
    for gl in np.unique(germlines):
        mask = germlines == gl
        if mask.sum() < 2:
            continue
        rmses.append(_rmse(y_true[mask], y_pred[mask]))
    return float(np.mean(rmses)) if rmses else float("nan")


# -----------------------
# Models: Ridge + baselines
# -----------------------
def fit_predict_ridge(Xtr, Ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    reg = Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)
    reg.fit(Xtr_s, Ytr)
    return reg.predict(Xte_s)

def fit_predict_global_mean(Ytr, n_test: int):
    mu = Ytr.mean(axis=0, keepdims=True)
    return np.repeat(mu, n_test, axis=0)

def fit_predict_germline_mean(Ytr, germ_tr, germ_te):
    global_mu = Ytr.mean(axis=0)
    means = {gl: Ytr[germ_tr == gl].mean(axis=0) for gl in np.unique(germ_tr)}
    return np.vstack([means.get(gl, global_mu) for gl in germ_te])


# -----------------------
# CV regimes
# -----------------------
def eval_tcr_groupkfold(X, Y, tcr_groups, germlines, target_names, angle_encoding, n_splits=10):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, Y, groups=tcr_groups)):
        Xtr, Xte = X[tr], X[te]
        Ytr, Yte = Y[tr], Y[te]
        gtr, gte = germlines[tr], germlines[te]

        pred = fit_predict_ridge(Xtr, Ytr, Xte)
        m = compute_metrics(Yte, pred, target_names, angle_encoding)
        m["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, pred, gte)

        pred0 = fit_predict_global_mean(Ytr, len(te))
        b0 = compute_metrics(Yte, pred0, target_names, angle_encoding)
        b0["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, pred0, gte)

        predg = fit_predict_germline_mean(Ytr, gtr, gte)
        bg = compute_metrics(Yte, predg, target_names, angle_encoding)
        bg["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, predg, gte)

        unseen = np.setdiff1d(np.unique(gte), np.unique(gtr)).size
        m["unseen_germlines_in_test"] = int(unseen)
        b0["unseen_germlines_in_test"] = int(unseen)
        bg["unseen_germlines_in_test"] = int(unseen)

        rows.append({"fold": fold, "model": "ridge", **m})
        rows.append({"fold": fold, "model": "global_mean", **b0})
        rows.append({"fold": fold, "model": "germline_mean", **bg})

    return pd.DataFrame(rows)

def eval_holdout_germlines_one_run(
    X, Y, tcr_groups, germlines, target_names, angle_encoding,
    n_splits=5, min_tcrs_per_germline=1, seed=0
):
    df_tmp = pd.DataFrame({"germline": germlines, "tcr": tcr_groups})
    counts = df_tmp.groupby("germline")["tcr"].nunique()
    keep_germ = counts[counts >= min_tcrs_per_germline].index.values
    mask = np.isin(germlines, keep_germ)

    X2 = X[mask]
    Y2 = Y[mask]
    germ2 = germlines[mask]
    tcr2 = tcr_groups[mask]

    unique_germs = np.unique(germ2)
    if len(unique_germs) < 2:
        raise ValueError("Not enough germlines after filtering.")

    n_splits_eff = min(n_splits, len(unique_germs))
    rng = np.random.default_rng(seed)
    germs_shuffled = unique_germs.copy()
    rng.shuffle(germs_shuffled)
    folds = np.array_split(germs_shuffled, n_splits_eff)

    rows = []
    for fold, test_germs in enumerate(folds):
        te_mask = np.isin(germ2, test_germs)
        tr_mask = ~te_mask

        Xtr, Xte = X2[tr_mask], X2[te_mask]
        Ytr, Yte = Y2[tr_mask], Y2[te_mask]
        gtr, gte = germ2[tr_mask], germ2[te_mask]

        unseen = np.setdiff1d(np.unique(gte), np.unique(gtr)).size  # all test germlines

        pred = fit_predict_ridge(Xtr, Ytr, Xte)
        m = compute_metrics(Yte, pred, target_names, angle_encoding)
        m["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, pred, gte)
        m["unseen_germlines_in_test"] = int(unseen)

        pred0 = fit_predict_global_mean(Ytr, Xte.shape[0])
        b0 = compute_metrics(Yte, pred0, target_names, angle_encoding)
        b0["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, pred0, gte)
        b0["unseen_germlines_in_test"] = int(unseen)

        predg = fit_predict_germline_mean(Ytr, gtr, gte)
        bg = compute_metrics(Yte, predg, target_names, angle_encoding)
        bg["macro_rmse_by_germline"] = macro_rmse_by_germline(Yte, predg, gte)
        bg["unseen_germlines_in_test"] = int(unseen)

        rows.append({"fold": fold, "model": "ridge", **m})
        rows.append({"fold": fold, "model": "global_mean", **b0})
        rows.append({"fold": fold, "model": "germline_mean", **bg})

    return pd.DataFrame(rows)

def eval_holdout_germlines_repeated(
    X, Y, tcr_groups, germlines, target_names, angle_encoding,
    n_splits=5, n_repeats=10, min_tcrs_per_germline=1, base_seed=0
):
    all_runs = []
    for r in range(n_repeats):
        df_r = eval_holdout_germlines_one_run(
            X, Y, tcr_groups, germlines, target_names, angle_encoding,
            n_splits=n_splits,
            min_tcrs_per_germline=min_tcrs_per_germline,
            seed=base_seed + r
        )
        df_r["repeat"] = r
        all_runs.append(df_r)
    return pd.concat(all_runs, ignore_index=True)


# -----------------------
# Main
# -----------------------
def main():
    handler = GeometryDatasetHandler(
        germline_col=GERMLINE_COL,
        group_col=GROUP_COL,
        target_cols=GEOM_COLS,
        drop_na_targets=True,
        collapse_rare_germlines_min_count=None,
        random_state=RANDOM_STATE,
    )

    df = pd.read_csv(CSV_PATH)
    df = ensure_variable_sequences(df)
    df = handler.clean_dataframe(df)
    df = df.dropna(subset=[ALPHA_SEQ_COL, BETA_SEQ_COL, GERMLINE_COL, GROUP_COL]).reset_index(drop=True)

    Y, target_names = handler.extract_targets(df, angle_encoding=ANGLE_ENCODING)
    tcr_groups = df[GROUP_COL].astype(str).to_numpy()
    germlines = df[GERMLINE_COL].astype(str).to_numpy()

    print(f"[INFO] N={len(df)} | unique germlines={pd.Series(germlines).nunique()} | target_dim={Y.shape[1]}")
    print(f"[INFO] angle_encoding={ANGLE_ENCODING} | model={ESMC_MODEL_NAME} | device={DEVICE}")

    embedder = ESMCEmbedderEmbeddingsOnly(model_name=ESMC_MODEL_NAME, device=DEVICE)

    print("[INFO] Embedding alpha...")
    alpha_vecs = embedder.embed_many(df[ALPHA_SEQ_COL].astype(str).tolist())
    print("[INFO] Embedding beta...")
    beta_vecs = embedder.embed_many(df[BETA_SEQ_COL].astype(str).tolist())

    # Build concat X (alpha+beta), using indices where both exist
    keep_ab = np.array([i for i, (a, b) in enumerate(zip(alpha_vecs, beta_vecs)) if (a is not None and b is not None)], dtype=int)
    if len(keep_ab) < 10:
        raise ValueError("Too few paired alpha+beta embeddings.")

    Xab = np.hstack([
        np.vstack([alpha_vecs[i] for i in keep_ab]),
        np.vstack([beta_vecs[i] for i in keep_ab]),
    ])
    Yab = Y[keep_ab]
    tcr_ab = tcr_groups[keep_ab]
    germ_ab = germlines[keep_ab]

    # ---------- 1) TCR-held-out: 10-fold GroupKFold ----------
    print(f"[INFO] TCR-held-out evaluation: GroupKFold n_splits={TCR_CV_FOLDS}")
    df_cv = eval_tcr_groupkfold(
        Xab, Yab, tcr_ab, germ_ab, target_names, ANGLE_ENCODING, n_splits=TCR_CV_FOLDS
    )
    df_cv["input"] = "alpha_beta_concat"
    df_cv.to_csv(os.path.join(RESULTS_DIR, "eval_tcr_groupkfold.csv"), index=False)

    # ---------- 2) Germline-holdout: 5 folds repeated R times ----------
    print(f"[INFO] Germline-holdout evaluation: {GERMLINE_HOLDOUT_FOLDS} folds x {GERMLINE_HOLDOUT_REPEATS} repeats "
          f"(min_tcrs_per_germline={MIN_TCRS_PER_GERMLINE})")
    df_gl_rep = eval_holdout_germlines_repeated(
        Xab, Yab, tcr_ab, germ_ab, target_names, ANGLE_ENCODING,
        n_splits=GERMLINE_HOLDOUT_FOLDS,
        n_repeats=GERMLINE_HOLDOUT_REPEATS,
        min_tcrs_per_germline=MIN_TCRS_PER_GERMLINE,
        base_seed=RANDOM_STATE,
    )
    df_gl_rep["input"] = "alpha_beta_concat"
    df_gl_rep.to_csv(os.path.join(RESULTS_DIR, "eval_germline_holdout_repeated.csv"), index=False)

    # Create a repeat-averaged germline file for plotting (so plots aren't cluttered)
    num_cols = df_gl_rep.select_dtypes(include=[np.number]).columns.tolist()
    # Keep fold/model/input and average numeric metrics across repeats
    df_gl_plot = (
        df_gl_rep
        .groupby(["input", "model", "fold"], as_index=False)[num_cols]
        .mean()
    )
    # For unseen_germlines_in_test, mean is fine; max would also be OK. This keeps it simple.
    df_gl_plot.to_csv(os.path.join(RESULTS_DIR, "eval_germline_holdout.csv"), index=False)

    # ---------- Summaries ----------
    def summarize_simple(df_eval: pd.DataFrame) -> pd.DataFrame:
        metrics = ["rmse_all", "macro_rmse_by_germline"]
        if "rmse_dc" in df_eval.columns:
            metrics.append("rmse_dc")
        # include angle MAE columns if present
        metrics += [c for c in df_eval.columns if c.endswith("_mae_deg")]

        out = (df_eval
               .groupby(["input", "model"], as_index=False)[metrics]
               .agg(["mean", "std"]))
        # flatten columns
        out.columns = ["_".join([c for c in col if c]) for col in out.columns.to_flat_index()]
        return out

    sum_cv = summarize_simple(df_cv)
    sum_gl_overall = summarize_simple(df_gl_rep)  # overall across folds+repeats

    sum_cv.to_csv(os.path.join(RESULTS_DIR, "summary_tcr_groupkfold.csv"), index=False)
    sum_gl_overall.to_csv(os.path.join(RESULTS_DIR, "summary_germline_holdout_overall.csv"), index=False)

    meta = {
        "csv_path": CSV_PATH,
        "model": ESMC_MODEL_NAME,
        "device": DEVICE,
        "angle_encoding": ANGLE_ENCODING,
        "ridge_alpha": RIDGE_ALPHA,
        "tcr_groupkfold_folds": TCR_CV_FOLDS,
        "germline_holdout_folds": GERMLINE_HOLDOUT_FOLDS,
        "germline_holdout_repeats": GERMLINE_HOLDOUT_REPEATS,
        "min_tcrs_per_germline": MIN_TCRS_PER_GERMLINE,
        "files_written": [
            "eval_tcr_groupkfold.csv",
            "eval_germline_holdout_repeated.csv",
            "eval_germline_holdout.csv",
            "summary_tcr_groupkfold.csv",
            "summary_germline_holdout_overall.csv",
        ],
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Wrote evaluation CSVs and summaries to RESULTS_DIR.")

    # Plot using your existing plotting helper (expects eval_tcr_groupkfold.csv and eval_germline_holdout.csv)
    visualize_evaluation_results(
        results_dir=RESULTS_DIR,
        focus_input="alpha_beta_concat",
    )


if __name__ == "__main__":
    main()
    visualize_evaluation_results(
        results_dir="/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/geometry_prediction/results",
        focus_input="alpha_beta_concat",
    )
