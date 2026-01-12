import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_plot_dir(results_dir: str) -> str:
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def plot_fold_metric(df_eval: pd.DataFrame, out_path: str, title: str, metric: str = "rmse_all"):
    """
    df_eval must contain: fold, model, input, metric
    """
    plt.figure(figsize=(8, 4))
    sns.pointplot(data=df_eval, x="fold", y=metric, hue="model", dodge=True, errorbar=None)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_micro_vs_macro(df_eval: pd.DataFrame, out_path: str, title: str):
    """
    Requires columns: rmse_all, macro_rmse_by_germline, model
    """
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df_eval,
        x="rmse_all",
        y="macro_rmse_by_germline",
        hue="model",
        style="model",
        s=90,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_angle_mae_bars(df_eval: pd.DataFrame, out_path: str, title: str):
    """
    Plots mean angle MAE (degrees) per model across folds.
    Works for raw or sincos as long as you have columns like BA_mae_deg, ...
    """
    angle_cols = [c for c in df_eval.columns if c.endswith("_mae_deg")]
    if not angle_cols:
        print(f"[WARN] No *_mae_deg columns found; skipping {out_path}")
        return

    long = df_eval.melt(
        id_vars=["model", "input", "fold"],
        value_vars=angle_cols,
        var_name="angle",
        value_name="mae_deg",
    )
    # Clean angle names for plotting
    long["angle"] = long["angle"].str.replace("_mae_deg", "", regex=False)

    agg = long.groupby(["model", "input", "angle"], as_index=False)["mae_deg"].mean()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=agg, x="angle", y="mae_deg", hue="model")
    plt.ylabel("MAE (deg)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_rmse_boxplot(df_eval: pd.DataFrame, out_path: str, title: str, metric: str = "rmse_all"):
    """
    Boxplot of per-fold metric by model.
    """
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df_eval, x="model", y=metric)
    sns.stripplot(data=df_eval, x="model", y=metric, color="black", alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_per_germline_error_distribution(
    df_predictions: pd.DataFrame,
    out_path: str,
    title: str,
    germline_col: str = "germline_vj_pair",
    error_col: str = "rmse_sample",
):
    """
    Optional advanced plot if you create a per-sample error table (not in your current CSVs).
    Shows distribution of error per germline.

    Expected df_predictions columns:
      - germline_vj_pair
      - rmse_sample (or another sample-level error)
    """
    if germline_col not in df_predictions.columns or error_col not in df_predictions.columns:
        print(f"[WARN] Missing {germline_col} or {error_col}; skipping {out_path}")
        return

    # Order germlines by median error
    order = (
        df_predictions.groupby(germline_col)[error_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_predictions, x=germline_col, y=error_col, order=order)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def visualize_evaluation_results(results_dir: str, focus_input: str = "alpha_beta_concat"):
    """
    Loads your evaluation outputs and writes a standard suite of plots.

    Expected files:
      - eval_tcr_groupkfold.csv
      - eval_germline_holdout.csv
    """
    plot_dir = ensure_plot_dir(results_dir)

    path_cv = os.path.join(results_dir, "eval_tcr_groupkfold.csv")
    path_gl = os.path.join(results_dir, "eval_germline_holdout.csv")

    if not os.path.isfile(path_cv) or not os.path.isfile(path_gl):
        raise FileNotFoundError("Missing eval_tcr_groupkfold.csv or eval_germline_holdout.csv in results_dir.")

    df_cv = pd.read_csv(path_cv)
    df_gl = pd.read_csv(path_gl)

    # Focus on a single input setting for cleaner plots
    df_cv_f = df_cv[df_cv["input"] == focus_input].copy()
    df_gl_f = df_gl[df_gl["input"] == focus_input].copy()

    # --- TCR-held-out (GroupKFold by tcr_name) ---
    plot_fold_metric(
        df_cv_f,
        out_path=os.path.join(plot_dir, "tcr_cv_rmse_by_fold.png"),
        title=f"TCR-held-out CV ({focus_input}): RMSE by fold",
        metric="rmse_all",
    )
    plot_rmse_boxplot(
        df_cv_f,
        out_path=os.path.join(plot_dir, "tcr_cv_rmse_boxplot.png"),
        title=f"TCR-held-out CV ({focus_input}): RMSE distribution",
        metric="rmse_all",
    )
    plot_fold_metric(
        df_cv_f,
        out_path=os.path.join(plot_dir, "tcr_cv_macro_rmse_by_fold.png"),
        title=f"TCR-held-out CV ({focus_input}): Macro RMSE by germline (by fold)",
        metric="macro_rmse_by_germline",
    )
    plot_micro_vs_macro(
        df_cv_f,
        out_path=os.path.join(plot_dir, "tcr_cv_micro_vs_macro.png"),
        title=f"TCR-held-out CV ({focus_input}): micro vs macro RMSE",
    )
    plot_angle_mae_bars(
        df_cv_f,
        out_path=os.path.join(plot_dir, "tcr_cv_angle_mae_deg.png"),
        title=f"TCR-held-out CV ({focus_input}): Angle MAE (deg) averaged over folds",
    )

    # --- Germline-holdout (unseen germlines) ---
    plot_fold_metric(
        df_gl_f,
        out_path=os.path.join(plot_dir, "germline_holdout_rmse_by_fold.png"),
        title=f"Germline-holdout ({focus_input}): RMSE by fold (unseen germlines)",
        metric="rmse_all",
    )
    plot_rmse_boxplot(
        df_gl_f,
        out_path=os.path.join(plot_dir, "germline_holdout_rmse_boxplot.png"),
        title=f"Germline-holdout ({focus_input}): RMSE distribution (unseen germlines)",
        metric="rmse_all",
    )
    plot_fold_metric(
        df_gl_f,
        out_path=os.path.join(plot_dir, "germline_holdout_macro_rmse_by_fold.png"),
        title=f"Germline-holdout ({focus_input}): Macro RMSE by germline (by fold)",
        metric="macro_rmse_by_germline",
    )
    plot_micro_vs_macro(
        df_gl_f,
        out_path=os.path.join(plot_dir, "germline_holdout_micro_vs_macro.png"),
        title=f"Germline-holdout ({focus_input}): micro vs macro RMSE",
    )
    plot_angle_mae_bars(
        df_gl_f,
        out_path=os.path.join(plot_dir, "germline_holdout_angle_mae_deg.png"),
        title=f"Germline-holdout ({focus_input}): Angle MAE (deg) averaged over folds",
    )

    print(f"[INFO] Wrote plots to: {plot_dir}")