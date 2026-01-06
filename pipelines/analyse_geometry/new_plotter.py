# new_plotter_nature.py
# Nature-style plotting for TCR geometry:
# - Split violins (Unpaired vs Paired; within group: unbound vs bound side-by-side) + IQR + median
# - Paired scatters (plain + germline-colored) with categorical palette + legend
# - Ridgelines (density per germline) with robust KDE (handles singular covariance)
#
# Outputs:
#   - PNG (300 dpi) and PDF (vector) for each plot

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


FEATURES = [
    "BA", "BC1", "AC1", "BC2", "AC2", "dc",
    "alpha_cdr3_bend_deg", "alpha_cdr3_apex_height_A", "alpha_cdr3_apex_resi",
    "beta_cdr3_bend_deg", "beta_cdr3_apex_height_A", "beta_cdr3_apex_resi",
]


# -----------------------
# Nature-like style
# -----------------------
def apply_nature_style() -> None:
    """
    A practical 'Nature journal' aesthetic:
      - clean axes (no top/right spines)
      - small fonts
      - thin strokes
      - inward ticks with minors
      - tight layout
      - vector-friendly output
    """
    mpl.rcParams.update({
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Fonts (Helvetica-like; falls back if unavailable)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,   # TrueType (editable text)
        "ps.fonttype": 42,

        # Sizes (Nature often ~7–9 pt)
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,

        # Lines
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.9,
        "patch.linewidth": 0.6,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # Axes/legend
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.2,
    })


def _finish_axes(ax: mpl.axes.Axes) -> None:
    ax.minorticks_on()
    ax.grid(False)


def _save_fig(fig: mpl.figure.Figure, out_base: Path) -> None:
    """
    Save both PDF (vector) and PNG (raster).
    out_base should be full path without extension.
    """
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=300)


# -----------------------
# IO helpers
# -----------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def infer_join_key(df: pd.DataFrame) -> str:
    candidates = [
        "tcr_id", "TCR_id", "tcr", "TCR", "tcr_name", "TCR_name",
        "pdb", "pdb_id", "pdbcode", "pdb_code", "structure_id", "id",
        "uuid", "chain_pair_id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return "__index__"


def ensure_germline_vj_pair(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "germline_vj_pair" in df.columns:
        return df

    vj_cols = {"alpha_v_gene", "alpha_j_gene", "beta_v_gene", "beta_j_gene"}
    if vj_cols.issubset(df.columns):
        df["germline_vj_pair"] = (
            df["alpha_v_gene"].astype(str) + "_" + df["alpha_j_gene"].astype(str)
            + "__"
            + df["beta_v_gene"].astype(str) + "_" + df["beta_j_gene"].astype(str)
        )
        return df

    if {"alpha_germline", "beta_germline"}.issubset(df.columns):
        df["germline_vj_pair"] = df["alpha_germline"].astype(str) + "__" + df["beta_germline"].astype(str)
        return df

    df["germline_vj_pair"] = "Unknown"
    return df


def load_four_conditions(data_dir: Path, prefix: str = "") -> Dict[Tuple[str, str], pd.DataFrame]:
    mapping = {
        ("unpaired", "unbound"): data_dir / f"{prefix}TCR3d_unpaired_unbound.csv",
        ("unpaired", "bound"):   data_dir / f"{prefix}TCR3d_unpaired_bound.csv",
        ("paired", "unbound"):   data_dir / f"{prefix}TCR3d_paired_unbound.csv",
        ("paired", "bound"):     data_dir / f"{prefix}TCR3d_paired_bound.csv",
    }

    out = {}
    for (pairing, state), path in mapping.items():
        df = _read_csv(path)
        df["pairing"] = pairing
        df["state"] = state
        df = ensure_germline_vj_pair(df)
        out[(pairing, state)] = df
    return out


# -----------------------
# Germline palette (categorical)
# -----------------------
def categorical_palette(n: int):
    colors = []
    # tab20 family yields reasonably separable categorical colors
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = mpl.colormaps.get_cmap(name)
        colors.extend([cmap(i) for i in range(cmap.N)])

    if n <= len(colors):
        return colors[:n]

    # fallback: hsv hues if truly many classes
    extra = n - len(colors)
    hsv = mpl.colormaps.get_cmap("hsv")
    hues = np.linspace(0, 1, extra, endpoint=False)
    colors.extend([hsv(h) for h in hues])
    return colors[:n]


# -----------------------
# Split violins (robust side-by-side within group)
# -----------------------
def _is_discreteish(values: np.ndarray, feat_name: str) -> bool:
    if feat_name.endswith("_resi"):
        return True
    v = values[np.isfinite(values)]
    if v.size == 0:
        return False
    uniq = np.unique(v)
    return (uniq.size <= 10) or (uniq.size / max(1, v.size) < 0.05)


def plot_split_violins(
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    out_dir: Path,
    features: List[str],
    rng_seed: int = 1234,
) -> None:
    """
    Nature-style split violin:
      - two x-groups: Unpaired, Paired
      - within each group: unbound vs bound side-by-side narrow violins
      - overlay IQR (25–75%) and median
    """
    _ensure_dir(out_dir)
    rng = np.random.default_rng(rng_seed)

    group_pos = {"unpaired": 1.0, "paired": 2.0}
    dx = 0.18
    viol_width = 0.28

    # Keep a restrained, publication-friendly two-condition scheme
    # (blue/orange is common and distinguishes well in print)
    color_unbound = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
    color_bound = mpl.rcParams["axes.prop_cycle"].by_key()["color"][1]

    def _vals(pairing, state, feat):
        df = dfs[(pairing, state)]
        if feat not in df.columns:
            return np.array([])
        arr = pd.to_numeric(df[feat], errors="coerce").to_numpy()
        return arr[np.isfinite(arr)]

    def _kde_vals(raw_vals, feat_name):
        if raw_vals.size == 0:
            return raw_vals
        if _is_discreteish(raw_vals, feat_name):
            # tiny jitter ONLY for drawing violin KDE so it does not collapse
            return raw_vals + rng.normal(0.0, 0.15, size=raw_vals.shape)
        return raw_vals

    def _draw_violin(ax, vals_for_kde, x, facecolor):
        if vals_for_kde.size == 0:
            return
        parts = ax.violinplot(
            [vals_for_kde],
            positions=[x],
            widths=viol_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        body.set_facecolor(facecolor)
        body.set_edgecolor("black")
        body.set_alpha(0.30)
        body.set_linewidth(0.6)

    def _draw_iqr_median(ax, raw_vals, x, color):
        if raw_vals.size == 0:
            return
        q1, med, q3 = np.percentile(raw_vals, [25, 50, 75])
        ax.plot([x, x], [q1, q3], linewidth=1.6, color=color, zorder=3)
        ax.plot([x], [med], marker="o", markersize=3.8, color=color, zorder=4)

    for feat in features:
        if not any(feat in df.columns for df in dfs.values()):
            continue

        u_unpaired = _vals("unpaired", "unbound", feat)
        b_unpaired = _vals("unpaired", "bound", feat)
        u_paired = _vals("paired", "unbound", feat)
        b_paired = _vals("paired", "bound", feat)

        u_unpaired_k = _kde_vals(u_unpaired.astype(float), feat)
        b_unpaired_k = _kde_vals(b_unpaired.astype(float), feat)
        u_paired_k = _kde_vals(u_paired.astype(float), feat)
        b_paired_k = _kde_vals(b_paired.astype(float), feat)

        fig, ax = plt.subplots(figsize=(3.35, 2.25))  # Nature single-column-ish

        # Unpaired group
        x0 = group_pos["unpaired"]
        _draw_violin(ax, u_unpaired_k, x0 - dx, color_unbound)
        _draw_violin(ax, b_unpaired_k, x0 + dx, color_bound)
        _draw_iqr_median(ax, u_unpaired, x0 - dx, color_unbound)
        _draw_iqr_median(ax, b_unpaired, x0 + dx, color_bound)

        # Paired group
        x1 = group_pos["paired"]
        _draw_violin(ax, u_paired_k, x1 - dx, color_unbound)
        _draw_violin(ax, b_paired_k, x1 + dx, color_bound)
        _draw_iqr_median(ax, u_paired, x1 - dx, color_unbound)
        _draw_iqr_median(ax, b_paired, x1 + dx, color_bound)

        ax.set_title(feat, pad=4)
        ax.set_ylabel(feat)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels(["Unpaired", "Paired"])

        _finish_axes(ax)

        # concise legend (Nature style: small, unobtrusive)
        handles = [
            mpl.patches.Patch(facecolor=color_unbound, edgecolor="black", alpha=0.30, label="Unbound"),
            mpl.patches.Patch(facecolor=color_bound, edgecolor="black", alpha=0.30, label="Bound"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True)

        caption = (
            f"Unpaired: unbound n={len(u_unpaired)}, bound n={len(b_unpaired)}; "
            f"Paired: unbound n={len(u_paired)}, bound n={len(b_paired)}"
        )
        fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=6)

        fig.tight_layout()
        _save_fig(fig, out_dir / f"split_violin_{feat}")
        plt.close(fig)


# -----------------------
# Paired scatters
# -----------------------
def _prepare_paired_merge(paired_unbound: pd.DataFrame, paired_bound: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df_u = paired_unbound.copy()
    df_b = paired_bound.copy()

    key_u = infer_join_key(df_u)
    key_b = infer_join_key(df_b)

    if key_u != "__index__" and key_u in df_u.columns and key_u in df_b.columns:
        key = key_u
    elif key_b != "__index__" and key_b in df_u.columns and key_b in df_b.columns:
        key = key_b
    else:
        df_u["__index__"] = np.arange(len(df_u))
        df_b["__index__"] = np.arange(len(df_b))
        key = "__index__"

    merged = pd.merge(df_u, df_b, on=key, suffixes=("_unbound", "_bound"), how="inner")
    return merged, key


def plot_paired_scatters(
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    out_dir: Path,
    features: List[str],
    max_legend_entries: int = 25,
) -> None:
    _ensure_dir(out_dir)

    paired_unbound = dfs[("paired", "unbound")]
    paired_bound = dfs[("paired", "bound")]
    merged, join_key = _prepare_paired_merge(paired_unbound, paired_bound)

    germ_col = None
    for c in ["germline_vj_pair_unbound", "germline_vj_pair_bound"]:
        if c in merged.columns:
            germ_col = c
            break
    if germ_col is None:
        raise ValueError("Could not find germline_vj_pair column in merged paired data.")

    merged[germ_col] = merged[germ_col].astype(str).fillna("NaN")

    def _counts_caption(counts: pd.Series, limit: int = 40) -> str:
        items = [f"{g}: n={int(c)}" for g, c in counts.items()][:limit]
        cap = " | ".join(items)
        if len(counts) > limit:
            cap += f" | ... (+{len(counts)-limit} more)"
        return cap

    for feat in features:
        xcol = f"{feat}_unbound"
        ycol = f"{feat}_bound"
        if xcol not in merged.columns or ycol not in merged.columns:
            continue

        tmp = merged[[xcol, ycol, germ_col]].copy()
        tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
        tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
        tmp = tmp.dropna(subset=[xcol, ycol])
        if tmp.empty:
            continue

        # Plain scatter (single-column size)
        fig, ax = plt.subplots(figsize=(3.35, 3.0))
        ax.scatter(tmp[xcol].values, tmp[ycol].values, s=10, alpha=0.85, linewidths=0)
        ax.set_title(feat, pad=4)
        ax.set_xlabel("Unbound")
        ax.set_ylabel("Bound")
        _finish_axes(ax)

        caption = f"Paired; join={join_key}; n={len(tmp)}"
        fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=6)

        fig.tight_layout()
        _save_fig(fig, out_dir / f"paired_scatter_{feat}")
        plt.close(fig)

        # Germline-colored scatter (wider figure to accommodate legend)
        counts_tmp = tmp[germ_col].value_counts()
        top_germs = counts_tmp.index[:max_legend_entries].tolist()
        tmp["_germ_legend"] = np.where(tmp[germ_col].isin(top_germs), tmp[germ_col], "Other")

        legend_counts = tmp["_germ_legend"].value_counts()
        legend_groups = legend_counts.index.tolist()

        colors = categorical_palette(len(legend_groups))
        group_to_color = {g: colors[i] for i, g in enumerate(legend_groups)}

        fig, ax = plt.subplots(figsize=(6.8, 3.0))  # two-column-ish width
        for g in legend_groups:
            sub = tmp[tmp["_germ_legend"] == g]
            ax.scatter(
                sub[xcol].values, sub[ycol].values,
                s=12, alpha=0.9, linewidths=0,
                color=group_to_color[g],
                label=f"{g} (n={int(legend_counts[g])})",
            )

        ax.set_title(f"{feat}", pad=4)
        ax.set_xlabel("Unbound")
        ax.set_ylabel("Bound")
        _finish_axes(ax)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=6,
            title="germline_vj_pair",
        )

        caption = _counts_caption(counts_tmp, limit=40)
        if len(counts_tmp) > max_legend_entries:
            caption += f" || Legend shows top {max_legend_entries}; remaining grouped as 'Other'."
        fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=6)

        fig.tight_layout()
        _save_fig(fig, out_dir / f"paired_scatter_germline_{feat}")
        plt.close(fig)


# -----------------------
# Ridgelines (robust KDE)
# -----------------------
def _safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    return float(np.std(x))


def _gaussian_spike(grid: np.ndarray, center: float, sigma: float) -> np.ndarray:
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = (grid.max() - grid.min()) / 200.0 if grid.max() > grid.min() else 1.0
    return np.exp(-0.5 * ((grid - center) / sigma) ** 2)


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.zeros_like(grid)

    vstd = _safe_std(values)
    grid_span = float(np.nanmax(grid) - np.nanmin(grid)) if grid.size else 1.0
    eps = max(1e-12, 1e-6 * grid_span)

    if vstd < eps:
        center = float(np.nanmedian(values))
        sigma = max(eps, grid_span / 200.0)
        return _gaussian_spike(grid, center, sigma)

    try:
        kde = gaussian_kde(values)
        y = kde(grid)
        if not np.all(np.isfinite(y)) or np.nanmax(y) <= 0:
            raise ValueError("KDE produced non-finite or zero density.")
        return y
    except Exception:
        n = values.size
        bw = vstd * (n ** (-1/5))
        bw = max(bw, eps, grid_span / 300.0)
        inv = 1.0 / bw
        y = np.zeros_like(grid, dtype=float)
        for v in values:
            y += np.exp(-0.5 * ((grid - v) * inv) ** 2)
        return y


def plot_ridgelines_for_pairing(
    df_unbound: pd.DataFrame,
    df_bound: pd.DataFrame,
    feature: str,
    out_path_base: Path,
    pairing_label: str,
    max_germlines: int = 25,
) -> None:
    if feature not in df_unbound.columns or feature not in df_bound.columns:
        return

    du = df_unbound.copy()
    db = df_bound.copy()
    du[feature] = pd.to_numeric(du[feature], errors="coerce")
    db[feature] = pd.to_numeric(db[feature], errors="coerce")
    du = du.dropna(subset=[feature])
    db = db.dropna(subset=[feature])
    if du.empty and db.empty:
        return

    all_germs = pd.concat([du["germline_vj_pair"], db["germline_vj_pair"]], ignore_index=True).astype(str)
    germ_order = all_germs.value_counts().head(max_germlines).index.tolist()

    vals_all = pd.concat([du[feature], db[feature]], ignore_index=True).dropna().values
    if len(vals_all) < 2:
        return

    xmin, xmax = np.nanpercentile(vals_all, [1, 99])
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = np.nanmin(vals_all), np.nanmax(vals_all)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
    if feature.endswith("_resi"):
        xmin -= 1.0
        xmax += 1.0
    grid = np.linspace(xmin, xmax, 400)

    germ_counts_u = du["germline_vj_pair"].astype(str).value_counts()
    germ_counts_b = db["germline_vj_pair"].astype(str).value_counts()

    # Two-column-ish figure, monochrome fills (Nature-friendly)
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(6.8, max(2.6, 0.20 * len(germ_order) + 1.6)),
        sharex=True
    )
    ax_u, ax_b = axes[0], axes[1]
    ax_u.set_title(f"{pairing_label} Unbound", pad=4)
    ax_b.set_title(f"{pairing_label} Bound", pad=4)

    y_positions = np.arange(len(germ_order))

    for i, germ in enumerate(germ_order):
        vu = du.loc[du["germline_vj_pair"].astype(str) == germ, feature].values
        vb = db.loc[db["germline_vj_pair"].astype(str) == germ, feature].values

        yu = _kde_curve(vu, grid)
        yb = _kde_curve(vb, grid)

        if yu.max() > 0:
            yu = yu / yu.max()
        if yb.max() > 0:
            yb = yb / yb.max()

        offset = y_positions[i]
        ax_u.fill_between(grid, offset, offset + 0.75 * yu, alpha=0.85)
        ax_b.fill_between(grid, offset, offset + 0.75 * yb, alpha=0.85)

    for ax in (ax_u, ax_b):
        ax.set_yticks(y_positions)
        ax.set_yticklabels(germ_order, fontsize=6)
        ax.set_xlabel(feature)
        _finish_axes(ax)

    caption_items = []
    for germ in germ_order[:25]:
        caption_items.append(
            f"{germ}: unbound n={int(germ_counts_u.get(germ, 0))}, bound n={int(germ_counts_b.get(germ, 0))}"
        )
    caption = " | ".join(caption_items)
    if len(germ_order) > 25:
        caption += f" | ... (+{len(germ_order)-25} more)"

    fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=6)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    _save_fig(fig, out_path_base)
    plt.close(fig)


def plot_ridgelines(
    dfs: Dict[Tuple[str, str], pd.DataFrame],
    out_dir: Path,
    features: List[str]
) -> None:
    _ensure_dir(out_dir)
    for feat in features:
        plot_ridgelines_for_pairing(
            df_unbound=dfs[("unpaired", "unbound")],
            df_bound=dfs[("unpaired", "bound")],
            feature=feat,
            out_path_base=out_dir / f"ridgeline_unpaired_{feat}",
            pairing_label="Unpaired",
        )
        plot_ridgelines_for_pairing(
            df_unbound=dfs[("paired", "unbound")],
            df_bound=dfs[("paired", "bound")],
            feature=feat,
            out_path_base=out_dir / f"ridgeline_paired_{feat}",
            pairing_label="Paired",
        )


# -----------------------
# Driver
# -----------------------
def run_all(data_dir: str, out_root: str) -> None:
    apply_nature_style()

    data_dir = Path(data_dir)
    out_root = Path(out_root)
    _ensure_dir(out_root)

    variants = [
        ("normal", ""),
        ("human_only", "human_"),
        ("mouse_only", "mouse_"),
    ]

    for variant_name, prefix in variants:
        print(f"\n=== Plotting variant: {variant_name} (prefix='{prefix}') ===")
        dfs = load_four_conditions(data_dir=data_dir, prefix=prefix)

        variant_dir = out_root / variant_name
        viol_dir = variant_dir / "violins_split_iqr_nature"
        scat_dir = variant_dir / "paired_scatters_nature"
        ridge_dir = variant_dir / "ridgelines_nature"

        _ensure_dir(viol_dir)
        _ensure_dir(scat_dir)
        _ensure_dir(ridge_dir)

        plot_split_violins(dfs, viol_dir, FEATURES)
        plot_paired_scatters(dfs, scat_dir, FEATURES, max_legend_entries=25)
        plot_ridgelines(dfs, ridge_dir, FEATURES)

        print(f"Saved: {variant_dir}")


if __name__ == "__main__":
    DATA_DIR = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/"
    OUT_ROOT = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/plots_new/"
    run_all(DATA_DIR, OUT_ROOT)
