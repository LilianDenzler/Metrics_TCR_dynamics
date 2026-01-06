from plotter_analysis import *

# ---------------------------
# PCA utilities + plots
# ---------------------------
def pca_fit_and_project(
    df_fit: pd.DataFrame,
    df_proj_list: List[Tuple[str, pd.DataFrame]],
    features: List[str],
    seed: int,
) -> Tuple[StandardScaler, PCA, Dict[str, pd.DataFrame]]:
    """
    Fit scaler+PCA on df_fit (using features), then transform each df in df_proj_list.
    Returns scaler, pca, and dict of coords dataframes with PC1/PC2 columns.
    """
    def prep(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        for c in features:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        # drop NaNs for these features
        mask = np.isfinite(out[features].values).all(axis=1)
        return out.loc[mask].copy()

    fit_clean = prep(df_fit)
    X_fit = fit_clean[features].values.astype(float)

    scaler = StandardScaler()
    Xs_fit = scaler.fit_transform(X_fit)

    pca = PCA(n_components=2, random_state=seed)
    Z_fit = pca.fit_transform(Xs_fit)
    fit_coords = fit_clean.copy()
    fit_coords["PC1"] = Z_fit[:, 0]
    fit_coords["PC2"] = Z_fit[:, 1]

    coords_dict: Dict[str, pd.DataFrame] = {"fit": fit_coords}

    for name, dproj in df_proj_list:
        proj_clean = prep(dproj)
        if proj_clean.empty:
            coords_dict[name] = proj_clean.assign(PC1=np.nan, PC2=np.nan)
            continue
        Xp = proj_clean[features].values.astype(float)
        Zp = pca.transform(scaler.transform(Xp))
        proj_coords = proj_clean.copy()
        proj_coords["PC1"] = Zp[:, 0]
        proj_coords["PC2"] = Zp[:, 1]
        coords_dict[name] = proj_coords

    return scaler, pca, coords_dict

def pca_plot_groups(
    groups: List[Tuple[str, pd.DataFrame]],
    out_path: Path,
    title: str,
    subtitle: str,
    var1: float,
    var2: float,
):
    fig = plt.figure(figsize=(9.0, 7.3))
    ax = fig.add_subplot(1, 1, 1)

    for gname, gdf in groups:
        if gdf is None or gdf.empty:
            continue
        ax.scatter(
            gdf["PC1"].values,
            gdf["PC2"].values,
            s=20.0,
            alpha=0.80,
            edgecolors="none",
            label=f"{gname} (n={len(gdf)})",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\n{subtitle}\nPC1={var1*100:.1f}%, PC2={var2*100:.1f}%")
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def pca_plot_color_germline_marker_group(
    groups: List[Tuple[str, pd.DataFrame]],
    out_path: Path,
    title: str,
    var1: float,
    var2: float,
    top_n_germlines: int,
    min_germline_count: int,
    seed: int,
):
    """
    Color by germline_collapsed, marker by group.
    """
    # concatenate for global germline palette + counts
    all_df = []
    for gname, gdf in groups:
        if gdf is None or gdf.empty:
            continue
        tmp = gdf[["PC1", "PC2", "germline_collapsed"]].copy()
        tmp["group"] = gname
        all_df.append(tmp)
    if not all_df:
        return
    d = pd.concat(all_df, ignore_index=True)
    vc = d["germline_collapsed"].value_counts()
    cats = vc.index.tolist()
    cm = category_colors(cats)

    marker_cycle = ["o", "^", "s", "D", "v", "P", "X"]
    group_names = [g for g, _ in groups]
    marker_map = {g: marker_cycle[i % len(marker_cycle)] for i, g in enumerate(group_names)}

    fig = plt.figure(figsize=(11.5, 7.6))
    ax = fig.add_subplot(1, 1, 1)

    for gname, gdf in groups:
        if gdf is None or gdf.empty:
            continue
        mk = marker_map[gname]
        # plot per germline for clean legend
        for cat in cats:
            m = (gdf["germline_collapsed"].values == cat)
            if m.sum() == 0:
                continue
            ax.scatter(
                gdf.loc[m, "PC1"].values,
                gdf.loc[m, "PC2"].values,
                s=22.0,
                alpha=0.80,
                edgecolors="none",
                marker=mk,
                c=[cm.get(cat, (0.6, 0.6, 0.6, 0.6))],
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{title}\nPC1={var1*100:.1f}%, PC2={var2*100:.1f}%")

    # Germline legend
    germ_handles = []
    for cat in cats:
        germ_handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                       markerfacecolor=cm.get(cat, (0.6, 0.6, 0.6, 0.6)),
                       markeredgecolor="none",
                       label=f"{cat} (n={int(vc[cat])})")
        )

    # Group legend (markers only)
    group_handles = []
    for gname in group_names:
        group_handles.append(
            plt.Line2D([0], [0], marker=marker_map[gname], linestyle="", markersize=7,
                       markerfacecolor="black", markeredgecolor="none",
                       label=gname)
        )

    leg1 = ax.legend(handles=germ_handles, title=f"germline (top {top_n_germlines}, min n={min_germline_count})",
                     loc="center left", bbox_to_anchor=(1.02, 0.65),
                     fontsize=8, title_fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=group_handles, title="dataset", loc="center left", bbox_to_anchor=(1.02, 0.20),
              fontsize=9, title_fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_pca_plots()
    #---------------------------
    # 4) PCA: fit on UNPAIRED UNBOUND using BA..dc only; project others
    # ---------------------------
    pca_dir = out_dir / "pca_fit_on_unpaired_unbound"
    pca_dir.mkdir(parents=True, exist_ok=True)

    pca_features = [f for f in BASE_FEATURES if f in df_all.columns]
    if len(pca_features) != 6:
        raise RuntimeError(f"PCA requires BA,BC1,AC1,BC2,AC2,dc; found: {pca_features}")

    df_fit = df_all[df_all["dataset"] == "unpaired_unbound"].copy()

    df_proj_unpaired_bound = df_all[df_all["dataset"] == "unpaired_bound"].copy()
    df_proj_paired_unbound = df_all[df_all["dataset"] == "paired_unbound"].copy()
    df_proj_paired_bound   = df_all[df_all["dataset"] == "paired_bound"].copy()

    scaler, pca, coords = pca_fit_and_project(
        df_fit=df_fit,
        df_proj_list=[
            ("unpaired_bound", df_proj_unpaired_bound),
            ("paired_unbound", df_proj_paired_unbound),
            ("paired_bound", df_proj_paired_bound),
        ],
        features=pca_features,
        seed=args.seed,
    )

    # Save coords tables
    coords["fit"].to_csv(pca_dir / "coords__fit_unpaired_unbound.csv", index=False)
    coords["unpaired_bound"].to_csv(pca_dir / "coords__proj_unpaired_bound.csv", index=False)
    coords["paired_unbound"].to_csv(pca_dir / "coords__proj_paired_unbound.csv", index=False)
    coords["paired_bound"].to_csv(pca_dir / "coords__proj_paired_bound.csv", index=False)

    # Save PCA loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_features,
        columns=["PC1_loading", "PC2_loading"],
    )
    loadings.to_csv(pca_dir / "pca_loadings.csv")

    # PCA plot: unpaired unbound (fit) + unpaired bound (projected)
    var1, var2 = float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])
    pca_plot_groups(
        groups=[
            ("unpaired_unbound (fit)", coords["fit"]),
            ("unpaired_bound (proj)", coords["unpaired_bound"]),
        ],
        out_path=pca_dir / "pca__fit_unpaired_unbound__project_unpaired_bound__plain.png",
        title="PCA fit on unpaired_unbound (BA,BC1,AC1,BC2,AC2,dc)",
        subtitle="Unpaired bound projected into PCA space",
        var1=var1,
        var2=var2,
    )

    # PCA plot: also project paired datasets
    pca_plot_groups(
        groups=[
            ("unpaired_unbound (fit)", coords["fit"]),
            ("paired_unbound (proj)", coords["paired_unbound"]),
            ("paired_bound (proj)", coords["paired_bound"]),
        ],
        out_path=pca_dir / "pca__fit_unpaired_unbound__project_paired__plain.png",
        title="PCA fit on unpaired_unbound (BA,BC1,AC1,BC2,AC2,dc)",
        subtitle="Paired unbound/bound projected into PCA space",
        var1=var1,
        var2=var2,
    )

    # Germline-collapsed columns for PCA coords
    # (reuse global collapse mapping for consistency)
    for k in coords:
        if coords[k] is None or coords[k].empty:
            continue
        coords[k]["germline_collapsed"] = coords[k]["germline_collapsed"].fillna("other").astype(str)

    # Germline-colored PCA: unpaired fit + unpaired bound
    pca_plot_color_germline_marker_group(
        groups=[
            ("unpaired_unbound (fit)", coords["fit"]),
            ("unpaired_bound (proj)", coords["unpaired_bound"]),
        ],
        out_path=pca_dir / "pca__fit_unpaired_unbound__project_unpaired_bound__by_germline.png",
        title="PCA fit on unpaired_unbound; color=germline, marker=dataset",
        var1=var1,
        var2=var2,
        top_n_germlines=args.top_n_germlines,
        min_germline_count=args.min_germline_count,
        seed=args.seed,
    )

    # Germline-colored PCA: unpaired fit + paired projections
    pca_plot_color_germline_marker_group(
        groups=[
            ("unpaired_unbound (fit)", coords["fit"]),
            ("paired_unbound (proj)", coords["paired_unbound"]),
            ("paired_bound (proj)", coords["paired_bound"]),
        ],
        out_path=pca_dir / "pca__fit_unpaired_unbound__project_paired__by_germline.png",
        title="PCA fit on unpaired_unbound; paired projected; color=germline, marker=dataset",
        var1=var1,
        var2=var2,
        top_n_germlines=args.top_n_germlines,
        min_germline_count=args.min_germline_count,
        seed=args.seed,
    )

    # Save a small PCA summary
    pd.DataFrame([{
        "fit_dataset": "unpaired_unbound",
        "features": ",".join(pca_features),
        "pc1_var": var1,
        "pc2_var": var2,
        "pc1_pc2_cum_var": var1 + var2,
        "n_fit": int(len(coords["fit"])),
        "n_proj_unpaired_bound": int(len(coords["unpaired_bound"])),
        "n_proj_paired_unbound": int(len(coords["paired_unbound"])),
        "n_proj_paired_bound": int(len(coords["paired_bound"])),
    }]).to_csv(pca_dir / "pca_summary.csv", index=False)

    print("Done.")
    print(f"Outputs written to: {out_dir}")
