import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_pca(proj_gt, proj_model, evr_gt, evr_pred, outfile):
    """
    Plot PC1 vs PC2 and label axes with variance captured on each dataset.
    evr_gt/evr_pred are 1D arrays of explained-variance ratios per PC
    computed under the fitted PCA basis (e.g., from evr_for_dataset).
    """
    import os, numpy as np, matplotlib.pyplot as plt


    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(proj_gt[:,0],    proj_gt[:,1], s=10, alpha=0.7, label="ground truth")
    if proj_model is not None:
        ax.scatter(proj_model[:,0], proj_model[:,1], s=10, alpha=0.7, label="model")
    if proj_gt is not None and proj_gt.shape[0] > 0:
        ax.scatter(
            proj_gt[0, 0],
            proj_gt[0, 1],
            marker="*",
            s=60,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label="Initial Frame (GT)",
        )
    if evr_gt is not None and evr_pred is not None:
        ax.set_xlabel(f"PC1 (GT {evr_gt[0]:.1f}%, Pred {evr_pred[0]:.1f}%)")
        ax.set_ylabel(f"PC2 (GT {evr_gt[1]:.1f}%, Pred {evr_pred[1]:.1f}%)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.set_title("PCA of selected atoms")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")  # optional
    fig.tight_layout()

    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_kde(
    fe_md,
    fe_model,
    fe_diff,
    xedges,
    yedges,
    outfile,
    gt_points=None,
    model_points=None,
    gt_initial=None,
    model_initial=None,
    gt_point_ids=None,
    model_point_ids=None,
):
    """
    Bare-bones plotting funnction for quick visualization of KDE PMF and their differences.

    gt_points / model_points:
        list/array of (x, y) points. The first point is treated as the
        highest-density point, and is plotted as 'peak'. Subsequent points
        are treated as selected extremes.

    gt_point_ids / model_point_ids:
        list of IDs (e.g. frame indices), same length/order as *_points.
        Used for labeling extremes as 'point_<ID>'.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 5))  # wider figure to fit 3 subplots

    # ---------- MD PMF ----------
    if fe_model is None:
        ax_md = plt.subplot(1, 1, 1)
    else:
        ax_md = plt.subplot(1, 3, 1)

    md_pmf = ax_md.imshow(
        fe_md,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="bilinear",
    )
    ax_md.set_title("MD PMF")
    cb = plt.colorbar(md_pmf, ax=ax_md)
    cb.set_label("FE (kJ/mol)")

    # overlay GT markers on MD panel
    if gt_points is not None and len(gt_points) > 0:
        pts = np.asarray(gt_points)
        # ids aligned with points (or None if not provided / mismatched)
        if gt_point_ids is None or len(gt_point_ids) != len(pts):
            gt_point_ids = [None] * len(pts)

        # first point = highest-density peak
        ax_md.scatter(
            pts[0, 0],
            pts[0, 1],
            marker="o",
            s=40,
            facecolors="red",
            edgecolors="black",
            linewidths=1.5,
            label="GT peak",
            zorder=6,
        )
        # label peak with its ID if we have one
        if gt_point_ids[0] is not None:
            ax_md.text(
                pts[0, 0],
                pts[0, 1],
                f"peak_{gt_point_ids[0]}",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
                zorder=7,
            )

        # remaining points = extremes (point_<id>)
        for i in range(1, len(pts)):
            ax_md.scatter(
                pts[i, 0],
                pts[i, 1],
                marker="o",
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                zorder=5,
            )
            pid = gt_point_ids[i]
            if pid is not None:
                ax_md.text(
                    pts[i, 0],
                    pts[i, 1],
                    f"point_{pid}",
                    fontsize=7,
                    color="red",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    if gt_initial is not None:
        ax_md.scatter(
            gt_initial[0],
            gt_initial[1],
            marker="*",
            s=60,
            edgecolors="red",
            facecolors="none",
            linewidths=1.5,
            label="GT frame 0",
            zorder=6,
        )

    if (gt_points is not None and len(gt_points) > 0) or (gt_initial is not None):
        ax_md.legend(frameon=False)

    # If no model PMF, save and exit
    if fe_model is None:
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        return

    # ---------- Model PMF ----------
    ax_model = plt.subplot(1, 3, 2)
    model_pmf = ax_model.imshow(
        fe_model,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="bilinear",
    )
    ax_model.set_title("Model PMF")
    cb = plt.colorbar(model_pmf, ax=ax_model)
    cb.set_label("FE (kJ/mol)")

    # overlay MODEL markers on model panel
    if model_points is not None and len(model_points) > 0:
        pts = np.asarray(model_points)
        if model_point_ids is None or len(model_point_ids) != len(pts):
            model_point_ids = [None] * len(pts)

        # first point = highest-density model peak
        ax_model.scatter(
            pts[0, 0],
            pts[0, 1],
            marker="s",
            s=40,
            facecolors="red",
            edgecolors="black",
            linewidths=1.5,
            label="Model peak",
            zorder=6,
        )
        if model_point_ids[0] is not None:
            ax_model.text(
                pts[0, 0],
                pts[0, 1],
                f"peak_{model_point_ids[0]}",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
                zorder=7,
            )

        # remaining points = extremes / nearest-to-GT, etc.
        for i in range(1, len(pts)):
            ax_model.scatter(
                pts[i, 0],
                pts[i, 1],
                marker="s",
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                zorder=5,
            )
            pid = model_point_ids[i]
            if pid is not None:
                ax_model.text(
                    pts[i, 0],
                    pts[i, 1],
                    f"point_{pid}",
                    fontsize=7,
                    color="red",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    if gt_initial is not None:
        ax_model.scatter(
            gt_initial[0],
            gt_initial[1],
            marker="*",
            s=60,
            edgecolors="red",
            facecolors="none",
            linewidths=1.5,
            label="GT frame 0",
            zorder=6,
        )

    if (model_points is not None and len(model_points) > 0) or (model_initial is not None):
        ax_model.legend(frameon=False)

    # ---------- Difference PMF ----------
    # ---------- Difference PMF ----------
    ax_diff = plt.subplot(1, 3, 3)

    if fe_diff is not None:
        import numpy as np
        import matplotlib.pyplot as plt

        #Make a float copy
        fe_diff_plot = np.array(fe_diff, dtype=float)

        # Symmetric colour limits around 0
        vmax = np.nanmax(np.abs(fe_diff_plot))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0  # safe fallback

        cmap = plt.cm.bwr.copy()      # centre of bwr is white
        ax_diff.set_facecolor("white")

        im = ax_diff.imshow(
            fe_diff,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",  # IMPORTANT: no bilinear blending, nearest for less blending
        )

        ax_diff.set_title("FE Difference (MD - Model)")
        ax_diff.set_xlabel("PC1")
        ax_diff.set_ylabel("PC2")

        cb = plt.colorbar(im, ax=ax_diff)
        cb.set_label("FE difference (kJ/mol)")
    else:
        ax_diff.axis("off")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()



def plot_pmf(
    fe_md,
    fe_model,
    fe_diff,
    xedges,
    yedges,
    outfile,
    gt_points=None,
    model_points=None,
    gt_initial=None,
    model_initial=None,
    gt_point_ids=None,
    model_point_ids=None,
):
    """
    Bare-bones plotting funnction for quick visualization of PMF and their differences.

    Same labeling convention as plot_kde:
      - first point in *_points = 'peak'
      - others are annotated as 'point_<ID>' using *_point_ids.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 5))

    # ---------- MD PMF ----------
    if fe_model is None:
        ax_md = plt.subplot(1, 1, 1)
    else:
        ax_md = plt.subplot(1, 3, 1)

    md_pmf = ax_md.imshow(
        fe_md,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    ax_md.set_title("MD PMF")
    cb = plt.colorbar(md_pmf, ax=ax_md)
    cb.set_label("FE (kJ/mol)")

    # overlay GT markers on MD panel
    if gt_points is not None and len(gt_points) > 0:
        pts = np.asarray(gt_points)
        if gt_point_ids is None or len(gt_point_ids) != len(pts):
            gt_point_ids = [None] * len(pts)

        # peak
        ax_md.scatter(
            pts[0, 0],
            pts[0, 1],
            marker="o",
            s=40,
            facecolors="red",
            edgecolors="black",
            linewidths=1.5,
            label="GT peak",
            zorder=6,
        )
        if gt_point_ids[0] is not None:
            ax_md.text(
                pts[0, 0],
                pts[0, 1],
                f"peak_{gt_point_ids[0]}",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
                zorder=7,
            )

        for i in range(1, len(pts)):
            ax_md.scatter(
                pts[i, 0],
                pts[i, 1],
                marker="o",
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                zorder=5,
            )
            pid = gt_point_ids[i]
            if pid is not None:
                ax_md.text(
                    pts[i, 0],
                    pts[i, 1],
                    f"point_{pid}",
                    fontsize=7,
                    color="red",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    if gt_initial is not None:
        ax_md.scatter(
            gt_initial[0],
            gt_initial[1],
            marker="*",
            s=60,
            edgecolors="red",
            facecolors="none",
            linewidths=1.5,
            label="GT frame 0",
            zorder=6,
        )

    if (gt_points is not None and len(gt_points) > 0) or (gt_initial is not None):
        ax_md.legend(frameon=False)

    # If no model PMF, save and exit
    if fe_model is None:
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
        return

    # ---------- Model PMF ----------
    ax_model = plt.subplot(1, 3, 2)
    model_pmf = ax_model.imshow(
        fe_model,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )
    ax_model.set_title("Model PMF")
    cb = plt.colorbar(model_pmf, ax=ax_model)
    cb.set_label("FE (kJ/mol)")

    if model_points is not None and len(model_points) > 0:
        pts = np.asarray(model_points)
        if model_point_ids is None or len(model_point_ids) != len(pts):
            model_point_ids = [None] * len(pts)

        # peak
        ax_model.scatter(
            pts[0, 0],
            pts[0, 1],
            marker="s",
            s=40,
            facecolors="red",
            edgecolors="black",
            linewidths=1.5,
            label="Model peak",
            zorder=6,
        )
        if model_point_ids[0] is not None:
            ax_model.text(
                pts[0, 0],
                pts[0, 1],
                f"peak_{model_point_ids[0]}",
                fontsize=7,
                color="black",
                ha="left",
                va="bottom",
                zorder=7,
            )

        for i in range(1, len(pts)):
            ax_model.scatter(
                pts[i, 0],
                pts[i, 1],
                marker="s",
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.2,
                zorder=5,
            )
            pid = model_point_ids[i]
            if pid is not None:
                ax_model.text(
                    pts[i, 0],
                    pts[i, 1],
                    f"point_{pid}",
                    fontsize=7,
                    color="red",
                    ha="left",
                    va="bottom",
                    zorder=6,
                )

    if gt_initial is not None:
        ax_model.scatter(
            gt_initial[0],
            gt_initial[1],
            marker="*",
            s=60,
            edgecolors="red",
            facecolors="none",
            linewidths=1.5,
            label="GT frame 0",
            zorder=6,
        )

    if (model_points is not None and len(model_points) > 0) or (model_initial is not None):
        ax_model.legend(frameon=False)

    # ---------- Difference PMF ----------
    ax_diff = plt.subplot(1, 3, 3)
    #Make a float copy
    fe_diff_plot = np.array(fe_diff, dtype=float)

    # Symmetric colour limits around 0
    vmax = np.nanmax(np.abs(fe_diff_plot))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0  # safe fallback

    cmap = plt.cm.bwr.copy()      # centre of bwr is white
    ax_diff.set_facecolor("white")

    im = ax_diff.imshow(
        fe_diff,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",  # IMPORTANT: no bilinear blending
    )
    ax_diff.set_title("FE Difference (MD - Model)")
    cb = plt.colorbar(im, ax=ax_diff)
    cb.set_label("FE difference (kJ/mol)")

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

