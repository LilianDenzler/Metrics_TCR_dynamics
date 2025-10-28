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
    ax.scatter(proj_model[:,0], proj_model[:,1], s=10, alpha=0.7, label="model")
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


def plot_kde(fe_md, fe_model, fe_diff, xedges, yedges, outfile):
    """
    Bare-bones plotting funnction for quick visualization of KDE PMF and their differences.
    """
    plt.figure(figsize=(18,5))  # wider figure to fit 3 subplots

    # MD PMF
    plt.subplot(1,3,1)
    plt.imshow(
        fe_md,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation='bilinear'
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('MD PMF')

    # Model PMF
    plt.subplot(1,3,2)
    plt.imshow(
        fe_model,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation='bilinear'
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('Model PMF')

    # Difference PMF
    plt.subplot(1,3,3)
    plt.imshow(
        fe_diff,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='bwr',  # blue-red diverging colormap for differences
        interpolation='bilinear'
    )
    plt.colorbar(label='FE difference (kJ/mol)')
    plt.title('FE Difference (MD - Model)')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_pmf(fe_md, fe_model, fe_diff, xedges, yedges, outfile):
    """
    Bare-bones plotting funnction for quick visualization of PMF and their differences.
    """

    plt.figure(figsize=(18,5))

    # MD PMF
    plt.subplot(1,3,1)
    plt.imshow(
        fe_md,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('MD PMF')

    # Model PMF
    plt.subplot(1,3,2)
    plt.imshow(
        fe_model,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )
    plt.colorbar(label='FE (kJ/mol)')
    plt.title('Model PMF')

    # Difference PMF
    plt.subplot(1,3,3)
    plt.imshow(
        fe_diff,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='bwr'  # blue-red diverging colormap for differences
    )
    plt.colorbar(label='FE difference (kJ/mol)')
    plt.title('FE Difference (MD - Model)')

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

