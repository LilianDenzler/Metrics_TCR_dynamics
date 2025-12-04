import argparse, numpy as np
from TCR_TOOLS.scoring.embeddings.features import TVWrap
from TCR_TOOLS.scoring.embeddings.dim_reduction import fit_pca_linear, fit_kpca,fit_pca_linear, fit_kpca, fit_tica, fit_diffmap

from TCR_TOOLS.scoring.embeddings.metrics import trustworthiness, mantel_rmsd_vs_embedding
from TCR_TOOLS.scoring.embeddings.utils import save_npz, save_json
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import *
import os

def run(tv_gt, tv_pred, outdir, regions, atoms=("CA","C","N","O"),
        reducer="pca", n_components=2, fit_on="gt", pred_share=0.2,
        use_gt_scaler=True, trust_k=10, subsample=None,
        mantel_perms=9999, mantel_method="spearman", seed=0, lag=5,temperature = 300,xbins=50,ybins = 50):
    os.makedirs(outdir, exist_ok=True)
    X_gt = TVWrap(tv_gt).coords(regions, atoms)
    if tv_pred==None:
        X_pr=None
    else:
        X_pr = TVWrap(tv_pred).coords(regions, atoms)


    if reducer in ("pca","pca_weighted","concat","gt","pred"):
        model, Zg, Zp, info = fit_pca_linear(X_gt, X_pr, n_components, fit_on=fit_on,
                                             pred_share=pred_share, use_gt_scaler=use_gt_scaler)
    elif reducer == "tica":
        # choose lag via extra arg; example default:
        model, Zg, Zp, info = fit_tica(
        X_gt, X_pr,
        n_components=n_components,
        lag=lag)
    elif reducer == "diffmap":
        model, Zg, Zp, info = fit_diffmap(X_gt, X_pr, n_components=n_components, epsilon=None, seed=seed, fit_on=fit_on)
    else:
        kernel = reducer.split("_",1)[1]   # kpca_rbf/kpca_cosine/kpca_poly
        model, Zg, Zp, info = fit_kpca(X_gt, X_pr, n_components, kernel=kernel, fit_on=fit_on)
    tw = {"gt": trustworthiness(X_gt, Zg, trust_k, subsample, seed),
          "pred": trustworthiness(X_pr, Zp, trust_k, subsample, seed)if X_pr is not None else None}
    mantel = mantel_rmsd_vs_embedding(tv_gt, tv_pred, Zg, Zp, regions,
                                      atom_names=set(atoms), method=mantel_method,
                                      permutations=mantel_perms, seed=seed,
                                      subsample_gt=subsample, subsample_pred=subsample)

    save_npz(f"{outdir}/coords_embedding.npz", Z_gt=Zg, Z_pred=Zp)
    save_json(f"{outdir}/coords_metrics.json", {"reducer": reducer, "info": {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in info.items()}, "trust": tw, "mantel": mantel})
    return Zg, Zp, info, tw, mantel
