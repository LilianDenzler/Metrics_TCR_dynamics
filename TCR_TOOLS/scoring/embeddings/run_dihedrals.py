from TCR_TOOLS.scoring.embeddings.features import features_dihedrals
from TCR_TOOLS.scoring.embeddings.dim_reduction import fit_pca_linear, fit_kpca,fit_pca_linear, fit_kpca, fit_tica, fit_diffmap

from TCR_TOOLS.scoring.embeddings.metrics import trustworthiness, mantel_rmsd_vs_embedding
from TCR_TOOLS.scoring.embeddings.utils import save_npz, save_json
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import *
import os

def run(tv_gt, tv_pred, outdir, regions,  rmsd_superpose=False,dihedrals=("phi","psi"), encode="sincos",
        reducer="pca", n_components=2, fit_on="gt", pred_share=0.2, lag=5,
        trust_k=10, subsample=None, mantel_perms=9999, mantel_method="spearman", seed=0,temperature = 300,xbins=50,ybins = 50):
    os.makedirs(outdir, exist_ok=True)
    X_gt = features_dihedrals(tv_gt, regions, dihedrals=dihedrals, encode=encode)
    if tv_pred==None:
        X_pr=None
    else:
        X_pr = features_dihedrals(tv_pred, regions, dihedrals=dihedrals, encode=encode)

    if reducer in ("pca","pca_weighted","concat","gt","pred"):
        model, Zg, Zp, info = fit_pca_linear(X_gt, X_pr, n_components, fit_on=fit_on, pred_share=pred_share)
    elif reducer == "tica":
        # choose lag via extra arg; example default:
        model, Zg, Zp, info = fit_tica(
        X_gt, X_pr,
        n_components=n_components,
        lag=lag
    )
    elif reducer == "diffmap":
        model, Zg, Zp, info = fit_diffmap(X_gt, X_pr, n_components=n_components, epsilon=None, seed=seed, fit_on=fit_on)
    else:
        kernel = reducer.split("_",1)[1]
        model, Zg, Zp, info = fit_kpca(X_gt, X_pr, n_components, kernel=kernel, fit_on=fit_on)
    tw = {"gt": trustworthiness(X_gt, Zg, trust_k, subsample, seed),
          "pred": trustworthiness(X_pr, Zp, trust_k, subsample, seed) if X_pr is not None else None}
    mantel = mantel_rmsd_vs_embedding(tv_gt, tv_pred, Zg, Zp, regions,
                                      atom_names={"CA"}, method=mantel_method,
                                      permutations=mantel_perms, seed=seed,
                                      subsample_gt=subsample, subsample_pred=subsample, rmsd_superpose=rmsd_superpose)

    save_npz(f"{outdir}/dihed_embedding.npz", Z_gt=Zg, Z_pred=Zp)
    save_json(f"{outdir}/dihed_metrics.json", {"reducer": reducer, "info": {k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in info.items()}, "trust": tw, "mantel": mantel})
    return Zg, Zp, info, tw, mantel

def make_all_space_model(tv_gt_list, tv_pred_list, regions, dihedrals=("phi","psi"), encode="sincos",
        reducer="pca", n_components=2, fit_on="gt", pred_share=0.2, lag=5, seed=0):
    all_features_gt = []
    all_features_pr = []
    for tv_gt, tv_pred in zip(tv_gt_list, tv_pred_list):
        X_gt = features_dihedrals(tv_gt, regions, dihedrals=dihedrals, encode=encode)
        if tv_pred==None:
            X_pr=None
        else:
            X_pr = features_dihedrals(tv_pred, regions, dihedrals=dihedrals, encode=encode)
        all_features_gt.append(X_gt)
        all_features_pr.append(X_pr)
    # Now fit all together
    X_global_gt = np.vstack(all_features_gt)
    if any(x is not None for x in all_features_pr):
        X_global_pr = np.vstack([x for x in all_features_pr if x is not None])
    else:
        X_global_pr = None

    if reducer in ("pca","pca_weighted","concat","gt","pred"):
        model, Zg, Zp, info = fit_pca_linear(X_global_gt, X_global_pr, n_components, fit_on=fit_on, pred_share=pred_share)
    elif reducer == "tica":
        # choose lag via extra arg; example default:
        model, Zg, Zp, info = fit_tica(
        X_global_gt, X_global_pr,
        n_components=n_components,
        lag=lag
    )
    elif reducer == "diffmap":
        model, Zg, Zp, info = fit_diffmap(X_global_gt, X_global_pr, n_components=n_components, epsilon=None, seed=seed, fit_on=fit_on)
    else:
        kernel = reducer.split("_",1)[1]
        model, Zg, Zp, info = fit_kpca(X_global_gt, X_global_pr, n_components, kernel=kernel, fit_on=fit_on)
    return model,Zg, Zp, info

def run_all_tcr_space(tv_gt_list, tv_pred_list, outdir_list, regions,  rmsd_superpose=False,dihedrals=("phi","psi"), encode="sincos",
        reducer="pca", n_components=2, fit_on="gt", pred_share=0.2, lag=5,
        trust_k=10, subsample=None, mantel_perms=9999, mantel_method="spearman", seed=0,temperature = 300,xbins=50,ybins = 50):

    model,Zg_all, Zp_all, info_all = make_all_space_model(tv_gt_list, tv_pred_list, regions, dihedrals=dihedrals, encode=encode, reducer=reducer, n_components=n_components, fit_on=fit_on, pred_share=pred_share, lag=lag, seed=seed)
    results=[]
    for tv_gt, tv_pred, outdir in zip(tv_gt_list, tv_pred_list, outdir_list):
        X_gt = features_dihedrals(tv_gt, regions, dihedrals=dihedrals, encode=encode)
        if tv_pred==None:
            X_pr=None
        else:
            X_pr = features_dihedrals(tv_pred, regions, dihedrals=dihedrals, encode=encode)

        #Now transform for each TCR seperately
        if X_pr is None:
            Zg= model.transform(X_gt)
            Zp=None
        else:
            Zg, Zp = model.transform(X_gt), model.transform(X_pr)

        tw = {"gt": trustworthiness(X_gt, Zg, trust_k, subsample, seed),
            "pred": trustworthiness(X_pr, Zp, trust_k, subsample, seed) if X_pr is not None else None}
        mantel = mantel_rmsd_vs_embedding(tv_gt, tv_pred, Zg, Zp, regions,
                                        atom_names={"CA"}, method=mantel_method,
                                        permutations=mantel_perms, seed=seed,
                                        subsample_gt=subsample, subsample_pred=subsample, rmsd_superpose=rmsd_superpose)

        save_npz(f"{outdir}/dihed_embedding.npz", Z_gt=Zg, Z_pred=Zp)
        save_json(f"{outdir}/dihed_metrics.json", {"reducer": reducer, "trust": tw, "mantel": mantel})
        results.append( (Zg, Zp, info_all, tw, mantel, outdir) )

    return Zg_all, Zp_all, results