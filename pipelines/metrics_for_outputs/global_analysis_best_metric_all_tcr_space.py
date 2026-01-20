#!/usr/bin/env python3
"""
global_analysis_best_metric_all_tcr_shared_space.py

Goal
----
Compute "best embedding method" per (alignment context, assessment region) BUT using a
single shared embedding space across *all* TCRs (GT-only), rather than fitting a reducer
per TCR.

Key design choices
------------------
1) FEATURE CONSISTENCY ACROSS TCRs:
   For each (assessment region, atom set), we build an insertion-aware atom key:
       (chain_id, resSeq, insertion_code, atom_name)
   and take the strict intersection across all TCRs in the current batch so that every
   TCR is embedded from the exact same ordered atom list -> same feature dimension.

   This is critical because your existing per-TCR embedding code assumes fixed dimensions
   inside a single trajectory; across TCRs, CDR lengths and insertions vary.

2) ALIGNMENT FOR COORDS FEATURES:
   For "coords" embeddings, we first align ALL GT trajectories into one global coordinate frame
   using a robust reference selection:
     - pick a representative frame per TCR (iterative medoid-to-mean within that TCR)
     - pick a global reference among those reps (medoid across TCR reps)
     - superpose each TCR’s full trajectory to that global reference using common atom keys

   For "ca_dist" and "dihedrals", global rigid-body alignment is not required (internal features),
   but we still keep the same alignment pipeline for consistency.

3) RMSD USED IN MANTEL:
   We DO NOT call md.rmsd() (which superposes by default).
   Instead, we compute RMSD matrices directly from already-aligned coordinates
   (no additional per-pair fitting). This is the correct behavior for:
     - align on framework, assess on loop  (loop RMSD should be relative to the framework frame)
     - align on region, assess on same region (RMSD should be in that alignment frame)

Output layout (important for get_best_metric)
--------------------------------------------
outdir/
  ALIGN_<context>/
    <region_label>/
      <TCR_NAME>/
        coords_pca/coords_metrics.json
        coords_pca/coords_embedding.npz
        ...
      <TCR_NAME2>/...
    best_methods_per_region.csv   (written once at end)

Then you can reuse analyse_metrics.get_best_metric(root=ALIGN_context/region_label)
because it groups by method folder name (coords_pca, ca_kpca_rbf, dihed_tica, etc.)
across many metrics.json files (one per TCR).

Run
---
python TCR_Metrics/pipelines/metrics_for_outputs/global_analysis_best_metric_all_tcr_shared_space.py \
  --gt-root /mnt/larry/lilian/DATA/Cory_data \
  --outdir /workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_all_tcr_shared \
  --max-tcr 0 \
  --subsample 200 \
  --mantel-perms 999 \
  --trust-k 10

Notes
-----
- --subsample applies to frames when computing trustworthiness and Mantel, to keep runtime manageable.
- If strict intersection yields too few atoms (<3) for a region, that region is skipped (and logged).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mdtraj as md
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.classes.tcr import TrajectoryView
from TCR_TOOLS.scoring.embeddings.metrics import trustworthiness as tw_metric
from TCR_TOOLS.scoring.embeddings.metrics import mantel as mantel_test
from TCR_TOOLS.scoring.embeddings import dim_reduction
from pipelines.metrics_for_outputs.analyse_metrics import get_best_metric


# -----------------------------
# Utilities: insertion-aware keys
# -----------------------------
def _get_chain_id(chain) -> str:
    cid = getattr(chain, "id", None)
    if cid is None or str(cid).strip() == "":
        return str(getattr(chain, "index", "0"))
    return str(cid)


def _get_insertion_code(res) -> str:
    # mdtraj typically exposes insertion code as residue.insertion_code if present in PDB
    for attr in ("insertion_code", "icode", "insertionCode", "ins_code"):
        if hasattr(res, attr):
            v = getattr(res, attr)
            if v is None:
                return ""
            v = str(v)
            v = v.strip()
            return "" if v == "" or v == " " else v
    return ""


def _atom_key(atom: md.core.topology.Atom) -> Tuple[str, int, str, str]:
    ch = atom.residue.chain
    chain_id = _get_chain_id(ch)
    res = atom.residue
    resseq = getattr(res, "resSeq", None)
    if resseq is None:
        # last resort
        resseq = int(getattr(res, "index", 0))
    icode = _get_insertion_code(res)
    return (chain_id, int(resseq), str(icode), str(atom.name))


def _selection_atom_keys(tv: TrajectoryView, region_names: Sequence[str], atom_names: Sequence[str]) -> Dict[Tuple[str, int, str, str], int]:
    """
    Map atom_key -> atom_index for the selection within tv.
    """
    idx = np.asarray(tv.domain_idx(region_names=region_names, atom_names=set(atom_names)), dtype=int)
    top = tv.mdtraj.topology
    out: Dict[Tuple[str, int, str, str], int] = {}
    dup = []
    for ai in idx:
        k = _atom_key(top.atom(int(ai)))
        if k in out:
            dup.append(k)
        out[k] = int(ai)
    if dup:
        sample = "\n".join([f"  {d}" for d in dup[:10]])
        raise ValueError(
            "Duplicate atom keys found in selection (likely insertion code loss / non-unique residue IDs).\n"
            f"Examples:\n{sample}\n"
            "Fix upstream topology so (chain,resSeq,icode) is unique and insertion codes are preserved."
        )
    return out


def _strict_common_keys(maps: List[Dict[Tuple[str, int, str, str], int]]) -> List[Tuple[str, int, str, str]]:
    """
    Strict intersection across all maps. Returns sorted list of keys.
    """
    if not maps:
        return []
    common = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())
    keys = sorted(common, key=lambda x: (x[0], x[1], x[2], x[3]))
    return keys


def _indices_for_keys(tv: TrajectoryView, key_to_idx: Dict[Tuple[str, int, str, str], int], keys: List[Tuple[str, int, str, str]]) -> np.ndarray:
    return np.asarray([key_to_idx[k] for k in keys], dtype=int)


# -----------------------------
# RMSD without additional superposition
# -----------------------------
def rmsd_matrix_no_superpose(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    """
    Compute pairwise RMSD matrix from coordinates directly (no per-pair fitting).
    traj.xyz is in nm -> convert to Å at the end.
    """
    xyz = traj.xyz[:, atom_indices, :]  # (n,k,3) in nm
    n = xyz.shape[0]
    # Flatten to (n, 3k) and use squared L2 per-frame differences
    X = xyz.reshape(n, -1)
    # Efficient squared Euclidean distances
    G = np.sum(X * X, axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    # RMSD: sqrt(mean over atoms of squared coord diffs). Since D2 is sum over coords,
    # divide by (k) * 3? No: RMSD definition is sqrt(mean over atoms of squared distance per atom).
    # D2 = sum_over_atoms sum_over_xyz (dx^2) = sum_over_atoms ||d||^2
    k = xyz.shape[1]
    M_nm = np.sqrt(D2 / max(k, 1))
    return M_nm * 10.0  # Å


# -----------------------------
# Representative frame selection within a single trajectory (iterative medoid-to-mean)
# -----------------------------
def representative_frame_index(tv: TrajectoryView, region_names: Sequence[str], atom_names: Sequence[str], max_iter: int = 20) -> int:
    sel_map = _selection_atom_keys(tv, region_names, atom_names)
    keys = sorted(sel_map.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
    if len(keys) < 3:
        raise ValueError(f"Too few atoms for representative frame selection: {len(keys)}")
    idx = _indices_for_keys(tv, sel_map, keys)

    T = tv.mdtraj
    ref_i = 0
    for _ in range(max_iter):
        ref = T[ref_i]
        work = T.slice(np.arange(T.n_frames), copy=True)
        work.superpose(ref, atom_indices=idx, ref_atom_indices=idx)  # aligns frames to ref for mean stability

        region = work.atom_slice(idx, inplace=False)
        mean_xyz = region.xyz.mean(axis=0, keepdims=True)
        mean_traj = md.Trajectory(mean_xyz, region.topology)
        rmsd_to_mean = md.rmsd(region, mean_traj)  # (n,) WITH superposition to mean_traj (ok for medoid pick)
        new_i = int(np.argmin(rmsd_to_mean))
        if new_i == ref_i:
            break
        ref_i = new_i
    return ref_i


# -----------------------------
# Global alignment: choose reference among per-TCR reps and align all trajectories into that frame
# -----------------------------
def choose_global_reference(
    tv_list: List[TrajectoryView],
    region_names: Sequence[str],
    atom_names: Sequence[str],
    max_iter: int = 10,
) -> Tuple[int, int]:
    """
    Returns (ref_tv_index, ref_frame_index).
    """
    rep_frame_idx: List[int] = []
    rep_maps: List[Dict[Tuple[str, int, str, str], int]] = []
    for tv in tv_list:
        fi = representative_frame_index(tv, region_names, atom_names)
        rep_frame_idx.append(fi)
        rep_maps.append(_selection_atom_keys(tv, region_names, atom_names))

    # Start from TV with most selected atoms
    n_atoms = [len(m) for m in rep_maps]
    ref_tv_index = int(np.argmax(n_atoms))

    for _ in range(max_iter):
        ref_tv = tv_list[ref_tv_index]
        ref_fi = rep_frame_idx[ref_tv_index]
        ref_frame = ref_tv.mdtraj[ref_fi]

        # Build strict common keys between current ref and each candidate, then compute RMSD to mean
        # in the reference frame (no further fitting beyond the superpose we do to the ref).
        # We pick the medoid candidate with smallest RMSD to the mean coords.
        ref_map = rep_maps[ref_tv_index]
        ref_keys = sorted(ref_map.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))

        aligned_coords = []
        valid_ids = []

        for i, tv in enumerate(tv_list):
            cand_map = rep_maps[i]
            common = sorted(set(ref_keys) & set(cand_map.keys()), key=lambda x: (x[0], x[1], x[2], x[3]))
            if len(common) < 3:
                continue
            idx_ref = np.asarray([ref_map[k] for k in common], dtype=int)
            idx_cand = np.asarray([cand_map[k] for k in common], dtype=int)

            cand_frame = tv.mdtraj[rep_frame_idx[i]].slice([0], copy=True)
            cand_frame.superpose(ref_frame, atom_indices=idx_cand, ref_atom_indices=idx_ref)
            aligned_coords.append(cand_frame.xyz[0, idx_cand, :])  # (k,3) aligned into ref frame
            valid_ids.append(i)

        if len(valid_ids) < 2:
            break

        # mean coords across candidates (after alignment to ref)
        K = min([c.shape[0] for c in aligned_coords])
        # NOTE: we used per-candidate 'common' keys; K may differ across candidates.
        # For medoid scoring, we use only candidates with full ref_keys intersection would be too strict.
        # Practical compromise: keep the current ref if variability is too high.
        # If you want stricter behavior, enforce strict intersection across ALL candidates here.
        # For now: stop refining if K < 3.
        if K < 3:
            break

        # Pick medoid by RMSD to mean for those candidates where K is consistent (rare). Stop.
        # This keeps the algorithm stable without complicated missing-key bookkeeping.
        break

    return ref_tv_index, rep_frame_idx[ref_tv_index]


def align_all_to_global_reference(
    tv_list: List[TrajectoryView],
    region_names: Sequence[str],
    atom_names: Sequence[str],
    outdir: Optional[Path] = None,
) -> Tuple[List[TrajectoryView], Dict[str, Any]]:
    """
    Align each trajectory (all frames) to a global reference frame using common atom keys.
    """
    ref_tv_i, ref_fi = choose_global_reference(tv_list, region_names, atom_names)
    ref_tv = tv_list[ref_tv_i]
    ref_frame = ref_tv.mdtraj[ref_fi]

    # Build reference selection map & keys
    ref_map = _selection_atom_keys(ref_tv, region_names, atom_names)
    ref_keys = sorted(ref_map.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))

    aligned: List[TrajectoryView] = []

    for i, tv in enumerate(tv_list):
        tv_map = _selection_atom_keys(tv, region_names, atom_names)
        common = sorted(set(ref_keys) & set(tv_map.keys()), key=lambda x: (x[0], x[1], x[2], x[3]))
        if len(common) < 3:
            raise ValueError(f"[align] Too few common atoms for TCR index {i} in {region_names}: {len(common)}")

        idx_ref = np.asarray([ref_map[k] for k in common], dtype=int)
        idx_tv = np.asarray([tv_map[k] for k in common], dtype=int)

        T = tv.mdtraj.slice(np.arange(tv.mdtraj.n_frames), copy=True)
        T.superpose(ref_frame, atom_indices=idx_tv, ref_atom_indices=idx_ref)
        aligned.append(TrajectoryView(T, tv._chain_map))

    info = {
        "ref_tv_index": ref_tv_i,
        "ref_frame_index": ref_fi,
        "align_region_names": list(region_names),
        "align_atom_names": list(atom_names),
        "n_tcr": len(tv_list),
    }

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        ref_frame.save_pdb(str(outdir / "global_ref_frame.pdb"))
        (outdir / "global_alignment_info.json").write_text(json.dumps(info, indent=2))

    return aligned, info


# -----------------------------
# Feature extraction on strict-common keys
# -----------------------------
def extract_coords_features(
    tv: TrajectoryView,
    keys: List[Tuple[str, int, str, str]],
    key_to_idx: Dict[Tuple[str, int, str, str], int],
) -> Tuple[np.ndarray, np.ndarray]:
    idx = _indices_for_keys(tv, key_to_idx, keys)
    xyz = tv.mdtraj.xyz[:, idx, :]  # (n,k,3)
    X = xyz.reshape(xyz.shape[0], -1)
    return X, idx


def extract_ca_dist_features(
    tv: TrajectoryView,
    keys: List[Tuple[str, int, str, str]],
    key_to_idx: Dict[Tuple[str, int, str, str], int],
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    keys must be CA-only keys.
    pairs: (p,2) indices into [0..K-1] of CA atoms.
    Returns X: (n_frames, p) distances in nm (consistent with other code; scaling not required).
    """
    idx = _indices_for_keys(tv, key_to_idx, keys)
    xyz = tv.mdtraj.xyz[:, idx, :]  # (n,K,3)
    i = pairs[:, 0]
    j = pairs[:, 1]
    d = xyz[:, i, :] - xyz[:, j, :]
    X = np.linalg.norm(d, axis=2)  # (n,p) nm
    return X, idx


def _dihedral_key_from_indices(top: md.Topology, inds4: np.ndarray, kind: str) -> Tuple[str, Tuple[str, int, str]]:
    """
    For a dihedral defined by 4 atom indices, associate it to the 'central' residue key.
    For phi, central residue is atom 1 (second atom). For psi, central is atom 2 (third atom).
    """
    if kind == "phi":
        a = top.atom(int(inds4[1]))
    else:
        a = top.atom(int(inds4[2]))
    res = a.residue
    chain_id = _get_chain_id(res.chain)
    resseq = int(getattr(res, "resSeq", res.index))
    icode = _get_insertion_code(res)
    return (kind, (chain_id, resseq, icode))


def extract_dihedral_features_shared_keys(
    tv: TrajectoryView,
    residue_keys: List[Tuple[str, int, str]],
    dihedrals: Tuple[str, ...] = ("phi", "psi"),
    encode: str = "sincos",
) -> np.ndarray:
    """
    Returns X: (n_frames, n_features) for selected dihedral keys (strict-common across TCRs).
    """
    top = tv.mdtraj.topology
    wanted = set(residue_keys)

    feats = []

    if "phi" in dihedrals:
        phi_inds, phi_vals = md.compute_phi(tv.mdtraj)  # inds: (m,4), vals: (n,m)
        keep = []
        for j in range(phi_inds.shape[0]):
            _, rk = _dihedral_key_from_indices(top, phi_inds[j], "phi")
            if rk in wanted:
                keep.append(j)
        if keep:
            A = phi_vals[:, keep]
            feats.append(A)

    if "psi" in dihedrals:
        psi_inds, psi_vals = md.compute_psi(tv.mdtraj)
        keep = []
        for j in range(psi_inds.shape[0]):
            _, rk = _dihedral_key_from_indices(top, psi_inds[j], "psi")
            if rk in wanted:
                keep.append(j)
        if keep:
            A = psi_vals[:, keep]
            feats.append(A)

    if not feats:
        return np.zeros((tv.mdtraj.n_frames, 0), dtype=float)

    X = np.concatenate(feats, axis=1)  # radians

    if encode == "sincos":
        X = np.concatenate([np.sin(X), np.cos(X)], axis=1)
    return X


def strict_common_residue_keys_for_dihedrals(
    tv_list: List[TrajectoryView],
    region_names: Sequence[str],
) -> List[Tuple[str, int, str]]:
    """
    Determine strict-common residue keys (chain,resSeq,icode) that actually have phi/psi defined
    within each trajectory for the chosen region.

    This avoids padding NaNs into sklearn reducers.
    """
    per_tv_sets = []

    for tv in tv_list:
        # Start from residues present in region selection (CA is sufficient to identify residues)
        sel_map = _selection_atom_keys(tv, region_names, atom_names=("CA",))
        # residue keys present in region
        residue_present = set((k[0], k[1], k[2]) for k in sel_map.keys())

        # dihedral-available residue keys
        top = tv.mdtraj.topology
        dihed_keys = set()

        phi_inds, _ = md.compute_phi(tv.mdtraj)
        for j in range(phi_inds.shape[0]):
            _, rk = _dihedral_key_from_indices(top, phi_inds[j], "phi")
            if rk in residue_present:
                dihed_keys.add(rk)

        psi_inds, _ = md.compute_psi(tv.mdtraj)
        for j in range(psi_inds.shape[0]):
            _, rk = _dihedral_key_from_indices(top, psi_inds[j], "psi")
            if rk in residue_present:
                dihed_keys.add(rk)

        per_tv_sets.append(dihed_keys)

    if not per_tv_sets:
        return []
    common = set(per_tv_sets[0])
    for s in per_tv_sets[1:]:
        common &= set(s)
    return sorted(common, key=lambda x: (x[0], x[1], x[2]))


# -----------------------------
# Shared reducer fit + per-TCR scoring + saving
# -----------------------------
def fit_reducer_and_project(
    reducer: str,
    X_all: np.ndarray,
    X_list: List[np.ndarray],
    n_components: int,
    fit_on: str,
    lag: int,
    use_gt_scaler: bool,
    seed: int,
) -> Tuple[Any, List[np.ndarray], Dict[str, Any]]:
    """
    Fit reducer on X_all, return Z_list (same splits as X_list), plus info.
    """
    # Optional GT scaling (same as your run_coords.run behavior if desired)
    if use_gt_scaler:
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xfit = (X_all - mu) / sd
        X_list_use = [(Xi - mu) / sd for Xi in X_list]
    else:
        Xfit = X_all
        X_list_use = X_list

    info: Dict[str, Any] = {"reducer": reducer, "n_components": n_components}

    if reducer in ("pca", "pca_weighted"):
        model, Zfit, _, inf = dim_reduction.fit_pca_linear(Xfit, None, n_components=n_components, fit_on="gt", use_gt_scaler=False)
        info.update(inf)
        Z_list = [model.transform(Xi) for Xi in X_list_use]

    elif reducer.startswith("kpca_"):
        kernel = reducer.split("_", 1)[1]
        model, _, _, inf = dim_reduction.fit_kpca(Xfit, None, n_components=n_components, kernel=kernel, fit_on="gt")
        info.update(inf)
        Z_list = [model.transform(Xi) for Xi in X_list_use]

    elif reducer == "diffmap":
        model, _, _, inf = dim_reduction.fit_diffmap(Xfit, None, n_components=n_components, epsilon=None, seed=seed, fit_on="gt")
        info.update(inf)
        Z_list = [model.transform(Xi) for Xi in X_list_use]

    elif reducer == "tica":
        # TICA expects time-ordered frames; X_all is concatenated across TCRs, so timescales are not strictly meaningful.
        # Still: you asked to "try what happens" in a shared space; we keep it, but flag in info.
        model, _, _, inf = dim_reduction.fit_tica(Xfit, None, n_components=n_components, lag=lag)
        info.update(inf)
        info["warning"] = "tica_on_concat_across_tcrs"
        Z_list = [model.transform(Xi)[:, :n_components] for Xi in X_list_use]

    else:
        raise ValueError(f"Unknown reducer: {reducer}")

    return model, Z_list, info


def save_metrics_and_embedding(
    outdir_method: Path,
    assess_type: str,
    reducer: str,
    Z: np.ndarray,
    X: np.ndarray,
    traj_aligned: md.Trajectory,
    rmsd_atom_indices: np.ndarray,
    trust_k: int,
    subsample: Optional[int],
    seed: int,
    mantel_method: str,
    mantel_perms: int,
):
    outdir_method.mkdir(parents=True, exist_ok=True)

    # Trustworthiness (frame-neighborhood preservation)
    tw = tw_metric(X, Z, n_neighbors=trust_k, subsample=subsample, seed=seed)

    # Mantel (structure distance vs embedding distance)
    # Subsample frames for tractability
    rs = np.random.default_rng(seed)
    n_frames = traj_aligned.n_frames
    if subsample is not None and subsample < n_frames:
        I = np.sort(rs.choice(n_frames, size=subsample, replace=False))
        Z_use = Z[I]
        T_use = traj_aligned[I]
    else:
        I = np.arange(n_frames)
        Z_use = Z
        T_use = traj_aligned

    D = rmsd_matrix_no_superpose(T_use, rmsd_atom_indices)
    from scipy.spatial.distance import pdist, squareform
    E = squareform(pdist(Z_use))
    r, p = mantel_test(D, E, method=mantel_method, permutations=mantel_perms, seed=seed)

    # Save embeddings
    np.savez_compressed(
        outdir_method / f"{assess_type}_embedding.npz",
        Z_gt=Z,
        Z_pred=None,
    )

    # Save metrics json (schema compatible with analyse_metrics.get_best_metric)
    metrics = {
        "reducer": reducer,
        "trust": {"gt": float(tw), "pred": None},
        "mantel": {"gt": {"r": float(r), "p": float(p)}, "pred": None, "combined": None},
    }
    (outdir_method / f"{assess_type}_metrics.json").write_text(json.dumps(metrics, indent=2))


# -----------------------------
# Per-region shared screening
# -----------------------------
def assess_screening_shared_space(
    gt_tv_aligned_list: List[TrajectoryView],
    tcr_names: List[str],
    outdir_region: Path,
    assess_region_names: Sequence[str],
    alignment_context: str,
    reducers: Sequence[str],
    trust_k: int,
    subsample: Optional[int],
    seed: int,
    mantel_method: str,
    mantel_perms: int,
    lag: int,
    n_components: int = 2,
    max_pairs: int = 20000,
):
    """
    For one region (assessment selection) in one alignment context:
      - build strict-common atoms for each assess_type
      - fit reducer once on concatenated features across all TCRs
      - score each TCR and write metrics.json under outdir_region/<TCR>/<method>/
    """
    outdir_region.mkdir(parents=True, exist_ok=True)

    # ---------------- coords (CA,C,N)
    coords_atoms = ("CA", "C", "N")
    maps_coords = []
    for tv in gt_tv_aligned_list:
        print(tv)
        print(tv._chain_map)
        maps_coords.append(_selection_atom_keys(tv, assess_region_names, coords_atoms))
        print(maps_coords[-1])
    common_keys_coords = _strict_common_keys(maps_coords)
    if len(common_keys_coords) >= 3:
        X_list = []
        idx_list = []
        for tv, m in zip(gt_tv_aligned_list, maps_coords):
            X, idx = extract_coords_features(tv, common_keys_coords, m)
            X_list.append(X)
            idx_list.append(idx)

        X_all = np.concatenate(X_list, axis=0)

        for red in reducers:
            out_method = outdir_region / f"coords_{red}"
            # Fit shared model and project each TCR
            try:
                _, Z_list, _ = fit_reducer_and_project(
                    reducer=red,
                    X_all=X_all,
                    X_list=X_list,
                    n_components=n_components,
                    fit_on="gt",
                    lag=lag,
                    use_gt_scaler=False,
                    seed=seed,
                )
            except Exception as e:
                os.makedirs(out_method, exist_ok=True)
                (out_method / "SKIP.txt").write_text(f"Failed reducer {red}: {e}\n")
                continue

            # Save per TCR
            for name, tv, Z, X, idx in zip(tcr_names, gt_tv_aligned_list, Z_list, X_list, idx_list):
                out_tcr = outdir_region / name / f"coords_{red}"
                save_metrics_and_embedding(
                    outdir_method=out_tcr,
                    assess_type="coords",
                    reducer=red,
                    Z=Z,
                    X=X,
                    traj_aligned=tv.mdtraj,
                    rmsd_atom_indices=idx,
                    trust_k=trust_k,
                    subsample=subsample,
                    seed=seed,
                    mantel_method=mantel_method,
                    mantel_perms=mantel_perms,
                )
    else:
        (outdir_region / "SKIP_coords.txt").write_text(
            f"coords: too few strict-common atoms ({len(common_keys_coords)}) for region={assess_region_names}\n"
        )

    # ---------------- ca_dist (CA only)
    ca_atoms = ("CA",)
    maps_ca = []
    for tv in gt_tv_aligned_list:
        maps_ca.append(_selection_atom_keys(tv, assess_region_names, ca_atoms))
    common_keys_ca = _strict_common_keys(maps_ca)

    if len(common_keys_ca) >= 3:
        K = len(common_keys_ca)
        # sample pairs consistently
        all_pairs = K * (K - 1) // 2
        rng = np.random.default_rng(seed)
        if all_pairs <= max_pairs:
            pairs = np.array([(i, j) for i in range(K) for j in range(i + 1, K)], dtype=int)
        else:
            # sample without replacement by indexing into upper-triangular pairs
            # build mapping by generating random pairs until enough unique (OK for max_pairs ~2e4)
            s = set()
            while len(s) < max_pairs:
                i = int(rng.integers(0, K))
                j = int(rng.integers(0, K))
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                s.add((i, j))
            pairs = np.array(list(s), dtype=int)

        X_list = []
        idx_list = []
        for tv, m in zip(gt_tv_aligned_list, maps_ca):
            X, idx = extract_ca_dist_features(tv, common_keys_ca, m, pairs)
            X_list.append(X)
            idx_list.append(idx)
        X_all = np.concatenate(X_list, axis=0)

        for red in reducers:
            try:
                _, Z_list, _ = fit_reducer_and_project(
                    reducer=red,
                    X_all=X_all,
                    X_list=X_list,
                    n_components=n_components,
                    fit_on="gt",
                    lag=lag,
                    use_gt_scaler=False,
                    seed=seed,
                )
            except Exception as e:
                (outdir_region / f"SKIP_ca_{red}.txt").write_text(f"Failed reducer {red}: {e}\n")
                continue

            for name, tv, Z, X, idx in zip(tcr_names, gt_tv_aligned_list, Z_list, X_list, idx_list):
                out_tcr = outdir_region / name / f"ca_{red}"
                save_metrics_and_embedding(
                    outdir_method=out_tcr,
                    assess_type="ca",
                    reducer=red,
                    Z=Z,
                    X=X,
                    traj_aligned=tv.mdtraj,
                    rmsd_atom_indices=idx,
                    trust_k=trust_k,
                    subsample=subsample,
                    seed=seed,
                    mantel_method=mantel_method,
                    mantel_perms=mantel_perms,
                )
    else:
        (outdir_region / "SKIP_ca.txt").write_text(
            f"ca: too few strict-common CA atoms ({len(common_keys_ca)}) for region={assess_region_names}\n"
        )

    # ---------------- dihedrals (phi/psi; strict-common residue keys that actually have dihedrals)
    common_res_keys = strict_common_residue_keys_for_dihedrals(gt_tv_aligned_list, assess_region_names)
    if len(common_res_keys) >= 1:
        X_list = []
        # For Mantel RMSD we still use CA strict-common keys (if available) else skip Mantel in dihed.
        # But we store anyway; if CA common <3 you can’t do RMSD matrix.
        use_ca_for_mantel = len(common_keys_ca) >= 3

        for tv in gt_tv_aligned_list:
            X = extract_dihedral_features_shared_keys(tv, common_res_keys, dihedrals=("phi", "psi"), encode="sincos")
            X_list.append(X)

        # Need same feature dimension; strict common_res_keys ensures that.
        if X_list and X_list[0].shape[1] > 0:
            X_all = np.concatenate(X_list, axis=0)

            for red in reducers:
                try:
                    _, Z_list, _ = fit_reducer_and_project(
                        reducer=red,
                        X_all=X_all,
                        X_list=X_list,
                        n_components=n_components,
                        fit_on="gt",
                        lag=lag,
                        use_gt_scaler=False,
                        seed=seed,
                    )
                except Exception as e:
                    (outdir_region / f"SKIP_dihed_{red}.txt").write_text(f"Failed reducer {red}: {e}\n")
                    continue

                for name, tv, Z, X, m_ca in zip(tcr_names, gt_tv_aligned_list, Z_list, X_list, maps_ca):
                    out_tcr = outdir_region / name / f"dihed_{red}"
                    if use_ca_for_mantel:
                        idx = _indices_for_keys(tv, m_ca, common_keys_ca)
                        save_metrics_and_embedding(
                            outdir_method=out_tcr,
                            assess_type="dihed",
                            reducer=red,
                            Z=Z,
                            X=X,
                            traj_aligned=tv.mdtraj,
                            rmsd_atom_indices=idx,
                            trust_k=trust_k,
                            subsample=subsample,
                            seed=seed,
                            mantel_method=mantel_method,
                            mantel_perms=mantel_perms,
                        )
                    else:
                        # Save without Mantel (write r/p as NaN)
                        out_tcr.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(out_tcr / "dihed_embedding.npz", Z_gt=Z, Z_pred=None)
                        metrics = {
                            "reducer": red,
                            "trust": {"gt": float(tw_metric(X, Z, n_neighbors=trust_k, subsample=subsample, seed=seed)), "pred": None},
                            "mantel": {"gt": {"r": float("nan"), "p": float("nan")}, "pred": None, "combined": None},
                            "warning": "dihed_mantel_skipped_no_common_ca_atoms",
                        }
                        (out_tcr / "dihed_metrics.json").write_text(json.dumps(metrics, indent=2))
        else:
            (outdir_region / "SKIP_dihed_empty.txt").write_text("dihed: extracted 0 dihedral features.\n")
    else:
        (outdir_region / "SKIP_dihed.txt").write_text(
            f"dihed: too few strict-common dihedral-bearing residues ({len(common_res_keys)}) for region={assess_region_names}\n"
        )


# -----------------------------
# Correct best-metric aggregation across regions & alignment contexts
# -----------------------------
def calc_best_metric_shared(outdir: Path, rank_by: str = "mantel.gt.r") -> Path:
    """
    Traverse:
      outdir/ALIGN_*/<region_label>/
    and call get_best_metric on each <region_label> folder, which contains many <TCR>/.../metrics.json files.

    Writes a CSV:
      outdir/best_methods_per_region_shared.csv
    """
    rows = []
    for align_dir in sorted([p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("ALIGN_")]):
        for region_dir in sorted([p for p in align_dir.iterdir() if p.is_dir()]):
            try:
                best_method, score = get_best_metric(region_dir, rank_by=rank_by)
                rows.append(
                    {
                        "alignment_context": align_dir.name.replace("ALIGN_", ""),
                        "region": region_dir.name,
                        "best_method": best_method,
                        "score": float(score) if score is not None else np.nan,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "alignment_context": align_dir.name.replace("ALIGN_", ""),
                        "region": region_dir.name,
                        "best_method": None,
                        "score": np.nan,
                        "error": str(e),
                    }
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = outdir / "best_methods_per_region_shared.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def subsample_tv_frames(tv: TrajectoryView, n_frames: int, seed: int = 0, method: str = "uniform"):
    """
    Return a NEW TrajectoryView containing a subset of frames from tv.
    Keeps topology identical and returns mapping to original frame indices.

    method:
      - "uniform": evenly spaced frames
      - "random": random frames without replacement
    """
    traj = tv.mdtraj
    N = traj.n_frames
    if n_frames <= 0 or n_frames >= N:
        idx = np.arange(N, dtype=int)
        return tv.__class__(traj, tv._chain_map), idx

    if method == "uniform":
        idx = np.linspace(0, N - 1, n_frames, dtype=int)
    elif method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_frames, replace=False)
        idx.sort()
    else:
        raise ValueError(f"Unknown method: {method}")

    sub = traj.slice(idx, copy=True)
    return tv.__class__(sub, tv._chain_map), idx


# -----------------------------
# Main orchestration
# -----------------------------
def discover_gt_pairs(gt_root: Path, max_tcr: int = 0) -> Tuple[List[str], List[Any], List[TrajectoryView]]:
    """
    Returns (tcr_names, pairviews, tv_list).
    """
    tcr_names = []
    pairviews = []
    tv_list = []

    folders = sorted([p for p in gt_root.iterdir() if p.is_dir()])
    if max_tcr and max_tcr > 0:
        folders = folders[:max_tcr]

    for folder in folders:
        try:
            name = folder.name
            pdb = folder / f"{name}.pdb"
            xtc = folder / f"{name}_Prod.xtc"
            if not (pdb.exists() and xtc.exists()):
                # fallback
                xtc2 = folder / f"{name}.xtc"
                if pdb.exists() and xtc2.exists():
                    xtc = xtc2
                else:
                    continue

            tcr = TCR(
                input_pdb=str(pdb),
                traj_path=str(xtc),
                contact_cutoff=5.0,
                min_contacts=50,
                legacy_anarci=True,
            )
            pair = tcr.pairs[0]
            tcr_names.append(name)
            pairviews.append(pair)
            tv_list.append(pair.traj)
        except Exception as e:
            print(f"[discover_gt_pairs] Skipping TCR in {folder}: {e}")
            continue

    return tcr_names, pairviews, tv_list


def main(gt_root,
    outdir,
    subsample=200, #frames subsample for TW/Mantel; 0 disables
    reducers="pca,kpca_rbf,kpca_cosine,kpca_poly,diffmap,tica",
    max_tcr=0,  #0 means no limit
    trust_k=10,
    seed=0,
    mantel_method="spearman", #or pearson
    mantel_perms=999,
    lag=5,
    max_pairs=20000,
    rank_by="mantel.gt.r"
    ):




    gt_root = Path(gt_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    subsample = None if subsample in (0, None) else int(subsample)
    reducers = [s.strip() for s in reducers.split(",") if s.strip()]

    tcr_names, _, tv_list = discover_gt_pairs(gt_root, max_tcr=max_tcr)
    if not tv_list:
        raise SystemExit("No GT TCR trajectories found.")

    print(f"Loaded {len(tv_list)} GT TCRs.")

    # Alignment contexts:
    # 1) align on each CDR itself and assess that CDR
    cdrs = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]

    # 2) align on framework per chain (FR1/FR2/FR3) and assess loops + variable + combined
    fw_contexts = [
        ("A_variable", ["A_FR1", "A_FR2", "A_FR3"]),
        ("B_variable", ["B_FR1", "B_FR2", "B_FR3"]),
    ]
    combined_regions = [
        ["A_CDR1", "A_CDR2", "A_CDR3"],
        ["B_CDR1", "B_CDR2", "B_CDR3"],
        ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"],
    ]

    # -------------------------
    # 1) CDR self-alignment contexts
    # -------------------------
    for cdr in cdrs:
        if cdr=="A_CDR1":
            continue
        print(f"\n=== ALIGN context: {cdr} (self) ===")
        align_dir = outdir / f"ALIGN_{cdr}"
        gt_aligned, info = align_all_to_global_reference(
            tv_list, region_names=[cdr], atom_names=("CA", "C", "N"), outdir=align_dir
        )
        # ---- SUBSAMPLE FOR FITTING REDUCERS (prevents X_all from exploding) ----
        fit_frames_per_tcr = 1000
        if fit_frames_per_tcr is not None:
            gt_aligned_fit = []
            fit_index_maps = {}  # optional: store mapping if you want later
            for name, tv in zip(tcr_names, gt_aligned):
                tv_fit, idx = subsample_tv_frames(tv, n_frames=fit_frames_per_tcr, seed=seed, method="uniform")
                gt_aligned_fit.append(tv_fit)
                fit_index_maps[name] = idx.tolist()
        else:
            gt_aligned_fit = gt_aligned
            fit_index_maps = None

        # assess same CDR region
        region_dir = align_dir / cdr
        assess_screening_shared_space(
            gt_tv_aligned_list=gt_aligned_fit,
            tcr_names=tcr_names,
            outdir_region=region_dir,
            assess_region_names=[cdr],
            alignment_context=cdr,
            reducers=reducers,
            trust_k=trust_k,
            subsample=subsample,
            seed=seed,
            mantel_method=mantel_method,
            mantel_perms=mantel_perms,
            lag=lag,
            n_components=2,
            max_pairs=max_pairs
        )

    # -------------------------
    # 2) Framework alignment contexts
    # -------------------------
    for ctx_name, fr_regions in fw_contexts:
        print(f"\n=== ALIGN context: {ctx_name} (framework {fr_regions}) ===")
        align_dir = outdir / f"ALIGN_{ctx_name}"
        gt_aligned, info = align_all_to_global_reference(
            tv_list, region_names=fr_regions, atom_names=("CA", "C", "N"), outdir=align_dir
        )
        # ---- SUBSAMPLE FOR FITTING REDUCERS (prevents X_all from exploding) ----
        fit_frames_per_tcr = 1000
        if fit_frames_per_tcr is not None:
            gt_aligned_fit = []
            fit_index_maps = {}  # optional: store mapping if you want later
            for name, tv in zip(tcr_names, gt_aligned):
                tv_fit, idx = subsample_tv_frames(tv, n_frames=fit_frames_per_tcr, seed=seed, method="uniform")
                gt_aligned_fit.append(tv_fit)
                fit_index_maps[name] = idx.tolist()
        else:
            gt_aligned_fit = gt_aligned
            fit_index_maps = None

        # Assess each CDR + the variable domain itself
        for region in cdrs + [ctx_name]:
            region_dir = align_dir / region
            assess_screening_shared_space(
                gt_tv_aligned_list=gt_aligned_fit,
                tcr_names=tcr_names,
                outdir_region=region_dir,
                assess_region_names=[region],
                alignment_context=ctx_name,
                reducers=reducers,
                trust_k=trust_k,
                subsample=subsample,
                seed=seed,
                mantel_method=mantel_method,
                mantel_perms=mantel_perms,
                lag=lag,
                n_components=2,
                max_pairs=max_pairs,
            )

        # Combined regions
        for region_list in combined_regions:
            label = "".join(region_list)
            region_dir = align_dir / label
            assess_screening_shared_space(
                gt_tv_aligned_list=gt_aligned,
                tcr_names=tcr_names,
                outdir_region=region_dir,
                assess_region_names=region_list,
                alignment_context=ctx_name,
                reducers=reducers,
                trust_k=trust_k,
                subsample=subsample,
                seed=seed,
                mantel_method=mantel_method,
                mantel_perms=mantel_perms,
                lag=lag,
                n_components=2,
                max_pairs=max_pairs,
            )

    # -------------------------
    # Best-method aggregation across all TCRs (shared-space)
    # -------------------------
    out_csv = calc_best_metric_shared(outdir, rank_by=rank_by)
    print(f"\nWrote best-method summary to: {out_csv}")


if __name__ == "__main__":
    gt_root="/mnt/larry/lilian/DATA/Cory_data"
    outdir="/workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_all_tcr_shared"
    subsample= 200
    max_tcr=0
    trust_k= 10
    mantel_perms= 999
    rank_by= "mantel.gt.r"
    main(gt_root=gt_root, outdir=outdir, subsample= 200, max_tcr=0, trust_k= 10,mantel_perms= 999, rank_by= "mantel.gt.r")
