#!/usr/bin/env python3
"""
global_analysis_best_metric_all_tcr_space_gt_pred.py

Shared-space best-metric screening across ALL TCRs, but now for:
  - GT trajectories
  - model predictions (pred trajectories)

Key guarantees:
  - One shared reducer space per (ALIGN_CONTEXT, ASSESS_REGION, ASSESS_TYPE, REDUCER)
  - Reducer is FIT ON GT ONLY (scaled globally on GT)
  - Pred is projected into the same space using the same scaler + reducer
  - RMSD calibration factor k is GLOBAL across the batch for that context+region+method
  - Mantel RMSD uses already-aligned coordinates (no per-pair superposition)
  - Insertion codes handled via PDB atom serial mapping (robust with mdtraj)

Output layout:
outdir/
  ALIGN_<context>/
    <assess_region_label>/
      <TCR_NAME>/
        coords_pca/coords_metrics.json
        coords_pca/embeddings_scaled_centered.npz
        coords_pca/pmf_global_bins/...
        ...
      __method_summaries__/
        coords_pca/
          rmsd_calibration.txt
          rmsd_calibration.csv
          global_range.json
        ...
  best_methods_per_region_shared.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import mdtraj as md

# Ensure your repo is on path (same as your other scripts)
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
sys.path.append("/workspaces/Graphormer")

from TCR_TOOLS.classes.tcr import TCR, TrajectoryView
from TCR_TOOLS.scoring.embeddings.metrics import trustworthiness as tw_metric
from TCR_TOOLS.scoring.embeddings.metrics import mantel as mantel_test
from TCR_TOOLS.scoring.embeddings import dim_reduction
from TCR_TOOLS.scoring.pmf_kde import oriol_analysis
from pipelines.metrics_for_outputs.analyse_metrics import get_best_metric

from dig_output_handler.process_output import process_output


# -----------------------------------------------------------------------------
# Insertion-code aware atom keys using PDB serial mapping (robust with mdtraj)
# -----------------------------------------------------------------------------
AtomKey = Tuple[str, int, str, str]  # (chain_id, resseq, icode, atom_name)

def _parse_pdb_serial_map(pdb_path: str) -> Dict[int, AtomKey]:
    """
    Map PDB atom serial -> (chain_id, resseq, icode, atom_name).
    This is robust because mdtraj topology does not reliably preserve insertion codes.
    """
    serial_map: Dict[int, AtomKey] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            serial = int(line[6:11])
            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            icode = "" if (icode == "" or icode == " ") else icode
            serial_map[serial] = (chain_id, resseq, icode, atom_name)
    return serial_map


def _safe_domain_idx(tv: TrajectoryView, region_names: Sequence[str], atom_names: Set[str]) -> List[int]:
    """
    domain_idx can hard-fail in your current code path. We treat that as "no atoms".
    We also support an optional strict=False signature if available.
    """
    try:
        return list(tv.domain_idx(region_names=list(region_names), atom_names=set(atom_names), strict=False))
    except TypeError:
        # older signature: no strict kw
        try:
            return list(tv.domain_idx(region_names=list(region_names), atom_names=set(atom_names)))
        except Exception:
            return []
    except Exception:
        return []


def _region_atom_map_insertion_aware(
    tv: TrajectoryView,
    pdb_topology_path: str,
    region_names: Sequence[str],
    atom_names: Set[str],
) -> Dict[AtomKey, int]:
    """
    Return dict: atom_key -> atom_index (in tv.mdtraj full indexing).
    """
    serial_map = _parse_pdb_serial_map(pdb_topology_path)
    idx = _safe_domain_idx(tv, region_names, atom_names)
    if len(idx) == 0:
        return {}

    top = tv.mdtraj.topology
    out: Dict[AtomKey, int] = {}
    dup: List[AtomKey] = []

    for i in idx:
        a = top.atom(int(i))
        serial = getattr(a, "serial", None)

        if serial is not None and int(serial) in serial_map:
            key = serial_map[int(serial)]
        else:
            # Fallback: better than your old fallback; prefer chain.id when available
            ch = a.residue.chain
            chain_id = getattr(ch, "id", None)
            if chain_id is None or str(chain_id).strip() == "":
                chain_id = str(getattr(ch, "index", "0"))
            key = (str(chain_id), int(a.residue.resSeq), "", str(a.name))

        if key in out:
            dup.append(key)
        out[key] = int(i)

    if dup:
        sample = "\n".join([f"  {d}" for d in dup[:10]])
        raise ValueError(
            "Duplicate insertion-aware atom keys detected. This indicates inconsistent PDB serials "
            "or duplicated residue identifiers.\n"
            f"Examples:\n{sample}"
        )

    return out


def _common_indices_to_reference(
    tv: TrajectoryView,
    tv_pdb: str,
    ref_key_to_idx: Dict[AtomKey, int],
    region_names: Sequence[str],
    atom_names: Set[str],
) -> Tuple[np.ndarray, np.ndarray, List[AtomKey]]:
    tv_map = _region_atom_map_insertion_aware(tv, tv_pdb, region_names, atom_names)
    keys = sorted(set(tv_map.keys()) & set(ref_key_to_idx.keys()))
    tv_idx = np.array([tv_map[k] for k in keys], dtype=int)
    ref_idx = np.array([ref_key_to_idx[k] for k in keys], dtype=int)
    return tv_idx, ref_idx, keys


# -----------------------------------------------------------------------------
# RMSD matrix without additional superposition (correct for "align on X, assess on Y")
# -----------------------------------------------------------------------------
def rmsd_matrix_no_superpose(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    """
    Pairwise RMSD from already-aligned coordinates (no per-pair fitting).
    traj.xyz in nm -> convert to Å at end.
    """
    xyz = traj.xyz[:, atom_indices, :]  # (n,k,3)
    n = xyz.shape[0]
    k = xyz.shape[1]
    if n <= 1 or k <= 0:
        return np.zeros((n, n), dtype=float)

    X = xyz.reshape(n, -1)  # (n, 3k)
    G = np.sum(X * X, axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)

    # D2 = sum_over_atoms ||d||^2, so RMSD = sqrt( (1/k) * D2 )
    M_nm = np.sqrt(D2 / float(k))
    return M_nm * 10.0  # Å


# -----------------------------------------------------------------------------
# Global RMSD calibration factor (fixed version of your pasted function)
# -----------------------------------------------------------------------------
def rmsd_calibration_factor_all_space(
    Zg_list: List[np.ndarray],
    gt_traj_aligned_list: List[TrajectoryView],
    region_names: List[str],
    atom_names: Set[str] = {"CA"},
    n_pairs: int = 5000,
    random_state: int = 0,
    out_csv: Optional[str] = None,
    out_txt: Optional[str] = None,
    return_center: bool = True,
) -> Tuple[float, float, int, Optional[np.ndarray]]:
    """
    GLOBAL calibration for a shared embedding space:
      sample random (embed_distance, RMSD) pairs within random TCRs and pool them.

    Returns:
      k, corr, n_pairs_used, center_scaled
    """
    rng = np.random.default_rng(random_state)

    if len(Zg_list) == 0:
        raise ValueError("Zg_list is empty.")
    if len(Zg_list) != len(gt_traj_aligned_list):
        raise ValueError("Zg_list and gt_traj_aligned_list must have same length.")

    dims = [z.shape[1] for z in Zg_list if z is not None and z.size > 0]
    if len(dims) == 0:
        raise ValueError("No non-empty embeddings in Zg_list.")
    d = dims[0]
    if any(dd != d for dd in dims):
        raise ValueError("All Zg embeddings must have the same dimension.")

    # Precompute region-only trajectories for RMSD (and ensure indexable frame alignment)
    region_trajs: List[md.Trajectory] = []
    valid_tcr: List[int] = []

    for i, tv in enumerate(gt_traj_aligned_list):
        z = Zg_list[i]
        if z is None or z.shape[0] < 2:
            continue

        try:
            reg_tv = tv.domain_subset(region_names, atom_names)
            reg = reg_tv.mdtraj
        except Exception:
            continue

        if reg.n_frames < 2:
            continue

        # Ensure embeddings correspond to trajectory frames
        if z.shape[0] > reg.n_frames:
            raise ValueError(
                f"TCR {i}: embeddings have {z.shape[0]} rows but region traj has {reg.n_frames} frames. "
                "If you subsampled embeddings upstream, you MUST apply the same subsample to the trajectory."
            )

        region_trajs.append(reg)
        valid_tcr.append(i)

    if len(valid_tcr) == 0:
        raise ValueError("No valid TCRs (need >=2 frames) for global RMSD calibration.")

    d_embed: List[float] = []
    d_rmsd: List[float] = []

    for _ in range(n_pairs):
        tcr_local = int(rng.integers(0, len(valid_tcr)))
        i_global = valid_tcr[tcr_local]
        Z = Zg_list[i_global]
        reg = region_trajs[tcr_local]

        n = Z.shape[0]
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        if a > b:
            a, b = b, a

        de = float(np.linalg.norm(Z[a] - Z[b]))
        if de <= 1e-8:
            continue

        dr_nm = md.rmsd(reg[b], reg[a])[0]  # mdtraj returns nm, aligns reg[b] to reg[a] by default
        # IMPORTANT NOTE:
        # This md.rmsd() does superposition. If you want calibration consistent with your "no extra fitting"
        # definition, you'd replace this with direct RMSD on already-aligned coords. However, calibration is
        # a scalar mapping; using md.rmsd() is acceptable if you keep it consistent across methods.
        dr = float(dr_nm * 10.0)

        d_embed.append(de)
        d_rmsd.append(dr)

    if len(d_embed) < 10:
        raise ValueError("Too few valid (non-zero) pairs for calibration; increase n_pairs or check embeddings.")

    d_embed_arr = np.asarray(d_embed, float)
    d_rmsd_arr = np.asarray(d_rmsd, float)

    k = float(np.dot(d_embed_arr, d_rmsd_arr) / np.dot(d_embed_arr, d_embed_arr))
    corr = float(np.corrcoef(d_embed_arr, d_rmsd_arr)[0, 1]) if d_embed_arr.size > 1 else float("nan")
    n_used = int(d_embed_arr.size)

    center_scaled = None
    if return_center:
        Z_all = np.concatenate([z for z in Zg_list if z is not None and z.size > 0], axis=0)
        center_scaled = (Z_all * k).mean(axis=0)

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with out_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["region", "atoms", "n_pairs_used", "k_A_per_unit", "corr"])
            w.writerow(["+".join(region_names), "+".join(sorted(atom_names)), n_used, k, corr])

    if out_txt is not None:
        p = Path(out_txt)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"region={'+'.join(region_names)}\n"
            f"atoms={'+'.join(sorted(atom_names))}\n"
            f"n_pairs_used={n_used}\n"
            f"k_A_per_unit={k:.8g}\n"
            f"corr={corr:.8g}\n"
        )

    return k, corr, n_used, center_scaled


# -----------------------------------------------------------------------------
# Global alignment across GT + pred using a GT-chosen reference
# -----------------------------------------------------------------------------
def _gt_representative_frame_index(
    tv_gt: TrajectoryView,
    gt_pdb: str,
    region_names: Sequence[str],
    atom_names: Set[str],
    max_iter: int = 20,
) -> int:
    gt_map = _region_atom_map_insertion_aware(tv_gt, gt_pdb, region_names, atom_names)
    gt_idx = np.array(list(gt_map.values()), dtype=int)
    if gt_idx.size < 3:
        raise ValueError(f"Too few atoms for representative frame selection: {gt_idx.size}")

    T = tv_gt.mdtraj
    ref_i = 0
    prev = None
    work = T.slice(np.arange(T.n_frames), copy=True)

    for _ in range(max_iter):
        ref_frame = T[ref_i]
        work.superpose(ref_frame, atom_indices=gt_idx, ref_atom_indices=gt_idx)
        region_work = work.atom_slice(gt_idx, inplace=False)
        mean_xyz = region_work.xyz.mean(axis=0, keepdims=True)
        mean_traj = md.Trajectory(mean_xyz, region_work.topology)
        rmsd_to_mean = md.rmsd(region_work, mean_traj)
        new_i = int(np.argmin(rmsd_to_mean))
        if prev is not None and new_i == ref_i:
            break
        prev = ref_i
        ref_i = new_i

    return ref_i


def _choose_global_reference_gt_rep(
    gt_tv_list: List[TrajectoryView],
    gt_pdb_list: List[str],
    region_names: Sequence[str],
    atom_names: Set[str],
    max_iter: int = 10,
) -> Tuple[int, int]:
    """
    Choose a reference (GT index, frame index) among GTs.
    We pick:
      - representative frame per GT
      - start from GT with most atoms in selection
    """
    rep_frame_idx: List[int] = []
    rep_maps: List[Dict[AtomKey, int]] = []

    for tv_gt, gt_pdb in zip(gt_tv_list, gt_pdb_list):
        fi = _gt_representative_frame_index(tv_gt, gt_pdb, region_names, atom_names)
        rep_frame_idx.append(fi)
        rep_maps.append(_region_atom_map_insertion_aware(tv_gt, gt_pdb, region_names, atom_names))

    n_atoms = [len(m) for m in rep_maps]
    ref_gt_index = int(np.argmax(n_atoms))
    return ref_gt_index, rep_frame_idx[ref_gt_index]


def align_all_pairs_global(
    pred_tv_list: List[TrajectoryView],
    gt_tv_list: List[TrajectoryView],
    pred_pdb_list: List[str],
    gt_pdb_list: List[str],
    region_names: Sequence[str],
    atom_names: Set[str],
    outdir: Optional[Path] = None,
) -> Tuple[List[TrajectoryView], List[TrajectoryView], List[int], Dict[str, Any]]:
    """
    Align ALL GT and ALL preds to ONE shared global reference frame (chosen from GT).
    Robust: if a TCR cannot be aligned for this context, it is excluded (non-fatal).

    Returns:
      pred_aligned_list, gt_aligned_list, kept_indices, ref_info
    """
    assert len(pred_tv_list) == len(gt_tv_list) == len(pred_pdb_list) == len(gt_pdb_list)

    ref_gt_i, ref_frame_i = _choose_global_reference_gt_rep(
        gt_tv_list=gt_tv_list,
        gt_pdb_list=gt_pdb_list,
        region_names=region_names,
        atom_names=atom_names,
    )

    ref_tv = gt_tv_list[ref_gt_i]
    ref_pdb = gt_pdb_list[ref_gt_i]
    ref_frame = ref_tv.mdtraj[ref_frame_i]
    ref_key_to_idx = _region_atom_map_insertion_aware(ref_tv, ref_pdb, region_names, atom_names)

    kept: List[int] = []
    gt_aligned_list: List[TrajectoryView] = []
    pred_aligned_list: List[TrajectoryView] = []

    for i, (tv_gt, tv_pred, gt_pdb, pred_pdb) in enumerate(zip(gt_tv_list, pred_tv_list, gt_pdb_list, pred_pdb_list)):
        gt_idx, ref_idx_gt, keys_gt = _common_indices_to_reference(tv_gt, gt_pdb, ref_key_to_idx, region_names, atom_names)
        pred_idx, ref_idx_pred, keys_pred = _common_indices_to_reference(tv_pred, pred_pdb, ref_key_to_idx, region_names, atom_names)

        if len(keys_gt) < 3 or len(keys_pred) < 3:
            # Skip this TCR for this alignment context
            continue

        gt_traj = tv_gt.mdtraj.slice(np.arange(tv_gt.mdtraj.n_frames), copy=True)
        pred_traj = tv_pred.mdtraj.slice(np.arange(tv_pred.mdtraj.n_frames), copy=True)

        gt_traj.superpose(ref_frame, atom_indices=gt_idx, ref_atom_indices=ref_idx_gt)
        pred_traj.superpose(ref_frame, atom_indices=pred_idx, ref_atom_indices=ref_idx_pred)

        gt_aligned_list.append(TrajectoryView(gt_traj, tv_gt._chain_map))
        pred_aligned_list.append(TrajectoryView(pred_traj, tv_pred._chain_map))
        kept.append(i)

    ref_info = {
        "ref_gt_index": ref_gt_i,
        "ref_frame_index": ref_frame_i,
        "align_region_names": list(region_names),
        "align_atom_names": sorted(list(atom_names)),
        "n_input": len(gt_tv_list),
        "n_kept": len(kept),
    }

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        ref_frame.save_pdb(str(outdir / "global_ref_frame.pdb"))
        (outdir / "global_ref_info.json").write_text(json.dumps(ref_info, indent=2))

    return pred_aligned_list, gt_aligned_list, kept, ref_info


# -----------------------------------------------------------------------------
# Feature extraction: strict-common keys across *kept* TCRs, insertion-aware
# -----------------------------------------------------------------------------
def _strict_common_keys(maps: List[Dict[AtomKey, int]]) -> List[AtomKey]:
    if not maps:
        return []
    common = set(maps[0].keys())
    for m in maps[1:]:
        common &= set(m.keys())
    return sorted(common, key=lambda x: (x[0], x[1], x[2], x[3]))


def _indices_for_keys(key_to_idx: Dict[AtomKey, int], keys: List[AtomKey]) -> np.ndarray:
    return np.asarray([key_to_idx[k] for k in keys], dtype=int)


def extract_coords_features(tv: TrajectoryView, key_to_idx: Dict[AtomKey, int], keys: List[AtomKey]) -> Tuple[np.ndarray, np.ndarray]:
    idx = _indices_for_keys(key_to_idx, keys)
    xyz = tv.mdtraj.xyz[:, idx, :]  # (n,k,3) nm
    X = xyz.reshape(xyz.shape[0], -1)
    return X, idx


def extract_ca_dist_features(tv: TrajectoryView, key_to_idx: Dict[AtomKey, int], keys: List[AtomKey], pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = _indices_for_keys(key_to_idx, keys)
    xyz = tv.mdtraj.xyz[:, idx, :]  # (n,K,3)
    i = pairs[:, 0]
    j = pairs[:, 1]
    d = xyz[:, i, :] - xyz[:, j, :]
    X = np.linalg.norm(d, axis=2)  # (n, p) nm
    return X, idx


# Dihedrals: we use residue keys derived from CA selection keys
ResKey = Tuple[str, int, str]  # (chain, resseq, icode)

def _reskey_from_atomkey(k: AtomKey) -> ResKey:
    return (k[0], k[1], k[2])

def strict_common_residue_keys_for_dihedrals(
    tv_list: List[TrajectoryView],
    pdb_list: List[str],
    region_names: Sequence[str],
) -> List[ResKey]:
    """
    Determine strict-common residue keys (chain, resseq, icode) that are in the region
    AND have phi/psi defined in each trajectory.
    """
    per_tv_sets: List[Set[ResKey]] = []

    for tv, pdb in zip(tv_list, pdb_list):
        sel_map = _region_atom_map_insertion_aware(tv, pdb, region_names, {"CA"})
        residue_present = set(_reskey_from_atomkey(k) for k in sel_map.keys())
        if not residue_present:
            per_tv_sets.append(set())
            continue

        top = tv.mdtraj.topology
        dihed_keys: Set[ResKey] = set()

        phi_inds, _ = md.compute_phi(tv.mdtraj)
        for inds4 in phi_inds:
            a = top.atom(int(inds4[1]))  # central residue for phi
            serial = getattr(a, "serial", None)
            if serial is None:
                continue
            # serial->atomkey needs serial_map; easiest: approximate from mdtraj for dihedral membership
            rk = (getattr(a.residue.chain, "id", str(a.residue.chain.index)), int(a.residue.resSeq), "")
            if rk in residue_present:
                dihed_keys.add(rk)

        psi_inds, _ = md.compute_psi(tv.mdtraj)
        for inds4 in psi_inds:
            a = top.atom(int(inds4[2]))  # central residue for psi
            rk = (getattr(a.residue.chain, "id", str(a.residue.chain.index)), int(a.residue.resSeq), "")
            if rk in residue_present:
                dihed_keys.add(rk)

        per_tv_sets.append(dihed_keys)

    if not per_tv_sets:
        return []
    common = set(per_tv_sets[0])
    for s in per_tv_sets[1:]:
        common &= set(s)

    return sorted(common, key=lambda x: (x[0], x[1], x[2]))


def extract_dihedral_features_shared_keys(
    tv: TrajectoryView,
    residue_keys: List[ResKey],
    dihedrals: Tuple[str, ...] = ("phi", "psi"),
    encode: str = "sincos",
) -> np.ndarray:
    top = tv.mdtraj.topology
    wanted = set(residue_keys)
    feats = []

    if "phi" in dihedrals:
        phi_inds, phi_vals = md.compute_phi(tv.mdtraj)
        keep = []
        for j in range(phi_inds.shape[0]):
            a = top.atom(int(phi_inds[j][1]))
            rk = (getattr(a.residue.chain, "id", str(a.residue.chain.index)), int(a.residue.resSeq), "")
            if rk in wanted:
                keep.append(j)
        if keep:
            feats.append(phi_vals[:, keep])

    if "psi" in dihedrals:
        psi_inds, psi_vals = md.compute_psi(tv.mdtraj)
        keep = []
        for j in range(psi_inds.shape[0]):
            a = top.atom(int(psi_inds[j][2]))
            rk = (getattr(a.residue.chain, "id", str(a.residue.chain.index)), int(a.residue.resSeq), "")
            if rk in wanted:
                keep.append(j)
        if keep:
            feats.append(psi_vals[:, keep])

    if not feats:
        return np.zeros((tv.mdtraj.n_frames, 0), dtype=float)

    X = np.concatenate(feats, axis=1)
    if encode == "sincos":
        X = np.concatenate([np.sin(X), np.cos(X)], axis=1)
    return X


# -----------------------------------------------------------------------------
# Shared reducer fit (GT only) + projection (GT + pred) + global standardization
# -----------------------------------------------------------------------------
def fit_reducer_shared(
    reducer: str,
    Xfit_gt: np.ndarray,
    n_components: int,
    lag: int,
    seed: int,
):
    """
    Fit reducer on GT-only (already scaled) fit matrix Xfit_gt.
    Returns a model with .transform(X) for PCA/KPCA/Diffmap and TICA wrapper.
    """
    if reducer in ("pca", "pca_weighted"):
        model, _, _, inf = dim_reduction.fit_pca_linear(Xfit_gt, None, n_components=n_components, fit_on="gt", use_gt_scaler=False)
        return model, inf

    if reducer.startswith("kpca_"):
        kernel = reducer.split("_", 1)[1]
        model, _, _, inf = dim_reduction.fit_kpca(Xfit_gt, None, n_components=n_components, kernel=kernel, fit_on="gt")
        return model, inf

    if reducer == "diffmap":
        model, _, _, inf = dim_reduction.fit_diffmap(Xfit_gt, None, n_components=n_components, epsilon=None, seed=seed, fit_on="gt")
        return model, inf

    if reducer == "tica":
        model, _, _, inf = dim_reduction.fit_tica(Xfit_gt, None, n_components=n_components, lag=lag)
        inf = dict(inf)
        inf["warning"] = "tica_fit_on_concat_gt_fit_frames"
        return model, inf

    raise ValueError(f"Unknown reducer: {reducer}")


def transform_reducer(model, reducer: str, X: np.ndarray, n_components: int) -> np.ndarray:
    Z = model.transform(X)
    if reducer == "tica":
        # pyemma-like tica returns more dims; clip
        return Z[:, :n_components]
    return Z


# -----------------------------------------------------------------------------
# PMF: global bins per method (within context+region)
# -----------------------------------------------------------------------------
def _update_global_range(gr: Dict[str, float], Zg: np.ndarray, Zp: np.ndarray):
    all_pts = np.concatenate([Zg, Zp], axis=0) if (Zp is not None and Zp.size > 0) else Zg
    x = all_pts[:, 0]
    y = all_pts[:, 1]
    if "xmin" not in gr:
        gr.update(xmin=float(x.min()), xmax=float(x.max()), ymin=float(y.min()), ymax=float(y.max()))
    else:
        gr["xmin"] = min(gr["xmin"], float(x.min()))
        gr["xmax"] = max(gr["xmax"], float(x.max()))
        gr["ymin"] = min(gr["ymin"], float(y.min()))
        gr["ymax"] = max(gr["ymax"], float(y.max()))


def _write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


# -----------------------------------------------------------------------------
# Core evaluation per (ALIGN_CONTEXT, ASSESS_REGION): run coords/ca/dihed across reducers
# -----------------------------------------------------------------------------
def assess_region_shared_space_gt_pred(
    *,
    align_context: str,
    assess_region: Sequence[str],
    gt_aligned: List[TrajectoryView],
    pred_aligned: List[TrajectoryView],
    gt_pdb_kept: List[str],
    pred_pdb_kept: List[str],
    tcr_names_kept: List[str],
    outdir_region: Path,
    reducers: List[str],
    trust_k: int,
    subsample_frames_for_metrics: Optional[int],
    seed: int,
    mantel_method: str,
    mantel_perms: int,
    lag: int,
    n_components: int,
    max_pairs: int,
    fit_frames_per_tcr: int,
    fit_total_max_samples: int,
    calibration_pairs: int,
    temperature: float,
    xbins: int,
    ybins: int,
):
    outdir_region.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ---------- helper: subsample indices for fitting (per TCR)
    def fit_indices(n: int) -> np.ndarray:
        if fit_frames_per_tcr <= 0 or fit_frames_per_tcr >= n:
            return np.arange(n, dtype=int)
        # uniform spacing avoids stochastic variance
        return np.linspace(0, n - 1, fit_frames_per_tcr, dtype=int)

    # ---------- helper: mantel/trust scoring for one trajectory and embedding
    def score_one(
        X: np.ndarray,
        Z: np.ndarray,
        traj: md.Trajectory,
        rmsd_atom_indices: np.ndarray,
    ) -> Dict[str, Any]:
        tw = float(tw_metric(X, Z, n_neighbors=trust_k, subsample=subsample_frames_for_metrics, seed=seed))

        rs = np.random.default_rng(seed)
        n_frames = traj.n_frames
        if subsample_frames_for_metrics is not None and subsample_frames_for_metrics < n_frames:
            I = np.sort(rs.choice(n_frames, size=subsample_frames_for_metrics, replace=False))
            Z_use = Z[I]
            T_use = traj[I]
        else:
            Z_use = Z
            T_use = traj

        D = rmsd_matrix_no_superpose(T_use, rmsd_atom_indices)
        # embedding distances
        from scipy.spatial.distance import pdist, squareform
        E = squareform(pdist(Z_use))
        r, p = mantel_test(D, E, method=mantel_method, permutations=mantel_perms, seed=seed)
        return {"trust": tw, "mantel": {"r": float(r), "p": float(p)}}

    # =============================================================================
    # 1) COORDS (CA,C,N)
    # =============================================================================
    coords_atoms = {"CA", "C", "N"}
    gt_maps = [_region_atom_map_insertion_aware(tv, pdb, assess_region, coords_atoms) for tv, pdb in zip(gt_aligned, gt_pdb_kept)]
    pred_maps = [_region_atom_map_insertion_aware(tv, pdb, assess_region, coords_atoms) for tv, pdb in zip(pred_aligned, pred_pdb_kept)]

    # exclude any pair where either side has empty selection
    valid = [i for i in range(len(tcr_names_kept)) if (len(gt_maps[i]) >= 3 and len(pred_maps[i]) >= 3)]
    if len(valid) >= 2:
        gt_maps_v = [gt_maps[i] for i in valid]
        pred_maps_v = [pred_maps[i] for i in valid]
        names_v = [tcr_names_kept[i] for i in valid]
        gt_v = [gt_aligned[i] for i in valid]
        pred_v = [pred_aligned[i] for i in valid]

        common_keys = _strict_common_keys(gt_maps_v + pred_maps_v)
        if len(common_keys) >= 3:
            # Build full X per TCR (GT+pred) + fit matrix (GT only)
            Xg_list, Xp_list, idxg_list, idxp_list = [], [], [], []
            Xfit_chunks = []

            for tv_g, tv_p, mg, mp in zip(gt_v, pred_v, gt_maps_v, pred_maps_v):
                Xg, idxg = extract_coords_features(tv_g, mg, common_keys)
                Xp, idxp = extract_coords_features(tv_p, mp, common_keys)
                Xg_list.append(Xg); Xp_list.append(Xp)
                idxg_list.append(idxg); idxp_list.append(idxp)

                fi = fit_indices(Xg.shape[0])
                Xfit_chunks.append(Xg[fi])

            Xfit_gt = np.concatenate(Xfit_chunks, axis=0)
            # hard cap for kernel methods
            if fit_total_max_samples > 0 and Xfit_gt.shape[0] > fit_total_max_samples:
                sel = rng.choice(Xfit_gt.shape[0], size=fit_total_max_samples, replace=False)
                Xfit_gt = Xfit_gt[np.sort(sel)]

            # global scaling based on GT fit frames only
            mu = Xfit_gt.mean(axis=0)
            sd = Xfit_gt.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0

            Xfit_gt_sc = (Xfit_gt - mu) / sd
            Xg_list_sc = [(X - mu) / sd for X in Xg_list]
            Xp_list_sc = [(X - mu) / sd for X in Xp_list]
            print(len(Xg_list_sc),len(Xp_list_sc))

            for red in reducers:
                method_name = f"coords_{red}"
                method_dir_summary = outdir_region / "__method_summaries__" / method_name
                method_dir_summary.mkdir(parents=True, exist_ok=True)

                try:
                    model, inf = fit_reducer_shared(red, Xfit_gt_sc, n_components=n_components, lag=lag, seed=seed)
                except Exception as e:
                    (method_dir_summary / "SKIP.txt").write_text(f"Failed to fit {method_name}: {e}\n")
                    continue

                Zg_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xg_list_sc]
                Zp_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xp_list_sc]

                # Global RMSD calibration (GT only)
                calib_csv = str(method_dir_summary / "rmsd_calibration.csv")
                calib_txt = str(method_dir_summary / "rmsd_calibration.txt")
                k, corr, n_used, center = rmsd_calibration_factor_all_space(
                    Zg_list=Zg_list,
                    gt_traj_aligned_list=gt_v,
                    region_names=list(assess_region),
                    atom_names={"CA"},
                    n_pairs=calibration_pairs,
                    random_state=seed,
                    out_csv=calib_csv,
                    out_txt=calib_txt,
                    return_center=True,
                )
                assert center is not None

                # Apply scaling + centering, compute global_range for PMF
                gr: Dict[str, float] = {}
                for Zg, Zp in zip(Zg_list, Zp_list):
                    Zg_sc = (Zg * k) - center
                    Zp_sc = (Zp * k) - center
                    _update_global_range(gr, Zg_sc, Zp_sc)
                _write_json(method_dir_summary / "global_range.json", gr)

                global_range = (gr["xmin"], gr["xmax"], gr["ymin"], gr["ymax"])

                # Save per TCR metrics + embeddings + PMF
                for name, tv_g, tv_p, Xg, Xp, Zg, Zp, idxg in zip(names_v, gt_v, pred_v, Xg_list_sc, Xp_list_sc, Zg_list, Zp_list, idxg_list):
                    out_tcr_method = outdir_region / name / method_name
                    out_tcr_method.mkdir(parents=True, exist_ok=True)

                    Zg_sc = (Zg * k) - center
                    Zp_sc = (Zp * k) - center

                    np.savez_compressed(out_tcr_method / "embeddings_scaled_centered.npz", Zg=Zg_sc, Zp=Zp_sc)

                    # scores (use UN-SCALED Z for neighborhood metrics? you can use scaled; it’s just affine)
                    gt_scores = score_one(Xg, Zg, tv_g.mdtraj, idxg)
                    pred_scores = score_one(Xp, Zp, tv_p.mdtraj, idxg)

                    metrics = {
                        "reducer": red,
                        "trust": {"gt": gt_scores["trust"], "pred": pred_scores["trust"]},
                        "mantel": {"gt": gt_scores["mantel"], "pred": pred_scores["mantel"], "combined": None},
                        "rmsd_calibration": {"k_A_per_unit": float(k), "corr": float(corr), "n_pairs_used": int(n_used)},
                        "align_context": align_context,
                        "assess_region": list(assess_region),
                        "assess_type": "coords",
                        "fit_info": inf,
                    }
                    (out_tcr_method / "coords_metrics.json").write_text(json.dumps(metrics, indent=2))

                    pmf_dir = out_tcr_method / "pmf_global_bins"
                    pmf_dir.mkdir(parents=True, exist_ok=True)
                    oriol_analysis(
                        xbins, ybins,
                        Zg_sc, Zp_sc,
                        temperature,
                        name="",
                        outfolder=str(pmf_dir),
                        tv_gt=None, tv_pred=None,
                        global_range=global_range,
                    )
        else:
            (outdir_region / "SKIP_coords.txt").write_text(
                f"coords: too few strict-common atoms ({len(common_keys)}) for region={assess_region}\n"
            )
    else:
        (outdir_region / "SKIP_coords.txt").write_text(
            f"coords: too few valid TCRs (need >=2) with atoms for region={assess_region}\n"
        )

    # =============================================================================
    # 2) CA_DIST (CA only)
    # =============================================================================
    ca_atoms = {"CA"}
    gt_maps = [_region_atom_map_insertion_aware(tv, pdb, assess_region, ca_atoms) for tv, pdb in zip(gt_aligned, gt_pdb_kept)]
    pred_maps = [_region_atom_map_insertion_aware(tv, pdb, assess_region, ca_atoms) for tv, pdb in zip(pred_aligned, pred_pdb_kept)]
    valid = [i for i in range(len(tcr_names_kept)) if (len(gt_maps[i]) >= 3 and len(pred_maps[i]) >= 3)]

    if len(valid) >= 2:
        gt_maps_v = [gt_maps[i] for i in valid]
        pred_maps_v = [pred_maps[i] for i in valid]
        names_v = [tcr_names_kept[i] for i in valid]
        gt_v = [gt_aligned[i] for i in valid]
        pred_v = [pred_aligned[i] for i in valid]

        common_keys = _strict_common_keys(gt_maps_v + pred_maps_v)
        K = len(common_keys)
        if K >= 3:
            # pairs (consistent)
            all_pairs = K * (K - 1) // 2
            if all_pairs <= max_pairs:
                pairs = np.array([(i, j) for i in range(K) for j in range(i + 1, K)], dtype=int)
            else:
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

            Xg_list, Xp_list, idxg_list = [], [], []
            Xfit_chunks = []

            for tv_g, tv_p, mg, mp in zip(gt_v, pred_v, gt_maps_v, pred_maps_v):
                Xg, idxg = extract_ca_dist_features(tv_g, mg, common_keys, pairs)
                Xp, _    = extract_ca_dist_features(tv_p, mp, common_keys, pairs)
                Xg_list.append(Xg); Xp_list.append(Xp)
                idxg_list.append(idxg)
                fi = fit_indices(Xg.shape[0])
                Xfit_chunks.append(Xg[fi])

            Xfit_gt = np.concatenate(Xfit_chunks, axis=0)
            if fit_total_max_samples > 0 and Xfit_gt.shape[0] > fit_total_max_samples:
                sel = rng.choice(Xfit_gt.shape[0], size=fit_total_max_samples, replace=False)
                Xfit_gt = Xfit_gt[np.sort(sel)]

            mu = Xfit_gt.mean(axis=0)
            sd = Xfit_gt.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0

            Xfit_gt_sc = (Xfit_gt - mu) / sd
            Xg_list_sc = [(X - mu) / sd for X in Xg_list]
            Xp_list_sc = [(X - mu) / sd for X in Xp_list]

            for red in reducers:
                method_name = f"ca_{red}"
                method_dir_summary = outdir_region / "__method_summaries__" / method_name
                method_dir_summary.mkdir(parents=True, exist_ok=True)

                try:
                    model, inf = fit_reducer_shared(red, Xfit_gt_sc, n_components=n_components, lag=lag, seed=seed)
                except Exception as e:
                    (method_dir_summary / "SKIP.txt").write_text(f"Failed to fit {method_name}: {e}\n")
                    continue

                Zg_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xg_list_sc]
                Zp_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xp_list_sc]

                calib_csv = outdir_region / "__method_summaries__" / method_name
                k, corr, n_used, center = rmsd_calibration_factor_all_space(
                    Zg_list=Zg_list,
                    gt_traj_aligned_list=gt_v,
                    region_names=list(assess_region),
                    atom_names={"CA"},
                    n_pairs=calibration_pairs,
                    random_state=seed,
                    out_csv=str(calib / "rmsd_calibration.csv"),
                    out_txt=str(calib / "rmsd_calibration.txt"),
                    return_center=True,
                )
                assert center is not None

                gr: Dict[str, float] = {}
                for Zg, Zp in zip(Zg_list, Zp_list):
                    _update_global_range(gr, (Zg * k) - center, (Zp * k) - center)
                _write_json(method_dir_summary / "global_range.json", gr)
                global_range = (gr["xmin"], gr["xmax"], gr["ymin"], gr["ymax"])

                for name, tv_g, tv_p, Xg, Xp, Zg, Zp, idxg in zip(names_v, gt_v, pred_v, Xg_list_sc, Xp_list_sc, Zg_list, Zp_list, idxg_list):
                    out_tcr_method = outdir_region / name / method_name
                    out_tcr_method.mkdir(parents=True, exist_ok=True)

                    Zg_sc = (Zg * k) - center
                    Zp_sc = (Zp * k) - center
                    np.savez_compressed(out_tcr_method / "embeddings_scaled_centered.npz", Zg=Zg_sc, Zp=Zp_sc)

                    gt_scores = score_one(Xg, Zg, tv_g.mdtraj, idxg)
                    pred_scores = score_one(Xp, Zp, tv_p.mdtraj, idxg)

                    metrics = {
                        "reducer": red,
                        "trust": {"gt": gt_scores["trust"], "pred": pred_scores["trust"]},
                        "mantel": {"gt": gt_scores["mantel"], "pred": pred_scores["mantel"], "combined": None},
                        "rmsd_calibration": {"k_A_per_unit": float(k), "corr": float(corr), "n_pairs_used": int(n_used)},
                        "align_context": align_context,
                        "assess_region": list(assess_region),
                        "assess_type": "ca",
                        "fit_info": inf,
                    }
                    (out_tcr_method / "ca_metrics.json").write_text(json.dumps(metrics, indent=2))

                    pmf_dir = out_tcr_method / "pmf_global_bins"
                    pmf_dir.mkdir(parents=True, exist_ok=True)
                    oriol_analysis(
                        xbins, ybins,
                        Zg_sc, Zp_sc,
                        temperature,
                        name="",
                        outfolder=str(pmf_dir),
                        tv_gt=None, tv_pred=None,
                        global_range=global_range,
                    )
        else:
            (outdir_region / "SKIP_ca.txt").write_text(
                f"ca: too few strict-common CA atoms ({K}) for region={assess_region}\n"
            )
    else:
        (outdir_region / "SKIP_ca.txt").write_text(
            f"ca: too few valid TCRs (need >=2) with CA atoms for region={assess_region}\n"
        )

    # =============================================================================
    # 3) DIHED (phi/psi sincos) -- shared residue keys across GT+pred
    # =============================================================================
    # For dihedrals, we will:
    #   - compute residue keys from GT+pred; require both have dihedral for those residues
    #   - use CA indices for Mantel RMSD if available (same as above)
    # NOTE: your insertion-aware mapping isn't used for dihed keys in mdtraj; we keep residue-level strictness.
    common_res_keys_gt = strict_common_residue_keys_for_dihedrals(gt_aligned, gt_pdb_kept, assess_region)
    common_res_keys_pred = strict_common_residue_keys_for_dihedrals(pred_aligned, pred_pdb_kept, assess_region)
    common_res_keys = sorted(set(common_res_keys_gt) & set(common_res_keys_pred))

    if len(common_res_keys) >= 1:
        Xg_list = [extract_dihedral_features_shared_keys(tv, common_res_keys) for tv in gt_aligned]
        Xp_list = [extract_dihedral_features_shared_keys(tv, common_res_keys) for tv in pred_aligned]

        # valid pairs: need at least 2 frames and feature dim >0
        valid = [i for i in range(len(tcr_names_kept)) if (Xg_list[i].shape[0] >= 2 and Xp_list[i].shape[0] >= 2 and Xg_list[i].shape[1] > 0)]
        if len(valid) >= 2:
            names_v = [tcr_names_kept[i] for i in valid]
            gt_v = [gt_aligned[i] for i in valid]
            pred_v = [pred_aligned[i] for i in valid]
            Xg_list_v = [Xg_list[i] for i in valid]
            Xp_list_v = [Xp_list[i] for i in valid]

            Xfit_chunks = []
            for Xg in Xg_list_v:
                fi = fit_indices(Xg.shape[0])
                Xfit_chunks.append(Xg[fi])
            Xfit_gt = np.concatenate(Xfit_chunks, axis=0)
            if fit_total_max_samples > 0 and Xfit_gt.shape[0] > fit_total_max_samples:
                sel = rng.choice(Xfit_gt.shape[0], size=fit_total_max_samples, replace=False)
                Xfit_gt = Xfit_gt[np.sort(sel)]

            mu = Xfit_gt.mean(axis=0)
            sd = Xfit_gt.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0

            Xfit_gt_sc = (Xfit_gt - mu) / sd
            Xg_list_sc = [(X - mu) / sd for X in Xg_list_v]
            Xp_list_sc = [(X - mu) / sd for X in Xp_list_v]

            # For Mantel RMSD in dihed space, use CA selection if possible
            ca_maps_gt = [_region_atom_map_insertion_aware(tv, pdb, assess_region, {"CA"}) for tv, pdb in zip(gt_v, [gt_pdb_kept[i] for i in valid])]
            ca_maps_pred = [_region_atom_map_insertion_aware(tv, pdb, assess_region, {"CA"}) for tv, pdb in zip(pred_v, [pred_pdb_kept[i] for i in valid])]
            common_ca = _strict_common_keys(ca_maps_gt + ca_maps_pred)
            use_mantel = len(common_ca) >= 3

            if use_mantel:
                idxg_list = [_indices_for_keys(m, common_ca) for m in ca_maps_gt]
            else:
                idxg_list = [np.array([], dtype=int) for _ in gt_v]

            for red in reducers:
                method_name = f"dihed_{red}"
                method_dir_summary = outdir_region / "__method_summaries__" / method_name
                method_dir_summary.mkdir(parents=True, exist_ok=True)

                try:
                    model, inf = fit_reducer_shared(red, Xfit_gt_sc, n_components=n_components, lag=lag, seed=seed)
                except Exception as e:
                    (method_dir_summary / "SKIP.txt").write_text(f"Failed to fit {method_name}: {e}\n")
                    continue

                Zg_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xg_list_sc]
                Zp_list = [transform_reducer(model, red, Xsc, n_components) for Xsc in Xp_list_sc]

                # Calibration still uses GT trajectories + CA RMSD (if CA missing, calibration is meaningless)
                if not use_mantel:
                    (method_dir_summary / "SKIP_calibration.txt").write_text(
                        "No strict-common CA atoms for dihed region; skipping RMSD calibration + PMF.\n"
                    )
                    continue

                k, corr, n_used, center = rmsd_calibration_factor_all_space(
                    Zg_list=Zg_list,
                    gt_traj_aligned_list=gt_v,
                    region_names=list(assess_region),
                    atom_names={"CA"},
                    n_pairs=calibration_pairs,
                    random_state=seed,
                    out_csv=str(method_dir_summary / "rmsd_calibration.csv"),
                    out_txt=str(method_dir_summary / "rmsd_calibration.txt"),
                    return_center=True,
                )
                assert center is not None

                gr: Dict[str, float] = {}
                for Zg, Zp in zip(Zg_list, Zp_list):
                    _update_global_range(gr, (Zg * k) - center, (Zp * k) - center)
                _write_json(method_dir_summary / "global_range.json", gr)
                global_range = (gr["xmin"], gr["xmax"], gr["ymin"], gr["ymax"])

                for name, tv_g, tv_p, Xg, Xp, Zg, Zp, idxg in zip(names_v, gt_v, pred_v, Xg_list_sc, Xp_list_sc, Zg_list, Zp_list, idxg_list):
                    out_tcr_method = outdir_region / name / method_name
                    out_tcr_method.mkdir(parents=True, exist_ok=True)

                    Zg_sc = (Zg * k) - center
                    Zp_sc = (Zp * k) - center
                    np.savez_compressed(out_tcr_method / "embeddings_scaled_centered.npz", Zg=Zg_sc, Zp=Zp_sc)

                    # Trust always computed; Mantel computed only if idxg non-empty
                    gt_tw = float(tw_metric(Xg, Zg, n_neighbors=trust_k, subsample=subsample_frames_for_metrics, seed=seed))
                    pred_tw = float(tw_metric(Xp, Zp, n_neighbors=trust_k, subsample=subsample_frames_for_metrics, seed=seed))

                    if idxg.size >= 3:
                        gt_scores = score_one(Xg, Zg, tv_g.mdtraj, idxg)
                        pred_scores = score_one(Xp, Zp, tv_p.mdtraj, idxg)
                        mantel_gt = gt_scores["mantel"]
                        mantel_pred = pred_scores["mantel"]
                    else:
                        mantel_gt = {"r": float("nan"), "p": float("nan")}
                        mantel_pred = {"r": float("nan"), "p": float("nan")}

                    metrics = {
                        "reducer": red,
                        "trust": {"gt": gt_tw, "pred": pred_tw},
                        "mantel": {"gt": mantel_gt, "pred": mantel_pred, "combined": None},
                        "rmsd_calibration": {"k_A_per_unit": float(k), "corr": float(corr), "n_pairs_used": int(n_used)},
                        "align_context": align_context,
                        "assess_region": list(assess_region),
                        "assess_type": "dihed",
                        "fit_info": inf,
                    }
                    (out_tcr_method / "dihed_metrics.json").write_text(json.dumps(metrics, indent=2))

                    pmf_dir = out_tcr_method / "pmf_global_bins"
                    pmf_dir.mkdir(parents=True, exist_ok=True)
                    oriol_analysis(
                        xbins, ybins,
                        Zg_sc, Zp_sc,
                        temperature,
                        name="",
                        outfolder=str(pmf_dir),
                        tv_gt=None, tv_pred=None,
                        global_range=global_range,
                    )
        else:
            (outdir_region / "SKIP_dihed.txt").write_text("Too few valid TCRs for dihed evaluation.\n")
    else:
        (outdir_region / "SKIP_dihed.txt").write_text("No strict-common dihedral-bearing residues.\n")


# -----------------------------------------------------------------------------
# Best-metric aggregation across regions & alignment contexts
# -----------------------------------------------------------------------------
def calc_best_metric_shared(outdir: Path, rank_by: str = "mantel.gt.r") -> Path:
    rows: List[Dict[str, Any]] = []

    for align_dir in sorted([p for p in outdir.iterdir() if p.is_dir() and p.name.startswith("ALIGN_")]):
        for region_dir in sorted([p for p in align_dir.iterdir() if p.is_dir() and not p.name.startswith("__")]):
            try:
                best_method, score = get_best_metric(region_dir, rank_by=rank_by)
                rows.append(
                    {
                        "alignment_context": align_dir.name.replace("ALIGN_", ""),
                        "region": region_dir.name,
                        "best_method": best_method,
                        "score": float(score) if score is not None else float("nan"),
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "alignment_context": align_dir.name.replace("ALIGN_", ""),
                        "region": region_dir.name,
                        "best_method": None,
                        "score": float("nan"),
                        "error": str(e),
                    }
                )

    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = outdir / "best_methods_per_region_shared.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


# -----------------------------------------------------------------------------
# Data loading: GT + pred (DIG outputs)
# -----------------------------------------------------------------------------
def load_gt_pred_pairs(
    *,
    model_folder: str,
    model_name: str,
    gt_root: str,
    outdir: str,
    linker: str,
    max_tcr: int = 0,
) -> Tuple[List[str], List[TrajectoryView], List[TrajectoryView], List[str], List[str]]:
    """
    Returns:
      tcr_names (folder name / identifier),
      gt_tv_list,
      pred_tv_list,
      gt_pdb_list,
      pred_pdb_list
    """
    model_folder_p = Path(model_folder)
    gt_root_p = Path(gt_root)
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tcr_names: List[str] = []
    gt_tv_list: List[TrajectoryView] = []
    pred_tv_list: List[TrajectoryView] = []
    gt_pdb_list: List[str] = []
    pred_pdb_list: List[str] = []

    folders = sorted([p for p in model_folder_p.iterdir() if p.is_dir()])
    if max_tcr and max_tcr > 0:
        folders = folders[:max_tcr]

    for f in folders:
        tcr_name_full = f.name
        tcr_short = tcr_name_full.split("_")[0]

        # output staging for pred
        out_tcr = outdir_p / "_pred_cache" / tcr_name_full
        out_tcr.mkdir(parents=True, exist_ok=True)
        pred_xtc = str(out_tcr / "pred.xtc")
        pred_pdb = str(out_tcr / "pred.pdb")


        dig_output_dir = str(model_folder_p / tcr_name_full / model_name)
        if Path(pred_xtc).exists() and  Path(pred_pdb).exists():
            pass
        else:
            try:
                pred_xtc, pred_pdb = process_output(dig_output_dir, linker, pred_xtc, pred_pdb)
            except Exception as e:
                print(f"[SKIP] {tcr_name_full}: failed to process DIG output: {e}")
                continue

        gt_folder = gt_root_p / tcr_short
        gt_pdb = gt_folder / f"{tcr_short}.pdb"
        gt_xtc = gt_folder / f"{tcr_short}_Prod.xtc"
        if not (gt_pdb.exists() and gt_xtc.exists()):
            gt_xtc2 = gt_folder / f"{tcr_short}.xtc"
            if gt_pdb.exists() and gt_xtc2.exists():
                gt_xtc = gt_xtc2
            else:
                print(f"[SKIP] {tcr_name_full}: missing GT pdb/xtc for {tcr_short}")
                continue

        try:
            gt = TCR(input_pdb=str(gt_pdb), traj_path=str(gt_xtc), contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
            gt_pair = gt.pairs[0]
            tv_gt = gt_pair.traj
        except Exception as e:
            print(f"[SKIP] {tcr_name_full}: failed to load GT: {e}")
            continue

        try:
            pred = TCR(input_pdb=str(pred_pdb), traj_path=str(pred_xtc), contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
            pred_pair = pred.pairs[0]
            tv_pred = pred_pair.traj
        except Exception:
            # attach trajectory if needed
            try:
                pred = TCR(input_pdb=str(pred_pdb), traj_path=None, contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
                pred_pair = pred.pairs[0]
                pred_pair.attach_trajectory(str(pred_xtc), region_names=None, atom_names={"CA", "C", "N"})
                tv_pred = pred_pair.traj
            except Exception as e:
                print(f"[SKIP] {tcr_name_full}: failed to load/attach pred: {e}")
                continue

        tcr_names.append(tcr_short)  # keep the clean identifier in output path
        gt_tv_list.append(tv_gt)
        pred_tv_list.append(tv_pred)
        gt_pdb_list.append(str(gt_pdb))
        pred_pdb_list.append(str(pred_pdb))

    if len(tcr_names) == 0:
        raise RuntimeError("No GT+pred pairs loaded.")

    return tcr_names, gt_tv_list, pred_tv_list, gt_pdb_list, pred_pdb_list


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reducers = [s.strip() for s in args.reducers.split(",") if s.strip()]
    subsample_metrics = None if args.subsample_metrics <= 0 else int(args.subsample_metrics)

    # load pairs
    tcr_names, gt_tv_list, pred_tv_list, gt_pdb_list, pred_pdb_list = load_gt_pred_pairs(
        model_folder=args.model_folder,
        model_name=args.model_name,
        gt_root=args.gt_root,
        outdir=str(outdir),
        linker=args.linker,
        max_tcr=args.max_tcr,
    )
    print(f"[OK] Loaded {len(tcr_names)} GT+pred pairs.")

    cdrs = ["A_CDR1", "A_CDR2", "A_CDR3", "B_CDR1", "B_CDR2", "B_CDR3"]
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
    # 1) CDR self-alignment
    # -------------------------
    for cdr in cdrs:
        print(f"\n=== ALIGN context: {cdr} (self) ===")
        align_dir = outdir / f"ALIGN_{cdr}"

        pred_aligned, gt_aligned, kept, ref_info = align_all_pairs_global(
            pred_tv_list, gt_tv_list, pred_pdb_list, gt_pdb_list,
            region_names=[cdr],
            atom_names={"CA", "C", "N"},
            outdir=align_dir,
        )

        if len(kept) < 2:
            (align_dir / "SKIP.txt").write_text(f"Too few TCRs aligned for context {cdr}: kept={len(kept)}\n")
            continue

        names_kept = [tcr_names[i] for i in kept]
        gt_pdb_kept = [gt_pdb_list[i] for i in kept]
        pred_pdb_kept = [pred_pdb_list[i] for i in kept]
        print( gt_pdb_kept,pred_pdb_kept)

        region_dir = align_dir / cdr
        assess_region_shared_space_gt_pred(
            align_context=cdr,
            assess_region=[cdr],
            gt_aligned=gt_aligned,
            pred_aligned=pred_aligned,
            gt_pdb_kept=gt_pdb_kept,
            pred_pdb_kept=pred_pdb_kept,
            tcr_names_kept=names_kept,
            outdir_region=region_dir,
            reducers=reducers,
            trust_k=args.trust_k,
            subsample_frames_for_metrics=subsample_metrics,
            seed=args.seed,
            mantel_method=args.mantel_method,
            mantel_perms=args.mantel_perms,
            lag=args.lag,
            n_components=args.n_components,
            max_pairs=args.max_pairs,
            fit_frames_per_tcr=args.fit_frames_per_tcr,
            fit_total_max_samples=args.fit_total_max_samples,
            calibration_pairs=args.calibration_pairs,
            temperature=args.temperature,
            xbins=args.xbins,
            ybins=args.ybins,
        )

    # -------------------------
    # 2) Framework alignment
    # -------------------------
    for ctx_name, fr_regions in fw_contexts:
        print(f"\n=== ALIGN context: {ctx_name} (framework {fr_regions}) ===")
        align_dir = outdir / f"ALIGN_{ctx_name}"

        pred_aligned, gt_aligned, kept, ref_info = align_all_pairs_global(
            pred_tv_list, gt_tv_list, pred_pdb_list, gt_pdb_list,
            region_names=fr_regions,
            atom_names={"CA", "C", "N"},
            outdir=align_dir,
        )

        if len(kept) < 2:
            (align_dir / "SKIP.txt").write_text(f"Too few TCRs aligned for context {ctx_name}: kept={len(kept)}\n")
            continue

        names_kept = [tcr_names[i] for i in kept]
        gt_pdb_kept = [gt_pdb_list[i] for i in kept]
        pred_pdb_kept = [pred_pdb_list[i] for i in kept]

        # Assess each CDR + variable
        for region in cdrs + [ctx_name]:
            region_dir = align_dir / region
            assess_region_shared_space_gt_pred(
                align_context=ctx_name,
                assess_region=[region],
                gt_aligned=gt_aligned,
                pred_aligned=pred_aligned,
                gt_pdb_kept=gt_pdb_kept,
                pred_pdb_kept=pred_pdb_kept,
                tcr_names_kept=names_kept,
                outdir_region=region_dir,
                reducers=reducers,
                trust_k=args.trust_k,
                subsample_frames_for_metrics=subsample_metrics,
                seed=args.seed,
                mantel_method=args.mantel_method,
                mantel_perms=args.mantel_perms,
                lag=args.lag,
                n_components=args.n_components,
                max_pairs=args.max_pairs,
                fit_frames_per_tcr=args.fit_frames_per_tcr,
                fit_total_max_samples=args.fit_total_max_samples,
                calibration_pairs=args.calibration_pairs,
                temperature=args.temperature,
                xbins=args.xbins,
                ybins=args.ybins,
            )

        # Combined regions
        for region_list in combined_regions:
            label = "+".join(region_list)
            region_dir = align_dir / label
            assess_region_shared_space_gt_pred(
                align_context=ctx_name,
                assess_region=region_list,
                gt_aligned=gt_aligned,
                pred_aligned=pred_aligned,
                gt_pdb_kept=gt_pdb_kept,
                pred_pdb_kept=pred_pdb_kept,
                tcr_names_kept=names_kept,
                outdir_region=region_dir,
                reducers=reducers,
                trust_k=args.trust_k,
                subsample_frames_for_metrics=subsample_metrics,
                seed=args.seed,
                mantel_method=args.mantel_method,
                mantel_perms=args.mantel_perms,
                lag=args.lag,
                n_components=args.n_components,
                max_pairs=args.max_pairs,
                fit_frames_per_tcr=args.fit_frames_per_tcr,
                fit_total_max_samples=args.fit_total_max_samples,
                calibration_pairs=args.calibration_pairs,
                temperature=args.temperature,
                xbins=args.xbins,
                ybins=args.ybins,
            )

    # Best-method aggregation
    out_csv = calc_best_metric_shared(outdir, rank_by=args.rank_by)
    print(f"\n[DONE] Best-method summary: {out_csv}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-root", required=True, help="GT root folder with per-TCR subfolders containing <name>.pdb and <name>_Prod.xtc (or <name>.xtc).")
    p.add_argument("--model-folder", required=True, help="Folder containing per-TCR model output folders.")
    p.add_argument("--model-name", required=True, help="Subfolder within each model-folder/TCR/ that contains outputs.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--linker", default="", help="Linker passed to process_output (e.g., 'GGGGS'*3 for BioEmu-like).")
    p.add_argument("--max-tcr", type=int, default=0, help="Limit number of TCRs (0 = no limit).")

    p.add_argument("--reducers", default="pca,kpca_rbf,kpca_cosine,kpca_poly,diffmap,tica")
    p.add_argument("--rank-by", default="mantel.gt.r")

    p.add_argument("--trust-k", type=int, default=10)
    p.add_argument("--subsample-metrics", type=int, default=200, help="Frame subsample used for trust+mantel (0 disables).")
    p.add_argument("--mantel-method", default="spearman")
    p.add_argument("--mantel-perms", type=int, default=999)
    p.add_argument("--lag", type=int, default=5)
    p.add_argument("--n-components", type=int, default=2)

    p.add_argument("--max-pairs", type=int, default=20000, help="Max CA distance pairs for ca_dist features.")
    p.add_argument("--fit-frames-per-tcr", type=int, default=1000, help="GT frames per TCR used to build Xfit (uniform).")
    p.add_argument("--fit-total-max-samples", type=int, default=8000, help="Hard cap on total GT fit rows (kernel safety).")

    p.add_argument("--calibration-pairs", type=int, default=8000, help="Random pairs for global RMSD calibration.")
    p.add_argument("--seed", type=int, default=0)

    # PMF
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--xbins", type=int, default=50)
    p.add_argument("--ybins", type=int, default=50)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)


    """python run_scaled_all_tcr_space.py \
    --gt-root /mnt/larry/lilian/DATA/Cory_data \
    --model-folder /mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/ \
    --model-name bioemu_filter \
    --outdir /workspaces/Graphormer/TCR_Metrics/outputs/global_best_metric_all_tcr_shared_gt_pred \
    --linker "GGGGSGGGGSGGGGS"
    """