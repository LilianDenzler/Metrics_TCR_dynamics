#!/usr/bin/env python3
import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

import os
from pathlib import Path
import yaml
import numpy as np
import mdtraj as md
from typing import Dict, Any, List, Tuple, Set, Optional

from TCR_TOOLS.classes.tcr import TCR, TrajectoryView
from dig_output_handler.process_output import process_output

from TCR_TOOLS.scoring.embeddings import run_coords, run_ca_dist, run_dihedrals
from TCR_TOOLS.scoring.plotters import plot_pca
from TCR_TOOLS.scoring.pmf_kde import oriol_analysis

from TCR_Metrics.pipelines.calc_metrics_model_outputs.Standardise_by_RMSD import (
    rmsd_calibration_factor_all_space,
)


HERE = Path(__file__).resolve().parent


# -----------------------------
# Insertion-code aware atom keys
# -----------------------------
def _parse_pdb_serial_map(pdb_path: str) -> Dict[int, Tuple[str, int, str, str]]:
    """
    Map PDB atom serial -> (chain_id, resseq, icode, atom_name).

    This is the only robust way to retain insertion codes with mdtraj,
    because mdtraj does not expose insertion codes in its topology API.
    """
    serial_map: Dict[int, Tuple[str, int, str, str]] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # PDB columns
            serial = int(line[6:11])
            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            if icode == "" or icode == " ":
                icode = ""
            serial_map[serial] = (chain_id, resseq, icode, atom_name)
    return serial_map


def _region_atom_map_insertion_aware(
    tv: TrajectoryView,
    pdb_topology_path: str,
    region_names: List[str],
    atom_names: Set[str],
) -> Dict[Tuple[str, int, str, str], int]:
    """
    Return dict: (chain_id, resseq, icode, atom_name) -> atom_index (in full traj)
    for the region selection of a TrajectoryView.
    """
    serial_map = _parse_pdb_serial_map(pdb_topology_path)

    idx = list(tv.domain_idx(region_names=region_names, atom_names=atom_names))
    top = tv._traj.topology

    out: Dict[Tuple[str, int, str, str], int] = {}
    collisions = []

    for i in idx:
        a = top.atom(int(i))
        serial = getattr(a, "serial", None)
        if serial is None or int(serial) not in serial_map:
            # fall back: no insertion code info
            chain_id = getattr(a.residue.chain, "chain_id", str(a.residue.chain.index))
            key = (str(chain_id), int(a.residue.resSeq), "", str(a.name))
        else:
            key = serial_map[int(serial)]
        if key in out:
            collisions.append(key)
        out[key] = int(i)

    if collisions:
        # if this triggers, it means your topology is inconsistent (or serials reused)
        sample = "\n".join([f"  {c}" for c in collisions[:10]])
        raise ValueError(
            "Duplicate insertion-aware atom keys detected. This usually indicates "
            "a PDB topology/serial inconsistency.\n"
            f"Example duplicates:\n{sample}"
        )

    return out


def _common_indices_to_reference(
    tv: TrajectoryView,
    tv_pdb: str,
    ref_key_to_idx: Dict[Tuple[str, int, str, str], int],
    region_names: List[str],
    atom_names: Set[str],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, str, str]]]:
    tv_map = _region_atom_map_insertion_aware(tv, tv_pdb, region_names, atom_names)
    keys = sorted(set(tv_map.keys()) & set(ref_key_to_idx.keys()))
    tv_idx = np.array([tv_map[k] for k in keys], dtype=int)
    ref_idx = np.array([ref_key_to_idx[k] for k in keys], dtype=int)
    return tv_idx, ref_idx, keys


def _gt_representative_frame_index(
    tv_gt: TrajectoryView,
    gt_pdb: str,
    region_names: List[str],
    atom_names: Set[str],
    max_iter: int = 20,
) -> int:
    """
    Iterative medoid-to-mean inside ONE trajectory using the selected region atoms.
    """
    gt_traj = tv_gt._traj
    gt_map = _region_atom_map_insertion_aware(tv_gt, gt_pdb, region_names, atom_names)
    gt_idx = np.array(list(gt_map.values()), dtype=int)
    if gt_idx.size < 3:
        raise ValueError(f"Too few atoms for representative frame selection: {gt_idx.size}")

    ref_i = 0
    prev = None
    work = gt_traj.slice(np.arange(gt_traj.n_frames), copy=True)

    for _ in range(max_iter):
        ref_frame = gt_traj[ref_i]
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
    region_names: List[str],
    atom_names: Set[str],
    min_support_frac: float = 0.5,
    max_iter: int = 15,
) -> Tuple[int, int]:
    """
    Choose ONE global reference frame among all GTs by:
      1) pick representative frame per GT
      2) iteratively pick the medoid to the mean structure over supported keys
    """
    rep_frame_idx = []
    rep_maps = []
    for tv_gt, gt_pdb in zip(gt_tv_list, gt_pdb_list):
        fi = _gt_representative_frame_index(tv_gt, gt_pdb, region_names, atom_names)
        rep_frame_idx.append(fi)
        rep_maps.append(_region_atom_map_insertion_aware(tv_gt, gt_pdb, region_names, atom_names))

    # start: GT with most region atoms
    n_atoms = [len(m) for m in rep_maps]
    ref_gt_index = int(np.argmax(n_atoms))

    for _ in range(max_iter):
        ref_tv = gt_tv_list[ref_gt_index]
        ref_fi = rep_frame_idx[ref_gt_index]
        ref_frame = ref_tv._traj[ref_fi]

        ref_key_to_idx = rep_maps[ref_gt_index]
        ref_keys = list(ref_key_to_idx.keys())

        key_coords = {k: [] for k in ref_keys}
        aligned_rep_coords = []

        for gi, (tv, gt_pdb) in enumerate(zip(gt_tv_list, gt_pdb_list)):
            fi = rep_frame_idx[gi]

            tv_idx, ref_idx, keys = _common_indices_to_reference(
                tv, gt_pdb, ref_key_to_idx, region_names, atom_names
            )
            if len(keys) < 3:
                aligned_rep_coords.append((gi, None, keys))
                continue

            rep = tv._traj[fi].slice([0], copy=True)
            rep.superpose(ref_frame, atom_indices=tv_idx, ref_atom_indices=ref_idx)
            coords = rep.xyz[0, tv_idx, :]
            aligned_rep_coords.append((gi, coords, keys))

            for k, c in zip(keys, coords):
                key_coords[k].append(c)

        support_min = max(2, int(np.ceil(min_support_frac * len(gt_tv_list))))
        supported_keys = [k for k, cs in key_coords.items() if len(cs) >= support_min]
        if len(supported_keys) < 3:
            supported_keys = [k for k, cs in key_coords.items() if len(cs) >= 2]
        if len(supported_keys) < 3:
            break

        mean_coords = np.stack([np.mean(key_coords[k], axis=0) for k in supported_keys], axis=0)

        best = None
        supported_set = set(supported_keys)
        supported_index = {k: i for i, k in enumerate(supported_keys)}

        for gi, coords, keys in aligned_rep_coords:
            if coords is None:
                continue
            key_to_pos = {k: coords[i] for i, k in enumerate(keys)}
            common = [k for k in supported_keys if k in set(keys)]
            if len(common) < 3:
                continue
            X = np.stack([key_to_pos[k] for k in common], axis=0)
            Y = np.stack([mean_coords[supported_index[k]] for k in common], axis=0)
            rmsd = float(np.sqrt(np.mean(np.sum((X - Y) ** 2, axis=1))))
            if best is None or rmsd < best[0]:
                best = (rmsd, gi)

        if best is None:
            break

        new_ref = int(best[1])
        if new_ref == ref_gt_index:
            break
        ref_gt_index = new_ref

    return ref_gt_index, rep_frame_idx[ref_gt_index]


def align_all_pairs_global(
    pred_tv_list: List[TrajectoryView],
    gt_tv_list: List[TrajectoryView],
    pred_pdb_list: List[str],
    gt_pdb_list: List[str],
    region_names: List[str],
    atom_names: Set[str],
    outdir: Optional[str] = None,
    min_support_frac: float = 0.5,
    max_iter_ref: int = 15,
) -> Tuple[List[TrajectoryView], List[TrajectoryView], Dict[str, Any]]:
    """
    Align ALL GT and ALL preds to ONE shared global reference frame (chosen from GT).
    Handles missing residues by aligning on the intersection of insertion-aware atom keys.
    """
    assert len(pred_tv_list) == len(gt_tv_list) == len(pred_pdb_list) == len(gt_pdb_list)

    ref_gt_i, ref_frame_i = _choose_global_reference_gt_rep(
        gt_tv_list=gt_tv_list,
        gt_pdb_list=gt_pdb_list,
        region_names=region_names,
        atom_names=atom_names,
        min_support_frac=min_support_frac,
        max_iter=max_iter_ref,
    )

    ref_tv = gt_tv_list[ref_gt_i]
    ref_pdb = gt_pdb_list[ref_gt_i]
    ref_frame = ref_tv._traj[ref_frame_i]
    ref_key_to_idx = _region_atom_map_insertion_aware(ref_tv, ref_pdb, region_names, atom_names)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        ref_frame.save_pdb(os.path.join(outdir, "global_ref_frame.pdb"))
        with open(os.path.join(outdir, "global_ref_info.txt"), "w") as f:
            f.write(f"ref_gt_index={ref_gt_i}\n")
            f.write(f"ref_frame_index={ref_frame_i}\n")
            f.write(f"region_names={region_names}\n")
            f.write(f"atom_names={sorted(list(atom_names))}\n")

    gt_aligned_list = []
    pred_aligned_list = []

    for i, (tv_gt, tv_pred, gt_pdb, pred_pdb) in enumerate(zip(gt_tv_list, pred_tv_list, gt_pdb_list, pred_pdb_list)):
        gt_idx, ref_idx_gt, keys_gt = _common_indices_to_reference(
            tv_gt, gt_pdb, ref_key_to_idx, region_names, atom_names
        )
        pred_idx, ref_idx_pred, keys_pred = _common_indices_to_reference(
            tv_pred, pred_pdb, ref_key_to_idx, region_names, atom_names
        )

        if len(keys_gt) < 3:
            raise ValueError(f"[GT {i}] Too few common atoms to global reference for {region_names}: {len(keys_gt)}")
        if len(keys_pred) < 3:
            raise ValueError(f"[PRED {i}] Too few common atoms to global reference for {region_names}: {len(keys_pred)}")

        gt_traj = tv_gt._traj.slice(np.arange(tv_gt._traj.n_frames), copy=True)
        pred_traj = tv_pred._traj.slice(np.arange(tv_pred._traj.n_frames), copy=True)

        gt_traj.superpose(ref_frame, atom_indices=gt_idx, ref_atom_indices=ref_idx_gt)
        pred_traj.superpose(ref_frame, atom_indices=pred_idx, ref_atom_indices=ref_idx_pred)

        gt_aligned_list.append(TrajectoryView(gt_traj, tv_gt._chain_map))
        pred_aligned_list.append(TrajectoryView(pred_traj, tv_pred._chain_map))

    ref_info = {
        "ref_gt_index": ref_gt_i,
        "ref_frame_index": ref_frame_i,
        "region_names": region_names,
        "atom_names": sorted(list(atom_names)),
    }
    return pred_aligned_list, gt_aligned_list, ref_info


# -----------------------------
# Global range handling (bins)
# -----------------------------
def make_region_key(assess_mode: str, alignment_context: str, region_names: List[str]) -> str:
    return f"{assess_mode}|{alignment_context}|{'+'.join(region_names)}"


def update_global_range(global_ranges: dict, region_key: str, Zg_sc: np.ndarray, Zp_sc: np.ndarray):
    all_pts = Zg_sc if (Zp_sc is None or Zp_sc.size == 0) else np.concatenate([Zg_sc, Zp_sc], axis=0)
    x = all_pts[:, 0]
    y = all_pts[:, 1]
    local = dict(xmin=float(x.min()), xmax=float(x.max()), ymin=float(y.min()), ymax=float(y.max()))
    if region_key not in global_ranges:
        global_ranges[region_key] = local
    else:
        g = global_ranges[region_key]
        g["xmin"] = min(g["xmin"], local["xmin"])
        g["xmax"] = max(g["xmax"], local["xmax"])
        g["ymin"] = min(g["ymin"], local["ymin"])
        g["ymax"] = max(g["ymax"], local["ymax"])


# -----------------------------
# Assessment: compute embeddings in one shared space
# -----------------------------
def assess_collect_all_tcr_space(
    assess_mode: str,
    gt_tv_aligned_list: List[TrajectoryView],
    pred_tv_aligned_list: List[TrajectoryView],
    per_tcr_outdirs: List[str],
    regions: List[str],
    fit_on: str,
    global_ranges: dict,
    alignment_context: str,
    rmsd_pairs: int = 8000,
    seed: int = 0,
):
    """
    1) Compute embeddings in a SINGLE shared reducer space across all TCRs.
    2) Compute one GLOBAL RMSD calibration factor k and a GLOBAL center.
    3) Save scaled+centered embeddings per TCR in its region folder.
    4) Update global_ranges for later global-binned PMF.
    """
    assess_type = assess_mode.split("_")[0]   # dihed / coords / ca
    reducer = "_".join(assess_mode.split("_")[1:])

    # ---- embeddings (you said you have run_all_tcr_space functions in your environment) ----
    if assess_type == "dihed":
        Zg_all, Zp_all, results = run_dihedrals.run_all_tcr_space(
            gt_tv_aligned_list,
            pred_tv_aligned_list,
            outdir=None,
            regions=regions,
            lag=5,
            reducer=reducer,
            fit_on=fit_on,
            dihedrals=("phi", "psi"),
            encode="sincos",
            subsample=100,
            mantel_perms=999,
        )
    elif assess_type == "coords":
        Zg_all, Zp_all, results = run_coords.run_all_tcr_space(
            tv_gt=gt_tv_aligned_list,
            tv_pred=pred_tv_aligned_list,
            outdir=None,
            regions=regions,
            lag=5,
            atoms=("CA", "C", "N"),
            reducer=reducer,
            n_components=2,
            fit_on=fit_on,
            use_gt_scaler=False,
            subsample=100,
            mantel_method="spearman",
            mantel_perms=999,
        )
    elif assess_type == "ca":
        Zg_all, Zp_all, results = run_ca_dist.run_all_tcr_space(
            gt_tv_aligned_list,
            pred_tv_aligned_list,
            outdir=None,
            regions=regions,
            fit_on=fit_on,
            reducer=reducer,
            n_components=2,
            max_pairs=20000,
            subsample=100,
            lag=5,
            mantel_perms=999,
        )
    else:
        raise ValueError(f"Unknown assess_type: {assess_type}")

    # results must be per-TCR: (Zg_i, Zp_i, info, tw, mantel, outdir_i) OR similar
    # We only need Zg_i and Zp_i; we will override output directories using per_tcr_outdirs.

    Zg_list = []
    Zp_list = []
    for item in results:
        # tolerate different tuple lengths by convention
        Zg_i = item[0]
        Zp_i = item[1]
        Zg_list.append(Zg_i)
        Zp_list.append(Zp_i)

    # ---- global RMSD calibration ----
    out_csv = os.path.join(str(Path(per_tcr_outdirs[0]).parents[1]), "rmsd_calibration_all_space.csv")
    k_scale, corr, center_scaled = rmsd_calibration_factor_all_space(
        Zg_list=Zg_list,
        gt_traj_aligned_list=gt_tv_aligned_list,
        region_names=regions,
        atom_names={"CA"},
        n_pairs=rmsd_pairs,
        random_state=seed,
        out_csv=out_csv,
        return_center=True,
    )
    if center_scaled is None:
        raise RuntimeError("Expected return_center=True to produce center_scaled.")

    # ---- save per-TCR embeddings ----
    region_key = make_region_key(assess_mode, alignment_context, regions)
    for i, (Zg_i, Zp_i, outdir_i) in enumerate(zip(Zg_list, Zp_list, per_tcr_outdirs)):
        Zg_sc = (Zg_i * k_scale) - center_scaled
        Zp_sc = (Zp_i * k_scale) - center_scaled

        os.makedirs(outdir_i, exist_ok=True)
        np.savez_compressed(os.path.join(outdir_i, "embeddings_scaled_centered.npz"), Zg=Zg_sc, Zp=Zp_sc)

        update_global_range(global_ranges, region_key, Zg_sc, Zp_sc)

        # quick sanity plot + PMF in local range (optional)
        plot_pca(Zg_i, Zp_i, None, None, outfile=os.path.join(outdir_i, "pca_projection_raw.png"))
        plot_pca(Zg_sc, Zp_sc, None, None, outfile=os.path.join(outdir_i, "pca_projection_scaled_centered.png"))

    print(f"[OK] {regions} ({alignment_context})  k={k_scale:.4g}  corr={corr:.3f}")


def assess_pmf_all_tcrs(
    assess_mode: str,
    per_tcr_outdirs: List[str],
    regions: List[str],
    global_ranges: dict,
    alignment_context: str,
):
    """
    Run global-binned PMF for each TCR, using a shared global_range computed over all TCRs.
    """
    region_key = make_region_key(assess_mode, alignment_context, regions)
    if region_key not in global_ranges:
        print(f"[PMF] Missing global range for {region_key}, skipping.")
        return
    gr = global_ranges[region_key]
    global_range = (gr["xmin"], gr["xmax"], gr["ymin"], gr["ymax"])

    xbins = 50
    ybins = 50
    temperature = 300.0

    for outdir_i in per_tcr_outdirs:
        emb_path = os.path.join(outdir_i, "embeddings_scaled_centered.npz")
        if not os.path.exists(emb_path):
            continue
        data = np.load(emb_path)
        Zg = data["Zg"]
        Zp = data["Zp"]

        pmf_dir = os.path.join(outdir_i, "pmf_global_bins")
        os.makedirs(pmf_dir, exist_ok=True)
        oriol_analysis(
            xbins,
            ybins,
            Zg,
            Zp,
            temperature,
            name="",
            outfolder=pmf_dir,
            tv_gt=None,
            tv_pred=None,
            global_range=global_range,
        )


# -----------------------------
# Data loading
# -----------------------------
def load_all_pairs(
    model_folder: str,
    model_name: str,
    output_dir_all: str,
    GT_MD_FOLDER: str,
    linker: str,
) -> Tuple[List[Any], List[Any], List[str], List[str], List[str]]:
    """
    Returns:
      pred_pairs, gt_pairs, pred_pdbs, gt_pdbs, tcr_names_full
    """
    pred_pairs = []
    gt_pairs = []
    pred_pdbs = []
    gt_pdbs = []
    tcr_names_full = []

    for folder in sorted(os.listdir(model_folder)):
        full_path = os.path.join(model_folder, folder)
        if not os.path.isdir(full_path):
            continue

        tcr_name_full = folder
        tcr_name = tcr_name_full.split("_")[0]

        out_tcr = os.path.join(output_dir_all, tcr_name_full)
        os.makedirs(out_tcr, exist_ok=True)

        pred_xtc = os.path.join(out_tcr, "pred.xtc")
        pred_pdb = os.path.join(out_tcr, "pred.pdb")

        dig_output_dir = os.path.join(model_folder, tcr_name_full, model_name)
        pred_xtc, pred_pdb = process_output(dig_output_dir, linker, pred_xtc, pred_pdb)

        gt_pdb = f"{GT_MD_FOLDER}/{tcr_name}/{tcr_name}.pdb"
        gt_xtc = f"{GT_MD_FOLDER}/{tcr_name}/{tcr_name}_Prod.xtc"
        if not (os.path.exists(gt_pdb) and os.path.exists(gt_xtc)):
            gt_xtc = f"{GT_MD_FOLDER}/{tcr_name}/{tcr_name}.xtc"
            if not (os.path.exists(gt_pdb) and os.path.exists(gt_xtc)):
                print(f"[SKIP] Missing GT for {tcr_name}")
                continue

        gt = TCR(input_pdb=gt_pdb, traj_path=gt_xtc, contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
        gt_pair = gt.pairs[0]

        try:
            pred = TCR(input_pdb=pred_pdb, traj_path=pred_xtc, contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
            pred_pair = pred.pairs[0]
        except Exception:
            pred = TCR(input_pdb=pred_pdb, traj_path=None, contact_cutoff=5.0, min_contacts=50, legacy_anarci=True)
            pred_pair = pred.pairs[0]
            pred_pair.attach_trajectory(pred_xtc, region_names=None, atom_names={"CA", "C", "N"})

        gt_pairs.append(gt_pair)
        pred_pairs.append(pred_pair)
        gt_pdbs.append(gt_pdb)
        pred_pdbs.append(pred_pdb)
        tcr_names_full.append(tcr_name_full)

    return pred_pairs, gt_pairs, pred_pdbs, gt_pdbs, tcr_names_full


# -----------------------------
# Main experiment
# -----------------------------
def run_shared_space_experiment(
    model_folder: str,
    model_name: str,
    config_path: str,
    output_dir_all: str,
    GT_MD_FOLDER: str,
    linker: str,
    fit_on: str = "concat",
):
    with open(config_path, "r") as f:
        region_metric_config = yaml.safe_load(f)

    pred_pairs, gt_pairs, pred_pdbs, gt_pdbs, tcr_names_full = load_all_pairs(
        model_folder=model_folder,
        model_name=model_name,
        output_dir_all=output_dir_all,
        GT_MD_FOLDER=GT_MD_FOLDER,
        linker=linker,
    )

    if len(pred_pairs) == 0:
        raise RuntimeError("No pairs loaded.")

    pred_tv_list = [p.traj for p in pred_pairs]
    gt_tv_list = [p.traj for p in gt_pairs]

    global_ranges: Dict[str, Any] = {}

    # Regions you want to assess
    base_regions = ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]

    # --- 1) Align-all on each CDR itself, then assess that same CDR in the shared frame ---
    for region in base_regions:
        assess_mode = region_metric_config.get(region, {}).get(region, "all")
        if assess_mode == "all":
            # You *can* support this, but global-space evaluation is usually per explicit mode
            print(f"[WARN] region={region} assess_mode=all; skipping in shared-space run.")
            continue

        align_out = os.path.join(output_dir_all, "_global_align", region)
        pred_aligned, gt_aligned, _ = align_all_pairs_global(
            pred_tv_list, gt_tv_list, pred_pdbs, gt_pdbs,
            region_names=[region],
            atom_names={"CA","C","N"},
            outdir=align_out,
        )

        per_tcr_outdirs = [os.path.join(output_dir_all, t, region, f"{assess_mode}") for t in tcr_names_full]
        assess_collect_all_tcr_space(
            assess_mode=assess_mode,
            gt_tv_aligned_list=gt_aligned,
            pred_tv_aligned_list=pred_aligned,
            per_tcr_outdirs=per_tcr_outdirs,
            regions=[region],
            fit_on=fit_on,
            global_ranges=global_ranges,
            alignment_context="region",
        )

        assess_pmf_all_tcrs(
            assess_mode=assess_mode,
            per_tcr_outdirs=per_tcr_outdirs,
            regions=[region],
            global_ranges=global_ranges,
            alignment_context="region",
        )

    # --- 2) Align-all on A_variable / B_variable, then assess subregions in that shared frame ---
    for domain in ["A_variable", "B_variable"]:
        align_out = os.path.join(output_dir_all, "_global_align", domain)
        pred_aligned, gt_aligned, _ = align_all_pairs_global(
            pred_tv_list, gt_tv_list, pred_pdbs, gt_pdbs,
            region_names=[domain],
            atom_names={"CA","C","N"},
            outdir=align_out,
        )

        # assess each CDR in that aligned frame
        for region in base_regions + [domain]:
            assess_mode = region_metric_config.get(region, {}).get(domain, "all")
            if assess_mode == "all":
                print(f"[WARN] domain={domain} region={region} assess_mode=all; skipping in shared-space run.")
                continue

            per_tcr_outdirs = [os.path.join(output_dir_all, t, domain, region, f"{assess_mode}") for t in tcr_names_full]
            assess_collect_all_tcr_space(
                assess_mode=assess_mode,
                gt_tv_aligned_list=gt_aligned,
                pred_tv_aligned_list=pred_aligned,
                per_tcr_outdirs=per_tcr_outdirs,
                regions=[region],
                fit_on=fit_on,
                global_ranges=global_ranges,
                alignment_context=domain,
            )
            assess_pmf_all_tcrs(
                assess_mode=assess_mode,
                per_tcr_outdirs=per_tcr_outdirs,
                regions=[region],
                global_ranges=global_ranges,
                alignment_context=domain,
            )

        # combined regions (example)
        combined = [
            ["A_CDR1","A_CDR2","A_CDR3"],
            ["B_CDR1","B_CDR2","B_CDR3"],
            ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"],
        ]
        for region_list in combined:
            label = "+".join(region_list)
            assess_mode = region_metric_config.get(label, {}).get(domain, "all")
            if assess_mode == "all":
                continue

            per_tcr_outdirs = [os.path.join(output_dir_all, t, domain, label, f"{assess_mode}") for t in tcr_names_full]
            assess_collect_all_tcr_space(
                assess_mode=assess_mode,
                gt_tv_aligned_list=gt_aligned,
                pred_tv_aligned_list=pred_aligned,
                per_tcr_outdirs=per_tcr_outdirs,
                regions=region_list,
                fit_on=fit_on,
                global_ranges=global_ranges,
                alignment_context=domain,
            )
            assess_pmf_all_tcrs(
                assess_mode=assess_mode,
                per_tcr_outdirs=per_tcr_outdirs,
                regions=region_list,
                global_ranges=global_ranges,
                alignment_context=domain,
            )

    # save global ranges for debugging / reuse
    np.save(os.path.join(output_dir_all, "global_ranges.npy"), global_ranges)
    print("[DONE] Saved global_ranges.npy")


if __name__ == "__main__":
    # defaults from your script
    model_folder = "/mnt/larry/lilian/DATA/VANILLA_DIG_OUTPUTS/CORY_PDBS/benchmarked_models/"
    GT_MD_FOLDER = "/mnt/larry/lilian/DATA/Cory_data/"
    model_name = "bioemu_filter"
    config_path = str(HERE / "config_assess_modes.yaml")
    output_dir_all = f"/mnt/dave/lilian/DIG_VARIATION_OUTPUTS/assessments_aaron/default/{model_name}"
    fit_on = "concat"

    linker = "" if model_name in ("AF3", "framedipt") else ("GGGGS" * 3)

    os.makedirs(output_dir_all, exist_ok=True)

    run_shared_space_experiment(
        model_folder=model_folder,
        model_name=model_name,
        config_path=config_path,
        output_dir_all=output_dir_all,
        GT_MD_FOLDER=GT_MD_FOLDER,
        linker=linker,
        fit_on=fit_on,
    )
