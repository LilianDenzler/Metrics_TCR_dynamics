from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Set
import os
import re
import subprocess
import tempfile
import logging

import numpy as np
import mdtraj as md
from Bio.PDB import Structure as BPStructure, PDBIO

# project constants (IMGT ranges)
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE
from TCR_TOOLS.core.io import load_pdb, write_pdb

# --------------------------------------------------------------------------------------
# Selection helpers (pure)
# --------------------------------------------------------------------------------------

Site = Tuple[str, int, str, str]  # (chain_id, resSeq, insertionCode, atom_name)

def _intervals_by_chain_from_regions(
    regions: List[str],
    chain_map: Dict[str, str],
    cdr_fr_ranges: Dict[str, Tuple[int, int]] = CDR_FR_RANGES,
    variable_range: Tuple[int, int] = VARIABLE_RANGE,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Expand a list of region names into {chain_id: [(start,end), ...]} using the given chain_map.
    """
    out: Dict[str, List[Tuple[int, int]]] = {}
    for name in regions:
        if name in ("A_variable", "B_variable"):
            start, end = variable_range
            cid = chain_map["alpha"] if name.startswith("A_") else chain_map["beta"]
        else:
            if name not in cdr_fr_ranges:
                raise KeyError(f"Unknown region: {name}")
            start, end = cdr_fr_ranges[name]
            cid = chain_map["alpha"] if name.startswith("A_") else chain_map["beta"]
        out.setdefault(cid, []).append((start, end))
    return out

def _alt_priority(alt: str) -> int:
    # lower is better
    if not alt or alt.isspace(): return 0
    if alt.upper() == "A":       return 1
    return 2

def _keys_for_selection(
    top: md.Topology,
    chain_map: Dict[str, str],
    regions: List[str],
    atom_names: Optional[Set[str]] = None,
) -> List[Site]:
    intervals_by_chain = _intervals_by_chain_from_regions(regions, chain_map)
    print("Intervals by chain:", intervals_by_chain)
    best_for_site: Dict[Site, str] = {}  # site -> best altloc
    for atom in top.atoms:
        res = atom.residue
        if res is None or res.chain is None:
            continue
        chain_id = getattr(res.chain, "chain_id", None) or str(res.chain.index)
        if chain_id not in intervals_by_chain:
            continue
        rnum  = int(getattr(res, "resSeq", res.index))
        icode = getattr(res, "insertionCode", "") or ""
        alt   = getattr(atom, "altloc", "") or ""
        for (s, e) in intervals_by_chain[chain_id]:
            if s <= rnum <= e and (atom_names is None or atom.name in atom_names):
                site = (chain_id, rnum, icode, atom.name)
                if site not in best_for_site or _alt_priority(alt) < _alt_priority(best_for_site[site]):
                    best_for_site[site] = alt
                break
    keys = sorted(best_for_site.keys(), key=lambda k: (k[0], k[1], k[2], k[3]))
    return keys

def _atom_indices_from_keys(top: md.Topology, keys: List[Site]) -> List[int]:
    site_to_best_alt: Dict[Site, str] = {}
    site_to_best_idx: Dict[Site, int] = {}
    for atom in top.atoms:
        res = atom.residue
        chain_id = getattr(res.chain, "chain_id", None) or str(res.chain.index)
        rnum  = int(getattr(res, "resSeq", res.index))
        icode = getattr(res, "insertionCode", "") or ""
        alt   = getattr(atom, "altloc", "") or ""
        site  = (chain_id, rnum, icode, atom.name)
        if site not in site_to_best_alt or _alt_priority(alt) < _alt_priority(site_to_best_alt[site]):
            site_to_best_alt[site] = alt
            site_to_best_idx[site] = atom.index
    idxs = [site_to_best_idx[k] for k in keys if k in site_to_best_idx]
    # uniqueness guaranteed by construction
    return idxs



def _common_selection_indices(
    ref_top: md.Topology,
    ref_chain_map: Dict[str, str],
    mov_top: md.Topology,
    mov_chain_map: Dict[str, str],
    regions: List[str],
    atom_names: Optional[Set[str]] = None,
) -> Tuple[List[int], List[int], List[Tuple[str, int, str]]]:
    """
    Build lists of atom indices in the SAME order for reference and moving topologies,
    using the intersection of selection keys.
    """
    keys_ref = _keys_for_selection(ref_top, ref_chain_map, regions, atom_names)
    keys_mov = _keys_for_selection(mov_top, mov_chain_map, regions, atom_names)

    common = sorted(set(keys_ref) & set(keys_mov), key=lambda k: (k[0], k[1], k[2]))
    if len(common) < 3:
        raise ValueError(
            f"Not enough common atoms to align (found {len(common)}). "
            f"Try atom_names={{'CA'}} or widen the regions."
        )

    idx_ref = _atom_indices_from_keys(ref_top, common)
    idx_mov = _atom_indices_from_keys(mov_top, common)
    return idx_ref, idx_mov, common


def _ca_indices_for_regions(tv, regions: List[str]) -> List[int]:
    """
    CA-only indices using an existing TrajectoryView's region machinery.
    """
    return tv._region_atom_indices(regions, CDR_FR_RANGES, VARIABLE_RANGE, atom_names={"CA"})


def _write_frame0_subset_pdb(traj: md.Trajectory, atom_indices: List[int], out_path: str) -> None:
    """
    Write a 1-frame PDB containing only the provided atom indices (frame 0).
    """
    traj[0].atom_slice(atom_indices).save_pdb(out_path)


# --------------------------------------------------------------------------------------
# TM-align runner + parser (pure)
# --------------------------------------------------------------------------------------
def run_tmalign(pdb_mov: str, pdb_ref: str, tmalign_bin: str = "TMalign") -> str:
    proc = subprocess.run(
        [tmalign_bin, pdb_mov, pdb_ref],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    return proc.stdout


_alignment_block_re = re.compile(r"^[A-Za-z\-]+$")

def _tmalign_chain_mapping(
    ref_traj: md.Trajectory, mov_traj: md.Trajectory,
    ref_top: md.Topology, mov_top: md.Topology,
    ref_chain_map: Dict[str,str], mov_chain_map: Dict[str,str],
    which: str,  # "alpha" or "beta"
    regions: List[str], tmalign_bin: str = "TMalign"
) -> Tuple[List[int], List[int], float]:
    # pick regions for a single chain only
    if which == "alpha":
        chain_regions = [r for r in regions if r.startswith("A_")]
    else:
        chain_regions = [r for r in regions if r.startswith("B_")]
    if not chain_regions:
        raise ValueError(f"No regions for chain '{which}' in {regions}.")

    # CA-only indices for that chain
    keys_ref = _keys_for_selection(ref_top, ref_chain_map, chain_regions, atom_names={"CA"})
    idx_ref_ca = _atom_indices_from_keys(ref_top, keys_ref)
    keys_mov = _keys_for_selection(mov_top, mov_chain_map, chain_regions, atom_names={"CA"})
    idx_mov_ca = _atom_indices_from_keys(mov_top, keys_mov)

    if len(idx_ref_ca) < 3 or len(idx_mov_ca) < 3:
        raise ValueError(f"Too few CA for chain {which}")

    with tempfile.TemporaryDirectory() as td:
        mov_pdb = os.path.join(td, "mov_ca.pdb")
        ref_pdb = os.path.join(td, "ref_ca.pdb")
        _write_frame0_subset_pdb(mov_traj, idx_mov_ca, mov_pdb)
        _write_frame0_subset_pdb(ref_traj, idx_ref_ca, ref_pdb)
        out = run_tmalign(mov_pdb, ref_pdb, tmalign_bin=tmalign_bin)

    pairs = parse_tmalign_map(out, len_mov=len(idx_mov_ca), len_ref=len(idx_ref_ca))
    idx_mov = [idx_mov_ca[i] for (i, j) in pairs]
    idx_ref = [idx_ref_ca[j] for (i, j) in pairs]

    # crude TM-score from TM-align stdout (optional)
    m = re.search(r"TM-score=\s*([0-9.]+)", out)
    tm = float(m.group(1)) if m else float("nan")
    return idx_ref, idx_mov, tm

_alignment_seq_re = re.compile(r"^[A-Za-z\-]+$")
_alignment_mid_re = re.compile(r"^[ :\.]+$")


def parse_tmalign_map(stdout: str, len_mov: int, len_ref: int) -> List[Tuple[int, int]]:
    lines = [ln.rstrip() for ln in stdout.splitlines()]
    pairs: List[Tuple[int, int]] = []
    # cumulative positions across blocks
    pos1 = pos2 = 0
    i = 0
    while i + 2 < len(lines):
        l1, l2, l3 = lines[i], lines[i+1], lines[i+2]
        if _alignment_seq_re.match(l1) and _alignment_mid_re.match(l2) and _alignment_seq_re.match(l3):
            # walk this block and ADVANCE cumulative positions
            for c1, c2 in zip(l1, l3):
                g1 = (c1 == "-"); g2 = (c2 == "-")
                if not g1 and not g2:
                    pairs.append((pos1, pos2))
                if not g1: pos1 += 1
                if not g2: pos2 += 1
            i += 3
            continue
        i += 1
    pairs = [(i1, i2) for (i1, i2) in pairs if 0 <= i1 < len_mov and 0 <= i2 < len_ref]
    if len(pairs) < 3:
        raise ValueError(f"TM-align mapping too small ({len(pairs)}).")
    return pairs


# --------------------------------------------------------------------------------------
# Geometry: Kabsch (batched) & optional TM-score helper
# --------------------------------------------------------------------------------------
def _batched_kabsch_to_single_target(P: np.ndarray, Q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kabsch for many frames P_n -> one fixed target Q0.
    P:  (n_frames, k_atoms, 3)
    Q0: (k_atoms, 3)
    returns R: (n_frames, 3, 3), t: (n_frames, 3)
    """
    Pc = P.mean(axis=1)            # (n,3)
    Qc0 = Q0.mean(axis=0)          # (3,)
    X = P - Pc[:, None, :]         # (n,k,3)
    Y0 = Q0 - Qc0[None, :]         # (k,3)

    C = np.matmul(X.transpose(0, 2, 1), Y0)  # (n,3,3)
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt
    # Ensure proper rotation (det=+1)
    detR = np.linalg.det(R)
    neg = detR < 0
    if np.any(neg):
        U[neg, :, 2] *= -1.0
        R[neg] = U[neg] @ Vt[neg]
    t = Qc0[None, :] - np.einsum("nij,nj->ni", R, Pc)
    return R, t


def _tm_per_frame(dist_A: np.ndarray, L: int) -> np.ndarray:
    """
    dist_A : (n_frames, k) distances in Å between mapped atom pairs
    L      : integer length used by TM-score (use k = mapped pairs)
    returns: (n_frames,) TM-score per frame
    """
    if L <= 0:
        raise ValueError("TM-score: L must be > 0")
    d0 = 1.24 * np.cbrt(L - 15.0) - 1.8
    d0 = float(max(d0, 0.5))
    if not np.isfinite(d0) or d0 <= 0.0:
        d0 = 0.5
    return np.mean(1.0 / (1.0 + (dist_A / d0) ** 2), axis=1)

def _print_mapping_preview(ref_top: md.Topology, mov_top: md.Topology,
                           idx_ref: List[int], idx_mov: List[int], max_rows: int = 25):
    def row(top, i):
        a = list(top.atoms)[i]
        r = a.residue
        chain_id = getattr(r.chain, "chain_id", None) or str(r.chain.index)
        resseq   = int(getattr(r, "resSeq", r.index))
        icode    = getattr(r, "insertionCode", "") or ""
        aname    = a.name
        return chain_id, resseq, icode, aname
    print("Mapping preview (mov -> ref):")
    for k, (im, ir) in enumerate(zip(idx_mov, idx_ref)):
        if k >= max_rows:
            print(f"... ({len(idx_mov)-max_rows} more)")
            break
        m = row(mov_top, im)
        r = row(ref_top, ir)
        print(f"{k:3d}: MOV {m[0]} {m[1]}{m[2]} {m[3]:>4s}  -->  REF {r[0]} {r[1]}{r[2]} {r[3]:>4s}")

# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def _mdtraj_from_pair_full_structure(pv) -> md.Trajectory:
    """
    Build a 1-frame mdtraj.Trajectory from pv.full_structure (Biopython).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    tmp.close()
    try:
        write_pdb(tmp.name, pv.full_structure)
        tr = md.load(tmp.name)  # 1 frame
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return tr


# --------------------------------------------------------------------------------------
# High-level aligners
# --------------------------------------------------------------------------------------
def align_traj_same_tcr_fast(pv_ref, pv_mov, regions: List[str], atom_names: Optional[Set[str]] = None):
    """
    Aligns:
      - reference trajectory: all frames -> reference frame 0 (on selection)
      - moving trajectory:   all frames -> reference frame 0 (on selection)

    Returns (ref_aligned, mov_aligned, summary) as (mdtraj.Trajectory, mdtraj.Trajectory, dict)
    """
    if pv_ref.tcr_owner is None or pv_ref.tcr_owner._traj is None:
        raise RuntimeError("Reference TCR has no attached trajectory.")
    if pv_mov.tcr_owner is None or pv_mov.tcr_owner._traj is None:
        raise RuntimeError("Moving TCR has no attached trajectory.")

    ref_traj = pv_ref.traj._traj
    mov_traj = pv_mov.traj._traj

    # indices in each topology (intersection by keys)
    idx_ref, idx_mov, _ = _common_selection_indices(
        ref_top=pv_ref.traj._top,
        ref_chain_map=pv_ref.traj._chain_map,
        mov_top=pv_mov.traj._top,
        mov_chain_map=pv_mov.traj._chain_map,
        regions=regions,
        atom_names=atom_names,
    )

    n = min(ref_traj.n_frames, mov_traj.n_frames)

    # Kabsch: MOV -> REF[0]
    P_mov = mov_traj.xyz[:n, idx_mov, :]    # (n,k,3)
    Q0 = ref_traj.xyz[0, idx_ref, :]        # (k,3)
    Rm, tm = _batched_kabsch_to_single_target(P_mov, Q0)

    mov_aligned = md.Trajectory(
        xyz=mov_traj.xyz[:n].copy(),
        topology=mov_traj.topology,
        time=None, unitcell_lengths=None, unitcell_angles=None,
    )
    mov_aligned.xyz[:] = np.einsum("nij,naj->nai", Rm, mov_aligned.xyz) + tm[:, None, :]

    # Kabsch: REF -> REF[0] (optional self-centering for numerical consistency)
    P_ref = ref_traj.xyz[:n, idx_ref, :]
    Rr, tr = _batched_kabsch_to_single_target(P_ref, Q0)
    ref_aligned = md.Trajectory(
        xyz=ref_traj.xyz[:n].copy(),
        topology=ref_traj.topology,
        time=None, unitcell_lengths=None, unitcell_angles=None,
    )
    ref_aligned.xyz[:] = np.einsum("nij,naj->nai", Rr, ref_aligned.xyz) + tr[:, None, :]

    # RMSD summary on selection (mdtraj RMSD in nm → convert to Å for reporting)
    rmsd_nm = md.rmsd(mov_aligned[:n], ref_aligned[:n], atom_indices=idx_mov, ref_atom_indices=idx_ref)
    rmsd_A = rmsd_nm * 10.0
    summary = {
        "frames_scored": int(n),
        "mapped_pairs": int(len(idx_ref)),
        "mean_rmsd_A": float(np.mean(rmsd_A)),
        "median_rmsd_A": float(np.median(rmsd_A)),
        "p95_rmsd_A": float(np.percentile(rmsd_A, 95)),
        "max_rmsd_A": float(np.max(rmsd_A)),
    }
    return ref_aligned, mov_aligned, summary


def align_traj_to_refstructure(
    pv_ref,
    pv_mov,
    regions: List[str],
    atom_names: Optional[Set[str]] = None,
    tmalign_step: bool = False,
    tmalign_bin: str = "TMalign",
) -> Tuple[md.Trajectory, Dict[str, float]]:
    """
    Align all frames of MOV trajectory to REF frame 0 (IMGT regions).
    pv_ref may or may not have a trajectory attached.

    Returns:
      mov_aligned (md.Trajectory), summary (dict)
    """
    if pv_mov.tcr_owner is None or pv_mov.tcr_owner._traj is None:
        raise RuntimeError("Moving TCR has no attached trajectory.")

    # Reference: trajectory if present; otherwise, 1-frame from full_structure
    if pv_ref.tcr_owner is not None and pv_ref.tcr_owner._traj is not None:
        ref_traj = pv_ref.traj._traj
        ref_top = pv_ref.traj._top
        ref_chain_map = pv_ref.traj._chain_map
    else:
        ref_traj = _mdtraj_from_pair_full_structure(pv_ref)
        ref_top = ref_traj.topology
        ref_chain_map = pv_ref.chain_map  # already the remapped A/B (or G/D) ids

    mov_traj = pv_mov.traj._traj
    mov_top = pv_mov.traj._top

    if tmalign_step:
        # do per-chain; pick the chain giving better TM-score (or choose explicitly)
        idx_ref_A, idx_mov_A, tmA = _tmalign_chain_mapping(ref_traj, mov_traj, ref_top, mov_top,
                                ref_chain_map, pv_mov.traj._chain_map, "alpha", regions, tmalign_bin)
        idx_ref_B, idx_mov_B, tmB = _tmalign_chain_mapping(ref_traj, mov_traj, ref_top, mov_top,
                                ref_chain_map, pv_mov.traj._chain_map, "beta", regions, tmalign_bin)

        # choose mapping (example: pick the better TM-score)
        if (tmA >= tmB) and len(idx_ref_A) >= 3:
            idx_ref, idx_mov, picked = idx_ref_A, idx_mov_A, "alpha"
        else:
            idx_ref, idx_mov, picked = idx_ref_B, idx_mov_B, "beta"
        print(f"TM-align mapping based on chain: {picked} (TM≈{max(tmA, tmB):.3f})")

        _print_mapping_preview(ref_top, mov_top, idx_ref, idx_mov, max_rows=20)

    else:
        idx_ref, idx_mov, _ = _common_selection_indices(
            ref_top=ref_top,
            ref_chain_map=ref_chain_map,
            mov_top=mov_top,
            mov_chain_map=pv_mov.traj._chain_map,
            regions=regions,
            atom_names=atom_names,
        )
    print("Number of indexes selected for alignment:" ,len(idx_ref), len(idx_mov))

    if len(idx_ref) < 3:
        raise ValueError(f"Selection too small for rigid fit (|idx_ref|={len(idx_ref)}).")

    n = min(ref_traj.n_frames, mov_traj.n_frames)

    # 1) MOV -> REF[0]
    P_mov = mov_traj.xyz[:n, idx_mov, :]
    Q0    = ref_traj.xyz[0, idx_ref, :]
    Rmov, tmov = _batched_kabsch_to_single_target(P_mov, Q0)

    mov_aligned = md.Trajectory(
        xyz=mov_traj.xyz[:n].copy(),
        topology=mov_traj.topology,
        time=None, unitcell_lengths=None, unitcell_angles=None,
    )
    mov_aligned.xyz[:] = np.einsum("nij,naj->nai", Rmov, mov_aligned.xyz) + tmov[:, None, :]

    P_after = mov_aligned.xyz[0, idx_mov, :]         # nm
    Q0      = ref_traj.xyz[0, idx_ref, :]            # nm
    errs_nm = np.linalg.norm(P_after - Q0, axis=1)
    print(f"[check] mapped CA median/95th/max error (Å): "
        f"{np.median(errs_nm)*10:.3f} / {np.percentile(errs_nm,95)*10:.3f} / {errs_nm.max()*10:.3f}")



    # 2) REF -> REF[0] (self-align for consistent overlay)
    P_ref = ref_traj.xyz[:n, idx_ref, :]
    Rref, tref = _batched_kabsch_to_single_target(P_ref, Q0)

    ref_self = md.Trajectory(
        xyz=ref_traj.xyz[:n].copy(),
        topology=ref_traj.topology,
        time=None, unitcell_lengths=None, unitcell_angles=None,
    )
    ref_self.xyz[:] = np.einsum("nij,naj->nai", Rref, ref_self.xyz) + tref[:, None, :]

    # 3) Score (Å) on the mapped set
    rmsd_nm = md.rmsd(mov_aligned, ref_self, atom_indices=idx_mov, ref_atom_indices=idx_ref)
    rmsd_A = rmsd_nm * 10.0

    summary = {
        "frames_scored": int(n),
        "mapped_pairs": int(len(idx_ref)),
        "mean_rmsd_A": float(np.mean(rmsd_A)),
        "median_rmsd_A": float(np.median(rmsd_A)),
        "p95_rmsd_A": float(np.percentile(rmsd_A, 95)),
        "max_rmsd_A": float(np.max(rmsd_A)),
    }
    return mov_aligned, summary


def align_traj_different_tcr_tmalign(
    pv_ref,
    pv_mov,
    regions: List[str],
    tmalign_bin: str = "TMalign",
) -> Tuple[md.Trajectory, md.Trajectory, Dict[str, float]]:
    """
    1) Build CA selections from both TCR pair views using IMGT regions.
    2) Write 1-frame PDBs of those CA selections (frame 0).
    3) Run TM-align once to get a residue mapping (CA positions).
    4) Self-align reference trajectory to frame 0 (mapped CA set).
    5) Batched Kabsch: align moving trajectory to the reference frame-by-frame using the mapped CA pairs.

    Returns (ref_aligned, mov_aligned, summary_scores).
    """
    if pv_ref.tcr_owner is None or pv_ref.tcr_owner._traj is None:
        raise RuntimeError("Reference TCR has no attached trajectory.")
    if pv_mov.tcr_owner is None or pv_mov.tcr_owner._traj is None:
        raise RuntimeError("Moving TCR has no attached trajectory.")

    ref_in = pv_ref.traj._traj
    mov_in = pv_mov.traj._traj

    # 1) CA selections
    idx_ref_ca = _ca_indices_for_regions(pv_ref.traj, regions)
    idx_mov_ca = _ca_indices_for_regions(pv_mov.traj, regions)

    # 2) TM-align mapping
    with tempfile.TemporaryDirectory() as td:
        mov_pdb = os.path.join(td, "mov_ca.pdb")
        ref_pdb = os.path.join(td, "ref_ca.pdb")
        _write_frame0_subset_pdb(mov_in, idx_mov_ca, mov_pdb)
        _write_frame0_subset_pdb(ref_in, idx_ref_ca, ref_pdb)
        out = run_tmalign(mov_pdb, ref_pdb, tmalign_bin=tmalign_bin)

    pairs = parse_tmalign_map(out, len_mov=len(idx_mov_ca), len_ref=len(idx_ref_ca))
    mapped_idx_mov = [idx_mov_ca[i] for (i, j) in pairs]
    mapped_idx_ref = [idx_ref_ca[j] for (i, j) in pairs]

    n = min(ref_in.n_frames, mov_in.n_frames)

    # 4) Self-align REF to its frame 0 (mapped CA)
    P_ref = ref_in.xyz[:n, mapped_idx_ref, :]
    Q0_ref = ref_in.xyz[0, mapped_idx_ref, :]
    Rr, tr = _batched_kabsch_to_single_target(P_ref, Q0_ref)
    ref_aligned = ref_in[:]
    ref_block = ref_aligned.xyz[:n]
    ref_block[:] = np.einsum("nij,naj->nai", Rr, ref_block) + tr[:, None, :]

    # 5) MOV -> REF[0] (mapped CA)
    P_mov = mov_in.xyz[:n, mapped_idx_mov, :]
    Q0 = ref_aligned.xyz[0, mapped_idx_ref, :]  # after ref self-align, but frame 0 unchanged by design
    Rm, tm = _batched_kabsch_to_single_target(P_mov, Q0)
    mov_aligned = mov_in[:]
    mov_block = mov_aligned.xyz[:n]
    mov_block[:] = np.einsum("nij,naj->nai", Rm, mov_block) + tm[:, None, :]

    # Scores (Å)
    rmsd_nm = md.rmsd(mov_aligned[:n], ref_aligned[:n], atom_indices=mapped_idx_mov, ref_atom_indices=mapped_idx_ref)
    rmsd_A = rmsd_nm * 10.0
    summary = {
        "frames_scored": int(n),
        "mapped_CA_pairs": int(len(mapped_idx_ref)),
        "mean_rmsd_A": float(np.mean(rmsd_A)),
        "median_rmsd_A": float(np.median(rmsd_A)),
        "p95_rmsd_A": float(np.percentile(rmsd_A, 95)),
        "max_rmsd_A": float(np.max(rmsd_A)),
    }
    return ref_aligned, mov_aligned, summary

def save_frames_to_new_tmp_dir(trajectory: md.Trajectory, prefix: str) -> (str, list):
    # 1. Create a new unique temporary directory in /tmp/
    output_dir = tempfile.mkdtemp(prefix=f"{prefix}_", dir="/tmp")
    logging.info(f"Saving {trajectory.n_frames} frames to new temp directory: {output_dir}")

    pdb_paths = []

    # 2. Determine padding for filenames
    n_frames = trajectory.n_frames
    padding = len(str(n_frames - 1))

    # 3. Iterate over each frame and save it
    for i in range(n_frames):
        frame = trajectory[i]
        filename = f"frame_{i:0{padding}}.pdb"
        output_path = os.path.join(output_dir, filename)

        frame.save_pdb(output_path)
        pdb_paths.append(output_path) # Store the full path

        if (i + 1) % 100 == 0 or (i + 1) == n_frames:
            logging.info(f"  ... Saved frame {i + 1} / {n_frames}")

    return output_dir, pdb_paths

def align_MD_same_TCR_profit(
    pv_ref,
    pv_mov,
    regions: List[str],
    atom_names: Optional[Set[str]] = None,
    inplace: bool = False,
    profit_bin: str = "/workspaces/Graphormer/ProFitV3.3/bin/profit",
) -> Tuple[md.Trajectory, md.Trajectory, Dict[str, float]]:
    '''
    Aligns:
      - reference trajectory: all frames -> reference frame 0 (on selection)
      - moving trajectory:   all frames -> reference frame 0 (on selection)

    Uses ProFit as the alignment engine.

    Returns (ref_aligned, mov_aligned, summary) as (mdtraj.Trajectory, mdtraj.Trajectory, dict)
    '''
    traj_ref=pv_ref.traj
    traj_mov=pv_mov.traj
    reftraj_idx, reftrajnames = traj_ref.domain_idx(region_names=regions, atom_names=atom_names,pass_names=True)
    movtraj_idx, movtrajnames = traj_mov.domain_idx(region_names=regions, atom_names=atom_names,pass_names=True)
    if len(reftrajnames) != len(movtrajnames):
        shared = [x for x in movtrajnames if x in reftrajnames]
        print(f"Warning: Different selection names between reference and moving trajecotries. Using shared names: {shared}")
    #need to write each frame to pdb files

    output_dir_ref, ref_pdb_paths = save_frames_to_new_tmp_dir(traj_ref._traj, "ref")
    output_dir_mov, mov_pdb_paths = save_frames_to_new_tmp_dir(traj_mov._traj, "mov")

    all_pdb_paths = ref_pdb_paths + mov_pdb_paths

    logging.info(f"Total PDB files created: {len(all_pdb_paths)}")

    # --- 3. Make the .txt file with all paths ---
    # We use NamedTemporaryFile to get a unique path in /tmp
    # 'delete=False' means the file will *not* be deleted when we close it.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir='/tmp', delete=False) as f:
        list_file_path = f.name
        logging.info(f"Creating path list file at: {list_file_path}")

        for path in all_pdb_paths:
            f.write(f"{path}\n")

    logging.info("Successfully wrote all paths to .txt file.")
    seq_dict_ref=pv_ref.cdr_fr_sequences()
    seq_dict_mov=pv_mov.cdr_fr_sequences()
    limit_lines=[]
    for region in regions:
        seq_ref=seq_dict_ref.get(region,"")
        seq_mov=seq_dict_mov.get(region,"")
        if seq_ref != seq_mov:
            print(f"Warning: Different sequences for region {region} between reference and moving trajectories.")
        if "A" in region:
            chain_id=traj_ref._chain_map["alpha"]
            full_chain=seq_dict_ref.get("full_A","")
            region_seq=seq_dict_ref.get(region,"")
            start_index = full_chain.find(region_seq)+1
            end_index = start_index + len(region_seq) - 1

        elif "B" in region:
            chain_id=traj_ref._chain_map["beta"]
            full_chain=seq_dict_ref.get("full_B","")
            region_seq=seq_dict_ref.get(region,"")
            start_index = full_chain.find(region_seq)+1
            end_index = start_index + len(region_seq) - 1
        limit_lines.append(f"LIMIT {chain_id}{str(start_index)} {chain_id}{str(end_index)}")

    #write prf file

    script_txt=f'''MULTI {list_file_path}
    iter 1.0
    {i+"/n" for i in limit_lines}
    MULTREF
    fit
    MWRITE
    WRITE REF file_to_write_ref.pdb
    '''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.prf', dir='/tmp', delete=False) as f:
        prf_file_path = f.name
        logging.info(f"Creating ProFit .prf file at: {prf_file_path}")
        f.write(script_txt)
    print(f"ProFit script:\n{script_txt}")
    #run bash command
    cmd = [profit_bin, prf_file_path]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )


