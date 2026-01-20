# score_alignment.py
import argparse, os
import numpy as np
import mdtraj as md




from TCR_TOOLS.aligners.aligning import (
    _keys_for_selection,
    _atom_indices_from_keys,
    _tm_per_frame,
    _batched_kabsch_to_single_target,
)
from TCR_TOOLS.__init__ import CDR_FR_RANGES, VARIABLE_RANGE
from TCR_TOOLS.core import io


def run(mov, ref, regions=None, atoms=None, superpose=False):

    ref_chains = sorted({getattr(c, "chain_id", str(i)) for i, c in enumerate(ref.topology.chains)})
    if set(ref_chains) >= {"A","B"}:
        ref_chain_map = {"alpha":"A","beta":"B"}
    elif set(ref_chains) >= {"G","D"}:
        ref_chain_map = {"alpha":"G","beta":"D"}
    mov_chains = sorted({getattr(c, "chain_id", None) for i, c in enumerate(mov.topology.chains)})
    # assume same mapping letters exist in mov topology; otherwise set explicitly
    if set(mov_chains) >= {"A","B"}:
        mov_chain_map = {"alpha":"A","beta":"B"}
    elif set(mov_chains) >= {"G","D"}:
        mov_chain_map = {"alpha":"G","beta":"D"}
    # Build common selections

    keys_ref = _keys_for_selection(ref.topology, ref_chain_map, regions, atom_names=atoms)
    keys_mov = _keys_for_selection(mov.topology, mov_chain_map, regions, atom_names=atoms)
    print(f"REF selected atoms: {keys_ref}")
    print(f"MOV selected atoms: {keys_mov}")
    common = sorted(set(keys_ref) & set(keys_mov), key=lambda k: (k[0], k[1], k[2]))
    if len(common) < 3:
        raise SystemExit(f"Too few common atoms ({len(common)}) for scoring. Check regions/atoms.")

    idx_ref = _atom_indices_from_keys(ref.topology, common)
    idx_mov = _atom_indices_from_keys(mov.topology, common)
    k = len(idx_ref)
    n = mov.n_frames

    # Coordinates (mdtraj uses nm internally)
    P = mov.mdtraj.xyz[:, idx_mov, :]          # (n,k,3)
    Q0 = ref.xyz[0, idx_ref, :]              # (k,3)
    k = P.shape[1]

    if superpose:
        # Fit P(t) -> Q0 on the *selected atoms* (batched Kabsch), then compute RMSD/TM from that P_aln
        R, t = _batched_kabsch_to_single_target(P, Q0)  # R:(n,3,3) t:(n,3)
        P_use = np.einsum("nij,nkj->nki", R, P) + t[:, None, :]
    else:
        # No refit: use P as-is in the current coordinate frame
        P_use = P

    diff = P_use - Q0[None, :, :]
    rmsd_nm = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))  # (n,)
    rmsd_A = rmsd_nm * 10.0

    dist_A = np.linalg.norm(diff, axis=2) * 10.0  # (n,k)
    tm = _tm_per_frame(dist_A, L=k)               # (n,)

    return mov.mdtraj.n_frames, k, rmsd_A, tm


def rmsd_tm(mov, ref_pair,regions=None, atoms=None, superpose=False):
    ref= io.mdtraj_from_biopython_path(ref_pair.full_structure)
    n, k, rmsd_A, tm=run(mov, ref, regions=regions, atoms=atoms, superpose=superpose)
    print(f"Frames scored: {n}, mapped_pairs: {k}")
    print(f"RMSD_A  mean={np.mean(rmsd_A):.3f}  median={np.median(rmsd_A):.3f}  p95={np.percentile(rmsd_A,95):.3f}  max={np.max(rmsd_A):.3f}")
    print(f"TMscore mean={np.mean(tm):.3f}  median={np.median(tm):.3f}  p95={np.percentile(tm,95):.3f}  min={np.min(tm):.3f}")
    return n, k, rmsd_A, tm


def main(mov_xtc=None, mov_top=None, ref_pdb=None, regions=None, atoms=None):
    # Load aligned MOV and REF frame-0
    mov = md.load(mov_xtc, top=mov_top)
    ref = md.load(ref_pdb)
    n, k, rmsd_A, tm=run(mov, ref, regions=regions, atoms=atoms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mov_xtc", required=True)
    ap.add_argument("--mov_top", required=True)
    ap.add_argument("--ref_pdb", required=True)
    ap.add_argument("--regions", default=["A_variable","B_variable"])
    ap.add_argument("--atoms", default="CA", choices=["CA","all"])
    args = ap.parse_args()

    main(
        mov_xtc=args.mov_xtc,
        mov_top=args.mov_top,
        ref_pdb=args.ref_pdb,
        regions=args.regions,
        atoms={args.atoms}
    )
