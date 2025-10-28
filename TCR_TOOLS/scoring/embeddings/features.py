import numpy as np
import mdtraj as md
from itertools import combinations
from typing import Iterable, List, Optional, Tuple

class TVWrap:
    def __init__(self, tv): self.tv = tv

    def indices(self, region_names: List[str], atom_names: Iterable[str]):
        return np.asarray(self.tv.domain_idx(region_names=region_names,
                                             atom_names=set(atom_names)), dtype=int)


    def coords(self, region_names: List[str], atom_names=("CA","C","N","O")) -> np.ndarray:
        idx = self.indices(region_names, atom_names)
        T = self.tv.mdtraj
        return T.xyz[:, idx, :].reshape(T.n_frames, -1)

    def ca_indices(self, region_names: List[str]) -> np.ndarray:
        idx = self.indices(region_names, {"CA"})
        # be strict that these are CA
        ca = [i for i in idx if self.tv.mdtraj.topology.atom(int(i)).name.strip().upper() == "CA"]
        if not ca: raise ValueError("No CA atoms found for selection.")
        return np.asarray(ca, dtype=int)

def features_ca_dist(tv, region_names: List[str], max_pairs: Optional[int]=None, rng: int=0) -> np.ndarray:
    wrap = TVWrap(tv)
    ca = wrap.ca_indices(region_names)
    pos_pairs = np.array(list(combinations(range(len(ca)), 2)), dtype=int)
    if max_pairs and len(pos_pairs) > max_pairs:
        rs = np.random.default_rng(rng)
        pos_pairs = pos_pairs[rs.choice(len(pos_pairs), size=max_pairs, replace=False)]
    atom_pairs = np.stack([ca[pos_pairs[:,0]], ca[pos_pairs[:,1]]], axis=1)
    D = md.compute_distances(tv.mdtraj, atom_pairs, periodic=False) * 10.0  # Å
    return D

def _compute_mdtraj_dihed(traj: md.Trajectory, kind: str):
    k = kind.lower()
    if k == "phi":   idx, ang = md.compute_phi(traj)
    elif k == "psi": idx, ang = md.compute_psi(traj)
    elif k == "omega": idx, ang = md.compute_omega(traj)
    elif k in ("chi1","chi2","chi3","chi4","chi5"):
        order = int(k[-1]); idx, ang = md.compute_chi(traj, order=order)
    else:
        raise ValueError(f"Unknown dihedral kind: {kind}")
    return idx, ang  # radians

def features_dihedrals(tv, region_names: List[str], dihedrals=("phi","psi"), encode="sincos") -> np.ndarray:
    wrap = TVWrap(tv); traj = tv.mdtraj
    ca_idx = set(wrap.ca_indices(region_names).tolist())
    allowed_res = {traj.topology.atom(int(i)).residue.index for i in ca_idx}
    parts = []
    for k in dihedrals:
        idx, ang = _compute_mdtraj_dihed(traj, k)
        centers = [traj.topology.atom(int(row[1])).residue.index for row in idx]
        keep = np.array([i for i, r in enumerate(centers) if r in allowed_res], dtype=int)
        if keep.size == 0: continue
        A = ang[:, keep]  # radians
        if encode == "sincos":
            parts += [np.cos(A), np.sin(A)]
        elif encode == "radians":
            parts += [A]
        else:
            raise ValueError("encode must be 'sincos' or 'radians'")
    if not parts:
        raise ValueError("No dihedral features selected.")
    return np.hstack(parts)
