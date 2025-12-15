from __future__ import annotations

import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")

import os
import re
from copy import deepcopy
from typing import Dict, Tuple, Union, List, Any, Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, Superimposer
from Bio.SeqUtils import seq1
from TCR_TOOLS.__init__ import CDR_FR_RANGES, A_FR_ALIGN,B_FR_ALIGN

# ============================================================
# Residue spec parsing / normalization
# ============================================================

def _parse_res_id(x):
    """
    Accept:
      - int (e.g., 105) => (105, ' ')
      - tuple (resseq, icode) => (resseq, icode)
      - string:
          "105"   => (105, ' ')
          "105A"  => (105, 'A')
          "105 "  => (105, ' ')
    """
    if isinstance(x, tuple) and len(x) == 2:
        resseq = int(x[0])
        icode = str(x[1]) if x[1] is not None else " "
        if icode == "":
            icode = " "
        return (resseq, icode)

    if isinstance(x, int):
        return (x, " ")

    if isinstance(x, str):
        s = x.strip()
        m = re.match(r"^(\d+)\s*([A-Za-z]?)$", s)
        if not m:
            raise ValueError(f"Unrecognized residue id: {x!r} (use 105, '105A', or (105,'A'))")
        resseq = int(m.group(1))
        icode = m.group(2) if m.group(2) else " "
        return (resseq, icode)

    raise ValueError(f"Unsupported residue id type: {type(x)}")


def _normalize_residue_spec(spec) -> set[Tuple[int, str]]:
    """
    Normalize residue specs into a set of (resseq, icode).

    spec can be:
      - list/tuple/set of residue ids: [105, "106A", (107,' ')]
      - dict with keys:
          {"ids": [...]} OR {"range": (start, end)} OR {"ranges": [(s,e),...]} OR combos
        where start/end can be int, "105A", or (105,'A')
    """
    out = set()

    if spec is None:
        return out

    if isinstance(spec, (list, tuple, set)):
        for x in spec:
            out.add(_parse_res_id(x))
        return out

    if isinstance(spec, dict):
        if "ids" in spec and spec["ids"] is not None:
            for x in spec["ids"]:
                out.add(_parse_res_id(x))

        if "range" in spec and spec["range"] is not None:
            s, e = spec["range"]
            s_res, s_ic = _parse_res_id(s)
            e_res, e_ic = _parse_res_id(e)
            if s_ic != " " or e_ic != " ":
                raise ValueError("For 'range', use plain residue numbers without insertion codes.")
            for r in range(min(s_res, e_res), max(s_res, e_res) + 1):
                out.add((r, " "))

        if "ranges" in spec and spec["ranges"] is not None:
            for (s, e) in spec["ranges"]:
                s_res, s_ic = _parse_res_id(s)
                e_res, e_ic = _parse_res_id(e)
                if s_ic != " " or e_ic != " ":
                    raise ValueError("For 'ranges', use plain residue numbers without insertion codes.")
                for r in range(min(s_res, e_res), max(s_res, e_res) + 1):
                    out.add((r, " "))

        return out

    raise ValueError(f"Unsupported residue spec type: {type(spec)}")


def _pymol_resi_token(resseq: int, icode: str) -> str:
    icode = (icode or " ").strip()
    return f"{resseq}{icode}" if icode else f"{resseq}"


def selection_to_pymol(sel: Dict[str, object]) -> str:
    """
    Convert a selection dict into a PyMOL selection string that is valid inside:
        (obj_name and (<returned_string>))
    """
    parts = []
    for chain_id, spec in sel.items():
        if spec is None or spec == [] or spec == {}:
            parts.append(f"(chain {chain_id})")
            continue

        if isinstance(spec, (list, tuple, set)):
            ids = []
            for x in spec:
                r, ic = _parse_res_id(x)
                ids.append(_pymol_resi_token(r, ic))
            if len(ids) == 0:
                parts.append(f"(chain {chain_id})")
            else:
                parts.append(f"(chain {chain_id} and resi {'+'.join(ids)})")
            continue

        if isinstance(spec, dict):
            resi_chunks = []

            if "ids" in spec and spec["ids"]:
                ids = []
                for x in spec["ids"]:
                    r, ic = _parse_res_id(x)
                    ids.append(_pymol_resi_token(r, ic))
                if ids:
                    resi_chunks.append("+".join(ids))

            if "range" in spec and spec["range"] is not None:
                s, e = spec["range"]
                s_res, s_ic = _parse_res_id(s)
                e_res, e_ic = _parse_res_id(e)
                if s_ic != " " or e_ic != " ":
                    raise ValueError("PyMOL 'range' selection here assumes no insertion codes.")
                resi_chunks.append(f"{min(s_res,e_res)}-{max(s_res,e_res)}")

            if "ranges" in spec and spec["ranges"] is not None:
                for (s, e) in spec["ranges"]:
                    s_res, s_ic = _parse_res_id(s)
                    e_res, e_ic = _parse_res_id(e)
                    if s_ic != " " or e_ic != " ":
                        raise ValueError("PyMOL 'ranges' selection here assumes no insertion codes.")
                    resi_chunks.append(f"{min(s_res,e_res)}-{max(s_res,e_res)}")

            if len(resi_chunks) == 0:
                parts.append(f"(chain {chain_id})")
            else:
                resi_expr = "+".join(resi_chunks)
                parts.append(f"(chain {chain_id} and resi {resi_expr})")
            continue

        raise ValueError(f"Unsupported selection spec for chain {chain_id}: {type(spec)}")

    if not parts:
        return "all"
    return " or ".join(parts)


# ============================================================
# Atom-mode normalization
# ============================================================

AtomMode = Union[str, List[str]]

def _normalize_atom_mode(atom_mode: AtomMode) -> Union[str, List[str]]:
    """
    Returns either:
      - "all" (special)
      - list of atom names (uppercase)
    Allowed inputs:
      - "CA" or ["CA"]
      - ["N","CA","C","O"] etc.
      - "all"
      - "heavy" (alias for "all" non-H)
    """
    if isinstance(atom_mode, str):
        s = atom_mode.strip()
        if s.lower() in ("all", "heavy"):
            return "all"
        return [s.upper()]

    if isinstance(atom_mode, (list, tuple)):
        names = [str(x).strip().upper() for x in atom_mode if str(x).strip()]
        if len(names) == 0:
            raise ValueError("atom_mode list is empty; pass e.g. ['CA'] or 'all'.")
        return names

    raise ValueError(f"Unsupported atom_mode type: {type(atom_mode)}")


# ============================================================
# Core atom matching / superposition / scoring
# ============================================================

def _collect_matched_atoms(
    struct_moving,
    struct_fixed,
    chain_to_resids: Dict[str, object],
    atom_mode: AtomMode = "CA",
    include_hetero: bool = False,
):
    """
    Return (fixed_atoms, moving_atoms) lists matched by:
      ((chain_id, resseq, icode), atom_name)
    atom_mode:
      - "all" => all non-H atoms in selection
      - list of atom names => only those atoms
    """
    atom_mode_n = _normalize_atom_mode(atom_mode)

    chain_to_set = {cid: _normalize_residue_spec(spec) for cid, spec in chain_to_resids.items()}

    def residue_selected(chain_id: str, residue) -> bool:
        if chain_id not in chain_to_set:
            return False
        wanted = chain_to_set[chain_id]
        hetflag, resseq, icode = residue.id
        if (not include_hetero) and (hetflag != " "):
            return False
        if len(wanted) == 0:
            return True  # empty spec => whole chain
        return (int(resseq), (icode if icode else " ")) in wanted

    def iter_atoms(structure):
        model = next(structure.get_models())
        for chain in model:
            cid = chain.id
            if cid not in chain_to_set:
                continue
            for res in chain:
                if not residue_selected(cid, res):
                    continue
                hetflag, resseq, icode = res.id
                rk = (cid, int(resseq), (icode if icode else " "))

                if atom_mode_n == "all":
                    for a in res.get_atoms():
                        # exclude hydrogens; Biopython sometimes has element empty, so fallback on name
                        el = getattr(a, "element", None)
                        if el == "H":
                            continue
                        if el is None and a.get_name().upper().startswith("H"):
                            continue
                        yield (rk, a.get_name()), a
                else:
                    # explicit atom list
                    for aname in atom_mode_n:
                        if aname in res:
                            yield (rk, aname), res[aname]

    moving_map = dict(iter_atoms(struct_moving))
    fixed_map  = dict(iter_atoms(struct_fixed))

    keys = sorted(set(moving_map.keys()) & set(fixed_map.keys()))
    if len(keys) < 3:
        raise RuntimeError(
            f"Not enough matched atoms for selection (matched={len(keys)}). "
            f"Check chain IDs / residue numbering / atom_mode. Selection: {chain_to_resids}, atom_mode={atom_mode}"
        )

    fixed_atoms  = [fixed_map[k]  for k in keys]
    moving_atoms = [moving_map[k] for k in keys]
    return fixed_atoms, moving_atoms


def _write_pdb(path: str, structure) -> None:
    io = PDBIO()
    io.set_structure(structure)
    io.save(path)


def _coords_from_atoms(atoms) -> np.ndarray:
    return np.asarray([a.get_coord() for a in atoms], dtype=float)


def _rmsd_from_coords(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _tm_score_fixed(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """
    TM-score computed on already-superposed coordinates with 1-to-1 correspondence.
    """
    coords_a = np.asarray(coords_a, float)
    coords_b = np.asarray(coords_b, float)
    assert coords_a.shape == coords_b.shape
    L = coords_a.shape[0]
    if L == 0:
        return float("nan")

    d = np.linalg.norm(coords_a - coords_b, axis=1)

    if L < 15:
        d0 = 0.5
    else:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)

    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def _atoms_to_coords_and_seq(atoms):
    coords = _coords_from_atoms(atoms)
    # NOTE: tmtools requires sequences; for non-CA atoms, this “sequence” still corresponds 1:1
    # to the residue of each atom. That’s not strictly what TM-align expects, so prefer tm_mode="fixed".
    seq = []
    for a in atoms:
        res = a.get_parent()
        try:
            seq.append(seq1(res.get_resname(), custom_map={"UNK": "X"}))
        except Exception:
            seq.append("X")
    return coords, "".join(seq)


# ============================================================
# Group-wise alignment: medoid + iterative consensus (GPA)
# ============================================================

def _build_common_key_list_and_coords(structures, sel, atom_mode: AtomMode = "CA", include_hetero: bool = False):
    """
    Extract selection atoms for every structure and build:
      - keys: sorted list of common atom keys across ALL structures
      - X: (K, N, 3) float array of coords in original frames, stacked by keys
    Key: ((chain_id, resseq, icode), atom_name)
    """
    atom_mode_n = _normalize_atom_mode(atom_mode)
    chain_to_set = {cid: _normalize_residue_spec(spec) for cid, spec in sel.items()}

    def residue_selected(chain_id: str, residue) -> bool:
        if chain_id not in chain_to_set:
            return False
        wanted = chain_to_set[chain_id]
        hetflag, resseq, icode = residue.id
        if (not include_hetero) and (hetflag != " "):
            return False
        if len(wanted) == 0:
            return True
        return (int(resseq), (icode if icode else " ")) in wanted

    def iter_atoms(structure):
        model = next(structure.get_models())
        out = {}
        for chain in model:
            cid = chain.id
            if cid not in chain_to_set:
                continue
            for res in chain:
                if not residue_selected(cid, res):
                    continue
                hetflag, resseq, icode = res.id
                rk = (cid, int(resseq), (icode if icode else " "))

                if atom_mode_n == "all":
                    for a in res.get_atoms():
                        el = getattr(a, "element", None)
                        if el == "H":
                            continue
                        if el is None and a.get_name().upper().startswith("H"):
                            continue
                        out[(rk, a.get_name())] = a.coord.astype(float)
                else:
                    for aname in atom_mode_n:
                        if aname in res:
                            out[(rk, aname)] = res[aname].coord.astype(float)
        return out

    dicts = [iter_atoms(obj.imgt_all_structure) for obj in structures]
    common = set(dicts[0].keys())
    for d in dicts[1:]:
        common &= set(d.keys())
    keys = sorted(common)

    if len(keys) < 3:
        raise RuntimeError(
            f"Too few common atoms across structures for align_sel (common={len(keys)}). "
            f"Try looser selections or atom_mode={atom_mode}."
        )

    X = np.stack([np.stack([d[k] for k in keys], axis=0) for d in dicts], axis=0)  # (K,N,3)
    return keys, X


def _kabsch_fit_RT(P: np.ndarray, Q: np.ndarray):
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ

    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cQ - (R @ cP)
    return R, t


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = _kabsch_fit_RT(P, Q)
    Pf = (R @ P.T).T + t
    d = Pf - Q
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _choose_medoid_index_from_X(X: np.ndarray) -> int:
    K = X.shape[0]
    Dsum = np.zeros(K, dtype=float)
    for i in range(K):
        Pi = X[i]
        for j in range(i + 1, K):
            d = _kabsch_rmsd(Pi, X[j])
            Dsum[i] += d
            Dsum[j] += d
    return int(np.argmin(Dsum))


def _gpa_consensus_template(
    X: np.ndarray,
    init_index: int,
    max_iter: int = 15,
    tol: float = 1e-4,
    center_template: bool = True,
):
    template = X[init_index].copy()
    if center_template:
        template = template - template.mean(axis=0, keepdims=True)

    prev_delta = None
    transforms: List[Tuple[np.ndarray, np.ndarray]] = [(np.eye(3), np.zeros(3)) for _ in range(X.shape[0])]

    for it in range(max_iter):
        X_aligned = np.zeros_like(X)
        new_transforms: List[Tuple[np.ndarray, np.ndarray]] = []

        for i in range(X.shape[0]):
            R, t = _kabsch_fit_RT(X[i], template)
            new_transforms.append((R, t))
            X_aligned[i] = (R @ X[i].T).T + t

        new_template = X_aligned.mean(axis=0)
        if center_template:
            new_template = new_template - new_template.mean(axis=0, keepdims=True)

        delta = float(np.sqrt(np.mean(np.sum((new_template - template) ** 2, axis=1))))
        template = new_template
        transforms = new_transforms

        if prev_delta is not None and abs(prev_delta - delta) < tol:
            return template, transforms, it + 1, delta
        prev_delta = delta

    return template, transforms, max_iter, prev_delta if prev_delta is not None else float("nan")


def _apply_RT_to_structure(structure, R: np.ndarray, t: np.ndarray):
    model0 = next(structure.get_models())
    for atom in model0.get_atoms():
        p = atom.coord.astype(float)
        atom.coord = (R @ p + t).astype(np.float32)


# ============================================================
# Align to a reference and score
# ============================================================

def align_to_ref(
    structures,
    ref_struct,
    atom_mode: AtomMode = "CA",
    align_sel=None,
    score_sel=None,
    tm_mode="fixed",
    tm_reduce="max",
    aligned_dir="aligned",
    write_aligned_pdbs=True,
    ref_index=0):
    rows = []
    aligned_paths = []

    for i, obj in enumerate(structures):
        if not hasattr(obj, "imgt_all_structure"):
            raise ValueError(f"Object at index {i} lacks .imgt_all_structure")
        name = getattr(obj, "name", f"obj{i}")

        mover_struct = deepcopy(obj.imgt_all_structure)

        fixed_atoms_align, moving_atoms_align = _collect_matched_atoms(
            struct_moving=mover_struct,
            struct_fixed=ref_struct,
            chain_to_resids=align_sel,
            atom_mode=atom_mode,
        )

        sup = Superimposer()
        sup.set_atoms(fixed_atoms_align, moving_atoms_align)
        model0 = next(mover_struct.get_models())
        sup.apply(list(model0.get_atoms()))
        align_rms = float(sup.rms)

        fixed_atoms_score, moving_atoms_score = _collect_matched_atoms(
            struct_moving=mover_struct,
            struct_fixed=ref_struct,
            chain_to_resids=score_sel,
            atom_mode=atom_mode,
        )

        coords_fixed = _coords_from_atoms(fixed_atoms_score)
        coords_moving = _coords_from_atoms(moving_atoms_score)
        score_rmsd = _rmsd_from_coords(coords_moving, coords_fixed)

        if tm_mode == "fixed":
            score_tm = _tm_score_fixed(coords_moving, coords_fixed)
        elif tm_mode == "tmalign":
            # Strong recommendation: use tm_mode="fixed" unless you are CA-only,
            # because TM-align assumes residue-level matching rather than arbitrary atom lists.
            try:
                from tmtools import tm_align
            except ImportError as e:
                raise ImportError("tmtools not installed. Install with: pip install tmtools") from e

            cf, sf = _atoms_to_coords_and_seq(fixed_atoms_score)
            cm, sm = _atoms_to_coords_and_seq(moving_atoms_score)
            res = tm_align(cm, cf, sm, sf)
            tm1 = float(res.tm_norm_chain1)
            tm2 = float(res.tm_norm_chain2)

            if tm_reduce == "chain1":
                score_tm = tm1
            elif tm_reduce == "chain2":
                score_tm = tm2
            elif tm_reduce == "mean":
                score_tm = 0.5 * (tm1 + tm2)
            else:
                score_tm = max(tm1, tm2)
        else:
            raise ValueError("tm_mode must be one of: 'fixed' | 'tmalign'")

        if aligned_dir is not None and aligned_dir.strip() != "":
            aligned_path = os.path.join(aligned_dir, f"{name}_aligned.pdb")
            if write_aligned_pdbs:
                _write_pdb(aligned_path, mover_struct)
            aligned_paths.append(aligned_path)
        else:
            aligned_path = None

        rows.append({
            "index": i,
            "name": name,
            "is_reference": (i == ref_index),
            "align_rmsd_on_align_sel": align_rms,
            "score_rmsd_on_score_sel": score_rmsd,
            "score_tm_on_score_sel": score_tm,
            "aligned_pdb": aligned_path,
        })

    results_df = pd.DataFrame(rows)
    return results_df, aligned_paths


# ============================================================
# Batch API
# ============================================================

def align_and_score_structures(
    structures: List[Any],
    align_sel: Dict[str, object],
    score_sel: Dict[str, object],
    out_dir: str,
    atom_mode: AtomMode = "CA",
    alignment_method: str = "align_to_first",  # "align_to_first" | "mediod" | "iterative_mediod"
    tm_mode: str = "fixed",                    # "fixed" | "tmalign"
    tm_reduce: str = "max",                    # if tm_mode="tmalign": "max" | "mean" | "chain1" | "chain2"
    write_aligned_pdbs: bool = True,
    make_pymol_viz: bool = True,
    pymol_png_name: str = "aligned_overlay.png",
    pymol_pse_name: str = "aligned_overlay.pse",
    align_color: str = "tv_yellow",
    score_color: str = "magenta",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Align each structure group-wise and score.

    atom_mode:
      - "CA" or ["CA"]
      - ["N","CA","C","O"] (backbone)
      - ["N","CA","C","O","CB"] (backbone+CB, common but not strictly backbone)
      - "all" (all non-H atoms in selected residues)
    """
    if out_dir is None or out_dir.strip() == "":
        print("Alignment PDBs are not being saved")
    else:
        os.makedirs(out_dir, exist_ok=True)
        aligned_dir = os.path.join(out_dir, "aligned_pdbs")
        os.makedirs(aligned_dir, exist_ok=True)

    if len(structures) < 2:
        raise ValueError("Need at least 2 structures to align/score.")
    if alignment_method not in ["align_to_first", "mediod", "iterative_mediod"]:
        raise ValueError("alignment_method must be one of: 'align_to_first' | 'mediod' | 'iterative_mediod'")

    # Ensure every object has a name
    for i, obj in enumerate(structures):
        if not hasattr(obj, "name"):
            setattr(obj, "name", getattr(obj, "pdb_id", f"obj{i}"))
    for i, obj in enumerate(structures):
        if not hasattr(obj, "imgt_all_structure"):
            setattr(obj, "imgt_all_structure", obj.full_structure) #TCRPAIRVIEW compatability

    if alignment_method == "align_to_first":
        ref_index = 0
        ref_struct = structures[0].imgt_all_structure

        results_df, aligned_paths = align_to_ref(
            structures,
            ref_struct,
            atom_mode=atom_mode,
            align_sel=align_sel,
            score_sel=score_sel,
            tm_mode=tm_mode,
            tm_reduce=tm_reduce,
            aligned_dir=aligned_dir,
            write_aligned_pdbs=write_aligned_pdbs,
            ref_index=ref_index,
        )

    else:
        # --- build correspondence on align_sel across the group ---
        _, X = _build_common_key_list_and_coords(structures, align_sel, atom_mode=atom_mode)

        medoid_idx = _choose_medoid_index_from_X(X)
        print(f"[align] Medoid index={medoid_idx} name={structures[medoid_idx].name} common_atoms={X.shape[1]}")

        if alignment_method == "mediod":
            ref_index = medoid_idx
            ref_struct = structures[ref_index].imgt_all_structure

            results_df, aligned_paths = align_to_ref(
                structures,
                ref_struct,
                atom_mode=atom_mode,
                align_sel=align_sel,
                score_sel=score_sel,
                tm_mode=tm_mode,
                tm_reduce=tm_reduce,
                aligned_dir=aligned_dir,
                write_aligned_pdbs=write_aligned_pdbs,
                ref_index=ref_index,
            )

        else:
            # --- iterative consensus (GPA) ---
            _, transforms, n_iters, final_delta = _gpa_consensus_template(
                X,
                init_index=medoid_idx,
                max_iter=15,
                tol=1e-4,
                center_template=True,
            )
            print(f"[align] GPA converged iters={n_iters} final_delta={final_delta:.6f}")

            # Apply GPA transforms to deep-copied structures
            gpa_structures = []
            for obj, (R, t) in zip(structures, transforms):
                o2 = deepcopy(obj)
                o2.imgt_all_structure = deepcopy(obj.imgt_all_structure)
                _apply_RT_to_structure(o2.imgt_all_structure, R, t)
                gpa_structures.append(o2)

            ref_index = medoid_idx
            ref_struct = gpa_structures[ref_index].imgt_all_structure

            results_df, aligned_paths = align_to_ref(
                gpa_structures,
                ref_struct,
                atom_mode=atom_mode,
                align_sel=align_sel,
                score_sel=score_sel,
                tm_mode=tm_mode,
                tm_reduce=tm_reduce,
                aligned_dir=aligned_dir,
                write_aligned_pdbs=write_aligned_pdbs,
                ref_index=ref_index,
            )

    outputs = {
        "aligned_pdb_dir": aligned_dir,
        "pymol_png": "",
        "pymol_pse": "",
        "pymol_script": "",
        "results_csv": "",
    }

    # ---------------- PyMOL visualization ----------------
    if make_pymol_viz and out_dir is not None and out_dir.strip() != "":
        pymol_script_path = os.path.join(out_dir, "pymol_align_overlay.py")
        pymol_png_path = os.path.join(out_dir, pymol_png_name)
        pymol_pse_path = os.path.join(out_dir, pymol_pse_name)

        align_sel_pml = selection_to_pymol(align_sel)
        score_sel_pml = selection_to_pymol(score_sel)

        obj_palette = [
            "cyan", "green", "orange", "violet", "lime", "salmon", "marine",
            "teal", "yelloworange", "slate", "wheat", "pink", "purple", "olive",
            "deepteal", "tv_blue", "tv_green", "tv_red", "tv_orange", "tv_purple"
        ]

        lines = []
        lines.append("from pymol import cmd")
        lines.append("cmd.reinitialize()")
        lines.append('cmd.bg_color("white")')

        for p in aligned_paths:
            obj_name = os.path.splitext(os.path.basename(p))[0]
            lines.append(f'cmd.load(r"{p}", "{obj_name}")')

        lines.append('cmd.hide("everything", "all")')

        for i, p in enumerate(aligned_paths):
            obj_name = os.path.splitext(os.path.basename(p))[0]
            obj_color = obj_palette[i % len(obj_palette)]
            lines.append(f'cmd.show("cartoon", "{obj_name}")')
            lines.append(f'cmd.color("{obj_color}", "{obj_name}")')

        for p in aligned_paths:
            obj_name = os.path.splitext(os.path.basename(p))[0]
            lines.append(f'cmd.select("align_{obj_name}", "{obj_name} and ({align_sel_pml})")')
            lines.append(f'cmd.select("score_{obj_name}", "{obj_name} and ({score_sel_pml})")')
            lines.append(f'cmd.color("{align_color}", "align_{obj_name}")')
            lines.append(f'cmd.color("{score_color}", "score_{obj_name}")')

        # Legend
        lines.append('mn, mx = cmd.get_extent("all")')
        lines.append("dx = mx[0]-mn[0]; dy = mx[1]-mn[1]; dz = mx[2]-mn[2]")
        lines.append("p_align = [mn[0] + 0.06*dx, mx[1] - 0.06*dy, mx[2] - 0.06*dz]")
        lines.append("p_score = [mn[0] + 0.06*dx, mx[1] - 0.14*dy, mx[2] - 0.06*dz]")

        lines.append('cmd.pseudoatom("legend_align", pos=p_align)')
        lines.append('cmd.pseudoatom("legend_score", pos=p_score)')
        lines.append('cmd.show("spheres", "legend_align or legend_score")')
        lines.append('cmd.set("sphere_scale", 1, "legend_align or legend_score")')
        lines.append(f'cmd.color("{align_color}", "legend_align")')
        lines.append(f'cmd.color("{score_color}", "legend_score")')
        lines.append('cmd.set("label_color", "black")')
        lines.append('cmd.set("label_size", 18)')
        lines.append('cmd.set("label_position", [1.5, 0.0, 0.0])')
        lines.append('cmd.set("label_outline_color", "white")')
        lines.append('cmd.label("legend_align", "\'Aligned region\'")')
        lines.append('cmd.label("legend_score", "\'Scored region\'")')

        lines.append('cmd.set("cartoon_transparency", 0.15, "all")')
        lines.append('cmd.orient("all")')
        lines.append('cmd.zoom("all", 1.15)')
        lines.append(f'cmd.png(r"{pymol_png_path}", dpi=300, ray=1)')
        lines.append(f'cmd.save(r"{pymol_pse_path}")')
        lines.append("cmd.quit()")

        with open(pymol_script_path, "w") as f:
            f.write("\n".join(lines))

        rc = os.system(f"pymol -cq {pymol_script_path}")
        if rc != 0:
            print(f"[WARN] PyMOL returned non-zero exit code: {rc}")
            print(f"[WARN] Script written to: {pymol_script_path}")
        else:
            outputs["pymol_png"] = pymol_png_path
            outputs["pymol_pse"] = pymol_pse_path
        outputs["pymol_script"] = pymol_script_path

    # Save results CSV
    if out_dir is None or out_dir.strip() == "":
        print("Results CSV is not being saved")
    else:
        results_csv = os.path.join(out_dir, "alignment_scores.csv")
        results_df.to_csv(results_csv, index=False)
    outputs["results_csv"] = results_csv

    return results_df, outputs

def CLASSIC_TCR_ALIGNMENTS(tcrpmhc_list, out_dir, atom_mode, alignment_method, tm_mode,make_pymol_viz):
    complete_df=pd.DataFrame()
    #FIRST ALIGN AND SCORE ON EACH INDIVIDUAL CDR
    for cdr_name, cdr_range in CDR_FR_RANGES.items():
        if "A_CDR" in cdr_name:
            align_sel = {"A": {"ranges": A_FR_ALIGN}}  # Adjust chain ID as needed
            score_sel = {"A": {"range": cdr_range}}  # Adjust chain ID as needed
        elif "B_CDR" in cdr_name:
            align_sel = {"B": {"ranges": B_FR_ALIGN}}  # Adjust chain ID as needed
            score_sel = {"B": {"range": cdr_range}}  # Adjust chain ID as needed
        else:
            continue
        df_aligned_cdr, outputs_aligned_cdr = align_and_score_structures(
            structures=tcrpmhc_list,
            align_sel=score_sel,
            score_sel=score_sel,
            out_dir=None if out_dir is None or out_dir.strip() == "" else os.path.join(out_dir, f"CDR_{cdr_name}_alignment"),
            atom_mode=atom_mode,
            alignment_method=alignment_method,
            tm_mode=tm_mode,
            make_pymol_viz=make_pymol_viz
        )
        df_aligned_fr, outputs_aligned_fr = align_and_score_structures(
            structures=tcrpmhc_list,
            align_sel=align_sel,
            score_sel=score_sel,
            out_dir=None if out_dir is None or out_dir.strip() == "" else os.path.join(out_dir, f"FR_{cdr_name}_alignment"),
            atom_mode=atom_mode,
            alignment_method=alignment_method,
            tm_mode=tm_mode,
            make_pymol_viz=make_pymol_viz
        )
        #only take rmsd, tm and name column from df
        df_aligned_cdr=df_aligned_cdr[["score_rmsd_on_score_sel","score_tm_on_score_sel", "name"]]
        df_aligned_fr=df_aligned_fr[["score_rmsd_on_score_sel","score_tm_on_score_sel", "name"]]
        #rename score_rmsd_on_score_sel to {cdr_name}_rmsd
        df_aligned_cdr.rename(columns={
            "score_rmsd_on_score_sel":f"{cdr_name}_rmsd",
            "score_tm_on_score_sel":f"{cdr_name}_tm"
        }, inplace=True)


        df_aligned_fr.rename(columns={
            "score_rmsd_on_score_sel":f"fr_aligned{cdr_name}_rmsd",
            "score_tm_on_score_sel":f"fr_aligned{cdr_name}_tm"
        }, inplace=True)
        if complete_df.empty:
            complete_df=df_aligned_cdr
            complete_df=pd.merge(complete_df, df_aligned_fr, on="name")

        else:
            complete_df=pd.merge(complete_df, df_aligned_cdr, on="name")
            complete_df=pd.merge(complete_df, df_aligned_fr, on="name")
    # ALIGN ON FR AND CALC ALL CDRS
    all_A_CDRS= {"A": {"ranges": [CDR_FR_RANGES["A_CDR1"], CDR_FR_RANGES["A_CDR2"], CDR_FR_RANGES["A_CDR3"]]}}
    all_B_CDRS= {"B": {"ranges": [CDR_FR_RANGES["B_CDR1"], CDR_FR_RANGES["B_CDR2"], CDR_FR_RANGES["B_CDR3"]]}}
    all_CDRS= {**all_A_CDRS, **all_B_CDRS}
    align_sel_A = {"A": {"ranges": A_FR_ALIGN}}  # Adjust chain ID as needed
    align_sel_B = {"B": {"ranges": B_FR_ALIGN}}  # Adjust chain ID as needed
    align_sell_AB = {**align_sel_A, **align_sel_B}
    df_aligned_allA, outputs_aligned_allA = align_and_score_structures(
            structures=tcrpmhc_list,
            align_sel=align_sel_A,
            score_sel=all_A_CDRS,
            out_dir=None if out_dir is None or out_dir.strip() == "" else os.path.join(out_dir, f"all_ACDRs_alignment"),
            atom_mode=atom_mode,
            alignment_method=alignment_method,
            tm_mode=tm_mode,
            make_pymol_viz=make_pymol_viz
        )
    df_aligned_allB, outputs_aligned_allB = align_and_score_structures(
            structures=tcrpmhc_list,
            align_sel=align_sel_B,
            score_sel=all_B_CDRS,
            out_dir=None if out_dir is None or out_dir.strip() == "" else os.path.join(out_dir, f"all_BCDRs_alignment"),
            atom_mode=atom_mode,
            alignment_method=alignment_method,
            tm_mode=tm_mode,
            make_pymol_viz=make_pymol_viz
        )
    df_aligned_allAB, outputs_aligned_allAB = align_and_score_structures(
            structures=tcrpmhc_list,
            align_sel=align_sell_AB,
            score_sel=all_CDRS,
            out_dir=None if out_dir is None or out_dir.strip() == "" else os.path.join(out_dir, f"all_CDRs_alignment"),
            atom_mode=atom_mode,
            alignment_method=alignment_method,
            tm_mode=tm_mode,
            make_pymol_viz=make_pymol_viz
        )
    #only take rmsd, tm and name column from df
    df_aligned_allA=df_aligned_allA[["score_rmsd_on_score_sel","score_tm_on_score_sel", "name"]]
    df_aligned_allB=df_aligned_allB[["score_rmsd_on_score_sel","score_tm_on_score_sel", "name"]]
    df_aligned_allAB=df_aligned_allAB[["score_rmsd_on_score_sel","score_tm_on_score_sel", "name"]]
    #rename score_rmsd_on_score_sel to all_ACDRs_rmsd
    df_aligned_allA.rename(columns={
        "score_rmsd_on_score_sel":"all_ACDRs_rmsd",
        "score_tm_on_score_sel":"all_ACDRs_tm"}, inplace=True)
    df_aligned_allB.rename(columns={
        "score_rmsd_on_score_sel":"all_BCDRs_rmsd",
        "score_tm_on_score_sel":"all_BCDRs_tm"}, inplace=True)
    df_aligned_allAB.rename(columns={
        "score_rmsd_on_score_sel":"all_CDRs_rmsd",
        "score_tm_on_score_sel":"all_CDRs_tm"}, inplace=True)
    complete_df=pd.merge(complete_df, df_aligned_allA, on="name")
    complete_df=pd.merge(complete_df, df_aligned_allB, on="name")
    complete_df=pd.merge(complete_df, df_aligned_allAB, on="name")
    #SAVE COMPLETE DF
    if out_dir is None or out_dir.strip() == "":
        print("Complete TCR CDR alignment CSV is not being saved")
    else:
        complete_df.to_csv(os.path.join(out_dir, "complete_TCR_CDR_alignments.csv"), index=False)
    return complete_df




# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC
    from TCR_TOOLS.classes.tcr import TCR


    #a = TCRpMHC("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1j8h.pdb", MHC_a_chain_id="A", MHC_b_chain_id="B", Peptide_chain_id="C")
    #b = TCRpMHC("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1fyt.pdb", MHC_a_chain_id="A", MHC_b_chain_id="B", Peptide_chain_id="C")
    #c = TCRpMHC("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1g6r.pdb", MHC_a_chain_id="A", MHC_b_chain_id="B", Peptide_chain_id="C")

    a = TCR("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1j8h.pdb").pairs[0]
    b = TCR("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1fyt.pdb").pairs[0]
    c = TCR("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1g6r.pdb").pairs[0]


    tcrpmhc_list = [a, b, c]
    out_dir="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/aligners/test"
    atom_mode=["N", "CA", "C", "O"]  # backbone
    alignment_method="iterative_mediod"
    tm_mode="fixed"                 # strongly recommended for non-CA atom sets
    make_pymol_viz=True
    CLASSIC_TCR_ALIGNMENTS(tcrpmhc_list, out_dir, atom_mode, alignment_method, tm_mode,make_pymol_viz)
