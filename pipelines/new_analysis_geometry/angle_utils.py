#!/usr/bin/env python3
"""
Compute TCR geometry + germlines for a directory of PDBs and save a single CSV.

What it does
------------
For each .pdb in --pdb-dir:
  - Initializes TCR_TOOLS.classes.tcr.TCR
  - Takes tcr.pairs[0]
  - Runs pair.calc_angles_tcr(out_path=None, vis=False)
  - Extracts alpha/beta germline calls (pair.alpha_germline / pair.beta_germline)
  - Normalizes germlines to:
        alpha_v_gene, alpha_v_score, alpha_j_gene, alpha_j_score
        beta_v_gene,  beta_v_score,  beta_j_gene,  beta_j_score
        alpha_vj, beta_vj, germline_vj_pair
  - Adds: tcr_name (filename stem), state (given), pdb_path
  - Saves CSV

Usage
-----
python build_tcr_geometry_table.py \
  --pdb-dir /path/to/pdbs \
  --state unbound \
  --out /path/to/out.csv

Optional:
  --min-contacts 50
  --contact-cutoff 5.0
  --legacy-anarci 1
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# Import your TCR tools
# --------------------------------------------------------------------
# If you do NOT need these, remove them and rely on a proper pip install.
sys.path.append("/workspaces/Graphormer/TCR_Metrics/")
sys.path.append("/workspaces/Graphormer/")

from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC


# --------------------------------------------------------------------
# Germline parsing helpers
# --------------------------------------------------------------------
def parse_germline_field(x: Any) -> Dict[str, Any]:
    """
    Accept dict or stringified dict; return dict or {} on failure.
    """
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    # pandas may store missing dicts as NaN floats
    if isinstance(x, float) and np.isnan(x):
        return {}
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
    return {}

def _extract_gene_and_score(germline_obj: Any, key: str) -> Tuple[str, float]:
    """
    germline_obj example:
      {'v_gene': [('human', 'TRAV12-2*01'), 0.9565], 'j_gene': [('human', 'TRAJ24*02'), 1.0]}
    Returns (gene_str, score_float) for requested key ('v_gene' or 'j_gene').
    """
    d = parse_germline_field(germline_obj)
    if key not in d:
        return ("unknown", float("nan"))
    item = d.get(key)
    try:
        gene_tuple = item[0]  # ('human', 'TRAV12-2*01')
        score = float(item[1])
        gene = gene_tuple[1] if isinstance(gene_tuple, (list, tuple)) and len(gene_tuple) >= 2 else "unknown"
        gene = str(gene) if gene else "unknown"
        return (gene, score)
    except Exception:
        return ("unknown", float("nan"))

def normalize_germlines_in_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds normalized germline columns to a row dict in-place style.
    Expects row has keys: alpha_germline, beta_germline (dict or string or None).
    """
    a = row.get("alpha_germline", None)
    b = row.get("beta_germline", None)

    alpha_v_gene, alpha_v_score = _extract_gene_and_score(a, "v_gene")
    alpha_j_gene, alpha_j_score = _extract_gene_and_score(a, "j_gene")
    beta_v_gene,  beta_v_score  = _extract_gene_and_score(b, "v_gene")
    beta_j_gene,  beta_j_score  = _extract_gene_and_score(b, "j_gene")

    row["alpha_v_gene"] = alpha_v_gene
    row["alpha_v_score"] = alpha_v_score
    row["alpha_j_gene"] = alpha_j_gene
    row["alpha_j_score"] = alpha_j_score
    row["beta_v_gene"] = beta_v_gene
    row["beta_v_score"] = beta_v_score
    row["beta_j_gene"] = beta_j_gene
    row["beta_j_score"] = beta_j_score

    row["alpha_vj"] = f"{alpha_v_gene}-{alpha_j_gene}"
    row["beta_vj"] = f"{beta_v_gene}-{beta_j_gene}"
    row["germline_vj_pair"] = f"{row['alpha_vj']}|{row['beta_vj']}"
    return row


# --------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------
def compute_one_pdb_unbound(
    pdb_path: Path,
    state: str,
    contact_cutoff: float,
    min_contacts: int,
    legacy_anarci: bool,
    vis: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Returns a single row dict (angles + germlines + metadata) or None if failed.
    """
    tcr_name = pdb_path.stem

    try:
        tcr = TCR(
            input_pdb=str(pdb_path),
            traj_path=None,
            contact_cutoff=float(contact_cutoff),
            min_contacts=int(min_contacts),
            legacy_anarci=bool(legacy_anarci),
        )
    except Exception as e:
        print(f"[WARN] init failed: {tcr_name}: {e}")
        return None

    if not hasattr(tcr, "pairs") or len(tcr.pairs) == 0:
        print(f"[WARN] no alpha/beta pair: {tcr_name}")
        return None

    pair = tcr.pairs[0]
    alpha_germline = getattr(pair, "alpha_germline", None)
    beta_germline = getattr(pair, "beta_germline", None)

    try:
        angle_dict = pair.calc_angles_tcr(out_path=None, vis=bool(vis))
        if not isinstance(angle_dict, dict) or len(angle_dict) == 0:
            print(f"[WARN] empty angles: {tcr_name}")
            return None
    except Exception as e:
        print(f"[WARN] calc_angles failed: {tcr_name}: {e}")
        return None

    # Build row
    row: Dict[str, Any] = dict(angle_dict)
    row["tcr_name"] = tcr_name
    row["state"] = str(state)
    row["pdb_path"] = str(pdb_path)

    # keep raw germline objects (dicts or strings), then normalize
    row["alpha_germline"] = alpha_germline
    row["beta_germline"] = beta_germline
    row = normalize_germlines_in_row(row)

    return row

def compute_one_pdb_bound(
    pdb_path: Path,
    state: str,
    contact_cutoff: float,
    min_contacts: int,
    legacy_anarci: bool,
    vis: bool = False,
    MHC_a_chain_id="A",
    MHC_b_chain_id="B",
    Peptide_chain_id="C",
    TCR_a_chain_id="D",
    TCR_b_chain_id="E"
    ) -> Optional[Dict[str, Any]]:
    """
    Returns a single row dict (angles + germlines + metadata) or None if failed.
    """
    tcr_name = pdb_path.stem

    try:
        tcrpMHC=TCRpMHC(
            input_pdb=str(pdb_path),
            contact_cutoff=float(contact_cutoff),
            min_contacts=int(min_contacts),
            legacy_anarci=bool(legacy_anarci)
        )
        pair = tcrpMHC.pairs[0]
    except Exception as e1:
        try:
            tcrpMHC=TCRpMHC(
                input_pdb=str(pdb_path),
                contact_cutoff=float(contact_cutoff),
                min_contacts=int(min_contacts),
                legacy_anarci=bool(legacy_anarci),
                MHC_a_chain_id= MHC_a_chain_id,
                MHC_b_chain_id= MHC_b_chain_id,
                Peptide_chain_id=Peptide_chain_id,
                TCR_a_chain_id=TCR_a_chain_id,
                TCR_b_chain_id=TCR_b_chain_id
            )
            pair = tcrpMHC.pairs[0]
        except Exception as e2:
            try:
                tcrpMHC=TCRpMHC(
                        input_pdb=str(pdb_path),
                        contact_cutoff=float(contact_cutoff),
                        min_contacts=int(min_contacts),
                        legacy_anarci=bool(legacy_anarci),
                        MHC_a_chain_id= MHC_a_chain_id,
                        MHC_b_chain_id= None,
                        Peptide_chain_id=Peptide_chain_id,
                        TCR_a_chain_id=TCR_a_chain_id,
                        TCR_b_chain_id=TCR_b_chain_id
                        )
                pair = tcrpMHC.pairs[0]
            except Exception as e3:
                print(f"[WARN] init failed: {tcr_name}: {e1} | {e2} | {e3}")
                return None

    alpha_germline = getattr(pair, "alpha_germline", None)
    beta_germline = getattr(pair, "beta_germline", None)
    mhc_alpha_allele = getattr(tcrpMHC, "mhc_alpha_allele", None)
    mhc_beta_allele = getattr(tcrpMHC, "mhc_beta_allele", None)
    print(pdb_path)
    angle_dict =tcrpMHC.calc_geometry(out_path=None)
    print(angle_dict)

    if not isinstance(angle_dict, dict) or len(angle_dict) == 0:
        print(f"[WARN] empty angles: {tcr_name}")
        return None

    # Build row
    row: Dict[str, Any] = dict(angle_dict)
    row["tcr_name"] = tcr_name
    row["state"] = str(state)
    row["pdb_path"] = str(pdb_path)

    # keep raw germline objects (dicts or strings), then normalize
    row["alpha_germline"] = alpha_germline
    row["beta_germline"] = beta_germline
    row["mhc_alpha_allele"]=mhc_alpha_allele["stitle"]
    row["mhc_beta_allele"]= mhc_beta_allele["stitle"] if mhc_beta_allele is not None else None
    row = normalize_germlines_in_row(row)

    return row




def compute_directory(
    pdb_dir: Path,
    state: str,
    contact_cutoff: float = 5.0,
    min_contacts: int = 50,
    legacy_anarci: bool = True,
    vis: bool = False) -> pd.DataFrame:

    fails=[]
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {pdb_dir}")

    pdb_files = sorted([p for p in pdb_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdb"])
    print(f"[INFO] Found {len(pdb_files)} PDB files in {pdb_dir} (state={state})")

    rows: List[Dict[str, Any]] = []
    for i, pdb_path in enumerate(pdb_files, start=1):
        if i % 50 == 0:
            print(f"[INFO] processed {i}/{len(pdb_files)}")
        if state=="unbound":
            row = compute_one_pdb_unbound(
                    pdb_path=pdb_path,
                    state=state,
                    contact_cutoff=contact_cutoff,
                    min_contacts=min_contacts,
                    legacy_anarci=legacy_anarci,
                    vis=vis)
            if row is not None:
                rows.append(row)
            else:
                print(f"failed for {pdb_path}")
                fails.append(pdb_path)
        if state=="bound":
            row = compute_one_pdb_bound(
                    pdb_path=pdb_path,
                    state=state,
                    contact_cutoff=contact_cutoff,
                    min_contacts=min_contacts,
                    legacy_anarci=legacy_anarci,
                    vis=vis)
            if row is not None:
                rows.append(row)
            else:
                print(f"failed for {pdb_path}")
                fails.append(pdb_path)

    if not rows:
        raise RuntimeError(f"No results produced for directory: {pdb_dir}")

    df = pd.DataFrame(rows)
    # Summary
    n_unique = df["tcr_name"].nunique() if "tcr_name" in df.columns else len(df)
    print(f"[OK] Collected {len(df)} rows ({n_unique} unique TCRs) for state={state}.")
    return df, fails


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(pdb_dir, out_csv, state):
    pdb_dir = Path(pdb_dir)
    fails_file=out_csv.replace(".csv","_fails.txt")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df, fails= compute_directory(
        pdb_dir=pdb_dir,
        state=state
    )

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote CSV: {out_csv}")

    with open(fails_file, 'w') as f:
        for line in fails:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
