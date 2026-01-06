import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import sys
sys.path.append("/workspaces/Graphormer")
from TCR_Metrics.pipelines.process_datasets.process_TRAIT.pMHC_seq import get_alleles


# ---------------------------
# pMHC parsing
# ---------------------------

_ALLELE_RE = re.compile(r"^([A-Z])(\d{2})(\d{2})$")
_NR_RE = re.compile(r"^NR\(([^)]+)\)$")


def allele_code_to_hla(code: str) -> str:
    m = _ALLELE_RE.fullmatch(str(code).strip())
    if not m:
        return str(code).strip()
    locus, f1, f2 = m.groups()
    return f"HLA-{locus}*{f1}:{f2}"


def _infer_mhc_class(hla: str) -> int:
    if hla.startswith("HLA-A*") or hla.startswith("HLA-B*") or hla.startswith("HLA-C*"):
        return 1
    if hla.startswith("HLA-DP") or hla.startswith("HLA-DQ") or hla.startswith("HLA-DR"):
        return 2
    raise ValueError(f"Unrecognized HLA allele format: {hla}")


def parse_pmhc_id(pmhc: str) -> Dict[str, Any]:
    parts = str(pmhc).split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected pMHC_ID format: {pmhc}")

    allele_token = parts[0].strip()
    peptide = parts[1].strip()
    source = "_".join(parts[2:]).replace("_binder", "") if len(parts) > 2 else ""

    nr_m = _NR_RE.fullmatch(allele_token)
    allele_code = nr_m.group(1) if nr_m else allele_token

    hla = allele_code_to_hla(allele_code)
    mhc_class = _infer_mhc_class(hla)

    MHC_A_seq = get_alleles(hla)
    MHC_B_seq = get_alleles("B2M") if mhc_class == 1 else None

    return {
        "hla": hla,
        "peptide": peptide,
        "MHC_A_seq": MHC_A_seq,
        "MHC_B_seq": MHC_B_seq,
        "class": mhc_class,
        "source": source,
    }


# ---------------------------
# Thimble
# ---------------------------

THIMBLE_COLUMNS = [
    "TCR_name",
    "TRAV", "TRAJ", "TRA_CDR3",
    "TRBV", "TRBJ", "TRB_CDR3",
    "TRAC", "TRBC",
    "TRA_leader", "TRB_leader",
    "Linker", "Link_order",
    "TRA_5_prime_seq", "TRA_3_prime_seq",
    "TRB_5_prime_seq", "TRB_3_prime_seq",
]


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable '{name}' not found on PATH.")


def make_thimble_input(df: pd.DataFrame, out_path: Path) -> Path:
    """
    Creates a Thimble input with a strict header template.
    Ensures stable row IDs (row0, row1, ...) for mapping back later.
    """
    required = ["TRAV", "TRAJ", "CDR3a", "TRBV", "TRBJ", "CDR3b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for Thimble: {missing}. Found: {list(df.columns)}")

    th = pd.DataFrame({
        "TCR_name": [f"row{i}" for i in range(len(df))],
        "TRAV": df["TRAV"].astype(str).str.strip(),
        "TRAJ": df["TRAJ"].astype(str).str.strip(),
        "TRA_CDR3": df["CDR3a"].astype(str).str.strip(),
        "TRBV": df["TRBV"].astype(str).str.strip(),
        "TRBJ": df["TRBJ"].astype(str).str.strip(),
        "TRB_CDR3": df["CDR3b"].astype(str).str.strip(),
    })

    # Add remaining required template columns as empty strings
    for col in THIMBLE_COLUMNS:
        if col not in th.columns:
            th[col] = ""

    th = th[THIMBLE_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    th.to_csv(out_path, sep="\t", index=False)
    return out_path


def run_thimble(thimble_in: Path, thimble_out: Path, receptor: str = "AB", species: str = "human") -> None:
    """
    Runs thimble, writes stdout/stderr logs next to output for debugging.
    """
    _require_executable("thimble")

    cmd = ["thimble", "-i", str(thimble_in), "-r", receptor, "-s", species, "-o", str(thimble_out)]
    p = subprocess.run(cmd, capture_output=True, text=True)

    # Persist logs for inspection
    thimble_out.with_suffix(".stdout.txt").write_text(p.stdout or "")
    thimble_out.with_suffix(".stderr.txt").write_text(p.stderr or "")

    if p.returncode != 0:
        raise RuntimeError(
            "Thimble failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"See logs:\n  {thimble_out.with_suffix('.stdout.txt')}\n  {thimble_out.with_suffix('.stderr.txt')}\n"
        )

    if not thimble_out.exists():
        raise RuntimeError(f"Thimble reported success but output not found: {thimble_out}")


def _pick_chain_cols(thimble_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Find alpha/beta AA columns in Thimble output.
    Adjust candidates if your thimble version names columns differently.
    """
    cols = list(thimble_df.columns)

    # Common names; this is where your install may differ.
    tra_candidates = ["TRA_aa", "TRA_AA", "TRA_protein", "TRA_sequence_aa", "alpha_aa", "alpha_protein"]
    trb_candidates = ["TRB_aa", "TRB_AA", "TRB_protein", "TRB_sequence_aa", "beta_aa", "beta_protein"]

    tra = next((c for c in tra_candidates if c in cols), None)
    trb = next((c for c in trb_candidates if c in cols), None)

    if tra is None or trb is None:
        raise KeyError(
            "Could not find TRA/TRB AA columns in Thimble output.\n"
            f"Columns present: {cols}\n"
            "Fix: open the thimble output file and add its actual AA column names to _pick_chain_cols()."
        )
    return tra, trb


# ---------------------------
# Main processing
# ---------------------------

def process_pmhc_file(file_path: str, pmhc_dict: Dict[str, Dict[str, Any]], outdir: Path) -> pd.DataFrame:
    file_path = str(file_path)
    in_path = Path(file_path)
    stem = in_path.stem


    # Force strings so pandas never turns IDs into floats
    print(f"[INFO] Processing file: {file_path}")
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)
    print(df.columns)

    pmhc_col = "pMHC_ID" if "pMHC_ID" in df.columns else ("pMHC" if "pMHC" in df.columns else None)
    if pmhc_col is None:
        raise KeyError(f"Missing pMHC column (expected pMHC_ID or pMHC). Columns: {list(df.columns)}")

    # Add stable key for mapping back
    df["TCR_name"] = [f"row{i}" for i in range(len(df))]

    # Thimble I/O per file (avoid overwriting between files)
    thimble_in = outdir / f"{stem}.thimble_input.tsv"
    thimble_out = outdir / f"{stem}.thimble_out.tsv"

    make_thimble_input(df, thimble_in)
    run_thimble(thimble_in, thimble_out, receptor="AB", species="human")

    # Load thimble output as strings only
    th = pd.read_csv(thimble_out, sep="\t", dtype=str, keep_default_na=False)
    if "TCR_name" not in th.columns:
        raise KeyError(f"Thimble output missing TCR_name column. Columns: {list(th.columns)}")

    tra_col, trb_col = _pick_chain_cols(th)
    # Build mapping (avoids merge dtype issues entirely)
    tra_map = dict(zip(th["TCR_name"], th[tra_col]))
    trb_map = dict(zip(th["TCR_name"], th[trb_col]))

    df["TRA_full_aa"] = df["TCR_name"].map(tra_map)
    df["TRB_full_aa"] = df["TCR_name"].map(trb_map)
    # Report stitching failures
    n_missing = (df["TRA_full_aa"].eq("") | df["TRA_full_aa"].isna()).sum() + (df["TRB_full_aa"].eq("") | df["TRB_full_aa"].isna()).sum()
    if n_missing > 0:
        print(f"[WARN] {stem}: {n_missing} missing stitched chain sequences. "
              f"Check {thimble_out.with_suffix('.stderr.txt')} for Stitchr/Thimble errors.")

        # Optional: save a diagnostics subset
        fail_mask = (df["TRA_full_aa"].eq("") | df["TRA_full_aa"].isna()) | (df["TRB_full_aa"].eq("") | df["TRB_full_aa"].isna())
        df.loc[fail_mask, ["TCR_name", "TRAV", "TRAJ", "CDR3a", "TRBV", "TRBJ", "CDR3b"]].to_csv(
            outdir / f"{stem}.stitch_failures.tsv", sep="\t", index=False
        )

    # Parse pMHC with caching
    parsed = []
    for pmhc_id in df[pmhc_col].astype(str).tolist():
        if pmhc_id in pmhc_dict:
            info = pmhc_dict[pmhc_id]
        else:
            info = parse_pmhc_id(pmhc_id)
            pmhc_dict[pmhc_id] = info
        parsed.append(info)

    df["pMHC_hla"] = [p["hla"] for p in parsed]
    df["pMHC_peptide"] = [p["peptide"] for p in parsed]
    df["pMHC_class"] = [p["class"] for p in parsed]
    df["pMHC_source"] = [p["source"] for p in parsed]
    df["MHC_A_seq"] = [p["MHC_A_seq"] for p in parsed]
    df["MHC_B_seq"] = [p["MHC_B_seq"] for p in parsed]

    # Save augmented
    out_path = outdir / f"{stem}.augmented.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote augmented file: {out_path}")

    return df


# ---------------------------
# Batch processing
# ---------------------------

omics_database_home = Path("/mnt/larry/lilian/DATA/TRAIT/OMICS_DATA_TRAIT")
pmhc_dict: Dict[str, Dict[str, Any]] = {}  # keep cache across all files
outdir=Path("/mnt/larry/lilian/DATA/TRAIT/omics_processed")
for file_name in sorted(os.listdir(omics_database_home)):
    if not file_name.endswith(".txt"):
        continue
    file_path = omics_database_home / file_name
    #try:
    process_pmhc_file(str(file_path), pmhc_dict, outdir=outdir)
    #except Exception as e:
    #    print(f"[ERROR] Failed on {file_name}: {e}")
cleaned_output_dir=os.path.join(outdir, "cleaned")
for file in os.listdir(outdir):
    if file.endswith("augmented.tsv"):
        df = pd.read_csv(outdir / file, sep="\t", dtype=str, keep_default_na=False)
        n_total = len(df)
        n_missing = (df["TRA_full_aa"].eq("") | df["TRA_full_aa"].isna()).sum() + (df["TRB_full_aa"].eq("") | df["TRB_full_aa"].isna()).sum()
        print(f"{file}: {n_missing}/{n_total*2} missing stitched chains ({(n_missing/(n_total*2))*100:.2f}%)")
        input()
        cleaned_df= df[~((df["TRA_full_aa"].eq("") | df["TRA_full_aa"].isna()) | (df["TRB_full_aa"].eq("") | df["TRB_full_aa"].isna()))]
        os.makedirs(cleaned_output_dir, exist_ok=True)
        cleaned_df.to_csv(os.path.join(cleaned_output_dir, file), sep="\t", index=False)