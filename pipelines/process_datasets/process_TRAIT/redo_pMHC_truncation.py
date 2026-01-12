#!/usr/bin/env python3
import os
import json
import re
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import requests


# -------------------------
# Constants / paths
# -------------------------
HUMAN_B2M_SEQ = (
    "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSREPAENGKPNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
)

TRUNCATION_CSV = "/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/process_TRAIT/MHC_TRUNCATION.csv"

omics_processed_dir = "/mnt/larry/lilian/DATA/TRAIT/omics_processed"
pmhc_col = "pMHC"     # your files use this
force = True          # overwrite existing columns

# Optional: persistent allele cache file (recommended)
CACHE_JSON = "/mnt/larry/lilian/DATA/TRAIT/omics_processed/mhc_fullseq_cache.json"


_ALLELE_RE = re.compile(r"^([A-Z])(\d{2})(\d{2})$")  # e.g. A0301 -> HLA-A*03:01
_NR_RE = re.compile(r"^NR\(([^)]+)\)$")             # NR(HLA-A*03:01)


# -------------------------
# pMHC parsing
# -------------------------
def allele_code_to_hla(code: str) -> str:
    code = str(code).strip()
    m = _ALLELE_RE.fullmatch(code)
    if not m:
        return code
    locus, f1, f2 = m.groups()
    return f"HLA-{locus}*{f1}:{f2}"

def infer_mhc_class(hla: str) -> int:
    if hla.startswith(("HLA-A*", "HLA-B*", "HLA-C*")):
        return 1
    if hla.startswith(("HLA-DP", "HLA-DQ", "HLA-DR")):
        return 2
    raise ValueError(f"Unrecognized HLA format: {hla}")

def parse_pmhc_id(pmhc: str) -> Dict[str, Any]:
    """
    Expected formats:
      - A0301_KLGGALQAK
      - A0301_KLGGALQAK_binder
      - NR(HLA-A*03:01)_KLGGALQAK
      - HLA-A*03:01_KLGGALQAK
    """
    parts = str(pmhc).split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected pMHC format: {pmhc}")

    allele_token = parts[0].strip()
    peptide = parts[1].strip()

    nr_m = _NR_RE.fullmatch(allele_token)
    allele_code = nr_m.group(1) if nr_m else allele_token

    hla = allele_code_to_hla(allele_code)
    mhc_class = infer_mhc_class(hla)

    return {"hla": hla, "peptide": peptide, "class": mhc_class}


# -------------------------
# Fetching from IPD-IMGT/HLA (EBI)
# -------------------------
def _generate_ebi_allele_query_url(allele_name: str) -> str:
    base_url = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele"
    allele_name = allele_name.replace("HLA-", "")  # EBI expects like A*03:01
    query = (
        f'or(startsWith(name,"{allele_name}"),'
        f'contains(previous_nomenclature,"{allele_name}"),'
        f'eq(accession,"{allele_name}"))'
    )
    params = {
        "limit": 1,
        "project": "HLA",
        "fields": "name,accession",
        "query": query,
    }
    return f"{base_url}?{urllib.parse.urlencode(params, safe='(),*')}"

def fetch_accession_code(hla: str, timeout_s: int = 30) -> str:
    url = _generate_ebi_allele_query_url(hla)
    headers = {"accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if "data" not in data or len(data["data"]) == 0:
        raise ValueError(f"No accession found for allele {hla} (query URL: {url})")
    return data["data"][0]["accession"]

def fetch_full_protein_sequence(accession_code: str, timeout_s: int = 30) -> str:
    url = f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=imgthlapro;id={accession_code}&format=fasta&style=raw"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    fasta = r.text.strip().splitlines()
    if len(fasta) < 2:
        raise ValueError(f"FASTA fetch too short for accession {accession_code}")
    seq = "".join(fasta[1:]).strip().upper()
    if not seq:
        raise ValueError(f"Empty sequence for accession {accession_code}")
    return seq


# -------------------------
# Truncation from YOUR CSV
# -------------------------
def load_trunc_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    if not {"MHC_name", "start", "end"}.issubset(df.columns):
        raise KeyError(f"Truncation table must contain MHC_name,start,end. Found: {list(df.columns)}")
    df["MHC_name"] = df["MHC_name"].astype(str).str.strip()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df

def mhc_key_from_hla(hla: str) -> str:
    """
    HLA-A*03:01 -> HLA-A
    HLA-DRB1*01:01 -> HLA-DRB1
    """
    hla = str(hla).strip()
    if hla.startswith("HLA-") and "*" in hla:
        return hla.split("*")[0]
    if hla.startswith("HLA-"):
        return hla
    if "*" in hla:
        return "HLA-" + hla.split("*")[0]
    return hla

def truncate_with_table(hla: str, seq: str, trunc_df: pd.DataFrame) -> str:
    """
    CSV uses UniProt-style coords: 1-based, inclusive.
    start=25,end=206 -> seq[24:206]
    """
    seq = (seq or "").strip().replace(" ", "").replace("\n", "").upper()
    if not seq:
        return seq

    key = mhc_key_from_hla(hla)
    row = trunc_df.loc[trunc_df["MHC_name"] == key]
    if row.empty:
        return seq

    start_1 = int(row.iloc[0]["start"])
    end_1 = int(row.iloc[0]["end"])

    start0 = max(start_1 - 1, 0)
    end_excl = min(end_1, len(seq))

    if start0 >= len(seq) or end_excl <= start0:
        return ""

    return seq[start0:end_excl]


# -------------------------
# Cache utilities
# -------------------------
def load_cache_json(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())

def save_cache_json(path: str, cache: Dict[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True))


# -------------------------
# Core: compute sequences once per unique pMHC
# -------------------------
def compute_pmhc_fields_for_value(
    pmhc: str,
    trunc_df: pd.DataFrame,
    allele_fullseq_cache: Dict[str, str],
) -> Dict[str, str]:
    """
    Returns dict with:
      pMHC_hla, pMHC_peptide, pMHC_class, MHC_A_seq, MHC_B_seq
    computed ONCE per unique pMHC.
    """
    info = parse_pmhc_id(pmhc)
    hla = info["hla"]
    pep = info["peptide"]
    mhc_class = info["class"]

    out = {
        "pMHC_hla": hla,
        "pMHC_peptide": pep,
        "pMHC_class": str(mhc_class),
        "MHC_A_seq": "",
        "MHC_B_seq": "",
    }

    if mhc_class == 1:
        # fetch full heavy chain once per allele
        if hla not in allele_fullseq_cache:
            acc = fetch_accession_code(hla)
            allele_fullseq_cache[hla] = fetch_full_protein_sequence(acc)

        full = allele_fullseq_cache[hla]
        out["MHC_A_seq"] = truncate_with_table(hla, full, trunc_df)
        out["MHC_B_seq"] = HUMAN_B2M_SEQ
        return out

    # class II not reconstructed here (needs two alleles). Keep blanks or existing if you prefer.
    return out


# -------------------------
# Main
# -------------------------
def main():
    trunc_df = load_trunc_table(TRUNCATION_CSV)

    # persistent allele cache across files/runs
    allele_fullseq_cache = load_cache_json(CACHE_JSON)

    for fn in sorted(os.listdir(omics_processed_dir)):
        if not fn.endswith("augmented.tsv"):
            continue

        in_path = os.path.join(omics_processed_dir, fn)
        print(f"\n[INFO] Processing: {in_path}")

        df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)
        df.columns = df.columns.astype(str).str.strip()

        if pmhc_col not in df.columns:
            print(f"[WARN] Skipping {fn}: missing column {pmhc_col}")
            continue

        # Ensure output columns exist / reset if force
        out_cols = ["pMHC_hla", "pMHC_peptide", "pMHC_class", "MHC_A_seq", "MHC_B_seq"]
        for c in out_cols:
            if c not in df.columns:
                df[c] = ""
            elif force:
                df[c] = ""

        # Compute per unique pMHC (usually 1 per file)
        unique_pmhc = pd.Series(df[pmhc_col].astype(str)).unique().tolist()
        print(f"[INFO] unique pMHC values in file: {len(unique_pmhc)}")

        pmhc_to_fields: Dict[str, Dict[str, str]] = {}
        for pmhc in unique_pmhc:
            try:
                pmhc_to_fields[pmhc] = compute_pmhc_fields_for_value(pmhc, trunc_df, allele_fullseq_cache)
            except Exception as e:
                pmhc_to_fields[pmhc] = {
                    "pMHC_hla": "",
                    "pMHC_peptide": "",
                    "pMHC_class": "",
                    "MHC_A_seq": "",
                    "MHC_B_seq": "",
                }
                if "pmhc_rebuild_error" not in df.columns:
                    df["pmhc_rebuild_error"] = ""
                # set an error string for all rows matching this pmhc
                df.loc[df[pmhc_col].astype(str) == pmhc, "pmhc_rebuild_error"] = repr(e)

        # Map fields onto rows (vectorized)
        df["pMHC_hla"] = df[pmhc_col].map(lambda x: pmhc_to_fields.get(str(x), {}).get("pMHC_hla", ""))
        df["pMHC_peptide"] = df[pmhc_col].map(lambda x: pmhc_to_fields.get(str(x), {}).get("pMHC_peptide", ""))
        df["pMHC_class"] = df[pmhc_col].map(lambda x: pmhc_to_fields.get(str(x), {}).get("pMHC_class", ""))
        df["MHC_A_seq"] = df[pmhc_col].map(lambda x: pmhc_to_fields.get(str(x), {}).get("MHC_A_seq", ""))
        df["MHC_B_seq"] = df[pmhc_col].map(lambda x: pmhc_to_fields.get(str(x), {}).get("MHC_B_seq", ""))

        # Write in place (your previous behavior)
        df.to_csv(in_path, sep="\t", index=False)
        print(f"[OK] Wrote updated file in-place: {in_path}")

    save_cache_json(CACHE_JSON, allele_fullseq_cache)
    print(f"\n[OK] Saved allele full-seq cache: {CACHE_JSON} (n={len(allele_fullseq_cache)})")


if __name__ == "__main__":
    main()
