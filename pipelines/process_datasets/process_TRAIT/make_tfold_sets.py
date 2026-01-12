#!/usr/bin/env python3
"""
TRAIT OMICS proof-of-concept for a single pMHC:
  - Load augmented pos/neg files
  - Filter to a target pMHC (exact match if available; otherwise prefix/contains fallback)
  - Create three 500-vs-500 subsets (pos and neg separately):
      1) random
      2) germline-balanced (TRAV+TRAJ+TRBV+TRBJ)
      3) highest Count
  - For each subset, create:
      <OUT_BASE>/<subset_name>/{pos,neg}/
    and run tFold to generate predicted complex PDBs into those folders.

Assumptions:
  - Augmented TSVs contain:
      Donor, pMHC, Binding, Count, TCR_name,
      TRA_full_aa, TRB_full_aa, MHC_A_seq, MHC_B_seq, pMHC_peptide
  - Binding labels are exactly: "Binding" or "Non-binding"
"""

import os
import re
import numpy as np
import pandas as pd
from anarci import anarci
import time
from datetime import datetime

# tFold
import tfold

human_beta2m_sequence="MSRSVALAVLALLSLSGLEAIQRTPKIQVYSREPAENGKPNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"

# -----------------------
# User config
# -----------------------
PMHC_TARGET = "A0301_KLGGALQAK"
N_PER_CLASS = 500
RANDOM_SEED = 7

omics_processed_dir = "/mnt/larry/lilian/DATA/TRAIT/omics_processed"
OUT_BASE = "/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/A0301_KLGGALQAK"

# If you want to restrict which augmented files to read, set this to True.
# It will only read files whose filename contains PMHC_TARGET.
FILTER_FILES_BY_FILENAME = True


# -----------------------
# Definitions
# -----------------------
TCR_ID_COLS = ["TRAV", "TRAJ", "CDR3a", "TRBV", "TRBD", "TRBJ", "CDR3b"]
GERMLINE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]

REQUIRED_TFOLD_COLS = [
    "Donor", "pMHC", "Binding", "Count", "TCR_name",
    "TRA_full_aa", "TRB_full_aa", "MHC_A_seq", "MHC_B_seq", "pMHC_peptide",
] + TCR_ID_COLS


# -----------------------
# Utilities
# -----------------------
def sanitize_name(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+-]", "_", s)
    return s


def clean_seq(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    # Some exports append "." at end of sequence
    s = s.rstrip(".")
    if s == "":
        return None
    return s


def list_augmented_files(base_dir: str, pmhc_target: str, filter_by_filename: bool = True):
    pos_files, neg_files = [], []
    for fn in os.listdir(base_dir):
        if not (fn.endswith("_binder_pos.augmented.tsv") or fn.endswith("_binder_neg.augmented.tsv")):
            continue
        if filter_by_filename and (pmhc_target not in fn):
            continue
        p = os.path.join(base_dir, fn)
        if fn.endswith("_binder_pos.augmented.tsv"):
            pos_files.append(p)
        else:
            neg_files.append(p)
    return sorted(pos_files), sorted(neg_files)


def read_augmented(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    for c in REQUIRED_TFOLD_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df = df[REQUIRED_TFOLD_COLS].copy()

    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    df["binding_class"] = df["Binding"].map({"Binding": "pos", "Non-binding": "neg"})
    df = df.dropna(subset=["Donor", "pMHC", "binding_class"])

    # keys
    for c in TCR_ID_COLS:
        df[c] = df[c].astype("string").fillna("")

    if df.empty:
        df["tcr_key"] = pd.Series([], index=df.index, dtype="string")
        df["germline_key"] = pd.Series([], index=df.index, dtype="string")
        return df

    df["tcr_key"] = df[TCR_ID_COLS].agg("|".join, axis=1).astype("string")
    df["germline_key"] = df[GERMLINE_COLS].astype("string").fillna("").agg("+".join, axis=1).astype("string")

    # Clean sequences (for robustness)
    for c in ["TRA_full_aa", "TRB_full_aa", "MHC_A_seq", "MHC_B_seq", "pMHC_peptide"]:
        df[c] = df[c].apply(clean_seq)

    return df


def filter_to_pmhc(df: pd.DataFrame, pmhc_target: str) -> pd.DataFrame:
    """
    Prefer exact match. If none, fall back to prefix match. If still none, fall back to contains.
    """
    if df.empty:
        return df

    exact = df[df["pMHC"] == pmhc_target]
    if len(exact) > 0:
        return exact

    prefix = df[df["pMHC"].astype(str).str.startswith(pmhc_target)]
    if len(prefix) > 0:
        return prefix

    contains = df[df["pMHC"].astype(str).str.contains(re.escape(pmhc_target), regex=True)]
    return contains


def deduplicate_by_tcr_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate within a class, keeping the highest Count representative per unique tcr_key.
    """
    if df.empty:
        return df
    df = df.sort_values("Count", ascending=False)
    return df.drop_duplicates(subset=["tcr_key"], keep="first")


def sample_random(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    n = min(n, len(df))
    return df.sample(n=n, replace=False, random_state=seed)


def sample_top_count(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("Count", ascending=False).head(n)


def sample_round_robin_balanced(df: pd.DataFrame, group_col: str, n: int, seed: int) -> pd.DataFrame:
    """
    Balanced sampling across groups:
      - shuffle within each group
      - iterate groups in shuffled order, picking 1 at a time until n reached or exhausted
    Produces near-uniform allocation across observed germlines.
    """
    if df.empty:
        return df

    rng = np.random.default_rng(seed)
    grouped = {}
    for g, gdf in df.groupby(group_col, dropna=False):
        gdf = gdf.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))  # shuffle
        grouped[g] = list(gdf.index)

    groups = list(grouped.keys())
    rng.shuffle(groups)

    picked = []
    exhausted = set()

    while len(picked) < n:
        progressed = False
        for g in groups:
            if g in exhausted:
                continue
            if grouped[g]:
                picked.append(grouped[g].pop(0))
                progressed = True
                if len(picked) >= n:
                    break
            else:
                exhausted.add(g)
        if not progressed:
            break  # all groups exhausted

    return df.loc[picked].copy()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

#-------------------------
#time helpers
#-------------------------
def _time_log_path(output_dir: str) -> str:
    return os.path.join(output_dir, "time.txt")


def init_time_log(output_dir: str):
    """
    Ensure time.txt exists with a header.
    """
    p = _time_log_path(output_dir)
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write("timestamp\ttfold_id\tstatus\tseconds\tnote\n")


def log_time(output_dir: str, tfold_id: str, status: str, seconds: float, note: str = ""):
    """
    Append one timing record to time.txt in output_dir.
    """
    init_time_log(output_dir)
    ts = datetime.now().isoformat(timespec="seconds")
    # keep note single-line
    note = (note or "").replace("\n", " ").replace("\t", " ")
    with open(_time_log_path(output_dir), "a") as f:
        f.write(f"{ts}\t{tfold_id}\t{status}\t{seconds:.6f}\t{note}\n")


# ------------------------
# anarci helpers to extract variable
# -------------------------
def _clean_protein_seq(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().replace(" ", "").replace("\n", "").upper()
    s = s.rstrip(".")
    s = s.replace("*", "")  # remove stop
    return s if s else None

def truncate_variable_anarci_imgt(seq: str, cache: dict) -> str:
    """
    Return variable-domain substring corresponding to IMGT positions 1..128 using ANARCI.

    Uses alignment_details["query_start"] and ["query_end"] (0-indexed, inclusive) to slice
    the original input sequence, then extends the slice if ANARCI numbered only up to <128.

    This matches your observation:
      - IMGT pos 1 corresponds to input position query_start (0-indexed)
      - If max IMGT pos present is 127, extend by 1 residue beyond query_end.

    Returns "" if ANARCI fails.
    """
    seq = _clean_protein_seq(seq)
    if not seq:
        return ""

    if seq in cache:
        return cache[seq]

    # Run ANARCI
    numbering, alignment_details, hit_tables = anarci([("q", seq)], scheme="imgt", output=False)

    # Guard: no hit
    if not numbering or numbering[0] is None:
        cache[seq] = ""
        return ""

    # Your ANARCI output structure (as in your other script):
    # numbering[0][0][0] is the numbering list: [((imgt_pos, ins), aa), ...]
    try:
        num_list = numbering[0][0][0]
    except Exception:
        cache[seq] = ""
        return ""

    # alignment_details[0][0] is a dict with query_start/query_end
    try:
        ad = alignment_details[0][0]
        query_start = int(ad["query_start"])   # 0-indexed
        query_end = int(ad["query_end"])       # 0-indexed, inclusive
    except Exception:
        cache[seq] = ""
        return ""

    # Determine the max IMGT position actually present (excluding gaps '-')
    max_imgt = 0
    for item in num_list:
        # item expected: ((pos, ins), aa)
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        pos_info, aa = item
        if aa is None:
            continue
        aa = str(aa)
        if aa == "-" or aa == "":
            continue
        if not isinstance(pos_info, (list, tuple)) or len(pos_info) < 1:
            continue
        pos = pos_info[0]
        if pos is None:
            continue
        try:
            pos_i = int(pos)
        except Exception:
            continue
        if pos_i > max_imgt:
            max_imgt = pos_i

    # Extend to IMGT 128 if needed
    extend = max(0, 128 - max_imgt)

    # Slice original sequence:
    # query_end is inclusive, so python end_excl = query_end + 1
    start0 = max(query_start, 0)
    end_excl = min(len(seq), query_end + 1 + extend)

    var = seq[start0:end_excl]

    cache[seq] = var
    return var


# caches so we don't ANARCI the same sequence thousands of times
_ANARCI_CACHE_ALPHA = {}
_ANARCI_CACHE_BETA = {}


# -----------------------
# tFold helpers
# -----------------------
def build_tfold_input_for_pdb(seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide):
    data = [
        {"id": "B", "sequence": seq_tcr_beta},   # TCR beta / delta
        {"id": "A", "sequence": seq_tcr_alpha},  # TCR alpha / gamma
        {"id": "M", "sequence": seq_mhc_heavy},  # MHC heavy / alpha
    ]
    if seq_mhc_light is not None:
        data.append({"id": "N", "sequence": seq_mhc_light})  # optional MHC light / beta
    data.append({"id": "P", "sequence": seq_peptide})        # peptide
    return data


def load_tfold_model():
    print("Loading tFold-TCR (TCRpMHCPredictor) model...")
    ppi_model_path = tfold.model.esm_ppi_650m_tcr()
    tfold_model_path = tfold.model.tfold_tcr_pmhc_trunk()
    model = tfold.deploy.TCRpMHCPredictor(ppi_model_path, tfold_model_path)
    return model

def process_sequences(seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide):
    """
    On-the-fly preprocessing for tFold:
      - Clean sequences
      - Truncate TCR alpha/beta to variable domain only using ANARCI (IMGT 1..128)
      - Keep MHC sequences as provided (already truncated upstream)
      - Optionally drop B2M chain (set seq_mhc_light=None) if you want (toggle below)
    """

    # ---- Clean everything ----
    seq_tcr_alpha = _clean_protein_seq(seq_tcr_alpha)
    seq_tcr_beta  = _clean_protein_seq(seq_tcr_beta)
    seq_mhc_heavy = _clean_protein_seq(seq_mhc_heavy)
    seq_mhc_light = _clean_protein_seq(seq_mhc_light)
    seq_peptide   = _clean_protein_seq(seq_peptide)

    # ---- Truncate TCRs to variable domain (ANARCI) ----
    # If ANARCI fails for a sequence, we fall back to the original sequence (so you still get a model run),
    # but we log it so you can inspect failure rates.
    if seq_tcr_alpha:
        try:
            trunc_a = truncate_variable_anarci_imgt(seq_tcr_alpha, _ANARCI_CACHE_ALPHA)
            if trunc_a:
                seq_tcr_alpha = trunc_a
            else:
                print("[WARN] ANARCI returned empty alpha variable domain; using full alpha sequence.")
        except Exception as e:
            print(f"[WARN] ANARCI failed for alpha; using full alpha sequence. Error: {e}")

    if seq_tcr_beta:
        try:
            trunc_b = truncate_variable_anarci_imgt(seq_tcr_beta, _ANARCI_CACHE_BETA)
            if trunc_b:
                seq_tcr_beta = trunc_b
            else:
                print("[WARN] ANARCI returned empty beta variable domain; using full beta sequence.")
        except Exception as e:
            print(f"[WARN] ANARCI failed for beta; using full beta sequence. Error: {e}")

    # ---- Optional: omit B2M chain if present ----
    # Toggle this depending on what tFold expects in your setup.
    DROP_B2M = True
    if DROP_B2M and seq_mhc_light:
        # If it's exactly the B2M sequence you used upstream, drop it
        if seq_mhc_light == human_beta2m_sequence:
            seq_mhc_light = None
    if seq_tcr_alpha==None or seq_tcr_beta==None or seq_mhc_heavy==None or seq_peptide==None:
        raise RuntimeError(f"one sequence is none")
    return seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide



def run_tfold_for_df(model, output_dir: str, df: pd.DataFrame):
    """
    Generates PDBs for each row in df into output_dir.
    Writes failures to fails.txt inside output_dir.
    Also writes timing per attempted sample to time.txt inside output_dir.
    """
    ensure_dir(output_dir)
    fails_file = os.path.join(output_dir, "fails.txt")

    # make sure time.txt exists
    init_time_log(output_dir)

    for i, row in df.iterrows():
        binding = row["Binding"]  # "Binding" or "Non-binding"
        donor = sanitize_name(row["Donor"])
        tcr_name = sanitize_name(row["TCR_name"])

        tfold_id = f"{binding}_{donor}_{tcr_name}_idx{i}"
        output_pdb = os.path.join(output_dir, tfold_id + ".pdb")

        # Start timer for the *full* per-row processing (including truncation + infer_pdb)
        t0 = time.perf_counter()

        # If already done, optionally log and continue
        if os.path.exists(output_pdb):
            dt = time.perf_counter() - t0
            log_time(output_dir, tfold_id, "skip_exists", dt, note="pdb_exists")
            continue

        # Pull sequences
        seq_tcr_alpha = row["TRA_full_aa"]
        seq_tcr_beta = row["TRB_full_aa"]
        seq_mhc_heavy = row["MHC_A_seq"]
        seq_mhc_light = row["MHC_B_seq"]  # can be None
        seq_peptide = row["pMHC_peptide"]

        try:
            # includes ANARCI truncation + any cleaning you do
            seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide = process_sequences(
                seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide
            )

            # Basic validation
            if not (seq_tcr_alpha and seq_tcr_beta and seq_mhc_heavy and seq_peptide):
                dt = time.perf_counter() - t0
                log_time(output_dir, tfold_id, "skip_missing_seq", dt, note="missing_seq_after_processing")
                with open(fails_file, "a") as f:
                    f.write(f"missing_seq\tindex={i}\t{tfold_id}\n")
                continue

            data = build_tfold_input_for_pdb(
                seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide
            )

            # tFold inference
            model.infer_pdb(data, output_pdb)

            dt = time.perf_counter() - t0
            log_time(output_dir, tfold_id, "ok", dt, note="")

        except Exception as e:
            dt = time.perf_counter() - t0
            log_time(output_dir, tfold_id, "fail", dt, note=repr(e)[:300])
            with open(fails_file, "a") as f:
                f.write(f"{tfold_id}\t{repr(e)}\n")



# -----------------------
# Main
# -----------------------
def main():
    ensure_dir(OUT_BASE)

    pos_files, neg_files = list_augmented_files(
        omics_processed_dir, PMHC_TARGET, filter_by_filename=FILTER_FILES_BY_FILENAME
    )

    # Fallback: if filename-filtering finds nothing, load all augmented files
    if len(pos_files) == 0 and len(neg_files) == 0:
        print("No files matched by filename. Falling back to loading all augmented files.")
        pos_files, neg_files = list_augmented_files(
            omics_processed_dir, pmhc_target="", filter_by_filename=False
        )

    # Load
    frames = []
    for f in pos_files + neg_files:
        frames.append(read_augmented(f))
        print("loaded:", os.path.basename(f))

    aug = pd.concat(frames, ignore_index=True)
    aug = filter_to_pmhc(aug, PMHC_TARGET)

    if aug.empty:
        raise RuntimeError(f"No rows found for pMHC target '{PMHC_TARGET}' after filtering.")

    # Split and deduplicate by TCR identity
    pos_df = deduplicate_by_tcr_key(aug[aug["binding_class"] == "pos"].copy())
    neg_df = deduplicate_by_tcr_key(aug[aug["binding_class"] == "neg"].copy())

    print(f"After filter+dedup: pos={len(pos_df)} unique TCRs, neg={len(neg_df)} unique TCRs")

    # Build subsets (each returns up to N_PER_CLASS rows)
    subsets = {}

    # 1) random
    subsets["random"] = {
        "pos": sample_random(pos_df, N_PER_CLASS, RANDOM_SEED),
        "neg": sample_random(neg_df, N_PER_CLASS, RANDOM_SEED),
    }

    # 2) germline-balanced (round-robin across germline_key)
    subsets["germline_balanced"] = {
        "pos": sample_round_robin_balanced(pos_df, "germline_key", N_PER_CLASS, RANDOM_SEED),
        "neg": sample_round_robin_balanced(neg_df, "germline_key", N_PER_CLASS, RANDOM_SEED),
    }

    # 3) highest Count
    subsets["highest_count"] = {
        "pos": sample_top_count(pos_df, N_PER_CLASS),
        "neg": sample_top_count(neg_df, N_PER_CLASS),
    }

    # Save manifests + run tFold
    model = load_tfold_model()

    for subset_name, d in subsets.items():
        subset_dir = os.path.join(OUT_BASE, sanitize_name(subset_name))
        pos_dir = os.path.join(subset_dir, "pos")
        neg_dir = os.path.join(subset_dir, "neg")
        ensure_dir(pos_dir)
        ensure_dir(neg_dir)

        # Save manifests
        d["pos"].to_csv(os.path.join(subset_dir, "selected_pos.csv"), index=False)
        d["neg"].to_csv(os.path.join(subset_dir, "selected_neg.csv"), index=False)

        print(f"\n=== tFold subset: {subset_name} ===")
        print(f"pos: {len(d['pos'])} | neg: {len(d['neg'])} | out: {subset_dir}")

        # Run tFold into class folders
        run_tfold_for_df(model, pos_dir, d["pos"])
        run_tfold_for_df(model, neg_dir, d["neg"])

    print("\nDone.")
    print("Outputs:", OUT_BASE)


if __name__ == "__main__":
    main()
