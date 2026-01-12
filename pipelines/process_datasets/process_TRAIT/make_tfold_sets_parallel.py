#!/usr/bin/env python3
"""
Run tFold structure generation for ALL pMHC datasets in:
  /mnt/larry/lilian/DATA/TRAIT/omics_processed

Rules:
- Each pMHC target is inferred from filenames like:
    <PMHC_TARGET>_binder_pos.augmented.tsv
    <PMHC_TARGET>_binder_neg.augmented.tsv
- For each PMHC_TARGET we create:
    /mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/<PMHC_TARGET>/<subset>/{pos,neg}/
  where subset in {random, germline_balanced, highest_count}

- If a PMHC_TARGET already finished, we skip it (DONE.txt marker).

Sampling / sizing:
- Positive target size = min(N_PER_CLASS, n_pos_candidates)
- Negative target size = min(N_PER_CLASS, n_neg_candidates)   (safe; avoids infinite fill)

Speed:
- Uses multi-GPU: one worker process per GPU (pinned with CUDA_VISIBLE_DEVICES).
- Worker loads tFold once and moves predictor to CUDA via .to("cuda").

Outputs per (subset,pos/neg):
- time.txt, fails.txt, tfold_inputs.csv
- PDB files (exactly target_n if possible; warns if not enough candidates)
"""

import os
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from anarci import anarci

# -----------------------
# Config
# -----------------------
N_PER_CLASS = 500
RANDOM_SEED = 7

omics_processed_dir = "/mnt/larry/lilian/DATA/TRAIT/omics_processed"
OUT_ROOT = "/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets"

# GPUs to use (one worker per GPU)
TFOLD_GPU_IDS = [0, 1]  # set to [0] if only one GPU

human_beta2m_sequence = (
    "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSREPAENGKPNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
)

# file pattern
AUG_RE = re.compile(r"^(?P<pmhc>.+)_binder_(?P<label>pos|neg)\.augmented\.tsv$")

TCR_ID_COLS = ["TRAV", "TRAJ", "CDR3a", "TRBV", "TRBD", "TRBJ", "CDR3b"]
GERMLINE_COLS = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
REQUIRED_TFOLD_COLS = [
    "Donor", "pMHC", "Binding", "Count", "TCR_name",
    "TRA_full_aa", "TRB_full_aa", "MHC_A_seq", "MHC_B_seq", "pMHC_peptide",
] + TCR_ID_COLS


# -----------------------
# Small utilities
# -----------------------

def normalize_pmhc(x: str) -> str:
    x = str(x).strip()
    # remove common suffixes
    x = x.replace("_binder", "")
    x = x.replace("_binder_pos", "")
    x = x.replace("_binder_neg", "")
    return x


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+-]", "_", s)
    return s

def clean_seq(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return None
    s = s.rstrip(".")
    return s if s else None

def _clean_protein_seq(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().replace(" ", "").replace("\n", "").upper()
    s = s.rstrip(".").replace("*", "")
    return s if s else None

def is_b2m_sequence(seq: str) -> bool:
    s = _clean_protein_seq(seq)
    if not s:
        return False
    b2m = _clean_protein_seq(human_beta2m_sequence)
    if s == b2m:
        return True
    if len(s) == len(b2m):
        ident = sum(a == b for a, b in zip(s, b2m)) / len(s)
        return ident > 0.95
    return False


# -----------------------
# Load / prep augmented
# -----------------------
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

    for c in TCR_ID_COLS:
        df[c] = df[c].astype("string").fillna("")

    if df.empty:
        df["tcr_key"] = pd.Series([], index=df.index, dtype="string")
        df["germline_key"] = pd.Series([], index=df.index, dtype="string")
        return df

    df["tcr_key"] = df[TCR_ID_COLS].agg("|".join, axis=1).astype("string")
    df["germline_key"] = df[GERMLINE_COLS].astype("string").fillna("").agg("+".join, axis=1).astype("string")

    for c in ["TRA_full_aa", "TRB_full_aa", "MHC_A_seq", "MHC_B_seq", "pMHC_peptide"]:
        df[c] = df[c].apply(clean_seq)

    return df

def deduplicate_by_tcr_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("Count", ascending=False)
    return df.drop_duplicates(subset=["tcr_key"], keep="first")


# -----------------------
# Candidate ordering (not truncating to N)
# -----------------------
def order_random(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sample(frac=1.0, replace=False, random_state=seed)

def order_top_count(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("Count", ascending=False)

def order_round_robin(df: pd.DataFrame, group_col: str, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    rng = np.random.default_rng(seed)

    grouped = {}
    for g, gdf in df.groupby(group_col, dropna=False):
        gdf = gdf.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        grouped[g] = list(gdf.index)

    groups = list(grouped.keys())
    rng.shuffle(groups)

    picked = []
    exhausted = set()
    while True:
        progressed = False
        for g in groups:
            if g in exhausted:
                continue
            if grouped[g]:
                picked.append(grouped[g].pop(0))
                progressed = True
            else:
                exhausted.add(g)
        if not progressed:
            break

    return df.loc[picked].copy()


# -------------------------
# time + manifest
# -------------------------
def _time_log_path(output_dir: str) -> str:
    return os.path.join(output_dir, "time.txt")

def init_time_log(output_dir: str):
    p = _time_log_path(output_dir)
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write("timestamp\ttfold_id\tstatus\tseconds\tnote\n")

def log_time(output_dir: str, tfold_id: str, status: str, seconds: float, note: str = ""):
    init_time_log(output_dir)
    ts = datetime.now().isoformat(timespec="seconds")
    note = (note or "").replace("\n", " ").replace("\t", " ")
    with open(_time_log_path(output_dir), "a") as f:
        f.write(f"{ts}\t{tfold_id}\t{status}\t{seconds:.6f}\t{note}\n")

def save_manifest(output_dir: str, rows: list[dict]):
    if not rows:
        return
    out_csv = os.path.join(output_dir, "tfold_inputs.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -------------------------
# ANARCI truncation (cached in-process)
# -------------------------
_ANARCI_CACHE_ALPHA = {}
_ANARCI_CACHE_BETA = {}

def truncate_variable_anarci_imgt(seq: str, cache: dict) -> str:
    # seq should already be cleaned
    if not seq:
        return ""
    if seq in cache:
        return cache[seq]

    numbering, alignment_details, hit_tables = anarci([("q", seq)], scheme="imgt", output=False)
    if not numbering or numbering[0] is None:
        cache[seq] = ""
        return ""

    try:
        num_list = numbering[0][0][0]
        ad = alignment_details[0][0]
        query_start = int(ad["query_start"])
        query_end = int(ad["query_end"])
    except Exception:
        cache[seq] = ""
        return ""

    max_imgt = 0
    for item in num_list:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        pos_info, aa = item
        aa = str(aa)
        if aa in {"-", ""}:
            continue
        if not isinstance(pos_info, (list, tuple)) or len(pos_info) < 1:
            continue
        pos = pos_info[0]
        if pos is None:
            continue
        try:
            max_imgt = max(max_imgt, int(pos))
        except Exception:
            pass

    extend = max(0, 128 - max_imgt)
    start0 = max(query_start, 0)
    end_excl = min(len(seq), query_end + 1 + extend)
    var = seq[start0:end_excl]

    cache[seq] = var
    return var

def process_sequences(seq_tcr_alpha, seq_tcr_beta, seq_mhc_heavy, seq_mhc_light, seq_peptide):
    a = _clean_protein_seq(seq_tcr_alpha)
    b = _clean_protein_seq(seq_tcr_beta)
    mhc = _clean_protein_seq(seq_mhc_heavy)
    mhc_light = _clean_protein_seq(seq_mhc_light)
    pep = _clean_protein_seq(seq_peptide)

    if a:
        va = truncate_variable_anarci_imgt(a, _ANARCI_CACHE_ALPHA)
        if va:
            a = va
    if b:
        vb = truncate_variable_anarci_imgt(b, _ANARCI_CACHE_BETA)
        if vb:
            b = vb

    if mhc_light and is_b2m_sequence(mhc_light):
        mhc_light = None

    return a, b, mhc, mhc_light, pep


# -----------------------
# Worker: GPU-safe, one per GPU
# -----------------------
_TFOLD_MODEL = None

def _tfold_worker_init(gpu_id: int):
    global _TFOLD_MODEL
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    import tfold

    ppi_model_path = tfold.model.esm_ppi_650m_tcr()
    tfold_model_path = tfold.model.tfold_tcr_pmhc_trunk()
    _TFOLD_MODEL = tfold.deploy.TCRpMHCPredictor(ppi_model_path, tfold_model_path).to("cuda")

    print(
        f"[WORKER pinned={gpu_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"cuda={torch.cuda.is_available()} count={torch.cuda.device_count()} current={torch.cuda.current_device()}"
    )

def build_tfold_input_for_pdb(B_seq, A_seq, M_seq, N_seq, P_seq):
    data = [
        {"id": "B", "sequence": B_seq},
        {"id": "A", "sequence": A_seq},
        {"id": "M", "sequence": M_seq},
    ]
    if N_seq is not None:
        data.append({"id": "N", "sequence": N_seq})
    data.append({"id": "P", "sequence": P_seq})
    return data

def _tfold_worker_run(task: dict):
    global _TFOLD_MODEL

    tfold_id = task["tfold_id"]
    output_pdb = task["output_pdb"]
    t0 = time.perf_counter()

    if os.path.exists(output_pdb):
        return {"tfold_id": tfold_id, "status": "skip_exists", "seconds": time.perf_counter() - t0, "note": "pdb_exists"}

    A = task["A_seq"]; B = task["B_seq"]; M = task["M_seq"]; N = task["N_seq"]; P = task["P_seq"]
    if not (A and B and M and P):
        return {"tfold_id": tfold_id, "status": "skip_missing_seq", "seconds": time.perf_counter() - t0, "note": "missing_seq_after_processing"}

    try:
        data = build_tfold_input_for_pdb(B, A, M, N, P)
        _TFOLD_MODEL.infer_pdb(data, output_pdb)
        return {"tfold_id": tfold_id, "status": "ok", "seconds": time.perf_counter() - t0, "note": ""}
    except Exception as e:
        return {"tfold_id": tfold_id, "status": "fail", "seconds": time.perf_counter() - t0, "note": repr(e)[:300]}


def _count_pdbs(output_dir: str) -> int:
    return len([fn for fn in os.listdir(output_dir) if fn.endswith(".pdb")])


def run_fill_to_target(output_dir: str, df_candidates: pd.DataFrame, target_n: int, gpu_ids):
    ensure_dir(output_dir)
    init_time_log(output_dir)
    fails_file = os.path.join(output_dir, "fails.txt")

    existing = _count_pdbs(output_dir)
    if existing >= target_n:
        print(f"[INFO] {output_dir}: already has {existing} pdbs >= target {target_n}, skipping.")
        return

    gpu_ids = list(gpu_ids)

    executors = {
        gid: ProcessPoolExecutor(max_workers=1, initializer=_tfold_worker_init, initargs=(gid,))
        for gid in gpu_ids
    }

    manifest_rows = []
    ok_count = existing
    next_idx = 0
    inflight = set()
    inflight_meta = {}

    def submit_one(slot: int) -> bool:
        nonlocal next_idx
        if next_idx >= len(df_candidates):
            return False

        row = df_candidates.iloc[next_idx]
        next_idx += 1

        binding = row["Binding"]
        donor_raw = row["Donor"]
        tcr_name_raw = row["TCR_name"]
        pmhc = row["pMHC"]
        count = row.get("Count", "")

        donor = sanitize_name(donor_raw)
        tcr_name = sanitize_name(tcr_name_raw)

        tfold_id = f"{binding}_{donor}_{tcr_name}_cand{next_idx:06d}"
        output_pdb = os.path.join(output_dir, tfold_id + ".pdb")

        try:
            a, b, mhc, mhc_light, pep = process_sequences(
                row["TRA_full_aa"], row["TRB_full_aa"], row["MHC_A_seq"], row["MHC_B_seq"], row["pMHC_peptide"]
            )
        except Exception:
            a = b = mhc = mhc_light = pep = None

        mrow = {
            "tfold_id": tfold_id,
            "output_pdb": output_pdb,
            "Binding": binding,
            "Donor": donor_raw,
            "TCR_name": tcr_name_raw,
            "pMHC": pmhc,
            "Count": count,
            "TCR_alpha_seq_tfold": a,
            "TCR_beta_seq_tfold": b,
            "MHC_heavy_seq_tfold": mhc,
            "MHC_light_seq_tfold": mhc_light,
            "peptide_seq_tfold": pep,
        }

        task = {"tfold_id": tfold_id, "output_pdb": output_pdb, "A_seq": a, "B_seq": b, "M_seq": mhc, "N_seq": mhc_light, "P_seq": pep}
        gid = gpu_ids[slot % len(gpu_ids)]
        fut = executors[gid].submit(_tfold_worker_run, task)
        inflight.add(fut)
        inflight_meta[fut] = mrow
        return True

    try:
        # keep only len(gpu_ids) in flight (prevents overshoot and keeps logic simple)
        for k in range(len(gpu_ids)):
            if ok_count >= target_n:
                break
            if not submit_one(k):
                break

        while inflight:
            fut = next(as_completed(inflight))
            inflight.remove(fut)
            mrow = inflight_meta.pop(fut)
            res = fut.result()

            status = res["status"]
            seconds = res["seconds"]
            note = res.get("note", "")

            log_time(output_dir, res["tfold_id"], status, seconds, note)

            kept = False
            pruned = False

            if status == "ok":
                if ok_count < target_n:
                    ok_count += 1
                    kept = True
                else:
                    # prune extras (rare because we only keep <=num_gpus in flight)
                    try:
                        if os.path.exists(mrow["output_pdb"]):
                            os.remove(mrow["output_pdb"])
                    except Exception:
                        pass
                    pruned = True

            mrow["status"] = status
            mrow["seconds"] = seconds
            mrow["note"] = note
            mrow["kept_in_final_set"] = kept
            mrow["pruned_extra_ok"] = pruned
            manifest_rows.append(mrow)

            if status in {"fail", "skip_missing_seq"}:
                with open(fails_file, "a") as f:
                    f.write(f"{res['tfold_id']}\t{note}\n")

            if ok_count < target_n:
                submit_one(next_idx)

        save_manifest(output_dir, manifest_rows)

        if ok_count < target_n:
            print(f"[WARN] {output_dir}: reached {ok_count}/{target_n}. Ran out of candidates ({len(df_candidates)}).")
        else:
            print(f"[OK] {output_dir}: reached {ok_count}/{target_n} successes.")

    finally:
        for ex in executors.values():
            ex.shutdown(wait=True)


def discover_pmhc_targets(base_dir: str) -> dict:
    """
    Returns dict: pmhc_target -> {"pos": [files], "neg": [files]}
    Based on filenames like <PMHC>_binder_pos.augmented.tsv
    """
    out = {}
    for fn in os.listdir(base_dir):
        m = AUG_RE.match(fn)
        if not m:
            continue
        pmhc = m.group("pmhc")
        lbl = m.group("label")
        out.setdefault(pmhc, {"pos": [], "neg": []})
        out[pmhc][lbl].append(os.path.join(base_dir, fn))
    # sort lists
    for pmhc in out:
        out[pmhc]["pos"] = sorted(out[pmhc]["pos"])
        out[pmhc]["neg"] = sorted(out[pmhc]["neg"])
    return out


def already_done(out_base: str) -> bool:
    return os.path.exists(os.path.join(out_base, "DONE.txt"))

def mark_done(out_base: str):
    with open(os.path.join(out_base, "DONE.txt"), "w") as f:
        f.write(datetime.now().isoformat(timespec="seconds") + "\n")


def main():
    ensure_dir(OUT_ROOT)

    pmhc_to_files = discover_pmhc_targets(omics_processed_dir)
    if not pmhc_to_files:
        raise RuntimeError(f"No '*_binder_(pos|neg).augmented.tsv' files found in {omics_processed_dir}")

    pmhc_targets = sorted(pmhc_to_files.keys())
    print(f"[INFO] Found {len(pmhc_targets)} pMHC targets in folder.")

    for pmhc_target in pmhc_targets:
        out_base = os.path.join(OUT_ROOT, sanitize_name(pmhc_target))
        ensure_dir(out_base)

        if already_done(out_base):
            print(f"[SKIP] {pmhc_target}: already done (DONE.txt exists).")
            continue

        pos_files = pmhc_to_files[pmhc_target]["pos"]
        neg_files = pmhc_to_files[pmhc_target]["neg"]

        if len(pos_files) == 0 or len(neg_files) == 0:
            print(f"[WARN] {pmhc_target}: missing pos or neg files. pos={len(pos_files)} neg={len(neg_files)}. Skipping.")
            continue

        # Load all pos+neg for this pMHC
        frames = []
        for f in pos_files + neg_files:
            frames.append(read_augmented(f))
        aug = pd.concat(frames, ignore_index=True)
        # Some safety filter in case (normalize both sides)
        target_norm = normalize_pmhc(pmhc_target)

        pmhc_series = aug["pMHC"].astype(str).map(normalize_pmhc)
        aug = aug[pmhc_series == target_norm].copy()

        if aug.empty:
            # optional: show a few unique values for debugging
            print(f"[WARN] {pmhc_target}: no rows after filtering. Example pMHC values: {aug['pMHC'].astype(str).head(3).tolist()}")
            continue


        pos_df = deduplicate_by_tcr_key(aug[aug["binding_class"] == "pos"].copy())
        neg_df = deduplicate_by_tcr_key(aug[aug["binding_class"] == "neg"].copy())

        print(f"\n=== {pmhc_target} ===")
        print(f"pos unique TCRs: {len(pos_df)} | neg unique TCRs: {len(neg_df)}")

        # If not enough positive samples, use all positives (your request).
        target_pos = min(N_PER_CLASS, len(pos_df))
        target_neg = min(N_PER_CLASS, len(neg_df))

        # Candidate pools per subset strategy (not truncated; fill-to-target handles failures)
        subsets = {
            "random": {
                "pos": order_random(pos_df, RANDOM_SEED),
                "neg": order_random(neg_df, RANDOM_SEED),
            },
            "germline_balanced": {
                "pos": order_round_robin(pos_df, "germline_key", RANDOM_SEED),
                "neg": order_round_robin(neg_df, "germline_key", RANDOM_SEED),
            },
            "highest_count": {
                "pos": order_top_count(pos_df),
                "neg": order_top_count(neg_df),
            },
        }

        for subset_name, d in subsets.items():
            subset_dir = os.path.join(out_base, sanitize_name(subset_name))
            pos_dir = os.path.join(subset_dir, "pos")
            neg_dir = os.path.join(subset_dir, "neg")
            ensure_dir(pos_dir); ensure_dir(neg_dir)

            # Save candidate pools for provenance
            d["pos"].to_csv(os.path.join(subset_dir, "candidates_pos.csv"), index=False)
            d["neg"].to_csv(os.path.join(subset_dir, "candidates_neg.csv"), index=False)

            print(f"[INFO] subset={subset_name} target_pos={target_pos} target_neg={target_neg} GPUs={TFOLD_GPU_IDS}")

            run_fill_to_target(pos_dir, d["pos"], target_n=target_pos, gpu_ids=TFOLD_GPU_IDS)
            run_fill_to_target(neg_dir, d["neg"], target_n=target_neg, gpu_ids=TFOLD_GPU_IDS)

        mark_done(out_base)
        print(f"[DONE] {pmhc_target} written under {out_base}")

    print("\nAll done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
