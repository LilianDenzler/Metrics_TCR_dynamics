#!/usr/bin/env python3
# precompute_mmseqs_gpu_then_openfold.py

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


# ---------- small utils ----------

def _run(cmd, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    if env is not None:
        env = {str(k): str(v) for k, v in env.items()}
    subprocess.run(cmd, check=True, env=env)

def _ensure_exists(path: str, what: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")

def _first_fasta_header(fasta_path: str) -> str:
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                return line[1:].strip().split()[0]
    return "seq"

def _rewrite_fasta_header(fasta_path: str, new_tag: str):
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    if lines and lines[0].startswith(">"):
        lines[0] = f">{new_tag}\n"
    else:
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        lines = [f">{new_tag}\n", seq + ("\n" if not seq.endswith("\n") else "")]
    with open(fasta_path, "w") as f:
        f.writelines(lines)

# ---------- MMseqs2-GPU builder (A3Ms in OpenFold layout) ----------

def run_mmseqs_gpu_to_align_dir(
    fasta_path: str,
    align_dir: str,
    tag: Optional[str],
    uniref90_db: str,
    pdb70_db: Optional[str] = None,
    envdb: Optional[str] = None,
    threads: int = 32,
    gpu_ids: str = "0",
    conda_env: Optional[str] = "nvcc",
) -> str:
    """
    Produces:
      <align_dir>/<tag>/uniref90_hits.a3m
      <align_dir>/<tag>/bfd_uniclust_hits.a3m   (if envdb provided)
    """
    _ensure_exists(fasta_path, "FASTA")
    _ensure_exists(uniref90_db, "MMseqs UniRef90 DB")
    if envdb is not None:
        _ensure_exists(envdb, "MMseqs envDB")
    if pdb70_db is not None:
        _ensure_exists(pdb70_db, "MMseqs PDB70 DB")

    if tag is None:
        tag = _first_fasta_header(fasta_path) or "seq"

    outdir = Path(align_dir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="mmseqs_gpu_"))
    work = tmp_root / "work"; work.mkdir(exist_ok=True)
    qdb = work / "qdb"
    tmp = work / "tmp"; tmp.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env.setdefault("MMSEQS_FORCEGPU", "1")  # some builds need this to pick GPU

    try:
        _run(["conda", "run", "-n", conda_env,"mmseqs", "createdb", fasta_path, str(qdb)], env=env)

        # 2) UniRef90 search on GPU (CRITICAL CHANGE: added '-a' flag)
        u90_res = work / "u90_res"
        _run([
            "conda", "run", "-n", conda_env, "mmseqs", "search", str(qdb), uniref90_db, str(u90_res), str(tmp),
            "--threads", str(threads), "--gpu", "1", "-v", "3", "--split", "0", "--db-load-mode", "2",
            "-a"  # <-- IMPORTANT: Add this flag to store alignments
        ], env=env)

        # 3) To A3M-like FASTA (FIXED: Replaced result2msa/msa2a3m with a single convertalis call)
        uniref90_a3m = outdir / "uniref90_hits.a3m"
        _run([
            "conda", "run", "-n", conda_env, "mmseqs", "convertalis", str(qdb), uniref90_db, str(u90_res), str(uniref90_a3m),
            "--format-mode", "2" # Format mode 2 outputs a FASTA file
        ], env=env)

        # 4) Optional environmental DB → deeper hits
        if envdb:
            env_res = work / "env_res"
            env_a3m = outdir / "bfd_uniclust_hits.a3m"
            _run([
                "conda", "run", "-n", conda_env, "mmseqs", "search", str(qdb), envdb, str(env_res), str(tmp),
                "--threads", str(threads), "--gpu", "0",
                "-a" # <-- IMPORTANT: Also add this flag here
            ], env=env)
            _run([
                "conda", "run", "-n", conda_env, "mmseqs", "convertalis", str(qdb), envdb, str(env_res), str(env_a3m),
                "--format-mode", "2"
            ], env=env)

        print(f"✔ A3Ms written to {outdir}")
        if pdb70_db:
            pdb70_res = work / "pdb70_res"
            pdb70_a3m = outdir / "pdb70_hits.a3m"
            _run([
                "conda", "run", "-n", conda_env, "mmseqs", "search", str(qdb), pdb70_db, str(pdb70_res), str(tmp),
                "--threads", str(threads), "--gpu", "0",
                "-a" # <-- IMPORTANT: Also add this flag here
            ], env=env)
            _run([
                "conda", "run", "-n", conda_env, "mmseqs", "convertalis", str(qdb), pdb70_db, str(pdb70_res), str(pdb70_a3m),
                "--format-mode", "2"
            ], env=env)

        print(f"✔ A3Ms written to {outdir}")
        return str(outdir)

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

# ---------- Driver that mirrors your working setup ----------

def run_pipeline_with_precomputed_alignments(
    fasta_path: str,
    align_dir: str,
    output_dir: str,
    uniref90_db: str,
    envdb: Optional[str] = None,
    model_device: str = "cuda:0",
    config_preset: str = "model_1_ptm",
    template_mmcif_dir: str = "/mnt/bob/shared/alphafold/pdb_mmcif/mmcif_files",
    threads: str = "32",
    gpu_ids: str = "0",
    conda_env: str = "nvcc",
):
    """
    1) PDB -> FASTA (header 'seq' to match OpenFold tag)
    2) MMseqs2-GPU -> <align_dir>/seq/*.a3m
    3) Call your existing OpenFold runner with --use_precomputed_alignments
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)

    #fasta_path = os.path.join(fasta_dir, "seq.fasta")
    #pdb_to_fasta(pdb_path, fasta_path)
    # ensure the tag is 'seq' so OpenFold and alignments agree
    #_rewrite_fasta_header(fasta_path, "seq")
    #tag = "seq"

    # 1) build A3Ms on GPU
    run_mmseqs_gpu_to_align_dir(
        fasta_path=fasta_path,
        align_dir=align_dir,
        tag="pdb_structure_A",
        uniref90_db=uniref90_db,
        envdb=envdb,
        threads=threads,
        gpu_ids=gpu_ids,
        conda_env=conda_env
    )
    print("✔ MSA generation complete.")
    print(align_dir)
    return align_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMseqs2-GPU + OpenFold MSA pipeline")
    parser.add_argument("--fasta_path", type=str, required=True, help="Input FASTA file path")
    parser.add_argument("--align_dir", type=str, required=True, help="Output alignment directory")
    parser.add_argument("--output_dir", type=str, required=True, help="OpenFold output directory")
    parser.add_argument("--uniref90_db", type=str, required=True, help="Path to MMseqs UniRef90 DB")
    parser.add_argument("--envdb", type=str, default=None, help="Path to MMseqs environmental DB (optional)")
    parser.add_argument("--model_device", type=str, default="cuda:0", help="Model device for OpenFold")
    parser.add_argument("--config_preset", type=str, default="model_1_ptm", help="OpenFold config preset")
    parser.add_argument("--template_mmcif_dir", type=str, default="/mnt/bob/shared/alphafold/pdb_mmcif/mmcif_files", help="Template mmCIF directory")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for MMseqs2-GPU")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs for MMseqs2-GPU")

    args = parser.parse_args()

    run_pipeline_with_precomputed_alignments(
        fasta_path=args.fasta_path,
        align_dir=args.align_dir,
        output_dir=args.output_dir,
        uniref90_db=args.uniref90_db,
        envdb=args.envdb,
        model_device=args.model_device,
        config_preset=args.config_preset,
        template_mmcif_dir=args.template_mmcif_dir,
        threads=args.threads,
        gpu_ids=args.gpu_ids,
    )
