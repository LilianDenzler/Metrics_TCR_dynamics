import subprocess
import tempfile
import pandas as pd
from io import StringIO
from pathlib import Path
import re
from typing import Optional, Dict, Any

_ALLELE_PAT = re.compile(
    r"\b(?P<prefix>[A-Za-z]+)-(?P<gene>[A-Za-z0-9]+)\*(?P<allele>[0-9:]+)\b"
)

def _parse_imgt_allele_from_title(stitle: str) -> Optional[Dict[str, str]]:
    """
    Extract e.g. 'HLA-A*03:01:01:01' from a BLAST subject title string.
    Returns {prefix, gene, allele, allele_full} or None.
    """
    if not stitle:
        return None
    m = _ALLELE_PAT.search(stitle)
    if not m:
        return None
    prefix = m.group("prefix")
    gene = m.group("gene")
    allele = m.group("allele")
    return {
        "prefix": prefix,
        "gene": gene,
        "allele": allele,
        "allele_full": f"{prefix}-{gene}*{allele}",
    }

def blastp_imgt_hla(
    query_sequence: str,
    imgt_db_path: str,
    blastp_bin: str = "blastp",
    max_hits: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    BLAST a protein sequence against the IPD-IMGT/HLA BLAST DB and return top-hit metadata
    including gene/allele parsed from stitle.
    """
    query_sequence = query_sequence.strip().replace("\n", "").replace(" ", "")
    if not query_sequence:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_fasta = tmpdir / "query.fasta"
        query_fasta.write_text(">query\n" + query_sequence + "\n")

        # IMPORTANT: include stitle so we can parse the allele/gene.
        outfmt = (
            "6 qseqid sseqid stitle pident length mismatch gapopen "
            "qstart qend sstart send evalue bitscore"
        )

        cmd = [
            blastp_bin,
            "-query", str(query_fasta),
            "-db", imgt_db_path,
            "-outfmt", outfmt,
            "-max_target_seqs", str(max_hits),
        ]

        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        tsv_text = res.stdout.strip()
        if not tsv_text:
            return None

        df = pd.read_csv(
            StringIO(tsv_text),
            sep="\t",
            header=None,
            names=[
                "qseqid", "sseqid", "stitle",
                "pident", "length", "mismatch", "gapopen",
                "qstart", "qend", "sstart", "send",
                "evalue", "bitscore",
            ],
        )

    df = df.sort_values(
        by=["pident", "length", "bitscore"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    top = df.iloc[0]
    parsed = _parse_imgt_allele_from_title(str(top["stitle"]))

    hit = {
        "sseqid": str(top["sseqid"]),
        "stitle": str(top["stitle"]),
        "identity": float(top["pident"]),
        "aln_len": int(top["length"]),
        "bitscore": float(top["bitscore"]),
        "evalue": float(top["evalue"]),
    }
    if parsed:
        hit.update(parsed)  # adds prefix/gene/allele/allele_full

    return hit
