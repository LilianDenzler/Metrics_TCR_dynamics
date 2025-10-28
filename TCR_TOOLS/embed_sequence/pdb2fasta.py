from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO


def pdb_to_fasta(pdb_path, fasta_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb_structure", pdb_path)

    seq_records = []

    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.get_id()[0] == " "]
            sequence = "".join(seq1(res.get_resname()) for res in residues)
            record = SeqRecord(
                Seq(sequence),
                id=f"{structure.id}_{chain.id}",
                description=f"Chain {chain.id} from {pdb_path}"
            )
            seq_records.append(record)

    SeqIO.write(seq_records, fasta_path, "fasta")
    print(f"✅ Wrote {len(seq_records)} chains to {fasta_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to the input PDB file")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the output FASTA file")
    args = parser.parse_args()

    pdb_to_fasta(args.pdb_path, args.fasta_path)