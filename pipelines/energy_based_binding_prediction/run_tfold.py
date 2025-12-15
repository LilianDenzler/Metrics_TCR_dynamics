#!/usr/bin/env python

import os
import sys
import pyrosetta
from pyrosetta import rosetta
import tfold


# ---------------------------------------------------------
# Helpers: extract sequences from PDB using PyRosetta
# ---------------------------------------------------------

def get_chain_sequence_from_pose(pose, chain_id):
    """
    Extract one-letter AA sequence for a given chain from a PyRosetta Pose.
    Only counts protein residues.
    Returns None if chain not present / no protein residues.
    """
    pdb_info = pose.pdb_info()
    seq_chars = []

    for i in range(1, pose.size() + 1):
        if pdb_info.chain(i) == chain_id and pose.residue(i).is_protein():
            aa = pose.residue(i).name1()
            # Skip non-standard or unknown residues if needed
            if aa == 'X':
                continue
            seq_chars.append(aa)

    if not seq_chars:
        return None

    return "".join(seq_chars)


def build_tfold_input_for_pdb(pdb_path):
    """
    Load a TCR–pMHC complex PDB and build the 'data' list
    expected by tfold.deploy.TCRpMHCPredictor, using the chain
    mapping you specified:

      - chain D -> TCR beta  -> id "B"
      - chain E -> TCR alpha -> id "A"
      - chain A -> MHC heavy -> id "M"
      - chain B -> MHC light -> id "N" (optional)
      - chain C -> peptide   -> id "P"
    """
    pose = rosetta.core.import_pose.pose_from_file(pdb_path)

    seq_tcr_beta = get_chain_sequence_from_pose(pose, 'D')   # TCR β
    seq_tcr_alpha = get_chain_sequence_from_pose(pose, 'E')  # TCR α
    seq_mhc_heavy = get_chain_sequence_from_pose(pose, 'A')  # MHC α
    seq_mhc_light = get_chain_sequence_from_pose(pose, 'B')  # MHC β / β2m (optional)
    seq_peptide = get_chain_sequence_from_pose(pose, 'C')    # peptide

    # Basic sanity checks
    missing = []
    if seq_tcr_beta is None:
        missing.append("TCR beta (chain D)")
    if seq_tcr_alpha is None:
        missing.append("TCR alpha (chain E)")
    if seq_mhc_heavy is None:
        missing.append("MHC heavy (chain A)")
    if seq_peptide is None:
        missing.append("peptide (chain C)")

    if missing:
        raise ValueError(
            f"{os.path.basename(pdb_path)} is missing required chains: "
            + ", ".join(missing)
        )

    # Assemble tFold-TCR input
    data = [
        {
            "id": "B",  # TCR beta / delta chain
            "sequence": seq_tcr_beta,
        },
        {
            "id": "A",  # TCR alpha / gamma chain
            "sequence": seq_tcr_alpha,
        },
        {
            "id": "M",  # MHC I/II heavy / alpha chain
            "sequence": seq_mhc_heavy,
        },
    ]

    if seq_mhc_light is not None:
        data.append(
            {
                "id": "N",  # MHC I/II light / beta chain (optional)
                "sequence": seq_mhc_light,
            }
        )

    data.append(
        {
            "id": "P",  # peptide
            "sequence": seq_peptide,
        }
    )

    return data


# ---------------------------------------------------------
# Main: loop over PDBs and run tFold-TCR
# ---------------------------------------------------------

def main(
    input_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes",
    output_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold",
):

    # 1) Init PyRosetta ONCE (for reading sequences)
    print("Initializing PyRosetta...")
    pyrosetta.init("-mute all")

    # 2) Init tFold-TCR model ONCE
    print("Loading tFold-TCR (TCRpMHCPredictor) model...")
    ppi_model_path = tfold.model.esm_ppi_650m_tcr()
    tfold_model_path = tfold.model.tfold_tcr_pmhc_trunk()
    model = tfold.deploy.TCRpMHCPredictor(ppi_model_path, tfold_model_path)

    # 3) Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4) Iterate over input PDBs
    pdb_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pdb"))

    if not pdb_files:
        print(f"No .pdb files found in {input_dir}")
        return

    for pdb_file in pdb_files:
        input_pdb = os.path.join(input_dir, pdb_file)
        output_pdb = os.path.join(output_dir, pdb_file)  # same name, new folder

        # Skip if already exists (optional)
        if os.path.exists(output_pdb):
            print(f"[skip] {pdb_file} -> already exists at {output_pdb}")
            continue

        print(f"\n=== Processing {pdb_file} ===")

        try:
            # Build tfold input from template structure
            data = build_tfold_input_for_pdb(input_pdb)
            print("  - Built tFold input with chains:")
            for entry in data:
                print(f"    id={entry['id']} len={len(entry['sequence'])}")

            # Run tFold-TCR and write predicted complex
            print(f"  - Running tFold-TCR, writing to {output_pdb}")
            model.infer_pdb(data, output_pdb)
            print(f"  ✓ Done: {output_pdb}")

        except Exception as e:
            print(f"  ✗ Error processing {pdb_file}: {e}")
            continue


if __name__ == "__main__":
    # You can also pass custom paths as CLI args if you want:
    #   python run_tfold_tcr_pmhc.py /path/to/in /path/to/out

    if len(sys.argv) >= 3:
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
    else:
        in_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes"
        out_dir="/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes_tfold"

    main(in_dir, out_dir)
