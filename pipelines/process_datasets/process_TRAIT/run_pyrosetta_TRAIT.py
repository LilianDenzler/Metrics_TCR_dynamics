import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from pipelines.energy_based_binding_prediction.run_pyrosetta import run_folder
from pipelines.process_datasets.process_TRAIT.plot_pyrosetta_trait_omics import plot_rosetta_csv

antigen_title_map = {
    "A0101_SLEGGGLGY_NC": "Antigen: NC pMHC (HLA-A*01:01, SLEGGGLGY)",
    "A0101_STEGGGLAY_NC": "Antigen: NC pMHC (HLA-A*01:01, STEGGGLAY)",
    "A0101_VTEHDTLLY_IE-1_CMV": "Antigen: Immediate-early 1 (IE-1) CMV pMHC (HLA-A*01:01, VTEHDTLLY)",

    "A0201_ALIAPVHAV_NC": "Antigen: NC pMHC (HLA-A*02:01, ALIAPVHAV)",
    "A0201_CLGGLLTMV_LMP-2A_EBV": "Antigen: LMP2A EBV pMHC (HLA-A*02:01, CLGGLLTMV)",
    "A0201_CLLGTYTQDV_Kanamycin-B-dioxygenase": "Antigen: Kanamycin B dioxygenase pMHC (HLA-A*02:01, CLLGTYTQDV)",
    "A0201_CLLWSFQTSA_Tyrosinase_Cancer": "Antigen: Tyrosinase (cancer) pMHC (HLA-A*02:01, CLLWSFQTSA)",
    "A0201_ELAGIGILTV_MART-1_Cancer": "Antigen: MART-1 (cancer) pMHC (HLA-A*02:01, ELAGIGILTV)",
    "A0201_FLASKIGRLV_Ca2-indepen-Plip-A2": "Antigen: Ca2+-independent phospholipase A2 pMHC (HLA-A*02:01, FLASKIGRLV)",
    "A0201_FLYALALLL_LMP2A_EBV": "Antigen: LMP2A EBV pMHC (HLA-A*02:01, FLYALALLL)",
    "A0201_GILGFVFTL_Flu-MP_Influenza": "Antigen: Influenza matrix protein (M1/MP) pMHC (HLA-A*02:01, GILGFVFTL)",
    "A0201_GLCTLVAML_BMLF1_EBV": "Antigen: BMLF1 EBV pMHC (HLA-A*02:01, GLCTLVAML)",
    "A0201_ILKEPVHGV_RT_HIV": "Antigen: HIV reverse transcriptase (RT) pMHC (HLA-A*02:01, ILKEPVHGV)",

    "A0201_IMDQVPFSV_gp100_Cancer": "Antigen: gp100 (cancer) pMHC (HLA-A*02:01, IMDQVPFSV)",
    "A0201_KTWGQYWQV_gp100_Cancer": "Antigen: gp100 (cancer) pMHC (HLA-A*02:01, KTWGQYWQV)",

    "A0201_KLQCVDLHV_PSA146-154": "Antigen: PSA 146–154 pMHC (HLA-A*02:01, KLQCVDLHV)",

    "A0201_KVAELVHFL_MAGE-A3_Cancer": "Antigen: MAGE-A3 (cancer) pMHC (HLA-A*02:01, KVAELVHFL)",
    "A0201_KVLEYVIKV_MAGE-A1_Cancer": "Antigen: MAGE-A1 (cancer) pMHC (HLA-A*02:01, KVLEYVIKV)",

    "A0201_LLDFVRFMGV_EBNA-3B_EBV": "Antigen: EBNA-3B EBV pMHC (HLA-A*02:01, LLDFVRFMGV)",
    "A0201_YLLEMLWRL_LMP1_EBV": "Antigen: LMP1 EBV pMHC (HLA-A*02:01, YLLEMLWRL)",

    "A0201_LLMGTLGIVC_HPV-16E7_82-91": "Antigen: HPV-16 E7 (82–91) pMHC (HLA-A*02:01, LLMGTLGIVC)",
    "A0201_MLDLQPETT_16E7_HPV": "Antigen: HPV E7 pMHC (HLA-A*02:01, MLDLQPETT)",

    "A0201_NLVPMVATV_pp65_CMV": "Antigen: pp65 CMV pMHC (HLA-A*02:01, NLVPMVATV)",

    "A0201_RMFPNAPYL_WT-1": "Antigen: WT1 pMHC (HLA-A*02:01, RMFPNAPYL)",
    "A0201_CYTWNQMNL_WT1-_235-243_236M_Y": "Antigen: WT1 (235–243; 236M→Y) pMHC (HLA-A*24:02, CYTWNQMNL)",

    "A0201_RTLNAWVKV_Gag-protein_HIV": "Antigen: HIV Gag pMHC (HLA-A*02:01, RTLNAWVKV)",
    "A0201_SLFNTVATL_Gag-protein_HIV": "Antigen: HIV Gag pMHC (HLA-A*02:01, SLFNTVATL)",
    "A0201_SLFNTVATLY_Gag-protein_HIV": "Antigen: HIV Gag pMHC (HLA-A*02:01, SLFNTVATLY)",
    "A0201_SLLMWITQV_NY-ESO-1_Cancer": "Antigen: NY-ESO-1 (cancer) pMHC (HLA-A*02:01, SLLMWITQV)",
    "A0201_SLLMWITQV_NY-ESO-1_Cancer": "Antigen: NY-ESO-1 (cancer) pMHC (HLA-A*02:01, SLLMWITQV)",
    "A0201_SLYNTVATLY_Gag-protein_HIV": "Antigen: HIV Gag pMHC (HLA-A*02:01, SLYNTVATLY)",
    "A0201_SLLMWITQV_NY-ESO-1_Cancer": "Antigen: NY-ESO-1 (cancer) pMHC (HLA-A*02:01, SLLMWITQV)",
    "A0201_LLFGYPVYV_HTLV-1": "Antigen: HTLV-1 pMHC (HLA-A*02:01, LLFGYPVYV)",
    "A0201_YLNDHLEPWI_BCL-X_Cancer": "Antigen: BCL-X (cancer) pMHC (HLA-A*02:01, YLNDHLEPWI)",

    "A0301_KLGGALQAK_IE-1_CMV": "Antigen: Immediate-early 1 (IE-1) CMV pMHC (HLA-A*03:01, KLGGALQAK)",
    "A0301_RIAAWMATY_BCL-2L1_Cancer": "Antigen: BCL-2L1 (cancer) pMHC (HLA-A*03:01, RIAAWMATY)",
    "A0301_RLRAEAQVK_EMNA-3A_EBV": "Antigen: EBNA-3A EBV pMHC (HLA-A*03:01, RLRAEAQVK)",

    "A1101_AVFDRKSDAK_EBNA-3B_EBV": "Antigen: EBNA-3B EBV pMHC (HLA-A*11:01, AVFDRKSDAK)",
    "A1101_IVTDFSVIK_EBNA-3B_EBV": "Antigen: EBNA-3B EBV pMHC (HLA-A*11:01, IVTDFSVIK)",

    "A2402_QYDPVAALF_pp65_CMV": "Antigen: pp65 CMV pMHC (HLA-A*24:02, QYDPVAALF)",
    "A2402_AYAQKIFKI_IE-1_CMV": "Antigen: Immediate-early 1 (IE-1) CMV pMHC (HLA-A*24:02, AYAQKIFKI)",
    "A2402_AYSSAGASI_NC": "Antigen: NC pMHC (HLA-A*24:02, AYSSAGASI)",

    "B0702_GPAESAAGL_NC": "Antigen: NC pMHC (HLA-B*07:02, GPAESAAGL)",
    "B0702_QPRAPIRPI_EBNA-6_EBV": "Antigen: EBNA-6 EBV pMHC (HLA-B*07:02, QPRAPIRPI)",
    "B0702_RPHERNGFTVL_pp65_CMV": "Antigen: pp65 CMV pMHC (HLA-B*07:02, RPHERNGFTVL)",
    "B0702_RPPIFIRRL_EBNA-3A_EBV": "Antigen: EBNA-3A EBV pMHC (HLA-B*07:02, RPPIFIRRL)",
    "B0702_TPRVTGGGAM_pp65_CMV": "Antigen: pp65 CMV pMHC (HLA-B*07:02, TPRVTGGGAM)",

    "B0801_ELRRKMMYM_IE-1_CMV": "Antigen: Immediate-early 1 (IE-1) CMV pMHC (HLA-B*08:01, ELRRKMMYM)",
    "B0801_FLRGRAYGL_EBNA-3A_EBV": "Antigen: EBNA-3A EBV pMHC (HLA-B*08:01, FLRGRAYGL)",
    "B0801_RAKFKQLL_BZLF1_EBV": "Antigen: BZLF1 EBV pMHC (HLA-B*08:01, RAKFKQLL)",

    "B3501_IPSINVHHY_pp65_CMV": "Antigen: pp65 CMV pMHC (HLA-B*35:01, IPSINVHHY)",

    "NR_B0801__AAKGRGAAL_NC": "Antigen: NC pMHC (HLA-B*08:01, AAKGRGAAL)",
}





out_folder=f"/workspaces/Graphormer/TCR_Metrics/pipelines/process_datasets/process_TRAIT/omics_tfold_rossetta_results"
tfold_subsets_dir="/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets"
for antigen_name in os.listdir(tfold_subsets_dir):
    antigen_dir=os.path.join(tfold_subsets_dir,antigen_name)
    if Path(os.path.join(antigen_dir,"DONE.txt")).exists():
        pass
    else:
        print(f"TFOLD not done for: {antigen_name}")
        continue
    for dataset_type in os.listdir(antigen_dir):
        dataset_path=os.path.join(antigen_dir,dataset_type)
        if Path(dataset_path).is_dir():
            pass
        else:
            continue
        for neg_pos in ["neg","pos"]:
            base_dir=f"/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/{antigen_name}/{dataset_type}/{neg_pos}"
            fixed_dir=f"/mnt/larry/lilian/DATA/TRAIT/omics_tfold_subsets/{antigen_name}/{dataset_type}/{neg_pos}_openmm_minimised"
            os.makedirs(fixed_dir, exist_ok=True)
            vis_folder=os.path.join(out_folder,antigen_name,dataset_type)
            os.makedirs(vis_folder, exist_ok=True)
            #tfold standardises, so that the chains are always named the same way
            tcr_alpha="A"
            tcr_beta="B"
            mhc_alpha="M"
            mhc_beta="N"
            peptide="P"
            modes = ["none", "rosetta_sidechain", "openmm"]#,"rosetta_full"]
            for mode in modes:
                if mode in ["rosetta_sidechain", "rosetta_full", "none"]:
                    num_workers=4
                else:
                    num_workers=2 #do 1 if using GPU
                mode_name=mode
                if mode=="rosetta_sidechain":
                    mode_name="sidechain"
                out_csv = os.path.join(vis_folder,f"{mode_name}_{neg_pos}.csv")
                plots_dir = os.path.join(vis_folder, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                if Path(out_csv).exists():
                    print(f"{out_csv} already exists, skipping...")
                else:
                    run_folder(
                        base_dir=base_dir,
                        fixed_dir=fixed_dir,
                        minimization_mode=mode,
                        out_csv=out_csv,
                        tcr_alpha_id=tcr_alpha,
                        tcr_beta_id=tcr_beta,
                        mhc_alpha_id=mhc_alpha,
                        mhc_beta_id=mhc_beta,
                        peptide_id=peptide,
                        num_workers=num_workers,
                    )
                # Always plot (idempotent + fast)
                plot_rosetta_csv(
                    csv_path=out_csv,
                    plots_dir=plots_dir,
                    method=mode,
                    antigen_name=antigen_title_map[antigen_name],
                )
