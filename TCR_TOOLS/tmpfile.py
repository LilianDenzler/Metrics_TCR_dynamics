import sys
sys.path.append("/workspaces/Graphormer")
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from Bio.PDB import PDBIO
import os
import shutil
import pandas as pd
a6="/mnt/larry/lilian/DATA/Cory_data/A6/A6.pdb"

true_complex = TCRpMHC(input_pdb=a6,
                        MHC_a_chain_id="A",
                        MHC_b_chain_id="B",
                        Peptide_chain_id="C")
                    #write to pdb
write_pdb("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/A6_imgt.pdb", true_complex.imgt_all_structure)

