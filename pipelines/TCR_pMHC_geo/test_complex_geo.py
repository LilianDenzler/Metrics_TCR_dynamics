import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC
from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
from Bio.PDB import PDBIO
input_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/2ESV.pdb"


complex=TCRpMHC(
    input_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/2ESV.pdb",
    MHC_a_chain_id= "A",
    MHC_b_chain_id= None,
    Peptide_chain_id= "C",
)
out_path="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/complex_calc"
os.makedirs(out_path,exist_ok=True)
results=complex.calc_geometry(out_path=out_path)
print(results)
output_dir="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/changed_geometry_output"
os.makedirs(output_dir, exist_ok=True)
rearranged_pdb=complex.change_geometry(output_dir, results)
print("Changed geometry pdb saved to:", rearranged_pdb)