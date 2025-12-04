import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr_pMHC import TCRpMHC
from TCR_TOOLS.classes.tcr import *
from Bio.PDB import PDBIO


tcr=TCR(
    input_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7.pdb"
)

tcrpair=tcr.pairs[0]
out_path="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/calc"
angles=tcrpair.calc_angles(out_path=out_path)
print(angles)
#-62.678106  106.895734  ...  159.943516  148.10108  26.80931

# -61.393677  99.820084  ...  158.547358  145.099283  26.833455
outpath="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7"
BA=angles["BA"].values[0]
BC1=angles["BC1"].values[0]
BC2=angles["BC2"].values[0]
AC1=angles["AC1"].values[0]
AC2=angles["AC2"].values[0]
dc=angles["dc"].values[0]
print("BA:", BA)
print("BC1:", BC1)
print("BC2:", BC2)
print("AC1:", AC1)
print("AC2:", AC2)
print("dc:", dc)
outpdb=tcrpair.change_angles(outpath, BA, BC1, BC2, AC1, AC2, dc)


tcr_changed=TCR(input_pdb=outpdb)
tcr_changedpairs=tcr_changed.pairs[0]
angles_changed=tcr_changedpairs.calc_angles()
print(angles_changed)
angles_changed_dict=angles_changed.to_dict()
angles_dict=angles.to_dict()
for key in angles_dict:
    print(f"{key}: original={angles_dict[key]}, changed={angles_changed_dict[key]}")

input()


complex=TCRpMHC(
    input_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7.pdb",
    MHC_a_chain_id= "A",
    MHC_b_chain_id= None,
    Peptide_chain_id= "C",
)

structure_complex_numbered=complex.imgt_all_structure
#write to pdb
io = PDBIO()
io.set_structure(structure_complex_numbered)
io.save("/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/check_numberings.pdb")
df=complex.calc_geometry()
print(df)
for idx, row in df.iterrows():
    print(f"Row {idx}:")
    print(row)
#df to dict
result_dict=row.to_dict()
print("Result dict:")
print(result_dict)
#new_ba=result_dict["d_alpha_mhc"]+10
#result_dict["d_alpha_mhc"]=new_ba
output_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/changed_geometry_output.pdb"
complex.change_geometry(output_pdb, result_dict)