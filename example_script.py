#import sys
#sys.path.append("path/to/where/TCR_Metrics/dir/is/for/you")
from TCR_TOOLS.classes.tcr import *
from TCR_TOOLS.core.io import write_pdb
tcr_pdb="/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/"
tcr = TCR(
        input_pdb=f"/workspaces/Graphormer/TCR_Metrics/TCR_TOOLS/geometry/test/1ao7.pdb",
        traj_path=None,         # or an XTC/DCD if you have one
        contact_cutoff=5.0, #can keep default, determines distance in Å that constitutes contact
        min_contacts=50, #can keep defaultm determines minimum number of contacts between TCR chains to be recognised as paired chains
        legacy_anarci=True) #if true, prioritise legacy anarci, and only run new anarci if legacy mode fails
print(f"Found {len(tcr.pairs)} pair(s).")
pv = tcr.pairs[0]   # TCRPairView for first pair
#write renumbered file:
write_pdb(os.path.join(tcr_pdb,"1ao7_imgt.pdb"),tcr.imgt_all_structure)


#write only TCR chains renumbered
write_pdb(os.path.join(tcr_pdb,"1ao7_tcr_imgt.pdb"),pv.full_structure)

#write only renumbered variable fragment
write_pdb(os.path.join(tcr_pdb,"1ao7_tcr_vf_imgt.pdb"),pv.variable_structure)


#write only cdr3 beta for example
write_pdb(os.path.join(tcr_pdb,"1ao7_tcr_cdr3b_imgt.pdb"),pv.domain_subset(region_names=["B_CDR3"]))

print(pv.cdr_fr_sequences())
cdr_dict=pv.cdr_fr_sequences()
#write to file
with open(os.path.join(tcr_pdb,"cdr_fr_seq.txt"),"w") as f:
    for cdr in cdr_dict:
        f.write(f">{cdr}\n{cdr_dict[cdr]}\n")

if tcr.original_CDR_FR_RANGES is None:
    pass
else:
    with open(os.path.join(tcr_pdb,"annotation_original_resnums.txt"),"w") as f:
        for cdr in tcr.original_CDR_FR_RANGES:
            f.write(f"{cdr}: {tcr.original_CDR_FR_RANGES[cdr]}\n")

