from TCR_TOOLS.classes.tcr import TCR
from TCR_TOOLS.core.io import write_pdb

# 1) Construct a TCR object from a PDB
tcr = TCR(
    input_pdb="/mnt/larry/lilian/DATA/Cory_data/6OVN/6OVN.pdb",
    traj_path=None,         # or an XTC/DCD if you have one
    contact_cutoff=5.0,
    min_contacts=50,
    legacy_anarci=True
)

# 2) See the paired receptors that were found
print(f"Found {len(tcr.pairs)} pair(s).")
pv = tcr.pairs[0]   # TCRPairView for pair 1
print(pv.alpha_type, pv.beta_type, pv.chain_map)   # e.g., "A", "B", {"alpha":"A","beta":"B"}

# 3) Get sequences (original vs IMGT, all chains)
print( tcr.get_sequence_dict())


# 4) Per-pair CDR/FR sequences
print(pv.cdr_fr_sequences())

# 5) Get the variable-domain structure (cached on first access)
fv = pv.variable_structure
write_pdb("/tmp/pair1_variable.pdb", fv)

# 6) Get arbitrary region subsets
cdr_struct = pv.domain_subset(["A_CDR1","A_CDR2","B_CDR3"])
write_pdb("/tmp/pair1_regions.pdb", cdr_struct)

# 7) Build the linked variable (cached). Uses a temp folder internally; nothing left on disk.
LINKER_SEQUENCE = "GGGGS"*3
linked = pv.linked_structure(LINKER_SEQUENCE)
write_pdb("/tmp/pair1_linked.pdb", linked)

# 8) (Optional) Attach a trajectory to the TCR, then slice per-pair regions
tcr.attach_trajectory("/path/to/trajectory.xtc", topology_choice="imgt")
sub_traj = pv.traj.domain_subset(["A_CDR1","A_CDR2"], atom_names={"CA"})  # returns a new mdtraj.Trajectory
print(sub_traj)



mov_aln,summary=align_traj_to_refstructure(pv1, pv2, regions, atom_names=None, tmalign_step=True)
print(summary)
mov_aln.save("/workspaces/Graphormer/TCR_Metrics/test/B_TM_topdb.xtc"); mov_aln[0].save_pdb("/workspaces/Graphormer/TCR_Metrics/test/B_TM_topdb.pdb")
