from angle_utils import *
from pathlib import Path

out=compute_one_pdb(Path("/mnt/larry/lilian/DATA/TCR3d_datasets/TCR_complexes/1fyt.pdb"),
                state= "bound",
                contact_cutoff= 5.0,
                min_contacts = 50,
                legacy_anarci= True,
                vis= True)

print(out)