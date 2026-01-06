from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

# 1. Load your PDB file
pdb_id = '/mnt/larry/lilian/DATA/Oriol_adaptive_sampling/2CDG/2CDG.pdb' # Change this to your filename
pdb = PDBFile(pdb_id)

# 2. Select a Forcefield
# 'amber14-all.xml' for protein, 'amber14/tip3pfb.xml' for water
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

# 3. Add Water and a Bounding Box
modeller = Modeller(pdb.topology, pdb.positions)

# This adds a cubic box with a 1.0 nm (10 Angstrom) buffer around the protein
# 'padding' defines the distance between the protein and the box edge
modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometer, ionicStrength=0.15*molar)

# 4. Save the "Boxed" PDB for PyMOL
with open('protein_in_box.pdb', 'w') as f:
    PDBFile.writeFile(modeller.topology, modeller.positions, f)

print("Successfully created 'protein_in_box.pdb' with a water box.")

# --- OPTIONAL: Brief Energy Minimization ---
# This fixes overlaps so the box looks 'cleaner' in PyMOL
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

print("Minimizing energy...")
simulation.minimizeEnergy()

# Save the minimized version
with open('protein_minimized_box.pdb', 'w') as f:
    PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)