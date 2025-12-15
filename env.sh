conda create -n tcr_metrics python=3.10 -y
conda activate tcr_metrics
pip install numpy
pip install mdtraj
conda install -c conda-forge biopython -y
conda install -c conda-forge biotite -y
pip install anarcii
conda install bioconda::anarci -y
pip install mdtraj
conda install -y -c conda-forge pymol-open-source
pip install pandas
pip install MDAnalysis
conda install -y -c conda-forge scikit-learn
pip install pyEMMA
pip install seaborn
pip install matplotlib
pip install tabulate

conda install conda-forge::openmm -y

conda install conda-forge::pdbfixer -y