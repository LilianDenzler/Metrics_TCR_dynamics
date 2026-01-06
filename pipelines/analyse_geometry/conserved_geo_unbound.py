from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
#!/usr/bin/env python
from pathlib import Path
from geo_analysis_tools import compute_TCR_angles
import os

INPUT_DIR = Path("/mnt/larry/lilian/DATA/TCR3d_datasets/ab_chain")
OUT_DIR = Path("/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/geo_conservation_by_germline4tuple")
os.makedirs(OUT_DIR, exist_ok=True)
ANGLE_COLUMNS = ["BA","BC1","AC1","BC2","AC2","dc"]

def main():
    df, failed = compute_TCR_angles(INPUT_DIR)
    df.to_csv(OUT_DIR / "angles_with_germlines.csv", index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "failed_tcrs.txt").write_text("\n".join(failed) + ("\n" if failed else ""))


if __name__ == "__main__":
    main()
