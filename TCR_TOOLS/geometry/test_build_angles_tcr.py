from copy import deepcopy
import os
import numpy as np
import warnings
from pathlib import Path
import math
import json
import tempfile
import biotite.structure as bts
import biotite.structure.io as btsio

import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.geometry.change_geometry import *
# 1) Pick some realistic angles
BA, BC1, BC2, AC1, AC2, dc = 120.0, 95.0, 10.0, 70.0, 150.0, 24.0

# 2) Build geometry
A_C, A_V1, A_V2, B_C, B_V1, B_V2 = build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc)

# 3) Plug into your calc-style geometry to see if you recover the same values
from math import degrees

def as_unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def angle_between(v1, v2):
    v1 = as_unit(v1); v2 = as_unit(v2)
    return degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

A1 = as_unit(A_V1 - A_C)
A2 = as_unit(A_V2 - A_C)
B1 = as_unit(B_V1 - B_C)
B2 = as_unit(B_V2 - B_C)
Cvec = as_unit(B_C - A_C)

nx = np.cross(A1, Cvec)
ny = np.cross(Cvec, nx)
Lp = as_unit([0.0, np.dot(A1, nx), np.dot(A1, ny)])
Hp = as_unit([0.0, np.dot(B1, nx), np.dot(B1, ny)])
BA_calc = angle_between(Lp, Hp)
if np.cross(Lp, Hp)[0] < 0:
    BA_calc = -BA_calc

BC1_calc = angle_between(B1, -Cvec)
AC1_calc = angle_between(A1,  Cvec)
BC2_calc = angle_between(B2, -Cvec)
AC2_calc = angle_between(A2,  Cvec)
dc_calc  = np.linalg.norm(B_C - A_C)

print("BA   target / calc:", BA,  BA_calc)
print("BC1  target / calc:", BC1, BC1_calc)
print("AC1  target / calc:", AC1, AC1_calc)
print("BC2  target / calc:", BC2, BC2_calc)
print("AC2  target / calc:", AC2, AC2_calc)
print("dc   target / calc:", dc,  dc_calc)
