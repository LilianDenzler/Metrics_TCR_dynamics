from __future__ import annotations
from typing import Dict, Tuple

from importlib.resources import files
from pathlib import Path
# -------------------------------------------------------------------
# IMGT region definitions
# -------------------------------------------------------------------
CDR_FR_RANGES: Dict[str, Tuple[int, int]] = {
    # Alpha-like (A)
    "A_FR1":   (3, 26),
    "A_CDR1":  (27, 38),
    "A_FR2":   (39, 55),
    "A_CDR2":  (56, 65),
    "A_FR3":   (66, 104),
    "A_CDR3":  (105, 117),
    "A_FR4":   (118, 125),
    # Beta-like (B)
    "B_FR1":   (3, 26),
    "B_CDR1":  (27, 38),
    "B_FR2":   (39, 55),
    "B_CDR2":  (56, 65),
    "B_FR3":   (66, 104),
    "B_CDR3":  (105, 117),
    "B_FR4":   (118, 125),
}
VARIABLE_RANGE: Tuple[int, int] = (1, 128)

A_FR_ALIGN = [(39,43),(78,79), (87,104)]
B_FR_ALIGN = [(5,24),(40,44), (78,79),(86,90),(100,104)]

__all__ = ["CDR_FR_RANGES", "VARIABLE_RANGE"]

_GEO_DEFAULT_DATA_PATH = files("TCR_TOOLS.geometry.data").joinpath("consensus_output")
GEO_DATA_PATH: Path = Path(str(_GEO_DEFAULT_DATA_PATH))

IMGT_DB_PATH="/mnt/larry/lilian/DATA/IPD-IMGT-HLA-DATABASE/imgt_hla_prot"