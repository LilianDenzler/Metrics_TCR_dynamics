from typing import Dict, Tuple
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

__all__ = ["CDR_FR_RANGES", "VARIABLE_RANGE"]