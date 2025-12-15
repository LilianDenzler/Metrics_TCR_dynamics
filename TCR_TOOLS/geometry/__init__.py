from __future__ import annotations
from typing import Dict, Tuple

from importlib.resources import files
from pathlib import Path
_DEFAULT_DATA_PATH = files("TCR_TOOLS.geometry.data").joinpath("consensus_output")
DATA_PATH: Path = Path(str(_DEFAULT_DATA_PATH))