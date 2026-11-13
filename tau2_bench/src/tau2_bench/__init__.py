"""tau2-bench customer-service agent recipe (airline and retail, English, text)."""

import os
from pathlib import Path


UPSTREAM_COMMIT = "989d3285ae669e49503324bdb9981bf14fccedeb"

# tau2 reads TAU2_DATA_DIR once, when it is first imported, and its wheel ships
# no data. Point it at our checksum-pinned cache before any tau2 import; every
# module in this package imports tau2 only after this runs.
os.environ.setdefault("TAU2_DATA_DIR", str(Path.home() / ".cache" / "tau2-bench" / UPSTREAM_COMMIT / "data"))
