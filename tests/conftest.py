from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Ensure test imports resolve to local source tree, not an installed wheel/editable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
