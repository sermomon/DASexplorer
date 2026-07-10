"""
DAS Explorer reader modules.

Shared utilities used across multiple reader modules.
"""
import sys
from typing import Optional

# Sanity checks:::

PROPRIETARY_TOOLS_PATH = None

# Ensure any external tools package is importable.

# In a standard pip install this function does nothing — all dependencies
# are already on the Python path. It only has effect when PROPRIETARY_TOOLS_PATH
# points to an external directory, which is useful when running from source
# without a full install or when proprietary binaries (e.g. hdas_reader*.pyd)
# live outside the package tree.

def _ensure_tools_importable() -> None:
    """Add PROPRIETARY_TOOLS_PATH to sys.path if needed."""
    if PROPRIETARY_TOOLS_PATH is not None and PROPRIETARY_TOOLS_PATH not in sys.path:
        sys.path.append(PROPRIETARY_TOOLS_PATH)