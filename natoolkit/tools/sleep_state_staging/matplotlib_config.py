from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache() -> Path:
    cache_directory = Path(tempfile.gettempdir()) / "natoolkit" / "matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    return Path(os.environ["MPLCONFIGDIR"])
