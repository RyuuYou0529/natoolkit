from __future__ import annotations

import numpy as np


def roi_ids(labels: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(labels) if value > 0]
