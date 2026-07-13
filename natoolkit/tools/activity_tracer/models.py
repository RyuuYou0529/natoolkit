from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ROISet:
    labels: np.ndarray


@dataclass
class MovieState:
    start: int = 0
    stop: int = 0
    layer_name: str = ""
    source_data: np.ndarray | None = None
    spatial_reference: str = ""
    temporal_reference: str = ""
    roi_set: ROISet | None = None
    traces: dict[int, np.ndarray] = field(default_factory=dict)
    trace_colors: dict[int, str] = field(default_factory=dict)
    visible_rois: set[int] = field(default_factory=set)
    spikes: dict[int, list[dict[str, float]]] = field(default_factory=dict)
