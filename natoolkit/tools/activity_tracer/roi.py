from __future__ import annotations

import numpy as np
from napari.layers import Labels
from napari.utils.colormaps import CyclicLabelColormap


def roi_colormap() -> CyclicLabelColormap:
    return CyclicLabelColormap(
        colors=[
            "transparent",
            "#00ffff",
            "#ffff00",
            "#ff4dff",
            "#7cff00",
            "#ff9f1c",
            "#00ff9f",
            "#ff1493",
            "#1e90ff",
            "#ffd700",
            "#adff2f",
            "#ff7f50",
            "#40e0d0",
            "#ee82ee",
            "#fa8072",
            "#7fffd4",
            "#f0e68c",
            "#ff69b4",
            "#87cefa",
            "#dfff00",
            "#ff4500",
            "#dda0dd",
            "#20e0c0",
            "#ff6347",
            "#e0ffff",
            "#90ee90",
            "#f5deb3",
            "#da70d6",
            "#6495ed",
            "#eee8aa",
            "#98ff98",
            "#ffdab9",
            "#ffffff",
        ],
        name="bright_roi_colors",
    )


def roi_centers(labels: np.ndarray, ids: list[int]) -> np.ndarray:
    centers = []
    for roi_id in ids:
        pixels = np.argwhere(labels == roi_id)
        center = pixels.mean(axis=0)
        centers.append(pixels[np.argmin(np.sum((pixels - center) ** 2, axis=1))])
    return np.asarray(centers, dtype=float).reshape(-1, 2)


class ROILabels(Labels):
    """Labels layer where each completed polygon replaces its label."""

    def paint_polygon(self, points, new_label):
        background = self.colormap.background_value
        if new_label == background:
            super().paint_polygon(points, new_label)
            return

        current = np.nonzero(np.asarray(self.data) == new_label)
        if current[0].size == 0:
            super().paint_polygon(points, new_label)
            return

        with self.block_history():
            self.data_setitem(current, background, refresh=False)
            super().paint_polygon(points, new_label)


def roi_ids(labels: np.ndarray) -> list[int]:
    return [int(value) for value in np.unique(labels) if value > 0]
