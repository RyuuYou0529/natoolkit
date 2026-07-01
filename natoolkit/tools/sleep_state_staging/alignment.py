from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelIntervals:
    starts_sec: np.ndarray
    stops_sec: np.ndarray
    labels: np.ndarray


def labels_to_intervals(
    labels: np.ndarray,
    label_times_sec: np.ndarray,
    step_sec: float = 1.0,
) -> LabelIntervals:
    centers = np.asarray(label_times_sec, dtype=np.float64)
    half_step = step_sec / 2.0
    return LabelIntervals(
        starts_sec=centers - half_step,
        stops_sec=centers + half_step,
        labels=np.asarray(labels, dtype=object),
    )


def assign_labels_to_times(
    query_times_sec: np.ndarray,
    labels: np.ndarray,
    label_times_sec: np.ndarray,
    step_sec: float = 1.0,
    default: str = "Unknown",
) -> np.ndarray:
    intervals = labels_to_intervals(labels, label_times_sec, step_sec)
    query = np.asarray(query_times_sec, dtype=np.float64)
    out = np.full(query.shape, default, dtype=object)
    idx = np.searchsorted(intervals.starts_sec, query, side="right") - 1
    valid = (idx >= 0) & (idx < len(intervals.labels))
    in_interval = valid.copy()
    in_interval[valid] = query[valid] < intervals.stops_sec[idx[valid]]
    valid = in_interval
    out[valid] = intervals.labels[idx[valid]]
    return out


def sd_frame_to_raw_frame(sd_frames: np.ndarray | int, context_radius: int = 10) -> np.ndarray:
    return np.asarray(sd_frames, dtype=int) + int(context_radius)


def frame_times_from_rate(
    n_frames: int,
    frame_rate: float,
    start_sec: float = 0.0,
    frame_offset: int = 0,
) -> np.ndarray:
    frames = np.arange(n_frames, dtype=np.float64) + frame_offset
    return float(start_sec) + frames / float(frame_rate)
