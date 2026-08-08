from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .experiment import RecordingSource, VideoSource
from .io import EEGEMGRecording
from .matplotlib_config import configure_matplotlib_cache
from .staging import NREM, REM, WAKE, StagingResult


STAGES = (WAKE, NREM, REM)
DRAFT_VERSION = 1


@dataclass
class ReviewSession:
    source: RecordingSource
    recording: EEGEMGRecording
    processed_eeg: np.ndarray
    processed_emg: np.ndarray
    result: StagingResult
    output_directory: Path
    draft_labels: np.ndarray = field(init=False)
    committed_labels: np.ndarray = field(init=False)
    video_status: dict[str, str] = field(init=False)
    committed_video_status: dict[str, str] = field(init=False)
    undo_stack: list[tuple[int, int, np.ndarray]] = field(default_factory=list, init=False)
    redo_stack: list[tuple[int, int, np.ndarray]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.draft_labels = self.result.labels.astype(object).copy()
        self.committed_labels = self.result.labels.astype(object).copy()
        self.video_status = {self.video_key(video): "unreviewed" for video in self.source.videos}
        self.committed_video_status = self.video_status.copy()
        self._load_commit()
        self._load_draft()

    @property
    def auto_labels(self) -> np.ndarray:
        return self.result.labels

    @property
    def label_interval_sec(self) -> float:
        return float(self.result.params.step_sec)

    @property
    def label_starts(self) -> np.ndarray:
        return self.result.times_sec - self.label_interval_sec / 2.0

    @property
    def label_stops(self) -> np.ndarray:
        return self.result.times_sec + self.label_interval_sec / 2.0

    @property
    def has_uncommitted_changes(self) -> bool:
        return bool(
            np.any(self.draft_labels != self.committed_labels)
            or self.video_status != self.committed_video_status
        )

    @staticmethod
    def video_key(video: VideoSource) -> str:
        return video.display_name

    def epoch_at(self, time_sec: float) -> int | None:
        if len(self.result.times_sec) == 0:
            return None
        index = int(np.searchsorted(self.label_starts, time_sec, side="right") - 1)
        if index < 0 or index >= len(self.result.times_sec) or time_sec >= self.label_stops[index]:
            return None
        return index

    def epoch_range(self, start_sec: float, stop_sec: float) -> tuple[int, int] | None:
        lo, hi = sorted((float(start_sec), float(stop_sec)))
        if hi == lo:
            index = self.epoch_at(lo)
            return (index, index + 1) if index is not None else None
        first = int(np.searchsorted(self.label_stops, lo, side="right"))
        last = int(np.searchsorted(self.label_starts, hi, side="left"))
        first = max(0, min(first, len(self.draft_labels)))
        last = max(first, min(last, len(self.draft_labels)))
        return (first, last) if last > first else None

    def selection_bounds(self, selection: tuple[int, int]) -> tuple[float, float]:
        first, last = selection
        return float(self.label_starts[first]), float(self.label_stops[last - 1])

    def apply_stage(self, selection: tuple[int, int], stage: str) -> bool:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        first, last = selection
        if first < 0 or last > len(self.draft_labels) or first >= last:
            raise ValueError("A non-empty label selection is required.")
        old = self.draft_labels[first:last].copy()
        if np.all(old == stage):
            return False
        self.undo_stack.append((first, last, old))
        self.redo_stack.clear()
        self.draft_labels[first:last] = stage
        self._invalidate_overlapping_videos(first, last)
        self.save_draft()
        return True

    def restore_auto(self, selection: tuple[int, int]) -> bool:
        first, last = selection
        old = self.draft_labels[first:last].copy()
        replacement = self.auto_labels[first:last]
        if np.array_equal(old, replacement):
            return False
        self.undo_stack.append((first, last, old))
        self.redo_stack.clear()
        self.draft_labels[first:last] = replacement
        self._invalidate_overlapping_videos(first, last)
        self.save_draft()
        return True

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        first, last, replacement = self.undo_stack.pop()
        current = self.draft_labels[first:last].copy()
        self.redo_stack.append((first, last, current))
        self.draft_labels[first:last] = replacement
        self._invalidate_overlapping_videos(first, last)
        self.save_draft()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        first, last, replacement = self.redo_stack.pop()
        current = self.draft_labels[first:last].copy()
        self.undo_stack.append((first, last, current))
        self.draft_labels[first:last] = replacement
        self._invalidate_overlapping_videos(first, last)
        self.save_draft()
        return True

    def confirm_video(self, video: VideoSource) -> None:
        self.video_status[self.video_key(video)] = "reviewed"
        self.save_draft()

    def commit(self) -> None:
        self.committed_labels = self.draft_labels.copy()
        self.committed_video_status = self.video_status.copy()
        payload = {
            "version": DRAFT_VERSION,
            "signature": self._signature(),
            "labels": [str(value) for value in self.committed_labels],
            "video_status": self.video_status,
        }
        _write_json_atomic(self.output_directory / "review_commit.json", payload)
        _write_corrected_csv(
            self.output_directory / "sleep_state_corrected_epochs.csv",
            self.result,
            self.committed_labels,
        )
        summary = self._summary_payload()
        _write_json_atomic(
            self.output_directory / "sleep_state_corrected_summary.json",
            summary,
        )
        corrected_result = replace(
            self.result,
            labels=self.committed_labels.copy(),
            summary={**self.result.summary, **summary},
        )
        _write_hypnogram(self, corrected_result)
        self.save_draft()

    def save_draft(self) -> None:
        payload = {
            "version": DRAFT_VERSION,
            "signature": self._signature(),
            "labels": [str(value) for value in self.draft_labels],
            "video_status": self.video_status,
        }
        _write_json_atomic(self.output_directory / "review_draft.json", payload)

    def _invalidate_overlapping_videos(self, first: int, last: int) -> None:
        start = float(self.label_starts[first])
        stop = float(self.label_stops[last - 1])
        for video in self.source.videos:
            duration = float(video.duration_sec or 0.0)
            if duration <= 0:
                continue
            if video.start_sec < stop and video.stop_sec > start:
                key = self.video_key(video)
                if self.video_status.get(key) == "reviewed":
                    self.video_status[key] = "needs_review"

    def _signature(self) -> dict[str, int | float | str]:
        times = self.result.times_sec
        return {
            "recording": str(self.recording.path),
            "n_labels": len(times),
            "first_time_sec": float(times[0]) if len(times) else 0.0,
            "last_time_sec": float(times[-1]) if len(times) else 0.0,
            "feature_window_sec": float(self.result.params.epoch_sec),
            "label_interval_sec": float(self.result.params.step_sec),
            "auto_labels_sha256": hashlib.sha256(
                "\0".join(str(value) for value in self.auto_labels).encode()
            ).hexdigest(),
        }

    def _load_draft(self) -> None:
        payload = _read_matching_payload(self.output_directory / "review_draft.json", self._signature())
        if payload is None:
            return
        labels = np.asarray(payload.get("labels", []), dtype=object)
        if len(labels) == len(self.draft_labels):
            self.draft_labels = labels
        self._restore_video_status(payload)

    def _load_commit(self) -> None:
        payload = _read_matching_payload(self.output_directory / "review_commit.json", self._signature())
        if payload is None:
            return
        labels = np.asarray(payload.get("labels", []), dtype=object)
        if len(labels) == len(self.committed_labels):
            self.committed_labels = labels
            self.draft_labels = labels.copy()
        self._restore_video_status(payload)
        self.committed_video_status = self.video_status.copy()

    def _restore_video_status(self, payload: dict) -> None:
        stored = payload.get("video_status", {})
        for key in self.video_status:
            value = stored.get(key)
            if value in {"unreviewed", "reviewed", "needs_review"}:
                self.video_status[key] = value

    def _summary_payload(self) -> dict:
        labels = self.committed_labels
        counts = {stage: int(np.sum(labels == stage)) for stage in STAGES}
        reviewed = sum(value == "reviewed" for value in self.video_status.values())
        return {
            "n_steps": len(labels),
            "label_interval_sec": self.label_interval_sec,
            "feature_window_sec": float(self.result.params.epoch_sec),
            "wake_steps": counts[WAKE],
            "nrem_steps": counts[NREM],
            "rem_steps": counts[REM],
            "corrected_steps": int(np.sum(labels != self.auto_labels)),
            "reviewed_videos": reviewed,
            "total_videos": len(self.video_status),
            "video_status": self.video_status,
        }


def write_auto_results(session: ReviewSession) -> None:
    records = session.result.to_records()
    path = session.output_directory / "sleep_state_epochs.csv"
    if records:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    payload = {
        "input": {
            "path": str(session.recording.path),
            "fs": session.recording.fs,
            "n_samples": session.recording.n_samples,
            "duration_sec": session.recording.duration_sec,
            "eeg_col": session.recording.eeg_col,
            "emg_col": session.recording.emg_col,
        },
        "staging": {
            "summary": session.result.summary,
            "thresholds": session.result.thresholds,
            "params": vars(session.result.params),
        },
    }
    _write_json_atomic(session.output_directory / "sleep_state_summary.json", payload)


def _write_hypnogram(session: ReviewSession, result: StagingResult) -> None:
    configure_matplotlib_cache()
    target = session.output_directory / "sleep_state_hypnogram.svg"
    with tempfile.NamedTemporaryFile(
        prefix=".sleep_state_hypnogram-",
        suffix=".svg",
        dir=session.output_directory,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    figure = None
    try:
        from .qc import plot_hypnogram

        figure = plot_hypnogram(
            session.processed_eeg,
            session.processed_emg,
            result,
            session.recording.fs,
            temporary,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        if figure is not None:
            from matplotlib import pyplot as plt

            plt.close(figure)


def _write_corrected_csv(path: Path, result: StagingResult, labels: np.ndarray) -> None:
    starts = result.times_sec - result.params.step_sec / 2.0
    stops = result.times_sec + result.params.step_sec / 2.0
    with path.open("w", newline="") as handle:
        fieldnames = (
            "step_idx",
            "start_sec",
            "end_sec",
            "auto_stage",
            "corrected_stage",
            "final_stage",
            "corrected",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (start, stop, auto, final) in enumerate(
            zip(starts, stops, result.labels, labels)
        ):
            corrected = bool(auto != final)
            writer.writerow(
                {
                    "step_idx": index,
                    "start_sec": round(float(start), 6),
                    "end_sec": round(float(stop), 6),
                    "auto_stage": str(auto),
                    "corrected_stage": str(final) if corrected else "",
                    "final_stage": str(final),
                    "corrected": int(corrected),
                }
            )


def _read_matching_payload(path: Path, signature: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != DRAFT_VERSION or payload.get("signature") != signature:
        return None
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
