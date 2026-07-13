from __future__ import annotations

import csv
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import tifffile


NOTE_ROW = re.compile(
    r"^\s*(Wake|NREM|REM)\s+(\d+)\s+(\d+):(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
MOVIE_KEY = re.compile(r"(Wake|NREM|REM).*?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SleepIntervals:
    starts_sec: tuple[float, ...]
    stops_sec: tuple[float, ...]
    labels: tuple[str, ...]

    def label_at(self, time_sec: float) -> str:
        index = bisect_right(self.starts_sec, time_sec) - 1
        if index < 0 or index >= len(self.labels):
            return "Unknown"
        return self.labels[index] if time_sec < self.stops_sec[index] else "Unknown"


@dataclass(frozen=True)
class AlignmentResult:
    output_path: Path
    row_count: int
    unknown_count: int


def align_activity_file(
    note_path: str | Path,
    labels_path: str | Path,
    activity_path: str | Path,
    tiff_dir: str | Path,
    output_path: str | Path,
) -> AlignmentResult:
    activity_path = Path(activity_path)
    destination = Path(output_path)
    if activity_path.resolve() == destination.resolve():
        raise ValueError("Output CSV must be different from the input activity CSV.")

    note_times = load_note_times(note_path)
    intervals = load_sleep_intervals(labels_path)
    fieldnames, rows = _load_activity_rows(activity_path)
    timestamps = _load_frame_timestamps(rows, Path(tiff_dir))

    unknown_count = 0
    for row in rows:
        movie = row["movie"]
        key = movie_key(movie)
        if key not in note_times:
            raise ValueError(f"No Note.txt entry matches activity movie: {movie}")
        frame = _parse_frame(row["source_frame"], movie)
        eeg_time_sec = note_times[key] + timestamps[(movie, frame)]
        row["sleep_state"] = intervals.label_at(eeg_time_sec)
        unknown_count += row["sleep_state"] == "Unknown"

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "sleep_state"])
        writer.writeheader()
        writer.writerows(rows)

    return AlignmentResult(destination, len(rows), unknown_count)


def load_note_times(path: str | Path) -> dict[tuple[str, int], float]:
    note_times: dict[tuple[str, int], float] = {}
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            match = NOTE_ROW.match(line)
            if match is None:
                continue
            stage, movie_number, minutes, seconds = match.groups()
            key = stage.upper(), int(movie_number)
            if key in note_times:
                raise ValueError(f"Duplicate Note.txt entry at line {line_number}: {stage} {movie_number}")
            note_times[key] = int(minutes) * 60 + float(seconds)
    if not note_times:
        raise ValueError("No movie start times were found in Note.txt.")
    return note_times


def load_sleep_intervals(path: str | Path) -> SleepIntervals:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)

    if not rows:
        raise ValueError("The sleep-state label CSV contains no rows.")
    if {"start_sec", "end_sec", "final_stage"} <= fieldnames:
        starts = [float(row["start_sec"]) for row in rows]
        stops = [float(row["end_sec"]) for row in rows]
        labels = [row["final_stage"].strip() for row in rows]
    elif {"time_sec", "stage"} <= fieldnames:
        centers = [float(row["time_sec"]) for row in rows]
        step = median(b - a for a, b in zip(centers, centers[1:])) if len(centers) > 1 else 1.0
        starts = [center - step / 2 for center in centers]
        stops = [center + step / 2 for center in centers]
        labels = [row["stage"].strip() for row in rows]
    else:
        raise ValueError(
            "Sleep-state CSV must contain start_sec/end_sec/final_stage "
            "or time_sec/stage columns."
        )

    if any(stop <= start for start, stop in zip(starts, stops)):
        raise ValueError("Sleep-state CSV contains a non-positive interval.")
    if any(current < previous for previous, current in zip(starts, starts[1:])):
        raise ValueError("Sleep-state CSV intervals are not ordered by time.")
    if any(not label for label in labels):
        raise ValueError("Sleep-state CSV contains an empty label.")
    return SleepIntervals(tuple(starts), tuple(stops), tuple(labels))


def movie_key(movie: str) -> tuple[str, int]:
    match = MOVIE_KEY.search(Path(movie).stem)
    if match is None:
        raise ValueError(f"Cannot identify sleep state and movie number from: {movie}")
    return match.group(1).upper(), int(match.group(2))


def _load_activity_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"movie", "source_frame"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Activity CSV is missing columns: {', '.join(sorted(missing))}")
        if "sleep_state" in fieldnames:
            raise ValueError("Activity CSV already contains a sleep_state column.")
        rows = list(reader)
    if not rows:
        raise ValueError("The activity CSV contains no rows.")
    return fieldnames, rows


def _load_frame_timestamps(
    rows: list[dict[str, str]],
    tiff_dir: Path,
) -> dict[tuple[str, int], float]:
    tiff_paths = _index_tiffs(tiff_dir)
    frames_by_movie: dict[str, set[int]] = {}
    for row in rows:
        movie = row["movie"]
        frames_by_movie.setdefault(movie, set()).add(_parse_frame(row["source_frame"], movie))

    timestamps: dict[tuple[str, int], float] = {}
    for movie, frames in frames_by_movie.items():
        path = tiff_paths.get(Path(movie).stem.casefold())
        if path is None:
            raise ValueError(f"No TIFF file matches activity movie: {movie}")
        with tifffile.TiffFile(path) as tif:
            page_count = len(tif.pages)
            for frame in frames:
                if frame >= page_count:
                    raise ValueError(
                        f"source_frame {frame} exceeds the {page_count} pages in {path.name}."
                    )
                tag = tif.pages[frame].tags.get("ImageDescription")
                if tag is None:
                    raise ValueError(f"Missing ImageDescription in {path.name} page {frame}.")
                description = tifffile.matlabstr2py(tag.value)
                if "frameTimestamps_sec" not in description:
                    raise ValueError(f"Missing frameTimestamps_sec in {path.name} page {frame}.")
                timestamps[(movie, frame)] = float(description["frameTimestamps_sec"])
    return timestamps


def _index_tiffs(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"TIFF directory does not exist: {directory}")
    paths: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}:
            key = path.stem.casefold()
            if key in paths:
                raise ValueError(f"Duplicate TIFF movie name: {path.stem}")
            paths[key] = path
    return paths


def _parse_frame(value: str, movie: str) -> int:
    try:
        frame = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid source_frame for {movie}: {value}") from exc
    if frame < 0:
        raise ValueError(f"Negative source_frame for {movie}: {frame}")
    return frame
