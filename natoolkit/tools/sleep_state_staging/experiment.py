from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tifffile

from .io import EEGEMGFileInfo, probe_eegemg_txt


NOTE_MOVIE_RE = re.compile(
    r"^\s*(Wake|NREM|REM)(-CNO)?\s+(\d+)\s+(\d+):(\d+(?:\.\d+)?)\s*(.*)$",
    re.IGNORECASE,
)
SESSION_RE = re.compile(
    r"^\s*EEG\s*(\d+)\b\s*(?:(\d+):(\d+(?:\.\d+)?))?\s*(.*)$",
    re.IGNORECASE,
)
FRAME_RATE_RE = re.compile(r"Frame\s*rate\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", re.IGNORECASE)
MOVIE_FILE_RE = re.compile(r"^(Wake|NREM|REM)(.*)_(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ExperimentEvent:
    time_sec: float | None
    label: str
    raw_text: str


@dataclass(frozen=True)
class VideoSource:
    segment_number: int
    stage: str
    number: int
    start_sec: float
    comment: str
    line_number: int
    cno: bool = False
    path: Path | None = None
    n_frames: int | None = None
    duration_sec: float | None = None
    frame_rate: float | None = None

    @property
    def stop_sec(self) -> float:
        return self.start_sec + float(self.duration_sec or 0.0)

    @property
    def display_name(self) -> str:
        if self.path is not None:
            return self.path.stem
        condition = "-CNO" if self.cno else ""
        return f"{self.stage}{condition}_{self.number:05d}"


@dataclass(frozen=True)
class RecordingSource:
    number: int
    path: Path | None
    info: EEGEMGFileInfo | None
    condition: str | None
    events: tuple[ExperimentEvent, ...]
    videos: tuple[VideoSource, ...]

    @property
    def name(self) -> str:
        suffix = f" / {self.condition}" if self.condition else ""
        return f"Session {self.number}{suffix}"


@dataclass(frozen=True)
class ExperimentSource:
    directory: Path
    note_path: Path
    tiff_directory: Path
    recordings: tuple[RecordingSource, ...]
    note_frame_rate: float | None
    note_header: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_ready(self) -> bool:
        return bool(self.recordings) and all(
            recording.path is not None and all(video.path is not None for video in recording.videos)
            for recording in self.recordings
        )


@dataclass(frozen=True)
class _ParsedNote:
    frame_rate: float | None
    header: tuple[str, ...]
    videos: tuple[VideoSource, ...]
    events: dict[int, tuple[ExperimentEvent, ...]]
    segment_numbers: tuple[int, ...]


def scan_experiment(
    directory: str | Path,
    *,
    note_path: str | Path | None = None,
    recording_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    tiff_directory: str | Path | None = None,
) -> ExperimentSource:
    """Discover one experiment without recursively consuming derived outputs."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Experiment directory does not exist: {root}")

    note = Path(note_path).expanduser().resolve() if note_path else root / "Note.txt"
    if not note.is_file():
        raise ValueError(f"Note file does not exist: {note}")
    tiff_root = Path(tiff_directory).expanduser().resolve() if tiff_directory else root
    if not tiff_root.is_dir():
        raise ValueError(f"TIFF directory does not exist: {tiff_root}")

    parsed = parse_note(note)
    warnings: list[str] = []
    infos = _recording_infos(root, note, recording_paths, warnings)
    paths_by_segment = _assign_recordings(parsed.segment_numbers, infos, warnings)
    video_paths = _index_tiffs(tiff_root)

    mapped_videos: list[VideoSource] = []
    for video in parsed.videos:
        matches = video_paths.get((video.stage.upper(), video.number, video.cno), [])
        if len(matches) != 1:
            if not matches:
                warnings.append(f"No TIFF matches {video.display_name} (Note line {video.line_number}).")
            else:
                warnings.append(f"Multiple TIFF files match {video.display_name}.")
            mapped_videos.append(video)
            continue
        try:
            mapped_videos.append(_with_tiff_metadata(video, matches[0], parsed.frame_rate))
        except (OSError, ValueError, KeyError, tifffile.TiffFileError) as exc:
            warnings.append(f"Unable to read TIFF metadata from {matches[0].name}: {exc}")
            mapped_videos.append(replace(video, path=matches[0]))

    measured_rates = [video.frame_rate for video in mapped_videos if video.frame_rate]
    if parsed.frame_rate and measured_rates:
        median_rate = float(np.median(measured_rates))
        if abs(median_rate - parsed.frame_rate) / parsed.frame_rate > 0.02:
            warnings.append(
                f"Note frame rate is {parsed.frame_rate:g} Hz, but TIFF timestamps indicate "
                f"approximately {median_rate:.4g} Hz. TIFF timestamps will be used."
            )

    has_cno = any(video.cno for video in mapped_videos) or any(
        "cno" in event.label.casefold() for events in parsed.events.values() for event in events
    )
    recordings: list[RecordingSource] = []
    for number in parsed.segment_numbers:
        segment_videos = tuple(video for video in mapped_videos if video.segment_number == number)
        if any(video.cno for video in segment_videos):
            condition = "Post-CNO"
        elif has_cno:
            condition = "Pre-CNO"
        else:
            condition = None
        info = paths_by_segment.get(number)
        recordings.append(
            RecordingSource(
                number=number,
                path=info.path if info else None,
                info=info,
                condition=condition,
                events=parsed.events.get(number, ()),
                videos=segment_videos,
            )
        )

    return ExperimentSource(
        directory=root,
        note_path=note,
        tiff_directory=tiff_root,
        recordings=tuple(recordings),
        note_frame_rate=parsed.frame_rate,
        note_header=parsed.header,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def parse_note(path: str | Path) -> _ParsedNote:
    note_path = Path(path)
    lines = note_path.read_text(encoding="utf-8-sig").splitlines()
    current_segment = 1
    segment_numbers = {1}
    videos: list[VideoSource] = []
    events: dict[int, list[ExperimentEvent]] = {}
    header: list[str] = []
    frame_rate: float | None = None

    for line_number, line in enumerate(lines, 1):
        if frame_rate is None:
            match = FRAME_RATE_RE.search(line)
            if match is not None:
                frame_rate = float(match.group(1))

        session_match = SESSION_RE.match(line)
        if session_match is not None:
            number, minutes, seconds, label = session_match.groups()
            current_segment = int(number)
            segment_numbers.add(current_segment)
            time_sec = None
            if minutes is not None and seconds is not None:
                time_sec = int(minutes) * 60 + float(seconds)
            if label.strip() or time_sec is not None:
                events.setdefault(current_segment, []).append(
                    ExperimentEvent(time_sec=time_sec, label=label.strip(), raw_text=line.strip())
                )
            continue

        movie_match = NOTE_MOVIE_RE.match(line)
        if movie_match is None:
            if line.strip():
                header.append(line.strip())
            continue
        stage, cno_token, number, minutes, seconds, comment = movie_match.groups()
        videos.append(
            VideoSource(
                segment_number=current_segment,
                stage=stage.upper() if stage.upper() != "WAKE" else "Wake",
                number=int(number),
                start_sec=int(minutes) * 60 + float(seconds),
                comment=comment.strip(),
                line_number=line_number,
                cno=bool(cno_token),
            )
        )

    if not videos:
        raise ValueError(f"No movie start times were found in {note_path}.")
    normalized_videos = tuple(replace(video, stage=_normalize_stage(video.stage)) for video in videos)
    return _ParsedNote(
        frame_rate=frame_rate,
        header=tuple(header),
        videos=normalized_videos,
        events={number: tuple(values) for number, values in events.items()},
        segment_numbers=tuple(sorted(segment_numbers)),
    )


def load_video_frame_times(video: VideoSource, fallback_rate: float | None = None) -> np.ndarray:
    if video.path is None:
        return np.asarray([], dtype=np.float64)
    with tifffile.TiffFile(video.path) as tif:
        times: list[float] = []
        for page in tif.pages:
            tag = page.tags.get("ImageDescription")
            description = tifffile.matlabstr2py(tag.value) if tag is not None else {}
            value = description.get("frameTimestamps_sec") if isinstance(description, dict) else None
            if value is None:
                times = []
                break
            times.append(float(value))
    if times:
        return np.asarray(times, dtype=np.float64)
    rate = video.frame_rate or fallback_rate
    if rate is None or rate <= 0 or video.n_frames is None:
        return np.asarray([], dtype=np.float64)
    return np.arange(video.n_frames, dtype=np.float64) / rate


def _recording_infos(
    root: Path,
    note_path: Path,
    recording_paths: list[str | Path] | tuple[str | Path, ...] | None,
    warnings: list[str],
) -> list[EEGEMGFileInfo]:
    candidates = (
        [Path(path).expanduser().resolve() for path in recording_paths]
        if recording_paths is not None
        else [path for path in root.glob("*.txt") if path.resolve() != note_path.resolve()]
    )
    infos: list[EEGEMGFileInfo] = []
    for path in candidates:
        try:
            infos.append(probe_eegemg_txt(path))
        except (OSError, UnicodeError, ValueError) as exc:
            if recording_paths is not None:
                warnings.append(f"Unable to use EEG/EMG file {path.name}: {exc}")
    return sorted(infos, key=lambda info: _natural_key(info.path.name))


def _assign_recordings(
    segment_numbers: tuple[int, ...],
    infos: list[EEGEMGFileInfo],
    warnings: list[str],
) -> dict[int, EEGEMGFileInfo]:
    assigned: dict[int, EEGEMGFileInfo] = {}
    unused = list(infos)
    for info in list(unused):
        hint = _recording_number_hint(info.path.stem)
        if hint in segment_numbers and hint not in assigned:
            assigned[hint] = info
            unused.remove(info)
    for number in segment_numbers:
        if number not in assigned and unused:
            assigned[number] = unused.pop(0)
    for number in segment_numbers:
        if number not in assigned:
            warnings.append(f"Session {number} has no EEG/EMG file.")
    if unused:
        warnings.append("Some EEG/EMG files were not assigned to a Note session: " + ", ".join(i.path.name for i in unused))
    if len(infos) != len(segment_numbers):
        warnings.append(
            f"Note defines {len(segment_numbers)} session(s), but {len(infos)} EEG/EMG file(s) were detected."
        )
    return assigned


def _index_tiffs(directory: Path) -> dict[tuple[str, int, bool], list[Path]]:
    indexed: dict[tuple[str, int, bool], list[Path]] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.casefold() not in {".tif", ".tiff"}:
            continue
        match = MOVIE_FILE_RE.match(path.stem)
        if match is None:
            continue
        stage, middle, number = match.groups()
        key = (_normalize_stage(stage).upper(), int(number), "cno" in middle.casefold())
        indexed.setdefault(key, []).append(path)
    return indexed


def _with_tiff_metadata(video: VideoSource, path: Path, note_rate: float | None) -> VideoSource:
    with tifffile.TiffFile(path) as tif:
        n_frames = len(tif.pages)
        first = _page_timestamp(tif.pages[0]) if n_frames else None
        last = _page_timestamp(tif.pages[-1]) if n_frames else None
    duration = None
    frame_rate = None
    if first is not None and last is not None and n_frames > 1 and last > first:
        duration = last - first
        frame_rate = (n_frames - 1) / duration
    elif note_rate is not None and note_rate > 0 and n_frames:
        duration = (n_frames - 1) / note_rate
        frame_rate = note_rate
    return replace(
        video,
        path=path,
        n_frames=n_frames,
        duration_sec=duration,
        frame_rate=frame_rate,
    )


def _page_timestamp(page) -> float | None:
    tag = page.tags.get("ImageDescription")
    if tag is None:
        return None
    description = tifffile.matlabstr2py(tag.value)
    if not isinstance(description, dict) or "frameTimestamps_sec" not in description:
        return None
    return float(description["frameTimestamps_sec"])


def _normalize_stage(stage: str) -> str:
    value = stage.upper()
    return "Wake" if value == "WAKE" else value


def _recording_number_hint(stem: str) -> int | None:
    match = re.search(r"(?:No|EEG)[_ -]?(\d+)$", stem, re.IGNORECASE)
    return int(match.group(1)) if match is not None else None


def _natural_key(text: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text))
