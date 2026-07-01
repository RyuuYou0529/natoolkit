from __future__ import annotations

import codecs
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EEGEMGRecording:
    eeg: np.ndarray
    emg: np.ndarray
    fs: float
    path: Path
    eeg_col: int
    emg_col: int
    signal_columns: tuple[int, ...]
    data_start_row: int
    encoding: str

    @property
    def n_samples(self) -> int:
        return int(min(len(self.eeg), len(self.emg)))

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.fs


def load_eegemg_txt(
    path: str | Path,
    eeg_col: int = 1,
    emg_col: int = 2,
    fs: float | None = None,
    encodings: Iterable[str] = ("gb18030", "utf-8", "latin_1"),
) -> EEGEMGRecording:
    """Load tabular EEG/EMG text data.

    Column indices are 1-based signal-channel indices after excluding a leading
    sample-number column when one is present. This matches the lab convention:
    channel 1 = EEG, channel 2 = EMG for the Zhou example data.
    """
    file_path = Path(path)
    encoding, data_start_row, signal_columns, delimiter, detected_fs = _detect_layout(
        file_path, encodings
    )
    if fs is None:
        fs = detected_fs if detected_fs is not None else 1000.0

    if eeg_col < 1 or emg_col < 1:
        raise ValueError("eeg_col and emg_col are 1-based and must be positive.")
    if eeg_col > len(signal_columns) or emg_col > len(signal_columns):
        raise ValueError(
            f"Requested EEG/EMG columns ({eeg_col}, {emg_col}) exceed "
            f"detected signal-channel count ({len(signal_columns)})."
        )

    usecols = (signal_columns[eeg_col - 1], signal_columns[emg_col - 1])
    with codecs.open(file_path, encoding=encoding) as handle:
        data = np.loadtxt(
            handle,
            delimiter=delimiter,
            skiprows=data_start_row,
            usecols=usecols,
            dtype=np.float64,
        )
    data = np.atleast_2d(data)
    if data.shape[0] == 1 and data.shape[1] > 2:
        data = data.T

    return EEGEMGRecording(
        eeg=np.asarray(data[:, 0], dtype=np.float64),
        emg=np.asarray(data[:, 1], dtype=np.float64),
        fs=float(fs),
        path=file_path,
        eeg_col=eeg_col,
        emg_col=emg_col,
        signal_columns=signal_columns,
        data_start_row=data_start_row,
        encoding=encoding,
    )


def _detect_layout(
    path: Path, encodings: Iterable[str]
) -> tuple[str, int, tuple[int, ...], str | None, float | None]:
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return _scan_layout(path, encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to read EEG/EMG text file: {path}")


def _scan_layout(path: Path, encoding: str) -> tuple[str, int, tuple[int, ...], str | None, float | None]:
    detected_fs: float | None = None
    with codecs.open(path, encoding=encoding) as handle:
        for row_idx, line in enumerate(handle):
            if detected_fs is None:
                detected_fs = _parse_sampling_rate(line)

            delimiter = "\t" if "\t" in line else None
            parts = line.strip().split(delimiter)
            numeric_cols = _numeric_column_indices(parts)
            signal_cols = _signal_columns(numeric_cols)
            if len(signal_cols) >= 2:
                return encoding, row_idx, tuple(signal_cols), delimiter, detected_fs

    raise ValueError(f"No data row with at least two signal columns found: {path}")


def _numeric_column_indices(parts: list[str]) -> list[int]:
    numeric_cols: list[int] = []
    for idx, value in enumerate(parts):
        try:
            float(value.strip())
        except ValueError:
            continue
        numeric_cols.append(idx)
    return numeric_cols


def _signal_columns(numeric_cols: list[int]) -> list[int]:
    if len(numeric_cols) >= 3:
        return numeric_cols[1:]
    return numeric_cols


def _parse_sampling_rate(line: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*Hz", line, re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))
