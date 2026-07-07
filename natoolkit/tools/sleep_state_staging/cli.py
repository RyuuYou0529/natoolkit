from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

from .io import load_eegemg_txt
from .preprocess import preprocess_eeg_emg
from .staging import WAKE_MODES, StagingParams, classify_sleep_state


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    eegemg_path = _resolve_path(args.eegemg, "EEG/EMG file path")
    eeg_col = _resolve_int(args.eeg_col, "EEG channel", default=1)
    emg_col = _resolve_int(args.emg_col, "EMG channel", default=2)
    fs = _resolve_float(args.fs, "Sampling rate Hz", default=1000.0)
    out_dir = _resolve_output_dir(args.out, eegemg_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    recording = load_eegemg_txt(eegemg_path, eeg_col=eeg_col, emg_col=emg_col, fs=fs)

    if args.no_preprocess:
        eeg, emg = recording.eeg, recording.emg
    else:
        eeg, emg = preprocess_eeg_emg(
            recording.eeg,
            recording.emg,
            recording.fs,
            eeg_hp_cutoff=args.eeg_hp,
            emg_hp_cutoff=args.emg_hp,
            line_freq=args.line_freq,
        )

    params = StagingParams(wake_mode=args.wake_mode, epoch_sec=args.epoch_sec, step_sec=args.step_sec)
    result = classify_sleep_state(eeg, emg, recording.fs, params=params)

    epochs_path = out_dir / "sleep_state_epochs.csv"
    summary_path = out_dir / "sleep_state_summary.json"
    hypnogram_path = out_dir / f"sleep_state_hypnogram.{args.plot_format}"

    _write_epoch_csv(epochs_path, result.to_records())
    _write_summary(summary_path, recording, result, args)
    _configure_matplotlib_cache(out_dir)
    from .qc import plot_hypnogram

    plot_hypnogram(
        eeg,
        emg,
        result,
        recording.fs,
        hypnogram_path,
        max_time_bins=args.qc_max_time_bins,
        dpi=args.qc_dpi,
    )

    print("Sleep-state staging complete.")
    print(f"  epochs:   {epochs_path}")
    print(f"  summary:  {summary_path}")
    print(f"  hypnogram:{hypnogram_path}")
    print(
        "  counts:   "
        f"Wake={result.summary['wake_steps']} "
        f"NREM={result.summary['nrem_steps']} "
        f"REM={result.summary['rem_steps']}"
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m natoolkit.tools.sleep_state_staging",
        description="Classify EEG/EMG recordings into Wake, NREM, and REM.",
    )
    parser.add_argument("--eegemg", type=Path, help="Path to EEG/EMG text file.")
    parser.add_argument("--eeg-col", type=int, help="1-based EEG signal channel index.")
    parser.add_argument("--emg-col", type=int, help="1-based EMG signal channel index.")
    parser.add_argument("--fs", type=float, help="Sampling rate in Hz.")
    parser.add_argument("--out", type=Path, help="Output directory.")
    parser.add_argument("--plot-format", default="pdf", choices=("pdf", "svg", "png"))
    parser.add_argument("--qc-max-time-bins", type=int, default=3000, help="Maximum spectrogram time bins in the QC plot.")
    parser.add_argument("--qc-dpi", type=int, default=150, help="DPI for rasterized QC plot layers.")
    parser.add_argument("--wake-mode", default="auto", choices=WAKE_MODES, help="Wake scoring mode; auto follows the reference dynamic-range rule.")
    parser.add_argument("--epoch-sec", type=float, default=5.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--eeg-hp", type=float, default=0.5, help="EEG high-pass cutoff in Hz.")
    parser.add_argument("--emg-hp", type=float, default=1.0, help="EMG high-pass cutoff in Hz.")
    parser.add_argument("--line-freq", type=float, default=50.0, help="Line-noise notch frequency in Hz.")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip filtering before staging.")
    return parser.parse_args(argv)


def _resolve_path(value: Path | None, prompt: str) -> Path:
    if value is not None:
        return value
    while True:
        text = input(f"{prompt}: ").strip().strip("'\"")
        if text:
            return Path(text)


def _resolve_int(value: int | None, prompt: str, default: int) -> int:
    if value is not None:
        return value
    text = input(f"{prompt} [{default}]: ").strip()
    return int(text) if text else default


def _resolve_float(value: float | None, prompt: str, default: float) -> float:
    if value is not None:
        return value
    text = input(f"{prompt} [{default:g}]: ").strip()
    return float(text) if text else default


def _resolve_output_dir(value: Path | None, eegemg_path: Path) -> Path:
    if value is not None:
        return value
    default = eegemg_path.with_name("sleep_state_staging_output")
    text = input(f"Output directory [{default}]: ").strip().strip("'\"")
    return Path(text) if text else default


def _write_epoch_csv(path: Path, records: list[dict[str, int | float | str]]) -> None:
    if records:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = ["step_idx", "time_sec", "stage"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_summary(path: Path, recording, result, args: argparse.Namespace) -> None:
    payload = {
        "input": {
            "path": str(recording.path),
            "fs": recording.fs,
            "n_samples": recording.n_samples,
            "duration_sec": recording.duration_sec,
            "eeg_col": recording.eeg_col,
            "emg_col": recording.emg_col,
            "encoding": recording.encoding,
            "data_start_row": recording.data_start_row,
            "signal_columns": list(recording.signal_columns),
        },
        "preprocessing": {
            "enabled": not args.no_preprocess,
            "eeg_hp": args.eeg_hp,
            "emg_hp": args.emg_hp,
            "line_freq": args.line_freq,
        },
        "staging": {
            "summary": result.summary,
            "thresholds": result.thresholds,
            "params": vars(result.params),
        },
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _configure_matplotlib_cache(out_dir: Path) -> None:
    cache_dir = out_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


if __name__ == "__main__":
    raise SystemExit(main())
