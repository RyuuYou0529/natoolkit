from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import spectrogram

from .staging import NREM, REM, WAKE, StagingResult


matplotlib.use("Agg", force=True)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

STAGE_COLORS = {WAKE: "#3863FF", NREM: "#E55E94", REM: "#FFA233"}


def plot_hypnogram(
    eeg: np.ndarray,
    emg: np.ndarray,
    result: StagingResult,
    fs: float,
    output_path: str | Path | None = None,
    title: str | None = None,
    start_sec: float = 0.0,
    stop_sec: float | None = None,
    max_time_bins: int = 3000,
    dpi: int = 150,
):
    eeg = np.asarray(eeg, dtype=np.float64)
    emg = np.asarray(emg, dtype=np.float64)
    n = min(len(eeg), len(emg))
    stop_sec = min(stop_sec if stop_sec is not None else n / fs, n / fs)
    start_sample = max(0, int(round(start_sec * fs)))
    stop_sample = min(n, int(round(stop_sec * fs)))
    eeg_seg = eeg[start_sample:stop_sample]
    emg_seg = emg[start_sample:stop_sample]
    duration_sec = len(eeg_seg) / fs

    freqs, spec_times, spec_power, vmin, vmax = _spectrogram(eeg_seg, fs, max_time_bins=max_time_bins)
    fig = plt.figure(figsize=(22, 7))
    grid = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[3.2, 1.4, 0.6],
        width_ratios=[1.0, 0.015],
        hspace=0.08,
        wspace=0.02,
        left=0.06,
        right=0.96,
        top=0.93,
        bottom=0.14,
    )
    ax_eeg = fig.add_subplot(grid[0, 0])
    ax_colorbar = fig.add_subplot(grid[0, 1])
    ax_emg = fig.add_subplot(grid[1, 0])
    fig.add_subplot(grid[1, 1]).axis("off")
    ax_stage = fig.add_subplot(grid[2, 0])
    fig.add_subplot(grid[2, 1]).axis("off")

    im = ax_eeg.imshow(
        spec_power,
        aspect="auto",
        origin="lower",
        extent=_image_extent(spec_times, freqs, duration_sec),
        cmap="jet",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    colorbar = fig.colorbar(im, cax=ax_colorbar)
    colorbar.set_label("Power (dB)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    ax_eeg.set_ylabel("EEG\nFreq. (Hz)", fontsize=11)
    ax_eeg.set_ylim(0, 30)
    ax_eeg.set_xlim(0, duration_sec)
    ax_eeg.tick_params(labelbottom=False)
    ax_eeg.set_title(_figure_title(result, title, duration_sec), fontsize=11, fontweight="bold")

    display_clip = max(float(np.percentile(np.abs(emg_seg), 99.5)), 1e-9)
    step = max(1, len(emg_seg) // 30000)
    emg_t = np.arange(len(emg_seg), dtype=np.float64) / fs
    ax_emg.plot(
        emg_t[::step],
        np.clip(emg_seg, -display_clip, display_clip)[::step],
        color="black",
        lw=0.35,
        rasterized=True,
    )
    ax_emg.set_ylabel("EMG", fontsize=11)
    ax_emg.set_xlim(0, duration_sec)
    ax_emg.set_ylim(-display_clip * 1.2, display_clip * 1.2)
    ax_emg.tick_params(labelbottom=False)

    _plot_stage_bar(ax_stage, result, start_sec, stop_sec)
    ax_stage.set_yticks([])
    ax_stage.set_xlabel(f"Time (s)  [mode: {result.summary.get('mode', 'unknown')}]", fontsize=10)
    ax_stage.set_xlim(0, duration_sec)
    handles = [Patch(facecolor=STAGE_COLORS[stage], label=stage) for stage in (WAKE, NREM, REM)]
    ax_stage.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -1.45), ncol=3, frameon=False)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def _spectrogram(eeg: np.ndarray, fs: float, max_time_bins: int):
    nperseg = min(len(eeg), int(round(5 * fs)))
    noverlap = min(max(0, nperseg - int(round(fs))), nperseg - 1)
    nfft = max(4096, nperseg)
    freqs, times, power = spectrogram(eeg, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_mask = (freqs >= 0.0) & (freqs <= 30.0)
    power_db = 10 * np.log10(power[freq_mask, :] + 1e-15)
    vmin = float(np.percentile(power_db, 5))
    vmax = vmin + 30.0
    if max_time_bins > 0 and power_db.shape[1] > max_time_bins:
        keep = np.linspace(0, power_db.shape[1] - 1, max_time_bins).astype(np.int64)
        times = times[keep]
        power_db = power_db[:, keep]
    power_db = np.clip(power_db, vmin, vmax)
    return freqs[freq_mask], times, power_db, vmin, vmax


def _figure_title(result: StagingResult, title: str | None, duration_sec: float) -> str:
    if title:
        return title
    return (
        "EEG/EMG Sleep-State Staging  |  "
        f"Wake={result.summary.get('wake_steps', 0)}  "
        f"NREM={result.summary.get('nrem_steps', 0)}  "
        f"REM={result.summary.get('rem_steps', 0)} steps "
        f"({duration_sec / 60.0:.0f} min)"
    )


def _image_extent(times: np.ndarray, freqs: np.ndarray, duration_sec: float) -> tuple[float, float, float, float]:
    x0 = float(times[0]) if len(times) else 0.0
    x1 = float(times[-1]) if len(times) > 1 else duration_sec
    if x1 <= x0:
        x1 = max(duration_sec, x0 + 1.0)
    return x0, x1, float(freqs[0]), float(freqs[-1])


def _plot_stage_bar(ax, result: StagingResult, start_sec: float, stop_sec: float) -> None:
    labels = np.asarray(result.labels)
    times = np.asarray(result.times_sec, dtype=np.float64)
    keep = (times >= start_sec) & (times <= stop_sec)
    labels = labels[keep]
    times = times[keep] - start_sec
    if labels.size == 0:
        return

    stage_ids = np.full(labels.shape, 3, dtype=np.int16)
    stage_ids[labels == WAKE] = 0
    stage_ids[labels == NREM] = 1
    stage_ids[labels == REM] = 2
    cmap = ListedColormap([STAGE_COLORS[WAKE], STAGE_COLORS[NREM], STAGE_COLORS[REM], "#AAAAAA"])
    ax.imshow(
        stage_ids[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        extent=_stage_extent(times, result.params.step_sec, stop_sec - start_sec),
        rasterized=True,
    )


def _stage_extent(times: np.ndarray, step_sec: float, duration_sec: float) -> tuple[float, float, float, float]:
    half_step = step_sec / 2.0
    x0 = max(0.0, float(times[0]) - half_step)
    x1 = min(duration_sec, float(times[-1]) + half_step)
    if x1 <= x0:
        x1 = duration_sec
    return x0, x1, 0.0, 1.0
