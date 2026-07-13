from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks


@dataclass(frozen=True)
class FrameSettings:
    fps: int
    baseline_ms: int
    lowpass_hz: float
    highpass_hz: float
    isi_ms: float
    threshold: float
    negative_signal: bool


def process_frame_trace(trace: np.ndarray, settings: FrameSettings) -> dict[str, np.ndarray | float]:
    trace = np.asarray(trace, dtype=float)
    nyquist = settings.fps / 2
    if settings.lowpass_hz >= nyquist or settings.highpass_hz >= nyquist:
        zeros = np.zeros_like(trace)
        return {
            "baseline": zeros,
            "detrended": zeros,
            "suprathreshold": zeros,
            "spike_trace": zeros,
            "noise": 0.0,
            "snr_trace": zeros,
        }
    baseline_frames = max(1, round(settings.baseline_ms * settings.fps / 1000))
    baseline = median_filter(trace, size=baseline_frames, mode="nearest")
    detrended = trace - baseline
    b, a = butter(5, settings.lowpass_hz / nyquist, btype="low")
    suprathreshold = filtfilt(b, a, detrended)
    spike_trace = -suprathreshold if settings.negative_signal else suprathreshold
    b, a = butter(5, settings.highpass_hz / nyquist, btype="high")
    noise = float(np.std(filtfilt(b, a, spike_trace)))
    snr_trace = spike_trace / noise if noise else np.zeros_like(spike_trace)
    return {
        "baseline": baseline,
        "detrended": detrended,
        "suprathreshold": suprathreshold,
        "spike_trace": spike_trace,
        "noise": noise,
        "snr_trace": snr_trace,
    }


def find_frame_spikes(trace: np.ndarray, settings: FrameSettings) -> list[dict[str, float]]:
    processed = process_frame_trace(trace, settings)
    distance = max(1, round(settings.isi_ms * settings.fps / 1000))
    peaks, _ = find_peaks(processed["snr_trace"], height=settings.threshold, distance=distance)
    return [
        {
            "frame": int(frame),
            "peak": float(processed["spike_trace"][frame]),
            "noise": float(processed["noise"]),
            "snr": float(processed["snr_trace"][frame]),
        }
        for frame in peaks
    ]


def normalize_trace(
    trace: np.ndarray,
    mode: str,
    f0_percent: int,
    frame_settings: FrameSettings,
) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    if mode == "FRAME_SNR":
        return process_frame_trace(trace, frame_settings)["snr_trace"]
    if mode in {"dF/F0", "SNR(dF/F0)"}:
        f0 = np.percentile(trace, f0_percent)
        normalized = (trace - f0) / f0 if f0 else np.zeros_like(trace)
        signal = -normalized if frame_settings.negative_signal else normalized
        if mode == "SNR(dF/F0)":
            baseline = np.median(signal)
            noise = 1.4826 * np.median(np.abs(signal - baseline))
            return (signal - baseline) / noise if noise else np.zeros_like(trace)
        return signal
    if mode == "Z-score":
        std = trace.std()
        return (trace - trace.mean()) / std if std else np.zeros_like(trace)
    if mode == "Min-max":
        span = trace.max() - trace.min()
        return (trace - trace.min()) / span if span else np.zeros_like(trace)
    return trace
