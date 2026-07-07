from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def remove_dc_offset(data: np.ndarray, fs: float, hp_cutoff: float = 0.5) -> np.ndarray:
    signal = np.asarray(data, dtype=np.float64)
    nyquist = fs / 2.0
    if hp_cutoff <= 0 or hp_cutoff >= nyquist:
        return signal.copy()
    b, a = butter(2, hp_cutoff / nyquist, "highpass")
    return filtfilt(b, a, signal)


def remove_power_interference(
    data: np.ndarray,
    fs: float,
    line_freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    signal = np.asarray(data, dtype=np.float64)
    nyquist = fs / 2.0
    if line_freq <= 0 or line_freq >= nyquist:
        return signal.copy()
    b, a = iirnotch(line_freq / nyquist, quality_factor)
    return filtfilt(b, a, signal)


def preprocess_eeg_emg(
    eeg: np.ndarray,
    emg: np.ndarray,
    fs: float,
    eeg_hp_cutoff: float = 0.5,
    emg_hp_cutoff: float = 1.0,
    line_freq: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_f = remove_dc_offset(eeg, fs, eeg_hp_cutoff)
    eeg_f = remove_power_interference(eeg_f, fs, line_freq)
    emg_f = remove_dc_offset(emg, fs, emg_hp_cutoff)
    emg_f = remove_power_interference(emg_f, fs, line_freq)
    n = min(len(eeg_f), len(emg_f))
    return eeg_f[:n], emg_f[:n]
