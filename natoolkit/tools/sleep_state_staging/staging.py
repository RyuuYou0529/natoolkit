from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, resample, welch


WAKE = "Wake"
NREM = "NREM"
REM = "REM"
STAGES = (WAKE, NREM, REM)
WAKE_MODES = ("auto", "emg_primary", "balanced", "eeg_primary")
EPS = 1e-12


@dataclass(frozen=True)
class StagingParams:
    wake_mode: str = "auto"
    epoch_sec: float = 5.0
    step_sec: float = 1.0
    target_eeg_fs: float = 100.0
    delta_band: tuple[float, float] = (0.5, 4.0)
    theta_band: tuple[float, float] = (6.0, 10.0)
    sigma_band: tuple[float, float] = (10.0, 15.0)
    beta_band: tuple[float, float] = (15.0, 30.0)
    gamma_band: tuple[float, float] = (30.0, 50.0)
    eeg_total_band: tuple[float, float] = (0.5, 50.0)
    high_frequency_band: tuple[float, float] = (20.0, 40.0)
    emg_low_band: tuple[float, float] = (10.0, 50.0)
    emg_mid_band: tuple[float, float] = (50.0, 150.0)
    emg_high_band: tuple[float, float] = (150.0, 300.0)
    emg_clip_percentile: float = 99.0
    emg_mid_weight: float = 0.35
    normalize_low_percentile: float = 2.0
    normalize_high_percentile: float = 98.0
    strong_emg_dynamic_range: float = 5.0
    moderate_emg_dynamic_range: float = 2.0
    max_initial_wake_fraction: float = 0.70
    fallback_wake_percentile: float = 70.0
    min_bout_sec: float = 10.0
    rem_lookback_sec: float = 50.0
    allowed_wake_gap_before_rem_sec: float = 10.0
    sleep_onset_nrem_sec: float = 20.0
    microarousal_sec: float = 10.0
    sustained_wake_sec: float = 2.0
    sustained_wake_factor: float = 2.5


@dataclass(frozen=True)
class StagingResult:
    labels: np.ndarray
    times_sec: np.ndarray
    features: dict[str, np.ndarray]
    thresholds: dict[str, float]
    summary: dict[str, int | float | str]
    params: StagingParams

    def to_records(self) -> list[dict[str, int | float | str]]:
        records: list[dict[str, int | float | str]] = []
        for idx, (time_sec, stage) in enumerate(zip(self.times_sec, self.labels)):
            row: dict[str, int | float | str] = {
                "step_idx": idx,
                "time_sec": float(time_sec),
                "stage": str(stage),
            }
            for name, values in self.features.items():
                if len(values) == len(self.labels):
                    row[name] = float(values[idx])
            records.append(row)
        return records


def classify_sleep_state(
    eeg: np.ndarray,
    emg: np.ndarray,
    fs: float,
    params: StagingParams | None = None,
) -> StagingResult:
    params = params or StagingParams()
    if params.wake_mode not in WAKE_MODES:
        raise ValueError(f"wake_mode must be one of {WAKE_MODES}, got {params.wake_mode!r}")

    eeg = np.asarray(eeg, dtype=np.float64)
    emg = np.asarray(emg, dtype=np.float64)
    n_samples = min(len(eeg), len(emg))
    eeg = eeg[:n_samples]
    emg = emg[:n_samples]

    eeg_ds, fs_ds = _downsample_eeg(eeg, fs, params.target_eeg_fs)
    windows = _window_sizes(fs, fs_ds, len(emg), len(eeg_ds), params)
    if windows["n_steps"] == 0:
        return _empty_result(params)

    clip_thr = float(np.percentile(np.abs(emg), params.emg_clip_percentile))
    emg_clipped = np.clip(emg, -clip_thr, clip_thr)
    features = _extract_features(eeg_ds, emg_clipped, fs_ds, fs, windows, params)

    emg_activity = _emg_activity(features, params)
    td_ratio = features["theta"] / (features["delta"] + EPS)
    emg_norm = _percentile_normalize(emg_activity, params)
    td_norm = _percentile_normalize(td_ratio, params)
    hf_norm = _percentile_normalize(features["eeg_hf"], params)
    delta_norm = _percentile_normalize(features["delta"], params)

    noise_floor = float(np.percentile(features["emg_rms"], 10))
    peak_rms = float(np.percentile(features["emg_rms"], 95))
    dynamic_range = peak_rms / (noise_floor + 1e-9)

    wake_score, mode = _wake_score(emg_norm, hf_norm, delta_norm, dynamic_range, params)
    wake_score_norm = _percentile_normalize(wake_score, params)
    wake_thr = _otsu_threshold(wake_score_norm)
    wake_mask = wake_score_norm >= wake_thr

    if float(np.mean(wake_mask)) > params.max_initial_wake_fraction:
        mode = "eeg_primary"
        wake_score = 0.15 * emg_norm + 0.55 * hf_norm + 0.30 * (1.0 - delta_norm)
        wake_score_norm = _percentile_normalize(wake_score, params)
        wake_thr = _otsu_threshold(wake_score_norm)
        wake_mask = wake_score_norm >= wake_thr
    if float(np.mean(wake_mask)) > params.max_initial_wake_fraction:
        mode = "percentile_fallback"
        wake_thr = float(np.percentile(wake_score_norm, params.fallback_wake_percentile))
        wake_mask = wake_score_norm >= wake_thr

    td_thr = _sleep_td_threshold(td_norm, wake_mask, params)
    labels = np.where(wake_mask, WAKE, np.where(td_norm >= td_thr, REM, NREM)).astype(object)
    labels = _postprocess_labels(
        labels,
        features["emg_p90"],
        noise_floor,
        wake_score_norm,
        td_norm,
        wake_thr,
        td_thr,
        params,
    )

    features.update(
        {
            "emg_activity": emg_activity,
            "emg_norm": emg_norm,
            "theta_delta_ratio": td_ratio,
            "theta_delta_norm": td_norm,
            "eeg_hf_norm": hf_norm,
            "delta_norm": delta_norm,
            "wake_score": wake_score_norm,
        }
    )
    thresholds = {
        "emg_clip": clip_thr,
        "wake": float(wake_thr),
        "theta_delta": float(td_thr),
        "emg_noise_floor": noise_floor,
    }
    times_sec = (
        np.arange(windows["n_steps"], dtype=np.float64) * params.step_sec
        + params.step_sec / 2.0
    )
    summary = _summary(labels, dynamic_range, mode)
    return StagingResult(labels=labels, times_sec=times_sec, features=features, thresholds=thresholds, summary=summary, params=params)


def _downsample_eeg(eeg: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    if fs <= target_fs:
        return eeg, fs
    n_samples = int(round(len(eeg) * target_fs / fs))
    return resample(eeg, n_samples), target_fs


def _window_sizes(
    fs_emg: float,
    fs_eeg: float,
    n_emg: int,
    n_eeg: int,
    params: StagingParams,
) -> dict[str, int]:
    win_eeg = int(round(params.epoch_sec * fs_eeg))
    win_emg = int(round(params.epoch_sec * fs_emg))
    step_eeg = int(round(params.step_sec * fs_eeg))
    step_emg = int(round(params.step_sec * fs_emg))
    if min(win_eeg, win_emg, step_eeg, step_emg) <= 0:
        return {"n_steps": 0}
    n_steps = min((n_eeg - win_eeg) // step_eeg + 1, (n_emg - win_emg) // step_emg + 1)
    return {
        "win_eeg": win_eeg,
        "win_emg": win_emg,
        "step_eeg": step_eeg,
        "step_emg": step_emg,
        "n_steps": max(int(n_steps), 0),
    }


def _extract_features(
    eeg_ds: np.ndarray,
    emg: np.ndarray,
    fs_eeg: float,
    fs_emg: float,
    windows: dict[str, int],
    params: StagingParams,
) -> dict[str, np.ndarray]:
    n_steps = windows["n_steps"]
    features = {
        "delta": np.zeros(n_steps, dtype=np.float64),
        "theta": np.zeros(n_steps, dtype=np.float64),
        "sigma": np.zeros(n_steps, dtype=np.float64),
        "beta": np.zeros(n_steps, dtype=np.float64),
        "gamma": np.zeros(n_steps, dtype=np.float64),
        "eeg_total": np.zeros(n_steps, dtype=np.float64),
        "eeg_hf": np.zeros(n_steps, dtype=np.float64),
        "eeg_delta_ratio": np.zeros(n_steps, dtype=np.float64),
        "eeg_theta_ratio": np.zeros(n_steps, dtype=np.float64),
        "eeg_spectral_entropy": np.zeros(n_steps, dtype=np.float64),
        "emg_rms": np.zeros(n_steps, dtype=np.float64),
        "emg_mean_abs": np.zeros(n_steps, dtype=np.float64),
        "emg_std": np.zeros(n_steps, dtype=np.float64),
        "emg_p90": np.zeros(n_steps, dtype=np.float64),
        "emg_p95": np.zeros(n_steps, dtype=np.float64),
        "emg_cv": np.zeros(n_steps, dtype=np.float64),
        "emg_power": np.zeros(n_steps, dtype=np.float64),
        "emg_burst_rate": np.zeros(n_steps, dtype=np.float64),
        "emg_low": np.zeros(n_steps, dtype=np.float64),
        "emg_mid": np.zeros(n_steps, dtype=np.float64),
        "emg_high": np.zeros(n_steps, dtype=np.float64),
        "emg_total": np.zeros(n_steps, dtype=np.float64),
        "emg_mid_low_ratio": np.zeros(n_steps, dtype=np.float64),
        "emg_high_mid_ratio": np.zeros(n_steps, dtype=np.float64),
        "emg_spectral_entropy": np.zeros(n_steps, dtype=np.float64),
        "emg_peak_freq": np.zeros(n_steps, dtype=np.float64),
    }
    emg_band = _bandpass_emg(emg, fs_emg, params.emg_low_band[0], params.emg_high_band[1])

    for idx in range(n_steps):
        eeg_start = idx * windows["step_eeg"]
        emg_start = idx * windows["step_emg"]
        eeg_epoch = eeg_ds[eeg_start : eeg_start + windows["win_eeg"]]
        emg_epoch = emg[emg_start : emg_start + windows["win_emg"]]
        emg_band_epoch = emg_band[emg_start : emg_start + windows["win_emg"]]

        _fill_eeg_features(features, idx, eeg_epoch, fs_eeg, params)
        _fill_emg_features(features, idx, emg_epoch, emg_band_epoch, fs_emg, params)

    return features


def _fill_eeg_features(
    features: dict[str, np.ndarray],
    idx: int,
    eeg_epoch: np.ndarray,
    fs: float,
    params: StagingParams,
) -> None:
    eeg = np.asarray(eeg_epoch, dtype=np.float64)
    eeg = eeg[np.isfinite(eeg)]
    if eeg.size < 4:
        return
    freqs, psd = _welch(eeg - np.mean(eeg), fs)
    delta = _band_power_from_psd(freqs, psd, params.delta_band)
    theta = _band_power_from_psd(freqs, psd, params.theta_band)
    total = _band_power_from_psd(freqs, psd, params.eeg_total_band)

    features["delta"][idx] = delta
    features["theta"][idx] = theta
    features["sigma"][idx] = _band_power_from_psd(freqs, psd, params.sigma_band)
    features["beta"][idx] = _band_power_from_psd(freqs, psd, params.beta_band)
    features["gamma"][idx] = _band_power_from_psd(freqs, psd, params.gamma_band)
    features["eeg_total"][idx] = total
    features["eeg_hf"][idx] = _band_power_from_psd(freqs, psd, params.high_frequency_band)
    features["eeg_delta_ratio"][idx] = delta / (total + EPS)
    features["eeg_theta_ratio"][idx] = theta / (total + EPS)
    features["eeg_spectral_entropy"][idx] = _spectral_entropy(freqs, psd, params.eeg_total_band)


def _fill_emg_features(
    features: dict[str, np.ndarray],
    idx: int,
    emg_epoch: np.ndarray,
    emg_band_epoch: np.ndarray,
    fs: float,
    params: StagingParams,
) -> None:
    emg = np.asarray(emg_epoch, dtype=np.float64)
    emg = emg[np.isfinite(emg)]
    if emg.size == 0:
        return

    emg = emg - np.mean(emg)
    abs_emg = np.abs(emg)
    p90 = float(np.percentile(abs_emg, 90))
    features["emg_rms"][idx] = float(np.sqrt(np.mean(emg * emg)))
    features["emg_mean_abs"][idx] = float(np.mean(abs_emg))
    features["emg_std"][idx] = float(np.std(emg))
    features["emg_p90"][idx] = p90
    features["emg_p95"][idx] = float(np.percentile(abs_emg, 95))
    features["emg_cv"][idx] = float(np.std(abs_emg) / (np.mean(abs_emg) + EPS))
    features["emg_power"][idx] = float(np.mean(emg * emg))
    features["emg_burst_rate"][idx] = float(np.mean(abs_emg > p90))

    if emg_band_epoch.size < 16:
        return
    freqs, psd = _welch(emg_band_epoch - np.mean(emg_band_epoch), fs)
    low = _band_power_from_psd(freqs, psd, params.emg_low_band)
    mid = _band_power_from_psd(freqs, psd, params.emg_mid_band)
    high = _band_power_from_psd(freqs, psd, params.emg_high_band)
    total = _band_power_from_psd(freqs, psd, (params.emg_low_band[0], params.emg_high_band[1]))

    features["emg_low"][idx] = low
    features["emg_mid"][idx] = mid
    features["emg_high"][idx] = high
    features["emg_total"][idx] = total
    features["emg_mid_low_ratio"][idx] = mid / (low + EPS)
    features["emg_high_mid_ratio"][idx] = high / (mid + EPS)
    emg_total_band = (params.emg_low_band[0], params.emg_high_band[1])
    features["emg_spectral_entropy"][idx] = _spectral_entropy(freqs, psd, emg_total_band)
    features["emg_peak_freq"][idx] = _peak_frequency(freqs, psd, emg_total_band)


def _emg_activity(features: dict[str, np.ndarray], params: StagingParams) -> np.ndarray:
    geometric = (features["emg_rms"] * features["emg_p90"] * features["emg_cv"]) ** (1.0 / 3.0)
    geometric = geometric / (np.max(geometric) + EPS)
    mid = _percentile_normalize(np.log1p(np.maximum(features["emg_mid"], 0.0)), params)
    mid_weight = float(np.clip(params.emg_mid_weight, 0.0, 1.0))
    features["emg_geometric"] = geometric
    return _percentile_normalize((1.0 - mid_weight) * geometric + mid_weight * mid, params)


def _wake_score(
    emg_norm: np.ndarray,
    hf_norm: np.ndarray,
    delta_norm: np.ndarray,
    dynamic_range: float,
    params: StagingParams,
) -> tuple[np.ndarray, str]:
    mode = params.wake_mode
    if mode == "auto":
        if dynamic_range >= params.strong_emg_dynamic_range:
            mode = "emg_primary"
        elif dynamic_range >= params.moderate_emg_dynamic_range:
            mode = "balanced"
        else:
            mode = "eeg_primary"
    if mode == "emg_primary":
        return 0.75 * emg_norm + 0.25 * hf_norm, mode
    if mode == "balanced":
        return 0.45 * emg_norm + 0.35 * hf_norm + 0.20 * (1.0 - delta_norm), mode
    return 0.20 * emg_norm + 0.50 * hf_norm + 0.30 * (1.0 - delta_norm), mode


def _sleep_td_threshold(td_norm: np.ndarray, wake_mask: np.ndarray, params: StagingParams) -> float:
    stride = max(1, int(round(params.epoch_sec / params.step_sec)))
    sleep_td = td_norm[::stride][~wake_mask[::stride]]
    return _otsu_threshold(sleep_td) if len(sleep_td) > 5 else 0.5


def _postprocess_labels(
    labels: np.ndarray,
    emg_p90: np.ndarray,
    noise_floor: float,
    wake_score_norm: np.ndarray,
    td_norm: np.ndarray,
    wake_thr: float,
    td_thr: float,
    params: StagingParams,
) -> np.ndarray:
    labels = _detect_sustained_wake(labels, emg_p90, noise_floor, params)
    labels = _merge_short(labels, _steps(params.min_bout_sec, params.step_sec))
    labels = _enforce_valid_transitions(labels, wake_score_norm, td_norm, wake_thr, td_thr)
    labels = _validate_rem_anchor(labels, wake_score_norm, td_norm, wake_thr, td_thr, params)
    labels = _merge_short(labels, _steps(params.min_bout_sec, params.step_sec))
    labels = _enforce_valid_transitions(labels, wake_score_norm, td_norm, wake_thr, td_thr)
    labels = _enforce_sleep_onset(labels, _steps(params.sleep_onset_nrem_sec, params.step_sec))
    labels = _absorb_microarousals(labels, _steps(params.microarousal_sec, params.step_sec))
    labels = _enforce_rem_after_nrem(labels)
    return _enforce_valid_transitions(labels, wake_score_norm, td_norm, wake_thr, td_thr)


def _detect_sustained_wake(labels: np.ndarray, emg_p90: np.ndarray, noise_floor: float, params: StagingParams) -> np.ndarray:
    out = labels.copy()
    threshold = noise_floor * params.sustained_wake_factor
    min_steps = _steps(params.sustained_wake_sec, params.step_sec)
    idx = 0
    while idx < len(out):
        if out[idx] in (NREM, REM) and emg_p90[idx] > threshold:
            stop = idx
            while stop < len(out) and emg_p90[stop] > threshold:
                stop += 1
            if stop - idx >= min_steps:
                out[idx:stop] = WAKE
            idx = stop
        else:
            idx += 1
    return out


def _merge_short(labels: np.ndarray, min_steps: int) -> np.ndarray:
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        idx = 0
        while idx < len(out):
            stop = idx
            while stop < len(out) and out[stop] == out[idx]:
                stop += 1
            if stop - idx < min_steps and idx > 0 and stop < len(out):
                out[idx:stop] = out[idx - 1]
                changed = True
            idx = stop
    return out


def _enforce_valid_transitions(
    labels: np.ndarray,
    wake_score_norm: np.ndarray,
    td_norm: np.ndarray,
    wake_thr: float,
    td_thr: float,
) -> np.ndarray:
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        for idx in range(1, len(out)):
            if out[idx - 1] == WAKE and out[idx] == REM:
                out[idx] = _resolve_non_rem(wake_score_norm[idx], td_norm[idx], wake_thr, td_thr)
                changed = True
    return out


def _validate_rem_anchor(
    labels: np.ndarray,
    wake_score_norm: np.ndarray,
    td_norm: np.ndarray,
    wake_thr: float,
    td_thr: float,
    params: StagingParams,
) -> np.ndarray:
    out = labels.copy()
    lookback = _steps(params.rem_lookback_sec, params.step_sec)
    allowed_wake_gap = _steps(params.allowed_wake_gap_before_rem_sec, params.step_sec)
    for idx in range(len(out)):
        if out[idx] != REM:
            continue
        found_nrem = False
        wake_gap = 0
        for prev in range(idx - 1, max(idx - lookback - 1, -1), -1):
            if out[prev] == REM:
                wake_gap = 0
            elif out[prev] == NREM:
                found_nrem = True
                break
            elif out[prev] == WAKE:
                wake_gap += 1
                if wake_gap > allowed_wake_gap:
                    break
        if not found_nrem:
            out[idx] = _resolve_non_rem(wake_score_norm[idx], td_norm[idx], wake_thr, td_thr)
    return out


def _enforce_sleep_onset(labels: np.ndarray, min_nrem_steps: int) -> np.ndarray:
    out = labels.copy()
    idx = 0
    while idx < len(out):
        if out[idx] == NREM and (idx == 0 or out[idx - 1] == WAKE):
            stop = idx
            while stop < len(out) and out[stop] == NREM:
                stop += 1
            if stop - idx < min_nrem_steps:
                out[idx:stop] = WAKE
                idx = stop
                continue
        idx += 1
    return out


def _absorb_microarousals(labels: np.ndarray, max_wake_steps: int) -> np.ndarray:
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        idx = 1
        while idx < len(out) - 1:
            if out[idx] != WAKE:
                idx += 1
                continue
            stop = idx
            while stop < len(out) and out[stop] == WAKE:
                stop += 1
            if stop - idx <= max_wake_steps and out[idx - 1] == NREM and stop < len(out) and out[stop] == NREM:
                out[idx:stop] = NREM
                changed = True
                idx = stop
                continue
            idx += 1
    return out


def _enforce_rem_after_nrem(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    idx = 0
    while idx < len(out):
        if out[idx] != REM:
            idx += 1
            continue
        stop = idx
        while stop < len(out) and out[stop] == REM:
            stop += 1
        if idx == 0 or out[idx - 1] != NREM:
            out[idx:stop] = NREM
        idx = stop
    return out


def _resolve_non_rem(wake_score: float, td_score: float, wake_thr: float, td_thr: float) -> str:
    wake_evidence = wake_score - wake_thr
    nrem_evidence = td_thr - td_score
    return WAKE if wake_evidence >= nrem_evidence else NREM


def _bandpass_emg(emg: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    signal = np.asarray(emg, dtype=np.float64)
    if signal.size < 12:
        return signal - np.mean(signal)
    nyquist = fs / 2.0
    high_eff = min(high, nyquist - 1.0)
    if low <= 0 or high_eff <= low:
        return signal - np.mean(signal)
    b, a = butter(4, [low / nyquist, high_eff / nyquist], btype="band")
    if signal.size <= 3 * max(len(a), len(b)):
        return signal - np.mean(signal)
    return filtfilt(b, a, signal)


def _welch(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    return welch(signal, fs=fs, nperseg=min(len(signal), int(round(2 * fs))))


def _band_power_from_psd(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    idx = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(idx):
        return 0.0
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(psd[idx], freqs[idx]))


def _spectral_entropy(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    idx = (freqs >= band[0]) & (freqs < band[1])
    power = np.asarray(psd[idx], dtype=np.float64)
    if power.size == 0:
        return 0.0
    prob = power / (np.sum(power) + EPS)
    return float(-np.sum(prob * np.log2(prob + EPS)) / np.log2(max(len(prob), 2)))


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    idx = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(idx):
        return 0.0
    return float(freqs[idx][np.argmax(psd[idx])])


def _otsu_threshold(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 0.5
    n = len(clean)
    best_thr = float(np.median(clean))
    best_var = 0.0
    low = float(np.percentile(clean, 2))
    high = float(np.percentile(clean, 98))
    for threshold in np.linspace(low, high, 300):
        lower = clean[clean <= threshold]
        upper = clean[clean > threshold]
        if len(lower) == 0 or len(upper) == 0:
            continue
        between_var = (len(lower) / n) * (len(upper) / n) * (np.mean(lower) - np.mean(upper)) ** 2
        if between_var > best_var:
            best_var = float(between_var)
            best_thr = float(threshold)
    return best_thr


def _percentile_normalize(values: np.ndarray, params: StagingParams) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.nanpercentile(values, params.normalize_low_percentile))
    high = float(np.nanpercentile(values, params.normalize_high_percentile))
    if high <= low:
        return np.zeros_like(values, dtype=np.float64)
    out = np.clip((values - low) / (high - low), 0.0, 1.0)
    out[~np.isfinite(values)] = 0.0
    return out


def _steps(duration_sec: float, step_sec: float) -> int:
    return max(1, int(round(duration_sec / step_sec)))


def _summary(labels: np.ndarray, dynamic_range: float, mode: str) -> dict[str, int | float | str]:
    counts = Counter(str(label) for label in labels)
    n_steps = len(labels)
    return {
        "n_steps": int(n_steps),
        "wake_steps": int(counts.get(WAKE, 0)),
        "nrem_steps": int(counts.get(NREM, 0)),
        "rem_steps": int(counts.get(REM, 0)),
        "wake_fraction": float(counts.get(WAKE, 0) / n_steps) if n_steps else 0.0,
        "nrem_fraction": float(counts.get(NREM, 0) / n_steps) if n_steps else 0.0,
        "rem_fraction": float(counts.get(REM, 0) / n_steps) if n_steps else 0.0,
        "dynamic_range": float(dynamic_range),
        "mode": mode,
    }


def _empty_result(params: StagingParams) -> StagingResult:
    return StagingResult(
        labels=np.asarray([], dtype=object),
        times_sec=np.asarray([], dtype=np.float64),
        features={},
        thresholds={},
        summary=_summary(np.asarray([], dtype=object), 0.0, "empty"),
        params=params,
    )
