# Sleep State Staging

`sleep_state_staging` classifies mouse EEG/EMG recordings into three
sleep/wake states:

```text
Wake
NREM
REM
```

The method is based on the reusable EEG/EMG staging logic in
`test/ref/from_JinXuan/CNO_Spectral_Temperature_Analysis.py`, but it removes the
CNO and temperature-specific parts. This tool is intended to produce a
reproducible hypnogram that can later be aligned to two-photon movie frames and
VideoSD outputs.

## Recommended Workflow

```text
EEG/EMG text file
  -> load EEG and EMG channels
  -> high-pass and notch filter
  -> classify Wake/NREM/REM at 1 s resolution
  -> inspect hypnogram QC plot
  -> align labels to raw or VideoSD frame times
```

## Basic Usage

### Command Line

Prompt-based run:

```bash
python -m natoolkit.tools.sleep_state_staging
```

Noninteractive run:

```bash
python -m natoolkit.tools.sleep_state_staging \
  --eegemg test/data/260605_PlxD1-CreER-G8s_SNI_D3_mice_1/EEGEMG_2026_06_05.txt \
  --eeg-col 1 \
  --emg-col 2 \
  --fs 1000 \
  --out outputs/sleep_state_260605 \
  --plot-format pdf \
  --wake-mode auto \
  --qc-max-time-bins 3000 \
  --qc-dpi 150
```

The command writes:

```text
sleep_state_epochs.csv
sleep_state_summary.json
sleep_state_hypnogram.pdf
```

### Python API

```python
from natoolkit.tools.sleep_state_staging import (
    classify_sleep_state,
    load_eegemg_txt,
    preprocess_eeg_emg,
)
from natoolkit.tools.sleep_state_staging.qc import plot_hypnogram

recording = load_eegemg_txt(
    "test/data/260605_PlxD1-CreER-G8s_SNI_D3_mice_1/EEGEMG_2026_06_05.txt",
    eeg_col=1,
    emg_col=2,
    fs=1000,
)

eeg, emg = preprocess_eeg_emg(recording.eeg, recording.emg, recording.fs)
result = classify_sleep_state(eeg, emg, recording.fs)

print(result.summary)
plot_hypnogram(eeg, emg, result, recording.fs, "hypnogram.pdf")
```

`result.labels` contains one label per 1 s step. `result.times_sec` contains the
center time of each 5 s staging window.

## Core Method

The classifier uses:

- 5 s rolling windows.
- 1 s step size.
- EEG downsampled to 100 Hz for spectral features.
- EMG RMS, EMG P90, and EMG coefficient of variation.
- EEG delta power, theta power, and 20-40 Hz high-frequency power.
- Adaptive Wake/Sleep scoring based on EMG dynamic range.
- Otsu thresholds for Wake/Sleep and REM/NREM separation.
- Theta/delta ratio for REM versus NREM.
- Post-processing rules for impossible Wake-to-REM transitions, short bouts,
  sleep onset, microarousals, and sustained EMG Wake overrides.

`--wake-mode auto` follows the reference script's adaptive rule:

```text
dynamic range >= 5x  -> emg_primary
dynamic range 2-5x   -> balanced
dynamic range < 2x   -> eeg_primary
```

The explicit modes `emg_primary`, `balanced`, and `eeg_primary` are also
available for diagnostic comparisons. The CNO-specific transfer mode from the
reference script is intentionally not included.

The default stages are stored as exact strings:

```text
Wake
NREM
REM
```

## Input Notes

`load_eegemg_txt()` treats `eeg_col` and `emg_col` as 1-based signal-channel
indices after excluding a leading sample-number column.

For the Zhou example EEG/EMG file:

```text
eeg_col=1
emg_col=2
fs=1000
```

The loader also supports the older four-signal-channel reference format:

```text
[mouse A EEG, mouse A EMG, mouse B EEG, mouse B EMG]
```

## Frame Alignment Helpers

Use `assign_labels_to_times()` to assign Wake/NREM/REM labels to arbitrary time
points, such as two-photon frame timestamps.

```python
from natoolkit.tools.sleep_state_staging import assign_labels_to_times

frame_labels = assign_labels_to_times(
    frame_times_sec,
    result.labels,
    result.times_sec,
    step_sec=result.params.step_sec,
)
```

For VideoSD denoised outputs that drop temporal context frames:

```python
from natoolkit.tools.sleep_state_staging import sd_frame_to_raw_frame

raw_frame_idx = sd_frame_to_raw_frame(sd_frame_idx, context_radius=10)
```

For context stack size 21, `context_radius=10`, so:

```text
SD frame i -> raw frame i + 10
```

## Quality Control

The QC hypnogram follows the reference report structure:

```text
EEG spectrogram
EMG trace
Wake/NREM/REM state bar
```

The dense plot elements are rasterized so full-night PDF/SVG outputs remain
small enough to open while preserving axes, labels, legend, and colorbar.

Before using labels for calcium activity analysis, check:

- The EEG spectrogram has plausible Wake/NREM/REM structure.
- EMG is high during Wake and low during sleep.
- REM bouts are preceded by NREM.
- Manual notes agree with automatic labels around recorded movie intervals.
- The time convention is consistent: labels are represented by window-center
  times in `result.times_sec`.

## Current Scope

This package is a library module, not a GUI. A review GUI can be added later,
but the analysis engine should remain scriptable and reproducible.
