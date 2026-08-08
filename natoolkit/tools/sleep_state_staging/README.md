# Sleep State Staging

Sleep State Staging classifies mouse EEG/EMG recordings into three exact state
labels:

```text
Wake
NREM
REM
```

It provides a reusable signal-processing API, a reproducible command-line
pipeline, and a Qt review application for experiment discovery, automatic
classification, local video-centered correction, and committed global review.
The resulting state intervals can later be joined to two-photon activity traces
with Activity–Sleep State Aligner.

## Design Principles

The automatic classifier is intentionally independent of the GUI. Loading,
preprocessing, feature extraction, classification, interval conversion, quality
control, and manual review are separate modules with explicit data structures.
This keeps the numerical method scriptable while allowing the GUI to manage
experiment-specific files and human corrections.

The review system distinguishes three label states:

- **Automatic labels** are the immutable result of the classifier.
- **Draft labels** include the current manual edits and are autosaved.
- **Committed labels** are the stable labels used by the global overview and
  downstream corrected CSV files.

This separation prevents incomplete local edits from silently changing the
dataset used for downstream analysis.

The classifier follows the reusable EEG/EMG logic of the v3.2 JinXuan reference
in `tests/ref/from_JinXuan/Sleep_Stage_Analysis_v3_2.py`. CNO-specific threshold
transfer, AccuSleePy import, and the reference script's interactive plotting are
not part of the classifier.

## Launch

Open the review application from the unified launcher:

```bash
natoolkit
```

or launch the GUI directly:

```bash
sleep-state-staging
```

The explicit GUI module is equivalent:

```bash
python -m natoolkit.tools.sleep_state_staging.gui
```

`python -m natoolkit.tools.sleep_state_staging` is intentionally different: it
runs the command-line classifier rather than the GUI.

## Recommended GUI Workflow

```text
Choose an experiment directory
  -> verify Note.txt, EEG/EMG files, sessions, and TIFF movies
  -> adjust source mapping if automatic discovery is incomplete
  -> review automatic-analysis parameters
  -> run preprocessing and classification in a worker thread
  -> inspect each movie in local review
  -> edit labels and confirm reviewed movies
  -> commit each Session
  -> inspect committed labels in the global overview
```

## Experiment Discovery

The GUI scans only the selected experiment directory; it does not recursively
consume derived output directories. By default it expects:

```text
experiment/
  Note.txt
  one or more EEG/EMG .txt files
  matching raw .tif or .tiff movies
```

Use **Manual source mapping** to select a different `Note.txt`, TIFF directory,
or explicit list of EEG/EMG files.

### Note.txt Syntax

Movie entries have this form:

```text
Wake 1 34:27 optional comment
NREM 2 41:03.5 optional comment
REM-CNO 3 52:10 optional comment
```

An optional metadata line such as `Frame rate 30 Hz` supplies a fallback rate
for TIFF files without usable frame timestamps.

Multiple EEG recordings are divided by lines beginning with `EEG` and a session
number:

```text
EEG1 00:00 Pre-CNO
Wake 1 01:20
NREM 2 03:40
EEG2 00:00 Post-CNO
REM-CNO 1 02:15
```

Movie entries following a session marker belong to that session. Without a
marker, they belong to Session 1. Session marker times and trailing text are
stored as experiment events.

### EEG/EMG Assignment

All top-level `.txt` files other than `Note.txt` are probed as possible
recordings. Filenames ending in patterns such as `EEG1`, `EEG_1`, or `No1` are
first matched to the corresponding Note session; remaining files are assigned
in natural filename order. The source page reports missing, surplus, or
unreadable recordings.

### TIFF Assignment and Timing

TIFF stems must start with `Wake`, `NREM`, or `REM`, contain any optional middle
text, and end in an underscore plus the movie number. `CNO` in the middle text
distinguishes CNO movies.

The scanner reads the first and last TIFF-page `frameTimestamps_sec` values to
estimate duration and frame rate. When those timestamps are unavailable, it uses
the Note frame rate. A difference greater than 2% between the Note rate and the
median measured TIFF rate produces a warning; TIFF timestamps take precedence.

The experiment is ready only when every Note session has an EEG/EMG recording
and every movie entry has exactly one TIFF match.

## EEG/EMG Input Loading

The loader tries `gb18030`, UTF-8, and Latin-1 encodings. It scans for the first
row with at least two numeric signal columns and supports whitespace- or
tab-delimited data. A line containing a value followed by `Hz` supplies the
sampling rate; otherwise the GUI defaults to 1000 Hz.

`eeg_col` and `emg_col` are 1-based signal-channel indices. If a row contains
three or more numeric columns, the first numeric column is treated as a sample
number and excluded from signal-channel indexing. Thus the common layout is:

```text
sample number | EEG channel 1 | EMG channel 2
```

The loader also supports older files with several signal channels, for example:

```text
sample number | mouse A EEG | mouse A EMG | mouse B EEG | mouse B EMG
```

The GUI currently loads signal channel 1 as EEG and channel 2 as EMG. The Python
and command-line APIs allow other channel selections.

## Preprocessing

Preprocessing is enabled by default and performs these steps independently on
EEG and EMG:

1. apply a second-order Butterworth high-pass filter with zero-phase
   `filtfilt`;
2. apply an IIR notch filter with quality factor 30, also with zero-phase
   filtering;
3. truncate EEG and EMG to their common sample length.

Default settings are:

```text
EEG high-pass: 0.5 Hz
EMG high-pass: 1.0 Hz
line notch:    50 Hz
```

A cutoff outside `(0, Nyquist)` is skipped rather than applied. The GUI's
**Automatic parameters** dialog can disable preprocessing or change these
values.

## Automatic Classification Logic

The default classifier uses a symmetric 5-second feature window to produce one
label every 1 second. Labels are represented by interval centers. Edge labels
are omitted whenever a complete symmetric feature window is unavailable.

### 1. Signal Preparation

- EEG is resampled to 100 Hz when its original rate is higher.
- EMG amplitudes are clipped symmetrically at the 99th percentile of absolute
  amplitude before feature extraction.
- EMG is additionally band-pass filtered across 10–300 Hz for spectral
  features, subject to Nyquist.

### 2. Per-Window Features

EEG power spectral density is estimated with Welch's method. The main bands are:

| Feature | Frequency band |
| --- | --- |
| Delta | 0.5–4 Hz |
| Theta | 6–10 Hz |
| Sigma | 10–15 Hz |
| Beta | 15–30 Hz |
| Gamma | 30–50 Hz |
| High-frequency Wake evidence | 20–40 Hz |
| Total EEG power | 0.5–50 Hz |

The result also stores EEG delta and theta ratios and spectral entropy.

EMG time-domain features include RMS, mean absolute amplitude, standard
deviation, P90, P95, coefficient of variation, power, and an above-P90 sample
fraction. Spectral features cover 10–50 Hz, 50–150 Hz, and 150–300 Hz power,
band ratios, entropy, and peak frequency.

### 3. EMG Activity and Wake Score

The primary EMG activity term is the geometric mean of EMG RMS, P90, and
coefficient of variation. It is blended with normalized log 50–150 Hz power;
the default mid-band weight is 0.35. Feature scores are normalized between their
2nd and 98th percentiles.

EMG dynamic range is:

```text
P95(EMG RMS) / P10(EMG RMS)
```

In `auto` mode it selects the Wake score:

```text
dynamic range >= 5  -> emg_primary
dynamic range >= 2  -> balanced
dynamic range < 2   -> eeg_primary
```

The corresponding weights are:

```text
emg_primary = 0.75 * EMG + 0.25 * EEG high frequency
balanced    = 0.45 * EMG + 0.35 * EEG high frequency
              + 0.20 * (1 - delta)
eeg_primary = 0.20 * EMG + 0.50 * EEG high frequency
              + 0.30 * (1 - delta)
```

An Otsu threshold separates initial Wake from Sleep. If this marks more than 70%
of the recording as Wake, the classifier retries with stronger EEG weighting.
If the fraction is still too high, the 70th percentile of Wake score is used as
a fallback threshold.

### 4. REM Versus NREM

For non-Wake windows, the classifier computes:

```text
theta/delta = theta power / delta power
```

An Otsu threshold on candidate Sleep windows separates higher-ratio REM from
lower-ratio NREM.

### 5. Temporal Post-Processing

The initial per-window labels are processed in this order:

1. sustained high-EMG Sleep intervals are overridden to Wake;
2. bouts shorter than the configured minimum, 10 seconds by default, are
   merged into the preceding state when bounded on both sides;
3. direct Wake-to-REM transitions are resolved as Wake or NREM from the current
   evidence;
4. REM requires a recent NREM anchor within the configured 50-second lookback,
   allowing at most a 10-second Wake gap;
5. short bouts and invalid transitions are checked again;
6. a new NREM sleep onset must last at least 20 seconds;
7. Wake microarousals of at most 10 seconds between NREM bouts are absorbed;
8. every REM bout must immediately follow NREM.

These rules encode the implemented workflow assumptions. They are not a learned
probabilistic state-transition model.

## Review GUI

Automatic analysis runs in a `QThread` worker. For each Note session the GUI
loads the recording, optionally preprocesses it, classifies it at a fixed
1-second label interval, and writes automatic results under:

```text
experiment/sleep_state_staging/
```

For multiple sessions the output is divided into `session_<number>`
subdirectories.

### Local Video Review

Local review centers the synchronized display on one Note movie and adds 60
seconds of context on either side by default. It shows EEG, EMG, automatic
labels, editable draft labels, movie extent, Note annotations, selection, and a
cursor. All tracks share one time axis.

Click one label or drag across a time range, then use:

```text
1  assign Wake
2  assign NREM
3  assign REM
A  restore automatic labels
Ctrl/Cmd+Z  undo
Ctrl/Cmd+Shift+Z or platform equivalent  redo
```

The time-width and position sliders change the visible range. **Fit window**
restores the complete local context. EEG and EMG gain controls change display
amplitude only and do not modify stored signal values.

Each edit is immediately written to `review_draft.json`. Editing a previously
confirmed movie marks it as needing review when the edited interval overlaps the
movie. **Confirm and next** updates review status but does not commit labels.

### Commit Semantics

**Commit Session** promotes the complete draft to the committed state. If some
movies are not confirmed, the GUI requests confirmation before a partial review
is committed. Commit writes corrected CSV and JSON files and atomically replaces
the corrected hypnogram.

Draft and commit files include a signature derived from the recording path,
label grid, feature-window settings, and a SHA-256 hash of automatic labels.
Incompatible files are ignored instead of being applied to a changed automatic
analysis.

### Global Overview

The global tab is read-only and displays committed labels across the full
recording with all movie intervals. Uncommitted draft edits remain hidden. A
selected movie can be opened back in local review.

## Output Files

Automatic GUI or CLI analysis writes:

```text
sleep_state_epochs.csv
sleep_state_summary.json
sleep_state_hypnogram.<format>
```

`sleep_state_epochs.csv` contains `step_idx`, center `time_sec`, `stage`, and all
per-step stored features. The summary records input metadata, preprocessing,
thresholds, parameters, state counts, fractions, dynamic range, and selected
Wake mode.

The GUI also maintains:

```text
review_draft.json
review_commit.json
sleep_state_corrected_epochs.csv
sleep_state_corrected_summary.json
sleep_state_hypnogram.svg
```

Corrected CSV columns are:

| Column | Meaning |
| --- | --- |
| `step_idx` | Zero-based label index. |
| `start_sec`, `end_sec` | Explicit half-open interval boundaries. |
| `auto_stage` | Original automatic state. |
| `corrected_stage` | Manual state when it differs, otherwise empty. |
| `final_stage` | State used for downstream analysis. |
| `corrected` | `1` when final differs from automatic, otherwise `0`. |

The hypnogram contains an EEG spectrogram, clipped EMG trace, and state bar.
Dense layers are rasterized while labels and axes remain vector-friendly.

## Command-Line Usage

Prompt-based operation:

```bash
python -m natoolkit.tools.sleep_state_staging
```

Noninteractive example:

```bash
python -m natoolkit.tools.sleep_state_staging \
  --eegemg experiment/EEGEMG.txt \
  --eeg-col 1 \
  --emg-col 2 \
  --fs 1000 \
  --out experiment/sleep_state_cli \
  --plot-format pdf \
  --wake-mode auto \
  --epoch-sec 5 \
  --step-sec 1 \
  --qc-max-time-bins 3000 \
  --qc-dpi 150
```

Use `--no-preprocess` to bypass filtering. The CLI permits a configurable label
step, while the GUI intentionally fixes it at 1 second.

## Python API

```python
from natoolkit.tools.sleep_state_staging import (
    classify_sleep_state,
    load_eegemg_txt,
    preprocess_eeg_emg,
)
from natoolkit.tools.sleep_state_staging.qc import plot_hypnogram

recording = load_eegemg_txt(
    "experiment/EEGEMG.txt",
    eeg_col=1,
    emg_col=2,
    fs=1000,
)
eeg, emg = preprocess_eeg_emg(recording.eeg, recording.emg, recording.fs)
result = classify_sleep_state(eeg, emg, recording.fs)

plot_hypnogram(eeg, emg, result, recording.fs, "hypnogram.pdf")
print(result.summary)
```

## Alignment Helpers

Assign labels to arbitrary EEG-relative times:

```python
from natoolkit.tools.sleep_state_staging import assign_labels_to_times

frame_labels = assign_labels_to_times(
    frame_times_sec,
    result.labels,
    result.times_sec,
    step_sec=result.params.step_sec,
)
```

Map a VideoSD output index back to the raw frame when temporal context frames
were dropped:

```python
from natoolkit.tools.sleep_state_staging import sd_frame_to_raw_frame

raw_frame = sd_frame_to_raw_frame(sd_frame, context_radius=10)
```

For a context stack of 21 frames, this gives `raw frame = SD frame + 10`.

## Quality-Control Checklist

- Confirm the detected EEG/EMG sampling rate and channel assignment.
- Resolve every missing or multiply matched Note movie before analysis.
- Check that TIFF-derived frame rates agree with the acquisition notes.
- Confirm that Wake has relatively high EMG and plausible EEG high-frequency
  activity.
- Confirm that NREM has plausible delta activity and REM follows NREM.
- Inspect state boundaries near recorded movie intervals.
- Commit each reviewed Session before using corrected labels downstream.
- Preserve the time convention: automatic `time_sec` values are interval
  centers; corrected outputs provide explicit start and end boundaries.

## Limitations

- Adaptive thresholds are recording-specific; labels from different recordings
  are not produced from one shared calibrated threshold.
- Temporal rules encode expected mouse sleep transitions and may suppress rare
  or artifact-driven patterns.
- Automatic staging is a review aid, not a substitute for EEG/EMG quality
  control and expert correction.
- The GUI maps channels 1 and 2 by convention. Use the CLI or Python API when a
  recording uses a different channel order.
- Movie synchronization assumes Note times and TIFF timestamps refer to the same
  EEG-relative clock and does not correct clock drift.
