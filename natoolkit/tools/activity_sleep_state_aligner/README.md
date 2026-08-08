# Activity–Sleep State Aligner

Activity–Sleep State Aligner is a focused Qt application that appends one
sleep-state label to every row of an activity CSV exported by Activity Tracer.
It combines three clocks: movie start times recorded in `Note.txt`, per-frame
timestamps stored in the raw TIFF files, and interval labels produced by Sleep
State Staging.

## Design Principles

The aligner does not resample, aggregate, or reorder activity data. It reads the
complete input CSV, computes one label for each row, preserves every existing
column and row, and adds a final `sleep_state` column. Validation is deliberately
strict so that a missing movie, malformed frame index, or ambiguous file match
fails visibly instead of silently assigning an incorrect state.

The core alignment function is independent of Qt and can be called from Python.
The GUI only collects paths, reports progress, and displays errors or the final
row count.

## Launch

Open the tool from the unified launcher:

```bash
natoolkit
```

or launch it directly:

```bash
activity-sleep-state-aligner
```

The module entry point is equivalent:

```bash
python -m natoolkit.tools.activity_sleep_state_aligner
```

## Required Inputs

The GUI requests five paths:

1. **Note.txt** — movie names and EEG-relative start times.
2. **Sleep-state CSV** — automatic or manually corrected state intervals.
3. **Activity CSV** — an export from Activity Tracer.
4. **TIFF directory** — raw TIFF movies named by the activity CSV.
5. **Output CSV** — a new destination file.

Selecting an activity CSV automatically suggests
`<activity-name>_sleep_state.csv` unless an output path is already present.

## File Conventions

### Note.txt

Recognized movie rows begin with a stage, a positive integer movie number, and a
`minutes:seconds` time:

```text
Wake 1 34:27
NREM 2 41:03.5
REM 3 52:10
```

Matching is case-insensitive and ignores text after the time. Each pair
`(stage, movie number)` must occur only once. Other lines are ignored.

### Activity CSV

The activity CSV must contain at least:

| Column | Meaning |
| --- | --- |
| `movie` | Movie name written by Activity Tracer. |
| `source_frame` | Zero-based frame index in the original raw movie. |

The normal Activity Tracer export also includes `roi`, `frame`,
`mean_intensity`, `normalization`, and `normalized`; all are preserved. The input
must not already contain a `sleep_state` column, and the output path must differ
from the input path.

The movie name is reduced to its filename stem and matched with the final stage
name and integer. Examples include `Wake 1`, `NREM_2`, and `REM-CNO-3`, provided
the stem ends in the integer movie number.

### Sleep-State CSV

Two schemas are accepted.

Automatic output from Sleep State Staging:

```text
time_sec,stage,...
```

Here, `time_sec` is an interval center. The aligner estimates the interval width
as the median difference between consecutive centers, or 1 second when only one
row exists.

Manually corrected output:

```text
start_sec,end_sec,final_stage,...
```

These explicit half-open intervals are used directly. Rows must be ordered by
start time, every stop must be greater than its start, and labels must be
non-empty.

### TIFF Directory

The directory is scanned non-recursively for `.tif` and `.tiff` files. Matching
uses the case-insensitive filename stem. Two TIFF files with the same normalized
stem are rejected as ambiguous.

Every referenced TIFF page must contain an `ImageDescription` value that
`tifffile.matlabstr2py` can decode and that provides:

```text
frameTimestamps_sec
```

The timestamp is interpreted as the time of that page relative to raw TIFF page
0.

## Alignment Logic

For every activity row, the program performs this calculation:

```text
movie key = (stage parsed from movie name, final movie number)
movie_start_sec = Note.txt[movie key]
frame_offset_sec = TIFF[source_frame].frameTimestamps_sec
eeg_time_sec = movie_start_sec + frame_offset_sec
sleep_state = label interval containing eeg_time_sec
```

Sleep intervals are treated as half-open ranges:

```text
start_sec <= eeg_time_sec < end_sec
```

Interval lookup uses binary search over ordered start times, so repeated rows
from multiple ROIs do not require scanning the label file from the beginning.
TIFF metadata is loaded only for source frames referenced by the activity CSV.

The central timing assumption is important: a movie time in `Note.txt` must be
the EEG-relative time of raw TIFF page 0. A TimeROI used in Activity Tracer does
not change this convention because the exported `source_frame` restores the raw
page index.

## Output

The output preserves the original header and appends:

| Column | Meaning |
| --- | --- |
| `sleep_state` | Label containing the computed EEG-relative frame time. |

When the computed time lies before the first sleep interval, after the final
interval, or in a gap between intervals, the value is:

```text
Unknown
```

Unknown rows are valid output and are counted in the completion message. Missing
files, malformed metadata, or naming mismatches are errors and stop the export.

## Python API

```python
from natoolkit.tools.activity_sleep_state_aligner import align_activity_file

result = align_activity_file(
    note_path="experiment/Note.txt",
    labels_path="experiment/sleep_state_staging/sleep_state_corrected_epochs.csv",
    activity_path="experiment/activities.csv",
    tiff_dir="experiment/raw_tiffs",
    output_path="experiment/activities_sleep_state.csv",
)

print(result.output_path)
print(result.row_count)
print(result.unknown_count)
```

## Validation and Common Errors

- **No Note.txt entry matches activity movie** — the activity movie stem does
  not resolve to a stage/number pair present in `Note.txt`.
- **Cannot identify sleep state and movie number** — the movie stem does not
  contain `Wake`, `NREM`, or `REM` followed eventually by a final integer.
- **No TIFF file matches activity movie** — the TIFF directory does not contain
  an equal filename stem.
- **source_frame exceeds pages** — Activity Tracer's source frame is not valid
  for the matched raw TIFF.
- **Missing frameTimestamps_sec** — the required per-page acquisition metadata
  is absent.
- **Sleep-state CSV intervals are not ordered** — regenerate or sort the label
  source without changing interval meaning.
- **Activity CSV already contains a sleep_state column** — select the original
  Activity Tracer export rather than a previously aligned output.

## Limitations

- The application trusts the clocks encoded in `Note.txt` and the TIFF metadata;
  it does not estimate or correct clock drift.
- TIFF matching is based on filename stems, so renaming either the activity
  movie or its raw TIFF can break the relationship.
- `Unknown` distinguishes times with no label coverage; it does not infer the
  most likely neighboring state.
- The GUI performs alignment synchronously. Very large files may temporarily
  make the window less responsive, although the unified launcher and other tools
  remain isolated in their own processes.
