# Activity–Sleep State Aligner

This Qt application appends sleep-state labels to an activity CSV exported by
Activity Tracer.

## Launch

```bash
activity-sleep-state-aligner
```

or:

```bash
python -m natoolkit.tools.activity_sleep_state_aligner
```

## Inputs and Output

Select these inputs in the GUI:

1. `Note.txt`, containing entries such as `Wake 1 34:27`.
2. `sleep_state_epochs.csv` or `sleep_state_corrected_epochs.csv`.
3. An activity CSV exported by Activity Tracer.
4. The directory containing the TIFF movies named by the activity CSV.
5. The output CSV path.

The output preserves every activity row and column and appends one column named
`sleep_state`.

The alignment assumes each time in `Note.txt` is the EEG-relative start time of
raw TIFF page 0. For each activity row, the tool adds the corresponding TIFF
page's `frameTimestamps_sec` value to that movie start time and looks up the
resulting time in the selected sleep-state CSV. Times outside the label CSV are
written as `Unknown`.
