# Neural Activity Toolkit

Lab-internal tools for neural activity data analysis.

## Current tools

- Activity Tracer: import movie data in napari, draw ROI labels, and export activity traces.
- Sleep State Staging: classify and manually review EEG/EMG sleep states.
- Activity–Sleep State Aligner: append sleep-state labels to exported activity traces.

## Install

Install from GitHub:

```bash
pip install "git+https://github.com/RyuuYou0529/natoolkit.git"
```

Install from a local checkout:

```bash
pip install -e .
```

After installation, napari can discover the plugin from the `natoolkit` manifest.
