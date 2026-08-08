# Neural Activity Toolkit

Neural Activity Toolkit is a collection of lab-internal Python applications for
two-photon activity extraction, EEG/EMG sleep-state staging, and alignment of
activity traces with sleep labels. The applications share one installation and
one Qt launcher but run as independent processes so that napari and standalone
Qt event loops remain isolated.

## Applications

| Application | Purpose | Documentation |
| --- | --- | --- |
| Activity Tracer | Import imaging movies in napari, manage ROI labels, extract and normalize traces, and detect FRAME-style events. | [Activity Tracer README](natoolkit/tools/activity_tracer/README.md) |
| Sleep State Staging | Discover EEG/EMG experiments, classify Wake/NREM/REM, review local movie intervals, and commit corrected labels. | [Sleep State Staging README](natoolkit/tools/sleep_state_staging/README.md) |
| Activity–Sleep State Aligner | Append sleep-state labels to Activity Tracer CSV rows using Note and TIFF timing metadata. | [Aligner README](natoolkit/tools/activity_sleep_state_aligner/README.md) |
| AI Guider | Answer project-scoped questions from approved documentation and selected source code. | [AI Guider README](natoolkit/ai_guider/README.md) |

## End-to-End Workflow

```text
Imaging movies
  -> Activity Tracer
  -> activity CSV with movie and source_frame

EEG/EMG recording + Note.txt + raw TIFF metadata
  -> Sleep State Staging
  -> automatic or manually corrected sleep-state CSV

activity CSV + sleep-state CSV + Note.txt + raw TIFF directory
  -> Activity–Sleep State Aligner
  -> activity CSV with an appended sleep_state column
```

The tools exchange explicit files rather than hidden in-memory state. This makes
each processing stage independently reviewable and reproducible.

## Installation

Install from GitHub:

```bash
pip install "git+https://github.com/RyuuYou0529/natoolkit.git"
```

Install an editable local checkout:

```bash
pip install -e .
```

Activity Tracer works without Suite2p. Install the optional motion-correction
and automatic ROI-detection integration with:

```bash
pip install -e ".[suite2p]"
```

Python 3.10 or newer is required. The default dependencies install napari with
PyQt6, QtPy, NumPy, SciPy, Matplotlib, PyQtGraph, Dask, tifffile, and the OpenAI
compatible client used by AI Guider. SOCKS transport support is included for
environments that route API traffic through a SOCKS proxy.

## Unified Launcher

After installation, run:

```bash
natoolkit
```

The launcher presents one card for each application. Clicking **Open** starts
the selected tool with the same Python interpreter in a detached child process.
Multiple tools can therefore remain open at the same time, and closing the
launcher does not close them.

The launcher can also be invoked as a module:

```bash
python -m natoolkit.launcher
```

## Direct Commands

Every application remains independently launchable:

```bash
activity-tracer
sleep-state-staging
activity-sleep-state-aligner
ai-guider
```

`sleep-state-staging` opens the Qt review GUI. To run the command-line
classifier, use:

```bash
python -m natoolkit.tools.sleep_state_staging --help
```

AI Guider requires a DeepSeek API key in its process environment:

```bash
export DEEPSEEK_API_KEY="your-token"
ai-guider
```

The key is not stored by the application. See the AI Guider README for model,
scope, source access, and privacy details.

napari can also discover Activity Tracer from the packaged `natoolkit` plugin
manifest.

## Repository Layout

```text
natoolkit/
  launcher/                         unified Qt launcher
  ai_guider/                        project-scoped conversational manual
  tools/
    activity_tracer/                napari imaging and ROI workflow
    sleep_state_staging/            EEG/EMG classifier and review GUI
    activity_sleep_state_aligner/   CSV and TIFF time alignment
  napari.yaml                       napari plugin contributions
tests/                              automated tests and reference material
```

The detailed tool READMEs describe design choices, algorithms, input
conventions, output schemas, and limitations. They are included in packaged
distributions as package data.

## Testing

Run the automated test suite from an environment containing the project
dependencies:

```bash
python -m unittest discover -s tests
```

GUI tests may require a display server or an offscreen Qt platform in headless
environments.

## License

Neural Activity Toolkit is licensed under the GNU General Public License v3.0.
