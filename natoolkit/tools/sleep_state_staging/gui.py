from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtSvg, QtWidgets
from scipy.signal import spectrogram

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"natoolkit_matplotlib_{os.getpid()}"),
)

from .io import load_eegemg_txt
from .preprocess import preprocess_eeg_emg
from .qc import plot_hypnogram
from .staging import NREM, REM, WAKE, WAKE_MODES, StagingParams, classify_sleep_state


STAGES = (WAKE, NREM, REM)
STAGE_IDS = {WAKE: 0, NREM: 1, REM: 2}
STAGE_COLORS = {
    WAKE: (56, 99, 255, 255),
    NREM: (229, 94, 148, 255),
    REM: (255, 162, 51, 255),
}
STAGE_LUT = np.asarray([STAGE_COLORS[stage] for stage in STAGES], dtype=np.ubyte)


class TimelineViewBox(pg.ViewBox):
    zoom_requested = QtCore.Signal(float, float)
    pan_requested = QtCore.Signal(float)
    cursor_requested = QtCore.Signal(float)
    epoch_range_requested = QtCore.Signal(float, float, bool)

    def __init__(self, epoch_selectable: bool = False) -> None:
        super().__init__(enableMenu=False)
        self.epoch_selectable = epoch_selectable
        self.setMouseEnabled(x=False, y=False)

    def wheelEvent(self, event, axis=None) -> None:
        center = self.mapSceneToView(event.scenePos()).x()
        factor = 0.80 if event.delta() > 0 else 1.25
        self.zoom_requested.emit(float(center), factor)
        event.accept()

    def mouseClickEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        time_sec = float(self.mapSceneToView(event.scenePos()).x())
        if self.epoch_selectable:
            self.epoch_range_requested.emit(time_sec, time_sec, True)
        else:
            self.cursor_requested.emit(time_sec)
        event.accept()

    def mouseDragEvent(self, event, axis=None) -> None:
        button = event.button()
        if self.epoch_selectable and button == QtCore.Qt.MouseButton.LeftButton:
            start = float(self.mapSceneToView(event.buttonDownScenePos(button)).x())
            stop = float(self.mapSceneToView(event.scenePos()).x())
            self.epoch_range_requested.emit(start, stop, bool(event.isFinish()))
            event.accept()
            return
        if button in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            old = float(self.mapSceneToView(event.lastScenePos()).x())
            new = float(self.mapSceneToView(event.scenePos()).x())
            self.pan_requested.emit(old - new)
            event.accept()
            return
        event.ignore()


class StageLegend(pg.GraphicsWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumWidth(86)
        self.setMaximumWidth(86)
        self.entries = ((WAKE, WAKE), (NREM, NREM), (REM, REM))

    def paint(self, painter, option, widget=None) -> None:
        rect = self.boundingRect()
        x = rect.left() + 10
        y = rect.top() + max(10, (rect.height() - 72) / 2)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(pg.mkPen(40, 40, 40))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        for row, (label, stage) in enumerate(self.entries):
            top = y + row * 24
            painter.setBrush(pg.mkBrush(*STAGE_COLORS[stage]))
            painter.drawRect(QtCore.QRectF(x, top + 3, 16, 12))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawText(QtCore.QPointF(x + 22, top + 14), label)


class SleepStateReviewWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sleep State Epoch Review")
        self.resize(1320, 860)

        self.recording = None
        self.eeg: np.ndarray | None = None
        self.emg: np.ndarray | None = None
        self.result = None
        self.auto_labels: np.ndarray | None = None
        self.corrected_labels: np.ndarray | None = None
        self.corrected_mask: np.ndarray | None = None

        self.duration_sec = 0.0
        self.view_start = 0.0
        self.view_stop = 1.0
        self.cursor_time = 0.0
        self.selection: tuple[int, int] | None = None
        self.undo_stack: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self.grid_lines = []
        self.spec_cmap = pg.colormap.get("turbo")

        pg.setConfigOptions(
            antialias=False,
            imageAxisOrder="row-major",
            background="w",
            foreground="k",
        )
        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.addWidget(self._controls())
        layout.addWidget(self._timeline(), stretch=1)

        footer = QtWidgets.QHBoxLayout()
        self.cursor_info = QtWidgets.QLabel("Cursor: -")
        self.selection_info = QtWidgets.QLabel("Selection: -")
        self.status = QtWidgets.QLabel("Load an EEG/EMG file and run auto staging.")
        footer.addWidget(self.cursor_info)
        footer.addWidget(self.selection_info)
        footer.addStretch(1)
        footer.addWidget(self.status)
        layout.addLayout(footer)
        self.setCentralWidget(root)

    def _controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Input, staging, and correction")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for col in (1, 3, 5):
            grid.setColumnStretch(col, 1)

        self.input_path = QtWidgets.QLineEdit()
        self.output_dir = QtWidgets.QLineEdit()
        self.eeg_col = _spin(1, 999, 1)
        self.emg_col = _spin(1, 999, 2)
        self.fs = _double_spin(1.0, 100000.0, 1000.0, 1)
        self.epoch_sec = _double_spin(0.1, 120.0, 5.0, 2)
        self.step_sec = _double_spin(0.1, 60.0, 1.0, 2)
        self.eeg_hp = _double_spin(0.0, 500.0, 0.5, 2)
        self.emg_hp = _double_spin(0.0, 500.0, 1.0, 2)
        self.line_freq = _double_spin(0.0, 500.0, 50.0, 2)
        self.figure_format = QtWidgets.QComboBox()
        self.figure_format.addItems(("svg", "png", "pdf"))
        self.preprocess = QtWidgets.QCheckBox("Preprocess")
        self.preprocess.setChecked(True)
        self.wake_mode = QtWidgets.QComboBox()
        self.wake_mode.addItems(WAKE_MODES)

        self.run_button = QtWidgets.QPushButton("Run Auto Staging")
        self.save_button = QtWidgets.QPushButton("Save Corrected Results")
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.wake_button = QtWidgets.QPushButton("Wake (1)")
        self.nrem_button = QtWidgets.QPushButton("NREM (2)")
        self.rem_button = QtWidgets.QPushButton("REM (3)")
        self.fit_button = QtWidgets.QPushButton("Fit")

        self.run_button.clicked.connect(self.run_staging)
        self.save_button.clicked.connect(self.save_corrected)
        self.undo_button.clicked.connect(self.undo)
        self.wake_button.clicked.connect(lambda: self.apply_stage(WAKE))
        self.nrem_button.clicked.connect(lambda: self.apply_stage(NREM))
        self.rem_button.clicked.connect(lambda: self.apply_stage(REM))
        self.fit_button.clicked.connect(self.fit_recording)

        self._add_path_row(grid, 0, "EEG/EMG file", self.input_path, self._browse_input)
        self._add_path_row(grid, 1, "Output directory", self.output_dir, self._browse_output)
        _add_row(grid, 2, "EEG col", self.eeg_col)
        _add_row(grid, 2, "EMG col", self.emg_col, col=2)
        _add_row(grid, 2, "Sampling Hz", self.fs, col=4)
        _add_row(grid, 3, "Feature window sec", self.epoch_sec)
        _add_row(grid, 3, "Epoch step sec", self.step_sec, col=2)
        _add_row(grid, 3, "Wake mode", self.wake_mode, col=4)
        _add_row(grid, 4, "EEG HP", self.eeg_hp)
        _add_row(grid, 4, "EMG HP", self.emg_hp, col=2)
        _add_row(grid, 4, "Line Hz", self.line_freq, col=4)
        _add_row(grid, 5, "Figure format", self.figure_format)

        buttons = QtWidgets.QHBoxLayout()
        for widget in (
            self.preprocess,
            self.run_button,
            self.wake_button,
            self.nrem_button,
            self.rem_button,
            self.undo_button,
            self.fit_button,
            self.save_button,
        ):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        grid.addLayout(buttons, 6, 0, 1, 6)
        return box

    def _timeline(self) -> pg.GraphicsLayoutWidget:
        self.timeline = pg.GraphicsLayoutWidget()
        self.overview_plot = self._add_plot(row=0, height=55)
        self.spec_plot = self._add_plot(row=1, height=330)
        self.emg_plot = self._add_plot(row=2, height=150)
        self.auto_label_plot = self._add_plot(row=3, height=46)
        self.final_label_plot = self._add_plot(row=4, height=54, epoch_selectable=True)

        self.overview_image = pg.ImageItem()
        self.spec_image = pg.ImageItem()
        self.auto_label_image = pg.ImageItem()
        self.final_label_image = pg.ImageItem()
        self.emg_curve = pg.PlotDataItem(pen=pg.mkPen("k", width=0.6))
        self.spec_image.setColorMap(self.spec_cmap)

        self.overview_plot.addItem(self.overview_image)
        self.spec_plot.addItem(self.spec_image)
        self.emg_plot.addItem(self.emg_curve)
        self.auto_label_plot.addItem(self.auto_label_image)
        self.final_label_plot.addItem(self.final_label_image)

        self.spec_colorbar = pg.ColorBarItem(
            values=(0, 1),
            width=28,
            colorMap=self.spec_cmap,
            label="EEG dB",
            interactive=False,
            pen=pg.mkPen(30, 30, 30),
            colorMapMenu=False,
        )
        self.timeline.addItem(self.spec_colorbar, row=1, col=2)
        self.spec_colorbar.setImageItem(self.spec_image)
        self.timeline.ci.layout.setColumnFixedWidth(0, 70)
        self.timeline.ci.layout.setColumnFixedWidth(2, 86)
        for row, text in enumerate(("Overview", "EEG", "EMG", "Auto", "Final")):
            self.timeline.addItem(_row_label(text), row=row, col=0)
        self.timeline.addItem(_stage_legend(), row=2, col=2, rowspan=3, colspan=1)

        self.focus_left_region = pg.LinearRegionItem(
            values=(0, 0),
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 165),
            pen=pg.mkPen(255, 255, 255, 0),
        )
        self.focus_right_region = pg.LinearRegionItem(
            values=(0, 0),
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 165),
            pen=pg.mkPen(255, 255, 255, 0),
        )
        self.focus_start_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(20, 20, 20, 255, width=3),
        )
        self.focus_stop_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(20, 20, 20, 255, width=3),
        )
        self.selection_region = pg.LinearRegionItem(
            values=(0, 0),
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 70),
            pen=pg.mkPen(255, 255, 255, 150),
        )
        for item in (
            self.focus_left_region,
            self.focus_right_region,
            self.focus_start_line,
            self.focus_stop_line,
        ):
            self.overview_plot.addItem(item)
            item.setZValue(10)
        self.final_label_plot.addItem(self.selection_region)
        self.selection_region.setZValue(10)
        self.selection_region.setVisible(False)

        self.cursor_lines = []
        cursor_pens = (
            pg.mkPen(30, 30, 30, 180),
            pg.mkPen(255, 255, 255, 220),
            pg.mkPen(30, 30, 30, 180),
            pg.mkPen(30, 30, 30, 180),
            pg.mkPen(30, 30, 30, 180),
        )
        for plot, pen in zip(
            (
                self.overview_plot,
                self.spec_plot,
                self.emg_plot,
                self.auto_label_plot,
                self.final_label_plot,
            ),
            cursor_pens,
        ):
            line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            plot.addItem(line)
            self.cursor_lines.append(line)
        return self.timeline

    def _add_plot(self, row: int, height: int, epoch_selectable: bool = False):
        view_box = TimelineViewBox(epoch_selectable=epoch_selectable)
        view_box.zoom_requested.connect(self.zoom_at)
        view_box.pan_requested.connect(self.pan_by)
        view_box.cursor_requested.connect(self.set_cursor)
        view_box.epoch_range_requested.connect(self.select_epochs_from_times)

        plot = self.timeline.addPlot(row=row, col=1, viewBox=view_box)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.setMinimumHeight(height)
        plot.showButtons()
        plot.hideButtons()
        return plot

    def _add_path_row(self, grid, row: int, label: str, edit: QtWidgets.QLineEdit, slot) -> None:
        grid.addWidget(_qt_label(label), row, 0)
        grid.addWidget(edit, row, 1, 1, 4)
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(slot)
        grid.addWidget(button, row, 5)

    def _bind_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self.undo)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if _editing_text():
            super().keyPressEvent(event)
            return
        key = event.key()
        modifiers = event.modifiers()
        if key in (QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3):
            stages = {
                QtCore.Qt.Key.Key_1: WAKE,
                QtCore.Qt.Key.Key_2: NREM,
                QtCore.Qt.Key.Key_3: REM,
            }
            self.apply_stage(stages[key])
        elif key == QtCore.Qt.Key.Key_A:
            self.accept_auto_label()
        elif key == QtCore.Qt.Key.Key_F:
            self.fit_recording()
        elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            step = -1 if key == QtCore.Qt.Key.Key_Left else 1
            jump = 10 if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier else 1
            self.step_cursor(
                step * jump,
                extend=bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier),
            )
        else:
            super().keyPressEvent(event)

    def _browse_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select EEG/EMG file")
        if path:
            self.input_path.setText(path)
            if not self.output_dir.text():
                self.output_dir.setText(str(Path(path).with_name("sleep_state_staging")))

    def _browse_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.output_dir.setText(path)

    @QtCore.Slot()
    def run_staging(self) -> None:
        try:
            input_path = Path(self.input_path.text()).expanduser()
            if not input_path.exists():
                raise ValueError("Input file does not exist.")
            output_dir = self._resolved_output_dir(input_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.recording = load_eegemg_txt(
                input_path,
                eeg_col=self.eeg_col.value(),
                emg_col=self.emg_col.value(),
                fs=self.fs.value(),
            )
            if self.preprocess.isChecked():
                self.eeg, self.emg = preprocess_eeg_emg(
                    self.recording.eeg,
                    self.recording.emg,
                    self.recording.fs,
                    eeg_hp_cutoff=self.eeg_hp.value(),
                    emg_hp_cutoff=self.emg_hp.value(),
                    line_freq=self.line_freq.value(),
                )
            else:
                self.eeg, self.emg = self.recording.eeg, self.recording.emg

            params = StagingParams(
                wake_mode=self.wake_mode.currentText(),
                epoch_sec=self.epoch_sec.value(),
                step_sec=self.step_sec.value(),
            )
            self.result = classify_sleep_state(self.eeg, self.emg, self.recording.fs, params=params)
            self.auto_labels = self.result.labels.astype(object).copy()
            self.corrected_labels = self.auto_labels.copy()
            self.corrected_mask = np.zeros(len(self.auto_labels), dtype=bool)
            self.undo_stack.clear()
            self.cursor_time = 0.0
            self.selection = (0, 1) if len(self.auto_labels) else None
            self.duration_sec = min(len(self.eeg), len(self.emg)) / self.recording.fs

            _write_rows(output_dir / "sleep_state_epochs.csv", self.result.to_records())
            _write_json(output_dir / "sleep_state_summary.json", self._summary_payload())
            self._write_qc_figure(output_dir / f"sleep_state_hypnogram.{self.plot_format()}")
            self._plot_results()
            self.fit_recording()
            self._set_status(f"Auto staging complete. {len(self.auto_labels)} one-second epochs loaded.")
        except Exception as exc:
            self._show_error(str(exc))

    def _plot_results(self) -> None:
        self._plot_overview()
        self._plot_spectrogram()
        self._plot_emg()
        self._refresh_labels()
        self._update_cursor()
        self._update_selection()

    def _plot_overview(self) -> None:
        ids = _stage_ids(self.corrected_labels)[np.newaxis, :]
        self.overview_image.setLookupTable(STAGE_LUT)
        self.overview_image.setLevels((0, 2))
        self.overview_image.setImage(ids, autoLevels=False)
        self.overview_image.setRect(QtCore.QRectF(0, 0, self._label_duration(), 1))
        self.overview_plot.setYRange(0, 1, padding=0)

    def _plot_spectrogram(self) -> None:
        freqs, times, power_db, vmin, vmax = _spectrogram(self.eeg, self.recording.fs)
        self.spec_image.setImage(np.clip(power_db, vmin, vmax), autoLevels=False)
        self.spec_image.setLevels((vmin, vmax))
        self.spec_colorbar.setLevels((vmin, vmax))
        rect = _image_rect(times, freqs, self.duration_sec)
        self.spec_image.setRect(rect)
        self.spec_plot.setYRange(0, 30, padding=0)

    def _plot_emg(self) -> None:
        step = max(1, len(self.emg) // 80000)
        clip = max(float(np.percentile(np.abs(self.emg), 99.5)), 1e-9)
        time = np.arange(len(self.emg), dtype=float) / self.recording.fs
        self.emg_curve.setData(time[::step], np.clip(self.emg, -clip, clip)[::step])
        self.emg_plot.setYRange(-clip * 1.2, clip * 1.2, padding=0)

    def _refresh_labels(self) -> None:
        if self.corrected_labels is None:
            return
        auto_ids = _stage_ids(self.auto_labels)[np.newaxis, :]
        final_ids = _stage_ids(self.corrected_labels)[np.newaxis, :]
        for item in (self.auto_label_image, self.final_label_image, self.overview_image):
            item.setLookupTable(STAGE_LUT)
            item.setLevels((0, 2))
        self.auto_label_image.setImage(auto_ids, autoLevels=False)
        self.final_label_image.setImage(final_ids, autoLevels=False)
        self.auto_label_image.setRect(QtCore.QRectF(0, 0, self._label_duration(), 1))
        self.final_label_image.setRect(QtCore.QRectF(0, 0, self._label_duration(), 1))
        self.auto_label_plot.setYRange(0, 1, padding=0)
        self.final_label_plot.setYRange(0, 1, padding=0)
        if self.auto_labels is not None:
            self.overview_image.setImage(final_ids, autoLevels=False)

    def fit_recording(self) -> None:
        if self.duration_sec <= 0:
            return
        self.set_view(0.0, self.duration_sec)

    def zoom_at(self, center: float, factor: float) -> None:
        if self.duration_sec <= 0:
            return
        width = max(self.step(), (self.view_stop - self.view_start) * factor)
        width = min(width, self.duration_sec)
        center = float(np.clip(center, 0.0, self.duration_sec))
        self.set_view(center - width / 2, center + width / 2)

    def pan_by(self, delta_sec: float) -> None:
        if self.duration_sec <= 0:
            return
        self.set_view(self.view_start + delta_sec, self.view_stop + delta_sec)

    def set_view(self, start: float, stop: float) -> None:
        width = max(self.step(), stop - start)
        width = min(width, max(self.duration_sec, self.step()))
        start = float(np.clip(start, 0.0, max(0.0, self.duration_sec - width)))
        stop = start + width
        self.view_start, self.view_stop = start, stop
        for plot in (
            self.spec_plot,
            self.emg_plot,
            self.auto_label_plot,
            self.final_label_plot,
        ):
            plot.setXRange(start, stop, padding=0)
        self.overview_plot.setXRange(0, max(self.duration_sec, self.step()), padding=0)
        self._update_overview_focus()
        self._update_epoch_grid()

    def _update_overview_focus(self) -> None:
        duration = max(self.duration_sec, self.step())
        self.focus_left_region.setRegion((0.0, self.view_start))
        self.focus_right_region.setRegion((self.view_stop, duration))
        self.focus_start_line.setPos(self.view_start)
        self.focus_stop_line.setPos(self.view_stop)

    def set_cursor(self, time_sec: float) -> None:
        if self.duration_sec <= 0:
            return
        self.cursor_time = float(np.clip(time_sec, 0.0, self.duration_sec))
        idx = self.epoch_at(self.cursor_time)
        if idx is not None:
            self.selection = (idx, idx + 1)
        self.ensure_cursor_visible()
        self._update_cursor()
        self._update_selection()

    def step_cursor(self, steps: int, extend: bool = False) -> None:
        if self.corrected_labels is None:
            return
        current = self.epoch_at(self.cursor_time) or 0
        target = int(np.clip(current + steps, 0, len(self.corrected_labels) - 1))
        self.cursor_time = target * self.step()
        if extend and self.selection is not None:
            i0, i1 = self.selection
            self.selection = (min(i0, target), max(i1, target + 1))
        else:
            self.selection = (target, target + 1)
        self.ensure_cursor_visible()
        self._update_cursor()
        self._update_selection()

    def ensure_cursor_visible(self) -> None:
        if self.view_start <= self.cursor_time <= self.view_stop:
            return
        width = self.view_stop - self.view_start
        self.set_view(self.cursor_time - width / 2, self.cursor_time + width / 2)

    def select_epochs_from_times(self, start: float, stop: float, final: bool) -> None:
        if self.corrected_labels is None:
            return
        i0, i1 = self.epoch_range(start, stop)
        self.selection = (i0, i1)
        self.cursor_time = i0 * self.step()
        self._update_cursor()
        self._update_selection()

    def apply_stage(self, stage: str) -> None:
        if self.corrected_labels is None or self.corrected_mask is None:
            self._set_status("Run auto staging before correcting labels.")
            return
        i0, i1 = self.current_selection()
        old_labels = self.corrected_labels[i0:i1].copy()
        old_mask = self.corrected_mask[i0:i1].copy()
        if np.all(old_labels == stage):
            self._set_status(f"{stage} already covers selected epoch(s).")
            return
        self.undo_stack.append((i0, i1, old_labels, old_mask))
        self.corrected_labels[i0:i1] = stage
        self.corrected_mask[i0:i1] = self.corrected_labels[i0:i1] != self.auto_labels[i0:i1]
        self._refresh_labels()
        self._update_selection()
        self._set_status(f"Set epochs {i0}-{i1 - 1} to {stage}.")

    def accept_auto_label(self) -> None:
        if self.corrected_labels is None:
            return
        i0, i1 = self.current_selection()
        old_labels = self.corrected_labels[i0:i1].copy()
        old_mask = self.corrected_mask[i0:i1].copy()
        self.undo_stack.append((i0, i1, old_labels, old_mask))
        self.corrected_labels[i0:i1] = self.auto_labels[i0:i1]
        self.corrected_mask[i0:i1] = False
        self._refresh_labels()
        self._set_status(f"Accepted auto labels for epochs {i0}-{i1 - 1}.")

    def undo(self) -> None:
        if not self.undo_stack or self.corrected_labels is None or self.corrected_mask is None:
            self._set_status("Nothing to undo.")
            return
        i0, i1, labels, mask = self.undo_stack.pop()
        self.corrected_labels[i0:i1] = labels
        self.corrected_mask[i0:i1] = mask
        self._refresh_labels()
        self._update_selection()
        self._set_status("Undo complete.")

    def save_corrected(self) -> None:
        try:
            if self.corrected_labels is None or self.corrected_mask is None:
                raise ValueError("Run auto staging before saving corrected results.")
            output_dir = self._resolved_output_dir(Path(self.input_path.text()).expanduser())
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_rows(
                output_dir / "sleep_state_corrected_epochs.csv",
                self._corrected_rows(),
            )
            _write_json(
                output_dir / "sleep_state_corrected_summary.json",
                self._corrected_summary_payload(),
            )
            self._write_qc_figure(
                output_dir / f"sleep_state_corrected_hypnogram.{self.plot_format()}",
                result=self._corrected_result(),
            )
            self._set_status(f"Saved corrected labels to {output_dir}.")
        except Exception as exc:
            self._show_error(str(exc))

    def _write_qc_figure(self, path: Path, result=None) -> None:
        if path.suffix.lower() == ".png":
            self._write_qc_png(path, result=result)
            return
        fig = plot_hypnogram(
            self.eeg,
            self.emg,
            result or self.result,
            self.recording.fs,
            path,
        )
        from matplotlib import pyplot as plt

        plt.close(fig)

    def _write_qc_png(self, path: Path, result=None) -> None:
        with tempfile.TemporaryDirectory(prefix="sleep_state_qc_") as tmp:
            svg_path = Path(tmp) / "hypnogram.svg"
            self._write_qc_figure(svg_path, result=result)
            _render_svg_to_png(svg_path, path)

    def _corrected_result(self):
        return replace(
            self.result,
            labels=self.corrected_labels.copy(),
            summary=_label_summary(self.corrected_labels, mode="corrected"),
        )

    def plot_format(self) -> str:
        return self.figure_format.currentText()

    def current_selection(self) -> tuple[int, int]:
        if self.selection is not None:
            return self.selection
        idx = self.epoch_at(self.cursor_time)
        if idx is None:
            raise ValueError("No epoch is selected.")
        return idx, idx + 1

    def epoch_at(self, time_sec: float) -> int | None:
        if self.corrected_labels is None or len(self.corrected_labels) == 0:
            return None
        return int(np.clip(np.floor(time_sec / self.step()), 0, len(self.corrected_labels) - 1))

    def epoch_range(self, start: float, stop: float) -> tuple[int, int]:
        labels = self.corrected_labels
        if labels is None or len(labels) == 0:
            return 0, 0
        lo, hi = sorted((start, stop))
        i0 = int(np.clip(np.floor(lo / self.step()), 0, len(labels) - 1))
        i1 = int(np.clip(np.floor(hi / self.step()) + 1, i0 + 1, len(labels)))
        return i0, i1

    def step(self) -> float:
        return float(self.result.params.step_sec) if self.result is not None else 1.0

    def _label_duration(self) -> float:
        return len(self.corrected_labels) * self.step() if self.corrected_labels is not None else 1.0

    def _update_cursor(self) -> None:
        for line in self.cursor_lines:
            line.setPos(self.cursor_time)
        idx = self.epoch_at(self.cursor_time)
        if idx is None:
            self.cursor_info.setText("Cursor: -")
            return
        auto = self.auto_labels[idx]
        final = self.corrected_labels[idx]
        self.cursor_info.setText(
            f"Cursor: {self.cursor_time:.1f}s | epoch {idx} | auto {auto} | final {final}"
        )

    def _update_selection(self) -> None:
        if self.selection is None:
            self.selection_region.setVisible(False)
            self.selection_info.setText("Selection: -")
            return
        i0, i1 = self.selection
        start, stop = i0 * self.step(), i1 * self.step()
        self.selection_region.setRegion((start, stop))
        self.selection_region.setVisible(True)
        self.selection_info.setText(f"Selection: epochs {i0}-{i1 - 1} ({start:.1f}-{stop:.1f}s)")

    def _update_epoch_grid(self) -> None:
        for plot, line in self.grid_lines:
            plot.removeItem(line)
        self.grid_lines.clear()
        if self.result is None or self.view_stop - self.view_start > 90:
            return
        start = int(np.floor(self.view_start / self.step()))
        stop = int(np.ceil(self.view_stop / self.step()))
        for idx in range(max(0, start), stop + 1):
            for plot in (self.auto_label_plot, self.final_label_plot):
                line = pg.InfiniteLine(
                    pos=idx * self.step(),
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(255, 255, 255, 65),
                )
                line.setZValue(20)
                plot.addItem(line)
                self.grid_lines.append((plot, line))

    def _corrected_rows(self) -> list[dict[str, int | float | str]]:
        step = self.step()
        rows: list[dict[str, int | float | str]] = []
        for idx, (auto, final, corrected) in enumerate(
            zip(self.auto_labels, self.corrected_labels, self.corrected_mask)
        ):
            rows.append(
                {
                    "step_idx": idx,
                    "start_sec": round(idx * step, 6),
                    "end_sec": round((idx + 1) * step, 6),
                    "auto_stage": str(auto),
                    "corrected_stage": str(final) if bool(corrected) else "",
                    "final_stage": str(final),
                    "corrected": int(bool(corrected)),
                }
            )
        return rows

    def _summary_payload(self) -> dict:
        return {
            "input": {
                "path": str(self.recording.path),
                "fs": self.recording.fs,
                "n_samples": self.recording.n_samples,
                "duration_sec": self.recording.duration_sec,
                "eeg_col": self.recording.eeg_col,
                "emg_col": self.recording.emg_col,
            },
            "preprocessing": {
                "enabled": self.preprocess.isChecked(),
                "eeg_hp": self.eeg_hp.value(),
                "emg_hp": self.emg_hp.value(),
                "line_freq": self.line_freq.value(),
            },
            "staging": {
                "summary": self.result.summary,
                "thresholds": self.result.thresholds,
                "params": vars(self.result.params),
            },
        }

    def _corrected_summary_payload(self) -> dict:
        counts = {stage: int(np.sum(self.corrected_labels == stage)) for stage in STAGES}
        n_labels = len(self.corrected_labels)
        return {
            "n_steps": n_labels,
            "wake_steps": counts[WAKE],
            "nrem_steps": counts[NREM],
            "rem_steps": counts[REM],
            "wake_fraction": counts[WAKE] / n_labels if n_labels else 0.0,
            "nrem_fraction": counts[NREM] / n_labels if n_labels else 0.0,
            "rem_fraction": counts[REM] / n_labels if n_labels else 0.0,
            "corrected_steps": int(np.sum(self.corrected_mask)),
            "step_sec": self.step(),
        }

    def _resolved_output_dir(self, input_path: Path) -> Path:
        text = self.output_dir.text().strip()
        return Path(text).expanduser() if text else input_path.with_name("sleep_state_staging")

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _show_error(self, text: str) -> None:
        self._set_status(text)
        QtWidgets.QMessageBox.critical(self, "Sleep State Review", text)


def _spectrogram(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    nperseg = min(len(eeg), int(round(5 * fs)))
    noverlap = min(max(0, nperseg - int(round(fs))), nperseg - 1)
    freqs, times, power = spectrogram(
        eeg,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(4096, nperseg),
    )
    keep_f = (freqs >= 0) & (freqs <= 30)
    power_db = 10 * np.log10(power[keep_f] + 1e-15)
    if power_db.shape[1] > 3000:
        keep_t = np.linspace(0, power_db.shape[1] - 1, 3000).astype(int)
        times = times[keep_t]
        power_db = power_db[:, keep_t]
    vmin = float(np.percentile(power_db, 5))
    return freqs[keep_f], times, power_db, vmin, vmin + 30


def _image_rect(times: np.ndarray, freqs: np.ndarray, duration_sec: float) -> QtCore.QRectF:
    x0 = float(times[0]) if len(times) else 0.0
    x1 = float(times[-1]) if len(times) > 1 else duration_sec
    y0 = float(freqs[0]) if len(freqs) else 0.0
    y1 = float(freqs[-1]) if len(freqs) > 1 else 30.0
    return QtCore.QRectF(x0, y0, max(1e-6, x1 - x0), max(1e-6, y1 - y0))


def _stage_ids(labels: np.ndarray) -> np.ndarray:
    return np.asarray([STAGE_IDS.get(str(label), 0) for label in labels], dtype=np.uint8)


def _label_summary(labels: np.ndarray, mode: str) -> dict[str, int | float | str]:
    n_labels = len(labels)
    counts = {stage: int(np.sum(labels == stage)) for stage in STAGES}
    return {
        "n_steps": n_labels,
        "wake_steps": counts[WAKE],
        "nrem_steps": counts[NREM],
        "rem_steps": counts[REM],
        "wake_fraction": counts[WAKE] / n_labels if n_labels else 0.0,
        "nrem_fraction": counts[NREM] / n_labels if n_labels else 0.0,
        "rem_fraction": counts[REM] / n_labels if n_labels else 0.0,
        "mode": mode,
    }


def _write_rows(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _render_svg_to_png(svg_path: Path, png_path: Path) -> None:
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    size = renderer.defaultSize()
    if not size.isValid():
        size = QtCore.QSize(2200, 700)
    image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.white)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(png_path)):
        raise ValueError(f"Failed to write PNG figure: {png_path}")


def _spin(low: int, high: int, value: int) -> QtWidgets.QSpinBox:
    widget = QtWidgets.QSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


def _double_spin(low: float, high: float, value: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def _add_row(grid, row: int, label: str, widget: QtWidgets.QWidget, col: int = 0) -> None:
    grid.addWidget(_qt_label(label), row, col)
    grid.addWidget(widget, row, col + 1)


def _qt_label(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
    return label


def _row_label(text: str) -> pg.LabelItem:
    return pg.LabelItem(
        f"<span style='color:#222; font-weight:600'>{text}</span>",
        justify="right",
    )


def _stage_legend() -> StageLegend:
    return StageLegend()


def _editing_text() -> bool:
    widget = QtWidgets.QApplication.focusWidget()
    return isinstance(
        widget,
        (
            QtWidgets.QLineEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QComboBox,
        ),
    )


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = SleepStateReviewWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
