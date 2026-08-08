from __future__ import annotations

import re
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from scipy.signal import resample_poly

from .experiment import ExperimentSource, VideoSource, load_video_frame_times, scan_experiment
from .io import load_eegemg_txt
from .preprocess import preprocess_eeg_emg
from .review import ReviewSession, write_auto_results
from .staging import NREM, REM, WAKE, WAKE_MODES, StagingParams, classify_sleep_state


STAGES = (WAKE, NREM, REM)
STAGE_IDS = {WAKE: 0, NREM: 1, REM: 2}
STAGE_COLORS = {
    WAKE: (56, 99, 255, 255),
    NREM: (229, 94, 148, 255),
    REM: (255, 162, 51, 255),
}
STAGE_LUT = np.asarray([STAGE_COLORS[stage] for stage in STAGES], dtype=np.ubyte)
STATUS_TEXT = {
    "unreviewed": "○ Unreviewed",
    "reviewed": "✓ Reviewed",
    "needs_review": "! Review again",
}
_SPECTROGRAM_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = {}
_NAVIGATION_STEPS = 1000


@dataclass
class AnalysisSettings:
    preprocess: bool = True
    eeg_hp: float = 0.5
    emg_hp: float = 1.0
    line_freq: float = 50.0
    feature_window_sec: float = 5.0
    wake_mode: str = "auto"


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [_format_time(value, decimals=spacing < 1.0) for value in values]


class ReviewViewBox(pg.ViewBox):
    cursor_requested = QtCore.Signal(float)
    selection_requested = QtCore.Signal(float, float, bool)
    pan_requested = QtCore.Signal(float)
    reset_requested = QtCore.Signal()

    def __init__(self, editable: bool) -> None:
        super().__init__(enableMenu=False)
        self.editable = editable
        self.setMouseEnabled(x=False, y=False)

    def wheelEvent(self, event, axis=None) -> None:
        event.accept()

    def mouseClickEvent(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        time_sec = float(self.mapSceneToView(event.scenePos()).x())
        self.cursor_requested.emit(time_sec)
        if self.editable:
            self.selection_requested.emit(time_sec, time_sec, True)
        event.accept()

    def mouseDragEvent(self, event, axis=None) -> None:
        button = event.button()
        if self.editable and button == QtCore.Qt.MouseButton.LeftButton:
            start = float(self.mapSceneToView(event.buttonDownScenePos(button)).x())
            stop = float(self.mapSceneToView(event.scenePos()).x())
            self.selection_requested.emit(start, stop, bool(event.isFinish()))
            event.accept()
            return
        if button in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            previous = float(self.mapSceneToView(event.lastScenePos()).x())
            current = float(self.mapSceneToView(event.scenePos()).x())
            self.pan_requested.emit(previous - current)
            event.accept()
            return
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.reset_requested.emit()
            event.accept()
            return
        event.ignore()


class NoWheelSlider(QtWidgets.QSlider):
    def wheelEvent(self, event) -> None:
        event.ignore()


class GainControl(QtWidgets.QWidget):
    gain_changed = QtCore.Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.label = QtWidgets.QLabel("1.00×")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.slider = NoWheelSlider(QtCore.Qt.Orientation.Vertical)
        self.slider.setRange(-20, 12)
        self.slider.setValue(0)
        self.slider.setToolTip("Display gain; lower values compress peaks")
        self.slider.installEventFilter(self)
        self.slider.valueChanged.connect(self._emit_gain)
        self.slider.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.slider.customContextMenuRequested.connect(lambda _pos: self.reset())
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        self.setMaximumWidth(58)

    def reset(self) -> None:
        self.slider.setValue(0)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.slider and event.type() == QtCore.QEvent.Type.MouseButtonDblClick:
            self.reset()
            return True
        return super().eventFilter(watched, event)

    def _emit_gain(self, value: int) -> None:
        gain = 10.0 ** (value / 20.0)
        self.label.setText(f"{gain:.2f}×")
        self.gain_changed.emit(gain)


class SignalView(QtWidgets.QWidget):
    selection_requested = QtCore.Signal(float, float, bool)
    cursor_requested = QtCore.Signal(float)

    def __init__(self, editable: bool) -> None:
        super().__init__()
        self.editable = editable
        self.session: ReviewSession | None = None
        self.domain = (0.0, 1.0)
        self.view_range = (0.0, 1.0)
        self._changing_navigation = False
        self._gain_base = {"eeg": (0.0, 1.0), "emg": (0.0, 1.0)}
        self._note_lines: list[tuple[pg.PlotItem, pg.InfiniteLine]] = []
        self._global_video_regions: list[tuple[pg.PlotItem, pg.LinearRegionItem]] = []

        pg.setConfigOptions(
            antialias=False,
            imageAxisOrder="row-major",
            background="w",
            foreground="k",
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        display_bar = QtWidgets.QHBoxLayout()
        self.eeg_source = QtWidgets.QComboBox()
        self.emg_source = QtWidgets.QComboBox()
        for combo in (self.eeg_source, self.emg_source):
            combo.addItems(("Preprocessed", "Original"))
            combo.currentIndexChanged.connect(self._signal_source_changed)
        display_bar.addWidget(QtWidgets.QLabel("EEG"))
        display_bar.addWidget(self.eeg_source)
        display_bar.addSpacing(12)
        display_bar.addWidget(QtWidgets.QLabel("EMG"))
        display_bar.addWidget(self.emg_source)
        display_bar.addStretch(1)
        self.cursor_label = QtWidgets.QLabel("Cursor: –")
        display_bar.addWidget(self.cursor_label)
        layout.addLayout(display_bar)

        tracks = QtWidgets.QWidget()
        track_layout = QtWidgets.QVBoxLayout(tracks)
        track_layout.setContentsMargins(0, 0, 0, 0)
        self.track_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        track_layout.addWidget(self.track_splitter)

        self.spec_plot = self._plot("EEG power", editable=self.editable)
        self.spec_image = pg.ImageItem()
        self.spec_image.setColorMap(pg.colormap.get("turbo"))
        self.spec_plot.addItem(self.spec_image)
        self.spec_colorbar = pg.ColorBarItem(
            values=(0, 1),
            width=14,
            colorMap=pg.colormap.get("turbo"),
            label="dB",
            interactive=False,
            colorMapMenu=False,
        )
        self.spec_colorbar.setFixedWidth(58)
        self.spec_colorbar.setImageItem(self.spec_image, insert_in=self.spec_plot.getPlotItem())
        self.track_splitter.addWidget(self.spec_plot)

        self.eeg_track, self.eeg_plot, self.eeg_gain = self._signal_track("EEG (µV)")
        self.eeg_curve = pg.PlotDataItem(pen=pg.mkPen("k", width=0.7))
        self.eeg_plot.addItem(self.eeg_curve)
        self.track_splitter.addWidget(self.eeg_track)

        self.emg_track, self.emg_plot, self.emg_gain = self._signal_track("EMG (mV)")
        self.emg_curve = pg.PlotDataItem(pen=pg.mkPen("k", width=0.7))
        self.emg_plot.addItem(self.emg_curve)
        self.track_splitter.addWidget(self.emg_track)

        self.auto_plot = self._plot("Auto", editable=self.editable)
        self.final_plot = self._plot("Draft" if self.editable else "Committed", editable=self.editable)
        for plot in (self.auto_plot, self.final_plot):
            plot.getAxis("left").setStyle(showValues=False, tickLength=0)
            right_axis = plot.getAxis("right")
            plot.showAxis("right")
            right_axis.setStyle(showValues=False, tickLength=0)
            right_axis.setPen(pg.mkPen(None))
            right_axis.setWidth(58)
            plot.setMinimumHeight(44)
            plot.setMaximumHeight(52)
            plot.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        self.auto_image = pg.ImageItem()
        self.final_image = pg.ImageItem()
        self.auto_plot.addItem(self.auto_image)
        self.final_plot.addItem(self.final_image)
        self.track_splitter.addWidget(self.auto_plot)
        self.track_splitter.addWidget(self.final_plot)
        self.track_splitter.setStretchFactor(0, 4)
        self.track_splitter.setStretchFactor(1, 2)
        self.track_splitter.setStretchFactor(2, 2)
        self.track_splitter.setStretchFactor(3, 0)
        self.track_splitter.setStretchFactor(4, 0)
        self.track_splitter.setCollapsible(3, False)
        self.track_splitter.setCollapsible(4, False)
        self.track_splitter.setSizes((240, 140, 140, 48, 48))

        self.plots = (self.spec_plot, self.eeg_plot, self.emg_plot, self.auto_plot, self.final_plot)
        self.cursor_lines: list[pg.InfiniteLine] = []
        self.selection_regions: list[pg.LinearRegionItem] = []
        self.video_regions: list[pg.LinearRegionItem] = []
        for plot in self.plots:
            cursor = pg.InfiniteLine(
                angle=90,
                movable=False,
                pen=pg.mkPen(255, 75, 20, 255, width=3),
            )
            cursor.addMarker("v", position=0.97, size=10)
            cursor.setZValue(60)
            plot.addItem(cursor)
            self.cursor_lines.append(cursor)
            label_track = plot in (self.auto_plot, self.final_plot)
            selection = pg.LinearRegionItem(
                values=(0, 0),
                movable=False,
                brush=pg.mkBrush(255, 255, 255, 125 if label_track else 70),
                pen=pg.mkPen(25, 25, 25, 245, width=3 if label_track else 2),
            )
            selection.setVisible(False)
            selection.setZValue(40)
            plot.addItem(selection)
            self.selection_regions.append(selection)
            video = pg.LinearRegionItem(
                values=(0, 0),
                movable=False,
                brush=pg.mkBrush(80, 150, 255, 18),
                pen=pg.mkPen(80, 150, 255, 80),
            )
            video.setVisible(False)
            video.setZValue(5)
            plot.addItem(video)
            self.video_regions.append(video)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tracks)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(360)
        layout.addWidget(scroll, 1)
        layout.addWidget(self._stage_legend())

        navigation = QtWidgets.QHBoxLayout()
        self.time_zoom_slider = NoWheelSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_zoom_slider.setRange(0, _NAVIGATION_STEPS)
        self.time_zoom_slider.setToolTip(
            "Visible time width: left is the full window, right is more detail"
        )
        self.time_width_label = QtWidgets.QLabel("00:01")
        self.time_width_label.setMinimumWidth(58)
        self.time_position_slider = NoWheelSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_position_slider.setRange(0, _NAVIGATION_STEPS)
        self.time_position_slider.setToolTip("Move the visible window through the current review range")
        self.time_position_label = QtWidgets.QLabel("00:00")
        self.time_position_label.setMinimumWidth(58)
        self.fit_button = QtWidgets.QPushButton("Fit window")
        self.fit_button.clicked.connect(self.fit_domain)
        self.time_zoom_slider.valueChanged.connect(self._time_zoom_changed)
        self.time_position_slider.valueChanged.connect(self._time_position_changed)
        navigation.addWidget(QtWidgets.QLabel("Time width"))
        navigation.addWidget(self.time_zoom_slider, 2)
        navigation.addWidget(self.time_width_label)
        navigation.addSpacing(12)
        navigation.addWidget(QtWidgets.QLabel("Window position"))
        navigation.addWidget(self.time_position_slider, 3)
        navigation.addWidget(self.time_position_label)
        navigation.addWidget(self.fit_button)
        layout.addLayout(navigation)

        self.eeg_gain.gain_changed.connect(lambda gain: self._apply_gain("eeg", gain))
        self.emg_gain.gain_changed.connect(lambda gain: self._apply_gain("emg", gain))

    def _plot(self, label: str, editable: bool) -> pg.PlotWidget:
        view_box = ReviewViewBox(editable=editable)
        view_box.cursor_requested.connect(self.set_cursor)
        view_box.selection_requested.connect(self.selection_requested)
        view_box.pan_requested.connect(self.pan_by)
        view_box.reset_requested.connect(self.fit_domain)
        plot = pg.PlotWidget(viewBox=view_box, axisItems={"bottom": TimeAxis(orientation="bottom")})
        plot.setLabel("left", label)
        plot.getAxis("left").setWidth(58)
        plot.getAxis("bottom").setHeight(24)
        plot.setMenuEnabled(False)
        plot.setMinimumHeight(48)
        return plot

    def _stage_legend(self) -> QtWidgets.QWidget:
        legend = QtWidgets.QWidget()
        legend.setMaximumHeight(30)
        row = QtWidgets.QHBoxLayout(legend)
        row.setContentsMargins(0, 2, 0, 2)
        row.addStretch(1)
        for stage in STAGES:
            red, green, blue, _alpha = STAGE_COLORS[stage]
            swatch = QtWidgets.QFrame()
            swatch.setFixedSize(16, 12)
            swatch.setStyleSheet(
                f"background-color: rgb({red}, {green}, {blue}); border: 1px solid #555"
            )
            row.addWidget(swatch)
            row.addWidget(QtWidgets.QLabel(stage))
            row.addSpacing(18)
        row.addStretch(1)
        return legend

    def _signal_track(self, label: str) -> tuple[QtWidgets.QWidget, pg.PlotWidget, GainControl]:
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        plot = self._plot(label, editable=self.editable)
        gain = GainControl()
        layout.addWidget(plot, 1)
        layout.addWidget(gain)
        return wrapper, plot, gain

    def set_session(self, session: ReviewSession, final_labels: np.ndarray) -> None:
        self.session = session
        self._set_spectrogram()
        self.refresh_labels(final_labels)
        self.eeg_gain.reset()
        self.emg_gain.reset()

    def set_domain(self, start: float, stop: float) -> None:
        if stop <= start:
            stop = start + 1.0
        self.domain = (float(start), float(stop))
        self._reset_gain_bases()
        self.fit_domain()

    def set_video(self, video: VideoSource | None, frame_times: np.ndarray | None = None) -> None:
        self._clear_note_lines()
        for region in self.video_regions:
            region.setVisible(video is not None)
            if video is not None:
                region.setRegion((video.start_sec, video.stop_sec))
        if video is None or frame_times is None or len(frame_times) == 0:
            return
        for frame in _note_frame_numbers(video.comment):
            if frame < 0 or frame >= len(frame_times):
                continue
            position = video.start_sec + float(frame_times[frame])
            line = pg.InfiniteLine(
                pos=position,
                angle=90,
                movable=False,
                pen=pg.mkPen(230, 140, 30, 190, style=QtCore.Qt.PenStyle.DashLine),
                label=f"f{frame}",
                labelOpts={"position": 0.85, "color": (150, 80, 0)},
            )
            self.final_plot.addItem(line)
            self._note_lines.append((self.final_plot, line))

    def show_global_videos(self, videos: tuple[VideoSource, ...]) -> None:
        for plot, region in self._global_video_regions:
            plot.removeItem(region)
        self._global_video_regions.clear()
        for video in videos:
            if not video.duration_sec:
                continue
            region = pg.LinearRegionItem(
                values=(video.start_sec, video.stop_sec),
                movable=False,
                brush=pg.mkBrush(80, 150, 255, 45),
                pen=pg.mkPen(80, 150, 255, 80),
            )
            region.setZValue(2)
            self.final_plot.addItem(region)
            self._global_video_regions.append((self.final_plot, region))

    def refresh_labels(self, final_labels: np.ndarray | None = None) -> None:
        if self.session is None:
            return
        auto = self.session.auto_labels
        final = auto if final_labels is None else final_labels
        self._set_label_image(self.auto_image, auto)
        self._set_label_image(self.final_image, final)
        for plot in (self.auto_plot, self.final_plot):
            plot.setYRange(0, 1, padding=0)

    def set_selection(self, bounds: tuple[float, float] | None) -> None:
        for region in self.selection_regions:
            region.setVisible(bounds is not None)
            if bounds is not None:
                region.setRegion(bounds)

    def set_cursor(self, time_sec: float) -> None:
        if self.session is None:
            return
        time_sec = float(np.clip(time_sec, *self.domain))
        for line in self.cursor_lines:
            line.setPos(time_sec)
        self.cursor_label.setText(f"Cursor: {_format_time(time_sec, decimals=True)}")
        self.cursor_requested.emit(time_sec)

    def fit_domain(self) -> None:
        self._set_view(*self.domain)

    def pan_by(self, delta_sec: float) -> None:
        start, stop = self.view_range
        self._set_view(start + delta_sec, stop + delta_sec)

    def _time_zoom_changed(self, value: int) -> None:
        if self._changing_navigation:
            return
        domain_width = self.domain[1] - self.domain[0]
        width = _view_width_from_slider(domain_width, value)
        center = sum(self.view_range) / 2.0
        self._set_view(center - width / 2.0, center + width / 2.0)

    def _time_position_changed(self, value: int) -> None:
        if self._changing_navigation:
            return
        domain_start, domain_stop = self.domain
        width = self.view_range[1] - self.view_range[0]
        available = max(0.0, domain_stop - domain_start - width)
        start = domain_start + available * value / _NAVIGATION_STEPS
        self._set_view(start, start + width)

    def _set_view(self, start: float, stop: float) -> None:
        domain_start, domain_stop = self.domain
        width = min(max(1.0, stop - start), domain_stop - domain_start)
        start = float(np.clip(start, domain_start, max(domain_start, domain_stop - width)))
        stop = start + width
        self.view_range = (start, stop)
        for plot in self.plots:
            plot.setXRange(start, stop, padding=0)
        self._sync_navigation_controls()
        self._update_waveforms()

    def _sync_navigation_controls(self) -> None:
        domain_start, domain_stop = self.domain
        start, stop = self.view_range
        domain_width = domain_stop - domain_start
        width = stop - start
        available = max(0.0, domain_width - width)
        zoom_value = _slider_from_view_width(domain_width, width)
        position_value = 0 if available <= 1e-9 else round(
            _NAVIGATION_STEPS * (start - domain_start) / available
        )
        self._changing_navigation = True
        self.time_zoom_slider.setValue(zoom_value)
        self.time_position_slider.setValue(position_value)
        self.time_position_slider.setEnabled(available > 1e-9)
        self._changing_navigation = False
        self.time_width_label.setText(_format_time(width, decimals=width < 10.0))
        self.time_position_label.setText(_format_time((start + stop) / 2.0, decimals=width < 10.0))

    def _set_spectrogram(self) -> None:
        assert self.session is not None
        key = (str(self.session.recording.path), len(self.session.processed_eeg))
        cached = _SPECTROGRAM_CACHE.get(key)
        if cached is None:
            cached = _spectrogram_preview(
                self.session.processed_eeg,
                self.session.recording.fs,
                max_time_bins=3000,
            )
            _SPECTROGRAM_CACHE[key] = cached
        freqs, times, power, low, high = cached
        self.spec_image.setImage(np.clip(power, low, high), autoLevels=False)
        self.spec_image.setLevels((low, high))
        self.spec_colorbar.setLevels((low, high))
        if len(times) and len(freqs):
            self.spec_image.setRect(
                QtCore.QRectF(
                    float(times[0]),
                    float(freqs[0]),
                    max(1e-6, float(times[-1] - times[0])),
                    max(1e-6, float(freqs[-1] - freqs[0])),
                )
            )
        self.spec_plot.setYRange(0, 30, padding=0)

    def _set_label_image(self, image: pg.ImageItem, labels: np.ndarray) -> None:
        assert self.session is not None
        ids = np.asarray([STAGE_IDS.get(str(label), 0) for label in labels], dtype=np.uint8)[None, :]
        image.setLookupTable(STAGE_LUT)
        image.setLevels((0, 2))
        image.setImage(ids, autoLevels=False)
        starts = self.session.label_starts
        stops = self.session.label_stops
        if len(starts):
            image.setRect(QtCore.QRectF(float(starts[0]), 0, float(stops[-1] - starts[0]), 1))

    def _signal_source_changed(self) -> None:
        if self.session is None:
            return
        self._reset_gain_bases()
        self._update_waveforms()

    def _source_signal(self, name: str) -> np.ndarray:
        assert self.session is not None
        combo = self.eeg_source if name == "eeg" else self.emg_source
        if combo.currentText() == "Original":
            return self.session.recording.eeg if name == "eeg" else self.session.recording.emg
        return self.session.processed_eeg if name == "eeg" else self.session.processed_emg

    def _reset_gain_bases(self) -> None:
        if self.session is None:
            return
        start, stop = self.domain
        fs = self.session.recording.fs
        first = max(0, int(start * fs))
        last = min(self.session.recording.n_samples, int(np.ceil(stop * fs)))
        for name in ("eeg", "emg"):
            values = self._source_signal(name)[first:last]
            if len(values) > 100_000:
                values = values[:: max(1, len(values) // 100_000)]
            low, high = np.percentile(values, (0.5, 99.5)) if len(values) else (-1.0, 1.0)
            center = float((low + high) / 2.0)
            half = max(float(high - low) * 0.6, 1e-9)
            self._gain_base[name] = (center, half)
        self._apply_gain("eeg", 10.0 ** (self.eeg_gain.slider.value() / 20.0))
        self._apply_gain("emg", 10.0 ** (self.emg_gain.slider.value() / 20.0))

    def _apply_gain(self, name: str, gain: float) -> None:
        center, base_half = self._gain_base[name]
        half = base_half / max(gain, 1e-6)
        plot = self.eeg_plot if name == "eeg" else self.emg_plot
        plot.setYRange(center - half, center + half, padding=0)

    def _update_waveforms(self) -> None:
        if self.session is None:
            return
        fs = self.session.recording.fs
        start, stop = self.view_range
        for name, curve in (("eeg", self.eeg_curve), ("emg", self.emg_curve)):
            times, values = _visible_envelope(self._source_signal(name), fs, start, stop)
            curve.setData(times, values)

    def _clear_note_lines(self) -> None:
        for plot, line in self._note_lines:
            plot.removeItem(line)
        self._note_lines.clear()


class SourcePage(QtWidgets.QWidget):
    browse_requested = QtCore.Signal()
    adjust_requested = QtCore.Signal()
    run_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Sleep-state staging experiment")
        title.setStyleSheet("font-size: 18px; font-weight: 600")
        layout.addWidget(title)
        description = QtWidgets.QLabel(
            "Choose an experiment directory, verify the detected structure, then run automatic analysis."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        path_row = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLineEdit()
        self.path_label.setReadOnly(True)
        browse = QtWidgets.QPushButton("Choose experiment directory…")
        browse.clicked.connect(self.browse_requested)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(("Detected item", "Value", "Status"))
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree, 1)

        self.warning_label = QtWidgets.QLabel()
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #a44b00")
        layout.addWidget(self.warning_label)

        buttons = QtWidgets.QHBoxLayout()
        self.adjust_button = QtWidgets.QPushButton("Manual source mapping…")
        self.adjust_button.clicked.connect(self.adjust_requested)
        self.run_button = QtWidgets.QPushButton("Run automatic analysis")
        self.run_button.clicked.connect(self.run_requested)
        buttons.addWidget(self.adjust_button)
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)
        self.set_source(None)

    def set_source(self, source: ExperimentSource | None) -> None:
        self.tree.clear()
        self.path_label.setText(str(source.directory) if source else "")
        self.adjust_button.setEnabled(source is not None)
        self.run_button.setEnabled(bool(source and source.is_ready))
        if source is None:
            self.warning_label.setText("")
            return
        note = QtWidgets.QTreeWidgetItem(("Note", str(source.note_path), "✓"))
        self.tree.addTopLevelItem(note)
        if source.note_frame_rate:
            QtWidgets.QTreeWidgetItem(note, ("Note frame rate", f"{source.note_frame_rate:g} Hz", "metadata"))
        for recording in source.recordings:
            root = QtWidgets.QTreeWidgetItem((recording.name, "", "✓" if recording.path else "missing"))
            self.tree.addTopLevelItem(root)
            if recording.path:
                duration = _format_time(recording.info.duration_sec) if recording.info and recording.info.duration_sec else "unknown"
                QtWidgets.QTreeWidgetItem(root, ("EEG/EMG", str(recording.path), f"{duration}, {recording.info.fs:g} Hz"))
            videos = QtWidgets.QTreeWidgetItem(root, ("Movies", f"{len(recording.videos)} detected", ""))
            for video in recording.videos:
                if video.path:
                    frame_text = f"{video.n_frames} frames" if video.n_frames is not None else "TIFF"
                    rate_text = f", {video.frame_rate:.4g} Hz" if video.frame_rate else ""
                    status = frame_text + rate_text
                else:
                    status = "missing"
                value = f"{_format_time(video.start_sec)} → {_format_time(video.stop_sec)}"
                QtWidgets.QTreeWidgetItem(videos, (video.display_name, value, status))
            for event in recording.events:
                when = _format_time(event.time_sec) if event.time_sec is not None else ""
                QtWidgets.QTreeWidgetItem(root, ("Event", event.label or event.raw_text, when))
        self.tree.expandToDepth(1)
        self.warning_label.setText("\n".join(f"• {warning}" for warning in source.warnings))


class SourceMappingDialog(QtWidgets.QDialog):
    def __init__(self, source: ExperimentSource, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual source mapping")
        self.resize(780, 420)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.note_path = QtWidgets.QLineEdit(str(source.note_path))
        self.tiff_path = QtWidgets.QLineEdit(str(source.tiff_directory))
        self.recording_paths = QtWidgets.QPlainTextEdit(
            "\n".join(str(recording.path) for recording in source.recordings if recording.path)
        )
        form.addRow("Note file", self._browse_row(self.note_path, directory=False))
        form.addRow("Raw TIFF directory", self._browse_row(self.tiff_path, directory=True))
        form.addRow("EEG/EMG files\n(one path per line)", self.recording_paths)
        layout.addLayout(form)
        choose_recordings = QtWidgets.QPushButton("Choose EEG/EMG files…")
        choose_recordings.clicked.connect(self._choose_recordings)
        layout.addWidget(choose_recordings)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[Path, list[Path], Path]:
        recordings = [Path(line.strip()) for line in self.recording_paths.toPlainText().splitlines() if line.strip()]
        return Path(self.note_path.text()), recordings, Path(self.tiff_path.text())

    def _browse_row(self, edit: QtWidgets.QLineEdit, directory: bool) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QtWidgets.QPushButton("Browse…")
        if directory:
            button.clicked.connect(lambda: self._choose_directory(edit))
        else:
            button.clicked.connect(lambda: self._choose_file(edit))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return widget

    def _choose_file(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose Note file", filter="Text files (*.txt)")
        if path:
            edit.setText(path)

    def _choose_directory(self, edit: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose raw TIFF directory")
        if path:
            edit.setText(path)

    def _choose_recordings(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose EEG/EMG files", filter="Text files (*.txt)")
        if paths:
            self.recording_paths.setPlainText("\n".join(paths))


class AnalysisSettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: AnalysisSettings, source: ExperimentSource | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Automatic analysis parameters")
        self.resize(500, 460)
        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        fs_text = "–"
        if source and source.recordings and source.recordings[0].info:
            fs_text = f"{source.recordings[0].info.fs:g} Hz (detected)"
        form.addRow("EEG/EMG sampling rate", QtWidgets.QLabel(fs_text))
        form.addRow("Label interval", QtWidgets.QLabel("1.0 s (fixed)"))
        self.preprocess = QtWidgets.QCheckBox("Enable preprocessing")
        self.preprocess.setChecked(settings.preprocess)
        self.eeg_hp = _double_spin(0.0, 500.0, settings.eeg_hp, 2)
        self.emg_hp = _double_spin(0.0, 500.0, settings.emg_hp, 2)
        self.line_freq = _double_spin(0.0, 500.0, settings.line_freq, 2)
        self.feature_window = _double_spin(1.0, 120.0, settings.feature_window_sec, 2)
        self.wake_mode = QtWidgets.QComboBox()
        self.wake_mode.addItems(WAKE_MODES)
        self.wake_mode.setCurrentText(settings.wake_mode)
        form.addRow("Preprocessing", self.preprocess)
        form.addRow("EEG high-pass (Hz)", self.eeg_hp)
        form.addRow("EMG high-pass (Hz)", self.emg_hp)
        form.addRow("Line notch (Hz)", self.line_freq)
        form.addRow("Symmetric feature window (s)", self.feature_window)
        form.addRow("Wake mode", self.wake_mode)
        note_rate = source.note_frame_rate if source else None
        form.addRow("Note video frame rate", QtWidgets.QLabel(f"{note_rate:g} Hz (alignment only)" if note_rate else "–"))
        scroll.setWidget(body)
        layout.addWidget(scroll)
        reset = QtWidgets.QPushButton("Restore defaults")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            preprocess=self.preprocess.isChecked(),
            eeg_hp=self.eeg_hp.value(),
            emg_hp=self.emg_hp.value(),
            line_freq=self.line_freq.value(),
            feature_window_sec=self.feature_window.value(),
            wake_mode=self.wake_mode.currentText(),
        )

    def _reset(self) -> None:
        defaults = AnalysisSettings()
        self.preprocess.setChecked(defaults.preprocess)
        self.eeg_hp.setValue(defaults.eeg_hp)
        self.emg_hp.setValue(defaults.emg_hp)
        self.line_freq.setValue(defaults.line_freq)
        self.feature_window.setValue(defaults.feature_window_sec)
        self.wake_mode.setCurrentText(defaults.wake_mode)


class AnalysisWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, source: ExperimentSource, settings: AnalysisSettings) -> None:
        super().__init__()
        self.source = source
        self.settings = settings

    @QtCore.Slot()
    def run(self) -> None:
        sessions: list[ReviewSession] = []
        try:
            for recording_source in self.source.recordings:
                if recording_source.path is None:
                    raise ValueError(f"{recording_source.name} has no EEG/EMG file.")
                self.progress.emit(f"Loading {recording_source.name}: {recording_source.path.name}")
                loaded = load_eegemg_txt(recording_source.path, eeg_col=1, emg_col=2, fs=None)
                if self.settings.preprocess:
                    self.progress.emit(f"Preprocessing {recording_source.name}")
                    eeg, emg = preprocess_eeg_emg(
                        loaded.eeg,
                        loaded.emg,
                        loaded.fs,
                        eeg_hp_cutoff=self.settings.eeg_hp,
                        emg_hp_cutoff=self.settings.emg_hp,
                        line_freq=self.settings.line_freq,
                    )
                else:
                    eeg, emg = loaded.eeg, loaded.emg
                self.progress.emit(f"Classifying {recording_source.name}")
                result = classify_sleep_state(
                    eeg,
                    emg,
                    loaded.fs,
                    params=StagingParams(
                        wake_mode=self.settings.wake_mode,
                        epoch_sec=self.settings.feature_window_sec,
                        step_sec=1.0,
                    ),
                )
                compact_recording = replace(
                    loaded,
                    eeg=np.asarray(loaded.eeg, dtype=np.float32),
                    emg=np.asarray(loaded.emg, dtype=np.float32),
                )
                output = self.source.directory / "sleep_state_staging"
                if len(self.source.recordings) > 1:
                    output = output / f"session_{recording_source.number}"
                session = ReviewSession(
                    source=recording_source,
                    recording=compact_recording,
                    processed_eeg=np.asarray(eeg, dtype=np.float32),
                    processed_emg=np.asarray(emg, dtype=np.float32),
                    result=result,
                    output_directory=output,
                )
                write_auto_results(session)
                sessions.append(session)
            self.finished.emit(sessions)
        except Exception:
            self.failed.emit(traceback.format_exc())


class LocalReviewPage(QtWidgets.QWidget):
    committed = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[ReviewSession] = []
        self.session: ReviewSession | None = None
        self.video: VideoSource | None = None
        self.frame_times = np.asarray([], dtype=np.float64)
        self.selection: tuple[int, int] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Session"))
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.currentIndexChanged.connect(self._session_changed)
        top.addWidget(self.session_combo)
        top.addSpacing(12)
        top.addWidget(QtWidgets.QLabel("Context ±"))
        self.context_sec = QtWidgets.QSpinBox()
        self.context_sec.setRange(0, 600)
        self.context_sec.setValue(60)
        self.context_sec.setSuffix(" s")
        self.context_sec.valueChanged.connect(self._video_changed)
        top.addWidget(self.context_sec)
        self.queue_button = QtWidgets.QPushButton("Hide queue")
        self.queue_button.clicked.connect(self._toggle_queue)
        top.addWidget(self.queue_button)
        top.addStretch(1)
        self.session_status = QtWidgets.QLabel()
        top.addWidget(self.session_status)
        layout.addLayout(top)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.queue_panel = QtWidgets.QWidget()
        queue_layout = QtWidgets.QVBoxLayout(self.queue_panel)
        queue_layout.addWidget(QtWidgets.QLabel("Video review queue"))
        self.video_list = QtWidgets.QListWidget()
        self.video_list.currentRowChanged.connect(self._video_changed)
        queue_layout.addWidget(self.video_list, 1)
        splitter.addWidget(self.queue_panel)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        self.video_title = QtWidgets.QLabel("No video")
        self.video_title.setStyleSheet("font-size: 15px; font-weight: 600")
        self.note_label = QtWidgets.QLabel()
        self.note_label.setWordWrap(True)
        content_layout.addWidget(self.video_title)
        content_layout.addWidget(self.note_label)
        self.signal_view = SignalView(editable=True)
        self.signal_view.selection_requested.connect(self._select_times)
        self.signal_view.cursor_requested.connect(self._cursor_changed)
        content_layout.addWidget(self.signal_view, 1)
        splitter.addWidget(content)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.queue_panel.setMinimumWidth(190)
        layout.addWidget(splitter, 1)

        label_actions = QtWidgets.QHBoxLayout()
        review_actions = QtWidgets.QHBoxLayout()
        self.wake_button = QtWidgets.QPushButton("Wake (1)")
        self.nrem_button = QtWidgets.QPushButton("NREM (2)")
        self.rem_button = QtWidgets.QPushButton("REM (3)")
        self.auto_button = QtWidgets.QPushButton("Restore Auto (A)")
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.redo_button = QtWidgets.QPushButton("Redo")
        self.previous_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        self.confirm_button = QtWidgets.QPushButton("Confirm and next")
        self.commit_button = QtWidgets.QPushButton("Commit Session")
        self.wake_button.clicked.connect(lambda: self.apply_stage(WAKE))
        self.nrem_button.clicked.connect(lambda: self.apply_stage(NREM))
        self.rem_button.clicked.connect(lambda: self.apply_stage(REM))
        self.auto_button.clicked.connect(self.restore_auto)
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        self.previous_button.clicked.connect(lambda: self._move_video(-1))
        self.next_button.clicked.connect(lambda: self._move_video(1))
        self.confirm_button.clicked.connect(self.confirm_and_next)
        self.commit_button.clicked.connect(self.commit_session)
        for button in (
            self.wake_button,
            self.nrem_button,
            self.rem_button,
            self.auto_button,
            self.undo_button,
            self.redo_button,
        ):
            label_actions.addWidget(button)
        label_actions.addStretch(1)
        for button in (
            self.previous_button,
            self.next_button,
            self.confirm_button,
            self.commit_button,
        ):
            review_actions.addWidget(button)
        review_actions.addStretch(1)
        layout.addLayout(label_actions)
        layout.addLayout(review_actions)
        self._update_edit_actions()

    def set_sessions(self, sessions: list[ReviewSession]) -> None:
        self.sessions = sessions
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems(session.source.name for session in sessions)
        self.session_combo.blockSignals(False)
        if sessions:
            self.session_combo.setCurrentIndex(0)
            self._session_changed(0)

    def open_video(self, session_index: int, video_index: int) -> None:
        self.session_combo.setCurrentIndex(session_index)
        self.video_list.setCurrentRow(video_index)

    def apply_stage(self, stage: str) -> None:
        if self.session is None or self.selection is None:
            return
        if self.session.apply_stage(self.selection, stage):
            self._refresh_after_edit()

    def restore_auto(self) -> None:
        if self.session is None or self.selection is None:
            return
        if self.session.restore_auto(self.selection):
            self._refresh_after_edit()

    def undo(self) -> None:
        if self.session and self.session.undo():
            self._refresh_after_edit()

    def redo(self) -> None:
        if self.session and self.session.redo():
            self._refresh_after_edit()

    def confirm_and_next(self) -> None:
        if self.session is None or self.video is None:
            return
        self.session.confirm_video(self.video)
        self._refresh_queue()
        self._move_video(1)

    def commit_session(self) -> None:
        if self.session is None:
            return
        unreviewed = sum(value != "reviewed" for value in self.session.video_status.values())
        if unreviewed:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Commit partial review",
                f"{unreviewed} video(s) are not confirmed. Commit the Session draft anyway?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.session.commit()
        self._update_session_status()
        self.committed.emit(self.session_combo.currentIndex())

    def _session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        self.session = self.sessions[index]
        self.signal_view.set_session(self.session, self.session.draft_labels)
        self._refresh_queue()
        self.video_list.setCurrentRow(0 if self.session.source.videos else -1)
        self._video_changed()
        self._update_session_status()

    def _refresh_queue(self) -> None:
        if self.session is None:
            return
        current = self.video_list.currentRow()
        self.video_list.blockSignals(True)
        self.video_list.clear()
        for video in self.session.source.videos:
            status = self.session.video_status[self.session.video_key(video)]
            item = QtWidgets.QListWidgetItem(f"{STATUS_TEXT[status]}  {video.display_name}")
            item.setToolTip(video.comment or "No Note comment")
            self.video_list.addItem(item)
        if self.video_list.count():
            self.video_list.setCurrentRow(max(0, min(current, self.video_list.count() - 1)))
        self.video_list.blockSignals(False)

    def _video_changed(self, _value=0) -> None:
        if self.session is None:
            return
        index = self.video_list.currentRow()
        if index < 0 or index >= len(self.session.source.videos):
            return
        self.video = self.session.source.videos[index]
        self.frame_times = load_video_frame_times(self.video)
        context = self.context_sec.value()
        start = max(0.0, self.video.start_sec - context)
        stop = min(self.session.recording.duration_sec, self.video.stop_sec + context)
        self.signal_view.set_domain(start, stop)
        self.signal_view.set_video(self.video, self.frame_times)
        self.signal_view.set_selection(None)
        self.selection = None
        self.video_title.setText(
            f"{self.video.display_name} · {self.session.source.name} · "
            f"{_format_time(self.video.start_sec)}–{_format_time(self.video.stop_sec)}"
        )
        self.note_label.setText(f"Note: {self.video.comment or 'No additional comment'}")
        self._update_edit_actions()

    def _select_times(self, start: float, stop: float, _final: bool) -> None:
        if self.session is None:
            return
        start = float(np.clip(start, *self.signal_view.domain))
        stop = float(np.clip(stop, *self.signal_view.domain))
        self.selection = self.session.epoch_range(start, stop)
        bounds = self.session.selection_bounds(self.selection) if self.selection else None
        self.signal_view.set_selection(bounds)
        self._update_edit_actions()

    def _cursor_changed(self, time_sec: float) -> None:
        if self.video is None or len(self.frame_times) == 0:
            return
        relative = time_sec - self.video.start_sec
        if relative < 0 or relative > self.frame_times[-1]:
            return
        frame = int(np.clip(np.searchsorted(self.frame_times, relative, side="left"), 0, len(self.frame_times) - 1))
        self.signal_view.cursor_label.setText(
            f"Cursor: {_format_time(time_sec, decimals=True)} · frame {frame} · video {relative:.1f}s"
        )

    def _move_video(self, step: int) -> None:
        if self.video_list.count() == 0:
            return
        target = int(np.clip(self.video_list.currentRow() + step, 0, self.video_list.count() - 1))
        self.video_list.setCurrentRow(target)

    def _toggle_queue(self) -> None:
        visible = self.queue_panel.isHidden()
        self.queue_panel.setVisible(visible)
        self.queue_button.setText("Hide queue" if visible else "Show queue")

    def _refresh_after_edit(self) -> None:
        assert self.session is not None
        self.signal_view.refresh_labels(self.session.draft_labels)
        self._refresh_queue()
        self._update_session_status()
        self._update_edit_actions()

    def _update_session_status(self) -> None:
        if self.session is None:
            return
        reviewed = sum(value == "reviewed" for value in self.session.video_status.values())
        dirty = " · uncommitted draft" if self.session.has_uncommitted_changes else ""
        self.session_status.setText(f"Reviewed {reviewed}/{len(self.session.video_status)}{dirty}")

    def _update_edit_actions(self) -> None:
        has_selection = self.session is not None and self.selection is not None
        for button in (self.wake_button, self.nrem_button, self.rem_button, self.auto_button):
            button.setEnabled(has_selection)
        self.undo_button.setEnabled(bool(self.session and self.session.undo_stack))
        self.redo_button.setEnabled(bool(self.session and self.session.redo_stack))


class GlobalOverviewPage(QtWidgets.QWidget):
    open_local_requested = QtCore.Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.sessions: list[ReviewSession] = []
        self.session: ReviewSession | None = None
        layout = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Session"))
        self.session_combo = QtWidgets.QComboBox()
        self.session_combo.currentIndexChanged.connect(self._session_changed)
        bar.addWidget(self.session_combo)
        bar.addWidget(QtWidgets.QLabel("Video"))
        self.video_combo = QtWidgets.QComboBox()
        bar.addWidget(self.video_combo, 1)
        open_local = QtWidgets.QPushButton("Open in local review")
        open_local.clicked.connect(self._open_local)
        bar.addWidget(open_local)
        self.status = QtWidgets.QLabel()
        bar.addWidget(self.status)
        layout.addLayout(bar)
        self.signal_view = SignalView(editable=False)
        layout.addWidget(self.signal_view, 1)

    def set_sessions(self, sessions: list[ReviewSession]) -> None:
        self.sessions = sessions
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems(session.source.name for session in sessions)
        self.session_combo.blockSignals(False)
        if sessions:
            self.session_combo.setCurrentIndex(0)
            self._session_changed(0)

    def refresh_session(self, index: int) -> None:
        if index == self.session_combo.currentIndex():
            self._session_changed(index)

    def _session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        self.session = self.sessions[index]
        self.video_combo.clear()
        self.video_combo.addItems(video.display_name for video in self.session.source.videos)
        self.signal_view.set_session(self.session, self.session.committed_labels)
        self.signal_view.set_domain(0.0, self.session.recording.duration_sec)
        self.signal_view.set_video(None)
        self.signal_view.show_global_videos(self.session.source.videos)
        draft = " · uncommitted draft hidden" if self.session.has_uncommitted_changes else ""
        self.status.setText(f"Read-only committed view{draft}")

    def _open_local(self) -> None:
        self.open_local_requested.emit(self.session_combo.currentIndex(), self.video_combo.currentIndex())


class SleepStateReviewWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sleep State Staging")
        self.resize(1380, 900)
        self.source: ExperimentSource | None = None
        self.settings = AnalysisSettings()
        self.sessions: list[ReviewSession] = []
        self._thread: QtCore.QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._progress: QtWidgets.QProgressDialog | None = None
        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self) -> None:
        toolbar = QtWidgets.QToolBar("Main")
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        open_action = toolbar.addAction("Open experiment")
        source_action = toolbar.addAction("Source structure")
        params_action = toolbar.addAction("Automatic parameters")
        run_action = toolbar.addAction("Run automatic analysis")
        self.review_action = toolbar.addAction("Review workspace")
        open_action.triggered.connect(self.choose_experiment)
        source_action.triggered.connect(lambda: self.stack.setCurrentWidget(self.source_page))
        params_action.triggered.connect(self.edit_settings)
        run_action.triggered.connect(self.run_analysis)
        self.review_action.triggered.connect(lambda: self.stack.setCurrentWidget(self.work_tabs))
        self.review_action.setEnabled(False)

        self.stack = QtWidgets.QStackedWidget()
        self.source_page = SourcePage()
        self.source_page.browse_requested.connect(self.choose_experiment)
        self.source_page.adjust_requested.connect(self.adjust_source)
        self.source_page.run_requested.connect(self.run_analysis)
        self.stack.addWidget(self.source_page)

        self.work_tabs = QtWidgets.QTabWidget()
        self.local_page = LocalReviewPage()
        self.global_page = GlobalOverviewPage()
        self.local_page.committed.connect(self.global_page.refresh_session)
        self.global_page.open_local_requested.connect(self._open_local)
        self.work_tabs.addTab(self.local_page, "Local video review")
        self.work_tabs.addTab(self.global_page, "Global read-only overview")
        self.stack.addWidget(self.work_tabs)
        self.setCentralWidget(self.stack)
        self.statusBar().showMessage("Choose an experiment directory.")

    def _bind_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence("1"), self, activated=lambda: self._local_action(WAKE))
        QtGui.QShortcut(QtGui.QKeySequence("2"), self, activated=lambda: self._local_action(NREM))
        QtGui.QShortcut(QtGui.QKeySequence("3"), self, activated=lambda: self._local_action(REM))
        QtGui.QShortcut(QtGui.QKeySequence("A"), self, activated=self._restore_auto)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self._undo)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self, activated=self._redo)

    def choose_experiment(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose experiment directory")
        if not path:
            return
        try:
            self.source = scan_experiment(path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.sessions = []
        self.review_action.setEnabled(False)
        self.source_page.set_source(self.source)
        self.stack.setCurrentWidget(self.source_page)
        self.statusBar().showMessage("Verify the detected source structure before analysis.")

    def adjust_source(self) -> None:
        if self.source is None:
            return
        dialog = SourceMappingDialog(self.source, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        note, recordings, tiffs = dialog.values()
        try:
            self.source = scan_experiment(
                self.source.directory,
                note_path=note,
                recording_paths=recordings,
                tiff_directory=tiffs,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.source_page.set_source(self.source)

    def edit_settings(self) -> None:
        dialog = AnalysisSettingsDialog(self.settings, self.source, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            changed = dialog.settings() != self.settings
            self.settings = dialog.settings()
            if changed and self.sessions:
                self.statusBar().showMessage("Automatic parameters changed; rerun analysis before editing.")
                self.review_action.setEnabled(False)
                self.stack.setCurrentWidget(self.source_page)

    def run_analysis(self) -> None:
        if self.source is None or not self.source.is_ready:
            self._show_error("Choose a valid experiment source before analysis.")
            return
        if self.sessions and any(session.has_uncommitted_changes for session in self.sessions):
            answer = QtWidgets.QMessageBox.warning(
                self,
                "Rerun automatic analysis",
                "Rerunning may make current drafts incompatible. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._progress = QtWidgets.QProgressDialog("Starting analysis…", "", 0, 0, self)
        self._progress.setCancelButton(None)
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.show()
        self._thread = QtCore.QThread(self)
        self._worker = AnalysisWorker(self.source, self.settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setLabelText)
        self._worker.finished.connect(self._analysis_finished)
        self._worker.failed.connect(self._analysis_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _analysis_finished(self, sessions: list[ReviewSession]) -> None:
        if self._progress:
            self._progress.close()
        self.sessions = sessions
        self.review_action.setEnabled(True)
        self.local_page.set_sessions(sessions)
        self.global_page.set_sessions(sessions)
        self.stack.setCurrentWidget(self.work_tabs)
        self.work_tabs.setCurrentWidget(self.local_page)
        self.statusBar().showMessage("Automatic analysis complete. Local review is ready.")

    def _analysis_failed(self, details: str) -> None:
        if self._progress:
            self._progress.close()
        self._show_error(details)

    def _open_local(self, session_index: int, video_index: int) -> None:
        self.work_tabs.setCurrentWidget(self.local_page)
        self.local_page.open_video(session_index, video_index)

    def _local_is_active(self) -> bool:
        return self.stack.currentWidget() is self.work_tabs and self.work_tabs.currentWidget() is self.local_page

    def _local_action(self, stage: str) -> None:
        if self._local_is_active() and not _editing_text():
            self.local_page.apply_stage(stage)

    def _restore_auto(self) -> None:
        if self._local_is_active() and not _editing_text():
            self.local_page.restore_auto()

    def _undo(self) -> None:
        if self._local_is_active() and not _editing_text():
            self.local_page.undo()

    def _redo(self) -> None:
        if self._local_is_active() and not _editing_text():
            self.local_page.redo()

    def _show_error(self, text: str) -> None:
        self.statusBar().showMessage(text.splitlines()[-1] if text else "Unknown error")
        QtWidgets.QMessageBox.critical(self, "Sleep State Staging", text)


def _spectrogram_preview(
    eeg: np.ndarray,
    fs: float,
    max_time_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    target_fs = min(float(fs), 100.0)
    down = max(1, int(round(fs / target_fs)))
    signal = resample_poly(np.asarray(eeg, dtype=np.float32), 1, down)
    actual_fs = fs / down
    window = min(len(signal), max(8, int(round(5.0 * actual_fs))))
    if len(signal) < window or window < 8:
        return np.asarray([]), np.asarray([]), np.empty((0, 0)), 0.0, 1.0
    possible = max(1, int((len(signal) - window) / actual_fs) + 1)
    count = min(max_time_bins, possible)
    starts = np.linspace(0, len(signal) - window, count, dtype=np.int64)
    indices = starts[:, None] + np.arange(window, dtype=np.int64)[None, :]
    segments = signal[indices]
    segments = segments - np.mean(segments, axis=1, keepdims=True)
    segments *= np.hanning(window).astype(np.float32)
    nfft = max(512, 1 << int(np.ceil(np.log2(window))))
    power = np.abs(np.fft.rfft(segments, n=nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / actual_fs)
    keep = freqs <= 30.0
    power_db = 10.0 * np.log10(power[:, keep].T + 1e-12)
    times = (starts + window / 2.0) / actual_fs
    low = float(np.percentile(power_db, 5))
    return freqs[keep], times, power_db.astype(np.float32), low, low + 30.0


def _visible_envelope(
    signal: np.ndarray,
    fs: float,
    start_sec: float,
    stop_sec: float,
    max_points: int = 120_000,
) -> tuple[np.ndarray, np.ndarray]:
    first = max(0, int(np.floor(start_sec * fs)))
    last = min(len(signal), int(np.ceil(stop_sec * fs)))
    values = np.asarray(signal[first:last])
    if len(values) <= max_points:
        times = np.arange(first, last, dtype=np.float64) / fs
        return times, values
    block = max(1, int(np.ceil(len(values) / (max_points // 2))))
    usable = len(values) // block * block
    reshaped = values[:usable].reshape(-1, block)
    lows = reshaped.min(axis=1)
    highs = reshaped.max(axis=1)
    centers = (first + np.arange(len(lows)) * block + block / 2.0) / fs
    times = np.repeat(centers, 2)
    envelope = np.column_stack((lows, highs)).reshape(-1)
    return times, envelope


def _note_frame_numbers(comment: str) -> list[int]:
    values = []
    for match in re.finditer(r"(?<![\d.])(\d+)\s*(?=(?:To|TO|to|[-–]))", comment):
        values.append(int(match.group(1)))
    return values


def _format_time(value: float | None, decimals: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    second_text = f"{seconds:04.1f}" if decimals else f"{int(seconds):02d}"
    return f"{hours}:{minutes:02d}:{second_text}" if hours else f"{minutes:02d}:{second_text}"


def _view_width_from_slider(domain_width: float, value: int) -> float:
    domain_width = max(0.0, float(domain_width))
    minimum_width = min(1.0, domain_width)
    if domain_width <= minimum_width:
        return domain_width
    fraction = float(np.clip(value, 0, _NAVIGATION_STEPS)) / _NAVIGATION_STEPS
    return domain_width * (minimum_width / domain_width) ** fraction


def _slider_from_view_width(domain_width: float, view_width: float) -> int:
    domain_width = max(0.0, float(domain_width))
    minimum_width = min(1.0, domain_width)
    if domain_width <= minimum_width:
        return 0
    view_width = float(np.clip(view_width, minimum_width, domain_width))
    fraction = np.log(view_width / domain_width) / np.log(minimum_width / domain_width)
    return round(_NAVIGATION_STEPS * fraction)


def _double_spin(low: float, high: float, value: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    return widget


def _editing_text() -> bool:
    widget = QtWidgets.QApplication.focusWidget()
    return isinstance(
        widget,
        (
            QtWidgets.QLineEdit,
            QtWidgets.QPlainTextEdit,
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
