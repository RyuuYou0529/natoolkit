from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from napari.layers import Image, Labels
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks


@dataclass
class MovieState:
    start: int = 0
    stop: int = 0
    layer_name: str = ""
    source_data: np.ndarray | None = None
    spatial_reference: str = ""
    temporal_reference: str = ""
    traces: dict[int, np.ndarray] = field(default_factory=dict)
    visible_rois: set[int] = field(default_factory=set)
    spikes: dict[int, list[dict[str, float]]] = field(default_factory=dict)


class ActivityPlot(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(7, 3))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.time_line = None
        self.current_frame = 0
        self.x_limits: tuple[float, float] | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        zoom_layout = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(1)
        self.zoom_slider.valueChanged.connect(lambda _=0: self.apply_zoom())
        zoom_layout.addWidget(QLabel("Zoom"))
        zoom_layout.addWidget(self.zoom_slider)
        layout.addLayout(zoom_layout)

        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.draw_empty()

    def draw_empty(self, message: str = "No traced activities") -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha="center", va="center", transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.time_line = None
        self.x_limits = None
        self.canvas.draw_idle()

    def draw_traces(self, states: list[MovieState], mode: str, normalize) -> None:
        self.ax.clear()
        self.ax.set_axis_on()
        shown = [
            (state, roi)
            for state in states
            for roi in sorted(state.visible_rois)
            if roi in state.traces
        ]
        if not shown:
            self.draw_empty("No visible traces")
            return

        for state, roi in shown:
            trace = normalize(state.traces[roi])
            frames = np.arange(len(trace))
            self.ax.plot(frames, trace, lw=1.2, label=f"{state.layer_name} / ROI {roi}")
            spike_frames = [int(spike["frame"]) for spike in state.spikes.get(roi, [])]
            spike_frames = [frame for frame in spike_frames if frame < len(trace)]
            if spike_frames:
                self.ax.scatter(spike_frames, trace[spike_frames], c="red", s=24, marker="v", zorder=5)

        stops = [len(state.traces[roi]) - 1 for state, roi in shown]
        self.ax.set_title(f"Activities ({mode})")
        self.ax.set_xlabel("Cropped frame")
        self.ax.set_ylabel(mode)
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, alpha=0.2)
        self.time_line = self.ax.axvline(self.current_frame, color="black", lw=1.0, alpha=0.8)
        self.x_limits = (0.0, float(max(max(stops), 1)))
        self.apply_zoom(redraw=False)
        self.figure.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.24)
        self.canvas.draw_idle()

    def set_frame(self, frame: int) -> None:
        self.current_frame = frame
        if self.time_line is not None:
            self.time_line.set_xdata([frame, frame])
            self.apply_zoom(redraw=False)
            self.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        step = 8 if event.button == "up" else -8
        value = min(max(self.zoom_slider.value() + step, self.zoom_slider.minimum()), self.zoom_slider.maximum())
        self.zoom_slider.setValue(value)

    def apply_zoom(self, redraw: bool = True) -> None:
        if self.x_limits is None:
            return
        xmin, xmax = self.x_limits
        width = xmax - xmin + 1
        zoom = self.zoom_slider.value()
        if zoom == 1:
            self.ax.set_xlim(xmin, xmax)
        else:
            half = width / zoom / 2
            center = min(max(self.current_frame, xmin + half), xmax - half)
            self.ax.set_xlim(center - half, center + half)
        if redraw:
            self.canvas.draw_idle()


class SimpleTracerWidget(QWidget):
    roi_layer_name = "Shared ROIs"

    def __init__(self, viewer: napari.Viewer, plot: ActivityPlot) -> None:
        super().__init__()
        self.viewer = viewer
        self.movie_states: dict[int, MovieState] = {}
        self.roi_layer: Labels | None = None
        self.syncing = False
        self.spatial_ref_name: str | None = None
        self.spatial_ref_shape: tuple[int, int] | None = None
        self.temporal_ref_name: str | None = None
        self.temporal_ref_length: int | None = None

        self.plot = plot
        self._build_ui()
        self._connect_events()
        self.sync_from_selection()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        self.setMinimumWidth(420)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setMinimumWidth(390)
        controls_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        root_layout.addWidget(scroll)

        self.active_label = QLabel("Active movie: none")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.active_label)

        movie_group = QGroupBox("Movie")
        movie_layout = QVBoxLayout(movie_group)
        import_button = QPushButton("Import Movie")
        import_button.clicked.connect(lambda _=False: self.import_movies())
        movie_layout.addWidget(import_button)
        self.movie_info_label = QLabel("No selected movie.")
        self.movie_info_label.setWordWrap(True)
        movie_layout.addWidget(self.movie_info_label)

        time_layout = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.stop_spin = QSpinBox()
        self.start_spin.setMaximum(10**9)
        self.stop_spin.setMaximum(10**9)
        time_layout.addWidget(QLabel("Start"))
        time_layout.addWidget(self.start_spin)
        time_layout.addWidget(QLabel("Stop"))
        time_layout.addWidget(self.stop_spin)
        movie_layout.addLayout(time_layout)
        full_range_button = QPushButton("Use Full Range")
        full_range_button.clicked.connect(lambda _=False: self.use_full_range())
        movie_layout.addWidget(full_range_button)

        spatial_ref_button = QPushButton("Set Spatial Center Pad Ref")
        pad_button = QPushButton("Apply Pad")
        temporal_ref_button = QPushButton("Set Temporal Center Clip Ref")
        clip_button = QPushButton("Apply Clip")
        spatial_ref_button.clicked.connect(lambda _=False: self.set_spatial_reference())
        pad_button.clicked.connect(lambda _=False: self.pad_selected_movie())
        temporal_ref_button.clicked.connect(lambda _=False: self.set_temporal_reference())
        clip_button.clicked.connect(lambda _=False: self.clip_selected_movie())
        spatial_row = QHBoxLayout()
        spatial_row.addWidget(spatial_ref_button, stretch=2)
        spatial_row.addWidget(pad_button, stretch=1)
        temporal_row = QHBoxLayout()
        temporal_row.addWidget(temporal_ref_button, stretch=2)
        temporal_row.addWidget(clip_button, stretch=1)
        movie_layout.addLayout(spatial_row)
        movie_layout.addLayout(temporal_row)

        self.reference_info_label = QLabel("")
        self.reference_info_label.setWordWrap(True)
        movie_layout.addWidget(self.reference_info_label)
        controls_layout.addWidget(movie_group)

        roi_group = QGroupBox("ROIs")
        roi_layout = QVBoxLayout(roi_group)
        create_roi_button = QPushButton("Create Shared Labels")
        load_roi_button = QPushButton("Load ROI Labels")
        save_roi_button = QPushButton("Export ROI Labels")
        create_roi_button.clicked.connect(lambda _=False: self.create_roi_layer())
        load_roi_button.clicked.connect(lambda _=False: self.load_roi_labels())
        save_roi_button.clicked.connect(lambda _=False: self.export_roi_labels())
        roi_row_1 = QHBoxLayout()
        roi_row_1.addWidget(create_roi_button)
        roi_row_1.addWidget(load_roi_button)
        roi_layout.addLayout(roi_row_1)
        roi_layout.addWidget(save_roi_button)
        controls_layout.addWidget(roi_group)

        trace_group = QGroupBox("Activities")
        trace_layout = QVBoxLayout(trace_group)
        norm_layout = QHBoxLayout()
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Raw", "dF/F0", "-dF/F0", "SNR(-dF/F0)", "FRAME_SNR", "Z-score", "Min-max"])
        self.f0_spin = QSpinBox()
        self.f0_spin.setRange(0, 100)
        self.f0_spin.setValue(10)
        self.f0_spin.setSuffix("%")
        self.norm_combo.currentTextChanged.connect(lambda _="": self.redraw_plot())
        self.f0_spin.valueChanged.connect(lambda _=0: self.redraw_plot())
        norm_layout.addWidget(QLabel("Normalize"))
        norm_layout.addWidget(self.norm_combo)
        norm_layout.addWidget(QLabel("F0"))
        norm_layout.addWidget(self.f0_spin)
        extract_button = QPushButton("Extract Activities")
        export_button = QPushButton("Export Activities")
        show_all_button = QPushButton("Show All Traces")
        hide_all_button = QPushButton("Hide All Traces")
        extract_button.clicked.connect(lambda _=False: self.extract_active_movie())
        export_button.clicked.connect(lambda _=False: self.export_activities())
        show_all_button.clicked.connect(lambda _=False: self.set_all_roi_visibility(True))
        hide_all_button.clicked.connect(lambda _=False: self.set_all_roi_visibility(False))
        trace_layout.addLayout(norm_layout)
        activity_row_1 = QHBoxLayout()
        activity_row_1.addWidget(extract_button)
        activity_row_1.addWidget(export_button)
        activity_row_2 = QHBoxLayout()
        activity_row_2.addWidget(show_all_button)
        activity_row_2.addWidget(hide_all_button)
        trace_layout.addLayout(activity_row_1)
        trace_layout.addLayout(activity_row_2)

        self.roi_check_widget = QWidget()
        self.roi_check_layout = QVBoxLayout(self.roi_check_widget)
        self.roi_scroll = QScrollArea()
        self.roi_scroll.setMinimumHeight(320)
        self.roi_scroll.setWidgetResizable(True)
        self.roi_scroll.setWidget(self.roi_check_widget)
        trace_layout.addWidget(self.roi_scroll)
        controls_layout.addWidget(trace_group)

        spike_group = QGroupBox("FRAME_Spike")
        spike_layout = QVBoxLayout(spike_group)
        self.spike_fps_spin = QSpinBox()
        self.spike_fps_spin.setRange(1, 100000)
        self.spike_fps_spin.setValue(1000)
        self.spike_fps_spin.setSuffix(" Hz")
        self.spike_baseline_spin = QSpinBox()
        self.spike_baseline_spin.setRange(1, 10000)
        self.spike_baseline_spin.setValue(500)
        self.spike_baseline_spin.setSuffix(" ms")
        self.spike_lowpass_spin = QDoubleSpinBox()
        self.spike_lowpass_spin.setRange(0.1, 10000.0)
        self.spike_lowpass_spin.setValue(250.0)
        self.spike_lowpass_spin.setSuffix(" Hz")
        self.spike_highpass_spin = QDoubleSpinBox()
        self.spike_highpass_spin.setRange(0.1, 10000.0)
        self.spike_highpass_spin.setValue(20.0)
        self.spike_highpass_spin.setSuffix(" Hz")
        self.spike_isi_spin = QDoubleSpinBox()
        self.spike_isi_spin.setRange(0.1, 1000.0)
        self.spike_isi_spin.setValue(3.0)
        self.spike_isi_spin.setSuffix(" ms")
        self.spike_threshold_spin = QDoubleSpinBox()
        self.spike_threshold_spin.setRange(0.1, 1000.0)
        self.spike_threshold_spin.setValue(4.0)
        self.spike_threshold_spin.setSingleStep(0.5)

        spike_row_1 = QHBoxLayout()
        spike_row_1.addWidget(QLabel("FPS"))
        spike_row_1.addWidget(self.spike_fps_spin)
        spike_row_1.addWidget(QLabel("SNR"))
        spike_row_1.addWidget(self.spike_threshold_spin)
        spike_row_2 = QHBoxLayout()
        spike_row_2.addWidget(QLabel("Median"))
        spike_row_2.addWidget(self.spike_baseline_spin)
        spike_row_2.addWidget(QLabel("ISI"))
        spike_row_2.addWidget(self.spike_isi_spin)
        spike_row_3 = QHBoxLayout()
        spike_row_3.addWidget(QLabel("Low"))
        spike_row_3.addWidget(self.spike_lowpass_spin)
        spike_row_3.addWidget(QLabel("High"))
        spike_row_3.addWidget(self.spike_highpass_spin)

        detect_button = QPushButton("Detect FRAME_Spike")
        export_spike_button = QPushButton("Export Spikes")
        detect_button.clicked.connect(lambda _=False: self.detect_frame_spikes())
        export_spike_button.clicked.connect(lambda _=False: self.export_spikes())
        spike_button_row = QHBoxLayout()
        spike_button_row.addWidget(detect_button)
        spike_button_row.addWidget(export_spike_button)
        spike_layout.addLayout(spike_row_1)
        spike_layout.addLayout(spike_row_2)
        spike_layout.addLayout(spike_row_3)
        spike_layout.addLayout(spike_button_row)
        controls_layout.addWidget(spike_group)

        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch()

    def _connect_events(self) -> None:
        self.viewer.layers.selection.events.changed.connect(self.sync_from_selection)
        self.viewer.dims.events.current_step.connect(self.sync_time_line)
        self.start_spin.valueChanged.connect(self.save_time_roi)
        self.stop_spin.valueChanged.connect(self.save_time_roi)

    def active_image_layer(self) -> Image | None:
        layer = self.viewer.layers.selection.active
        return layer if isinstance(layer, Image) else None

    def state_for(self, layer: Image) -> MovieState:
        key = id(layer)
        state = self.movie_states.setdefault(key, MovieState())
        if state.source_data is None:
            state.source_data = layer.data
            state.stop = int(layer.data.shape[0])
        state.layer_name = layer.name
        return state

    def update_movie_info(self, layer: Image | None) -> None:
        if layer is None:
            self.movie_info_label.setText("No selected movie.")
        else:
            state = self.state_for(layer)
            self.movie_info_label.setText(
                f"Name: {layer.name}\n"
                f"Source shape: {tuple(state.source_data.shape)}\n"
                f"Displayed shape: {tuple(layer.data.shape)}\n"
                f"Dtype: {state.source_data.dtype}"
            )

        spatial_ref = "None"
        if self.spatial_ref_name and self.spatial_ref_shape:
            spatial_ref = f"{self.spatial_ref_name} ({self.spatial_ref_shape[0]} x {self.spatial_ref_shape[1]})"
        temporal_ref = "None"
        if self.temporal_ref_name and self.temporal_ref_length:
            temporal_ref = f"{self.temporal_ref_name} ({self.temporal_ref_length} frames)"

        usage = "Selected uses: none"
        if layer is not None:
            state = self.state_for(layer)
            spatial = state.spatial_reference or "not padded"
            temporal = state.temporal_reference or "manual/full range"
            usage = f"Selected uses:\nSpatial: {spatial}\nTemporal: {temporal}"

        self.reference_info_label.setText(
            f"Spatial ref: {spatial_ref}\n"
            f"Temporal ref: {temporal_ref}\n"
            f"{usage}"
        )

    def sync_from_selection(self, event=None) -> None:
        layer = self.active_image_layer()
        self.syncing = True
        if layer is None:
            self.active_label.setText("Active movie: none")
            self.start_spin.setEnabled(False)
            self.stop_spin.setEnabled(False)
            self.rebuild_roi_controls()
            self.redraw_plot()
        else:
            state = self.state_for(layer)
            frame_count = int(state.source_data.shape[0])
            if state.stop == 0 or state.stop > frame_count:
                state.stop = frame_count
            self.active_label.setText(f"Active movie: {layer.name}")
            self.start_spin.setEnabled(True)
            self.stop_spin.setEnabled(True)
            self.start_spin.setRange(0, frame_count)
            self.stop_spin.setRange(0, frame_count)
            self.start_spin.setValue(state.start)
            self.stop_spin.setValue(state.stop)
            self.rebuild_roi_controls()
            self.redraw_plot()
        self.update_movie_info(layer)
        self.syncing = False
        self.sync_time_line()

    def save_time_roi(self) -> None:
        if self.syncing:
            return
        layer = self.active_image_layer()
        if layer is None:
            return
        state = self.state_for(layer)
        new_range = (self.start_spin.value(), self.stop_spin.value())
        if new_range[0] >= new_range[1]:
            self.set_status("TimeROI needs Start smaller than Stop.")
            return
        if new_range != (state.start, state.stop):
            previous_visible = set(state.visible_rois)
            should_retrace = bool(state.traces)
            state.start, state.stop = new_range
            state.temporal_reference = ""
            self.apply_time_roi(layer, state)
            if should_retrace:
                if self.trace_movie(layer, previous_visible) is None:
                    state.traces.clear()
                    state.visible_rois.clear()
                    state.spikes.clear()
            self.rebuild_roi_controls()
            self.redraw_plot()
            self.update_movie_info(layer)

    def use_full_range(self) -> None:
        layer = self.active_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        state = self.state_for(layer)
        self.start_spin.setValue(0)
        self.stop_spin.setValue(int(state.source_data.shape[0]))

    def apply_time_roi(self, layer: Image, state: MovieState) -> None:
        layer.data = state.source_data[state.start : state.stop]
        self.viewer.dims.set_current_step(0, 0)
        self.update_movie_info(layer)

    def set_spatial_reference(self) -> None:
        layer = self.active_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        state = self.state_for(layer)
        self.spatial_ref_name = layer.name
        self.spatial_ref_shape = tuple(state.source_data.shape[-2:])
        self.update_movie_info(layer)
        self.set_status(f"Spatial pad reference set to {layer.name}.")

    def set_temporal_reference(self) -> None:
        layer = self.active_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        self.temporal_ref_name = layer.name
        self.temporal_ref_length = int(layer.data.shape[0])
        self.update_movie_info(layer)
        self.set_status(f"Temporal clip reference set to {layer.name}.")

    def pad_selected_movie(self) -> None:
        layer = self.active_image_layer()
        if layer is None or self.spatial_ref_shape is None:
            self.set_status("Select a movie and set a spatial reference first.")
            return
        state = self.state_for(layer)
        source = state.source_data
        target_y, target_x = self.spatial_ref_shape
        source_y, source_x = source.shape[-2:]
        if source_y > target_y or source_x > target_x:
            self.set_status("Selected movie is larger than the spatial reference.")
            return

        previous_visible = set(state.visible_rois)
        should_retrace = bool(state.traces)
        pad_y = target_y - source_y
        pad_x = target_x - source_x
        pad_width = [(0, 0)] * source.ndim
        pad_width[-2] = (pad_y // 2, pad_y - pad_y // 2)
        pad_width[-1] = (pad_x // 2, pad_x - pad_x // 2)
        state.source_data = np.pad(source, pad_width, mode="constant")
        state.spatial_reference = self.spatial_ref_name or ""
        self.apply_time_roi(layer, state)
        self.retrace_after_change(layer, previous_visible, should_retrace)
        self.set_status(f"Padded {layer.name} to {target_y} x {target_x}.")

    def clip_selected_movie(self) -> None:
        layer = self.active_image_layer()
        if layer is None or self.temporal_ref_length is None:
            self.set_status("Select a movie and set a temporal reference first.")
            return
        state = self.state_for(layer)
        length = int(state.source_data.shape[0])
        if length < self.temporal_ref_length:
            self.set_status("Selected movie is shorter than the temporal reference.")
            return

        previous_visible = set(state.visible_rois)
        should_retrace = bool(state.traces)
        diff = length - self.temporal_ref_length
        state.start = diff // 2
        state.stop = length - (diff - state.start)
        state.temporal_reference = self.temporal_ref_name or ""
        self.apply_time_roi(layer, state)
        self.syncing = True
        self.start_spin.setValue(state.start)
        self.stop_spin.setValue(state.stop)
        self.syncing = False
        self.retrace_after_change(layer, previous_visible, should_retrace)
        self.set_status(f"Center-clipped {layer.name} to {self.temporal_ref_length} frames.")

    def retrace_after_change(self, layer: Image, previous_visible: set[int], should_retrace: bool) -> None:
        state = self.state_for(layer)
        if should_retrace and self.trace_movie(layer, previous_visible) is None:
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()
        self.rebuild_roi_controls()
        self.redraw_plot()
        self.update_movie_info(layer)

    def import_movies(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Open movie")
        if not paths:
            return
        last_layer = None
        for file_name in paths:
            path = Path(file_name)
            if path.suffix.lower() == ".npy":
                last_layer = self.viewer.add_image(np.load(path, mmap_mode="r"), name=path.stem)
            else:
                opened = self.viewer.open(str(path), layer_type="image")
                last_layer = opened[0]
                last_layer.name = path.stem
            self.state_for(last_layer)
        self.viewer.layers.selection.active = last_layer
        self.sync_from_selection()

    def current_roi_layer(self) -> Labels | None:
        if self.roi_layer is not None:
            for layer in self.viewer.layers:
                if layer is self.roi_layer:
                    return layer
        for layer in self.viewer.layers:
            if isinstance(layer, Labels) and layer.name == self.roi_layer_name:
                self.roi_layer = layer
                return layer
        return None

    def create_roi_layer(self) -> None:
        shape = self.reference_image_shape()
        if shape is None:
            self.set_status("Import or select a movie before creating ROIs.")
            return
        layer = self.current_roi_layer()
        if layer is None:
            layer = self.viewer.add_labels(np.zeros(shape, dtype=np.uint16), name=self.roi_layer_name)
            self.roi_layer = layer
        self.viewer.layers.selection.active = layer
        self.set_status("Draw labels with nonzero IDs. Re-extract after editing ROIs.")

    def reference_image_shape(self) -> tuple[int, int] | None:
        layer = self.active_image_layer()
        if layer is not None:
            return tuple(layer.data.shape[-2:])
        for candidate in self.viewer.layers:
            if isinstance(candidate, Image):
                return tuple(candidate.data.shape[-2:])
        return None

    def load_roi_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI labels", filter="NumPy labels (*.npy)")
        if not path:
            return
        labels = np.load(path)
        layer = self.current_roi_layer()
        if layer is None:
            self.roi_layer = self.viewer.add_labels(labels, name=self.roi_layer_name)
        else:
            layer.data = labels
        self.set_status("ROI labels loaded. Re-extract activities for updated ROIs.")

    def export_roi_labels(self) -> None:
        layer = self.current_roi_layer()
        if layer is None:
            self.set_status("Create or load ROI labels first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ROI labels", filter="NumPy labels (*.npy)")
        if not path:
            return
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, np.asarray(layer.data))
        self.set_status(f"ROI labels exported to {path}.")

    def extract_active_movie(self) -> None:
        layer = self.active_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        count = self.trace_movie(layer)
        if count is not None:
            self.rebuild_roi_controls()
            self.redraw_plot()
            self.set_status(f"Extracted {count} ROI traces from {layer.name}.")

    def trace_movie(self, layer: Image, visible_rois: set[int] | None = None) -> int | None:
        roi_layer = self.current_roi_layer()
        if roi_layer is None:
            self.set_status("Create or load ROI labels first.")
            return None

        state = self.state_for(layer)
        labels = np.asarray(roi_layer.data)
        if state.start >= state.stop:
            self.set_status("TimeROI is empty. Set Start smaller than Stop.")
            return None
        if labels.shape != tuple(layer.data.shape[-2:]):
            self.set_status("ROI labels must match the movie Y/X shape.")
            return None
        movie = np.asarray(layer.data)
        roi_ids = np.unique(labels)
        roi_ids = roi_ids[roi_ids > 0]
        flat_movie = movie.reshape(movie.shape[0], -1)
        flat_labels = labels.reshape(-1)

        state.traces = {
            int(roi_id): flat_movie[:, flat_labels == roi_id].mean(axis=1)
            for roi_id in roi_ids
        }
        state.spikes.clear()
        state.visible_rois = set(state.traces) if visible_rois is None else set(state.traces) & visible_rois
        if not state.visible_rois:
            state.visible_rois = set(state.traces)
        return len(state.traces)

    def detect_frame_spikes(self) -> None:
        layer = self.active_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        state = self.state_for(layer)
        if not state.traces:
            self.set_status("Extract activities before detecting spikes.")
            return
        nyquist = self.spike_fps_spin.value() / 2
        if self.spike_lowpass_spin.value() >= nyquist or self.spike_highpass_spin.value() >= nyquist:
            self.set_status("FRAME_Spike filter cutoffs must be below Nyquist.")
            return

        state.spikes = {
            roi: self.frame_spikes_for_trace(trace)
            for roi, trace in state.traces.items()
        }
        count = sum(len(spikes) for spikes in state.spikes.values())
        self.redraw_plot()
        self.set_status(f"Detected {count} FRAME_Spike events in {layer.name}.")

    def frame_spikes_for_trace(self, trace: np.ndarray) -> list[dict[str, float]]:
        processed = self.frame_processed_trace(trace)
        snr_trace = processed["snr_trace"]
        noise = float(processed["noise"])
        distance = max(1, int(round(self.spike_isi_spin.value() * self.spike_fps_spin.value() / 1000)))
        peaks, _ = find_peaks(
            snr_trace,
            height=self.spike_threshold_spin.value(),
            distance=distance,
        )
        return [
            {
                "frame": int(frame),
                "peak": float(processed["spike_trace"][frame]),
                "noise": noise,
                "snr": float(snr_trace[frame]),
            }
            for frame in peaks
        ]

    def frame_processed_trace(self, trace: np.ndarray) -> dict[str, np.ndarray | float]:
        fps = self.spike_fps_spin.value()
        nyquist = fps / 2
        baseline_frames = max(1, int(round(self.spike_baseline_spin.value() * fps / 1000)))
        lowpass_hz = self.spike_lowpass_spin.value()
        highpass_hz = self.spike_highpass_spin.value()
        if lowpass_hz >= nyquist or highpass_hz >= nyquist:
            return {
                "baseline": np.zeros_like(trace, dtype=float),
                "detrended": np.zeros_like(trace, dtype=float),
                "suprathreshold": np.zeros_like(trace, dtype=float),
                "spike_trace": np.zeros_like(trace, dtype=float),
                "noise": 0.0,
                "snr_trace": np.zeros_like(trace, dtype=float),
            }

        baseline = median_filter(trace, size=baseline_frames, mode="nearest")
        detrended = np.asarray(trace, dtype=float) - baseline
        b, a = butter(5, lowpass_hz / nyquist, btype="low")
        suprathreshold = filtfilt(b, a, detrended)
        spike_trace = -suprathreshold
        b, a = butter(5, highpass_hz / nyquist, btype="high")
        noise = float(np.std(filtfilt(b, a, spike_trace)))
        snr_trace = spike_trace / noise if noise else np.zeros_like(spike_trace)
        return {
            "baseline": baseline,
            "detrended": detrended,
            "suprathreshold": suprathreshold,
            "spike_trace": spike_trace,
            "noise": noise,
            "snr_trace": snr_trace,
        }

    def export_spikes(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export FRAME_Spike events", filter="CSV (*.csv)")
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"
        fps = self.spike_fps_spin.value()
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["movie", "roi", "frame", "source_frame", "time_ms", "source_time_ms", "peak", "noise", "snr"])
            for state in self.traced_states():
                for roi, spikes in sorted(state.spikes.items()):
                    for spike in spikes:
                        frame = int(spike["frame"])
                        source_frame = state.start + frame
                        writer.writerow(
                            [
                                state.layer_name,
                                roi,
                                frame,
                                source_frame,
                                frame / fps * 1000,
                                source_frame / fps * 1000,
                                spike["peak"],
                                spike["noise"],
                                spike["snr"],
                            ]
                        )
        self.set_status(f"FRAME_Spike events exported to {path}.")

    def export_activities(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export activities", filter="CSV (*.csv)")
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"
        mode = self.normalization_name()
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["movie", "roi", "frame", "source_frame", "mean_intensity", "normalization", "normalized"])
            for state in self.traced_states():
                for roi, trace in sorted(state.traces.items()):
                    normalized = self.normalize_trace(trace)
                    for offset, value in enumerate(trace):
                        writer.writerow(
                            [
                                state.layer_name,
                                roi,
                                offset,
                                state.start + offset,
                                float(value),
                                mode,
                                float(normalized[offset]),
                            ]
                        )
        self.set_status(f"Activities exported to {path}.")

    def normalization_name(self) -> str:
        mode = self.norm_combo.currentText()
        if mode in {"dF/F0", "-dF/F0", "SNR(-dF/F0)"}:
            return f"{mode} F0={self.f0_spin.value()}%"
        return mode

    def normalize_trace(self, trace: np.ndarray) -> np.ndarray:
        mode = self.norm_combo.currentText()
        trace = np.asarray(trace, dtype=float)
        if mode == "FRAME_SNR":
            return self.frame_processed_trace(trace)["snr_trace"]
        if mode in {"dF/F0", "-dF/F0", "SNR(-dF/F0)"}:
            f0 = np.percentile(trace, self.f0_spin.value())
            normalized = (trace - f0) / f0 if f0 else np.zeros_like(trace)
            negative = -normalized
            if mode == "-dF/F0":
                return negative
            if mode == "SNR(-dF/F0)":
                baseline = np.median(negative)
                noise = 1.4826 * np.median(np.abs(negative - baseline))
                return (negative - baseline) / noise if noise else np.zeros_like(trace)
            return normalized
        if mode == "Z-score":
            std = trace.std()
            return (trace - trace.mean()) / std if std else np.zeros_like(trace)
        if mode == "Min-max":
            span = trace.max() - trace.min()
            return (trace - trace.min()) / span if span else np.zeros_like(trace)
        return trace

    def traced_states(self) -> list[MovieState]:
        states = []
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and id(layer) in self.movie_states:
                state = self.state_for(layer)
                if state.traces:
                    states.append(state)
        return states

    def rebuild_roi_controls(self) -> None:
        while self.roi_check_layout.count():
            item = self.roi_check_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        states = self.traced_states()
        if not states:
            self.roi_check_layout.addWidget(QLabel("No extracted traces"))
            self.roi_check_layout.addStretch()
            return

        for state in states:
            self.roi_check_layout.addWidget(QLabel(state.layer_name))
            for roi in sorted(state.traces):
                checkbox = QCheckBox(f"ROI {roi}")
                checkbox.setChecked(roi in state.visible_rois)
                checkbox.toggled.connect(partial(self.set_roi_visibility, state, roi))
                self.roi_check_layout.addWidget(checkbox)
        self.roi_check_layout.addStretch()

    def set_roi_visibility(self, state: MovieState, roi: int, checked: bool) -> None:
        if checked:
            state.visible_rois.add(roi)
        else:
            state.visible_rois.discard(roi)
        self.redraw_plot()

    def set_all_roi_visibility(self, visible: bool) -> None:
        for state in self.traced_states():
            state.visible_rois = set(state.traces) if visible else set()
        self.rebuild_roi_controls()
        self.redraw_plot()

    def redraw_plot(self) -> None:
        states = self.traced_states()
        if not states:
            self.plot.draw_empty("No traced activities")
        else:
            self.plot.draw_traces(states, self.normalization_name(), self.normalize_trace)
        self.sync_time_line()

    def sync_time_line(self, event=None) -> None:
        current_step = self.viewer.dims.current_step
        if current_step:
            self.plot.set_frame(int(current_step[0]))

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


def clear_viewer_workspace(viewer: napari.Viewer) -> None:
    viewer.layers.clear()
    viewer.window.remove_dock_widget("all")


class ClearWorkspaceDockWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        clear_viewer_workspace(napari_viewer)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Workspace cleared."))
        QTimer.singleShot(0, lambda: self._close_self(napari_viewer))

    def _close_self(self, napari_viewer: napari.Viewer) -> None:
        try:
            napari_viewer.window.remove_dock_widget(self)
        except LookupError:
            pass


def main() -> None:
    viewer = napari.Viewer()
    widget = activity_tracer_widget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Activity Tracer")
    napari.run()


class ActivityTracerDockWidget(SimpleTracerWidget):
    def __init__(self, napari_viewer: napari.Viewer) -> None:
        clear_viewer_workspace(napari_viewer)
        plot = ActivityPlot()
        super().__init__(napari_viewer, plot)
        napari_viewer.window.add_dock_widget(plot, area="bottom", name="Activities")


def activity_tracer_widget(viewer: napari.Viewer) -> SimpleTracerWidget:
    return ActivityTracerDockWidget(viewer)


if __name__ == "__main__":
    main()
