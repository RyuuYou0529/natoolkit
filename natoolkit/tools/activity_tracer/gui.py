from __future__ import annotations

import csv
from collections.abc import Callable
from functools import partial
from pathlib import Path

import dask.array as da
import napari
import numpy as np
from napari.layers import Image, Labels, Points
from napari.qt.threading import create_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .movie_manager import ManagedMovie, MovieManager
from .models import MovieState, ROISet
from .plot import ActivityPlot
from .processing import (
    FrameSettings,
    find_frame_spikes,
    normalize_trace as normalize_activity_trace,
    process_frame_trace,
)
from .roi import ROILabels, roi_centers, roi_colormap, roi_ids
from .suite2p_adapter import (
    MovieInput,
    ROIDetectionResult,
    Suite2PSession,
    run_motion_correction as execute_motion_correction,
    run_roi_detection as execute_roi_detection,
)
from .suite2p_dialogs import MotionCorrectionDialog, ROIDetectionDialog


class ROIManagerWidget(QWidget):
    def __init__(self, select_roi: Callable[[int], None]) -> None:
        super().__init__()
        self.select_roi = select_roi
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._select_item)
        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget)
        self.setMinimumSize(180, 280)

    def set_rois(
        self,
        ids: list[int],
        colors: dict[int, np.ndarray],
        selected: int,
    ) -> None:
        self.list_widget.clear()
        for roi_id in ids:
            color = colors[roi_id]
            pixmap = QPixmap(14, 14)
            pixmap.fill(QColor.fromRgbF(*(float(value) for value in color[:3])))
            item = QListWidgetItem(QIcon(pixmap), f"ROI {roi_id}")
            item.setData(Qt.ItemDataRole.UserRole, roi_id)
            self.list_widget.addItem(item)
            if roi_id == selected:
                self.list_widget.setCurrentItem(item)

    def _select_item(self, item: QListWidgetItem) -> None:
        self.select_roi(int(item.data(Qt.ItemDataRole.UserRole)))


class SimpleTracerWidget(QWidget):
    roi_layer_name = "ROIs"

    def __init__(self, viewer: napari.Viewer, plot: ActivityPlot) -> None:
        super().__init__()
        self.viewer = viewer
        self.movie_manager = MovieManager()
        self.movie_states = self.movie_manager.states
        self.roi_layer: Labels | None = None
        self.roi_ids_layer: Points | None = None
        self.shared_roi_set: ROISet | None = None
        self.roi_target_movie: Image | None = None
        self.roi_reference_movie: Image | None = None
        self.selected_movie: Image | None = None
        self.roi_mode_name = "Shared"
        self.next_trace_color = 0
        self.removing_roi_layer = False
        self.syncing = False
        self.spatial_ref_name: str | None = None
        self.spatial_ref_shape: tuple[int, int] | None = None
        self.temporal_ref_name: str | None = None
        self.temporal_ref_length: int | None = None
        self.roi_colors = roi_colormap()
        self.roi_manager = ROIManagerWidget(self.select_roi_from_manager)
        self.roi_manager_dock = None
        self.suite2p_session: Suite2PSession | None = None
        self.suite2p_worker = None
        self.suite2p_busy = False
        self.movie_view_busy = False
        self.merged_movie_layer: Image | None = None
        self.merged_movie_keys: tuple[int, ...] = ()
        self.switching_movie_view = False
        self.activity_states: list[MovieState] = []
        self.activity_export_states: list[MovieState] = []
        self.activity_merged = False

        self.plot = plot
        self._build_ui()
        self.roi_feedback_timer = QTimer(self)
        self.roi_feedback_timer.setSingleShot(True)
        self.roi_feedback_timer.timeout.connect(self.reset_roi_feedback)
        self._connect_events()
        self.sync_from_selection()
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        QTimer.singleShot(0, self.reload_roi_layer)

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

        self.movie_group = QGroupBox("Movie")
        movie_layout = QVBoxLayout(self.movie_group)
        self.import_button = QPushButton("Import Movie")
        self.import_button.clicked.connect(lambda _=False: self.import_movies())
        movie_layout.addWidget(self.import_button)
        self.motion_correction_button = QPushButton("Motion Correction")
        self.motion_correction_button.clicked.connect(
            lambda _=False: self.start_motion_correction()
        )
        movie_layout.addWidget(self.motion_correction_button)
        self.merged_view_checkbox = QCheckBox("Merged view")
        self.merged_view_checkbox.toggled.connect(self.toggle_merged_view)
        movie_layout.addWidget(self.merged_view_checkbox)
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
        controls_layout.addWidget(self.movie_group)

        self.roi_group = QGroupBox("ROIs")
        roi_layout = QVBoxLayout(self.roi_group)
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["Shared", "Unique"])
        self.roi_mode_combo.currentTextChanged.connect(self.switch_roi_mode)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addWidget(self.roi_mode_combo)
        roi_layout.addLayout(mode_row)

        self.roi_detection_button = QPushButton("ROI Detection")
        self.roi_detection_button.clicked.connect(
            lambda _=False: self.start_roi_detection()
        )
        self.initialize_unique_button = QPushButton("Initialize Unique ROIs from Shared")
        self.initialize_unique_button.clicked.connect(
            lambda _=False: self.initialize_unique_rois_from_shared()
        )
        roi_layout.addWidget(self.roi_detection_button)
        roi_layout.addWidget(self.initialize_unique_button)

        self.show_roi_checkbox = QCheckBox("Show ROI layer")
        self.show_roi_checkbox.setChecked(False)
        self.show_roi_checkbox.toggled.connect(self.toggle_roi_layer)
        roi_layout.addWidget(self.show_roi_checkbox)

        target_label = QLabel("Target video")
        self.roi_target_combo = QComboBox()
        self.roi_target_combo.currentIndexChanged.connect(self.switch_roi_target)
        roi_layout.addWidget(target_label)
        roi_layout.addWidget(self.roi_target_combo)

        self.roi_reference_label = QLabel("Reference: none")
        set_reference_button = QPushButton("Set as Reference")
        copy_reference_button = QPushButton("Copy from Reference")
        set_reference_button.clicked.connect(lambda _=False: self.set_roi_reference())
        copy_reference_button.clicked.connect(lambda _=False: self.copy_roi_reference())
        reference_row = QHBoxLayout()
        reference_row.addWidget(set_reference_button)
        reference_row.addWidget(copy_reference_button)
        roi_layout.addWidget(self.roi_reference_label)
        roi_layout.addLayout(reference_row)

        load_roi_button = QPushButton("Load ROI Labels")
        save_roi_button = QPushButton("Export ROI Labels")
        load_roi_button.clicked.connect(lambda _=False: self.load_roi_labels())
        save_roi_button.clicked.connect(lambda _=False: self.export_roi_labels())
        roi_file_row = QHBoxLayout()
        roi_file_row.addWidget(load_roi_button)
        roi_file_row.addWidget(save_roi_button)
        roi_layout.addLayout(roi_file_row)
        self.roi_count_label = QLabel("ROIs: 0")
        self.roi_action_label = QLabel("Drawing ROI 1")
        open_manager_button = QPushButton("Open ROI Manager")
        open_manager_button.clicked.connect(lambda _=False: self.open_roi_manager())
        roi_layout.addWidget(self.roi_count_label)
        roi_layout.addWidget(self.roi_action_label)
        roi_layout.addWidget(open_manager_button)
        self.unique_roi_widgets = [
            target_label,
            self.roi_target_combo,
            self.roi_reference_label,
            set_reference_button,
            copy_reference_button,
        ]
        for widget in self.unique_roi_widgets:
            widget.setEnabled(False)
        controls_layout.addWidget(self.roi_group)

        trace_group = QGroupBox("Activities")
        trace_layout = QVBoxLayout(trace_group)
        norm_layout = QHBoxLayout()
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Raw", "dF/F0", "SNR(dF/F0)", "FRAME_SNR", "Z-score", "Min-max"])
        self.negative_signal_checkbox = QCheckBox("Negative-going signal")
        self.f0_spin = QSpinBox()
        self.f0_spin.setRange(0, 100)
        self.f0_spin.setValue(10)
        self.f0_spin.setSuffix("%")
        self.norm_combo.currentTextChanged.connect(lambda _="": self.redraw_plot())
        self.f0_spin.valueChanged.connect(lambda _=0: self.redraw_plot())
        self.negative_signal_checkbox.toggled.connect(self.change_signal_polarity)
        norm_layout.addWidget(QLabel("Normalize"))
        norm_layout.addWidget(self.norm_combo)
        norm_layout.addWidget(QLabel("F0"))
        norm_layout.addWidget(self.f0_spin)
        extract_button = QPushButton("Extract Activities")
        export_button = QPushButton("Export Activities")
        show_all_button = QPushButton("Show All Traces")
        hide_all_button = QPushButton("Hide All Traces")
        extract_button.clicked.connect(lambda _=False: self.extract_all_movies())
        export_button.clicked.connect(lambda _=False: self.export_activities())
        show_all_button.clicked.connect(lambda _=False: self.set_all_roi_visibility(True))
        hide_all_button.clicked.connect(lambda _=False: self.set_all_roi_visibility(False))
        trace_layout.addLayout(norm_layout)
        trace_layout.addWidget(self.negative_signal_checkbox)
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

        spike_group = QGroupBox("Spike")
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

        detect_button = QPushButton("Detect Spike")
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
        self.viewer.layers.events.inserted.connect(self.on_layer_inserted)
        self.viewer.layers.events.removing.connect(self.on_layer_removing)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)
        self.viewer.dims.events.current_step.connect(self.sync_time_line)
        self.start_spin.valueChanged.connect(self.save_time_roi)
        self.stop_spin.valueChanged.connect(self.save_time_roi)

    def active_image_layer(self) -> Image | None:
        layer = self.viewer.layers.selection.active
        return layer if isinstance(layer, Image) else None

    def current_image_layer(self) -> Image | None:
        return self.active_image_layer() or self.selected_movie

    def state_for(self, layer: Image) -> MovieState:
        return self.movie_manager.state_for(layer)

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
        ids_layer = self.current_roi_ids_layer()
        if (
            ids_layer is not None
            and self.viewer.layers.selection.active is ids_layer
        ):
            QTimer.singleShot(0, self.activate_roi_layer)
            return
        active = self.active_image_layer()
        target_changed = active is not None and active is not self.roi_target_movie
        if target_changed and self.roi_mode_name == "Unique":
            self.save_roi_layer()
        if active is not None:
            self.selected_movie = active
            self.roi_target_movie = active
            index = self.roi_target_combo.findData(active)
            if index >= 0:
                self.roi_target_combo.blockSignals(True)
                self.roi_target_combo.setCurrentIndex(index)
                self.roi_target_combo.blockSignals(False)

        layer = self.current_image_layer()
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
        if target_changed and self.roi_mode_name == "Unique":
            self.reload_roi_layer()
        self.refresh_roi_manager()
        self.sync_time_line()

    def activate_roi_layer(self) -> None:
        layer = self.current_roi_layer()
        if layer is not None:
            self.viewer.layers.selection.active = layer

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
        if self.merged_view_checkbox.isChecked():
            self.merged_view_checkbox.setChecked(False)
            QTimer.singleShot(0, self.import_movies)
            return
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
            self.movie_manager.mark_imported(last_layer, path)
        self.viewer.layers.selection.active = last_layer
        self.sync_from_selection()
        self.refresh_roi_targets()
        self.update_suite2p_controls()

    def suite2p_movie_inputs(self) -> list[MovieInput]:
        return [
            MovieInput(
                key=record.key,
                name=record.layer.name,
                data=record.motion_input,
                source_path=record.source_path,
            )
            for record in self.ordered_imported_movies()
        ]

    def ordered_imported_movies(self) -> list[ManagedMovie]:
        records = self.movie_manager.records
        return [
            records[id(layer)]
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and id(layer) in records
            and records[id(layer)].imported
        ]

    def session_matches_imported_movies(self) -> bool:
        if self.suite2p_session is None:
            return False
        imported_keys = {record.key for record in self.movie_manager.imported_movies()}
        return set(self.suite2p_session.movie_keys) == imported_keys

    def default_suite2p_output(self) -> Path:
        movies = self.ordered_imported_movies()
        if movies and movies[0].source_path is not None:
            return movies[0].source_path.parent
        return Path.cwd()

    def has_existing_rois(self) -> bool:
        if self.shared_roi_set is not None and (
            self.shared_roi_set.submitted_ids or np.any(self.shared_roi_set.labels)
        ):
            return True
        return any(
            record.state.roi_set is not None
            and (
                record.state.roi_set.submitted_ids
                or np.any(record.state.roi_set.labels)
            )
            for record in self.movie_manager.imported_movies()
        )

    def start_motion_correction(self) -> None:
        if self.suite2p_busy:
            return
        inputs = self.suite2p_movie_inputs()
        if not inputs:
            self.set_status("Import movies through Activity Tracer before motion correction.")
            return
        dialog = MotionCorrectionDialog(self.default_suite2p_output(), self)
        if not dialog.exec():
            return
        output_root, parameters = dialog.values()
        suite2p_dir = output_root.expanduser().resolve() / "suite2p"
        replace_existing = suite2p_dir.exists() and any(suite2p_dir.iterdir())
        if replace_existing:
            answer = QMessageBox.warning(
                self,
                "Replace existing Suite2p results?",
                f"{suite2p_dir} already contains results. They will be permanently "
                "deleted and cannot be recovered. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self.has_existing_rois():
            answer = QMessageBox.question(
                self,
                "Replace existing ROIs?",
                "Motion correction changes pixel coordinates. Existing ROIs and traces "
                "for imported movies will be cleared for this run.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        if (
            replace_existing
            and self.suite2p_session is not None
            and self.suite2p_session.suite2p_dir.resolve() == suite2p_dir
        ):
            self.discard_current_suite2p_session()

        self._set_suite2p_busy(True)
        self.set_status("Running Suite2p motion correction...")
        worker = create_worker(
            execute_motion_correction,
            inputs,
            output_root,
            parameters,
            replace_existing,
            _start_thread=False,
            _progress={"desc": "Suite2p motion correction"},
            _ignore_errors=True,
        )
        worker.returned.connect(self.on_motion_correction_complete)
        worker.errored.connect(self.on_suite2p_error)
        worker.finished.connect(self.on_suite2p_finished)
        self.suite2p_worker = worker
        worker.start()

    def discard_current_suite2p_session(self) -> None:
        if self.show_roi_checkbox.isChecked():
            self.show_roi_checkbox.setChecked(False)
        self.shared_roi_set = None
        for record in self.movie_manager.imported_movies():
            state = self.movie_manager.restore_motion_input(record.layer)
            state.roi_set = None
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()
        self.suite2p_session = None

    def on_motion_correction_complete(self, session: Suite2PSession) -> None:
        if session.movie_keys != tuple(
            record.key for record in self.ordered_imported_movies()
        ):
            self.set_status(
                "Movie layers or their order changed during motion correction; "
                "results were not applied."
            )
            return

        self.suite2p_session = session
        self.shared_roi_set = None
        for state in self.movie_states.values():
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()
        for record in self.movie_manager.imported_movies():
            data = session.registered_movie(record.key)
            state = self.movie_manager.apply_registered_data(record.layer, data)
            state.roi_set = None
            state.stop = min(state.stop, data.shape[0])
            record.layer.data = data[state.start : state.stop]

        if self.show_roi_checkbox.isChecked():
            self.reload_roi_layer()
        self.sync_from_selection()
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        metric_status = (
            " Registration metrics were saved."
            if "regDX" in session.reg_outputs
            else " Registration metrics require at least 1500 combined frames."
        )
        self.set_status(
            f"Motion-corrected {len(session.movie_keys)} movies. "
            f"Suite2p outputs: {session.suite2p_dir}.{metric_status}"
        )

    def start_roi_detection(self) -> None:
        if self.suite2p_busy:
            return
        if self.roi_mode_name != "Shared":
            self.set_status("Switch to Shared mode before Suite2p ROI detection.")
            return
        session = self.suite2p_session
        if session is None or not self.session_matches_imported_movies():
            self.set_status("Run motion correction for the current imported movies first.")
            return
        dialog = ROIDetectionDialog(self)
        if not dialog.exec():
            return
        if self.shared_roi_set is not None and (
            self.shared_roi_set.submitted_ids or np.any(self.shared_roi_set.labels)
        ):
            answer = QMessageBox.question(
                self,
                "Replace shared ROIs?",
                "Suite2p candidates will replace the current shared ROIs.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_suite2p_busy(True)
        self.set_status("Running Suite2p ROI detection...")
        worker = create_worker(
            execute_roi_detection,
            session,
            dialog.values(),
            _start_thread=False,
            _progress={"desc": "Suite2p ROI detection"},
            _ignore_errors=True,
        )
        worker.returned.connect(self.on_roi_detection_complete)
        worker.errored.connect(self.on_suite2p_error)
        worker.finished.connect(self.on_suite2p_finished)
        self.suite2p_worker = worker
        worker.start()

    def on_roi_detection_complete(self, result: ROIDetectionResult) -> None:
        active_label = max(result.roi_ids, default=0) + 1
        self.shared_roi_set = ROISet(result.labels, set(result.roi_ids), active_label)
        for state in self.movie_states.values():
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()

        movies = self.movie_manager.imported_movies()
        target = self.merged_movie_layer if self.merged_movie_layer is not None else (
            movies[0].layer if movies else None
        )
        if target is not None:
            self.roi_target_movie = target
            self.selected_movie = target
            self.viewer.layers.selection.active = target
        if not self.show_roi_checkbox.isChecked():
            self.show_roi_checkbox.setChecked(True)
        else:
            self.reload_roi_layer()
        self.refresh_roi_manager()
        self.rebuild_roi_controls()
        self.redraw_plot()
        self.update_suite2p_controls()
        self.set_status(
            f"Imported {len(result.roi_ids)} Suite2p candidate ROIs from {result.stat_path}."
        )

    def initialize_unique_rois_from_shared(self) -> None:
        if self.roi_mode_name != "Shared":
            self.set_status("Switch to Shared mode before initializing unique ROIs.")
            return
        self.save_roi_layer(mode="Shared")
        source = self.shared_roi_set
        if source is None or not source.submitted_ids:
            self.set_status("Detect, draw, or load shared ROIs first.")
            return

        copied = 0
        skipped = 0
        for record in self.movie_manager.imported_movies():
            target = record.state.roi_set
            has_unique = target is not None and (
                target.submitted_ids or np.any(target.labels)
            )
            if has_unique or source.labels.shape != tuple(record.state.source_data.shape[-2:]):
                skipped += 1
                continue
            record.state.roi_set = ROISet(
                source.labels.copy(),
                set(source.submitted_ids),
                source.active_label,
            )
            copied += 1

        if copied:
            self.roi_mode_combo.setCurrentText("Unique")
        self.update_suite2p_controls()
        self.set_status(
            f"Initialized unique ROIs for {copied} movies; skipped {skipped} with "
            "existing or incompatible unique ROIs."
        )

    def _set_suite2p_busy(self, busy: bool) -> None:
        self.suite2p_busy = busy
        enabled = not busy and not self.movie_view_busy
        self.movie_group.setEnabled(enabled)
        self.roi_group.setEnabled(enabled)
        if not busy:
            self.refresh_roi_targets()
            self.update_suite2p_controls()

    def _set_movie_view_busy(self, busy: bool) -> None:
        self.movie_view_busy = busy
        enabled = not busy and not self.suite2p_busy
        self.movie_group.setEnabled(enabled)
        self.roi_group.setEnabled(enabled)
        if not busy:
            self.refresh_roi_targets()
            self.update_suite2p_controls()

    def on_suite2p_error(self, error: Exception) -> None:
        self.set_status(f"Suite2p failed: {error}")
        QMessageBox.critical(self, "Suite2p failed", str(error))

    def on_suite2p_finished(self) -> None:
        self.suite2p_worker = None
        self._set_suite2p_busy(False)

    def update_suite2p_controls(self) -> None:
        has_movies = bool(self.movie_manager.imported_movies())
        valid_session = self.session_matches_imported_movies()
        merged = self.merged_movie_layer is not None
        has_shared = self.shared_roi_set is not None and bool(
            self.shared_roi_set.submitted_ids
        )
        self.motion_correction_button.setEnabled(
            has_movies
            and not merged
            and not self.suite2p_busy
            and not self.movie_view_busy
        )
        self.merged_view_checkbox.setEnabled(
            valid_session
            and self.roi_mode_name == "Shared"
            and not self.suite2p_busy
            and not self.movie_view_busy
        )
        self.roi_detection_button.setEnabled(
            valid_session
            and self.roi_mode_name == "Shared"
            and not self.suite2p_busy
            and not self.movie_view_busy
        )
        self.initialize_unique_button.setEnabled(
            has_shared
            and self.roi_mode_name == "Shared"
            and not merged
            and not self.suite2p_busy
            and not self.movie_view_busy
        )

    def toggle_merged_view(self, checked: bool) -> None:
        if self.movie_view_busy:
            return
        self._set_movie_view_busy(True)
        self.set_status(
            "Building merged view..." if checked else "Restoring separate views..."
        )
        QTimer.singleShot(0, lambda: self.apply_movie_view(checked))

    def apply_movie_view(self, merged: bool) -> None:
        try:
            if merged:
                self.show_merged_view()
            else:
                self.show_separate_view()
        finally:
            self._set_movie_view_busy(False)

    def show_merged_view(self) -> None:
        if not self.session_matches_imported_movies():
            self.merged_view_checkbox.blockSignals(True)
            self.merged_view_checkbox.setChecked(False)
            self.merged_view_checkbox.blockSignals(False)
            self.set_status("Run motion correction before enabling merged view.")
            return
        records = self.ordered_imported_movies()
        if not records:
            return
        if any(record.state.start >= record.state.stop for record in records):
            self.merged_view_checkbox.blockSignals(True)
            self.merged_view_checkbox.setChecked(False)
            self.merged_view_checkbox.blockSignals(False)
            self.set_status("Each movie needs a non-empty TimeROI before merging.")
            return

        self.save_roi_layer()
        for record in records:
            record.state.layer_name = record.layer.name
        self.merged_movie_keys = tuple(record.key for record in records)
        chunks = [
            da.from_array(
                record.state.source_data[record.state.start : record.state.stop],
                chunks=(
                    max(1, min(256, record.state.stop - record.state.start)),
                    *record.state.source_data.shape[-2:],
                ),
                asarray=False,
            )
            for record in records
        ]
        merged_data = da.concatenate(chunks, axis=0)

        self.switching_movie_view = True
        try:
            for record in records:
                self.viewer.layers.remove(record.layer)
            merged_layer = Image(merged_data, name="Merged registered movies")
            self.viewer.layers.insert(0, merged_layer)
        finally:
            self.switching_movie_view = False

        self.merged_movie_layer = merged_layer
        self.selected_movie = merged_layer
        self.roi_target_movie = merged_layer
        self.viewer.layers.selection.active = merged_layer
        self.roi_mode_combo.setEnabled(False)
        self.sync_from_selection()
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        self.set_status(
            f"Merged {len(records)} movies in the current bottom-to-top layer order."
        )

    def show_separate_view(self) -> None:
        merged_layer = self.merged_movie_layer
        if merged_layer is None:
            return
        records = [
            self.movie_manager.records[key]
            for key in self.merged_movie_keys
            if key in self.movie_manager.records
        ]

        self.switching_movie_view = True
        try:
            if merged_layer in self.viewer.layers:
                self.viewer.layers.remove(merged_layer)
            self.movie_manager.remove(merged_layer)
            for index, record in enumerate(records):
                self.viewer.layers.insert(index, record.layer)
        finally:
            self.switching_movie_view = False

        self.merged_movie_layer = None
        self.merged_movie_keys = ()
        self.roi_mode_combo.setEnabled(True)
        target = records[0].layer if records else None
        self.selected_movie = target
        self.roi_target_movie = target
        if target is not None:
            self.viewer.layers.selection.active = target
        self.sync_from_selection()
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        self.set_status("Restored separate movie layers.")

    def current_roi_layer(self) -> Labels | None:
        return self.roi_layer if self.roi_layer is not None and self.roi_layer in self.viewer.layers else None

    def current_roi_ids_layer(self) -> Points | None:
        return (
            self.roi_ids_layer
            if self.roi_ids_layer is not None and self.roi_ids_layer in self.viewer.layers
            else None
        )

    def roi_set_for(
        self,
        mode: str | None = None,
        movie: Image | None = None,
        create: bool = True,
    ) -> ROISet | None:
        mode = mode or self.roi_mode_name
        movie = movie or self.roi_target_movie or self.selected_movie
        if movie is None:
            return None
        if mode == "Shared":
            if self.shared_roi_set is None and create:
                self.shared_roi_set = ROISet(np.zeros(movie.data.shape[-2:], dtype=np.uint16))
            return self.shared_roi_set
        state = self.state_for(movie)
        if state.roi_set is None and create:
            state.roi_set = ROISet(np.zeros(movie.data.shape[-2:], dtype=np.uint16))
        return state.roi_set

    def roi_color(self, roi_id: int) -> np.ndarray:
        return np.asarray(self.roi_colors.map(roi_id), dtype=float)

    def refresh_roi_manager(self) -> None:
        roi_set = self.roi_set_for(create=False)
        ids = sorted(roi_set.submitted_ids) if roi_set is not None else []
        selected = roi_set.active_label if roi_set is not None else 1
        colors = {roi_id: self.roi_color(roi_id) for roi_id in ids}
        self.roi_manager.set_rois(ids, colors, selected)
        self.roi_count_label.setText(f"ROIs: {len(ids)}")
        if not self.roi_feedback_timer.isActive():
            self.roi_action_label.setText(f"Drawing ROI {selected}")
        if self.roi_manager_dock is not None:
            title = "Shared ROI Manager"
            if self.roi_mode_name == "Unique" and self.roi_target_movie is not None:
                title = f"ROI Manager — {self.roi_target_movie.name}"
            self.roi_manager_dock.setWindowTitle(title)

    def open_roi_manager(self) -> None:
        if self.roi_manager_dock is None:
            self.roi_manager_dock = self.viewer.window.add_dock_widget(
                self.roi_manager,
                name="ROI Manager",
                area="right",
            )
            self.roi_manager_dock.setFloating(True)
        self.refresh_roi_manager()
        self.roi_manager_dock.show()
        self.roi_manager_dock.raise_()

    def select_roi_from_manager(self, roi_id: int) -> None:
        if not self.show_roi_checkbox.isChecked():
            self.show_roi_checkbox.setChecked(True)
        layer = self.current_roi_layer()
        roi_set = self.roi_set_for(create=False)
        if layer is None or roi_set is None:
            return
        roi_set.active_label = roi_id
        layer.selected_label = roi_id
        self.viewer.layers.selection.active = layer
        self.viewer.window._qt_viewer.setFocus()
        self.refresh_roi_manager()

    def on_roi_label_selected(self, event=None) -> None:
        layer = self.current_roi_layer()
        roi_set = self.roi_set_for(create=False)
        if layer is None or roi_set is None:
            return
        roi_set.active_label = int(layer.selected_label)
        if layer.show_selected_label:
            self.reload_roi_ids_layer()
        self.refresh_roi_manager()

    def show_roi_feedback(self, text: str, color: str) -> None:
        self.roi_action_label.setText(text)
        self.roi_action_label.setStyleSheet(
            f"QLabel {{ background: {color}; color: black; padding: 4px; }}"
        )
        self.roi_feedback_timer.start(900)

    def reset_roi_feedback(self) -> None:
        self.roi_action_label.setStyleSheet("")
        self.refresh_roi_manager()

    def reload_roi_ids_layer(self, event=None) -> None:
        if not self.show_roi_checkbox.isChecked():
            return
        roi_set = self.roi_set_for(create=False)
        roi_layer = self.current_roi_layer()
        if roi_set is None or roi_layer is None:
            return
        labels = np.asarray(roi_layer.data)
        ids = [
            roi_id
            for roi_id in sorted(roi_set.submitted_ids)
            if np.any(labels == roi_id)
        ]
        if roi_layer.show_selected_label:
            ids = [roi_id for roi_id in ids if roi_id == roi_layer.selected_label]
        positions = roi_centers(labels, ids)
        features = {"roi_id": np.asarray(ids, dtype=int)}
        colors = np.asarray([self.roi_color(roi_id) for roi_id in ids])
        name = (
            "Shared ROI IDs"
            if self.roi_mode_name == "Shared"
            else f"ROI IDs: {self.roi_target_movie.name}"
        )
        layer = self.current_roi_ids_layer()
        if layer is None:
            layer = Points(
                positions,
                border_color=colors if ids else "white",
                border_width=0,
                face_color=[0, 0, 0, 0],
                features=features,
                name=name,
                size=18,
                text={
                    "string": "{roi_id}",
                    "color": "white",
                    "size": 12,
                    "anchor": "center",
                },
            )
            layer.editable = False
            self.roi_ids_layer = layer
            self.viewer.layers.append(layer)
        else:
            layer.data = positions
            layer.features = features
            layer.border_color = colors if ids else "white"
            layer.name = name

    def save_roi_layer(
        self,
        mode: str | None = None,
        movie: Image | None = None,
        layer: Labels | None = None,
    ) -> None:
        layer = layer or self.current_roi_layer()
        roi_set = self.roi_set_for(mode, movie)
        if layer is None or roi_set is None:
            return
        labels = np.asarray(layer.data)
        if np.array_equal(labels, roi_set.labels):
            return
        roi_set.labels = labels.copy()
        if (mode or self.roi_mode_name) == "Shared":
            states = self.movie_states.values()
        else:
            target = movie or self.roi_target_movie
            states = [self.state_for(target)] if target is not None else []
        for state in states:
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()

    def reload_roi_layer(self) -> None:
        if not self.show_roi_checkbox.isChecked():
            return
        roi_set = self.roi_set_for()
        if roi_set is None:
            return
        layer = self.current_roi_layer()
        name = "Shared ROIs" if self.roi_mode_name == "Shared" else f"ROIs: {self.roi_target_movie.name}"
        if layer is None:
            layer = ROILabels(
                roi_set.labels.copy(),
                colormap=self.roi_colors,
                name=name,
                opacity=0.5,
            )
            layer.contour = 2
            layer.preserve_labels = True
            layer.selected_label = roi_set.active_label
            self.roi_layer = layer
            self.viewer.layers.append(layer)
            self.bind_roi_shortcuts(layer)
            layer.events.selected_label.connect(self.on_roi_label_selected)
            layer.events.show_selected_label.connect(self.reload_roi_ids_layer)
        else:
            layer.data = roi_set.labels.copy()
            layer.name = name
            layer.selected_label = roi_set.active_label
        self.reload_roi_ids_layer()
        self.activate_roi_layer()
        self.refresh_roi_manager()

    def bind_roi_shortcuts(self, layer: Labels) -> None:
        @layer.bind_key("T")
        def submit_roi(active_layer: Labels) -> None:
            label_id = int(active_layer.selected_label)
            background = int(active_layer.colormap.background_value)
            labels = np.asarray(active_layer.data)
            if label_id == background or not np.any(labels == label_id):
                self.set_status("Draw an ROI before submitting it.")
                self.show_roi_feedback(f"ROI {label_id} is empty", "#f4c95d")
                return

            self.save_roi_layer(layer=active_layer)
            roi_set = self.roi_set_for()
            roi_set.submitted_ids.add(label_id)
            next_label = int(labels.max()) + 1
            roi_set.active_label = next_label
            active_layer.selected_label = next_label
            self.reload_roi_ids_layer()
            self.refresh_roi_manager()
            self.update_suite2p_controls()
            self.set_status(f"Submitted ROI {label_id}. Drawing ROI {next_label}.")
            self.show_roi_feedback(f"✓ ROI {label_id} submitted", "#7bd88f")

        @layer.bind_key("C")
        def clear_roi(active_layer: Labels) -> None:
            label_id = int(active_layer.selected_label)
            background = int(active_layer.colormap.background_value)
            labels = np.asarray(active_layer.data)
            indices = np.nonzero(labels == label_id)
            if label_id == background or indices[0].size == 0:
                self.set_status("The selected ROI is already empty.")
                self.show_roi_feedback(f"ROI {label_id} is empty", "#f4c95d")
                return

            active_layer.data_setitem(indices, background)
            self.save_roi_layer(layer=active_layer)
            roi_set = self.roi_set_for()
            roi_set.submitted_ids.discard(label_id)
            roi_set.active_label = label_id
            self.reload_roi_ids_layer()
            self.refresh_roi_manager()
            self.update_suite2p_controls()
            self.set_status(f"Cleared ROI {label_id}.")
            self.show_roi_feedback(f"ROI {label_id} cleared", "#f4c95d")

    def toggle_roi_layer(self, checked: bool) -> None:
        if checked:
            self.reload_roi_layer()
            return
        layer = self.current_roi_layer()
        ids_layer = self.current_roi_ids_layer()
        if layer is None:
            return
        self.save_roi_layer()
        self.removing_roi_layer = True
        if ids_layer is not None:
            self.viewer.layers.remove(ids_layer)
        self.viewer.layers.remove(layer)
        self.removing_roi_layer = False

    def on_layer_removing(self, event) -> None:
        layer = self.viewer.layers[event.index]
        if layer is self.roi_ids_layer:
            self.roi_ids_layer = None
            return
        if isinstance(layer, Image):
            if self.switching_movie_view:
                return
            if layer is self.roi_target_movie and self.roi_mode_name == "Unique":
                self.save_roi_layer(movie=layer)
            return
        if layer is not self.roi_layer:
            return
        if not self.removing_roi_layer:
            self.save_roi_layer(layer=layer)
        self.roi_layer = None
        self.show_roi_checkbox.blockSignals(True)
        self.show_roi_checkbox.setChecked(False)
        self.show_roi_checkbox.blockSignals(False)

    def on_layer_inserted(self, event) -> None:
        layer = event.value
        if not isinstance(layer, Image) or self.switching_movie_view:
            return
        self.state_for(layer)
        if self.roi_target_movie is None:
            self.roi_target_movie = layer
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        if self.show_roi_checkbox.isChecked() and self.current_roi_layer() is None:
            QTimer.singleShot(0, self.reload_roi_layer)

    def on_layer_removed(self, event) -> None:
        layer = event.value
        if not isinstance(layer, Image) or self.switching_movie_view:
            return
        previous_target = self.roi_target_movie
        if layer is self.selected_movie:
            self.selected_movie = self.active_image_layer()
        self.movie_manager.remove(layer)
        self.refresh_roi_targets()
        self.update_suite2p_controls()
        if previous_target is not self.roi_target_movie and self.roi_mode_name == "Unique":
            QTimer.singleShot(0, self.reload_roi_layer)

    def switch_roi_mode(self, mode: str) -> None:
        if mode == self.roi_mode_name:
            return
        self.save_roi_layer(mode=self.roi_mode_name)
        self.roi_mode_name = mode
        for widget in self.unique_roi_widgets:
            widget.setEnabled(mode == "Unique" and self.roi_target_combo.count() > 0)
        self.reload_roi_layer()
        self.refresh_roi_manager()
        self.update_suite2p_controls()
        self.set_status(f"ROI mode changed to {mode}.")

    def refresh_roi_targets(self) -> None:
        movies = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        if self.roi_target_movie not in movies:
            self.roi_target_movie = (
                self.selected_movie
                if self.selected_movie in movies
                else (movies[0] if movies else None)
            )
        if self.roi_reference_movie not in movies:
            self.roi_reference_movie = None
        self.roi_target_combo.blockSignals(True)
        self.roi_target_combo.clear()
        for movie in movies:
            self.roi_target_combo.addItem(movie.name, movie)
        index = self.roi_target_combo.findData(self.roi_target_movie)
        if index >= 0:
            self.roi_target_combo.setCurrentIndex(index)
        self.roi_target_combo.blockSignals(False)
        reference = self.roi_reference_movie.name if self.roi_reference_movie is not None else "none"
        self.roi_reference_label.setText(f"Reference: {reference}")
        self.show_roi_checkbox.setEnabled(bool(movies))
        for widget in self.unique_roi_widgets:
            widget.setEnabled(self.roi_mode_name == "Unique" and bool(movies))
        self.refresh_roi_manager()

    def switch_roi_target(self, index: int) -> None:
        if self.syncing:
            return
        movie = self.roi_target_combo.itemData(index)
        if movie is None or movie is self.roi_target_movie:
            return
        if self.roi_mode_name == "Unique":
            self.save_roi_layer()
        self.roi_target_movie = movie
        self.selected_movie = movie
        self.viewer.layers.selection.active = movie
        if self.roi_mode_name == "Unique":
            self.reload_roi_layer()
        self.refresh_roi_manager()

    def set_roi_reference(self) -> None:
        self.save_roi_layer()
        if self.roi_target_movie is None:
            self.set_status("Select a target video first.")
            return
        self.roi_reference_movie = self.roi_target_movie
        self.roi_reference_label.setText(f"Reference: {self.roi_reference_movie.name}")
        self.set_status(f"ROI reference set to {self.roi_reference_movie.name}.")

    def copy_roi_reference(self) -> None:
        self.save_roi_layer()
        source = self.roi_set_for("Unique", self.roi_reference_movie, create=False)
        target = self.roi_target_movie
        if source is None or target is None:
            self.set_status("Set an ROI reference and select a target video first.")
            return
        if source.labels.shape != tuple(target.data.shape[-2:]):
            self.set_status("Reference ROIs must match the target movie Y/X shape.")
            return
        state = self.state_for(target)
        state.roi_set = ROISet(
            source.labels.copy(),
            set(source.submitted_ids),
            source.active_label,
        )
        state.traces.clear()
        state.visible_rois.clear()
        state.spikes.clear()
        self.reload_roi_layer()
        self.refresh_roi_manager()
        self.set_status(f"Copied ROIs from {self.roi_reference_movie.name} to {target.name}.")

    def roi_labels_for(self, movie: Image) -> np.ndarray | None:
        roi_set = self.roi_set_for(self.roi_mode_name, movie, create=False)
        return None if roi_set is None else roi_set.labels

    def load_roi_labels(self) -> None:
        movie = self.roi_target_movie or self.current_image_layer()
        if movie is None:
            self.set_status("Select a movie before loading ROI labels.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI labels", filter="NumPy labels (*.npy)")
        if not path:
            return
        labels = np.load(path)
        if labels.shape != tuple(movie.data.shape[-2:]):
            self.set_status("ROI labels must match the movie Y/X shape.")
            return
        ids = set(roi_ids(labels))
        roi_set = ROISet(labels, ids, max(ids, default=0) + 1)
        if self.roi_mode_name == "Shared":
            self.shared_roi_set = roi_set
            states = self.movie_states.values()
        else:
            state = self.state_for(movie)
            state.roi_set = roi_set
            states = [state]
        for state in states:
            state.traces.clear()
            state.visible_rois.clear()
            state.spikes.clear()
        self.reload_roi_layer()
        self.refresh_roi_manager()
        self.update_suite2p_controls()
        self.set_status(f"ROI labels loaded in {self.roi_mode_name} mode.")

    def export_roi_labels(self) -> None:
        self.save_roi_layer()
        roi_set = self.roi_set_for(create=False)
        if roi_set is None:
            self.set_status("Create, copy, or load ROI labels first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ROI labels", filter="NumPy labels (*.npy)")
        if not path:
            return
        if not path.endswith(".npy"):
            path += ".npy"
        np.save(path, roi_set.labels)
        self.set_status(f"ROI labels exported to {path}.")

    def extract_all_movies(self) -> None:
        self.save_roi_layer()
        if self.merged_movie_layer is not None:
            records = [
                self.movie_manager.records[key]
                for key in self.merged_movie_keys
                if key in self.movie_manager.records
            ]
            results = [(record, self.trace_record(record)) for record in records]
            completed = [(record, count) for record, count in results if count is not None]
            failed = [record.state.layer_name for record, count in results if count is None]
            export_states = [record.state for record, _count in completed]
            if export_states:
                merged_state = self.merge_activity_states(export_states)
                self.activity_states = [merged_state]
                self.activity_export_states = export_states
                self.activity_merged = True
                completed = [(merged_state, len(merged_state.traces))]
            else:
                self.activity_states = []
                self.activity_export_states = []
                self.activity_merged = False
            self.finish_activity_extraction(completed, failed)
            return

        movies = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        if not movies:
            self.set_status("Import movies before extracting activities.")
            return
        results = [(layer, self.trace_movie(layer)) for layer in movies]
        completed = [(layer, count) for layer, count in results if count is not None]
        failed = [layer.name for layer, count in results if count is None]
        self.activity_states = [self.state_for(layer) for layer, _count in completed]
        self.activity_export_states = list(self.activity_states)
        self.activity_merged = False
        self.finish_activity_extraction(completed, failed)

    def merge_activity_states(self, states: list[MovieState]) -> MovieState:
        common_rois = set(states[0].traces)
        for state in states[1:]:
            common_rois.intersection_update(state.traces)

        merged = MovieState(
            stop=sum(state.stop - state.start for state in states),
            layer_name="Merged registered movies",
        )
        merged.traces = {
            roi: np.concatenate([state.traces[roi] for state in states])
            for roi in sorted(common_rois)
        }
        merged.trace_colors = {
            roi: states[0].trace_colors[roi]
            for roi in merged.traces
        }
        merged.visible_rois = set(merged.traces)
        return merged

    def finish_activity_extraction(self, completed: list, failed: list[str]) -> None:
        self.rebuild_roi_controls()
        self.redraw_plot()
        total = sum(count for _, count in completed)
        if self.activity_merged:
            message = (
                f"Extracted {total} merged ROI traces from "
                f"{len(self.activity_export_states)} movies."
            )
        else:
            message = f"Extracted {total} ROI traces from {len(completed)} movies."
        if failed:
            message += f" Skipped: {', '.join(failed)}."
        self.set_status(message)

    def trace_movie(self, layer: Image, visible_rois: set[int] | None = None) -> int | None:
        self.save_roi_layer()
        state = self.state_for(layer)
        labels = self.roi_labels_for(layer)
        return self.trace_state(state, np.asarray(layer.data), labels, visible_rois)

    def trace_record(
        self,
        record: ManagedMovie,
        visible_rois: set[int] | None = None,
    ) -> int | None:
        state = record.state
        roi_set = self.shared_roi_set if self.roi_mode_name == "Shared" else state.roi_set
        labels = None if roi_set is None else roi_set.labels
        movie = np.asarray(state.source_data[state.start : state.stop])
        return self.trace_state(state, movie, labels, visible_rois)

    def trace_state(
        self,
        state: MovieState,
        movie: np.ndarray,
        labels: np.ndarray | None,
        visible_rois: set[int] | None = None,
    ) -> int | None:
        if labels is None:
            self.set_status("Create, copy, or load ROI labels first.")
            return None
        if state.start >= state.stop:
            self.set_status("TimeROI is empty. Set Start smaller than Stop.")
            return None
        if labels.shape != tuple(movie.shape[-2:]):
            self.set_status("ROI labels must match the movie Y/X shape.")
            return None
        ids = roi_ids(labels)
        flat_movie = movie.reshape(movie.shape[0], -1)
        flat_labels = labels.reshape(-1)

        state.traces = {
            int(roi_id): flat_movie[:, flat_labels == roi_id].mean(axis=1)
            for roi_id in ids
        }
        for roi in state.traces:
            if roi not in state.trace_colors:
                hue = self.next_trace_color * 137 % 360
                state.trace_colors[roi] = QColor.fromHsv(hue, 180, 220).name()
                self.next_trace_color += 1
        state.spikes.clear()
        state.visible_rois = set(state.traces) if visible_rois is None else set(state.traces) & visible_rois
        if not state.visible_rois:
            state.visible_rois = set(state.traces)
        return len(state.traces)

    def detect_frame_spikes(self) -> None:
        layer = self.current_image_layer()
        if layer is None:
            self.set_status("Select a movie layer first.")
            return
        state = (
            self.activity_states[0]
            if self.activity_merged and self.activity_states
            else self.state_for(layer)
        )
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
        return find_frame_spikes(trace, self.frame_settings())

    def frame_processed_trace(self, trace: np.ndarray) -> dict[str, np.ndarray | float]:
        return process_frame_trace(trace, self.frame_settings())

    def frame_settings(self) -> FrameSettings:
        return FrameSettings(
            fps=self.spike_fps_spin.value(),
            baseline_ms=self.spike_baseline_spin.value(),
            lowpass_hz=self.spike_lowpass_spin.value(),
            highpass_hz=self.spike_highpass_spin.value(),
            isi_ms=self.spike_isi_spin.value(),
            threshold=self.spike_threshold_spin.value(),
            negative_signal=self.negative_signal_checkbox.isChecked(),
        )

    def change_signal_polarity(self, _checked: bool) -> None:
        for state in self.movie_states.values():
            state.spikes.clear()
        for state in self.activity_states:
            state.spikes.clear()
        self.redraw_plot()

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
            for state in self.activity_export_states:
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
        signal = "-dF/F0" if self.negative_signal_checkbox.isChecked() else "dF/F0"
        if mode == "dF/F0":
            return f"{signal} F0={self.f0_spin.value()}%"
        if mode == "SNR(dF/F0)":
            return f"SNR({signal}) F0={self.f0_spin.value()}%"
        if mode == "FRAME_SNR":
            direction = "valley" if self.negative_signal_checkbox.isChecked() else "peak"
            return f"{mode} ({direction})"
        return mode

    def normalize_trace(self, trace: np.ndarray) -> np.ndarray:
        return normalize_activity_trace(
            trace,
            self.norm_combo.currentText(),
            self.f0_spin.value(),
            self.frame_settings(),
        )

    def traced_states(self) -> list[MovieState]:
        return [state for state in self.activity_states if state.traces]

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
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                checkbox = QCheckBox(f"ROI {roi}")
                checkbox.setChecked(roi in state.visible_rois)
                checkbox.toggled.connect(partial(self.set_roi_visibility, state, roi))
                color_block = QLabel()
                color_block.setFixedSize(18, 18)
                color_button = QPushButton()
                color_button.setFixedWidth(90)
                color_button.clicked.connect(
                    partial(self.choose_trace_color, state, roi, color_block, color_button)
                )
                self.update_trace_color_widgets(color_block, color_button, state.trace_colors[roi])
                row_layout.addWidget(checkbox)
                row_layout.addStretch()
                row_layout.addWidget(color_block)
                row_layout.addWidget(color_button)
                self.roi_check_layout.addWidget(row)
        self.roi_check_layout.addStretch()

    def choose_trace_color(
        self,
        state: MovieState,
        roi: int,
        block: QLabel,
        button: QPushButton,
        _checked: bool = False,
    ) -> None:
        color = QColorDialog.getColor(QColor(state.trace_colors[roi]), self, f"Color for ROI {roi}")
        if not color.isValid():
            return
        state.trace_colors[roi] = color.name()
        self.update_trace_color_widgets(block, button, color.name())
        self.redraw_plot()

    @staticmethod
    def update_trace_color_widgets(block: QLabel, button: QPushButton, color: str) -> None:
        block.setStyleSheet(f"background-color: {color}; border: 1px solid #808080;")
        button.setText(color.upper())

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
