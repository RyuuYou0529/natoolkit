from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from .models import MovieState


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

    def draw_traces(
        self,
        states: list[MovieState],
        mode: str,
        normalize,
    ) -> None:
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
            self.ax.plot(
                frames,
                trace,
                color=state.trace_colors[roi],
                ls="-",
                lw=1.2,
                label=f"{state.layer_name} / ROI {roi}",
            )
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
