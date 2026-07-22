from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .suite2p_adapter import MotionCorrectionParameters, ROIDetectionParameters


def _device_combo() -> QComboBox:
    combo = QComboBox()
    for label, value in (("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("MPS", "mps")):
        combo.addItem(label, value)
    return combo


def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def _float_spin(
    minimum: float,
    maximum: float,
    value: float,
    decimals: int = 2,
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


class MotionCorrectionDialog(QDialog):
    def __init__(self, output_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Suite2p Motion Correction")
        self.setMinimumWidth(460)
        layout = QFormLayout(self)

        self.output_edit = QLineEdit(str(output_root))
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self._browse_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_button)
        layout.addRow("Output parent", output_row)

        self.device_combo = _device_combo()
        self.nonrigid_check = QCheckBox()
        self.nonrigid_check.setChecked(True)
        self.nimg_init_spin = _int_spin(1, 1_000_000, 400)
        self.batch_size_spin = _int_spin(1, 1_000_000, 100)
        self.maxregshift_spin = _float_spin(0.0, 1.0, 0.1, 3)
        self.smooth_sigma_spin = _float_spin(0.25, 100.0, 1.15, 2)
        self.block_y_spin = _int_spin(1, 100_000, 128)
        self.block_x_spin = _int_spin(1, 100_000, 128)
        block_row = QWidget()
        block_layout = QHBoxLayout(block_row)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.addWidget(self.block_y_spin)
        block_layout.addWidget(self.block_x_spin)
        self.maxregshift_nr_spin = _int_spin(0, 100_000, 5)
        self.bidiphase_check = QCheckBox()

        layout.addRow("Compute device", self.device_combo)
        layout.addRow("Non-rigid registration", self.nonrigid_check)
        layout.addRow("Reference frames", self.nimg_init_spin)
        layout.addRow("Batch size", self.batch_size_spin)
        layout.addRow("Maximum rigid shift", self.maxregshift_spin)
        layout.addRow("XY smoothing", self.smooth_sigma_spin)
        layout.addRow("Non-rigid block Y/X", block_row)
        layout.addRow("Maximum non-rigid shift", self.maxregshift_nr_spin)
        layout.addRow("Compute bidiphase", self.bidiphase_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Suite2p output parent",
            self.output_edit.text(),
        )
        if path:
            self.output_edit.setText(path)

    def values(self) -> tuple[Path, MotionCorrectionParameters]:
        return Path(self.output_edit.text()), MotionCorrectionParameters(
            device=str(self.device_combo.currentData()),
            nonrigid=self.nonrigid_check.isChecked(),
            nimg_init=self.nimg_init_spin.value(),
            batch_size=self.batch_size_spin.value(),
            maxregshift=self.maxregshift_spin.value(),
            smooth_sigma=self.smooth_sigma_spin.value(),
            block_size=(self.block_y_spin.value(), self.block_x_spin.value()),
            maxregshift_nr=self.maxregshift_nr_spin.value(),
            do_bidiphase=self.bidiphase_check.isChecked(),
        )


class ROIDetectionDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Suite2p ROI Detection")
        self.setMinimumWidth(420)
        layout = QFormLayout(self)

        self.device_combo = _device_combo()
        self.fs_spin = _float_spin(0.01, 1_000_000.0, 10.0, 3)
        self.tau_spin = _float_spin(0.01, 10.0, 1.0, 3)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["sparsery", "sourcery", "cellpose"])
        self.diameter_y_spin = _float_spin(1.0, 100_000.0, 12.0, 1)
        self.diameter_x_spin = _float_spin(1.0, 100_000.0, 12.0, 1)
        diameter_row = QWidget()
        diameter_layout = QHBoxLayout(diameter_row)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.addWidget(self.diameter_y_spin)
        diameter_layout.addWidget(self.diameter_x_spin)
        self.threshold_spin = _float_spin(0.0, 1000.0, 1.0, 3)
        self.max_rois_spin = _int_spin(1, 10_000_000, 5000)
        self.spatial_scale_spin = _int_spin(0, 4, 0)
        self.max_overlap_spin = _float_spin(0.0, 1.0, 0.75, 2)
        self.nbins_spin = _int_spin(1, 10_000_000, 5000)
        self.denoise_check = QCheckBox()

        layout.addRow("Compute device", self.device_combo)
        layout.addRow("Sampling frequency (Hz)", self.fs_spin)
        layout.addRow("Calcium timescale (s)", self.tau_spin)
        layout.addRow("Detection algorithm", self.algorithm_combo)
        layout.addRow("ROI diameter Y/X", diameter_row)
        layout.addRow("Threshold scaling", self.threshold_spin)
        layout.addRow("Maximum ROIs", self.max_rois_spin)
        layout.addRow("Spatial scale (0 = auto)", self.spatial_scale_spin)
        layout.addRow("Maximum overlap", self.max_overlap_spin)
        layout.addRow("Maximum binned frames", self.nbins_spin)
        layout.addRow("PCA denoising", self.denoise_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> ROIDetectionParameters:
        return ROIDetectionParameters(
            device=str(self.device_combo.currentData()),
            fs=self.fs_spin.value(),
            tau=self.tau_spin.value(),
            algorithm=self.algorithm_combo.currentText(),
            diameter=(self.diameter_y_spin.value(), self.diameter_x_spin.value()),
            threshold_scaling=self.threshold_spin.value(),
            max_rois=self.max_rois_spin.value(),
            spatial_scale=self.spatial_scale_spin.value(),
            max_overlap=self.max_overlap_spin.value(),
            nbins=self.nbins_spin.value(),
            denoise=self.denoise_check.isChecked(),
        )
