from __future__ import annotations

import sys
from pathlib import Path

from qtpy import QtWidgets

from .core import align_activity_file


class ActivitySleepStateAlignerWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Activity–Sleep State Aligner")
        self.resize(760, 290)

        self.note_path = QtWidgets.QLineEdit()
        self.labels_path = QtWidgets.QLineEdit()
        self.activity_path = QtWidgets.QLineEdit()
        self.tiff_dir = QtWidgets.QLineEdit()
        self.output_path = QtWidgets.QLineEdit()
        self.status = QtWidgets.QLabel("Select the inputs and output path, then click Execute.")
        self.status.setWordWrap(True)

        form = QtWidgets.QGridLayout()
        self._add_file_row(form, 0, "Note.txt", self.note_path, "Text files (*.txt);;All files (*)")
        self._add_file_row(form, 1, "Sleep-state CSV", self.labels_path, "CSV files (*.csv)")
        self._add_file_row(form, 2, "Activity CSV", self.activity_path, "CSV files (*.csv)")
        self._add_directory_row(form, 3, "TIFF directory", self.tiff_dir)
        self._add_output_row(form, 4, "Output CSV", self.output_path)

        execute = QtWidgets.QPushButton("Execute")
        execute.clicked.connect(self.execute)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(execute)
        layout.addWidget(self.status)

        self.activity_path.textChanged.connect(self._suggest_output_path)

    def _add_file_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label: str,
        edit: QtWidgets.QLineEdit,
        file_filter: str,
    ) -> None:
        button = QtWidgets.QPushButton("Browse…")
        button.clicked.connect(lambda: self._choose_file(edit, file_filter))
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)

    def _add_directory_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label: str,
        edit: QtWidgets.QLineEdit,
    ) -> None:
        button = QtWidgets.QPushButton("Browse…")
        button.clicked.connect(lambda: self._choose_directory(edit))
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)

    def _add_output_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label: str,
        edit: QtWidgets.QLineEdit,
    ) -> None:
        button = QtWidgets.QPushButton("Browse…")
        button.clicked.connect(lambda: self._choose_output(edit))
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)

    def _choose_file(self, edit: QtWidgets.QLineEdit, file_filter: str) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", filter=file_filter)
        if path:
            edit.setText(path)

    def _choose_directory(self, edit: QtWidgets.QLineEdit) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select TIFF directory")
        if path:
            edit.setText(path)

    def _choose_output(self, edit: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save output", filter="CSV files (*.csv)")
        if path:
            edit.setText(path if path.lower().endswith(".csv") else f"{path}.csv")

    def _suggest_output_path(self, activity_path: str) -> None:
        if not activity_path or self.output_path.text().strip():
            return
        path = Path(activity_path)
        self.output_path.setText(str(path.with_name(f"{path.stem}_sleep_state.csv")))

    def execute(self) -> None:
        values = {
            "Note.txt": self.note_path.text().strip(),
            "sleep-state CSV": self.labels_path.text().strip(),
            "activity CSV": self.activity_path.text().strip(),
            "TIFF directory": self.tiff_dir.text().strip(),
            "output CSV": self.output_path.text().strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            self._show_error(f"Select: {', '.join(missing)}")
            return

        try:
            self.status.setText("Aligning activity rows…")
            QtWidgets.QApplication.processEvents()
            result = align_activity_file(
                values["Note.txt"],
                values["sleep-state CSV"],
                values["activity CSV"],
                values["TIFF directory"],
                values["output CSV"],
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        message = f"Complete: wrote {result.row_count} rows to {result.output_path}."
        if result.unknown_count:
            message += f" {result.unknown_count} rows have sleep_state=Unknown."
        self.status.setText(message)
        QtWidgets.QMessageBox.information(self, "Alignment complete", message)

    def _show_error(self, message: str) -> None:
        self.status.setText(message)
        QtWidgets.QMessageBox.critical(self, "Alignment failed", message)


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = ActivitySleepStateAlignerWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
