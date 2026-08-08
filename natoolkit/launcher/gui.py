from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial

from qtpy import QtCore, QtWidgets


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    module: str


TOOLS = (
    ToolSpec(
        "Activity Tracer",
        "Import imaging movies, manage ROIs, and export activity traces.",
        "natoolkit.tools.activity_tracer.gui",
    ),
    ToolSpec(
        "Sleep State Staging",
        "Classify EEG/EMG recordings and review Wake, NREM, and REM labels.",
        "natoolkit.tools.sleep_state_staging.gui",
    ),
    ToolSpec(
        "Activity–Sleep State Aligner",
        "Append sleep-state labels to activity traces using movie timestamps.",
        "natoolkit.tools.activity_sleep_state_aligner",
    ),
    ToolSpec(
        "AI Guider",
        "Ask project-scoped questions about usage, algorithms, and implementation.",
        "natoolkit.ai_guider",
    ),
)


def start_tool(tool: ToolSpec) -> bool:
    result = QtCore.QProcess.startDetached(sys.executable, ["-m", tool.module])
    return bool(result[0] if isinstance(result, tuple) else result)


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neural Activity Toolkit")
        self.resize(720, 520)

        title = QtWidgets.QLabel("Neural Activity Toolkit")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Choose a lab tool. Each application opens in an independent process."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtitle")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        for tool in TOOLS:
            layout.addWidget(self._tool_card(tool))

        layout.addStretch()
        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        self.setStyleSheet(_STYLE)

    def _tool_card(self, tool: ToolSpec) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("toolCard")
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        name = QtWidgets.QLabel(tool.name)
        name.setObjectName("toolName")
        description = QtWidgets.QLabel(tool.description)
        description.setWordWrap(True)
        description.setObjectName("toolDescription")

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.setSpacing(4)
        text_layout.addWidget(name)
        text_layout.addWidget(description)

        button = QtWidgets.QPushButton("Open")
        button.setMinimumWidth(92)
        button.clicked.connect(partial(self._launch, tool))

        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(18, 14, 16, 14)
        card_layout.addLayout(text_layout, 1)
        card_layout.addWidget(button)
        return card

    def _launch(self, tool: ToolSpec, _checked: bool = False) -> None:
        try:
            started = start_tool(tool)
        except Exception as exc:
            self._show_launch_error(tool, str(exc))
            return
        if not started:
            self._show_launch_error(tool, "The child process could not be started.")
            return
        self.status.setText(f"Opened {tool.name}.")

    def _show_launch_error(self, tool: ToolSpec, details: str) -> None:
        message = f"Unable to open {tool.name}: {details}"
        self.status.setText(message)
        QtWidgets.QMessageBox.critical(self, "Launch failed", message)


_STYLE = """
QWidget {
    background: #f4f6f8;
    color: #17202a;
    font-size: 13px;
}
QLabel#title {
    font-size: 25px;
    font-weight: 700;
}
QLabel#subtitle, QLabel#toolDescription, QLabel#status {
    color: #566573;
}
QFrame#toolCard {
    background: white;
    border: 1px solid #d5d8dc;
    border-radius: 8px;
}
QLabel#toolName {
    background: transparent;
    font-size: 16px;
    font-weight: 600;
}
QLabel#toolDescription {
    background: transparent;
}
QPushButton {
    background: #21618c;
    border: 0;
    border-radius: 5px;
    color: white;
    font-weight: 600;
    padding: 8px 16px;
}
QPushButton:hover {
    background: #2874a6;
}
QPushButton:pressed {
    background: #1b4f72;
}
"""


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
