from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

from .client import DEFAULT_MODEL, DeepSeekClient
from .knowledge import KnowledgeError, build_context, ground_answer
from .router import RouteError


MAX_QUESTION_CHARS = 4_000
OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions about Neural Activity Toolkit, its applications, "
    "documented workflows, file formats, algorithms, and implementation."
)
ROUTER_FAILURE_RESPONSE = (
    "I could not verify that this question is within the Neural Activity Toolkit "
    "scope, so I did not generate an answer."
)
MARKDOWN_SPECIAL_CHARS = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|>])")


@dataclass
class ChatMessage:
    role: str
    content: str


def _escape_markdown(text: str) -> str:
    """Return user-provided text that Markdown renders literally."""
    escaped = MARKDOWN_SPECIAL_CHARS.sub(r"\\\1", text)
    return escaped.replace("\n", "  \n")


def _transcript_markdown(messages: list[ChatMessage]) -> str:
    if not messages:
        return (
            "# AI Guider\n\n"
            "I answer only Neural Activity Toolkit questions and cite the "
            "approved documentation or source code used for each answer."
        )

    blocks = []
    for message in messages:
        author = "You" if message.role == "user" else "AI Guider"
        content = message.content or "…"
        if message.role == "user":
            content = _escape_markdown(content)
        blocks.append(f"## {author}\n\n{content}")
    return "\n\n---\n\n".join(blocks)


class PromptEdit(QtWidgets.QPlainTextEdit):
    submit_requested = QtCore.Signal()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        submit_modifier = (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        if (
            event.key()
            in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)
            and event.modifiers() & submit_modifier
        ):
            self.submit_requested.emit()
            return
        super().keyPressEvent(event)


class AskWorker(QtCore.QObject):
    chunk = QtCore.Signal(str)
    status = QtCore.Signal(str)
    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(
        self, question: str, history: list[tuple[str, str]]
    ) -> None:
        super().__init__()
        self.question = question
        self.history = history
        self._cancelled = threading.Event()

    @QtCore.Slot()
    def run(self) -> None:
        client: DeepSeekClient | None = None
        try:
            client = DeepSeekClient.from_environment()
            self.status.emit("Classifying the question…")
            route = client.route(self.question, self.history)
            if self._cancelled.is_set():
                self.completed.emit("Request cancelled.")
                return
            if not route.in_scope:
                self.completed.emit(OUT_OF_SCOPE_RESPONSE)
                return

            self.status.emit("Loading approved project context…")
            context = build_context(route)
            self.status.emit("Generating a source-grounded answer…")
            chunks: list[str] = []
            for chunk in client.stream_answer(self.question, self.history, context):
                if self._cancelled.is_set():
                    self.completed.emit("Request cancelled.")
                    return
                chunks.append(chunk)
                self.chunk.emit(chunk)
            self.completed.emit(ground_answer("".join(chunks), context))
        except RouteError:
            self.completed.emit(ROUTER_FAILURE_RESPONSE)
        except KnowledgeError as exc:
            self.completed.emit(f"I could not find approved project evidence: {exc}")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if client is not None:
                client.close()

    def cancel(self) -> None:
        self._cancelled.set()


class AIGuiderWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neural Activity Toolkit — AI Guider")
        self.resize(900, 680)
        self.messages: list[ChatMessage] = []
        self._thread: QtCore.QThread | None = None
        self._worker: AskWorker | None = None
        self._build_ui()
        self._render_transcript()
        self._show_initial_status()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("AI Guider")
        title.setStyleSheet("font-size: 22px; font-weight: 700")
        description = QtWidgets.QLabel(
            "Ask how to use the toolkit or how its documented algorithms and code work."
        )
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        self.transcript = QtWidgets.QTextBrowser()
        self.transcript.setReadOnly(True)
        self.transcript.setOpenExternalLinks(False)
        self.transcript.setOpenLinks(False)
        self.transcript.setPlaceholderText("No conversation yet.")
        self.transcript.document().setDefaultStyleSheet(
            "body { color: #243447; } "
            "h1, h2 { color: #17202a; } "
            "code { color: #7d3c98; background-color: #f4f6f7; } "
            "pre { background-color: #f4f6f7; margin: 8px; padding: 8px; } "
            "blockquote { color: #566573; border-left: 3px solid #aab7b8; }"
        )
        layout.addWidget(self.transcript, 1)

        self.prompt = PromptEdit()
        self.prompt.setPlaceholderText(
            "Ask a project question. Press Ctrl+Enter or Cmd+Enter to send."
        )
        self.prompt.setMaximumHeight(120)
        self.prompt.submit_requested.connect(self.send_question)
        layout.addWidget(self.prompt)

        buttons = QtWidgets.QHBoxLayout()
        self.send_button = QtWidgets.QPushButton("Send")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.new_button = QtWidgets.QPushButton("New Conversation")
        self.send_button.clicked.connect(self.send_question)
        self.stop_button.clicked.connect(self.stop_request)
        self.new_button.clicked.connect(self.new_conversation)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.new_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #566573")
        layout.addWidget(self.status)

    def _show_initial_status(self) -> None:
        model = os.environ.get("NATOOLKIT_DEEPSEEK_MODEL", DEFAULT_MODEL)
        if os.environ.get("DEEPSEEK_API_KEY", "").strip():
            self.status.setText(f"Ready · model: {model}")
        else:
            self.status.setText(
                f"Set DEEPSEEK_API_KEY and restart AI Guider · model: {model}"
            )

    @QtCore.Slot()
    def send_question(self) -> None:
        if self._thread is not None:
            return
        question = self.prompt.toPlainText().strip()
        if not question:
            return
        if len(question) > MAX_QUESTION_CHARS:
            QtWidgets.QMessageBox.warning(
                self,
                "Question too long",
                f"Questions are limited to {MAX_QUESTION_CHARS} characters.",
            )
            return
        if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            QtWidgets.QMessageBox.information(
                self,
                "DeepSeek API key required",
                "Set DEEPSEEK_API_KEY in the environment and restart AI Guider.",
            )
            return

        history = [(message.role, message.content) for message in self.messages[-8:]]
        self.messages.append(ChatMessage("user", question))
        self.messages.append(ChatMessage("assistant", ""))
        self.prompt.clear()
        self._render_transcript()
        self._set_busy(True)

        self._thread = QtCore.QThread(self)
        self._worker = AskWorker(question, history)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.chunk.connect(self._append_chunk)
        self._worker.status.connect(self.status.setText)
        self._worker.completed.connect(self._complete_answer)
        self._worker.failed.connect(self._fail_answer)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.completed.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._request_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @QtCore.Slot(str)
    def _append_chunk(self, chunk: str) -> None:
        if self.messages and self.messages[-1].role == "assistant":
            self.messages[-1].content += chunk
            self._render_transcript()

    @QtCore.Slot(str)
    def _complete_answer(self, answer: str) -> None:
        if self.messages and self.messages[-1].role == "assistant":
            self.messages[-1].content = answer
        self.status.setText("Ready")
        self._render_transcript()

    @QtCore.Slot(str)
    def _fail_answer(self, error: str) -> None:
        if self.messages and self.messages[-1].role == "assistant":
            self.messages[-1].content = f"Unable to complete the request: {error}"
        self.status.setText("Request failed")
        self._render_transcript()

    @QtCore.Slot()
    def _request_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def stop_request(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping after the current API operation…")

    def new_conversation(self) -> None:
        if self._thread is not None:
            return
        self.messages.clear()
        self._render_transcript()
        self._show_initial_status()

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.prompt.setEnabled(not busy)
        self.stop_button.setEnabled(busy)

    def _render_transcript(self) -> None:
        self.transcript.document().setMarkdown(
            _transcript_markdown(self.messages),
            QtGui.QTextDocument.MarkdownDialectGitHub,
        )
        scrollbar = self.transcript.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = AIGuiderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
