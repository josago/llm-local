import html
import math
import os
import re
import sys
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QTextDocument, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm import ensure_ollama_server, list_installed_ollama_models, stream_ollama_chat

_HTML_LIKE_RE = re.compile(r"^(<!DOCTYPE html|<html\b|<body\b|<[a-zA-Z][\w:-]*(\s|>|/))")
_BODY_RE = re.compile(r"<body[^>]*>(.*)</body>", flags=re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body([^>]*)>", flags=re.IGNORECASE)
_STYLE_TAG_RE = re.compile(r"<style[^>]*>.*?</style>", flags=re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RE = re.compile(r"""style\s*=\s*(['"])(.*?)\1""", flags=re.IGNORECASE | re.DOTALL)
_LOADING_FRAMES = (
    "Thinking ⠋",
    "Thinking ⠙",
    "Thinking ⠹",
    "Thinking ⠸",
    "Thinking ⠼",
    "Thinking ⠴",
    "Thinking ⠦",
    "Thinking ⠧",
    "Thinking ⠇",
    "Thinking ⠏",
)
_USER_BUBBLE_STYLE = (
    "background: #edf3ff; border: 1px solid #ccdbff; border-radius: 18px; padding: 12px 14px;"
)
_ASSISTANT_BUBBLE_STYLE = "padding: 8px 10px;"
_NOTICE_BUBBLE_STYLE = "padding: 6px 8px; color: #666666;"


class ServerWarmupWorker(QThread):
    ready = pyqtSignal(list)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            ensure_ollama_server()
            self.ready.emit(list_installed_ollama_models())
        except Exception as exc:
            self.failed.emit(str(exc))


class OllamaPromptWorker(QThread):
    chunk = pyqtSignal(str, str)
    completed = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, model_name: str, messages: list[dict[str, str]]) -> None:
        super().__init__()
        self.model_name = model_name
        self.messages = messages

    def run(self) -> None:
        try:
            for chunk_type, chunk_text in stream_ollama_chat(
                model_name=self.model_name,
                messages=self.messages,
            ):
                self.chunk.emit(chunk_type, chunk_text)
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class AutoHeightTextBrowser(QTextBrowser):
    _HEIGHT_PADDING = 1
    _MIN_HEIGHT = 28

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_adjusting_height = False
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportMargins(0, 0, 0, 0)
        self.viewport().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.document().setDocumentMargin(0)
        text_option = self.document().defaultTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.document().setDefaultTextOption(text_option)
        self.document().documentLayout().documentSizeChanged.connect(self._adjust_height)

    def setHtml(self, html_text: str) -> None:
        wrapped_html = (
            "<html><head><meta charset='utf-8'/>"
            "<style>html, body { margin: 0; padding: 0; width: 100%; }</style>"
            "</head><body>"
            f"<div style='width: 100%; margin: 0; padding: 0;'>{html_text}</div>"
            "</body></html>"
        )
        super().setHtml(wrapped_html)
        self._adjust_height()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        event.ignore()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._adjust_height()

    def _adjust_height(self, *_args) -> None:
        if self._is_adjusting_height:
            return

        self._is_adjusting_height = True
        try:
            # Keep QTextBrowser in control of wrap width; we only auto-size height.
            doc_height = self.document().documentLayout().documentSize().height()
            chrome_height = max(0, self.height() - self.viewport().height())
            height = math.ceil(doc_height + chrome_height + self._HEIGHT_PADDING)
            self.setFixedHeight(max(self._MIN_HEIGHT, height))
        finally:
            self._is_adjusting_height = False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.default_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.available_models: list[str] = []
        self.chat_messages: list[dict[str, str]] = []
        self.response_markdown = ""
        self.thinking_markdown = ""
        self.current_assistant_browser: Optional[AutoHeightTextBrowser] = None
        self.server_ready = False
        self.server_worker: Optional[ServerWarmupWorker] = None
        self.prompt_worker: Optional[OllamaPromptWorker] = None
        self.query_running = False
        self.send_icon = self._theme_icon("mail-send", QStyle.StandardPixmap.SP_ArrowRight)
        self.loading_frames = _LOADING_FRAMES
        self.loading_frame_index = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(120)
        self.loading_timer.timeout.connect(self._animate_send_button)

        self.setWindowTitle("llm-local")
        self.resize(900, 700)
        self._build_ui()
        self._start_server_warmup()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        model_row = QWidget(root)
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)

        model_label = QLabel("Model", model_row)
        model_label.setStyleSheet("font-size: 12px; color: #6f6f6f;")
        model_layout.addWidget(model_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.model_combo = QComboBox(model_row)
        self.model_combo.addItem("Loading installed models...")
        self.model_combo.setEnabled(False)
        self.model_combo.currentTextChanged.connect(self._update_send_button_state)
        model_layout.addWidget(self.model_combo, 1)

        layout.addWidget(model_row, 0)

        self.chat_scroll = QScrollArea(root)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.chat_container = QWidget(self.chat_scroll)
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        input_row = QWidget(root)
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_input = QTextEdit(input_row)
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setFixedHeight(92)
        self.prompt_input.setPlaceholderText("Write your Markdown message here and click send.")
        self.prompt_input.textChanged.connect(self._update_send_button_state)
        input_layout.addWidget(self.prompt_input, 1)

        self.send_button = QPushButton("Send", input_row)
        self.send_button.setIcon(self.send_icon)
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._send_prompt)
        input_layout.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)

        layout.addWidget(input_row, 0, Qt.AlignmentFlag.AlignBottom)

        self.setCentralWidget(root)

    def _start_server_warmup(self) -> None:
        self.server_worker = ServerWarmupWorker()
        self.server_worker.ready.connect(self._on_server_ready)
        self.server_worker.failed.connect(self._on_server_failed)
        self.server_worker.start()

    def _on_server_ready(self, model_names: list[str]) -> None:
        self.server_ready = True
        self._set_model_choices(model_names)
        self._update_send_button_state()

    def _on_server_failed(self, error_message: str) -> None:
        self.server_ready = False
        self.available_models = []
        self.model_combo.clear()
        self.model_combo.addItem("No models available")
        self._sync_model_selector_state()
        self._set_query_running(False)
        self._append_notice(f"**Server startup failed**\n\n```\n{error_message}\n```")

    def _send_prompt(self) -> None:
        if self.prompt_worker and self.prompt_worker.isRunning():
            return
        if not self.server_ready and self.server_worker and self.server_worker.isRunning():
            self._append_notice("_Ollama is still starting, please wait._")
            return

        prompt_markdown = self.prompt_input.toPlainText().strip()
        if not prompt_markdown:
            return

        model_name = self._selected_model_name()
        if not model_name:
            self._append_notice(
                "**No installed model selected.**\n\nPull one with `ollama pull <model>` and restart."
            )
            return

        self._append_user_message(prompt_markdown)
        self.chat_messages.append({"role": "user", "content": prompt_markdown})

        self._reset_active_response()
        self.prompt_input.clear()

        self.current_assistant_browser = self._append_assistant_message("")
        self._update_current_assistant_message()
        self._set_query_running(True)

        conversation_snapshot = [dict(message) for message in self.chat_messages]
        self.prompt_worker = OllamaPromptWorker(model_name, conversation_snapshot)
        self.prompt_worker.chunk.connect(self._on_prompt_chunk)
        self.prompt_worker.completed.connect(self._on_prompt_completed)
        self.prompt_worker.failed.connect(self._on_prompt_failed)
        self.prompt_worker.finished.connect(self._on_prompt_finished)
        self.prompt_worker.start()

    def _on_prompt_chunk(self, chunk_type: str, chunk_text: str) -> None:
        if chunk_type == "thinking":
            self.thinking_markdown += chunk_text
        else:
            self.response_markdown += chunk_text
        self._update_current_assistant_message()

    def _on_prompt_completed(self) -> None:
        assistant_message = self.response_markdown.strip()
        if assistant_message:
            self.chat_messages.append({"role": "assistant", "content": assistant_message})
        self._update_current_assistant_message(final=True)
        self._reset_active_response(clear_browser=True)

    def _on_prompt_failed(self, error_message: str) -> None:
        self.response_markdown = f"**Request failed**\n\n```\n{error_message}\n```"
        self._update_current_assistant_message(final=True)
        self._reset_active_response(clear_browser=True)

    def _on_prompt_finished(self) -> None:
        self._set_query_running(False)

    def _set_query_running(self, is_running: bool) -> None:
        self.query_running = is_running
        self._sync_model_selector_state()
        if is_running:
            self.loading_frame_index = 0
            self.send_button.setIcon(QIcon())
            self._animate_send_button()
            self.loading_timer.start()
            self.send_button.setEnabled(False)
            return

        self.loading_timer.stop()
        self.send_button.setIcon(self.send_icon)
        self.send_button.setText("Send")
        self._update_send_button_state()

    def _update_send_button_state(self) -> None:
        has_text = bool(self.prompt_input.toPlainText().strip())
        has_model = bool(self._selected_model_name())
        can_send = self.server_ready and has_model and has_text and not self.query_running
        self.send_button.setEnabled(can_send)

    def _sync_model_selector_state(self) -> None:
        self.model_combo.setEnabled(
            self.server_ready and bool(self.available_models) and not self.query_running
        )

    def _selected_model_name(self) -> str:
        selected = self.model_combo.currentText().strip()
        return selected if selected in self.available_models else ""

    def _set_model_choices(self, model_names: list[str]) -> None:
        self.available_models = model_names

        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            if model_names:
                self.model_combo.addItems(model_names)
                preferred = self.default_model if self.default_model in model_names else model_names[0]
                self.model_combo.setCurrentText(preferred)
            else:
                self.model_combo.addItem("No installed models found")
        finally:
            self.model_combo.blockSignals(False)

        self._sync_model_selector_state()

        if not model_names:
            self._append_notice(
                "**No local Ollama models found.**\n\nRun `ollama pull <model>` and restart the app."
            )
        elif self.default_model not in model_names:
            self._append_notice(
                f"_Default model `{self.default_model}` is not installed. Using `{self._selected_model_name()}`._"
            )

    def _animate_send_button(self) -> None:
        frame = self.loading_frames[self.loading_frame_index % len(self.loading_frames)]
        self.send_button.setText(frame)
        self.loading_frame_index += 1

    def _theme_icon(self, theme_name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon.fromTheme(theme_name)
        if icon.isNull():
            return self.style().standardIcon(fallback)
        return icon

    def _reset_active_response(self, clear_browser: bool = False) -> None:
        self.response_markdown = ""
        self.thinking_markdown = ""
        if clear_browser:
            self.current_assistant_browser = None

    def _add_message_row(
        self,
        role_label: str,
        html_content: str,
        bubble_style: str,
    ) -> AutoHeightTextBrowser:
        row = QWidget(self.chat_container)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        label = QLabel(role_label, row)
        label.setStyleSheet("font-size: 12px; color: #6f6f6f;")
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        row_layout.addWidget(label)

        body = AutoHeightTextBrowser(row)
        body.setStyleSheet(f"QTextBrowser {{{bubble_style}}}")
        body.setHtml(html_content)
        row_layout.addWidget(body)

        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        QTimer.singleShot(0, self._scroll_response_to_bottom)
        return body

    def _append_user_message(self, prompt_text: str) -> None:
        self._add_message_row(
            role_label="User",
            html_content=self._escaped_plain_text_html(prompt_text),
            bubble_style=_USER_BUBBLE_STYLE,
        )

    def _append_assistant_message(self, html_content: str) -> AutoHeightTextBrowser:
        return self._add_message_row(
            role_label="Assistant",
            html_content=html_content,
            bubble_style=_ASSISTANT_BUBBLE_STYLE,
        )

    def _append_notice(self, markdown_text: str) -> None:
        self._add_message_row(
            role_label="System",
            html_content=self._markdown_to_html_fragment(markdown_text),
            bubble_style=_NOTICE_BUBBLE_STYLE,
        )

    def _update_current_assistant_message(self, final: bool = False) -> None:
        if self.current_assistant_browser is None:
            return

        thinking = self.thinking_markdown.strip()
        answer = self.response_markdown.strip()
        html_blocks: list[str] = []

        if thinking:
            html_blocks.append(
                self._escaped_plain_text_html(
                    thinking,
                    style="color: #8a8a8a; font-style: italic; white-space: pre-wrap;",
                )
            )

        if answer:
            html_blocks.append(self._renderable_answer_html(answer))

        if not html_blocks:
            if final:
                html_blocks.append(
                    self._markdown_to_html_fragment("_Model returned an empty response._")
                )
            else:
                html_blocks.append(
                    self._escaped_plain_text_html(
                        "Thinking",
                        style="color: #8a8a8a; font-style: italic; white-space: pre-wrap;",
                    )
                )

        joined_html = "<div style='margin-top: 10px;'></div>".join(html_blocks)
        self.current_assistant_browser.setHtml(joined_html)
        QTimer.singleShot(0, self._scroll_response_to_bottom)

    def _scroll_response_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _renderable_answer_html(self, answer_text: str) -> str:
        return (
            self._html_to_fragment(answer_text)
            if self._looks_like_html(answer_text)
            else self._markdown_to_html_fragment(answer_text)
        )

    def _escaped_plain_text_html(
        self, text: str, style: str = "white-space: pre-wrap; margin: 0; width: 100%;"
    ) -> str:
        return f"<div style='{style}'>{html.escape(text)}</div>"

    def _looks_like_html(self, text: str) -> bool:
        return bool(_HTML_LIKE_RE.match(text.lstrip()))

    def _markdown_to_html_fragment(self, markdown_text: str) -> str:
        doc = QTextDocument()
        doc.setMarkdown(markdown_text)
        return self._html_to_fragment(doc.toHtml())

    def _html_to_fragment(self, html_text: str) -> str:
        body_match = _BODY_RE.search(html_text)
        if body_match:
            body_fragment = body_match.group(1)
            body_tag_match = _BODY_TAG_RE.search(html_text)
            body_style = ""
            if body_tag_match:
                style_match = _STYLE_ATTR_RE.search(body_tag_match.group(1))
                if style_match:
                    body_style = style_match.group(2).strip()
            if body_style:
                escaped_style = html.escape(body_style, quote=True)
                body_fragment = f"<div style=\"{escaped_style}\">{body_fragment}</div>"
            style_blocks = "".join(_STYLE_TAG_RE.findall(html_text))
            return f"{style_blocks}{body_fragment}"
        return html_text


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
