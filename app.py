import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QTextDocument, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QStyle,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm import (
    ensure_ollama_server,
    get_ollama_model_configuration,
    list_installed_ollama_models,
    stream_ollama_chat,
)

_HTML_LIKE_RE = re.compile(r"^(<!DOCTYPE html|<html\b|<body\b|<[a-zA-Z][\w:-]*(\s|>|/))")
_BODY_RE = re.compile(r"<body[^>]*>(.*)</body>", flags=re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body([^>]*)>", flags=re.IGNORECASE)
_STYLE_TAG_RE = re.compile(r"<style[^>]*>.*?</style>", flags=re.IGNORECASE | re.DOTALL)
_STYLE_ATTR_RE = re.compile(r"""style\s*=\s*(['"])(.*?)\1""", flags=re.IGNORECASE | re.DOTALL)
_BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
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
_PERSISTED_THREAD_PATH = Path(__file__).resolve().parent / "threads" / "conversation.json"
_VALID_CHAT_ROLES = {"user", "assistant"}


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
        self.persisted_thread_path = _PERSISTED_THREAD_PATH
        self.available_models: list[str] = []
        self.model_config_cache: dict[str, dict[str, list[str]]] = {}
        self.chat_messages: list[dict[str, str]] = []
        self.active_response_model_name = ""
        self.response_markdown = ""
        self.thinking_markdown = ""
        self.current_assistant_browser: Optional[AutoHeightTextBrowser] = None
        self.server_ready = False
        self.server_worker: Optional[ServerWarmupWorker] = None
        self.prompt_worker: Optional[OllamaPromptWorker] = None
        self.query_running = False
        self.send_icon = self._theme_icon("mail-send", QStyle.StandardPixmap.SP_ArrowRight)
        self.hide_sidebar_icon = self._theme_icon("go-previous", QStyle.StandardPixmap.SP_ArrowLeft)
        self.show_sidebar_icon = self._theme_icon("go-next", QStyle.StandardPixmap.SP_ArrowRight)
        self.loading_frames = _LOADING_FRAMES
        self.loading_frame_index = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(120)
        self.loading_timer.timeout.connect(self._animate_send_button)
        self.sidebar_splitter: Optional[QSplitter] = None
        self.sidebar_toggle_button: Optional[QPushButton] = None
        self.thread_list: Optional[QListWidget] = None
        self.sidebar_last_width = 240
        self.sidebar_expanded = True

        recent_threads = self._list_thread_paths_by_recent_update()
        if recent_threads:
            self.persisted_thread_path = recent_threads[0]

        self.setWindowTitle("llm-local")
        self.resize(900, 700)
        self._build_ui()
        self._refresh_thread_list()
        self._load_persisted_conversation()
        self._start_server_warmup()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        self.sidebar_splitter = QSplitter(Qt.Orientation.Horizontal, root)
        self.sidebar_splitter.setChildrenCollapsible(False)

        sidebar_panel = QWidget(self.sidebar_splitter)
        sidebar_panel.setMinimumWidth(0)
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(6)

        sidebar_header = QLabel("Threads", sidebar_panel)
        sidebar_header.setStyleSheet("font-size: 12px; color: #6f6f6f;")
        sidebar_layout.addWidget(sidebar_header)
        self.thread_list = QListWidget(sidebar_panel)
        self.thread_list.setFrameShape(QFrame.Shape.NoFrame)
        self.thread_list.setSpacing(4)
        self.thread_list.setStyleSheet(
            "QListWidget::item { padding-top: 8px; padding-bottom: 8px; }"
        )
        self.thread_list.currentTextChanged.connect(self._on_thread_selected)
        sidebar_layout.addWidget(self.thread_list, 1)

        main_panel = QWidget(self.sidebar_splitter)
        layout = QVBoxLayout(main_panel)

        self.chat_scroll = QScrollArea(main_panel)
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.chat_container = QWidget(self.chat_scroll)
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, 1)

        input_row = QWidget(main_panel)
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self.prompt_input = QTextEdit(input_row)
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setFixedHeight(92)
        self.prompt_input.setPlaceholderText("Write your Markdown message here and click send.")
        self.prompt_input.textChanged.connect(self._update_send_button_state)
        input_layout.addWidget(self.prompt_input, 1)

        layout.addWidget(input_row, 0)

        controls_row = QWidget(main_panel)
        controls_layout = QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.sidebar_toggle_button = QPushButton("Hide threads", controls_row)
        self.sidebar_toggle_button.clicked.connect(self._toggle_sidebar)
        self._update_sidebar_toggle_button()
        controls_layout.addWidget(self.sidebar_toggle_button, 0, Qt.AlignmentFlag.AlignVCenter)

        model_label = QLabel("Model: ", controls_row)
        model_label.setStyleSheet("font-size: 12px; color: #6f6f6f;")
        controls_layout.addWidget(model_label, 0, Qt.AlignmentFlag.AlignVCenter)

        self.model_combo = QComboBox(controls_row)
        self.model_combo.addItem("Loading installed models...")
        self.model_combo.setEnabled(False)
        self.model_combo.currentTextChanged.connect(self._on_model_selection_changed)
        controls_layout.addWidget(self.model_combo, 1)

        self.send_button = QPushButton("Send", controls_row)
        self.send_button.setIcon(self.send_icon)
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._send_prompt)
        controls_layout.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(controls_row, 0, Qt.AlignmentFlag.AlignBottom)

        self.sidebar_splitter.addWidget(sidebar_panel)
        self.sidebar_splitter.addWidget(main_panel)
        self.sidebar_splitter.setStretchFactor(0, 0)
        self.sidebar_splitter.setStretchFactor(1, 1)
        self.sidebar_splitter.setCollapsible(0, True)
        self.sidebar_splitter.setCollapsible(1, False)
        self.sidebar_splitter.setSizes([self.sidebar_last_width, 900 - self.sidebar_last_width])
        self.sidebar_splitter.splitterMoved.connect(self._on_sidebar_splitter_moved)

        root_layout.addWidget(self.sidebar_splitter, 1)
        self.setCentralWidget(root)

    def _toggle_sidebar(self) -> None:
        if self.sidebar_splitter is None:
            return

        sizes = self.sidebar_splitter.sizes()
        if not sizes or len(sizes) < 2:
            return

        if self.sidebar_expanded:
            if sizes[0] > 0:
                self.sidebar_last_width = sizes[0]
            self.sidebar_splitter.setSizes([0, max(1, sum(sizes))])
            self.sidebar_expanded = False
        else:
            target_width = max(160, self.sidebar_last_width)
            total_width = max(1, sum(sizes))
            main_width = max(1, total_width - target_width)
            self.sidebar_splitter.setSizes([target_width, main_width])
            self.sidebar_expanded = True

        self._update_sidebar_toggle_button()

    def _on_sidebar_splitter_moved(self, _pos: int, _index: int) -> None:
        if self.sidebar_splitter is None:
            return

        sizes = self.sidebar_splitter.sizes()
        if not sizes or len(sizes) < 2:
            return

        sidebar_width = sizes[0]
        if sidebar_width > 0:
            self.sidebar_last_width = sidebar_width

        is_expanded = sidebar_width > 0
        if is_expanded != self.sidebar_expanded:
            self.sidebar_expanded = is_expanded
            self._update_sidebar_toggle_button()

    def _update_sidebar_toggle_button(self) -> None:
        if self.sidebar_toggle_button is None:
            return
        if self.sidebar_expanded:
            self.sidebar_toggle_button.setText("Hide threads")
            self.sidebar_toggle_button.setIcon(self.hide_sidebar_icon)
            return
        self.sidebar_toggle_button.setText("Show threads")
        self.sidebar_toggle_button.setIcon(self.show_sidebar_icon)

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
        self._save_persisted_conversation()

        self._reset_active_response()
        self.active_response_model_name = model_name
        self.prompt_input.clear()

        self.current_assistant_browser = self._append_assistant_message("", model_name=model_name)
        self._update_current_assistant_message()
        self._set_query_running(True)

        conversation_snapshot = [
            {"role": message["role"], "content": message["content"]}
            for message in self.chat_messages
        ]
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
        assistant_thinking = self.thinking_markdown.strip()
        if assistant_message or assistant_thinking:
            message_record = {"role": "assistant", "content": assistant_message}
            if assistant_thinking:
                message_record["thinking"] = assistant_thinking
            model_name = self.active_response_model_name.strip()
            if model_name:
                message_record["model"] = model_name
            self.chat_messages.append(message_record)
        self._save_persisted_conversation()
        self._update_current_assistant_message(final=True)
        self._reset_active_response(clear_browser=True)

    def _on_prompt_failed(self, error_message: str) -> None:
        self.response_markdown = f"**Request failed**\n\n```\n{error_message}\n```"
        self._save_persisted_conversation()
        self._update_current_assistant_message(final=True)
        self._reset_active_response(clear_browser=True)

    def _on_prompt_finished(self) -> None:
        self._set_query_running(False)

    def _on_model_selection_changed(self, _selected_text: str) -> None:
        self._update_send_button_state()
        self._print_selected_model_configuration()

    def _set_query_running(self, is_running: bool) -> None:
        self.query_running = is_running
        self._sync_model_selector_state()
        if self.thread_list is not None:
            self.thread_list.setEnabled(not is_running)
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
        self.model_config_cache = {
            name: config for name, config in self.model_config_cache.items() if name in model_names
        }

        self.model_combo.blockSignals(True)
        try:
            self.model_combo.clear()
            if model_names:
                self.model_combo.addItems(model_names)
                self.model_combo.setCurrentIndex(0)
            else:
                self.model_combo.addItem("No installed models found")
        finally:
            self.model_combo.blockSignals(False)

        self._sync_model_selector_state()
        self._print_selected_model_configuration()

        if not model_names:
            self._append_notice(
                "**No local Ollama models found.**\n\nRun `ollama pull <model>` and restart the app."
            )

    def _animate_send_button(self) -> None:
        frame = self.loading_frames[self.loading_frame_index % len(self.loading_frames)]
        self.send_button.setText(frame)
        self.loading_frame_index += 1

    def _load_persisted_conversation(self) -> None:
        self._reset_active_response(clear_browser=True)
        self.chat_messages = []
        self._clear_chat_view()

        if not self.persisted_thread_path.exists():
            return

        try:
            with self.persisted_thread_path.open("r", encoding="utf-8") as handle:
                persisted_text = handle.read()
        except OSError as exc:
            print(
                f"Failed to read persisted conversation from {self.persisted_thread_path}: {exc}",
                file=sys.stderr,
            )
            return

        if not persisted_text.strip():
            return

        try:
            loaded_payload = json.loads(persisted_text)
        except json.JSONDecodeError as exc:
            print(
                f"Failed to read persisted conversation from {self.persisted_thread_path}: {exc}",
                file=sys.stderr,
            )
            return

        loaded_messages = self._coerce_persisted_messages(loaded_payload)
        if loaded_messages is None:
            print(
                f"Persisted conversation at {self.persisted_thread_path} has an invalid format.",
                file=sys.stderr,
            )
            return

        self.chat_messages = loaded_messages
        for message in self.chat_messages:
            if message["role"] == "user":
                self._append_user_message(message["content"])
                continue
            self._append_assistant_message(
                html_content=self._assistant_response_html(
                    thinking=message.get("thinking", ""),
                    answer=message["content"],
                    final=True,
                ),
                model_name=message.get("model", ""),
            )
        QTimer.singleShot(0, self._scroll_response_to_bottom)

    def _coerce_persisted_messages(self, payload: Any) -> Optional[list[dict[str, str]]]:
        if isinstance(payload, dict):
            payload = payload.get("messages")
        if not isinstance(payload, list):
            return None

        normalized_messages: list[dict[str, str]] = []
        for message in payload:
            if not isinstance(message, dict):
                return None
            role = message.get("role")
            content = message.get("content")
            if role not in _VALID_CHAT_ROLES or not isinstance(content, str):
                return None
            normalized_message = {"role": role, "content": content}
            if role == "assistant":
                model_name = message.get("model")
                if model_name is not None and not isinstance(model_name, str):
                    return None
                if isinstance(model_name, str) and model_name.strip():
                    normalized_message["model"] = model_name.strip()
                thinking = message.get("thinking")
                if thinking is not None and not isinstance(thinking, str):
                    return None
                if isinstance(thinking, str) and thinking.strip():
                    normalized_message["thinking"] = thinking.strip()
            normalized_messages.append(normalized_message)
        return normalized_messages

    def _save_persisted_conversation(self) -> None:
        try:
            self.persisted_thread_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persisted_thread_path.open("w", encoding="utf-8") as handle:
                json.dump({"messages": self.chat_messages}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            self._refresh_thread_list()
        except OSError as exc:
            print(
                f"Failed to write persisted conversation to {self.persisted_thread_path}: {exc}",
                file=sys.stderr,
            )

    def _refresh_thread_list(self) -> None:
        if self.thread_list is None:
            return

        threads_dir = self.persisted_thread_path.parent
        thread_paths = self._list_thread_paths_by_recent_update()
        thread_names = [path.stem for path in thread_paths]
        active_thread_name = self.persisted_thread_path.stem
        if thread_names and active_thread_name not in thread_names:
            active_thread_name = thread_names[0]
            self.persisted_thread_path = threads_dir / f"{active_thread_name}.json"

        self.thread_list.blockSignals(True)
        try:
            self.thread_list.clear()
            self.thread_list.addItems(thread_names)
            if active_thread_name in thread_names:
                self.thread_list.setCurrentRow(thread_names.index(active_thread_name))
        finally:
            self.thread_list.blockSignals(False)

    def _on_thread_selected(self, thread_name: str) -> None:
        clean_thread_name = thread_name.strip()
        if not clean_thread_name:
            return

        selected_path = self.persisted_thread_path.parent / f"{clean_thread_name}.json"
        if selected_path == self.persisted_thread_path and self.chat_messages:
            return

        self.persisted_thread_path = selected_path
        self._load_persisted_conversation()

    def _list_thread_paths_by_recent_update(self) -> list[Path]:
        threads_dir = self.persisted_thread_path.parent
        try:
            thread_paths = [
                path
                for path in threads_dir.iterdir()
                if path.is_file() and path.name.endswith(".json")
            ]
        except OSError:
            return []

        def _sort_key(path: Path) -> tuple[float, str]:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = float("-inf")
            return (-mtime, path.stem.lower())

        return sorted(thread_paths, key=_sort_key)

    def _theme_icon(self, theme_name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon = QIcon.fromTheme(theme_name)
        if icon.isNull():
            return self.style().standardIcon(fallback)
        return icon

    def _reset_active_response(self, clear_browser: bool = False) -> None:
        self.response_markdown = ""
        self.thinking_markdown = ""
        self.active_response_model_name = ""
        if clear_browser:
            self.current_assistant_browser = None

    def _clear_chat_view(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

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

    def _append_assistant_message(self, html_content: str, model_name: str) -> AutoHeightTextBrowser:
        role_label = "Assistant"
        clean_model_name = model_name.strip()
        if clean_model_name:
            role_label = f"Assistant ({clean_model_name})"

        return self._add_message_row(
            role_label=role_label,
            html_content=html_content,
            bubble_style=_ASSISTANT_BUBBLE_STYLE,
        )

    def _append_notice(self, markdown_text: str) -> None:
        self._add_message_row(
            role_label="System",
            html_content=self._markdown_to_html_fragment(markdown_text),
            bubble_style=_NOTICE_BUBBLE_STYLE,
        )

    def _print_selected_model_configuration(self) -> None:
        model_name = self._selected_model_name()
        if not model_name:
            return

        if model_name in self.model_config_cache:
            config = self.model_config_cache[model_name]
        else:
            try:
                config = get_ollama_model_configuration(model_name=model_name, request_timeout=3.0)
            except Exception as exc:
                print(f"[model-config] Failed to inspect `{model_name}`: {exc}", file=sys.stderr)
                return
            self.model_config_cache[model_name] = config

        capabilities = config.get("capabilities", [])
        parameter_options = config.get("parameter_options", [])
        capabilities_text = ", ".join(capabilities) if capabilities else "(none reported)"
        options_text = ", ".join(parameter_options) if parameter_options else "(none reported)"

        print(f"[model-config] {model_name}")
        print(f"  capabilities: {capabilities_text}")
        print(f"  configuration options: {options_text}")

    def _update_current_assistant_message(self, final: bool = False) -> None:
        if self.current_assistant_browser is None:
            return

        joined_html = self._assistant_response_html(
            thinking=self.thinking_markdown,
            answer=self.response_markdown,
            final=final,
        )
        self.current_assistant_browser.setHtml(joined_html)
        QTimer.singleShot(0, self._scroll_response_to_bottom)

    def _assistant_response_html(self, thinking: str, answer: str, final: bool) -> str:
        clean_thinking = thinking.strip()
        clean_answer = answer.strip()
        html_blocks: list[str] = []

        if clean_thinking:
            html_blocks.append(self._renderable_thinking_html(clean_thinking))

        if clean_answer:
            html_blocks.append(self._renderable_answer_html(clean_answer))

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

        return "<div style='margin-top: 10px;'></div>".join(html_blocks)

    def _scroll_response_to_bottom(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _renderable_answer_html(self, answer_text: str) -> str:
        return (
            self._html_to_fragment(answer_text)
            if self._looks_like_html(answer_text)
            else self._markdown_to_html_fragment(answer_text)
        )

    def _renderable_thinking_html(self, thinking_text: str) -> str:
        rendered_thinking = self._renderable_answer_html(thinking_text)
        return (
            "<div style='color: #8a8a8a; font-style: italic;'>"
            f"{rendered_thinking}"
            "</div>"
        )

    def _escaped_plain_text_html(
        self, text: str, style: str = "white-space: pre-wrap; margin: 0; width: 100%;"
    ) -> str:
        return f"<div style='{style}'>{html.escape(text)}</div>"

    def _looks_like_html(self, text: str) -> bool:
        return bool(_HTML_LIKE_RE.match(text.lstrip()))

    def _markdown_to_html_fragment(self, markdown_text: str) -> str:
        doc = QTextDocument()
        # Qt's markdown parser mishandles bare <br> in tables; normalize to XHTML-style breaks.
        normalized_markdown = _BR_TAG_RE.sub("<br />", markdown_text)
        doc.setMarkdown(normalized_markdown)
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
