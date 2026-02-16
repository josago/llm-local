import json
import re
import shutil
import socket
import subprocess
import time
from typing import Iterator, Optional, Tuple
from urllib import error, request


_OLLAMA_HOST = "127.0.0.1"
_OLLAMA_PORT = 11434
_OLLAMA_CHAT_URL = f"http://{_OLLAMA_HOST}:{_OLLAMA_PORT}/api/chat"
_OLLAMA_TAGS_URL = f"http://{_OLLAMA_HOST}:{_OLLAMA_PORT}/api/tags"
_ALLOWED_ROLES = {"system", "user", "assistant"}
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _validate_model_name(model_name: str) -> None:
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")


def _validate_prompt(prompt_markdown: str) -> None:
    if prompt_markdown is None:
        raise ValueError("prompt_markdown must not be None")


def _single_user_message(prompt_markdown: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt_markdown}]


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("messages must not be empty")

    normalized_messages: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")
        if role not in _ALLOWED_ROLES:
            raise ValueError("message role must be one of: system, user, assistant")
        if content is None:
            raise ValueError("message content must not be None")
        normalized_messages.append({"role": role, "content": str(content)})
    return normalized_messages


def _ollama_server_up(host: str = _OLLAMA_HOST, port: int = _OLLAMA_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _annotate_ollama_error(detail: str) -> str:
    message = detail.strip()
    lowered = message.lower()
    hints: list[str] = []

    if "do load request" in lowered and "eof" in lowered:
        hints.append("Hint: Ollama runner crashed while loading the model.")
        hints.append("Hint: Run `ollama run <model>` in a terminal for the full crash reason.")
        hints.append("Hint: Retry with a smaller model, or restart/update Ollama.")

    if "nsrangeexception" in lowered and "libmlx" in lowered:
        hints.append(
            "Hint: This looks like an Ollama + mlx-c crash on macOS. Updating or downgrading Ollama may be required."
        )

    if not hints:
        return message
    return f"{message}\n\n" + "\n".join(hints)


def _chat_request(model_name: str, normalized_messages: list[dict[str, str]]) -> request.Request:
    payload = {
        "model": model_name,
        "messages": normalized_messages,
        "stream": True,
        "think": "medium" if model_name.startswith("gpt-oss") else True,
    }
    return request.Request(
        _OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def ensure_ollama_server(startup_timeout: float = 20.0) -> None:
    if shutil.which("ollama") is None:
        raise RuntimeError("`ollama` CLI not found in PATH")

    if _ollama_server_up():
        return

    server = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                "`ollama serve` exited before becoming ready. "
                "Run `ollama serve` in a terminal to inspect the root crash output."
            )
        if _ollama_server_up():
            return
        time.sleep(0.2)

    server.terminate()
    try:
        server.wait(timeout=1)
    except subprocess.TimeoutExpired:
        server.kill()
    raise TimeoutError("Timed out waiting for `ollama serve` to start")


def run_ollama_model(model_name: str, startup_timeout: float = 20.0) -> int:
    _validate_model_name(model_name)

    ensure_ollama_server(startup_timeout=startup_timeout)

    completed = subprocess.run(["ollama", "run", model_name], check=False)
    return completed.returncode


def list_installed_ollama_models(
    startup_timeout: float = 20.0,
    request_timeout: Optional[float] = 5.0,
) -> list[str]:
    if shutil.which("ollama") is None:
        raise RuntimeError("`ollama` CLI not found in PATH")

    ensure_ollama_server(startup_timeout=startup_timeout)

    req = request.Request(_OLLAMA_TAGS_URL, method="GET")
    try:
        with request.urlopen(req, timeout=request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = _annotate_ollama_error(error_body or str(exc))
        raise RuntimeError(detail) from exc
    except error.URLError as exc:
        raise RuntimeError(_annotate_ollama_error(f"Failed to connect to Ollama API: {exc}")) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama API returned an invalid response for model list") from exc

    names: list[str] = []
    seen: set[str] = set()
    for item in payload.get("models", []):
        name = str(item.get("name", "")).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def run_ollama_prompt(
    model_name: str,
    prompt_markdown: str,
    startup_timeout: float = 20.0,
    request_timeout: Optional[float] = None,
) -> str:
    _validate_model_name(model_name)
    _validate_prompt(prompt_markdown)

    return "".join(
        chunk_text
        for chunk_type, chunk_text in stream_ollama_chat(
            model_name=model_name,
            messages=_single_user_message(prompt_markdown),
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
        )
        if chunk_type == "content"
    ).strip()


def stream_ollama_prompt(
    model_name: str,
    prompt_markdown: str,
    startup_timeout: float = 20.0,
    request_timeout: Optional[float] = None,
) -> Iterator[Tuple[str, str]]:
    _validate_model_name(model_name)
    _validate_prompt(prompt_markdown)

    yield from stream_ollama_chat(
        model_name=model_name,
        messages=_single_user_message(prompt_markdown),
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
    )


def stream_ollama_chat(
    model_name: str,
    messages: list[dict[str, str]],
    startup_timeout: float = 20.0,
    request_timeout: Optional[float] = None,
) -> Iterator[Tuple[str, str]]:
    _validate_model_name(model_name)
    normalized_messages = _normalize_messages(messages)

    ensure_ollama_server(startup_timeout=startup_timeout)

    req = _chat_request(model_name, normalized_messages)

    try:
        with request.urlopen(req, timeout=request_timeout) as response:
            for line in response:
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue

                item = json.loads(raw)
                message = item.get("message") or {}
                thinking = _strip_ansi(message.get("thinking", "") or item.get("thinking", ""))
                content = _strip_ansi(message.get("content", "") or item.get("response", ""))

                if thinking:
                    yield ("thinking", thinking)
                if content:
                    yield ("content", content)

                if item.get("done"):
                    break
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = _annotate_ollama_error(error_body or str(exc))
        raise RuntimeError(detail) from exc
    except error.URLError as exc:
        raise RuntimeError(_annotate_ollama_error(f"Failed to connect to Ollama API: {exc}")) from exc
