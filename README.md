# llm-local

Basic vibe-coded app using Python 3.14 and QT 6 that provides a similar interaction as the one you would expect from an online service such as ChatGPT, but running everything 100% locally, for the privacy freaks. Tested in an M4 Mac with `ollama` v0.17.1, `gpt-oss:20b`, `qwen3:30b` and `qwen3.5:35b`.

## Requirements

- `ollama` available in `PATH`, with at least one LLM already pulled and available.

## Installation

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
