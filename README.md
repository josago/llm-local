# llm-local

`llm-local` is a small PyQt6 desktop chat client for running local LLM conversations through Ollama.

## Requirements

- Python 3.14+
- `ollama` CLI available in `PATH`
- A local model pulled in Ollama (default: `gpt-oss:20b`)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Optional model override:

```bash
OLLAMA_MODEL=llama3.2:3b python3 app.py
```

## Notes

- The app auto-starts `ollama serve` if the API is not already running.
- Prompts and responses are streamed incrementally into the UI.
