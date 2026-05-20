# XYZ — AI Coding Assistant TUI

A Gemini CLI-inspired terminal UI for AI-powered coding, with sandboxed file operations,
verbose activity logging, and support for Ollama (local LLMs), OpenAI, and CUDA acceleration.

```
██╗  ██╗██╗   ██╗███████╗
╚██╗██╔╝╚██╗ ██╔╝╚══███╔╝
 ╚███╔╝  ╚████╔╝   ███╔╝ 
 ██╔██╗   ╚██╔╝   ███╔╝  
██╔╝ ██╗   ██║   ███████╗
╚═╝  ╚═╝   ╚═╝   ╚══════╝
```

---

## Features

- **Gemini CLI-style TUI** — two-panel layout: chat left, verbose activity log right
- **Ollama backend** — runs local LLMs (deepseek-coder, codellama, llama3, mistral…)
- **CUDA auto-detection** — Ollama uses your GPU automatically when available
- **AirLLM support** — use environment variable to point at any OpenAI-compatible API
- **8 built-in tools** the AI can invoke:
  - `read_file` — read any sandbox file
  - `write_file` — create/overwrite files
  - `list_files` — directory listing
  - `run_shell` — execute commands in the sandbox (30s timeout)
  - `search_replace` — surgical text replacement in files
  - `create_dir` — make directories
  - `delete_file` — delete files
  - `web_search` — DuckDuckGo instant answers (no API key needed)
- **Sandboxed** — all file ops are restricted to `/tmp/XYZ_sandbox` (configurable)
- **Verbose log** — every token count, tool call, exit code, and timing shown live
- **Multi-turn** — full conversation history maintained, auto-fed tool results back

---

## Quick Start

### 1. Install dependencies

```bash
bash setup.sh
```

Or manually:
```bash
pip install textual rich httpx requests psutil pygments
```

### 2. Start Ollama (recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a coding model
ollama pull deepseek-coder-v2   # Best for coding
ollama pull codellama:7b         # Lighter option
ollama pull llama3.1             # General purpose

# Start the server
ollama serve
```

### 3. Launch XYZ

```bash
python3 XYZ.py
# or
python3 XYZ.py --model deepseek-coder-v2
python3 XYZ.py --backend openai --model gpt-4o
```

---

## Controls

| Key | Action |
|-----|--------|
| `Ctrl+J` | Send message |
| `Ctrl+L` | Clear chat |
| `Ctrl+S` | Show sandbox files |
| `Ctrl+M` | Cycle through Ollama models |
| `F1` | Help |
| `Ctrl+C` | Quit |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `XYZ_MODEL` | `llama3` | Model name |
| `XYZ_BACKEND` | `ollama` | `ollama` or `openai` |
| `XYZ_SANDBOX` | `/tmp/XYZ_sandbox` | Sandbox directory |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | *(none)* | For OpenAI/compatible APIs |
| `OPENAI_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL |

---

## Performance Tips

### CUDA / GPU Acceleration
Ollama detects and uses NVIDIA GPUs automatically. The status bar shows:
- GPU name and VRAM
- Token generation speed (tok/s)
- CPU & RAM usage

### Recommended Models for Coding

| Model | Size | Notes |
|-------|------|-------|
| `deepseek-coder-v2` | 16B | Best code quality |
| `codellama:34b` | 34B | Meta's largest coding model |
| `codellama:7b` | 7B | Fast, good quality |
| `qwen2.5-coder:7b` | 7B | Excellent for code |
| `mistral:7b` | 7B | Good general + code |
| `llama3.1:8b` | 8B | Balanced |

### AirLLM (Large models on consumer GPU)
AirLLM splits large models across GPU layers. To use:
```bash
pip install airllm
# Then use as OpenAI-compatible server on localhost
export OPENAI_BASE=http://localhost:8080/v1
python3 XYZ.py --backend openai --model your-airllm-model
```

---

## Example Session

```
YOU: Create a Python web scraper that extracts article titles from Hacker News

XYZ: I'll create a complete HN scraper. Let me set it up...
(tool call)

✓ tool:write_file
Created hn_scraper.py (1.2kb)

✓ tool:run_shell
Exit 0: Fetched 30 stories: [1] "New AI model...", [2] "Rust 2.0..."
```

---

## Architecture

```
XYZ.py
├── XYZApp (Textual App)
│   ├── Title bar (ASCII logo)
│   ├── Chat panel (scrollable messages)
│   ├── Activity log panel (real-time verbose log)
│   ├── Input area (TextArea)
│   └── Status bar (model/GPU/CPU/RAM)
├── Tool Engine (execute_tool)
│   ├── Sandbox path validation
│   ├── 8 tools with full logging
│   └── Async shell execution
├── Backends
│   ├── stream_ollama (streaming, token counting)
│   └── stream_openai (OpenAI-compatible)
└── AI Loop
    ├── Build messages with system prompt
    ├── Stream response
    ├── Extract <tool> calls via regex
    ├── Execute tools
    └── Feed results back → loop (max depth 5)
```
