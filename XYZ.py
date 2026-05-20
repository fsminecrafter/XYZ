#!/usr/bin/env python3
"""
XYZ - AI Coding Assistant TUI
Gemini CLI-inspired interface with Ollama/OpenAI backend,
verbose logging, sandboxed file tools, and CUDA awareness.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    TextArea,
)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENAI_BASE = os.environ.get("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")

DEFAULT_MODEL = os.environ.get("XYZ_MODEL", "llama3")  # or "gpt-4o", "mistral", etc.
BACKEND       = os.environ.get("XYZ_BACKEND", "ollama")  # ollama | openai | anthropic

SANDBOX_DIR   = Path(os.environ.get("XYZ_SANDBOX", "/tmp/xyz_sandbox"))
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are XYZ, an elite AI coding assistant. You operate inside a sandboxed environment.

You can use the following tools by outputting a JSON block wrapped in <tool> tags:

<tool>{"name": "read_file", "path": "relative/path"}</tool>
<tool>{"name": "write_file", "path": "relative/path", "content": "..."}</tool>
<tool>{"name": "list_files", "path": "."}</tool>
<tool>{"name": "run_shell", "command": "python3 script.py"}</tool>
<tool>{"name": "search_replace", "path": "file.py", "search": "old_text", "replace": "new_text"}</tool>
<tool>{"name": "create_dir", "path": "new_folder"}</tool>
<tool>{"name": "delete_file", "path": "file.py"}</tool>
<tool>{"name": "web_search", "query": "python asyncio tutorial"}</tool>
<tool>{"name": "diff_file", "path": "file.py", "before": "...", "after": "..."}</tool>

Rules:
- ALWAYS use tools to actually write/read code. Never just describe what you would do.
- After using a tool, explain what you did and why.
- Be concise but thorough. Output clean, production-ready code.
- All file paths are relative to the sandbox directory.
- For shell commands, prefer safe operations. Never rm -rf / or anything destructive outside the sandbox.
- When writing code, include proper error handling and comments.
"""

# ─────────────────────────────────────────────
#  Tool Execution Engine
# ─────────────────────────────────────────────

class ToolResult:
    def __init__(self, tool_name: str, success: bool, output: str, metadata: dict = None):
        self.tool_name = tool_name
        self.success   = success
        self.output    = output
        self.metadata  = metadata or {}
        self.timestamp = datetime.now()


def sandbox_path(rel: str) -> Path:
    """Resolve a relative path safely inside the sandbox."""
    p = (SANDBOX_DIR / rel).resolve()
    if not str(p).startswith(str(SANDBOX_DIR.resolve())):
        raise ValueError(f"Path escape attempt blocked: {rel}")
    return p


async def execute_tool(tool: dict, log_cb=None) -> ToolResult:
    name = tool.get("name", "unknown")

    def log(msg: str):
        if log_cb:
            log_cb(msg)

    try:
        # ── read_file ──────────────────────────────────────────────────────
        if name == "read_file":
            path = sandbox_path(tool["path"])
            log(f"[tool:read_file] Reading → {path}")
            if not path.exists():
                return ToolResult(name, False, f"File not found: {tool['path']}")
            content = path.read_text(errors="replace")
            log(f"[tool:read_file] Read {len(content)} chars from {path.name}")
            return ToolResult(name, True, content, {"path": str(path), "size": len(content)})

        # ── write_file ─────────────────────────────────────────────────────
        elif name == "write_file":
            path = sandbox_path(tool["path"])
            content = tool["content"]
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            path.write_text(content)
            log(f"[tool:write_file] {'Overwrote' if existed else 'Created'} {path} ({len(content)} chars)")
            return ToolResult(name, True, f"Written {len(content)} bytes to {tool['path']}", {"path": str(path)})

        # ── list_files ─────────────────────────────────────────────────────
        elif name == "list_files":
            path = sandbox_path(tool.get("path", "."))
            log(f"[tool:list_files] Listing {path}")
            if not path.exists():
                return ToolResult(name, False, f"Directory not found: {tool.get('path','.')}")
            items = []
            for p in sorted(path.rglob("*")):
                rel = p.relative_to(SANDBOX_DIR)
                size = p.stat().st_size if p.is_file() else 0
                items.append(f"{'📁' if p.is_dir() else '📄'} {rel}  ({size}B)" if p.is_file() else f"{'📁'} {rel}/")
            result = "\n".join(items) if items else "(empty)"
            log(f"[tool:list_files] Found {len(items)} items")
            return ToolResult(name, True, result)

        # ── run_shell ──────────────────────────────────────────────────────
        elif name == "run_shell":
            cmd = tool["command"]
            log(f"[tool:run_shell] Executing: {cmd}")
            log(f"[tool:run_shell] CWD: {SANDBOX_DIR}")
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(SANDBOX_DIR),
                env={**os.environ, "PYTHONPATH": str(SANDBOX_DIR)},
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(name, False, "Command timed out after 30s")
            output = stdout.decode(errors="replace").strip()
            rc = proc.returncode
            log(f"[tool:run_shell] Exit code: {rc} | Output: {len(output)} chars")
            return ToolResult(name, rc == 0, output or "(no output)", {"returncode": rc})

        # ── search_replace ─────────────────────────────────────────────────
        elif name == "search_replace":
            path = sandbox_path(tool["path"])
            log(f"[tool:search_replace] Patching {path}")
            if not path.exists():
                return ToolResult(name, False, f"File not found: {tool['path']}")
            original = path.read_text()
            search  = tool["search"]
            replace = tool["replace"]
            if search not in original:
                return ToolResult(name, False, f"Search string not found in {tool['path']}")
            patched = original.replace(search, replace, 1)
            path.write_text(patched)
            log(f"[tool:search_replace] Replaced 1 occurrence in {path.name}")
            return ToolResult(name, True, f"Replaced in {tool['path']}")

        # ── create_dir ─────────────────────────────────────────────────────
        elif name == "create_dir":
            path = sandbox_path(tool["path"])
            path.mkdir(parents=True, exist_ok=True)
            log(f"[tool:create_dir] Created directory: {path}")
            return ToolResult(name, True, f"Directory created: {tool['path']}")

        # ── delete_file ────────────────────────────────────────────────────
        elif name == "delete_file":
            path = sandbox_path(tool["path"])
            log(f"[tool:delete_file] Deleting: {path}")
            if not path.exists():
                return ToolResult(name, False, f"File not found: {tool['path']}")
            path.unlink()
            log(f"[tool:delete_file] Deleted {path.name}")
            return ToolResult(name, True, f"Deleted: {tool['path']}")

        # ── web_search ─────────────────────────────────────────────────────
        elif name == "web_search":
            query = tool["query"]
            log(f"[tool:web_search] Searching: {query}")
            # DuckDuckGo instant answer API (no key needed)
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1"},
                )
            data = resp.json()
            abstract = data.get("AbstractText") or data.get("Answer") or ""
            topics = [t.get("Text", "") for t in data.get("RelatedTopics", [])[:5]]
            result = (abstract or "(no abstract)") + "\n\nRelated:\n" + "\n".join(f"• {t}" for t in topics if t)
            log(f"[tool:web_search] Got {len(result)} chars of results")
            return ToolResult(name, True, result)

        # ── diff_file ──────────────────────────────────────────────────────
        elif name == "diff_file":
            path = sandbox_path(tool["path"])
            before = tool.get("before", "")
            after  = tool.get("after", "")
            log(f"[tool:diff_file] Diffing {path.name}")
            with tempfile.NamedTemporaryFile("w", suffix=".before", delete=False) as f1:
                f1.write(before); fa = f1.name
            with tempfile.NamedTemporaryFile("w", suffix=".after", delete=False) as f2:
                f2.write(after); fb = f2.name
            proc = subprocess.run(["diff", "-u", fa, fb], capture_output=True, text=True)
            os.unlink(fa); os.unlink(fb)
            return ToolResult(name, True, proc.stdout or "(no diff)")

        else:
            return ToolResult(name, False, f"Unknown tool: {name}")

    except Exception as e:
        tb = traceback.format_exc()
        log(f"[tool:{name}] ERROR: {e}")
        return ToolResult(name, False, f"Error: {e}\n{tb}")


# ─────────────────────────────────────────────
#  Backend Clients
# ─────────────────────────────────────────────

async def stream_ollama(model: str, messages: list, log_cb=None):
    """Stream from Ollama API."""
    if log_cb:
        log_cb(f"[ollama] Connecting to {OLLAMA_BASE}")
        log_cb(f"[ollama] Model: {model} | Messages: {len(messages)}")
        log_cb(f"[ollama] Estimating prompt tokens: ~{sum(len(str(m)) for m in messages)//4}")

    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }

    collected = ""
    token_count = 0
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    raise RuntimeError(f"Ollama HTTP {resp.status_code}: {err.decode()}")
                if log_cb:
                    log_cb(f"[ollama] Stream started (HTTP 200)")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        collected += chunk
                        token_count += 1
                        yield chunk
                    if data.get("done"):
                        elapsed = time.time() - start
                        tps = token_count / elapsed if elapsed > 0 else 0
                        if log_cb:
                            log_cb(f"[ollama] Done. Tokens: {token_count} | {tps:.1f} tok/s | {elapsed:.2f}s")
                        break
    except httpx.ConnectError:
        msg = f"\n\n[ERROR] Cannot connect to Ollama at {OLLAMA_BASE}\nStart Ollama: `ollama serve` then `ollama pull {model}`"
        yield msg
        if log_cb:
            log_cb(f"[ollama] Connection refused – is Ollama running?")


async def stream_openai(model: str, messages: list, log_cb=None):
    """Stream from OpenAI-compatible API."""
    if not OPENAI_KEY:
        yield "\n[ERROR] OPENAI_API_KEY not set. Export it or switch to Ollama backend."
        return

    if log_cb:
        log_cb(f"[openai] Model: {model} | Endpoint: {OPENAI_BASE}")

    url = f"{OPENAI_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "stream": True, "temperature": 0.3}

    start = time.time()
    token_count = 0
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                yield f"\n[ERROR] OpenAI HTTP {resp.status_code}: {err.decode()}"
                return
            if log_cb:
                log_cb(f"[openai] Stream started")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    elapsed = time.time() - start
                    if log_cb:
                        log_cb(f"[openai] Done. Tokens: {token_count} | {elapsed:.2f}s")
                    break
                try:
                    data  = json.loads(data_str)
                    chunk = data["choices"][0]["delta"].get("content", "")
                    if chunk:
                        token_count += 1
                        yield chunk
                except (json.JSONDecodeError, KeyError):
                    pass


async def list_ollama_models(log_cb=None) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            if log_cb:
                log_cb(f"[ollama] Available models: {', '.join(models) or 'none'}")
            return models
    except Exception as e:
        if log_cb:
            log_cb(f"[ollama] Could not list models: {e}")
        return []


def detect_cuda() -> str:
    """Detect CUDA / GPU availability."""
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "No NVIDIA GPU detected (CPU mode)"


def get_system_stats() -> dict:
    cpu  = psutil.cpu_percent(interval=0.1)
    mem  = psutil.virtual_memory()
    return {
        "cpu": f"{cpu:.0f}%",
        "ram": f"{mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB",
        "ram_pct": mem.percent,
    }


# ─────────────────────────────────────────────
#  Parse AI response for tool calls
# ─────────────────────────────────────────────

def extract_tools(text: str) -> list[dict]:
    """Find all <tool>{...}</tool> blocks in AI output."""
    tools = []
    for match in re.finditer(r"<tool>(.*?)</tool>", text, re.DOTALL):
        raw = match.group(1).strip()
        try:
            tools.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return tools


# ─────────────────────────────────────────────
#  Textual App
# ─────────────────────────────────────────────

LOGO = """
██╗  ██╗██╗   ██╗███████╗
╚██╗██╔╝╚██╗ ██╔╝╚══███╔╝
 ╚███╔╝  ╚████╔╝   ███╔╝ 
 ██╔██╗   ╚██╔╝   ███╔╝  
██╔╝ ██╗   ██║   ███████╗
╚═╝  ╚═╝   ╚═╝   ╚══════╝"""


class StatusBar(Static):
    """Bottom status bar showing model, backend, GPU, stats."""
    
    def update_stats(self, model: str, backend: str, gpu: str, stats: dict):
        gpu_short = gpu.split("\n")[0][:40] if "No NVIDIA" not in gpu else "CPU"
        self.update(
            f"[bold cyan]⚡ {backend.upper()}[/] [dim]│[/] "
            f"[green]{model}[/] [dim]│[/] "
            f"[yellow]GPU: {gpu_short}[/] [dim]│[/] "
            f"[dim]CPU:{stats['cpu']} RAM:{stats['ram']}[/]"
        )


class ChatMessage(Static):
    """A single message bubble in the chat."""
    
    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role    = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "user":
            yield Label(f"[bold cyan]YOU[/]  [dim]{datetime.now().strftime('%H:%M:%S')}[/]", classes="msg-header user-header")
            yield Static(self.content, classes="msg-body user-body")
        elif self.role == "assistant":
            yield Label(f"[bold green]XYZ[/] [dim]{datetime.now().strftime('%H:%M:%S')}[/]", classes="msg-header ai-header")
            yield Static(self.content, classes="msg-body ai-body", markup=False)
        elif self.role == "tool":
            yield Static(self.content, classes="tool-result")
        elif self.role == "system_info":
            yield Static(self.content, classes="sys-info")


class XYZApp(App):
    """XYZ - AI Coding Assistant"""

    CSS = """
    Screen {
        background: #0a0e1a;
        layout: vertical;
    }

    #title-bar {
        height: 9;
        background: #0a0e1a;
        border-bottom: solid #1a2540;
        padding: 0 2;
        content-align: center middle;
        color: #00d4ff;
        text-style: bold;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #chat-panel {
        width: 1fr;
        height: 100%;
        border-right: solid #1a2540;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 2;
        scrollbar-color: #1a2540 #0a0e1a;
        scrollbar-size: 1 1;
    }

    #log-panel {
        width: 42;
        height: 100%;
        background: #050810;
    }

    #log-header {
        height: 1;
        background: #0d1628;
        color: #4a5568;
        padding: 0 1;
        text-style: bold;
        content-align: left middle;
    }

    #activity-log {
        height: 1fr;
        padding: 0 1;
        scrollbar-color: #1a2540 #050810;
        scrollbar-size: 1 1;
        background: #050810;
    }

    #input-area {
        height: auto;
        min-height: 5;
        max-height: 10;
        border-top: solid #1a2540;
        background: #0a0e1a;
        padding: 1 2;
        layout: vertical;
    }

    #input-hint {
        height: 1;
        color: #2a3a5a;
        padding: 0 1;
    }

    #user-input {
        height: auto;
        min-height: 3;
        background: #0d1628;
        border: solid #1e3a5f;
        color: #e2e8f0;
        padding: 0 1;
    }

    #user-input:focus {
        border: solid #00d4ff;
    }

    #status-bar {
        height: 1;
        background: #050810;
        border-top: solid #1a2540;
        padding: 0 2;
        color: #718096;
    }

    .msg-header {
        height: 1;
        margin-top: 1;
    }

    .user-header {
        color: #00d4ff;
    }

    .ai-header {
        color: #48bb78;
    }

    .msg-body {
        padding: 0 1;
        margin-bottom: 1;
        color: #e2e8f0;
    }

    .user-body {
        color: #cbd5e0;
        border-left: thick #00d4ff 30%;
        padding-left: 1;
    }

    .ai-body {
        color: #e2e8f0;
    }

    .tool-result {
        background: #0d1a0d;
        border-left: thick #48bb78;
        padding: 0 1;
        margin: 0 0 1 0;
        color: #9ae6b4;
    }

    .sys-info {
        color: #4a5568;
        margin: 0 0 1 0;
        padding: 0 1;
    }

    #streaming-indicator {
        height: 1;
        color: #00d4ff;
        padding: 0 2;
        display: none;
    }

    #streaming-indicator.active {
        display: block;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+s", "show_sandbox", "Sandbox"),
        Binding("ctrl+m", "change_model", "Model"),
        Binding("f1", "show_help", "Help"),
    ]

    model   = reactive(DEFAULT_MODEL)
    backend = reactive(BACKEND)
    busy    = reactive(False)

    def __init__(self):
        super().__init__()
        self.conversation_history: list[dict] = []
        self.gpu_info   = detect_cuda()
        self.stats      = get_system_stats()
        self._stream_widget: Static | None = None

    # ── Compose ───────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="title-bar"):
            yield Static(LOGO)

        with Container(id="main-container"):
            with Vertical(id="chat-panel"):
                with ScrollableContainer(id="chat-scroll"):
                    yield ChatMessage("system_info",
                        f"[dim]Sandbox: {SANDBOX_DIR}  |  GPU: {self.gpu_info.split(chr(10))[0]}[/]",
                        id="welcome-msg")
                yield Static(id="streaming-indicator")
                with Container(id="input-area"):
                    yield Static("  [dim]Enter your coding request. Ctrl+Enter to send.[/]", id="input-hint")
                    yield TextArea(id="user-input", language="markdown")

            with Vertical(id="log-panel"):
                yield Static("  ◈ ACTIVITY LOG", id="log-header")
                yield RichLog(id="activity-log", highlight=True, markup=True, wrap=True)

        yield StatusBar(id="status-bar")
        yield Footer()

    # ── App ready ─────────────────────────────────────────────────────────

    async def on_mount(self):
        self.query_one("#user-input", TextArea).focus()
        self._refresh_status()
        self.log_activity("[bold cyan]XYZ started[/]")
        self.log_activity(f"[dim]Backend: {self.backend} | Model: {self.model}[/]")
        self.log_activity(f"[dim]Sandbox: {SANDBOX_DIR}[/]")
        self.log_activity(f"[dim]GPU: {self.gpu_info.split(chr(10))[0]}[/]")
        self.log_activity("[dim]─────────────────────────────[/]")

        # check Ollama in background
        if self.backend == "ollama":
            self.run_worker(self._check_ollama(), exclusive=False)

        # start stats refresh loop
        self.set_interval(5, self._refresh_status)

    # ── Helpers ───────────────────────────────────────────────────────────

    def log_activity(self, msg: str):
        log = self.query_one("#activity-log", RichLog)
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:12]
        log.write(f"[dim]{ts}[/] {msg}")

    def _refresh_status(self):
        self.stats = get_system_stats()
        sb = self.query_one("#status-bar", StatusBar)
        sb.update_stats(self.model, self.backend, self.gpu_info, self.stats)

    async def _check_ollama(self):
        self.log_activity("[yellow]Checking Ollama connection…[/]")
        models = await list_ollama_models(self.log_activity)
        if models:
            self.log_activity(f"[green]✓ Ollama online. Models: {', '.join(models[:4])}[/]")
            if self.model not in models and models:
                self.log_activity(f"[yellow]⚠ Model '{self.model}' not found. Defaulting to {models[0]}[/]")
                self.model = models[0]
        else:
            self.log_activity("[red]✗ Ollama not reachable. Start: `ollama serve`[/]")

    def _add_message_widget(self, role: str, content: str) -> ChatMessage:
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        msg    = ChatMessage(role, content)
        scroll.mount(msg)
        scroll.scroll_end(animate=False)
        return msg

    def _set_streaming_indicator(self, active: bool, text: str = ""):
        ind = self.query_one("#streaming-indicator", Static)
        if active:
            ind.update(f"[bold cyan blink]◈[/] [cyan]{text}[/]")
            ind.add_class("active")
        else:
            ind.remove_class("active")

    # ── Key events ────────────────────────────────────────────────────────

    @on(TextArea.Changed, "#user-input")
    async def _on_input_change(self, event: TextArea.Changed):
        pass  # could add live token counting here

    async def on_key(self, event) -> None:
        if event.key == "ctrl+j" or event.key == "enter":
            # Check if TextArea is focused and handle submission  
            ta = self.query_one("#user-input", TextArea)
            if ta.has_focus and event.key == "ctrl+j":
                await self._submit()
                event.prevent_default()

    async def _submit(self):
        if self.busy:
            self.log_activity("[yellow]⚠ Already processing. Please wait.[/]")
            return
        ta   = self.query_one("#user-input", TextArea)
        text = ta.text.strip()
        if not text:
            return
        ta.clear()
        await self._handle_user_message(text)

    # ── Actions ───────────────────────────────────────────────────────────

    def action_quit(self):
        self.exit()

    def action_clear_chat(self):
        scroll = self.query_one("#chat-scroll", ScrollableContainer)
        for child in list(scroll.children):
            if child.id != "welcome-msg":
                child.remove()
        self.conversation_history.clear()
        self.log_activity("[cyan]Chat cleared[/]")

    def action_show_sandbox(self):
        files = list(SANDBOX_DIR.rglob("*"))
        msg   = f"[dim]Sandbox ({SANDBOX_DIR}):\n" + "\n".join(
            f"  {f.relative_to(SANDBOX_DIR)}" for f in files if f.is_file()
        ) + ("[/]" if files else "  (empty)[/]")
        self._add_message_widget("system_info", msg)

    def action_change_model(self):
        self.run_worker(self._pick_model(), exclusive=True)

    async def _pick_model(self):
        models = await list_ollama_models(self.log_activity)
        if models:
            self.model = models[(models.index(self.model) + 1) % len(models)] if self.model in models else models[0]
            self.log_activity(f"[cyan]Switched to model: {self.model}[/]")
            self._add_message_widget("system_info", f"[dim]Model → {self.model}[/]")

    def action_show_help(self):
        help_text = (
            "[bold cyan]XYZ Help[/]\n\n"
            "[bold]Keyboard shortcuts:[/]\n"
            "  Ctrl+J    Send message\n"
            "  Ctrl+L    Clear chat\n"
            "  Ctrl+S    Show sandbox files\n"
            "  Ctrl+M    Cycle model\n"
            "  F1        This help\n\n"
            "[bold]Tools available to AI:[/]\n"
            "  read_file, write_file, list_files\n"
            "  run_shell, search_replace\n"
            "  create_dir, delete_file, web_search\n\n"
            f"[bold]Sandbox:[/] {SANDBOX_DIR}\n"
            f"[bold]Backend:[/] {self.backend} → {OLLAMA_BASE}\n"
            f"[bold]GPU:[/] {self.gpu_info.split(chr(10))[0]}"
        )
        self._add_message_widget("system_info", help_text)

    # ── Main AI flow ──────────────────────────────────────────────────────

    async def _handle_user_message(self, text: str):
        self.busy = True
        self._add_message_widget("user", text)
        self.log_activity(f"[cyan]User → {text[:60]}{'…' if len(text)>60 else ''}[/]")

        self.conversation_history.append({"role": "user", "content": text})

        # Build message list with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history

        self.log_activity(f"[dim]Context: {len(self.conversation_history)} turns | ~{sum(len(str(m)) for m in messages)//4} tokens[/]")

        try:
            await self._stream_and_execute(messages)
        except Exception as e:
            self.log_activity(f"[red]Fatal error: {e}[/]")
            self._add_message_widget("system_info", f"[red]Error: {e}[/]")
        finally:
            self.busy = False
            self._set_streaming_indicator(False)
            self._refresh_status()

    async def _stream_and_execute(self, messages: list, depth: int = 0):
        """Stream AI response, detect tool calls, execute them, loop back."""

        if depth > 5:
            self.log_activity("[yellow]Max tool-call depth (5) reached[/]")
            return

        self.log_activity(f"[green]→ Streaming response (depth={depth})…[/]")
        self._set_streaming_indicator(True, f"Thinking with {self.model}…")

        # Create streaming output area
        scroll   = self.query_one("#chat-scroll", ScrollableContainer)
        ai_label = Label(f"[bold green]XYZ[/] [dim]{datetime.now().strftime('%H:%M:%S')}[/]", classes="msg-header ai-header")
        ai_body  = Static("", classes="msg-body ai-body")
        wrapper  = Container()
        wrapper.mount(ai_label)
        wrapper.mount(ai_body)
        scroll.mount(wrapper)

        full_response = ""
        char_count    = 0

        stream_gen = (
            stream_ollama(self.model, messages, self.log_activity)
            if self.backend == "ollama"
            else stream_openai(self.model, messages, self.log_activity)
        )

        async for chunk in stream_gen:
            full_response += chunk
            char_count    += len(chunk)
            # Update display (strip tool tags for cleaner output)
            display = re.sub(r"<tool>.*?</tool>", "[dim](tool call)[/dim]", full_response, flags=re.DOTALL)
            ai_body.update(display)
            scroll.scroll_end(animate=False)

            if char_count % 200 == 0:
                self.log_activity(f"[dim]  …{char_count} chars received[/]")

        self.log_activity(f"[green]✓ Response complete ({len(full_response)} chars)[/]")
        self._set_streaming_indicator(False)

        # Store in history
        self.conversation_history.append({"role": "assistant", "content": full_response})

        # Extract & execute tools
        tools = extract_tools(full_response)
        if tools:
            self.log_activity(f"[cyan]Found {len(tools)} tool call(s)[/]")
            tool_outputs = []

            for i, tool in enumerate(tools, 1):
                tname = tool.get("name", "?")
                self.log_activity(f"[yellow]  [{i}/{len(tools)}] Executing: {tname}[/]")
                self._set_streaming_indicator(True, f"Running tool: {tname}…")

                result = await execute_tool(tool, self.log_activity)

                # Show tool result in chat
                icon  = "✓" if result.success else "✗"
                color = "green" if result.success else "red"
                preview = result.output[:300] + ("…" if len(result.output) > 300 else "")
                tool_display = (
                    f"[bold {color}]{icon} tool:{tname}[/]\n"
                    f"[dim]{preview}[/]"
                )
                self._add_message_widget("tool", tool_display)
                self.log_activity(f"[{'green' if result.success else 'red'}]  ↳ {tname}: {'OK' if result.success else 'FAIL'} | {len(result.output)} chars[/]")

                tool_outputs.append({
                    "tool": tname,
                    "success": result.success,
                    "output": result.output
                })

            # Feed results back to AI for follow-up
            tool_summary = "\n\n".join(
                f"Tool `{t['tool']}` {'succeeded' if t['success'] else 'failed'}:\n```\n{t['output'][:1500]}\n```"
                for t in tool_outputs
            )
            messages.append({"role": "user", "content": f"[TOOL RESULTS]\n{tool_summary}\n\nContinue based on these results."})
            self.log_activity(f"[cyan]Feeding {len(tool_outputs)} result(s) back to model…[/]")
            self._set_streaming_indicator(False)

            await self._stream_and_execute(messages, depth=depth + 1)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main():
    global OLLAMA_BASE, SANDBOX_DIR, DEFAULT_MODEL, BACKEND
    import argparse
    parser = argparse.ArgumentParser(description="XYZ - AI Coding Assistant TUI")
    parser.add_argument("--model",   default=DEFAULT_MODEL,    help="Model name (default: %(default)s)")
    parser.add_argument("--backend", default=BACKEND,          choices=["ollama", "openai"], help="Backend")
    parser.add_argument("--ollama",  default=OLLAMA_BASE,      help="Ollama base URL")
    parser.add_argument("--sandbox", default=str(SANDBOX_DIR), help="Sandbox directory")
    args = parser.parse_args()
    OLLAMA_BASE   = args.ollama
    DEFAULT_MODEL = args.model
    BACKEND       = args.backend
    SANDBOX_DIR   = Path(args.sandbox)
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  XYZ — AI Coding Assistant")
    print(f"  Backend : {args.backend}")
    print(f"  Model   : {args.model}")
    print(f"  Sandbox : {SANDBOX_DIR}")
    print(f"  GPU     : {detect_cuda().split(chr(10))[0]}\n")

    app = XYZApp()
    app.run()


if __name__ == "__main__":
    main()
