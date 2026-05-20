#!/usr/bin/env bash
# xyz Setup Script
set -e

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║   xyz - AI Coding Assistant    ║"
echo "  ║   Setup Script                   ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# ── Python deps ─────────────────────────────────────────────────────────
echo "→ Installing Python dependencies…"
pip install textual rich httpx requests psutil pygments prompt_toolkit --break-system-packages -q
echo "  ✓ Python deps installed"

# ── Ollama check ─────────────────────────────────────────────────────────
echo "→ Checking Ollama…"
if command -v ollama &>/dev/null; then
    echo "  ✓ Ollama found: $(ollama --version)"
    echo "  → Pulling recommended coding models…"
    echo "     • deepseek-coder-v2  (excellent for code)"
    echo "     • codellama          (Meta's coding model)"
    echo "     • llama3.1           (general purpose)"
    echo ""
    echo "  Run to pull a model:"
    echo "    ollama pull deepseek-coder-v2"
    echo "    ollama pull codellama:7b"
    echo "    ollama pull llama3.1"
else
    echo "  ⚠ Ollama not found."
    echo "  Install from: https://ollama.com/download"
    echo "  Or via curl:  curl -fsSL https://ollama.com/install.sh | sh"
fi

# ── CUDA check ───────────────────────────────────────────────────────────
echo ""
echo "→ GPU Detection…"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    echo "  ✓ CUDA GPU detected — Ollama will use it automatically"
else
    echo "  ℹ No NVIDIA GPU — running in CPU mode (slower)"
fi

# ── Sandbox ──────────────────────────────────────────────────────────────
SANDBOX="${xyz_SANDBOX:-/tmp/xyz_sandbox}"
mkdir -p "$SANDBOX"
echo ""
echo "→ Sandbox directory: $SANDBOX"
echo ""
echo "  ══════════════════════════════════"
echo "  ✓ Setup complete!"
echo ""
echo "  To launch:"
echo "    python3 XYZ.py"
echo "    python3 XYZ.py --model deepseek-coder-v2"
echo "    python3 XYZ.py --backend openai --model gpt-4o"
echo ""
echo "  Environment variables:"
echo "    xyz_MODEL=deepseek-coder-v2"
echo "    xyz_BACKEND=ollama"
echo "    xyz_SANDBOX=/tmp/my_sandbox"
echo "    OLLAMA_HOST=http://localhost:11434"
echo "    OPENAI_API_KEY=sk-..."
echo "  ══════════════════════════════════"
