#!/bin/bash
# Bootstrap: deps -> venv -> MCP registration. Idempotent; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  echo "mlx requires Apple Silicon (Darwin arm64); found: $(uname -sm)" >&2
  exit 1
fi

# 1. uv (manages its own Python; avoids broken system/homebrew venvs)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

# 2. venv + dependencies
echo "Creating venv and installing mlx-lm, mcp, httpx..."
uv venv --python 3.12
uv pip install --python .venv/bin/python mlx-lm "mcp>=1.2" httpx
chmod +x llm

# 3. Register the MCP server with Claude Code (user scope = all projects)
if command -v claude >/dev/null 2>&1; then
  claude mcp remove --scope user local-llm >/dev/null 2>&1 || true
  claude mcp add --scope user local-llm -- "$PWD/.venv/bin/python" "$PWD/server.py"
  echo "Registered MCP server 'local-llm' with Claude Code."
else
  echo "claude CLI not found. Register manually once it's installed:"
  echo "  claude mcp add --scope user local-llm -- $PWD/.venv/bin/python $PWD/server.py"
fi

echo
echo "Done. Next steps:"
echo "  ./llm pull mlx-community/Qwen3-14B-4bit   # download a model (~8 GB)"
echo "  ./llm start                                # load it (optional - auto-starts on first use)"
echo "  ./llm status                               # verify"
