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

# 4. Register the large-read-guard hook in user settings (idempotent)
.venv/bin/python - "$PWD" <<'PY'
import json, sys
from pathlib import Path

project = sys.argv[1]
settings_path = Path.home() / ".claude" / "settings.json"
settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
entry = {
    "matcher": "Read",
    "hooks": [{
        "type": "command",
        "command": f"{project}/.venv/bin/python {project}/hooks/read_guard.py",
        "timeout": 10,
        "statusMessage": "Checking file size for local-llm delegation",
    }],
}
pre = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
pre[:] = [e for e in pre if "read_guard.py" not in json.dumps(e)]
pre.append(entry)
# Status line: register only if the user doesn't already have one.
if "statusLine" not in settings:
    settings["statusLine"] = {
        "type": "command",
        "command": f"{project}/.venv/bin/python {project}/statusline.py",
        "refreshInterval": 30,
    }
settings_path.parent.mkdir(parents=True, exist_ok=True)
settings_path.write_text(json.dumps(settings, indent=2) + "\n")
print(f"Registered large-read-guard hook (and status line) in {settings_path}")
PY

echo
echo "Done. Next steps:"
echo "  ./llm pull mlx-community/Qwen3-14B-4bit   # download a model (~8 GB)"
echo "  ./llm start                                # load it (optional - auto-starts on first use)"
echo "  ./llm status                               # verify"
