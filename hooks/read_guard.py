"""PreToolUse hook on Read: steer large file reads to local-llm summarize.

Blocks full Reads of large text files with a message pointing Claude to
mcp__local-llm__summarize. Ranged Reads (offset/limit) always pass, so
Claude keeps an escape hatch when exact contents matter (e.g. before Edit).
"""

import json
import os
import sys
from pathlib import Path

MAX_LINES = int(os.environ.get("LOCAL_LLM_HOOK_MAX_LINES", "400"))
MAX_BYTES = int(os.environ.get("LOCAL_LLM_HOOK_MAX_BYTES", "65536"))
# Read handles these natively (images/PDFs/notebooks); line counts are meaningless.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb"}


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> None:
    try:
        tool_input = json.load(sys.stdin).get("tool_input") or {}
    except json.JSONDecodeError:
        return  # malformed input: never block
    file_path = tool_input.get("file_path")
    if not file_path:
        return
    # Explicit ranged reads are the escape hatch - always allow.
    if tool_input.get("offset") is not None or tool_input.get("limit") is not None:
        return
    path = Path(file_path)
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        return

    size = path.stat().st_size
    if size <= MAX_BYTES:
        try:
            with open(path, errors="replace") as f:
                lines = sum(1 for _ in f)
        except OSError:
            return
        if lines <= MAX_LINES:
            return
        detail = f"{lines} lines"
    else:
        detail = f"{size // 1024} KB"

    deny(
        f"large-read-guard: {path} is {detail}. To save context, prefer "
        f'mcp__local-llm__summarize with input_files=["{path}"] and a focus '
        f"question. If you need exact contents (e.g. before an Edit), call "
        f"Read again with offset/limit - ranged Reads always pass this guard."
    )


if __name__ == "__main__":
    main()
