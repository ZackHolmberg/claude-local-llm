"""PreToolUse hook on Read: steer large file reads to local-llm summarize.

Blocks full Reads of large text files with a message pointing Claude to
mcp__local-llm__summarize. Thresholds are tiered by file kind, encoding *why*
a file gets read:

- data (logs, dumps, lockfiles, minified/vendored/generated): the gist is
  almost always enough -> aggressive threshold
- source code: Claude often needs it verbatim to edit correctly -> lenient
- everything else (docs, configs): in between

Ranged Reads (offset/limit) always pass, so Claude keeps an escape hatch
when exact contents matter (e.g. before an Edit).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "usage.jsonl"

# (max_lines, max_bytes) per tier
TIERS = {
    "data": (150, 32_768),
    "source": (800, 131_072),
    "default": (400, 65_536),
}

DATA_SUFFIXES = {".log", ".csv", ".tsv", ".jsonl", ".ndjson"}
DATA_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock",
    "poetry.lock", "cargo.lock", "gemfile.lock", "composer.lock", "go.sum",
}
DATA_PATH_PARTS = {
    "node_modules", "vendor", "dist", "build", ".venv", "venv",
    "target", "__pycache__", ".next", "coverage",
}
SOURCE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs", ".java",
    ".kt", ".c", ".cc", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift",
    ".m", ".cs", ".sh", ".bash", ".zsh", ".sql", ".vue", ".svelte",
}
# Read handles these natively (images/PDFs/notebooks); line counts are meaningless.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ipynb"}


def classify(path: Path) -> str:
    name = path.name.lower()
    if (
        path.suffix.lower() in DATA_SUFFIXES
        or name in DATA_NAMES
        or ".min." in name
        or DATA_PATH_PARTS.intersection(path.parts)
    ):
        return "data"
    if path.suffix.lower() in SOURCE_SUFFIXES:
        return "source"
    return "default"


def log_deflection(path: Path, tier: str, size: int, lines: int | None) -> None:
    """Record the deflection in the shared ledger so the savings report can
    show the funnel: deflections -> summarize calls. Never blocks on failure."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "deflection",
            "file": str(path),
            "tier": tier,
            "size_bytes": size,
            "lines": lines,
        }
        with open(LEDGER, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


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

    tier = classify(path)
    max_lines, max_bytes = TIERS[tier]
    # Env vars override the applicable tier's limits (whichever tier matched).
    max_lines = int(os.environ.get("LOCAL_LLM_HOOK_MAX_LINES", max_lines))
    max_bytes = int(os.environ.get("LOCAL_LLM_HOOK_MAX_BYTES", max_bytes))

    size = path.stat().st_size
    lines: int | None = None
    if size <= max_bytes:
        try:
            with open(path, errors="replace") as f:
                lines = sum(1 for _ in f)
        except OSError:
            return
        if lines <= max_lines:
            return
        detail = f"{lines} lines"
    else:
        detail = f"{size // 1024} KB"

    log_deflection(path, tier, size, lines)
    deny(
        f"large-read-guard: {path} is {detail}, over the {max_lines}-line/"
        f"{max_bytes // 1024} KB limit for {tier} files. To save context, "
        f'prefer mcp__local-llm__summarize with input_files=["{path}"] and a '
        f"focus question. If you need exact contents (e.g. before an Edit), "
        f"call Read again with offset/limit - ranged Reads always pass this "
        f"guard."
    )


if __name__ == "__main__":
    main()
