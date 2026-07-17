"""MCP server that offloads mechanical tasks from Claude to a local Qwen model via mlx-lm.

Exposes delegation tools to Claude Code. On first use it auto-starts
`mlx_lm.server` (OpenAI-compatible, shared across sessions) if it isn't
already listening.

Token-saving design: tools accept file *paths* as input and an optional
output path, so large content flows disk -> local model -> disk without
ever entering Claude's context.
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

PROJECT_DIR = Path(__file__).resolve().parent
VENV_BIN = PROJECT_DIR / ".venv" / "bin"

CONFIG_PATH = PROJECT_DIR / "config.json"


def _config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Precedence: env var > config.json (written by `llm use`) > default.
MODEL = os.environ.get("LOCAL_LLM_MODEL") or _config().get(
    "model", "mlx-community/Qwen3-14B-4bit"
)
PORT = int(os.environ.get("LOCAL_LLM_PORT") or _config().get("port", 8734))
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVER_LOG = PROJECT_DIR / "mlx-server.log"
USAGE_LOG = PROJECT_DIR / "usage.jsonl"

# Claude pricing, USD per million tokens (input, output).
CLAUDE_PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-fable-5": (10.00, 50.00),
}

# Qwen3-14B supports 32k tokens of context; leave headroom for the
# instruction and the response.
MAX_INPUT_CHARS = 80_000
GENERATION_TIMEOUT_S = 600
STARTUP_TIMEOUT_S = 240

SYSTEM_PROMPT = (
    "/no_think You are a precise assistant handling a delegated subtask. "
    "Follow the instruction exactly. Output only the requested result - no "
    "preamble, no commentary, no markdown fences unless asked for them."
)

logging.getLogger("httpx").setLevel(logging.WARNING)

mcp = FastMCP("local-llm")


def _strip_fences(text: str) -> str:
    match = re.fullmatch(r"```[\w+-]*\n(.*?)\n?```", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


def _server_alive() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/v1/models", timeout=3)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _ensure_server() -> None:
    if _server_alive():
        return
    with open(SERVER_LOG, "ab") as log:
        subprocess.Popen(
            [str(VENV_BIN / "mlx_lm.server"), "--model", MODEL, "--port", str(PORT)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _server_alive():
            return
        time.sleep(2)
    raise RuntimeError(
        f"Local model server failed to start within {STARTUP_TIMEOUT_S}s. "
        f"Check {SERVER_LOG} - the model may still be downloading."
    )


def _gather_input(
    input_files: list[str] | None, input_text: str | None
) -> tuple[str, int]:
    """Returns (combined input text, chars that came from files).

    File-derived chars are what Claude avoided reading; `input_text` came from
    Claude's context already, so it never counts as savings.
    """
    parts: list[str] = []
    file_chars = 0
    budget = MAX_INPUT_CHARS
    for raw_path in input_files or []:
        path = Path(raw_path).expanduser()
        text = path.read_text(errors="replace")
        if len(text) > budget:
            keep = max(budget, 2_000)
            head, tail = text[: keep * 2 // 3], text[-keep // 3 :]
            text = f"{head}\n\n...[middle truncated]...\n\n{tail}"
        budget -= len(text)
        file_chars += len(text)
        parts.append(f'<file path="{path}">\n{text}\n</file>')
        if budget <= 0:
            parts.append("[input budget exhausted; remaining files omitted]")
            break
    if input_text:
        parts.append(input_text[:MAX_INPUT_CHARS])
    return "\n\n".join(parts), file_chars


def _est_tokens(chars: int) -> int:
    return chars // 4


def _log_usage(
    tool: str,
    usage: dict,
    input_tokens_avoided: int,
    output_tokens_avoided: int,
    returned_chars: int,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": tool,
        "model": MODEL,
        "local_prompt_tokens": usage.get("prompt_tokens", 0),
        "local_completion_tokens": usage.get("completion_tokens", 0),
        "input_tokens_avoided": input_tokens_avoided,
        "output_tokens_avoided": output_tokens_avoided,
        "tokens_returned_to_claude": _est_tokens(returned_chars),
    }
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _generate(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, dict]:
    _ensure_server()
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=GENERATION_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip(), data.get("usage", {})


@mcp.tool()
def delegate(
    instruction: str,
    input_files: list[str] | None = None,
    input_text: str | None = None,
    output_file: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Offload a mechanical, non-reasoning task to a fast local model to save
    Claude tokens. Ideal for: generating boilerplate, test fixtures, mock
    data, docstrings, or repetitive code; extracting or reformatting data;
    converting between formats; drafting routine docs; bulk find-and-describe
    work over files.

    NOT for tasks needing careful reasoning, debugging, architectural
    decisions, or correctness-critical logic - do those yourself.

    Pass large inputs via `input_files` (absolute paths) instead of pasting
    content, and set `output_file` for bulk output - then the content never
    enters your context and you only see a short confirmation. Give a
    self-contained instruction: the local model sees ONLY what you pass here,
    not the conversation.
    """
    gathered, file_chars = _gather_input(input_files, input_text)
    prompt = f"{instruction}\n\n{gathered}" if gathered else instruction
    result, usage = _generate(prompt, max_tokens)
    if output_file:
        result = _strip_fences(result)
        out = Path(output_file).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result + "\n")
        lines = result.count("\n") + 1
        preview = "\n".join(result.splitlines()[:8])
        reply = (
            f"Wrote {lines} lines ({len(result)} chars) to {out}\n"
            f"Preview:\n{preview}"
        )
        _log_usage(
            "delegate",
            usage,
            input_tokens_avoided=_est_tokens(file_chars),
            output_tokens_avoided=usage.get("completion_tokens", 0),
            returned_chars=len(reply),
        )
        return reply
    _log_usage(
        "delegate",
        usage,
        input_tokens_avoided=_est_tokens(file_chars),
        output_tokens_avoided=0,
        returned_chars=len(result),
    )
    return result


@mcp.tool()
def summarize(
    input_files: list[str] | None = None,
    input_text: str | None = None,
    focus: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Summarize large files, logs, diffs, or docs with a local model instead
    of reading them yourself - a major token saving whenever a file is big
    and you only need the gist or a specific answer from it. Pass absolute
    paths via `input_files`; use `focus` to ask a pointed question (e.g.
    "which requests failed and why?"). Prefer this over Read for any file
    over a few hundred lines when you don't need exact contents.
    """
    gathered, file_chars = _gather_input(input_files, input_text)
    if not gathered:
        return "Error: provide input_files and/or input_text."
    task = (
        f"Answer this about the content below, citing specifics: {focus}"
        if focus
        else "Summarize the content below: purpose, structure, and notable details."
    )
    result, usage = _generate(f"{task}\n\n{gathered}", max_tokens)
    _log_usage(
        "summarize",
        usage,
        input_tokens_avoided=_est_tokens(file_chars),
        output_tokens_avoided=0,
        returned_chars=len(result),
    )
    return result


def _build_report() -> str:
    if not USAGE_LOG.exists():
        return "No delegated calls logged yet."
    entries = [json.loads(line) for line in USAGE_LOG.read_text().splitlines() if line]
    calls = len(entries)
    input_avoided = sum(e["input_tokens_avoided"] for e in entries)
    output_avoided = sum(e["output_tokens_avoided"] for e in entries)
    returned = sum(e["tokens_returned_to_claude"] for e in entries)
    net_input = input_avoided - returned
    first = entries[0]["ts"][:10]

    lines = [
        f"Local-LLM savings since {first} ({calls} delegated calls)",
        "",
        f"  Input tokens avoided (files Claude never read):    {input_avoided:>10,}",
        f"  Output tokens avoided (content written to disk):   {output_avoided:>10,}",
        f"  Tokens returned into Claude's context (cost):      {returned:>10,}",
        f"  Net input tokens saved:                            {net_input:>10,}",
        "",
        "  Estimated cost saved (net input + avoided output):",
    ]
    for model, (in_price, out_price) in CLAUDE_PRICING.items():
        saved = (net_input * in_price + output_avoided * out_price) / 1_000_000
        lines.append(f"    at {model} rates:  ${saved:,.4f}")
    lines.append("")
    lines.append(
        "  Counts are approximate: file/return sizes use chars/4; avoided "
        "output uses the local model's exact completion tokens. Tokenizers "
        "differ slightly from Claude's."
    )
    return "\n".join(lines)


@mcp.tool()
def savings_report() -> str:
    """Report approximate Claude tokens and dollars saved so far by
    delegating work to the local model. Use when the user asks how much the
    local-LLM offloading has saved."""
    return _build_report()


if __name__ == "__main__":
    if "--report" in sys.argv:
        print(_build_report())
    else:
        mcp.run()
