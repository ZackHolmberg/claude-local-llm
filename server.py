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
# Small models degrade on long contexts (MinionS, arXiv:2502.15964), so
# summarize switches to chunked map-reduce with abstention above this size.
#
# Measured on Qwen3-14B-4bit with a 40k-char source file and a focus question
# with known ground truth (12 CLI flags to recover):
#     single-pass       70.2s   12/12 flags
#     map-reduce @24k   74.5s   12/12 flags
#     map-reduce @7k    89.0s   12/12 flags
# No accuracy cliff at 40k, and map-reduce costs ~20% more wall-clock there
# (each chunk re-pays prompt-processing overhead), so the threshold sits above
# it. Beyond ~40k the MinionS degradation risk is real and untested, so keep
# chunking -- but with chunks large enough not to shred the input.
MAP_REDUCE_THRESHOLD = 40_000
CHUNK_CHARS = 20_000
ABSTAIN = "NOT_RELEVANT"
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


def _html_to_text(html_src: str) -> str:
    from html import unescape

    text = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html_src)
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr|/section|/article)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _fetch_url(url: str) -> str:
    resp = httpx.get(
        url,
        timeout=30,
        follow_redirects=True,
        headers={"User-Agent": "claude-local-llm/1.0"},
    )
    resp.raise_for_status()
    if "html" in resp.headers.get("content-type", ""):
        return _html_to_text(resp.text)
    return resp.text


def _gather_input(
    input_files: list[str] | None, input_text: str | None
) -> tuple[str, int]:
    """Returns (combined input text, chars that came from files/URLs).

    File- and URL-derived chars are what Claude avoided reading; `input_text`
    came from Claude's context already, so it never counts as savings.
    """
    parts: list[str] = []
    file_chars = 0
    budget = MAX_INPUT_CHARS
    for raw_path in input_files or []:
        if raw_path.startswith(("http://", "https://")):
            try:
                text = _fetch_url(raw_path)
            except httpx.HTTPError as e:
                parts.append(f'<url href="{raw_path}" error="{e}"/>')
                continue
            if len(text) > budget:
                text = text[: max(budget, 2_000)] + "\n...[truncated]..."
            budget -= len(text)
            file_chars += len(text)
            parts.append(f'<url href="{raw_path}">\n{text}\n</url>')
            if budget <= 0:
                parts.append("[input budget exhausted; remaining inputs omitted]")
                break
            continue
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


ACTIVE_MARKER = PROJECT_DIR / ".active"


def _mark_active(tool: str) -> None:
    try:
        ACTIVE_MARKER.write_text(json.dumps({"tool": tool, "started": time.time()}))
    except OSError:
        pass


def _clear_active() -> None:
    ACTIVE_MARKER.unlink(missing_ok=True)


def _log_usage(
    tool: str,
    usage: dict,
    input_tokens_avoided: int,
    output_tokens_avoided: int,
    returned_chars: int,
    *,
    duration_ms: float | None = None,
    sources: list[str] | None = None,
    mode: str | None = None,
    n_chunks: int | None = None,
    error: str | None = None,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "call",
        "tool": tool,
        "model": MODEL,
        "local_prompt_tokens": usage.get("prompt_tokens", 0),
        "local_completion_tokens": usage.get("completion_tokens", 0),
        "input_tokens_avoided": input_tokens_avoided,
        "output_tokens_avoided": output_tokens_avoided,
        "tokens_returned_to_claude": _est_tokens(returned_chars),
    }
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms)
    if sources:
        entry["sources"] = sources
    if mode:
        entry["mode"] = mode
    if n_chunks:
        entry["n_chunks"] = n_chunks
    if error:
        entry["error"] = error
    with open(USAGE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _generate(prompt: str, max_tokens: int, temperature: float = 0.2) -> tuple[str, dict]:
    _mark_active("local model")
    try:
        return _generate_inner(prompt, max_tokens, temperature)
    finally:
        _clear_active()


def _generate_inner(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
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
    choice = data["choices"][0]
    # mlx-lm >=0.31 returns chain-of-thought in a separate `reasoning` field and
    # omits `content` entirely when the token budget runs out mid-thought. The
    # /no_think system prompt normally prevents that, but a small max_tokens can
    # still trip it, and a bare ["content"] would raise KeyError instead of
    # reporting a usable error.
    message = choice.get("message") or {}
    raw = message.get("content")
    if not raw:
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning:
            raise RuntimeError(
                f"local model produced only reasoning, no answer "
                f"(finish_reason={choice.get('finish_reason')}). Raise max_tokens."
            )
        raw = ""
    content = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    usage = data.get("usage", {})
    usage["finish_reason"] = choice.get("finish_reason")
    return content.strip(), usage


@mcp.tool()
def delegate(
    instruction: str,
    input_files: list[str] | None = None,
    input_text: str | None = None,
    output_file: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Offload a mechanical task to a fast local model to save Claude tokens.

    The rule is compression ratio: delegate when your instruction is much
    smaller than the content it produces or consumes, and correctness is
    checkable at a glance. ALWAYS delegate (with `output_file`): test
    fixtures, mock/sample data, boilerplate and scaffolding, format
    conversions, docstring/comment passes, README/doc drafts, repetitive
    near-identical code. NEVER delegate: logic, algorithms, debugging,
    anything you would need to reason about line-by-line - if specifying the
    task takes as many tokens as doing it, do it yourself.

    Pass large inputs via `input_files` (absolute paths or http(s) URLs)
    instead of pasting content, and set `output_file` for bulk output - then
    the content never enters your context and you only see a short
    confirmation. Give a self-contained instruction: the local model sees
    ONLY what you pass here, not the conversation.
    """
    t0 = time.monotonic()
    try:
        gathered, file_chars = _gather_input(input_files, input_text)
        prompt = f"{instruction}\n\n{gathered}" if gathered else instruction
        result, usage = _generate(prompt, max_tokens)
    except Exception as e:
        _log_usage(
            "delegate", {}, 0, 0, 0,
            duration_ms=(time.monotonic() - t0) * 1000,
            sources=input_files,
            error=f"{type(e).__name__}: {e}",
        )
        raise
    duration_ms = (time.monotonic() - t0) * 1000
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
            duration_ms=duration_ms,
            sources=input_files,
        )
        return reply
    _log_usage(
        "delegate",
        usage,
        input_tokens_avoided=_est_tokens(file_chars),
        output_tokens_avoided=0,
        returned_chars=len(result),
        duration_ms=duration_ms,
        sources=input_files,
    )
    return result


@mcp.tool()
def summarize(
    input_files: list[str] | None = None,
    input_text: str | None = None,
    focus: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Summarize large files, logs, diffs, docs, or web pages with a local
    model instead of reading them yourself - a major token saving whenever
    the content is big and you only need the gist or a specific answer from
    it. Pass absolute paths or http(s) URLs via `input_files`; use `focus`
    to ask a pointed question (e.g. "which requests failed and why?").
    Prefer this over Read for any file over a few hundred lines, and over
    WebFetch for long documentation pages, when you don't need exact
    contents.
    """
    t0 = time.monotonic()
    try:
        gathered, file_chars = _gather_input(input_files, input_text)
        if not gathered:
            return "Error: provide input_files and/or input_text."
        task = (
            f"Answer this about the content below, citing specifics: {focus}"
            if focus
            else "Summarize the content below: purpose, structure, and notable details."
        )
        if len(gathered) > MAP_REDUCE_THRESHOLD:
            result, usage = _map_reduce(task, gathered, max_tokens)
            mode = "map_reduce"
        else:
            result, usage = _generate(f"{task}\n\n{gathered}", max_tokens)
            mode = "single_shot"
    except Exception as e:
        _log_usage(
            "summarize", {}, 0, 0, 0,
            duration_ms=(time.monotonic() - t0) * 1000,
            sources=input_files,
            error=f"{type(e).__name__}: {e}",
        )
        raise
    _log_usage(
        "summarize",
        usage,
        input_tokens_avoided=_est_tokens(file_chars),
        output_tokens_avoided=0,
        returned_chars=len(result),
        duration_ms=(time.monotonic() - t0) * 1000,
        sources=input_files,
        mode=mode,
        n_chunks=usage.pop("n_chunks", None),
    )
    return result


def _chunk(text: str) -> list[str]:
    """Split on line boundaries into ~CHUNK_CHARS pieces."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines(keepends=True):
        if size + len(line) > CHUNK_CHARS and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _acc(total: dict, usage: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens"):
        total[key] = total.get(key, 0) + usage.get(key, 0)


def _map_reduce(task: str, gathered: str, max_tokens: int) -> tuple[str, dict]:
    """MinionS-style summarize: single-step question per small chunk, with
    per-chunk abstention, then a local reduce pass over the survivors."""
    chunks = _chunk(gathered)
    total: dict = {}
    answers: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        # Map = single-step evidence extraction, not the full task: a chunk
        # alone often can't answer a compound question, and asking it to
        # causes mass abstention. The reduce step answers the actual task.
        prompt = (
            f"You are seeing excerpt {i} of {len(chunks)} of a larger input. "
            f"Extract every fact, quote, or detail from THIS excerpt that is "
            f"relevant to the task below, as brief bullet points. Do not try "
            f"to answer the task itself - other excerpts exist. Only if "
            f"nothing in this excerpt relates to the task at all, reply "
            f"exactly {ABSTAIN}.\n\nTask: {task}\n\n{chunk}"
        )
        text, usage = _generate(prompt, max_tokens=min(max_tokens, 512))
        _acc(total, usage)
        if ABSTAIN in text and len(text) < 40:
            continue
        answers.append(f"[excerpt {i}] {text}")
    total["n_chunks"] = len(chunks)
    if not answers:
        return (
            "No relevant content found for this task in the provided input.",
            total,
        )
    if len(answers) == 1:
        return answers[0].split("] ", 1)[1], total
    combined = "\n\n".join(answers)[: MAX_INPUT_CHARS // 2]
    text, usage = _generate(
        f"Below is evidence extracted from consecutive excerpts of one "
        f"large input. Using only this evidence, give a single coherent, "
        f"non-redundant answer to this task: {task}\n\n{combined}",
        max_tokens,
    )
    _acc(total, usage)
    return text, total


EDIT_MAX_CHARS = 40_000


@mcp.tool()
def edit(
    instruction: str,
    input_file: str,
    output_file: str | None = None,
) -> str:
    """Architect/editor split: you describe a mechanical edit in prose, the
    local model rewrites the file, and you review cheaply with `git diff` -
    reading a diff costs input tokens instead of generating the whole change
    as 5x-priced output tokens.

    Use for bulk mechanical rewrites of one file: renaming a symbol
    throughout, docstring/comment passes, reformatting or style migrations,
    converting a repeated pattern, reordering/sorting entries. NOT for logic
    changes, bug fixes, or anything needing line-by-line reasoning - use
    your own Edit tool for those.

    The instruction must be self-contained (the local model sees only the
    file, not the conversation). `output_file` defaults to editing in place.
    ALWAYS review the result afterwards - `git diff <file>` if tracked,
    otherwise a ranged Read of the changed regions."""
    import difflib

    t0 = time.monotonic()

    def _fail(reply: str, code: str, usage: dict | None = None) -> str:
        _log_usage(
            "edit", usage or {}, 0, 0, len(reply),
            duration_ms=(time.monotonic() - t0) * 1000,
            sources=[input_file], error=code,
        )
        return reply

    src = Path(input_file).expanduser()
    original = src.read_text(errors="replace")
    if len(original) > EDIT_MAX_CHARS:
        return _fail(
            f"Error: {src} is {len(original)} chars, over the "
            f"{EDIT_MAX_CHARS}-char single-pass edit limit. Edit it "
            f"yourself with ranged Reads, or split the work.",
            "file_too_large",
        )
    prompt = (
        "Apply the following edit instruction to the file below. Output the "
        "COMPLETE edited file and nothing else - no code fences, no "
        "commentary. Preserve everything not affected by the instruction "
        "exactly, including whitespace, comments, and blank lines. Apply the "
        "instruction EXHAUSTIVELY: if it says 'every' or 'all', first "
        "mentally enumerate every occurrence in the file, then make sure "
        "each one is covered - do not stop after the first few.\n\n"
        f"Edit instruction: {instruction}\n\n"
        f'<file path="{src}">\n{original}\n</file>'
    )
    max_tokens = min(16_384, max(2_048, len(original) // 2))
    try:
        result, usage = _generate(prompt, max_tokens=max_tokens)
    except Exception as e:
        _fail("", f"{type(e).__name__}: {e}")
        raise
    if usage.get("finish_reason") == "length":
        return _fail(
            "Error: the local model's output was truncated before the end "
            "of the file; nothing was written. The file is too large or the "
            "edit too expansive for a single pass - do this edit yourself.",
            "output_truncated",
            usage,
        )
    result = _strip_fences(result)
    if not result.strip():
        return _fail(
            "Error: the local model returned empty output; nothing was written.",
            "empty_output",
            usage,
        )

    out = Path(output_file).expanduser() if output_file else src
    out.write_text(result + ("" if result.endswith("\n") else "\n"))

    diff = list(
        difflib.unified_diff(
            original.splitlines(), result.splitlines(),
            fromfile=str(src), tofile=str(out), lineterm="", n=1,
        )
    )
    changed = [l for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    added_chars = sum(len(l) for l in changed if l.startswith("+"))
    preview = "\n".join(diff[:14])
    reply = (
        f"Edited {out}: {len(original.splitlines())} -> "
        f"{len(result.splitlines())} lines, {len(changed)} diff lines. "
        f"REVIEW REQUIRED (git diff or ranged Read).\nDiff preview:\n{preview}"
    )
    _log_usage(
        "edit",
        usage,
        input_tokens_avoided=_est_tokens(len(original)),
        output_tokens_avoided=_est_tokens(added_chars),
        returned_chars=len(reply),
        duration_ms=(time.monotonic() - t0) * 1000,
        sources=[input_file],
    )
    return reply


def _build_report() -> str:
    if not USAGE_LOG.exists():
        return "No delegated calls logged yet."
    entries = [json.loads(line) for line in USAGE_LOG.read_text().splitlines() if line]
    calls = [e for e in entries if e.get("event", "call") == "call"]
    deflections = [e for e in entries if e.get("event") == "deflection"]
    if not calls and not deflections:
        return "No delegated calls logged yet."
    input_avoided = sum(e["input_tokens_avoided"] for e in calls)
    output_avoided = sum(e["output_tokens_avoided"] for e in calls)
    returned = sum(e["tokens_returned_to_claude"] for e in calls)
    net_input = input_avoided - returned
    first = entries[0]["ts"][:10]
    failed = sum(1 for e in calls if "error" in e)

    lines = [
        f"Local-LLM savings since {first} "
        f"({len(calls)} delegated calls, {failed} failed)",
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

    lines.extend(["", "  Reliability / latency by tool:"])
    for tool in sorted({e["tool"] for e in calls}):
        tcalls = [e for e in calls if e["tool"] == tool]
        errs = [e for e in tcalls if "error" in e]
        durations = [e["duration_ms"] for e in tcalls if "duration_ms" in e]
        stat = f"    {tool}: {len(tcalls)} calls, {len(errs)} failed"
        if durations:
            stat += (
                f", avg {sum(durations) / len(durations) / 1000:.1f}s"
                f", max {max(durations) / 1000:.1f}s"
            )
        mr = [e for e in tcalls if e.get("mode") == "map_reduce"]
        if mr:
            chunks = [e["n_chunks"] for e in mr if e.get("n_chunks")]
            avg_chunks = f", avg {sum(chunks) / len(chunks):.0f} chunks" if chunks else ""
            stat += f" (map-reduce x{len(mr)}{avg_chunks})"
        lines.append(stat)
        for code, count in sorted(
            {e["error"]: sum(1 for x in errs if x["error"] == e["error"]) for e in errs}.items()
        ):
            lines.append(f"      error: {code} x{count}")

    if deflections:
        deflected_files = {e["file"] for e in deflections}
        summarized_files = {
            s for e in calls if e["tool"] == "summarize" for s in e.get("sources") or []
        }
        followed = sum(1 for f in deflected_files if f in summarized_files)
        lines.extend(
            [
                "",
                "  Read-guard funnel:",
                f"    deflections: {len(deflections)} "
                f"({len(deflected_files)} distinct files)",
                f"    later summarized via local model: {followed} files",
                f"    not delegated (ranged-Read workaround or dropped): "
                f"{len(deflected_files) - followed} files",
            ]
        )

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
