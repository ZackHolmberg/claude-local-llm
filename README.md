# claude-local-llm

Offloads mechanical, non-reasoning tasks from Claude Code to a local Qwen
model running on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm),
to cut Claude token usage.

**New here?** [ARTICLE.md](ARTICLE.md) is the story — the problem, the
design intuitions, the research this borrows from (MinionS, aider), and
measured A/B results. [IDEAS.md](IDEAS.md) is the research log and roadmap.

## How it works

```
Claude Code ──(MCP stdio)──> server.py ──(HTTP :8734)──> mlx_lm.server (Qwen3-14B-4bit)
```

- `server.py` is registered as the `local-llm` MCP server (user scope, all
  projects). It auto-starts `mlx_lm.server` on first tool call, and the model
  server is shared across all Claude Code sessions.
- Tools accept **file paths** as input and an optional **output file**, so
  large content flows disk → local model → disk without entering Claude's
  context.
- `~/.claude/CLAUDE.md` tells Claude when to delegate.

## Installation

Requires Apple Silicon. Clone, then:

```sh
./install.sh                               # uv + venv + deps + Claude Code MCP registration
./llm pull mlx-community/Qwen3-14B-4bit    # download a model (~8 GB; needs ~24 GB RAM)
./llm start                                # optional - the server auto-starts on first tool call
./llm status                               # verify
```

Smaller machines: `mlx-community/Qwen3-8B-4bit` (~4.5 GB) then `./llm use` it.
Re-running `install.sh` is safe (idempotent).

## CLI

`./llm <command>` (a wrapper around `cli.py` running in the project venv):

| Command | What it does |
|---|---|
| `status` | Configured model, server state, ledger call count |
| `start` / `stop` / `restart` | Manage the mlx model server (`stop` frees ~8 GB RAM) |
| `pull <model>` | Download a model from HuggingFace |
| `models` | List locally cached models, marking the configured one |
| `use <model>` | Switch models (persists to `config.json`; restarts if running) |
| `ask "<prompt>"` | One-off generation, for testing (`--max-tokens N`) |
| `report` | Token/cost savings report (same data as the `savings_report` MCP tool) |
| `logs [-f]` | Show (or follow) the mlx server log |

## Tools

- **`summarize`** — summarize / answer focused questions about large files,
  logs, diffs, docs, or **web pages** (pass http(s) URLs alongside paths).
  Used instead of Read for big files and instead of WebFetch for long pages.
  Inputs over ~12K chars are processed MinionS-style
  ([arXiv:2502.15964](https://arxiv.org/abs/2502.15964)): split into ~7K-char
  chunks, evidence extracted per chunk (with per-chunk abstention on
  irrelevant chunks), then a local reduce pass answers the actual question —
  small models degrade badly when handed one huge context, so this recovers
  near-frontier quality on big inputs. Expect ~1 min per 15K chars of input;
  very large files take a few minutes.
- **`delegate`** — general mechanical generation/transformation: boilerplate,
  fixtures, mock data, format conversion, docstrings. With `output_file`,
  bulk output goes straight to disk.
- **`edit`** — architect/editor split (inspired by
  [aider](https://aider.chat/2024/09/26/architect.html)): Claude describes a
  mechanical rewrite in prose (rename a symbol throughout, docstring/comment
  pass, style migration), the local model emits the edited file, and Claude
  reviews via `git diff` — cheap input tokens instead of 5x-priced output
  tokens. Single file, ≤40K chars; refuses to write anything if the local
  model's output was truncated; the confirmation includes a diff preview and
  a review reminder.

## Large-read guard (hook)

Steering via CLAUDE.md is advisory — Claude doesn't always remember to
delegate. `hooks/read_guard.py` makes the clear-cut case deterministic: a
`PreToolUse` hook (registered in `~/.claude/settings.json` by `install.sh`)
intercepts Claude's `Read` calls and **blocks** full reads of large text
files with a message steering Claude to `summarize` instead.

Thresholds are **tiered by file kind**, encoding *why* a file gets read:

| Tier | Matches | Limit | Rationale |
|---|---|---|---|
| data | `.log` `.csv` `.jsonl`, lockfiles, `.min.*`, `node_modules`/`vendor`/`dist`/`build` paths | 150 lines / 32 KB | The gist is almost always enough |
| source | `.py` `.ts` `.go` `.rs` etc. | 800 lines / 128 KB | Claude often needs code verbatim to edit correctly |
| default | everything else (docs, configs) | 400 lines / 64 KB | In between |

Other rules:

- `LOCAL_LLM_HOOK_MAX_LINES` / `LOCAL_LLM_HOOK_MAX_BYTES` env vars override
  whichever tier matched.
- **Ranged Reads (offset/limit) always pass** — the escape hatch when Claude
  needs exact contents, e.g. before an Edit.
- Images, PDFs, and notebooks are exempt (Read handles those natively).
- Fails open: malformed input or unreadable files never block.

Judgment stays with Claude for everything fuzzy; the hook only enforces the
unambiguous case. Disable it by removing the `PreToolUse` entry from
`~/.claude/settings.json` (or via the `/hooks` menu in Claude Code).

## Status line (live visibility)

`statusline.py` renders a Claude Code status line so delegation is visible
without digging through the transcript:

```
Fable 5 | ⚡ local model working            <- a delegated call is running now
Fable 5 | 🦙 3 delegated · ~12,400 tok saved today
Fable 5 | 🦙 local-llm ready               <- no calls yet today
```

The server writes a `.active` marker for the duration of every local
generation; the status line (refreshed every 30s) picks it up, and shows
today's call count / net tokens saved / failures from the ledger otherwise.
`install.sh` registers it only if you don't already have a status line.
Delegated calls also appear inline in the transcript as
`mcp__local-llm__*` tool uses.

## Usage tracking & savings report

Every `delegate`/`summarize` call appends one line to `usage.jsonl` (in this
directory), using the mlx server's exact token counts where available.

### Viewing the report

Two equivalent ways:

1. **Ask Claude** — e.g. *"how much has the local model saved me?"* Claude
   calls the `savings_report` MCP tool.
2. **From the shell:** `./llm report`

Sample output:

```
Local-LLM savings since 2026-07-17 (2 delegated calls)

  Input tokens avoided (files Claude never read):         2,595
  Output tokens avoided (content written to disk):          487
  Tokens returned into Claude's context (cost):             282
  Net input tokens saved:                                 2,313

  Estimated cost saved (net input + avoided output):
    at claude-opus-4-8 rates:  $0.0237
    at claude-sonnet-5 rates:  $0.0142
    at claude-fable-5 rates:  $0.0475
```

On a subscription plan, the token counts are the meaningful numbers (they map
to how fast you burn usage limits); the dollar figures are the pay-per-token
API equivalent, at pricing current as of July 2026 (`CLAUDE_PRICING` in
`server.py`).

### Ledger format

`usage.jsonl` — one JSON object per event. `event: "call"` entries (one per
tool invocation, including failures):

| Field | Meaning |
|---|---|
| `ts`, `tool`, `model` | UTC timestamp, tool name, local model ID |
| `local_prompt_tokens` / `local_completion_tokens` | Exact counts from the mlx server (Qwen tokenizer) |
| `input_tokens_avoided` | File/URL content the local model read instead of Claude (chars/4) |
| `output_tokens_avoided` | Content written to disk instead of generated by Claude |
| `tokens_returned_to_claude` | Tool output that entered Claude's context — counted against savings (chars/4) |
| `duration_ms` | Wall-clock latency of the call |
| `sources` | The file paths / URLs processed |
| `mode` / `n_chunks` | summarize only: `single_shot` or `map_reduce`, and chunk count |
| `error` | Present on failures: refusal codes (`file_too_large`, `output_truncated`, `empty_output`) or the exception |

`event: "deflection"` entries are written by the read-guard hook whenever it
blocks a Read (`file`, `tier`, `size_bytes`, `lines`). The report joins
these against summarize `sources` to show the **funnel**: how many
deflections were followed by delegation vs. worked around — plus per-tool
reliability (failure rates by error code) and latency (avg/max, map-reduce
chunk counts). Together these answer the after-a-week questions: how much
was saved, how often delegation was used vs. evaded, how often it failed,
and what it cost in time. (Answer *quality* isn't capturable in a ledger —
a summarize call on a file Claude then re-reads anyway is the
dissatisfaction signal to grep for.)

To start a fresh measurement period, delete `usage.jsonl` — day-one dev and
test entries are in it otherwise.

Since it's JSONL, ad-hoc slicing is easy, e.g. savings by day:

```sh
jq -r '[.ts[:10], .input_tokens_avoided + .output_tokens_avoided] | @tsv' usage.jsonl |
  awk '{sum[$1] += $2} END {for (d in sum) print d, sum[d]}' | sort
```

### Accounting rules

- **File inputs** count as input tokens avoided — Claude never read them.
- **`output_file` content** counts as output tokens avoided — Claude never
  generated it (output tokens cost 5x input, so this is the biggest lever).
- **Anything a tool returns** counts *against* savings — it enters Claude's
  context as input.
- **`input_text`** counts as zero savings — it came from Claude's context.

Counts are approximate: chars/4 estimates plus the Qwen tokenizer as a proxy
for Claude's. Treat them as directionally right, not billing-grade.

## Configuration

Environment variables (set on the MCP server entry in `~/.claude.json`):

| Variable          | Default                        |
|-------------------|--------------------------------|
| `LOCAL_LLM_MODEL` | `mlx-community/Qwen3-14B-4bit` |
| `LOCAL_LLM_PORT`  | `8734`                         |

## Operations without the CLI

Everything the CLI does maps to plain commands if you need them:

```sh
curl -s http://127.0.0.1:8734/v1/models        # status
pkill -f mlx_lm.server                          # stop
tail -f mlx-server.log                          # logs
.venv/bin/python server.py --report             # savings report
```

Model/port precedence is `LOCAL_LLM_MODEL`/`LOCAL_LLM_PORT` env vars (set on
the MCP entry in `~/.claude.json`) > `config.json` (written by `llm use`) >
built-in default. After moving this directory, re-run `./install.sh` to
re-register the MCP server with the new paths.
