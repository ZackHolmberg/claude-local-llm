# Research notes & roadmap

What others are doing with the frontier-orchestrator / local-worker pattern,
and what's worth stealing. (Researched 2026-07.)

## Prior art

### Minions / MinionS — Stanford Hazy Research (ICML 2025)

[Paper](https://arxiv.org/abs/2502.15964). Exactly our pattern, formalized:
a small on-device model with access to local data collaborates with a
frontier cloud model. Two protocols, benchmarked on long-document reasoning:

- **Minion (naive chat):** local and remote models converse; only the local
  model reads the full context. 30x cloud-cost reduction but only **87%** of
  frontier quality — small models fail at multi-step instructions (-56%) and
  degrade on long contexts (-13% beyond 65K tokens).
- **MinionS (decompose-execute-aggregate):** the remote model writes a
  *decomposition* (single-step instructions over small chunks); the local
  model executes chunks in parallel with structured output
  (explanation/citation/answer) and may **abstain** per chunk; abstentions
  are filtered; the remote model synthesizes survivors and can loop.
  **97.9%** of frontier quality at 5.7x cost reduction.

Lessons that map directly onto this project:

1. **Never hand the local model a huge context in one shot** — our current
   `summarize` truncates >80K chars head/tail and stuffs the rest into one
   prompt, which is the exact failure mode the paper measures. Chunked
   map-reduce with abstention is the fix.
2. **Single-step instructions per chunk** beat clever multi-part prompts.
3. **Local tokens are free** — self-consistency (sample k answers, merge)
   buys reliability for latency only.
4. Knobs: chunk size (smaller = better quality, more prefill), samples per
   task, rounds of remote-local iteration.

### Aider architect/editor mode

[Writeup](https://aider.chat/2024/09/26/architect.html). The inverse split:
a strong model *describes* the change in prose; a weaker model materializes
the actual file edits. Produced SOTA on aider's editing benchmark. Validates
the compression-ratio rule from the other direction and suggests a concrete
tool we don't have: Claude writes a prose edit spec, Qwen emits the edited
file, Claude reviews via `git diff` (cheap: reading a diff is input tokens,
writing the file would have been 5x-priced output tokens).

### MCP ecosystem cousins

Several projects wire Claude Code to Ollama the same way we wire to mlx-lm:
[cc_token_saver_mcp](https://github.com/csabakecskemeti/cc_token_saver_mcp),
[claude-sidekick](https://github.com/andrewbrereton/claude-sidekick),
[mcp-local-llm](https://github.com/aplaceforallmystuff/mcp-local-llm),
[OllamaClaude](https://github.com/Jadael/OllamaClaude),
[local-delegate](https://glama.ai/mcp/servers/ZahiriNatZuke/local-delegate).
Same architecture (Claude orchestrates, local model grunts, files read
server-side). None appear to have: the enforcement hook, the savings ledger,
or MinionS-style chunking. Our differentiators are measurement + enforcement.

## Roadmap candidates (rough priority)

1. **Map-reduce summarize with abstention** — replace head/tail truncation:
   chunk big inputs (~6-8K chars), run the focus question per chunk with an
   explicit "reply NOT_RELEVANT if the chunk doesn't bear on the question"
   escape, drop abstentions, then a final local reduce pass over survivors.
   Fixes our biggest quality gap on exactly the files the hook deflects.
2. **`edit` tool (architect/editor)** — `edit(instruction, input_file,
   output_file)`: Claude specs the change in prose, Qwen emits the full
   edited file, Claude spot-checks with `git diff`. Targets bulk mechanical
   rewrites (renames across a file, docstring passes, format migrations).
3. **Self-consistency knob** — `samples: int` on delegate/summarize; run k
   generations and have Qwen merge/vote. Local tokens are free; only costs
   latency. Worth it for extraction where a wrong answer is worse than slow.
4. **Multi-round loop** — let Claude ask a follow-up against the *same*
   gathered input without re-reading files (cache the last gather server-side,
   keyed by hash). Saves re-prefill on iterative interrogation of one artifact.
