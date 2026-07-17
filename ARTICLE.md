# Teaching Claude Code to Delegate: A Local LLM as the Grunt-Work Engine

*How we cut Claude token burn by pairing it with a Qwen3-14B running on the
Mac it already lives on — the intuitions, the research we borrowed, and a
staged evaluation showing where each iteration earned its keep.*

## The problem

Claude Code is a frontier-model agent, and frontier tokens are the most
expensive tokens you can buy — output tokens especially, at 5x the price of
input ($10 vs $50 per million on Fable 5). Yet a large share of what a
coding agent actually does is *not* frontier work: reading a 3,000-line log
to find the one error, generating 500 rows of mock data, adding docstrings
to a file. Every one of those tokens flows through the same expensive
context window, burning API dollars or subscription usage limits.

Meanwhile, an M4 Pro with 24 GB of RAM sits under the session doing
nothing. A quantized Qwen3-14B runs on it comfortably via
[mlx-lm](https://github.com/ml-explore/mlx-lm), and its tokens are free.

So: keep Claude as the brain, make the local model the hands. We built this
in three iterations — intuition first, research second, refinement third —
and measured each stage against a raw-Claude baseline.

## The stages

**Stage 0 — raw Fable 5.** No delegation. Claude reads every file into
context and generates every byte of output itself.

**v1 — intuition.** An MCP server exposing `delegate` and `summarize`,
backed by an auto-started `mlx_lm.server`. Two intuitions did the heavy
lifting: (1) tools take **file paths** as input and an **output file** for
results, so bulk content flows disk → local model → disk without entering
Claude's context; (2) the criterion for delegating is **compression
ratio** — delegate when the instruction is much smaller than the content it
produces or consumes. Plus a flat 400-line read-blocking hook, a management
CLI, and — crucially — a ledger logging every call, so later stages could
be judged with numbers.

**v2 — research.** We went looking for prior art and found
[MinionS](https://arxiv.org/abs/2502.15964) (Stanford, ICML 2025), which
formalized exactly this local/frontier split: naive delegation recovers
only **87%** of frontier quality because small models fail on long contexts
and multi-step instructions; decomposing into single-step questions over
small chunks, with per-chunk *abstention*, recovers **97.9%** at 5.7x cost
reduction. v2 rebuilt `summarize` as chunked map-reduce, tiered the read
guard by file kind, and added URL support.

**v3 — the edit tool.** Borrowed from
[aider's architect/editor mode](https://aider.chat/2024/09/26/architect.html):
Claude describes a mechanical rewrite in prose, Qwen emits the edited file,
Claude reviews via `git diff` — paying cheap input tokens to review instead
of 5x-priced output tokens to generate.

## Staged evaluation

**Method.** A four-task suite built from the artifacts we actually tested
with, scored per stage on Claude-side tokens. File sizes, local-model
outputs, failures, and timings are measured; Claude-side tool-call overhead
(instructions emitted, confirmations read) is estimated at chars/4 and
labeled as such. Dollar figures use Fable 5 rates ($10/M input, $50/M
output). Hardware: M4 Pro, 24 GB, Qwen3-14B-4bit.

The suite:

- **A.** Diagnose a 43K-char incident log (three related errors planted at
  start/middle/end among ~670 noise lines): what errors, what order, what
  causal chain?
- **B.** Add a docstring to every function in a Python file (7 functions).
- **C.** Rename a class and its attributes throughout a file.
- **D.** Generate 20 rows of mock CSV data.

**Claude tokens per task per stage** (input / output, estimates marked ~):

| Task | Stage 0 (raw) | v1 (intuition) | v2 (+MinionS) | v3 (+edit) |
|---|---|---|---|---|
| A: log diagnosis | 10,768 / ~180 | **failed** → fell back to raw: ~10,830 / ~260 | ~150 / ~70 | ~150 / ~70 |
| B: docstring pass | ~150 / ~385 | same as raw | same as raw | ~330 / ~50 |
| C: rename | ~170 / ~300 | same as raw | same as raw | ~350 / ~55 |
| D: mock CSV | 0 / ~320 | ~100 / ~45 | ~100 / ~45 | ~100 / ~45 |
| **Suite total** | **11,088 / 1,185** | **11,250 / 990** | **730 / 800** | **930 / 220** |
| **Est. cost (Fable 5)** | **$0.170** | **$0.162** | **$0.047** | **$0.020** |
| vs. baseline | — | 1.05x | **3.6x** | **8.4x** |

Three things this table shows that a headline number wouldn't:

**v1 barely moved the needle — and regressed on the hardest task.** Task A
is the measured centerpiece: v1's `summarize` stuffed the whole 43K-char
log into one prompt, and the request **crashed the GPU with a Metal
out-of-memory error** after 10 minutes, wedging the model server. Claude's
fallback is reading the file raw, so v1 cost slightly *more* than no
delegation at all. (v1 worked fine on inputs under ~12K chars — the suite's
log is exactly the file the system exists for, which is the point.) On
consumer hardware, chunking isn't a quality optimization; it's feasibility.

**v2 is where the input savings actually landed.** Map-reduce processed the
same log in 3m 58s: 3/3 planted errors, correct order, correct causal
chain, ~150 tokens returned instead of 10,768 read. Suite input tokens
dropped 15x.

**v3 is where the *output* savings landed — the 5x-priced kind.** Suite
output tokens fell from 1,185 to 220. Reviewing a diff (input) replaced
generating edits (output), which is why v3 nearly halves the cost again
despite *raising* input tokens slightly.

### The A/Bs inside the stage transitions

The stage jumps were not free — two of them shipped broken first, and
side-by-side runs caught both:

**v2's map prompt, first attempt vs. fix** (same log, same question):

| Map-step prompt design | Time | Result |
|---|---|---|
| Full compound question per chunk | 2m 24s | **Wrong** — every chunk abstained ("no relevant content") |
| Extract evidence per chunk, answer in reduce | 3m 58s | **Correct** — 3/3 errors, order, causal chain |

We violated the MinionS "single-step instructions" rule we had just cited:
no single chunk can answer "in what order?", so each chunk *correctly*
concluded it couldn't answer and abstained. Splitting map (extract) from
reduce (answer) fixed it.

**v3's edit prompt, baseline vs. exhaustiveness demand** (same file, same
instruction):

| Prompt | Coverage | Collateral changes |
|---|---|---|
| Baseline | 4/7 functions | none |
| + "enumerate every occurrence, apply EXHAUSTIVELY" | **7/7** | none |

Small models under-apply "every X" instructions — they quietly stop early.
One explicit demand closed the gap. The rename test added a qualitative
lesson: every *code* occurrence was converted perfectly, but a docstring
mention of the old name survived — caught by the mandatory `git diff`
review for a few dozen input tokens. The architect/editor split only works
if the architect actually reviews.

### The running ledger

Independent of the suite, every real call is logged. After day one of
building and testing (9 delegated calls): 24,730 input tokens avoided, 701
output tokens avoided, 1,112 tokens returned into context — net ~23.6K
Claude tokens that never existed. Small absolute numbers, but they're
measured, and the ledger is what will decide the next iteration.

## Honest limitations

- **Latency.** Qwen3-14B generates at ~25 tok/s. Map-reduce on a 40K-char
  file takes ~4 minutes. Delegation pays in tokens, not time.
- **Small-model compliance.** "Every X" under-application, judgment-call
  remnants, occasional fence-wrapping. Mitigations: exhaustiveness prompts,
  truncation guards (never write a half-file), mandatory review.
- **Estimates are estimates.** The suite's Claude-side overheads are chars/4
  approximations; the stage *rankings* are robust to that, the exact
  multipliers less so.
- **The suite is favorable terrain.** It's built from delegable tasks; a
  session of pure debugging would delegate little and save little. The
  ledger, not the suite, is the long-run scoreboard.

## Takeaways

1. Keep orchestration with the strongest model. The delegation *decision*
   is the reasoning task; pushing it down-stack re-creates the problem.
2. Enforce what's rule-detectable (a read hook can block spend before it
   happens); steer what isn't (a write's tokens are spent before any hook
   fires — only planning-time steering can save them).
3. Never hand a small model a huge context or a compound instruction —
   decompose. The paper said so; our A/B proved it within the hour.
4. Intuition got the architecture right (v1's path-based I/O survived
   unchanged); research got the *protocols* right (v2/v3 is where 8x came
   from); measurement kept everyone honest.

*Code: [github.com/ZackHolmberg/claude-local-llm](https://github.com/ZackHolmberg/claude-local-llm).
Research notes and roadmap in `IDEAS.md`.*
