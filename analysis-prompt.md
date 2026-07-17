# Week-one analysis prompt

Paste the block below into a fresh Claude Code session after ~a week of
normal use with the local-llm delegation system.

---

Analyze the first real measurement period of my local-llm delegation system
(repo: `~/dev/claude-local-llm` — read its README.md for the ledger schema,
IDEAS.md for the roadmap, and ARTICLE.md for the staged evaluation this
data will be compared against).

**Part 1 — Analysis.** Work from `usage.jsonl` (and `./llm report`). All
numbers must come from the ledger — no estimates presented as measurements.
Answer:

1. **Adoption**: delegated calls per day (trend over the week), split by
   tool. Did delegation actually happen in daily work, or only when the
   hook forced it?
2. **Savings**: net tokens saved (input vs the 5x-priced output separately),
   dollar equivalents, and which tool/file-type contributed most. Compare
   the input:output savings ratio against the day-one expectation in
   ARTICLE.md that output savings would grow as write-side steering got
   exercised — did they?
3. **Funnel & evasion**: deflections vs. deflections followed by a
   summarize of the same file. A low follow-through rate means the hook is
   being worked around with ranged Reads — if so, say it plainly and
   diagnose (thresholds too aggressive? tiers misclassifying?).
4. **Reliability**: failure rate per tool, by error code. Any timeouts or
   OOM-shaped failures (look for long-duration errors on large inputs)?
5. **Latency**: distribution per tool (median/p90/max), map-reduce chunk
   counts, and the total wall-clock time spent waiting on the local model
   this week vs. tokens saved — state the implied "tokens saved per minute
   of waiting" so the trade is explicit.
6. **Quality proxy**: sources that were summarized but show repeated
   summarize calls or a deflection *after* a summarize (signs the summary
   didn't suffice).

Before drawing conclusions, ask me 2–3 questions about my subjective
experience (Was the hook ever annoying? Did any summary mislead you? Did
you notice the latency?) — the ledger can't see quality or friction, and
the writeup should include the human verdict alongside the numbers.

**Part 2 — Recommendations.** Based on the data, propose (don't apply
without asking): threshold/tier tuning for the read guard, steering changes
if a tool went unused, and which IDEAS.md roadmap item the data now
justifies (or doesn't). If the data says something we built isn't earning
its keep, recommend removing it — deletions count as optimizations.

**Part 3 — Write it up.** Append a "Week One: Real-World Results" section
to ARTICLE.md: the numbers, how they compare to the day-one staged
evaluation (the suite was favorable terrain — did the 8.4x hold, shrink, or
collapse in real use?), the subjective verdict, and what changes next.
Match the article's existing voice: lead with outcomes, honest about
disappointments, every number sourced. Show me the section before
committing; after I approve, commit and push.
