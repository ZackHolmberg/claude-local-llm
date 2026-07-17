"""Claude Code status line: live visibility into local-llm delegation.

Shows the Claude model, whether the local model is generating right now,
and today's delegation stats from the ledger.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    try:
        session = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        session = {}
    parts = []
    model = (session.get("model") or {}).get("display_name")
    if model:
        parts.append(model)

    active = ROOT / ".active"
    working = False
    try:
        if active.exists() and time.time() - active.stat().st_mtime < 900:
            working = True
            parts.append("⚡ local model working")
    except OSError:
        pass

    calls = 0
    saved = 0
    failed = 0
    today = datetime.now(timezone.utc).isoformat()[:10]
    ledger = ROOT / "usage.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event", "call") != "call" or e.get("ts", "")[:10] != today:
                continue
            calls += 1
            if "error" in e:
                failed += 1
            saved += (
                e.get("input_tokens_avoided", 0)
                + e.get("output_tokens_avoided", 0)
                - e.get("tokens_returned_to_claude", 0)
            )
    if calls:
        stat = f"🦙 {calls} delegated · ~{saved:,} tok saved today"
        if failed:
            stat += f" · {failed} failed"
        parts.append(stat)
    elif not working:
        parts.append("🦙 local-llm ready")

    print(" | ".join(parts))


if __name__ == "__main__":
    main()
