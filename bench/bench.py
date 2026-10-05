"""Benchmark candidate local models on the delegation workloads server.py runs.

Each config is served by its own mlx_lm.server on BENCH_PORT (the live server
on 8734 should be stopped first so two models don't compete for memory), and
hit with the same request shape server.py sends. Tasks have exact ground
truth, so accuracy is scored automatically:

    flags_40k/80k/160k     summarize-style: recover the 12 CLI flags defined in
                           a generated Python file of that size (the original
                           Qwen3-14B threshold experiment, plus a long variant)
    csv_json               delegate-style: convert a 60-row CSV to JSON
                           (output-heavy, so it measures generation speed)
    rename                 edit-style: rename one function throughout a file

Usage:
    .venv/bin/python bench/bench.py                    # all configs
    .venv/bin/python bench/bench.py --only gemma4-26b-think --reps 3
"""

import argparse
import csv
import difflib
import io
import json
import random
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_BIN = PROJECT_DIR / ".venv" / "bin"
OUT_DIR = Path(__file__).resolve().parent / "results"
BENCH_PORT = 8735
BASE_URL = f"http://127.0.0.1:{BENCH_PORT}"
TIMEOUT_S = 900

BASE_SYSTEM = (
    "You are a precise assistant handling a delegated subtask. "
    "Follow the instruction exactly. Output only the requested result - no "
    "preamble, no commentary, no markdown fences unless asked for them."
)

# name -> model, system prompt, extra request body
CONFIGS = {
    "qwen3-14b-nothink": {
        "model": "mlx-community/Qwen3-14B-4bit",
        "system": "/no_think " + BASE_SYSTEM,  # what server.py does today
        "extra": {},
    },
    "qwen3.5-9b-nothink": {
        "model": "mlx-community/Qwen3.5-9B-4bit",
        "system": BASE_SYSTEM,  # Qwen3.5 ignores /no_think
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "gemma4-26b-nothink": {
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "system": BASE_SYSTEM,
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "gemma4-26b-think": {
        "model": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "system": BASE_SYSTEM,
        "extra": {"chat_template_kwargs": {"enable_thinking": True}},
    },
}

# ---------------------------------------------------------------- fixtures

TRUE_FLAGS = [
    "--input", "--output-dir", "--format", "--max-retries", "--timeout",
    "--dry-run", "--verbose", "--config", "--workers", "--log-file",
    "--no-cache", "--since",
]
REMOVED_FLAGS = ["--legacy-mode", "--fast"]  # mentioned only in comments

VERBS = ["filter", "merge", "score", "normalize", "dedupe", "rank", "bucket",
         "reconcile", "expand", "collapse", "tag", "validate", "index", "split"]
NOUNS = ["records", "orders", "events", "sessions", "invoices", "shipments",
         "accounts", "metrics", "batches", "segments", "tickets", "payloads"]
FIELDS = ["score", "weight", "amount", "latency", "count", "priority", "age"]


def _filler_fn(rng: random.Random, used: set) -> str:
    while True:
        name = f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{rng.randint(1, 99)}"
        if name not in used:
            used.add(name)
            break
    field = rng.choice(FIELDS)
    other = rng.choice([f for f in FIELDS if f != field])
    n, k = rng.randint(1, 500), rng.randint(2, 9)
    style = rng.randint(0, 2)
    if style == 0:
        return f'''
def {name}(records, threshold={n}):
    """Keep entries whose {field} exceeds threshold, scaling {other} by {k}."""
    result = []
    for rec in records:
        if rec.get("{field}", 0) > threshold:
            result.append({{"id": rec["id"], "{other}": rec.get("{other}", 0) * {k}}})
    logger.debug("{name}: kept %d of %d", len(result), len(records))
    return result
'''
    if style == 1:
        return f'''
def {name}(items, window={k}):
    """Rolling mean of {field} over a window of {k} items."""
    out = []
    acc = 0.0
    for i, item in enumerate(items):
        acc += item["{field}"]
        if i >= window:
            acc -= items[i - window]["{field}"]
        out.append(acc / min(i + 1, window))
    if len(out) > {n}:
        logger.warning("{name}: long series (%d points)", len(out))
    return out
'''
    return f'''
class {name.title().replace("_", "")}:
    """Accumulates {field} totals keyed by {other}."""

    def __init__(self, limit={n}):
        self.limit = limit
        self.totals = {{}}

    def add(self, rec):
        key = rec.get("{other}")
        self.totals[key] = self.totals.get(key, 0) + rec.get("{field}", 0)
        if len(self.totals) > self.limit:
            self.totals.pop(next(iter(self.totals)))

    def top(self, n={k}):
        return sorted(self.totals.items(), key=lambda kv: -kv[1])[:n]
'''


FLAG_BLOCKS = [
    ("register_io_args", [
        ('"--input"', 'required=True, help="Path to the input file or directory."'),
        ('"--output-dir"', 'default="out", help="Where results are written."'),
        ('"--format"', 'choices=["json", "csv", "parquet"], default="json", help="Output format."'),
    ]),
    ("register_runtime_args", [
        ('"--max-retries"', 'type=int, default=3, help="Retries per failed batch."'),
        ('"--timeout"', 'type=float, default=30.0, help="Per-request timeout in seconds."'),
        ('"--workers"', 'type=int, default=4, help="Number of worker processes."'),
    ]),
    ("register_logging_args", [
        ('"--verbose"', 'action="store_true", help="Enable debug logging."'),
        ('"--log-file"', 'default=None, help="Also write logs to this file."'),
        ('"--dry-run"', 'action="store_true", help="Plan the run without writing output."'),
    ]),
    ("register_misc_args", [
        ('"--config"', 'default="pipeline.toml", help="Config file to load."'),
        ('"--no-cache"', 'action="store_true", help="Ignore the on-disk cache."'),
        ('"--since"', 'default=None, help="Only process records newer than this ISO date."'),
    ]),
]


def make_cli_source(target_chars: int, seed: int = 42) -> str:
    """A Python CLI whose 12 flags are scattered through filler code."""
    rng = random.Random(seed)
    used: set = set()
    header = (
        '"""pipeline.py - batch processor for nightly exports."""\n\n'
        "import argparse\nimport logging\n\nlogger = logging.getLogger(__name__)\n\n"
        f"# NOTE: {REMOVED_FLAGS[0]} was removed in v2.0 and must not be reintroduced.\n"
    )
    blocks = []
    for fn, args in FLAG_BLOCKS:
        body = "".join(f"    parser.add_argument({a}, {kw})\n" for a, kw in args)
        blocks.append(f"\n\ndef {fn}(parser):\n{body}")
    footer = (
        "\n\ndef main():\n"
        "    parser = argparse.ArgumentParser(description=__doc__)\n"
        + "".join(f"    {fn}(parser)\n" for fn, _ in FLAG_BLOCKS)
        + f"    # {REMOVED_FLAGS[1]} was dropped; the speedup is now the default.\n"
        "    args = parser.parse_args()\n"
        '    logger.info("starting with %s", vars(args))\n\n\n'
        'if __name__ == "__main__":\n    main()\n'
    )
    budget = target_chars - len(header) - len(footer) - sum(map(len, blocks))
    fillers = []
    while sum(map(len, fillers)) < budget:
        fillers.append(_filler_fn(rng, used))
    # Place the flag blocks at roughly 10%, 35%, 60%, 90% of the filler.
    parts = [header]
    marks = [int(len(fillers) * f) for f in (0.10, 0.35, 0.60, 0.90)]
    for i, f in enumerate(fillers):
        if i in marks:
            parts.append(blocks[marks.index(i)])
        parts.append(f)
    parts.append(footer)
    return "".join(parts)


def score_flags(answer: str) -> dict:
    found = set(re.findall(r"--[a-z][a-z0-9-]*", answer))
    truth = set(TRUE_FLAGS)
    return {
        "score": len(found & truth) / len(truth),
        "detail": f"{len(found & truth)}/{len(truth)} flags",
        "missing": sorted(truth - found),
        "extra": sorted(found - truth),
    }


def make_csv(seed: int = 7, rows: int = 60) -> tuple[str, list[dict]]:
    rng = random.Random(seed)
    names = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fatima", "Gus", "Hana",
             "Ivan", "Jade", "Kofi", "Lena", "Mateo", "Nora", "Omar", "Pia"]
    cities = ["Oslo", "Lima", "Accra", "Hanoi", "Porto", "Quito", "Riga", "Sendai"]
    expected = []
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "name", "city", "qty", "price", "active"])
    for i in range(1, rows + 1):
        row = {
            "id": 1000 + i,
            "name": rng.choice(names),
            "city": rng.choice(cities),
            "qty": rng.randint(0, 250),
            "price": round(rng.uniform(1, 999), 2),
            "active": rng.random() < 0.6,
        }
        expected.append(row)
        w.writerow([row["id"], row["name"], row["city"], row["qty"],
                    f'{row["price"]:.2f}', str(row["active"]).lower()])
    return buf.getvalue(), expected


def _strip_fences(text: str) -> str:
    m = re.search(r"```[\w+-]*\n(.*?)\n?```", text, flags=re.DOTALL)
    return m.group(1) if m else text.strip()


def score_csv(answer: str, expected: list[dict]) -> dict:
    try:
        got = json.loads(_strip_fences(answer))
    except json.JSONDecodeError as e:
        return {"score": 0.0, "detail": f"invalid JSON: {e}"}
    if not isinstance(got, list):
        return {"score": 0.0, "detail": "not a list"}

    def same(a, b):
        if not isinstance(a, dict) or a.keys() != b.keys():
            return False
        for k, v in b.items():
            x = a[k]
            if k == "price":  # 12.0 may legitimately come back as 12
                ok = isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x - v) < 0.005
            else:
                ok = type(x) is type(v) and x == v
            if not ok:
                return False
        return True

    ok = sum(1 for a, b in zip(got, expected) if same(a, b))
    return {"score": ok / len(expected), "detail": f"{ok}/{len(expected)} rows"}


def make_rename_source(seed: int = 3) -> tuple[str, str]:
    rng = random.Random(seed)
    used: set = set()
    parts = ['"""loader.py"""\n\nimport json\nimport logging\n\nlogger = logging.getLogger(__name__)\n',
             '\n\ndef load_records(path):\n    """Read newline-delimited JSON records."""\n'
             '    with open(path) as f:\n        return [json.loads(line) for line in f if line.strip()]\n',
             '\n\ndef load_records_from_cache(key):\n    """Cached variant - must NOT be renamed."""\n'
             '    return _CACHE.get(key) or []\n']
    for i in range(14):
        parts.append(_filler_fn(rng, used))
        if i % 2 == 0:
            parts.append(
                f'\n\ndef pipeline_step_{i}(path):\n'
                f'    rows = load_records(path)\n'
                f'    # load_records returns a list; empty files give []\n'
                f'    return len(rows)\n')
    parts.append('\n_CACHE = {}\n')
    src = "".join(parts)
    expected = re.sub(r"\bload_records\b", "fetch_records", src)
    return src, expected


def score_rename(answer: str, expected: str) -> dict:
    norm = lambda s: "\n".join(l.rstrip() for l in _strip_fences(s).strip().splitlines())
    a, e = norm(answer), norm(expected)
    if a == e:
        return {"score": 1.0, "detail": "exact"}
    ratio = difflib.SequenceMatcher(None, a, e).ratio()
    return {"score": 0.0, "detail": f"mismatch (similarity {ratio:.3f})"}


def build_tasks() -> list[dict]:
    focus = ("Answer this about the content below, citing specifics: list every "
             "command-line flag this program currently accepts, one per line.")
    csv_text, csv_expected = make_csv()
    rename_src, rename_expected = make_rename_source()
    tasks = []
    for label, size in (("flags_40k", 40_000), ("flags_80k", 80_000),
                        ("flags_160k", 160_000)):
        src = make_cli_source(size)
        tasks.append({"name": label, "chars": len(src), "max_tokens": 8192,
                      "prompt": f"{focus}\n\n{src}", "score": score_flags})
    tasks.append({
        "name": "csv_json", "chars": len(csv_text), "max_tokens": 16384,
        "prompt": ("Convert the CSV below to a JSON array of objects, one per row, "
                   "using the header names as keys. id and qty are integers, price "
                   "is a number, active is a boolean. Output only the JSON.\n\n" + csv_text),
        "score": lambda a: score_csv(a, csv_expected),
    })
    tasks.append({
        "name": "rename", "chars": len(rename_src), "max_tokens": 16384,
        "prompt": ("Rewrite the file below, renaming the function load_records to "
                   "fetch_records everywhere it appears (definition, calls, comments). "
                   "Do not change any other identifier - load_records_from_cache keeps "
                   "its name. Output the complete file and nothing else.\n\n" + rename_src),
        "score": lambda a: score_rename(a, rename_expected),
    })
    return tasks

# ------------------------------------------------------------------ runner


def _alive() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/v1/models", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def start_server(model: str, log_path: Path) -> subprocess.Popen:
    if _alive():
        sys.exit(f"Something is already listening on {BENCH_PORT}; stop it first.")
    log = open(log_path, "ab")
    proc = subprocess.Popen(
        # Uncapped, the prompt cache grew to 7.4 GB beside an 8 GB model and
        # pushed a 24 GB Mac into swap (80k prefill slowed 10x, then timed out).
        # Every request is unique here, so the cache can only hurt.
        [str(VENV_BIN / "mlx_lm.server"), "--model", model, "--port", str(BENCH_PORT),
         "--prompt-cache-bytes", str(2 * 1024**3)],
        stdout=log, stderr=log, start_new_session=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        if _alive():
            return proc
        if proc.poll() is not None:
            sys.exit(f"server for {model} exited; see {log_path}")
        time.sleep(2)
    proc.terminate()
    sys.exit(f"server for {model} did not come up; see {log_path}")


def call(cfg: dict, prompt: str, max_tokens: int) -> dict:
    # mlx_lm.server reuses the KV cache for a repeated prompt prefix, which made
    # rep 2 of a 40k-char prompt take 3s instead of 70s. Real delegations rarely
    # resend the same input, so a nonce ahead of the content forces a full
    # prefill of everything but the short system prompt.
    nonce = f"[request {uuid.uuid4().hex[:12]}]"
    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": cfg["system"]},
                     {"role": "user", "content": f"{nonce}\n{prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        **cfg["extra"],
    }
    t0 = time.monotonic()
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=TIMEOUT_S)
    wall = time.monotonic() - t0
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    raw = msg.get("content") or ""
    # Qwen inlines <think>; Gemma 4 uses <|channel>thought ... <channel|>. mlx-lm
    # normally moves both into `reasoning`, but strip any leak so it can't score.
    think_re = r"<think>(.*?)</think>|<\|channel>(.*?)<channel\|>"
    inline = "".join(a + b for a, b in re.findall(think_re, raw, flags=re.DOTALL))
    answer = re.sub(think_re, "", raw, flags=re.DOTALL).strip()
    usage = data.get("usage", {})
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(msg.get("reasoning") or "") + len(inline),
        "finish_reason": choice.get("finish_reason"),
        "answer": answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS), help="configs to run")
    ap.add_argument("--tasks", nargs="*", help="task names to run (default: all)")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUT_DIR / f"run-{stamp}.jsonl"
    tasks = [t for t in build_tasks() if not args.tasks or t["name"] in args.tasks]

    for name in args.only or list(CONFIGS):
        cfg = CONFIGS[name]
        print(f"\n=== {name} ({cfg['model']})", flush=True)
        proc = start_server(cfg["model"], OUT_DIR / f"server-{stamp}.log")
        try:
            call(cfg, "Reply with the single word: ready", 64)  # warm-up / load
            for task in tasks:
                for rep in range(1, args.reps + 1):
                    try:
                        res = call(cfg, task["prompt"], task["max_tokens"])
                        sc = task["score"](res["answer"])
                    except Exception as e:  # keep going; record the failure
                        res, sc = {"wall_s": None, "error": repr(e), "answer": ""}, {
                            "score": 0.0, "detail": "error"}
                    row = {"config": name, "model": cfg["model"], "task": task["name"],
                           "input_chars": task["chars"], "rep": rep, **res, **sc}
                    with open(out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    tps = (res.get("completion_tokens") or 0) / res["wall_s"] if res.get("wall_s") else 0
                    print(f"  {task['name']:<10} rep{rep}  {res.get('wall_s')}s  "
                          f"{res.get('completion_tokens')} tok ({tps:.0f}/s incl. prefill)  "
                          f"reasoning={res.get('reasoning_chars')}ch  "
                          f"finish={res.get('finish_reason')}  -> {sc['detail']}"
                          + (f"  extra={sc['extra']}" if sc.get("extra") else ""),
                          flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            time.sleep(3)
    summarize(out_path)


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    shown = path.relative_to(PROJECT_DIR) if path.is_relative_to(PROJECT_DIR) else path
    print(f"\nResults: {shown}\n")
    print(f"{'config':<22}{'task':<11}{'median s':>9}{'accuracy':>10}{'compl tok':>11}")
    keys = sorted({(r["config"], r["task"]) for r in rows},
                  key=lambda k: (list(CONFIGS).index(k[0]), k[1]))
    for cfg, task in keys:
        rs = [r for r in rows if r["config"] == cfg and r["task"] == task]
        walls = [r["wall_s"] for r in rs if r.get("wall_s")]
        toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        acc = statistics.mean(r["score"] for r in rs)
        print(f"{cfg:<22}{task:<11}"
              f"{statistics.median(walls) if walls else float('nan'):>9.1f}"
              f"{acc:>10.0%}"
              f"{int(statistics.median(toks)) if toks else 0:>11}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--summarize":
        summarize(Path(sys.argv[2]))
    else:
        main()
