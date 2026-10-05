"""Operations CLI for the local-llm delegation server. Run via ./llm <command>."""

import argparse
import json
import subprocess
import sys
import time

import httpx

import server

HF_HUB = server.Path.home() / ".cache" / "huggingface" / "hub"
STOP_PATTERN = f"mlx_lm.server.*--port {server.PORT}"


def cmd_status(_args) -> None:
    print(f"Configured model: {server.MODEL}")
    print(f"Port:             {server.PORT}")
    if server._server_alive():
        served = httpx.get(f"{server.BASE_URL}/v1/models", timeout=3).json()
        ids = ", ".join(m["id"] for m in served.get("data", []))
        print(f"Server:           running (serving {ids})")
    else:
        print("Server:           not running (auto-starts on first tool call)")
    if server.USAGE_LOG.exists():
        calls = sum(1 for _ in server.USAGE_LOG.open())
        print(f"Ledger:           {calls} delegated calls logged ({server.USAGE_LOG})")
    else:
        print("Ledger:           no calls logged yet")


def cmd_start(_args) -> None:
    if server._server_alive():
        print("Already running.")
        return
    print(f"Starting mlx_lm.server with {server.MODEL} (first run downloads the model)...")
    server._ensure_server()
    print(f"Ready on {server.BASE_URL}")


def cmd_stop(_args) -> None:
    result = subprocess.run(["pkill", "-f", STOP_PATTERN])
    print("Stopped." if result.returncode == 0 else "Not running.")


def cmd_restart(args) -> None:
    cmd_stop(args)
    time.sleep(2)
    cmd_start(args)


def cmd_pull(args) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.model}...")
    path = snapshot_download(args.model)
    print(f"Cached at {path}")


def cmd_models(_args) -> None:
    if not HF_HUB.exists():
        print("No HuggingFace cache found.")
        return
    for entry in sorted(HF_HUB.glob("models--*")):
        name = entry.name.removeprefix("models--").replace("--", "/")
        size = subprocess.run(
            ["du", "-sh", str(entry)], capture_output=True, text=True
        ).stdout.split("\t")[0]
        marker = "  * " if name == server.MODEL else "    "
        print(f"{marker}{name}  ({size})")
    print("\n  * = configured model. Switch with: llm use <model>")


def cmd_use(args) -> None:
    config = server._config()
    config["model"] = args.model
    server.CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Configured model set to {args.model}")
    if server._server_alive():
        print("Restarting server on the new model...")
        subprocess.run(["pkill", "-f", f"mlx_lm.server.*--port {server.PORT}"])
        time.sleep(2)
        # Re-derive module-level MODEL for this process before starting
        server.MODEL = args.model
        server._ensure_server()
        print("Done.")
    else:
        print("Server not running; the new model loads on next start.")


def cmd_ask(args) -> None:
    text, usage = server._generate(args.prompt, max_tokens=args.max_tokens)
    print(text)
    print(
        f"\n[{usage.get('prompt_tokens', '?')} prompt + "
        f"{usage.get('completion_tokens', '?')} completion tokens]",
        file=sys.stderr,
    )


def cmd_report(_args) -> None:
    print(server._build_report())


def cmd_dashboard(args) -> None:
    import dashboard

    dashboard.serve(args.port, open_browser=not args.no_open)


def cmd_logs(args) -> None:
    if not server.SERVER_LOG.exists():
        print("No log file yet.")
        return
    if args.follow:
        subprocess.run(["tail", "-f", str(server.SERVER_LOG)])
    else:
        subprocess.run(["tail", "-n", "40", str(server.SERVER_LOG)])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm", description="Manage the local-llm delegation server."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show model, server, and ledger status").set_defaults(fn=cmd_status)
    sub.add_parser("start", help="Start the mlx model server").set_defaults(fn=cmd_start)
    sub.add_parser("stop", help="Stop the mlx model server (frees RAM)").set_defaults(fn=cmd_stop)
    sub.add_parser("restart", help="Restart the mlx model server").set_defaults(fn=cmd_restart)

    p = sub.add_parser("pull", help="Download a model from HuggingFace")
    p.add_argument("model", help="e.g. mlx-community/Qwen3-8B-4bit")
    p.set_defaults(fn=cmd_pull)

    sub.add_parser("models", help="List locally cached models").set_defaults(fn=cmd_models)

    p = sub.add_parser("use", help="Switch the configured model")
    p.add_argument("model")
    p.set_defaults(fn=cmd_use)

    p = sub.add_parser("ask", help="One-off prompt against the local model")
    p.add_argument("prompt")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.set_defaults(fn=cmd_ask)

    sub.add_parser("report", help="Show the token/cost savings report").set_defaults(fn=cmd_report)

    p = sub.add_parser("dashboard", help="Open the live usage dashboard in a browser")
    p.add_argument("--port", type=int, default=8740)
    p.add_argument("--no-open", action="store_true", help="Serve without opening a browser")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("logs", help="Show mlx server logs")
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(fn=cmd_logs)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
