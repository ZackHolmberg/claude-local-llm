"""Local usage dashboard for the delegation server. Run via ./llm dashboard.

Serves dashboard.html plus a JSON endpoint built from usage.jsonl, bound to
127.0.0.1 only: the ledger holds local file paths, so it never leaves the
machine. The page polls the endpoint, so it stays live while calls land.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import server

PAGE = Path(__file__).resolve().parent / "dashboard.html"
DEFAULT_PORT = 8740


def _entries() -> list[dict]:
    if not server.USAGE_LOG.exists():
        return []
    entries = []
    for line in server.USAGE_LOG.read_text().splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written line from a call in progress
    return entries


def _status() -> dict:
    status = {
        "configured_model": server.MODEL,
        "port": server.PORT,
        "running": server._server_alive(),
        "active": None,
    }
    try:
        status["active"] = json.loads(server.ACTIVE_MARKER.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return status


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path  # ignore ?theme= and other query params
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif path == "/api/data":
            body = json.dumps({
                "entries": _entries(),
                "status": _status(),
                # USD per million tokens (input, output), same table as the report
                "pricing": server.CLAUDE_PRICING,
            }).encode()
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass  # the page polls every few seconds; don't spam the terminal


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Dashboard at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
