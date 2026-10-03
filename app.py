#!/usr/bin/env python3
"""Discovery Market – local results app.

One page that runs the simulation and the tests and shows the results.
Standard library only; listens on localhost.

Run: uv run python app.py   (then open http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bounty
import report

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
COMMANDS = {
    "/api/run": [sys.executable, "run.py"],
    "/api/test": [sys.executable, "-m", "pytest", "-q", "--color=no"],
}
STATIC = {"/index.html": "index.html"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_command(cmd: list[str]) -> dict:
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "seconds": round(time.time() - start, 1),
        "output": ANSI.sub("", proc.stdout + proc.stderr).strip(),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, status: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self) -> None:
        if self.path in ("/", "/report"):
            html = (ROOT / "report_template.html").read_text()
            self._send(html.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            events, ledger, summary = report.load_outputs()
            self._json(report.build_data(events, ledger, summary))
        elif self.path == "/api/bounty/info":
            self._json({"example": bounty.EXAMPLE_REQUEST, "live": bounty.api_key() is not None,
                        "model": bounty.MODEL, "max_bsl": bounty.MAX_BSL})
        elif self.path in STATIC:
            self._send((ROOT / STATIC[self.path]).read_bytes(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/api/bounty":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                prize = float(body.get("prize", 0))
            except (ValueError, TypeError):
                self._json({"ok": False, "error": "Could not read the form."}, 400)
                return
            self._json(bounty.post_bounty(str(body.get("hypothesis", "")), str(body.get("criterion", "")), prize))
        elif self.path in COMMANDS:
            self._json(run_command(COMMANDS[self.path]))
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path}\n")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Discovery Market app running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
