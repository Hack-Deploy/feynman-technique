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
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import report

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
COMMANDS = {
    "/api/run": [sys.executable, "run.py"],
    "/api/test": [sys.executable, "-m", "pytest", "-q", "--color=no"],
}
STATIC = {"/index.html": "index.html"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BUSY = threading.Lock()


def run_command(cmd: list[str]) -> dict:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "seconds": round(time.time() - start, 1),
            "output": "Timed out after 600 s.",
        }
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

    def _local_request(self) -> bool:
        allowed_hosts = {"localhost", "127.0.0.1"}
        host = self.headers.get("Host")
        if not host:
            return False
        try:
            hostname = urllib.parse.urlsplit(f"//{host}").hostname
        except ValueError:
            return False
        if hostname not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin is not None:
            try:
                origin_host = urllib.parse.urlsplit(origin).hostname
            except ValueError:
                return False
            if origin_host not in allowed_hosts:
                return False
        return True

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self._local_request():
            self._json({"error": "forbidden"}, 403)
            return
        if path in ("/", "/report"):
            html = (ROOT / "report_template.html").read_text()
            self._send(html.encode(), "text/html; charset=utf-8")
        elif path == "/api/data":
            try:
                events, ledger, summary = report.load_outputs()
                data = report.build_data(events, ledger, summary)
            except Exception as exc:
                self._json({"error": f"could not read outputs: {exc}"}, 503)
                return
            self._json(data)
        elif path in STATIC:
            self._send((ROOT / STATIC[path]).read_bytes(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self._local_request():
            self._json({"error": "forbidden"}, 403)
            return
        if path in COMMANDS:
            if not _BUSY.acquire(blocking=False):
                self._json(
                    {
                        "ok": False,
                        "seconds": 0,
                        "output": (
                            "Another command is already running; try again when it finishes."
                        ),
                    },
                    409,
                )
                return
            try:
                self._json(run_command(COMMANDS[path]))
            finally:
                _BUSY.release()
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
