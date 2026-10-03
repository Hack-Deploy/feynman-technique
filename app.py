#!/usr/bin/env python3
"""Discovery Market – local results app.

One page that runs the simulation and the tests and shows the results.
Standard library only; listens on localhost.

Run: uv run python app.py   (then open http://localhost:8000)
"""

from __future__ import annotations

import errno
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

import bounty
import real_data
import report

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
COMMANDS = {
    "/api/run": [sys.executable, "run.py"],
    "/api/test": [sys.executable, "-m", "pytest", "-q", "--color=no"],
    "/api/real/grid": [sys.executable, "-m", "tests.forcebench_settle"],
}
# Commands run in order; the first failure stops the chain.
CHAINS = {
    "/api/real/replay": [
        [sys.executable, "-m", "dm.importers.ara"],
        [sys.executable, "-m", "dm.replay", "ara"],
    ],
}
TIMEOUTS = {"/api/real/grid": 1800}
STATIC = {"/index.html": "index.html"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BUSY = threading.Lock()


def run_command(cmd: list[str], timeout: int = 600) -> dict:
    start = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "seconds": round(time.time() - start, 1),
            "output": f"Timed out after {timeout} s.",
        }
    return {
        "ok": proc.returncode == 0,
        "seconds": round(time.time() - start, 1),
        "output": ANSI.sub("", proc.stdout + proc.stderr).strip(),
    }


def run_chain(cmds: list[list[str]]) -> dict:
    results = []
    for cmd in cmds:
        results.append(run_command(cmd))
        if not results[-1]["ok"]:
            break
    return {
        "ok": all(r["ok"] for r in results) and len(results) == len(cmds),
        "seconds": round(sum(r["seconds"] for r in results), 1),
        "output": "\n\n".join(r["output"] for r in results),
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
        elif path == "/real":
            self._send((ROOT / "real.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/real":
            try:
                self._json(real_data.build_real_data())
            except Exception as exc:
                self._json({"error": f"could not read real-attempt outputs: {exc}"}, 503)
        elif path == "/api/data":
            try:
                events, ledger, summary = report.load_outputs()
                data = report.build_data(events, ledger, summary)
            except Exception as exc:
                self._json({"error": f"could not read outputs: {exc}"}, 503)
                return
            self._json(data)
        elif path == "/api/bounty/info":
            self._json({"example": bounty.EXAMPLE_REQUEST, "live": bounty.api_key() is not None,
                        "model": bounty.MODEL, "max_bsl": bounty.MAX_BSL})
        elif path in STATIC:
            self._send((ROOT / STATIC[path]).read_bytes(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self._local_request():
            self._json({"error": "forbidden"}, 403)
            return
        if path == "/api/bounty":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                prize = float(body.get("prize", 0))
            except (ValueError, TypeError):
                self._json({"ok": False, "error": "Could not read the form."}, 400)
                return
            self._json(bounty.post_bounty(str(body.get("hypothesis", "")), str(body.get("criterion", "")), prize))
        elif path == "/api/real/attempt":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                world, solver, seed = str(body["world"]), str(body["solver"]), int(body["seed"])
            except (ValueError, TypeError, KeyError):
                self._json({"ok": False, "error": "Could not read the request."}, 400)
                return
            if not _BUSY.acquire(blocking=False):
                self._json({"ok": False, "error": "Another run is in progress; try again when it finishes."}, 409)
                return
            try:
                self._json(real_data.run_forcebench_attempt(world, solver, seed))
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:
                self._json({"ok": False, "error": f"The attempt failed: {exc}"}, 500)
            finally:
                _BUSY.release()
        elif path in COMMANDS or path in CHAINS:
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
                if path in CHAINS:
                    self._json(run_chain(CHAINS[path]))
                else:
                    self._json(run_command(COMMANDS[path], TIMEOUTS.get(path, 600)))
            finally:
                _BUSY.release()
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path}\n")


def _bind(port: int, tries: int = 20) -> ThreadingHTTPServer:
    """Bind to ``port``, or the next free port if it is taken (e.g. by another app)."""
    for p in range(port, port + tries):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Handler)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            print(f"Port {p} is in use, trying {p + 1}...")
    raise SystemExit(f"No free port in {port}-{port + tries - 1}; set PORT to choose another.")


def main() -> None:
    server = _bind(PORT)
    port = server.server_address[1]
    url = f"http://localhost:{port}"
    print(f"Discovery Market app running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
