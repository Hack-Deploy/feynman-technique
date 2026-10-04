#!/usr/bin/env python3
"""Discovery Market – local app for the vision, simulation and live market pages.

Run: uv run python app.py   (then open http://localhost:8000)
"""

from __future__ import annotations

import errno
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import live_market
import real_data

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PORT = int(os.environ.get("PORT", "8000"))
# Extra address to listen on besides loopback, e.g. a Tailscale IP
# (HOST=100.64.50.64). Avoid 0.0.0.0: the app can run simulations and tests.
HOST = os.environ.get("HOST", "127.0.0.1")
COMMANDS = {
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
ANSI = re.compile(r"\x1b\[[0-9;]*m")
_BUSY = threading.Lock()
PAGE_ROUTES = {
    "/": "vision.html",
    "/simulation": "simulation.html",
    "/live": "live.html",
    "/pitch": "pitch.html",
}
STATIC_EXTENSIONS = {".css", ".js", ".svg", ".png", ".json"}


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
        # Host/Origin allowlist (DNS-rebinding guard): the loopback names, plus the
        # extra address the operator chose with HOST (e.g. a Tailscale IP).
        allowed_hosts = {"localhost", "127.0.0.1", "::1", HOST}
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
        if path in PAGE_ROUTES:
            self._send((WEB / PAGE_ROUTES[path]).read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/real":
            try:
                self._json(real_data.build_real_data())
            except Exception as exc:
                self._json({"error": f"could not read real-attempt outputs: {exc}"}, 503)
        elif path == "/api/live/info":
            self._json(live_market.info())
        elif path == "/api/live/runs":
            self._json(live_market.runs())
        elif path == "/api/live/recorded":
            self._json(live_market.recorded())
        elif path == "/api/live/recorded/run":
            attempt_id = urllib.parse.parse_qs(
                urllib.parse.urlsplit(self.path).query
            ).get("id", [""])[0]
            result = live_market.recorded_run(attempt_id)
            if result is None:
                self._json({"error": "not found"}, 404)
            else:
                self._json(result)
        elif path == "/api/live/job":
            job_id = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("id", [""])[0]
            result = live_market.job(job_id)
            if result is None:
                self._json({"error": "not found"}, 404)
            else:
                self._json(result)
        elif path.startswith("/web/"):
            name = urllib.parse.unquote(path.removeprefix("/web/"))
            relative_path = Path(name)
            if (not name or relative_path.is_absolute() or ".." in relative_path.parts
                    or "\\" in name or "\x00" in name
                    or Path(name).suffix.lower() not in STATIC_EXTENSIONS):
                self._json({"error": "not found"}, 404)
                return
            file_path = (WEB / relative_path).resolve()
            if WEB.resolve() not in file_path.parents or not file_path.is_file():
                self._json({"error": "not found"}, 404)
                return
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type += "; charset=utf-8"
            self._send(file_path.read_bytes(), content_type)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if not self._local_request():
            self._json({"error": "forbidden"}, 403)
            return
        if path == "/api/real/attempt":
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
        elif path == "/api/live/start":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                hypothesis_id = str(body["hypothesis_id"])
                model = str(body.get("model", ""))
                scripted = body.get("scripted", False)
                if not isinstance(scripted, bool):
                    raise ValueError("scripted must be a boolean")
            except (ValueError, TypeError, KeyError) as exc:
                self._json({"ok": False, "error": str(exc) or "Could not read the request."}, 400)
                return
            try:
                result = live_market.start(hypothesis_id, model, scripted)
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except PermissionError as exc:
                self._json({"ok": False, "error": str(exc)}, 403)
            except RuntimeError as exc:
                self._json({"ok": False, "error": str(exc)}, 409)
            else:
                self._json(result)
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


def _bind(port: int, tries: int = 20) -> tuple[ThreadingHTTPServer, int]:
    """Bind to ``port``, or the next free port if it is taken (e.g. by another app).
    Returns the server and the port it bound."""
    for p in range(port, port + tries):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Handler), p
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            print(f"Port {p} is in use, trying {p + 1}...")
    raise SystemExit(f"No free port in {port}-{port + tries - 1}; set PORT to choose another.")


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _also_ipv6(port: int) -> None:
    """Serve the same app on IPv6 loopback too: on many Linux systems ``localhost``
    resolves to ::1 first, and a browser would otherwise get 'connection refused'."""
    try:
        v6 = _V6Server(("::1", port), Handler)
    except OSError:
        return  # no IPv6 loopback, or ::1 port taken; 127.0.0.1 still works
    threading.Thread(target=v6.serve_forever, daemon=True).start()


def _also_host(host: str, port: int) -> str | None:
    """Serve on ``host`` too (e.g. a Tailscale IP). Returns its URL, or None."""
    if host in ("127.0.0.1", "localhost", "::1"):
        return None
    try:
        extra = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"Could not listen on {host}:{port} ({e}); serving on this machine only.")
        return None
    threading.Thread(target=extra.serve_forever, daemon=True).start()
    return f"http://{host}:{port}"


def main() -> None:
    server, port = _bind(PORT)
    _also_ipv6(port)
    remote = _also_host(HOST, port)
    url = f"http://127.0.0.1:{port}"
    print(f"Discovery Market app running at {url}  (also http://localhost:{port})  (Ctrl+C to stop)")
    if remote:
        print(f"Also reachable at {remote}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
