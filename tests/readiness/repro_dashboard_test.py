"""Adversarial regression tests for the simulation dashboard."""

import hashlib
import http.client
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import analysis
import app
import report


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAMES = ("events.json", "ledger.json", "summary.json")


def _copy_simulation(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for source in REPO_ROOT.glob("*.py"):
        shutil.copy2(source, destination / source.name)
    shutil.copytree(REPO_ROOT / "data", destination / "data")
    shutil.copy2(REPO_ROOT / "config.yaml", destination / "config.yaml")
    shutil.copy2(
        REPO_ROOT / "report_template.html", destination / "report_template.html"
    )
    return destination


@pytest.fixture(scope="module")
def sim_copy(tmp_path_factory):
    return _copy_simulation(tmp_path_factory.mktemp("sim-copy"))


@pytest.fixture
def app_server(monkeypatch):
    monkeypatch.setattr(
        app,
        "COMMANDS",
        {
            "/api/run": [sys.executable, "-c", "print('ran')"],
            "/api/test": [sys.executable, "-c", "import sys; sys.exit(3)"],
        },
    )
    server = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request(url, method="GET", data=None, headers=None):
    request = urllib.request.Request(
        url, data=data, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def _output_hashes(output_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in OUTPUT_NAMES
    }


def test_run_outputs_identical_across_hashseeds(sim_copy):
    baseline_path = REPO_ROOT / "tests" / "fixtures" / "baseline_sha256.json"
    baseline = json.loads(baseline_path.read_text())
    expected = {name: baseline[name] for name in OUTPUT_NAMES}
    hashes_by_seed = []

    for seed in ("0", "1"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        subprocess.run(
            [sys.executable, "run.py"],
            cwd=sim_copy,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        hashes_by_seed.append(_output_hashes(sim_copy / "output"))

    assert hashes_by_seed[0] == hashes_by_seed[1] == expected


def test_report_html_deterministic(sim_copy):
    rendered = []
    for _ in range(2):
        subprocess.run(
            [sys.executable, "report.py"],
            cwd=sim_copy,
            check=True,
            capture_output=True,
            text=True,
        )
        rendered.append((sim_copy / "output" / "report.html").read_bytes())
    assert rendered[0] == rendered[1]


def test_concurrent_runs_do_not_corrupt_outputs(tmp_path):
    copied_repo = _copy_simulation(tmp_path / "concurrent-sim-copy")
    processes = []
    for index in range(11):
        processes.append(
            subprocess.Popen(
                [sys.executable, "run.py"],
                cwd=copied_repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        if index < 10:
            time.sleep(0.05)
    results = [process.communicate(timeout=180) for process in processes]
    failures = [
        (process.returncode, stdout, stderr)
        for process, (stdout, stderr) in zip(processes, results)
        if process.returncode != 0
    ]
    assert not failures, f"run.py subprocess failures: {failures}"

    baseline = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "baseline_sha256.json").read_text()
    )
    expected = {name: baseline[name] for name in OUTPUT_NAMES}
    assert _output_hashes(copied_repo / "output") == expected


def test_app_binds_localhost_only(monkeypatch):
    bound = {}

    class FakeServer:
        def __init__(self, address, handler):
            bound["address"] = address
            bound["handler"] = handler

        def serve_forever(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(app, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(sys, "argv", ["app.py", "--no-browser"])

    app.main()

    assert bound["address"][0] == "127.0.0.1"


def test_app_unknown_paths_404(app_server):
    for path in ("/api/nope", "/../STATUS.md", "/api/run"):
        status, headers, body = _request(f"{app_server}{path}")
        assert status == 404
        assert "json" in headers.get("Content-Type", "").lower()
        assert json.loads(body) == {"error": "not found"}


def test_app_command_failure_reported(app_server):
    status, _, body = _request(f"{app_server}/api/test", method="POST", data=b"")
    assert status == 200
    result = json.loads(body)
    assert result["ok"] is False

    status, _, body = _request(f"{app_server}/api/run", method="POST", data=b"")
    assert status == 200
    result = json.loads(body)
    assert result["ok"] is True
    assert "ran" in result["output"]


def test_api_data_with_corrupt_output_returns_json_error(
    app_server, monkeypatch, tmp_path
):
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "events.json").write_text('[{"a":')
    (tmp_path / "ledger.json").write_text("[]")
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)

    try:
        status, headers, body = _request(f"{app_server}/api/data")
    except (ConnectionError, OSError, urllib.error.URLError, http.client.HTTPException) as error:
        pytest.fail(f"expected an HTTP error response, connection failed: {error}")

    assert status >= 500
    assert "json" in headers.get("Content-Type", "").lower()
    json.loads(body)


def test_app_rejects_cross_origin_post(app_server):
    status, _, _ = _request(
        f"{app_server}/api/run",
        method="POST",
        data=b"",
        headers={"Origin": "http://evil.example"},
    )
    assert status == 403


def test_app_root_with_query_string(app_server):
    status, _, _ = _request(f"{app_server}/?x=1")
    assert status == 200


def test_report_embed_escapes_script_close(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "load_outputs", lambda: ([], [], {}))
    monkeypatch.setattr(
        report,
        "build_data",
        lambda *args: {"law": "</script><script>window.pwned=1</script>"},
    )
    monkeypatch.setattr(report, "OUTPUT_DIR", tmp_path)

    report.main()

    template = (REPO_ROOT / "report_template.html").read_text()
    rendered = (tmp_path / "report.html").read_text()
    assert rendered.count("</script>") == template.count("</script>")


@pytest.mark.xfail(
    strict=True,
    reason="READINESS: dashboard must load no external hosts except cdnjs",
)
def test_report_has_no_external_hosts_except_cdnjs():
    template = (REPO_ROOT / "report_template.html").read_text()
    urls = re.findall(
        r"""(?:src|href)=["'](https?://[^"']+)""", template, flags=re.IGNORECASE
    )
    urls.extend(
        re.findall(
            r"""@import\s+(?:url\()?["']?(https?://[^"')\s;]+)""",
            template,
            flags=re.IGNORECASE,
        )
    )
    urls.extend(
        re.findall(
            r"""url\(\s*["']?(https?://[^"')\s]+)""",
            template,
            flags=re.IGNORECASE,
        )
    )

    assert all(urlsplit(url).hostname == "cdnjs.cloudflare.com" for url in urls)


@pytest.mark.xfail(
    strict=True,
    reason="READINESS: dashboard balances must retain unknown solver names",
)
def test_report_balance_series_keeps_unknown_solvers():
    events = [
        {
            "run_id": "track_a_replay_seed0",
            "tick": 0,
            "seq": 0,
            "type": "account_funded",
            "from": "external",
            "to": "agent:fable",
            "amount": 100,
        },
        {
            "run_id": "track_a_replay_seed0",
            "tick": 1,
            "seq": 1,
            "type": "round_charged",
            "from": "agent:fable",
            "to": "lab",
            "amount": 10,
        },
    ]

    assert "fable" in report.balance_series(events)["track_a_replay_seed0"]["points"]


def test_report_data_carries_provenance(sim_copy, monkeypatch):
    subprocess.run(
        [sys.executable, "run.py"],
        cwd=sim_copy,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setattr(report, "OUTPUT_DIR", sim_copy / "output")

    data = report.build_data(*report.load_outputs())

    assert "provenance" in data


def test_dashboard_clearing_caption_matches_analysis():
    template = (REPO_ROOT / "report_template.html").read_text()
    assert "at least 3 of 5 seeds" in template
    assert "in at least one seed" not in template


def test_clearing_rule_is_3_of_5():
    source = inspect.getsource(analysis.compute_clearing_prizes)
    assert ">= 3" in source


def test_readme_commands_exist():
    readme = (REPO_ROOT / "README.md").read_text()
    inside_code_block = False
    scripts = []
    for line in readme.splitlines():
        if line.startswith("```"):
            inside_code_block = not inside_code_block
            continue
        if inside_code_block:
            match = re.match(r"\s*uv run python ([^\s]+\.py)(?:\s|$)", line)
            if match:
                scripts.append(match.group(1))

    assert scripts
    assert all((REPO_ROOT / script).is_file() for script in scripts)


def test_pitch_pages_present():
    for filename in ("index.html", "slides.html"):
        path = REPO_ROOT / filename
        assert path.is_file()
        assert path.stat().st_size > 0

    slides_pdf = REPO_ROOT / "slides.pdf"
    assert slides_pdf.is_file()
    assert slides_pdf.read_bytes().startswith(b"%PDF")


def test_app_second_command_while_busy_gets_409(app_server, monkeypatch):
    monkeypatch.setitem(
        app.COMMANDS,
        "/api/run",
        [sys.executable, "-c", "import time; time.sleep(1.5)"],
    )
    first_response = []

    def send_first_request():
        first_response.append(
            _request(f"{app_server}/api/run", method="POST", data=b"")
        )

    thread = threading.Thread(target=send_first_request)
    thread.start()
    time.sleep(0.3)

    status, _, body = _request(
        f"{app_server}/api/run", method="POST", data=b""
    )
    assert status == 409
    assert json.loads(body)["ok"] is False

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert first_response[0][0] == 200


def test_app_rejects_foreign_host_header(app_server):
    status, _, body = _request(
        f"{app_server}/api/data", headers={"Host": "evil.example"}
    )

    assert status == 403
    assert json.loads(body) == {"error": "forbidden"}


def test_replay_ledger_records_attempt_source():
    from tests.test_phase1_engine import _replay_run

    _, _, ledger = _replay_run(0)
    assert ledger
    assert all(
        row["context"]["attempt_source"] == "sim"
        and row["context"]["protocol"] == "fixture"
        for row in ledger
    )


def test_bounty_routes_reject_foreign_origin_and_host(app_server, monkeypatch):
    calls = []

    def fake_post_bounty(*args):
        calls.append(args)
        return {}

    monkeypatch.setattr(app.bounty, "post_bounty", fake_post_bounty)

    status, _, body = _request(
        f"{app_server}/api/bounty",
        method="POST",
        data=json.dumps({"hypothesis": "x", "criterion": "y", "prize": 1}).encode(),
        headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
    )
    assert status == 403
    assert json.loads(body) == {"error": "forbidden"}

    status, _, body = _request(
        f"{app_server}/api/bounty/info",
        headers={"Host": "evil.example"},
    )
    assert status == 403
    assert json.loads(body) == {"error": "forbidden"}
    assert calls == []
