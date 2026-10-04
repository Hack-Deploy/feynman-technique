from __future__ import annotations

import json
from pathlib import Path

import pytest

from poc import config as C, hf_data, live_cache


class FakeHfApi:
    def __init__(self):
        self.created = []
        self.uploads = []

    def create_repo(self, repo, **kwargs):
        self.created.append((repo, kwargs))

    def upload_folder(self, **kwargs):
        folder = Path(kwargs["folder_path"])
        self.uploads.append({
            **kwargs,
            "files": {
                path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*")
                if path.is_file()
            },
        })


def _seed_upload_tree(root: Path) -> tuple[str, Path]:
    cfg = C.load()
    claim = cfg.hypotheses[0].id
    runs = root / "attempts" / "fixtures" / "live" / "runs.jsonl"
    runs.parent.mkdir(parents=True, exist_ok=True)
    runs.write_text(json.dumps({
        "record": {"attempt_id": "real-attempt"},
        "key": {"hypothesis_id": claim, "model": "claude", "seed": 0},
        "source": "real",
        "usd": 1.234567,
        "order_seed": 77,
    }) + "\n")
    (runs.parent / "runs.summary.json").write_text("{}")
    store = root / "attempts" / "poc_dp_bench.jsonl"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{}\n")
    transcript = root / "attempts" / "transcripts" / "poc_dp" / "one.json"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"transcript": true}')
    trajectory = root / "attempts" / "poc_trajectories" / "one.csv"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text("x,y\n0,1\n")
    return claim, runs


def test_push_uses_only_allowlisted_files_and_private_dataset_by_default(tmp_path):
    claim, runs = _seed_upload_tree(tmp_path)
    excluded = (
        "poc/.env",
        "attempts/live_spend.jsonl",
        "attempts/live_spend.jsonl.lock",
        "attempts/archive/old/runs.jsonl",
        "attempts/cache/ara.jsonl",
        "other/secret.json",
    )
    for relative in excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not upload")
    api = FakeHfApi()

    uploaded = hf_data.push("org/data", root=tmp_path, api=api)

    assert set(uploaded) == {
        "attempts/fixtures/live/runs.jsonl",
        "attempts/fixtures/live/runs.summary.json",
        "attempts/poc_dp_bench.jsonl",
        "attempts/transcripts/poc_dp/one.json",
        "attempts/poc_trajectories/one.csv",
    }
    assert len(api.created) == 1
    assert api.created[0] == (
        "org/data",
        {"repo_type": "dataset", "private": True, "exist_ok": True},
    )
    assert len(api.uploads) == 1
    uploaded_files = api.uploads[0]["files"]
    assert set(uploaded_files) == set(uploaded) | {"README.md"}
    assert all(name not in uploaded_files for name in excluded)
    card = uploaded_files["README.md"].decode()
    assert f"`{claim}`: 1 run(s)" in card
    assert "`order_seed`: 77" in card
    assert "$1.234567" in card
    assert "`max_rounds`:" in card
    assert "max_tokens" in card
    assert runs.exists()


@pytest.mark.parametrize(
    ("secret", "environment"),
    (
        ("sk-ant-Abcdefghijklmnop", None),
        ("anthropic-test-secret", "ANTHROPIC_API_KEY"),
        ("hf-test-secret", "HF_TOKEN"),
    ),
)
def test_push_aborts_on_secret_without_disclosing_match(
    tmp_path, monkeypatch, secret, environment
):
    _claim, runs = _seed_upload_tree(tmp_path)
    if environment:
        monkeypatch.setenv(environment, secret)
    runs.write_text(runs.read_text() + secret)
    api = FakeHfApi()

    with pytest.raises(ValueError, match="secret detected") as exc_info:
        hf_data.push("org/data", root=tmp_path, api=api)

    assert "runs.jsonl" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert api.created == []
    assert api.uploads == []


def test_dry_run_lists_sizes_without_api_calls(tmp_path):
    _seed_upload_tree(tmp_path)
    api = FakeHfApi()
    output = []

    hf_data.push("org/data", root=tmp_path, api=api, dry_run=True, out=output.append)

    assert output
    assert all("(" in line and " bytes)" in line for line in output)
    assert any(line.startswith("README.md") for line in output)
    assert api.created == []
    assert api.uploads == []


def test_pull_archives_existing_files_then_copies_allowed_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = tmp_path
    local_runs = root / "attempts" / "fixtures" / "live" / "runs.jsonl"
    local_runs.parent.mkdir(parents=True)
    local_runs.write_text("old runs")
    local_summary = live_cache.summary_path(local_runs)
    local_summary.write_text("old summary")
    local_store = root / "attempts" / "poc_dp_bench.jsonl"
    local_store.write_text("old store")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        snapshot = Path(kwargs["local_dir"])
        runs = snapshot / "attempts" / "fixtures" / "live" / "runs.jsonl"
        runs.parent.mkdir(parents=True, exist_ok=True)
        runs.write_text("new runs")
        live_cache.summary_path(runs).write_text("new summary")
        excluded = snapshot / "attempts" / "live_spend.jsonl"
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_text("do not copy")
        return snapshot

    pulled = hf_data.pull(
        "org/data",
        root=root,
        revision="stable",
        download=download,
        out=lambda _line: None,
    )

    assert calls[0]["repo_id"] == "org/data"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "stable"
    assert calls[0]["allow_patterns"] == list(hf_data.ALLOW_PATTERNS)
    assert "token" not in calls[0]
    assert set(pulled) == {
        "attempts/fixtures/live/runs.jsonl",
        "attempts/fixtures/live/runs.summary.json",
    }
    assert local_runs.read_text() == "new runs"
    assert local_summary.read_text() == "new summary"
    assert not local_store.exists()
    archives = list((root / "attempts" / "archive").glob("*/attempts/fixtures/live/*"))
    assert {path.name for path in archives} == {"runs.jsonl", "runs.summary.json"}
    assert {path.read_text() for path in archives} == {"old runs", "old summary"}
    archived_store = list(
        (root / "attempts" / "archive").glob("*/attempts/poc_dp_bench.jsonl")
    )
    assert [path.read_text() for path in archived_store] == ["old store"]
    assert not (root / "attempts" / "live_spend.jsonl").exists()


def test_repo_resolution_prefers_argument_then_environment_then_default(monkeypatch):
    monkeypatch.setenv("DM_HF_REPO", "org/environment")
    assert hf_data._repo_id("org/argument") == "org/argument"
    assert hf_data._repo_id(None) == "org/environment"

    monkeypatch.delenv("DM_HF_REPO")
    assert hf_data._repo_id(None) == hf_data.DEFAULT_HF_REPO
