"""Share or warm-start the live-market record with a Hugging Face dataset."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

from poc import archive, bench, config as C, live_cache, spend

ALLOWLIST_FILES = (
    "attempts/fixtures/live/runs.jsonl",
    "attempts/fixtures/live/runs.summary.json",
    "attempts/poc_dp_bench.jsonl",
)
ALLOWLIST_DIRS = (
    "attempts/transcripts/poc_dp",
    "attempts/poc_trajectories",
)
ALLOW_PATTERNS = (
    *ALLOWLIST_FILES,
    "attempts/transcripts/poc_dp/**",
    "attempts/poc_trajectories/**",
)
_SECRET_PATTERN = re.compile(rb"sk-ant-[A-Za-z0-9_-]{10,}")


def _repo_id(repo: str | None) -> str:
    resolved = (repo or os.environ.get("DM_HF_REPO", "")).strip()
    if not resolved:
        raise ValueError("set --repo ORG/NAME or DM_HF_REPO")
    return resolved


def _is_allowlisted(relative: str) -> bool:
    if relative in ALLOWLIST_FILES:
        return True
    return any(relative.startswith(f"{directory}/") for directory in ALLOWLIST_DIRS)


def _files_to_upload(root: Path) -> list[tuple[str, Path]]:
    root = Path(root)
    files: dict[str, Path] = {}
    for relative in ALLOWLIST_FILES:
        path = root / relative
        if path.is_file():
            files[relative] = path
    for directory in ALLOWLIST_DIRS:
        base = root / directory
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    if _is_allowlisted(relative):
                        files[relative] = path
    root_resolved = root.resolve()
    for relative, path in files.items():
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            raise ValueError(f"allow-listed path escapes repository root: {relative}") from None
    return sorted(files.items())


def _secret_values() -> list[bytes]:
    return [
        value.encode("utf-8")
        for name in ("ANTHROPIC_API_KEY", "HF_TOKEN")
        if (value := os.environ.get(name))
    ]


def _check_secret_content(contents: bytes, relative: str, secrets: list[bytes]) -> None:
    if _SECRET_PATTERN.search(contents) or any(secret in contents for secret in secrets):
        raise ValueError(f"secret detected in allow-listed file: {relative}")


def _scan_secrets(files: list[tuple[str, Path]], secrets: list[bytes]) -> None:
    for relative, path in files:
        _check_secret_content(path.read_bytes(), relative, secrets)


def _dataset_card(repo: str, files: list[tuple[str, Path]], root: Path) -> str:
    settings = spend.load_settings()
    cfg = C.load()
    cache_path = Path(root) / ALLOWLIST_FILES[0]
    entries = live_cache.load(cache_path)
    seeds = sorted({
        entry["order_seed"]
        for entry in entries
        if type(entry.get("order_seed")) is int
    })
    order_seed = ", ".join(map(str, seeds)) if seeds else "not recorded"
    run_counts = {hyp.id: 0 for hyp in cfg.hypotheses}
    total_real_usd = 0.0
    for entry in entries:
        claim = entry.get("key", {}).get("hypothesis_id")
        if claim in run_counts:
            run_counts[claim] += 1
        if entry.get("source") == "real":
            total_real_usd += float(entry.get("usd", 0.0) or 0.0)

    model_rows = "\n".join(
        f"- `{model.id}` — {model.label}" for model in settings.models
    )
    claim_rows = "\n".join(
        f"- `{claim}`: {count} run(s)" for claim, count in run_counts.items()
    )
    content_rows = "\n".join(f"- `{relative}`" for relative, _ in files) or "- No run files found"
    return f"""---
license: other
---

# Discovery Market live data

Append-only run records and supporting live-market files for `{repo}`.

## Contents

{content_rows}

## Models and run settings

{model_rows}

- `max_rounds`: {settings.max_rounds}
- `max_tokens`: {settings.max_tokens}

## Runs by claim

{claim_rows}

- `order_seed`: {order_seed}
- Total real API spend recorded in `runs.jsonl`: ${total_real_usd:.6f}

## Pull the data

```bash
uv run python -m poc.hf_data pull --repo {repo}
```
"""


def push(
    repo: str | None = None,
    *,
    root: Path | None = None,
    public: bool = False,
    dry_run: bool = False,
    message: str = "Discovery Market live data",
    api=None,
    out=print,
) -> list[str]:
    repo = _repo_id(repo)
    root = Path(root or C.ROOT)
    files = _files_to_upload(root)
    secrets = _secret_values()
    _scan_secrets(files, secrets)
    card = _dataset_card(repo, files, root)
    _check_secret_content(card.encode("utf-8"), "README.md", secrets)
    if dry_run:
        for relative, path in files:
            out(f"{relative} ({path.stat().st_size} bytes)")
        out(f"README.md ({len(card.encode('utf-8'))} bytes)")
        return [relative for relative, _ in files]

    if api is None:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to push a dataset")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="dm-hf-push-") as staging:
        stage = Path(staging)
        for relative, source in files:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (stage / "README.md").write_text(card, encoding="utf-8")
        api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(stage),
            commit_message=message,
        )
    return [relative for relative, _ in files]


def pull(
    repo: str | None = None,
    *,
    root: Path | None = None,
    revision: str | None = None,
    download=None,
    out=print,
) -> list[str]:
    repo = _repo_id(repo)
    root = Path(root or C.ROOT)
    token = os.environ.get("HF_TOKEN", "").strip()
    if download is None:
        if not token:
            raise ValueError("HF_TOKEN must be set to pull a dataset")
        from huggingface_hub import snapshot_download

        download = snapshot_download

    with tempfile.TemporaryDirectory(prefix="dm-hf-pull-") as staging:
        snapshot = download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision or None,
            allow_patterns=list(ALLOW_PATTERNS),
            local_dir=staging,
            **({"token": token} if token else {}),
        )
        snapshot_root = Path(snapshot or staging)
        downloads = []
        for path in snapshot_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(snapshot_root).as_posix()
            if _is_allowlisted(relative):
                downloads.append((relative, path))
        downloads.sort()
        conflicts = [
            Path(root) / relative
            for relative, _ in downloads
            if (Path(root) / relative).exists()
        ]
        protected = [
            Path(spend.LEDGER_PATH),
            Path(f"{spend.LEDGER_PATH}.lock"),
        ]
        archive.archive_paths(conflicts, root, out=out, protected_paths=protected)
        for relative, source in downloads:
            target = Path(root) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            out(f"Pulled {relative}")
    return [relative for relative, _ in downloads]


def main(argv: list[str] | None = None) -> None:
    bench.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    push_parser = commands.add_parser("push", help="upload live-market records")
    push_parser.add_argument("--repo")
    push_parser.add_argument("--public", action="store_true")
    push_parser.add_argument("--dry-run", action="store_true")
    push_parser.add_argument("--message", default="Discovery Market live data")
    pull_parser = commands.add_parser("pull", help="download live-market records")
    pull_parser.add_argument("--repo")
    pull_parser.add_argument("--revision")
    args = parser.parse_args(argv)
    try:
        repo = _repo_id(args.repo)
        if args.command == "push":
            push(
                repo,
                public=args.public,
                dry_run=args.dry_run,
                message=args.message,
            )
        else:
            pull(repo, revision=args.revision)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
