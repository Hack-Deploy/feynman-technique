"""Archive append-only live-market data without deleting it."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def archive_paths(
    paths: list[Path],
    root: Path,
    out=print,
    protected_paths: list[Path] | None = None,
) -> list[tuple[Path, Path]]:
    root = Path(root)
    root_resolved = root.resolve()
    protected = {Path(path).resolve() for path in (protected_paths or [])}
    blocked = [
        (root / "attempts" / "archive").resolve(),
        (root / "attempts" / "cache").resolve(),
    ]
    scripted_dir = (root / "attempts" / "fixtures" / "live").resolve()
    sources = []
    seen = set()
    for path in paths:
        source = Path(path)
        resolved = source.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (
            (resolved.parent == scripted_dir and source.name.startswith("scripted_demo"))
            or resolved in protected
            or any(resolved == blocked_root or blocked_root in resolved.parents for blocked_root in blocked)
            or not source.exists()
        ):
            continue
        sources.append(source)
    if not sources:
        return []

    archive_dir = (
        root
        / "attempts"
        / "archive"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    if archive_dir.exists():
        raise FileExistsError(f"archive destination already exists: {archive_dir}")

    archived = []
    for source in sources:
        try:
            relative = source.resolve().relative_to(root_resolved)
        except ValueError:
            relative = Path("external") / source.name
        destination = archive_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        archived.append((source, destination))
        out(f"Archived {source} -> {destination}")
    return archived
