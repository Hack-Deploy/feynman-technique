"""Export the app as a static, read-only site for Vercel.

The Python app needs jax and the simulator (far over Vercel's function size limit) and makes
paid model calls, so the hosted copy is static: every page, ``web/`` assets, and the JSON each
read-only API endpoint returns, with ``vercel.json`` rewrites mapping the API paths onto those
files. Starting a new paid run is disabled.

    uv run python scripts/export_static.py              # writes output/discovery-market/
    vercel deploy output/discovery-market --prod
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import live_market  # noqa: E402
import marketplace  # noqa: E402
import real_data  # noqa: E402

OUT = ROOT / "output" / "discovery-market"


def _write(rel: str, obj) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(app.WEB, OUT / "web")

    rewrites = []
    for route, page in app.PAGE_ROUTES.items():
        # Each route is a real file, so no page depends on a rewrite.
        target = OUT / ("index.html" if route == "/" else f"{route.strip('/')}.html")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(app.WEB / page, target)

    info = live_market.info()
    info["live"] = {**info["live"], "enabled": False,
                    "reasons": ["This hosted copy is read-only; new paid runs start only from a local install."]}
    _write("api/live/info.json", info)
    _write("api/live/runs.json", live_market.runs())
    recorded = live_market.recorded()
    _write("api/live/recorded.json", recorded)
    for row in recorded["runs"]:
        _write(f"api/live/recorded/run/{row['attempt_id']}.json",
               live_market.recorded_run(row["attempt_id"]))
    _write("api/real.json", real_data.build_real_data())

    _write("api/market/venues.json", marketplace.venues())
    for venue in marketplace.venues()["venues"]:
        vid = venue["id"]
        _write(f"api/market/venue/{vid}.json", marketplace.venue(vid, None))
        for h in live_market.info()["hypotheses"]:
            try:
                scoped = marketplace.venue(vid, h["id"])
            except ValueError:
                continue
            if scoped is not None:
                _write(f"api/market/venue/{vid}/bounty/{h['id']}.json", scoped)
        rewrites.append({"source": f"/api/market/venue/{vid}",
                         "has": [{"type": "query", "key": "bounty", "value": "(?<b>.+)"}],
                         "destination": f"/api/market/venue/{vid}/bounty/:b.json"})

    rewrites += [
        {"source": "/api/live/recorded/run",
         "has": [{"type": "query", "key": "id", "value": "(?<id>[0-9a-f-]+)"}],
         "destination": "/api/live/recorded/run/:id.json"},
        {"source": "/api/:path*", "destination": "/api/:path*.json"},
    ]
    (OUT / "vercel.json").write_text(json.dumps(
        {"cleanUrls": True, "trailingSlash": False, "rewrites": rewrites}, indent=2))
    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT.relative_to(ROOT)}: {files} files, {len(recorded['runs'])} recorded runs")


if __name__ == "__main__":
    main()
