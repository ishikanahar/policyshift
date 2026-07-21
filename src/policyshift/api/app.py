"""FastAPI artifact playback server (Phase 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "experiments"
EXAMPLE_EXPORT = REPO_ROOT / "artifacts" / "example" / "web_export"
WEB_ROOT = REPO_ROOT / "apps" / "web"
PORTFOLIO_ROOT = REPO_ROOT / "portfolio_export"

app = FastAPI(
    title="PolicyShift Artifact Playback",
    description="Serve experiment artifacts for website/resume demos. Playback first; no live inference required.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(404, f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "policyshift-playback"}


@app.get("/api/experiments")
def list_experiments() -> dict[str, Any]:
    if not ARTIFACT_ROOT.exists():
        return {"experiments": []}
    exps = []
    for path in sorted(ARTIFACT_ROOT.iterdir()):
        if not path.is_dir():
            continue
        manifest = path / "manifest.json"
        summary = path / "metrics" / "summary.json"
        item: dict[str, Any] = {"id": path.name}
        if manifest.exists():
            item["manifest"] = _read_json(manifest)
        if summary.exists():
            item["has_summary"] = True
        exps.append(item)
    return {"experiments": exps}


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    base = ARTIFACT_ROOT / experiment_id
    if not base.exists():
        raise HTTPException(404, "Experiment not found")
    payload: dict[str, Any] = {"id": experiment_id}
    for name in ("manifest.json", "config.json", "phase5_summary.json", "phase6_summary.json", "phase7_summary.json", "phase3_summary.json", "phase4_summary.json"):
        path = base / name
        if path.exists():
            payload[name.replace(".json", "")] = _read_json(path)
    summary = base / "metrics" / "summary.json"
    if summary.exists():
        payload["summary"] = _read_json(summary)
    failures = base / "failures" / "report.json"
    if failures.exists():
        payload["failures"] = _read_json(failures)
    explorer = base / "preferences" / "preference_explorer.json"
    if explorer.exists():
        payload["preference_explorer"] = _read_json(explorer)
    return payload


@app.get("/api/experiments/{experiment_id}/trajectories")
def get_trajectories(experiment_id: str, limit: int = 20) -> dict[str, Any]:
    path = ARTIFACT_ROOT / experiment_id / "traces" / "trajectories.jsonl"
    if not path.exists():
        raise HTTPException(404, "No trajectories")
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return {"experiment_id": experiment_id, "n": len(rows), "trajectories": rows}


@app.get("/api/portfolio")
def portfolio() -> dict[str, Any]:
    path = PORTFOLIO_ROOT / "website_card.json"
    if path.exists():
        return _read_json(path)
    # Fallback to example export
    if (EXAMPLE_EXPORT / "results-summary.json").exists():
        return {
            "title": "PolicyShift",
            "subtitle": "Artifact playback",
            "example": _read_json(EXAMPLE_EXPORT / "results-summary.json"),
        }
    raise HTTPException(404, "Run scripts/export_portfolio.py first")


@app.get("/api/example")
def example_export() -> dict[str, Any]:
    if not EXAMPLE_EXPORT.exists():
        raise HTTPException(404, "No example export")
    out = {}
    for path in EXAMPLE_EXPORT.glob("*.json"):
        out[path.stem] = _read_json(path)
    return out


# Static website
if WEB_ROOT.exists():
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "policyshift.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
