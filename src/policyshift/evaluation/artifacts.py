"""Artifact export for experiment traces and metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyshift.schemas import AgentTrajectory
from policyshift.utils.hashing import sha256_json
from policyshift.utils.io import ensure_dir, write_json, write_jsonl


def new_experiment_id(prefix: str = "exp") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def export_experiment(
    root: str | Path,
    *,
    experiment_id: str,
    config: dict[str, Any],
    trajectories: list[AgentTrajectory],
    per_case_metrics: list[dict[str, Any]],
    summary_metrics: dict[str, Any],
    retrieval_rows: list[dict[str, Any]] | None = None,
    retrieval_summary: dict[str, Any] | None = None,
    failures: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a self-contained experiment artifact directory."""
    base = ensure_dir(Path(root) / experiment_id)
    traces_dir = ensure_dir(base / "traces")
    metrics_dir = ensure_dir(base / "metrics")
    retrieval_dir = ensure_dir(base / "retrieval")
    failures_dir = ensure_dir(base / "failures")

    paths: dict[str, Path] = {}
    paths["config"] = write_json(base / "config.json", config)
    paths["trajectories"] = write_jsonl(traces_dir / "trajectories.jsonl", trajectories)
    paths["per_case_metrics"] = write_json(metrics_dir / "per_case.json", per_case_metrics)
    paths["summary"] = write_json(metrics_dir / "summary.json", summary_metrics)
    if retrieval_rows is not None:
        paths["retrieval_rows"] = write_json(retrieval_dir / "per_case.json", retrieval_rows)
    if retrieval_summary is not None:
        paths["retrieval_summary"] = write_json(retrieval_dir / "summary.json", retrieval_summary)
    if failures is not None:
        paths["failures"] = write_json(failures_dir / "report.json", failures)

    manifest = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_trajectories": len(trajectories),
        "files": {k: str(v.relative_to(base)) for k, v in paths.items()},
        "config_checksum": sha256_json(config),
        "summary_checksum": sha256_json(summary_metrics),
    }
    paths["manifest"] = write_json(base / "manifest.json", manifest)
    return paths
