#!/usr/bin/env python3
"""Export web-ready JSON from a Phase 2 experiment artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyshift.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/example/web_export"))
    args = parser.parse_args()

    exp = args.experiment_dir
    out = ensure_dir(args.out)
    summary = read_json(exp / "metrics" / "summary.json")
    failures = read_json(exp / "failures" / "report.json")
    retrieval = read_json(exp / "retrieval" / "summary.json")
    manifest = read_json(exp / "manifest.json")

    write_json(out / "results-summary.json", summary)
    write_json(out / "failure-gallery.json", failures)
    write_json(out / "retrieval-summary.json", retrieval)
    write_json(out / "manifest.json", manifest)

    # Sample first few trajectories for playback
    traces_path = exp / "traces" / "trajectories.jsonl"
    samples = []
    with traces_path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= 20:
                break
            samples.append(json.loads(line))
    write_json(out / "sample-trajectories.json", samples)
    print(f"Wrote web export to {out}")


if __name__ == "__main__":
    main()
