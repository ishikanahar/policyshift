#!/usr/bin/env python3
"""Build preference dataset quality report (markdown + JSON)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from policyshift.utils.io import ensure_dir, write_json


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    rows = _load_jsonl(args.dpo)
    by_version = Counter(str(r.get("policy_version", "unknown")) for r in rows)
    by_source = Counter(str(r.get("source", "unknown")) for r in rows)
    by_fail = Counter()
    for r in rows:
        for f in r.get("failure_categories") or []:
            by_fail[str(f)] += 1

    chosen_lens = [len(str(r.get("chosen", ""))) for r in rows]
    rejected_lens = [len(str(r.get("rejected", ""))) for r in rows]
    margins = [float(r.get("reward_margin", 0.0)) for r in rows]

    # Duplicate detection on (case_id, source)
    keys = [(r.get("case_id"), r.get("source")) for r in rows]
    dupes = len(keys) - len(set(keys))

    forbidden = [r for r in rows if str(r.get("policy_version")) == "2.0"]
    stats = {
        "n_pairs": len(rows),
        "by_policy_version": dict(by_version),
        "by_source": dict(by_source),
        "by_failure_category": dict(by_fail),
        "mean_chosen_len": sum(chosen_lens) / len(chosen_lens) if chosen_lens else 0,
        "mean_rejected_len": sum(rejected_lens) / len(rejected_lens) if rejected_lens else 0,
        "mean_reward_margin": sum(margins) / len(margins) if margins else 0,
        "duplicate_case_source_pairs": dupes,
        "n_v2_leakage_rows": len(forbidden),
        "leakage_passed": len(forbidden) == 0,
        "manual_review_sample_ids": [r.get("id") for r in rows[:50]],
    }

    out = ensure_dir(args.out_dir)
    write_json(out / "preference_dataset_stats.json", stats)

    md = out / "preference_dataset_report.md"
    lines = [
        "# Preference dataset report",
        "",
        f"- Pairs: **{stats['n_pairs']}**",
        f"- Leakage check (no v2.0 in prefs): **{'PASS' if stats['leakage_passed'] else 'FAIL'}**",
        f"- Duplicate (case_id, source): **{dupes}**",
        f"- Mean reward margin: **{stats['mean_reward_margin']:.3f}**",
        "",
        "## By policy version",
        "",
    ]
    for k, v in sorted(by_version.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## By preference source", ""]
    for k, v in sorted(by_source.items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Failure categories on rejected", ""]
    for k, v in by_fail.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Manual review sample",
        "",
        "First 50 pair ids (review for hard-negative quality):",
        "",
    ]
    for pid in stats["manual_review_sample_ids"]:
        lines.append(f"- `{pid}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(md), **{k: stats[k] for k in ("n_pairs", "leakage_passed")}}, indent=2))


if __name__ == "__main__":
    main()
