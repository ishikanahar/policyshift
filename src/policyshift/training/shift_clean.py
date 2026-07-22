"""Clean temporal policy-shift dataset preparation and leakage validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import CaseEvent, Split
from policyshift.training.leakage import collect_train_versions, load_jsonl
from policyshift.training.preferences import (
    build_preference_dataset,
    pairs_to_dpo_examples,
)
from policyshift.training.sft_data import build_sft_dataset, case_to_prompt
from policyshift.training.teacher import TeacherTrajectoryGenerator
from policyshift.training.version_filters import filter_cases_by_versions
from policyshift.utils.hashing import sha256_json, sha256_text
from policyshift.utils.io import ensure_dir, write_json, write_jsonl

TRAIN_VERSIONS = ("1.0", "1.1")
EVAL_VERSIONS = ("2.0",)
DEFAULT_DATA_ROOT = Path("data/shift_clean")
VALIDATION_STAMP_NAME = "validation_stamp.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preference_type_bucket(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "")
    failure = str(row.get("failure_type") or "")
    cats = [str(c) for c in (row.get("failure_categories") or [])]
    blob = " ".join([source, failure, *cats]).lower()
    if "stale" in blob or source == "current_vs_stale":
        return "stale_policy"
    if "unsafe" in blob or source == "safe_vs_unsafe":
        return "unsafe_action"
    if "invalid_tool" in blob or "tool" in failure:
        return "invalid_tool_action"
    return "other_easy_negative"


def _enrich_common(
    row: dict[str, Any],
    *,
    case: CaseEvent,
    split_label: str,
    generation_seed: int,
) -> dict[str, Any]:
    out = dict(row)
    out["case_id"] = case.case_id
    out["policy_version"] = case.expected_policy_version
    out["policy_id"] = case.expected_policy_id
    out["expected_policy_id"] = case.expected_policy_id
    out["domain"] = case.domain.value
    out["effective_date"] = case.occurred_at.isoformat()
    out["split"] = split_label
    out["generation_seed"] = generation_seed
    return out


def prepare_shift_clean_data(
    *,
    out_root: Path = DEFAULT_DATA_ROOT,
    seed: int = 42,
    n_train_cases: int = 80,
    n_eval_cases: int = 24,
) -> dict[str, Any]:
    """Build leakage-free train (1.0/1.1) + held-out v2.0 eval under ``data/shift_clean``."""
    store = PolicyStore.from_builtin()
    all_cases = generate_cases(
        seed=seed, n_cases=max(n_train_cases * 3, n_eval_cases * 3, 240)
    )

    eval_pool = [c for c in all_cases if c.split == Split.VALIDATION]
    eval_cases = filter_cases_by_versions(eval_pool, list(EVAL_VERSIONS))[:n_eval_cases]
    if len(eval_cases) < max(4, n_eval_cases // 3):
        eval_cases = filter_cases_by_versions(all_cases, list(EVAL_VERSIONS))[:n_eval_cases]
    eval_ids = {c.case_id for c in eval_cases}

    # Train only on 1.0/1.1 and never on held-out eval case IDs.
    train_pool = [
        c
        for c in all_cases
        if c.split == Split.TRAIN and c.case_id not in eval_ids
    ]
    train_cases = filter_cases_by_versions(train_pool, list(TRAIN_VERSIONS))[:n_train_cases]
    if len(train_cases) < n_train_cases:
        extra = [
            c
            for c in filter_cases_by_versions(all_cases, list(TRAIN_VERSIONS))
            if c.case_id not in eval_ids and c.case_id not in {x.case_id for x in train_cases}
        ]
        train_cases = (train_cases + extra)[:n_train_cases]

    root = ensure_dir(out_root)
    sft_dir = ensure_dir(root / "sft")
    dpo_dir = ensure_dir(root / "dpo")
    eval_dir = ensure_dir(root / "eval")
    man_dir = ensure_dir(root / "manifests")

    gen = TeacherTrajectoryGenerator(store, source="oracle")
    accepted, teacher_report = gen.generate_batch(train_cases)
    sft_examples = build_sft_dataset(train_cases, accepted)
    sft_rows = [
        _enrich_common(
            ex,
            case=next(c for c in train_cases if c.case_id == ex["case_id"]),
            split_label="train",
            generation_seed=seed,
        )
        for ex in sft_examples
    ]

    prefs = build_preference_dataset(train_cases, policy_store=store)
    traj_map = {t.trajectory_id: t for t in prefs["chosen"] + prefs["rejected"]}
    dpo_examples = pairs_to_dpo_examples(train_cases, prefs["pairs"], traj_map)
    case_by_id = {c.case_id: c for c in train_cases}
    dpo_rows = [
        _enrich_common(
            ex,
            case=case_by_id[ex["case_id"]],
            split_label="train",
            generation_seed=seed,
        )
        for ex in dpo_examples
        if ex["case_id"] in case_by_id
    ]
    # Hard filter: never keep v2 rows even if a bug slips through.
    sft_rows = [r for r in sft_rows if str(r.get("policy_version")) in TRAIN_VERSIONS]
    dpo_rows = [r for r in dpo_rows if str(r.get("policy_version")) in TRAIN_VERSIONS]

    pair_type_counts = Counter(preference_type_bucket(r) for r in dpo_rows)

    eval_rows: list[dict[str, Any]] = []
    for case in eval_cases:
        eval_rows.append(
            {
                "case_id": case.case_id,
                "policy_version": case.expected_policy_version,
                "policy_id": case.expected_policy_id,
                "domain": case.domain.value,
                "effective_date": case.occurred_at.isoformat(),
                "split": "heldout_v2",
                "generation_seed": seed,
                "expected_resolution": case.expected_resolution,
                "expected_policy_key": case.expected_policy_key,
                "prompt": case_to_prompt(case),
                "case": case.model_dump(mode="json"),
            }
        )

    sft_path = write_jsonl(sft_dir / "sft_train.jsonl", sft_rows)
    dpo_path = write_jsonl(dpo_dir / "dpo_train.jsonl", dpo_rows)
    eval_path = write_jsonl(eval_dir / "v2_eval.jsonl", eval_rows)

    hashes = {
        "sft_train.jsonl": file_sha256(sft_path),
        "dpo_train.jsonl": file_sha256(dpo_path),
        "v2_eval.jsonl": file_sha256(eval_path),
    }

    split_manifest = {
        "experiment_name": "shift-clean",
        "train_versions": list(TRAIN_VERSIONS),
        "eval_versions": list(EVAL_VERSIONS),
        "generation_seed": seed,
        "n_train_cases": len(train_cases),
        "n_eval_cases": len(eval_cases),
        "train_case_ids": [c.case_id for c in train_cases],
        "eval_case_ids": [c.case_id for c in eval_cases],
        "paths": {
            "sft": str(sft_path),
            "dpo": str(dpo_path),
            "eval": str(eval_path),
        },
        "dataset_hashes": hashes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_report": {
            "n_accepted": getattr(teacher_report, "n_accepted", len(accepted)),
            "n_rejected": getattr(teacher_report, "n_rejected", 0),
        },
    }
    sft_manifest = {
        "path": str(sft_path),
        "n_examples": len(sft_rows),
        "policy_versions": sorted(collect_train_versions(sft_rows)),
        "dataset_hash": hashes["sft_train.jsonl"],
        "split": "train",
        "generation_seed": seed,
    }
    dpo_manifest = {
        "path": str(dpo_path),
        "n_pairs": len(dpo_rows),
        "policy_versions": sorted(collect_train_versions(dpo_rows)),
        "dataset_hash": hashes["dpo_train.jsonl"],
        "split": "train",
        "generation_seed": seed,
        "preference_pair_types": dict(pair_type_counts),
        "preference_pair_type_notes": (
            "Distribution disclosed for limitations; current generator emphasizes "
            "stale/unsafe/unsupported negatives rather than rich invalid-tool pairs."
        ),
    }

    write_json(man_dir / "split_manifest.json", split_manifest)
    write_json(man_dir / "sft_manifest.json", sft_manifest)
    write_json(man_dir / "dpo_manifest.json", dpo_manifest)
    # Invalidate prior validation stamp when data is regenerated.
    stamp = man_dir / VALIDATION_STAMP_NAME
    if stamp.exists():
        stamp.unlink()

    return {
        "out_root": str(root),
        "n_sft": len(sft_rows),
        "n_dpo": len(dpo_rows),
        "n_eval": len(eval_rows),
        "train_versions": list(TRAIN_VERSIONS),
        "eval_versions": list(EVAL_VERSIONS),
        "dataset_hashes": hashes,
        "preference_pair_types": dict(pair_type_counts),
        "paths": split_manifest["paths"],
    }


def _versions_in_rows(rows: list[dict[str, Any]]) -> set[str]:
    return collect_train_versions(rows)


def _row_text_blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("prompt") or ""),
        str(row.get("chosen") or ""),
        str(row.get("rejected") or ""),
        str(row.get("text") or ""),
        json.dumps(row.get("messages") or [], default=str),
        json.dumps(row.get("messages_chosen") or [], default=str),
        json.dumps(row.get("messages_rejected") or [], default=str),
    ]
    return "\n".join(parts)


def validate_shift_split(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    write_stamp: bool = True,
) -> dict[str, Any]:
    """Hard leakage gate for the clean shift split. Raises AssertionError on failure."""
    sft_path = data_root / "sft" / "sft_train.jsonl"
    dpo_path = data_root / "dpo" / "dpo_train.jsonl"
    eval_path = data_root / "eval" / "v2_eval.jsonl"
    man_dir = data_root / "manifests"
    for path in (sft_path, dpo_path, eval_path):
        if not path.exists():
            raise AssertionError(f"Missing required dataset file: {path}")

    sft_rows = load_jsonl(sft_path)
    dpo_rows = load_jsonl(dpo_path)
    eval_rows = load_jsonl(eval_path)
    sft_manifest = json.loads((man_dir / "sft_manifest.json").read_text(encoding="utf-8"))
    dpo_manifest = json.loads((man_dir / "dpo_manifest.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((man_dir / "split_manifest.json").read_text(encoding="utf-8"))

    sft_versions = _versions_in_rows(sft_rows)
    dpo_versions = _versions_in_rows(dpo_rows)
    eval_versions = _versions_in_rows(eval_rows)

    leakage_events: list[dict[str, Any]] = []

    if sft_versions - set(TRAIN_VERSIONS):
        leakage_events.append({"type": "sft_forbidden_version", "versions": sorted(sft_versions)})
    if dpo_versions - set(TRAIN_VERSIONS):
        leakage_events.append({"type": "dpo_forbidden_version", "versions": sorted(dpo_versions)})
    if eval_versions != set(EVAL_VERSIONS):
        leakage_events.append({"type": "eval_version_mismatch", "versions": sorted(eval_versions)})

    train_case_ids = {str(r.get("case_id")) for r in sft_rows} | {str(r.get("case_id")) for r in dpo_rows}
    eval_case_ids = {str(r.get("case_id")) for r in eval_rows}
    overlap = sorted(train_case_ids & eval_case_ids)
    if overlap:
        leakage_events.append({"type": "case_id_overlap", "n": len(overlap), "examples": overlap[:20]})

    # Duplicated case IDs that appear with both train and heldout labels.
    split_by_case: dict[str, set[str]] = {}
    for row in sft_rows + dpo_rows + eval_rows:
        split_by_case.setdefault(str(row.get("case_id")), set()).add(str(row.get("split")))
    cross = {cid: sorted(splits) for cid, splits in split_by_case.items() if len(splits) > 1}
    if cross:
        leakage_events.append(
            {"type": "cross_split_case_ids", "n": len(cross), "examples": list(cross.items())[:10]}
        )

    store = PolicyStore.from_builtin()
    other_clause_texts: set[str] = set()
    v2_clause_texts: list[str] = []
    for doc in store.all():
        for clause in doc.clauses:
            text = (clause.text or "").strip()
            if not text:
                continue
            if str(doc.version) == "2.0":
                v2_clause_texts.append(text)
            else:
                other_clause_texts.add(text)
    v2_unique = [t for t in v2_clause_texts if t not in other_clause_texts and len(t) >= 40]

    train_blob = "\n".join(_row_text_blob(r) for r in sft_rows + dpo_rows)
    if "@2.0" in train_blob or 'version": "2.0"' in train_blob or "version': '2.0'" in train_blob:
        leakage_events.append({"type": "v2_version_marker_in_train"})
    for snippet in v2_unique:
        if snippet in train_blob:
            leakage_events.append(
                {"type": "v2_policy_text_in_train", "snippet": snippet[:120]}
            )
            break

    # Exact eval reference answers / preference strings must not appear in train.
    eval_refs = set()
    for r in eval_rows:
        ans = str(r.get("expected_resolution") or "").strip()
        if ans:
            eval_refs.add(ans)
        prompt = str(r.get("prompt") or "").strip()
        if prompt:
            eval_refs.add(prompt)
    for row in sft_rows + dpo_rows:
        for key in ("chosen", "rejected", "text", "prompt"):
            val = str(row.get(key) or "").strip()
            if val and val in eval_refs:
                leakage_events.append(
                    {
                        "type": "eval_reference_in_train",
                        "field": key,
                        "row_id": row.get("id") or row.get("case_id"),
                    }
                )

    hashes = {
        "sft_train.jsonl": file_sha256(sft_path),
        "dpo_train.jsonl": file_sha256(dpo_path),
        "v2_eval.jsonl": file_sha256(eval_path),
    }
    if hashes["sft_train.jsonl"] != sft_manifest.get("dataset_hash"):
        leakage_events.append({"type": "sft_manifest_hash_mismatch"})
    if hashes["dpo_train.jsonl"] != dpo_manifest.get("dataset_hash"):
        leakage_events.append({"type": "dpo_manifest_hash_mismatch"})
    if hashes != split_manifest.get("dataset_hashes"):
        leakage_events.append({"type": "split_manifest_hash_mismatch"})

    if Path(sft_manifest.get("path", "")).resolve() != sft_path.resolve():
        leakage_events.append({"type": "sft_manifest_path_mismatch"})
    if Path(dpo_manifest.get("path", "")).resolve() != dpo_path.resolve():
        leakage_events.append({"type": "dpo_manifest_path_mismatch"})

    report = {
        "passed": len(leakage_events) == 0,
        "sft_training_versions": sorted(sft_versions),
        "dpo_training_versions": sorted(dpo_versions),
        "evaluation_versions": sorted(eval_versions),
        "n_sft_examples": len(sft_rows),
        "n_dpo_pairs": len(dpo_rows),
        "n_heldout_v2_cases": len(eval_rows),
        "leakage_count": len(leakage_events),
        "leakage_events": leakage_events[:50],
        "dataset_hashes": hashes,
        "preference_pair_types": dpo_manifest.get("preference_pair_types", {}),
        "validated_paths": {
            "sft": str(sft_path),
            "dpo": str(dpo_path),
            "eval": str(eval_path),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if not report["passed"]:
        raise AssertionError(
            f"Shift split validation failed with {len(leakage_events)} leakage event(s): "
            f"{leakage_events[:5]}"
        )

    if write_stamp:
        stamp = {
            **report,
            "stamp": sha256_json(
                {
                    "hashes": hashes,
                    "paths": report["validated_paths"],
                    "train_versions": list(TRAIN_VERSIONS),
                    "eval_versions": list(EVAL_VERSIONS),
                }
            ),
        }
        write_json(man_dir / VALIDATION_STAMP_NAME, stamp)
    return report


def require_shift_clean_validation(train_file: str | Path) -> Path | None:
    """If training on shift_clean data, require a valid validation stamp."""
    path = Path(train_file).resolve()
    parts = path.parts
    if "shift_clean" not in parts:
        return None
    # .../data/shift_clean/...
    try:
        idx = parts.index("shift_clean")
    except ValueError:
        return None
    data_root = Path(*parts[: idx + 1])
    stamp_path = data_root / "manifests" / VALIDATION_STAMP_NAME
    if not stamp_path.exists():
        raise SystemExit(
            f"Refuse to train on {train_file}: missing {stamp_path}. "
            "Run: python scripts/validate_shift_split.py"
        )
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    if not stamp.get("passed"):
        raise SystemExit(f"Refuse to train: validation stamp not passed ({stamp_path})")
    # Re-check hashes against current files.
    sft_path = data_root / "sft" / "sft_train.jsonl"
    dpo_path = data_root / "dpo" / "dpo_train.jsonl"
    expected = stamp.get("dataset_hashes") or {}
    if sft_path.exists() and expected.get("sft_train.jsonl") != file_sha256(sft_path):
        raise SystemExit("Refuse to train: SFT dataset hash changed since validation.")
    if dpo_path.exists() and expected.get("dpo_train.jsonl") != file_sha256(dpo_path):
        raise SystemExit("Refuse to train: DPO dataset hash changed since validation.")
    return stamp_path


def write_provenance(
    out_path: Path,
    *,
    experiment_name: str = "shift-clean",
    base_model: str,
    initialization_checkpoint: str | None,
    adapter_path: str,
    training_policy_versions: list[str],
    evaluation_policy_versions: list[str],
    dataset_path: str,
    dataset_hash: str,
    seed: int,
    training_stage: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    import subprocess

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = "unknown"

    payload: dict[str, Any] = {
        "experiment_name": experiment_name,
        "base_model": base_model,
        "initialization_checkpoint": initialization_checkpoint,
        "adapter_path": adapter_path,
        "training_policy_versions": training_policy_versions,
        "evaluation_policy_versions": evaluation_policy_versions,
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "seed": seed,
        "smoke": False,
        "leakage_count": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if training_stage:
        payload["training_stage"] = training_stage
    if extra:
        payload.update(extra)
    return write_json(out_path, payload)
