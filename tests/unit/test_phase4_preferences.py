"""Phase 4 unit tests: preference pairs, DPO smoke, checkpoint load."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import PreferenceSource, Split
from policyshift.training import (
    DPOStudentAgent,
    DPOTrainConfig,
    build_preference_dataset,
    load_dpo_checkpoint,
    pairs_to_dpo_examples,
    run_dpo,
    run_phase4_smoke,
    write_preference_artifacts,
)

pytestmark = pytest.mark.phase4


def test_preference_pairs_cover_core_sources() -> None:
    store = PolicyStore.from_builtin()
    cases = [c for c in generate_cases(seed=42, n_cases=30) if c.split == Split.TRAIN][:6]
    dataset = build_preference_dataset(cases, policy_store=store)
    assert dataset["n_pairs"] >= len(cases)  # at least one pair per case (actually 3)
    assert dataset["n_pairs"] == len(cases) * 3
    sources = {p.source for p in dataset["pairs"]}
    assert PreferenceSource.CURRENT_VS_STALE in sources
    assert PreferenceSource.GROUNDED_VS_UNSUPPORTED in sources
    assert PreferenceSource.SAFE_VS_UNSAFE in sources
    for pair in dataset["pairs"]:
        assert pair.chosen_trajectory_id != pair.rejected_trajectory_id
        assert pair.preference_reason


def test_preference_artifacts_and_dpo_jsonl(tmp_path: Path) -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=1, n_cases=8)
    dataset = build_preference_dataset(cases[:4], policy_store=store)
    traj_map = {t.trajectory_id: t for t in dataset["chosen"] + dataset["rejected"]}
    examples = pairs_to_dpo_examples(cases, dataset["pairs"], traj_map)
    assert examples
    assert "prompt" in examples[0] and "chosen" in examples[0] and "rejected" in examples[0]
    paths = write_preference_artifacts(
        tmp_path / "prefs",
        pairs=dataset["pairs"],
        chosen=dataset["chosen"],
        rejected=dataset["rejected"],
        dpo_examples=examples,
        skipped=dataset["skipped"],
    )
    assert paths["explorer"].exists()
    assert paths["dpo_jsonl"].exists()
    explorer = paths["explorer"].read_text(encoding="utf-8")
    assert "preference_reason" in explorer


def test_smoke_dpo_trains_and_loads(tmp_path: Path) -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=0, n_cases=6)
    dataset = build_preference_dataset(cases, policy_store=store)
    traj_map = {t.trajectory_id: t for t in dataset["chosen"] + dataset["rejected"]}
    examples = pairs_to_dpo_examples(cases, dataset["pairs"], traj_map)
    paths = write_preference_artifacts(
        tmp_path / "prefs",
        pairs=dataset["pairs"],
        chosen=dataset["chosen"],
        rejected=dataset["rejected"],
        dpo_examples=examples,
    )
    metrics = run_dpo(
        DPOTrainConfig(
            output_dir=str(tmp_path / "ckpt"),
            train_file=str(paths["dpo_jsonl"]),
            smoke=True,
            max_steps=2,
            seed=0,
        )
    )
    assert metrics["status"] == "completed"
    assert metrics["checkpoint_load"]["ok"] is True
    loaded = load_dpo_checkpoint(metrics["training"]["checkpoint"])
    assert loaded["path"]


def test_phase4_smoke_exports_comparison(tmp_path: Path) -> None:
    result = run_phase4_smoke(
        seed=42,
        n_cases=16,
        n_eval=8,
        artifact_root=tmp_path / "experiments",
        experiment_id="phase4-test",
    )
    assert result["experiment_id"] == "phase4-test"
    assert result["summary"]["preferences"]["n_pairs"] >= 1
    assert result["summary"]["checkpoint_loaded"]["path"]
    conditions = result["summary"]["conditions"]
    assert set(conditions) >= {"baseline", "rag", "dpo"}
    assert conditions["dpo"]["success"] >= conditions["baseline"]["success"]
    assert Path(result["paths"]["preference_explorer"]).exists()
    assert Path(result["paths"]["manifest"]).exists()


def test_dpo_student_replays_chosen() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=42, n_cases=5)[0]
    dataset = build_preference_dataset([case], policy_store=store)
    chosen = dataset["chosen"][0]
    student = DPOStudentAgent({case.case_id: chosen}, store)
    traj = student.resolve(case)
    assert traj.model_id == "dpo-student-smoke"
    assert traj.final_answer == chosen.final_answer
    assert traj.success is True
