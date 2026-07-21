"""Phase 3 unit tests: teacher filtering, SFT data, smoke training, checkpoint load."""

from __future__ import annotations

from pathlib import Path

import pytest

from policyshift.data_generation.cases import generate_cases
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import Split
from policyshift.training import (
    DistilledStudentAgent,
    TeacherTrajectoryGenerator,
    TrainConfig,
    build_sft_dataset,
    load_checkpoint,
    run_phase3_smoke,
    run_sft,
    write_sft_dataset,
    write_teacher_artifacts,
)

pytestmark = pytest.mark.phase3


def test_teacher_oracle_accepts_verified_trajectories() -> None:
    store = PolicyStore.from_builtin()
    cases = [c for c in generate_cases(seed=42, n_cases=30) if c.split == Split.TRAIN][:8]
    gen = TeacherTrajectoryGenerator(store, source="oracle")
    accepted, report = gen.generate_batch(cases)
    assert report.teacher_calls == len(cases)
    assert report.n_accepted == len(cases)
    assert len(accepted) == len(cases)
    assert all(t.success for t in accepted)


def test_teacher_rejects_bad_trajectory() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=1, n_cases=3)[0]
    gen = TeacherTrajectoryGenerator(store)
    good = gen.generate_for_case(case)
    good.final_answer = "teleport_to_mars"
    good.cited_policy_versions = ["POL-FAKE@9.9"]
    ok, reasons = gen.filter_trajectory(case, good)
    assert ok is False
    assert reasons


def test_sft_dataset_build_and_write(tmp_path: Path) -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=42, n_cases=10)
    accepted, _ = TeacherTrajectoryGenerator(store).generate_batch(cases[:5])
    examples = build_sft_dataset(cases, accepted)
    assert len(examples) == len(accepted)
    assert examples[0]["messages"][-1]["role"] == "assistant"
    paths = write_sft_dataset(examples, tmp_path / "sft")
    assert paths["jsonl"].exists()
    assert paths["stats"].exists()


def test_smoke_sft_trains_and_checkpoint_loads(tmp_path: Path) -> None:
    store = PolicyStore.from_builtin()
    cases = generate_cases(seed=0, n_cases=8)
    accepted, _ = TeacherTrajectoryGenerator(store).generate_batch(cases)
    examples = build_sft_dataset(cases, accepted)
    write_sft_dataset(examples, tmp_path / "sft")
    metrics = run_sft(
        TrainConfig(
            output_dir=str(tmp_path / "ckpt"),
            train_file=str(tmp_path / "sft" / "sft_train.jsonl"),
            smoke=True,
            max_steps=2,
            seed=0,
        )
    )
    assert metrics["status"] == "completed"
    assert metrics["checkpoint_load"]["ok"] is True
    loaded = load_checkpoint(metrics["training"]["checkpoint"])
    assert loaded["path"]


def test_phase3_smoke_exports_comparison(tmp_path: Path) -> None:
    result = run_phase3_smoke(
        seed=42,
        n_cases=16,
        n_eval=8,
        artifact_root=tmp_path / "experiments",
        experiment_id="phase3-test",
    )
    assert result["experiment_id"] == "phase3-test"
    assert result["summary"]["teacher"]["n_accepted"] >= 1
    assert result["summary"]["sft_examples"] >= 1
    assert result["summary"]["checkpoint_loaded"]["path"]
    conditions = result["summary"]["conditions"]
    assert set(conditions) >= {"baseline", "rag", "distilled"}
    # Distilled should beat or match baseline on smoke eval
    assert conditions["distilled"]["success"] >= conditions["baseline"]["success"]
    assert Path(result["paths"]["manifest"]).exists()


def test_distilled_student_replays_teacher() -> None:
    store = PolicyStore.from_builtin()
    case = generate_cases(seed=42, n_cases=5)[0]
    teacher, _ = TeacherTrajectoryGenerator(store).generate_batch([case])
    student = DistilledStudentAgent({teacher[0].case_id: teacher[0]}, store)
    traj = student.resolve(case)
    assert traj.model_id == "distilled-student-smoke"
    assert traj.final_answer == teacher[0].final_answer
    assert traj.success is True
