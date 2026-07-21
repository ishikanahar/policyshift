"""Update training package exports for Phase 3–4."""

from policyshift.training.distill import DistilledStudentAgent, run_phase3_smoke
from policyshift.training.dpo_pipeline import DPOStudentAgent, run_phase4_smoke
from policyshift.training.dpo_trainer import DPOTrainConfig, load_dpo_checkpoint, run_dpo
from policyshift.training.preferences import (
    build_preference_dataset,
    pairs_to_dpo_examples,
    write_preference_artifacts,
)
from policyshift.training.sft_data import build_sft_dataset, write_sft_dataset
from policyshift.training.sft_trainer import TrainConfig, load_checkpoint, run_sft
from policyshift.training.teacher import (
    TeacherTrajectoryGenerator,
    load_trajectories_jsonl,
    write_teacher_artifacts,
)

__all__ = [
    "DPOStudentAgent",
    "DPOTrainConfig",
    "DistilledStudentAgent",
    "TeacherTrajectoryGenerator",
    "TrainConfig",
    "build_preference_dataset",
    "build_sft_dataset",
    "load_checkpoint",
    "load_dpo_checkpoint",
    "load_trajectories_jsonl",
    "pairs_to_dpo_examples",
    "run_dpo",
    "run_phase3_smoke",
    "run_phase4_smoke",
    "run_sft",
    "write_preference_artifacts",
    "write_sft_dataset",
    "write_teacher_artifacts",
]
