"""Training utilities: teacher distillation, SFT data, LoRA smoke/full trainers."""

from policyshift.training.distill import DistilledStudentAgent, run_phase3_smoke
from policyshift.training.sft_data import build_sft_dataset, write_sft_dataset
from policyshift.training.sft_trainer import TrainConfig, load_checkpoint, run_sft
from policyshift.training.teacher import (
    TeacherTrajectoryGenerator,
    load_trajectories_jsonl,
    write_teacher_artifacts,
)

__all__ = [
    "DistilledStudentAgent",
    "TeacherTrajectoryGenerator",
    "TrainConfig",
    "build_sft_dataset",
    "load_checkpoint",
    "load_trajectories_jsonl",
    "run_phase3_smoke",
    "run_sft",
    "write_sft_dataset",
    "write_teacher_artifacts",
]
