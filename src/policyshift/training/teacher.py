"""Teacher trajectory generation for distillation.

Smoke default uses the deterministic OracleAgent as a verified teacher
(no API key required). Optional: load pre-generated JSONL, or call an API teacher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from policyshift.agents.oracle import OracleAgent
from policyshift.environment.policy_store import PolicyStore
from policyshift.schemas import AgentTrajectory, CaseEvent, TrainingMethod
from policyshift.utils.hashing import sha256_text
from policyshift.utils.io import ensure_dir, write_json, write_jsonl
from policyshift.verification.verifiers import TrajectoryVerifier

TeacherSource = Literal["oracle", "file", "api"]


@dataclass
class TeacherCallRecord:
    case_id: str
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    trajectory_id: str | None = None
    token_count: int | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None


@dataclass
class TeacherGenerationReport:
    source: str
    n_cases: int
    n_accepted: int
    n_rejected: int
    teacher_calls: int
    total_tokens: int
    estimated_cost_usd: float
    records: list[TeacherCallRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_cases": self.n_cases,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "teacher_calls": self.teacher_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "records": [
                {
                    "case_id": r.case_id,
                    "accepted": r.accepted,
                    "rejection_reasons": r.rejection_reasons,
                    "trajectory_id": r.trajectory_id,
                    "token_count": r.token_count,
                    "latency_ms": r.latency_ms,
                    "estimated_cost_usd": r.estimated_cost_usd,
                }
                for r in self.records
            ],
        }


class TeacherTrajectoryGenerator:
    """Generate and verifier-filter teacher trajectories."""

    def __init__(
        self,
        policy_store: PolicyStore | None = None,
        *,
        source: TeacherSource = "oracle",
        file_path: str | Path | None = None,
        cost_per_1k_tokens: float = 0.0,
    ) -> None:
        self.policy_store = policy_store or PolicyStore.from_builtin()
        self.source = source
        self.file_path = Path(file_path) if file_path else None
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.verifier = TrajectoryVerifier(self.policy_store)
        self.oracle = OracleAgent(self.policy_store)

    def generate_for_case(self, case: CaseEvent) -> AgentTrajectory:
        if self.source == "oracle":
            traj = self.oracle.resolve(case)
            traj.training_method = TrainingMethod.DISTILLATION
            traj.model_id = "teacher-oracle"
            # Deterministic teacher id
            traj.trajectory_id = f"traj-teacher-{sha256_text(case.case_id)[:12]}"
            return traj
        if self.source == "file":
            raise RuntimeError("Use generate_batch with file source, not generate_for_case")
        if self.source == "api":
            raise RuntimeError(
                "API teacher not configured for smoke. Set TEACHER_API_KEY and implement "
                "adapter, or use source=oracle / pre-generated file."
            )
        raise ValueError(f"Unknown teacher source: {self.source}")

    def filter_trajectory(
        self, case: CaseEvent, trajectory: AgentTrajectory
    ) -> tuple[bool, list[str]]:
        results = self.verifier.verify(case, trajectory)
        trajectory.verifier_results = results
        trajectory.failure_categories = self.verifier.categorize_failures(
            case, trajectory, results
        )
        trajectory.success = self.verifier.success(results)
        reasons: list[str] = []
        if not trajectory.success:
            for r in results:
                if not r.passed:
                    reasons.append(f"{r.name}:{r.detail}")
        return trajectory.success, reasons

    def generate_batch(
        self,
        cases: list[CaseEvent],
        *,
        trajectories_from_file: list[AgentTrajectory] | None = None,
    ) -> tuple[list[AgentTrajectory], TeacherGenerationReport]:
        accepted: list[AgentTrajectory] = []
        records: list[TeacherCallRecord] = []
        total_tokens = 0
        teacher_calls = 0

        file_map: dict[str, AgentTrajectory] = {}
        if self.source == "file":
            if trajectories_from_file is None and self.file_path:
                trajectories_from_file = load_trajectories_jsonl(self.file_path)
            if not trajectories_from_file:
                raise ValueError("file teacher source requires trajectories")
            file_map = {t.case_id: t for t in trajectories_from_file}

        for case in cases:
            teacher_calls += 1
            if self.source == "file":
                if case.case_id not in file_map:
                    records.append(
                        TeacherCallRecord(
                            case_id=case.case_id,
                            accepted=False,
                            rejection_reasons=["missing_teacher_trajectory"],
                        )
                    )
                    continue
                traj = file_map[case.case_id]
            else:
                traj = self.generate_for_case(case)

            # Approximate tokens from action text length for cost accounting
            token_count = estimate_trajectory_tokens(traj)
            total_tokens += token_count
            cost = (token_count / 1000.0) * self.cost_per_1k_tokens

            ok, reasons = self.filter_trajectory(case, traj)
            records.append(
                TeacherCallRecord(
                    case_id=case.case_id,
                    accepted=ok,
                    rejection_reasons=reasons,
                    trajectory_id=traj.trajectory_id,
                    token_count=token_count,
                    latency_ms=traj.latency_ms,
                    estimated_cost_usd=cost,
                )
            )
            if ok:
                accepted.append(traj)

        report = TeacherGenerationReport(
            source=self.source,
            n_cases=len(cases),
            n_accepted=len(accepted),
            n_rejected=len(cases) - len(accepted),
            teacher_calls=teacher_calls,
            total_tokens=total_tokens,
            estimated_cost_usd=(total_tokens / 1000.0) * self.cost_per_1k_tokens,
            records=records,
        )
        return accepted, report


def estimate_trajectory_tokens(trajectory: AgentTrajectory) -> int:
    text = trajectory.final_answer or ""
    for action in trajectory.actions:
        text += " " + action.thought_summary + " " + action.tool_name
        text += " " + json.dumps(action.arguments, default=str)
    # Rough char/4 heuristic
    return max(1, len(text) // 4)


def load_trajectories_jsonl(path: str | Path) -> list[AgentTrajectory]:
    rows: list[AgentTrajectory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(AgentTrajectory.model_validate(json.loads(line)))
    return rows


def write_teacher_artifacts(
    out_dir: str | Path,
    accepted: list[AgentTrajectory],
    report: TeacherGenerationReport,
    *,
    rejected: list[AgentTrajectory] | None = None,
) -> dict[str, Path]:
    root = ensure_dir(out_dir)
    paths = {
        "accepted": write_jsonl(root / "accepted_trajectories.jsonl", accepted),
        "report": write_json(root / "teacher_report.json", report.to_dict()),
    }
    if rejected:
        paths["rejected"] = write_jsonl(root / "rejected_trajectories.jsonl", rejected)
    return paths
