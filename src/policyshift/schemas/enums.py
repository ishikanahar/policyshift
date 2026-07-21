"""Shared enumerations for PolicyShift schemas."""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    MATERIALS = "materials"
    LABORATORY = "laboratory"
    AI_GOVERNANCE = "ai_governance"


class RuleType(str, Enum):
    REQUIREMENT = "requirement"
    PERMISSION = "permission"
    PROHIBITION = "prohibition"
    EXCEPTION = "exception"
    ESCALATION = "escalation"
    DEFINITION = "definition"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TrainingMethod(str, Enum):
    BASE = "base"
    RAG = "rag"
    SFT_SEQUENTIAL = "sft_sequential"
    SFT_REPLAY = "sft_replay"
    DISTILLATION = "distillation"
    DPO = "dpo"
    RL = "rl"
    ORACLE = "oracle"
    DEMO = "demo"


class FailureCategory(str, Enum):
    STALE_POLICY_SELECTED = "stale_policy_selected"
    CORRECT_POLICY_IGNORED = "correct_policy_retrieved_but_ignored"
    INCORRECT_CLAUSE = "incorrect_clause_selected"
    MISSING_EVIDENCE_OVERLOOKED = "missing_evidence_overlooked"
    INVALID_TOOL = "invalid_tool"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    UNSUPPORTED_FINAL_ANSWER = "unsupported_final_answer"
    HALLUCINATED_EVIDENCE = "hallucinated_evidence"
    HALLUCINATED_POLICY = "hallucinated_policy"
    PREMATURE_ACTION = "premature_action"
    UNSAFE_ACTION = "unsafe_action"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"
    EXCESSIVE_REFUSAL = "excessive_refusal"
    EXCESSIVE_TOOL_USE = "excessive_tool_use"
    REWARD_HACKING = "reward_hacking"
    INVALID_REASONING_PATH = "correct_result_through_invalid_reasoning_path"
    RETRIEVAL_FAILURE = "retrieval_failure"
    VERSION_BOUNDARY_CONFUSION = "version_boundary_confusion"
    EXCEPTION_HANDLING_FAILURE = "exception_handling_failure"


class PreferenceSource(str, Enum):
    SUCCESS_VS_FAILURE = "success_vs_failure"
    CURRENT_VS_STALE = "current_vs_stale"
    GROUNDED_VS_UNSUPPORTED = "grounded_vs_unsupported"
    SAFE_VS_UNSAFE = "safe_vs_unsafe"
    EFFICIENT_VS_VERBOSE = "efficient_vs_verbose"
    TEACHER_VS_STUDENT = "teacher_vs_student"
    MANUAL = "manual"
