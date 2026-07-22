"""Shared response-aware DPO formatting / tokenization for train, validate, and eval.

Never concatenates prompt+completion and keep_start-truncates the result.
Prompt and completions are tokenized separately; completion budget is reserved first.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 1536
DEFAULT_MAX_PROMPT_LENGTH = 1152
DEFAULT_MAX_COMPLETION_LENGTH = 384

_SYSTEM_RE = re.compile(
    r"^(You are a policy-aware enterprise operations agent\.)",
    re.MULTILINE,
)
_CURRENT_POLICY_RE = re.compile(
    r"(Retrieved current policies:|Current policy(?: clause)?:|Active policy:)",
    re.IGNORECASE,
)
_TOOL_SCHEMA_RE = re.compile(
    r"(Available tools:|Tool schema:|Tools:\s*\[|tools\s*=\s*\[)",
    re.IGNORECASE,
)
_EXAMPLES_RE = re.compile(
    r"\n(?:Optional examples|Examples|Few-shot examples)\s*:\s*\n",
    re.IGNORECASE,
)
_POLICY_HISTORY_RE = re.compile(
    r"\n(?:Policy history|Superseded policies|Older policies|Redundant policy history)\s*:\s*\n",
    re.IGNORECASE,
)
_VERSION_CLAUSE_RE = re.compile(
    r"(\[[^\]\n]+@[\d.]+\][^\n]*|policy(?:_id| version)?[^\n]{0,80}(?:effective|version)[^\n]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DPOBudgetConfig:
    """Production token budgets for PolicyShift DPO."""

    max_length: int = DEFAULT_MAX_LENGTH
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH
    max_completion_length: int = DEFAULT_MAX_COMPLETION_LENGTH

    def __post_init__(self) -> None:
        if self.max_prompt_length + self.max_completion_length > self.max_length:
            raise ValueError(
                "max_prompt_length + max_completion_length must be <= max_length "
                f"({self.max_prompt_length}+{self.max_completion_length}>{self.max_length})"
            )


DEFAULT_BUDGET = DPOBudgetConfig()


@dataclass
class BudgetedDPOPair:
    """One preference pair after response-aware budgeting."""

    prompt: str
    chosen: str
    rejected: str
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]
    prompt_tokens: int
    chosen_completion_tokens: int
    rejected_completion_tokens: int
    chosen_loss_tokens: int
    rejected_loss_tokens: int
    first_differing_completion_token: int | None
    difference_survived_truncation: bool
    current_policy_clause_retained: bool
    tool_schema_retained: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def prompt_chosen_ids(self) -> list[int]:
        return self.prompt_ids + self.chosen_ids

    @property
    def prompt_rejected_ids(self) -> list[int]:
        return self.prompt_ids + self.rejected_ids

    def to_conversational(self) -> dict[str, list[dict[str, str]]]:
        return to_conversational_dpo_example(self.prompt, self.chosen, self.rejected)

    def report_fields(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "chosen_completion_tokens": self.chosen_completion_tokens,
            "rejected_completion_tokens": self.rejected_completion_tokens,
            "chosen_loss_tokens": self.chosen_loss_tokens,
            "rejected_loss_tokens": self.rejected_loss_tokens,
            "first_differing_completion_token": self.first_differing_completion_token,
            "difference_survived_truncation": self.difference_survived_truncation,
            "current_policy_clause_retained": self.current_policy_clause_retained,
            "tool_schema_retained": self.tool_schema_retained,
            "warnings": list(self.warnings),
        }


def to_conversational_dpo_example(
    prompt: str,
    chosen: str,
    rejected: str,
) -> dict[str, list[dict[str, str]]]:
    """Build the conversational preference dict TRL expects for chat templates."""
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def normalize_completion(text: str) -> str:
    """Put preference-critical fields first so end-truncation keeps the signal."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if not isinstance(data, dict) or "final_resolution" not in data:
        return text
    ordered: dict[str, Any] = {
        "final_resolution": data.get("final_resolution"),
        "cited_policy_versions": data.get("cited_policy_versions", []),
    }
    if "steps" in data:
        ordered["steps"] = data["steps"]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return json.dumps(ordered, indent=2, default=str)


def format_dpo_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: str,
    rejected: str,
) -> dict[str, str]:
    """Apply TRL's preference chat template (shared train / validate path)."""
    from trl.data_utils import apply_chat_template

    example = to_conversational_dpo_example(prompt, chosen, rejected)
    return apply_chat_template(example, tokenizer)


def _tokenize_prompt_ids(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    from trl.data_utils import _tokenize

    return list(
        _tokenize(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )["input_ids"]
    )


def _tokenize_completion_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    prompt_ids: list[int] | None = None,
) -> list[int]:
    from trl.data_utils import _tokenize

    if prompt_ids is None:
        prompt_ids = _tokenize_prompt_ids(tokenizer, prompt)
    full = list(
        _tokenize(
            tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ],
        )["input_ids"]
    )
    if full[: len(prompt_ids)] != prompt_ids:
        # Fall back to suffix after longest common prefix.
        n = 0
        for a, b in zip(prompt_ids, full, strict=False):
            if a != b:
                break
            n += 1
        return full[n:]
    return full[len(prompt_ids) :]


def tokenize_dpo_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: str,
    rejected: str,
) -> dict[str, list[int]]:
    """Tokenize prompt / chosen / rejected as separate components (no concat truncation)."""
    prompt_ids = _tokenize_prompt_ids(tokenizer, prompt)
    chosen_ids = _tokenize_completion_ids(tokenizer, prompt, chosen, prompt_ids)
    rejected_ids = _tokenize_completion_ids(tokenizer, prompt, rejected, prompt_ids)
    return {
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_ids": rejected_ids,
        "prompt_chosen_ids": prompt_ids + chosen_ids,
        "prompt_rejected_ids": prompt_ids + rejected_ids,
    }


def first_diff_token_index(a: list[int], b: list[int]) -> int | None:
    """Return index of first differing token, or None if sequences are identical."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def first_diff_text_span(
    tokenizer: PreTrainedTokenizerBase,
    a_ids: list[int],
    b_ids: list[int],
    *,
    context_tokens: int = 8,
) -> dict[str, Any]:
    """Decode a short span around the first differing token for inspection."""
    idx = first_diff_token_index(a_ids, b_ids)
    if idx is None:
        return {"index": None, "chosen_span": "", "rejected_span": ""}
    start = max(0, idx - context_tokens)
    end_a = min(len(a_ids), idx + context_tokens)
    end_b = min(len(b_ids), idx + context_tokens)
    return {
        "index": idx,
        "chosen_span": tokenizer.decode(a_ids[start:end_a], skip_special_tokens=False),
        "rejected_span": tokenizer.decode(b_ids[start:end_b], skip_special_tokens=False),
    }


def _drop_marked_section(text: str, marker: re.Pattern[str]) -> str:
    match = marker.search(text)
    if not match:
        return text
    start = match.start()
    # Drop from marker to next major section or end.
    rest = text[match.end() :]
    next_hits = [
        m.start()
        for m in (
            _CURRENT_POLICY_RE.search(rest),
            _TOOL_SCHEMA_RE.search(rest),
            _EXAMPLES_RE.search(rest),
            _POLICY_HISTORY_RE.search(rest),
        )
        if m is not None
    ]
    if next_hits:
        end = match.end() + min(next_hits)
        return (text[:start] + text[end:]).strip()
    return text[:start].rstrip()


def _has_current_policy_clause(prompt: str) -> bool:
    if _CURRENT_POLICY_RE.search(prompt):
        return True
    # Bare prompts without retrieval still cite active policy in the instruction.
    return "Cite the active policy version" in prompt or bool(_VERSION_CLAUSE_RE.search(prompt))


def _has_tool_schema(prompt: str) -> bool:
    if _TOOL_SCHEMA_RE.search(prompt):
        return True
    # Default PolicyShift prompts instruct tool use without an inline JSON schema.
    return "Use tools as needed" in prompt


def _trim_older_policy_context(prompt: str) -> str:
    """Keep the first/current policy block; drop later superseded/history lines."""
    match = _CURRENT_POLICY_RE.search(prompt)
    if not match:
        # No explicit block: drop trailing history-like paragraphs.
        return _drop_marked_section(prompt, _POLICY_HISTORY_RE)

    head = prompt[: match.end()]
    body = prompt[match.end() :]
    # Keep first policy snippet paragraph; drop subsequent ones.
    chunks = re.split(r"\n(?=\[)", body)
    if len(chunks) <= 1:
        return _drop_marked_section(prompt, _POLICY_HISTORY_RE)
    kept = chunks[0] + "".join(
        c for c in chunks[1:] if "supersed" not in c.lower() and "history" not in c.lower()
    )
    # If still many chunks, keep only the first retrieved policy.
    first_only = chunks[0]
    # Prefer first-only when body is large.
    if len(body) > 800:
        kept = first_only
    return (head + kept).rstrip()


def truncate_prompt_text(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int,
) -> tuple[str, list[str]]:
    """Shrink prompt while preserving system, case, current policy, and tool schema."""
    warnings: list[str] = []
    text = prompt
    had_policy = _has_current_policy_clause(prompt)
    had_tools = _has_tool_schema(prompt)

    def _n_tokens(value: str) -> int:
        return len(_tokenize_prompt_ids(tokenizer, value))

    if _n_tokens(text) <= max_prompt_length:
        return text, warnings

    # 1) Remove optional examples.
    trimmed = _drop_marked_section(text, _EXAMPLES_RE)
    if trimmed != text:
        warnings.append("removed_optional_examples")
        text = trimmed
        if _n_tokens(text) <= max_prompt_length:
            return text, warnings

    # 2) Remove redundant policy history.
    trimmed = _drop_marked_section(text, _POLICY_HISTORY_RE)
    if trimmed != text:
        warnings.append("removed_policy_history")
        text = trimmed
        if _n_tokens(text) <= max_prompt_length:
            return text, warnings

    # 3) Truncate older irrelevant policy context before current policy context.
    trimmed = _trim_older_policy_context(text)
    if trimmed != text:
        warnings.append("trimmed_older_policy_context")
        text = trimmed
        if _n_tokens(text) <= max_prompt_length:
            return text, warnings

    # 4) Last resort: keep system + user case (+ current policy / tools if present),
    #    then right-trim remaining body while protecting markers.
    sys_m = _SYSTEM_RE.match(text.strip())
    system = sys_m.group(1) if sys_m else "You are a policy-aware enterprise operations agent."
    body = text[sys_m.end() :].lstrip() if sys_m else text

    # Protect current policy / tool schema slices by moving them after the case core.
    policy_m = _CURRENT_POLICY_RE.search(body)
    tool_m = _TOOL_SCHEMA_RE.search(body)
    protected_parts: list[str] = []
    if policy_m:
        protected_parts.append(body[policy_m.start() :])
        body = body[: policy_m.start()].rstrip()
    elif tool_m:
        protected_parts.append(body[tool_m.start() :])
        body = body[: tool_m.start()].rstrip()

    # Binary-trim the case body from the end (drop trailing redundancy).
    lo, hi = 0, len(body)
    best = body
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = "\n".join(p for p in (system, body[:mid], *protected_parts) if p).strip()
        if _n_tokens(candidate) <= max_prompt_length:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    text = best
    warnings.append("truncated_user_case_tail")
    if had_policy and not _has_current_policy_clause(text):
        warnings.append("current_policy_clause_lost")
    if had_tools and not _has_tool_schema(text):
        warnings.append("tool_schema_lost")
    return text, warnings


def _truncate_completion_tokens_from_end(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    prompt_ids: list[int],
    *,
    max_completion_length: int,
    label: str,
) -> tuple[str, list[int], list[str]]:
    """Truncate an overlong completion by dropping tokens from the end only."""
    warnings: list[str] = []
    ids = _tokenize_completion_ids(tokenizer, prompt, completion, prompt_ids)
    if len(ids) <= max_completion_length:
        return completion, ids, warnings
    warnings.append(
        f"{label}_completion_truncated_from_end:{len(ids)}->{max_completion_length}"
    )
    logger.debug(
        "DPO %s completion length %d exceeds max_completion_length=%d; truncating from end",
        label,
        len(ids),
        max_completion_length,
    )
    ids = ids[:max_completion_length]
    # Decode back to text so train/eval share the same truncated string.
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    # Re-tokenize to stay consistent with chat-template boundaries.
    ids = _tokenize_completion_ids(tokenizer, prompt, text, prompt_ids)[:max_completion_length]
    return text, ids, warnings


def budget_dpo_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: str,
    rejected: str,
    budget: DPOBudgetConfig | None = None,
) -> BudgetedDPOPair:
    """Apply production response-aware token budgeting to one preference pair."""
    cfg = budget or DEFAULT_BUDGET
    warnings: list[str] = []

    chosen_n = normalize_completion(chosen)
    rejected_n = normalize_completion(rejected)
    if chosen_n != chosen or rejected_n != rejected:
        warnings.append("normalized_completion_field_order")

    # Reserve completion space first; shrink prompt against remaining budget.
    prompt_budget = min(cfg.max_prompt_length, cfg.max_length - cfg.max_completion_length)
    prompt_t, prompt_warnings = truncate_prompt_text(
        prompt,
        tokenizer,
        max_prompt_length=prompt_budget,
    )
    warnings.extend(prompt_warnings)

    prompt_ids = _tokenize_prompt_ids(tokenizer, prompt_t)
    # If prompt still overflows (edge case), hard-trim prompt token ids from the end
    # after protected decode — should be rare after truncate_prompt_text.
    if len(prompt_ids) > prompt_budget:
        warnings.append(f"prompt_hard_trim:{len(prompt_ids)}->{prompt_budget}")
        prompt_ids = prompt_ids[:prompt_budget]
        prompt_t = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
        prompt_ids = _tokenize_prompt_ids(tokenizer, prompt_t)[:prompt_budget]

    chosen_t, chosen_ids, w_c = _truncate_completion_tokens_from_end(
        tokenizer,
        prompt_t,
        chosen_n,
        prompt_ids,
        max_completion_length=cfg.max_completion_length,
        label="chosen",
    )
    rejected_t, rejected_ids, w_r = _truncate_completion_tokens_from_end(
        tokenizer,
        prompt_t,
        rejected_n,
        prompt_ids,
        max_completion_length=cfg.max_completion_length,
        label="rejected",
    )
    warnings.extend(w_c)
    warnings.extend(w_r)

    # Enforce total max_length without stealing completion budget.
    max_prompt_for_pair = cfg.max_length - max(len(chosen_ids), len(rejected_ids), 1)
    if len(prompt_ids) > max_prompt_for_pair:
        warnings.append(f"prompt_fit_to_max_length:{len(prompt_ids)}->{max_prompt_for_pair}")
        prompt_t, extra = truncate_prompt_text(
            prompt_t,
            tokenizer,
            max_prompt_length=max_prompt_for_pair,
        )
        warnings.extend(extra)
        prompt_ids = _tokenize_prompt_ids(tokenizer, prompt_t)[:max_prompt_for_pair]
        chosen_ids = _tokenize_completion_ids(tokenizer, prompt_t, chosen_t, prompt_ids)[
            : cfg.max_completion_length
        ]
        rejected_ids = _tokenize_completion_ids(tokenizer, prompt_t, rejected_t, prompt_ids)[
            : cfg.max_completion_length
        ]

    if chosen_ids and rejected_ids and chosen_ids == rejected_ids:
        warnings.append("chosen_rejected_identical_after_completion_truncation")
        logger.debug("Chosen/rejected completions identical after truncation")

    diff_idx = first_diff_token_index(chosen_ids, rejected_ids)
    survived = chosen_ids != rejected_ids
    return BudgetedDPOPair(
        prompt=prompt_t,
        chosen=chosen_t,
        rejected=rejected_t,
        prompt_ids=prompt_ids,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
        prompt_tokens=len(prompt_ids),
        chosen_completion_tokens=len(chosen_ids),
        rejected_completion_tokens=len(rejected_ids),
        chosen_loss_tokens=len(chosen_ids),
        rejected_loss_tokens=len(rejected_ids),
        first_differing_completion_token=diff_idx,
        difference_survived_truncation=survived,
        current_policy_clause_retained=_has_current_policy_clause(prompt_t),
        tool_schema_retained=_has_tool_schema(prompt_t),
        warnings=warnings,
    )


def format_inference_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    budget: DPOBudgetConfig | None = None,
) -> dict[str, Any]:
    """Format/tokenize an eval prompt with the same prompt budget as DPO training."""
    cfg = budget or DEFAULT_BUDGET
    prompt_budget = min(cfg.max_prompt_length, cfg.max_length - cfg.max_completion_length)
    prompt_t, warnings = truncate_prompt_text(
        prompt,
        tokenizer,
        max_prompt_length=prompt_budget,
    )
    messages = [{"role": "user", "content": prompt_t}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = _tokenize_prompt_ids(tokenizer, prompt_t)
    return {
        "prompt": prompt_t,
        "formatted_prompt": text,
        "input_ids": input_ids,
        "prompt_tokens": len(input_ids),
        "current_policy_clause_retained": _has_current_policy_clause(prompt_t),
        "tool_schema_retained": _has_tool_schema(prompt_t),
        "warnings": warnings,
        "budget": asdict(cfg),
    }


def validate_budgeted_pairs(
    rows: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    budget: DPOBudgetConfig | None = None,
    *,
    n_samples: int = 10,
    min_tokenized_diff_pct: float = 95.0,
) -> dict[str, Any]:
    """Validate preference data under production response-aware budgets."""
    cfg = budget or DEFAULT_BUDGET
    total = len(rows)
    empty_fields = 0
    raw_identical = 0
    tokenized_identical = 0
    empty_loss_masks = 0
    no_diff_in_loss = 0
    survived = 0
    prompt_tokens: list[int] = []
    chosen_tokens: list[int] = []
    rejected_tokens: list[int] = []
    chosen_loss: list[int] = []
    rejected_loss: list[int] = []
    first_diffs: list[int] = []
    policy_retained = 0
    tools_retained = 0
    all_warnings: list[str] = []
    samples: list[dict[str, Any]] = []
    per_row: list[dict[str, Any]] = []

    for row in rows:
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not (
            isinstance(prompt, str)
            and prompt.strip()
            and isinstance(chosen, str)
            and chosen.strip()
            and isinstance(rejected, str)
            and rejected.strip()
        ):
            empty_fields += 1
        if chosen == rejected:
            raw_identical += 1

        budgeted = budget_dpo_pair(tokenizer, str(prompt), str(chosen), str(rejected), cfg)
        all_warnings.extend(budgeted.warnings)
        prompt_tokens.append(budgeted.prompt_tokens)
        chosen_tokens.append(budgeted.chosen_completion_tokens)
        rejected_tokens.append(budgeted.rejected_completion_tokens)
        chosen_loss.append(budgeted.chosen_loss_tokens)
        rejected_loss.append(budgeted.rejected_loss_tokens)
        if budgeted.chosen_loss_tokens == 0 or budgeted.rejected_loss_tokens == 0:
            empty_loss_masks += 1
        if not budgeted.difference_survived_truncation:
            tokenized_identical += 1
            no_diff_in_loss += 1
        else:
            survived += 1
            if budgeted.first_differing_completion_token is not None:
                first_diffs.append(budgeted.first_differing_completion_token)
        if budgeted.current_policy_clause_retained:
            policy_retained += 1
        if budgeted.tool_schema_retained:
            tools_retained += 1

        row_report = {
            "id": row.get("id"),
            "case_id": row.get("case_id"),
            **budgeted.report_fields(),
        }
        per_row.append(row_report)
        if len(samples) < n_samples:
            span = first_diff_text_span(
                tokenizer,
                budgeted.chosen_ids,
                budgeted.rejected_ids,
            )
            samples.append(
                {
                    **row_report,
                    "chosen_span": span["chosen_span"],
                    "rejected_span": span["rejected_span"],
                    "prompt_preview": (budgeted.prompt[:160] + "…")
                    if len(budgeted.prompt) > 160
                    else budgeted.prompt,
                }
            )

    from statistics import mean

    tokenized_diff_pct = (100.0 * survived / total) if total else 0.0
    raw_diff_pct = (100.0 * (total - raw_identical) / total) if total else 0.0
    # 100% raw diffs; >=95% tokenized diffs; no empty loss masks; surviving
    # pairs must include at least one differing completion token in the loss mask.
    has_diff_in_loss = all(
        (not r["difference_survived_truncation"])
        or (
            r["first_differing_completion_token"] is not None
            and r["chosen_loss_tokens"] > 0
            and r["rejected_loss_tokens"] > 0
        )
        for r in per_row
    )
    passed = (
        total > 0
        and empty_fields == 0
        and raw_identical == 0
        and tokenized_diff_pct >= min_tokenized_diff_pct
        and empty_loss_masks == 0
        and has_diff_in_loss
    )

    return {
        "budget": asdict(cfg),
        "total_pairs": total,
        "empty_field_pairs": empty_fields,
        "raw_identical_pairs": raw_identical,
        "raw_difference_pct": raw_diff_pct,
        "identical_pairs_after_tokenization": tokenized_identical,
        "percentage_retaining_chosen_rejected_difference": tokenized_diff_pct,
        "empty_loss_mask_pairs": empty_loss_masks,
        "pairs_without_differing_loss_token": no_diff_in_loss,
        "prompt_tokens": {
            "mean": mean(prompt_tokens) if prompt_tokens else 0.0,
            "max": max(prompt_tokens) if prompt_tokens else 0,
            "min": min(prompt_tokens) if prompt_tokens else 0,
        },
        "chosen_completion_tokens": {
            "mean": mean(chosen_tokens) if chosen_tokens else 0.0,
            "max": max(chosen_tokens) if chosen_tokens else 0,
            "min": min(chosen_tokens) if chosen_tokens else 0,
        },
        "rejected_completion_tokens": {
            "mean": mean(rejected_tokens) if rejected_tokens else 0.0,
            "max": max(rejected_tokens) if rejected_tokens else 0,
            "min": min(rejected_tokens) if rejected_tokens else 0,
        },
        "chosen_loss_tokens": {
            "mean": mean(chosen_loss) if chosen_loss else 0.0,
            "max": max(chosen_loss) if chosen_loss else 0,
            "min": min(chosen_loss) if chosen_loss else 0,
        },
        "rejected_loss_tokens": {
            "mean": mean(rejected_loss) if rejected_loss else 0.0,
            "max": max(rejected_loss) if rejected_loss else 0,
            "min": min(rejected_loss) if rejected_loss else 0,
        },
        "first_differing_completion_token": {
            "count_with_diff": len(first_diffs),
            "mean_index": mean(first_diffs) if first_diffs else None,
            "min_index": min(first_diffs) if first_diffs else None,
            "max_index": max(first_diffs) if first_diffs else None,
        },
        "difference_survived_truncation": survived,
        "difference_survived_truncation_pct": tokenized_diff_pct,
        "current_policy_clause_retained": policy_retained,
        "tool_schema_retained": tools_retained,
        "completion_truncation_warning_count": sum(
            1 for w in all_warnings if "completion_truncated_from_end" in w
        ),
        "warnings": sorted(set(all_warnings)),
        "min_tokenized_diff_pct_required": min_tokenized_diff_pct,
        "passed": passed,
        "sample_pairs": samples,
        "rows": per_row,
    }
