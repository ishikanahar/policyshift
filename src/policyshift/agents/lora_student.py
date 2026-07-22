"""LoRA student agent: generate structured resolutions from PEFT adapters.

Used for held-out policy-shift evaluation of real SFT/DPO checkpoints.
Not a full multi-step tool loop — prompts the adapter and verifies the parsed decision.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policyshift.environment.policy_store import PolicyStore
from policyshift.rewards.scorer import RewardScorer
from policyshift.retrieval.retriever import PolicyRetriever, RetrievalMode
from policyshift.schemas import AgentAction, AgentTrajectory, CaseEvent, TrainingMethod
from policyshift.training.dpo_format import DEFAULT_BUDGET, format_inference_prompt
from policyshift.training.sft_data import case_to_prompt
from policyshift.utils.hashing import sha256_text
from policyshift.verification.verifiers import TrajectoryVerifier


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _parse_generation(text: str) -> dict[str, Any]:
    """Extract final_resolution / citations from model text (JSON preferred)."""
    blob = text.strip()
    # Prefer last JSON object in the string
    candidates = re.findall(r"\{[\s\S]*\}", blob)
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                final = data.get("final_resolution") or data.get("final_answer") or data.get("decision")
                cites = data.get("cited_policy_versions") or data.get("policy_citations") or []
                if final:
                    return {
                        "final_answer": str(final),
                        "cited_policy_versions": [str(c) for c in cites] if isinstance(cites, list) else [str(cites)],
                        "raw_json": data,
                    }
        except Exception:
            continue
    # Heuristic line scrape
    final = None
    for line in blob.splitlines():
        if "final_resolution" in line.lower() or "decision" in line.lower():
            final = line.split(":", 1)[-1].strip().strip('",')
            break
    if not final:
        # last non-empty short line
        lines = [ln.strip() for ln in blob.splitlines() if ln.strip()]
        final = lines[-1][:200] if lines else "abstain"
    return {"final_answer": final, "cited_policy_versions": [], "raw_json": None}


class LoRAStudentAgent:
    """PEFT LoRA adapter over an instruct base model."""

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: str | Path,
        model_id: str,
        training_method: TrainingMethod,
        policy_store: PolicyStore | None = None,
        use_retrieval: bool = False,
        retrieval_mode: RetrievalMode = "date_filtered_rerank",
        max_new_tokens: int = 96,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.model_id = model_id
        self.training_method = training_method
        self.policy_store = policy_store or PolicyStore.from_builtin()
        self.use_retrieval = use_retrieval
        self.retrieval_mode = retrieval_mode
        self.retriever = PolicyRetriever.from_store(self.policy_store) if use_retrieval else None
        self.max_new_tokens = max_new_tokens
        self.device = device or _pick_device()
        self.verifier = TrajectoryVerifier(self.policy_store)
        self.scorer = RewardScorer(self.policy_store)
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(self.base_model, torch_dtype=dtype)
        self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
        self._model.to(self.device)
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        import torch

        if self._model is None or self._tokenizer is None:
            self.load()
        # Same prompt budgeting / chat template as DPO train + validate_dpo_data.
        packed = format_inference_prompt(self._tokenizer, prompt, budget=DEFAULT_BUDGET)
        input_ids = torch.tensor([packed["input_ids"]], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(self.max_new_tokens, DEFAULT_BUDGET.max_completion_length),
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def resolve(self, case: CaseEvent) -> AgentTrajectory:
        prompt = case_to_prompt(case)
        retrieval_meta: dict[str, Any] = {}
        citations: list[str] = []
        if self.use_retrieval and self.retriever is not None:
            result = self.retriever.retrieve(case, mode=self.retrieval_mode, top_k=3)
            snippets = []
            for hit in result.hits[:3]:
                doc = hit.document
                key = f"{doc.policy_id}@{doc.version}"
                citations.append(key)
                snippets.append(f"[{key}] {doc.text[:400]}")
            retrieval_meta = {
                "mode": self.retrieval_mode,
                "n_hits": len(result.hits),
                "keys": citations,
            }
            prompt = (
                prompt
                + "\n\nRetrieved current policies:\n"
                + "\n".join(snippets)
                + "\n\nRespond with JSON including final_resolution and cited_policy_versions."
            )
        else:
            prompt = prompt + "\n\nRespond with JSON including final_resolution and cited_policy_versions."

        raw = self._generate(prompt)
        parsed = _parse_generation(raw)
        cites = parsed["cited_policy_versions"] or citations[:1]
        final = parsed["final_answer"]

        actions = [
            AgentAction(
                step_number=1,
                thought_summary="LoRA student generation" + (" + RAG" if self.use_retrieval else ""),
                tool_name="lora_generate",
                arguments={"adapter": str(self.adapter_path), "use_retrieval": self.use_retrieval},
                tool_output={"ok": True, "text_preview": raw[:500]},
                policy_citations=cites,
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ]
        traj = AgentTrajectory(
            trajectory_id=f"traj-lora-{sha256_text(case.case_id + self.model_id)[:12]}",
            case_id=case.case_id,
            model_id=self.model_id,
            training_method=self.training_method,
            actions=actions,
            final_answer=final,
            cited_policy_versions=cites,
            metadata={
                "lora_adapter": str(self.adapter_path),
                "retrieval": retrieval_meta,
                "generation_preview": raw[:1000],
                "parsed": parsed.get("raw_json"),
            },
        )
        results = self.verifier.verify(case, traj)
        traj.verifier_results = results
        traj.failure_categories = self.verifier.categorize_failures(case, traj, results)
        breakdown = self.scorer.score(case, traj)
        traj.reward_components = breakdown
        traj.total_reward = breakdown.total
        traj.success = self.verifier.success(results)
        return traj
