"""Optional Hugging Face instruct model adapter (GPU path).

Smoke evaluation does not require this. Full command documented in configs.
"""

from __future__ import annotations

from typing import Any


class HFInstructAdapter:
    """Thin wrapper around a local HF chat model for later Phase 2+/3 runs."""

    def __init__(self, model_id: str, *, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "transformers is required for HFInstructAdapter. "
                "Install with: pip install 'policyshift[training]'"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device:
            self._model.to(self.device)

    def generate(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() before generate(), or use heuristic agents for smoke.")
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def smoke_status(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "transformers_installed": self.available(),
            "loaded": self._model is not None,
            "note": "Phase 2 smoke uses heuristic-baseline / heuristic-rag agents.",
        }
