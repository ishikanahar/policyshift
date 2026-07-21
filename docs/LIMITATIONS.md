# Limitations

- Synthetic policies and cases; not real enterprise or regulatory documents.
- Phase 1 includes an oracle resolver, not learned agents.
- Phase 2 smoke agents are heuristic tool-users (`heuristic-baseline` / `heuristic-rag`), not trained LLM checkpoints. They exist so CPU smoke can produce real traces and retrieval metrics without requiring a GPU.
- Phase 3 distilled smoke student replays verifier-accepted teacher trajectories; SFT smoke trains a tiny CPU adapter (torch or numpy). Perfect distilled success on covered cases is expected under replay and is not a claim of Qwen-scale LoRA quality. Full GPU path is documented in `configs/sft/full_gpu.yaml`.
- Phase 4 DPO smoke student replays preference-chosen (oracle) trajectories; DPO smoke trains a tiny preference adapter. Perfect DPO success on covered cases is expected under replay and is not a claim of TRL/Qwen DPO quality. Full GPU path is documented in `configs/dpo/full_gpu.yaml`.
- Optional Sentence Transformers / FAISS / HF adapters need extras; defaults use hashing embeddings + NumPy.
- Small open models are the intended students for later phases; this is not frontier-scale training.
- Optional LLM judges (later) are never the sole evaluator.
- Results tables must remain empty or link to artifacts until real experiments exist.
- Transfer claims require held-out evaluations that are not yet run for trained models.
