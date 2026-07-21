# PolicyShift Implementation Plan

## Project identity

**PolicyShift** studies continual post-training for tool-using language models that must operate under evolving enterprise policies. The system is synthetic end-to-end: policies, cases, tools, rewards, and trajectories are independently authored for research reproducibility. No proprietary employer data, private regulations, or confidential prompts are used.

**Primary research question:** How do supervised fine-tuning, teacher–student distillation, DPO, and verifier-guided RL differ in teaching a small language model to follow changing enterprise policies while minimizing stale-policy errors, forgetting, unsafe actions, and teacher cost?

## Repository audit (2026-07-20)

| Finding | Detail |
| --- | --- |
| Prior code | Empty workspace; no reusable training, evaluation, serving, or continual-learning utilities |
| User files | None present beyond the empty directory |
| Decision | Greenfield monorepo under `src/policyshift/` with config-driven experiments |

## Design principles

1. **Executable truth.** Policy correctness is judged by deterministic verifiers and tools, not LLM judges alone.
2. **Artifact integrity.** Every metric, chart, and UI trace must map to a stored experiment artifact or be labeled demonstration-only.
3. **Smoke-first.** Full GPU runs are optional; every training path has a CPU/tiny smoke configuration.
4. **Model-agnostic.** Defaults are small open models (e.g., Qwen2.5-0.5B-Instruct); adapters hide provider details.
5. **Phase gates.** Do not start the next phase until the current acceptance criteria pass.

## Phase plan

### Phase 1 — Functional environment

**Build:** Pydantic schemas; three synthetic domains × three policy versions; deterministic tool environment; case generator; verifiers; decomposable rewards; unit tests; CLI demo; ≥100 deterministic cases.

**Acceptance:** Cases resolve deterministically; effective-date / supersession logic works; stale policies detectable; tests pass; CLI demo runs. **Done.**

**Validate:** `make phase1` (generate policies/cases + pytest Phase 1 suite).

### Phase 2 — Baseline agent and retrieval

Local hashing / optional Sentence-Transformers + FAISS retriever; date-filtered and metadata-aware variants; base and RAG-only agents; evaluation harness; failure taxonomy; artifact export.

**Acceptance:** Base and RAG conditions produce real metrics/traces; retrieval metrics correct. **Done 2026-07-21.**

**Validate:** `make evaluate-phase2` and `pytest -m phase2`.

### Phase 3 — SFT and distillation

Teacher trajectory generation (API / local / pre-generated file); verifier filtering; LoRA SFT; smoke training; documented GPU command.

**Acceptance:** Smoke SFT completes; checkpoint loads; base vs SFT comparison artifacts exported.

**Validate:** `make evaluate-phase3` and `pytest -m phase3`.

### Phase 4 — Preference optimization

Preference-pair construction from verified trajectories; DPO via TRL; preference explorer artifacts.

**Acceptance:** Inspectable pairs; DPO smoke training; evaluation artifacts.

**Validate:** `make evaluate-phase4` and `pytest -m phase4`.

### Phase 5 — Continual learning

Sequential policy updates; version-aware replay strategies; forgetting / backward-transfer metrics.

**Acceptance:** Accuracy matrix, forgetting, stale-policy errors reported from real runs.

### Phase 6 — TeacherBudget

Active selection under fixed teacher budgets; cost accounting; comparison report.

### Phase 7 — Optional RL (after stability)

GRPO or RLOO with verifier rewards; smoke config; reward-hacking diagnostics; no fabricated RL results.

### Phase 8 — Visual application

Next.js + FastAPI; artifact playback first; live inference only after playback is reliable.

### Phase 9 — Report and portfolio export

Technical report, figures from real artifacts, `portfolio_export/`, resume language with measured values only.

## Dependency strategy

| Layer | Default | Optional |
| --- | --- | --- |
| Core (Phase 1) | pydantic, pyyaml, numpy, rich, typer | — |
| Retrieval (Phase 2) | sentence-transformers, faiss-cpu | Cohere Embed/Rerank |
| Training (Phase 3+) | transformers, peft, trl, datasets, accelerate | bitsandbytes, vLLM |
| API / Web (Phase 8) | fastapi, uvicorn | Next.js frontend |

Optional dependencies degrade gracefully; smoke paths never require API keys or GPUs.

## Data and safety constraints

- Synthetic only; labeled as such in `docs/DATA_CARD.md` (Phase 1 draft, expanded later).
- No unrestricted chain-of-thought storage; trajectories store short structured decision summaries.
- Secrets via environment variables; `.env.example` only.
- No committed model weights or large generated corpora; `.gitignore` enforces this.

## Configuration layout

```
configs/
  smoke/          # tiny CPU paths for every method
  retrieval/
  sft/ replay/ distillation/ dpo/ rl/
  evaluation/
```

Each config is YAML with seed, paths, model ids, and hardware notes.

## Validation gates (per phase)

1. Unit / integration tests green for that phase’s scope.
2. Deterministic seed replay produces identical artifacts for generation code.
3. `docs/BUILD_STATUS.md` updated with pass/fail evidence (commands + exit codes).
4. No fabricated metrics in README, report, or UI.

## Immediate Phase 1 work breakdown

1. Schemas (`PolicyDocument`, `PolicyClause`, `CaseEvent`, `AgentAction`, `AgentTrajectory`, `PreferencePair`) + JSON Schema export.
2. Author policies for materials, laboratory, ai_governance (v1.0, v1.1, v2.0) with real change logs.
3. Tool registry with typed args, JSON Schema, permissions, failure modes, audit log.
4. Environment that enforces active-policy constraints and immutable evidence.
5. Seeded case generator with difficulty / adversarial templates and leakage-safe splits.
6. Verifiers + configurable reward components with per-trajectory breakdowns.
7. Scripts: `generate_policies.py`, `generate_cases.py`; CLI demo; Phase 1 tests.
8. Update `BUILD_STATUS.md` after test run.

## Out of scope for Phase 1

Model adapters, training, retrieval embeddings, FastAPI/Next.js UI, teacher APIs, fabricated results tables.
