# PolicyShift Build Status

Last updated: 2026-07-20 (Phase 1 acceptance re-audit)

## Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` complete and validated
- `[!]` blocked / needs follow-up

## Phase checklist

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Repo audit + docs plan | `[x]` | Empty repo; plan + research design written |
| 1. Functional environment | `[x]` | Re-audited against master Phase 1 criteria; hardened; **40 tests passed** |
| 2. Baseline agent + retrieval | `[ ]` | Next |
| 3. SFT + distillation | `[ ]` | Smoke + GPU docs |
| 4. Preference / DPO | `[ ]` | |
| 5. Continual learning + replay | `[ ]` | |
| 6. TeacherBudget | `[ ]` | |
| 7. Optional RL | `[ ]` | After stability |
| 8. Visual application | `[ ]` | Artifact playback first |
| 9. Report + portfolio export | `[ ]` | Measured results only |

## Phase 1 master-spec acceptance matrix

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Schemas (`PolicyDocument`, `PolicyClause`, `CaseEvent`, `AgentAction`, `AgentTrajectory`, `PreferencePair`) + JSON Schema | `[x]` | `schema_version=1.0.0` on models; `policies/schemas/*.json` |
| 2 | 3 domains × 3 versions with required policy fields | `[x]` | 9 JSON policies under `policies/{materials,laboratory,ai_governance}/` |
| 3 | Genuine changes, unchanged clauses, new/removed exceptions, stricter/relaxed, looks-conflicting, date bounds, near-boundary events | `[x]` | Policy changelogs + `looks_conflicting` clause; boundary templates |
| 4 | 15 required tools with typed args, JSON Schema, permissions, failure conditions, audit logs, tests | `[x]` | Registry + env handlers; permissions enforced; audit on tool calls; +2 held-out tools |
| 5 | Env rejects unknown tools, bad args, prohibited actions, expired-policy actions, nonexistent citations, unsupported conclusions, evidence mutation | `[x]` | Hardened `PolicyShiftEnvironment` + unit tests |
| 6 | Case axes: missing / conflicting / irrelevant / adversarial stale docs / ambiguous / escalation / safe refusal / multi-step | `[x]` | Templates + tags; stale docs as evidence; feature coverage test |
| 7 | Difficulties easy/medium/hard/adversarial | `[x]` | Seed 42: 12 / 57 / 39 / 12 |
| 8 | Template + combination leakage-safe splits; held-out format, tool, policy updates | `[x]` | `check_split_leakage`; markdown held-out formats; `heldout_*` tools gated by metadata |
| 9 | ≥100 cases; all resolve deterministically | `[x]` | 120 cases; oracle **120/120** |
| 10 | Verifiers + decomposable configurable rewards | `[x]` | `TrajectoryVerifier`, `RewardScorer`, ablations |
| 11 | Unit/integration tests + CLI demo | `[x]` | `pytest -m phase1` → 40 passed; `policyshift demo` |
| 12 | Stale policies detectable; versions affect active policy | `[x]` | `PolicyStore.is_stale` / `resolve_active` + tests |

## Phase 1 detailed checklist

- [x] `docs/IMPLEMENTATION_PLAN.md` / `docs/RESEARCH_DESIGN.md`
- [x] Monorepo scaffold
- [x] Versioned Pydantic schemas (`schema_version`) + JSON Schema export
- [x] 9 synthetic policies with real version diffs + held-out markdown serializations
- [x] Deterministic tool environment (15 core + 2 held-out)
- [x] Permission enforcement, full audit logging, unsupported-resolution whitelist
- [x] Expired/stale-policy action rejection; immutable evidence rejection
- [x] Case generator with conflict / irrelevant / ambiguous / safe-refusal / stale-doc coverage
- [x] Leakage checks on templates **and** combinations
- [x] Held-out policy format (markdown table) + held-out tools
- [x] Verifiers + rewards (non-vacuous ablation test)
- [x] Deterministic oracle trajectory IDs
- [x] Phase 1 tests green

## Validation log

| Date | Command | Result |
| --- | --- | --- |
| 2026-07-20 | repository inspection | Empty workspace; no prior code |
| 2026-07-20 | initial Phase 1 suite | 26 passed; oracle 120/120 |
| 2026-07-20 | Phase 1 acceptance re-audit + hardening | Gaps fixed (env, cases, held-outs, permissions, tests) |
| 2026-07-20 | `python scripts/generate_policies.py` | 9 JSON policies (+ 3 held-out markdown exports) |
| 2026-07-20 | `python scripts/generate_cases.py --seed 42 --n-cases 120` | 120 cases; `leakage_ok=True`; 27 templates |
| 2026-07-20 | `pytest tests/unit tests/integration -m phase1 -v` | **40 passed** in 0.34s |
| 2026-07-20 | `policyshift resolve-all --seed 42 --n-cases 120` | Oracle success **120/120** |

## Fixes applied in re-audit (not Phase 2)

1. Enforced tool permissions; audit-log all tool calls
2. Whitelisted finalize resolutions; reject unsupported conclusions
3. Reject actions based on selected/cited stale policies
4. Explicit immutable-evidence mutation rejection (`alter_evidence`)
5. Case templates for conflicting evidence, irrelevant evidence, ambiguous wording, safe refusal
6. Real stale/misleading documents in evidence payloads
7. Held-out markdown policy format files + case evidence; real held-out tools gated by case metadata
8. Combination-level split leakage check
9. `schema_version` on data models; deterministic oracle trajectory IDs
10. Expanded Phase 1 tests from 26 → 40

## Intentionally deferred (later phases)

- Semantic retrieval / FAISS / Sentence Transformers (Phase 2)
- Base/RAG LLM agents and evaluation artifact export (Phase 2)
- SFT, distillation, DPO, RL, TeacherBudget (Phases 3–7)
- FastAPI / Next.js UI (Phase 8)
- Experimental metrics in README/report (only after real runs)
