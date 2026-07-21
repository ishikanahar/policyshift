# PolicyShift Research Design

## Motivation

Enterprise AI agents operate under policies that change: receiving procedures, lab access rules, data-handling constraints, approval thresholds, and escalation criteria. When a policy version is superseded, the agent must apply the version that was effective at the event time, adapt after updates, retain still-valid procedures, and avoid stale or unsupported actions. Retrieval alone does not guarantee correct following. Continual post-training can help, but risks forgetting, reward hacking, and expensive teacher labeling.

PolicyShift builds a synthetic, fully executable benchmark to study these tradeoffs at research-accessible scale. It does not claim to reproduce frontier-scale training.

## Primary question

How do supervised fine-tuning (SFT), teacher–student distillation, Direct Preference Optimization (DPO), and verifier-guided reinforcement learning differ in teaching a small tool-using language model to follow changing enterprise policies?

## Secondary questions

1. How often does each model follow an outdated policy after a new policy becomes active?
2. Does adapting to a new policy cause forgetting of unchanged procedures?
3. Can replay preserve earlier capabilities without preventing policy updates?
4. Can DPO teach preference for current, grounded trajectories over stale or unsupported ones?
5. Does verifier-guided RL improve task success, or introduce reward hacking?
6. Can active example selection reduce teacher queries without reducing student performance?
7. Which method best trades off task success, policy freshness, groundedness, safety, efficiency, and compute/teacher cost?
8. Do gains transfer to held-out tools, policy formats, and domains?

## Hypotheses

| ID | Hypothesis |
| --- | --- |
| H1 | RAG alone retrieves current policies but does not always cause the agent to follow them correctly. |
| H2 | Sequential SFT improves newest-policy performance but increases forgetting on older, still-valid procedures. |
| H3 | Replay-based SFT reduces forgetting but may increase stale-policy confusion if replay is not version-aware. |
| H4 | Teacher distillation improves multi-step tool use but may copy undesirable teacher behaviors (verbosity, unnecessary tools, overconfidence). |
| H5 | DPO on current- vs stale-policy trajectory pairs reduces stale-policy errors more than SFT alone. |
| H6 | Verifier-guided RL improves executable task success, but poorly balanced rewards may encourage shortcuts or excessive refusals. |
| H7 | Failure-aware and diversity-aware teacher selection matches near-full labeling quality under a fixed teacher budget. |

## Synthetic domains

All content is independently authored and labeled synthetic.

### Domain A — Scientific Materials Receiving (`materials`)

Temperature-sensitive biological material, missing supplier documentation, damaged packaging, lot-number mismatch, expired certificates, quantity mismatch, storage location, quarantine and escalation.

### Domain B — Laboratory Access and Equipment (`laboratory`)

Training requirements, instrument reservation, calibration validity, maintenance windows, access-level restrictions, after-hours use, failed QC checks, supervisor approval.

### Domain C — Enterprise Data and AI Use (`ai_governance`)

Sensitive-data handling, approved models, external API restrictions, human approval for high-impact actions, output verification, retention rules, tool permissions, incident escalation.

Each domain has sequential versions **1.0 → 1.1 → 2.0** with effective dates, supersession links, scope, definitions, required evidence, permitted/prohibited actions, exceptions, escalation criteria, examples, and change logs. Versions include genuine changes, unchanged clauses, added/removed exceptions, stricter and relaxed requirements, near-boundary events, and non-conflicting clauses that look conflicting.

## Experimental conditions

| ID | Condition | Description |
| --- | --- | --- |
| C0 | Base | No post-training |
| C1 | RAG-only | Base model + version-aware retrieval + tools |
| C2 | Sequential SFT | Chronological fine-tuning without replay |
| C3 | Replay SFT | Sequential SFT with version-aware replay |
| C4 | Distillation | Student trained on verifier-accepted teacher trajectories |
| C5 | DPO | Preference optimization on trajectory pairs |
| C6 | Verifier RL | Optional GRPO/RLOO with executable rewards |

Default student: Qwen2.5-0.5B-Instruct (or similar). Optional larger student when GPU allows. Teachers: configurable API, larger local model, or pre-generated trajectories for reproducibility.

## Environment and evaluation philosophy

Agents interact with a deterministic Python tool environment. Correctness is primarily decided by executable verifiers (active policy version, citations, tool validity, evidence coverage, prohibited actions, final resolution). An optional LLM judge may be compared to verifiers but is never the sole evaluator.

### Core metrics

Task success; active-policy selection accuracy; stale-policy error rate; citation precision/recall; tool selection and argument validity; evidence coverage; groundedness; unsupported-claim and unsafe-action rates; escalation and over-refusal rates; steps, tokens, latency, reward; teacher cost; training time; peak GPU memory; inference throughput.

### Continual-learning metrics

Per-version accuracy matrix; backward transfer; average forgetting; performance on unchanged / newly updated / superseded policies.

### Failure taxonomy

Stale policy selected; retrieved-but-ignored; incorrect clause; missing evidence overlooked; invalid tool/args; unsupported final answer; hallucinated evidence/policy; premature or unsafe action; unnecessary escalation; excessive refusal or tool use; reward hacking; correct result via invalid path; retrieval failure; version-boundary confusion; exception-handling failure.

## Reward design (summary)

Decomposable, configurable weights (see `docs/REWARD_DESIGN.md` when authored). Positive credit for correct resolution, active version, citations, tools, evidence, escalation, grounding. Penalties for stale/expired policy use, hallucinations, prohibited/unsafe actions, premature finals, over-refusal, inefficient tools. Ablations: outcome-only; +freshness; +grounding; +efficiency; safety-heavy; balanced full.

## TeacherBudget experiment

Under a fixed teacher-call budget, compare: label all; random; student uncertainty; student failure; embedding diversity; policy-change coverage; combined uncertainty/failure/diversity. Measure student success, stale-policy errors, teacher calls/tokens/cost, and coverage.

## Capability regression

Fixed suite covering instruction following, basic reasoning, JSON generation, tool-call formatting, reading comprehension, safe refusal, short factual tasks, out-of-domain prompts. Goal: detect post-training regressions, not claim broad superiority.

## Statistical reporting

When compute permits: ≥3 seeds; mean, std, bootstrap CIs; paired tests when justified. Never invent scores. Until experiments complete, README/report use placeholders only.

## Ethical and claim boundaries

- Synthetic data only; no private or proprietary content.
- No claim of frontier-scale, SOTA, novel algorithm, or production deployment without repository evidence.
- Resume language stays templated until measured results exist (`docs/resume_after_results.md`).

## Success criteria for the research artifact

A recruiter can grasp the problem in 30 seconds; an ML engineer can inspect methods, artifacts, and limitations in 30 minutes; every displayed number links to a real run or is explicitly marked non-experimental.
