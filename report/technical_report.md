# PolicyShift Technical Report

> Draft scaffold. Sections will be completed after real experiments. Do not invent results.

## Abstract

TBD after measured results.

## Motivation

Enterprise AI agents must follow evolving policies. Retrieval alone may surface the right document without ensuring correct tool use. PolicyShift studies post-training methods for small tool-using models under sequential synthetic policy versions.

## Related work

TBD (continual learning, tool-use agents, distillation, DPO, RL from verifiable rewards).

## Environment design

See Phase 1 implementation: versioned policies, deterministic tools, case generator, verifiers, rewards.

## Dataset construction

Synthetic domains and versions; leakage-safe template splits. Details in `docs/DATA_CARD.md`.

## Models

TBD after training runs.

## Post-training methods

Planned: base, RAG-only, sequential SFT, replay SFT, distillation, DPO, optional verifier RL.

## Reward design

See `docs/REWARD_DESIGN.md`.

## Evaluation

Executable metrics preferred over LLM judges. Full harness in Phase 2+.

## Results

No results yet.

## Ablations

TBD.

## Failure analysis

TBD.

## Compute and cost

TBD.

## Limitations

See `docs/LIMITATIONS.md`.

## Future work

TBD.
