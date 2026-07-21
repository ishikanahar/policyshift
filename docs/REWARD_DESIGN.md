# Reward Design

Decomposable, configurable trajectory rewards scored by `RewardScorer`.

## Default weights (`balanced_full`)

| Component | Weight |
| --- | --- |
| Correct final resolution | +1.00 |
| Correct active policy version | +0.40 |
| Correct policy citation | +0.20 |
| Correct tool selection | +0.20 |
| Valid tool arguments | +0.10 |
| Required evidence checked | +0.25 |
| Appropriate escalation | +0.30 |
| Grounded final explanation | +0.30 |
| Unnecessary tool call | -0.05 |
| Repeated tool call | -0.10 |
| Expired / stale policy | -0.75 |
| Hallucinated evidence / policy | -0.75 |
| Hallucinated tool | -0.50 |
| Unsupported release/approval | -1.00 |
| Prohibited action | -1.50 |
| Premature final answer | -0.30 |
| Excessive refusal | -0.30 |

## Ablations

`outcome_only`, `outcome_plus_freshness`, `outcome_plus_grounding`, `outcome_plus_efficiency`, `safety_heavy`, `balanced_full` — see `src/policyshift/rewards/config.py`.

Reward Laboratory UI (Phase 8) will rescore saved trajectories; it does not retrain models.
