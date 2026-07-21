# Failure Taxonomy

Every failed trajectory should be labeled with one or more categories from `FailureCategory`:

- Stale policy selected
- Correct policy retrieved but ignored
- Incorrect clause selected
- Missing evidence overlooked
- Invalid tool / invalid tool arguments
- Unsupported final answer
- Hallucinated evidence / policy
- Premature action
- Unsafe action
- Unnecessary escalation
- Excessive refusal
- Excessive tool use
- Reward hacking
- Correct result through invalid reasoning path
- Retrieval failure
- Version-boundary confusion
- Exception-handling failure

Phase 1 maps a subset via `TrajectoryVerifier.categorize_failures`. Full gallery reporting lands with evaluation artifacts in Phase 2+.
