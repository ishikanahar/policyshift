# Website Integration

PolicyShift will later connect to a personal portfolio site. Until Phase 8–9, treat this as a planning document.

## Embeddable components

- Research overview blurb + system diagram
- Policy timeline (artifact-backed)
- Agent Arena playback from saved traces
- Preference pair explorer
- Results tables linked to `artifacts/experiments/`

## Artifact-only mode

Deploy the FastAPI app (Phase 8) with `POLICYSHIFT_ARTIFACT_DIR` pointing at exported traces. No GPU required for playback.

## Static export (`portfolio_export/`)

Generated later from real artifacts:

- `results-summary.json`, `model-comparison.json`, `sample-trajectories.json`, etc.
- Demo video instructions; do not commit fabricated videos as results.

## Environment variables

See `.env.example`. Never hardcode credentials.
