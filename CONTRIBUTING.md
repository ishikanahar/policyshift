# Contributing to PolicyShift

Thanks for interest in improving PolicyShift.

## Principles

1. Do not fabricate metrics, charts, or model outputs.
2. Keep data synthetic or public; never add proprietary or private content.
3. Prefer deterministic tests and config-driven experiments.
4. Expand phase by phase; do not skip acceptance gates.

## Development setup

```bash
python -m pip install -e ".[dev]"
make phase1
```

## Pull requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Update `docs/BUILD_STATUS.md` when completing a phase gate.
- Do not commit secrets, model weights, or bulk generated corpora.
