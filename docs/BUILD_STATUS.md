# PolicyShift Build Status

Last updated: 2026-07-21 (Phases 0–9 smoke complete)

## Phase checklist

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Repo audit + docs plan | `[x]` | |
| 1. Functional environment | `[x]` | |
| 2. Baseline + retrieval | `[x]` | |
| 3. SFT + distillation | `[x]` | |
| 4. Preference / DPO | `[x]` | |
| 5. Continual learning + replay | `[x]` | Accuracy matrices + forgetting |
| 6. TeacherBudget | `[x]` | 80% teacher-call reduction @ matched smoke success |
| 7. Verifier RL smoke | `[x]` | Tiny RLOO-style + hacking diagnostics |
| 8. Visual application | `[x]` | FastAPI + static artifact playback |
| 9. Report + portfolio export | `[x]` | `portfolio_export/`, resume bullets, technical report |

## Headline measured smoke (not fabricated)

| Metric | Value |
| --- | --- |
| Baseline → RAG task success | 0.58 → **0.75** |
| Retrieval stale@5 naive → date-filtered | 0.45 → **0.00** |
| Preference pairs (Phase 4) | **120** |
| TeacherBudget call reduction (combined vs label-all) | **80%** (task_success 0.75 vs 0.75) |
| Tests | **74 passed** |

Distillation / DPO / RL smoke students may **replay** verifier-accepted teachers — label clearly on resume/website; not Qwen-scale LoRA/TRL/GRPO claims.

## Validate

```bash
pytest tests/unit tests/integration -q
python scripts/export_portfolio.py
python scripts/serve_playback.py   # http://127.0.0.1:8000
```

Resume copy: `portfolio_export/RESUME_BULLETS.md`  
Website JSON: `portfolio_export/website_card.json`  
Report: `docs/TECHNICAL_REPORT.md`  
**Public site:** https://ishikanahar.github.io/policyshift/  
Optional GPU: `docs/FULL_GPU.md`, `notebooks/PolicyShift_Full_GPU.ipynb`

