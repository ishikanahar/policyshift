# Synthetic Enterprise Data and AI Use Policy

- policy_id: `POL-AI-USE`
- version: `1.0`
- effective_at: `2024-01-01T00:00:00+00:00`

| clause_id | rule_type | text |
| --- | --- | --- |
| AI-1.0-SENS | prohibition | Sensitive data must not be sent to external APIs. |
| AI-1.0-MODEL | requirement | Only approved models may be used for production workflows. |
| AI-1.0-HUMAN | escalation | High-impact actions require human approval before execution. |
| AI-1.0-VERIFY | requirement | Model outputs used for operational decisions must be verified against source records. |
| AI-1.0-RETAIN | requirement | Prompt and output logs for sensitive workflows are retained for 30 days. |
| AI-1.0-INCIDENT | escalation | Suspected policy violations must be reported as AI incidents. |
