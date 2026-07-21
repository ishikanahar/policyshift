# Synthetic Enterprise Data and AI Use Policy

- policy_id: `POL-AI-USE`
- version: `1.1`
- effective_at: `2024-07-01T00:00:00+00:00`

| clause_id | rule_type | text |
| --- | --- | --- |
| AI-1.1-SENS | prohibition | Sensitive data must not be sent to external APIs. |
| AI-1.1-MODEL | requirement | Only approved models may be used for production workflows. |
| AI-1.1-TOOLS | requirement | Agents may invoke only tools listed on the user tool-access grant. |
| AI-1.1-HUMAN | escalation | High-impact actions require human approval before execution. |
| AI-1.1-VERIFY | requirement | Model outputs used for operational decisions must be verified against source records. |
| AI-1.1-RETAIN | requirement | Prompt and output logs for sensitive workflows are retained for 60 days. |
| AI-1.1-INCIDENT | escalation | Suspected policy violations must be reported as AI incidents. |
