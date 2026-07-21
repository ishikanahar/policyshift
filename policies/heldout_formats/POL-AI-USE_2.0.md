# Synthetic Enterprise Data and AI Use Policy

- policy_id: `POL-AI-USE`
- version: `2.0`
- effective_at: `2025-01-01T00:00:00+00:00`

| clause_id | rule_type | text |
| --- | --- | --- |
| AI-2.0-SENS | prohibition | Sensitive or confidential data must not be sent to external APIs. |
| AI-2.0-PUBLIC-API | permission | External APIs may be used for public metadata that is not sensitive. |
| AI-2.0-MODEL | requirement | Only approved models may be used for production workflows. |
| AI-2.0-TOOLS | requirement | Agents may invoke only tools listed on the user tool-access grant. |
| AI-2.0-HUMAN | escalation | High-impact actions require human approval before execution. |
| AI-2.0-VERIFY | requirement | Model outputs used for operational decisions must be verified against source records. |
| AI-2.0-RETAIN | requirement | Prompt and output logs for sensitive workflows are retained for 60 days. |
| AI-2.0-NO-REDACT-EX | prohibition | Redacted confidential excerpts are not exempt from external-API prohibition. |
| AI-2.0-INCIDENT | escalation | Suspected policy violations must be reported as AI incidents. |
