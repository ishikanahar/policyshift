# Data Card (Phase 1 draft)

## Summary

PolicyShift uses independently authored **synthetic** policy documents and case events. Nothing is copied from employer systems, private regulations, patient/customer data, or confidential prompts.

## Domains

| Domain | Policy ID | Versions |
| --- | --- | --- |
| Scientific materials receiving | `POL-MAT-RECV` | 1.0, 1.1, 2.0 |
| Laboratory access and equipment | `POL-LAB-ACCESS` | 1.0, 1.1, 2.0 |
| Enterprise data and AI use | `POL-AI-USE` | 1.0, 1.1, 2.0 |

## Cases

Generated deterministically via `scripts/generate_cases.py` with configurable seed. Default smoke generation: 120 cases. Splits are assigned by **template identity** to prevent train/val/test leakage.

## Provenance

- Generator: `src/policyshift/data_generation/`
- Label: `metadata.synthetic = true`
- Source URI pattern: `synthetic://policyshift/...`

## Intended use

Research benchmark for policy-aware tool-use agents and post-training methods. Not for production compliance decisions.
