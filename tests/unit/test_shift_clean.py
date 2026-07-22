"""CPU tests for leakage-free shift-clean data prep + validation."""

from __future__ import annotations

from pathlib import Path

from policyshift.training.shift_clean import (
    prepare_shift_clean_data,
    require_shift_clean_validation,
    validate_shift_split,
)


def test_prepare_and_validate_shift_clean(tmp_path: Path) -> None:
    root = tmp_path / "shift_clean"
    report = prepare_shift_clean_data(
        out_root=root, seed=42, n_train_cases=24, n_eval_cases=8
    )
    assert report["n_sft"] >= 8
    assert report["n_dpo"] >= 8
    assert report["n_eval"] >= 4
    assert set(report["train_versions"]) == {"1.0", "1.1"}
    assert report["eval_versions"] == ["2.0"]

    val = validate_shift_split(data_root=root, write_stamp=True)
    assert val["passed"]
    assert val["leakage_count"] == 0
    assert val["evaluation_versions"] == ["2.0"]
    assert "2.0" not in val["sft_training_versions"]
    assert "2.0" not in val["dpo_training_versions"]

    stamp = require_shift_clean_validation(root / "sft" / "sft_train.jsonl")
    assert stamp is not None and stamp.exists()
