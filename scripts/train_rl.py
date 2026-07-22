#!/usr/bin/env python3
"""Optional full-GPU RL trainer (TRL GRPO when available; else documents fallback)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from policyshift.training.rl_smoke import RLTrainConfig, run_rl_smoke_train
from policyshift.utils.io import ensure_dir, write_json


def load_rl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(
                {
                    "case_id": row.get("case_id", "x"),
                    "good": row.get("chosen") or row.get("good") or "",
                    "bad": row.get("rejected") or row.get("bad") or "",
                    "reward_good": float(row.get("reward_good", 1.0)),
                    "reward_bad": float(row.get("reward_bad", 0.0)),
                }
            )
    if not rows:
        raise SystemExit(f"No RL rows in {path}")
    return rows


def run_full_trl_grpo(cfg: dict, rows: list[dict], out_dir: Path) -> dict:
    """Best-effort TRL GRPO / reward trainer; raises with install hint if unavailable."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Full RL requires policyshift[training]. Install: pip install 'policyshift[training]'"
        ) from exc

    # Prefer GRPOTrainer if present in this TRL version; otherwise fall back to smoke adapter
    # after writing a readiness note (keeps CI/CPU environments safe).
    try:
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError:
        note = (
            "TRL GRPOTrainer not available in this environment. "
            "Wrote CPU RLOO-style smoke adapter instead. "
            "Upgrade trl or use Colab notebook for GRPO."
        )
        metrics = run_rl_smoke_train(
            rows,
            RLTrainConfig(
                output_dir=str(out_dir),
                smoke=True,
                max_steps=int(cfg.get("max_steps", 8)),
                seed=int(cfg.get("seed", 42)),
                notes=note,
            ),
        )
        metrics["note"] = note
        return metrics

    model_id = str(cfg.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct"))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    try:
        model = get_peft_model(model, lora)
    except ImportError as exc:
        # Common on Colab: peft expects newer torchao than the environment ships.
        note = (
            f"Full RL LoRA unavailable ({exc}). "
            "Falling back to CPU RLOO-style smoke adapter. "
            "SFT/DPO are the primary post-training evidence; RL is optional."
        )
        metrics = run_rl_smoke_train(
            rows,
            RLTrainConfig(
                output_dir=str(out_dir),
                smoke=True,
                max_steps=int(cfg.get("max_steps", 8)),
                seed=int(cfg.get("seed", 42)),
                notes=note,
            ),
        )
        metrics["note"] = note
        return metrics

    # Minimal prompt dataset for GRPO; reward is length/proxy — replace with verifier reward in research runs.
    prompts = [f"Resolve policy case {r['case_id']}: prefer {r['good']}" for r in rows]

    def reward_fn(completions: list[str], **kwargs):  # noqa: ANN003
        scores = []
        for text in completions:
            # Toy executable-style reward: prefer non-empty grounded-looking answers
            score = 0.1 * min(len(text), 200) / 200.0
            if any(k in text.lower() for k in ("policy", "approve", "quarantine", "refuse")):
                score += 0.5
            scores.append(float(score))
        return scores

    ds = Dataset.from_dict({"prompt": prompts})
    args = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=int(cfg.get("max_steps", 20)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        learning_rate=float(cfg.get("learning_rate", 5e-6)),
        logging_steps=1,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    result = trainer.train()
    adapter = ensure_dir(out_dir / "adapter")
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    return {
        "backend": "trl-grpo-lora",
        "checkpoint": str(adapter),
        "train_loss": [float(getattr(result, "training_loss", 0.0) or 0.0)],
        "steps": int(cfg.get("max_steps", 20)),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rl/full_gpu.yaml"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    raw = {}
    if args.config.exists():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    train_file = args.train_file or Path(raw.get("train_file", "data/full/rl/rl_pairs.jsonl"))
    output_dir = ensure_dir(
        args.output_dir or Path(raw.get("output_dir", "artifacts/experiments/rl-full/checkpoints"))
    )
    smoke = bool(raw.get("smoke", True)) if args.smoke is None else bool(args.smoke)

    if not train_file.exists():
        raise SystemExit(
            f"Missing {train_file}. Run: python scripts/prepare_full_training_data.py"
        )

    rows = load_rl_rows(train_file)
    write_json(output_dir / "train_config.json", {**raw, "smoke": smoke, "train_file": str(train_file)})

    if smoke:
        metrics = run_rl_smoke_train(
            rows,
            RLTrainConfig(
                output_dir=str(output_dir),
                smoke=True,
                max_steps=int(raw.get("max_steps", 4)),
                seed=int(raw.get("seed", 42)),
            ),
        )
    else:
        metrics = run_full_trl_grpo(raw, rows, output_dir)

    write_json(output_dir / "train_metrics.json", metrics)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
