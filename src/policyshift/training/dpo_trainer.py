"""DPO training with a CPU smoke path (no large model download required).

Smoke mode runs a tiny preference-margin loop (torch if available, else numpy)
and writes a loadable checkpoint. Full GPU TRL DPO is documented under
configs/dpo/full_gpu.yaml.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from policyshift.training.sft_trainer import load_checkpoint
from policyshift.utils.hashing import sha256_text
from policyshift.utils.io import ensure_dir, write_json


@dataclass
class DPOTrainConfig:
    output_dir: str
    train_file: str
    model_name_or_path: str = "smoke-tiny-dpo"
    # Full DPO must warm-start from an SFT LoRA adapter (not raw base Qwen).
    sft_adapter_path: str | None = None
    smoke: bool = True
    max_steps: int | None = 2
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-3
    beta: float = 0.1
    seed: int = 42
    lora_r: int = 4
    lora_alpha: int = 8
    max_seq_length: int = 1536
    max_prompt_length: int = 1152
    max_completion_length: int = 384
    per_device_train_batch_size: int = 1
    notes: str = ""
    policy_versions: list[str] | None = None
    skip_data_validation: bool = False


def resolve_sft_adapter_path(path: str | Path) -> Path:
    """Accept either the adapter dir or its parent checkpoints/ folder."""
    root = Path(path)
    if (root / "adapter_config.json").exists():
        return root
    nested = root / "adapter"
    if (nested / "adapter_config.json").exists():
        return nested
    raise FileNotFoundError(
        f"No PEFT adapter_config.json under {root} or {nested}. "
        "Train SFT first (configs/sft/full_gpu.yaml)."
    )


def load_dpo_rows(
    train_file: str | Path,
    limit: int | None = None,
    *,
    policy_versions: list[str] | None = None,
) -> list[dict[str, Any]]:
    from policyshift.training.version_filters import filter_rows_by_versions

    rows: list[dict[str, Any]] = []
    with Path(train_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    rows = filter_rows_by_versions(rows, policy_versions)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(
            f"No DPO examples in {train_file}"
            + (f" for policy_versions={policy_versions}" if policy_versions else "")
        )
    return rows


def _text_vec(text: str, dim: int = 32) -> list[float]:
    """Deterministic bag-of-bytes feature vector for smoke preference scoring."""
    import numpy as np

    vec = np.zeros(dim, dtype=np.float64)
    raw = text.encode("utf-8", errors="ignore")
    for i, b in enumerate(raw[:512]):
        vec[i % dim] += (b / 255.0) * (1.0 / (1 + i // dim))
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).tolist()


def _smoke_dpo_torch(cfg: DPOTrainConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(cfg.seed)
    dim = 32
    model = nn.Sequential(nn.Linear(dim, cfg.lora_r), nn.Tanh(), nn.Linear(cfg.lora_r, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    losses: list[float] = []
    started = time.perf_counter()
    step = 0
    max_steps = int(cfg.max_steps or 2)
    while step < max_steps:
        for row in rows:
            if step >= max_steps:
                break
            prompt = row["prompt"]
            chosen = _text_vec(prompt + "\n" + row["chosen"], dim)
            rejected = _text_vec(prompt + "\n" + row["rejected"], dim)
            c = model(torch.tensor(chosen, dtype=torch.float32))
            r = model(torch.tensor(rejected, dtype=torch.float32))
            # DPO-style logistic preference loss with temperature beta
            loss = -torch.nn.functional.logsigmoid(cfg.beta * (c - r))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1

    elapsed = time.perf_counter() - started
    out = ensure_dir(cfg.output_dir)
    ckpt_path = out / "smoke_dpo_adapter.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "dim": dim,
                "lora_r": cfg.lora_r,
                "beta": cfg.beta,
                "seed": cfg.seed,
            },
        },
        ckpt_path,
    )
    return {
        "backend": "torch-smoke-dpo",
        "checkpoint": str(ckpt_path),
        "train_loss": losses,
        "final_loss": losses[-1] if losses else None,
        "steps": len(losses),
        "elapsed_sec": elapsed,
        "peak_memory_mb": None,
        "beta": cfg.beta,
    }


def _smoke_dpo_numpy(cfg: DPOTrainConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(cfg.seed)
    dim = 32
    w = rng.normal(0, 0.01, size=(dim,))
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(int(cfg.max_steps or 2)):
        row = rows[step % len(rows)]
        c = np.array(_text_vec(row["prompt"] + "\n" + row["chosen"], dim))
        r = np.array(_text_vec(row["prompt"] + "\n" + row["rejected"], dim))
        margin = float(cfg.beta * (c @ w - r @ w))
        # logistic preference loss; update favors chosen
        loss = math.log1p(math.exp(-margin))
        grad = (1.0 / (1.0 + math.exp(margin))) * cfg.beta * (r - c)
        w = w - cfg.learning_rate * grad
        losses.append(float(loss))
    elapsed = time.perf_counter() - started
    out = ensure_dir(cfg.output_dir)
    payload = {
        "format": "numpy-smoke-dpo-v1",
        "seed": cfg.seed,
        "beta": cfg.beta,
        "lora_r": cfg.lora_r,
        "data_checksum": sha256_text("".join(r["id"] for r in rows)),
        "weights": w.tolist(),
    }
    ckpt_path = out / "smoke_dpo_adapter.json"
    write_json(ckpt_path, payload)
    return {
        "backend": "numpy-smoke-dpo",
        "checkpoint": str(ckpt_path),
        "train_loss": losses,
        "final_loss": losses[-1] if losses else None,
        "steps": len(losses),
        "elapsed_sec": elapsed,
        "peak_memory_mb": None,
        "beta": cfg.beta,
        "note": (
            "Torch unavailable; wrote deterministic smoke DPO adapter. "
            "Use configs/dpo/full_gpu.yaml for real TRL DPO on Qwen2.5-0.5B-Instruct."
        ),
    }


def load_dpo_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load smoke DPO checkpoint; reuses SFT loader for .pt/.json."""
    ckpt = Path(path)
    if ckpt.is_dir():
        for name in ("smoke_dpo_adapter.pt", "smoke_dpo_adapter.json", "adapter_config.json"):
            candidate = ckpt / name
            if candidate.exists():
                return load_checkpoint(candidate)
    return load_checkpoint(ckpt)


def run_dpo(cfg: DPOTrainConfig) -> dict[str, Any]:
    """Run smoke or (documented) full DPO. Always writes metrics + checkpoint metadata."""
    rows = load_dpo_rows(
        cfg.train_file,
        limit=64 if cfg.smoke else None,
        policy_versions=cfg.policy_versions,
    )
    out = ensure_dir(cfg.output_dir)
    write_json(out / "train_config.json", asdict(cfg))

    if cfg.smoke:
        try:
            import torch  # noqa: F401

            result = _smoke_dpo_torch(cfg, rows)
        except ImportError:
            result = _smoke_dpo_numpy(cfg, rows)
    else:
        result = _run_full_trl_dpo(cfg, rows)

    loaded = load_dpo_checkpoint(result["checkpoint"])
    metrics = {
        "status": "completed",
        "smoke": cfg.smoke,
        "model_name_or_path": cfg.model_name_or_path,
        "policy_versions": cfg.policy_versions,
        "n_train_pairs": len(rows),
        "training": result,
        "checkpoint_load": {
            "ok": True,
            "path": loaded["path"],
            "format": loaded["format"],
        },
    }
    write_json(out / "train_metrics.json", metrics)
    (out / "CHECKPOINT_NOTES.txt").write_text(
        "Smoke DPO checkpoints are tiny preference adapters, not production weights.\n"
        "Full GPU command:\n"
        "  pip install 'policyshift[training]'\n"
        "  python scripts/train_dpo.py --config configs/dpo/full_gpu.yaml\n",
        encoding="utf-8",
    )
    return metrics


def _load_sft_peft_pair(
    *,
    base_model: str,
    sft_adapter: Path,
):
    """Load trainable policy + frozen reference, both initialized from the SFT adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    try:
        policy_base = AutoModelForCausalLM.from_pretrained(base_model)
        ref_base = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(policy_base, str(sft_adapter), is_trainable=True)
        ref_model = PeftModel.from_pretrained(ref_base, str(sft_adapter), is_trainable=False)
    except ImportError as exc:
        if "torchao" in str(exc).lower():
            raise ImportError(
                "PEFT/torchao version conflict. In Colab run:\n"
                '  pip install -U "torchao>=0.16.0"\n'
                "or:\n"
                "  pip uninstall -y torchao\n"
                "then re-run training."
            ) from exc
        raise
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return model, ref_model


def _run_full_trl_dpo(cfg: DPOTrainConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Full TRL DPO path (requires GPU for practical Qwen runs).

    Initializes policy + reference from the SFT LoRA checkpoint (not raw base Qwen).
    """
    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Full DPO requires transformers/peft/trl/datasets. "
            "Install with: pip install 'policyshift[training]'"
        ) from exc

    from policyshift.training.dpo_format import (
        DPOBudgetConfig,
        budget_dpo_pair,
        validate_budgeted_pairs,
    )

    if not cfg.sft_adapter_path:
        raise ValueError(
            "Full DPO requires sft_adapter_path (SFT LoRA checkpoint). "
            "Refusing to initialize from raw base Qwen. "
            "Set sft_adapter_path in configs/dpo/full_gpu.yaml."
        )
    sft_adapter = resolve_sft_adapter_path(cfg.sft_adapter_path)

    print(
        "DPO init:\n"
        f"  policy/ref from SFT adapter: {sft_adapter}\n"
        f"  base model: {cfg.model_name_or_path}\n"
        f"  epochs: {cfg.num_train_epochs}\n"
        f"  training pairs: {len(rows)}\n"
        f"  sequence budgets: {cfg.max_seq_length} / {cfg.max_prompt_length} / {cfg.max_completion_length}"
    )

    # Tokenizer-only first: fail closed on data before loading model weights.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    budget = DPOBudgetConfig(
        max_length=cfg.max_seq_length,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
    )
    validation_path = ensure_dir(Path(cfg.output_dir)) / "dpo_validation.json"
    if not cfg.skip_data_validation:
        report = validate_budgeted_pairs(rows, tokenizer, budget)
        compact = {k: v for k, v in report.items() if k != "rows"}
        write_json(validation_path, compact)
        if not report["passed"]:
            raise ValueError(
                "DPO data validation failed before model loading. "
                f"See {validation_path}. "
                f"retain_pct={report['percentage_retaining_chosen_rejected_difference']:.2f} "
                f"raw_identical={report['raw_identical_pairs']} "
                f"empty_loss_masks={report['empty_loss_mask_pairs']}"
            )

    budgeted_rows = [
        budget_dpo_pair(tokenizer, r["prompt"], r["chosen"], r["rejected"], budget)
        for r in rows
    ]
    ds = Dataset.from_list([b.to_conversational() for b in budgeted_rows])

    model, ref_model = _load_sft_peft_pair(
        base_model=cfg.model_name_or_path,
        sft_adapter=sft_adapter,
    )

    # Rows are pre-budgeted; TRL max_* must not re-apply keep_start on concat.
    import inspect

    cfg_kwargs: dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "learning_rate": cfg.learning_rate,
        "beta": cfg.beta,
        "logging_steps": 1,
        "report_to": [],
        "max_length": cfg.max_seq_length,
        "max_prompt_length": cfg.max_prompt_length,
        "truncation_mode": "keep_start",
        "remove_unused_columns": False,
        "save_strategy": "epoch",
        "logging_first_step": True,
    }
    # Prefer epoch-based training; only honor max_steps when explicitly > 0.
    if cfg.max_steps is not None and int(cfg.max_steps) > 0:
        cfg_kwargs["max_steps"] = int(cfg.max_steps)
    supported = set(inspect.signature(DPOConfig.__init__).parameters)
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in supported}
    args = DPOConfig(**cfg_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer_supported = set(inspect.signature(DPOTrainer.__init__).parameters)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_supported}
    trainer = DPOTrainer(**trainer_kwargs)
    started = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - started
    adapter_dir = str(ensure_dir(Path(cfg.output_dir) / "adapter"))
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    global_step = int(getattr(train_result, "global_step", 0) or 0)
    return {
        "backend": "trl-dpo-lora",
        "checkpoint": adapter_dir,
        "init_from": str(sft_adapter),
        "n_train_pairs": len(rows),
        "num_train_epochs": cfg.num_train_epochs,
        "train_loss": [float(getattr(train_result, "training_loss", 0.0) or 0.0)],
        "final_loss": float(getattr(train_result, "training_loss", 0.0) or 0.0),
        "steps": global_step,
        "elapsed_sec": elapsed,
        "peak_memory_mb": None,
        "beta": cfg.beta,
        "budget": {
            "max_length": cfg.max_seq_length,
            "max_prompt_length": cfg.max_prompt_length,
            "max_completion_length": cfg.max_completion_length,
        },
        "validation_report": str(validation_path),
    }
