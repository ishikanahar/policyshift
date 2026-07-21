"""JSON/YAML IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    elif isinstance(payload, list) and payload and isinstance(payload[0], BaseModel):
        data = [item.model_dump(mode="json") for item in payload]
    else:
        data = payload
    target.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_models(path: str | Path, model: Type[T]) -> list[T]:
    raw = read_json(path)
    if isinstance(raw, list):
        return [model.model_validate(item) for item in raw]
    return [model.model_validate(raw)]


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, BaseModel):
                handle.write(json.dumps(row.model_dump(mode="json"), default=str) + "\n")
            else:
                handle.write(json.dumps(row, default=str) + "\n")
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data
