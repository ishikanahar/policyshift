"""Deterministic seeding utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> random.Random:
    """Seed Python and NumPy RNGs; return an isolated Random instance."""
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def stable_choice(rng: random.Random, items: list[Any]) -> Any:
    if not items:
        raise ValueError("Cannot choose from empty list")
    return items[rng.randrange(len(items))]
