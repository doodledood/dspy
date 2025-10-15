"""Training set sampling helpers for APEX."""

from __future__ import annotations

import random
from typing import Sequence

from dspy.primitives import Example

from .types import SamplerFn


def sample_trainset(
    trainset: Sequence[Example],
    *,
    iteration: int,
    rng: random.Random,
    sampler: None | int | SamplerFn,
) -> list[Example]:
    """Return a sampled view of the training set for the given iteration."""

    examples = list(trainset)
    if sampler is None:
        sampler = len(examples)

    if isinstance(sampler, int):
        k = min(sampler, len(examples))
        return rng.sample(examples, k=k)

    sampled = sampler(examples, iteration)
    if not isinstance(sampled, list):
        msg = "Custom train_sample callable must return a list of Examples."
        raise TypeError(msg)
    return sampled


__all__ = ["sample_trainset"]
