"""Utilities for converting APEX payload objects into JSON-serializable data."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from dspy.primitives import Example, Prediction


def to_serializable(value: Any) -> Any:
    """Best-effort conversion of mixed DSPy, LiteLLM, and Pydantic objects to plain data."""
    if isinstance(value, Prediction | Example):
        return value.toDict()

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()  # type: ignore[no-any-return]
        except Exception:  # pragma: no cover - defensive
            pass

    if hasattr(value, "model_dump"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                return value.model_dump()  # type: ignore[no-untyped-call]
        except Exception:  # pragma: no cover - defensive
            pass

    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}

    if isinstance(value, set):
        return [to_serializable(v) for v in value]

    if isinstance(value, str | bytes | bytearray):
        return value

    if isinstance(value, Sequence):
        return [to_serializable(v) for v in value]

    return value
