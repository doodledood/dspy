"""Shared type definitions for the APEX optimizer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Literal, Mapping, TypeAlias, TypeVar

from dspy.primitives import Example, Prediction

TraceEntry: TypeAlias = tuple[Any, Mapping[str, Any], Prediction]
LogLevel: TypeAlias = Literal["info", "warning", "debug", "error"]

ItemT = TypeVar("ItemT")
MetricFn = Callable[[Example, Prediction, list[TraceEntry]], Any]
SamplerFn = Callable[[list[Example], int], list[Example]]


class Verbosity(str, Enum):
    """Verbosity levels supported by the APEX optimizer."""

    NONE = "none"
    NORMAL = "normal"
    HIGH = "high"

    @classmethod
    def parse(cls, value: str | Verbosity | None) -> Verbosity:
        """Parse a verbosity value from user input."""
        if value is None:
            return cls.NORMAL
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        for member in cls:
            if member.value == normalized:
                return member
        msg = "Unsupported verbosity level '{value}'. Use one of: none, normal, high."
        raise ValueError(msg.format(value=value))


def verbosity_rank(level: Verbosity) -> int:
    """Return a numeric ranking for a verbosity level."""

    return {
        Verbosity.NONE: 0,
        Verbosity.NORMAL: 1,
        Verbosity.HIGH: 2,
    }[level]


__all__ = [
    "TraceEntry",
    "LogLevel",
    "ItemT",
    "MetricFn",
    "SamplerFn",
    "Verbosity",
    "verbosity_rank",
]
