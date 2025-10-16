"""Utilities for structuring MLflow trace artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class TraceBatch:
    """Normalized execution trace payload ready for MLflow logging."""

    iteration: int
    stage: str
    traces: list[Mapping[str, Any]] = field(default_factory=list)

    @classmethod
    def from_iterable(
        cls,
        *,
        iteration: int,
        stage: str,
        traces: Iterable[Mapping[str, Any]],
    ) -> TraceBatch:
        return cls(iteration=iteration, stage=stage, traces=list(traces))

    def to_artifact(self) -> Mapping[str, Any]:
        """Convert to JSON-serializable structure."""
        return {
            "iteration": self.iteration,
            "stage": self.stage,
            "count": len(self.traces),
            "traces": self.traces,
        }
