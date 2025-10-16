"""MLflow tracing utilities for APEX."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow import MlflowClient

    MLFLOW_AVAILABLE = True
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


@dataclass(slots=True)
class TraceHandle:
    """Minimal handle for closing out an MLflow trace."""

    client: MlflowClient
    trace_id: str


_MLFLOW_CLIENT: Optional[MlflowClient] = None


def _get_client() -> Optional[MlflowClient]:
    global _MLFLOW_CLIENT
    if not MLFLOW_AVAILABLE:
        return None
    if _MLFLOW_CLIENT is None:
        try:
            _MLFLOW_CLIENT = MlflowClient()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to initialize MlflowClient: %s", exc)
            return None
    return _MLFLOW_CLIENT


def _start_trace(name: str, *, inputs: Mapping[str, Any] | None, attributes: Mapping[str, Any]) -> Optional[TraceHandle]:
    client = _get_client()
    if client is None:
        return None

    try:
        span = client.start_trace(
            name=name,
            inputs=dict(inputs or {}),
            attributes=dict(attributes or {}),
        )
        return TraceHandle(client=client, trace_id=span.trace_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to start MLflow trace '%s': %s", name, exc)
        return None


def create_evaluation_trace(
    example_idx: int,
    stage: str,
    iteration: int,
    hypothesis_info: Mapping[str, Any] | None = None,
) -> Optional[TraceHandle]:
    if not MLFLOW_AVAILABLE:
        return None

    attributes: dict[str, Any] = {
        "stage": stage,
        "iteration": iteration,
    }

    inputs: dict[str, Any] = {"example_index": example_idx}

    if hypothesis_info:
        attributes["hypothesis_strategy"] = hypothesis_info.get("strategy", "unknown")
        attributes["hypothesis_impact"] = hypothesis_info.get("impact_score", 0.0)

    return _start_trace(
        name=f"apex_evaluate_{stage}",
        inputs=inputs,
        attributes=attributes,
    )


def end_trace(
    trace: Optional[TraceHandle],
    outputs: Mapping[str, Any] | None = None,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> None:
    if not trace:
        return

    try:
        trace.client.end_trace(
            trace_id=trace.trace_id,
            outputs=dict(outputs or {}),
            attributes=dict(attributes or {}),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to end MLflow trace '%s': %s", trace.trace_id, exc)


def create_analysis_trace(
    analysis_type: str,
    iteration: int,
    example_data: Mapping[str, Any],
) -> Optional[TraceHandle]:
    if not MLFLOW_AVAILABLE:
        return None

    attributes = {
        "analysis_type": analysis_type,
        "iteration": iteration,
        "metric_score": example_data.get("metric_score", 0.0),
    }

    inputs = {"example_index": example_data.get("example_index", -1)}

    return _start_trace(
        name=f"apex_analyze_{analysis_type}",
        inputs=inputs,
        attributes=attributes,
    )


def create_hypothesis_trace(
    iteration: int,
    num_failures: int,
    num_successes: int,
) -> Optional[TraceHandle]:
    if not MLFLOW_AVAILABLE:
        return None

    attributes = {
        "iteration": iteration,
        "num_failures": num_failures,
        "num_successes": num_successes,
    }

    return _start_trace(
        name="apex_generate_hypotheses",
        inputs={},
        attributes=attributes,
    )
