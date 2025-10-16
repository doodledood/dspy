"""Minimal MLflow experiment tracker for APEX."""

from __future__ import annotations

import json
import logging
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Thin wrapper around the MLflow fluent API."""

    def __init__(
        self,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = "http://127.0.0.1:5000",
        mlflow_experiment_name: str | None = "APEX",
    ) -> None:
        self.use_mlflow = bool(use_mlflow and MLFLOW_AVAILABLE)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name or "APEX"
        self._run_active = False

        if self.use_mlflow and mlflow is not None:
            try:
                if self.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                if self.mlflow_experiment_name:
                    mlflow.set_experiment(self.mlflow_experiment_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Disabling MLflow tracking: %s", exc)
                self.use_mlflow = False

    def __enter__(self) -> ExperimentTracker:
        if self.use_mlflow and not self._run_active and mlflow is not None:
            try:
                mlflow.start_run()
                self._run_active = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to start MLflow run: %s", exc)
                self.use_mlflow = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._run_active and mlflow is not None:
            status = "FINISHED" if exc_type is None else "FAILED"
            try:
                mlflow.end_run(status=status)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to end MLflow run: %s", exc)
        self._run_active = False
        return False

    # ------------------------------------------------------------------ helpers
    def _active(self) -> bool:
        return self.use_mlflow and self._run_active and mlflow is not None

    def _log_safe(self, func, *args, **kwargs) -> None:
        if not self._active():
            return
        try:
            func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MLflow logging failed: %s", exc)

    # ------------------------------------------------------------------ metrics & params
    def log_params(self, params: Mapping[str, Any]) -> None:
        if not params:
            return
        for key, value in params.items():
            if value is None:
                continue
            self._log_safe(mlflow.log_param, key, str(value)[:500])

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            if isinstance(value, int | float) and math.isfinite(float(value)):
                self._log_safe(mlflow.log_metric, key, float(value), step=step)

    def log_artifact_text(self, text: str, filename: str, artifact_path: str = "apex_outputs") -> None:
        if not self._active():
            return
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8") as handle:
                handle.write(text)
                temp_path = handle.name
            self._log_safe(mlflow.log_artifact, temp_path, artifact_path=artifact_path)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------ tracing
    def is_tracing_enabled(self) -> bool:
        return self._active() and hasattr(mlflow, "start_span")

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ):
        if not self.is_tracing_enabled():
            yield None
            return

        span_cm = None
        try:
            kwargs: dict[str, Any] = {}
            if attributes:
                kwargs["attributes"] = attributes
            span_cm = mlflow.start_span(name=name, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to start MLflow span '%s': %s", name, exc)
            yield None
            return

        with span_cm as span:
            if inputs and hasattr(span, "set_inputs"):
                try:
                    span.set_inputs(inputs)
                except Exception:  # pragma: no cover - defensive
                    pass
            yield span

    # ------------------------------------------------------------------ misc
    def get_run_id(self) -> str | None:
        if not self._active():
            return None
        run = mlflow.active_run()
        return run.info.run_id if run is not None else None

    def is_active(self) -> bool:
        return self._active()

    def log_trace_batch(self, batch: Any) -> None:  # backwards compatibility for tests
        if not batch:
            return
        payload = json.dumps(batch, default=str)
        self.log_artifact_text(payload, "trace_batch.json")

    # ------------------------------------------------------------------ high-level helpers
    def log_iteration(self, iteration: int, iteration_data: Mapping[str, Any]) -> None:
        metrics = {
            "iteration": iteration,
            "num_failures": iteration_data.get("num_failures", 0),
            "num_successes": iteration_data.get("num_successes", 0),
            "num_hypotheses": iteration_data.get("num_hypotheses", 0),
            "num_candidates": iteration_data.get("num_candidates", 0),
        }
        if "best_score" in iteration_data:
            metrics["best_score_in_iteration"] = iteration_data["best_score"]
        self.log_metrics(metrics, step=iteration)

        hypotheses = iteration_data.get("hypotheses")
        if hypotheses:
            self.log_artifact_text(
                json.dumps({"iteration": iteration, "hypotheses": hypotheses}, indent=2, ensure_ascii=False),
                f"hypotheses_iter_{iteration}.json",
            )

    def log_candidate(self, candidate_data: Mapping[str, Any], iteration: int, candidate_idx: int) -> None:
        metrics: dict[str, float] = {}
        score = candidate_data.get("overall_score")
        if isinstance(score, int | float) and math.isfinite(float(score)):
            metrics[f"candidate_{candidate_idx}_score"] = float(score)

        scores = candidate_data.get("per_example_scores")
        if isinstance(scores, list) and scores:
            numeric_scores = [float(s) for s in scores if isinstance(s, int | float) and math.isfinite(float(s))]
            if numeric_scores:
                metrics[f"candidate_{candidate_idx}_mean_score"] = sum(numeric_scores) / len(numeric_scores)
                metrics[f"candidate_{candidate_idx}_min_score"] = min(numeric_scores)
                metrics[f"candidate_{candidate_idx}_max_score"] = max(numeric_scores)

        if metrics:
            self.log_metrics(metrics, step=iteration)

    def log_best_program(self, program_data: Mapping[str, Any]) -> None:
        if not program_data:
            return

        self.log_artifact_text(
            json.dumps(program_data, indent=2, ensure_ascii=False),
            "best_program.json",
        )

        score = program_data.get("overall_score")
        if isinstance(score, int | float) and math.isfinite(float(score)):
            self.log_metrics({"final_best_score": float(score)})
