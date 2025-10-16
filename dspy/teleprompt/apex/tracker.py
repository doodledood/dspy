"""MLflow experiment tracker for APEX optimizer."""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .tracking_session import TraceBatch

logger = logging.getLogger(__name__)

try:
    from mlflow.exceptions import MlflowException  # type: ignore
    from mlflow.tracking import MlflowClient  # type: ignore

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MlflowClient = None  # type: ignore[assignment]
    MlflowException = Exception  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Experiment tracker for APEX with MLflow integration."""

    def __init__(
        self,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = "http://127.0.0.1:5000",
        mlflow_experiment_name: str | None = "APEX",
    ):
        """Initialize the experiment tracker.

        Args:
            use_mlflow: Whether to use MLflow for tracking.
            mlflow_tracking_uri: MLflow tracking server URI.
            mlflow_experiment_name: Experiment name to create or reuse.
        """
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name or "APEX"

        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._run_id: str | None = None
        self._run_active = False

        if use_mlflow and not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow tracking requested but MLflow is not installed. Install with: pip install mlflow>=2.9.0"
            )
            return

        if self.use_mlflow:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Set up MLflow client and experiment."""
        if not self.use_mlflow or MlflowClient is None:
            return

        try:
            self._client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            experiment = self._client.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                self._experiment_id = self._client.create_experiment(self.mlflow_experiment_name)
            else:
                self._experiment_id = experiment.experiment_id
        except MlflowException as exc:  # pragma: no cover - network/config failures
            logger.warning(f"Failed to initialize MLflow client: {exc}")
            self.use_mlflow = False
            self._client = None
            self._experiment_id = None

    def __enter__(self) -> ExperimentTracker:
        """Context manager entry."""
        if self.use_mlflow:
            self._ensure_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if self.use_mlflow and self._run_active and self._client and self._run_id:
            status = "FINISHED" if exc_type is None else "FAILED"
            try:
                self._client.set_terminated(self._run_id, status=status)
            except MlflowException as exc:  # pragma: no cover - defensive
                logger.debug(f"Failed to terminate MLflow run {self._run_id}: {exc}")
        self._run_active = False
        return False

    def _ensure_run(self) -> str | None:
        """Create the MLflow run if needed."""
        if not self.use_mlflow or not self._client or not self._experiment_id:
            return None
        if self._run_id is not None:
            self._run_active = True
            return self._run_id

        try:
            run = self._client.create_run(self._experiment_id)
        except MlflowException as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to create MLflow run: {exc}")
            self.use_mlflow = False
            return None

        self._run_id = run.info.run_id
        self._run_active = True
        return self._run_id

    def _can_log(self) -> bool:
        if not self.use_mlflow:
            return False
        if self._client is None or self._experiment_id is None:
            return False
        if self._run_id is None:
            self._ensure_run()
        return self._client is not None and self._run_id is not None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self._can_log():
            return

        for key, value in params.items():
            if value is None:
                continue
            try:
                value_str = str(value)
                self._client.log_param(self._run_id, key, value_str[:500])
            except MlflowException as exc:  # pragma: no cover - defensive
                logger.debug(f"Failed to log param '{key}': {exc}")

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        if not self._can_log():
            return

        timestamp = int(time.time() * 1000)
        metric_step = step if step is not None else 0

        for key, value in metrics.items():
            if not isinstance(value, int | float):
                continue
            if not math.isfinite(float(value)):
                continue
            try:
                self._client.log_metric(self._run_id, key, float(value), timestamp=timestamp, step=metric_step)
            except MlflowException as exc:  # pragma: no cover - defensive
                logger.debug(f"Failed to log metric '{key}': {exc}")

    def log_artifact_text(self, text: str, filename: str, artifact_path: str = "apex_outputs") -> None:
        """Log plain text content as an MLflow artifact."""
        if not self._can_log():
            return

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8") as handle:
                handle.write(text)
                temp_path = handle.name
            assert self._client is not None  # for type-checkers
            self._client.log_artifact(self._run_id, temp_path, artifact_path=artifact_path)
        except (OSError, MlflowException) as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to log artifact '{filename}': {exc}")
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def log_iteration(self, iteration: int, iteration_data: Mapping[str, Any]) -> None:
        """Log high-level iteration metrics and hypothesis artifacts."""
        if not self._can_log():
            return

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
            payload = json.dumps({"iteration": iteration, "hypotheses": hypotheses}, indent=2, ensure_ascii=False)
            self.log_artifact_text(payload, f"hypotheses_iter_{iteration}.json")

    def log_candidate(self, candidate_data: Mapping[str, Any], iteration: int, candidate_idx: int) -> None:
        """Log metrics for an evaluated candidate."""
        if not self._can_log():
            return

        metrics: dict[str, float] = {}
        score = candidate_data.get("overall_score")
        if isinstance(score, int | float) and math.isfinite(float(score)):
            metrics[f"candidate_{candidate_idx}_score"] = float(score)

        scores = candidate_data.get("per_example_scores")
        if isinstance(scores, Sequence) and scores:
            numeric_scores = [float(s) for s in scores if isinstance(s, int | float) and math.isfinite(float(s))]
            if numeric_scores:
                metrics[f"candidate_{candidate_idx}_mean_score"] = sum(numeric_scores) / len(numeric_scores)
                metrics[f"candidate_{candidate_idx}_min_score"] = min(numeric_scores)
                metrics[f"candidate_{candidate_idx}_max_score"] = max(numeric_scores)

        if metrics:
            self.log_metrics(metrics, step=iteration)

    def log_best_program(self, program_data: Mapping[str, Any]) -> None:
        """Persist the best program summary as both metrics and artifact."""
        if not self._can_log():
            return

        payload = json.dumps(program_data, indent=2, ensure_ascii=False)
        self.log_artifact_text(payload, "best_program.json")

        score = program_data.get("overall_score")
        if isinstance(score, int | float) and math.isfinite(float(score)):
            self.log_metrics({"final_best_score": float(score)})

    def log_trace_batch(self, batch: TraceBatch) -> None:
        """Log execution traces for a batch of examples."""
        if not self._can_log() or not batch.traces:
            return

        artifact = batch.to_artifact()
        payload = json.dumps(artifact, indent=2, ensure_ascii=False)
        filename = f"{batch.stage}_iter_{batch.iteration}_traces.json"
        self.log_artifact_text(payload, filename, artifact_path="apex_traces")

    def get_run_id(self) -> str | None:
        """Get the current MLflow run ID."""
        return self._run_id if self.use_mlflow else None

    def is_active(self) -> bool:
        """Check if tracking is currently active."""
        return self._can_log() and self._run_active
