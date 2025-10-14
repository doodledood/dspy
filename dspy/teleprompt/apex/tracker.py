"""MLflow experiment tracker for APEX optimizer."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class ExperimentTracker:
    """Experiment tracker for APEX with MLflow integration."""

    def __init__(
        self,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ):
        """Initialize the experiment tracker.

        Args:
            use_mlflow: Whether to use MLflow for tracking
            mlflow_tracking_uri: Optional MLflow tracking server URI
            mlflow_experiment_name: Optional experiment name
        """
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.active_run = None
        self.run_id = None

        if use_mlflow and not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow tracking requested but MLflow is not installed. Install with: pip install mlflow>=2.18.0"
            )

    def __enter__(self):
        """Context manager entry."""
        if self.use_mlflow:
            self.initialize()
            self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.use_mlflow:
            self.end_run()
        return False

    def initialize(self):
        """Initialize MLflow settings."""
        if not self.use_mlflow:
            return

        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            if self.mlflow_experiment_name:
                mlflow.set_experiment(self.mlflow_experiment_name)
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.use_mlflow = False

    def start_run(self, run_name: str | None = None, nested: bool = False):
        """Start an MLflow run."""
        if not self.use_mlflow:
            return

        try:
            self.active_run = mlflow.start_run(run_name=run_name, nested=nested)
            self.run_id = self.active_run.info.run_id
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            self.use_mlflow = False

    def end_run(self):
        """End the current MLflow run."""
        if not self.use_mlflow or not self.active_run:
            return

        try:
            mlflow.end_run()
            self.active_run = None
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

    def log_params(self, params: dict[str, Any]):
        """Log parameters to MLflow."""
        if not self.use_mlflow:
            return

        try:
            for key, value in params.items():
                if value is not None:
                    value_str = str(value)[:500]
                    mlflow.log_param(key, value_str)
        except Exception as e:
            logger.debug(f"Failed to log params: {e}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Log metrics to MLflow."""
        if not self.use_mlflow:
            return

        try:
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, int | float) and value == value:
                    numeric_metrics[key] = float(value)

            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")

    def log_artifact_text(self, text: str, filename: str):
        """Log text as an artifact."""
        if not self.use_mlflow:
            return

        try:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False) as f:
                f.write(text)
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="apex_outputs")
            Path(temp_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Failed to log artifact: {e}")

    def log_iteration(self, iteration: int, iteration_data: dict[str, Any]):
        """Log data for a specific iteration."""
        if not self.use_mlflow:
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

        if iteration_data.get("hypotheses"):
            hypotheses_text = json.dumps(iteration_data["hypotheses"], indent=2)
            self.log_artifact_text(hypotheses_text, f"hypotheses_iter_{iteration}.json")

    def log_candidate(self, candidate_data: dict[str, Any], iteration: int, candidate_idx: int):
        """Log data for a specific candidate evaluation."""
        if not self.use_mlflow:
            return

        metrics = {
            f"candidate_{candidate_idx}_score": candidate_data.get("overall_score", 0.0),
        }

        if "per_example_scores" in candidate_data:
            scores = candidate_data["per_example_scores"]
            if scores:
                metrics[f"candidate_{candidate_idx}_mean_score"] = sum(scores) / len(scores)
                metrics[f"candidate_{candidate_idx}_min_score"] = min(scores)
                metrics[f"candidate_{candidate_idx}_max_score"] = max(scores)

        self.log_metrics(metrics, step=iteration)

    def log_best_program(self, program_data: dict[str, Any]):
        """Log the final best program."""
        if not self.use_mlflow:
            return

        program_text = json.dumps(program_data, indent=2)
        self.log_artifact_text(program_text, "best_program.json")

        if "overall_score" in program_data:
            self.log_metrics({"final_best_score": program_data["overall_score"]})

    def log_execution_flow(self, flow_text: str, iteration: int):
        """Log execution flow as artifact."""
        if not self.use_mlflow:
            return

        self.log_artifact_text(flow_text, f"execution_flow_iter_{iteration}.txt")

    def get_run_id(self) -> str | None:
        """Get the current MLflow run ID if available."""
        return self.run_id if self.use_mlflow else None

    def is_active(self) -> bool:
        """Check if tracking is currently active."""
        return self.use_mlflow and self.active_run is not None
