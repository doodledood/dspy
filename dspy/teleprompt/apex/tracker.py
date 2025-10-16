"""Minimal MLflow experiment tracker for APEX."""

from __future__ import annotations

import json
import logging
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

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
        self._autolog_kwargs: dict[str, Any] = {}
        self._autolog_active = False
        self._supports_tracing = False

        if self.use_mlflow and mlflow is not None:
            if not hasattr(mlflow, "dspy") or not hasattr(mlflow.dspy, "autolog"):
                raise RuntimeError("mlflow.dspy.autolog is required for MLflow tracking.")
            self._configure_remote_tracking()
            self._supports_tracing = hasattr(mlflow, "start_span")

    def __enter__(self) -> ExperimentTracker:
        if self.use_mlflow and not self._run_active and mlflow is not None:
            try:
                mlflow.start_run()
                self._run_active = True
                self._enable_autolog()
            except Exception as exc:  # pragma: no cover - defensive
                self._disable_autolog(silent=True)
                raise RuntimeError(f"Failed to start MLflow run: {exc}") from exc
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            self._disable_autolog(silent=True)
        finally:
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

    def _configure_remote_tracking(self) -> None:
        if not self.mlflow_tracking_uri:
            raise ValueError("An MLflow tracking URI is required when use_mlflow=True.")
        parsed = urlparse(self.mlflow_tracking_uri)
        if parsed.scheme not in {"http", "https"}:
            msg = f"MLflow tracking URI must use http or https. Received '{self.mlflow_tracking_uri}'."
            raise ValueError(msg)

        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            resolved_uri = mlflow.get_tracking_uri()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to configure MLflow tracking URI '{self.mlflow_tracking_uri}': {exc}") from exc

        if resolved_uri and resolved_uri.startswith("file:"):
            raise RuntimeError(
                "MLflow resolved the tracking URI to a local file store. Configure an MLflow server and supply its HTTP(S) URI."
            )

        if self.mlflow_experiment_name:
            try:
                mlflow.set_experiment(self.mlflow_experiment_name)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Failed to set MLflow experiment '{self.mlflow_experiment_name}': {exc}") from exc

        self._autolog_kwargs: dict[str, Any] = {
            "log_traces": True,
            "log_traces_from_compile": True,
            "log_traces_from_eval": True,
            "log_compiles": False,
            "log_evals": False,
        }

    def _enable_autolog(self) -> None:
        if self._autolog_active or mlflow is None:
            return
        try:
            mlflow.dspy.autolog(**self._autolog_kwargs, silent=True)
            self._autolog_active = True
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to enable MLflow DSPy autolog: {exc}") from exc

    def _disable_autolog(self, *, silent: bool = False) -> None:
        if not self._autolog_active or mlflow is None:
            return
        try:
            mlflow.dspy.autolog(disable=True, silent=True)
        except Exception as exc:  # pragma: no cover - defensive
            if not silent:
                logger.warning("Failed to disable MLflow DSPy autolog: %s", exc)
        finally:
            self._autolog_active = False

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
        return self._active() and self._supports_tracing

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
