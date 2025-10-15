"""Module wrapper for automatic MLflow tracking of all module and predictor calls."""

import json
import logging
import time
from typing import Any

import dspy

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class TrackedPredictor:
    """Wraps a predictor to track its calls."""

    def __init__(self, predictor, predictor_name: str, run_id: str | None, iteration: int | None):
        self._wrapped_predictor = predictor
        self._predictor_name = predictor_name
        self._run_id = run_id
        self._iteration = iteration
        self._call_count = 0
        self._tracking_enabled = MLFLOW_AVAILABLE and run_id is not None

    def __call__(self, *args, **kwargs):
        self._call_count += 1

        if not self._tracking_enabled:
            return self._wrapped_predictor(*args, **kwargs)

        start_time = time.time()
        call_info = {
            "type": "predictor",
            "predictor_name": self._predictor_name,
            "predictor_type": self._wrapped_predictor.__class__.__name__,
            "call_count": self._call_count,
            "iteration": self._iteration,
            "timestamp": start_time,
        }

        # Capture signature instructions if available
        if hasattr(self._wrapped_predictor, "signature") and hasattr(self._wrapped_predictor.signature, "instructions"):
            call_info["instructions"] = self._wrapped_predictor.signature.instructions[:500]  # Truncate for logging

        # Log inputs
        if kwargs:
            call_info["input_keys"] = list(kwargs.keys())
            # Sample first 200 chars of each input for logging
            call_info["input_samples"] = {k: str(v)[:200] for k, v in kwargs.items()}

        try:
            # Execute the actual predictor call
            result = self._wrapped_predictor(*args, **kwargs)

            call_info["execution_time"] = time.time() - start_time
            call_info["status"] = "success"

            # Log output info
            if isinstance(result, dspy.Prediction):
                call_info["output_fields"] = list(result.keys())
                # Sample first 200 chars of each output
                call_info["output_samples"] = {k: str(v)[:200] for k, v in result.toDict().items()}

            self._log_call(call_info)
            return result

        except Exception as e:
            call_info["execution_time"] = time.time() - start_time
            call_info["status"] = "error"
            call_info["error"] = str(e)
            self._log_call(call_info)
            raise

    def _log_call(self, call_info: dict[str, Any]):
        """Log predictor call to MLflow."""
        if not self._tracking_enabled:
            return

        try:
            # Log execution time as metric
            if "execution_time" in call_info:
                metric_name = f"predictor_{self._predictor_name}_exec_time"
                step = call_info.get("call_count", 0)
                mlflow.log_metric(metric_name, call_info["execution_time"], step=step)

            # Log full trace as artifact
            artifact_name = f"predictor_traces_iter_{self._iteration}.jsonl" if self._iteration else "predictor_traces.jsonl"

            import os
            import tempfile

            with tempfile.NamedTemporaryFile(mode="a", suffix=f"_{artifact_name}", delete=False) as f:
                f.write(json.dumps(call_info) + "\n")
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="execution_traces")
            os.unlink(temp_path)

        except Exception as e:
            logger.debug(f"Failed to log predictor call to MLflow: {e}")

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped predictor."""
        return getattr(self._wrapped_predictor, name)


class TrackedModule(dspy.Module):
    """Recursively wraps a dspy.Module to track ALL forward() and predictor calls."""

    def __init__(self, module: dspy.Module, run_id: str | None = None, iteration: int | None = None, depth: int = 0):
        """Initialize the tracked module wrapper.

        Args:
            module: The dspy.Module to wrap
            run_id: MLflow run ID for logging (if MLflow is enabled)
            iteration: Current iteration number for logging context
            depth: Depth in the module hierarchy (for nested tracking)
        """
        super().__init__()
        self._wrapped_module = module
        self._run_id = run_id
        self._iteration = iteration
        self._depth = depth
        self._call_count = 0
        self._tracking_enabled = MLFLOW_AVAILABLE and run_id is not None

        # Recursively wrap all sub-modules and predictors
        if self._tracking_enabled:
            self._wrap_submodules()

    def _wrap_submodules(self):
        """Recursively wrap all sub-modules and predictors for tracking."""
        # Wrap all named predictors
        for name, predictor in self._wrapped_module.named_predictors():
            if not isinstance(predictor, TrackedPredictor):
                wrapped = TrackedPredictor(predictor, name, self._run_id, self._iteration)
                # Replace the predictor in the module
                parts = name.split(".")
                obj = self._wrapped_module
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], wrapped)

        # Wrap all sub-modules
        for attr_name in dir(self._wrapped_module):
            if not attr_name.startswith("_"):
                attr = getattr(self._wrapped_module, attr_name)
                if isinstance(attr, dspy.Module) and not isinstance(attr, TrackedModule):
                    # Recursively wrap the sub-module
                    wrapped = TrackedModule(attr, self._run_id, self._iteration, self._depth + 1)
                    setattr(self._wrapped_module, attr_name, wrapped)

    def forward(self, *args, **kwargs):
        """Forward call with automatic MLflow tracking."""
        self._call_count += 1

        if not self._tracking_enabled:
            return self._wrapped_module.forward(*args, **kwargs)

        start_time = time.time()
        call_info = {
            "type": "module",
            "module_type": self._wrapped_module.__class__.__name__,
            "module_depth": self._depth,
            "call_count": self._call_count,
            "iteration": self._iteration,
            "timestamp": start_time,
        }

        # Log input info
        if args:
            call_info["args_count"] = len(args)
        if kwargs:
            call_info["kwargs_keys"] = list(kwargs.keys())

        try:
            # Execute the actual forward call
            result = self._wrapped_module.forward(*args, **kwargs)

            call_info["execution_time"] = time.time() - start_time
            call_info["status"] = "success"

            # Log output info
            if isinstance(result, dspy.Prediction):
                call_info["output_fields"] = list(result.keys())

            self._log_call(call_info)
            return result

        except Exception as e:
            call_info["execution_time"] = time.time() - start_time
            call_info["status"] = "error"
            call_info["error"] = str(e)
            self._log_call(call_info)
            raise

    def _log_call(self, call_info: dict[str, Any]):
        """Log module call to MLflow."""
        if not self._tracking_enabled:
            return

        try:
            # Log execution time as metric
            if "execution_time" in call_info:
                metric_name = f"module_{call_info['module_type']}_exec_time"
                step = call_info.get("call_count", 0)
                mlflow.log_metric(metric_name, call_info["execution_time"], step=step)

            # Log full trace as artifact
            artifact_name = f"module_traces_iter_{self._iteration}.jsonl" if self._iteration else "module_traces.jsonl"

            import os
            import tempfile

            with tempfile.NamedTemporaryFile(mode="a", suffix=f"_{artifact_name}", delete=False) as f:
                f.write(json.dumps(call_info) + "\n")
                temp_path = f.name

            mlflow.log_artifact(temp_path, artifact_path="execution_traces")
            os.unlink(temp_path)

        except Exception as e:
            logger.debug(f"Failed to log module call to MLflow: {e}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped module."""
        return getattr(self._wrapped_module, name)

    def update_iteration(self, iteration: int):
        """Update the iteration context for logging."""
        self._iteration = iteration
        # Update all wrapped predictors
        for _, predictor in self._wrapped_module.named_predictors():
            if isinstance(predictor, TrackedPredictor):
                predictor._iteration = iteration


def track_module(module: dspy.Module, run_id: str | None = None, iteration: int | None = None) -> dspy.Module:
    """Recursively wrap a module and all its sub-modules/predictors with tracking.

    Args:
        module: The module to wrap
        run_id: MLflow run ID (if tracking is enabled)
        iteration: Current iteration for context

    Returns:
        TrackedModule if tracking is enabled, otherwise returns the module unchanged
    """
    if MLFLOW_AVAILABLE and run_id:
        return TrackedModule(module, run_id, iteration)
    return module
