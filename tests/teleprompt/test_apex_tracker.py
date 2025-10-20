"""Unit tests for the lightweight MLflow tracker used by APEX."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dspy.teleprompt.apex import tracker as tracker_module
from dspy.teleprompt.apex.tracker import ExperimentTracker


class _DummySpan:
    def __init__(self) -> None:
        self.inputs: Mapping[str, object] | None = None

    def __enter__(self) -> "_DummySpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - nothing to clean up
        return None

    def set_inputs(self, inputs: Mapping[str, object]) -> None:
        self.inputs = inputs


class _DummyMLflow:
    def __init__(self) -> None:
        self.started = False
        self.ended: str | None = None
        self.logged_params: list[tuple[str, str]] = []
        self.logged_metrics: list[tuple[str, float, int | None]] = []
        self.logged_artifacts: list[tuple[str, str | None]] = []
        self.autolog_kwargs: dict[str, object] | None = None
        self.disabled_autolog = False
        self.dspy = self

    # ------------------------------------------------------------------ run lifecycle
    def start_run(self) -> None:
        self.started = True

    def end_run(self, status: str) -> None:
        self.ended = status

    # ------------------------------------------------------------------ experiment config
    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def get_tracking_uri(self) -> str:
        return "http://mlflow.test"

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    # ------------------------------------------------------------------ logging helpers
    def log_param(self, key: str, value: str) -> None:
        self.logged_params.append((key, value))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.logged_metrics.append((key, value, step))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.logged_artifacts.append((path, artifact_path))

    def autolog(self, *, disable: bool = False, silent: bool = True, **kwargs) -> None:
        if disable:
            self.disabled_autolog = True
        else:
            self.autolog_kwargs = kwargs

    # ------------------------------------------------------------------ tracing API
    def start_span(self, name: str, **kwargs) -> _DummySpan:
        span = _DummySpan()
        span.name = name  # type: ignore[attr-defined]
        span.attributes = kwargs.get("attributes")  # type: ignore[attr-defined]
        return span


@pytest.fixture
def dummy_mlflow(monkeypatch: pytest.MonkeyPatch) -> _DummyMLflow:
    stub = _DummyMLflow()
    monkeypatch.setattr(tracker_module, "mlflow", stub)
    monkeypatch.setattr(tracker_module, "MLFLOW_AVAILABLE", True)
    return stub


def test_experiment_tracker_uses_mlflow_when_available(dummy_mlflow: _DummyMLflow) -> None:
    tracker = ExperimentTracker(
        use_mlflow=True,
        mlflow_tracking_uri="http://example",  # validates scheme
        mlflow_experiment_name="apex-tests",
    )

    with tracker as active:
        assert active.is_tracing_enabled()
        active.log_params({"iterations": 3, "unused": None})
        active.log_metrics({"score": 0.9, "bad": float("nan")}, step=2)
        active.log_artifact_text("payload", filename="summary.json")
        with active.span("apex.unit", inputs={"foo": "bar"}) as span:
            assert span.inputs == {"foo": "bar"}

    assert dummy_mlflow.started is True
    assert dummy_mlflow.ended == "FINISHED"
    assert dummy_mlflow.autolog_kwargs == {
        "log_traces": True,
        "log_traces_from_compile": True,
        "log_traces_from_eval": True,
        "log_compiles": False,
        "log_evals": False,
    }
    assert dummy_mlflow.logged_params == [("iterations", "3")]
    assert dummy_mlflow.logged_metrics == [("score", 0.9, 2)]
    assert dummy_mlflow.logged_artifacts  # ensure artifact logging occurred
    assert dummy_mlflow.disabled_autolog is True


def test_experiment_tracker_without_mlflow_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracker_module, "MLFLOW_AVAILABLE", False)
    monkeypatch.setattr(tracker_module, "mlflow", None)
    tracker = ExperimentTracker(use_mlflow=True)

    with tracker as active:
        assert not active.is_tracing_enabled()
        with active.span("noop") as span:
            assert span is None
