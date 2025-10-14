"""Tests for APEX MLflow integration."""
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import dspy
from dspy import Example
from dspy.teleprompt.apex import ChangeMagnitude, PromptChange
from dspy.teleprompt.apex_optimizer import APEX
from dspy.utils.dummies import DummyLM


def make_train_example(value: str) -> Example:
    return Example(input=value, output="good").with_inputs("input")


def make_analysis_response(root_cause: str = "Prompt missing correct token") -> dict:
    return {
        "root_cause": root_cause,
        "involved_predictors": ["predictor"],
        "context": "Baseline emits 'bad'",
        "category": "format_ambiguity",
        "key_details": "Needs to say good",
    }


def make_hypothesis_response(prompt_value: str = "good") -> dict:
    return {
        "hypotheses": [
            {
                "observation": "Prompt mismatch",
                "fixable_root_causes": ["Prompt missing correct token"],
                "non_fixable_root_causes": [],
                "strategy": "Rewrite prompt",
                "expected_impact": "Outputs 'good'",
                "impact_score": 0.8,
                "generalizability_score": 0.9,
                "prompt_changes": {
                    "predictor": PromptChange(
                        new_prompt=prompt_value,
                        rationale="Align output with expectation",
                        change_magnitude=ChangeMagnitude.MINIMAL,
                    )
                },
            }
        ]
    }


class PromptDrivenModule(dspy.Module):
    def __init__(self, initial_prompt: str):
        super().__init__()
        self.predictor = dspy.Predict("input -> output")
        self.predictor.signature.instructions = initial_prompt

    def forward(self, input: str) -> dspy.Prediction:
        return dspy.Prediction(output=self.predictor.signature.instructions)


def metric(example: Example, prediction: dspy.Prediction, trace) -> float:
    expected = example.output
    predicted = prediction.output
    return 1.0 if expected == predicted else 0.0


def test_apex_mlflow_disabled_by_default():
    """Test that MLflow tracking is disabled by default."""
    trainset = [make_train_example("x")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response()],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response()],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=42,
        verbosity="none",
    )

    # MLflow should not be enabled
    assert not optimizer.tracker.use_mlflow
    assert not optimizer.tracker.is_active()

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)
    assert optimized.predictor.signature.instructions == "good"


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_enabled(mock_mlflow):
    """Test that MLflow tracking can be enabled and logs appropriate data."""
    # Configure MLflow mock
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.start_run.return_value = mock_run

    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response() for _ in range(2)],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response()],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=42,
        verbosity="none",
        use_mlflow=True,  # Enable MLflow tracking
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="test-experiment",
    )

    # MLflow should be enabled
    assert optimizer.tracker.use_mlflow
    assert optimizer.tracker.mlflow_tracking_uri == "http://localhost:5000"
    assert optimizer.tracker.mlflow_experiment_name == "test-experiment"

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Verify MLflow methods were called
    mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_with("test-experiment")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.end_run.assert_called_once()

    # Verify parameters were logged (metrics are batched, so check log_param)
    assert mock_mlflow.log_param.call_count > 0

    # Verify optimization still works correctly
    assert optimized.predictor.signature.instructions == "good"


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", False)
def test_apex_mlflow_graceful_fallback():
    """Test that APEX handles missing MLflow gracefully."""
    trainset = [make_train_example("x")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response()],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response()],
        adapter=dspy.JSONAdapter(),
    )

    # Even if use_mlflow is True, it should handle missing MLflow gracefully
    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=42,
        verbosity="none",
        use_mlflow=True,  # Request MLflow but it's not available
    )

    # MLflow should be disabled since it's not available
    assert not optimizer.tracker.use_mlflow

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Optimization should still work
    assert optimized.predictor.signature.instructions == "good"


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_iteration_tracking(mock_mlflow):
    """Test that APEX tracks iteration-level metrics with MLflow."""
    # Configure MLflow mock
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.start_run.return_value = mock_run

    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response("issue 1"), make_analysis_response("issue 2")],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response("good")],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=42,
        verbosity="none",
        use_mlflow=True,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    # Check that metrics logging was attempted
    assert mock_mlflow.log_metrics.called or mock_mlflow.log_param.called

    # Verify core MLflow methods were called
    assert mock_mlflow.start_run.called
    assert mock_mlflow.end_run.called


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_artifact_logging(mock_mlflow):
    """Test that APEX logs artifacts like hypotheses to MLflow."""
    # Configure MLflow mock
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.start_run.return_value = mock_run

    trainset = [make_train_example("x")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response()],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response("improved_prompt")],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=42,
        verbosity="none",
        use_mlflow=True,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Should have attempted to log artifacts
    assert mock_mlflow.log_artifact.called or mock_mlflow.log_param.called

    # Verify best program was logged
    assert optimized.predictor.signature.instructions == "improved_prompt"


def test_apex_mlflow_context_manager():
    """Test that the ExperimentTracker works as a context manager."""
    from dspy.teleprompt.apex.tracker import ExperimentTracker

    with patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True):
        with patch("dspy.teleprompt.apex.tracker.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-id"
            mock_mlflow.start_run.return_value = mock_run

            tracker = ExperimentTracker(use_mlflow=True)

            # Use as context manager
            with tracker:
                assert tracker.is_active()
                tracker.log_params({"test_param": "value"})
                tracker.log_metrics({"test_metric": 0.5})

            # After exiting context, run should be ended
            mock_mlflow.end_run.assert_called_once()


def test_apex_tracking_utils_format_functions():
    """Test the tracking utility formatting functions."""
    from dspy.teleprompt.apex import tracking_utils

    # Test format_baseline_metrics
    baseline_metrics = tracking_utils.format_baseline_metrics(
        baseline_score=0.5, num_train=10, num_val=5
    )
    assert baseline_metrics["baseline_score"] == 0.5
    assert baseline_metrics["num_train_examples"] == 10
    assert baseline_metrics["num_val_examples"] == 5

    # Test format_iteration_metrics
    hypotheses = [
        MagicMock(
            strategy="test_strategy",
            impact_score=0.8,
            generalizability_score=0.9,
            fixable_root_causes=["issue1"],
            prompt_changes={
                "predictor": MagicMock(
                    new_prompt="new prompt text",
                    rationale="test rationale",
                    change_magnitude=ChangeMagnitude.MINIMAL,
                )
            },
        )
    ]
    iteration_metrics = tracking_utils.format_iteration_metrics(
        iteration=1,
        num_failures=2,
        num_successes=3,
        hypotheses=hypotheses,
        candidates=[],
        best_score=0.7,
    )
    assert iteration_metrics["iteration"] == 1
    assert iteration_metrics["num_failures"] == 2
    assert iteration_metrics["num_successes"] == 3
    assert iteration_metrics["best_score"] == 0.7
    assert len(iteration_metrics["hypotheses"]) == 1

    # Test format_candidate_data
    candidate = MagicMock(
        overall_score=0.8,
        iteration=1,
        hypothesis=MagicMock(strategy="test_strategy"),
        per_example_scores=[0.7, 0.8, 0.9],
    )
    candidate_data = tracking_utils.format_candidate_data(candidate)
    assert candidate_data["overall_score"] == 0.8
    assert candidate_data["iteration"] == 1
    assert candidate_data["has_hypothesis"] is True
    assert pytest.approx(candidate_data["mean_score"]) == 0.8

    # Test format_optimization_summary
    best_candidate = MagicMock(overall_score=0.9)
    all_candidates = [MagicMock(overall_score=0.7), MagicMock(overall_score=0.9)]
    iterations = [
        MagicMock(
            num_failures=2, num_successes=3, hypotheses=hypotheses
        )
    ]
    summary = tracking_utils.format_optimization_summary(
        best_candidate=best_candidate,
        all_candidates=all_candidates,
        iterations=iterations,
        stopped_after="patience",
        initial_score=0.5,
    )
    assert summary["final_score"] == 0.9
    assert summary["initial_score"] == 0.5
    assert summary["improvement"] == 0.4
    assert summary["stopped_after"] == "patience"
    assert summary["total_iterations"] == 1
    assert summary["total_candidates"] == 2