"""Unit tests covering the primary Pydantic models exposed by ``dspy.teleprompt.apex``."""

from __future__ import annotations

import dspy
from dspy import Example
from dspy.teleprompt.apex.models import (
    ApexCheckpoint,
    ApexIterationLog,
    ApexOptimizationResult,
    CandidateRecord,
    ChangeMagnitude,
    CheckpointConfig,
    ExecutionFlowEntry,
    HypothesisSpec,
    PromptChange,
    TrainExampleRecord,
)


class _ConstantModule(dspy.Module):
    """Minimal module used to satisfy the ``Module`` requirement in ``CandidateRecord``."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self._value = value

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return dspy.Prediction(output=self._value)


def test_prompt_change_and_hypothesis_spec_defaults_are_independent() -> None:
    prompt_change = PromptChange(
        new_prompt="new instructions",
        change_summary="Provide clearer structure",
        change_magnitude=ChangeMagnitude.MODERATE,
    )

    hypothesis = HypothesisSpec(
        observation="Baseline misses structured reasoning",
        fixable_root_causes=["missing reasoning"],
        strategy="Add explicit reasoning steps",
        expected_impact="Improves structured answers",
        prompt_changes={"predict": prompt_change},
    )

    assert hypothesis.prompt_changes["predict"].change_magnitude is ChangeMagnitude.MODERATE
    assert hypothesis.non_fixable_root_causes == []
    # Mutating defaults on one instance must not affect new instances.
    hypothesis.fixable_root_causes.append("missing verification")
    other = HypothesisSpec(
        observation="Independent hypothesis",
        strategy="No-op",
        expected_impact="None",
    )

    assert other.fixable_root_causes == []


def test_candidate_record_accepts_modules_and_scores() -> None:
    module = _ConstantModule("value")
    candidate = CandidateRecord(
        program=module,
        overall_score=0.75,
        per_example_scores=[0.5, 1.0],
        iteration=2,
        hypothesis=None,
    )

    assert candidate.overall_score == 0.75
    assert candidate.per_example_scores == [0.5, 1.0]
    assert candidate.program is module


def test_apex_checkpoint_round_trip_serialization() -> None:
    module = _ConstantModule("baseline")
    candidate = CandidateRecord(
        program=module,
        overall_score=1.0,
        per_example_scores=[1.0],
        iteration=0,
        hypothesis=None,
    )
    iteration_log = ApexIterationLog(
        iteration=0,
        sampled_train_size=1,
        num_failures=0,
        num_successes=1,
        hypotheses=[],
        candidates=[candidate],
    )
    result = ApexOptimizationResult(
        best_candidate=candidate,
        all_candidates=[candidate],
        iterations=[iteration_log],
        stopped_after="success",
    )
    checkpoint = ApexCheckpoint(
        iteration=1,
        current_program=module,
        best_candidate=candidate,
        all_candidates=[candidate],
        iteration_logs=[iteration_log],
        no_improvement_count=0,
        iteration_baseline=candidate,
        rng_state={"state": 123},
        config=CheckpointConfig(
            max_iterations=10,
            num_hypotheses=3,
            num_eval_runs=2,
            train_sample=None,
            success_threshold=0.9,
            min_metric=0.0,
            max_metric=1.0,
            convergence_patience=2,
            seed=42,
            candidate_selection="best_on_val",
            pareto_merge_probability=0.1,
        ),
    )

    dumped = checkpoint.model_dump()
    restored = ApexCheckpoint.model_validate(dumped)

    assert restored.iteration == 1
    assert restored.best_candidate.overall_score == 1.0
    assert restored.config.num_hypotheses == 3
    # Ensure the optimization result structure serializes without error as well.
    assert result.best_candidate.iteration == 0


def test_train_example_record_supports_prediction_feedback() -> None:
    example = Example(input="x", output="y").with_inputs("input")
    record = TrainExampleRecord(
        example=example,
        prediction=dspy.Prediction(output="y"),
        metric_score=1.0,
        metric_feedback=None,
        is_success=True,
        error=None,
    )

    dumped = record.model_dump()

    assert dumped["example"].output == "y"
    assert dumped["prediction"].output == "y"


def test_execution_flow_entry_new_fields_default_values() -> None:
    entry = ExecutionFlowEntry(
        predictor_name="test_predictor",
        predictor_type="Predict",
        inputs='{"input": "test"}',
        outputs='{"output": "result"}',
        instructions="Test instructions",
    )

    assert entry.executed is True
    assert entry.execution_order is None
    assert entry.predictor_name == "test_predictor"


def test_execution_flow_entry_with_execution_tracking() -> None:
    entry = ExecutionFlowEntry(
        predictor_name="tracked_predictor",
        predictor_type="ChainOfThought",
        inputs='{"query": "test"}',
        outputs='{"answer": "response"}',
        instructions="Process the query",
        dependencies=["previous_predictor"],
        input_sources={"query": ["previous_predictor"]},
        executed=False,
        execution_order=5,
    )

    assert entry.executed is False
    assert entry.execution_order == 5
    assert entry.predictor_name == "tracked_predictor"
    assert entry.dependencies == ["previous_predictor"]


def test_execution_flow_entry_serialization_with_new_fields() -> None:
    original = ExecutionFlowEntry(
        predictor_name="serializable_predictor",
        predictor_type="Predict",
        inputs='{"x": 1}',
        outputs='{"y": 2}',
        instructions="Execute task",
        dependencies=["dep1", "dep2"],
        input_sources={"x": ["dep1"]},
        executed=True,
        execution_order=3,
    )

    dumped = original.model_dump()
    restored = ExecutionFlowEntry.model_validate(dumped)

    assert restored.predictor_name == "serializable_predictor"
    assert restored.executed is True
    assert restored.execution_order == 3
    assert restored.dependencies == ["dep1", "dep2"]
    assert restored.input_sources == {"x": ["dep1"]}
