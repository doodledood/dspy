"""Unit tests for the lifecycle helpers used by the APEX optimizer."""

from __future__ import annotations

import random
from unittest import mock

import pytest

import dspy
from dspy import Example
from dspy.teleprompt.apex.lifecycle import (
    finalize_optimization,
    initialize_new_state,
    resume_from_checkpoint,
)
from dspy.teleprompt.apex.models import (
    ApexCheckpoint,
    ApexIterationLog,
    CandidateRecord,
    CheckpointConfig,
)
from dspy.teleprompt.apex.state import OptimizationState
from dspy.teleprompt.apex.tracker import ExperimentTracker
from dspy.teleprompt.apex.types import Verbosity


class SimpleModule(dspy.Module):
    """Minimal module used for lifecycle tests."""

    def forward(self, **kwargs):  # type: ignore[override]
        return dspy.Prediction(**kwargs)


def make_candidate(program: dspy.Module, *, score: float, iteration: int) -> CandidateRecord:
    return CandidateRecord(
        program=program,
        overall_score=score,
        per_example_scores=[score],
        iteration=iteration,
    )


def test_resume_from_checkpoint_updates_configuration() -> None:
    base_program = SimpleModule()
    baseline = make_candidate(base_program, score=0.2, iteration=0)
    checkpoint_rng = random.Random(123)
    checkpoint = ApexCheckpoint(
        iteration=3,
        current_program=base_program,
        best_candidate=baseline,
        all_candidates=[baseline],
        iteration_logs=[],
        no_improvement_count=1,
        iteration_baseline=baseline,
        rng_state=checkpoint_rng.getstate(),
        config=CheckpointConfig(
            max_iterations=10,
            num_hypotheses=1,
            num_eval_runs=1,
            train_sample=5,
            success_threshold=1.0,
            min_metric=0.0,
            max_metric=1.0,
            convergence_patience=2,
            seed=7,
            candidate_selection="pareto",
            pareto_merge_probability=0.4,
        ),
    )

    log = mock.Mock()
    rng = random.Random(999)

    state, selection, merge_probability = resume_from_checkpoint(
        checkpoint=checkpoint,
        log=log,
        rng=rng,
        candidate_selection="best_on_val",
        pareto_merge_probability=0.1,
    )

    assert state.iteration == 3
    assert selection == "pareto"
    assert merge_probability == pytest.approx(0.4)
    assert rng.getstate() == checkpoint.rng_state
    log.assert_any_call("APEX: Resuming from iteration 3", Verbosity.NORMAL)


def test_initialize_new_state_evaluates_baseline_and_saves_checkpoint() -> None:
    student = SimpleModule()
    trainset = [Example(question="q", answer="a").with_inputs("question")]
    valset = [Example(question="q", answer="a").with_inputs("question")]

    baseline_program = student.deepcopy()
    baseline_candidate = make_candidate(baseline_program, score=0.5, iteration=0)

    evaluator = mock.Mock()
    evaluator.evaluate_candidate.return_value = baseline_candidate

    tracker = mock.Mock(spec=ExperimentTracker)
    tracker.is_active.return_value = True

    checkpoints = mock.Mock()

    config = CheckpointConfig(
        max_iterations=5,
        num_hypotheses=1,
        num_eval_runs=1,
        train_sample=1,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        convergence_patience=1,
        seed=3,
    )

    def build_config() -> CheckpointConfig:
        return config

    log = mock.Mock()
    rng = random.Random(42)

    state = initialize_new_state(
        student=student,
        trainset=trainset,
        valset=valset,
        evaluator=evaluator,
        tracker=tracker,
        checkpoints=checkpoints,
        build_checkpoint_config=build_config,
        log=log,
        rng=rng,
        num_threads=2,
        max_iterations=5,
        num_hypotheses=1,
        success_threshold=1.0,
        convergence_patience=1,
        seed=7,
    )

    evaluator.evaluate_candidate.assert_called_once()
    assert state.current_program is not student
    assert state.iteration_baseline is baseline_candidate
    tracker.log_metrics.assert_called_once()
    checkpoints.save.assert_called_once()
    kwargs = checkpoints.save.call_args.kwargs
    assert kwargs["iteration"] == 0
    assert kwargs["best_candidate"] is baseline_candidate
    assert kwargs["config"] is config


def test_finalize_optimization_sets_result_and_logs() -> None:
    baseline_program = SimpleModule()
    baseline_candidate = make_candidate(baseline_program, score=0.2, iteration=0)
    state = OptimizationState.initialize(program=baseline_program, baseline=baseline_candidate)

    best_program = baseline_program.deepcopy()
    best_candidate = make_candidate(best_program, score=0.9, iteration=2)
    state.best_candidate = best_candidate
    state.all_candidates.append(best_candidate)
    state.iteration_logs.append(
        ApexIterationLog(
            iteration=1,
            sampled_train_size=1,
            num_failures=1,
            num_successes=0,
            hypotheses=[],
            candidates=[baseline_candidate, best_candidate],
        )
    )
    state.iteration = 2
    state.stop_reason = ""

    tracker = mock.Mock(spec=ExperimentTracker)
    tracker.is_active.return_value = True

    def is_enabled(level: Verbosity) -> bool:
        return True

    log = mock.Mock()
    print_fn = mock.Mock()

    optimized = finalize_optimization(
        state,
        log=log,
        is_enabled=is_enabled,
        candidate_selection="pareto",
        pareto_merge_probability=0.3,
        tracker=tracker,
        print_fn=print_fn,
    )

    assert state.stop_reason == "completed"
    assert optimized is best_candidate.program
    assert getattr(optimized, "_compiled", False) is True
    assert hasattr(optimized, "apex_result")
    tracker.log_metrics.assert_called_once()
    print_fn.assert_called_once()
    log.assert_any_call(mock.ANY, Verbosity.NORMAL)
