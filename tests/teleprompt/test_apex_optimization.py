"""Unit tests for the optimization loop that powers APEX."""

from __future__ import annotations

import random
from typing import Callable
from unittest import mock

import pytest

import dspy
from dspy import Example
from dspy.teleprompt.apex.candidate_selection import SelectionResult
from dspy.teleprompt.apex.models import (
    CandidateRecord,
    ChangeMagnitude,
    HypothesisSpec,
    PromptChange,
)
from dspy.teleprompt.apex.optimization import (
    IterationContext,
    IterationResult,
    LoopCollaborators,
    LoopSettings,
    OptimizationLoop,
)
from dspy.teleprompt.apex.state import OptimizationState
from dspy.teleprompt.apex.types import Verbosity


class SimpleModule(dspy.Module):
    """Minimal module used across optimization tests."""

    def forward(self, **kwargs):  # type: ignore[override]
        return dspy.Prediction(**kwargs)


def make_candidate(program: dspy.Module, *, score: float, iteration: int) -> CandidateRecord:
    return CandidateRecord(
        program=program,
        overall_score=score,
        per_example_scores=[score],
        iteration=iteration,
    )


def make_loop(
    *,
    analysis_fn: Callable | None = None,
    candidate_selection: str = "best_on_val",
    pareto_merge_probability: float = 0.0,
    convergence_patience: int | None = 1,
    is_enabled: Callable[[Verbosity], bool] | None = None,
) -> OptimizationLoop:
    settings = LoopSettings(
        max_iterations=None,
        convergence_patience=convergence_patience,
        train_sample=None,
        num_hypotheses=1,
        include_hypothesis_history=False,
        candidate_selection=candidate_selection,  # type: ignore[arg-type]
        pareto_merge_probability=pareto_merge_probability,
        success_threshold=0.8,
        min_metric=0.0,
        max_metric=1.0,
    )

    tracker = mock.Mock()
    tracker.is_active.return_value = False

    collaborators = LoopCollaborators(
        runtime=mock.Mock(),
        evaluator=mock.Mock(),
        tracker=tracker,
        checkpoints=mock.Mock(),
        rng=random.Random(0),
        build_checkpoint_config=lambda: mock.sentinel.checkpoint_config,
        log=mock.Mock(),
        is_enabled=is_enabled or (lambda _: False),
        analysis_lm=mock.Mock(),
        analysis_adapter=mock.Mock(),
        hypothesis_lm=mock.Mock(),
        hypothesis_adapter=mock.Mock(),
        analysis_fn=analysis_fn,
        format_execution_flow=mock.Mock(),
        generate_hypotheses=mock.Mock(),
        generate_merge_hypotheses=mock.Mock(),
    )
    collaborators.checkpoints.enabled = False

    return OptimizationLoop(settings, collaborators)


def make_hypothesis(label: str) -> HypothesisSpec:
    return HypothesisSpec(
        observation=f"Observation {label}",
        fixable_root_causes=["missing step"],
        non_fixable_root_causes=[],
        impact_score=0.5,
        generalizability_score=0.4,
        strategy=f"Strategy {label}",
        expected_impact="Fixes observed issues",
        prompt_changes={
            "predictor": PromptChange(
                new_prompt=f"Prompt {label}",
                change_summary="Adds reasoning",
                change_magnitude=ChangeMagnitude.MODERATE,
            )
        },
    )


def test_run_analysis_uses_custom_hook_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = make_loop(analysis_fn=lambda *args, **kwargs: None)

    calls: list[tuple[str, int]] = []

    def fake_analysis(record, *, mode: str, iteration: int, example_index: int, **_: object):
        calls.append((mode, example_index))
        return {"record": record, "mode": mode, "iteration": iteration, "index": example_index}

    loop.cb.analysis_fn = fake_analysis

    analyzer = mock.Mock(side_effect=AssertionError("Batch analyzer should not run"))
    monkeypatch.setattr(
        "dspy.teleprompt.apex.optimization.analyze_failures_and_successes",
        analyzer,
    )

    failures = [mock.Mock(name="failure0"), mock.Mock(name="failure1")]
    successes = [mock.Mock(name="success0")]

    failure_summaries, success_summaries = loop._run_analysis(
        failures=failures,
        successes=successes,
        iteration=3,
        available_predictor_names=["predictor"],
    )

    assert len(failure_summaries) == 2
    assert len(success_summaries) == 1
    assert calls == [("failure", 0), ("failure", 1), ("success", 0)]
    analyzer.assert_not_called()


def test_run_analysis_uses_batch_pipeline_when_hook_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = make_loop(analysis_fn=None)

    expected_failures = [mock.sentinel.failure_summary]
    expected_successes = [mock.sentinel.success_summary]

    analyzer = mock.Mock(return_value=(expected_failures, expected_successes))
    monkeypatch.setattr(
        "dspy.teleprompt.apex.optimization.analyze_failures_and_successes",
        analyzer,
    )

    failure_summaries, success_summaries = loop._run_analysis(
        failures=[mock.Mock()],
        successes=[mock.Mock()],
        iteration=1,
        available_predictor_names=["predictor"],
    )

    assert failure_summaries == expected_failures
    assert success_summaries == expected_successes
    analyzer.assert_called_once()


def test_maybe_generate_merge_hypotheses_returns_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = make_loop(
        candidate_selection="pareto",
        pareto_merge_probability=1.0,
        convergence_patience=None,
    )

    baseline_program = SimpleModule()
    partner_program = SimpleModule()
    pareto_baseline = make_candidate(baseline_program, score=0.6, iteration=1)
    partner_candidate = make_candidate(partner_program, score=0.7, iteration=2)

    selection = SelectionResult(
        baseline=pareto_baseline,
        frontier=[pareto_baseline, partner_candidate],
        weights=[0.4, 0.6],
    )

    hypotheses = [make_hypothesis("A"), make_hypothesis("B")]
    loop.cb.generate_merge_hypotheses.return_value = hypotheses

    monkeypatch.setattr(
        "dspy.teleprompt.apex.optimization.draw_weighted_candidate",
        lambda *args, **kwargs: partner_candidate,
    )

    merge_hypotheses, overrides = loop._maybe_generate_merge_hypotheses(
        iteration=4,
        pareto_baseline=pareto_baseline,
        selection_result=selection,
    )

    assert merge_hypotheses == hypotheses
    assert overrides == {id(hypotheses[1]): partner_candidate.program}
    loop.cb.generate_merge_hypotheses.assert_called_once_with(
        baseline_candidate=pareto_baseline,
        partner_candidate=partner_candidate,
        runtime=loop.cb.runtime,
        hypothesis_lm=loop.cb.hypothesis_lm,
        hypothesis_adapter=loop.cb.hypothesis_adapter,
        iteration=4,
        tracker=loop.cb.tracker,
    )


def test_handle_perfect_performance_respects_convergence_patience() -> None:
    loop = make_loop(convergence_patience=1)

    baseline_program = SimpleModule()
    baseline_candidate = make_candidate(baseline_program, score=0.5, iteration=0)
    state = OptimizationState.initialize(program=baseline_program, baseline=baseline_candidate)

    sampled_train = [Example(question="q", answer="a").with_inputs("question")]

    result = loop._handle_perfect_performance(
        state=state,
        iteration=2,
        sampled_train=sampled_train,
        success_count=len(sampled_train),
    )

    assert isinstance(result, IterationResult)
    assert result.stop is True
    assert result.skip_to_next is False
    assert state.no_improvement_count == 1
    assert state.stop_reason == "patience"
    assert len(state.iteration_logs) == 1
    loop.cb.checkpoints.save.assert_not_called()
    loop.cb.log.assert_any_call(
        "APEX: Iteration 2 - Perfect performance! All 1 examples succeeded",
        Verbosity.NORMAL,
    )


def test_handle_interrupt_appends_iteration_log_and_saves_checkpoint() -> None:
    loop = make_loop()
    loop.cb.checkpoints.enabled = True

    baseline_program = SimpleModule()
    baseline_candidate = make_candidate(baseline_program, score=0.4, iteration=0)
    state = OptimizationState.initialize(program=baseline_program, baseline=baseline_candidate)
    state.iteration = 3

    partner_program = SimpleModule()
    partner_candidate = make_candidate(partner_program, score=0.6, iteration=3)

    context = IterationContext(
        iteration=3,
        sampled_train=[Example(question="q", answer="a").with_inputs("question")],
        failure_summaries=[mock.sentinel.failure_summary],
        success_summaries=[mock.sentinel.success_summary],
        hypotheses=[make_hypothesis("context")],
        candidates=[partner_candidate],
    )

    loop._handle_interrupt(state, context)

    assert state.stop_reason == "interrupted"
    assert len(state.iteration_logs) == 1
    log_entry = state.iteration_logs[0]
    assert log_entry.iteration == 3
    assert log_entry.num_failures == 1
    assert log_entry.num_successes == 1
    assert log_entry.candidates == [partner_candidate]
    loop.cb.checkpoints.save.assert_called_once()
    kwargs = loop.cb.checkpoints.save.call_args.kwargs
    assert kwargs["iteration"] == state.iteration
    loop.cb.log.assert_any_call(
        "APEX: Optimization interrupted by user (Ctrl+C)",
        Verbosity.NORMAL,
    )
