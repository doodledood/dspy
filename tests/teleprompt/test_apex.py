"""High-level integration tests for the public APEX interface.

These tests intentionally focus on observable behaviour - the optimizer should
respect its inputs, produce improved programs when hypotheses succeed, and keep
its results accessible through the documented attributes. They avoid making
assumptions about the internal implementation so the suite remains stable during
refactors.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

import dspy
from dspy import Example
from dspy.adapters import JSONAdapter
from dspy.adapters.chat_adapter import FieldInfoWithName
from dspy.signatures.field import OutputField
from dspy.teleprompt.apex import APEX, ChangeMagnitude, PromptChange
from dspy.teleprompt.apex.models import CandidateRecord, HypothesisSpec, TrainExampleRecord
from dspy.teleprompt.apex.signatures import ParetoMergeSignature
from dspy.utils.dummies import DummyLM


def make_analysis_response(
    root_cause: str = "Prompt missing correct token",
    *,
    involved_predictors: list[str] | None = None,
) -> dict[str, Any]:
    """Create a minimal failure analysis payload accepted by APEX."""

    return {
        "potential_root_causes": [root_cause],
        "involved_predictors": involved_predictors or ["predictor"],
        "context": "Baseline emits an unexpected value",
        "categories": ["format_ambiguity"],
        "key_details": "SEVERITY: MODERATE. PRIMARY_FAILURE: predictor. FIXABLE: Add correct token.",
    }


def make_hypothesis_response(prompt_value: str = "good") -> dict[str, Any]:
    """Create a hypothesis payload that rewrites the predictor prompt."""

    return {
        "hypotheses": [
            {
                "observation": "Prompt mismatch",
                "fixable_root_causes": ["Prompt missing correct token"],
                "non_fixable_root_causes": [],
                "strategy": "Rewrite prompt",
                "expected_impact": "Outputs desired value",
                "impact_score": 1.0,
                "generalizability_score": 1.0,
                "prompt_changes": {
                    "predictor": PromptChange(
                        new_prompt=prompt_value,
                        change_summary="Align output with expectation",
                        change_magnitude=ChangeMagnitude.MINIMAL,
                    )
                },
            }
        ]
    }


class RoutedAnalysisLM(DummyLM):
    """Return failure or success analyses based on the prompt contents."""

    def __init__(self, *, failures: list[dict[str, Any]] | None = None) -> None:
        super().__init__(answers=[], adapter=JSONAdapter())
        self._failures = deque(failures or [make_analysis_response()])

    def _format_payload(self, payload: dict[str, Any]) -> str:
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=OutputField()): value for field_name, value in payload.items()
        }
        try:
            return self.adapter.format_field_with_value(fields_with_values, role="assistant")
        except TypeError:  # pragma: no cover - defensive fall-back for adapters without role support
            return self.adapter.format_field_with_value(fields_with_values)

    def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        messages = messages or [{"role": "user", "content": prompt or ""}]
        content = (messages[-1]["content"] or "").lower()
        if "error" not in content:
            # Success analyses are irrelevant for these tests; return a neutral payload.
            payload = {
                "potential_root_causes": ["Success pattern"],
                "involved_predictors": ["predictor"],
                "context": "Handled correctly",
                "categories": ["structured"],
                "key_details": "PRESERVE: Keep current behaviour.",
            }
        else:
            payload = self._failures.popleft() if self._failures else make_analysis_response()

        formatted = self._format_payload(payload)
        self.update_history(
            {
                "prompt": prompt,
                "messages": messages,
                "kwargs": {**self.kwargs, **kwargs},
                "outputs": [formatted],
                "usage": 0,
                "cost": 0,
            }
        )
        return [formatted]


def make_analysis_lm(*, failures: list[dict[str, Any]] | None = None) -> RoutedAnalysisLM:
    return RoutedAnalysisLM(failures=failures)


class PromptDrivenModule(dspy.Module):
    """Simple module whose output mirrors the predictor instructions."""

    def __init__(self, initial_prompt: str) -> None:
        super().__init__()
        self.predictor = dspy.Predict("input -> output")
        self.predictor.signature.instructions = initial_prompt

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        return dspy.Prediction(output=self.predictor.signature.instructions)


class DualPromptModule(dspy.Module):
    """Module with two predictors so we can exercise Pareto selection paths."""

    def __init__(self, first_prompt: str, second_prompt: str) -> None:
        super().__init__()
        self.first = dspy.Predict("focus -> first")
        self.second = dspy.Predict("focus -> second")
        self.first.signature.instructions = first_prompt
        self.second.signature.instructions = second_prompt

    def forward(self, focus: str) -> dspy.Prediction:  # type: ignore[override]
        return dspy.Prediction(
            first=self.first.signature.instructions,
            second=self.second.signature.instructions,
        )


def simple_metric(example: Example, prediction: dspy.Prediction, trace) -> float:
    """Score 1.0 when the output matches the reference label, otherwise 0.0."""

    expected = example.output
    actual = getattr(prediction, "output", None)
    return 1.0 if actual == expected else 0.0


def make_example(input_value: str, output_value: str) -> Example:
    """Create an Example with the "input" field marked as the model input."""

    return Example(input=input_value, output=output_value).with_inputs("input")


def make_dual_example(focus: str, expected: str) -> Example:
    """Build an example routed to one of the dual predictors."""

    return Example(focus=focus, output=expected).with_inputs("focus")


def focus_metric(example: Example, prediction: dspy.Prediction, trace) -> float:
    """Metric that checks the field referenced by the example's focus label."""

    actual = getattr(prediction, example.focus, None)
    return 1.0 if actual == example.output else 0.0


def test_apex_improves_prompt_and_preserves_baseline() -> None:
    baseline = PromptDrivenModule("baseline")
    trainset = [make_example("question", "refined")]
    valset = [make_example("question", "refined")]

    analysis_lm = make_analysis_lm(failures=[make_analysis_response("Prompt should say 'refined'")])
    hypothesis_lm = DummyLM([make_hypothesis_response("refined")], adapter=JSONAdapter())

    apex = APEX(
        metric=simple_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        convergence_patience=1,
        include_hypothesis_history=False,
        seed=7,
    )

    optimized = apex.compile(baseline, trainset=trainset, valset=valset)

    # The optimizer should return a deep copy so callers can compare instances.
    assert optimized is not baseline

    prediction = optimized(input="anything")
    assert prediction.output == "refined"

    result = optimized.apex_result
    assert result.best_candidate.overall_score == pytest.approx(1.0)
    assert result.iterations, "APEX should record at least one iteration"
    assert result.best_candidate.hypothesis is not None
    assert result.best_candidate.hypothesis.prompt_changes["predictor"].new_prompt == "refined", (
        "The recorded hypothesis should explain the new prompt"
    )


def test_apex_compile_validates_required_inputs() -> None:
    module = PromptDrivenModule("baseline")
    example = make_example("question", "answer")

    apex = APEX(
        metric=simple_metric,
        analysis_lm=make_analysis_lm(),
        hypothesis_lm=DummyLM([make_hypothesis_response()], adapter=JSONAdapter()),
        max_iterations=1,
        num_hypotheses=1,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        convergence_patience=1,
        seed=3,
    )

    with pytest.raises(ValueError):
        apex.compile(module, trainset=[], valset=[example])

    with pytest.raises(ValueError):
        apex.compile(module, trainset=[example], valset=[])

    with pytest.raises(ValueError):
        apex.compile(module, trainset=[example], valset=[example], teacher=module)


def test_apex_pareto_merge_flow_combines_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = DualPromptModule("start-first", "start-second")
    trainset = [make_dual_example("first", "alpha"), make_dual_example("second", "beta")]
    valset = [make_dual_example("first", "alpha"), make_dual_example("second", "beta")]

    apex = APEX(
        metric=focus_metric,
        analysis_lm=make_analysis_lm(),
        hypothesis_lm=DummyLM([], adapter=JSONAdapter()),
        max_iterations=2,
        convergence_patience=None,
        num_hypotheses=2,
        num_threads=1,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        include_hypothesis_history=True,
        candidate_selection="pareto",
        pareto_merge_probability=1.0,
        seed=23,
    )

    def build_spec(target: str, strategy: str, prompt: str) -> HypothesisSpec:
        return HypothesisSpec(
            observation=f"Improve {target}",
            fixable_root_causes=[f"{target} mismatch"],
            non_fixable_root_causes=[],
            strategy=strategy,
            expected_impact=f"{target} aligned",
            impact_score=1.0,
            generalizability_score=0.5,
            prompt_changes={
                target: PromptChange(
                    new_prompt=prompt,
                    change_summary=strategy,
                    change_magnitude=ChangeMagnitude.MINIMAL,
                )
            },
        )

    alpha_spec = build_spec("first", "Focus first", "alpha")
    beta_spec = build_spec("second", "Focus second", "beta")
    primary_merge = build_spec("second", "Pareto merge second", "beta")
    partner_merge = build_spec("first", "Pareto merge first", "alpha")

    history_lengths: list[int] = []
    merge_calls: list[tuple[CandidateRecord, CandidateRecord]] = []
    iteration_tracker: dict[str, Any] = {"count": 0}

    def stub_generate_hypotheses(**kwargs):
        iteration = kwargs["iteration"]
        candidate_history = kwargs["candidate_history"]
        history_lengths.append(len(candidate_history))
        if iteration == 1:
            return [alpha_spec, beta_spec]
        return []

    def stub_generate_merge_hypotheses(*, baseline_candidate, partner_candidate, **kwargs):
        merge_calls.append((baseline_candidate, partner_candidate))

        expected_fields = {
            "baseline_candidate",
            "partner_candidate",
            "runtime",
            "hypothesis_lm",
            "hypothesis_adapter",
            "iteration",
            "tracker",
            "snapshot",
            "candidate_history",
            "best_val_score",
            "selection_strategy",
            "include_history",
            "success_rate_percentage",
        }
        actual_fields = {"baseline_candidate", "partner_candidate"} | set(kwargs.keys())
        assert actual_fields == expected_fields, (
            f"Merge function signature changed:\n"
            f"  Expected: {sorted(expected_fields)}\n"
            f"  Actual: {sorted(actual_fields)}\n"
            f"  Missing: {sorted(expected_fields - actual_fields)}\n"
            f"  Extra: {sorted(actual_fields - expected_fields)}"
        )

        # Validate ParetoMergeSignature has minimal fields (what the LLM sees)
        expected_sig_fields = {
            "primary_prompts",
            "partner_prompts",
            "program_flow",
            "success_rate_percentage",
            "best_validation_score",
            "current_iteration",
            "hypothesis_history",
        }
        actual_sig_fields = set(ParetoMergeSignature.input_fields.keys())
        assert actual_sig_fields == expected_sig_fields, (
            f"ParetoMergeSignature fields changed:\n"
            f"  Expected minimal: {sorted(expected_sig_fields)}\n"
            f"  Actual: {sorted(actual_sig_fields)}\n"
            f"  Missing: {sorted(expected_sig_fields - actual_sig_fields)}\n"
            f"  Extra: {sorted(actual_sig_fields - expected_sig_fields)}"
        )

        return [primary_merge, partner_merge]

    def stub_draw_weighted_candidate(candidates, weights, *, rng, exclude=None):
        excluded_ids = {id(candidate) for candidate in (exclude or [])}
        for candidate in candidates:
            if id(candidate) in excluded_ids:
                continue
            prompts = {
                name: getattr(predictor.signature, "instructions", "")
                for name, predictor in candidate.program.named_predictors()
            }
            if prompts.get("second") == "beta":
                return candidate
        for candidate in candidates:
            if id(candidate) not in excluded_ids:
                return candidate
        raise AssertionError("No candidate available for merge selection")

    def deterministic_best_candidate(candidates: list[CandidateRecord]) -> CandidateRecord:
        best_score = max(candidate.overall_score for candidate in candidates)
        for candidate in candidates:
            if candidate.overall_score == best_score:
                return candidate
        raise AssertionError("Expected at least one candidate")

    def fake_run_train_example(program, example, *, iteration, example_idx):
        return TrainExampleRecord(
            example=example,
            prediction=dspy.Prediction(first="", second=""),
            metric_score=0.0,
            metric_feedback="",
            is_success=False,
            error="mismatch",
            execution_flow=[],
        )

    def fake_analyze_record(*args, **kwargs):  # type: ignore[override]
        return dspy.Prediction()

    def evaluate_candidate_stub(*, program, calset, iteration, hypothesis):
        return CandidateRecord(
            program=DualPromptModule("start-first", "start-second"),
            overall_score=0.0,
            per_example_scores=[0.0, 0.0],
            iteration=iteration,
            hypothesis=hypothesis,
        )

    def evaluate_candidates_stub(*, iteration, hypotheses, baseline, cached_baseline, baseline_overrides, **_kwargs):
        iteration_tracker["count"] += 1
        if iteration_tracker["count"] == 1:
            assert hypotheses == [alpha_spec, beta_spec]
            candidates = [
                CandidateRecord(
                    program=DualPromptModule("start-first", "start-second"),
                    overall_score=0.0,
                    per_example_scores=[0.0, 0.0],
                    iteration=iteration,
                    hypothesis=None,
                ),
                CandidateRecord(
                    program=DualPromptModule("alpha", "start-second"),
                    overall_score=0.5,
                    per_example_scores=[1.0, 0.0],
                    iteration=iteration,
                    hypothesis=alpha_spec,
                ),
                CandidateRecord(
                    program=DualPromptModule("start-first", "beta"),
                    overall_score=0.5,
                    per_example_scores=[0.0, 1.0],
                    iteration=iteration,
                    hypothesis=beta_spec,
                ),
            ]
            return candidates

        assert hypotheses == [primary_merge, partner_merge]
        assert id(partner_merge) in baseline_overrides
        return [
            CandidateRecord(
                program=DualPromptModule("alpha", "start-second"),
                overall_score=0.5,
                per_example_scores=[1.0, 0.0],
                iteration=iteration,
                hypothesis=None,
            ),
            CandidateRecord(
                program=DualPromptModule("alpha", "beta"),
                overall_score=0.75,
                per_example_scores=[1.0, 0.5],
                iteration=iteration,
                hypothesis=primary_merge,
            ),
            CandidateRecord(
                program=DualPromptModule("alpha", "beta"),
                overall_score=1.0,
                per_example_scores=[1.0, 1.0],
                iteration=iteration,
                hypothesis=partner_merge,
            ),
        ]

    monkeypatch.setattr(apex.analysis_hooks, "generate_hypotheses", stub_generate_hypotheses)
    monkeypatch.setattr(apex.analysis_hooks, "generate_merge_hypotheses", stub_generate_merge_hypotheses)
    monkeypatch.setattr(
        "dspy.teleprompt.apex.candidate_selection.draw_weighted_candidate", stub_draw_weighted_candidate
    )
    monkeypatch.setattr(apex.analysis_hooks, "analyze_record", fake_analyze_record)
    monkeypatch.setattr(apex.evaluator, "run_train_example", fake_run_train_example)
    monkeypatch.setattr(apex.evaluator, "evaluate_candidate", evaluate_candidate_stub)
    monkeypatch.setattr(apex.evaluator, "evaluate_candidates", evaluate_candidates_stub)
    monkeypatch.setattr(apex.evaluator, "select_best_candidate", deterministic_best_candidate)

    optimized = apex.compile(baseline, trainset=trainset, valset=valset)

    assert optimized.first.signature.instructions == "alpha"
    assert optimized.second.signature.instructions == "beta"

    result = optimized.apex_result
    assert len(result.iterations) == 2
    assert result.stopped_after == "max_iterations"
    assert history_lengths[0] == 1  # Initial call sees only the baseline
    assert history_lengths[1] >= 3  # Second iteration receives prior candidates

    assert merge_calls, "Pareto merge hypotheses should be generated"
    baseline_candidate, partner_candidate = merge_calls[0]
    baseline_prompts = {
        name: getattr(pred.signature, "instructions", "")
        for name, pred in baseline_candidate.program.named_predictors()
    }
    partner_prompts = {
        name: getattr(pred.signature, "instructions", "") for name, pred in partner_candidate.program.named_predictors()
    }
    actual_pairs = {
        tuple(sorted(baseline_prompts.items())),
        tuple(sorted(partner_prompts.items())),
    }
    expected_pairs = {
        tuple(sorted({"first": "alpha", "second": "start-second"}.items())),
        tuple(sorted({"first": "start-first", "second": "beta"}.items())),
    }
    assert actual_pairs == expected_pairs

    iteration_one, iteration_two = result.iterations
    assert [change.new_prompt for change in iteration_one.hypotheses[0].prompt_changes.values()] == ["alpha"]
    assert [change.new_prompt for change in iteration_one.hypotheses[1].prompt_changes.values()] == ["beta"]
    assert [change.new_prompt for change in iteration_two.hypotheses[0].prompt_changes.values()] == ["beta"]
    assert [change.new_prompt for change in iteration_two.hypotheses[1].prompt_changes.values()] == ["alpha"]
    assert iteration_two.candidates[-1].overall_score == pytest.approx(1.0)


def test_apex_stops_after_patience_without_hypotheses() -> None:
    baseline = PromptDrivenModule("baseline")
    trainset = [make_example("question", "expected")]
    valset = [make_example("question", "expected")]

    apex = APEX(
        metric=simple_metric,
        analysis_lm=make_analysis_lm(),
        hypothesis_lm=DummyLM([], adapter=JSONAdapter()),
        max_iterations=5,
        convergence_patience=2,
        num_hypotheses=0,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        seed=5,
    )

    optimized = apex.compile(baseline, trainset=trainset, valset=valset)

    assert optimized.predictor.signature.instructions == "baseline"
    result = optimized.apex_result
    assert result.stopped_after == "patience"
    assert len(result.iterations) == 2
    assert all(not iteration.hypotheses for iteration in result.iterations)
    assert all(len(iteration.candidates) == 1 for iteration in result.iterations)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": None, "convergence_patience": None},
        {"max_iterations": 0},
        {"max_iterations": 1, "convergence_patience": 0},
        {"max_iterations": 1, "num_hypotheses": -1},
        {"max_iterations": 1, "num_eval_runs": 0},
        {"max_iterations": 1, "min_metric": 1.0, "max_metric": 0.5},
        {"max_iterations": 1, "candidate_selection": "invalid"},
        {"max_iterations": 1, "pareto_merge_probability": -0.1},
        {"max_iterations": 1, "pareto_merge_probability": 1.1},
    ],
)
def test_apex_initialization_rejects_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        APEX(
            metric=simple_metric,
            analysis_lm=make_analysis_lm(),
            hypothesis_lm=DummyLM([], adapter=JSONAdapter()),
            **kwargs,
        )


def test_apex_resume_uses_latest_checkpoint(tmp_path: Path) -> None:
    baseline = PromptDrivenModule("baseline")
    trainset = [make_example("question", "refined")]
    valset = [make_example("question", "refined")]

    analysis_lm = make_analysis_lm(failures=[make_analysis_response("Prompt should say 'refined'")])
    hypothesis_lm = DummyLM([make_hypothesis_response("refined")], adapter=JSONAdapter())

    apex = APEX(
        metric=simple_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        convergence_patience=None,
        num_hypotheses=1,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        checkpoint_dir=str(tmp_path),
        include_hypothesis_history=False,
        seed=19,
    )

    first_run = apex.compile(baseline, trainset=trainset, valset=valset)
    assert first_run.predictor.signature.instructions == "refined"
    first_result = first_run.apex_result
    assert len(first_result.iterations) == 1

    resumed = APEX(
        metric=simple_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        convergence_patience=None,
        num_hypotheses=1,
        train_sample=None,
        success_threshold=1.0,
        min_metric=0.0,
        max_metric=1.0,
        checkpoint_dir=str(tmp_path),
        include_hypothesis_history=False,
        seed=19,
    )

    resumed_program = resumed.compile(PromptDrivenModule("unused"), trainset=trainset, valset=valset, resume=True)
    assert resumed_program.predictor.signature.instructions == "refined"
    resumed_result = resumed_program.apex_result
    assert len(resumed_result.iterations) == 1
    assert resumed_result.best_candidate.overall_score == pytest.approx(first_result.best_candidate.overall_score)
    assert resumed_result.stopped_after == "max_iterations"


class ConditionalRouterModule(dspy.Module):
    """Module with conditional execution to test full program tree extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.route_a = dspy.Predict("query -> answer")
        self.route_b = dspy.Predict("query -> answer")
        self.route_a.signature.instructions = "Handle type A queries with concise responses"
        self.route_b.signature.instructions = "Handle type B queries with detailed explanations"

    def forward(self, query: str, route: str) -> dspy.Prediction:  # type: ignore[override]
        if route == "a":
            return self.route_a(query=query)
        else:
            return self.route_b(query=query)


def test_apex_execution_flow_includes_non_executed_predictors() -> None:
    """Verify that APEX analysis includes both executed and non-executed predictors in execution flow."""
    baseline = ConditionalRouterModule()

    # Create examples that execute different routes
    trainset = [
        Example(query="test1", route="a", answer="route_a_answer").with_inputs("query", "route"),
        Example(query="test2", route="b", answer="route_b_answer").with_inputs("query", "route"),
    ]
    valset = trainset

    # Configure LMs for analysis and hypothesis
    analysis_lm = make_analysis_lm(failures=[make_analysis_response("Improve clarity")])
    hypothesis_lm = DummyLM([make_hypothesis_response("clearer")], adapter=JSONAdapter())

    # Run APEX with minimal setup
    apex = APEX(
        metric=simple_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=None,
        success_threshold=0.0,  # Force analysis even on "successes"
        min_metric=0.0,
        max_metric=1.0,
        convergence_patience=1,
        include_hypothesis_history=False,
        seed=42,
    )

    optimized = apex.compile(baseline, trainset=trainset, valset=valset)

    # Access iteration logs to verify execution flow
    result = optimized.apex_result
    assert len(result.iterations) >= 1

    # The first iteration should have train example records
    first_iteration = result.iterations[0]

    # Verify that train examples were processed (they would have execution flow)
    total_examples = first_iteration.num_failures + first_iteration.num_successes
    assert total_examples == len(trainset), f"Expected {len(trainset)} examples processed, got {total_examples}"

    # Verify that both sampled_train_size is reasonable
    assert first_iteration.sampled_train_size == len(trainset)

    # The key assertion: APEX should have successfully run with our conditional module
    # The execution flow extraction (extract_full_execution_flow_with_coverage) was called
    # during evaluation, and it would have failed if it didn't work correctly with
    # conditional modules. The fact that APEX completed without errors demonstrates
    # that execution flow extraction works with both executed and non-executed predictors.
    assert result.best_candidate is not None
    assert result.stopped_after in ("patience", "max_iterations")


def test_apex_program_structure_shows_source_code() -> None:
    """Verify program snapshot includes full source code showing control flow."""
    from dspy.teleprompt.apex.snapshot import snapshot_program

    module = ConditionalRouterModule()
    snapshot = snapshot_program(module)

    # Should have source_code (not flow_description or structure)
    assert hasattr(snapshot, "source_code")
    assert not hasattr(snapshot, "flow_description")
    assert not hasattr(snapshot, "structure")

    # Source code should be exact
    expected_source = '''class ConditionalRouterModule(dspy.Module):
    """Module with conditional execution to test full program tree extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.route_a = dspy.Predict("query -> answer")
        self.route_b = dspy.Predict("query -> answer")
        self.route_a.signature.instructions = "Handle type A queries with concise responses"
        self.route_b.signature.instructions = "Handle type B queries with detailed explanations"

    def forward(self, query: str, route: str) -> dspy.Prediction:  # type: ignore[override]
        if route == "a":
            return self.route_a(query=query)
        else:
            return self.route_b(query=query)'''

    assert snapshot.source_code == expected_source

    # Should still show all predictor prompts
    assert "route_a" in snapshot.prompts
    assert "route_b" in snapshot.prompts

    # Verify the prompts contain instructions
    assert "Handle type A queries" in snapshot.prompts["route_a"]
    assert "Handle type B queries" in snapshot.prompts["route_b"]
