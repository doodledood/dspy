import random
from unittest.mock import MagicMock, patch

import pytest

import dspy
import dspy.teleprompt.apex.runtime as runtime_module
from dspy import Example
from dspy.teleprompt.apex import (
    APEX,
    CandidateRecord,
    ChangeMagnitude,
    ExperimentTracker,
    HypothesisSpec,
    PromptChange,
    Verbosity,
)
from dspy.teleprompt.apex.evaluation import EvaluationEngine
from dspy.utils.dummies import DummyLM


def make_analysis_response(root_cause: str = "Prompt missing correct token") -> dict:
    return {
        "root_cause": root_cause,
        "involved_predictors": ["predictor"],
        "context": "Baseline emits 'bad'",
        "category": "format_ambiguity",
        "key_details": "Needs to say good",
    }


def make_success_response(pattern: str = "Prompt handled well") -> dict:
    return {
        "success_pattern": pattern,
        "contributing_predictors": ["predictor"],
        "context": "Handled correctly",
        "category": "clear_format_compliance",
        "key_details": "Keep current instructions",
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
        # Emit the current prompt text to keep the module deterministic for tests.
        return dspy.Prediction(output=self.predictor.signature.instructions)


def make_train_example(value: str) -> Example:
    return Example(input=value, output="good").with_inputs("input")


def metric(example: Example, prediction: dspy.Prediction, trace) -> float:
    expected = example.output
    predicted = prediction.output
    return 1.0 if expected == predicted else 0.0


class InMemoryTracker(ExperimentTracker):
    """Tracker stub that captures logged trace batches in-memory."""

    def __init__(self):
        super().__init__(use_mlflow=False)
        self.logged_batches: list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def is_active(self):
        return True

    def log_trace_batch(self, batch):
        self.logged_batches.append(batch)


def test_evaluate_candidate_logs_traces_without_mutating_predictors():
    tracker = InMemoryTracker()
    runtime = runtime_module.RuntimeTools(verbosity=Verbosity.HIGH, num_threads=1)
    engine = EvaluationEngine(
        metric=metric,
        runtime=runtime,
        tracker=tracker,
        min_metric=0.0,
        max_metric=1.0,
        success_threshold=1.0,
        num_eval_runs=1,
        rng=random.Random(0),
        log=lambda message, level, _: None,
        is_enabled=lambda _: False,
    )

    class TraceableModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor.lm = DummyLM([{"output": "good"}])

        def forward(self, input: str) -> dspy.Prediction:
            return self.predictor(input=input)

    program = TraceableModule()
    example = make_train_example("x")

    record = engine.evaluate_candidate(program=program, calset=[example], iteration=1, hypothesis=None)

    predictor_names = [name for name, _ in record.program.named_predictors()]
    assert predictor_names == ["predictor"]

    assert tracker.logged_batches, "Expected execution traces to be logged"
    artifact = tracker.logged_batches[0].to_artifact()
    assert artifact["stage"] == "baseline_evaluation"
    execution_flows = artifact["traces"][0]["runs"][0]["execution_flow"]
    predictor_names_in_trace = {entry["predictor_name"] for entry in execution_flows}
    assert predictor_names_in_trace == {"predictor"}


@pytest.mark.parametrize(
    "kwargs, error_match",
    [
        ({"max_iterations": None, "convergence_patience": None}, "At least one"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"convergence_patience": 0}, "convergence_patience"),
        ({"num_hypotheses": -1}, "num_hypotheses"),
        ({"num_eval_runs": 0}, "num_eval_runs"),
        ({"min_metric": 1.1, "max_metric": 1.0}, "min_metric"),
    ],
)
def test_apex_constructor_validation(kwargs, error_match):
    analysis_lm = DummyLM([], adapter=dspy.JSONAdapter())

    with pytest.raises(ValueError, match=error_match):
        params = {
            "metric": metric,
            "analysis_lm": analysis_lm,
            "hypothesis_lm": analysis_lm,
            "max_iterations": 1,
            "convergence_patience": 1,
        }
        params.update(kwargs)
        APEX(**params)


def test_apex_requires_non_empty_train_and_valset():
    analysis_lm = DummyLM([], adapter=dspy.JSONAdapter())
    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=analysis_lm,
        max_iterations=1,
        convergence_patience=1,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")

    with pytest.raises(ValueError, match="trainset must be non-empty"):
        optimizer.compile(student, trainset=[], valset=[make_train_example("x")])

    with pytest.raises(ValueError, match="calibration set"):
        optimizer.compile(student, trainset=[make_train_example("x")], valset=[])


def test_apex_builds_history_text_when_enabled():
    analysis_lm = DummyLM([], adapter=dspy.JSONAdapter())
    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=analysis_lm,
        max_iterations=1,
        convergence_patience=1,
        verbosity="none",
    )

    hypothesis = HypothesisSpec(
        observation="obs",
        fixable_root_causes=["missing token"],
        non_fixable_root_causes=[],
        impact_score=0.5,
        generalizability_score=0.2,
        strategy="Improve prompt",
        expected_impact="better",
        prompt_changes={
            "predictor": PromptChange(
                new_prompt="better",
                rationale="Clean whitespace",
                change_magnitude=ChangeMagnitude.MINIMAL,
            )
        },
    )

    candidate = CandidateRecord(
        program=PromptDrivenModule(initial_prompt="bad"),
        overall_score=0.8,
        per_example_scores=[0.8],
        iteration=3,
        hypothesis=hypothesis,
    )

    history_text = optimizer._build_hypothesis_history_text([candidate])

    assert "Iteration 3" in history_text
    assert "predictor" in history_text
    assert "Clean whitespace" in history_text


def test_apex_history_disabled_returns_na():
    analysis_lm = DummyLM([], adapter=dspy.JSONAdapter())
    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=analysis_lm,
        max_iterations=1,
        convergence_patience=1,
        verbosity="none",
        include_hypothesis_history=False,
    )

    candidate = CandidateRecord(
        program=PromptDrivenModule(initial_prompt="bad"),
        overall_score=1.0,
        per_example_scores=[1.0],
        iteration=1,
        hypothesis=None,
    )

    assert optimizer._build_hypothesis_history_text([candidate]) == "N/A"


def test_apex_sample_callable_must_return_list():
    analysis_lm = DummyLM([make_analysis_response()], adapter=dspy.JSONAdapter())
    hypothesis_lm = DummyLM([make_hypothesis_response()], adapter=dspy.JSONAdapter())

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=1,
        verbosity="none",
        train_sample=lambda data, iteration: tuple(data),
    )

    student = PromptDrivenModule(initial_prompt="bad")

    with pytest.raises(TypeError, match="Custom train_sample callable"):
        optimizer.compile(
            student,
            trainset=[make_train_example("x"), make_train_example("y")],
            valset=[make_train_example("x")],
        )


def test_apex_apply_hypothesis_rejects_unknown_predictor():
    analysis_lm = DummyLM([], adapter=dspy.JSONAdapter())
    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=analysis_lm,
        max_iterations=1,
        convergence_patience=1,
        verbosity="none",
    )

    hypothesis = HypothesisSpec(
        observation="obs",
        fixable_root_causes=[],
        non_fixable_root_causes=[],
        impact_score=0.1,
        generalizability_score=0.1,
        strategy="",
        expected_impact="",
        prompt_changes={
            "unknown": PromptChange(
                new_prompt="new",
                rationale="",
                change_magnitude=ChangeMagnitude.MINIMAL,
            )
        },
    )

    with pytest.raises(ValueError, match="unknown predictor"):
        optimizer._apply_hypothesis(PromptDrivenModule(initial_prompt="bad"), hypothesis)


def test_apex_improves_and_tracks_history():
    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response() for _ in range(len(trainset))],
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
        max_iterations=5,
        num_hypotheses=1,
        convergence_patience=2,
        num_eval_runs=1,
        train_sample=None,
        seed=42,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    assert optimized.predictor.signature.instructions == "good"
    result = optimized.apex_result
    best = result.best_candidate
    assert pytest.approx(best.overall_score) == 1.0
    assert result.stopped_after in {"patience", "max_iterations"}
    # Ensure at least two iterations logged (improvement + patience stop)
    assert len(result.iterations) >= 2


def test_apex_train_sampling_controls_analysis_calls():
    trainset = [make_train_example(str(i)) for i in range(6)]
    calset = trainset[:2]

    analysis_lm = DummyLM(
        [make_analysis_response("always wrong")],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [
            {
                "hypotheses": [
                    {
                        "observation": "fix",
                        "fixable_root_causes": ["always wrong"],
                        "non_fixable_root_causes": [],
                        "strategy": "swap prompt",
                        "expected_impact": "",
                        "prompt_changes": {
                            "predictor": PromptChange(
                                new_prompt="good",
                                rationale="",
                                change_magnitude=ChangeMagnitude.MINIMAL,
                            )
                        },
                    }
                ]
            }
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=1,
        num_eval_runs=1,
        convergence_patience=1,
        seed=0,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert len(analysis_lm.history) == 1


def test_apex_sampling_callable_receives_iteration():
    def sampler(dataset, iteration):
        assert iteration == 1
        return dataset[:2]

    analysis_lm = DummyLM(
        [make_analysis_response("wrong") for _ in range(2)],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [
            {
                "hypotheses": [
                    {
                        "observation": "fix",
                        "fixable_root_causes": ["wrong"],
                        "non_fixable_root_causes": [],
                        "strategy": "",
                        "expected_impact": "",
                        "prompt_changes": {},
                    }
                ]
            }
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=sampler,
        num_eval_runs=1,
        convergence_patience=1,
        seed=13,
        verbosity="none",
    )

    trainset = [make_train_example("x"), make_train_example("y"), make_train_example("z")]
    calset = trainset[:1]
    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert len(analysis_lm.history) == 2


def test_apex_execution_flow_captures_branching_dependencies():
    class BranchingModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.a = dspy.Predict("x -> a")
            self.b = dspy.Predict("a -> b")
            self.c = dspy.Predict("a -> c")
            self.d = dspy.Predict("b, c -> d")

        def forward(self, x: str) -> dspy.Prediction:
            out_a = self.a(x=x)
            out_b = self.b(a=out_a.a)
            out_c = self.c(a=out_a.a)
            return self.d(b=out_b.b, c=out_c.c)

    apex = APEX(
        metric=metric,
        analysis_lm=DummyLM([make_analysis_response()], adapter=dspy.JSONAdapter()),
        hypothesis_lm=DummyLM([make_hypothesis_response()], adapter=dspy.JSONAdapter()),
        max_iterations=1,
        num_hypotheses=0,
        train_sample=None,
        num_eval_runs=1,
        convergence_patience=1,
        seed=0,
        verbosity="none",
    )

    branching_program = BranchingModule()

    execution_lm = DummyLM(
        [
            {"a": "A"},
            {"b": "B"},
            {"c": "C"},
            {"d": "D"},
        ]
    )

    with dspy.settings.context(lm=execution_lm, trace=[], max_trace_size=50):
        branching_program(x="input")
        trace = list(dspy.settings.trace or [])

    execution_flow = apex._extract_execution_flow(trace, branching_program)

    assert [entry.predictor_name for entry in execution_flow] == ["a", "b", "c", "d"]

    entries = {entry.predictor_name: entry for entry in execution_flow}
    assert entries["a"].dependencies == []
    assert entries["b"].dependencies == ["a"]
    assert entries["c"].dependencies == ["a"]
    assert entries["d"].dependencies == ["b", "c"]
    assert entries["d"].input_sources == {"b": ["b"], "c": ["c"]}

    graph = apex._format_execution_flow_as_graph(execution_flow)
    assert "Program DAG" in graph
    assert "↳ a" in graph
    assert "a (Predict)" in graph
    assert "depends on: Input" in graph
    assert "feeds: b, c" in graph
    assert "d (Predict)" in graph
    assert "depends on: b, c" in graph
    assert "feeds: Output" in graph


def test_apex_uses_configured_num_threads(monkeypatch):
    calls: list[int] = []

    def fake_execute(self, function, data):
        calls.append(self.num_threads)
        return [function(item) for item in data]

    monkeypatch.setattr(runtime_module.ParallelExecutor, "execute", fake_execute)

    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response("thread test") for _ in range(len(trainset))],
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
        train_sample=None,
        num_eval_runs=1,
        convergence_patience=1,
        seed=11,
        verbosity="none",
        num_threads=2,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert any(num == 2 for num in calls)


def test_apex_rejects_invalid_analysis_json():
    analysis_lm = DummyLM(
        [{"invalid_field": "missing required fields"}],  # Invalid response, missing required fields
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
        seed=0,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    trainset = [make_train_example("x")]
    from dspy.utils.exceptions import AdapterParseError

    with pytest.raises(AdapterParseError):
        optimizer.compile(student, trainset=trainset, valset=trainset)


def test_apex_trims_hypotheses_to_limit():
    analysis_lm = DummyLM(
        [make_analysis_response()],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [
            {
                "hypotheses": [
                    {
                        "observation": "option A",
                        "fixable_root_causes": ["Prompt missing correct token"],
                        "non_fixable_root_causes": [],
                        "strategy": "Rewrite prompt A",
                        "expected_impact": "Outputs 'good'",
                        "prompt_changes": {
                            "predictor": PromptChange(
                                new_prompt="good",
                                rationale="Align",
                                change_magnitude=ChangeMagnitude.MINIMAL,
                            )
                        },
                    },
                    {
                        "observation": "option B",
                        "fixable_root_causes": ["Prompt missing correct token"],
                        "non_fixable_root_causes": [],
                        "strategy": "Rewrite prompt B",
                        "expected_impact": "Outputs 'great'",
                        "prompt_changes": {
                            "predictor": PromptChange(
                                new_prompt="great",
                                rationale="Align alt",
                                change_magnitude=ChangeMagnitude.MODERATE,
                            )
                        },
                    },
                ]
            }
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=0,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=[make_train_example("x")], valset=[make_train_example("x")])

    iteration = optimized.apex_result.iterations[0]
    assert len(iteration.hypotheses) == 1
    assert iteration.hypotheses[0].prompt_changes["predictor"].new_prompt == "good"


def test_apex_handles_fewer_successes_than_failures():
    def mixed_metric(example: Example, prediction: dspy.Prediction, trace) -> float:
        # Treat inputs ending with "success" as automatic successes.
        if example.input.endswith("success"):
            return 1.0
        return 1.0 if prediction.output == "good" else 0.0

    trainset = [
        Example(input="a_success", output="good").with_inputs("input"),
        Example(input="b_failure", output="good").with_inputs("input"),
        Example(input="c_failure", output="good").with_inputs("input"),
    ]
    calset = trainset

    analysis_payloads = [
        make_analysis_response("mixed failure 1"),
        make_analysis_response("mixed failure 2"),
        make_success_response("success pattern"),
    ]
    analysis_lm = DummyLM(analysis_payloads, adapter=dspy.JSONAdapter())
    hypothesis_lm = DummyLM([make_hypothesis_response()], adapter=dspy.JSONAdapter())

    optimizer = APEX(
        metric=mixed_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=2,
        num_hypotheses=1,
        convergence_patience=1,
        seed=123,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Ensures we analyzed all failures plus the single available success (no duplication).
    assert len(analysis_lm.history) == len(analysis_payloads)
    assert optimized.apex_result.best_candidate.overall_score >= 1.0


def test_apex_end_to_end_fake_data():
    trainset = [
        Example(input="sample_success", output="baseline").with_inputs("input"),
        make_train_example("fix_a"),
        make_train_example("fix_b"),
    ]
    calset = [
        Example(input="sample_success", output="baseline").with_inputs("input"),
        make_train_example("fix_a"),
        make_train_example("fix_b"),
    ]

    analysis_responses = [
        make_analysis_response("fix format for fix_a"),
        make_analysis_response("fix format for fix_b"),
        make_success_response("baseline prompt handles sample_success"),
        make_analysis_response("baseline prompt now mismatched"),
        make_success_response("good prompt stable"),
        make_success_response("good prompt handles remaining cases"),
    ]
    analysis_lm = DummyLM(analysis_responses, adapter=dspy.JSONAdapter())
    hypothesis_lm = DummyLM(
        [
            make_hypothesis_response("good"),
            {"hypotheses": []},
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=3,
        num_hypotheses=1,
        convergence_patience=1,
        num_eval_runs=1,
        seed=99,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="baseline")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)
    result = optimized.apex_result

    # Best candidate should apply the "good" prompt and improve average score from 1/3 to 2/3.
    assert optimized.predictor.signature.instructions == "good"
    assert pytest.approx(result.best_candidate.overall_score, rel=0.0, abs=1e-9) == 2 / 3
    assert result.stopped_after == "patience"

    # Two iterations: first with a winning hypothesis, second with no improvement.
    assert len(result.iterations) == 2
    first_iter, second_iter = result.iterations
    assert first_iter.num_failures == 2 and first_iter.num_successes == 1
    assert len(first_iter.hypotheses) == 1
    assert first_iter.hypotheses[0].prompt_changes["predictor"].new_prompt == "good"
    assert first_iter.candidates[0].hypothesis is None  # baseline evaluated first
    assert first_iter.candidates[1].hypothesis == first_iter.hypotheses[0]
    assert len(first_iter.candidates[0].per_example_scores) == len(calset)
    assert len(first_iter.candidates[1].per_example_scores) == len(calset)
    assert second_iter.num_failures == 1 and second_iter.num_successes == 2
    assert second_iter.hypotheses == []
    assert second_iter.candidates[0].hypothesis is None  # re-evaluated champion
    assert len(second_iter.candidates[0].per_example_scores) == len(calset)
    assert all(
        score <= result.best_candidate.overall_score + 1e-9
        for score in (cand.overall_score for cand in result.all_candidates)
    )

    # Candidate history should contain initial baseline + baseline + new hypothesis + final baseline re-evaluation.
    assert len(result.all_candidates) == 4
    assert result.all_candidates[0].hypothesis is None  # Initial baseline (iteration 0)
    assert result.all_candidates[1].hypothesis is None  # First iteration baseline
    assert result.all_candidates[2].hypothesis is not None  # First iteration hypothesis
    assert result.all_candidates[3].hypothesis is None  # Second iteration baseline


def test_apex_normal_verbosity_logs_candidates_only():
    trainset = [make_train_example("x")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response("normal verbosity check")],
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
        seed=7,
        verbosity="normal",
    )

    messages: list[tuple[Verbosity, str]] = []

    def capture(message: str, level: Verbosity = Verbosity.NORMAL) -> None:
        if optimizer._is_enabled(level):
            messages.append((level, message))

    optimizer._log = capture  # type: ignore[assignment]

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert any("hypothesis score" in msg for _, msg in messages)
    assert all("failure analysis" not in msg for _, msg in messages)


def test_apex_high_verbosity_logs_analysis():
    trainset = [make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response("high verbosity issue")],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response("better")],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=8,
        verbosity="high",
    )

    messages: list[tuple[Verbosity, str]] = []

    def capture(message: str, level: Verbosity = Verbosity.NORMAL) -> None:
        if optimizer._is_enabled(level):
            messages.append((level, message))

    optimizer._log = capture  # type: ignore[assignment]

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert any("hypothesis score" in msg for _, msg in messages)
    assert any("failure analysis" in msg for _, msg in messages)


def test_apex_public_api_exposed():
    import dspy.teleprompt as teleprompt_module

    assert teleprompt_module.APEX is APEX


def test_apex_checkpoint_and_resume(tmp_path):
    """Verify APEX can save checkpoints and resume from them."""

    def dummy_metric(example: Example, prediction: dspy.Prediction, trace) -> float:
        return 1.0 if prediction.output == "good" else 0.0

    trainset = [make_train_example("x"), make_train_example("y")]
    valset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response() for _ in range(10)],  # Enough for multiple iterations
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response() for _ in range(10)],  # Enough for multiple iterations
        adapter=dspy.JSONAdapter(),
    )

    # First run - will complete after 2 iterations
    optimizer1 = APEX(
        metric=dummy_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=2,
        num_hypotheses=1,
        convergence_patience=None,  # Rely on max_iterations
        checkpoint_dir=tmp_path,
        verbosity="none",
        seed=42,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized1 = optimizer1.compile(student=student, trainset=trainset, valset=valset)
    initial_iterations = len(optimized1.apex_result.iterations)
    assert initial_iterations == 2
    assert optimized1.apex_result.stopped_after == "max_iterations"

    # Verify checkpoint files were created
    checkpoint_files = list(tmp_path.glob("checkpoint_iter_*.pkl"))
    assert len(checkpoint_files) > 0, "No checkpoint files created"
    latest_file = tmp_path / "latest_checkpoint.json"
    assert latest_file.exists(), "latest_checkpoint.json not created"

    # Second run - resume from checkpoint and continue for more iterations
    optimizer2 = APEX(
        metric=dummy_metric,
        analysis_lm=analysis_lm,
        hypothesis_lm=hypothesis_lm,
        max_iterations=4,  # Continue for more iterations
        num_hypotheses=1,
        convergence_patience=None,
        checkpoint_dir=tmp_path,
        verbosity="none",
        seed=42,
    )

    # Resume from checkpoint
    optimized2 = optimizer2.compile(student=student, trainset=trainset, valset=valset, resume=True)
    final_iterations = len(optimized2.apex_result.iterations)

    # Should have continued from where it left off
    assert final_iterations == 4
    assert optimized2.apex_result.stopped_after == "max_iterations"


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

    assert not optimizer.tracker.use_mlflow
    assert not optimizer.tracker.is_active()

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)
    assert optimized.predictor.signature.instructions == "good"


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_enabled(mock_mlflow):
    """Test that MLflow tracking can be enabled and logs appropriate data."""
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
        use_mlflow=True,
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="test-experiment",
    )

    assert optimizer.tracker.use_mlflow
    assert optimizer.tracker.mlflow_tracking_uri == "http://localhost:5000"
    assert optimizer.tracker.mlflow_experiment_name == "test-experiment"

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    mock_mlflow.set_tracking_uri.assert_called_with("http://localhost:5000")
    mock_mlflow.set_experiment.assert_called_with("test-experiment")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.end_run.assert_called_once()

    assert mock_mlflow.log_param.call_count > 0
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

    assert not optimizer.tracker.use_mlflow

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)
    assert optimized.predictor.signature.instructions == "good"


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_iteration_tracking(mock_mlflow):
    """Test that APEX tracks iteration-level metrics with MLflow."""
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

    assert mock_mlflow.log_metrics.called or mock_mlflow.log_param.called
    assert mock_mlflow.start_run.called
    assert mock_mlflow.end_run.called


@patch("dspy.teleprompt.apex.tracker.MLFLOW_AVAILABLE", True)
@patch("dspy.teleprompt.apex.tracker.mlflow")
def test_apex_mlflow_artifact_logging(mock_mlflow):
    """Test that APEX logs artifacts like hypotheses to MLflow."""
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

    assert mock_mlflow.log_artifact.called or mock_mlflow.log_param.called
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

            with tracker:
                assert tracker.is_active()
                tracker.log_params({"test_param": "value"})
                tracker.log_metrics({"test_metric": 0.5})

            mock_mlflow.end_run.assert_called_once()


def test_apex_tracking_utils_format_functions():
    """Test the tracking utility formatting functions."""
    from dspy.teleprompt.apex import tracking_utils

    baseline_metrics = tracking_utils.format_baseline_metrics(baseline_score=0.5, num_train=10, num_val=5)
    assert baseline_metrics["baseline_score"] == 0.5
    assert baseline_metrics["num_train_examples"] == 10
    assert baseline_metrics["num_val_examples"] == 5

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

    best_candidate = MagicMock(overall_score=0.9)
    all_candidates = [MagicMock(overall_score=0.7), MagicMock(overall_score=0.9)]
    iterations = [MagicMock(num_failures=2, num_successes=3, hypotheses=hypotheses)]
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


def test_apex_predictor_names_in_execution_flow_after_deepcopy():
    """Predictor names must be correctly resolved in execution flow after deepcopy.

    This tests the specific bug where predictor names show as 'unknown' in
    execution flow, causing hypothesis LM to generate invalid predictor references.
    """
    from dspy.teleprompt.apex.execution_flow import extract_execution_flow

    class QuestionAnswer(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    program = dspy.ChainOfThought(QuestionAnswer)

    # Simulate APEX's flow: deepcopy the program
    program_copy = program.deepcopy()

    # Run with tracing (simulating evaluate_train_examples)
    example = dspy.Example(question="Test?", answer="42").with_inputs("question")

    fake_lm = DummyLM([{"reasoning": "thinking", "answer": "42"}])
    with dspy.settings.context(lm=fake_lm, trace=[], max_trace_size=50):
        _ = program_copy(question=example.question)
        trace = list(dspy.settings.trace or [])

    assert len(trace) > 0, "Trace should contain predictor calls"

    # Extract execution flow from the SAME program instance
    execution_flow = extract_execution_flow(trace, program_copy)

    # Critical assertion: predictor names must NOT be "unknown"
    for entry in execution_flow:
        assert entry.predictor_name != "unknown", (
            f"BUG: Predictor name is 'unknown' (type: {entry.predictor_type}). "
            f"This causes hypothesis LM to see 'unknown (Predict)' and generate "
            f"invalid predictor references like 'Predict' instead of 'predict'."
        )
        # For ChainOfThought, the single predictor is named "predict"
        assert entry.predictor_name == "predict", f"Expected 'predict', got '{entry.predictor_name}'"


def test_apex_end_to_end_with_deepcopy():
    """Verify APEX works end-to-end with deepcopied programs."""
    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    analysis_lm = DummyLM(
        [make_analysis_response("needs fix") for _ in range(len(trainset))],
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
        num_eval_runs=1,
        train_sample=None,
        seed=99,
        verbosity="none",
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Verify optimization succeeded (predictor names were resolved correctly)
    assert optimized.predictor.signature.instructions == "good"
    assert optimized.apex_result.best_candidate.overall_score == 1.0

    # Verify predictor names appeared correctly in failure analysis
    # (this would have failed with the "unknown" bug)
    iteration = optimized.apex_result.iterations[0]
    assert len(iteration.hypotheses) == 1
    assert "predictor" in iteration.hypotheses[0].prompt_changes
