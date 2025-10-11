import pytest

import dspy
import dspy.teleprompt.apex_optimizer as apex_module
from dspy import Example
from dspy.teleprompt.apex_optimizer import APEX, ChangeMagnitude, PromptChange, Verbosity
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


def test_apex_uses_configured_num_threads(monkeypatch):
    calls: list[int] = []

    def fake_execute(self, function, data):
        calls.append(self.num_threads)
        return [function(item) for item in data]

    monkeypatch.setattr(apex_module.ParallelExecutor, "execute", fake_execute)

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
    assert second_iter.num_failures == 1 and second_iter.num_successes == 1
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
