import pytest

import dspy
from dspy import Example
from dspy.teleprompt.apex_optimizer import APEX
from dspy.utils.dummies import DummyLM


def make_analysis_response(root_cause: str = "Prompt missing correct token") -> dict:
    return {
        "json_response": {
            "root_cause": root_cause,
            "involved_predictors": ["predictor"],
            "context": "Baseline emits 'bad'",
            "category": "format_ambiguity",
            "key_details": "Needs to say good",
        }
    }


def make_success_response(pattern: str = "Prompt handled well") -> dict:
    return {
        "json_response": {
            "success_pattern": pattern,
            "contributing_predictors": ["predictor"],
            "context": "Handled correctly",
            "category": "clear_format_compliance",
            "key_details": "Keep current instructions",
        }
    }


def make_hypothesis_response(prompt_value: str = "good") -> dict:
    return {
        "json_response": [
            {
                "observation": "Prompt mismatch",
                "fixable_root_causes": ["Prompt missing correct token"],
                "non_fixable_root_causes": [],
                "strategy": "Rewrite prompt",
                "expected_impact": "Outputs 'good'",
                "prompt_changes": {
                    "predictor": {
                        "new_prompt": prompt_value,
                        "rationale": "Align output with expectation",
                        "change_magnitude": "minimal",
                    }
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
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=5,
        num_hypotheses=1,
        convergence_patience=2,
        num_eval_runs=1,
        train_sample=None,
        seed=42,
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
                "json_response": [
                    {
                        "observation": "fix",
                        "fixable_root_causes": ["always wrong"],
                        "non_fixable_root_causes": [],
                        "strategy": "swap prompt",
                        "expected_impact": "",
                        "prompt_changes": {
                            "predictor": {
                                "new_prompt": "good",
                                "rationale": "",
                                "change_magnitude": "minimal",
                            }
                        },
                    }
                ]
            }
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=1,
        num_eval_runs=1,
        convergence_patience=1,
        seed=0,
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
                "json_response": [
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
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=sampler,
        num_eval_runs=1,
        convergence_patience=1,
        seed=13,
    )

    trainset = [make_train_example("x"), make_train_example("y"), make_train_example("z")]
    calset = trainset[:1]
    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert len(analysis_lm.history) == 2


def test_apex_rejects_invalid_analysis_json():
    analysis_lm = DummyLM(
        [{"json_response": "not json"}],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [make_hypothesis_response()],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=0,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    trainset = [make_train_example("x")]
    with pytest.raises(ValueError):
        optimizer.compile(student, trainset=trainset, valset=trainset)


def test_apex_trims_hypotheses_to_limit():
    analysis_lm = DummyLM(
        [make_analysis_response()],
        adapter=dspy.JSONAdapter(),
    )
    hypothesis_lm = DummyLM(
        [
            {
                "json_response": [
                    {
                        "observation": "option A",
                        "fixable_root_causes": ["Prompt missing correct token"],
                        "non_fixable_root_causes": [],
                        "strategy": "Rewrite prompt A",
                        "expected_impact": "Outputs 'good'",
                        "prompt_changes": {
                            "predictor": {
                                "new_prompt": "good",
                                "rationale": "Align",
                                "change_magnitude": "minimal",
                            }
                        },
                    },
                    {
                        "observation": "option B",
                        "fixable_root_causes": ["Prompt missing correct token"],
                        "non_fixable_root_causes": [],
                        "strategy": "Rewrite prompt B",
                        "expected_impact": "Outputs 'great'",
                        "prompt_changes": {
                            "predictor": {
                                "new_prompt": "great",
                                "rationale": "Align alt",
                                "change_magnitude": "moderate",
                            }
                        },
                    },
                ]
            }
        ],
        adapter=dspy.JSONAdapter(),
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=1,
        num_hypotheses=1,
        convergence_patience=1,
        seed=0,
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
        analysis_llm=analysis_lm,
        hypothesis_llm=hypothesis_lm,
        max_iterations=2,
        num_hypotheses=1,
        convergence_patience=1,
        seed=123,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimized = optimizer.compile(student, trainset=trainset, valset=calset)

    # Ensures we analyzed all failures plus the single available success (no duplication).
    assert len(analysis_lm.history) == len(analysis_payloads)
    assert optimized.apex_result.best_candidate.overall_score >= 1.0


def test_apex_public_api_exposed():
    import dspy.teleprompt as teleprompt_module

    assert teleprompt_module.APEX is APEX
