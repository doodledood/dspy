import json

import pytest

import dspy
from dspy import Example
from dspy.teleprompt.apex_optimizer import APEX


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


class LinearResponder:
    """Deterministic responder returning predefined JSON payloads in order."""

    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.call_count = 0

    def __call__(self, prompt: str | None = None):
        if self.call_count >= len(self._responses):
            raise RuntimeError("Responder exhausted responses.")
        payload = self._responses[self.call_count]
        self.call_count += 1
        return json.dumps(payload)


def test_apex_improves_and_tracks_history():
    trainset = [make_train_example("x"), make_train_example("y")]
    calset = trainset

    failure_responses = [
        {
            "root_cause": "Prompt missing correct token",
            "involved_predictors": ["predictor"],
            "context": "Baseline emits 'bad'",
            "category": "format_ambiguity",
            "key_details": "Needs to say good",
        }
        for _ in range(len(trainset))
    ]
    analysis_responder = LinearResponder(failure_responses)
    hypothesis_responder = LinearResponder(
        [
            [
                {
                    "observation": "Prompt mismatch",
                    "fixable_root_causes": ["Prompt missing correct token"],
                    "non_fixable_root_causes": [],
                    "strategy": "Rewrite prompt",
                    "expected_impact": "Outputs 'good'",
                    "prompt_changes": {
                        "predictor": {
                            "new_prompt": "good",
                            "rationale": "Align output with expectation",
                            "change_magnitude": "minimal",
                        }
                    },
                }
            ]
        ]
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_responder,
        hypothesis_llm=hypothesis_responder,
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

    analysis_responder = LinearResponder(
        [
            {
                "root_cause": "always wrong",
                "involved_predictors": ["predictor"],
                "context": "",
                "category": "generic",
                "key_details": "",
            }
        ]
    )
    hypothesis_responder = LinearResponder(
        [
            [
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
        ]
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_responder,
        hypothesis_llm=hypothesis_responder,
        max_iterations=1,
        num_hypotheses=1,
        train_sample=1,
        num_eval_runs=1,
        convergence_patience=1,
        seed=0,
    )

    student = PromptDrivenModule(initial_prompt="bad")
    optimizer.compile(student, trainset=trainset, valset=calset)

    assert analysis_responder.call_count == 1


def test_apex_sampling_callable_receives_iteration():
    def sampler(dataset, iteration):
        assert iteration == 1
        return dataset[:2]

    analysis_responder = LinearResponder(
        [
            {
                "root_cause": "wrong",
                "involved_predictors": ["predictor"],
                "context": "",
                "category": "generic",
                "key_details": "",
            },
            {
                "root_cause": "wrong",
                "involved_predictors": ["predictor"],
                "context": "",
                "category": "generic",
                "key_details": "",
            },
        ]
    )
    hypothesis_responder = LinearResponder(
        [
            [
                {
                    "observation": "fix",
                    "fixable_root_causes": ["wrong"],
                    "non_fixable_root_causes": [],
                    "strategy": "",
                    "expected_impact": "",
                    "prompt_changes": {},
                }
            ]
        ]
    )

    optimizer = APEX(
        metric=metric,
        analysis_llm=analysis_responder,
        hypothesis_llm=hypothesis_responder,
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

    assert analysis_responder.call_count == 2


def test_apex_public_api_exposed():
    import dspy.teleprompt as teleprompt_module

    assert teleprompt_module.APEX is APEX
