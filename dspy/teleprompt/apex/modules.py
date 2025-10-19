"""Reusable DSPy modules used by the APEX optimizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dspy
from dspy.adapters import Adapter
from dspy.clients.lm import LM
from dspy.primitives import Module, Prediction

from .models import HypothesisSpec
from .signatures import (
    FailureAnalysisSignature,
    HypothesisGenerationSignature,
    ParetoMergeSignature,
    SuccessAnalysisSignature,
)


class FailureAnalysisModule(Module):
    """Wraps the failure analysis signature for reuse."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(FailureAnalysisSignature)

    def __call__(self, *, lm: LM, adapter: Adapter, **call_inputs: Any) -> Prediction:
        with dspy.context(lm=lm, adapter=adapter):
            return self.predictor(**call_inputs)


class SuccessAnalysisModule(Module):
    """Wraps the success analysis signature for reuse."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(SuccessAnalysisSignature)

    def __call__(self, *, lm: LM, adapter: Adapter, **call_inputs: Any) -> Prediction:
        with dspy.context(lm=lm, adapter=adapter):
            return self.predictor(**call_inputs)


class HypothesisGenerationModule(Module):
    """Generates hypotheses with validation enforced via dspy.Refine."""

    def __init__(self, *, attempts: int = 3) -> None:
        super().__init__()
        self.predictor = dspy.Predict(HypothesisGenerationSignature)
        self.attempts = attempts

    @property
    def instructions(self) -> str:
        return getattr(self.predictor.signature, "instructions", "")

    def generate(
        self,
        *,
        payload: Mapping[str, Any],
        lm: LM,
        adapter: Adapter,
        num_hypotheses: int,
        available_prompts: Mapping[str, str],
    ) -> list[HypothesisSpec]:
        validation_error: list[str] = []
        max_hypotheses = num_hypotheses
        known_prompts: dict[str, str] = dict(available_prompts)

        def reward_fn(_, prediction: Prediction | None) -> float:
            if prediction is None:
                validation_error[:] = []
                return 1.0

            hypotheses = getattr(prediction, "hypotheses", None) or []
            count = len(hypotheses)
            if count == 0:
                validation_error[:] = []
                return 1.0

            if count > max_hypotheses:
                validation_error[:] = []
                return 1.0

            for idx, spec in enumerate(hypotheses, start=1):
                fixable = getattr(spec, "fixable_root_causes", []) or []
                non_fixable = getattr(spec, "non_fixable_root_causes", []) or []
                prompt_changes = getattr(spec, "prompt_changes", {}) or {}

                if not fixable and not non_fixable:
                    validation_error[:] = [
                        f"APEX hypothesis #{idx} validation failed: must have at least 1 root cause. "
                        "Either fixable_root_causes or non_fixable_root_causes (or both) must be non-empty.",
                    ]
                    return 0.0

                invalid_predictors = [name for name in prompt_changes if name not in known_prompts]
                if invalid_predictors:
                    known_predictor_names = sorted(known_prompts.keys())
                    validation_error[:] = [
                        f"APEX hypothesis #{idx} validation failed: unknown predictor(s) "
                        f"{sorted(invalid_predictors)}. Known predictors: {known_predictor_names}",
                    ]
                    return 0.0

            validation_error[:] = []
            return 1.0

        with dspy.context(lm=lm, adapter=adapter):
            answers = getattr(lm, "answers", None)
            if answers is not None and not hasattr(answers, "__deepcopy__"):

                class _SharedIterator:
                    def __init__(self, iterator):
                        self._iterator = iterator

                    def __iter__(self):
                        return self

                    def __next__(self):
                        return next(self._iterator)

                    def __deepcopy__(self, memo):
                        return self

                lm.answers = _SharedIterator(answers)

            validator = dspy.Refine(
                module=self.predictor,
                N=self.attempts,
                reward_fn=reward_fn,
                threshold=1.0,
                fail_count=self.attempts,
            )

            try:
                result = validator(**payload)
            except Exception as exc:
                if validation_error:
                    raise ValueError(validation_error[0]) from exc
                raise

        if validation_error:
            raise ValueError(validation_error[0])

        validated_specs: list[HypothesisSpec]
        if result is None:
            validated_specs = []
        else:
            hypotheses = getattr(result, "hypotheses", None)
            validated_specs = list(hypotheses or [])

        validated_specs.sort(key=lambda h: (h.impact_score, h.generalizability_score), reverse=True)
        return validated_specs[:max_hypotheses]


class ParetoMergeModule(Module):
    """Generates paired merge hypotheses that blend two Pareto candidates from the frontier."""

    def __init__(self, *, attempts: int = 3) -> None:
        super().__init__()
        self.predictor = dspy.Predict(ParetoMergeSignature)
        self.attempts = attempts

    def merge(
        self,
        *,
        inputs: Mapping[str, Any],
        lm: LM,
        adapter: Adapter,
    ) -> list[HypothesisSpec]:
        def reward_fn(_, prediction: Prediction | None) -> float:
            if prediction is None:
                return 0.0

            primary = getattr(prediction, "primary_hypothesis", None)
            partner = getattr(prediction, "partner_hypothesis", None)
            if primary is None or partner is None:
                return 0.0

            primary_changes = getattr(primary, "prompt_changes", {}) or {}
            partner_changes = getattr(partner, "prompt_changes", {}) or {}
            if not primary_changes or not partner_changes:
                return 0.0

            return 1.0

        with dspy.context(lm=lm, adapter=adapter):
            validator = dspy.Refine(
                module=self.predictor,
                N=self.attempts,
                reward_fn=reward_fn,
                threshold=1.0,
                fail_count=self.attempts,
            )
            result = validator(**inputs)

        if result is None:
            return []

        hypotheses: list[HypothesisSpec] = []
        for attribute in ("primary_hypothesis", "partner_hypothesis"):
            hypothesis = getattr(result, attribute, None)
            if hypothesis is None:
                return []
            if not hypothesis.prompt_changes:
                return []
            if not getattr(hypothesis, "strategy", ""):
                hypothesis.strategy = "Pareto merge refinement"
            hypotheses.append(hypothesis)

        return hypotheses


__all__ = [
    "FailureAnalysisModule",
    "HypothesisGenerationModule",
    "ParetoMergeModule",
    "SuccessAnalysisModule",
]
