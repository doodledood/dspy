"""Unit tests for the public signature classes exposed by the APEX package."""

from __future__ import annotations

from dspy.teleprompt.apex.signatures import (
    FailureAnalysisSignature,
    HypothesisGenerationSignature,
    ParetoMergeSignature,
    SuccessAnalysisSignature,
)


def test_failure_analysis_signature_declares_expected_fields() -> None:
    input_names = set(FailureAnalysisSignature.input_fields.keys())
    output_names = set(FailureAnalysisSignature.output_fields.keys())

    assert {"problem", "prediction", "expected"}.issubset(input_names)
    assert {"potential_root_causes", "categories"}.issubset(output_names)
    assert "instructions" in FailureAnalysisSignature.instructions.lower()


def test_hypothesis_generation_signature_outputs_hypothesis_specs() -> None:
    field = HypothesisGenerationSignature.model_fields["hypotheses"]

    assert "HypothesisSpec" in repr(field.annotation)
    assert "list" in repr(field.annotation).lower()
    assert "failure_analyses" in HypothesisGenerationSignature.input_fields


def test_pareto_merge_signature_exposes_pair_outputs() -> None:
    outputs = ParetoMergeSignature.output_fields

    assert "primary_hypothesis" in outputs
    assert "partner_hypothesis" in outputs
    assert "hypotheses" not in outputs  # ensures merge signature differs from generator


def test_success_analysis_signature_tracks_success_patterns() -> None:
    outputs = SuccessAnalysisSignature.output_fields

    assert "potential_success_patterns" in outputs
    assert "key_details" in outputs
    assert "metric_score" in SuccessAnalysisSignature.input_fields
