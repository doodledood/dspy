from __future__ import annotations

import random
from typing import Callable, Sequence

import dspy
from dspy.adapters import Adapter
from dspy.clients.lm import LM
from dspy.primitives import Prediction

from .models import (
    CandidateRecord,
    ExecutionFlowEntry,
    HypothesisSpec,
    ProgramSnapshot,
    TrainExampleRecord,
)
from .runtime import RuntimeTools
from .signatures import (
    FailureAnalysisSignature,
    HypothesisGenerationSignature,
    SuccessAnalysisSignature,
)
from .types import Verbosity


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_hypothesis_history_text(
    *, include_history: bool, candidate_history: Sequence[CandidateRecord] | None
) -> str:
    if not include_history:
        return "N/A"

    lines: list[str] = ["Previous hypotheses evaluated (oldest first):"]

    if not candidate_history:
        lines.append("- No hypotheses have been tried yet.")
        return "\n".join(lines)

    sorted_candidates = sorted(
        (candidate for candidate in candidate_history if candidate.hypothesis),
        key=lambda candidate: candidate.iteration,
    )

    if not sorted_candidates:
        lines.append("- No hypotheses have been tried yet.")
        return "\n".join(lines)

    for candidate in sorted_candidates:
        score = candidate.overall_score
        score_text = f"{score:.4f}" if score is not None else "N/A"
        lines.append(f"- Iteration {candidate.iteration} (score={score_text}):")

        changes = candidate.hypothesis.prompt_changes
        if not changes:
            lines.append("    * No prompt changes recorded")
            continue

        for predictor_name, change in changes.items():
            rationale_text = (
                _normalize_whitespace(change.rationale) if change.rationale else "No rationale provided"
            )
            magnitude = change.change_magnitude.value
            lines.append(f"    * {predictor_name} [{magnitude}]: {rationale_text}")

    return "\n".join(lines)


def analyze_examples(
    records: list[TrainExampleRecord],
    *,
    mode: str,
    analysis_lm: LM,
    analysis_adapter: Adapter,
    runtime: RuntimeTools,
    success_threshold: float,
    min_metric: float,
    max_metric: float,
    format_execution_flow: Callable[[list[ExecutionFlowEntry]], str],
    log: Callable[[str, Verbosity], None] | None = None,
) -> list[Prediction]:
    if not records:
        return []

    signature_class = FailureAnalysisSignature if mode == "failure" else SuccessAnalysisSignature
    log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))

    def process(record: TrainExampleRecord) -> Prediction:
        with dspy.context(lm=analysis_lm, adapter=analysis_adapter):
            predictor = dspy.Predict(signature_class)

            inputs = record.example.inputs().toDict()
            expected = record.example.labels().toDict()

            execution_flow_str = format_execution_flow(record.execution_flow)

            if mode == "failure":
                call_inputs = {
                    "problem": str(inputs),
                    "prediction": str(record.prediction) if record.prediction else "",
                    "expected": str(expected),
                    "error": record.error or "",
                    "execution_flow": execution_flow_str,
                    "metric_score": record.metric_score,
                    "metric_feedback": record.metric_feedback or "N/A",
                    "success_threshold": success_threshold,
                    "min_metric": min_metric,
                    "max_metric": max_metric,
                }
                result = predictor(**call_inputs)
            else:
                result = predictor(
                    problem=str(inputs),
                    prediction=str(record.prediction) if record.prediction else "",
                    expected=str(expected),
                    execution_flow=execution_flow_str,
                    metric_score=record.metric_score,
                    metric_feedback=record.metric_feedback or "N/A",
                    success_threshold=success_threshold,
                    min_metric=min_metric,
                    max_metric=max_metric,
                )

        return result

    analyses = runtime.parallel_execute(
        records,
        process,
        description=f"APEX: analyzing {mode}s",
        level=Verbosity.HIGH,
    )

    if runtime.is_enabled(Verbosity.HIGH):
        for index, analysis in enumerate(analyses, start=1):
            if mode == "failure":
                log_fn(
                    f"APEX: failure analysis #{index} ({analysis.category}) → {analysis.root_cause}",
                    level=Verbosity.HIGH,
                )
            else:
                log_fn(
                    f"APEX: success analysis #{index} ({analysis.category}) → {analysis.success_pattern}",
                    level=Verbosity.HIGH,
                )
    return analyses


def analyze_successes(
    success_records: list[TrainExampleRecord],
    *,
    failure_count: int,
    analysis_lm: LM,
    analysis_adapter: Adapter,
    runtime: RuntimeTools,
    success_threshold: float,
    min_metric: float,
    max_metric: float,
    format_execution_flow: Callable[[list[ExecutionFlowEntry]], str],
    log: Callable[[str, Verbosity], None] | None = None,
) -> list[Prediction]:
    if not success_records or failure_count == 0:
        return []
    return analyze_examples(
        success_records,
        mode="success",
        analysis_lm=analysis_lm,
        analysis_adapter=analysis_adapter,
        runtime=runtime,
        success_threshold=success_threshold,
        min_metric=min_metric,
        max_metric=max_metric,
        format_execution_flow=format_execution_flow,
        log=log,
    )


def generate_hypotheses(
    *,
    failure_summaries: list[Prediction],
    success_summaries: list[Prediction],
    snapshot: ProgramSnapshot,
    candidate_history: Sequence[CandidateRecord] | None,
    current_val_score: float | None,
    runtime: RuntimeTools,
    hypothesis_lm: LM,
    hypothesis_adapter: Adapter,
    num_hypotheses: int,
    include_history: bool,
    rng: random.Random,
    log: Callable[[str, Verbosity], None] | None = None,
) -> list[HypothesisSpec]:
    if not failure_summaries or num_hypotheses == 0:
        log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))
        log_fn("APEX: No hypotheses to generate (no failures or num_hypotheses=0)", Verbosity.HIGH)
        return []

    log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))
    shuffled_failures = list(failure_summaries)
    rng.shuffle(shuffled_failures)

    log_fn(
        f"APEX: Generating up to {num_hypotheses} hypotheses from {len(failure_summaries)} failures",
        Verbosity.HIGH,
    )

    failure_text = "\n".join([f"- {f.root_cause} (category: {f.category})" for f in shuffled_failures])
    success_text = (
        "\n".join([f"- {s.success_pattern} (category: {s.category})" for s in success_summaries])
        if success_summaries
        else "No success patterns available"
    )

    program_flow = snapshot.flow_description

    history_text = build_hypothesis_history_text(
        include_history=include_history, candidate_history=candidate_history
    )
    current_val_text = f"{current_val_score:.4f}" if current_val_score is not None else "N/A"

    with dspy.context(lm=hypothesis_lm, adapter=hypothesis_adapter):
        predictor = dspy.Predict(HypothesisGenerationSignature)
        result = predictor(
            failure_analyses=failure_text,
            success_analyses=success_text,
            program_flow=program_flow,
            current_validation_score=current_val_text,
            hypothesis_history=history_text,
            num_hypotheses=num_hypotheses,
        )

    validated_specs = result.hypotheses if result.hypotheses else []
    validated_specs.sort(key=lambda h: (h.impact_score, h.generalizability_score), reverse=True)
    validated_specs = validated_specs[:num_hypotheses]

    if runtime.is_enabled(Verbosity.HIGH):
        for idx, spec in enumerate(validated_specs, start=1):
            log_fn(
                "APEX: hypothesis #{} ({}) targeting {} [impact={:.2f}, generalizability={:.2f}]".format(
                    idx,
                    spec.strategy,
                    ", ".join(spec.fixable_root_causes) or "no fixable causes",
                    spec.impact_score,
                    spec.generalizability_score,
                ),
                Verbosity.HIGH,
            )
            for predictor_name, changes in spec.prompt_changes.items():
                if len(changes.new_prompt) > 200:
                    prompt_preview = f"{changes.new_prompt[:200]}..."
                else:
                    prompt_preview = changes.new_prompt
                log_fn(
                    f"  → {predictor_name}: {prompt_preview}",
                    Verbosity.HIGH,
                )
                if changes.rationale:
                    log_fn(
                        f"     Rationale: {changes.rationale}",
                        Verbosity.HIGH,
                    )

    return validated_specs
