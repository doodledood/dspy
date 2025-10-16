from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Callable, Sequence

import dspy
from dspy.adapters import Adapter
from dspy.clients.lm import LM
from dspy.primitives import Prediction

from .models import (
    CandidateRecord,
    ExecutionFlowEntry,
    FailureSummaryRecord,
    HypothesisSpec,
    ProgramSnapshot,
    SuccessSummaryRecord,
    TrainExampleRecord,
)
from .runtime import RuntimeTools
from .serialization import to_serializable
from .signatures import (
    FailureAnalysisSignature,
    HypothesisGenerationSignature,
    SuccessAnalysisSignature,
)
from .tracker import ExperimentTracker
from .types import Verbosity


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_hypothesis_history_text(*, include_history: bool, candidate_history: Sequence[CandidateRecord] | None) -> str:
    if not include_history:
        return "N/A"

    lines: list[str] = ["Previous hypotheses evaluated (oldest first):"]

    if not candidate_history:
        lines.append("- No hypotheses have been tried yet.")
        return "\n".join(lines)

    baseline_candidate = min(
        (candidate for candidate in candidate_history if candidate.iteration == 0),
        default=None,
        key=lambda candidate: candidate.iteration,
    )
    if baseline_candidate is None:
        baseline_candidate = min(candidate_history, key=lambda candidate: candidate.iteration)

    baseline_score: float | None = None
    best_so_far = None
    if baseline_candidate is not None:
        baseline_score = baseline_candidate.overall_score
        baseline_score_text = f"{baseline_score:.4f}" if baseline_score is not None else "N/A"
        lines.append(f"- Iteration {baseline_candidate.iteration} baseline score={baseline_score_text} (best so far)")
        best_so_far = baseline_score

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
        delta_text = ""
        previous_best = best_so_far
        if previous_best is not None and score is not None:
            delta = score - previous_best
            delta_text = f", delta={'+' if delta >= 0 else ''}{delta:.4f} vs prior best"
        lines.append(f"- Iteration {candidate.iteration} (score={score_text}{delta_text}):")

        if score is not None:
            best_so_far = score if previous_best is None else max(previous_best, score)

        changes = candidate.hypothesis.prompt_changes
        if not changes:
            lines.append("    * No prompt changes recorded")
            continue

        for predictor_name, change in changes.items():
            rationale_text = _normalize_whitespace(change.rationale) if change.rationale else "No rationale provided"
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
    tracker: ExperimentTracker,
    success_threshold: float,
    min_metric: float,
    max_metric: float,
    format_execution_flow: Callable[[list[ExecutionFlowEntry]], str],
    log: Callable[[str, Verbosity], None] | None = None,
    iteration: int | None = None,
) -> list[Prediction]:
    if not records:
        return []

    signature_class = FailureAnalysisSignature if mode == "failure" else SuccessAnalysisSignature
    log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))

    indexed_records = list(enumerate(records))

    def process(item: tuple[int, TrainExampleRecord]) -> Prediction:
        idx, record = item

        predictor = dspy.Predict(signature_class)
        prompt_text = getattr(predictor.signature, "instructions", "")

        example_inputs = record.example.inputs().toDict()
        example_labels = record.example.labels().toDict()
        execution_flow_str = format_execution_flow(record.execution_flow)

        call_inputs = {
            "problem": str(example_inputs),
            "prediction": str(record.prediction) if record.prediction else "",
            "expected": str(example_labels),
            "execution_flow": execution_flow_str,
            "metric_score": record.metric_score,
            "metric_feedback": record.metric_feedback or "N/A",
            "success_threshold": success_threshold,
            "min_metric": min_metric,
            "max_metric": max_metric,
        }

        if mode == "failure":
            call_inputs["error"] = record.error or ""

        span_inputs = {
            "inputs": example_inputs,
            "labels": example_labels,
            "metric_score": record.metric_score,
            "metric_feedback": record.metric_feedback,
            "prompt": prompt_text,
            "analysis_payload": to_serializable(call_inputs),
        }

        attributes = {
            "analysis_mode": mode,
            "iteration": iteration if iteration is not None else -1,
            "example_index": idx,
        }

        with tracker.span(f"apex.analysis.{mode}", inputs=span_inputs, attributes=attributes) as span:
            try:
                with dspy.context(lm=analysis_lm, adapter=analysis_adapter):
                    result = predictor(**call_inputs)

                if span and hasattr(span, "set_outputs"):
                    try:
                        payload = {
                            "prompt": prompt_text,
                            "analysis": to_serializable(result),
                        }
                        if mode == "failure":
                            payload["root_cause"] = getattr(result, "root_cause", "")
                        else:
                            payload["success_pattern"] = getattr(result, "success_pattern", "")
                        span.set_outputs(payload)
                    except Exception:  # pragma: no cover - defensive
                        pass

                return result
            except Exception as exc:
                if span and hasattr(span, "set_attribute"):
                    try:
                        span.set_attribute("error", str(exc)[:200])
                    except Exception:  # pragma: no cover - defensive
                        pass
                raise

    analyses = runtime.parallel_execute(
        indexed_records,
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
    tracker: ExperimentTracker,
    success_threshold: float,
    min_metric: float,
    max_metric: float,
    format_execution_flow: Callable[[list[ExecutionFlowEntry]], str],
    log: Callable[[str, Verbosity], None] | None = None,
    iteration: int | None = None,
) -> list[Prediction]:
    if not success_records or failure_count == 0:
        return []
    return analyze_examples(
        success_records,
        mode="success",
        analysis_lm=analysis_lm,
        analysis_adapter=analysis_adapter,
        runtime=runtime,
        tracker=tracker,
        success_threshold=success_threshold,
        min_metric=min_metric,
        max_metric=max_metric,
        format_execution_flow=format_execution_flow,
        log=log,
        iteration=iteration,
    )


def generate_hypotheses(
    *,
    failure_summaries: list[Prediction],
    success_summaries: list[Prediction],
    snapshot: ProgramSnapshot,
    candidate_history: Sequence[CandidateRecord] | None,
    best_val_score: float | None,
    runtime: RuntimeTools,
    hypothesis_lm: LM,
    hypothesis_adapter: Adapter,
    num_hypotheses: int,
    include_history: bool,
    rng: random.Random,
    log: Callable[[str, Verbosity], None] | None = None,
    iteration: int | None = None,
    tracker: ExperimentTracker | None = None,
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

    span_attributes = {
        "iteration": iteration if iteration is not None else -1,
        "num_failures": len(failure_summaries),
        "num_successes": len(success_summaries),
    }

    predictor = dspy.Predict(HypothesisGenerationSignature)
    prompt_text = getattr(predictor.signature, "instructions", "")

    failure_records = [
        FailureSummaryRecord(
            root_cause=getattr(f, "root_cause", ""),
            involved_predictors=list(getattr(f, "involved_predictors", []) or []),
            context=getattr(f, "context", "") or "",
            category=getattr(f, "category", "") or "",
            key_details=getattr(f, "key_details", "") or "",
        )
        for f in shuffled_failures
    ]

    success_records = [
        SuccessSummaryRecord(
            success_pattern=getattr(s, "success_pattern", ""),
            contributing_predictors=list(getattr(s, "contributing_predictors", []) or []),
            context=getattr(s, "context", "") or "",
            category=getattr(s, "category", "") or "",
            key_details=getattr(s, "key_details", "") or "",
        )
        for s in success_summaries
    ]

    program_flow = snapshot.flow_description
    history_text = build_hypothesis_history_text(
        include_history=include_history,
        candidate_history=candidate_history,
    )
    best_val_text = f"{best_val_score:.4f}" if best_val_score is not None else "N/A"

    generation_payload = {
        "failure_analyses": failure_records,
        "success_analyses": success_records,
        "program_flow": program_flow,
        "best_validation_score": best_val_text,
        "current_iteration": iteration if iteration is not None else -1,
        "hypothesis_history": history_text,
        "num_hypotheses": num_hypotheses,
    }

    span_inputs = {
        "best_val_score": best_val_score,
        "prompt": prompt_text,
        "generation_payload": to_serializable(generation_payload),
    }

    span_cm = (
        tracker.span("apex.generate_hypotheses", inputs=span_inputs, attributes=span_attributes)
        if tracker is not None
        else nullcontext(None)
    )

    with span_cm as span:
        with dspy.context(lm=hypothesis_lm, adapter=hypothesis_adapter):
            result = predictor(**generation_payload)

        validated_specs = result.hypotheses if result.hypotheses else []
        validated_specs.sort(key=lambda h: (h.impact_score, h.generalizability_score), reverse=True)
        validated_specs = validated_specs[:num_hypotheses]

        if span and hasattr(span, "set_outputs"):
            try:
                payload = {
                    "num_hypotheses_generated": len(validated_specs),
                    "best_val_score": best_val_score or 0.0,
                    "current_iteration": iteration if iteration is not None else -1,
                    "failure_analyses": [to_serializable(record) for record in failure_records],
                    "success_analyses": [to_serializable(record) for record in success_records],
                    "prompt": prompt_text,
                    "hypotheses": [to_serializable(spec) for spec in validated_specs],
                }
                span.set_outputs(payload)
            except Exception:  # pragma: no cover - defensive
                pass

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
