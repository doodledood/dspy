from __future__ import annotations

import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import dspy
from dspy.adapters import Adapter
from dspy.clients.lm import LM
from dspy.primitives import Prediction
from dspy.utils.exceptions import AdapterParseError

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


Mode = Literal["failure", "success"]


@dataclass(frozen=True)
class _AnalysisTask:
    mode: Mode
    index: int
    record: TrainExampleRecord


def _analyze_single_example(
    record: TrainExampleRecord,
    *,
    index: int,
    mode: Mode,
    analysis_lm: LM,
    analysis_adapter: Adapter,
    runtime: RuntimeTools,
    tracker: ExperimentTracker,
    success_threshold: float,
    min_metric: float,
    max_metric: float,
    format_execution_flow: Callable[[list[ExecutionFlowEntry]], str],
    iteration: int | None,
    available_predictor_names: list[str] | None = None,
) -> Prediction | None:
    signature_class = FailureAnalysisSignature if mode == "failure" else SuccessAnalysisSignature
    predictor = dspy.Predict(signature_class)
    prompt_text = getattr(predictor.signature, "instructions", "")

    example_inputs = record.example.inputs().toDict()
    example_labels = record.example.labels().toDict()
    execution_flow_str = format_execution_flow(record.execution_flow)

    call_inputs: dict[str, object] = {
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
        "example_index": index,
    }

    validation_error: list[str] = []

    def analysis_reward_fn(_, prediction: Prediction | None) -> float:
        """Validate analysis predictor names - all logic embedded for dspy.Refine."""
        # Capture context variables needed for validation
        known_predictor_names = available_predictor_names
        analysis_mode = mode

        if prediction is None or known_predictor_names is None:
            validation_error[:] = []
            return 1.0

        predictor_field = "involved_predictors" if analysis_mode == "failure" else "contributing_predictors"
        predictors = getattr(prediction, predictor_field, []) or []

        invalid_predictors = [p for p in predictors if p and p not in known_predictor_names]
        if invalid_predictors:
            error_message = (
                f"Analysis returned unknown predictor(s) {invalid_predictors}. "
                f"Valid predictors: {sorted(known_predictor_names)}"
            )
            validation_error[:] = [error_message]
            return 0.0

        validation_error[:] = []
        return 1.0

    with tracker.span(f"apex.analysis.{mode}", inputs=span_inputs, attributes=attributes) as span:
        try:
            with dspy.context(lm=analysis_lm, adapter=analysis_adapter):
                if available_predictor_names:
                    validator = dspy.Refine(
                        module=predictor,
                        N=3,
                        reward_fn=analysis_reward_fn,
                        threshold=1.0,
                        fail_count=3,
                    )
                    result = validator(**call_inputs)
                else:
                    result = predictor(**call_inputs)

            if validation_error:
                raise ValueError(validation_error[0])

            if span and hasattr(span, "set_outputs"):
                try:
                    payload = {
                        "prompt": prompt_text,
                        "analysis": to_serializable(result),
                    }
                    if mode == "failure":
                        payload["potential_root_causes"] = getattr(result, "potential_root_causes", [])
                    else:
                        payload["potential_success_patterns"] = getattr(result, "potential_success_patterns", [])
                    span.set_outputs(payload)
                except Exception:  # pragma: no cover - defensive
                    pass

            return result
        except AdapterParseError as exc:
            if span and hasattr(span, "set_attribute"):
                try:
                    span.set_attribute("error", str(exc)[:200])
                except Exception:  # pragma: no cover - defensive
                    pass
            if mode == "success":
                return None
            raise
        except Exception as exc:
            if span and hasattr(span, "set_attribute"):
                try:
                    span.set_attribute("error", str(exc)[:200])
                except Exception:  # pragma: no cover - defensive
                    pass
            raise


def _log_analysis_results(
    mode: Mode, analyses: Sequence[Prediction], log_fn: Callable[[str, Verbosity], None], runtime: RuntimeTools
) -> None:
    if not analyses or not runtime.is_enabled(Verbosity.DETAILED):
        return

    for index, analysis in enumerate(analyses, start=1):
        categories = getattr(analysis, "categories", [])
        if len(categories) > 1:
            category_str = f"{categories[0]}+{len(categories) - 1}"
        else:
            category_str = categories[0] if categories else "unknown"

        if mode == "failure":
            root_causes = getattr(analysis, "potential_root_causes", [])
            cause_str = root_causes[0] if root_causes else "unknown cause"
            if len(root_causes) > 1:
                cause_str += f" (+{len(root_causes) - 1} alt)"
            log_fn(
                f"APEX: failure analysis #{index} ({category_str}) → {cause_str}",
                level=Verbosity.DETAILED,
            )
        else:
            patterns = getattr(analysis, "potential_success_patterns", [])
            pattern_str = patterns[0] if patterns else "unknown pattern"
            if len(patterns) > 1:
                pattern_str += f" (+{len(patterns) - 1} alt)"
            log_fn(
                f"APEX: success analysis #{index} ({category_str}) → {pattern_str}",
                level=Verbosity.DETAILED,
            )


def _run_analysis_tasks(
    tasks: Sequence[_AnalysisTask],
    *,
    description: str,
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
    level: Verbosity = Verbosity.DETAILED,
    available_predictor_names: list[str] | None = None,
) -> dict[Mode, list[Prediction]]:
    if not tasks:
        return {"failure": [], "success": []}

    log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))

    def process(task: _AnalysisTask) -> Prediction:
        return _analyze_single_example(
            task.record,
            index=task.index,
            mode=task.mode,
            analysis_lm=analysis_lm,
            analysis_adapter=analysis_adapter,
            runtime=runtime,
            tracker=tracker,
            success_threshold=success_threshold,
            min_metric=min_metric,
            max_metric=max_metric,
            format_execution_flow=format_execution_flow,
            iteration=iteration,
            available_predictor_names=available_predictor_names,
        )

    analyses = runtime.parallel_execute(
        tasks,
        process,
        description=description,
        level=level,
    )

    failure_results: list[Prediction] = []
    success_results: list[Prediction] = []

    for task, analysis in zip(tasks, analyses, strict=False):
        if isinstance(analysis, Exception):
            raise analysis
        if analysis is None:
            continue
        if task.mode == "failure":
            failure_results.append(analysis)
        else:
            success_results.append(analysis)

    _log_analysis_results("failure", failure_results, log_fn, runtime)
    _log_analysis_results("success", success_results, log_fn, runtime)

    return {"failure": failure_results, "success": success_results}


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
            summary_text = (
                _normalize_whitespace(change.change_summary) if change.change_summary else "No summary provided"
            )
            magnitude = change.change_magnitude.value
            lines.append(f"    * {predictor_name} [{magnitude}]: {summary_text}")

    return "\n".join(lines)


def analyze_examples(
    records: list[TrainExampleRecord],
    *,
    mode: Mode,
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
    available_predictor_names: list[str] | None = None,
) -> list[Prediction]:
    if not records:
        return []

    tasks = [_AnalysisTask(mode=mode, index=idx, record=record) for idx, record in enumerate(records)]
    results = _run_analysis_tasks(
        tasks,
        description=f"APEX: analyzing {mode}s",
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
        level=Verbosity.DETAILED,
        available_predictor_names=available_predictor_names,
    )
    return results[mode]


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
    available_predictor_names: list[str] | None = None,
) -> list[Prediction]:
    if not success_records or failure_count == 0:
        return []
    tasks = [_AnalysisTask(mode="success", index=idx, record=record) for idx, record in enumerate(success_records)]
    results = _run_analysis_tasks(
        tasks,
        description="APEX: analyzing successes",
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
        level=Verbosity.DETAILED,
        available_predictor_names=available_predictor_names,
    )
    return results["success"]


def analyze_record(
    record: TrainExampleRecord,
    *,
    mode: Mode,
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
    example_index: int = 0,
    available_predictor_names: list[str] | None = None,
) -> Prediction | None:
    return _analyze_single_example(
        record,
        index=example_index,
        mode=mode,
        analysis_lm=analysis_lm,
        analysis_adapter=analysis_adapter,
        runtime=runtime,
        tracker=tracker,
        success_threshold=success_threshold,
        min_metric=min_metric,
        max_metric=max_metric,
        format_execution_flow=format_execution_flow,
        iteration=iteration,
        available_predictor_names=available_predictor_names,
    )


def analyze_failures_and_successes(
    *,
    failure_records: list[TrainExampleRecord],
    success_records: list[TrainExampleRecord],
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
    available_predictor_names: list[str] | None = None,
) -> tuple[list[Prediction], list[Prediction]]:
    if not failure_records:
        return [], []

    tasks: list[_AnalysisTask] = [
        _AnalysisTask(mode="failure", index=idx, record=record) for idx, record in enumerate(failure_records)
    ]

    include_success = bool(success_records)
    if include_success:
        tasks.extend(
            _AnalysisTask(mode="success", index=idx, record=record) for idx, record in enumerate(success_records)
        )

    results = _run_analysis_tasks(
        tasks,
        description="APEX: analyzing failures/successes",
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
        available_predictor_names=available_predictor_names,
    )

    success_summaries = results["success"] if include_success else []
    return results["failure"], success_summaries


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
        log_fn("APEX: No hypotheses to generate (no failures or num_hypotheses=0)", Verbosity.DETAILED)
        return []

    log_fn = log or (lambda message, level=Verbosity.NORMAL: runtime.log(message, level))
    shuffled_failures = list(failure_summaries)
    rng.shuffle(shuffled_failures)

    log_fn(
        f"APEX: Generating up to {num_hypotheses} hypotheses from {len(failure_summaries)} failures",
        Verbosity.DETAILED,
    )

    span_attributes = {
        "iteration": iteration if iteration is not None else -1,
        "num_failures": len(failure_summaries),
        "num_successes": len(success_summaries),
    }

    available_prompts = snapshot.prompts or {}
    available_predictor_names = list(available_prompts.keys())

    def normalize_predictor_name(raw_name: str) -> str:
        """Return the canonical predictor name or raise when it cannot be resolved."""

        normalized = (raw_name or "").strip()
        if not normalized:
            return normalized
        if normalized in available_prompts:
            return normalized

        matches = [candidate for candidate in available_predictor_names if candidate.lower() == normalized.lower()]
        if len(matches) == 1:
            return matches[0]

        known_predictors = sorted(available_prompts.keys())
        raise ValueError(
            f"APEX hypothesis generation aborted: unknown predictor '{raw_name}'. Known predictors: {known_predictors}"
        )

    def build_program_flow_text() -> str:
        """Return the text describing the DAG structure plus each predictor prompt."""

        if not available_prompts:
            return snapshot.flow_description

        lines: list[str] = [
            f"Program structure: {snapshot.flow_description}",
            "",
            "Predictor prompts and configurations:",
        ]
        for predictor_name, prompt in available_prompts.items():
            prompt_text = prompt if prompt else "(no prompt provided)"
            lines.append(f"\n### {predictor_name} ###")
            lines.append(prompt_text)
        return "\n".join(lines)

    def build_failure_records() -> tuple[list[FailureSummaryRecord], Counter[str]]:
        """Normalize predictor names and collect aggregate failure metadata."""

        category_counts: Counter[str] = Counter()
        records: list[FailureSummaryRecord] = []
        for failure in shuffled_failures:
            categories = getattr(failure, "categories", []) or ["uncategorized"]
            for category in categories:
                category_counts[category] += 1
            normalized_predictors = [
                normalize_predictor_name(predictor)
                for predictor in list(getattr(failure, "involved_predictors", []) or [])
            ]
            records.append(
                FailureSummaryRecord(
                    potential_root_causes=getattr(failure, "potential_root_causes", []) or ["Unknown cause"],
                    involved_predictors=[p for p in normalized_predictors if p],
                    categories=categories,
                    context=getattr(failure, "context", ""),
                    key_details=getattr(failure, "key_details", ""),
                )
            )
        return records, category_counts

    def build_success_records() -> tuple[list[SuccessSummaryRecord], Counter[str]]:
        """Normalize predictor names and collect aggregate success metadata."""

        category_counts: Counter[str] = Counter()
        records: list[SuccessSummaryRecord] = []
        for success in success_summaries:
            categories = getattr(success, "categories", []) or ["uncategorized"]
            for category in categories:
                category_counts[category] += 1
            normalized_predictors = [
                normalize_predictor_name(predictor)
                for predictor in list(getattr(success, "contributing_predictors", []) or [])
            ]
            records.append(
                SuccessSummaryRecord(
                    potential_root_causes=getattr(success, "potential_success_patterns", []) or ["Unknown pattern"],
                    contributing_predictors=[p for p in normalized_predictors if p],
                    categories=categories,
                    context=getattr(success, "context", ""),
                    key_details=getattr(success, "key_details", ""),
                )
            )
        return records, category_counts

    failure_records, failure_category_counts = build_failure_records()
    success_records, success_category_counts = build_success_records()

    total_examples = len(failure_summaries) + len(success_summaries)
    success_rate_percentage = (len(success_summaries) / total_examples * 100.0) if total_examples else 0.0

    predictor_module = dspy.Predict(HypothesisGenerationSignature)
    prompt_text = getattr(predictor_module.signature, "instructions", "")

    validation_error: list[str] = []

    def hypothesis_reward_fn(_, prediction: Prediction | None) -> float:
        """Validate hypothesis generation - all logic embedded for dspy.Refine."""
        # Capture context variables needed for validation
        max_hypotheses = num_hypotheses
        known_prompts = available_prompts

        if prediction is None:
            error_message = "APEX hypothesis generation failed: no prediction returned for validation"
            validation_error[:] = [error_message]
            return 0.0

        hypotheses = getattr(prediction, "hypotheses", None) or []
        num_hypotheses_generated = len(hypotheses)

        # Must have 1-N hypotheses (never 0, never > N)
        if num_hypotheses_generated == 0:
            error_message = (
                "APEX hypothesis generation failed: must generate at least 1 hypothesis. "
                "Even if all issues are non-fixable via prompts, generate 1 hypothesis documenting the non-fixable issues "
                "with empty prompt_changes."
            )
            validation_error[:] = [error_message]
            return 0.0

        if num_hypotheses_generated > max_hypotheses:
            error_message = (
                f"APEX hypothesis generation failed: generated {num_hypotheses_generated} hypotheses "
                f"but maximum is {max_hypotheses}. Return at most {max_hypotheses} hypotheses."
            )
            validation_error[:] = [error_message]
            return 0.0

        # Validate each hypothesis
        for idx, spec in enumerate(hypotheses, start=1):
            fixable = getattr(spec, "fixable_root_causes", []) or []
            non_fixable = getattr(spec, "non_fixable_root_causes", []) or []
            prompt_changes = getattr(spec, "prompt_changes", {}) or {}

            # Rule 1: Must have at least one root cause (fixable OR non-fixable)
            if not fixable and not non_fixable:
                error_message = (
                    f"APEX hypothesis #{idx} validation failed: must have at least 1 root cause. "
                    "Either fixable_root_causes or non_fixable_root_causes (or both) must be non-empty."
                )
                validation_error[:] = [error_message]
                return 0.0

            # Rule 2: Empty prompt_changes ONLY allowed when no fixable causes and some non-fixable causes
            if not prompt_changes:
                if fixable:
                    error_message = (
                        f"APEX hypothesis #{idx} validation failed: prompt_changes is empty but fixable_root_causes is not empty. "
                        f"If there are fixable issues ({len(fixable)} found), prompt_changes must contain at least one change."
                    )
                    validation_error[:] = [error_message]
                    return 0.0
                if not non_fixable:
                    error_message = (
                        f"APEX hypothesis #{idx} validation failed: prompt_changes is empty but non_fixable_root_causes is also empty. "
                        "Empty prompt_changes is only valid when documenting non-fixable issues (non_fixable_root_causes must be non-empty)."
                    )
                    validation_error[:] = [error_message]
                    return 0.0

            # Validate predictor names in prompt_changes
            invalid_predictors = [name for name in prompt_changes if name not in known_prompts]
            if invalid_predictors:
                known_predictor_names = sorted(known_prompts.keys())
                error_message = (
                    f"APEX hypothesis #{idx} validation failed: unknown predictor(s) "
                    f"{sorted(invalid_predictors)}. Known predictors: {known_predictor_names}"
                )
                validation_error[:] = [error_message]
                return 0.0

        # All validations passed
        validation_error[:] = []
        return 1.0

    program_flow = build_program_flow_text()
    history_text = build_hypothesis_history_text(
        include_history=include_history,
        candidate_history=candidate_history,
    )
    best_val_text = f"{best_val_score:.4f}" if best_val_score is not None else "N/A"

    generation_payload = {
        "failure_analyses": failure_records,
        "success_analyses": success_records,
        "program_flow": program_flow,
        "failure_category_counts": dict(failure_category_counts),
        "success_category_counts": dict(success_category_counts),
        "success_rate_percentage": success_rate_percentage,
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
            answers = getattr(hypothesis_lm, "answers", None)
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

                hypothesis_lm.answers = _SharedIterator(answers)

            validator = dspy.Refine(
                module=predictor_module,
                N=3,
                reward_fn=hypothesis_reward_fn,
                threshold=1.0,
                fail_count=3,
            )

            try:
                result = validator(**generation_payload)
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
            if hypotheses:
                validated_specs = list(hypotheses)
            else:
                validated_specs = []
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
                    "failure_category_counts": dict(failure_category_counts),
                    "success_category_counts": dict(success_category_counts),
                    "success_rate_percentage": success_rate_percentage,
                    "prompt": prompt_text,
                    "hypotheses": [to_serializable(spec) for spec in validated_specs],
                }
                span.set_outputs(payload)
            except Exception:  # pragma: no cover - defensive
                pass

    if runtime.is_enabled(Verbosity.DETAILED):
        for idx, spec in enumerate(validated_specs, start=1):
            log_fn(
                "APEX: hypothesis #{} ({}) targeting {} [impact={:.2f}, generalizability={:.2f}]".format(
                    idx,
                    spec.strategy,
                    ", ".join(spec.fixable_root_causes) or "no fixable causes",
                    spec.impact_score,
                    spec.generalizability_score,
                ),
                Verbosity.DETAILED,
            )
            for predictor_name, changes in spec.prompt_changes.items():
                if len(changes.new_prompt) > 200:
                    prompt_preview = f"{changes.new_prompt[:200]}..."
                else:
                    prompt_preview = changes.new_prompt
                log_fn(
                    f"  → {predictor_name}: {prompt_preview}",
                    Verbosity.DETAILED,
                )
                if changes.change_summary:
                    log_fn(
                        f"     Summary: {changes.change_summary}",
                        Verbosity.DETAILED,
                    )

    return validated_specs
