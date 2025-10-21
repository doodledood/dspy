from __future__ import annotations

import random
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

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
from .modules import (
    FailureAnalysisModule,
    HypothesisGenerationModule,
    ParetoMergeModule,
    SuccessAnalysisModule,
)
from .runtime import RuntimeTools
from .serialization import to_serializable
from .tracker import ExperimentTracker
from .types import Verbosity


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


Mode = Literal["failure", "success"]
_SCORE_TOLERANCE = 1e-9


def _prompt_map_from_candidate(candidate: CandidateRecord) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for name, predictor in candidate.program.named_predictors():
        prompts[name] = getattr(predictor.signature, "instructions", "")
    return prompts


def _summarize_prompt_changes(candidate: CandidateRecord) -> dict[str, str]:
    if candidate.hypothesis is None or not candidate.hypothesis.prompt_changes:
        return {}

    summaries: dict[str, str] = {}
    for predictor_name, change in candidate.hypothesis.prompt_changes.items():
        summary_pieces = [
            change.change_summary or "No summary provided",
            f"magnitude={change.change_magnitude.value}",
        ]
        summaries[predictor_name] = " | ".join(summary_pieces)
    return summaries


def _summarize_candidate(candidate: CandidateRecord, label: str) -> str:
    lines = [
        f"{label}",
        f"iteration={candidate.iteration}",
        f"overall_score={candidate.overall_score:.4f}",
    ]
    if candidate.hypothesis is not None:
        lines.append(f"strategy={candidate.hypothesis.strategy}")
        if candidate.hypothesis.prompt_changes:
            predictors = ", ".join(candidate.hypothesis.prompt_changes.keys())
            lines.append(f"prompt_changes={predictors}")
    return "\n".join(lines)


def _format_per_example_notes(primary: CandidateRecord, partner: CandidateRecord) -> str:
    max_len = max(len(primary.per_example_scores), len(partner.per_example_scores))
    if max_len == 0:
        return "No per-example scores available."

    primary_wins = 0
    partner_wins = 0
    ties = 0
    for idx in range(max_len):
        primary_score = primary.per_example_scores[idx] if idx < len(primary.per_example_scores) else float("-inf")
        partner_score = partner.per_example_scores[idx] if idx < len(partner.per_example_scores) else float("-inf")
        if primary_score > partner_score + _SCORE_TOLERANCE:
            primary_wins += 1
        elif partner_score > primary_score + _SCORE_TOLERANCE:
            partner_wins += 1
        else:
            ties += 1

    return (
        f"Primary wins {primary_wins} example(s); "
        f"Partner wins {partner_wins} example(s); "
        f"Ties {ties} example(s); Total compared {max_len}."
    )


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
    if mode == "failure":
        analysis_module: FailureAnalysisModule | SuccessAnalysisModule = FailureAnalysisModule()
        prompt_text = getattr(analysis_module.predictor.signature, "instructions", "")
    else:
        analysis_module = SuccessAnalysisModule()
        prompt_text = getattr(analysis_module.predictor.signature, "instructions", "")

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

    with tracker.span(f"apex.analysis.{mode}", inputs=span_inputs, attributes=attributes) as span:
        try:
            result = analysis_module(lm=analysis_lm, adapter=analysis_adapter, **call_inputs)

            if available_predictor_names:
                predictor_field = "involved_predictors" if mode == "failure" else "contributing_predictors"
                predictors = getattr(result, predictor_field, []) or []
                invalid_predictors = [p for p in predictors if p and p not in available_predictor_names]
                if invalid_predictors:
                    raise ValueError(
                        "Analysis returned unknown predictor(s) %s. Valid predictors: %s"
                        % (invalid_predictors, sorted(available_predictor_names))
                    )

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


def build_hypothesis_history_text(
    *,
    candidate_history: Sequence[CandidateRecord] | None,
    selection_strategy: str,
) -> str:
    lines: list[str] = ["Previous hypotheses evaluated (oldest first):"]
    lines.append(f"Selection strategy in effect: {selection_strategy}")

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
        best_label = "best so far" if selection_strategy == "best_on_val" else "pareto frontier candidate"
        lines.append(f"- Iteration {baseline_candidate.iteration} baseline score={baseline_score_text} ({best_label})")
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
    selection_strategy: str = "best_on_val",
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
        """Return the text describing the program source code and each predictor prompt."""

        lines: list[str] = [
            "Program Source Code:",
            "```python",
            '"""',
            snapshot.source_code,
            '"""',
            "```",
        ]

        if available_prompts:
            lines.extend(["", "Predictor prompts and configurations:"])
            for predictor_name, prompt in available_prompts.items():
                prompt_text = prompt if prompt else "(no prompt provided)"
                lines.append(f"\n### {predictor_name} ###")
                lines.append(f'"""\n{prompt_text}\n"""')

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

    hypothesis_module = HypothesisGenerationModule()
    prompt_text = hypothesis_module.instructions

    program_flow = build_program_flow_text()
    history_text = (
        build_hypothesis_history_text(
            candidate_history=candidate_history,
            selection_strategy=selection_strategy,
        )
        if include_history
        else "N/A"
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

    validated_specs: list[HypothesisSpec] = []

    with span_cm as span:
        try:
            validated_specs = hypothesis_module.generate(
                payload=generation_payload,
                lm=hypothesis_lm,
                adapter=hypothesis_adapter,
                num_hypotheses=num_hypotheses,
                available_prompts=available_prompts,
            )
        except AdapterParseError as exc:  # pragma: no cover - defensive
            runtime.log(
                f"APEX: Hypothesis generation parse error: {str(exc)[:200]}",
                Verbosity.DETAILED,
                log_level="warning",
            )
            return []
        except ValueError as exc:
            runtime.log(
                f"APEX: Hypothesis generation validation failed: {str(exc)[:200]}",
                Verbosity.DETAILED,
                log_level="warning",
            )
            raise
        else:
            if not validated_specs:
                log_fn(
                    "APEX: Hypothesis LM returned no hypotheses; treating as no-op for this iteration",
                    Verbosity.DETAILED,
                )
                return []

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


def generate_merge_hypotheses(
    *,
    baseline_candidate: CandidateRecord,
    partner_candidate: CandidateRecord,
    runtime: RuntimeTools,
    hypothesis_lm: LM,
    hypothesis_adapter: Adapter,
    iteration: int | None,
    tracker: ExperimentTracker | None,
    snapshot: ProgramSnapshot,
    candidate_history: list[CandidateRecord],
    best_val_score: float | None,
    selection_strategy: str,
    include_history: bool,
    success_rate_percentage: float,
) -> list[HypothesisSpec]:
    """Generate paired merged hypotheses combining two Pareto candidates from the frontier."""

    # Build available prompts from snapshot
    available_prompts = snapshot.prompts if snapshot.prompts else {}

    def build_program_flow_text() -> str:
        """Return the text describing the program source code and each predictor prompt."""

        lines: list[str] = [
            "Program Source Code:",
            "```python",
            '"""',
            snapshot.source_code,
            '"""',
            "```",
        ]

        if available_prompts:
            lines.extend(["", "Predictor prompts and configurations:"])
            for predictor_name, prompt in available_prompts.items():
                prompt_text = prompt if prompt else "(no prompt provided)"
                lines.append(f"\n### {predictor_name} ###")
                lines.append(f'"""\n{prompt_text}\n"""')

        return "\n".join(lines)

    program_flow = build_program_flow_text()
    history_text = (
        build_hypothesis_history_text(
            candidate_history=candidate_history,
            selection_strategy=selection_strategy,
        )
        if include_history
        else "N/A"
    )
    best_val_text = f"{best_val_score:.4f}" if best_val_score is not None else "N/A"

    call_inputs = {
        "primary_prompts": _prompt_map_from_candidate(baseline_candidate),
        "partner_prompts": _prompt_map_from_candidate(partner_candidate),
        "program_flow": program_flow,
        "success_rate_percentage": success_rate_percentage,
        "best_validation_score": best_val_text,
        "current_iteration": iteration if iteration is not None else -1,
        "hypothesis_history": history_text,
    }

    attributes = {
        "iteration": iteration if iteration is not None else -1,
        "baseline_iteration": baseline_candidate.iteration,
        "partner_iteration": partner_candidate.iteration,
    }

    merge_inputs = to_serializable(call_inputs)
    merge_module = ParetoMergeModule()

    span_cm = (
        tracker.span("apex.merge_hypothesis", inputs=merge_inputs, attributes=attributes)
        if tracker is not None
        else nullcontext(None)
    )

    with span_cm as span:
        try:
            hypotheses = merge_module.merge(
                inputs=call_inputs,
                lm=hypothesis_lm,
                adapter=hypothesis_adapter,
            )
        except AdapterParseError as exc:  # pragma: no cover - defensive
            runtime.log(
                f"APEX: Merge hypothesis generation parse error: {str(exc)[:200]}",
                Verbosity.DETAILED,
                log_level="warning",
            )
            return []
        except Exception as exc:  # pragma: no cover - defensive
            runtime.log(
                f"APEX: Merge hypothesis generation failed: {str(exc)[:200]}",
                Verbosity.DETAILED,
                log_level="warning",
            )
            return []

        if len(hypotheses) != 2:
            runtime.log(
                "APEX: Merge hypothesis generation produced insufficient prompt changes; skipping merge candidates.",
                Verbosity.DETAILED,
                log_level="warning",
            )
            return []

        if span and hasattr(span, "set_outputs"):
            try:
                span.set_outputs(
                    {
                        "primary_hypothesis": to_serializable(hypotheses[0]),
                        "partner_hypothesis": to_serializable(hypotheses[1]),
                    }
                )
            except Exception:  # pragma: no cover - defensive
                pass

        runtime.log(
            "APEX: Generated Pareto merge hypotheses blending baseline and partner candidates",
            Verbosity.DETAILED,
        )
        return hypotheses
