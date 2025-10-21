"""Evaluation utilities for the APEX optimizer."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import dspy
from dspy.primitives import Example, Module, Prediction

from .execution_flow import extract_full_execution_flow_with_coverage
from .models import CandidateRecord, HypothesisSpec, TrainExampleRecord
from .runtime import RuntimeTools
from .serialization import to_serializable
from .tracker import ExperimentTracker
from .types import LogLevel, MetricFn, TraceEntry, Verbosity

_LM_HISTORY_KEYS = (
    "prompt",
    "messages",
    "outputs",
    "usage",
    "kwargs",
    "cost",
    "model",
    "response_model",
    "timestamp",
    "uuid",
)


def _serialize_trace_entries(trace_entries: list[TraceEntry]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = [None] * len(trace_entries)  # type: ignore[list-item]
    history_offsets: dict[int, int] = {}

    for idx in range(len(trace_entries) - 1, -1, -1):
        predictor_obj, inputs, outputs = trace_entries[idx]

        entry: dict[str, Any] = {
            "predictor_type": type(predictor_obj).__name__,
            "inputs": to_serializable(inputs),
            "outputs": to_serializable(outputs),
        }
        predictor_name = getattr(predictor_obj, "_predictor_name", None)
        if predictor_name:
            entry["predictor_name"] = predictor_name

        history_list = getattr(predictor_obj, "history", None)
        if history_list:
            key = id(predictor_obj)
            history_offsets[key] = history_offsets.get(key, 0) + 1
            offset = history_offsets[key]
            if offset <= len(history_list):
                history_entry = history_list[-offset]
                history_payload = {
                    k: to_serializable(history_entry.get(k)) for k in _LM_HISTORY_KEYS if k in history_entry
                }
                if history_payload:
                    entry["lm_history"] = history_payload

        serialized[idx] = entry

    return serialized  # type: ignore[return-value]


@dataclass
class EvaluationEngine:
    """Coordinates program and candidate evaluation for APEX."""

    metric: MetricFn
    runtime: RuntimeTools
    tracker: ExperimentTracker
    min_metric: float
    max_metric: float
    success_threshold: float
    num_eval_runs: int
    rng: random.Random
    log: Callable[[str, Verbosity, LogLevel], None]
    is_enabled: Callable[[Verbosity], bool]

    def evaluate_train_examples(
        self,
        program: Module,
        trainset: Iterable[Example],
        iteration: int | None = None,
    ) -> tuple[list[TrainExampleRecord], list[TrainExampleRecord]]:
        """Run the training set through the program and bucket by success."""

        examples = list(trainset)
        if not examples:
            return [], []

        indexed_examples = list(enumerate(examples))

        def process(item: tuple[int, Example]) -> TrainExampleRecord:
            idx, example = item
            return self.run_train_example(program, example, iteration=iteration, example_idx=idx)

        records = self.runtime.parallel_execute(
            indexed_examples,
            process,
            description="APEX: evaluating trainset",
            level=Verbosity.NORMAL,
        )

        failures: list[TrainExampleRecord] = []
        successes: list[TrainExampleRecord] = []
        for record in records:
            if record is None:
                continue
            if record.is_success:
                successes.append(record)
            else:
                failures.append(record)

        return failures, successes

    def run_train_example(
        self,
        program: Module,
        example: Example,
        *,
        iteration: int | None = None,
        example_idx: int = 0,
    ) -> TrainExampleRecord:
        return self._run_single_example(program, example, iteration=iteration, example_idx=example_idx)

    def evaluate_candidates(
        self,
        *,
        baseline: Module,
        hypotheses: Sequence[HypothesisSpec],
        calset: Sequence[Example],
        iteration: int,
        cached_baseline: CandidateRecord | None = None,
        baseline_overrides: Mapping[int, Module] | None = None,
    ) -> list[CandidateRecord]:
        """Evaluate the baseline and each hypothesis on the calibration set."""

        candidates: list[CandidateRecord] = []

        candidate_specs: list[tuple[int, str, Module, Module, HypothesisSpec | None]] = []
        spec_counter = 0

        baseline_map = dict(baseline_overrides or {})

        if cached_baseline is not None:
            baseline_record = CandidateRecord(
                program=baseline,
                overall_score=cached_baseline.overall_score,
                per_example_scores=cached_baseline.per_example_scores,
                iteration=iteration,
                hypothesis=None,
            )
            candidates.append(baseline_record)
            self.log(
                f"APEX: iteration {iteration} baseline score={baseline_record.overall_score:.4f}",
                Verbosity.NORMAL,
                "info",
            )
        else:
            candidate_specs.append((spec_counter, "baseline", baseline, baseline.deepcopy(), None))
            spec_counter += 1

        for hypothesis in hypotheses:
            hypothesis_baseline = baseline_map.get(id(hypothesis), baseline)
            candidate_program = self.apply_hypothesis(hypothesis_baseline, hypothesis)
            candidate_specs.append(
                (spec_counter, "hypothesis", candidate_program, candidate_program.deepcopy(), hypothesis)
            )
            spec_counter += 1

        if not candidate_specs:
            return candidates

        indexed_calset = list(enumerate(calset))
        tasks = [
            (spec_id, label, eval_program, hypothesis, example_idx, example)
            for spec_id, label, _record_program, eval_program, hypothesis in candidate_specs
            for example_idx, example in indexed_calset
        ]

        def process(
            task: tuple[
                int,
                str,
                Module,
                HypothesisSpec | None,
                int,
                Example,
            ],
        ) -> tuple[int, int, float | None]:
            (
                spec_id,
                label,
                eval_program,
                hypothesis,
                example_idx,
                example,
            ) = task

            example_inputs = example.inputs().toDict()
            labels_dict = example.labels().toDict()

            prompts_map = {
                name: getattr(predictor.signature, "instructions", "")
                for name, predictor in eval_program.named_predictors()
            }

            span_inputs: dict[str, Any] = {
                "inputs": example_inputs,
                "labels": labels_dict,
                "prompts": prompts_map,
            }
            if hypothesis is not None:
                span_inputs["hypothesis"] = to_serializable(hypothesis)

            attributes = {
                "stage": label,
                "iteration": iteration,
                "example_index": example_idx,
            }

            with self.tracker.span(
                f"apex.{label}_example",
                inputs=span_inputs,
                attributes=attributes,
            ) as span:
                try:
                    per_runs: list[float] = []
                    run_details: list[dict[str, Any]] = []
                    for run_index in range(self.num_eval_runs):
                        with dspy.settings.context(trace=[]):
                            prediction = eval_program(**example_inputs)
                            trace_entries = list(dspy.settings.trace or [])

                        score, feedback = self._evaluate_metric(example, prediction, trace_entries)
                        clipped_score = max(self.min_metric, min(self.max_metric, score))
                        per_runs.append(clipped_score)

                        trace_payload = _serialize_trace_entries(trace_entries)

                        run_details.append(
                            {
                                "run_index": run_index,
                                "raw_score": score,
                                "clipped_score": clipped_score,
                                "feedback": feedback,
                                "prediction": to_serializable(prediction),
                                "trace": trace_payload,
                            }
                        )

                        if span and hasattr(span, "set_attribute"):
                            try:
                                span.set_attribute(f"run_{run_index}_score", clipped_score)
                            except Exception:  # pragma: no cover - defensive
                                pass

                    median_score = median(per_runs) if per_runs else self.min_metric
                    if span and hasattr(span, "set_outputs"):
                        try:
                            span.set_outputs({"median_score": median_score, "runs": run_details})
                        except Exception:  # pragma: no cover - defensive
                            pass
                    return spec_id, example_idx, median_score
                except Exception as exc:  # pragma: no cover - defensive
                    self.log(
                        f"APEX: Error evaluating example: {str(exc)[:200]}",
                        Verbosity.NORMAL,
                        "warning",
                    )
                    if span and hasattr(span, "set_attribute"):
                        try:
                            span.set_attribute("error", str(exc)[:200])
                        except Exception:
                            pass
                    return spec_id, example_idx, None

        results = self.runtime.parallel_execute(
            tasks,
            process,
            description="APEX: evaluating candidates",
            level=Verbosity.NORMAL,
        )

        scores_by_candidate: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for spec_id, example_idx, score in results:
            if score is None:
                # Use min_metric for failed examples to maintain alignment
                score = self.min_metric
            scores_by_candidate[spec_id].append((example_idx, score))

        num_calset_examples = len(calset)
        for spec_id, label, record_program, _, hypothesis in candidate_specs:
            scored_examples = sorted(scores_by_candidate.get(spec_id, []), key=lambda item: item[0])
            per_example_scores = [score for _, score in scored_examples]

            # Ensure we have exactly num_calset_examples scores for alignment
            if len(per_example_scores) != num_calset_examples:
                self.log(
                    f"APEX: Warning - {label} has {len(per_example_scores)} scores, expected {num_calset_examples}. "
                    f"Filling missing examples with min_metric={self.min_metric}",
                    Verbosity.NORMAL,
                    "warning",
                )
                # Fill missing indices with min_metric
                scored_dict = dict(scored_examples)
                per_example_scores = [scored_dict.get(idx, self.min_metric) for idx in range(num_calset_examples)]

            if not per_example_scores:
                self.log(
                    f"APEX: Warning - no valid scores obtained for {label}",
                    Verbosity.NORMAL,
                    "warning",
                )
                per_example_scores = [self.min_metric]

            overall_score = sum(per_example_scores) / len(per_example_scores)

            record = CandidateRecord(
                program=record_program,
                overall_score=overall_score,
                per_example_scores=per_example_scores,
                iteration=iteration,
                hypothesis=hypothesis,
            )

            if label == "baseline":
                candidates.insert(0, record)
                self.log(
                    f"APEX: iteration {iteration} baseline score={record.overall_score:.4f}",
                    Verbosity.NORMAL,
                    "info",
                )
            else:
                candidates.append(record)
                self.log(
                    f"APEX: iteration {iteration} hypothesis score={record.overall_score:.4f}",
                    Verbosity.NORMAL,
                    "info",
                )
                if self.is_enabled(Verbosity.DETAILED) and hypothesis is not None:
                    self.log(
                        f"APEX: hypothesis details → {hypothesis.model_dump()}",
                        Verbosity.DETAILED,
                        "info",
                    )

        return candidates

    def evaluate_candidate(
        self,
        *,
        program: Module,
        calset: Sequence[Example],
        iteration: int,
        hypothesis: HypothesisSpec | None,
    ) -> CandidateRecord:
        """Evaluate a single candidate program."""

        label = "baseline" if hypothesis is None else "hypothesis"
        program_to_eval = program.deepcopy()

        indexed_calset = list(enumerate(calset))

        def process(item: tuple[int, Example]) -> float:
            example_idx, example = item
            inputs_dict = example.inputs().toDict()
            labels_dict = example.labels().toDict()

            prompts_map = {
                name: getattr(predictor.signature, "instructions", "")
                for name, predictor in program_to_eval.named_predictors()
            }

            span_inputs: dict[str, Any] = {
                "inputs": inputs_dict,
                "labels": labels_dict,
                "prompts": prompts_map,
            }
            if hypothesis is not None:
                span_inputs["hypothesis"] = to_serializable(hypothesis)

            attributes = {
                "stage": label,
                "iteration": iteration,
                "example_index": example_idx,
            }

            with self.tracker.span(
                f"apex.{label}_example",
                inputs=span_inputs,
                attributes=attributes,
            ) as span:
                try:
                    per_runs: list[float] = []
                    run_details: list[dict[str, Any]] = []
                    for run_index in range(self.num_eval_runs):
                        with dspy.settings.context(trace=[]):
                            prediction = program_to_eval(**inputs_dict)
                            trace_entries = list(dspy.settings.trace or [])

                        score, feedback = self._evaluate_metric(example, prediction, trace_entries)
                        clipped_score = max(self.min_metric, min(self.max_metric, score))
                        per_runs.append(clipped_score)

                        trace_payload = _serialize_trace_entries(trace_entries)

                        run_details.append(
                            {
                                "run_index": run_index,
                                "raw_score": score,
                                "clipped_score": clipped_score,
                                "feedback": feedback,
                                "prediction": to_serializable(prediction),
                                "trace": trace_payload,
                            }
                        )

                        if span and hasattr(span, "set_attribute"):
                            try:
                                span.set_attribute(f"run_{run_index}_score", clipped_score)
                            except Exception:  # pragma: no cover - defensive
                                pass

                    median_score = median(per_runs) if per_runs else self.min_metric
                    if span and hasattr(span, "set_outputs"):
                        try:
                            span.set_outputs({"median_score": median_score, "runs": run_details})
                        except Exception:  # pragma: no cover - defensive
                            pass
                    return median_score
                except Exception as exc:  # pragma: no cover - defensive
                    self.log(
                        f"APEX: Error evaluating example: {str(exc)[:200]}",
                        Verbosity.NORMAL,
                        "warning",
                    )
                    if span and hasattr(span, "set_attribute"):
                        try:
                            span.set_attribute("error", str(exc)[:200])
                        except Exception:
                            pass
                    return self.min_metric

        results = self.runtime.parallel_execute(
            indexed_calset,
            process,
            description=f"APEX: evaluating {label}",
            level=Verbosity.NORMAL,
        )

        valid_scores = [score for score in results if score is not None]
        if not valid_scores:
            self.log(
                f"APEX: Warning - no valid scores obtained for {label}",
                Verbosity.NORMAL,
                "warning",
            )
            valid_scores = [self.min_metric]

        overall = sum(valid_scores) / len(valid_scores)

        return CandidateRecord(
            program=program,
            overall_score=overall,
            per_example_scores=valid_scores,
            iteration=iteration,
            hypothesis=hypothesis,
        )

    def select_best_candidate(self, candidates: Sequence[CandidateRecord]) -> CandidateRecord:
        """Return the highest-scoring candidate, using RNG to break ties."""

        best_score = max(candidate.overall_score for candidate in candidates)
        best_candidates = [c for c in candidates if c.overall_score == best_score]
        return self.rng.choice(best_candidates)

    def apply_hypothesis(self, baseline: Module, hypothesis: HypothesisSpec) -> Module:
        """Return a deep-copied program with the hypothesis' prompt changes applied."""

        candidate = baseline.deepcopy()
        name_to_predictor = dict(candidate.named_predictors())
        for predictor_name, changes in hypothesis.prompt_changes.items():
            if predictor_name not in name_to_predictor:
                msg = f"Hypothesis references unknown predictor '{predictor_name}'."
                raise ValueError(msg)
            predictor = name_to_predictor[predictor_name]
            predictor.signature.instructions = changes.new_prompt
        return candidate

    def _run_single_example(
        self,
        program: Module,
        example: Example,
        iteration: int | None = None,
        example_idx: int = 0,
    ) -> TrainExampleRecord:
        input_kwargs = example.inputs().toDict()

        prediction_obj: Prediction | None = None
        error_message: str | None = None
        raw_trace: list[TraceEntry] = []

        try:
            labels_dict = example.labels().toDict()
        except Exception:  # pragma: no cover - defensive
            labels_dict = {}

        prompts_map = {
            name: getattr(predictor.signature, "instructions", "") for name, predictor in program.named_predictors()
        }

        with self.tracker.span(
            "apex.train_example",
            inputs={"inputs": input_kwargs, "labels": labels_dict, "prompts": prompts_map},
            attributes={
                "stage": "train",
                "iteration": iteration if iteration is not None else -1,
                "example_index": example_idx,
            },
        ) as span:
            with dspy.settings.context(trace=[]):
                try:
                    prediction_obj = program(**input_kwargs)
                except Exception as exc:  # pragma: no cover - defensive
                    self.log(
                        f"APEX: Program execution failed on example: {str(exc)[:200]}",
                        Verbosity.DETAILED,
                        "warning",
                    )
                    error_message = f"execution_error: {exc}"
                finally:
                    raw_trace = list(dspy.settings.trace or [])

        execution_flow = extract_full_execution_flow_with_coverage(raw_trace, program)
        trace_serialized = _serialize_trace_entries(raw_trace)

        metric_score = self.min_metric
        metric_feedback: str | None = None
        try:
            if prediction_obj is not None:
                metric_score, metric_feedback = self._evaluate_metric(example, prediction_obj, raw_trace)
            elif error_message:
                self.log(
                    f"APEX: No prediction to evaluate due to error: {error_message[:100]}",
                    Verbosity.DETAILED,
                    "debug",
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.log(
                f"APEX: Metric evaluation failed: {str(exc)[:200]}",
                Verbosity.DETAILED,
                "warning",
            )
            metric_score = self.min_metric
            metric_feedback = f"metric_error: {exc}"

        metric_score = max(self.min_metric, min(self.max_metric, metric_score))
        is_success = metric_score >= self.success_threshold
        if span:
            if hasattr(span, "set_outputs"):
                try:
                    span.set_outputs(
                        {
                            "metric_score": metric_score,
                            "is_success": is_success,
                            "has_error": bool(error_message),
                            "prediction": to_serializable(prediction_obj),
                            "trace": trace_serialized,
                            "execution_flow": to_serializable(execution_flow),
                        }
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            if error_message and hasattr(span, "set_attribute"):
                try:
                    span.set_attribute("error", error_message[:200])
                except Exception:  # pragma: no cover - defensive
                    pass

        return TrainExampleRecord(
            example=example,
            prediction=prediction_obj,
            metric_score=metric_score,
            metric_feedback=metric_feedback,
            is_success=is_success,
            error=error_message,
            execution_flow=execution_flow,
        )

    def _evaluate_metric(
        self,
        example: Example,
        prediction: Prediction,
        trace_entries: list[TraceEntry],
    ) -> tuple[float, str | None]:
        result = self.metric(example, prediction, trace_entries)
        if isinstance(result, dict):
            if "score" not in result:
                msg = "Metric dict must contain a 'score' key."
                raise ValueError(msg)
            score = float(result["score"])
            feedback = result.get("feedback")
            return score, feedback
        if isinstance(result, Prediction):
            score = float(result["score"])
            feedback = result.get("feedback") if "feedback" in result else None
            return score, feedback
        if isinstance(result, bool):
            return (1.0 if result else 0.0), None
        if isinstance(result, int | float):
            return float(result), None
        msg = f"Unsupported metric return type: {type(result)}"
        raise TypeError(msg)


__all__ = ["EvaluationEngine"]
