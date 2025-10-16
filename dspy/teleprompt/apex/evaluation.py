"""Evaluation utilities for the APEX optimizer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import dspy
from dspy.primitives import Example, Module, Prediction

from .execution_flow import extract_execution_flow
from .models import CandidateRecord, HypothesisSpec, TrainExampleRecord
from .runtime import RuntimeTools
from .tracker import ExperimentTracker
from .tracking_session import TraceBatch
from .types import LogLevel, MetricFn, TraceEntry, Verbosity


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

        def process(example: Example) -> TrainExampleRecord:
            return self._run_single_example(program, example)

        records = self.runtime.parallel_execute(
            examples,
            process,
            description="APEX: evaluating trainset",
            level=Verbosity.NORMAL,
        )

        failures: list[TrainExampleRecord] = []
        successes: list[TrainExampleRecord] = []
        for record in records:
            if record.is_success:
                successes.append(record)
            else:
                failures.append(record)

        if iteration is not None and self.tracker.is_active():
            if failures:
                failure_batch = TraceBatch.from_iterable(
                    iteration=iteration,
                    stage="train_failures",
                    traces=[self._format_train_record(record) for record in failures],
                )
                self.tracker.log_trace_batch(failure_batch)
            if successes:
                success_batch = TraceBatch.from_iterable(
                    iteration=iteration,
                    stage="train_successes",
                    traces=[self._format_train_record(record) for record in successes],
                )
                self.tracker.log_trace_batch(success_batch)
        return failures, successes

    def evaluate_candidates(
        self,
        *,
        baseline: Module,
        hypotheses: Sequence[HypothesisSpec],
        calset: Sequence[Example],
        iteration: int,
        cached_baseline: CandidateRecord | None = None,
    ) -> list[CandidateRecord]:
        """Evaluate the baseline and each hypothesis on the calibration set."""

        candidates: list[CandidateRecord] = []

        if cached_baseline is not None:
            baseline_record = CandidateRecord(
                program=baseline,
                overall_score=cached_baseline.overall_score,
                per_example_scores=cached_baseline.per_example_scores,
                iteration=iteration,
                hypothesis=None,
            )
        else:
            baseline_record = self.evaluate_candidate(
                program=baseline.deepcopy(),
                calset=list(calset),
                iteration=iteration,
                hypothesis=None,
            )

        candidates.append(baseline_record)
        self.log(
            f"APEX: iteration {iteration} baseline score={baseline_record.overall_score:.4f}",
            Verbosity.NORMAL,
            "info",
        )

        for hypothesis in hypotheses:
            candidate_program = self.apply_hypothesis(baseline, hypothesis)
            record = self.evaluate_candidate(
                program=candidate_program,
                calset=list(calset),
                iteration=iteration,
                hypothesis=hypothesis,
            )
            candidates.append(record)
            self.log(
                f"APEX: iteration {iteration} hypothesis score={record.overall_score:.4f}",
                Verbosity.NORMAL,
                "info",
            )
            if self.is_enabled(Verbosity.HIGH):
                self.log(
                    f"APEX: hypothesis details → {hypothesis.model_dump()}",
                    Verbosity.HIGH,
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

        def process(example: Example) -> tuple[float | None, Mapping[str, Any] | None]:
            inputs_dict = example.inputs().toDict()
            labels_dict = example.labels().toDict()
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

                    execution_flow = extract_execution_flow(trace_entries, program_to_eval)
                    run_details.append(
                        {
                            "run_index": run_index,
                            "raw_score": score,
                            "clipped_score": clipped_score,
                            "feedback": feedback,
                            "prediction": prediction.toDict() if isinstance(prediction, Prediction) else prediction,
                            "execution_flow": [entry.model_dump() for entry in execution_flow],
                        }
                    )

                median_score = median(per_runs) if per_runs else None
                payload: dict[str, Any] = {
                    "inputs": inputs_dict,
                    "labels": labels_dict,
                    "runs": run_details,
                }
                if median_score is not None:
                    payload["median_score"] = median_score
                if hypothesis is not None:
                    payload["hypothesis"] = {
                        "strategy": hypothesis.strategy,
                        "fixable_root_causes": hypothesis.fixable_root_causes,
                        "impact_score": hypothesis.impact_score,
                        "generalizability_score": hypothesis.generalizability_score,
                    }
                return median_score, payload
            except Exception as exc:  # pragma: no cover - defensive
                self.log(
                    f"APEX: Error evaluating example: {str(exc)[:200]}",
                    Verbosity.NORMAL,
                    "warning",
                )
                payload: dict[str, Any] = {
                    "inputs": inputs_dict,
                    "labels": labels_dict,
                    "error": str(exc),
                }
                if hypothesis is not None:
                    payload["hypothesis"] = {
                        "strategy": hypothesis.strategy,
                        "fixable_root_causes": hypothesis.fixable_root_causes,
                    }
                return self.min_metric, payload

        results = self.runtime.parallel_execute(
            list(calset),
            process,
            description=f"APEX: evaluating {label}",
            level=Verbosity.NORMAL,
        )

        valid_scores: list[float] = []
        trace_payloads: list[Mapping[str, Any]] = []
        for score, payload in results:
            if score is not None:
                valid_scores.append(score)
            if payload:
                trace_payloads.append(payload)

        if not valid_scores:
            self.log(
                f"APEX: Warning - no valid scores obtained for {label}",
                Verbosity.NORMAL,
                "warning",
            )
            valid_scores = [self.min_metric]

        overall = sum(valid_scores) / len(valid_scores)

        if trace_payloads and self.tracker.is_active():
            stage_name = f"{label}_evaluation"
            batch = TraceBatch.from_iterable(iteration=iteration, stage=stage_name, traces=trace_payloads)
            self.tracker.log_trace_batch(batch)

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

    def _run_single_example(self, program: Module, example: Example) -> TrainExampleRecord:
        input_kwargs = example.inputs().toDict()

        prediction_obj: Prediction | None = None
        error_message: str | None = None

        with dspy.settings.context(trace=[]):
            try:
                prediction_obj = program(**input_kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                self.log(
                    f"APEX: Program execution failed on example: {str(exc)[:200]}",
                    Verbosity.HIGH,
                    "warning",
                )
                error_message = f"execution_error: {exc}"

        raw_trace = list(dspy.settings.trace or [])
        execution_flow = extract_execution_flow(raw_trace, program)

        metric_score = self.min_metric
        metric_feedback: str | None = None
        try:
            if prediction_obj is not None:
                metric_score, metric_feedback = self._evaluate_metric(example, prediction_obj, raw_trace)
            elif error_message:
                self.log(
                    f"APEX: No prediction to evaluate due to error: {error_message[:100]}",
                    Verbosity.HIGH,
                    "debug",
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.log(
                f"APEX: Metric evaluation failed: {str(exc)[:200]}",
                Verbosity.HIGH,
                "warning",
            )
            metric_score = self.min_metric
            metric_feedback = f"metric_error: {exc}"

        metric_score = max(self.min_metric, min(self.max_metric, metric_score))
        is_success = metric_score >= self.success_threshold

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

    def _format_train_record(self, record: TrainExampleRecord) -> Mapping[str, Any]:
        """Convert a training evaluation record into a trace artifact payload."""

        prediction_dict = record.prediction.toDict() if isinstance(record.prediction, Prediction) else None
        return {
            "inputs": record.example.inputs().toDict(),
            "labels": record.example.labels().toDict(),
            "metric_score": record.metric_score,
            "metric_feedback": record.metric_feedback,
            "is_success": record.is_success,
            "error": record.error,
            "prediction": prediction_dict,
            "execution_flow": [entry.model_dump() for entry in record.execution_flow],
        }


__all__ = ["EvaluationEngine"]
