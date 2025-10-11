from __future__ import annotations

import json
import logging
import os
import random
from enum import Enum
from statistics import median
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from tqdm.auto import tqdm

import dspy
from dspy.adapters import JSONAdapter
from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.signatures import InputField, OutputField
from dspy.teleprompt.prompts import render_failure_prompt, render_hypothesis_prompt, render_success_prompt
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

JsonObject: TypeAlias = dict[str, JsonValue]
TraceEntry: TypeAlias = tuple[Any, Mapping[str, Any], Prediction]

ModelT = TypeVar("ModelT", bound=BaseModel)
ItemT = TypeVar("ItemT")
MetricFn = Callable[[Example, Prediction, list[TraceEntry]], Any]
SamplerFn = Callable[[list[Example], int], list[Example]]


class Verbosity(str, Enum):
    NONE = "none"
    NORMAL = "normal"
    HIGH = "high"

    @classmethod
    def parse(cls, value: str | Verbosity | None) -> Verbosity:
        if value is None:
            return cls.NORMAL
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported verbosity level '{value}'. Use one of: none, normal, high.")


def _verbosity_rank(level: Verbosity) -> int:
    return {
        Verbosity.NONE: 0,
        Verbosity.NORMAL: 1,
        Verbosity.HIGH: 2,
    }[level]


class JsonResponseSignature(dspy.Signature):
    analysis_prompt: str = InputField(desc="Fully formatted prompt to send to the language model.")
    json_response: JsonValue = OutputField(desc="Strict JSON object or array encoded as text.")


class FailureAnalysis(BaseModel):
    root_cause: str
    involved_predictors: list[str] = Field(default_factory=list)
    context: str
    category: str
    key_details: str


class SuccessAnalysis(BaseModel):
    success_pattern: str
    contributing_predictors: list[str] = Field(default_factory=list)
    context: str
    category: str
    key_details: str


class FailureExamplePayload(BaseModel):
    input: JsonValue
    expected: JsonValue
    prediction: JsonValue | None
    metric_score: float
    metric_feedback: str | None = None
    trace: list[PredictorTraceRecord]
    error: str | None = None


class SuccessExamplePayload(BaseModel):
    input: JsonValue
    expected: JsonValue
    prediction: JsonValue | None
    metric_score: float
    metric_feedback: str | None = None
    trace: list[PredictorTraceRecord]


class FailurePromptPayload(BaseModel):
    program_structure: str
    predictor_flow: str
    predictor_prompts: dict[str, str]
    failed_example: FailureExamplePayload
    success_threshold: float


class SuccessPromptPayload(BaseModel):
    program_structure: str
    predictor_flow: str
    predictor_prompts: dict[str, str]
    successful_example: SuccessExamplePayload
    success_threshold: float


class HypothesisPromptPayload(BaseModel):
    error_summaries: list[JsonObject]
    success_summaries: list[JsonObject]
    predictor_prompts: dict[str, str]
    program_structure: str
    predictor_flow: str
    num_hypotheses: int


class PredictorTraceRecord(BaseModel):
    name: str
    prompt: str
    inputs: JsonValue
    outputs: JsonValue

    model_config = ConfigDict(frozen=True)


class TrainExampleRecord(BaseModel):
    example: Example
    inputs: JsonValue
    expected: JsonValue
    prediction: JsonValue | None
    metric_score: float
    metric_feedback: str | None = None
    is_success: bool
    traces: list[PredictorTraceRecord] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def failure_payload(self, snapshot: ProgramSnapshot, success_threshold: float) -> FailurePromptPayload:
        return FailurePromptPayload(
            program_structure=snapshot.structure,
            predictor_flow=snapshot.flow_description,
            predictor_prompts=snapshot.prompts,
            failed_example=FailureExamplePayload(
                input=self.inputs,
                expected=self.expected,
                prediction=self.prediction,
                metric_score=self.metric_score,
                metric_feedback=self.metric_feedback,
                trace=self.traces,
                error=self.error,
            ),
            success_threshold=success_threshold,
        )

    def success_payload(self, snapshot: ProgramSnapshot, success_threshold: float) -> SuccessPromptPayload:
        return SuccessPromptPayload(
            program_structure=snapshot.structure,
            predictor_flow=snapshot.flow_description,
            predictor_prompts=snapshot.prompts,
            successful_example=SuccessExamplePayload(
                input=self.inputs,
                expected=self.expected,
                prediction=self.prediction,
                metric_score=self.metric_score,
                metric_feedback=self.metric_feedback,
                trace=self.traces,
            ),
            success_threshold=success_threshold,
        )


class HypothesisPromptChange(BaseModel):
    new_prompt: str
    rationale: str | None = None
    change_magnitude: str | None = None


class HypothesisSpec(BaseModel):
    observation: str
    fixable_root_causes: list[str] = Field(default_factory=list)
    non_fixable_root_causes: list[str] = Field(default_factory=list)
    strategy: str
    expected_impact: str
    prompt_changes: dict[str, HypothesisPromptChange] = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    program: Module
    overall_score: float
    per_example_scores: list[float]
    iteration: int
    hypothesis: HypothesisSpec | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProgramSnapshot(BaseModel):
    structure: str
    flow_description: str
    prompts: dict[str, str]
    predictor_name_by_id: dict[int, str]


class ApexIterationLog(BaseModel):
    iteration: int
    sampled_train_size: int
    num_failures: int
    num_successes: int
    hypotheses: list[HypothesisSpec]
    candidates: list[CandidateRecord]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ApexOptimizationResult(BaseModel):
    best_candidate: CandidateRecord
    all_candidates: list[CandidateRecord]
    iterations: list[ApexIterationLog]
    stopped_after: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class APEX(Teleprompter):
    """APEX teleprompter implementing systematic prompt optimization."""

    def __init__(
        self,
        *,
        metric: MetricFn,
        analysis_llm: LM | None = None,
        analysis_lm: LM | None = None,
        max_iterations: int,
        hypothesis_llm: LM | None = None,
        analysis_adapter: JSONAdapter | None = None,
        hypothesis_adapter: JSONAdapter | None = None,
        verbosity: Verbosity | str | None = None,
        num_threads: int | None = None,
        num_hypotheses: int = 1,
        num_eval_runs: int = 1,
        train_sample: None | int | SamplerFn = None,
        success_threshold: float | None = None,
        min_metric: float = 0.0,
        max_metric: float = 1.0,
        convergence_patience: int = 3,
        seed: int | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0.")
        if num_hypotheses < 0:
            raise ValueError("num_hypotheses must be >= 0.")
        if num_eval_runs <= 0:
            raise ValueError("num_eval_runs must be > 0.")
        if convergence_patience <= 0:
            raise ValueError("convergence_patience must be > 0.")
        if min_metric > max_metric:
            raise ValueError("min_metric cannot exceed max_metric.")

        analysis_model = analysis_llm or analysis_lm
        if analysis_model is None:
            raise ValueError("analysis_llm must be provided.")
        if analysis_llm is not None and analysis_lm is not None and analysis_llm is not analysis_lm:
            raise ValueError("Provide only one of analysis_llm or analysis_lm.")

        hypothesis_model = hypothesis_llm or analysis_model

        self.metric = metric
        self.analysis_lm = analysis_model
        self.hypothesis_lm = hypothesis_model
        default_threads = num_threads if num_threads is not None else (os.cpu_count() or 0)
        if default_threads is None or default_threads <= 0:
            default_threads = dspy.settings.num_threads or 1
        self.num_threads = max(1, int(default_threads))
        self.max_iterations = max_iterations
        self.num_hypotheses = num_hypotheses
        self.num_eval_runs = num_eval_runs
        self.train_sample = train_sample
        self.verbosity = Verbosity.parse(verbosity)
        self.min_metric = float(min_metric)
        self.max_metric = float(max_metric)
        self.success_threshold = float(success_threshold) if success_threshold is not None else float(max_metric)
        self.convergence_patience = convergence_patience
        self.seed = seed if seed is not None else random.randint(1, 1_000_000)
        self._rng = random.Random(self.seed)
        self._analysis_predictor = dspy.Predict(JsonResponseSignature, lm=analysis_model)
        self._hypothesis_predictor = dspy.Predict(JsonResponseSignature, lm=hypothesis_model)
        self._analysis_predictor.lm = analysis_model
        self._hypothesis_predictor.lm = hypothesis_model
        self.analysis_adapter = analysis_adapter or JSONAdapter()
        self.hypothesis_adapter = hypothesis_adapter or JSONAdapter()

    # --- Logging & progress helpers ---------------------------------------------

    def _is_enabled(self, level: Verbosity) -> bool:
        return _verbosity_rank(self.verbosity) >= _verbosity_rank(level)

    def _log(self, message: str, level: Verbosity = Verbosity.NORMAL) -> None:
        if self._is_enabled(level):
            logger.info(message)

    def _parallel_execute(
        self,
        items: Iterable[ItemT],
        func: Callable[[ItemT], Any],
        *,
        description: str,
        level: Verbosity,
    ) -> list[Any]:
        items_list = list(items)
        if not items_list:
            return []
        if self.num_threads <= 1 or len(items_list) <= 1:
            results: list[Any] = []
            for item in self._iter_with_progress(items_list, description=description, level=level):
                results.append(func(item))
            return results
        executor = ParallelExecutor(
            num_threads=self.num_threads,
            disable_progress_bar=not self._is_enabled(level),
            max_errors=max(len(items_list), 1),
            provide_traceback=self._is_enabled(Verbosity.HIGH),
        )
        return executor.execute(func, items_list)

    def _iter_with_progress(
        self,
        iterable: Iterable[ItemT],
        *,
        description: str,
        level: Verbosity,
        total: int | None = None,
    ) -> Iterator[ItemT]:
        if not self._is_enabled(level):
            yield from iterable
            return
        progress_total = total
        if progress_total is None and hasattr(iterable, "__len__"):
            progress_total = len(iterable)  # type: ignore[arg-type]
        with tqdm(iterable, total=progress_total, desc=description, leave=False) as progress:
            yield from progress

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
    ) -> Module:
        if teacher is not None:
            raise ValueError("APEX does not support teacher programs.")
        if not trainset:
            raise ValueError("trainset must be non-empty.")
        if not valset:
            raise ValueError("calibration set (valset) must be provided and non-empty.")

        current_program = student.deepcopy()
        assert not getattr(current_program, "_compiled", False), "Student must be uncompiled."

        all_candidates: list[CandidateRecord] = []
        iteration_logs: list[ApexIterationLog] = []

        self._log(f"APEX: running with num_threads={self.num_threads}", Verbosity.NORMAL)

        no_improvement_count = 0
        best_candidate: CandidateRecord | None = None
        stop_reason = ""

        for iteration in range(1, self.max_iterations + 1):
            sampled_train = self._sample_trainset(trainset, iteration)
            self._log(
                f"APEX: iteration {iteration} started (train sample={len(sampled_train)}, val size={len(valset)})",
                Verbosity.NORMAL,
            )
            baseline_for_analysis = current_program.deepcopy()
            snapshot = self._snapshot_program(baseline_for_analysis)

            failures, successes = self._evaluate_train_examples(baseline_for_analysis, sampled_train, snapshot)
            failure_summaries = cast(
                list[FailureAnalysis],
                self._analyze_examples(
                    failures,
                    snapshot,
                    success_threshold=self.success_threshold,
                    mode="failure",
                ),
            )
            success_summaries = cast(
                list[SuccessAnalysis],
                self._analyze_successes(
                    successes,
                    snapshot,
                    failure_count=len(failure_summaries),
                    success_threshold=self.success_threshold,
                ),
            )
            if self._is_enabled(Verbosity.HIGH):
                self._log(
                    f"APEX: iteration {iteration} analyzed {len(failure_summaries)} failure(s) and {len(success_summaries)} success(es)",
                    Verbosity.HIGH,
                )

            hypotheses = self._generate_hypotheses(
                failure_summaries=failure_summaries,
                success_summaries=success_summaries,
                snapshot=snapshot,
            )
            self._log(
                f"APEX: iteration {iteration} produced {len(hypotheses)} hypothesis(es)",
                Verbosity.NORMAL,
            )

            candidates = self._evaluate_candidates(
                baseline=current_program,
                hypotheses=hypotheses,
                calset=valset,
                iteration=iteration,
            )

            best_candidate_for_iteration = self._select_best_candidate(candidates)

            all_candidates.extend(candidates)
            iteration_logs.append(
                ApexIterationLog(
                    iteration=iteration,
                    sampled_train_size=len(sampled_train),
                    num_failures=len(failure_summaries),
                    num_successes=len(success_summaries),
                    hypotheses=hypotheses,
                    candidates=candidates,
                )
            )

            if best_candidate is None or best_candidate_for_iteration.overall_score > best_candidate.overall_score:
                best_candidate = best_candidate_for_iteration

            baseline_candidate = candidates[0]
            self._log(
                f"APEX: iteration {iteration} best score={best_candidate_for_iteration.overall_score:.4f}",
                Verbosity.NORMAL,
            )
            if best_candidate_for_iteration is baseline_candidate:
                no_improvement_count += 1
                if no_improvement_count >= self.convergence_patience:
                    stop_reason = "patience"
                    break
            else:
                no_improvement_count = 0
                current_program = best_candidate_for_iteration.program

            if iteration == self.max_iterations:
                stop_reason = "max_iterations"

        assert best_candidate is not None, "APEX failed to evaluate any candidates."
        optimized_program = best_candidate.program
        optimized_program._compiled = True
        optimized_program.apex_result = ApexOptimizationResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iterations=iteration_logs,
            stopped_after=stop_reason,
        )
        return optimized_program

    # --- Train evaluation helpers -------------------------------------------------

    def _sample_trainset(self, trainset: Sequence[Example], iteration: int) -> list[Example]:
        if self.train_sample is None:
            sampled = list(trainset)
            self._rng.shuffle(sampled)
            return sampled

        if isinstance(self.train_sample, int):
            k = min(self.train_sample, len(trainset))
            return self._rng.sample(list(trainset), k=k)

        sampled = self.train_sample(list(trainset), iteration)
        if not isinstance(sampled, list):
            raise TypeError("Custom train_sample callable must return a list of Examples.")
        return sampled

    def _evaluate_train_examples(
        self,
        program: Module,
        trainset: Iterable[Example],
        snapshot: ProgramSnapshot,
    ) -> tuple[list[TrainExampleRecord], list[TrainExampleRecord]]:
        failure_records: list[TrainExampleRecord] = []
        success_records: list[TrainExampleRecord] = []

        examples = list(trainset)

        def process(example: Example) -> TrainExampleRecord:
            return self._run_single_example(program, example, snapshot.predictor_name_by_id)

        records = self._parallel_execute(
            examples,
            process,
            description="APEX: evaluating trainset",
            level=Verbosity.NORMAL,
        )

        for record in records:
            if record.is_success:
                success_records.append(record)
            else:
                failure_records.append(record)
        return failure_records, success_records

    def _run_single_example(
        self,
        program: Module,
        example: Example,
        name_lookup: dict[int, str],
    ) -> TrainExampleRecord:
        inputs_ex = example.inputs()
        labels = example.labels()
        input_kwargs = inputs_ex.toDict()
        expected_kwargs = labels.toDict()

        prediction_obj: Prediction | None = None
        trace_entries: list[TraceEntry] = []
        error_message: str | None = None

        with dspy.settings.context(trace=[]):
            try:
                prediction_obj = program(**input_kwargs)
                if prediction_obj is not None:
                    prediction_obj = self._sanitize_prediction(prediction_obj)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("APEX failed to execute program on example.", exc_info=exc)
                error_message = f"execution_error: {exc}"
            finally:
                trace_entries = cast(list[TraceEntry], list(dspy.settings.trace or []))

        metric_score = self.min_metric
        metric_feedback: str | None = None
        try:
            if prediction_obj is not None:
                metric_score, metric_feedback = self._evaluate_metric(example, prediction_obj, trace_entries)
            else:
                metric_score = self.min_metric
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("APEX metric raised an exception; defaulting to min_metric.", exc_info=exc)
            metric_score = self.min_metric
            metric_feedback = f"metric_error: {exc}"

        metric_score = max(self.min_metric, min(self.max_metric, metric_score))
        is_success = metric_score >= self.success_threshold
        prediction_json = self._serialize_prediction(prediction_obj) if prediction_obj is not None else None
        traces = self._serialize_traces(trace_entries, name_lookup)

        return TrainExampleRecord(
            example=example,
            inputs=self._serialize_mapping(input_kwargs),
            expected=self._serialize_mapping(expected_kwargs),
            prediction=prediction_json,
            metric_score=metric_score,
            metric_feedback=metric_feedback,
            is_success=is_success,
            traces=traces,
            error=error_message,
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
                raise ValueError("Metric dict must contain a 'score' key.")
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
        raise TypeError(f"Unsupported metric return type: {type(result)}")

    # --- Analysis helpers ---------------------------------------------------------

    def _analyze_examples(
        self,
        records: list[TrainExampleRecord],
        snapshot: ProgramSnapshot,
        success_threshold: float,
        mode: str,
    ) -> list[BaseModel]:
        if not records:
            return []

        prompt_builder = self._build_failure_prompt if mode == "failure" else self._build_success_prompt

        target_model = FailureAnalysis if mode == "failure" else SuccessAnalysis

        def process(record: TrainExampleRecord) -> BaseModel:
            payload = (
                record.failure_payload(snapshot, success_threshold)
                if mode == "failure"
                else record.success_payload(snapshot, success_threshold)
            )
            prompt = prompt_builder(payload)
            with dspy.settings.context(adapter=self.analysis_adapter):
                prediction = self._analysis_predictor(analysis_prompt=prompt)
            json_payload = self._normalize_json_response(prediction.json_response)
            return target_model.model_validate(json_payload)

        analyses = self._parallel_execute(
            records,
            process,
            description=f"APEX: analyzing {mode}s",
            level=Verbosity.HIGH,
        )

        if self._is_enabled(Verbosity.HIGH):
            for index, analysis in enumerate(analyses, start=1):
                if isinstance(analysis, FailureAnalysis):
                    self._log(
                        f"APEX: failure analysis #{index} ({analysis.category}) → {analysis.root_cause}",
                        Verbosity.HIGH,
                    )
                elif isinstance(analysis, SuccessAnalysis):
                    self._log(
                        f"APEX: success analysis #{index} ({analysis.category}) → {analysis.success_pattern}",
                        Verbosity.HIGH,
                    )
        return analyses

    def _analyze_successes(
        self,
        success_records: list[TrainExampleRecord],
        snapshot: ProgramSnapshot,
        failure_count: int,
        success_threshold: float,
    ) -> list[SuccessAnalysis]:
        if not success_records or failure_count == 0:
            return []
        if len(success_records) > failure_count:
            success_records = self._rng.sample(success_records, k=failure_count)
        return cast(
            list[SuccessAnalysis],
            self._analyze_examples(
                success_records,
                snapshot,
                success_threshold=success_threshold,
                mode="success",
            ),
        )

    def _generate_hypotheses(
        self,
        *,
        failure_summaries: list[FailureAnalysis],
        success_summaries: list[SuccessAnalysis],
        snapshot: ProgramSnapshot,
    ) -> list[HypothesisSpec]:
        if not failure_summaries or self.num_hypotheses == 0:
            return []

        shuffled_failures = list(failure_summaries)
        self._rng.shuffle(shuffled_failures)

        payload = HypothesisPromptPayload(
            error_summaries=[analysis.model_dump() for analysis in shuffled_failures],
            success_summaries=[analysis.model_dump() for analysis in success_summaries],
            predictor_prompts=snapshot.prompts,
            program_structure=snapshot.structure,
            predictor_flow=snapshot.flow_description,
            num_hypotheses=self.num_hypotheses,
        )
        prompt = self._build_hypothesis_prompt(payload)
        with dspy.settings.context(adapter=self.hypothesis_adapter):
            prediction = self._hypothesis_predictor(analysis_prompt=prompt)
        data = self._normalize_json_response(prediction.json_response)
        if not isinstance(data, list):
            raise ValueError("APEX expected a JSON array of hypotheses.")
        specs: list[HypothesisSpec] = []
        for item in data:
            try:
                specs.append(HypothesisSpec.model_validate(item))
            except ValidationError as exc:  # pragma: no cover - debug aid
                raise ValueError("APEX received an invalid hypothesis JSON payload.") from exc
        if self._is_enabled(Verbosity.HIGH):
            for idx, spec in enumerate(specs, start=1):
                self._log(
                    f"APEX: hypothesis #{idx} ({spec.strategy}) targeting {', '.join(spec.fixable_root_causes) or 'no fixable causes'}",
                    Verbosity.HIGH,
                )
        return specs[: self.num_hypotheses]

    # --- Candidate evaluation -----------------------------------------------------

    def _evaluate_candidates(
        self,
        *,
        baseline: Module,
        hypotheses: list[HypothesisSpec],
        calset: list[Example],
        iteration: int,
    ) -> list[CandidateRecord]:
        candidates: list[CandidateRecord] = []

        baseline_clone = baseline.deepcopy()
        baseline_record = self._evaluate_candidate(
            program=baseline_clone,
            calset=calset,
            iteration=iteration,
            hypothesis=None,
        )
        candidates.append(baseline_record)
        self._log(
            f"APEX: iteration {iteration} baseline score={baseline_record.overall_score:.4f}",
            Verbosity.NORMAL,
        )

        for hypothesis in hypotheses:
            candidate_program = self._apply_hypothesis(baseline, hypothesis)
            record = self._evaluate_candidate(
                program=candidate_program,
                calset=calset,
                iteration=iteration,
                hypothesis=hypothesis,
            )
            candidates.append(record)
            self._log(
                f"APEX: iteration {iteration} hypothesis score={record.overall_score:.4f}",
                Verbosity.NORMAL,
            )
            if self._is_enabled(Verbosity.HIGH):
                self._log(
                    f"APEX: hypothesis details → {hypothesis.model_dump()}",
                    Verbosity.HIGH,
                )
        return candidates

    def _evaluate_candidate(
        self,
        *,
        program: Module,
        calset: list[Example],
        iteration: int,
        hypothesis: HypothesisSpec | None,
    ) -> CandidateRecord:
        label = "baseline" if hypothesis is None else "hypothesis"
        cal_examples = list(calset)

        def process(example: Example) -> float:
            per_runs: list[float] = []
            for _ in range(self.num_eval_runs):
                with dspy.settings.context(trace=[]):
                    prediction = program(**example.inputs().toDict())
                    prediction = self._sanitize_prediction(prediction)
                    trace_entries = list(dspy.settings.trace or [])
                score, _ = self._evaluate_metric(example, prediction, trace_entries)
                score = max(self.min_metric, min(self.max_metric, score))
                per_runs.append(score)
            return median(per_runs)

        scores = self._parallel_execute(
            cal_examples,
            process,
            description=f"APEX: evaluating {label}",
            level=Verbosity.NORMAL,
        )

        overall = sum(scores) / len(scores)
        return CandidateRecord(
            program=program,
            overall_score=overall,
            per_example_scores=scores,
            iteration=iteration,
            hypothesis=hypothesis,
        )

    def _apply_hypothesis(self, baseline: Module, hypothesis: HypothesisSpec) -> Module:
        candidate = baseline.deepcopy()
        name_to_predictor = dict(candidate.named_predictors())
        for predictor_name, change in hypothesis.prompt_changes.items():
            if predictor_name not in name_to_predictor:
                raise ValueError(f"Hypothesis references unknown predictor '{predictor_name}'.")
            predictor = name_to_predictor[predictor_name]
            predictor.signature.instructions = change.new_prompt
        return candidate

    def _select_best_candidate(self, candidates: list[CandidateRecord]) -> CandidateRecord:
        best_score = max(c.overall_score for c in candidates)
        best_candidates = [c for c in candidates if c.overall_score == best_score]
        return self._rng.choice(best_candidates)

    # --- Snapshot & serialization -------------------------------------------------

    def _snapshot_program(self, program: Module) -> ProgramSnapshot:
        prompts: dict[str, str] = {}
        flow = []
        lookup: dict[int, str] = {}

        for name, predictor in program.named_predictors():
            flow.append(name)
            prompts[name] = getattr(predictor.signature, "instructions", "")
            lookup[id(predictor)] = name

        structure = repr(program)
        flow_description = " -> ".join(flow) if flow else "No predictors"
        return ProgramSnapshot(
            structure=structure,
            flow_description=flow_description,
            prompts=prompts,
            predictor_name_by_id=lookup,
        )

    def _serialize_traces(
        self,
        trace_entries: list[TraceEntry],
        name_lookup: dict[int, str],
    ) -> list[PredictorTraceRecord]:
        records: list[PredictorTraceRecord] = []
        for predictor, inputs, prediction in trace_entries:
            name = name_lookup.get(id(predictor), predictor.__class__.__name__)
            prompt = getattr(getattr(predictor, "signature", None), "instructions", "")
            records.append(
                PredictorTraceRecord(
                    name=name,
                    prompt=prompt,
                    inputs=self._serialize_mapping(inputs),
                    outputs=self._serialize_prediction(prediction),
                )
            )
        return records

    def _serialize_mapping(self, mapping: Mapping[str, Any]) -> JsonObject:
        return {str(key): self._serialize_value(value) for key, value in mapping.items()}

    def _serialize_prediction(self, prediction: Prediction) -> JsonObject:
        return self._serialize_mapping(prediction.toDict())

    def _serialize_value(self, value: Any) -> JsonValue:
        if hasattr(value, "message"):
            message_value = value.message
            if message_value is not None:
                return self._serialize_value(message_value)
        if hasattr(value, "choices"):
            choices_value = value.choices
            if choices_value is not None:
                return self._serialize_value(list(choices_value))
        if hasattr(value, "content") and not isinstance(value, str | bytes):
            content_value = value.content
            if isinstance(content_value, list):
                joined = "".join(
                    part
                    if isinstance(part, str)
                    else str(part.get("text", ""))
                    for part in content_value
                )
                return joined
            return self._serialize_value(content_value)
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, Mapping):
            return {str(key): self._serialize_value(val) for key, val in value.items()}
        if hasattr(value, "model_dump"):
            return self._serialize_value(value.model_dump())
        if hasattr(value, "toDict"):
            return self._serialize_value(value.toDict())
        if hasattr(value, "dict"):
            return self._serialize_value(value.dict())
        return str(value)

    def _sanitize_prediction(self, prediction: Prediction) -> Prediction:
        sanitized: dict[str, JsonValue] = {}
        for key in prediction.keys():
            sanitized[key] = self._serialize_value(prediction[key])
        return Prediction(**sanitized)

    # --- Prompt builders & invocation helpers ------------------------------------

    def _build_failure_prompt(self, payload: FailurePromptPayload) -> str:
        return render_failure_prompt(payload.model_dump())

    def _build_success_prompt(self, payload: SuccessPromptPayload) -> str:
        return render_success_prompt(payload.model_dump())

    def _build_hypothesis_prompt(self, payload: HypothesisPromptPayload) -> str:
        return render_hypothesis_prompt(payload.model_dump())

    def _normalize_json_response(self, response: JsonValue | str) -> JsonValue:
        if isinstance(response, dict | list):
            return cast(JsonValue, response)
        if isinstance(response, str):
            candidate = response.strip()
            if candidate.startswith("```"):
                candidate = candidate.strip("`")
                newline = candidate.find("\n")
                if newline != -1:
                    candidate = candidate[newline + 1 :]
            try:
                loaded = json.loads(candidate)
            except json.JSONDecodeError as exc:  # pragma: no cover - debug aid
                raise ValueError("APEX expected JSON output from language model.") from exc
            if not isinstance(loaded, dict | list):
                raise TypeError("APEX expected a JSON object or array from language model.")
            return cast(JsonValue, loaded)
        raise TypeError("APEX expected a JSON object or array from language model.")
