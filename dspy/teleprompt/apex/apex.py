from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Sequence

import dspy
from dspy.adapters import Adapter, JSONAdapter
from dspy.clients.lm import LM
from dspy.primitives import Example, Module
from dspy.teleprompt.teleprompt import Teleprompter

from . import tracking_utils
from .analysis import (
    _log_analysis_results,
    analyze_failures_and_successes,
    generate_hypotheses,
    generate_merge_hypotheses,
)
from .analysis import (
    analyze_record as _analyze_single_record,
)
from .candidate_selection import (
    CandidateSelectionStrategy,
    SelectionResult,
    candidates_are_equivalent,
    draw_weighted_candidate,
    select_baseline_candidate,
)
from .checkpoint_manager import CheckpointManager
from .evaluation import EvaluationEngine
from .execution_flow import (
    extract_execution_flow,
    format_execution_flow_as_graph,
    format_execution_flow_with_details,
)
from .lifecycle import finalize_optimization, initialize_new_state, resume_from_checkpoint
from .models import (
    ApexCheckpoint,
    ApexIterationLog,
    CandidateRecord,
    CheckpointConfig,
    ExecutionFlowEntry,
    HypothesisSpec,
)
from .runtime import RuntimeTools
from .sampling import sample_trainset
from .snapshot import snapshot_program
from .state import OptimizationState
from .tracker import ExperimentTracker
from .types import LogLevel, MetricFn, SamplerFn, TraceEntry, Verbosity

logger = logging.getLogger(__name__)


# Backward-compatible alias so downstream users (and tests) can monkeypatch
# ``analyze_record`` directly from this module.
analyze_record = _analyze_single_record


class APEX(Teleprompter):
    """Analysis-based Prompt Engineering eXpert (APEX) optimizer.

    APEX orchestrates a map-reduce style optimization loop that mirrors how
    human prompt engineers diagnose and remediate model errors.  Each iteration
    samples a subset of the training data, records full DSPy execution traces,
    and performs parallel failure/success analyses with an "analysis" language
    model.  A separate "hypothesis" model synthesizes those summaries into
    candidate prompt revisions which are then evaluated on a calibration set.
    The highest-scoring candidate becomes the new baseline while the global
    best-so-far is tracked for the final result.  Optional checkpointing and
    MLflow logging make long-running jobs resumable and observable.

    Args:
        metric: Callable that scores (example, prediction, trace) tuples.  It may
            return a float, bool, ``Prediction`` with a ``score`` field, or a
            dict containing ``{"score": float, "feedback": str | None}``.
        analysis_lm: Language model used for per-example root-cause and success
            pattern analyses.
        analysis_adapter: Adapter applied to ``analysis_lm`` calls.  Defaults to
            :class:`~dspy.adapters.JSONAdapter` for structured outputs.
        max_iterations: Hard limit on optimization iterations.  ``None`` means
            run until ``convergence_patience`` is exhausted.
        hypothesis_lm: Optional language model for hypothesis synthesis.  Falls
            back to ``analysis_lm`` when omitted.
        hypothesis_adapter: Adapter for hypothesis generation calls.  Defaults to
            :class:`~dspy.adapters.JSONAdapter`.
        verbosity: Verbosity level controlling console logging and progress
            bars.  Accepts either a :class:`Verbosity` enum or its string name.
        num_threads: Maximum number of worker threads used for example
            evaluation and analysis.  Defaults to CPU count (falling back to
            DSPy's global ``num_threads`` setting) and is clamped to ``>= 1``.
        num_hypotheses: Maximum number of hypotheses to request per iteration.
        num_eval_runs: Number of repeated executions per calibration example.
            Median aggregation across runs stabilizes metric estimates.
        train_sample: Defaults to ``20`` examples per iteration.  ``None`` uses
            the full (shuffled) train set by sampling ``len(trainset)`` examples
            without replacement, while a callable ``SamplerFn`` receives
            ``(trainset, iteration)`` and returns a list of
            :class:`~dspy.primitives.Example` objects.
        success_threshold: Metric score at or above which a training example is
            treated as a success.  Defaults to ``max_metric``.
        min_metric: Lower bound used to clip metric outputs and to backstop
            failures.
        max_metric: Upper bound used to clip metric outputs and define the
            default ``success_threshold``.
        convergence_patience: Number of consecutive iterations without an
            improved candidate before stopping.  Defaults to ``5``. ``None``
            disables patience.
        seed: Random seed for sampling, tie-breaking, and hypothesis ordering.
        checkpoint_dir: Directory for serialized :class:`ApexCheckpoint`
            snapshots.  When provided, checkpoints are saved at the start and
            end of each iteration and can be reloaded via ``compile(...,
            resume=True)``.
        include_hypothesis_history: Whether hypothesis generation prompts are
            augmented with a summary of previously tested changes.
        candidate_selection: Strategy for choosing the iteration baseline
            candidate.  ``"best_on_val"`` picks the highest validation score,
            while ``"pareto"`` samples from the Pareto frontier using per-example
            win weights.
        pareto_merge_probability: Probability (0-1) of sampling an additional
            Pareto merge hypothesis each iteration. Only applies when
            ``candidate_selection="pareto"``.
        use_mlflow: Enables MLflow tracking of iterations, candidates, and
            scores via :class:`ExperimentTracker`.
        mlflow_tracking_uri: Optional MLflow tracking URI forwarded to the
            experiment tracker.  Defaults to ``"http://127.0.0.1:5000"``.
        mlflow_experiment_name: Optional experiment name when MLflow logging is
            enabled.  Defaults to ``"APEX"``.
    """

    def __init__(
        self,
        *,
        metric: MetricFn,
        analysis_lm: LM,
        analysis_adapter: Adapter | None = None,
        max_iterations: int | None = None,
        hypothesis_lm: LM | None = None,
        hypothesis_adapter: Adapter | None = None,
        verbosity: Verbosity | str | None = None,
        num_threads: int | None = None,
        num_hypotheses: int = 1,
        num_eval_runs: int = 1,
        train_sample: None | int | SamplerFn = 20,
        success_threshold: float | None = None,
        min_metric: float = 0.0,
        max_metric: float = 1.0,
        convergence_patience: int | None = 5,
        seed: int | None = None,
        checkpoint_dir: str | Path | None = None,
        include_hypothesis_history: bool = True,
        candidate_selection: CandidateSelectionStrategy = "pareto",
        pareto_merge_probability: float = 1.0,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = "http://127.0.0.1:5000",
        mlflow_experiment_name: str | None = "APEX",
    ) -> None:
        if max_iterations is None and convergence_patience is None:
            raise ValueError("At least one of max_iterations or convergence_patience must be specified.")
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be > 0 if specified.")
        if convergence_patience is not None and convergence_patience <= 0:
            raise ValueError("convergence_patience must be > 0 if specified.")
        if num_hypotheses < 0:
            raise ValueError("num_hypotheses must be >= 0.")
        if num_eval_runs <= 0:
            raise ValueError("num_eval_runs must be > 0.")
        if min_metric > max_metric:
            raise ValueError("min_metric cannot exceed max_metric.")
        if candidate_selection not in {"best_on_val", "pareto"}:
            raise ValueError("candidate_selection must be 'best_on_val' or 'pareto'.")
        if not 0.0 <= pareto_merge_probability <= 1.0:
            raise ValueError("pareto_merge_probability must be between 0.0 and 1.0.")

        self.metric = metric
        self.analysis_lm = analysis_lm
        self.hypothesis_lm = hypothesis_lm or analysis_lm
        self.analysis_adapter = analysis_adapter or JSONAdapter()
        self.hypothesis_adapter = hypothesis_adapter or JSONAdapter()

        default_threads = num_threads if num_threads is not None else (os.cpu_count() or 1)
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
        self.include_hypothesis_history = include_hypothesis_history
        self.candidate_selection: CandidateSelectionStrategy = candidate_selection
        self.pareto_merge_probability = float(pareto_merge_probability)

        self.runtime = RuntimeTools(verbosity=self.verbosity, num_threads=self.num_threads, logger=logger)
        self.checkpoints = CheckpointManager(checkpoint_dir, runtime=self.runtime)

        self.tracker = ExperimentTracker(
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
        )
        self.evaluator = EvaluationEngine(
            metric=self.metric,
            runtime=self.runtime,
            tracker=self.tracker,
            min_metric=self.min_metric,
            max_metric=self.max_metric,
            success_threshold=self.success_threshold,
            num_eval_runs=self.num_eval_runs,
            rng=self._rng,
            log=self._dispatch_log,
            is_enabled=self._is_enabled,
        )
        if use_mlflow:
            self.runtime.log("APEX: MLflow tracking enabled", Verbosity.NORMAL)

    def _is_enabled(self, level: Verbosity) -> bool:
        return self.runtime.is_enabled(level)

    def _log(self, message: str, level: Verbosity = Verbosity.NORMAL, log_level: LogLevel = "info") -> None:
        self.runtime.log(message, level=level, log_level=log_level)

    def _dispatch_log(
        self,
        message: str,
        level: Verbosity = Verbosity.NORMAL,
        log_level: LogLevel = "info",
    ) -> None:
        log_fn = self._log
        try:
            log_fn(message, level, log_level)  # type: ignore[misc]
        except TypeError:
            log_fn(message, level)  # type: ignore[misc]

    def _extract_execution_flow(
        self,
        trace: list[TraceEntry],
        program: Module,
    ) -> list[ExecutionFlowEntry]:
        return extract_execution_flow(trace, program)

    def _format_execution_flow_as_graph(self, execution_flow: list[ExecutionFlowEntry]) -> str:
        return format_execution_flow_as_graph(execution_flow)

    def _format_execution_flow_with_details(self, execution_flow: list[ExecutionFlowEntry]) -> str:
        return format_execution_flow_with_details(execution_flow)

    def _initialize_iteration_baseline(
        self,
        state: OptimizationState,
    ) -> tuple[CandidateRecord | None, SelectionResult | None]:
        if self.candidate_selection != "pareto":
            return None, None

        selection = select_baseline_candidate(
            candidates=state.all_candidates,
            strategy="pareto",
            rng=self._rng,
        )
        baseline_record = selection.baseline
        state.prev_iteration_best = baseline_record
        state.current_program = baseline_record.program.deepcopy()
        if self._is_enabled(Verbosity.DETAILED):
            frontier_note = f", frontier_size={len(selection.frontier)}"
            self._log(
                "APEX: Pareto baseline selected (iteration="
                f"{baseline_record.iteration}, score={baseline_record.overall_score:.4f}{frontier_note})",
                Verbosity.DETAILED,
            )
        return baseline_record, selection

    def _build_checkpoint_config(self) -> CheckpointConfig:
        return CheckpointConfig(
            max_iterations=self.max_iterations,
            num_hypotheses=self.num_hypotheses,
            num_eval_runs=self.num_eval_runs,
            train_sample=self.train_sample if isinstance(self.train_sample, int) else None,
            success_threshold=self.success_threshold,
            min_metric=self.min_metric,
            max_metric=self.max_metric,
            convergence_patience=self.convergence_patience,
            seed=self.seed,
            candidate_selection=self.candidate_selection,
            pareto_merge_probability=self.pareto_merge_probability,
        )

    def _evaluate_candidates(
        self,
        *,
        baseline: Module,
        hypotheses: Sequence[HypothesisSpec],
        calset: Sequence[Example],
        iteration: int,
        cached_baseline: CandidateRecord | None = None,
        baseline_overrides: dict[int, Module] | None = None,
    ) -> list[CandidateRecord]:
        return self.evaluator.evaluate_candidates(
            baseline=baseline,
            hypotheses=hypotheses,
            calset=calset,
            iteration=iteration,
            cached_baseline=cached_baseline,
            baseline_overrides=baseline_overrides,
        )

    def _evaluate_candidate(
        self,
        *,
        program: Module,
        calset: Sequence[Example],
        iteration: int,
        hypothesis: HypothesisSpec | None,
    ) -> CandidateRecord:
        return self.evaluator.evaluate_candidate(
            program=program,
            calset=calset,
            iteration=iteration,
            hypothesis=hypothesis,
        )

    def _apply_hypothesis(self, baseline: Module, hypothesis: HypothesisSpec) -> Module:
        return self.evaluator.apply_hypothesis(baseline, hypothesis)

    def _select_best_candidate(self, candidates: Sequence[CandidateRecord]) -> CandidateRecord:
        return self.evaluator.select_best_candidate(candidates)

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        resume: bool = False,
    ) -> Module:
        """Optimize ``student`` by iteratively refining its predictor prompts.

        The compile loop performs the following operations until the iteration
        budget or convergence patience is exhausted:

        1. Sample the training set (according to ``train_sample``), execute the
           current program on each example, and collect full execution traces.
        2. Run the analysis language model on every failure (and a balanced set
           of successes) to obtain structured root-cause and contrastive
           summaries.
        3. Ask the hypothesis model to synthesize those summaries into up to
           ``num_hypotheses`` candidate prompt revisions.
        4. Evaluate the baseline and each hypothesis on ``valset`` with
           ``num_eval_runs`` repetitions per example and median aggregation.
        5. Adopt the highest-scoring candidate as the new baseline while tracking
           the global best for the final return value.

        Checkpoints are optionally written at the beginning and end of each
        iteration, enabling interrupted runs to resume when ``resume`` is
        ``True`` and ``checkpoint_dir`` was specified during initialization.

        Args:
            student: Uncompiled DSPy :class:`~dspy.primitives.Module` to be
                optimized.  The module is deep-copied before modification.
            trainset: Labeled training examples used for error analysis and
                success pattern mining.
            teacher: Unsupported parameter required by the ``Teleprompter``
                interface.  Passing a value raises ``ValueError``.
            valset: Calibration examples that determine candidate scores.  Must
                be non-empty.
            resume: When ``True`` and checkpoints are available, resume from the
                most recent saved state instead of starting from ``student``.

        Returns:
            Module: A deep copy of the program associated with the best overall
                calibration score observed during optimization.

        Raises:
            ValueError: If ``teacher`` is provided or if either ``trainset`` or
                ``valset`` is empty.
        """
        if teacher is not None:
            raise ValueError("APEX does not support teacher programs.")
        if not trainset:
            raise ValueError("trainset must be non-empty.")
        if not valset:
            raise ValueError("calibration set (valset) must be provided and non-empty.")

        checkpoint: ApexCheckpoint | None = None
        state: OptimizationState | None = None

        with self.tracker:
            if self.tracker.is_active():
                self.tracker.log_params(
                    {
                        "max_iterations": self.max_iterations,
                        "num_hypotheses": self.num_hypotheses,
                        "num_eval_runs": self.num_eval_runs,
                        "success_threshold": self.success_threshold,
                        "convergence_patience": self.convergence_patience,
                        "seed": self.seed,
                        "candidate_selection": self.candidate_selection,
                        "pareto_merge_probability": self.pareto_merge_probability,
                        "train_size": len(trainset),
                        "val_size": len(valset),
                        "verbosity": str(self.verbosity),
                    }
                )

            if resume and self.checkpoints.enabled:
                checkpoint = self.checkpoints.load()

            if checkpoint:
                state, self.candidate_selection, self.pareto_merge_probability = resume_from_checkpoint(
                    checkpoint=checkpoint,
                    log=self._log,
                    rng=self._rng,
                    candidate_selection=self.candidate_selection,
                    pareto_merge_probability=self.pareto_merge_probability,
                )
            else:
                state = initialize_new_state(
                    student=student,
                    trainset=trainset,
                    valset=valset,
                    evaluator=self.evaluator,
                    tracker=self.tracker,
                    checkpoints=self.checkpoints,
                    build_checkpoint_config=self._build_checkpoint_config,
                    log=self._log,
                    rng=self._rng,
                    num_threads=self.num_threads,
                    max_iterations=self.max_iterations,
                    num_hypotheses=self.num_hypotheses,
                    success_threshold=self.success_threshold,
                    convergence_patience=self.convergence_patience,
                    seed=self.seed,
                )

            assert state is not None
            state = self._run_optimization_loop(
                state=state,
                trainset=trainset,
                valset=valset,
            )

        assert state is not None
        return finalize_optimization(
            state,
            log=self._log,
            is_enabled=self._is_enabled,
            candidate_selection=self.candidate_selection,
            pareto_merge_probability=self.pareto_merge_probability,
            tracker=self.tracker,
        )

    def _run_optimization_loop(
        self,
        *,
        state: OptimizationState,
        trainset: Sequence[Example],
        valset: Sequence[Example],
    ) -> OptimizationState:
        sampled_train: list[Example] = []
        failure_summaries = []
        success_summaries = []
        hypotheses: list[HypothesisSpec] = []
        iteration_candidates: list[CandidateRecord] | None = None
        selection_result: SelectionResult | None = None
        pareto_baseline: CandidateRecord | None = None

        try:
            while True:
                iteration_candidates = None
                iteration = state.start_next_iteration()

                if self.max_iterations is not None and iteration > self.max_iterations:
                    state.stop_reason = "max_iterations"
                    self._log(
                        "APEX: Stopping due to max iterations reached",
                        Verbosity.NORMAL,
                    )
                    break

                pareto_baseline, selection_result = self._initialize_iteration_baseline(state)

                sampler = self.train_sample if self.train_sample is not None else len(trainset)
                sampled_train = sample_trainset(
                    trainset,
                    sampler=sampler,
                    rng=self._rng,
                    iteration=iteration,
                )
                snapshot = snapshot_program(state.current_program)
                available_predictor_names = list(snapshot.prompts.keys()) if snapshot.prompts else []

                failures, successes = self.evaluator.evaluate_train_examples(
                    program=state.current_program,
                    trainset=sampled_train,
                    iteration=iteration,
                )
                if not failures:
                    self._log(
                        f"APEX: Iteration {iteration} - Perfect performance! All {len(successes)} examples succeeded",
                        Verbosity.NORMAL,
                    )
                    state.iteration_logs.append(
                        ApexIterationLog(
                            iteration=iteration,
                            sampled_train_size=len(sampled_train),
                            num_failures=0,
                            num_successes=len(successes),
                            hypotheses=[],
                            candidates=[],
                        )
                    )
                    state.no_improvement_count += 1
                    if self.convergence_patience is not None:
                        self._log(
                            "APEX: No improvement ("
                            f"{state.no_improvement_count}/{self.convergence_patience} patience)",
                            Verbosity.DETAILED,
                        )
                        if state.no_improvement_count >= self.convergence_patience:
                            state.stop_reason = "patience"
                            self._log(
                                "APEX: Stopping due to convergence patience reached (all successes)",
                                Verbosity.NORMAL,
                            )
                            break
                    self.checkpoints.save(
                        iteration=iteration,
                        current_program=state.current_program,
                        best_candidate=state.best_candidate,
                        all_candidates=state.all_candidates,
                        iteration_logs=state.iteration_logs,
                        no_improvement_count=state.no_improvement_count,
                        iteration_baseline=state.iteration_baseline,
                        rng_state=self._rng.getstate(),
                        config=self._build_checkpoint_config(),
                    )
                    continue

                analysis_fn = analyze_record
                if analysis_fn is _analyze_single_record:
                    failure_summaries, success_summaries = analyze_failures_and_successes(
                        failure_records=failures,
                        success_records=successes,
                        analysis_lm=self.analysis_lm,
                        analysis_adapter=self.analysis_adapter,
                        runtime=self.runtime,
                        tracker=self.tracker,
                        success_threshold=self.success_threshold,
                        min_metric=self.min_metric,
                        max_metric=self.max_metric,
                        format_execution_flow=self._format_execution_flow_with_details,
                        log=self._log,
                        iteration=iteration,
                        available_predictor_names=available_predictor_names,
                    )
                else:
                    failure_summaries = []
                    success_summaries = []
                    for idx, record in enumerate(failures):
                        summary = analysis_fn(  # type: ignore[misc]
                            record,
                            mode="failure",
                            analysis_lm=self.analysis_lm,
                            analysis_adapter=self.analysis_adapter,
                            runtime=self.runtime,
                            tracker=self.tracker,
                            success_threshold=self.success_threshold,
                            min_metric=self.min_metric,
                            max_metric=self.max_metric,
                            format_execution_flow=self._format_execution_flow_with_details,
                            log=self._log,
                            iteration=iteration,
                            example_index=idx,
                            available_predictor_names=available_predictor_names,
                        )
                        if summary is not None:
                            failure_summaries.append(summary)
                    for idx, record in enumerate(successes):
                        summary = analysis_fn(  # type: ignore[misc]
                            record,
                            mode="success",
                            analysis_lm=self.analysis_lm,
                            analysis_adapter=self.analysis_adapter,
                            runtime=self.runtime,
                            tracker=self.tracker,
                            success_threshold=self.success_threshold,
                            min_metric=self.min_metric,
                            max_metric=self.max_metric,
                            format_execution_flow=self._format_execution_flow_with_details,
                            log=self._log,
                            iteration=iteration,
                            example_index=idx,
                            available_predictor_names=None,
                        )
                        if summary is not None:
                            success_summaries.append(summary)
                self._log(
                    f"APEX: Train evaluation complete - {len(failures)} failures, {len(successes)} successes",
                    Verbosity.DETAILED,
                )
                if self._is_enabled(Verbosity.DETAILED):
                    _log_analysis_results(
                        failure_summaries=failure_summaries,
                        success_summaries=success_summaries,
                        log=self._log,
                    )

                if not failure_summaries and not success_summaries:
                    self._log(
                        "APEX: No analyses generated; continuing to next iteration",
                        Verbosity.DETAILED,
                    )
                    self.checkpoints.save(
                        iteration=iteration,
                        current_program=state.current_program,
                        best_candidate=state.best_candidate,
                        all_candidates=state.all_candidates,
                        iteration_logs=state.iteration_logs,
                        no_improvement_count=state.no_improvement_count,
                        iteration_baseline=state.iteration_baseline,
                        rng_state=self._rng.getstate(),
                        config=self._build_checkpoint_config(),
                    )
                    continue

                if self._is_enabled(Verbosity.DETAILED):
                    self._log(
                        "APEX: iteration "
                        f"{iteration} analyzed {len(failure_summaries)} failure(s) and {len(success_summaries)} success(es)",
                        Verbosity.DETAILED,
                    )

                hypotheses = generate_hypotheses(
                    failure_summaries=failure_summaries,
                    success_summaries=success_summaries,
                    snapshot=snapshot,
                    candidate_history=state.all_candidates,
                    best_val_score=state.best_candidate.overall_score,
                    runtime=self.runtime,
                    hypothesis_lm=self.hypothesis_lm,
                    hypothesis_adapter=self.hypothesis_adapter,
                    num_hypotheses=self.num_hypotheses,
                    include_history=self.include_hypothesis_history,
                    rng=self._rng,
                    log=self._log,
                    iteration=iteration,
                    tracker=self.tracker,
                    selection_strategy=self.candidate_selection,
                )

                merge_hypotheses: list[HypothesisSpec] = []
                merge_baseline_overrides: dict[int, Module] = {}
                if (
                    self.candidate_selection == "pareto"
                    and selection_result is not None
                    and len(selection_result.frontier) > 1
                    and self.pareto_merge_probability > 0.0
                ):
                    merge_roll = self._rng.random()
                    if merge_roll < self.pareto_merge_probability:
                        try:
                            partner_candidate = draw_weighted_candidate(
                                selection_result.frontier,
                                selection_result.weights,
                                rng=self._rng,
                                exclude=[pareto_baseline] if pareto_baseline is not None else None,
                            )
                        except ValueError:
                            partner_candidate = None

                        if partner_candidate is not None and pareto_baseline is not None:
                            if candidates_are_equivalent(partner_candidate, pareto_baseline):
                                self._log(
                                    "APEX: Skipped Pareto merge hypotheses (partner matches baseline)",
                                    Verbosity.DETAILED,
                                )
                            else:
                                merge_hypotheses = generate_merge_hypotheses(
                                    baseline_candidate=pareto_baseline,
                                    partner_candidate=partner_candidate,
                                    runtime=self.runtime,
                                    hypothesis_lm=self.hypothesis_lm,
                                    hypothesis_adapter=self.hypothesis_adapter,
                                    iteration=iteration,
                                    tracker=self.tracker,
                                )
                                if merge_hypotheses:
                                    hypotheses.extend(merge_hypotheses)
                                    if len(merge_hypotheses) > 1:
                                        partner_hypothesis = merge_hypotheses[1]
                                        merge_baseline_overrides[id(partner_hypothesis)] = partner_candidate.program
                                    count = len(merge_hypotheses)
                                    noun = "hypothesis" if count == 1 else "hypotheses"
                                    self._log(
                                        f"APEX: Generated {count} Pareto merge {noun}",
                                        Verbosity.DETAILED,
                                    )
                        else:
                            self._log(
                                "APEX: Skipped Pareto merge hypotheses (no suitable partner candidate found)",
                                Verbosity.DETAILED,
                            )

                total_hypotheses = len(hypotheses)
                merge_note = (
                    f" (including {len(merge_hypotheses)} Pareto merge"
                    f"{'s' if len(merge_hypotheses) != 1 else ''})"
                    if merge_hypotheses
                    else ""
                )
                self._log(
                    f"APEX: Generated {total_hypotheses} hypothesis{'es' if total_hypotheses != 1 else ''}{merge_note} for iteration {iteration}",
                    Verbosity.NORMAL,
                )

                if hypotheses and self._is_enabled(Verbosity.NORMAL):
                    merge_ids = {id(h) for h in merge_hypotheses}
                    for idx, hypothesis in enumerate(hypotheses, start=1):
                        predictors_updated = (
                            list(hypothesis.prompt_changes.keys()) if hypothesis.prompt_changes else []
                        )
                        is_merge = id(hypothesis) in merge_ids
                        hypothesis_label = f"Hypothesis #{idx} (Pareto merge)" if is_merge else f"Hypothesis #{idx}"
                        self._log(
                            f"  → {hypothesis_label}: {hypothesis.strategy} | Impact: {hypothesis.impact_score:.2f} | "
                            f"Targets: {', '.join(predictors_updated) if predictors_updated else 'none'}",
                            Verbosity.NORMAL,
                        )
                if self._is_enabled(Verbosity.DETAILED) and hypotheses:
                    self._log(
                        "APEX: Detailed hypothesis info follows...",
                        Verbosity.DETAILED,
                    )

                iteration_candidates = self.evaluator.evaluate_candidates(
                    baseline=state.current_program,
                    hypotheses=hypotheses,
                    calset=valset,
                    iteration=iteration,
                    cached_baseline=state.prev_iteration_best,
                    baseline_overrides=merge_baseline_overrides,
                )

                best_candidate_for_iteration = self.evaluator.select_best_candidate(iteration_candidates)

                state.all_candidates.extend(iteration_candidates)
                state.iteration_logs.append(
                    ApexIterationLog(
                        iteration=iteration,
                        sampled_train_size=len(sampled_train),
                        num_failures=len(failure_summaries),
                        num_successes=len(success_summaries),
                        hypotheses=hypotheses,
                        candidates=iteration_candidates,
                    )
                )

                if self.tracker.is_active():
                    iteration_data = tracking_utils.format_iteration_metrics(
                        iteration=iteration,
                        num_failures=len(failure_summaries),
                        num_successes=len(success_summaries),
                        hypotheses=hypotheses,
                        candidates=iteration_candidates,
                        best_score=best_candidate_for_iteration.overall_score,
                    )
                    self.tracker.log_iteration(iteration, iteration_data)

                    for idx, candidate in enumerate(iteration_candidates):
                        candidate_data = tracking_utils.format_candidate_data(candidate)
                        self.tracker.log_candidate(candidate_data, iteration, idx)

                if best_candidate_for_iteration.overall_score > state.best_candidate.overall_score:
                    state.best_candidate = best_candidate_for_iteration
                    self._log(
                        f"APEX: New best candidate found with score {state.best_candidate.overall_score:.4f}",
                        Verbosity.NORMAL,
                    )
                    if state.best_candidate.hypothesis and state.best_candidate.hypothesis.prompt_changes:
                        self._log(
                            "APEX: Improved "
                            f"{len(state.best_candidate.hypothesis.prompt_changes)} predictor prompt(s) - "
                            f"strategy: {state.best_candidate.hypothesis.strategy}",
                            Verbosity.NORMAL,
                        )
                        if self._is_enabled(Verbosity.DETAILED):
                            self._log(
                                "APEX: Detailed improved prompts:",
                                Verbosity.DETAILED,
                            )
                            for (
                                predictor_name,
                                changes,
                            ) in state.best_candidate.hypothesis.prompt_changes.items():
                                preview = (
                                    f"  → {predictor_name}: {changes.new_prompt[:300]}..."
                                    if len(changes.new_prompt) > 300
                                    else f"  → {predictor_name}: {changes.new_prompt}"
                                )
                                self._log(preview, Verbosity.DETAILED)

                state.iteration_baseline = iteration_candidates[0]
                self._log(
                    f"APEX: Iteration {iteration} best score: {best_candidate_for_iteration.overall_score:.4f}",
                    Verbosity.NORMAL,
                )

                if self._is_enabled(Verbosity.DETAILED):
                    score_improvements = [
                        candidate.overall_score - state.iteration_baseline.overall_score
                        for candidate in iteration_candidates[1:]
                    ]
                    if score_improvements:
                        self._log(
                            f"APEX: Score improvements from iteration baseline: {score_improvements}",
                            Verbosity.DETAILED,
                        )

                if best_candidate_for_iteration is state.iteration_baseline:
                    state.no_improvement_count += 1
                    if self.convergence_patience is not None:
                        self._log(
                            "APEX: No improvement ("
                            f"{state.no_improvement_count}/{self.convergence_patience} patience)",
                            Verbosity.DETAILED,
                        )
                        if state.no_improvement_count >= self.convergence_patience:
                            state.stop_reason = "patience"
                            self._log(
                                "APEX: Stopping due to convergence patience reached",
                                Verbosity.NORMAL,
                            )
                            break
                    else:
                        self._log(
                            f"APEX: No improvement in iteration {iteration} (patience disabled)",
                            Verbosity.DETAILED,
                        )
                else:
                    state.no_improvement_count = 0
                    state.current_program = best_candidate_for_iteration.program
                    self._log(
                        "APEX: Updating program with hypothesis improvements",
                        Verbosity.DETAILED,
                    )

                state.prev_iteration_best = best_candidate_for_iteration

                self.checkpoints.save(
                    iteration=iteration,
                    current_program=state.current_program,
                    best_candidate=state.best_candidate,
                    all_candidates=state.all_candidates,
                    iteration_logs=state.iteration_logs,
                    no_improvement_count=state.no_improvement_count,
                    iteration_baseline=state.iteration_baseline,
                    rng_state=self._rng.getstate(),
                    config=self._build_checkpoint_config(),
                )

        except KeyboardInterrupt:
            state.stop_reason = "interrupted"
            self._log(
                "APEX: Optimization interrupted by user (Ctrl+C)",
                Verbosity.NORMAL,
            )

            if iteration_candidates:
                state.iteration_logs.append(
                    ApexIterationLog(
                        iteration=state.iteration,
                        sampled_train_size=len(sampled_train),
                        num_failures=len(failure_summaries),
                        num_successes=len(success_summaries),
                        hypotheses=hypotheses,
                        candidates=iteration_candidates,
                    )
                )

            if self.checkpoints.enabled:
                baseline_for_checkpoint = state.iteration_baseline or state.best_candidate
                self.checkpoints.save(
                    iteration=state.iteration,
                    current_program=state.current_program,
                    best_candidate=state.best_candidate,
                    all_candidates=state.all_candidates,
                    iteration_logs=state.iteration_logs,
                    no_improvement_count=state.no_improvement_count,
                    iteration_baseline=baseline_for_checkpoint,
                    rng_state=self._rng.getstate(),
                    config=self._build_checkpoint_config(),
                )
                self._log(
                    f"APEX: Checkpoint saved at iteration {state.iteration} - resume with resume=True",
                    Verbosity.NORMAL,
                )

        return state
