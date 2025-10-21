from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import dspy
from dspy.adapters import Adapter, JSONAdapter
from dspy.clients.lm import LM
from dspy.primitives import Example, Module
from dspy.teleprompt.teleprompt import Teleprompter

from .analysis import (
    generate_hypotheses as _generate_hypotheses,
)
from .analysis import (
    generate_merge_hypotheses as _generate_merge_hypotheses,
)
from .candidate_selection import CandidateSelectionStrategy
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
    CandidateRecord,
    CheckpointConfig,
    ExecutionFlowEntry,
    HypothesisSpec,
)
from .optimization import LoopCollaborators, LoopSettings, OptimizationLoop
from .runtime import RuntimeTools
from .state import OptimizationState
from .tracker import ExperimentTracker
from .types import LogLevel, MetricFn, SamplerFn, TraceEntry, Verbosity

logger = logging.getLogger(__name__)


@dataclass
class AnalysisHooks:
    """Configurable callables that drive analysis and hypothesis generation.

    ``analyze_record`` defaults to ``None``, which instructs the optimization loop
    to use the built-in batch analysis pipeline. Supplying a callable enables
    per-record overrides while keeping the hypothesis generators pluggable.
    """

    analyze_record: Callable[..., object] | None = None
    generate_hypotheses: Callable[..., object] = _generate_hypotheses
    generate_merge_hypotheses: Callable[..., object] = _generate_merge_hypotheses


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

        self.analysis_hooks: AnalysisHooks = AnalysisHooks()

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

    def _format_execution_flow_with_details(
        self, execution_flow: list[ExecutionFlowEntry], program: Module | None = None
    ) -> str:
        return format_execution_flow_with_details(execution_flow, program=program)

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
                program=student,
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
        program: Module,
    ) -> OptimizationState:
        # Create closure that captures program for execution flow formatting
        def format_with_program(execution_flow: list[ExecutionFlowEntry]) -> str:
            return self._format_execution_flow_with_details(execution_flow, program=program)

        loop = OptimizationLoop(
            settings=LoopSettings(
                max_iterations=self.max_iterations,
                convergence_patience=self.convergence_patience,
                train_sample=self.train_sample,
                num_hypotheses=self.num_hypotheses,
                include_hypothesis_history=self.include_hypothesis_history,
                candidate_selection=self.candidate_selection,
                pareto_merge_probability=self.pareto_merge_probability,
                success_threshold=self.success_threshold,
                min_metric=self.min_metric,
                max_metric=self.max_metric,
            ),
            collaborators=LoopCollaborators(
                runtime=self.runtime,
                evaluator=self.evaluator,
                tracker=self.tracker,
                checkpoints=self.checkpoints,
                rng=self._rng,
                build_checkpoint_config=self._build_checkpoint_config,
                log=self._log,
                is_enabled=self._is_enabled,
                analysis_lm=self.analysis_lm,
                analysis_adapter=self.analysis_adapter,
                hypothesis_lm=self.hypothesis_lm,
                hypothesis_adapter=self.hypothesis_adapter,
                analysis_fn=self.analysis_hooks.analyze_record,
                format_execution_flow=format_with_program,
                generate_hypotheses=self.analysis_hooks.generate_hypotheses,
                generate_merge_hypotheses=self.analysis_hooks.generate_merge_hypotheses,
            ),
        )

        return loop.run(state=state, trainset=trainset, valset=valset)
