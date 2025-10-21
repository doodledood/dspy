from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from dspy.adapters import Adapter
from dspy.clients.lm import LM
from dspy.primitives import Example, Module

from . import tracking_utils
from .analysis import analyze_failures_and_successes
from .candidate_selection import (
    CandidateSelectionStrategy,
    SelectionResult,
    candidates_are_equivalent,
    draw_weighted_candidate,
    select_baseline_candidate,
)
from .checkpoint_manager import CheckpointManager
from .evaluation import EvaluationEngine
from .models import ApexIterationLog, CandidateRecord, CheckpointConfig, HypothesisSpec
from .runtime import RuntimeTools
from .sampling import sample_trainset
from .snapshot import snapshot_program
from .state import OptimizationState
from .tracker import ExperimentTracker
from .types import SamplerFn, Verbosity

LogFn = Callable[[str, Verbosity], None]
VerbosityPredicate = Callable[[Verbosity], bool]


@dataclass(frozen=True)
class LoopSettings:
    max_iterations: int | None
    convergence_patience: int | None
    train_sample: None | int | SamplerFn
    num_hypotheses: int
    include_hypothesis_history: bool
    candidate_selection: CandidateSelectionStrategy
    pareto_merge_probability: float
    success_threshold: float
    min_metric: float
    max_metric: float


@dataclass
class LoopCollaborators:
    runtime: RuntimeTools
    evaluator: EvaluationEngine
    tracker: ExperimentTracker
    checkpoints: CheckpointManager
    rng: random.Random
    build_checkpoint_config: Callable[[], CheckpointConfig]
    log: LogFn
    is_enabled: VerbosityPredicate
    analysis_lm: LM
    analysis_adapter: Adapter
    hypothesis_lm: LM
    hypothesis_adapter: Adapter
    analysis_fn: Callable[..., object] | None
    format_execution_flow: Callable[[Sequence], str]
    generate_hypotheses: Callable[..., object]
    generate_merge_hypotheses: Callable[..., object]


@dataclass
class IterationContext:
    iteration: int
    sampled_train: list[Example] = field(default_factory=list)
    failure_summaries: list = field(default_factory=list)
    success_summaries: list = field(default_factory=list)
    hypotheses: list[HypothesisSpec] = field(default_factory=list)
    candidates: list[CandidateRecord] | None = None


@dataclass
class IterationResult:
    stop: bool = False
    skip_to_next: bool = False


class OptimizationLoop:
    def __init__(self, settings: LoopSettings, collaborators: LoopCollaborators) -> None:
        self.settings = settings
        self.cb = collaborators

    def run(
        self,
        *,
        state: OptimizationState,
        trainset: Sequence[Example],
        valset: Sequence[Example],
    ) -> OptimizationState:
        context = IterationContext(iteration=state.iteration)

        try:
            while True:
                iteration = state.start_next_iteration()
                context = IterationContext(iteration=iteration)

                if self.settings.max_iterations is not None and iteration > self.settings.max_iterations:
                    state.stop_reason = "max_iterations"
                    self.cb.log("APEX: Stopping due to max iterations reached", Verbosity.NORMAL)
                    break

                pareto_baseline, selection_result = self._initialize_iteration_baseline(state)

                result = self._run_iteration(
                    state=state,
                    trainset=trainset,
                    valset=valset,
                    iteration=iteration,
                    context=context,
                    pareto_baseline=pareto_baseline,
                    selection_result=selection_result,
                )

                if result.stop:
                    break
                if result.skip_to_next:
                    continue

        except KeyboardInterrupt:
            self._handle_interrupt(state, context)

        return state

    def _initialize_iteration_baseline(
        self, state: OptimizationState
    ) -> tuple[CandidateRecord | None, SelectionResult | None]:
        if self.settings.candidate_selection != "pareto":
            return None, None

        selection = select_baseline_candidate(
            candidates=state.all_candidates,
            strategy="pareto",
            rng=self.cb.rng,
        )
        baseline_record = selection.baseline
        state.prev_iteration_best = baseline_record
        state.current_program = baseline_record.program.deepcopy()

        if self.cb.is_enabled(Verbosity.DETAILED):
            frontier_note = f", frontier_size={len(selection.frontier)}"
            self.cb.log(
                "APEX: Pareto baseline selected (iteration="
                f"{baseline_record.iteration}, score={baseline_record.overall_score:.4f}{frontier_note})",
                Verbosity.DETAILED,
            )

        return baseline_record, selection

    def _run_iteration(
        self,
        *,
        state: OptimizationState,
        trainset: Sequence[Example],
        valset: Sequence[Example],
        iteration: int,
        context: IterationContext,
        pareto_baseline: CandidateRecord | None,
        selection_result: SelectionResult | None,
    ) -> IterationResult:
        sampler = self.settings.train_sample if self.settings.train_sample is not None else len(trainset)
        sampled_train = sample_trainset(
            trainset,
            sampler=sampler,
            rng=self.cb.rng,
            iteration=iteration,
        )
        context.sampled_train = sampled_train

        snapshot = snapshot_program(state.current_program)
        available_predictor_names = list(snapshot.prompts.keys()) if snapshot.prompts else []

        failures, successes = self.cb.evaluator.evaluate_train_examples(
            program=state.current_program,
            trainset=sampled_train,
            iteration=iteration,
        )

        if not failures:
            return self._handle_perfect_performance(
                state=state,
                iteration=iteration,
                sampled_train=sampled_train,
                success_count=len(successes),
            )

        failure_summaries, success_summaries = self._run_analysis(
            failures=failures,
            successes=successes,
            iteration=iteration,
            available_predictor_names=available_predictor_names,
        )
        context.failure_summaries = failure_summaries
        context.success_summaries = success_summaries

        self.cb.log(
            f"APEX: Train evaluation complete - {len(failures)} failures, {len(successes)} successes",
            Verbosity.DETAILED,
        )

        if not failure_summaries and not success_summaries:
            self.cb.log(
                "APEX: No analyses generated; continuing to next iteration",
                Verbosity.DETAILED,
            )
            self._save_checkpoint(state, iteration)
            return IterationResult(skip_to_next=True)

        if self.cb.is_enabled(Verbosity.DETAILED):
            self.cb.log(
                "APEX: iteration "
                f"{iteration} analyzed {len(failure_summaries)} failure(s) and {len(success_summaries)} success(es)",
                Verbosity.DETAILED,
            )

        hypotheses = self.cb.generate_hypotheses(
            failure_summaries=failure_summaries,
            success_summaries=success_summaries,
            snapshot=snapshot,
            candidate_history=state.all_candidates,
            best_val_score=state.best_candidate.overall_score,
            runtime=self.cb.runtime,
            hypothesis_lm=self.cb.hypothesis_lm,
            hypothesis_adapter=self.cb.hypothesis_adapter,
            num_hypotheses=self.settings.num_hypotheses,
            include_history=self.settings.include_hypothesis_history,
            rng=self.cb.rng,
            log=self.cb.log,
            iteration=iteration,
            tracker=self.cb.tracker,
            selection_strategy=self.settings.candidate_selection,
        )

        merge_hypotheses, merge_overrides = self._maybe_generate_merge_hypotheses(
            iteration=iteration,
            pareto_baseline=pareto_baseline,
            selection_result=selection_result,
            snapshot=snapshot,
            failure_summaries=failure_summaries,
            success_summaries=success_summaries,
            state=state,
        )
        if merge_hypotheses:
            hypotheses.extend(merge_hypotheses)

        context.hypotheses = hypotheses

        total_hypotheses = len(hypotheses)
        merge_note = (
            f" (including {len(merge_hypotheses)} Pareto merge{'s' if len(merge_hypotheses) != 1 else ''})"
            if merge_hypotheses
            else ""
        )
        self.cb.log(
            f"APEX: Generated {total_hypotheses} hypothesis{'es' if total_hypotheses != 1 else ''}{merge_note} for iteration {iteration}",
            Verbosity.NORMAL,
        )

        if hypotheses and self.cb.is_enabled(Verbosity.NORMAL):
            merge_ids = {id(h) for h in merge_hypotheses}
            for idx, hypothesis in enumerate(hypotheses, start=1):
                predictors_updated = list(hypothesis.prompt_changes.keys()) if hypothesis.prompt_changes else []
                is_merge = id(hypothesis) in merge_ids
                label = f"Hypothesis #{idx} (Pareto merge)" if is_merge else f"Hypothesis #{idx}"
                self.cb.log(
                    f"  → {label}: {hypothesis.strategy} | Impact: {hypothesis.impact_score:.2f} | "
                    f"Targets: {', '.join(predictors_updated) if predictors_updated else 'none'}",
                    Verbosity.NORMAL,
                )
        if self.cb.is_enabled(Verbosity.DETAILED) and hypotheses:
            self.cb.log("APEX: Detailed hypothesis info follows...", Verbosity.DETAILED)

        iteration_candidates = self.cb.evaluator.evaluate_candidates(
            baseline=state.current_program,
            hypotheses=hypotheses,
            calset=valset,
            iteration=iteration,
            cached_baseline=state.prev_iteration_best,
            baseline_overrides=merge_overrides,
        )
        context.candidates = iteration_candidates

        best_candidate = self.cb.evaluator.select_best_candidate(iteration_candidates)

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

        if self.cb.tracker.is_active():
            iteration_data = tracking_utils.format_iteration_metrics(
                iteration=iteration,
                num_failures=len(failure_summaries),
                num_successes=len(success_summaries),
                hypotheses=hypotheses,
                candidates=iteration_candidates,
                best_score=best_candidate.overall_score,
            )
            self.cb.tracker.log_iteration(iteration, iteration_data)

            for idx, candidate in enumerate(iteration_candidates):
                candidate_data = tracking_utils.format_candidate_data(candidate)
                self.cb.tracker.log_candidate(candidate_data, iteration, idx)

        if best_candidate.overall_score > state.best_candidate.overall_score:
            state.best_candidate = best_candidate
            self.cb.log(
                f"APEX: New best candidate found with score {state.best_candidate.overall_score:.4f}",
                Verbosity.NORMAL,
            )
            if state.best_candidate.hypothesis and state.best_candidate.hypothesis.prompt_changes:
                self.cb.log(
                    "APEX: Improved "
                    f"{len(state.best_candidate.hypothesis.prompt_changes)} predictor prompt(s) - "
                    f"strategy: {state.best_candidate.hypothesis.strategy}",
                    Verbosity.NORMAL,
                )
                if self.cb.is_enabled(Verbosity.DETAILED):
                    self.cb.log("APEX: Detailed improved prompts:", Verbosity.DETAILED)
                    for predictor_name, changes in state.best_candidate.hypothesis.prompt_changes.items():
                        preview = (
                            f"  → {predictor_name}: {changes.new_prompt[:300]}..."
                            if len(changes.new_prompt) > 300
                            else f"  → {predictor_name}: {changes.new_prompt}"
                        )
                        self.cb.log(preview, Verbosity.DETAILED)

        state.iteration_baseline = iteration_candidates[0]
        self.cb.log(
            f"APEX: Iteration {iteration} best score: {best_candidate.overall_score:.4f}",
            Verbosity.NORMAL,
        )

        if self.cb.is_enabled(Verbosity.DETAILED):
            score_improvements = [
                candidate.overall_score - state.iteration_baseline.overall_score
                for candidate in iteration_candidates[1:]
            ]
            if score_improvements:
                self.cb.log(
                    f"APEX: Score improvements from iteration baseline: {score_improvements}",
                    Verbosity.DETAILED,
                )

        if best_candidate is state.iteration_baseline:
            state.no_improvement_count += 1
            if self.settings.convergence_patience is not None:
                self.cb.log(
                    "APEX: No improvement ("
                    f"{state.no_improvement_count}/{self.settings.convergence_patience} patience)",
                    Verbosity.DETAILED,
                )
                if state.no_improvement_count >= (self.settings.convergence_patience or 0):
                    state.stop_reason = "patience"
                    self.cb.log("APEX: Stopping due to convergence patience reached", Verbosity.NORMAL)
                    return IterationResult(stop=True)
            else:
                self.cb.log(
                    f"APEX: No improvement in iteration {iteration} (patience disabled)",
                    Verbosity.DETAILED,
                )
        else:
            state.no_improvement_count = 0
            state.current_program = best_candidate.program
            self.cb.log("APEX: Updating program with hypothesis improvements", Verbosity.DETAILED)

        state.prev_iteration_best = best_candidate

        self._save_checkpoint(state, iteration)
        return IterationResult()

    def _handle_perfect_performance(
        self,
        *,
        state: OptimizationState,
        iteration: int,
        sampled_train: Sequence[Example],
        success_count: int,
    ) -> IterationResult:
        self.cb.log(
            f"APEX: Iteration {iteration} - Perfect performance! All {success_count} examples succeeded",
            Verbosity.NORMAL,
        )
        state.iteration_logs.append(
            ApexIterationLog(
                iteration=iteration,
                sampled_train_size=len(sampled_train),
                num_failures=0,
                num_successes=success_count,
                hypotheses=[],
                candidates=[],
            )
        )
        state.no_improvement_count += 1
        if self.settings.convergence_patience is not None:
            self.cb.log(
                f"APEX: No improvement ({state.no_improvement_count}/{self.settings.convergence_patience} patience)",
                Verbosity.DETAILED,
            )
            if state.no_improvement_count >= (self.settings.convergence_patience or 0):
                state.stop_reason = "patience"
                self.cb.log(
                    "APEX: Stopping due to convergence patience reached (all successes)",
                    Verbosity.NORMAL,
                )
                return IterationResult(stop=True)

        self._save_checkpoint(state, iteration)
        return IterationResult(skip_to_next=True)

    def _run_analysis(
        self,
        *,
        failures,
        successes,
        iteration: int,
        available_predictor_names: list[str],
    ) -> tuple[list, list]:
        analysis_fn = self.cb.analysis_fn

        if analysis_fn is None:
            failure_summaries, success_summaries = analyze_failures_and_successes(
                failure_records=failures,
                success_records=successes,
                analysis_lm=self.cb.analysis_lm,
                analysis_adapter=self.cb.analysis_adapter,
                runtime=self.cb.runtime,
                tracker=self.cb.tracker,
                success_threshold=self.settings.success_threshold,
                min_metric=self.settings.min_metric,
                max_metric=self.settings.max_metric,
                format_execution_flow=self.cb.format_execution_flow,
                log=self.cb.log,
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
                    analysis_lm=self.cb.analysis_lm,
                    analysis_adapter=self.cb.analysis_adapter,
                    runtime=self.cb.runtime,
                    tracker=self.cb.tracker,
                    success_threshold=self.settings.success_threshold,
                    min_metric=self.settings.min_metric,
                    max_metric=self.settings.max_metric,
                    format_execution_flow=self.cb.format_execution_flow,
                    log=self.cb.log,
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
                    analysis_lm=self.cb.analysis_lm,
                    analysis_adapter=self.cb.analysis_adapter,
                    runtime=self.cb.runtime,
                    tracker=self.cb.tracker,
                    success_threshold=self.settings.success_threshold,
                    min_metric=self.settings.min_metric,
                    max_metric=self.settings.max_metric,
                    format_execution_flow=self.cb.format_execution_flow,
                    log=self.cb.log,
                    iteration=iteration,
                    example_index=idx,
                    available_predictor_names=available_predictor_names,
                )
                if summary is not None:
                    success_summaries.append(summary)

        return failure_summaries, success_summaries

    def _maybe_generate_merge_hypotheses(
        self,
        *,
        iteration: int,
        pareto_baseline: CandidateRecord | None,
        selection_result: SelectionResult | None,
        snapshot,
        failure_summaries: list,
        success_summaries: list,
        state: OptimizationState,
    ) -> tuple[list[HypothesisSpec], dict[int, Module]]:
        if self.settings.candidate_selection != "pareto":
            self.cb.log(
                f"APEX: Skipping merge (not using Pareto selection, using {self.settings.candidate_selection})",
                Verbosity.DETAILED,
            )
            return [], {}

        if selection_result is None:
            self.cb.log(
                "APEX: Skipping merge (selection_result is None)",
                Verbosity.DETAILED,
            )
            return [], {}

        frontier_size = len(selection_result.frontier) if selection_result else 0
        if frontier_size <= 1:
            self.cb.log(
                f"APEX: Skipping merge (frontier too small: {frontier_size} <= 1)",
                Verbosity.DETAILED,
            )
            return [], {}

        if self.settings.pareto_merge_probability <= 0.0:
            self.cb.log(
                f"APEX: Skipping merge (probability too low: {self.settings.pareto_merge_probability})",
                Verbosity.DETAILED,
            )
            return [], {}

        merge_roll = self.cb.rng.random()
        if merge_roll >= self.settings.pareto_merge_probability:
            self.cb.log(
                f"APEX: Skipping merge (random roll {merge_roll:.4f} >= probability {self.settings.pareto_merge_probability})",
                Verbosity.DETAILED,
            )
            return [], {}

        self.cb.log(
            f"APEX: Attempting merge generation (frontier_size={frontier_size}, probability={self.settings.pareto_merge_probability}, roll={merge_roll:.4f})",
            Verbosity.DETAILED,
        )

        try:
            partner_candidate = draw_weighted_candidate(
                selection_result.frontier,
                selection_result.weights,
                rng=self.cb.rng,
                exclude=[pareto_baseline] if pareto_baseline is not None else None,
            )
        except ValueError:
            partner_candidate = None

        if partner_candidate is None or pareto_baseline is None:
            self.cb.log(
                "APEX: Skipped Pareto merge hypotheses (no suitable partner candidate found)",
                Verbosity.DETAILED,
            )
            return [], {}

        if candidates_are_equivalent(partner_candidate, pareto_baseline):
            self.cb.log(
                "APEX: Skipped Pareto merge hypotheses (partner matches baseline)",
                Verbosity.DETAILED,
            )
            return [], {}

        # Calculate general optimization health metric
        total_examples = len(failure_summaries) + len(success_summaries)
        success_rate_pct = (len(success_summaries) / total_examples * 100.0) if total_examples else 0.0

        merge_hypotheses = self.cb.generate_merge_hypotheses(
            baseline_candidate=pareto_baseline,
            partner_candidate=partner_candidate,
            runtime=self.cb.runtime,
            hypothesis_lm=self.cb.hypothesis_lm,
            hypothesis_adapter=self.cb.hypothesis_adapter,
            iteration=iteration,
            tracker=self.cb.tracker,
            snapshot=snapshot,
            candidate_history=state.all_candidates,
            best_val_score=state.best_candidate.overall_score,
            selection_strategy=self.settings.candidate_selection,
            include_history=self.settings.include_hypothesis_history,
            success_rate_percentage=success_rate_pct,
        )

        if not merge_hypotheses:
            return [], {}

        if len(merge_hypotheses) > 1:
            partner_hypothesis = merge_hypotheses[1]
            overrides = {id(partner_hypothesis): partner_candidate.program}
        else:
            overrides = {}

        count = len(merge_hypotheses)
        noun = "hypothesis" if count == 1 else "hypotheses"
        self.cb.log(
            f"APEX: Generated {count} Pareto merge {noun}",
            Verbosity.DETAILED,
        )
        return merge_hypotheses, overrides

    def _save_checkpoint(self, state: OptimizationState, iteration: int) -> None:
        self.cb.checkpoints.save(
            iteration=iteration,
            current_program=state.current_program,
            best_candidate=state.best_candidate,
            all_candidates=state.all_candidates,
            iteration_logs=state.iteration_logs,
            no_improvement_count=state.no_improvement_count,
            iteration_baseline=state.iteration_baseline,
            rng_state=self.cb.rng.getstate(),
            config=self.cb.build_checkpoint_config(),
        )

    def _handle_interrupt(self, state: OptimizationState, context: IterationContext) -> None:
        state.stop_reason = "interrupted"
        self.cb.log("APEX: Optimization interrupted by user (Ctrl+C)", Verbosity.NORMAL)

        if context.candidates:
            state.iteration_logs.append(
                ApexIterationLog(
                    iteration=context.iteration,
                    sampled_train_size=len(context.sampled_train),
                    num_failures=len(context.failure_summaries),
                    num_successes=len(context.success_summaries),
                    hypotheses=context.hypotheses,
                    candidates=context.candidates,
                )
            )

        if self.cb.checkpoints.enabled:
            baseline = state.iteration_baseline or state.best_candidate
            self.cb.checkpoints.save(
                iteration=state.iteration,
                current_program=state.current_program,
                best_candidate=state.best_candidate,
                all_candidates=state.all_candidates,
                iteration_logs=state.iteration_logs,
                no_improvement_count=state.no_improvement_count,
                iteration_baseline=baseline,
                rng_state=self.cb.rng.getstate(),
                config=self.cb.build_checkpoint_config(),
            )
            self.cb.log(
                f"APEX: Checkpoint saved at iteration {state.iteration} - resume with resume=True",
                Verbosity.NORMAL,
            )
