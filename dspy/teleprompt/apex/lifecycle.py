"""Lifecycle utilities for the APEX optimizer.

This module houses the coarse-grained orchestration helpers that manage
initialization, checkpoint resumption, and optimization finalization.  The
functions are intentionally pure (aside from the collaborators passed in) so
they can be unit tested in isolation and keep :mod:`apex` focused on the
high-level control flow.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from dspy.primitives import Example, Module

from . import tracking_utils
from .checkpoint_manager import CheckpointManager
from .models import ApexCheckpoint, ApexOptimizationResult, CandidateRecord, CheckpointConfig
from .state import OptimizationState
from .summary import generate_optimization_summary
from .tracker import ExperimentTracker
from .types import Verbosity

LogFn = Callable[[str, Verbosity], None]
VerbosityPredicate = Callable[[Verbosity], bool]


def resume_from_checkpoint(
    *,
    checkpoint: ApexCheckpoint,
    log: LogFn,
    rng: random.Random,
    candidate_selection: str,
    pareto_merge_probability: float,
) -> tuple[OptimizationState, str, float]:
    """Rehydrate the optimization state from an existing checkpoint.

    Returns the recovered :class:`OptimizationState` along with any updated
    ``candidate_selection`` or ``pareto_merge_probability`` values persisted in
    the checkpoint configuration.
    """

    state = OptimizationState.from_checkpoint(checkpoint)
    config = checkpoint.config

    if getattr(config, "candidate_selection", None) is not None:
        if config.candidate_selection != candidate_selection:
            log(
                "APEX: Overriding candidate_selection with checkpoint configuration.",
                Verbosity.DETAILED,
            )
        candidate_selection = config.candidate_selection

    if getattr(config, "pareto_merge_probability", None) is not None:
        if config.pareto_merge_probability != pareto_merge_probability:
            log(
                "APEX: Overriding pareto_merge_probability with checkpoint configuration.",
                Verbosity.DETAILED,
            )
        pareto_merge_probability = config.pareto_merge_probability

    rng.setstate(checkpoint.rng_state)
    log(f"APEX: Resuming from iteration {state.iteration}", Verbosity.NORMAL)

    return state, candidate_selection, pareto_merge_probability


def initialize_new_state(
    *,
    student: Module,
    trainset: Sequence[Example],
    valset: Sequence[Example],
    evaluator,
    tracker: ExperimentTracker,
    checkpoints: CheckpointManager,
    build_checkpoint_config: Callable[[], CheckpointConfig],
    log: LogFn,
    rng: random.Random,
    num_threads: int,
    max_iterations: int | None,
    num_hypotheses: int,
    success_threshold: float,
    convergence_patience: int | None,
    seed: int,
) -> OptimizationState:
    """Evaluate the initial baseline program and persist the first checkpoint."""

    current_program = student.deepcopy()
    assert not getattr(current_program, "_compiled", False), "Student must be uncompiled."

    log(f"APEX: running with num_threads={num_threads}", Verbosity.DETAILED)
    max_iter_str = f"{max_iterations}" if max_iterations is not None else "until convergence"
    patience_str = f"{convergence_patience}" if convergence_patience is not None else "disabled"
    log(
        "APEX: Configuration - "
        f"max_iterations={max_iter_str}, "
        f"num_hypotheses={num_hypotheses}, "
        f"success_threshold={success_threshold:.2f}, "
        f"convergence_patience={patience_str}",
        Verbosity.DETAILED,
    )
    log(f"APEX: Using seed={seed} for reproducibility", Verbosity.DETAILED)
    log("APEX: Evaluating initial baseline on validation set", Verbosity.NORMAL)

    baseline_candidate: CandidateRecord = evaluator.evaluate_candidate(
        program=current_program.deepcopy(),
        calset=valset,
        iteration=0,
        hypothesis=None,
    )

    state = OptimizationState.initialize(program=current_program, baseline=baseline_candidate)
    log(
        f"APEX: Initial baseline score={state.iteration_baseline.overall_score:.4f}",
        Verbosity.NORMAL,
    )

    if tracker.is_active():
        baseline_metrics = tracking_utils.format_baseline_metrics(
            baseline_score=state.iteration_baseline.overall_score,
            num_train=len(trainset),
            num_val=len(valset),
        )
        tracker.log_metrics(baseline_metrics, step=0)

    checkpoints.save(
        iteration=0,
        current_program=state.current_program,
        best_candidate=state.best_candidate,
        all_candidates=state.all_candidates,
        iteration_logs=state.iteration_logs,
        no_improvement_count=state.no_improvement_count,
        iteration_baseline=state.iteration_baseline,
        rng_state=rng.getstate(),
        config=build_checkpoint_config(),
    )

    return state


def finalize_optimization(
    state: OptimizationState,
    *,
    log: LogFn,
    is_enabled: VerbosityPredicate,
    candidate_selection: str,
    pareto_merge_probability: float,
    tracker: ExperimentTracker,
    print_fn: Callable[[str], None] = print,
) -> Module:
    """Emit the final optimization summary and annotate the optimized program."""

    if not state.stop_reason:
        state.stop_reason = "completed"

    log(
        f"APEX: ✓ Optimization complete | {len(state.iteration_logs)} iterations | Reason: {state.stop_reason}",
        Verbosity.NORMAL,
    )
    improvement = state.best_candidate.overall_score - state.initial_baseline.overall_score
    log(
        f"APEX: Final score: {state.best_candidate.overall_score:.4f} "
        f"({'+' if improvement >= 0 else ''}{improvement:.4f} "
        f"from baseline {state.initial_baseline.overall_score:.4f})",
        Verbosity.NORMAL,
    )

    if is_enabled(Verbosity.NORMAL):
        summary_table = generate_optimization_summary(
            iterations=state.iteration_logs,
            best_candidate=state.best_candidate,
            initial_score=state.initial_baseline.overall_score,
            selection_strategy=candidate_selection,
            pareto_merge_probability=(pareto_merge_probability if candidate_selection == "pareto" else None),
        )
        print_fn(summary_table)

    if tracker.is_active():
        summary = tracking_utils.format_optimization_summary(
            best_candidate=state.best_candidate,
            all_candidates=state.all_candidates,
            iterations=state.iteration_logs,
            stopped_after=state.stop_reason,
            initial_score=state.initial_baseline.overall_score,
        )
        tracker.log_metrics(summary)

        if state.best_candidate.hypothesis:
            best_program_data = {
                "overall_score": state.best_candidate.overall_score,
                "iteration": state.best_candidate.iteration,
                "hypothesis_strategy": (
                    state.best_candidate.hypothesis.strategy
                    if hasattr(state.best_candidate.hypothesis, "strategy")
                    else "unknown"
                ),
                "prompt_changes": {},
            }
            if state.best_candidate.hypothesis.prompt_changes:
                for pred_name, change in state.best_candidate.hypothesis.prompt_changes.items():
                    best_program_data["prompt_changes"][pred_name] = {
                        "new_prompt": change.new_prompt,
                        "change_summary": (change.change_summary if hasattr(change, "change_summary") else ""),
                    }
            tracker.log_best_program(best_program_data)

    if is_enabled(Verbosity.DETAILED):
        total_candidates = sum(len(log_entry.candidates) for log_entry in state.iteration_logs)
        total_hypotheses = sum(len(log_entry.hypotheses) for log_entry in state.iteration_logs)
        log(
            f"APEX: Summary - evaluated {total_candidates} candidates from {total_hypotheses} hypotheses",
            Verbosity.DETAILED,
        )
        score_trajectory = [
            max((c.overall_score for c in log_entry.candidates), default=0.0) if log_entry.candidates else 0.0
            for log_entry in state.iteration_logs
        ]
        log(
            f"APEX: Best score trajectory across iterations: {score_trajectory}",
            Verbosity.DETAILED,
        )

    optimized_program = state.best_candidate.program
    optimized_program._compiled = True
    optimized_program.apex_result = ApexOptimizationResult(
        best_candidate=state.best_candidate,
        all_candidates=state.all_candidates,
        iterations=state.iteration_logs,
        stopped_after=state.stop_reason,
    )
    return optimized_program


__all__ = [
    "finalize_optimization",
    "initialize_new_state",
    "resume_from_checkpoint",
]
