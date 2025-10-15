"""State tracking helpers for the APEX optimization loop."""

from __future__ import annotations

from dataclasses import dataclass

from dspy.primitives import Module

from .models import (
    ApexCheckpoint,
    ApexIterationLog,
    CandidateRecord,
)


@dataclass
class OptimizationState:
    """Mutable container for tracking optimization progress."""

    iteration: int
    current_program: Module
    baseline_candidate: CandidateRecord
    current_baseline_candidate: CandidateRecord
    best_candidate: CandidateRecord
    all_candidates: list[CandidateRecord]
    iteration_logs: list[ApexIterationLog]
    no_improvement_count: int
    initial_baseline: CandidateRecord
    stop_reason: str = ""

    @classmethod
    def initialize(cls, program: Module, baseline: CandidateRecord) -> OptimizationState:
        return cls(
            iteration=0,
            current_program=program,
            baseline_candidate=baseline,
            current_baseline_candidate=baseline,
            best_candidate=baseline,
            all_candidates=[baseline],
            iteration_logs=[],
            no_improvement_count=0,
            initial_baseline=baseline,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: ApexCheckpoint) -> OptimizationState:
        initial = checkpoint.all_candidates[0] if checkpoint.all_candidates else checkpoint.baseline_candidate
        return cls(
            iteration=checkpoint.iteration,
            current_program=checkpoint.current_program,
            baseline_candidate=checkpoint.baseline_candidate,
            current_baseline_candidate=checkpoint.baseline_candidate,
            best_candidate=checkpoint.best_candidate,
            all_candidates=list(checkpoint.all_candidates),
            iteration_logs=list(checkpoint.iteration_logs),
            no_improvement_count=checkpoint.no_improvement_count,
            initial_baseline=initial,
        )

    def start_next_iteration(self) -> int:
        """Increment and return the current iteration number."""

        self.iteration += 1
        return self.iteration


__all__ = ["OptimizationState"]
