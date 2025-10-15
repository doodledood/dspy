from __future__ import annotations

import json
from pathlib import Path

import cloudpickle

from .models import (
    ApexCheckpoint,
    ApexIterationLog,
    CandidateRecord,
    CheckpointConfig,
)
from .runtime import RuntimeTools
from .types import Verbosity


class CheckpointManager:
    """Handles serialization and recovery of APEX checkpoints."""

    def __init__(self, directory: str | Path | None, runtime: RuntimeTools) -> None:
        self.runtime = runtime
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.runtime.log(
                f"APEX: Checkpointing enabled at {self.directory}",
                level=Verbosity.NORMAL,
            )

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def save(
        self,
        *,
        iteration: int,
        current_program,
        best_candidate: CandidateRecord,
        all_candidates: list[CandidateRecord],
        iteration_logs: list[ApexIterationLog],
        no_improvement_count: int,
        baseline_candidate: CandidateRecord,
        rng_state,
        config: CheckpointConfig,
    ) -> None:
        if not self.enabled:
            return

        checkpoint = ApexCheckpoint(
            iteration=iteration,
            current_program=current_program,
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iteration_logs=iteration_logs,
            no_improvement_count=no_improvement_count,
            baseline_candidate=baseline_candidate,
            rng_state=rng_state,
            config=config,
        )

        checkpoint_path = self.directory / f"checkpoint_iter_{iteration}.pkl"
        with open(checkpoint_path, "wb") as f:
            cloudpickle.dump(checkpoint, f)

        latest_path = self.directory / "latest_checkpoint.json"
        with open(latest_path, "w") as f:
            json.dump({"iteration": iteration, "checkpoint_file": checkpoint_path.name}, f)

        self.runtime.log(
            f"APEX: Saved checkpoint at iteration {iteration}",
            level=Verbosity.HIGH,
        )

    def load(self) -> ApexCheckpoint | None:
        if not self.enabled:
            return None

        latest_path = self.directory / "latest_checkpoint.json"
        if not latest_path.exists():
            return None

        with open(latest_path) as f:
            latest_info = json.load(f)

        checkpoint_path = self.directory / latest_info["checkpoint_file"]
        if not checkpoint_path.exists():
            self.runtime.log(
                f"APEX: Checkpoint file {checkpoint_path} not found",
                level=self.runtime.verbosity,
                log_level="warning",
            )
            return None

        with open(checkpoint_path, "rb") as f:
            checkpoint = cloudpickle.load(f)

        if not isinstance(checkpoint, ApexCheckpoint):
            raise TypeError(
                f"Invalid checkpoint type: expected ApexCheckpoint, got {type(checkpoint)}"
            )

        self.runtime.log(
            f"APEX: Loaded checkpoint from iteration {checkpoint.iteration}",
            level=Verbosity.NORMAL,
        )
        return checkpoint
