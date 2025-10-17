"""Data models used by the APEX optimizer."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.primitives import Example, Module, Prediction


class ChangeMagnitude(str, Enum):
    """Magnitude of a prompt change."""

    MINIMAL = "minimal"
    MODERATE = "moderate"
    SUBSTANTIAL = "substantial"


class PromptChange(BaseModel):
    """A single prompt change for a predictor."""

    new_prompt: str = Field(description="Complete replacement text for the predictor's prompt")
    rationale: str = Field(description="Why this change fixes the identified issues")
    change_magnitude: ChangeMagnitude = Field(description="How significant this change is")


class HypothesisSpec(BaseModel):
    """Specification for a hypothesis to improve the program."""

    observation: str = Field(description="Synthesized description of patterns found across the targeted errors")
    fixable_root_causes: list[str] = Field(
        default_factory=list,
        description="Specific fixable issues that this hypothesis addresses through prompt changes (may be subset of all issues)",
    )
    non_fixable_root_causes: list[str] = Field(
        default_factory=list,
        description="Issues that cannot be fixed with prompt changes (e.g., 'needs retrieval system', 'requires multi-step architecture')",
    )
    impact_score: float = Field(
        default=0.0,
        description="Estimated impact score (0-1) based on volume and criticality of issues addressed",
    )
    generalizability_score: float = Field(
        default=0.0,
        description="How generalizable this hypothesis is (0-1) - will it help with future unseen examples?",
    )
    strategy: str = Field(description="Description of the approach this hypothesis takes. What makes it unique?")
    expected_impact: str = Field(
        description="Specific prediction of which errors this should fix and estimated success rate."
    )
    prompt_changes: dict[str, PromptChange] = Field(
        default_factory=dict,
        description="Mapping of predictor_name to PromptChange objects",
    )


class ExecutionFlowEntry(BaseModel):
    """Structured representation of a single predictor execution in the flow."""

    predictor_name: str
    predictor_type: str
    inputs: str
    outputs: str
    instructions: str
    dependencies: list[str] = Field(default_factory=list)
    input_sources: dict[str, list[str]] = Field(default_factory=dict)


class FailureSummaryRecord(BaseModel):
    """Compact failure summary provided to hypothesis generation."""

    root_cause: str = Field(description="Fundamental issue identified by FailureDetective")
    involved_predictors: list[str] = Field(
        default_factory=list,
        description="Predictors in causal order (primary failure first, then affected downstream)",
    )
    category: str = Field(description="Canonical category label for grouping similar failures")


class SuccessSummaryRecord(BaseModel):
    """Compact success summary provided to hypothesis generation."""

    root_cause: str = Field(description="Mechanism or pattern that produced the success")
    contributing_predictors: list[str] = Field(
        default_factory=list,
        description="Predictors responsible for the success pattern",
    )
    category: str = Field(description="Categorization of the success pattern")


class TrainExampleRecord(BaseModel):
    """Record of a single training example evaluation."""

    example: Example
    prediction: Prediction | None
    metric_score: float
    metric_feedback: str | None = None
    is_success: bool
    error: str | None = None
    execution_flow: list[ExecutionFlowEntry] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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


class CheckpointConfig(BaseModel):
    """Configuration saved in checkpoint."""

    max_iterations: int | None
    num_hypotheses: int
    num_eval_runs: int
    train_sample: int | None
    success_threshold: float
    min_metric: float
    max_metric: float
    convergence_patience: int | None
    seed: int


class ApexCheckpoint(BaseModel):
    """Checkpoint for resuming APEX optimization."""

    iteration: int
    current_program: Module
    best_candidate: CandidateRecord
    all_candidates: list[CandidateRecord]
    iteration_logs: list[ApexIterationLog]
    no_improvement_count: int
    baseline_candidate: CandidateRecord
    rng_state: Any
    config: CheckpointConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = [
    "ChangeMagnitude",
    "PromptChange",
    "HypothesisSpec",
    "ExecutionFlowEntry",
    "FailureSummaryRecord",
    "SuccessSummaryRecord",
    "TrainExampleRecord",
    "CandidateRecord",
    "ProgramSnapshot",
    "ApexIterationLog",
    "ApexOptimizationResult",
    "CheckpointConfig",
    "ApexCheckpoint",
]
