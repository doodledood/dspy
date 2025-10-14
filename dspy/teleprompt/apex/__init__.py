"""APEX optimizer package."""
from dspy.teleprompt.apex.apex import APEX
from dspy.teleprompt.apex.models import (
    ApexCheckpoint,
    ApexIterationLog,
    ApexOptimizationResult,
    CandidateRecord,
    ChangeMagnitude,
    CheckpointConfig,
    ExecutionFlowEntry,
    HypothesisSpec,
    ProgramSnapshot,
    PromptChange,
    TrainExampleRecord,
)
from dspy.teleprompt.apex.signatures import (
    FailureAnalysisSignature,
    HypothesisGenerationSignature,
    SuccessAnalysisSignature,
)
from dspy.teleprompt.apex.tracker import ExperimentTracker
from dspy.teleprompt.apex.types import (
    ItemT,
    LogLevel,
    MetricFn,
    SamplerFn,
    TraceEntry,
    Verbosity,
    verbosity_rank,
)

__all__ = [
    "APEX",
    "ApexCheckpoint",
    "ApexIterationLog",
    "ApexOptimizationResult",
    "CandidateRecord",
    "ChangeMagnitude",
    "CheckpointConfig",
    "ExecutionFlowEntry",
    "ExperimentTracker",
    "FailureAnalysisSignature",
    "HypothesisGenerationSignature",
    "HypothesisSpec",
    "ItemT",
    "LogLevel",
    "MetricFn",
    "ProgramSnapshot",
    "PromptChange",
    "SamplerFn",
    "SuccessAnalysisSignature",
    "TraceEntry",
    "TrainExampleRecord",
    "Verbosity",
    "verbosity_rank",
]
