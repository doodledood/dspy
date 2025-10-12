# ruff: noqa: RUF002
from __future__ import annotations

import json
import logging
import os
import random
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence, TypeAlias, TypeVar

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

import dspy
from dspy.adapters import Adapter, JSONAdapter
from dspy.clients.lm import LM
from dspy.primitives import Example, Module, Prediction
from dspy.signatures import InputField, OutputField, Signature
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)

TraceEntry: TypeAlias = tuple[Any, Mapping[str, Any], Prediction]
LogLevel: TypeAlias = Literal["info", "warning", "debug", "error"]

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


class FailureAnalysisSignature(Signature):
    """Analyze a failure in a DSPy program to identify its root cause.

    Trace through the execution to find the ROOT CAUSE. The root cause may involve:
    - A single predictor's prompt being unclear, incomplete, or incorrect
    - Multiple predictors where an upstream predictor's output causes downstream failures
    - Interaction issues between predictors
    - Missing constraints or examples in prompts

    Be thorough but concise. Focus on actionable insights for fixing the prompt(s).
    """

    problem: str = InputField(desc="The problem statement or input to the program")
    prediction: str = InputField(desc="The model's actual prediction/output")
    expected: str = InputField(desc="The expected correct output")
    error: str = InputField(desc="Error message if execution failed", default="")
    execution_flow: str = InputField(desc="Program flow showing predictor relationships and instructions", default="")

    root_cause: str = OutputField(
        desc="Detailed description of what fundamentally caused this failure. "
        "Be specific about which predictor(s) and what aspect of their behavior caused the issue."
    )
    involved_predictors: list[str] = OutputField(
        desc="List of predictor names that contributed to the failure", default_factory=list
    )
    context: str = OutputField(
        desc="Relevant characteristics of this example that are important for understanding when/why this failure occurs. "
        "Include input characteristics, intermediate state issues, or patterns that would help generalize to similar failures."
    )
    category: str = OutputField(desc="Short label categorizing this failure type")
    key_details: str = OutputField(
        desc="Additional important information that would help someone design a fix. "
        "What specifically went wrong in the predictor's processing? What should have happened instead?"
    )


class SuccessAnalysisSignature(Signature):
    """Analyze a SUCCESS in a DSPy program to understand what worked well.

    This will be contrasted with failures to identify what differentiates successful executions.

    Focus on:
    - What aspects of the prompts guided correct behavior
    - How predictors handled this input well
    - What patterns in the execution led to success
    - What characteristics distinguish this from potential failures

    Be thorough but concise. Focus on actionable insights that contrast with failures.
    """

    problem: str = InputField(desc="The problem statement or input to the program")
    prediction: str = InputField(desc="The model's actual prediction/output")
    expected: str = InputField(desc="The expected correct output")
    execution_flow: str = InputField(desc="Program flow showing predictor relationships and instructions", default="")

    success_pattern: str = OutputField(
        desc="Clear description of what made this execution successful. What did the predictors do right?"
    )
    contributing_predictors: list[str] = OutputField(
        desc="List of predictors that worked well in this execution", default_factory=list
    )
    context: str = OutputField(
        desc="Relevant characteristics of this example that help explain the success. "
        "What about the input, intermediate outputs, or execution made this work?"
    )
    category: str = OutputField(desc="Short label for this success type")
    key_details: str = OutputField(
        desc="What specifically worked well? What aspects of the prompts or execution should be preserved or amplified?"
    )


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
    """Specification for a hypothesis to improve the program.

    Each hypothesis can address one or more fixable issues, prioritizing by impact.
    Different hypotheses may target different numbers of problems based on their generalizability.
    """

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


class HypothesisGenerationSignature(Signature):
    """# DSPy Hypothesis Generation Prompt

You are HypothesisEngine, a prompt optimization specialist for DSPy programs.

Core principle: The smallest change that solves the biggest problem wins.

## Operational Context

You’re part of an iterative optimizer (APEX) that:

- Samples different training examples each iteration
- Tests hypotheses on a separate validation set
- Keeps the baseline if no improvements found
- Continues until convergence or max iterations
- May be working with an already-partially-optimized program (not always starting from scratch)

This means: Focus on generalizable patterns, not overfitting to specific examples. Your hypotheses face real evaluation - they must actually work, not just sound good. Each hypothesis is evaluated multiple times, so changes must be consistently beneficial, not just occasionally helpful.

## Task

Generate hypotheses to improve a DSPy program based on failure and success patterns. Output a list of HypothesisSpec objects prioritizing minimal effective changes that preserve what works.

## Input Understanding

You receive string summaries (not raw data) from a SAMPLE of training examples:

- **failure_analyses**: Root causes and categories from failed examples in this iteration’s sample
- **success_analyses**: Patterns that worked well and must be preserved
- **program_flow**: Predictor dependencies forming a directed acyclic graph (DAG) of relationships
- **current_prompts**: Existing predictor prompts that may need modification

Key insight: A predictor might succeed on some inputs and fail on others. Look for consistent patterns, not one-off issues. Use categories to group related failures for more effective targeting.

## Understanding Program Flow

From program_flow, understand:

- Graph structure: Shows a directed acyclic graph of predictor dependencies from inputs through intermediate nodes to outputs
- Topology: Edges indicate data flow; a node may have multiple parents or children. Respect the DAG when reasoning about impacts
- Execution order: Consider valid topological orders when coordinating changes across branches
- Cascade potential: Changes to upstream predictors propagate along all outgoing edges and can affect multiple downstream branches
- Bottlenecks: Identify hub predictors whose outputs feed many successors—they are high-leverage intervention points

This helps identify when a MINIMAL fix suffices vs when MODERATE coordinated changes are needed.

## Hypothesis Generation Strategy

Choose approach based on failure patterns:

**Single Dominant Pattern**
When one root cause appears repeatedly across the sample:
→ MINIMAL hypothesis: Add single constraint/example/clarification
Example: “Missing format specification” → Add JSON schema
Note: If this pattern represents most failures, fixing it alone may be sufficient

**Multiple Related Failures**
When several issues share underlying cause:
→ TARGETED hypothesis: Fix root cause with small coordinated changes
Example: “Ambiguous terminology” across predictors → Standardize terms
Note: More efficient than fixing each individually

**Cascade Failures** (Check program_flow carefully)
When upstream errors cause downstream problems:
→ MODERATE hypothesis: Align dependent predictors
Example: Extractor output incompatible with Validator → Fix both
Note: Must fix source AND affected predictors together

**Fundamental Issues**
When core approach flawed (use sparingly):
→ SUBSTANTIAL hypothesis: Restructure while preserving working elements
Only when patterns show no smaller fix possible
Note: High risk - only if confident no alternative exists

## Success Preservation

From success_analyses, identify patterns that work. When generating hypotheses:

1. Note which predictors/approaches succeed
1. Ensure changes don’t contradict successful patterns
1. If conflict exists, find alternative approach or skip
1. In rationale, state what successful patterns are preserved

Key: We see pattern summaries, not specific instructions, so preserve general approaches that work.

## Output Format

Return list of HypothesisSpec objects (maximum num_hypotheses):

```json
{
  "observation": "Pattern identified from failures",
  "fixable_root_causes": ["Issues addressable via prompts"],
  "non_fixable_root_causes": ["Issues needing architecture changes"],
  "impact_score": 0.0-1.0,
  "generalizability_score": 0.0-1.0,
  "strategy": "Approach description",
  "expected_impact": "Specific, testable prediction (e.g., 'Eliminates JSON parsing errors in 30% of cases' not 'should work better')",
  "prompt_changes": {
    "PredictorName": {
      "new_prompt": "COMPLETE replacement text",
      "rationale": "Why this fixes issue + what's preserved",
      "change_magnitude": "MINIMAL|MODERATE|SUBSTANTIAL"
    }
  }
}
```

**Critical Requirements:**

- PredictorName must EXACTLY match names from current_prompts
- new_prompt is COMPLETE replacement (all original + changes)
- Sort by impact_score descending, then generalizability_score
- change_magnitude must be exactly: MINIMAL, MODERATE, or SUBSTANTIAL

## Scoring Guidelines

**impact_score**: How many failures will this address?
Count the actual failure patterns mentioned:

- High (0.7-1.0): Addresses the most frequently mentioned root cause OR multiple related causes
- Medium (0.4-0.7): Addresses a moderately frequent cause OR several minor ones
- Low (0.0-0.4): Addresses only rarely mentioned causes

Concrete approach: If a root cause appears in many failure summaries, score it higher. Count mentions.

**generalizability_score**: Will this prevent future similar errors?
Assess the breadth of the fix:

- High (0.7-1.0): Adds systematic constraint (e.g., format spec fixes ALL format errors)
- Medium (0.4-0.7): Fixes specific cases but pattern may vary (e.g., one ambiguous term)
- Low (0.0-0.4): Very specific to exact scenario

**Important**: Scores are for sorting hypotheses - relative ordering matters more than exact values. Be consistent across hypotheses rather than perfect on absolute values.

## Risk Management

Since the optimizer keeps baseline if no improvement:

- Prefer high-confidence small changes over ambitious rewrites
- Conservative fixes that definitely work beat risky comprehensive changes
- When uncertain between approaches, choose the smaller change
- Remember: You compete against a working baseline

Convergence mindset:

- Small consistent improvements accumulate over iterations
- Even 5-10% improvement per iteration leads to convergence
- Maintaining performance while simplifying code is valuable
- The goal is steady progress, not perfection in one shot

## Hypothesis Diversity

If num_hypotheses > 1:

1. First: Most confident fix for biggest problem
1. Additional: Different approaches (different predictors, fix types, or scopes)
1. Never generate minor variations of same fix

Diversity matters because:

- Each hypothesis gets evaluated separately on validation set
- Different approaches help explore solution space
- Future iterations will see different training samples
- Diverse hypotheses provide more learning signal

## What Can/Cannot Be Fixed

**Fixable via prompts:**

- Missing/unclear instructions
- Format specifications
- Ambiguous language
- Missing examples
- Inconsistent terminology

**Not fixable (need architecture):**

- Missing data/tools
- Model limitations
- Need different program flow
- Data quality issues

## When to Return Empty List

Return [] if:

- No clear patterns in failures (just random errors across sample)
- All issues need architecture changes
- Fixes would likely break successful patterns
- Very low confidence in proposed changes
- Errors appear sample-specific rather than generalizable

Better to return [] than low-quality hypotheses that won’t survive validation.

## Success Preservation

From success_analyses, identify patterns that work. When generating hypotheses:

1. Note which predictors/approaches succeed
1. Ensure changes don’t contradict successful patterns
1. If conflict exists, find alternative approach or skip
1. In rationale, explicitly state what successful patterns are preserved

Remember: Successful patterns in the sample likely generalize to validation set. Breaking them risks degrading overall performance even if training errors decrease.

## Example Hypothesis

```json
{
  "observation": "JSON format errors dominate failures while extraction logic succeeds",
  "fixable_root_causes": ["Missing JSON format specification"],
  "non_fixable_root_causes": [],
  "impact_score": 0.85,
  "generalizability_score": 0.9,
  "strategy": "Add format specification without changing extraction logic",
  "expected_impact": "Eliminate JSON parsing errors affecting 35% of cases",
  "prompt_changes": {
    "ExtractorPredictor": {
      "new_prompt": "Extract key information from the provided text.\n\nRequirements:\n- Identify main entities and relationships\n- Preserve numerical data exactly\n- Include confidence scores\n\nOutput MUST be valid JSON:\n{\n  \"entities\": [...],\n  \"relationships\": [...],\n  \"confidence\": 0.0-1.0\n}\n\nFormat rules:\n- Use double quotes for strings\n- No trailing commas\n- Numbers without quotes",
      "rationale": "Adds format spec to fix parsing. Preserves successful extraction approach.",
      "change_magnitude": "MINIMAL"
    }
  }
}
```

## Key Principles

1. **Minimal effective change** - Smallest fix that solves the problem
1. **Preserve success** - Don’t modify what works
1. **Complete replacements** - new_prompt contains everything
1. **Pattern-based** - Work from summaries, not detailed instructions
1. **Testable impact** - Clear, measurable predictions

Remember: You’re working with pattern summaries. Focus on fixing clear problems while preserving successful approaches. Conservative improvements beat risky rewrites."""

    failure_analyses: str = InputField(
        desc="Root cause summaries from failure analyses, showing patterns and issues to fix"
    )
    success_analyses: str = InputField(
        desc="Success pattern summaries for contrast, showing what works well and should be preserved"
    )
    program_flow: str = InputField(
        desc="Program structure showing predictor relationships as a directed acyclic graph"
    )
    current_prompts: str = InputField(desc="Current predictor prompts in the program that may need modification")
    num_hypotheses: int = InputField(desc="Maximum number of hypotheses to generate (ordered by impact)")

    hypotheses: list[HypothesisSpec] = OutputField(
        desc="List of improvement hypotheses ordered by impact_score (highest first). "
        "Each may address different numbers of issues based on impact/generalizability tradeoffs. "
        "May be empty if no actionable improvements found. Limited to num_hypotheses."
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
    train_sample: int | None  # Only save if it's an int
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
    rng_state: Any  # Random state is opaque
    config: CheckpointConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)


class APEX(Teleprompter):
    """APEX teleprompter implementing systematic prompt optimization."""

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
        train_sample: None | int | SamplerFn = None,
        success_threshold: float | None = None,
        min_metric: float = 0.0,
        max_metric: float = 1.0,
        convergence_patience: int | None = 3,
        seed: int | None = None,
        checkpoint_dir: str | Path | None = None,
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

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"APEX: Checkpointing enabled at {self.checkpoint_dir}", Verbosity.NORMAL)

    def _is_enabled(self, level: Verbosity) -> bool:
        return _verbosity_rank(self.verbosity) >= _verbosity_rank(level)

    def _log(self, message: str, level: Verbosity = Verbosity.NORMAL, log_level: LogLevel = "info") -> None:
        """Log a message if verbosity level permits.

        Args:
            message: The message to log
            level: The verbosity level required to show this message
            log_level: The logging level to use (strongly typed: info, warning, debug, error)
        """
        if self._is_enabled(level):
            if log_level == "warning":
                logger.warning(message)
            elif log_level == "debug":
                logger.debug(message)
            elif log_level == "error":
                logger.error(message)
            else:
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

    def _save_checkpoint(
        self,
        iteration: int,
        current_program: Module,
        best_candidate: CandidateRecord,
        all_candidates: list[CandidateRecord],
        iteration_logs: list[ApexIterationLog],
        no_improvement_count: int,
        baseline_candidate: CandidateRecord,
    ) -> None:
        """Save checkpoint to disk."""
        if not self.checkpoint_dir:
            return

        config = CheckpointConfig(
            max_iterations=self.max_iterations,
            num_hypotheses=self.num_hypotheses,
            num_eval_runs=self.num_eval_runs,
            train_sample=self.train_sample if isinstance(self.train_sample, int) else None,
            success_threshold=self.success_threshold,
            min_metric=self.min_metric,
            max_metric=self.max_metric,
            convergence_patience=self.convergence_patience,
            seed=self.seed,
        )

        checkpoint = ApexCheckpoint(
            iteration=iteration,
            current_program=current_program,
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iteration_logs=iteration_logs,
            no_improvement_count=no_improvement_count,
            baseline_candidate=baseline_candidate,
            rng_state=self._rng.getstate(),
            config=config,
        )

        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        with open(checkpoint_path, "wb") as f:
            cloudpickle.dump(checkpoint, f)

        latest_path = self.checkpoint_dir / "latest_checkpoint.json"
        with open(latest_path, "w") as f:
            json.dump({"iteration": iteration, "checkpoint_file": f"checkpoint_iter_{iteration}.pkl"}, f)

        self._log(f"APEX: Saved checkpoint at iteration {iteration}", Verbosity.HIGH)

    def _load_checkpoint(self) -> ApexCheckpoint | None:
        """Load the latest checkpoint if it exists."""
        if not self.checkpoint_dir:
            return None

        latest_path = self.checkpoint_dir / "latest_checkpoint.json"
        if not latest_path.exists():
            return None

        with open(latest_path) as f:
            latest_info = json.load(f)

        checkpoint_path = self.checkpoint_dir / latest_info["checkpoint_file"]
        if not checkpoint_path.exists():
            self._log(f"APEX: Checkpoint file {checkpoint_path} not found", Verbosity.HIGH, "warning")
            return None

        with open(checkpoint_path, "rb") as f:
            checkpoint = cloudpickle.load(f)

        if not isinstance(checkpoint, ApexCheckpoint):
            raise TypeError(f"Invalid checkpoint type: expected ApexCheckpoint, got {type(checkpoint)}")

        self._log(f"APEX: Loaded checkpoint from iteration {checkpoint.iteration}", Verbosity.NORMAL)
        return checkpoint

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
        resume: bool = False,
    ) -> Module:
        if teacher is not None:
            raise ValueError("APEX does not support teacher programs.")
        if not trainset:
            raise ValueError("trainset must be non-empty.")
        if not valset:
            raise ValueError("calibration set (valset) must be provided and non-empty.")

        checkpoint = None
        if resume and self.checkpoint_dir:
            checkpoint = self._load_checkpoint()

        if checkpoint:
            current_program = checkpoint.current_program
            all_candidates = checkpoint.all_candidates
            iteration_logs = checkpoint.iteration_logs
            best_candidate = checkpoint.best_candidate
            baseline_candidate = checkpoint.baseline_candidate
            current_baseline_candidate = checkpoint.baseline_candidate
            no_improvement_count = checkpoint.no_improvement_count
            iteration = checkpoint.iteration
            self._rng.setstate(checkpoint.rng_state)
            self._log(f"APEX: Resuming from iteration {iteration}", Verbosity.NORMAL)
        else:
            current_program = student.deepcopy()
            assert not getattr(current_program, "_compiled", False), "Student must be uncompiled."

            all_candidates: list[CandidateRecord] = []
            iteration_logs: list[ApexIterationLog] = []

            self._log(f"APEX: running with num_threads={self.num_threads}", Verbosity.NORMAL)
            max_iter_str = f"{self.max_iterations}" if self.max_iterations is not None else "until convergence"
            patience_str = f"{self.convergence_patience}" if self.convergence_patience is not None else "disabled"
            self._log(
                f"APEX: Configuration - max_iterations={max_iter_str}, num_hypotheses={self.num_hypotheses}, "
                f"success_threshold={self.success_threshold:.2f}, convergence_patience={patience_str}",
                Verbosity.HIGH,
            )
            self._log(f"APEX: Using seed={self.seed} for reproducibility", Verbosity.HIGH)

            self._log("APEX: Evaluating initial baseline on validation set", Verbosity.NORMAL)
            baseline_candidate = self._evaluate_candidate(
                program=current_program.deepcopy(),
                calset=valset,
                iteration=0,
                hypothesis=None,
            )
            all_candidates.append(baseline_candidate)
            best_candidate = baseline_candidate
            current_baseline_candidate = baseline_candidate
            self._log(f"APEX: Initial baseline score={baseline_candidate.overall_score:.4f}", Verbosity.NORMAL)

            no_improvement_count = 0
            iteration = 0

            self._save_checkpoint(
                iteration=0,
                current_program=current_program,
                best_candidate=best_candidate,
                all_candidates=all_candidates,
                iteration_logs=iteration_logs,
                no_improvement_count=no_improvement_count,
                baseline_candidate=baseline_candidate,
            )

        stop_reason = ""

        try:
            while True:
                iteration += 1

                if self.max_iterations is not None and iteration > self.max_iterations:
                    stop_reason = "max_iterations"
                    self._log("APEX: Stopping due to max iterations reached", Verbosity.NORMAL)
                    break
                sampled_train = self._sample_trainset(trainset, iteration)
                self._log(
                    f"APEX: iteration {iteration} started (train sample={len(sampled_train)}, val size={len(valset)})",
                    Verbosity.NORMAL,
                )
                self._log(
                    f"APEX: Sampled {len(sampled_train)} training examples from {len(trainset)} total", Verbosity.HIGH
                )
                baseline_for_analysis = current_program.deepcopy()
                snapshot = self._snapshot_program(baseline_for_analysis)

                failures, successes = self._evaluate_train_examples(baseline_for_analysis, sampled_train)
                self._log(
                    f"APEX: Train evaluation complete - {len(failures)} failures, {len(successes)} successes",
                    Verbosity.HIGH,
                )

                if not failures:
                    self._log(
                        f"APEX: iteration {iteration} - No failures found! All examples succeeded. Skipping to next iteration.",
                        Verbosity.NORMAL,
                    )
                    iteration_logs.append(
                        ApexIterationLog(
                            iteration=iteration,
                            sampled_train_size=len(sampled_train),
                            num_failures=0,
                            num_successes=len(successes),
                            hypotheses=[],
                            candidates=[],
                        )
                    )
                    no_improvement_count += 1
                    if self.convergence_patience is not None:
                        if no_improvement_count >= self.convergence_patience:
                            stop_reason = "patience"
                            self._log(
                                "APEX: Stopping due to convergence patience reached (all successes)", Verbosity.NORMAL
                            )
                            break
                    self._save_checkpoint(
                        iteration=iteration,
                        current_program=current_program,
                        best_candidate=best_candidate,
                        all_candidates=all_candidates,
                        iteration_logs=iteration_logs,
                        no_improvement_count=no_improvement_count,
                        baseline_candidate=baseline_candidate,
                    )
                    continue

                failure_summaries = self._analyze_examples(failures, mode="failure")
                success_summaries = self._analyze_successes(successes, failure_count=len(failure_summaries))
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
                if hypotheses and self._is_enabled(Verbosity.NORMAL):
                    for idx, h in enumerate(hypotheses, start=1):
                        predictors_updated = list(h.prompt_changes.keys()) if h.prompt_changes else []
                        self._log(
                            f"APEX: hypothesis #{idx} - strategy: {h.strategy}, impact: {h.impact_score:.2f}, "
                            f"updating: {', '.join(predictors_updated) if predictors_updated else 'no predictors'}",
                            Verbosity.NORMAL,
                        )
                if self._is_enabled(Verbosity.HIGH) and hypotheses:
                    self._log(
                        "APEX: Detailed hypothesis info follows...",
                        Verbosity.HIGH,
                    )

                candidates = self._evaluate_candidates(
                    baseline=current_program,
                    hypotheses=hypotheses,
                    calset=valset,
                    iteration=iteration,
                    cached_baseline=current_baseline_candidate if iteration > 1 else None,
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

                if best_candidate_for_iteration.overall_score > best_candidate.overall_score:
                    best_candidate = best_candidate_for_iteration
                    self._log(
                        f"APEX: New best candidate found with score {best_candidate.overall_score:.4f}",
                        Verbosity.NORMAL,
                    )
                    if best_candidate.hypothesis and best_candidate.hypothesis.prompt_changes:
                        self._log(
                            f"APEX: Improved {len(best_candidate.hypothesis.prompt_changes)} predictor prompt(s) - strategy: {best_candidate.hypothesis.strategy}",
                            Verbosity.NORMAL,
                        )
                        if self._is_enabled(Verbosity.HIGH):
                            self._log(
                                "APEX: Detailed improved prompts:",
                                Verbosity.HIGH,
                            )
                            for predictor_name, changes in best_candidate.hypothesis.prompt_changes.items():
                                self._log(
                                    f"  → {predictor_name}: {changes.new_prompt[:300]}..."
                                    if len(changes.new_prompt) > 300
                                    else f"  → {predictor_name}: {changes.new_prompt}",
                                    Verbosity.HIGH,
                                )

                baseline_candidate = candidates[0]
                self._log(
                    f"APEX: iteration {iteration} best score={best_candidate_for_iteration.overall_score:.4f}",
                    Verbosity.NORMAL,
                )

                if self._is_enabled(Verbosity.HIGH):
                    score_improvements = [c.overall_score - baseline_candidate.overall_score for c in candidates[1:]]
                    if score_improvements:
                        self._log(
                            f"APEX: Score improvements from baseline: {score_improvements}",
                            Verbosity.HIGH,
                        )

                if best_candidate_for_iteration is baseline_candidate:
                    no_improvement_count += 1
                    if self.convergence_patience is not None:
                        self._log(
                            f"APEX: No improvement ({no_improvement_count}/{self.convergence_patience} patience)",
                            Verbosity.HIGH,
                        )
                        if no_improvement_count >= self.convergence_patience:
                            stop_reason = "patience"
                            self._log("APEX: Stopping due to convergence patience reached", Verbosity.NORMAL)
                            break
                    else:
                        self._log(
                            f"APEX: No improvement in iteration {iteration} (patience disabled)",
                            Verbosity.HIGH,
                        )
                else:
                    no_improvement_count = 0
                    current_program = best_candidate_for_iteration.program
                    current_baseline_candidate = best_candidate_for_iteration
                    self._log(
                        "APEX: Updating program with hypothesis improvements",
                        Verbosity.HIGH,
                    )

                self._save_checkpoint(
                    iteration=iteration,
                    current_program=current_program,
                    best_candidate=best_candidate,
                    all_candidates=all_candidates,
                    iteration_logs=iteration_logs,
                    no_improvement_count=no_improvement_count,
                    baseline_candidate=baseline_candidate,
                )
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            self._log("APEX: Optimization interrupted by user (Ctrl+C)", Verbosity.NORMAL)

            if "candidates" in locals() and candidates:
                iteration_logs.append(
                    ApexIterationLog(
                        iteration=iteration,
                        sampled_train_size=len(sampled_train) if "sampled_train" in locals() else 0,
                        num_failures=len(failure_summaries) if "failure_summaries" in locals() else 0,
                        num_successes=len(success_summaries) if "success_summaries" in locals() else 0,
                        hypotheses=hypotheses if "hypotheses" in locals() else [],
                        candidates=candidates,
                    )
                )

            if self.checkpoint_dir:
                self._save_checkpoint(
                    iteration=iteration,
                    current_program=current_program,
                    best_candidate=best_candidate,
                    all_candidates=all_candidates,
                    iteration_logs=iteration_logs,
                    no_improvement_count=no_improvement_count,
                    baseline_candidate=baseline_candidate if "baseline_candidate" in locals() else best_candidate,
                )
                self._log(
                    f"APEX: Checkpoint saved at iteration {iteration} - resume with resume=True", Verbosity.NORMAL
                )

        self._log(
            f"APEX: Optimization complete - stopped after {len(iteration_logs)} iterations ({stop_reason})",
            Verbosity.NORMAL,
        )
        self._log(
            f"APEX: Final score: {best_candidate.overall_score:.4f} (initial baseline: {baseline_candidate.overall_score:.4f})",
            Verbosity.NORMAL,
        )
        if self._is_enabled(Verbosity.HIGH):
            total_candidates = sum(len(log.candidates) for log in iteration_logs)
            total_hypotheses = sum(len(log.hypotheses) for log in iteration_logs)
            self._log(
                f"APEX: Summary - evaluated {total_candidates} candidates from {total_hypotheses} hypotheses",
                Verbosity.HIGH,
            )
            score_trajectory = [
                max(c.overall_score for c in log.candidates) if log.candidates else 0.0 for log in iteration_logs
            ]
            self._log(
                f"APEX: Best score trajectory across iterations: {score_trajectory}",
                Verbosity.HIGH,
            )

        optimized_program = best_candidate.program
        optimized_program._compiled = True
        optimized_program.apex_result = ApexOptimizationResult(
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iterations=iteration_logs,
            stopped_after=stop_reason,
        )
        return optimized_program

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
    ) -> tuple[list[TrainExampleRecord], list[TrainExampleRecord]]:
        failure_records: list[TrainExampleRecord] = []
        success_records: list[TrainExampleRecord] = []

        examples = list(trainset)

        def process(example: Example) -> TrainExampleRecord:
            return self._run_single_example(program, example)

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
    ) -> TrainExampleRecord:
        input_kwargs = example.inputs().toDict()

        prediction_obj: Prediction | None = None
        error_message: str | None = None
        raw_trace: list[TraceEntry] = []

        with dspy.settings.context(trace=[]):
            try:
                prediction_obj = program(**input_kwargs)
            except Exception as exc:
                self._log(f"APEX: Program execution failed on example: {str(exc)[:200]}", Verbosity.HIGH, "warning")
                error_message = f"execution_error: {exc}"

        raw_trace = list(dspy.settings.trace or [])
        execution_flow = self._extract_execution_flow(raw_trace, program)

        metric_score = self.min_metric
        metric_feedback: str | None = None
        try:
            if prediction_obj is not None:
                metric_score, metric_feedback = self._evaluate_metric(example, prediction_obj, raw_trace)
            else:
                metric_score = self.min_metric
                if error_message:
                    self._log(
                        f"APEX: No prediction to evaluate due to error: {error_message[:100]}", Verbosity.HIGH, "debug"
                    )
        except Exception as exc:
            self._log(f"APEX: Metric evaluation failed: {str(exc)[:200]}", Verbosity.HIGH, "warning")
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

    def _extract_execution_flow(
        self,
        trace: list[TraceEntry],
        program: Module,
    ) -> list[ExecutionFlowEntry]:
        """Extract structured execution flow from raw trace."""
        execution_flow: list[ExecutionFlowEntry] = []
        predictor_lookup = dict(program.named_predictors())

        def normalize_value(value: Any) -> Any:
            if isinstance(value, Prediction | Example):
                return {k: normalize_value(v) for k, v in value.toDict().items()}
            if isinstance(value, dict):
                return {str(k): normalize_value(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize_value(v) for v in value]
            if isinstance(value, set):
                return sorted(normalize_value(v) for v in value)
            return value

        def value_key(value: Any) -> str:
            normalized = normalize_value(value)
            try:
                return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
            except TypeError:
                return repr(normalized)

        def stringify(value: Any) -> str:
            normalized = normalize_value(value)
            try:
                return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
            except TypeError:
                return repr(normalized)

        value_sources_by_field: dict[str, list[str]] = {}
        value_sources_by_value: dict[str, list[str]] = {}

        for predictor_obj, inputs, outputs in trace:
            predictor_name = "unknown"
            predictor_type = type(predictor_obj).__name__

            for name, pred in predictor_lookup.items():
                if pred is predictor_obj:
                    predictor_name = name
                    break

            instructions = ""
            if hasattr(predictor_obj, "signature") and hasattr(predictor_obj.signature, "instructions"):
                instructions = predictor_obj.signature.instructions

            normalized_inputs = {str(k): normalize_value(v) for k, v in dict(inputs).items()}

            normalized_outputs: dict[str, Any] = {}
            if isinstance(outputs, Prediction | Example):
                normalized_outputs = {
                    str(k): normalize_value(v)
                    for k, v in outputs.toDict().items()
                }
            elif outputs is not None:
                normalized_outputs = {"value": normalize_value(outputs)}

            dependencies: set[str] = set()
            input_sources: dict[str, list[str]] = {}

            for input_name, input_value in normalized_inputs.items():
                key = value_key(input_value)
                field_key = f"{input_name}::{key}"
                source_candidates: list[str] = []

                if field_key in value_sources_by_field and value_sources_by_field[field_key]:
                    source_candidates = [value_sources_by_field[field_key][-1]]
                elif key in value_sources_by_value and value_sources_by_value[key]:
                    source_candidates = [value_sources_by_value[key][-1]]

                if source_candidates:
                    unique_sources = list(dict.fromkeys(source_candidates))
                    dependencies.update(unique_sources)
                    input_sources[input_name] = unique_sources

            execution_flow.append(
                ExecutionFlowEntry(
                    predictor_name=predictor_name,
                    predictor_type=predictor_type,
                    inputs=stringify(normalized_inputs),
                    outputs=stringify(normalized_outputs) if normalized_outputs else "{}",
                    instructions=instructions,
                    dependencies=sorted(dependencies),
                    input_sources={k: sorted(v) for k, v in sorted(input_sources.items())},
                )
            )

            for output_name, output_value in normalized_outputs.items():
                key = value_key(output_value)
                field_key = f"{output_name}::{key}"
                value_sources_by_field.setdefault(field_key, []).append(predictor_name)
                value_sources_by_value.setdefault(key, []).append(predictor_name)

        return execution_flow

    def _format_execution_flow_as_graph(self, execution_flow: list[ExecutionFlowEntry]) -> str:
        """Format execution flow as a relationship graph showing predictor dependencies."""
        if not execution_flow:
            return "No execution flow available"

        if len(execution_flow) == 1:
            entry = execution_flow[0]
            return (
                "Program DAG:\n"
                "  Input\n"
                f"    ↳ {entry.predictor_name}\n"
                f"  {entry.predictor_name} ({entry.predictor_type})\n"
                "    depends on: Input\n"
                "    feeds: Output"
            )

        flow_lines: list[str] = ["Program DAG:"]

        children_map: dict[str, set[str]] = {entry.predictor_name: set() for entry in execution_flow}
        root_nodes: list[str] = []

        for entry in execution_flow:
            if entry.dependencies:
                for dependency in entry.dependencies:
                    children_map.setdefault(dependency, set()).add(entry.predictor_name)
            else:
                root_nodes.append(entry.predictor_name)

        if root_nodes:
            flow_lines.append("  Input")
            flow_lines.append(f"    ↳ {', '.join(sorted(root_nodes))}")
        else:
            flow_lines.append("  Input (no predictors depend directly on program input)")

        for entry in execution_flow:
            flow_lines.append(f"  {entry.predictor_name} ({entry.predictor_type})")
            if entry.dependencies:
                flow_lines.append(f"    depends on: {', '.join(entry.dependencies)}")
            else:
                flow_lines.append("    depends on: Input")

            children = sorted(children_map.get(entry.predictor_name, set()))
            if children:
                flow_lines.append(f"    feeds: {', '.join(children)}")
            else:
                flow_lines.append("    feeds: Output")

        return "\n".join(flow_lines)

    def _format_execution_flow_with_details(self, execution_flow: list[ExecutionFlowEntry]) -> str:
        """Format execution flow with instructions but without I/O values for analysis."""
        if not execution_flow:
            return "No execution flow available"

        flow_parts: list[str] = []

        flow_parts.append(self._format_execution_flow_as_graph(execution_flow))
        flow_parts.append("\nPredictor Instructions:")

        for idx, entry in enumerate(execution_flow, start=1):
            instructions = entry.instructions if entry.instructions else "No instructions"
            dependencies = ", ".join(entry.dependencies) if entry.dependencies else "Input"
            flow_parts.append(
                f"\n{idx}. {entry.predictor_name} ({entry.predictor_type}):\n"
                f"   Instructions: {instructions}\n"
                f"   Depends on: {dependencies}"
            )

            if entry.input_sources:
                flow_parts.append("   Inputs sourced from:")
                for input_name, sources in entry.input_sources.items():
                    flow_parts.append(f"     - {input_name}: {', '.join(sources)}")
            else:
                flow_parts.append("   Inputs sourced from: program input or constants")

        return "\n".join(flow_parts)

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

    def _analyze_examples(
        self,
        records: list[TrainExampleRecord],
        mode: str,
    ) -> list[Prediction]:
        if not records:
            return []

        signature_class = FailureAnalysisSignature if mode == "failure" else SuccessAnalysisSignature
        analysis_lm = self.analysis_lm
        analysis_adapter = self.analysis_adapter

        def process(record: TrainExampleRecord) -> Prediction:
            with dspy.context(lm=analysis_lm, adapter=analysis_adapter):
                predictor = dspy.Predict(signature_class)

                inputs = record.example.inputs().toDict()
                expected = record.example.labels().toDict()

                execution_flow_str = self._format_execution_flow_with_details(record.execution_flow)

                if mode == "failure":
                    result = predictor(
                        problem=str(inputs),
                        prediction=str(record.prediction) if record.prediction else "",
                        expected=str(expected),
                        error=record.error or "",
                        execution_flow=execution_flow_str,
                    )
                else:
                    result = predictor(
                        problem=str(inputs),
                        prediction=str(record.prediction) if record.prediction else "",
                        expected=str(expected),
                        execution_flow=execution_flow_str,
                    )

            return result

        analyses = self._parallel_execute(
            records,
            process,
            description=f"APEX: analyzing {mode}s",
            level=Verbosity.HIGH,
        )

        if self._is_enabled(Verbosity.HIGH):
            for index, analysis in enumerate(analyses, start=1):
                if mode == "failure":
                    self._log(
                        f"APEX: failure analysis #{index} ({analysis.category}) → {analysis.root_cause}",
                        Verbosity.HIGH,
                    )
                else:
                    self._log(
                        f"APEX: success analysis #{index} ({analysis.category}) → {analysis.success_pattern}",
                        Verbosity.HIGH,
                    )
        return analyses

    def _analyze_successes(
        self,
        success_records: list[TrainExampleRecord],
        failure_count: int,
    ) -> list[Prediction]:
        if not success_records or failure_count == 0:
            return []
        if len(success_records) > failure_count:
            success_records = self._rng.sample(success_records, k=failure_count)
        return self._analyze_examples(success_records, mode="success")

    def _generate_hypotheses(
        self,
        *,
        failure_summaries: list[Prediction],
        success_summaries: list[Prediction],
        snapshot: ProgramSnapshot,
    ) -> list[HypothesisSpec]:
        if not failure_summaries or self.num_hypotheses == 0:
            self._log("APEX: No hypotheses to generate (no failures or num_hypotheses=0)", Verbosity.HIGH)
            return []

        shuffled_failures = list(failure_summaries)
        self._rng.shuffle(shuffled_failures)

        self._log(
            f"APEX: Generating up to {self.num_hypotheses} hypotheses from {len(failure_summaries)} failures",
            Verbosity.HIGH,
        )

        failure_text = "\n".join([f"- {f.root_cause} (category: {f.category})" for f in failure_summaries])
        success_text = (
            "\n".join([f"- {s.success_pattern} (category: {s.category})" for s in success_summaries])
            if success_summaries
            else "No success patterns available"
        )

        prompt_text = "\n".join(
            [
                f"- {name}: {prompt[:200]}..." if len(prompt) > 200 else f"- {name}: {prompt}"
                for name, prompt in snapshot.prompts.items()
            ]
        )

        program_flow = snapshot.flow_description

        with dspy.context(lm=self.hypothesis_lm, adapter=self.hypothesis_adapter):
            predictor = dspy.Predict(HypothesisGenerationSignature)
            result = predictor(
                failure_analyses=failure_text,
                success_analyses=success_text,
                program_flow=program_flow,
                current_prompts=prompt_text,
                num_hypotheses=self.num_hypotheses,
            )

        validated_specs = result.hypotheses if result.hypotheses else []

        # Sort by impact_score (highest first), then by generalizability_score if tied
        validated_specs.sort(key=lambda h: (h.impact_score, h.generalizability_score), reverse=True)

        validated_specs = validated_specs[: self.num_hypotheses]

        if self._is_enabled(Verbosity.HIGH):
            for idx, spec in enumerate(validated_specs, start=1):
                self._log(
                    f"APEX: hypothesis #{idx} ({spec.strategy}) targeting {', '.join(spec.fixable_root_causes) or 'no fixable causes'} "
                    f"[impact={spec.impact_score:.2f}, generalizability={spec.generalizability_score:.2f}]",
                    Verbosity.HIGH,
                )
                for predictor_name, changes in spec.prompt_changes.items():
                    self._log(
                        f"  → {predictor_name}: {changes.new_prompt[:200]}..."
                        if len(changes.new_prompt) > 200
                        else f"  → {predictor_name}: {changes.new_prompt}",
                        Verbosity.HIGH,
                    )
                    if changes.rationale:
                        self._log(
                            f"     Rationale: {changes.rationale}",
                            Verbosity.HIGH,
                        )
        return validated_specs

    def _evaluate_candidates(
        self,
        *,
        baseline: Module,
        hypotheses: list[HypothesisSpec],
        calset: list[Example],
        iteration: int,
        cached_baseline: CandidateRecord | None = None,
    ) -> list[CandidateRecord]:
        candidates: list[CandidateRecord] = []

        if cached_baseline:
            baseline_record = CandidateRecord(
                program=baseline,
                overall_score=cached_baseline.overall_score,
                per_example_scores=cached_baseline.per_example_scores,
                iteration=iteration,
                hypothesis=None,
            )
        else:
            baseline_record = self._evaluate_candidate(
                program=baseline.deepcopy(),
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
            try:
                per_runs: list[float] = []
                for _ in range(self.num_eval_runs):
                    with dspy.settings.context(trace=[]):
                        prediction = program(**example.inputs().toDict())
                        trace_entries = list(dspy.settings.trace or [])
                    score, _ = self._evaluate_metric(example, prediction, trace_entries)
                    score = max(self.min_metric, min(self.max_metric, score))
                    per_runs.append(score)
                return median(per_runs)
            except Exception as e:
                self._log(f"APEX: Error evaluating example: {str(e)[:200]}", Verbosity.NORMAL, "warning")
                return self.min_metric

        scores = self._parallel_execute(
            cal_examples,
            process,
            description=f"APEX: evaluating {label}",
            level=Verbosity.NORMAL,
        )

        # Filter out None values that may result from parallel execution errors
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            self._log(f"APEX: Warning - no valid scores obtained for {label}", Verbosity.NORMAL, "warning")
            valid_scores = [self.min_metric]  # Use minimum metric as fallback

        overall = sum(valid_scores) / len(valid_scores)
        return CandidateRecord(
            program=program,
            overall_score=overall,
            per_example_scores=valid_scores,
            iteration=iteration,
            hypothesis=hypothesis,
        )

    def _apply_hypothesis(self, baseline: Module, hypothesis: HypothesisSpec) -> Module:
        candidate = baseline.deepcopy()
        name_to_predictor = dict(candidate.named_predictors())
        for predictor_name, changes in hypothesis.prompt_changes.items():
            if predictor_name not in name_to_predictor:
                raise ValueError(f"Hypothesis references unknown predictor '{predictor_name}'.")
            predictor = name_to_predictor[predictor_name]
            predictor.signature.instructions = changes.new_prompt
        return candidate

    def _select_best_candidate(self, candidates: list[CandidateRecord]) -> CandidateRecord:
        best_score = max(c.overall_score for c in candidates)
        best_candidates = [c for c in candidates if c.overall_score == best_score]
        return self._rng.choice(best_candidates)

    def _snapshot_program(self, program: Module) -> ProgramSnapshot:
        prompts: dict[str, str] = {}
        flow = []
        lookup: dict[int, str] = {}

        for name, predictor in program.named_predictors():
            flow.append(name)
            prompts[name] = getattr(predictor.signature, "instructions", "")
            lookup[id(predictor)] = name

        structure = repr(program)

        if not flow:
            flow_description = "No predictors"
        elif len(flow) == 1:
            flow_description = f"Single predictor: {flow[0]}"
        else:
            flow_description = "Input → " + " → ".join(flow) + " → Output"

        return ProgramSnapshot(
            structure=structure,
            flow_description=flow_description,
            prompts=prompts,
            predictor_name_by_id=lookup,
        )
