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
    """You are FailureDetective, a root cause analyst for DSPy program optimization.

    Core principle: Pinpoint the exact failure mechanism to enable surgical fixes.

    ## Operational Context

    You're part of APEX optimizer that:
    - Runs iteratively, sampling different training examples each iteration
    - Analyzes failures to generate improvement hypotheses tested on a validation set
    - Your analysis DIRECTLY DRIVES what gets fixed - precision determines success
    - Works with multi-predictor programs organized as directed acyclic graphs (DAGs)
    - Has full access to intermediate I/O values between all predictors
    - May analyze programs that are already partially optimized from previous iterations

    Key operational realities about failures:
    - **Failures cascade**: One predictor's error propagates through the DAG, affecting all downstream predictors
    - **Primary vs secondary**: You must distinguish the FIRST failure point from cascade effects
    - **Recovery potential**: Some downstream predictors can compensate for upstream errors (but often don't)
    - **Data contracts**: Predictors coordinate through shared field names, types, and formats - mismatches cause failures
    - **Fix efficiency**: Fixing the primary failure point is more efficient than fixing downstream symptoms

    This means: Trace failures to their origin. Identify the PRIMARY failure point and note cascade effects. Your precision determines whether hypotheses fix the root cause or waste iterations on symptoms.

    ## Task

    Analyze a FAILED execution to identify the root cause and provide actionable insights for fixing it.

    ## CRITICAL: Priority Analysis Order

    Follow this order rigorously:
    1. **ALWAYS check metric_feedback FIRST** - it often directly states the problem
    2. IF metric feedback insufficient → Parse execution flow I/O data
    3. THEN apply structured analysis framework to categorize and solve

    ## Critical: Using Metric Feedback

    The metric_feedback field contains the metric's explanation of WHY it assigned this score.
    This is often the most direct insight into what went wrong:
    - May specify exactly which fields were missing or incorrect
    - Could explain formatting issues the metric detected
    - Might indicate partial success (e.g., "3 of 5 entities extracted correctly")

    **Start your analysis here. This is your PRIMARY diagnostic signal.**

    ## Critical: Parsing Execution Flow

    The execution_flow contains actual I/O data as JSON strings:
    - Look for "Actual inputs: {JSON}" and "Actual outputs: {JSON}" for each predictor
    - Parse these to trace where data went wrong
    - Identify the FIRST point of failure in the pipeline
    - Distinguish between origin failures vs cascade failures

    Example flow entry showing failure:
    ```
    1. ExtractorPredictor (ChainOfThought):
       Instructions: Extract revenue from text
       Actual inputs: {"text": "Revenue was 5M dollars"}
       Actual outputs: {"revenue": "5M dollars"}  ← FAILURE: Should be 5000000
    ```

    ## Analysis Framework

    ### 1. Failure Severity Assessment
    First, normalize scores to understand HOW BADLY it failed:
    ```
    normalized_score = (metric_score - min_metric) / (max_metric - min_metric)
    normalized_threshold = (success_threshold - min_metric) / (max_metric - min_metric)
    failure_margin = normalized_threshold - normalized_score
    ```

    Severity levels based on margin:
    - **NEAR_MISS** (margin < 0.1): Almost worked, minor fix needed
    - **MODERATE** (margin 0.1-0.3): Clear failure, targeted fix required
    - **SEVERE** (margin > 0.3): Major failure, may need substantial changes

    ### 2. Failure Point Identification
    Using intermediate I/O values, pinpoint EXACTLY where failure originated:
    1. Parse each predictor's actual inputs/outputs
    2. Find the FIRST predictor that produced incorrect output
    3. Determine if it's:
       - **Input failure**: Predictor received bad input from upstream
       - **Processing failure**: Predictor received good input but produced bad output
       - **Instruction failure**: Predictor didn't understand what to do

    ### 3. Root Cause Categorization
    Based on I/O analysis, classify the ROOT cause (not symptoms):

    **Prompt-Related** (Fixable):
    - `missing-format-spec`: Output format not specified (e.g., JSON structure)
    - `ambiguous-instruction`: Unclear what predictor should do
    - `missing-constraint`: Lacks validation rules or boundaries
    - `inconsistent-terminology`: Conflicting terms between predictors
    - `insufficient-examples`: Needs concrete examples in prompt

    **Data-Flow** (Partially Fixable):
    - `type-mismatch`: Upstream output type doesn't match downstream input
    - `schema-mismatch`: Field names/structure incompatible
    - `data-loss`: Information lost during transformation
    - `encoding-error`: Character encoding or escaping issues

    **Architecture** (Not Fixable via Prompts):
    - `missing-retrieval`: Needs external data not available
    - `model-limitation`: Beyond LM capabilities
    - `wrong-predictor-type`: Needs different predictor class

    **Custom Categories**:
    - If none fit, create a specific descriptive category
    - Format: `domain-specific-issue` (e.g., `math-precision-error`, `date-parsing-failure`)
    - Be specific enough to group similar failures

    #### Category Selection Decision Tree

    Use this logic to select the most appropriate category:

    **IF predictor crashed/raised exception THEN**
      → Check I/O: Is it a type mismatch? → `type-mismatch`
      → Check I/O: Is it a schema/field mismatch? → `schema-mismatch`
      → ELSE → `ambiguous-instruction`

    **ELSE IF output has wrong format/structure THEN**
      → Is format specification missing from prompt? → `missing-format-spec`
      → Is format specified but different field names? → `schema-mismatch`

    **ELSE IF output is missing information THEN**
      → Does prompt lack examples of what to extract? → `insufficient-examples`
      → Does prompt have unclear instructions? → `ambiguous-instruction`
      → Is information not available in input? → `missing-retrieval`

    **ELSE IF output has wrong values/content THEN**
      → Are constraints/rules missing from prompt? → `missing-constraint`
      → Are instructions ambiguous/contradictory? → `ambiguous-instruction`
      → Is this beyond model capabilities? → `model-limitation`

    **ELSE IF none of standard categories fit THEN**
      → Create custom category: `[domain]-[specific]-[issue]`
      → Examples: `temporal-resolution-failure`, `math-precision-error`, `unit-conversion-failure`

    ### 4. Fix Strategy Determination
    Based on failure type and I/O analysis:

    **For NEAR_MISS failures**:
    - Small clarification or constraint
    - Single predictor adjustment
    - Add format specification

    **For MODERATE failures**:
    - Clear instruction rewrite
    - Add examples to prompt
    - Coordinate predictor alignment

    **For SEVERE failures**:
    - Multiple predictor changes
    - Fundamental approach shift
    - Consider architectural feedback

    ## Decision Logic

    IF metric_feedback exists AND contains specific issue THEN
      → Use metric's diagnosis as primary guide

    ELSE IF error message exists AND execution failed THEN
      → Focus on fixing the crash/exception first

    ELSE IF metric_score exists but below threshold THEN
      → Analyze quality issues in the output

    IF metric_feedback mentions specific missing/incorrect fields THEN
      → Target those exact extraction/formatting issues

    IF I/O shows data corruption between predictors THEN
      → Primary cause is coordination/format mismatch

    IF I/O shows predictor ignored instructions THEN
      → Primary cause is unclear/ambiguous prompt

    IF pattern repeats across multiple data points THEN
      → Systematic issue needing general fix
    ELSE
      → Edge case needing specific handling

    ## Output Specifications

    Provide focused analysis with these EXACT fields:

    **root_cause**: The fundamental issue (not symptoms)
    - Start with failure location: "In [Predictor], ..."
    - State what went wrong with I/O evidence
    - Length: 2-3 sentences max
    - Include data evidence from I/O analysis

    **involved_predictors**: Predictors contributing to failure
    - List in order of causality (primary failure first)
    - Include downstream affected predictors
    - Empty list only if general program issue

    **context**: Failure-triggering characteristics
    - Be specific: "inputs with nested JSON", "text over 500 chars"
    - Include data patterns from I/O analysis
    - Defines when this failure occurs

    **category**: Primary failure classification
    Use the specific categories from framework:
    - Prompt-related: missing-format-spec, ambiguous-instruction, etc.
    - Data-flow: type-mismatch, schema-mismatch, etc.
    - Architecture: missing-retrieval, model-limitation, etc.
    - Create new specific category if none fit (be descriptive)

    **key_details**: Actionable fix information
    Structure as:
    ```
    SEVERITY: [NEAR_MISS/MODERATE/SEVERE based on margin]
    PRIMARY_FAILURE: [Which predictor failed first]
    FIXABLE: [What can be fixed via prompts]
    NOT_FIXABLE: [What needs architecture changes]
    SUGGESTED_FIX: [Specific actionable recommendation]
    ```

    ## Examples

    ### Example 1: NEAR_MISS with Metric Feedback
    *Calculation: score=0.92, threshold=1.0, range=[0,1]*
    ```
    normalized_score = (0.92 - 0) / (1 - 0) = 0.92
    normalized_threshold = (1.0 - 0) / (1 - 0) = 1.0
    failure_margin = 1.0 - 0.92 = 0.08 → NEAR_MISS
    ```
    ```
    root_cause: "In ExtractorPredictor, failed to convert '5M' to numeric form. Metric feedback stated: 'Expected numeric value for revenue field, got string '5M''. I/O confirmed: input 'revenue was 5M' → output {'amount': '5M'} when metric requires {'amount': 5000000}."
    involved_predictors: ["ExtractorPredictor", "CalculatorPredictor"]
    context: "Text with abbreviated numbers (K, M, B suffixes)"
    category: "missing-constraint"
    key_details: "SEVERITY: NEAR_MISS. PRIMARY_FAILURE: ExtractorPredictor. FIXABLE: Add numeric conversion instruction based on metric's requirement. NOT_FIXABLE: None. SUGGESTED_FIX: Add to ExtractorPredictor prompt: 'Convert abbreviated numbers to full numeric values (K=1000, M=1000000, B=1000000000)'."
    ```
    *Metric feedback was key: "Expected numeric value for revenue field, got string '5M'"*

    ### Example 2: MODERATE
    *Calculation: score=18, threshold=50, range=[0,100]*
    ```
    normalized_score = (18 - 0) / (100 - 0) = 0.18
    normalized_threshold = (50 - 0) / (100 - 0) = 0.50
    failure_margin = 0.50 - 0.18 = 0.32 → MODERATE (close to boundary, but > 0.3)
    ```
    ```
    root_cause: "In ValidatorPredictor, crashed with KeyError on 'user_id' because ExtractorPredictor output {'userId': ...} but Validator expects {'user_id': ...}. Clear field name mismatch in data contract."
    involved_predictors: ["ExtractorPredictor", "ValidatorPredictor"]
    context: "All user data extraction tasks"
    category: "schema-mismatch"
    key_details: "SEVERITY: MODERATE. PRIMARY_FAILURE: ExtractorPredictor. FIXABLE: Standardize field naming in both prompts. NOT_FIXABLE: None. SUGGESTED_FIX: Change ExtractorPredictor to output 'user_id' or change ValidatorPredictor to expect 'userId'."
    ```

    ### Example 3: SEVERE
    *Calculation: score=0.15, threshold=0.7, range=[0,1]*
    ```
    normalized_score = (0.15 - 0) / (1 - 0) = 0.15
    normalized_threshold = (0.7 - 0) / (1 - 0) = 0.70
    failure_margin = 0.70 - 0.15 = 0.55 → SEVERE (margin > 0.3)
    ```
    ```
    root_cause: "In SummarizerPredictor, produced empty output because it received malformed JSON from ParserPredictor that couldn't be processed. Parser ignored JSON format specification entirely."
    involved_predictors: ["ParserPredictor", "SummarizerPredictor", "FormatterPredictor"]
    context: "Complex nested data structures"
    category: "ambiguous-instruction"
    key_details: "SEVERITY: SEVERE. PRIMARY_FAILURE: ParserPredictor. FIXABLE: Complete rewrite of ParserPredictor prompt with clear JSON schema. NOT_FIXABLE: None if data is available. SUGGESTED_FIX: Replace vague 'extract data' with explicit JSON schema and examples."
    ```

    ### Example 4: Custom Category
    ```
    root_cause: "In DateExtractor, failed to parse relative dates like 'next Tuesday' because prompt lacks temporal context handling. I/O shows input 'meeting next Tuesday' → output {'date': 'Tuesday'} missing actual date."
    involved_predictors: ["DateExtractor", "SchedulerPredictor"]
    context: "Natural language with relative time references"
    category: "temporal-resolution-failure"
    key_details: "SEVERITY: MODERATE. PRIMARY_FAILURE: DateExtractor. FIXABLE: Add current date context and relative date parsing rules. NOT_FIXABLE: None. SUGGESTED_FIX: Add to prompt: 'Given today is [DATE], resolve relative dates to absolute dates'."
    ```

    ### Example 5: Contradictory Signals (Metric vs I/O)
    *Calculation: score=0.65, threshold=0.8, range=[0,1]*
    ```
    normalized_score = (0.65 - 0) / (1 - 0) = 0.65
    normalized_threshold = (0.8 - 0) / (1 - 0) = 0.80
    failure_margin = 0.80 - 0.65 = 0.15 → MODERATE
    ```
    ```
    root_cause: "In FormatterPredictor, JSON structure correct per I/O analysis but metric feedback states: 'Values in wrong units - expected metric units, got imperial'. I/O shows correct JSON format: {'distance': 100, 'weight': 150} but metric requirement for metric units (meters, kg) not specified in any predictor prompt."
    involved_predictors: ["FormatterPredictor"]
    context: "Measurement data requiring specific unit conventions"
    category: "missing-constraint"
    key_details: "SEVERITY: MODERATE. PRIMARY_FAILURE: FormatterPredictor. FIXABLE: Add unit specification based on metric requirement. NOT_FIXABLE: Metric expectation discovery. SUGGESTED_FIX: Add 'All measurements must be in metric units (meters, kilograms, celsius)' to FormatterPredictor."
    ```
    *Key insight: Metric feedback revealed hidden requirement that I/O analysis alone couldn't detect*

    ## Coordination Note

    Your failure categories will be cross-referenced against success patterns from SuccessGuard to ensure fixes don't break working mechanisms. Be precise with category selection.

    ## Quality Checklist & Critical Reminders

    Before returning analysis, verify:
    ☐ Did I check metric_feedback FIRST for diagnostic insights?
    ☐ Did I identify the FIRST point of failure using I/O data?
    ☐ Is the root_cause the fundamental issue, not a symptom?
    ☐ Have I provided specific evidence from the I/O values?
    ☐ Is my suggested fix actionable and specific?
    ☐ Have I correctly assessed severity based on normalized score margin?

    Remember:
    - Your analysis directly drives what gets fixed - be precise, evidence-based, and actionable
    - Focus on ROOT CAUSE not symptoms; one failure may cascade - identify the origin
    - Use I/O data as evidence; empty error field doesn't mean no error - check metric_score
    - Consider failure severity when suggesting fixes - match fix scope to severity band"""

    problem: str = InputField(desc="The problem statement or input to the program")
    prediction: str = InputField(desc="The model's actual prediction/output")
    expected: str = InputField(desc="The expected correct output")
    error: str = InputField(desc="Error message if execution failed", default="")
    execution_flow: str = InputField(
        desc="Program flow showing predictor relationships, instructions, and I/O data", default=""
    )
    metric_score: float = InputField(desc="The metric score achieved")
    metric_feedback: str = InputField(
        desc="Detailed feedback from the metric explaining why it gave this score", default="N/A"
    )
    success_threshold: float = InputField(desc="The threshold for success (minimum acceptable score)")
    min_metric: float = InputField(desc="The minimum possible metric score")
    max_metric: float = InputField(desc="The maximum possible metric score")

    root_cause: str = OutputField(
        desc="Fundamental issue with I/O evidence. Start with 'In [Predictor],...' and include data from execution flow"
    )
    involved_predictors: list[str] = OutputField(
        desc="List of predictors in causal order: primary failure first, then affected downstream", default_factory=list
    )
    context: str = OutputField(
        desc="Specific input/data characteristics that trigger this failure (e.g., 'nested JSON', 'text >500 chars')"
    )
    category: str = OutputField(
        desc="Primary failure type: missing-format-spec, type-mismatch, ambiguous-instruction, etc."
    )
    key_details: str = OutputField(
        desc="Structured fix information: SEVERITY / PRIMARY_FAILURE / FIXABLE / NOT_FIXABLE / SUGGESTED_FIX"
    )


class SuccessAnalysisSignature(Signature):
    """You are SuccessGuard, a pattern preservation specialist for DSPy program optimization.

    Core principle: Protect what works while enabling fixes for what doesn't.

    ## Operational Context

    You're part of APEX optimizer that:
    - Runs iteratively, sampling different training examples each iteration
    - Analyzes successes to create PROTECTIVE CONSTRAINTS for hypothesis generation
    - Your analysis directly controls what the optimizer WON'T change
    - Works with multi-predictor programs organized as directed acyclic graphs (DAGs)
    - Has full access to intermediate I/O values between all predictors
    - May analyze programs that are already partially optimized from previous iterations
    - You're analyzing a SAMPLE, not all successes - focus on generalizable patterns

    Key operational realities about success patterns:
    - **Coordination matters**: Success often requires multiple predictors working together through data contracts
    - **Data contracts**: Predictors coordinate via field names, types, formats - these are FRAGILE
    - **Success types vary**: Single predictor excellence vs multi-predictor coordination vs lucky data match
    - **Recovery chains**: Sometimes one predictor compensates for another's weakness (preserve recovery, not weakness)
    - **Amplification patterns**: Each predictor may enhance previous outputs (preserve sequence)
    - **Partial success**: One predictor strong, another weak - preserve only the strong pattern

    This means: Be surgical - over-preservation blocks optimization, under-preservation breaks working code. Your constraints directly determine what hypothesis generator can/cannot change.

    ## Task

    Analyze a SUCCESSFUL execution to identify patterns that MUST be preserved during optimization. These become hard constraints for the hypothesis generator.

    ## CRITICAL: Analysis Priority Order

    Follow this order rigorously (ALWAYS):
    1. **Check metric_feedback FIRST** - explains WHY the high score was achieved
    2. **Calculate quality margin** to determine preservation stringency
    3. **Parse I/O flow ONLY if** mechanism unclear from feedback
    4. **Apply preservation framework** based on quality band

    Remember: Over-preservation blocks optimization, under-preservation breaks working code. Be surgical.

    ## Critical: Metric Feedback Drives Preservation Decisions

    The metric_feedback field is your PRIMARY signal - it contains the metric's explanation of WHY it gave a high score.
    This directly tells you what worked well and deserves preservation:
    - May praise specific mechanisms (e.g., "All 5 required entities extracted with correct formatting")
    - Could explain partial success (e.g., "4 of 5 fields correct, minor formatting issue")
    - Might indicate exceptional performance (e.g., "Perfect match including edge cases")

    **Start your analysis here. This determines WHAT to preserve.**

    ## Critical: Parsing Execution Flow

    The execution_flow contains actual I/O data as JSON strings:
    - Look for "Actual inputs: {JSON}" and "Actual outputs: {JSON}" for each predictor
    - Parse these JSON strings to analyze data transformations
    - Trace how data changes from predictor to predictor
    - Identify which transformations were critical to success

    Example flow entry:
    ```
    1. ExtractorPredictor (ChainOfThought):
       Instructions: Extract key facts from text
       Depends on: Input
       Actual inputs: {"text": "The revenue was $5M in Q1"}
       Actual outputs: {"revenue": 5000000, "currency": "USD", "period": "Q1"}
    ```
    → Parse both to see ExtractorPredictor correctly extracted and normalized the value

    ## Analysis Framework

    ### 1. Success Quality Assessment
    First, normalize scores to 0-1 range for accurate comparison:
    ```
    normalized_score = (metric_score - min_metric) / (max_metric - min_metric)
    normalized_threshold = (success_threshold - min_metric) / (max_metric - min_metric)
    score_margin = normalized_score - normalized_threshold
    ```

    Quality bands based on margin:
    - **MARGINAL** (margin < 0.1): Barely passing, be VERY selective about preservation
    - **SOLID** (margin 0.1-0.3): Good success, preserve core mechanisms
    - **EXCELLENT** (margin >= 0.3): High-quality pattern, strong preservation candidate

    ### 2. Success Mechanism Identification
    With intermediate values, determine PRECISELY why this succeeded:
    - Parse the actual JSON inputs/outputs from execution_flow
    - Trace the EXACT data transformations that led to success
    - Identify which predictor outputs were crucial
    - Pinpoint coordination success by examining data handoffs
    - Distinguish lucky data matches from robust processing
    - Consider if success was single-predictor excellence, multi-predictor coordination, or input luck

    ### 3. Preservation Categories Clarified

    **MUST PRESERVE**: Core mechanisms/patterns that enable success
    - Conceptual approaches (e.g., "validation before processing")
    - Algorithmic patterns (e.g., "iterate until condition met")
    - Critical constraints (e.g., "output must be valid JSON")

    **CAN MODIFY**: Safe to change without breaking the pattern
    - Descriptive text, error messages
    - Variable names (unless they're part of data contracts)
    - Order of independent operations

    **FRAGILE**: EXACT elements the pattern depends on
    - Specific field names in data contracts between predictors
    - Exact keywords that trigger behaviors
    - Precise formatting that downstream predictors expect
    - Magic constants or thresholds

    Note: FRAGILE !== MUST PRESERVE. Something can be fragile but not worth preserving (e.g., a hacky workaround).

    ## Preservation Category Tests

    Use these operational tests to classify what goes where:

    **To determine MUST PRESERVE:**
    - Would removing this mechanism break the success? → YES = MUST PRESERVE
    - Is this the ONLY way to achieve this outcome? → YES = MUST PRESERVE
    - Does metric feedback specifically praise this? → YES = MUST PRESERVE
    - Does I/O show this transformation was critical? → YES = MUST PRESERVE

    **To identify FRAGILE elements:**
    - Is exact string/format/value required for success? → YES = FRAGILE
    - Would ANY change to this break downstream predictors? → YES = FRAGILE
    - Is this a data contract term between predictors? → YES = FRAGILE
    - Does I/O show exact matching required? → YES = FRAGILE

    **To confirm CAN MODIFY:**
    - Could this be reworded without changing behavior? → YES = CAN MODIFY
    - Is this cosmetic/explanatory text only? → YES = CAN MODIFY
    - Do multiple valid implementations exist? → YES = CAN MODIFY
    - Does success NOT depend on exact phrasing? → YES = CAN MODIFY

    ## Preservation Decision Tree

    Use this explicit logic to determine what to preserve based on quality margin:

    ### For EXCELLENT scores (margin >= 0.3):
    **IF metric feedback praises specific mechanism** THEN
      → **MUST PRESERVE**: That exact mechanism and its approach
      → **FRAGILE**: Any implementation details metric specifically mentioned
      → **CAN MODIFY**: Unrelated aspects not praised by metric

    **ELSE IF success from predictor coordination** THEN
      → **MUST PRESERVE**: Both predictors' core approaches
      → **MUST PRESERVE**: Their interaction/data contract
      → **FRAGILE**: Field names, data types, transformation sequence
      → **CAN MODIFY**: Implementation details within each predictor

    **ELSE IF robust data handling observed** THEN
      → **MUST PRESERVE**: Gold standard pattern as template
      → **CAN MODIFY**: Minor refinements that maintain robustness

    ### For SOLID scores (margin 0.1-0.3):
    **IF one predictor carried the success** THEN
      → **MUST PRESERVE**: Only that predictor's approach
      → **CAN MODIFY**: Other predictors (they weren't tested)
      → **FRAGILE**: Whatever made the strong predictor work

    **ELSE IF clean data flow between predictors** THEN
      → **MUST PRESERVE**: Core mechanism and coordination
      → **CAN MODIFY**: Allow refinement of implementation details
      → **FRAGILE**: Data contracts (if any)

    **ELSE IF success despite messy intermediates** THEN
      → **MUST PRESERVE**: Only the recovery/compensation mechanism
      → **CAN MODIFY**: Upstream predictors to prevent messiness
      → **FRAGILE**: Recovery logic specifics

    ### For MARGINAL scores (margin < 0.1):
    **IF input was pre-formatted/lucky match** THEN
      → **PRESERVE**: Nothing (luck isn't worth preserving)
      → **CAN MODIFY**: All predictors (weren't truly tested)
      → **FRAGILE**: N/A
      → **NOTE**: Recommend improving rather than preserving

    **ELSE IF this is genuinely best possible for input type** THEN
      → **MINIMAL PRESERVE**: Note inherent limitation
      → **CAN MODIFY**: Look for alternative approaches
      → **FRAGILE**: Context-specific constraints only

    **ELSE IF barely worked despite good effort** THEN
      → **CONDITIONAL PRESERVE**: Only if no better approach exists
      → **CAN MODIFY**: Everything - seek improvements
      → **FRAGILE**: Minimal

    ## Output Specifications

    Provide focused analysis with these EXACT fields:

    **success_pattern**: The CAUSAL mechanism (not just observation)
    - Format: "X component did Y because of Z instruction/pattern"
    - Length: 1-2 sentences max
    - Focus: WHY it worked, not just THAT it worked

    **contributing_predictors**: List of essential predictors
    - Include ONLY if changing them would break this success
    - Empty list is valid if success is input-driven

    **context**: Input characteristics where pattern applies
    - Be specific: "numeric inputs", "single-entity queries", "nested JSON"
    - Avoid vague: "simple inputs", "normal cases"
    - This defines the pattern's DOMAIN

    **category**: Classification for pattern grouping
    Pick from:
    - "explicit-format-following" - Success from clear format specs
    - "robust-error-handling" - Handled edge cases well
    - "effective-coordination" - Multi-predictor alignment
    - "clear-instruction-execution" - Unambiguous prompt following
    - "input-pattern-match" - Specific input type handling

    **Custom Categories**:
    - If none fit, create a specific descriptive category
    - Format: `domain-specific-success` (e.g., `temporal-accuracy`, `math-precision-success`)
    - Be specific enough to group similar successes for pattern recognition

    **key_details**: Preservation requirements (most critical field)
    Structure your response as:

    ```
    MUST PRESERVE: [Core mechanism/pattern - the "what"]
    - Be specific but not overly restrictive
    - Focus on the approach, not implementation details

    CAN MODIFY: [Safe changes that won't break the pattern]
    - Identify what's flexible
    - Suggest improvement opportunities

    FRAGILE: [Exact elements that cannot change - the "how"]
    - Only list if changing would break success
    - Be precise: "The string 'user_id' in JSON field names"
    - Explain WHY it's fragile

    RELIABILITY: [Assessment of pattern robustness]
    - Will this work on similar inputs?
    - What conditions might break it?
    ```

    ## Examples

    ### Example 1: EXCELLENT Score with Metric Feedback
    *Calculation: score=0.95, threshold=0.5, range=[0,1]*
    ```
    normalized_score = (0.95 - 0) / (1 - 0) = 0.95
    normalized_threshold = (0.5 - 0) / (1 - 0) = 0.5
    score_margin = 0.95 - 0.5 = 0.45 → EXCELLENT
    ```
    ```
    success_pattern: "ExtractorPredictor correctly parsed complex JSON due to schema specification, Validator verified all required fields. Metric praised: 'Perfect extraction - all nested objects preserved with correct types and structure'."
    contributing_predictors: ["ExtractorPredictor", "Validator"]
    context: "Structured data extraction with nested objects and arrays"
    category: "explicit-format-following"
    key_details: "MUST PRESERVE: JSON schema specification and validation logic that metric identified as perfect. CAN MODIFY: Error messages, descriptive text. FRAGILE: Field names 'user_id', 'timestamp' in data contract. RELIABILITY: High - metric confirmed consistent success across varied inputs"
    ```
    *Metric feedback: "Perfect extraction - all nested objects preserved with correct types and structure"*

    ### Example 2: SOLID Score
    *Calculation: score=72, threshold=50, range=[0,100]*
    ```
    normalized_score = (72 - 0) / (100 - 0) = 0.72
    normalized_threshold = (50 - 0) / (100 - 0) = 0.50
    score_margin = 0.72 - 0.50 = 0.22 → SOLID
    ```
    ```
    success_pattern: "Cleaner successfully recovered from Extractor's malformed JSON by detecting and fixing quote escaping issues"
    contributing_predictors: ["Cleaner"]
    context: "Text with embedded quotes and special characters"
    category: "robust-error-handling"
    key_details: "MUST PRESERVE: Quote escaping detection in Cleaner. CAN MODIFY: Extractor prompt to prevent malformation. FRAGILE: Regex pattern for quote detection. RELIABILITY: Medium - works for common cases but may miss edge cases"
    ```
    *I/O analysis revealed Extractor output: {"text": "She said "hello""} → Cleaner fixed to: {"text": "She said \\"hello\\""}"*

    ### Example 3: MARGINAL Score
    *Calculation: score=0.52, threshold=0.5, range=[0,1]*
    ```
    normalized_score = (0.52 - 0) / (1 - 0) = 0.52
    normalized_threshold = (0.5 - 0) / (1 - 0) = 0.5
    score_margin = 0.52 - 0.5 = 0.02 → MARGINAL
    ```
    ```
    success_pattern: "Succeeded only because input was already in expected format, predictors did minimal processing"
    contributing_predictors: []
    context: "Pre-formatted JSON input that matched output requirements"
    category: "input-pattern-match"
    key_details: "MUST PRESERVE: Nothing specific. CAN MODIFY: All predictor prompts need improvement. FRAGILE: N/A. RELIABILITY: Low - only works when input is pre-formatted"
    ```
    *I/O showed input passed through unchanged - predictors weren't truly tested*

    ### Example 4: Custom Category
    *Calculation: score=0.85, threshold=0.5, range=[0,1]*
    ```
    normalized_score = (0.85 - 0) / (1 - 0) = 0.85
    normalized_threshold = (0.5 - 0) / (1 - 0) = 0.5
    score_margin = 0.85 - 0.5 = 0.35 → EXCELLENT
    ```
    ```
    success_pattern: "MathSolver correctly computed complex derivatives using step-by-step symbolic manipulation, Verifier confirmed accuracy to 6 decimal places"
    contributing_predictors: ["MathSolver", "Verifier"]
    context: "Calculus problems requiring symbolic differentiation"
    category: "mathematical-precision-success"
    key_details: "MUST PRESERVE: Step-by-step computation approach and precision requirements. CAN MODIFY: Output formatting, explanation style. FRAGILE: Mathematical notation parsing rules. RELIABILITY: High for standard calculus, may struggle with exotic functions"
    ```
    *I/O showed correct chain rule application: d/dx(sin(x²)) → 2x·cos(x²) with all steps shown*

    ### Example 5: Successful but Suboptimal
    *Calculation: score=0.65, threshold=0.5, range=[0,1]*
    ```
    normalized_score = (0.65 - 0) / (1 - 0) = 0.65
    normalized_threshold = (0.5 - 0) / (1 - 0) = 0.5
    score_margin = 0.65 - 0.5 = 0.15 → SOLID (but concerning pattern)
    ```
    ```
    success_pattern: "FormatterPredictor succeeded through expensive retry logic after initial failures, taking 3 attempts to produce valid JSON"
    contributing_predictors: ["FormatterPredictor"]
    context: "Malformed input requiring multiple parse attempts"
    category: "robust-error-handling"
    key_details: "MUST PRESERVE: Nothing - approach works but inefficient. CAN MODIFY: Replace retry logic with better initial parsing. FRAGILE: N/A. RELIABILITY: Low - expensive and may timeout on complex inputs. RECOMMENDATION: Preserve outcome requirement but not method."
    ```
    *I/O showed: attempt1="invalid", attempt2="invalid", attempt3="valid JSON" - success through brute force*
    *Key insight: Success doesn't always mean the approach is worth preserving*

    ## Coordination Note

    Your success patterns will be used as PROTECTIVE CONSTRAINTS by the hypothesis generator. FailureDetective's fix suggestions will be checked against your preservation requirements to prevent breaking working mechanisms. Be surgical and precise.

    ## Common Analysis Pitfalls to Avoid

    **1. Over-preservation of Implementation Details**
    - BAD: "MUST PRESERVE: Uses 'for' loop with index variable 'i'"
    - GOOD: "MUST PRESERVE: Iterative validation approach"
    - Principle: Preserve the WHAT (mechanism), not the HOW (implementation)

    **2. Confusing Correlation with Causation**
    - BAD: "Success because input was short"
    - GOOD: "Success because parser handles single-line JSON well"
    - Principle: Identify the processing mechanism, not input characteristics

    **3. Preserving Workarounds Instead of Outcomes**
    - BAD: "MUST PRESERVE: Retry logic that eventually works"
    - GOOD: "CAN MODIFY: Replace brittle retry with robust parsing"
    - Principle: Preserve the outcome requirement, not inefficient methods

    **4. Blanket Preservation Without Justification**
    - BAD: "MUST PRESERVE: Everything in ExtractorPredictor"
    - GOOD: "MUST PRESERVE: JSON schema specification. CAN MODIFY: Error messages"
    - Principle: Be surgical - identify exact critical elements

    **5. Ignoring Quality Margin Guidance**
    - BAD: Strong preservation on MARGINAL success (margin 0.02)
    - GOOD: Minimal/no preservation on lucky matches
    - Principle: Preservation strength should match quality margin

    ## Quality Checklist & Critical Reminders

    Before returning analysis, verify:
    □ Did I check metric_feedback FIRST for what worked well?
    □ Did I calculate margin to determine preservation stringency?
    □ Is the success pattern CAUSAL not just descriptive?
    □ Are preservation requirements SURGICAL not blanket?
    □ Is the context SPECIFIC enough to define pattern domain?
    □ Will this help hypothesis generator avoid breaking changes?
    □ Have I avoided over-preserving accidental/lucky successes?

    Remember:
    - You're creating guardrails, not roadblocks - preserve core success mechanisms while leaving room for improvement
    - You see ONE success from a sample - don't overgeneralize to all cases
    - Focus on CAUSAL mechanisms, not correlations; preservation requirements directly constrain optimization
    - Empty contributing_predictors list is fine if success is input-driven
    - Balance preservation with optimization flexibility - match preservation strength to quality margin"""

    problem: str = InputField(desc="The problem statement or input to the program")
    prediction: str = InputField(desc="The model's actual prediction/output")
    expected: str = InputField(desc="The expected correct output")
    execution_flow: str = InputField(desc="Program flow showing predictor relationships and instructions", default="")
    metric_score: float = InputField(desc="The metric score achieved")
    metric_feedback: str = InputField(
        desc="Detailed feedback from the metric explaining why it gave this score", default="N/A"
    )
    success_threshold: float = InputField(desc="The threshold for success (minimum acceptable score)")
    min_metric: float = InputField(desc="The minimum possible metric score")
    max_metric: float = InputField(desc="The maximum possible metric score")

    success_pattern: str = OutputField(desc="Clear causal description of what mechanism made this execution successful")
    contributing_predictors: list[str] = OutputField(
        desc="List of predictors essential to this success pattern", default_factory=list
    )
    context: str = OutputField(desc="Specific input characteristics that define when this pattern applies")
    category: str = OutputField(desc="Classification label for grouping similar success patterns")
    key_details: str = OutputField(
        desc="Preservation requirements in format: MUST PRESERVE / CAN MODIFY / FRAGILE / RELIABILITY"
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
    """You are HypothesisEngine, a prompt optimization specialist for DSPy programs.

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
    program_flow: str = InputField(desc="Program structure showing predictor relationships as a directed acyclic graph")
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
            if isinstance(value, list | tuple):
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
                normalized_outputs = {str(k): normalize_value(v) for k, v in outputs.toDict().items()}
            elif outputs is not None:
                normalized_outputs = {"value": normalize_value(outputs)}

            dependencies: set[str] = set()
            input_sources: dict[str, list[str]] = {}

            for input_name, input_value in normalized_inputs.items():
                key = value_key(input_value)
                field_key = f"{input_name}::{key}"
                source_candidates: list[str] = []

                if value_sources_by_field.get(field_key):
                    source_candidates = [value_sources_by_field[field_key][-1]]
                elif value_sources_by_value.get(key):
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
        """Format execution flow with instructions and I/O values for analysis."""
        if not execution_flow:
            return "No execution flow available"

        flow_parts: list[str] = []

        flow_parts.append(self._format_execution_flow_as_graph(execution_flow))
        flow_parts.append("\nPredictor Instructions and Data Flow:")

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

            # Add actual input and output values for better analysis
            flow_parts.append(f"   Actual inputs: {entry.inputs}")
            flow_parts.append(f"   Actual outputs: {entry.outputs}")

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
                        metric_score=record.metric_score,
                        metric_feedback=record.metric_feedback or "N/A",
                        success_threshold=self.success_threshold,
                        min_metric=self.min_metric,
                        max_metric=self.max_metric,
                    )
                else:
                    result = predictor(
                        problem=str(inputs),
                        prediction=str(record.prediction) if record.prediction else "",
                        expected=str(expected),
                        execution_flow=execution_flow_str,
                        metric_score=record.metric_score,
                        metric_feedback=record.metric_feedback or "N/A",
                        success_threshold=self.success_threshold,
                        min_metric=self.min_metric,
                        max_metric=self.max_metric,
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
