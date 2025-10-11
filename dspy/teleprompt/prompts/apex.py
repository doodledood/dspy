from __future__ import annotations

import json
from textwrap import dedent
from typing import Any


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_failure_prompt(payload: dict[str, Any]) -> str:
    example = payload["failed_example"]
    return dedent(
        f"""
        You are analyzing a failure in a DSPy program to identify its root cause.

        ## DSPy Program Structure

        {payload["program_structure"]}

        Predictor flow: {payload["predictor_flow"]}

        ## Current Predictor Prompts

        {_format_json(payload["predictor_prompts"])}

        ## Failed Example

        **Input:**
        {_format_json(example["input"])}

        **Execution Trace:**
        {_format_json(example["trace"])}

        **Final Output:**
        {_format_json(example["prediction"])}

        **Expected Output:**
        {_format_json(example["expected"])}

        **Metric Score:** {example["metric_score"]} (threshold for success: {payload["success_threshold"]})
        {example["metric_feedback"] or ""}
        {f"**Error:** {example['error']}" if example["error"] else ""}

        ---

        ## Your Task

        Analyze this failure deeply. Trace through the execution to find the ROOT CAUSE.

        The root cause may involve:
        - A single predictor's prompt being unclear, incomplete, or incorrect
        - Multiple predictors where an upstream predictor's output causes downstream failures
        - Interaction issues between predictors
        - Missing constraints or examples in prompts

        Provide your analysis in this exact JSON format:

        {{
          "root_cause": "Detailed description of what fundamentally caused this failure. Be specific about which predictor(s) and what aspect of their behavior caused the issue.",
          "involved_predictors": ["list", "of", "predictor", "names"],
          "context": "Relevant characteristics of this example that are important for understanding when/why this failure occurs. Include input characteristics, intermediate state issues, or patterns that would help generalize to similar failures.",
          "category": "A short label categorizing this failure type (e.g., 'format_ambiguity', 'incomplete_reasoning', 'upstream_error_propagation', 'missing_constraints')",
          "key_details": "Any additional important information that would help someone design a fix. What specifically went wrong in the predictor's processing? What should have happened instead?"
        }}

        Be thorough but concise. Focus on actionable insights for fixing the prompt(s).
        """
    ).strip()


def render_success_prompt(payload: dict[str, Any]) -> str:
    example = payload["successful_example"]
    return dedent(
        f"""
        You are analyzing a SUCCESS in a DSPy program to understand what worked well.

        This will be contrasted with failures to identify what differentiates successful executions.

        ## DSPy Program Structure

        {payload["program_structure"]}

        Predictor flow: {payload["predictor_flow"]}

        ## Current Predictor Prompts

        {_format_json(payload["predictor_prompts"])}

        ## Successful Example

        **Input:**
        {_format_json(example["input"])}

        **Execution Trace:**
        {_format_json(example["trace"])}

        **Final Output:**
        {_format_json(example["prediction"])}

        **Expected Output:**
        {_format_json(example["expected"])}

        **Metric Score:** {example["metric_score"]} (threshold for success: {payload["success_threshold"]})
        {example["metric_feedback"] or ""}

        ---

        ## Your Task

        Analyze why this example succeeded. What did the predictors do correctly?

        Focus on:
        - What aspects of the prompts guided correct behavior
        - How predictors handled this input well
        - What patterns in the execution led to success
        - What characteristics distinguish this from potential failures

        Provide your analysis in this exact JSON format:

        {{
          "success_pattern": "Clear description of what made this execution successful. What did the predictors do right?",
          "contributing_predictors": ["list", "of", "predictors", "that", "worked", "well"],
          "context": "Relevant characteristics of this example that help explain the success. What about the input, intermediate outputs, or execution made this work?",
          "category": "A short label for this success type (e.g., 'clear_format_compliance', 'complete_reasoning', 'robust_handling')",
          "key_details": "What specifically worked well? What aspects of the prompts or execution should be preserved or amplified?"
        }}

        Be thorough but concise. Focus on actionable insights that contrast with failures.
        """
    ).strip()


def render_hypothesis_prompt(payload: dict[str, Any]) -> str:
    return dedent(
        f"""
        You are a prompt engineering expert tasked with improving a DSPy program based on systematic error analysis.

        ## DSPy Program Structure

        {payload["program_structure"]}

        Predictor flow: {payload["predictor_flow"]}

        ## Current Predictor Prompts

        {_format_json(payload["predictor_prompts"])}

        ## Error Analyses

        We analyzed {len(payload["error_summaries"])} failures. Here are their root cause summaries:

        {_format_json(payload["error_summaries"])}

        ## Success Analyses

        We analyzed {len(payload["success_summaries"])} successful examples for contrast:

        {_format_json(payload["success_summaries"])}

        ---

        ## Your Task

        Synthesize these analyses and generate hypotheses for fixing ALL fixable issues.

        ### Step 1: Pattern Synthesis

        Identify:
        1. Common patterns across errors
        2. How successes differ from failures
        3. Which issues are fixable by prompt changes
        4. Which issues need architecture/tools/data (mark as non-fixable)

        ### Step 2: Hypothesis Generation

        Generate between 0 and {payload["num_hypotheses"]} hypotheses.

        **Critical Requirements:**
        - Each hypothesis must address ALL fixable root causes together
        - If you generate multiple hypotheses, they should represent DIFFERENT STRATEGIES for fixing the same issues
        - Examples of different strategies:
          - Minimal localized changes vs substantial rewrites
          - Fix upstream predictor vs make downstream robust
          - Add explicit constraints vs add examples
          - Different predictor combinations
        - Bias toward minimal effective change (simplest intervention that works)
        - Specify COMPLETE REPLACEMENT PROMPTS for each affected predictor
        - Explain rationale for each change

        **When to generate 0 hypotheses:**
        - All root causes are non-fixable (need architecture/data/tools)
        - No clear improvement strategy emerges from the analysis
        - Errors are too diverse/unclear to form actionable hypothesis

        **When to generate multiple hypotheses:**
        - There are genuinely different ways to address the same root causes
        - You want to explore different intervention levels (minimal vs substantial)
        - Different architectural approaches are viable

        ### Step 3: Output Format

        Return a JSON array of hypotheses (may be empty):

        [
          {{
            "observation": "Synthesized description of patterns found across all errors",
            "fixable_root_causes": ["Specific fixable issue 1", "Specific fixable issue 2"],
            "non_fixable_root_causes": ["Issue X: needs retrieval system", "Issue Y: requires multi-step architecture"],
            "strategy": "Description of the approach this hypothesis takes. What makes it different from alternative approaches?",
            "expected_impact": "Specific prediction of which errors this should fix and why. Be concrete.",
            "prompt_changes": {{
              "predictor_name": {{
                "new_prompt": "COMPLETE REPLACEMENT PROMPT TEXT HERE. This is the full new instruction/prompt for this predictor, not a diff or partial change.",
                "rationale": "Detailed explanation of why this change addresses the root causes. How does this fix the identified issues?",
                "change_magnitude": "minimal | moderate | substantial"
              }}
            }}
          }}
        ]

        Important notes:
        - You can modify 1 predictor or multiple predictors in a single hypothesis
        - Different hypotheses should NOT address different subsets of issues - they should all address ALL fixable issues
        - Be specific and actionable in new prompts
        - Preserve what works (insights from success analyses)
        - Consider predictor interactions and dependencies
        - If truly stuck with no good ideas, return an empty array []
        """
    ).strip()
