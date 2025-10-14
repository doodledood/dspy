"""Tracking utilities for APEX optimizer."""

from typing import Any


def format_iteration_metrics(
    iteration: int,
    num_failures: int,
    num_successes: int,
    hypotheses: list,
    candidates: list,
    best_score: float | None = None,
) -> dict[str, Any]:
    """Format iteration data for logging.

    Args:
        iteration: Iteration number
        num_failures: Number of failed examples
        num_successes: Number of successful examples
        hypotheses: List of hypotheses generated
        candidates: List of candidates evaluated
        best_score: Best score in this iteration

    Returns:
        Dictionary formatted for logging
    """
    data = {
        "iteration": iteration,
        "num_failures": num_failures,
        "num_successes": num_successes,
        "num_hypotheses": len(hypotheses),
        "num_candidates": len(candidates),
        "hypotheses": [],
        "candidates": [],
    }

    if best_score is not None:
        data["best_score"] = best_score

    # Format hypotheses
    for h in hypotheses:
        hypothesis_data = {
            "strategy": h.strategy if hasattr(h, "strategy") else "unknown",
            "impact_score": h.impact_score if hasattr(h, "impact_score") else 0.0,
            "generalizability_score": h.generalizability_score if hasattr(h, "generalizability_score") else 0.0,
            "fixable_root_causes": h.fixable_root_causes if hasattr(h, "fixable_root_causes") else [],
            "prompt_changes": {},
        }

        # Format prompt changes
        if hasattr(h, "prompt_changes") and h.prompt_changes:
            for pred_name, change in h.prompt_changes.items():
                hypothesis_data["prompt_changes"][pred_name] = {
                    "new_prompt": change.new_prompt,  # Full prompt, no truncation
                    "rationale": change.rationale if hasattr(change, "rationale") else "",
                    "magnitude": str(change.change_magnitude) if hasattr(change, "change_magnitude") else "unknown",
                }

        data["hypotheses"].append(hypothesis_data)

    return data


def format_candidate_data(candidate) -> dict[str, Any]:
    """Format candidate data for logging.

    Args:
        candidate: CandidateRecord object

    Returns:
        Dictionary formatted for logging
    """
    data = {
        "overall_score": candidate.overall_score,
        "iteration": candidate.iteration,
        "has_hypothesis": candidate.hypothesis is not None,
    }

    if hasattr(candidate, "per_example_scores") and candidate.per_example_scores:
        scores = candidate.per_example_scores
        data["per_example_scores"] = scores
        data["num_examples"] = len(scores)
        data["mean_score"] = sum(scores) / len(scores) if scores else 0.0

    if candidate.hypothesis:
        data["hypothesis_strategy"] = (
            candidate.hypothesis.strategy if hasattr(candidate.hypothesis, "strategy") else "unknown"
        )

    return data


def format_baseline_metrics(baseline_score: float, num_train: int, num_val: int) -> dict[str, Any]:
    """Format baseline metrics for logging.

    Args:
        baseline_score: Initial baseline score
        num_train: Number of training examples
        num_val: Number of validation examples

    Returns:
        Dictionary formatted for logging
    """
    return {
        "baseline_score": baseline_score,
        "num_train_examples": num_train,
        "num_val_examples": num_val,
        "initial_score": baseline_score,
    }


def format_optimization_summary(
    best_candidate,
    all_candidates: list,
    iterations: list,
    stopped_after: str,
    initial_score: float,
) -> dict[str, Any]:
    """Format final optimization summary for logging.

    Args:
        best_candidate: Best candidate found
        all_candidates: All candidates evaluated
        iterations: All iteration logs
        stopped_after: Reason for stopping
        initial_score: Initial baseline score

    Returns:
        Dictionary formatted for logging
    """
    summary = {
        "final_score": best_candidate.overall_score,
        "initial_score": initial_score,
        "improvement": best_candidate.overall_score - initial_score,
        "stopped_after": stopped_after,
        "total_iterations": len(iterations),
        "total_candidates": len(all_candidates),
    }

    # Calculate additional statistics
    if iterations:
        total_hypotheses = sum(len(it.hypotheses) for it in iterations)
        summary["total_hypotheses"] = total_hypotheses

        total_failures = sum(it.num_failures for it in iterations)
        total_successes = sum(it.num_successes for it in iterations)
        summary["total_failures_analyzed"] = total_failures
        summary["total_successes_analyzed"] = total_successes

    # Score trajectory
    if all_candidates:
        score_trajectory = [c.overall_score for c in all_candidates]
        summary["score_trajectory"] = score_trajectory
        summary["max_score_achieved"] = max(score_trajectory)
        summary["min_score_achieved"] = min(score_trajectory)

    return summary


def format_hypothesis_details(hypothesis) -> str:
    """Format hypothesis details as readable text.

    Args:
        hypothesis: HypothesisSpec object

    Returns:
        Formatted string for display/logging
    """
    lines = []

    if hasattr(hypothesis, "observation"):
        lines.append(f"Observation: {hypothesis.observation}")

    if hasattr(hypothesis, "strategy"):
        lines.append(f"Strategy: {hypothesis.strategy}")

    if hasattr(hypothesis, "expected_impact"):
        lines.append(f"Expected Impact: {hypothesis.expected_impact}")

    if hasattr(hypothesis, "impact_score"):
        lines.append(f"Impact Score: {hypothesis.impact_score:.2f}")

    if hasattr(hypothesis, "generalizability_score"):
        lines.append(f"Generalizability Score: {hypothesis.generalizability_score:.2f}")

    if hasattr(hypothesis, "fixable_root_causes") and hypothesis.fixable_root_causes:
        lines.append(f"Fixable Root Causes: {', '.join(hypothesis.fixable_root_causes)}")

    if hasattr(hypothesis, "prompt_changes") and hypothesis.prompt_changes:
        lines.append("\nPrompt Changes:")
        for pred_name, change in hypothesis.prompt_changes.items():
            lines.append(f"  {pred_name}:")
            if hasattr(change, "rationale"):
                lines.append(f"    Rationale: {change.rationale}")
            if hasattr(change, "change_magnitude"):
                lines.append(f"    Magnitude: {change.change_magnitude}")
            if hasattr(change, "new_prompt"):
                # Show full prompt, no truncation
                lines.append(f"    New Prompt: {change.new_prompt}")

    return "\n".join(lines)


def format_execution_flow_for_logging(execution_flow: list) -> str:
    """Format execution flow for logging as text.

    Args:
        execution_flow: List of ExecutionFlowEntry objects

    Returns:
        Formatted string for logging
    """
    if not execution_flow:
        return "No execution flow captured"

    lines = ["Execution Flow:"]
    lines.append("-" * 40)

    for i, entry in enumerate(execution_flow, 1):
        predictor_name = entry.predictor_name if hasattr(entry, "predictor_name") else f"Step {i}"
        predictor_type = entry.predictor_type if hasattr(entry, "predictor_type") else "Unknown"

        lines.append(f"\n{i}. {predictor_name} ({predictor_type})")

        if hasattr(entry, "dependencies") and entry.dependencies:
            lines.append(f"   Dependencies: {', '.join(entry.dependencies)}")
        else:
            lines.append("   Dependencies: None (input)")

        if hasattr(entry, "instructions") and entry.instructions:
            # Show full instructions, no truncation
            lines.append(f"   Instructions: {entry.instructions}")

    return "\n".join(lines)
