"""Summary table generation for APEX optimization results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ApexIterationLog, CandidateRecord


def generate_optimization_summary(
    iterations: list[ApexIterationLog],
    best_candidate: CandidateRecord,
    initial_score: float,
) -> str:
    """Generate a formatted summary table of the optimization process."""

    if not iterations:
        return "No iterations to summarize."

    lines = []
    lines.append("\n╔═══════════════════════════════════════════════════════════════╗")
    lines.append("║                    APEX Optimization Summary                     ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    # Overall stats
    total_iterations = len(iterations)
    total_hypotheses = sum(len(it.hypotheses) for it in iterations)
    total_candidates = sum(len(it.candidates) for it in iterations)
    improvement = best_candidate.overall_score - initial_score

    lines.append(f"║ Total Iterations: {total_iterations:<47} ║")
    lines.append(f"║ Hypotheses Tested: {total_hypotheses:<46} ║")
    lines.append(f"║ Candidates Evaluated: {total_candidates:<43} ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    # Score progress
    lines.append(f"║ Initial Score: {initial_score:<50.4f} ║")
    lines.append(f"║ Final Score: {best_candidate.overall_score:<52.4f} ║")
    improvement_str = f"{'+' if improvement >= 0 else ''}{improvement:.4f}"
    lines.append(f"║ Improvement: {improvement_str:<52} ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    # Iteration details table
    lines.append("║ Iter │ Train F/S │ Hypotheses │ Best Score │ Δ from prev ║")
    lines.append("╟──────┼───────────┼────────────┼────────────┼──────────────╢")

    prev_best = initial_score
    for it in iterations:
        failures = it.num_failures
        successes = it.num_successes
        num_hyp = len(it.hypotheses)

        if it.candidates:
            best_score = max(c.overall_score for c in it.candidates)
            delta = best_score - prev_best
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            prev_best = max(prev_best, best_score)
        else:
            best_score = prev_best
            delta_str = "0.0000"

        lines.append(
            f"║ {it.iteration:^4} │ {failures:>3}/{successes:<5} │ {num_hyp:^10} │ {best_score:^10.4f} │ {delta_str:^12} ║"
        )

    lines.append("╚══════╧═══════════╧════════════╧════════════╧══════════════╝")

    # Best hypothesis details if available
    if best_candidate.hypothesis:
        lines.append("\n╔═══════════════════════════════════════════════════════════════╗")
        lines.append("║                    Best Hypothesis Details                       ║")
        lines.append("╠═══════════════════════════════════════════════════════════════╣")
        lines.append(f"║ Strategy: {best_candidate.hypothesis.strategy[:54]:<54} ║")
        if len(best_candidate.hypothesis.strategy) > 54:
            remaining = best_candidate.hypothesis.strategy[54:]
            for i in range(0, len(remaining), 63):
                chunk = remaining[i:i+63]
                lines.append(f"║ {chunk:<63} ║")

        lines.append(f"║ Impact Score: {best_candidate.hypothesis.impact_score:<50.2f} ║")
        lines.append(f"║ Iteration: {best_candidate.iteration:<54} ║")

        if best_candidate.hypothesis.prompt_changes:
            lines.append(f"║ Updated Predictors: {', '.join(best_candidate.hypothesis.prompt_changes.keys())[:43]:<43} ║")

        lines.append("╚═══════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


__all__ = ["generate_optimization_summary"]