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

    # Prompt evolution lineage - only show improvements that became best so far
    lines.append("\n╔═══════════════════════════════════════════════════════════════╗")
    lines.append("║                  Prompt Evolution Lineage                    ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    # Build lineage by tracking best score improvements
    lineage = []
    best_so_far = initial_score

    for it in iterations:
        # Find the best candidate in this iteration
        if it.candidates:
            iteration_best = max(it.candidates, key=lambda c: c.overall_score)

            # Only include if it improved over previous best and has a hypothesis
            if iteration_best.overall_score > best_so_far and iteration_best.hypothesis is not None:
                lineage.append((iteration_best, best_so_far))
                best_so_far = iteration_best.overall_score

    if lineage:
        for candidate, prev_score in lineage:
            score = candidate.overall_score
            delta = score - prev_score
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"

            header = f"Iteration {candidate.iteration} (score={score:.4f}, Δ={delta_str})"
            lines.append(f"║ {header:<61} ║")

            if candidate.hypothesis and candidate.hypothesis.prompt_changes:
                for predictor_name, change in candidate.hypothesis.prompt_changes.items():
                    summary = change.change_summary or "No summary provided"
                    magnitude = change.change_magnitude.value
                    prefix = f"  * {predictor_name} [{magnitude}]: "

                    # Word-wrap the summary
                    words = summary.split()
                    current_line = prefix

                    for word in words:
                        test_line = current_line + (" " if current_line != prefix else "") + word
                        if len(test_line) <= 61:
                            current_line = test_line
                        else:
                            lines.append(f"║ {current_line:<61} ║")
                            current_line = "    " + word

                    if current_line:
                        lines.append(f"║ {current_line:<61} ║")
    else:
        lines.append("║ No improvements found during optimization                     ║")

    lines.append("╚═══════════════════════════════════════════════════════════════╝")

    # Best hypothesis details if available
    if best_candidate.hypothesis:
        lines.append("\n╔═══════════════════════════════════════════════════════════════╗")
        lines.append("║                    Best Hypothesis Details                       ║")
        lines.append("╠═══════════════════════════════════════════════════════════════╣")
        lines.append(f"║ Strategy: {best_candidate.hypothesis.strategy[:54]:<54} ║")
        if len(best_candidate.hypothesis.strategy) > 54:
            remaining = best_candidate.hypothesis.strategy[54:]
            for i in range(0, len(remaining), 63):
                chunk = remaining[i : i + 63]
                lines.append(f"║ {chunk:<63} ║")

        lines.append(f"║ Impact Score: {best_candidate.hypothesis.impact_score:<50.2f} ║")
        lines.append(f"║ Iteration: {best_candidate.iteration:<54} ║")

        if best_candidate.hypothesis.prompt_changes:
            lines.append("╟───────────────────────────────────────────────────────────────╢")
            for predictor_name, change in best_candidate.hypothesis.prompt_changes.items():
                # Predictor header with magnitude
                magnitude = change.change_magnitude.value
                header = f"{predictor_name} [{magnitude}]"
                lines.append(f"║ {header:<61} ║")

                # Change summary
                summary = change.change_summary or "No summary provided"
                summary_prefix = "  Summary: "
                words = summary.split()
                current_line = summary_prefix

                for word in words:
                    test_line = current_line + (" " if current_line != summary_prefix else "") + word
                    if len(test_line) <= 61:
                        current_line = test_line
                    else:
                        lines.append(f"║ {current_line:<61} ║")
                        current_line = "    " + word

                if current_line:
                    lines.append(f"║ {current_line:<61} ║")

                # Full prompt (word-wrapped)
                lines.append(f"║ {'  Prompt:':<61} ║")
                prompt = change.new_prompt
                prompt_words = prompt.split()
                current_line = "    "

                for word in prompt_words:
                    test_line = current_line + (" " if len(current_line) > 4 else "") + word
                    if len(test_line) <= 61:
                        current_line = test_line
                    else:
                        lines.append(f"║ {current_line:<61} ║")
                        current_line = "    " + word

                if current_line.strip():
                    lines.append(f"║ {current_line:<61} ║")

                # Add separator if there are multiple predictors
                if len(best_candidate.hypothesis.prompt_changes) > 1:
                    lines.append("╟───────────────────────────────────────────────────────────────╢")

        lines.append("╚═══════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


__all__ = ["generate_optimization_summary"]
