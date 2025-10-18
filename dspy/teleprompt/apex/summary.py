"""Summary table generation for APEX optimization results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ApexIterationLog, CandidateRecord


def generate_optimization_summary(
    iterations: list[ApexIterationLog],
    best_candidate: CandidateRecord,
    initial_score: float,
    selection_strategy: str = "best_on_val",
    pareto_merge_probability: float | None = None,
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
    lines.append(f"║ Selection Strategy: {selection_strategy:<41} ║")
    if pareto_merge_probability is not None:
        lines.append(f"║ Pareto Merge Probability: {pareto_merge_probability:<32.2f} ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    # Score progress
    lines.append(f"║ Initial Score: {initial_score:<50.4f} ║")
    lines.append(f"║ Final Score: {best_candidate.overall_score:<52.4f} ║")
    improvement_str = f"{'+' if improvement >= 0 else ''}{improvement:.4f}"
    lines.append(f"║ Improvement: {improvement_str:<52} ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    frame_width = len("╔═══════════════════════════════════════════════════════════════╗")
    content_width = frame_width - 4
    iter_width = 4
    candidate_width = 13
    score_width = 8
    delta_width = 8
    summary_width = 18

    def wrap_text(text: str, width: int) -> list[str]:
        if not text:
            return [""]
        words = text.split()
        if not words:
            return [""]
        wrapped: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                if len(word) > width:
                    for start in range(0, len(word), width):
                        wrapped.append(word[start : start + width])
                    current = ""
                else:
                    current = word
        if current:
            wrapped.append(current)
        return wrapped or [""]

    def build_row(iter_text: str, candidate_text: str, score_text: str, delta_text: str, summary_text: str) -> str:
        summary_text = summary_text[:summary_width]
        columns = [
            f"{iter_text:^{iter_width}}",
            f"{candidate_text:<{candidate_width}}",
            f"{score_text:^{score_width}}",
            f"{delta_text:^{delta_width}}",
            f"{summary_text:<{summary_width}}",
        ]
        content = " │ ".join(columns)
        content = content[:content_width]
        row = f"║ {content}"
        padding = frame_width - len(row) - 1
        if padding < 0:
            row = row[: frame_width - 1]
            padding = 0
        row += " " * padding
        row += "║"
        return row

    def build_separator(char: str, junction: str, left: str, right: str) -> str:
        segments = [
            char * (iter_width + 2),
            char * (candidate_width + 2),
            char * (score_width + 2),
            char * (delta_width + 2),
            char * (summary_width + 2),
        ]
        content = junction.join(segments)
        line = f"{left}{content}"
        padding = frame_width - len(line) - 1
        if padding > 0:
            line += char * padding
        line += right
        return line

    header = build_row("Iter", "Candidate", "Score", "Δ vs best", "Prompt Summary")
    separator = build_separator("─", "┼", "╟", "╢")
    footer = build_separator("═", "╧", "╚", "╝")

    lines.append(header)
    lines.append(separator)

    best_so_far = initial_score
    first_iteration = True

    for iteration in iterations:
        if not iteration.candidates:
            if not first_iteration:
                lines.append(separator)
            first_iteration = False
            lines.append(
                build_row(
                    str(iteration.iteration),
                    "baseline",
                    f"{best_so_far:.4f}",
                    "+0.0000",
                    "No candidates evaluated",
                )
            )
            continue

        if not first_iteration:
            lines.append(separator)
        first_iteration = False

        for idx, candidate in enumerate(iteration.candidates):
            score = candidate.overall_score
            delta = score - best_so_far
            delta_text = f"{delta:+.4f}"
            if score > best_so_far:
                best_so_far = score

            candidate_label = "baseline" if idx == 0 else f"hyp #{idx}"
            base_summary: list[str] = []

            if idx == 0:
                base_summary.append(f"Train F/S: {iteration.num_failures}/{iteration.num_successes}")
                base_summary.append(f"Hyp eval: {len(iteration.hypotheses)}")
                base_summary.append("Original program")
            else:
                hypothesis = candidate.hypothesis
                if hypothesis is None:
                    base_summary.append("No hypothesis metadata")
                else:
                    base_summary.append(hypothesis.strategy or "Unknown strategy")
                    if hypothesis.prompt_changes:
                        for predictor_name, change in hypothesis.prompt_changes.items():
                            summary = change.change_summary or "No summary provided"
                            base_summary.append(f"{predictor_name}: {summary}")
                    else:
                        base_summary.append("No prompt changes")

            summary_lines: list[str] = []
            for entry in base_summary:
                summary_lines.extend(wrap_text(entry, summary_width))

            if not summary_lines:
                summary_lines = [""]

            for line_index, summary_line in enumerate(summary_lines):
                lines.append(
                    build_row(
                        str(candidate.iteration) if line_index == 0 else "",
                        candidate_label[:candidate_width] if line_index == 0 else "",
                        f"{score:.4f}" if line_index == 0 else "",
                        delta_text if line_index == 0 else "",
                        summary_line,
                    )
                )

    lines.append(footer)

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
