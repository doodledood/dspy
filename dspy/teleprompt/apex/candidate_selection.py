"""Candidate selection helpers for APEX."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .models import CandidateRecord

CandidateSelectionStrategy = Literal["best_on_val", "pareto"]

_FLOAT_TOLERANCE = 1e-9
_ROUND_PRECISION = 12


@dataclass(frozen=True)
class SelectionResult:
    """Summary of a baseline selection."""

    baseline: CandidateRecord
    frontier: list[CandidateRecord]
    weights: list[float]


def deduplicate_candidates(candidates: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """Collapse equivalent candidates, keeping the latest iteration for each unique prompt profile."""

    unique: dict[tuple[object, ...], CandidateRecord] = {}
    for candidate in candidates:
        key = _candidate_identity(candidate)
        existing = unique.get(key)
        if existing is None or candidate.iteration >= existing.iteration:
            unique[key] = candidate

    return list(unique.values())


def select_baseline_candidate(
    *,
    candidates: Sequence[CandidateRecord],
    strategy: CandidateSelectionStrategy,
    rng: random.Random,
) -> SelectionResult:
    if not candidates:
        raise ValueError("At least one candidate is required for selection.")

    pruned_candidates = deduplicate_candidates(candidates)

    if strategy == "best_on_val":
        max_score = max(candidate.overall_score for candidate in pruned_candidates)
        best = [c for c in pruned_candidates if math.isclose(c.overall_score, max_score, rel_tol=_FLOAT_TOLERANCE)]
        baseline = rng.choice(best) if len(best) > 1 else best[0]
        return SelectionResult(baseline=baseline, frontier=[baseline], weights=[1.0])

    frontier = non_dominated_candidates(pruned_candidates)
    weights = compute_win_weights(frontier)
    baseline = draw_weighted_candidate(frontier, weights, rng=rng)
    return SelectionResult(baseline=baseline, frontier=frontier, weights=weights)


def non_dominated_candidates(candidates: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """Return Pareto non-dominated candidates based on per-example scores."""

    frontier: list[CandidateRecord] = []

    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            if _dominates(other.per_example_scores, candidate.per_example_scores):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    return frontier


def compute_win_weights(candidates: Sequence[CandidateRecord]) -> list[float]:
    """Compute per-candidate weights based on per-example wins."""

    if not candidates:
        return []

    max_len = max(len(candidate.per_example_scores) for candidate in candidates)
    wins = [0.0 for _ in candidates]

    for idx in range(max_len):
        scores = []
        for candidate in candidates:
            score = candidate.per_example_scores[idx] if idx < len(candidate.per_example_scores) else float("-inf")
            scores.append(score)
        best_score = max(scores)
        for cand_idx, score in enumerate(scores):
            if math.isclose(score, best_score, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE):
                wins[cand_idx] += 1.0

    total_wins = sum(wins)
    if total_wins <= 0:
        return [1.0 for _ in wins]
    return wins


def draw_weighted_candidate(
    candidates: Sequence[CandidateRecord],
    weights: Sequence[float],
    *,
    rng: random.Random,
    exclude: Iterable[CandidateRecord] | None = None,
) -> CandidateRecord:
    if not candidates:
        raise ValueError("Cannot draw from an empty candidate list.")

    if len(candidates) != len(weights):
        raise ValueError("Candidates and weights must have the same length.")

    excluded_ids = {id(candidate) for candidate in (exclude or [])}
    pool: list[tuple[CandidateRecord, float]] = []
    for candidate, weight in zip(candidates, weights, strict=True):
        if id(candidate) in excluded_ids:
            continue
        pool.append((candidate, max(weight, 0.0)))

    if not pool:
        raise ValueError("Excluded every candidate; cannot draw a weighted selection.")

    weights_only = [weight for _, weight in pool]
    total = sum(weights_only)
    if total <= 0:
        return rng.choice([candidate for candidate, _ in pool])

    pick = rng.uniform(0.0, total)
    cumulative = 0.0
    for candidate, weight in pool:
        cumulative += weight
        if pick <= cumulative:
            return candidate

    return pool[-1][0]


def _dominates(values_a: Sequence[float], values_b: Sequence[float]) -> bool:
    if not values_a and values_b:
        return False
    if not values_b and values_a:
        return True

    any_strictly_better = False
    max_len = max(len(values_a), len(values_b))
    for idx in range(max_len):
        a = values_a[idx] if idx < len(values_a) else float("-inf")
        b = values_b[idx] if idx < len(values_b) else float("-inf")
        if a + _FLOAT_TOLERANCE < b:
            return False
        if a > b + _FLOAT_TOLERANCE:
            any_strictly_better = True
    return any_strictly_better


def _candidate_identity(candidate: CandidateRecord) -> tuple[object, ...]:
    if candidate.per_example_scores:
        score_key = tuple(round(score, _ROUND_PRECISION) for score in candidate.per_example_scores)
    else:
        score_key = (round(candidate.overall_score, _ROUND_PRECISION),)

    if candidate.hypothesis is None:
        prompt_key: tuple[object, ...] = ("baseline",)
    else:
        strategy = getattr(candidate.hypothesis, "strategy", "")
        prompt_changes = getattr(candidate.hypothesis, "prompt_changes", {}) or {}
        normalized_changes = tuple(
            sorted(
                (
                    predictor_name,
                    change.new_prompt,
                    change.change_summary or "",
                    change.change_magnitude.value,
                )
                for predictor_name, change in prompt_changes.items()
            )
        )
        prompt_key = ("hypothesis", strategy, normalized_changes)

    return (prompt_key, score_key)


def candidates_are_equivalent(a: CandidateRecord, b: CandidateRecord) -> bool:
    """Return True if two candidates represent the same underlying prompt configuration."""

    return _candidate_identity(a) == _candidate_identity(b)


__all__ = [
    "CandidateSelectionStrategy",
    "SelectionResult",
    "deduplicate_candidates",
    "compute_win_weights",
    "draw_weighted_candidate",
    "non_dominated_candidates",
    "select_baseline_candidate",
    "candidates_are_equivalent",
]
