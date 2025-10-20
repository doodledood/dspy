"""Unit tests for the public candidate selection helpers used by APEX."""

from __future__ import annotations

import random

import dspy
from dspy.teleprompt.apex.candidate_selection import (
    CandidateSelectionStrategy,
    SelectionResult,
    compute_win_weights,
    deduplicate_candidates,
    draw_weighted_candidate,
    select_baseline_candidate,
)
from dspy.teleprompt.apex.models import CandidateRecord


class _StaticModule(dspy.Module):
    """Minimal module used to populate ``CandidateRecord.program``."""

    def __init__(self, value: str) -> None:
        super().__init__()
        self._value = value

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return dspy.Prediction(output=self._value)


_DEF_PER_EXAMPLE = [0.2, 0.4, 0.6]


def _make_candidate(iteration: int, *, overall: float, label: str) -> CandidateRecord:
    return CandidateRecord(
        program=_StaticModule(label),
        overall_score=overall,
        per_example_scores=list(_DEF_PER_EXAMPLE),
        iteration=iteration,
        hypothesis=None,
    )


def test_deduplicate_candidates_prefers_latest_iteration() -> None:
    """Candidates with the same identity should collapse to the most recent iteration."""

    older = _make_candidate(1, overall=0.5, label="same")
    newer = _make_candidate(3, overall=0.5, label="same")

    pruned = deduplicate_candidates([older, newer, older])

    assert pruned == [newer]


def test_select_baseline_candidate_best_on_val_returns_selection_result() -> None:
    candidates = [
        _make_candidate(0, overall=0.8, label="first"),
        _make_candidate(1, overall=0.8, label="second"),
    ]
    rng = random.Random(0)

    result = select_baseline_candidate(
        candidates=candidates,
        strategy=CandidateSelectionStrategy.__args__[0],  # "best_on_val"
        rng=rng,
    )

    assert isinstance(result, SelectionResult)
    assert result.baseline in candidates
    assert result.frontier == [result.baseline]
    assert result.weights == [1.0]


def test_select_baseline_candidate_pareto_strategy_respects_frontier() -> None:
    better_first = CandidateRecord(
        program=_StaticModule("better-first"),
        overall_score=0.6,
        per_example_scores=[1.0, 0.0],
        iteration=1,
        hypothesis=None,
    )
    better_second = CandidateRecord(
        program=_StaticModule("better-second"),
        overall_score=0.6,
        per_example_scores=[0.0, 1.0],
        iteration=1,
        hypothesis=None,
    )
    dominated = CandidateRecord(
        program=_StaticModule("dominated"),
        overall_score=0.4,
        per_example_scores=[0.0, 0.0],
        iteration=1,
        hypothesis=None,
    )
    rng = random.Random(4)

    result = select_baseline_candidate(
        candidates=[better_first, better_second, dominated],
        strategy="pareto",
        rng=rng,
    )

    frontier_ids = {id(candidate) for candidate in result.frontier}
    assert frontier_ids == {id(better_first), id(better_second)}
    assert len(result.weights) == 2
    assert result.baseline in result.frontier


def test_compute_win_weights_counts_per_example_wins() -> None:
    candidates = [
        CandidateRecord(
            program=_StaticModule("win"),
            overall_score=1.0,
            per_example_scores=[1.0, 1.0],
            iteration=0,
            hypothesis=None,
        ),
        CandidateRecord(
            program=_StaticModule("loss"),
            overall_score=0.0,
            per_example_scores=[0.0, 0.5],
            iteration=0,
            hypothesis=None,
        ),
    ]

    weights = compute_win_weights(candidates)

    assert weights == [2.0, 0.0]


def test_draw_weighted_candidate_supports_exclusions() -> None:
    included = _make_candidate(0, overall=0.6, label="included")
    excluded = _make_candidate(0, overall=0.9, label="excluded")
    rng = random.Random(0)

    selection = draw_weighted_candidate(
        [included, excluded],
        [1.0, 10.0],
        rng=rng,
        exclude=[excluded],
    )

    assert selection is included
