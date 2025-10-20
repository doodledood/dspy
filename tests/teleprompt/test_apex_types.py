"""Unit tests for the shared type helpers used by the APEX optimizer."""

from __future__ import annotations

import pytest

from dspy.teleprompt.apex.types import Verbosity, verbosity_rank


def test_verbosity_parse_handles_strings_and_members() -> None:
    assert Verbosity.parse(None) is Verbosity.NORMAL
    assert Verbosity.parse("silent") is Verbosity.SILENT
    assert Verbosity.parse(Verbosity.DETAILED) is Verbosity.DETAILED


def test_verbosity_parse_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        Verbosity.parse("INVALID")


def test_verbosity_rank_matches_expected_ordering() -> None:
    assert verbosity_rank(Verbosity.SILENT) == 0
    assert verbosity_rank(Verbosity.NORMAL) == 1
    assert verbosity_rank(Verbosity.DETAILED) == 2
