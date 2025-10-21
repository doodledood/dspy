"""Unit tests for APEX execution flow extraction and formatting."""

from __future__ import annotations

import pytest

import dspy
from dspy.teleprompt.apex.execution_flow import (
    extract_full_execution_flow_with_coverage,
    format_execution_flow_as_graph,
    format_execution_flow_with_details,
)
from dspy.teleprompt.apex.models import ExecutionFlowEntry
from dspy.utils.dummies import DummyLM


@pytest.fixture(autouse=True)
def configure_lm():
    """Configure a DummyLM for all tests."""
    # Provide enough responses for multi-predictor tests
    responses = [{"intermediate": "step1", "output": "step2", "final": "step3", "answer": "result"} for _ in range(10)]
    lm = DummyLM(responses)
    dspy.configure(lm=lm)
    yield
    dspy.configure(lm=None)


class SimpleModule(dspy.Module):
    """Module with a single predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict("input -> output")
        self.predictor.signature.instructions = "Process the input"

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        return self.predictor(input=input)


class ConditionalModule(dspy.Module):
    """Module with conditional execution based on input."""

    def __init__(self) -> None:
        super().__init__()
        self.route_a = dspy.Predict("query -> answer")
        self.route_b = dspy.Predict("query -> answer")
        self.route_a.signature.instructions = "Handle route A queries"
        self.route_b.signature.instructions = "Handle route B queries"

    def forward(self, query: str, use_route_a: bool = True) -> dspy.Prediction:  # type: ignore[override]
        if use_route_a:
            return self.route_a(query=query)
        else:
            return self.route_b(query=query)


class MultiPredictorModule(dspy.Module):
    """Module with multiple predictors in sequence."""

    def __init__(self) -> None:
        super().__init__()
        self.first = dspy.Predict("input -> intermediate")
        self.second = dspy.Predict("intermediate -> output")
        self.third = dspy.Predict("output -> final")
        self.first.signature.instructions = "First step"
        self.second.signature.instructions = "Second step"
        self.third.signature.instructions = "Third step"

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        result1 = self.first(input=input)
        result2 = self.second(intermediate=result1.intermediate)
        result3 = self.third(output=result2.output)
        return result3


def test_extract_full_flow_all_predictors_executed() -> None:
    """Test extraction when all predictors are executed."""
    program = MultiPredictorModule()

    with dspy.settings.context(trace=[]):
        _ = program(input="test")
        trace = list(dspy.settings.trace or [])

    flow = extract_full_execution_flow_with_coverage(trace, program)

    assert len(flow) == 3
    for idx, entry in enumerate(flow):
        assert entry.executed is True
        assert entry.execution_order == idx
        assert entry.inputs != "[not executed]"
        assert entry.outputs != "[not executed]"

    assert flow[0].predictor_name == "first"
    assert flow[1].predictor_name == "second"
    assert flow[2].predictor_name == "third"


def test_extract_full_flow_with_conditional_execution() -> None:
    """Test extraction with conditional branches (some predictors not executed)."""
    program = ConditionalModule()

    # Execute only route_a
    with dspy.settings.context(trace=[]):
        _ = program(query="test", use_route_a=True)
        trace = list(dspy.settings.trace or [])

    flow = extract_full_execution_flow_with_coverage(trace, program)

    assert len(flow) == 2

    # First entry should be executed (route_a)
    executed_entry = flow[0]
    assert executed_entry.predictor_name == "route_a"
    assert executed_entry.executed is True
    assert executed_entry.execution_order == 0
    assert executed_entry.inputs != "[not executed]"
    assert executed_entry.outputs != "[not executed]"
    assert "Handle route A queries" in executed_entry.instructions

    # Second entry should be not executed (route_b)
    not_executed_entry = flow[1]
    assert not_executed_entry.predictor_name == "route_b"
    assert not_executed_entry.executed is False
    assert not_executed_entry.execution_order is None
    assert not_executed_entry.inputs == "[not executed]"
    assert not_executed_entry.outputs == "[not executed]"
    assert "Handle route B queries" in not_executed_entry.instructions


def test_extract_full_flow_no_predictors_executed() -> None:
    """Test extraction when trace is empty but program has predictors."""
    program = MultiPredictorModule()
    empty_trace = []

    flow = extract_full_execution_flow_with_coverage(empty_trace, program)

    assert len(flow) == 3
    for entry in flow:
        assert entry.executed is False
        assert entry.execution_order is None
        assert entry.inputs == "[not executed]"
        assert entry.outputs == "[not executed]"
        assert entry.instructions  # Instructions should still be present


def test_extract_full_flow_single_predictor() -> None:
    """Test extraction with a single predictor program."""
    program = SimpleModule()

    with dspy.settings.context(trace=[]):
        _ = program(input="test")
        trace = list(dspy.settings.trace or [])

    flow = extract_full_execution_flow_with_coverage(trace, program)

    assert len(flow) == 1
    entry = flow[0]
    assert entry.predictor_name == "predictor"
    assert entry.executed is True
    assert entry.execution_order == 0
    assert "Process the input" in entry.instructions


def test_format_graph_with_execution_labels() -> None:
    """Test graph formatting includes execution status labels."""
    # Create mixed execution flow
    entries = [
        ExecutionFlowEntry(
            predictor_name="executed_pred",
            predictor_type="Predict",
            inputs='{"x": 1}',
            outputs='{"y": 2}',
            instructions="Executed predictor",
            executed=True,
            execution_order=0,
        ),
        ExecutionFlowEntry(
            predictor_name="not_executed_pred",
            predictor_type="ChainOfThought",
            inputs="[not executed]",
            outputs="[not executed]",
            instructions="Not executed predictor",
            executed=False,
            execution_order=None,
        ),
    ]

    graph = format_execution_flow_as_graph(entries)

    assert "[executed]" in graph
    assert "[not executed]" in graph
    assert "executed_pred (Predict) [executed]" in graph
    assert "not_executed_pred (ChainOfThought) [not executed]" in graph


def test_format_details_with_execution_coverage() -> None:
    """Test detailed formatting handles executed vs non-executed predictors correctly."""
    entries = [
        ExecutionFlowEntry(
            predictor_name="active",
            predictor_type="Predict",
            inputs='{"input": "test"}',
            outputs='{"output": "result"}',
            instructions="Active predictor",
            dependencies=["input"],
            input_sources={"input": ["previous"]},
            executed=True,
            execution_order=0,
        ),
        ExecutionFlowEntry(
            predictor_name="inactive",
            predictor_type="Predict",
            inputs="[not executed]",
            outputs="[not executed]",
            instructions="Inactive predictor",
            executed=False,
            execution_order=None,
        ),
    ]

    details = format_execution_flow_with_details(entries)

    # Check executed predictor shows real input sources
    assert "active (Predict) [executed]" in details
    assert "Inputs sourced from:" in details
    assert "input: previous" in details or "input:" in details
    assert "Active predictor" in details

    # Check non-executed predictor shows [not executed] for input sources
    assert "inactive (Predict) [not executed]" in details
    assert "Inputs sourced from: [not executed]" in details
    assert "Inactive predictor" in details
    assert details.count("[not executed]") >= 3  # inputs, outputs, and input sources


def test_extract_full_flow_empty_trace_empty_program() -> None:
    """Test extraction with empty trace and empty program."""

    class EmptyModule(dspy.Module):
        def forward(self, **kwargs):  # type: ignore[override]
            return dspy.Prediction(**kwargs)

    program = EmptyModule()
    empty_trace = []

    flow = extract_full_execution_flow_with_coverage(empty_trace, program)

    assert len(flow) == 0


def test_extract_full_flow_empty_trace_with_predictors() -> None:
    """Test extraction with empty trace but program has predictors."""
    program = MultiPredictorModule()
    empty_trace = []

    flow = extract_full_execution_flow_with_coverage(empty_trace, program)

    assert len(flow) == 3
    for entry in flow:
        assert entry.executed is False
        assert entry.execution_order is None
        assert entry.inputs == "[not executed]"
        assert entry.outputs == "[not executed]"
        assert entry.instructions  # Instructions should be present


def test_extract_full_flow_preserves_execution_order() -> None:
    """Test that execution_order is correctly assigned and preserved."""
    program = MultiPredictorModule()

    with dspy.settings.context(trace=[]):
        _ = program(input="test")
        trace = list(dspy.settings.trace or [])

    flow = extract_full_execution_flow_with_coverage(trace, program)

    # Verify execution order is sequential for executed predictors
    executed_entries = [e for e in flow if e.executed]
    assert len(executed_entries) == 3
    assert executed_entries[0].execution_order == 0
    assert executed_entries[1].execution_order == 1
    assert executed_entries[2].execution_order == 2

    # Verify non-executed predictors (if any) have execution_order=None
    non_executed_entries = [e for e in flow if not e.executed]
    for entry in non_executed_entries:
        assert entry.execution_order is None


def test_format_graph_single_predictor_not_executed() -> None:
    """Test graph formatting for single non-executed predictor."""
    entries = [
        ExecutionFlowEntry(
            predictor_name="unused",
            predictor_type="Predict",
            inputs="[not executed]",
            outputs="[not executed]",
            instructions="Never called",
            executed=False,
            execution_order=None,
        )
    ]

    graph = format_execution_flow_as_graph(entries)

    assert "[not executed]" in graph
    assert "unused (Predict) [not executed]" in graph


def test_format_details_empty_flow() -> None:
    """Test details formatting with empty flow."""
    empty_flow = []

    details = format_execution_flow_with_details(empty_flow)

    assert details == "No execution flow available"
