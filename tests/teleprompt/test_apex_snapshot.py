"""Unit tests for APEX snapshot utilities."""

from __future__ import annotations

import pytest

import dspy
from dspy.teleprompt.apex.snapshot import (
    extract_program_source,
    generate_program_structure_from_introspection,
    snapshot_program,
)
from dspy.utils.dummies import DummyLM


@pytest.fixture(autouse=True)
def configure_lm():
    """Configure a DummyLM for all tests."""
    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)
    yield
    dspy.configure(lm=None)


class NormalModule(dspy.Module):
    """Normal module defined in file - source available."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        return self.predictor(input=input)


def test_extract_program_source_with_file_source() -> None:
    """Test that source extraction works for normal modules defined in files."""
    program = NormalModule()
    source = extract_program_source(program)

    # Should get exact source code
    expected = '''class NormalModule(dspy.Module):
    """Normal module defined in file - source available."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        return self.predictor(input=input)'''

    assert source == expected
    # Should NOT have introspection fallback comment
    assert "# Source not available" not in source


def test_extract_program_source_with_dynamic_class() -> None:
    """Test that source extraction falls back to introspection for dynamic classes."""
    # Create dynamic class using type() - source unavailable
    DynamicModule = type(  # noqa: N806
        "DynamicModule",
        (dspy.Module,),
        {
            "__init__": lambda self: (
                super(DynamicModule, self).__init__(),
                setattr(self, "pred", dspy.Predict("x -> y")),
            ),
        },
    )
    program = DynamicModule()

    source = extract_program_source(program)

    # Should have exact introspection fallback format
    expected = """# Source not available - structure generated from introspection

class DynamicModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.Predict("StringSignature")

    def forward(self, **kwargs):
        # Control flow not available - defined at runtime
        pass"""

    assert source == expected
    # Note: Warning is logged (visible in test output) but not easily testable with caplog


def test_generate_program_structure_from_introspection() -> None:
    """Test introspection-based structure generation."""

    class TestModule(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = dspy.Predict("a -> b")
            self.second = dspy.Predict("b -> c")

        def forward(self, a: str) -> dspy.Prediction:  # type: ignore[override]
            result = self.first(a=a)
            return self.second(b=result.b)

    program = TestModule()
    source = generate_program_structure_from_introspection(program)

    # Should have exact introspection format
    expected = """# Source not available - structure generated from introspection

class TestModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.first = dspy.Predict("StringSignature")
        self.second = dspy.Predict("StringSignature")

    def forward(self, **kwargs):
        # Control flow not available - defined at runtime
        pass"""

    assert source == expected


def test_generate_program_structure_no_predictors() -> None:
    """Test introspection with module that has no predictors."""

    class EmptyModule(dspy.Module):
        def forward(self, **kwargs):  # type: ignore[override]
            return dspy.Prediction(**kwargs)

    program = EmptyModule()
    source = generate_program_structure_from_introspection(program)

    # Should have exact format with pass statement
    expected = """# Source not available - structure generated from introspection

class EmptyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, **kwargs):
        # Control flow not available - defined at runtime
        pass"""

    assert source == expected


def test_snapshot_program_with_source_code() -> None:
    """Test that snapshot_program includes source_code field."""
    program = NormalModule()
    snapshot = snapshot_program(program)

    # Should have source_code field (not structure or flow_description)
    assert hasattr(snapshot, "source_code")
    assert hasattr(snapshot, "prompts")
    assert hasattr(snapshot, "predictor_name_by_id")
    assert not hasattr(snapshot, "structure")
    assert not hasattr(snapshot, "flow_description")

    # Source code should be exact
    expected_source = '''class NormalModule(dspy.Module):
    """Normal module defined in file - source available."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, input: str) -> dspy.Prediction:  # type: ignore[override]
        return self.predictor(input=input)'''

    assert snapshot.source_code == expected_source

    # Should have predictor prompts
    assert "predictor" in snapshot.prompts


def test_snapshot_program_with_dynamic_class() -> None:
    """Test snapshot_program with dynamic class uses introspection."""
    DynamicModule = type(  # noqa: N806
        "DynamicTestModule",
        (dspy.Module,),
        {
            "__init__": lambda self: (
                super(DynamicModule, self).__init__(),
                setattr(self, "pred", dspy.Predict("x -> y")),
            ),
        },
    )
    program = DynamicModule()
    snapshot = snapshot_program(program)

    # Should have exact introspection fallback format
    expected_source = """# Source not available - structure generated from introspection

class DynamicTestModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.Predict("StringSignature")

    def forward(self, **kwargs):
        # Control flow not available - defined at runtime
        pass"""

    assert snapshot.source_code == expected_source

    # Should still have prompts
    assert "pred" in snapshot.prompts
