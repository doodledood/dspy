"""Utilities for extracting and formatting APEX execution flow data."""

from __future__ import annotations

import json
from typing import Any

from dspy.primitives import Example, Module, Prediction

from .models import ExecutionFlowEntry
from .snapshot import extract_program_source
from .types import TraceEntry


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Prediction | Example):
        return {k: _normalize_value(v) for k, v in value.toDict().items()}
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_value(v) for v in value)
    return value


def _value_key(value: Any) -> str:
    normalized = _normalize_value(value)
    try:
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(normalized)


def _stringify(value: Any) -> str:
    normalized = _normalize_value(value)
    try:
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(normalized)


def extract_execution_flow(trace: list[TraceEntry], program: Module) -> list[ExecutionFlowEntry]:
    execution_flow: list[ExecutionFlowEntry] = []
    predictor_lookup = dict(program.named_predictors())

    value_sources_by_field: dict[str, list[str]] = {}
    value_sources_by_value: dict[str, list[str]] = {}

    for predictor_obj, inputs, outputs in trace:
        predictor_name = None
        predictor_type = type(predictor_obj).__name__

        # First check if the predictor has a stored name (TrackedPredictor)
        if hasattr(predictor_obj, "_predictor_name"):
            predictor_name = predictor_obj._predictor_name
        else:
            # Try identity-based lookup
            trace_predictor = predictor_obj
            if hasattr(predictor_obj, "_wrapped_predictor"):
                trace_predictor = predictor_obj._wrapped_predictor

            for name, pred in predictor_lookup.items():
                lookup_predictor = pred
                if hasattr(pred, "_wrapped_predictor"):
                    lookup_predictor = pred._wrapped_predictor

                if pred is predictor_obj or lookup_predictor is trace_predictor:
                    predictor_name = name
                    break

            # If identity lookup fails and there's only one predictor, use its name
            # This handles the deepcopy case where identity is lost
            if predictor_name is None and len(predictor_lookup) == 1:
                predictor_name = next(iter(predictor_lookup))

        # This should NEVER happen - if it does, it's a bug
        if predictor_name is None:
            raise ValueError(
                f"Failed to resolve predictor name for {predictor_type} in execution flow. "
                f"This is a bug in APEX - predictor identity was lost (likely due to deepcopy). "
                f"Trace contains {predictor_type} at {id(predictor_obj)}, "
                f"but program.named_predictors() has: {list(predictor_lookup.keys())} "
                f"at ids: {[id(p) for p in predictor_lookup.values()]}. "
                f"This causes APEX to fail with confusing 'unknown predictor' errors."
            )

        instructions = ""
        if hasattr(predictor_obj, "signature") and hasattr(predictor_obj.signature, "instructions"):
            instructions = predictor_obj.signature.instructions

        normalized_inputs = {str(k): _normalize_value(v) for k, v in dict(inputs).items()}

        normalized_outputs: dict[str, Any] = {}
        if isinstance(outputs, Prediction | Example):
            normalized_outputs = {str(k): _normalize_value(v) for k, v in outputs.toDict().items()}
        elif outputs is not None:
            normalized_outputs = {"value": _normalize_value(outputs)}

        dependencies: set[str] = set()
        input_sources: dict[str, list[str]] = {}

        for input_name, input_value in normalized_inputs.items():
            key = _value_key(input_value)
            field_key = f"{input_name}::{key}"
            source_candidates: list[str] = []

            if value_sources_by_field.get(field_key):
                source_candidates = [value_sources_by_field[field_key][-1]]
            elif value_sources_by_value.get(key):
                source_candidates = [value_sources_by_value[key][-1]]

            if source_candidates:
                unique_sources = list(dict.fromkeys(source_candidates))
                dependencies.update(unique_sources)
                input_sources[input_name] = unique_sources

        execution_flow.append(
            ExecutionFlowEntry(
                predictor_name=predictor_name,
                predictor_type=predictor_type,
                inputs=_stringify(normalized_inputs),
                outputs=_stringify(normalized_outputs) if normalized_outputs else "{}",
                instructions=instructions,
                dependencies=sorted(dependencies),
                input_sources={k: sorted(v) for k, v in sorted(input_sources.items())},
            )
        )

        for output_name, output_value in normalized_outputs.items():
            key = _value_key(output_value)
            field_key = f"{output_name}::{key}"
            value_sources_by_field.setdefault(field_key, []).append(predictor_name)
            value_sources_by_value.setdefault(key, []).append(predictor_name)

    return execution_flow


def extract_full_execution_flow_with_coverage(trace: list[TraceEntry], program: Module) -> list[ExecutionFlowEntry]:
    """Extract full program structure with execution coverage overlay.

    Returns all predictors in the program, marking which were executed and which were not.
    Executed predictors include actual I/O data and dependencies; non-executed predictors
    show their instructions but mark I/O as '[not executed]'.
    """
    # Get executed predictors with full data
    executed_flow = extract_execution_flow(trace, program)

    # Build map of executed predictor names and assign execution order
    executed_names: set[str] = set()
    for idx, entry in enumerate(executed_flow):
        executed_names.add(entry.predictor_name)
        entry.execution_order = idx

    # Get all predictors from the program
    all_predictors = dict(program.named_predictors())

    # Create entries for non-executed predictors
    non_executed_flow: list[ExecutionFlowEntry] = []
    for predictor_name, predictor_obj in all_predictors.items():
        if predictor_name not in executed_names:
            predictor_type = type(predictor_obj).__name__

            # Extract instructions if available
            instructions = ""
            if hasattr(predictor_obj, "signature") and hasattr(predictor_obj.signature, "instructions"):
                instructions = predictor_obj.signature.instructions

            non_executed_flow.append(
                ExecutionFlowEntry(
                    predictor_name=predictor_name,
                    predictor_type=predictor_type,
                    inputs="[not executed]",
                    outputs="[not executed]",
                    instructions=instructions,
                    dependencies=[],
                    input_sources={},
                    executed=False,
                    execution_order=None,
                )
            )

    # Combine: executed first (sorted by execution_order), then non-executed (alphabetically)
    combined = executed_flow + sorted(non_executed_flow, key=lambda e: e.predictor_name)
    return combined


def format_execution_flow_as_graph(execution_flow: list[ExecutionFlowEntry]) -> str:
    if not execution_flow:
        return "No execution flow available"

    if len(execution_flow) == 1:
        entry = execution_flow[0]
        exec_status = "[executed]" if entry.executed else "[not executed]"
        return (
            "Program DAG:\n"
            "  Input\n"
            f"    ↳ {entry.predictor_name}\n"
            f"  {entry.predictor_name} ({entry.predictor_type}) {exec_status}\n"
            "    depends on: Input\n"
            "    feeds: Output"
        )

    flow_lines: list[str] = ["Program DAG:"]

    children_map: dict[str, set[str]] = {entry.predictor_name: set() for entry in execution_flow}
    root_nodes: list[str] = []

    for entry in execution_flow:
        if entry.dependencies:
            for dependency in entry.dependencies:
                children_map.setdefault(dependency, set()).add(entry.predictor_name)
        else:
            root_nodes.append(entry.predictor_name)

    if root_nodes:
        flow_lines.append("  Input")
        flow_lines.append(f"    ↳ {', '.join(sorted(root_nodes))}")
    else:
        flow_lines.append("  Input (no predictors depend directly on program input)")

    for entry in execution_flow:
        exec_status = "[executed]" if entry.executed else "[not executed]"
        flow_lines.append(f"  {entry.predictor_name} ({entry.predictor_type}) {exec_status}")
        if entry.dependencies:
            flow_lines.append(f"    depends on: {', '.join(entry.dependencies)}")
        else:
            flow_lines.append("    depends on: Input")

        children = sorted(children_map.get(entry.predictor_name, set()))
        if children:
            flow_lines.append(f"    feeds: {', '.join(children)}")
        else:
            flow_lines.append("    feeds: Output")

    return "\n".join(flow_lines)


def format_execution_flow_with_details(execution_flow: list[ExecutionFlowEntry], program: Module | None = None) -> str:
    if not execution_flow:
        return "No execution flow available"

    flow_parts: list[str] = []

    # Show program source code first if available
    if program is not None:
        source_code = extract_program_source(program)
        flow_parts.extend(
            [
                "Program Source Code:",
                "```python",
                '"""',
                source_code,
                '"""',
                "```",
                "",
            ]
        )

    flow_parts.append(format_execution_flow_as_graph(execution_flow))
    flow_parts.append("\nPredictor Instructions and Data Flow:")

    for idx, entry in enumerate(execution_flow, start=1):
        instructions = entry.instructions if entry.instructions else "No instructions"
        dependencies = ", ".join(entry.dependencies) if entry.dependencies else "Input"
        exec_status = "[executed]" if entry.executed else "[not executed]"
        flow_parts.append(
            f"\n{idx}. {entry.predictor_name} ({entry.predictor_type}) {exec_status}:\n"
            f'   Instructions: """\n{instructions}\n"""\n'
            f"   Depends on: {dependencies}"
        )

        if entry.executed:
            if entry.input_sources:
                flow_parts.append("   Inputs sourced from:")
                for input_name, sources in entry.input_sources.items():
                    flow_parts.append(f"     - {input_name}: {', '.join(sources)}")
            else:
                flow_parts.append("   Inputs sourced from: program input or constants")
        else:
            flow_parts.append("   Inputs sourced from: [not executed]")

        flow_parts.append(f'   Actual inputs: """\n{entry.inputs}\n"""')
        flow_parts.append(f'   Actual outputs: """\n{entry.outputs}\n"""')

    return "\n".join(flow_parts)
