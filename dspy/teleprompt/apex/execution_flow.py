"""Utilities for extracting and formatting APEX execution flow data."""

from __future__ import annotations

import json
from typing import Any

from dspy.primitives import Example, Module, Prediction

from .models import ExecutionFlowEntry
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
        predictor_name = "unknown"
        predictor_type = type(predictor_obj).__name__

        for name, pred in predictor_lookup.items():
            if pred is predictor_obj:
                predictor_name = name
                break

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


def format_execution_flow_as_graph(execution_flow: list[ExecutionFlowEntry]) -> str:
    if not execution_flow:
        return "No execution flow available"

    if len(execution_flow) == 1:
        entry = execution_flow[0]
        return (
            "Program DAG:\n"
            "  Input\n"
            f"    ↳ {entry.predictor_name}\n"
            f"  {entry.predictor_name} ({entry.predictor_type})\n"
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
        flow_lines.append(f"  {entry.predictor_name} ({entry.predictor_type})")
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


def format_execution_flow_with_details(execution_flow: list[ExecutionFlowEntry]) -> str:
    if not execution_flow:
        return "No execution flow available"

    flow_parts: list[str] = []

    flow_parts.append(format_execution_flow_as_graph(execution_flow))
    flow_parts.append("\nPredictor Instructions and Data Flow:")

    for idx, entry in enumerate(execution_flow, start=1):
        instructions = entry.instructions if entry.instructions else "No instructions"
        dependencies = ", ".join(entry.dependencies) if entry.dependencies else "Input"
        flow_parts.append(
            f"\n{idx}. {entry.predictor_name} ({entry.predictor_type}):\n"
            f"   Instructions: {instructions}\n"
            f"   Depends on: {dependencies}"
        )

        if entry.input_sources:
            flow_parts.append("   Inputs sourced from:")
            for input_name, sources in entry.input_sources.items():
                flow_parts.append(f"     - {input_name}: {', '.join(sources)}")
        else:
            flow_parts.append("   Inputs sourced from: program input or constants")

        flow_parts.append(f"   Actual inputs: {entry.inputs}")
        flow_parts.append(f"   Actual outputs: {entry.outputs}")

    return "\n".join(flow_parts)
