"""Program snapshot helpers for APEX."""

from __future__ import annotations

from dspy.primitives import Module

from .models import ProgramSnapshot


def snapshot_program(program: Module) -> ProgramSnapshot:
    """Capture the structural summary of a program for analysis prompts."""

    prompts: dict[str, str] = {}
    flow: list[str] = []
    lookup: dict[int, str] = {}

    for name, predictor in program.named_predictors():
        flow.append(name)

        # Capture the full prompt including instructions and field descriptions
        signature = predictor.signature
        prompt_parts = []

        # Add the main instructions
        instructions = getattr(signature, "instructions", "")
        if instructions:
            prompt_parts.append(f"Instructions: {instructions}")

        # Add input field descriptions
        if signature.input_fields:
            input_desc = []
            for field_name, field_info in signature.input_fields.items():
                desc = field_info.json_schema_extra.get("desc", f"${{{field_name}}}")
                input_desc.append(f"  - {field_name}: {desc}")
            prompt_parts.append("Input fields:\n" + "\n".join(input_desc))

        # Add output field descriptions
        if signature.output_fields:
            output_desc = []
            for field_name, field_info in signature.output_fields.items():
                desc = field_info.json_schema_extra.get("desc", f"${{{field_name}}}")
                output_desc.append(f"  - {field_name}: {desc}")
            prompt_parts.append("Output fields:\n" + "\n".join(output_desc))

        # Combine all parts into the full prompt
        prompts[name] = "\n\n".join(prompt_parts) if prompt_parts else "(no prompt provided)"
        lookup[id(predictor)] = name

    if not flow:
        flow_description = "No predictors"
    elif len(flow) == 1:
        flow_description = f"Single predictor: {flow[0]}"
    else:
        flow_description = "Input → " + " → ".join(flow) + " → Output"

    return ProgramSnapshot(
        structure=repr(program),
        flow_description=flow_description,
        prompts=prompts,
        predictor_name_by_id=lookup,
    )


__all__ = ["snapshot_program"]
