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
        prompts[name] = getattr(predictor.signature, "instructions", "")
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
