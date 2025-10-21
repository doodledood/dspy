"""Program snapshot helpers for APEX."""

from __future__ import annotations

import inspect
import logging

from dspy.primitives import Module

from .models import ProgramSnapshot

logger = logging.getLogger(__name__)


def snapshot_program(program: Module) -> ProgramSnapshot:
    """Capture the structural summary of a program for analysis prompts."""

    prompts: dict[str, str] = {}
    lookup: dict[int, str] = {}

    for name, predictor in program.named_predictors():
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

    return ProgramSnapshot(
        source_code=extract_program_source(program),
        prompts=prompts,
        predictor_name_by_id=lookup,
    )


def generate_program_structure_from_introspection(program: Module) -> str:
    """Generate pseudo-code showing program structure when source is unavailable.

    This is used as fallback for notebooks, REPL, and dynamically created classes.
    """
    class_name = program.__class__.__name__
    predictors = list(program.named_predictors())

    lines = [
        "# Source not available - structure generated from introspection",
        "",
        f"class {class_name}(dspy.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    if predictors:
        for name, predictor in predictors:
            predictor_type = type(predictor).__name__
            # Try to get signature representation
            signature_repr = "..."
            if hasattr(predictor, "signature"):
                sig = predictor.signature
                if hasattr(sig, "__name__"):
                    signature_repr = f'"{sig.__name__}"'
                elif hasattr(sig, "__class__"):
                    signature_repr = f'"{sig.__class__.__name__}"'
            lines.append(f"        self.{name} = dspy.{predictor_type}({signature_repr})")
    else:
        lines.append("        pass")

    lines.extend(
        [
            "",
            "    def forward(self, **kwargs):",
            "        # Control flow not available - defined at runtime",
            "        pass",
        ]
    )

    return "\n".join(lines)


def extract_program_source(program: Module) -> str:
    """Extract program source code, falling back to introspection if unavailable.

    Tries to get actual source code from file. If not available (notebooks, REPL,
    dynamic classes), generates structure from introspection and logs warning.
    """
    try:
        source = inspect.getsource(program.__class__)
        return source.strip()
    except (TypeError, OSError):
        logger.warning(
            "Could not extract source code for %s - using introspection fallback. "
            "This typically happens in notebooks, REPL, or with dynamically created classes.",
            program.__class__.__name__,
        )
        return generate_program_structure_from_introspection(program)


__all__ = ["snapshot_program"]
