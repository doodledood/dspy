# dspy.APEX

**APEX** (_Analysis-based Prompt Engineering eXpert_) is a GEPA-style optimizer with additional stability guards.
Each iteration still samples a subset of your training data, records full DSPy traces, and runs a map-reduce
analysis pass before synthesizing hypotheses, but APEX augments the loop with history-aware prompts,
Pareto-frontier candidate selection (the default), and repeated calibration runs whose medians dampen noisy
metrics. These additions yield smoother performance curves while retaining GEPA's ability to explore multiple
competitive candidates per step. 【F:dspy/teleprompt/apex/apex.py†L61-L207】

You can choose between a Pareto frontier (multi-objective) sampler or a deterministic "best on validation" selector,
toggle hypothesis-history summaries, and control how frequently merge hypotheses are introduced. Combined with
checkpoints and MLflow tracking, long-running jobs remain auditable and resumable. 【F:dspy/teleprompt/apex/apex.py†L119-L207】

<!-- START_API_REF -->
::: dspy.APEX
    handler: python
    options:
        members:
            - compile
            - get_params
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->

## Key capabilities

- **Map-reduce style analysis loop.** Training batches are evaluated once, analyzed in parallel, and
  summarized into hypotheses before any prompts change. 【F:dspy/teleprompt/apex/apex.py†L77-L119】
- **History-aware exploration.** Hypothesis prompts can include summaries of prior iterations so new
  candidates build on what already worked (or failed). 【F:dspy/teleprompt/apex/apex.py†L183-L199】
- **Flexible candidate selection.** Choose Pareto-frontier sampling (default) for smoother overall gains or
  `"best_on_val"` to enforce strictly monotonic calibration scores. 【F:dspy/teleprompt/apex/apex.py†L119-L176】
- **Median-of-runs calibration.** Repeat validation executions and aggregate with the median to dampen
  metric noise when evaluating hypotheses. 【F:dspy/teleprompt/apex/apex.py†L95-L118】
- **Rich operational tooling.** Optional checkpoints and MLflow logging make it simple to resume an interrupted
  optimization or inspect iteration-level telemetry. 【F:dspy/teleprompt/apex/apex.py†L156-L207】

## Quick start

```python
import dspy
from dspy import Example
from dspy.teleprompt import APEX

def metric(example: Example, prediction: dspy.Prediction, trace):
    result = prediction.output.strip().lower() == example.output.strip().lower()
    return {"score": 1.0 if result else 0.0, "feedback": "Matches reference" if result else "Mismatch"}

apex = APEX(
    metric=metric,
    analysis_llm=dspy.LM("openai/gpt-4o", max_tokens=1200, temperature=0.0),
    hypothesis_lm=dspy.LM("openai/gpt-4o-mini", max_tokens=1800, temperature=0.2),
    max_iterations=10,
    num_hypotheses=2,
    num_eval_runs=3,
    train_sample=80,
    candidate_selection="pareto",  # default: smooth multi-objective progress
    convergence_patience=3,
)

optimized_program = apex.compile(
    student=my_program,
    trainset=train_examples,
    valset=calibration_examples,
)

print(optimized_program.apex_result.best_candidate.overall_score)
```

For a full walkthrough—complete with MLflow tracking and Pareto candidate selection—follow the
[APEX tutorial](../../tutorials/apex_optimizer.md). 【F:docs/docs/tutorials/apex_optimizer.md†L1-L140】
