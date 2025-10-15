# APEX Optimizer (`dspy.APEX`)

APEX is a teleprompter that mirrors how experienced prompt engineers improve programs:
systematically inspect failures, synthesize hypotheses that fix every actionable issue,
test each proposal on a calibration set, and advance only when the best candidate
outperforms the current baseline. The loop is intentionally simple—no Pareto frontier
or stochastic branching—so you can understand, debug, and reproduce every prompt
change that lands in your system.

[👉 Run the hands-on notebook on the PAPILLON dataset.](./apex_optimizer/index.ipynb)

## When to use APEX

- You can capture full execution traces of your DSPy program.
- The metric supplies either a scalar score or a `(score, feedback)` dictionary.
- You want interpretable hypotheses with explicit rationales and minimal prompt edits.
- Monotonic calibration performance matters—APEX never accepts regressions.

If you need evolutionary exploration or Pareto tracking, see `dspy.GEPA`. If you are
bootstrapping examples or finetuning weights, reach for `dspy.BootstrapFewShot`,
`dspy.MIPROv2`, or `dspy.BootstrapFinetune`.

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
    analysis_llm=my_analysis_llm,      # per-example root-cause analysis
    hypothesis_llm=my_hypothesis_llm,  # reducer generating hypotheses (defaults to analysis_llm)
    max_iterations=10,
    num_hypotheses=2,
    num_eval_runs=3,
    train_sample=80,                   # sample subset of trainset each iteration
    convergence_patience=3,
)

optimized_program = apex.compile(
    student=my_program,
    trainset=train_examples,
    valset=calibration_examples,  # required
)

result = optimized_program.apex_result
print(f"Best calibration score: {result.best_candidate.overall_score:.3f}")
```

The `apex_result` object stores the full history: every iteration, hypotheses,
per-example scores, and why optimization stopped (`"patience"` or
`"max_iterations"`).

## Algorithm overview

Each iteration performs:

1. **Train sampling & evaluation** – sample the trainset, run the current baseline
   once per example, record traces, metric scores, and success classification
   (>= `success_threshold`).
2. **Root cause analysis** – issue one LLM call per failure (and matching successes)
   using full traces and prompts; responses must be structured JSON.
3. **Hypothesis generation** – single synthesis call that receives all analyses +
   current prompts and returns up to `num_hypotheses` comprehensive proposals.
4. **Calibration testing** – evaluate baseline plus all hypotheses on the full
   calibration set, repeating each example `num_eval_runs` times and taking the median.
5. **Selection** – choose the candidate with the highest mean calibration score
   (ties broken randomly). Only adopt the candidate if it strictly beats the baseline.
6. **Convergence** – stop after `convergence_patience` non-improving iterations or when
   `max_iterations` is reached. Results are monotonic because APEX never accepts a
   regression.

All candidates are tracked across iterations, so you can audit how prompts evolved.

## Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `metric` | **required** | Callable returning `float` or `{"score": float, "feedback": str}`. |
| `analysis_llm` | **required** | `dspy.clients.lm.LM` used for per-example analysis (expects JSON-compatible output). |
| `analysis_adapter` | `JSONAdapter()` | Adapter used when invoking the analysis LM. |
| `hypothesis_llm` | `analysis_llm` | LM that synthesizes hypotheses (defaults to `analysis_llm`). |
| `hypothesis_adapter` | `analysis_adapter` | Adapter used when invoking the hypothesis LM. |
| `max_iterations` | **required** | Hard iteration cap for the optimization loop. |
| `num_hypotheses` | `1` | Max hypotheses generated per iteration. |
| `num_eval_runs` | `1` | Evaluation repeats per calibration example (median reduces variance). |
| `train_sample` | `20` | Number of train examples sampled per iteration. `None` = shuffle entire trainset, `int` = random subset size, `callable(trainset, iteration)` for custom sampling. |
| `success_threshold` | `max_metric` | Score threshold for classifying successes. |
| `min_metric`, `max_metric` | `0.0`, `1.0` | Bounds used to clamp metric scores. |
| `convergence_patience` | `5` | Stop after this many consecutive non-improving iterations. |
| `seed` | random | RNG seed controlling sampling and tie-breaking. |

## Configuring the analysis models

`analysis_llm` and `hypothesis_llm` should be instances of `dspy.clients.lm.LM` (or a
test double such as `DummyLM`). APEX automatically wraps each call with `JSONAdapter`,
so you only need to supply the underlying LM:

```python
import dspy

analysis_llm = dspy.LM("openai/gpt-4o", max_tokens=1200, temperature=0.0)
hypothesis_llm = dspy.LM("openai/gpt-4o-mini", max_tokens=1800, temperature=0.2)

apex = dspy.APEX(
    metric=metric,
    analysis_llm=analysis_llm,
    hypothesis_llm=hypothesis_llm,
    max_iterations=8,
    train_sample=64,
    num_eval_runs=3,
)
```

For unit tests or deterministic runs you can plug in `dspy.utils.dummies.DummyLM`,
providing structured responses that mimic the expected JSON payloads.

## Inspecting results

```python
history = optimized_program.apex_result.iterations
for iteration in history:
    print(f"Iteration {iteration.iteration}: "
          f"{len(iteration.hypotheses)} hypotheses, "
          f"best score={max(c.overall_score for c in iteration.candidates):.3f}")

best = optimized_program.apex_result.best_candidate
print(best.hypothesis.observation)
for predictor, change in best.hypothesis.prompt_changes.items():
    print(f"\nPredictor: {predictor}")
    print(change.new_prompt)
```

Use this information to audit prompt changes, reproduce analysis decisions, or feed the
history into downstream visualizations.
