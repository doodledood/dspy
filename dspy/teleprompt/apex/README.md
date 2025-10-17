# APEX - Analysis-based Prompt Engineering eXpert

APEX is DSPy's teleprompter for people who believe prompt engineering should
feel like disciplined debugging.  Instead of nudging prompts at random, it walks
through the same loop an experienced practitioner would follow: study how the
program fails, contrast those failures with the wins, generate targeted
hypotheses, and keep only the change that meaningfully improves quality.  The
optimizer's job is to turn that philosophy into a repeatable process that scales
as your programs and datasets grow.

## Why this approach works

* **Evidence before edits.** APEX never rewrites prompts without first building a
  concrete picture of what went wrong.  Detailed traces and structured root cause
  summaries keep the focus on actual bugs instead of speculation.
* **Contrastive reasoning.** Successful executions are analysed alongside
  failures, helping the language model tease apart signal from noise and preserve
  the behaviours that already work.
* **Comprehensive yet minimal fixes.** Each hypothesis is expected to tackle all
  fixable issues it sees while biasing toward the smallest effective change.  The
  loop therefore explores meaningfully different strategies without thrashing the
  program.
* **Stable evaluation.** Candidates compete on a dedicated calibration set with
  repeated runs and median aggregation, so improvements stick instead of being
  artifacts of stochastic metrics.
* **Progress you can trust.** APEX tracks every candidate and keeps the global
  best in reserve, allowing you to stop when progress plateaus without losing the
  highest-quality variant.

## How the optimization loop unfolds

1. **Establish a baseline.** The uncompiled student module is deep-copied and
   evaluated on the calibration set to anchor the optimization.
2. **Sample and observe.** Each iteration draws a training subset (full set by
   default), runs the current baseline with tracing enabled, and labels successes
   versus failures using the chosen metric.
3. **Analyse in parallel.** A language model reviews each failure—and a matching
   slice of successes—to produce structured summaries of root causes, involved
   predictors, and noteworthy context.
4. **Synthesize hypotheses.** Another model aggregates those summaries and
   proposes up to ``num_hypotheses`` complete prompt rewrites that would fix all
   identified, fixable issues.  It can abstain when nothing promising emerges.
5. **Evaluate candidates.** The baseline and every hypothesis-derived program are
   scored on the calibration set.  Median aggregation across repeated runs
   reduces variance and prevents lucky streaks from winning.
6. **Select and repeat.** The highest-scoring candidate becomes the new baseline.
   If no improvement occurs for ``convergence_patience`` iterations—or the
   iteration budget is met—the process halts and returns the best candidate seen
   across the entire run.

## What you bring to the table

* **A student program.** Any DSPy module whose predictors you want to optimize.
* **Two datasets.** A training set for analysis (the "lab notebook") and a
  calibration set that acts as the judge.  The calibration set is the only signal
  used when choosing between candidates, so make it representative of your
  deployment environment.
* **A metric.** A callable that scores predictions.  Returning optional feedback
  helps the analysis model reason about errors, but a simple float works too.
* **Optional persistence.** Provide a ``checkpoint_dir`` to resume long runs or
  an MLflow configuration to log iteration history.

## Using APEX in practice

1. Instantiate ``APEX`` with the metric, analysis model, and any optional knobs
   such as ``num_hypotheses`` or a ``train_sample`` policy.
2. Call ``compile(student, trainset=train, valset=cal)``.  The teleprompter will
   manage sampling, language model calls, evaluation, and selection.
3. Inspect the returned module and history (if requested) to understand what
   changed and why.  Because every hypothesis includes change summary text, you can
   audit improvements or feed them into additional tooling.

APEX provides a structured bridge between human-style error analysis and the
automation DSPy offers.  It excels when you want systematic, interpretable
progress on complex prompt graphs without juggling dozens of hyperparameters or
one-off experiments.
