# Benchmark scoring audit

Run `bash redteam/<case>/run.sh` from the gym container to install that case's
persistent prediction/data changes, invoke visible training evaluation, and
invoke `bash scripts/evaluate_test.sh` **without arguments**, in a new process.
The Python environment and hidden evaluator are never modified. A context
manager restores the original dataset, score files, and output artifacts after
each case. Cases are sequential because they share the default output paths.

`method.json` in each folder defines the actual method/shortcut. The common
implementation is `redteam/audit.py`. `fixtures/witb_bayes.py` is the original
agent-visible inference implementation. Prediction generation uses only that
code, visible data, and (for calibration) the visible train scenario list.
The original hidden scores and reference metadata in `fixtures/` are used only
for audit comparisons and initializing identical workspace snapshots, never
for candidate probabilities. Explicit answer-copy and target-rewrite shortcuts
use the labels already present in the visible benchmark file.

All reference/quality runs preserve the original 1000-sample seed-17 ablation
artifacts as fixed comparison methods. Three reference runs change only the
candidate's random seed. This measures the noise relevant to improving a method
against fixed baselines. The higher-sample variant uses 10000 samples.

The calibration variant retains the complete Bayesian inference algorithm and
fits one shrinkage strength on the nine visible train queries to minimize Brier
error. It never fits on test judgments. The weaker audio/vision variants remove
one modality. `late_fusion` is an honest simple baseline, not answer copying.

The all-case job is `python3 redteam/audit.py all`. Regenerate aggregate rank
analysis using `python3 redteam/audit.py summarize`. Successful test runs are
ranked using the absolute coefficients, better directions, experiment weight,
and population standard deviations across the named audit cohort. Missing
metrics are skipped as specified. Future attempt variances are unknown, so raw
metrics and Pareto comparisons are also retained.

Failed evaluations retain a copy of the score file left on disk and their
nonzero exit code. Such a stale file is **not** reported as a successful score.
Scores and `execution.json` are tracked evidence; generated predictions,
intermediate caches and full stdout logs are ignored.
