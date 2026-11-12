# Evaluation Targets

All targets are measured in `scoring/scores_test.json` (hidden test, written
by `bash scripts/evaluate_test.sh`, no arguments) under
`experiments.witb_placement_inference.results.<method>`. The same metrics on
the visible train split are in `scoring/scores_train.json`
(`bash scripts/evaluate_train.sh`).

## Primary: placement judgments track the truth

- **Metric:** `truth_correlation` (higher is better; coefficient 1.0)
- **Reference (paper's full model):** 0.802 on the hidden test (0.799 on train)

## Constraint: judgments are calibrated probabilities

Better ranking must not come from distorted confidence.

- **Metric:** `brier_score` (lower is better; coefficient 1.0). Reference 0.0893.
- **Metric:** `log_loss` (lower is better; coefficient 0.5, probabilities floored at 1e-4). Reference 0.346.

## Constraint: valid output on every hidden scenario

A method that crashes, times out (2 h), or omits or mis-specifies any
scenario/object distribution is scored at the worst value of all three
metrics (−1, 1, 9.21).

## Comparison diagnostics (not ranked)

`_baselines` in both score files: sound-only (paper Eq. 3), size-only, the
average of the two, and chance. On the hidden test the paper's full model is
above sound-only (0.770), size-only (0.454) and the average (0.764) in
correlation and Brier score, which is the paper's claim in its direction.
