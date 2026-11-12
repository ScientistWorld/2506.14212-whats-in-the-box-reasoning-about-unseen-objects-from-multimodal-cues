# Evaluation

## How a method is run

A method is an executable shell script `method/predictors/<name>.sh`. The
evaluators call every such script, as

```
bash method/predictors/<name>.sh INPUT_JSON OUTPUT_JSON
```

with the workspace root as working directory. Every script on disk is
scored under its file name (`<name>`).

- `INPUT_JSON` has `{"scenarios": [...]}` with the fields described in
  `briefing/problem.md` and `data/README.md`, but **without** `true_box`.
- `OUTPUT_JSON` must be
  `{"scenarios": [{"id": "<scenario id>", "box_probs": {"<object name>": [p_left, p_right], ...}}, ...]}`.
  Every scenario id and every object name must be present; each list has one
  finite probability per box in [0, 1] (floating-point slack of 1e-6 is
  clipped) and sums to 1 (tolerance 1e-3). Anything else in the file is
  ignored.
- The script runs offline on one CPU core and must finish within 2 hours.
  It may use code and files in the workspace (for example parameters fitted
  on the train split).

A script that crashes, times out, or writes an incomplete or invalid output
**fails**. A failed method receives the worst possible value of every scored
metric (correlation −1, Brier 1, log loss −ln 1e-4 ≈ 9.21).

## Splits

- **Train (visible):** `data/witb_train.json`, 240 scenarios (881 objects)
  with their true placements. `bash scripts/evaluate_train.sh` (no arguments)
  runs every predictor on them and writes `scoring/scores_train.json`.
  `bash scripts/evaluate.sh` does the same and writes `scoring/scores.json`.
- **Test (hidden):** 2000 scenarios (7075 objects) from the same simulated
  world, held only by the hidden evaluator. `bash scripts/evaluate_test.sh`
  (no arguments) runs every predictor on them and writes
  `scoring/scores_test.json`. Predictors get the unlabelled inputs in a
  temporary directory; the evaluator's own directory is unreadable while they
  run. Every method sees exactly the same test inputs.

## Metrics (experiment `witb_placement_inference`)

All metrics are recomputed by the evaluator from `box_probs` and the true
placements.

| metric | better | coefficient | meaning |
|---|---|---:|---|
| `truth_correlation` (primary) | higher | 1.0 | Pearson correlation over all (object, box) queries between the predicted probability and whether the object is in that box |
| `brier_score` | lower | 1.0 | mean over objects of ½ Σ_box (p − 1[true box])² |
| `log_loss` | lower | 0.5 | mean over objects of −ln max(p_true box, 1e-4) |

Correlation measures how well a method ranks likely and unlikely placements
(the statistic the original study reports). The two proper scoring rules also
require the probabilities to be calibrated, so a method cannot gain by
sharpening or flattening its outputs.

## Reference and comparison numbers

`scoring/scores_test.json` holds the reference method's hidden-test scores
(correlation 0.802, Brier 0.0893, log loss 0.346). Each score file also lists,
under `_baselines`, fixed comparison predictions stored with the evaluators
(single-cue systems, an average of the two single-cue systems, and chance).
They are not ranked.

| comparison (hidden test) | correlation | Brier | log loss |
|---|---:|---:|---:|
| sound cue only | 0.770 | 0.104 | 0.340 |
| size cue only | 0.454 | 0.203 | 0.655 |
| average of the two single-cue outputs | 0.764 | 0.119 | 0.394 |
| chance (0.5 / 0.5) | 0.000 | 0.250 | 0.693 |

The original study reports correlations with human ratings (0.78 for its
model, 0.55 sound only, 0.52 vision only, 0.31 for a large vision-language
model). Those are on a different target and are kept in
`scoring/reference.json` under `_paper_reported`, not as numbers to match.
